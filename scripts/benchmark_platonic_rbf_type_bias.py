#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from matterformer.models.platonic.layers import flash_attn_varlen_func
from matterformer.models.platonic.triton_attention import (
    TRITON_PLATONIC_ATTENTION_AVAILABLE,
    platonic_attention_flat_torch_reference,
    platonic_attention_flat_triton,
)
from scripts.benchmark_omol_mixed_backend import (
    _compile_model,
    _copy_equivalent_platonic_modules,
    _copy_matching_state,
    _forward,
    _load_json,
    _loss_from_outputs,
    _make_batch,
    _make_model,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _max_abs(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _make_cu_seqlens(total_tokens: int, num_graphs: int, device: torch.device) -> torch.Tensor:
    if total_tokens < num_graphs:
        raise ValueError("total_tokens must be >= num_graphs")
    base = total_tokens // num_graphs
    counts = torch.full((num_graphs,), base, dtype=torch.int32, device=device)
    counts[: total_tokens - base * num_graphs] += 1
    cu = torch.zeros(num_graphs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, dim=0)
    return cu


def _attention_inputs(
    *,
    total_tokens: int,
    num_graphs: int,
    num_heads: int,
    head_dim: int,
    heads_per_frame: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | int]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype, generator=generator)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype, generator=generator)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype, generator=generator)
    pos = torch.randn(total_tokens, 3, device=device, dtype=torch.float32, generator=generator)
    atom_types = torch.randint(1, 31, (total_tokens,), device=device, generator=generator)
    centers = torch.linspace(0.0, 6.0, 4, device=device)
    gamma = torch.tensor(0.75, device=device)
    rbf_weight = 0.01 * torch.randn(heads_per_frame, 4, device=device, generator=generator)
    type_bias = 0.01 * torch.randn(31, 31, heads_per_frame, device=device, generator=generator)
    cu = _make_cu_seqlens(total_tokens, num_graphs, device)
    return {
        "q": q,
        "k": k,
        "v": v,
        "pos": pos,
        "atom_types": atom_types,
        "centers": centers,
        "gamma": gamma,
        "rbf_weight": rbf_weight,
        "type_bias": type_bias,
        "cu_seqlens": cu,
        "max_seqlen": int((cu[1:] - cu[:-1]).max().item()),
        "heads_per_frame": int(heads_per_frame),
    }


def _attention_parity(device: torch.device, *, precision: str) -> dict[str, Any]:
    torch.manual_seed(100)
    data = _attention_inputs(
        total_tokens=64,
        num_graphs=4,
        num_heads=12,
        head_dim=16,
        heads_per_frame=3,
        device=device,
    )
    q = data["q"].detach().clone().requires_grad_(True)
    k = data["k"].detach().clone().requires_grad_(True)
    v = data["v"].detach().clone().requires_grad_(True)
    w = data["rbf_weight"].detach().clone().requires_grad_(True)
    tb = data["type_bias"].detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    tb_ref = tb.detach().clone().requires_grad_(True)

    out = platonic_attention_flat_triton(
        q,
        k,
        v,
        cu_seqlens=data["cu_seqlens"],
        max_seqlen=int(data["max_seqlen"]),
        pos=data["pos"],
        atom_types=data["atom_types"],
        heads_per_frame=int(data["heads_per_frame"]),
        rbf_weight=w,
        type_bias=tb,
        centers=data["centers"],
        gamma=data["gamma"],
        cutoff=6.0,
        max_atomic_number=30,
        radial_bias_kind="rbf_type_enveloped",
        strict=True,
        precision=precision,
    )
    ref = platonic_attention_flat_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens=data["cu_seqlens"],
        max_seqlen=int(data["max_seqlen"]),
        pos=data["pos"],
        atom_types=data["atom_types"],
        heads_per_frame=int(data["heads_per_frame"]),
        rbf_weight=w_ref,
        type_bias=tb_ref,
        centers=data["centers"],
        gamma=data["gamma"],
        cutoff=6.0,
        radial_bias_kind="rbf_type_enveloped",
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)

    zero_w = torch.zeros_like(w.detach()).requires_grad_(True)
    zero_tb = torch.zeros_like(tb.detach()).requires_grad_(True)
    q_zero = data["q"].detach().clone().requires_grad_(True)
    k_zero = data["k"].detach().clone().requires_grad_(True)
    v_zero = data["v"].detach().clone().requires_grad_(True)
    out_zero = platonic_attention_flat_triton(
        q_zero,
        k_zero,
        v_zero,
        cu_seqlens=data["cu_seqlens"],
        max_seqlen=int(data["max_seqlen"]),
        pos=data["pos"],
        atom_types=data["atom_types"],
        heads_per_frame=int(data["heads_per_frame"]),
        rbf_weight=zero_w,
        type_bias=zero_tb,
        centers=data["centers"],
        gamma=data["gamma"],
        cutoff=6.0,
        max_atomic_number=30,
        radial_bias_kind="rbf_type_enveloped",
        strict=True,
        precision=precision,
    )
    out_plain = platonic_attention_flat_triton(
        data["q"].detach().clone(),
        data["k"].detach().clone(),
        data["v"].detach().clone(),
        cu_seqlens=data["cu_seqlens"],
        max_seqlen=int(data["max_seqlen"]),
        strict=True,
        precision=precision,
    )

    return {
        "precision": precision,
        "triton_available": bool(TRITON_PLATONIC_ATTENTION_AVAILABLE),
        "forward_max_abs": _max_abs(out, ref),
        "dq_max_abs": _max_abs(q.grad, q_ref.grad),
        "dk_max_abs": _max_abs(k.grad, k_ref.grad),
        "dv_max_abs": _max_abs(v.grad, v_ref.grad),
        "drbf_weight_max_abs": _max_abs(w.grad, w_ref.grad),
        "dtype_bias_max_abs": _max_abs(tb.grad, tb_ref.grad),
        "zero_init_vs_plain_triton_forward_max_abs": _max_abs(out_zero, out_plain),
    }


def _bench_callable(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    backward: bool,
) -> dict[str, Any]:
    for _ in range(warmup):
        out = fn()
        if backward:
            out.float().square().mean().backward()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        start = time.perf_counter()
        out = fn()
        if backward:
            out.float().square().mean().backward()
        _sync()
        times.append(1000.0 * (time.perf_counter() - start))
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    return {"name": name, "pass": "fwd_bwd" if backward else "fwd", **_stats_ms(times), "peak_allocated_gib": peak_gib}


def _attention_perf(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    data = _attention_inputs(
        total_tokens=args.attn_tokens,
        num_graphs=args.attn_graphs,
        num_heads=args.n_heads,
        head_dim=args.head_dim,
        heads_per_frame=args.heads_per_frame,
        device=device,
    )
    rows: list[dict[str, Any]] = []

    def make_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            data["q"].detach().clone().requires_grad_(True),
            data["k"].detach().clone().requires_grad_(True),
            data["v"].detach().clone().requires_grad_(True),
        )

    def triton_plain() -> torch.Tensor:
        q, k, v = make_qkv()
        return platonic_attention_flat_triton(
            q,
            k,
            v,
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=int(data["max_seqlen"]),
            strict=True,
            precision="tf32x3",
        )

    def triton_rbf_type() -> torch.Tensor:
        q, k, v = make_qkv()
        return platonic_attention_flat_triton(
            q,
            k,
            v,
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=int(data["max_seqlen"]),
            pos=data["pos"],
            atom_types=data["atom_types"],
            heads_per_frame=int(data["heads_per_frame"]),
            rbf_weight=data["rbf_weight"].detach().clone().requires_grad_(True),
            type_bias=data["type_bias"].detach().clone().requires_grad_(True),
            centers=data["centers"],
            gamma=data["gamma"],
            cutoff=6.0,
            max_atomic_number=30,
            radial_bias_kind="rbf_type_enveloped",
            strict=True,
            precision="tf32x3",
        )

    def triton_rbf_type_zero() -> torch.Tensor:
        q, k, v = make_qkv()
        return platonic_attention_flat_triton(
            q,
            k,
            v,
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=int(data["max_seqlen"]),
            pos=data["pos"],
            atom_types=data["atom_types"],
            heads_per_frame=int(data["heads_per_frame"]),
            rbf_weight=torch.zeros_like(data["rbf_weight"]).requires_grad_(True),
            type_bias=torch.zeros_like(data["type_bias"]).requires_grad_(True),
            centers=data["centers"],
            gamma=data["gamma"],
            cutoff=6.0,
            max_atomic_number=30,
            radial_bias_kind="rbf_type_enveloped",
            strict=True,
            precision="tf32x3",
        )

    if flash_attn_varlen_func is not None:
        def flash_varlen() -> torch.Tensor:
            q, k, v = make_qkv()
            out = flash_attn_varlen_func(
                q.contiguous().to(torch.bfloat16),
                k.contiguous().to(torch.bfloat16),
                v.contiguous().to(torch.bfloat16),
                cu_seqlens_q=data["cu_seqlens"],
                cu_seqlens_k=data["cu_seqlens"],
                max_seqlen_q=int(data["max_seqlen"]),
                max_seqlen_k=int(data["max_seqlen"]),
                dropout_p=0.0,
                causal=False,
            )
            return out.to(torch.float32)

        rows.append(_bench_callable("attention_flash_varlen_bf16", flash_varlen, warmup=args.warmup, repeats=args.repeats, backward=True))
    rows.append(_bench_callable("attention_triton_plain_tf32x3", triton_plain, warmup=args.warmup, repeats=args.repeats, backward=True))
    rows.append(_bench_callable("attention_triton_rbf_type4_zero_tf32x3", triton_rbf_type_zero, warmup=args.warmup, repeats=args.repeats, backward=True))
    rows.append(_bench_callable("attention_triton_rbf_type4_nonzero_tf32x3", triton_rbf_type, warmup=args.warmup, repeats=args.repeats, backward=True))
    return rows


def _with_tetra_common(base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    tetra = cfg.setdefault("tetra", {})
    tetra["linear_backend"] = "spatial"
    tetra["attention_linear_backend"] = "spatial"
    tetra["ffn_linear_backend"] = "spatial"
    tetra["rope_cache"] = True
    tetra["constant_key_fastpath"] = True
    tetra["fused_qv"] = True
    return cfg


def _full_model_cfg(base_cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = _with_tetra_common(base_cfg)
    tetra = cfg.setdefault("tetra", {})
    tetra.pop("local_attention_mod", None)
    if variant == "flash_fast":
        tetra["attention_backend"] = "flash"
        tetra.pop("attention_bias", None)
    elif variant == "triton_plain":
        tetra["attention_backend"] = "triton"
        tetra["attention_bias"] = {"precision": "tf32x3", "strict": True}
    elif variant == "local_rbf_type_every4":
        tetra["attention_backend"] = "flash"
        tetra.pop("attention_bias", None)
        tetra["local_attention_mod"] = {
            "enabled": True,
            "backend": "triton",
            "every": 4,
            "offset": 3,
            "kind": "rbf_type_enveloped",
            "num_rbf": 4,
            "cutoff": 6.0,
            "max_atomic_number": 118,
            "diag_zero": True,
            "zero_init": True,
            "precision": "tf32x3",
            "strict": True,
            "block_m": 16,
            "block_n": 32,
        }
    elif variant == "local_rbf_type_all":
        tetra["attention_backend"] = "triton"
        tetra["attention_bias"] = {
            "kind": "rbf_type_enveloped",
            "num_rbf": 4,
            "cutoff": 6.0,
            "max_atomic_number": 118,
            "diag_zero": True,
            "zero_init": True,
            "precision": "tf32x3",
            "strict": True,
            "block_m": 16,
            "block_n": 32,
        }
    else:
        raise ValueError(f"Unknown full-model variant {variant!r}")
    return cfg


def _full_model_perf(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, float]]:
    base_cfg = _load_json(args.hybrid_config_json)
    batch_tensors = _make_batch(
        total_atoms=args.total_atoms,
        num_graphs=args.num_graphs,
        max_atoms_per_graph=args.max_atoms_per_graph,
        device=device,
    )
    batch = batch_tensors[:5]
    stats = batch_tensors[5]

    ref_cfg = _full_model_cfg(base_cfg, "flash_fast")
    ref_model = _make_model(args, ref_cfg).to(device).eval()
    ref_outputs = _forward(ref_model, batch)
    ref_loss = float(_loss_from_outputs(ref_outputs).detach().item())

    rows: list[dict[str, Any]] = []
    for compile_scope in args.compile_scope:
        for variant in args.full_variants:
            cfg = _full_model_cfg(base_cfg, variant)
            model = _make_model(args, cfg).to(device).eval()
            _copy_matching_state(model, ref_model)
            _copy_equivalent_platonic_modules(model, ref_model)
            model = _compile_model(model, compile_scope)

            with torch.no_grad():
                outputs = _forward(model, batch)
                energy_diff = float((outputs["energy"].detach().float() - ref_outputs["energy"].detach().float()).abs().max().item())
                force_diff = float((outputs["forces"].detach().float() - ref_outputs["forces"].detach().float()).abs().max().item())
                loss = float(_loss_from_outputs(outputs).detach().item())

            for _ in range(args.warmup):
                model.zero_grad(set_to_none=True)
                _loss_from_outputs(_forward(model, batch)).backward()
            _sync()
            torch.cuda.reset_peak_memory_stats()
            times: list[float] = []
            for _ in range(args.repeats):
                model.zero_grad(set_to_none=True)
                start = time.perf_counter()
                _loss_from_outputs(_forward(model, batch)).backward()
                _sync()
                times.append(1000.0 * (time.perf_counter() - start))
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            rows.append(
                {
                    "variant": variant,
                    "compile_scope": compile_scope,
                    **_stats_ms(times),
                    "peak_allocated_gib": float(peak_gib),
                    "energy_max_abs_vs_flash_fast": energy_diff,
                    "force_max_abs_vs_flash_fast": force_diff,
                    "loss": loss,
                    "loss_delta_vs_flash_fast": loss - ref_loss,
                }
            )
            del model
            torch.cuda.empty_cache()
    del ref_model
    return rows, {key: float(value) for key, value in stats.items()}


def _write_outputs(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md = path.with_suffix(".md")
    lines = ["# Platonic RBF/Type Bias Benchmark", ""]
    lines.append("## Attention Parity")
    for key, value in payload["attention_parity"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Attention Microbenchmark")
    lines.append("| name | mean ms | median ms | peak GiB |")
    lines.append("|---|---:|---:|---:|")
    for row in payload["attention_perf"]:
        lines.append(f"| {row['name']} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | {row['peak_allocated_gib']:.3f} |")
    lines.append("")
    lines.append("## Full Model Fwd+Bwd")
    lines.append("| variant | compile | mean ms | median ms | peak GiB | energy max diff | force max diff |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in payload["full_model_perf"]:
        lines.append(
            f"| {row['variant']} | {row['compile_scope']} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | "
            f"{row['peak_allocated_gib']:.3f} | {row['energy_max_abs_vs_flash_fast']:.3e} | "
            f"{row['force_max_abs_vs_flash_fast']:.3e} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Platonic RBF/type local attention bias.")
    parser.add_argument("--hybrid-config-json", default="configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale.json")
    parser.add_argument("--d-model", type=int, default=1920)
    parser.add_argument("--n-heads", type=int, default=60)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--chgspin-mode", default="add", choices=["off", "add", "concat"])
    parser.add_argument("--chgspin-emb-dim", type=int, default=160)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--pair-n-rbf", type=int, default=16)
    parser.add_argument("--pair-rbf-max", type=float, default=6.0)
    parser.add_argument("--readout-activation", default="sin", choices=["gelu", "silu", "relu", "mish", "sin"])
    parser.add_argument("--total-atoms", type=int, default=12000)
    parser.add_argument("--num-graphs", type=int, default=218)
    parser.add_argument("--max-atoms-per-graph", type=int, default=220)
    parser.add_argument("--attn-tokens", type=int, default=4096)
    parser.add_argument("--attn-graphs", type=int, default=64)
    parser.add_argument("--heads-per-frame", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile-scope", nargs="+", default=["none", "trunk_flat"], choices=["none", "trunk_flat", "model"])
    parser.add_argument("--full-variants", nargs="+", default=["flash_fast", "triton_plain", "local_rbf_type_every4", "local_rbf_type_all"])
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--output-json", default="oracle_context/rbf_type_bias_delta_bench/results.json")
    parser.add_argument("--skip-full-model", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(0)
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "matmul_precision": args.matmul_precision,
        "args": vars(args),
        "attention_parity": {
            "ieee": _attention_parity(device, precision="ieee"),
            "tf32x3": _attention_parity(device, precision="tf32x3"),
        },
        "attention_perf": _attention_perf(args, device),
    }
    if args.skip_full_model:
        payload["full_model_batch"] = {}
        payload["full_model_perf"] = []
    else:
        full_rows, batch_stats = _full_model_perf(args, device)
        payload["full_model_batch"] = batch_stats
        payload["full_model_perf"] = full_rows
    output_json = Path(args.output_json)
    _write_outputs(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
