#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

try:
    import torch._dynamo as torch_dynamo
except Exception:  # pragma: no cover
    torch_dynamo = None

from scripts.benchmark_omol_mixed_backend import (
    _compile_model,
    _copy_matching_state,
    _forward,
    _load_json,
    _loss_from_outputs,
    _make_batch,
    _make_model,
)
from matterformer.models.platonic.layers import PlatonicAttention
from matterformer.models.platonic.linear import PlatonicLinear


ATTENTION_MODES: dict[str, dict[str, Any]] = {
    "sdpa": {"attention_backend": "sdpa", "attention_bias": None},
    "flash": {"attention_backend": "flash", "attention_bias": None},
    "triton_tf32": {"attention_backend": "triton", "attention_bias": {"precision": "tf32", "strict": True}},
    "triton_tf32x3": {"attention_backend": "triton", "attention_bias": {"precision": "tf32x3", "strict": True}},
    "triton_ieee": {"attention_backend": "triton", "attention_bias": {"precision": "ieee", "strict": True}},
    "triton_bf16compat": {
        "attention_backend": "triton",
        "attention_bias": {"precision": "bf16_flash_compat", "strict": True},
    },
    "triton_radial_r2": {
        "attention_backend": "triton_radial_r2",
        "attention_bias": {
            "kind": "radial_r2",
            "zero_init": True,
            "gate_init": 0.0,
            "diag_zero": True,
            "precision": "tf32x3",
            "strict": True,
        },
    },
    "triton_radial_slope": {
        "attention_backend": "triton_radial_slope",
        "attention_bias": {
            "kind": "radial_slope",
            "zero_init": True,
            "gate_init": 0.0,
            "diag_zero": True,
            "precision": "tf32x3",
            "strict": True,
        },
    },
    "triton_radial_rbf8": {
        "attention_backend": "triton_radial_rbf",
        "attention_bias": {
            "kind": "radial_rbf",
            "num_rbf": 8,
            "rbf_min": 0.0,
            "rbf_max": 6.0,
            "zero_init": True,
            "gate_init": 0.0,
            "diag_zero": True,
            "precision": "tf32x3",
            "strict": True,
        },
    },
}


LINEAR_MODES: dict[str, dict[str, Any]] = {
    "spatial_recompute": {
        "linear_backend": "spatial",
        "attention_linear_backend": "spatial",
        "ffn_linear_backend": "spatial",
        "rope_cache": False,
        "constant_key_fastpath": False,
        "fused_qv": False,
    },
    "spatial_fast": {
        "linear_backend": "spatial",
        "attention_linear_backend": "spatial",
        "ffn_linear_backend": "spatial",
        "rope_cache": True,
        "constant_key_fastpath": True,
        "fused_qv": True,
    },
    "ffn_fourier_scaffold": {
        "linear_backend": "spatial",
        "attention_linear_backend": "spatial",
        "ffn_linear_backend": "fourier",
        "rope_cache": True,
        "constant_key_fastpath": True,
        "fused_qv": True,
    },
    "ffn_fourier_direct": {
        "linear_backend": "spatial",
        "attention_linear_backend": "spatial",
        "ffn_linear_backend": "fourier_direct",
        "rope_cache": True,
        "constant_key_fastpath": True,
        "fused_qv": True,
    },
    "all_fourier_scaffold": {
        "linear_backend": "fourier",
        "attention_linear_backend": "fourier",
        "ffn_linear_backend": "fourier",
        "rope_cache": True,
        "constant_key_fastpath": True,
        "fused_qv": True,
    },
    "all_fourier_direct": {
        "linear_backend": "fourier_direct",
        "attention_linear_backend": "fourier_direct",
        "ffn_linear_backend": "fourier_direct",
        "rope_cache": True,
        "constant_key_fastpath": True,
        "fused_qv": True,
    },
}


PRECISION_MODES: dict[str, dict[str, Any]] = {
    "fp32_highest": {"matmul_precision": "highest", "amp_bf16": False},
    "fp32_high": {"matmul_precision": "high", "amp_bf16": False},
    "fp32_medium": {"matmul_precision": "medium", "amp_bf16": False},
    "ampbf16_high": {"matmul_precision": "high", "amp_bf16": True},
}


def _config_for_combo(base_cfg: dict[str, Any], *, attention_mode: str, linear_mode: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    tetra = cfg.setdefault("tetra", {})
    attention = ATTENTION_MODES[attention_mode]
    linear = LINEAR_MODES[linear_mode]
    tetra.update(linear)
    tetra["attention_backend"] = attention["attention_backend"]
    if attention["attention_bias"] is None:
        tetra.pop("attention_bias", None)
    else:
        tetra["attention_bias"] = dict(attention["attention_bias"])
    return cfg


def _run_with_autocast(model: nn.Module, batch: tuple[torch.Tensor, ...], *, amp_bf16: bool) -> dict[str, torch.Tensor]:
    if amp_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return _forward(model, batch)  # type: ignore[arg-type]
    return _forward(model, batch)  # type: ignore[arg-type]


def _time_fwd_bwd(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    repeats: int,
    amp_bf16: bool,
) -> tuple[float, float]:
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _loss_from_outputs(_run_with_autocast(model, batch, amp_bf16=amp_bf16)).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        _loss_from_outputs(_run_with_autocast(model, batch, amp_bf16=amp_bf16)).backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    return 1000.0 * elapsed / float(repeats), torch.cuda.max_memory_allocated() / 1024**3


def _time_forward(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    repeats: int,
    amp_bf16: bool,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _run_with_autocast(model, batch, amp_bf16=amp_bf16)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _run_with_autocast(model, batch, amp_bf16=amp_bf16)
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / float(repeats)


def _reset_runtime_state() -> None:
    if torch_dynamo is not None:
        try:
            torch_dynamo.reset()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def _copy_equivalent_platonic_modules_for_matrix(target: nn.Module, source: nn.Module) -> None:
    """Copy spatial-reference weights into equivalent benchmark backends.

    The shared mixed-backend helper handles spatial fused-qv and direct-Fourier
    non-attention linears. The matrix also benchmarks all-Fourier attention,
    where a fused qv projection must be initialized from separate spatial q/v
    kernels by converting the concatenated spatial kernel into the target
    backend.
    """
    source_modules = dict(source.named_modules())
    for name, target_module in target.named_modules():
        source_module = source_modules.get(name)
        if isinstance(target_module, PlatonicAttention) and isinstance(source_module, PlatonicAttention):
            if target_module.qv_proj is not None and source_module.q_proj is not None and source_module.v_proj is not None:
                if (
                    source_module.q_proj.linear_backend == source_module.v_proj.linear_backend
                    and source_module.q_proj.linear_backend == target_module.qv_proj.linear_backend
                ):
                    target_module.set_fused_qv_from_separate_(source_module.q_proj, source_module.v_proj)
                elif source_module.q_proj.kernel is not None and source_module.v_proj.kernel is not None:
                    kernel = torch.cat([source_module.q_proj.kernel, source_module.v_proj.kernel], dim=1)
                    bias = None
                    if source_module.q_proj.bias is not None and source_module.v_proj.bias is not None:
                        bias = torch.cat([source_module.q_proj.bias, source_module.v_proj.bias], dim=0)
                    target_module.qv_proj.set_spatial_parameters_(kernel, bias)
        if isinstance(target_module, PlatonicLinear) and isinstance(source_module, PlatonicLinear):
            if target_module.linear_backend == "fourier_direct" and source_module.kernel is not None:
                target_module.set_spatial_parameters_(source_module.kernel, source_module.bias)


def _write_outputs(output_prefix: Path, payload: dict[str, Any]) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows = payload["rows"]
    fieldnames = [
        "status",
        "attention_mode",
        "linear_mode",
        "compile_scope",
        "precision_mode",
        "forward_ms",
        "fwd_bwd_ms",
        "max_mem_gb",
        "energy_max_abs",
        "force_max_abs",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    valid_rows.sort(key=lambda row: float(row["fwd_bwd_ms"]))
    lines = [
        "# OMol Backend Matrix Benchmark",
        "",
        "## Shape",
        "",
        "```json",
        json.dumps(payload["batch"], indent=2, sort_keys=True),
        "```",
        "",
        "## Fastest fwd+bwd Rows",
        "",
        "| rank | attention | linear | compile | precision | fwd+bwd ms | forward ms | mem GB | force max abs |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(valid_rows[:20], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["attention_mode"]),
                    str(row["linear_mode"]),
                    str(row["compile_scope"]),
                    str(row["precision_mode"]),
                    f"{float(row['fwd_bwd_ms']):.3f}",
                    f"{float(row['forward_ms']):.3f}",
                    f"{float(row['max_mem_gb']):.3f}",
                    f"{float(row['force_max_abs']):.3e}",
                ]
            )
            + " |"
        )
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "## Failed Rows", ""])
        for row in failed:
            lines.append(
                f"- `{row.get('attention_mode')}` / `{row.get('linear_mode')}` / "
                f"`{row.get('compile_scope')}` / `{row.get('precision_mode')}`: {row.get('error')}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive OMol backend matrix benchmark.")
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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attention-modes", nargs="+", default=["flash", "triton_tf32x3", "triton_bf16compat", "triton_radial_r2", "triton_radial_slope", "triton_radial_rbf8"])
    parser.add_argument("--linear-modes", nargs="+", default=["spatial_recompute", "spatial_fast", "ffn_fourier_direct", "all_fourier_direct"])
    parser.add_argument("--compile-scopes", nargs="+", default=["none", "trunk_flat"])
    parser.add_argument("--precision-modes", nargs="+", default=["fp32_high"])
    parser.add_argument("--output-prefix", type=Path, default=Path("oracle_context/omol_backend_matrix_20260516/backend_matrix"))
    parser.add_argument("--stop-after", type=int, default=None, help="Debug helper: stop after this many combinations.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    for name in args.attention_modes:
        if name not in ATTENTION_MODES:
            raise ValueError(f"Unknown attention mode {name!r}; choices={sorted(ATTENTION_MODES)}")
    for name in args.linear_modes:
        if name not in LINEAR_MODES:
            raise ValueError(f"Unknown linear mode {name!r}; choices={sorted(LINEAR_MODES)}")
    for name in args.precision_modes:
        if name not in PRECISION_MODES:
            raise ValueError(f"Unknown precision mode {name!r}; choices={sorted(PRECISION_MODES)}")

    torch.manual_seed(0)
    device = torch.device("cuda")
    base_cfg = _load_json(args.hybrid_config_json)
    batch_tensors = _make_batch(
        total_atoms=args.total_atoms,
        num_graphs=args.num_graphs,
        max_atoms_per_graph=args.max_atoms_per_graph,
        device=device,
    )
    batch = batch_tensors[:5]
    batch_stats = batch_tensors[5]

    reference_cfg = _config_for_combo(base_cfg, attention_mode="flash", linear_mode="spatial_recompute")
    reference = _make_model(args, reference_cfg).to(device).eval()
    reference_outputs_by_precision: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for precision_name in args.precision_modes:
            precision = PRECISION_MODES[precision_name]
            torch.set_float32_matmul_precision(str(precision["matmul_precision"]))
            outputs = _run_with_autocast(reference, batch, amp_bf16=bool(precision["amp_bf16"]))  # type: ignore[arg-type]
            reference_outputs_by_precision[precision_name] = {
                key: value.detach().clone() for key, value in outputs.items()
            }
    reference = reference.cpu()
    _reset_runtime_state()

    combos = list(itertools.product(args.precision_modes, args.compile_scopes, args.attention_modes, args.linear_modes))
    if args.stop_after is not None:
        combos = combos[: int(args.stop_after)]
    payload: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch": batch_stats,
        "args": vars(args) | {"output_prefix": str(args.output_prefix)},
        "rows": [],
    }
    print("matrix " + json.dumps({k: payload[k] for k in ("gpu", "torch", "cuda", "batch")}, sort_keys=True))
    print(f"matrix_combinations={len(combos)}")

    for index, (precision_mode, compile_scope, attention_mode, linear_mode) in enumerate(combos, start=1):
        precision = PRECISION_MODES[precision_mode]
        torch.set_float32_matmul_precision(str(precision["matmul_precision"]))
        amp_bf16 = bool(precision["amp_bf16"])
        row: dict[str, Any] = {
            "status": "ok",
            "index": index,
            "total": len(combos),
            "precision_mode": precision_mode,
            "compile_scope": compile_scope,
            "attention_mode": attention_mode,
            "linear_mode": linear_mode,
        }
        print("start " + json.dumps(row, sort_keys=True), flush=True)
        start_wall = time.perf_counter()
        try:
            _reset_runtime_state()
            cfg = _config_for_combo(base_cfg, attention_mode=attention_mode, linear_mode=linear_mode)
            model = _make_model(args, cfg).to(device).eval()
            _copy_matching_state(model, reference)
            _copy_equivalent_platonic_modules_for_matrix(model, reference)
            with torch.no_grad():
                outputs = _run_with_autocast(model, batch, amp_bf16=amp_bf16)  # type: ignore[arg-type]
            reference_outputs = reference_outputs_by_precision[precision_mode]
            energy_error = (outputs["energy"].float() - reference_outputs["energy"].float()).abs()
            force_error = (outputs["forces"].float() - reference_outputs["forces"].float()).abs()
            row.update(
                {
                    "energy_max_abs": float(energy_error.max().item()),
                    "energy_mean_abs": float(energy_error.mean().item()),
                    "force_max_abs": float(force_error.max().item()),
                    "force_mean_abs": float(force_error.mean().item()),
                }
            )
            model = _compile_model(model, str(compile_scope))
            row["forward_ms"] = _time_forward(
                model,
                batch,  # type: ignore[arg-type]
                warmup=max(1, int(args.warmup)),
                repeats=int(args.repeats),
                amp_bf16=amp_bf16,
            )
            fwd_bwd_ms, mem_gb = _time_fwd_bwd(
                model,
                batch,  # type: ignore[arg-type]
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                amp_bf16=amp_bf16,
            )
            row["fwd_bwd_ms"] = fwd_bwd_ms
            row["max_mem_gb"] = mem_gb
            row["wall_s"] = time.perf_counter() - start_wall
            del model
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["wall_s"] = time.perf_counter() - start_wall
        finally:
            payload["rows"].append(row)
            _write_outputs(args.output_prefix, payload)
            print("done " + json.dumps(row, sort_keys=True), flush=True)
            _reset_runtime_state()

    _write_outputs(args.output_prefix, payload)
    valid = [row for row in payload["rows"] if row.get("status") == "ok"]
    valid.sort(key=lambda row: float(row["fwd_bwd_ms"]))
    print("top " + json.dumps(valid[:10], indent=2, sort_keys=True))
    failed = [row for row in payload["rows"] if row.get("status") != "ok"]
    if failed:
        print("failed " + json.dumps(failed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
