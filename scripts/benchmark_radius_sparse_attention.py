#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from matterformer.models.platonic.radius_sparse import build_radius_block_sparse_layout
from matterformer.models.platonic.triton_attention import (
    TRITON_PLATONIC_ATTENTION_AVAILABLE,
    platonic_attention_flat_torch_reference,
    platonic_attention_flat_triton,
    platonic_radius_block_sparse_attention_torch_reference,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _make_cu_seqlens(total_tokens: int, num_segments: int, device: torch.device) -> torch.Tensor:
    base = total_tokens // num_segments
    counts = torch.full((num_segments,), base, dtype=torch.int32, device=device)
    counts[: total_tokens - base * num_segments] += 1
    cu = torch.zeros(num_segments + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, dim=0)
    return cu


def _make_positions(cu: torch.Tensor, *, cutoff: float, spread: float, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20260518)
    chunks: list[torch.Tensor] = []
    starts = cu[:-1].detach().cpu().tolist()
    ends = cu[1:].detach().cpu().tolist()
    for seg_idx, (start, end) in enumerate(zip(starts, ends)):
        count = int(end) - int(start)
        center = torch.tensor([seg_idx * spread * 2.5, 0.0, 0.0], device=device)
        chunk = center + torch.randn((count, 3), device=device, generator=generator) * float(spread)
        chunks.append(chunk)
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 3), device=device)


def _make_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.heads % args.heads_per_frame != 0:
        raise ValueError("--heads must be divisible by --heads-per-frame")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cu = _make_cu_seqlens(args.tokens, args.segments, device)
    q = torch.randn((args.tokens, args.heads, args.head_dim), device=device, generator=generator)
    k = torch.randn((args.tokens, args.heads, args.head_dim), device=device, generator=generator)
    v = torch.randn((args.tokens, args.heads, args.head_dim), device=device, generator=generator)
    pos = _make_positions(cu, cutoff=args.cutoff, spread=args.position_spread, device=device)
    atom_types = torch.randint(1, args.max_atomic_number + 1, (args.tokens,), device=device, generator=generator)
    centers = torch.linspace(0.0, args.cutoff, args.num_rbf, device=device)
    delta = float(args.cutoff) / max(int(args.num_rbf) - 1, 1)
    gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), device=device)
    rbf_weight = 0.01 * torch.randn(
        (args.heads_per_frame, args.num_rbf),
        device=device,
        generator=generator,
    )
    type_bias = 0.01 * torch.randn(
        (args.max_atomic_number + 1, args.max_atomic_number + 1, args.heads_per_frame),
        device=device,
        generator=generator,
    )
    layout = build_radius_block_sparse_layout(
        pos,
        cu,
        cutoff=args.cutoff,
        block_m=args.block_m,
        block_n=args.block_n,
        sort=args.sort,
        include_self=args.include_self,
    )
    perm = layout.perm
    return {
        "q": q.index_select(0, perm).contiguous(),
        "k": k.index_select(0, perm).contiguous(),
        "v": v.index_select(0, perm).contiguous(),
        "pos": pos.index_select(0, perm).contiguous(),
        "atom_types": atom_types.index_select(0, perm).contiguous(),
        "centers": centers,
        "gamma": gamma,
        "rbf_weight": rbf_weight,
        "type_bias": type_bias,
        "cu_seqlens": cu.to(device=device, dtype=torch.int32),
        "max_seqlen": int((cu[1:] - cu[:-1]).max().item()),
        "layout": layout,
    }


def _dense_radius(data: dict[str, Any], args: argparse.Namespace, q, k, v, w, tb) -> torch.Tensor:
    return platonic_attention_flat_torch_reference(
        q,
        k,
        v,
        cu_seqlens=data["cu_seqlens"],
        max_seqlen=data["max_seqlen"],
        pos=data["pos"],
        atom_types=data["atom_types"],
        heads_per_frame=args.heads_per_frame,
        rbf_weight=w,
        type_bias=tb,
        centers=data["centers"],
        gamma=data["gamma"],
        cutoff=args.cutoff,
        radial_bias_kind="radius_rbf_type_enveloped",
        diag_zero=True,
        include_self=args.include_self,
        envelope_in_score=True,
    )


def _sparse_radius(data: dict[str, Any], args: argparse.Namespace, q, k, v, w, tb) -> torch.Tensor:
    return platonic_radius_block_sparse_attention_torch_reference(
        q,
        k,
        v,
        pos=data["pos"],
        atom_types=data["atom_types"],
        heads_per_frame=args.heads_per_frame,
        rbf_weight=w,
        type_bias=tb,
        centers=data["centers"],
        gamma=data["gamma"],
        cutoff=args.cutoff,
        diag_zero=True,
        include_self=args.include_self,
        envelope_in_score=True,
        radius_layout=data["layout"],
    )


def _dense_rbf_type_triton(data: dict[str, Any], args: argparse.Namespace, q, k, v, w, tb) -> torch.Tensor:
    return platonic_attention_flat_triton(
        q,
        k,
        v,
        cu_seqlens=data["cu_seqlens"],
        max_seqlen=data["max_seqlen"],
        pos=data["pos"],
        atom_types=data["atom_types"],
        heads_per_frame=args.heads_per_frame,
        rbf_weight=w,
        type_bias=tb,
        centers=data["centers"],
        gamma=data["gamma"],
        cutoff=args.cutoff,
        max_atomic_number=args.max_atomic_number,
        radial_bias_kind="rbf_type_enveloped",
        precision=args.precision,
        block_m=args.block_m,
        block_n=args.block_n,
        strict=True,
    )


def _clone_train_inputs(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        data["q"].detach().clone().requires_grad_(True),
        data["k"].detach().clone().requires_grad_(True),
        data["v"].detach().clone().requires_grad_(True),
        data["rbf_weight"].detach().clone().requires_grad_(True),
        data["type_bias"].detach().clone().requires_grad_(True),
    )


def _parity(data: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    q, k, v, w, tb = _clone_train_inputs(data)
    q_ref, k_ref, v_ref, w_ref, tb_ref = _clone_train_inputs(data)
    sparse = _sparse_radius(data, args, q, k, v, w, tb)
    dense = _dense_radius(data, args, q_ref, k_ref, v_ref, w_ref, tb_ref)
    grad = torch.randn_like(sparse)
    sparse.backward(grad)
    dense.backward(grad)
    result: dict[str, Any] = {
        "radius_sparse_torch_vs_dense_torch": {
            "forward_max_abs": _max_abs(sparse, dense),
            "dq_max_abs": _max_abs(q.grad, q_ref.grad),
            "dk_max_abs": _max_abs(k.grad, k_ref.grad),
            "dv_max_abs": _max_abs(v.grad, v_ref.grad),
            "drbf_weight_max_abs": _max_abs(w.grad, w_ref.grad),
            "dtype_bias_max_abs": _max_abs(tb.grad, tb_ref.grad),
        }
    }
    if device.type == "cuda" and TRITON_PLATONIC_ATTENTION_AVAILABLE:
        q_tri, k_tri, v_tri, w_tri, tb_tri = _clone_train_inputs(data)
        q_den, k_den, v_den, w_den, tb_den = _clone_train_inputs(data)
        triton = _dense_rbf_type_triton(data, args, q_tri, k_tri, v_tri, w_tri, tb_tri)
        dense_rbf = platonic_attention_flat_torch_reference(
            q_den,
            k_den,
            v_den,
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=data["max_seqlen"],
            pos=data["pos"],
            atom_types=data["atom_types"],
            heads_per_frame=args.heads_per_frame,
            rbf_weight=w_den,
            type_bias=tb_den,
            centers=data["centers"],
            gamma=data["gamma"],
            cutoff=args.cutoff,
            radial_bias_kind="rbf_type_enveloped",
            diag_zero=True,
        )
        triton.backward(grad)
        dense_rbf.backward(grad)
        result["dense_rbf_type_triton_vs_dense_torch"] = {
            "forward_max_abs": _max_abs(triton, dense_rbf),
            "dq_max_abs": _max_abs(q_tri.grad, q_den.grad),
            "dk_max_abs": _max_abs(k_tri.grad, k_den.grad),
            "dv_max_abs": _max_abs(v_tri.grad, v_den.grad),
            "drbf_weight_max_abs": _max_abs(w_tri.grad, w_den.grad),
            "dtype_bias_max_abs": _max_abs(tb_tri.grad, tb_den.grad),
        }
        try:
            platonic_attention_flat_triton(
                data["q"],
                data["k"],
                data["v"],
                cu_seqlens=data["cu_seqlens"],
                max_seqlen=data["max_seqlen"],
                pos=data["pos"],
                atom_types=data["atom_types"],
                heads_per_frame=args.heads_per_frame,
                rbf_weight=data["rbf_weight"],
                type_bias=data["type_bias"],
                centers=data["centers"],
                gamma=data["gamma"],
                cutoff=args.cutoff,
                max_atomic_number=args.max_atomic_number,
                radial_bias_kind="radius_rbf_type_enveloped",
                precision=args.precision,
                block_m=args.block_m,
                block_n=args.block_n,
                strict=True,
            )
            result["radius_sparse_triton"] = {"available": True}
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend availability.
            result["radius_sparse_triton"] = {"available": False, "reason": str(exc)}
    else:
        result["dense_rbf_type_triton_vs_dense_torch"] = {
            "skipped": True,
            "reason": "CUDA or Triton is unavailable",
        }
        result["radius_sparse_triton"] = {
            "available": False,
            "reason": "CUDA or Triton is unavailable",
        }
    return result


def _bench(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    data: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    backward: bool,
) -> dict[str, Any]:
    def step() -> torch.Tensor:
        if backward:
            q, k, v, w, tb = _clone_train_inputs(data)
            out = fn(q, k, v, w, tb)
            out.float().square().mean().backward()
            return out
        with torch.no_grad():
            return fn(data["q"], data["k"], data["v"], data["rbf_weight"], data["type_bias"])

    for _ in range(args.warmup):
        step()
    _sync(device)
    times: list[float] = []
    peaks: list[int] = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        step()
        _sync(device)
        times.append(1000.0 * (time.perf_counter() - start))
        if device.type == "cuda":
            peaks.append(int(torch.cuda.max_memory_allocated(device)))
    row: dict[str, Any] = {
        "name": name,
        "pass": "fwd_bwd" if backward else "fwd",
        **_stats_ms(times),
    }
    if peaks:
        row["peak_cuda_allocated_bytes"] = int(max(peaks))
    return row


def _memory_estimates(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    layout = data["layout"]
    dtype_bytes = torch.tensor([], dtype=data["q"].dtype).element_size()
    dense_score_elems = int(layout.dense_pair_count) * int(args.heads)
    live_tile_score_elems = int(layout.num_live_block_pairs) * int(args.block_m) * int(args.block_n) * int(args.heads)
    exact_block_score_elems = 0
    block_ptr = layout.block_ptr.detach().cpu().tolist()
    block_col = layout.block_col.detach().cpu().tolist()
    q_starts = layout.q_block_start.detach().cpu().tolist()
    q_ends = layout.q_block_end.detach().cpu().tolist()
    k_starts = layout.k_block_start.detach().cpu().tolist()
    k_ends = layout.k_block_end.detach().cpu().tolist()
    for q_block, (m_start, m_end) in enumerate(zip(q_starts, q_ends)):
        m_len = int(m_end) - int(m_start)
        n_len = 0
        for ptr in range(int(block_ptr[q_block]), int(block_ptr[q_block + 1])):
            k_block = int(block_col[ptr])
            n_len += int(k_ends[k_block]) - int(k_starts[k_block])
        exact_block_score_elems += m_len * n_len * int(args.heads)
    return {
        "dense_score_bytes_estimate": dense_score_elems * dtype_bytes,
        "live_tile_score_bytes_estimate": live_tile_score_elems * dtype_bytes,
        "block_sparse_reference_score_bytes_estimate": exact_block_score_elems * dtype_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark eSEN-like radius-local Platonic attention reference paths.")
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--heads-per-frame", type=int, default=1)
    parser.add_argument("--num-rbf", type=int, default=4)
    parser.add_argument("--max-atomic-number", type=int, default=30)
    parser.add_argument("--cutoff", type=float, default=1.5)
    parser.add_argument("--position-spread", type=float, default=4.0)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--sort", type=str, default="cell")
    parser.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", type=str, default="tf32x3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    data = _make_inputs(args, device)
    layout = data["layout"]
    layout_summary = {
        "num_q_blocks": layout.num_q_blocks,
        "num_k_blocks": layout.num_k_blocks,
        "num_live_block_pairs": layout.num_live_block_pairs,
        "dense_pair_count": layout.dense_pair_count,
        "radius_edge_count": layout.radius_edge_count,
        "mean_radius_degree": layout.mean_radius_degree,
        "true_edge_density": layout.true_edge_density,
        "effective_tile_density": layout.effective_tile_density,
        "max_block_row_length": layout.max_block_row_length,
        **_memory_estimates(data, args),
    }

    parity = _parity(data, args, device)
    rows = [
        _bench("dense_torch_radius_reference", lambda q, k, v, w, tb: _dense_radius(data, args, q, k, v, w, tb), data, args, device, backward=False),
        _bench("block_sparse_torch_radius_reference", lambda q, k, v, w, tb: _sparse_radius(data, args, q, k, v, w, tb), data, args, device, backward=False),
    ]
    if args.backward:
        rows.extend(
            [
                _bench("dense_torch_radius_reference", lambda q, k, v, w, tb: _dense_radius(data, args, q, k, v, w, tb), data, args, device, backward=True),
                _bench("block_sparse_torch_radius_reference", lambda q, k, v, w, tb: _sparse_radius(data, args, q, k, v, w, tb), data, args, device, backward=True),
            ]
        )
    if device.type == "cuda" and TRITON_PLATONIC_ATTENTION_AVAILABLE:
        rows.append(
            _bench(
                "dense_triton_rbf_type_reference_semantics",
                lambda q, k, v, w, tb: _dense_rbf_type_triton(data, args, q, k, v, w, tb),
                data,
                args,
                device,
                backward=False,
            )
        )
        if args.backward:
            rows.append(
                _bench(
                    "dense_triton_rbf_type_reference_semantics",
                    lambda q, k, v, w, tb: _dense_rbf_type_triton(data, args, q, k, v, w, tb),
                    data,
                    args,
                    device,
                    backward=True,
                )
            )

    result = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "triton_platonic_attention_available": bool(TRITON_PLATONIC_ATTENTION_AVAILABLE),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "layout": layout_summary,
        "parity": parity,
        "benchmarks": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
