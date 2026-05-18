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

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
except Exception:
    flash_attn_varlen_func = None

from matterformer.data.omol import FairChemOMolDataset, OMolDynamicBatchSampler, collate_omol
from matterformer.models.platonic.radius_sparse import RadiusBlockSparseLayout, build_radius_block_sparse_layout
from matterformer.models.platonic.triton_attention import (
    TRITON_PLATONIC_ATTENTION_AVAILABLE,
    platonic_radius_sparse_attention_flat_triton,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary_ms(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p10_ms": float(values_sorted[max(0, int(0.10 * (len(values_sorted) - 1)))]),
        "p90_ms": float(values_sorted[min(len(values_sorted) - 1, int(0.90 * (len(values_sorted) - 1)))]),
    }


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), torch.tensor(float(q), device=values.device)).item())


def _load_omol_flat_batch(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dataset = FairChemOMolDataset(args.data_path, keep_in_memory=False)
    sampler = OMolDynamicBatchSampler(
        dataset,
        max_batch_size=args.max_batch_size,
        max_atoms=args.max_atoms,
        max_edges=args.max_edges,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        batching_mode=args.batching_mode,
        bucket_window_size=args.bucket_window_size,
        bucket_shuffle_groups=args.bucket_shuffle_groups,
    )
    iterator = iter(sampler)
    batch_indices: list[int] | None = None
    for _ in range(max(0, args.batch_index) + 1):
        batch_indices = next(iterator)
    if batch_indices is None:
        raise RuntimeError("failed to draw an OMol batch")
    batch = collate_omol([dataset[idx] for idx in batch_indices])
    dataset.close()

    valid = ~batch.pad_mask
    counts = batch.num_atoms.to(device=device, dtype=torch.int32)
    cu = torch.zeros(counts.numel() + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(counts, dim=0)
    pos = batch.coords[valid].to(device=device, dtype=torch.float32).contiguous()
    atom_types = batch.atomic_numbers[valid].to(device=device, dtype=torch.long).contiguous()
    max_seqlen = int(counts.max().item()) if counts.numel() else 0
    return {
        "batch_indices": [int(idx) for idx in batch_indices],
        "counts": counts,
        "cu_seqlens": cu,
        "max_seqlen": max_seqlen,
        "pos": pos,
        "atom_types": atom_types,
    }


def _exact_degree_stats(pos: torch.Tensor, cu: torch.Tensor, radius: float) -> dict[str, float | int]:
    degrees: list[torch.Tensor] = []
    starts = cu[:-1].detach().cpu().tolist()
    ends = cu[1:].detach().cpu().tolist()
    total_edges_excluding_self = 0
    for start, end in zip(starts, ends):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        pos_seg = pos[start_i:end_i].float()
        dist = torch.cdist(pos_seg, pos_seg)
        mask = dist < float(radius)
        mask.fill_diagonal_(False)
        degree = mask.sum(dim=1).float()
        total_edges_excluding_self += int(degree.sum().item())
        degrees.append(degree.detach().cpu())
    degree_all = torch.cat(degrees, dim=0) if degrees else torch.zeros(0)
    if degree_all.numel() == 0:
        return {
            "mean_neighbors_excluding_self": 0.0,
            "median_neighbors_excluding_self": 0.0,
            "p90_neighbors_excluding_self": 0.0,
            "p99_neighbors_excluding_self": 0.0,
            "max_neighbors_excluding_self": 0,
            "radius_edges_excluding_self": 0,
        }
    return {
        "mean_neighbors_excluding_self": float(degree_all.mean().item()),
        "median_neighbors_excluding_self": float(degree_all.median().item()),
        "p90_neighbors_excluding_self": _percentile(degree_all, 0.90),
        "p99_neighbors_excluding_self": _percentile(degree_all, 0.99),
        "max_neighbors_excluding_self": int(degree_all.max().item()),
        "radius_edges_excluding_self": int(total_edges_excluding_self),
    }


def _layout_stats(
    pos: torch.Tensor,
    cu: torch.Tensor,
    *,
    radius: float,
    block_m: int,
    block_n: int,
    sort: str,
    include_self: bool,
) -> tuple[RadiusBlockSparseLayout, dict[str, Any]]:
    start = time.perf_counter()
    layout = build_radius_block_sparse_layout(
        pos,
        cu,
        cutoff=radius,
        block_m=block_m,
        block_n=block_n,
        sort=sort,
        include_self=include_self,
    )
    build_ms = 1000.0 * (time.perf_counter() - start)
    exact = _exact_degree_stats(pos, cu, radius)
    stats = {
        "radius": float(radius),
        "layout_build_ms": float(build_ms),
        "num_live_block_pairs": layout.num_live_block_pairs,
        "max_block_row_length": layout.max_block_row_length,
        "effective_tile_density": layout.effective_tile_density,
        "true_edge_density": layout.true_edge_density,
        "mean_radius_degree_including_self_when_enabled": layout.mean_radius_degree,
        "dense_pair_count": layout.dense_pair_count,
        "radius_edge_count_layout": layout.radius_edge_count,
        **exact,
    }
    return layout, stats


def _make_attention_inputs(args: argparse.Namespace, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + 1729)
    tokens = int(batch["pos"].shape[0])
    q = torch.randn((tokens, args.heads, args.head_dim), device=device, generator=generator, dtype=torch.float32)
    k = torch.randn((tokens, args.heads, args.head_dim), device=device, generator=generator, dtype=torch.float32)
    v = torch.randn((tokens, args.heads, args.head_dim), device=device, generator=generator, dtype=torch.float32)
    centers = torch.linspace(0.0, args.radius, args.num_rbf, device=device, dtype=torch.float32)
    delta = float(args.radius) / max(int(args.num_rbf) - 1, 1)
    gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), device=device, dtype=torch.float32)
    rbf_weight = 0.01 * torch.randn(
        (args.heads_per_frame, args.num_rbf),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    type_bias = 0.01 * torch.randn(
        (args.max_atomic_number + 1, args.max_atomic_number + 1, args.heads_per_frame),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "centers": centers,
        "gamma": gamma,
        "rbf_weight": rbf_weight,
        "type_bias": type_bias,
    }


def _flash_global(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash_attn is not importable in this environment")
    orig_dtype = q.dtype
    return flash_attn_varlen_func(
        q.contiguous().to(torch.bfloat16),
        k.contiguous().to(torch.bfloat16),
        v.contiguous().to(torch.bfloat16),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(max_seqlen),
        max_seqlen_k=int(max_seqlen),
        dropout_p=0.0,
        causal=False,
    ).to(orig_dtype)


def _sparse_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    layout: RadiusBlockSparseLayout,
    args: argparse.Namespace,
    rbf_weight: torch.Tensor,
    type_bias: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    return platonic_radius_sparse_attention_flat_triton(
        q,
        k,
        v,
        pos=pos,
        atom_types=atom_types,
        heads_per_frame=args.heads_per_frame,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=args.radius,
        max_atomic_number=args.max_atomic_number,
        diag_zero=True,
        include_self=args.include_self,
        envelope_in_score=True,
        radius_layout=layout,
        precision=args.precision,
        strict=True,
    )


def _bench(
    name: str,
    fn: Callable[..., torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    *,
    backward: bool,
    clone_inputs: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    def step() -> torch.Tensor:
        tensors = clone_inputs()
        out = fn(**tensors)
        if backward:
            out.float().square().mean().backward()
        return out

    for _ in range(args.warmup):
        step()
    _sync(device)

    times: list[float] = []
    peaks: list[int] = []
    peak_deltas: list[int] = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = int(torch.cuda.memory_allocated(device))
        else:
            before = 0
        start = time.perf_counter()
        step()
        _sync(device)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        times.append(elapsed_ms)
        if device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(device))
            peaks.append(peak)
            peak_deltas.append(max(0, peak - before))
    row: dict[str, Any] = {
        "name": name,
        "pass": "fwd_bwd" if backward else "fwd",
        **_summary_ms(times),
    }
    if peaks:
        row["peak_cuda_allocated_bytes"] = int(max(peaks))
        row["peak_cuda_delta_bytes"] = int(max(peak_deltas))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real-OMol global FlashAttention vs radius-sparse Triton attention.")
    parser.add_argument("--data-path", type=str, default="/home/ebekker/data/omol/open_mol/train_4M")
    parser.add_argument("--max-batch-size", type=int, default=999999)
    parser.add_argument("--max-atoms", type=int, default=12000)
    parser.add_argument("--max-edges", type=int, default=2400000)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batching-mode", type=str, default="random")
    parser.add_argument("--bucket-window-size", type=int, default=4096)
    parser.add_argument("--bucket-shuffle-groups", type=int, default=8)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--radii", type=float, nargs="+", default=[3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--heads", type=int, default=60)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--heads-per-frame", type=int, default=5)
    parser.add_argument("--num-rbf", type=int, default=4)
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--sort", type=str, default="cell")
    parser.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--precision", type=str, default="tf32x3")
    parser.add_argument("--matmul-precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.heads % args.heads_per_frame != 0:
        raise ValueError("--heads must be divisible by --heads-per-frame")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda" and flash_attn_varlen_func is None:
        raise SystemExit("flash_attn is required for the global FlashAttention benchmark")
    if device.type == "cuda" and not TRITON_PLATONIC_ATTENTION_AVAILABLE:
        raise SystemExit("Matterformer Platonic Triton attention is unavailable")

    batch = _load_omol_flat_batch(args, device)
    lengths = batch["counts"].detach().cpu()
    radius_stats: list[dict[str, Any]] = []
    chosen_layout: RadiusBlockSparseLayout | None = None
    chosen_stats: dict[str, Any] | None = None
    for radius in args.radii:
        layout, stats = _layout_stats(
            batch["pos"],
            batch["cu_seqlens"],
            radius=float(radius),
            block_m=args.block_m,
            block_n=args.block_n,
            sort=args.sort,
            include_self=args.include_self,
        )
        radius_stats.append(stats)
        if abs(float(radius) - float(args.radius)) < 1.0e-8:
            chosen_layout = layout
            chosen_stats = stats
    if chosen_layout is None:
        chosen_layout, chosen_stats = _layout_stats(
            batch["pos"],
            batch["cu_seqlens"],
            radius=float(args.radius),
            block_m=args.block_m,
            block_n=args.block_n,
            sort=args.sort,
            include_self=args.include_self,
        )
        radius_stats.append(chosen_stats)
    assert chosen_stats is not None

    attn = _make_attention_inputs(args, batch, device)
    perm = chosen_layout.perm
    q_sorted = attn["q"].index_select(0, perm).contiguous()
    k_sorted = attn["k"].index_select(0, perm).contiguous()
    v_sorted = attn["v"].index_select(0, perm).contiguous()
    pos_sorted = batch["pos"].index_select(0, perm).contiguous()
    atom_sorted = batch["atom_types"].index_select(0, perm).contiguous()

    def clone_flash() -> dict[str, Any]:
        return {
            "q": attn["q"].detach().clone().requires_grad_(True),
            "k": attn["k"].detach().clone().requires_grad_(True),
            "v": attn["v"].detach().clone().requires_grad_(True),
            "cu_seqlens": batch["cu_seqlens"],
            "max_seqlen": batch["max_seqlen"],
        }

    def clone_sparse() -> dict[str, Any]:
        return {
            "q": q_sorted.detach().clone().requires_grad_(True),
            "k": k_sorted.detach().clone().requires_grad_(True),
            "v": v_sorted.detach().clone().requires_grad_(True),
            "pos": pos_sorted,
            "atom_types": atom_sorted,
            "layout": chosen_layout,
            "args": args,
            "rbf_weight": attn["rbf_weight"].detach().clone().requires_grad_(True),
            "type_bias": attn["type_bias"].detach().clone().requires_grad_(True),
            "centers": attn["centers"],
            "gamma": attn["gamma"],
        }

    rows = [
        _bench("global_flash_attention_bf16", _flash_global, args, device, backward=False, clone_inputs=clone_flash),
        _bench("radius_sparse_triton", _sparse_triton, args, device, backward=False, clone_inputs=clone_sparse),
        _bench("global_flash_attention_bf16", _flash_global, args, device, backward=True, clone_inputs=clone_flash),
        _bench("radius_sparse_triton", _sparse_triton, args, device, backward=True, clone_inputs=clone_sparse),
    ]

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "flash_attn_available": flash_attn_varlen_func is not None,
        "triton_platonic_attention_available": bool(TRITON_PLATONIC_ATTENTION_AVAILABLE),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "batch": {
            "num_graphs": int(lengths.numel()),
            "num_atoms": int(batch["pos"].shape[0]),
            "max_atoms_per_graph": int(lengths.max().item()) if lengths.numel() else 0,
            "mean_atoms_per_graph": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
            "median_atoms_per_graph": float(lengths.float().median().item()) if lengths.numel() else 0.0,
            "sum_n2": int((lengths.to(torch.int64) ** 2).sum().item()),
            "batch_indices_first_10": batch["batch_indices"][:10],
        },
        "chosen_radius": chosen_stats,
        "radius_sweep": radius_stats,
        "benchmarks": rows,
        "notes": {
            "flash_attention": "global per molecule, FlashAttention varlen, q/k/v cast to bf16 like PlatonicAttention._flash_attention_flat",
            "radius_sparse": "Triton sparse forward over radius block layout. For non-split head dimensions such as 128, backward uses the sparse Triton dq/dkv kernels; split-D dimensions still fall back to the torch block-sparse reference.",
            "position_gradients": "positions are treated as conditioning tensors here; dpos is not benchmarked",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
