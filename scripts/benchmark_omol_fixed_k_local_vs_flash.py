#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
from matterformer.geometry import build_triton_nonperiodic_knn_geometry_cache
from matterformer.geometry.cache import FlatGeometryCache, flatten_padded_geometry_cache
from matterformer.models.platonic import ESENEnvelopedRBFTypeFixedKBias, prepare_esen_fixed_k_local_attention_features
from matterformer.models.platonic.local_attention import fixed_k_local_attention_torch_reference
from matterformer.models.platonic.local_attention_triton import (
    TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE,
    fixed_k_local_attention_triton,
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


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    return str(dtype).replace("torch.", "")


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), torch.tensor(float(q), device=values.device)).item())


def _load_omol_padded_batch(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dataset = FairChemOMolDataset(args.data_path, keep_in_memory=False)
    try:
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
    finally:
        dataset.close()

    coords = batch.coords.to(device=device, dtype=torch.float32).contiguous()
    pad_mask = batch.pad_mask.to(device=device, dtype=torch.bool).contiguous()
    atom_types_padded = batch.atomic_numbers.to(device=device, dtype=torch.long).contiguous()
    lengths = batch.num_atoms.to(device=device, dtype=torch.long).contiguous()
    cu = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(lengths.to(dtype=torch.int32), dim=0)
    valid = ~pad_mask
    batch_index = torch.repeat_interleave(torch.arange(lengths.numel(), device=device, dtype=torch.long), lengths)
    return {
        "batch_indices": [int(idx) for idx in batch_indices],
        "coords": coords,
        "pad_mask": pad_mask,
        "atom_types_padded": atom_types_padded,
        "atom_types": atom_types_padded[valid].contiguous(),
        "lengths": lengths,
        "cu_seqlens": cu,
        "max_seqlen": int(lengths.max().item()) if lengths.numel() else 0,
        "valid": valid,
        "batch_index": batch_index,
    }


def _make_qkv(
    args: argparse.Namespace,
    total_atoms: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + 9001)
    shape = (int(total_atoms), int(args.heads), int(args.head_dim))
    q = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
    k = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
    v = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
    return q.to(dtype=dtype), k.to(dtype=dtype), v.to(dtype=dtype)


def _make_bias(args: argparse.Namespace, *, radius: float, device: torch.device) -> ESENEnvelopedRBFTypeFixedKBias:
    generator = torch.Generator(device=device).manual_seed(args.seed + 4242 + int(round(radius * 100)))
    centers = torch.linspace(0.0, float(radius), int(args.num_rbf), device=device, dtype=torch.float32)
    delta = float(radius) / max(int(args.num_rbf) - 1, 1)
    gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), device=device, dtype=torch.float32)
    rbf_weight = args.bias_scale * torch.randn(
        (int(args.heads_per_frame), int(args.num_rbf)),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    type_bias = args.bias_scale * torch.randn(
        (int(args.max_atomic_number) + 1, int(args.max_atomic_number) + 1, int(args.heads_per_frame)),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    return ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=float(radius),
        heads_per_frame=int(args.heads_per_frame),
        diag_zero=True,
        envelope_in_score=True,
        trainable=True,
    ).to(device)


def _build_flat_geometry(
    args: argparse.Namespace,
    batch: dict[str, Any],
    *,
    radius: float,
    k_neighbors: int,
) -> tuple[FlatGeometryCache, dict[str, Any]]:
    device = batch["coords"].device
    for _ in range(max(0, int(args.geometry_warmup))):
        geom_warmup = build_triton_nonperiodic_knn_geometry_cache(
            batch["coords"],
            pad_mask=batch["pad_mask"],
            k_neighbors=int(k_neighbors),
            rbf_dim=int(args.num_rbf),
            cutoff=float(radius),
            seq_len=int(batch["coords"].shape[1]),
            strict=bool(args.strict_geometry),
            include_self=True,
            self_as_first_neighbor=True,
            mask_by_cutoff=True,
        )
        _ = flatten_padded_geometry_cache(
            geom_warmup,
            valid=batch["valid"],
            batch_index=batch["batch_index"],
            cu_seqlens=batch["cu_seqlens"],
        )
    _sync(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = int(torch.cuda.memory_allocated(device))
    else:
        before = 0
    start = time.perf_counter()
    geom = build_triton_nonperiodic_knn_geometry_cache(
        batch["coords"],
        pad_mask=batch["pad_mask"],
        k_neighbors=int(k_neighbors),
        rbf_dim=int(args.num_rbf),
        cutoff=float(radius),
        seq_len=int(batch["coords"].shape[1]),
        strict=bool(args.strict_geometry),
        include_self=True,
        self_as_first_neighbor=True,
        mask_by_cutoff=True,
    )
    flat = flatten_padded_geometry_cache(
        geom,
        valid=batch["valid"],
        batch_index=batch["batch_index"],
        cu_seqlens=batch["cu_seqlens"],
    )
    _sync(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    peak_delta = 0
    if device.type == "cuda":
        peak_delta = max(0, int(torch.cuda.max_memory_allocated(device)) - before)
    counts = flat.neighbor_mask.sum(dim=-1).float()
    stats = {
        "radius": float(radius),
        "k_neighbors": int(k_neighbors),
        "geometry_build_and_flatten_ms": float(elapsed_ms),
        "valid_slots_mean": float(counts.mean().item()) if counts.numel() else 0.0,
        "valid_slots_median": float(counts.median().item()) if counts.numel() else 0.0,
        "valid_slots_p90": _percentile(counts, 0.90),
        "valid_slots_p99": _percentile(counts, 0.99),
        "valid_slots_max": int(counts.max().item()) if counts.numel() else 0,
    }
    if device.type == "cuda":
        stats["geometry_peak_cuda_delta_bytes"] = int(peak_delta)
    return flat, stats


def _prepare_esen_features(
    args: argparse.Namespace,
    batch: dict[str, Any],
    flat_geom: FlatGeometryCache,
    bias: ESENEnvelopedRBFTypeFixedKBias,
    *,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = int(torch.cuda.memory_allocated(device))
    else:
        before = 0
    _sync(device)
    start = time.perf_counter()
    features = prepare_esen_fixed_k_local_attention_features(
        neighbor_idx=flat_geom.neighbor_idx,
        neighbor_mask=flat_geom.neighbor_mask,
        dist=flat_geom.dist,
        centers=bias.centers,
        gamma=bias.gamma,
        cutoff=float(bias.cutoff),
        heads_per_frame=int(bias.heads_per_frame),
        atom_types=batch["atom_types"],
        max_atomic_number=int(args.max_atomic_number),
        diag_zero=bool(bias.diag_zero),
        envelope_in_score=bool(bias.envelope_in_score),
    )
    _sync(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    stats: dict[str, Any] = {
        "esen_feature_prepare_ms": float(elapsed_ms),
        "esen_feature_rho_env_bytes": int(features.rho_env.numel() * features.rho_env.element_size()),
        "esen_feature_total_bytes": int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    features.local_mask,
                    features.env_bias,
                    features.log_env,
                    features.rho_env,
                    features.type_base,
                )
            )
        ),
    }
    if device.type == "cuda":
        stats["esen_feature_peak_cuda_delta_bytes"] = max(0, int(torch.cuda.max_memory_allocated(device)) - before)
    return features, stats


def _flash_global(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash_attn is not importable")
    # FlashAttention varlen does not run fp32 attention; keep the dense-global
    # baseline on bf16 compute and cast back to the caller's q dtype.
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
    ).to(q.dtype)


def _bench(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            fn()
    _sync(device)
    times: list[float] = []
    peaks: list[int] = []
    peak_deltas: list[int] = []
    with torch.no_grad():
        for _ in range(max(1, repeat)):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                before = int(torch.cuda.memory_allocated(device))
            else:
                before = 0
            start = time.perf_counter()
            out = fn()
            _sync(device)
            times.append(1000.0 * (time.perf_counter() - start))
            if device.type == "cuda":
                peak = int(torch.cuda.max_memory_allocated(device))
                peaks.append(peak)
                peak_deltas.append(max(0, peak - before))
            del out
    row: dict[str, Any] = {"name": name, **_summary_ms(times)}
    if peaks:
        row["peak_cuda_allocated_bytes"] = int(max(peaks))
        row["peak_cuda_delta_bytes"] = int(max(peak_deltas))
    return row


def _zero_grads(*values: torch.Tensor | torch.nn.Module | None) -> None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value.grad = None
        elif isinstance(value, torch.nn.Module):
            value.zero_grad(set_to_none=True)


def _bench_fwd_bwd(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    grad_seed: torch.Tensor,
    grad_owner: torch.nn.Module | None = None,
) -> dict[str, Any]:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    for _ in range(max(0, warmup)):
        _zero_grads(q, k, v, grad_owner)
        out = fn(q, k, v)
        (out * grad_seed.to(dtype=out.dtype)).sum().backward()
    _zero_grads(q, k, v, grad_owner)
    _sync(device)

    times: list[float] = []
    peaks: list[int] = []
    peak_deltas: list[int] = []
    for _ in range(max(1, repeat)):
        _zero_grads(q, k, v, grad_owner)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = int(torch.cuda.memory_allocated(device))
        else:
            before = 0
        start = time.perf_counter()
        out = fn(q, k, v)
        (out * grad_seed.to(dtype=out.dtype)).sum().backward()
        _sync(device)
        times.append(1000.0 * (time.perf_counter() - start))
        if device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(device))
            peaks.append(peak)
            peak_deltas.append(max(0, peak - before))
        del out
    _zero_grads(q, k, v, grad_owner)
    row: dict[str, Any] = {"name": name, **_summary_ms(times)}
    if peaks:
        row["peak_cuda_allocated_bytes"] = int(max(peaks))
        row["peak_cuda_delta_bytes"] = int(max(peak_deltas))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OMol fixed-K eSEN local attention vs global FlashAttention.")
    parser.add_argument("--data-path", type=str, default="/home/ebekker/data/omol/open_mol/train_4M")
    parser.add_argument("--max-batch-size", type=int, default=999999)
    parser.add_argument("--max-atoms", type=int, default=12000)
    parser.add_argument("--max-edges", type=int, default=2400000)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batching-mode", type=str, default="random")
    parser.add_argument("--bucket-window-size", type=int, default=4096)
    parser.add_argument("--bucket-shuffle-groups", type=int, default=8)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--radii", type=float, nargs="+", default=[4.0, 5.0, 6.0])
    parser.add_argument("--k-values", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--heads-per-frame", type=int, default=1)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--bias-scale", type=float, default=0.01)
    parser.add_argument("--precision", type=str, default="tf32x3")
    parser.add_argument(
        "--local-precisions",
        type=str,
        nargs="+",
        default=None,
        help="Fixed-K Triton precision modes to benchmark. Defaults to --precision.",
    )
    parser.add_argument(
        "--qkv-dtypes",
        type=str,
        nargs="+",
        default=["float32"],
        help="q/k/v storage dtypes to benchmark, e.g. float32 bfloat16.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backward-warmup", type=int, default=None)
    parser.add_argument("--backward-repeat", type=int, default=None)
    parser.add_argument("--geometry-warmup", type=int, default=1)
    parser.add_argument("--torch-repeat", type=int, default=3)
    parser.add_argument("--torch-warmup", type=int, default=1)
    parser.add_argument("--skip-torch-reference", action="store_true")
    parser.add_argument("--check-parity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
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
    if device.type == "cuda" and not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE:
        raise SystemExit("fixed-K local Triton attention is unavailable")

    batch = _load_omol_padded_batch(args, device)
    lengths = batch["lengths"].detach().cpu()
    local_precisions = list(args.local_precisions) if args.local_precisions is not None else [str(args.precision)]
    qkv_dtypes = [_dtype_from_name(name) for name in args.qkv_dtypes]
    results: dict[str, Any] = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "flash_attn_available": flash_attn_varlen_func is not None,
        "triton_fixed_k_local_attention_available": bool(TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "batch": {
            "num_graphs": int(lengths.numel()),
            "num_atoms": int(lengths.sum().item()) if lengths.numel() else 0,
            "max_atoms_per_graph": int(lengths.max().item()) if lengths.numel() else 0,
            "mean_atoms_per_graph": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
            "median_atoms_per_graph": float(lengths.float().median().item()) if lengths.numel() else 0.0,
            "sum_n2": int((lengths.to(torch.int64) ** 2).sum().item()) if lengths.numel() else 0,
            "batch_indices_first_10": batch["batch_indices"][:10],
        },
        "benchmarks": [],
        "local_configs": [],
        "parity": [],
        "notes": {
            "pass": "forward and optional forward+backward attention-only timings",
            "flash": "global per-molecule FlashAttention varlen with q/k/v cast to bf16, matching existing benchmark convention",
            "local": "fixed-K eSEN local attention over flattened OMol atoms; geometry build is reported separately",
            "local_backward": "fixed-K Triton backward includes q/k/v gradients and trainable eSEN rbf/type-bias gradients",
        },
    }
    backward_warmup = int(args.backward_warmup if args.backward_warmup is not None else args.warmup)
    backward_repeat = int(args.backward_repeat if args.backward_repeat is not None else args.repeat)

    local_configs: list[dict[str, Any]] = []
    for radius in args.radii:
        bias = _make_bias(args, radius=float(radius), device=device)
        for k_neighbors in args.k_values:
            flat_geom, geom_stats = _build_flat_geometry(args, batch, radius=float(radius), k_neighbors=int(k_neighbors))
            esen_features, feature_stats = _prepare_esen_features(
                args,
                batch,
                flat_geom,
                bias,
                device=device,
            )
            geom_stats.update(feature_stats)
            results["local_configs"].append(geom_stats)
            local_configs.append(
                {
                    "radius": float(radius),
                    "k_neighbors": int(k_neighbors),
                    "flat_geom": flat_geom,
                    "bias": bias,
                    "esen_features": esen_features,
                }
            )

    for qkv_dtype in qkv_dtypes:
        qkv_dtype_name = _dtype_name(qkv_dtype)
        q, k, v = _make_qkv(args, int(lengths.sum().item()), device, dtype=qkv_dtype)
        grad_seed = torch.randn_like(q)
        results["benchmarks"].append(
            _bench(
                f"global_flash_attention_bf16_compute_fwd_qkv_{qkv_dtype_name}",
                lambda q=q, k=k, v=v: _flash_global(
                    q,
                    k,
                    v,
                    cu_seqlens=batch["cu_seqlens"],
                    max_seqlen=batch["max_seqlen"],
                ),
                device=device,
                warmup=int(args.warmup),
                repeat=int(args.repeat),
            )
        )
        if bool(args.backward):
            results["benchmarks"].append(
                _bench_fwd_bwd(
                    f"global_flash_attention_bf16_compute_fwd_bwd_qkv_{qkv_dtype_name}",
                    lambda q, k, v: _flash_global(
                        q,
                        k,
                        v,
                        cu_seqlens=batch["cu_seqlens"],
                        max_seqlen=batch["max_seqlen"],
                    ),
                    q,
                    k,
                    v,
                    device=device,
                    warmup=backward_warmup,
                    repeat=backward_repeat,
                    grad_seed=grad_seed,
                )
            )

        for local_config in local_configs:
            radius = float(local_config["radius"])
            k_neighbors = int(local_config["k_neighbors"])
            flat_geom = local_config["flat_geom"]
            bias = local_config["bias"]
            esen_features = local_config["esen_features"]
            local_inputs = {
                "neighbor_idx": flat_geom.neighbor_idx,
                "neighbor_mask": flat_geom.neighbor_mask,
                "dist": flat_geom.dist,
                "rbf": flat_geom.rbf,
                "atom_types": batch["atom_types"],
                "bias": bias,
            }
            if not bool(args.skip_torch_reference):
                results["benchmarks"].append(
                    _bench(
                        f"fixed_k_esen_torch_reference_fwd_qkv_{qkv_dtype_name}_r{radius:g}_k{k_neighbors}",
                        lambda local_inputs=local_inputs: fixed_k_local_attention_torch_reference(q, k, v, **local_inputs),
                        device=device,
                        warmup=int(args.torch_warmup),
                        repeat=int(args.torch_repeat),
                    )
                )

            for precision in local_precisions:
                if bool(args.check_parity):
                    with torch.no_grad():
                        if str(precision) == "bf16_flash_compat":
                            q_ref = q.to(torch.bfloat16).float()
                            k_ref = k.to(torch.bfloat16).float()
                            v_ref = v.to(torch.bfloat16).float()
                            expected_dtype = torch.bfloat16
                        elif qkv_dtype is torch.bfloat16:
                            q_ref = q.float()
                            k_ref = k.float()
                            v_ref = v.float()
                            expected_dtype = torch.bfloat16
                        else:
                            q_ref, k_ref, v_ref = q, k, v
                            expected_dtype = q.dtype
                        ref = fixed_k_local_attention_torch_reference(q_ref, k_ref, v_ref, **local_inputs).to(expected_dtype).to(q.dtype)
                        tri = fixed_k_local_attention_triton(
                            q,
                            k,
                            v,
                            neighbor_idx=flat_geom.neighbor_idx,
                            neighbor_mask=flat_geom.neighbor_mask,
                            dist=flat_geom.dist,
                            atom_types=batch["atom_types"],
                            bias=bias,
                            esen_features=esen_features,
                            precision=precision,
                            strict=True,
                        )
                        diff = (tri - ref).abs()
                    results["parity"].append(
                        {
                            "qkv_dtype": qkv_dtype_name,
                            "precision": str(precision),
                            "radius": float(radius),
                            "k_neighbors": int(k_neighbors),
                            "max_abs_diff": float(diff.max().item()),
                            "mean_abs_diff": float(diff.mean().item()),
                        }
                    )
                    del ref, tri, diff
                    _sync(device)
                results["benchmarks"].append(
                    _bench(
                        f"fixed_k_esen_triton_fwd_qkv_{qkv_dtype_name}_{precision}_r{radius:g}_k{k_neighbors}",
                        lambda flat_geom=flat_geom, bias=bias, esen_features=esen_features, precision=precision, q=q, k=k, v=v: fixed_k_local_attention_triton(
                            q,
                            k,
                            v,
                            neighbor_idx=flat_geom.neighbor_idx,
                            neighbor_mask=flat_geom.neighbor_mask,
                            dist=flat_geom.dist,
                            atom_types=batch["atom_types"],
                            bias=bias,
                            esen_features=esen_features,
                            precision=precision,
                            strict=True,
                        ),
                        device=device,
                        warmup=int(args.warmup),
                        repeat=int(args.repeat),
                    )
                )
                if bool(args.backward):
                    results["benchmarks"].append(
                        _bench_fwd_bwd(
                            f"fixed_k_esen_triton_fwd_bwd_qkv_{qkv_dtype_name}_{precision}_r{radius:g}_k{k_neighbors}",
                            lambda q, k, v, flat_geom=flat_geom, bias=bias, esen_features=esen_features, precision=precision: fixed_k_local_attention_triton(
                                q,
                                k,
                                v,
                                neighbor_idx=flat_geom.neighbor_idx,
                                neighbor_mask=flat_geom.neighbor_mask,
                                dist=flat_geom.dist,
                                atom_types=batch["atom_types"],
                                bias=bias,
                                esen_features=esen_features,
                                precision=precision,
                                strict=True,
                            ),
                            q,
                            k,
                            v,
                            device=device,
                            warmup=backward_warmup,
                            repeat=backward_repeat,
                            grad_seed=grad_seed,
                            grad_owner=bias,
                        )
                    )

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
