#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
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

from matterformer.geometry import TRITON_NONPERIODIC_KNN_AVAILABLE, build_triton_nonperiodic_knn_geometry_cache


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), torch.tensor(float(q), device=values.device)).item())


def _summary_ms(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p90_ms": float(values_sorted[min(len(values_sorted) - 1, int(0.90 * (len(values_sorted) - 1)))]),
    }


def _load_omol_padded_batch(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from matterformer.data.omol import FairChemOMolDataset, OMolDynamicBatchSampler, collate_omol

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
    lengths = batch.num_atoms.to(device=device, dtype=torch.long).contiguous()
    return {
        "source": "omol",
        "batch_indices": [int(idx) for idx in batch_indices],
        "coords": coords,
        "pad_mask": pad_mask,
        "lengths": lengths,
    }


def _synthetic_padded_batch(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    lengths = torch.randint(
        low=max(2, int(args.synthetic_min_atoms)),
        high=max(3, int(args.synthetic_max_atoms) + 1),
        size=(int(args.synthetic_graphs),),
        generator=generator,
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())
    coords = torch.zeros((lengths.numel(), max_len, 3), dtype=torch.float32)
    pad_mask = torch.ones((lengths.numel(), max_len), dtype=torch.bool)
    for graph_idx, length in enumerate(lengths.tolist()):
        # A compact random cloud is enough to exercise neighbor masking and truncation stats.
        coords[graph_idx, :length] = torch.randn((length, 3), generator=generator) * float(args.synthetic_scale)
        pad_mask[graph_idx, :length] = False
    return {
        "source": "synthetic",
        "batch_indices": [],
        "coords": coords.to(device=device).contiguous(),
        "pad_mask": pad_mask.to(device=device).contiguous(),
        "lengths": lengths.to(device=device),
    }


def _exact_radius_degrees(
    coords: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    radius: float,
    include_self: bool,
) -> torch.Tensor:
    valid = ~pad_mask.bool()
    degrees: list[torch.Tensor] = []
    eye_cache: dict[int, torch.Tensor] = {}
    for graph_idx in range(coords.shape[0]):
        graph_valid = valid[graph_idx]
        count = int(graph_valid.sum().item())
        if count == 0:
            continue
        pos = coords[graph_idx, graph_valid].float()
        dist = torch.cdist(pos, pos)
        mask = dist < float(radius)
        if not include_self:
            eye = eye_cache.get(count)
            if eye is None:
                eye = torch.eye(count, device=coords.device, dtype=torch.bool)
                eye_cache[count] = eye
            mask = mask & ~eye
        degrees.append(mask.sum(dim=1).to(dtype=torch.float32))
    return torch.cat(degrees, dim=0) if degrees else torch.zeros(0, device=coords.device)


def _geometry_stats(
    coords: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    radius: float,
    k_neighbors: int,
    rbf_dim: int,
    seq_len: int,
    include_self: bool,
    self_as_first_neighbor: bool,
    strict: bool,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = coords.device
    cache = None
    for _ in range(max(0, warmup)):
        cache = build_triton_nonperiodic_knn_geometry_cache(
            coords,
            pad_mask=pad_mask,
            k_neighbors=k_neighbors,
            rbf_dim=rbf_dim,
            cutoff=radius,
            seq_len=seq_len,
            strict=strict,
            include_self=include_self,
            self_as_first_neighbor=self_as_first_neighbor,
            mask_by_cutoff=True,
        )
    _sync(device)

    times: list[float] = []
    peak_deltas: list[int] = []
    for _ in range(max(1, repeat)):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = int(torch.cuda.memory_allocated(device))
        else:
            before = 0
        start = time.perf_counter()
        cache = build_triton_nonperiodic_knn_geometry_cache(
            coords,
            pad_mask=pad_mask,
            k_neighbors=k_neighbors,
            rbf_dim=rbf_dim,
            cutoff=radius,
            seq_len=seq_len,
            strict=strict,
            include_self=include_self,
            self_as_first_neighbor=self_as_first_neighbor,
            mask_by_cutoff=True,
        )
        _sync(device)
        times.append(1000.0 * (time.perf_counter() - start))
        if device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(device))
            peak_deltas.append(max(0, peak - before))

    assert cache is not None
    valid_atoms = ~pad_mask.bool()
    valid_counts = cache.neighbor_mask[valid_atoms].sum(dim=-1).float()
    exact_degrees = _exact_radius_degrees(coords, pad_mask, radius=radius, include_self=include_self)
    if include_self and self_as_first_neighbor:
        exact_capacity = max(int(k_neighbors) - 1, 0)
        exact_excluding_self = torch.clamp(exact_degrees - 1.0, min=0.0)
        dropped = torch.clamp(exact_excluding_self - float(exact_capacity), min=0.0)
    else:
        dropped = torch.clamp(exact_degrees - float(k_neighbors), min=0.0)

    row: dict[str, Any] = {
        "radius": float(radius),
        "k_neighbors": int(k_neighbors),
        "include_self": bool(include_self),
        "self_as_first_neighbor": bool(self_as_first_neighbor),
        "valid_slots_mean": float(valid_counts.mean().item()) if valid_counts.numel() else 0.0,
        "valid_slots_median": float(valid_counts.median().item()) if valid_counts.numel() else 0.0,
        "valid_slots_p90": _percentile(valid_counts, 0.90),
        "valid_slots_p99": _percentile(valid_counts, 0.99),
        "valid_slots_max": int(valid_counts.max().item()) if valid_counts.numel() else 0,
        "exact_radius_degree_mean": float(exact_degrees.mean().item()) if exact_degrees.numel() else 0.0,
        "exact_radius_degree_p90": _percentile(exact_degrees, 0.90),
        "exact_radius_degree_p99": _percentile(exact_degrees, 0.99),
        "exact_radius_degree_max": int(exact_degrees.max().item()) if exact_degrees.numel() else 0,
        "truncated_rows": int((dropped > 0).sum().item()) if dropped.numel() else 0,
        "truncated_row_fraction": float((dropped > 0).float().mean().item()) if dropped.numel() else 0.0,
        "dropped_neighbor_count": int(dropped.sum().item()) if dropped.numel() else 0,
        "dropped_neighbor_fraction_of_exact_edges": (
            float(dropped.sum().item() / max(float(exact_degrees.sum().item()), 1.0)) if exact_degrees.numel() else 0.0
        ),
        "build_time": _summary_ms(times),
    }
    if peak_deltas:
        row["peak_cuda_delta_bytes"] = int(max(peak_deltas))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed-K nonperiodic geometry cache and radius truncation stats.")
    parser.add_argument("--data-path", type=str, default="/home/ebekker/data/omol/open_mol/train_4M")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-graphs", type=int, default=16)
    parser.add_argument("--synthetic-min-atoms", type=int, default=16)
    parser.add_argument("--synthetic-max-atoms", type=int, default=96)
    parser.add_argument("--synthetic-scale", type=float, default=2.0)
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
    parser.add_argument("--rbf-dim", type=int, default=16)
    parser.add_argument("--include-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-as-first-neighbor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.self_as_first_neighbor and not args.include_self:
        raise ValueError("--self-as-first-neighbor requires --include-self")
    if max(args.k_values) > 64:
        raise ValueError("triton_nonperiodic kNN supports K <= 64")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    batch = _synthetic_padded_batch(args, device) if args.synthetic else _load_omol_padded_batch(args, device)
    coords = batch["coords"]
    pad_mask = batch["pad_mask"]
    lengths = batch["lengths"].detach().cpu()
    rows = []
    for radius in args.radii:
        for k_neighbors in args.k_values:
            rows.append(
                _geometry_stats(
                    coords,
                    pad_mask,
                    radius=float(radius),
                    k_neighbors=int(k_neighbors),
                    rbf_dim=int(args.rbf_dim),
                    seq_len=int(coords.shape[1]),
                    include_self=bool(args.include_self),
                    self_as_first_neighbor=bool(args.self_as_first_neighbor),
                    strict=bool(args.strict),
                    warmup=int(args.warmup),
                    repeat=int(args.repeat),
                )
            )

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "triton_nonperiodic_knn_available": bool(TRITON_NONPERIODIC_KNN_AVAILABLE),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "batch": {
            "source": batch["source"],
            "num_graphs": int(lengths.numel()),
            "num_atoms": int(lengths.sum().item()) if lengths.numel() else 0,
            "max_atoms_per_graph": int(lengths.max().item()) if lengths.numel() else 0,
            "mean_atoms_per_graph": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
            "median_atoms_per_graph": float(lengths.float().median().item()) if lengths.numel() else 0.0,
            "sum_n2": int((lengths.to(torch.int64) ** 2).sum().item()) if lengths.numel() else 0,
            "batch_indices_first_10": batch["batch_indices"][:10],
        },
        "rows": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
