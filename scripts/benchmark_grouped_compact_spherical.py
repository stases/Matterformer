#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time

import torch

from matterformer.models.triton_grouped_compact_simplicial_attention import (
    TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE,
    _expand_compact_spherical_coefficients,
    _spherical_basis_lmax2,
    triton_grouped_compact_simplicial_attention,
)


def _make_neighbors(batch: int, atoms: int, k_neighbors: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(atoms, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :] + 1
    neighbor_idx = ((base + offsets) % atoms).expand(batch, -1, -1).contiguous()
    neighbor_mask = torch.ones(batch, atoms, k_neighbors, device=device, dtype=torch.bool)
    return neighbor_idx, neighbor_mask


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _make_case(
    *,
    batch: int,
    group: int,
    heads: int,
    atoms: int,
    k_neighbors: int,
    head_dim: int,
    channels_by_l: tuple[int, int, int],
    device: torch.device,
) -> dict[str, torch.Tensor | tuple[int, int, int] | int]:
    qkv = [torch.randn(batch, group, heads, atoms, head_dim, device=device, dtype=torch.float32) for _ in range(5)]
    neighbor_idx, neighbor_mask = _make_neighbors(batch, atoms, k_neighbors, device)
    unit = torch.nn.functional.normalize(torch.randn(batch, atoms, k_neighbors, 3, device=device), dim=-1).contiguous()
    coeff_dim = sum(channels_by_l)
    angle_rank = channels_by_l[0] + 3 * channels_by_l[1] + 5 * channels_by_l[2]
    u = torch.randn(batch, heads, atoms, k_neighbors, device=device, dtype=torch.float32)
    v_bias = torch.randn(batch, heads, atoms, k_neighbors, device=device, dtype=torch.float32)
    gate = torch.randn(batch, heads, atoms, device=device, dtype=torch.float32)
    left_coeff = torch.randn(batch, heads, atoms, k_neighbors, coeff_dim, device=device, dtype=torch.float32)
    right_coeff = torch.randn(batch, heads, atoms, k_neighbors, coeff_dim, device=device, dtype=torch.float32)
    angle_gate = torch.randn(batch, heads, atoms, device=device, dtype=torch.float32)
    return {
        "qkv": qkv,
        "neighbor_idx": neighbor_idx,
        "neighbor_mask": neighbor_mask,
        "unit": unit,
        "channels_by_l": channels_by_l,
        "angle_rank": angle_rank,
        "u": u,
        "v_bias": v_bias,
        "gate": gate,
        "left_coeff": left_coeff,
        "right_coeff": right_coeff,
        "angle_gate": angle_gate,
    }


def _run_once(case: dict[str, object], *, representation: str, precision: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    q, k1, v1, k2, v2 = [_leaf(t) for t in case["qkv"]]  # type: ignore[index]
    u = _leaf(case["u"])  # type: ignore[arg-type]
    v_bias = _leaf(case["v_bias"])  # type: ignore[arg-type]
    gate = _leaf(case["gate"])  # type: ignore[arg-type]
    left_coeff = _leaf(case["left_coeff"])  # type: ignore[arg-type]
    right_coeff = _leaf(case["right_coeff"])  # type: ignore[arg-type]
    angle_gate = _leaf(case["angle_gate"])  # type: ignore[arg-type]
    leaves = [q, k1, v1, k2, v2, u, v_bias, gate, left_coeff, right_coeff, angle_gate]
    if representation == "expanded":
        basis = _spherical_basis_lmax2(case["unit"])  # type: ignore[arg-type]
        angle_left = _expand_compact_spherical_coefficients(
            left_coeff,
            basis=basis,
            channels_by_l=case["channels_by_l"],  # type: ignore[arg-type]
        )
        angle_right = _expand_compact_spherical_coefficients(
            right_coeff,
            basis=basis,
            channels_by_l=case["channels_by_l"],  # type: ignore[arg-type]
        )
        out = triton_grouped_compact_simplicial_attention(
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx=case["neighbor_idx"],  # type: ignore[arg-type]
            neighbor_mask=case["neighbor_mask"],  # type: ignore[arg-type]
            u=u,
            v_bias=v_bias,
            gate=gate,
            angle_left=angle_left,
            angle_right=angle_right,
            angle_gate=angle_gate,
            precision=precision,
            strict=True,
        )
    elif representation == "compact":
        out = triton_grouped_compact_simplicial_attention(
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx=case["neighbor_idx"],  # type: ignore[arg-type]
            neighbor_mask=case["neighbor_mask"],  # type: ignore[arg-type]
            u=u,
            v_bias=v_bias,
            gate=gate,
            unit=case["unit"],  # type: ignore[arg-type]
            angle_left_coeff=left_coeff,
            angle_right_coeff=right_coeff,
            angle_channels_by_l=case["channels_by_l"],  # type: ignore[arg-type]
            angle_rank=case["angle_rank"],  # type: ignore[arg-type]
            angle_gate=angle_gate,
            precision=precision,
            strict=True,
        )
    else:
        raise ValueError(f"unknown representation: {representation}")
    return out, leaves


def _check_parity(case: dict[str, object], *, precision: str) -> dict[str, float]:
    expanded, expanded_leaves = _run_once(case, representation="expanded", precision=precision)
    compact, compact_leaves = _run_once(case, representation="compact", precision=precision)
    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    compact.backward(grad)
    torch.cuda.synchronize()
    grad_max = 0.0
    for lhs, rhs in zip(compact_leaves, expanded_leaves):
        if lhs.grad is None or rhs.grad is None:
            raise RuntimeError("missing gradient in parity check")
        grad_max = max(grad_max, float((lhs.grad - rhs.grad).abs().max().item()))
    return {
        "out_max_abs_diff": float((compact - expanded).abs().max().item()),
        "grad_max_abs_diff": grad_max,
    }


def _benchmark(
    case: dict[str, object],
    *,
    representation: str,
    precision: str,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        out, _ = _run_once(case, representation=representation, precision=precision)
        out.square().mean().backward()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out, _ = _run_once(case, representation=representation, precision=precision)
        out.square().mean().backward()
    stop.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(stop))
    return {
        "fw_bw_ms": elapsed_ms / float(iters),
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--group", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--atoms", type=int, default=160)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=28)
    parser.add_argument("--channels-by-l", type=str, default="2,3,1")
    parser.add_argument("--precision", type=str, default="tf32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE:
        raise RuntimeError("Grouped compact Triton is unavailable")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    channels_by_l = tuple(int(part) for part in args.channels_by_l.split(","))
    if len(channels_by_l) != 3:
        raise ValueError("--channels-by-l must have exactly three comma-separated values")
    case = _make_case(
        batch=args.batch,
        group=args.group,
        heads=args.heads,
        atoms=args.atoms,
        k_neighbors=args.k_neighbors,
        head_dim=args.head_dim,
        channels_by_l=channels_by_l,  # type: ignore[arg-type]
        device=device,
    )
    parity = _check_parity(case, precision="ieee_fp32")
    expanded = _benchmark(case, representation="expanded", precision=args.precision, warmup=args.warmup, iters=args.iters)
    compact = _benchmark(case, representation="compact", precision=args.precision, warmup=args.warmup, iters=args.iters)
    print(
        json.dumps(
            {
                "shape": {
                    "B": args.batch,
                    "G": args.group,
                    "H": args.heads,
                    "N": args.atoms,
                    "K": args.k_neighbors,
                    "D": args.head_dim,
                    "channels_by_l": channels_by_l,
                },
                "parity": parity,
                "expanded": expanded,
                "compact": compact,
                "speedup_expanded_over_compact": expanded["fw_bw_ms"] / compact["fw_bw_ms"],
                "memory_ratio_expanded_over_compact": expanded["peak_allocated_gb"] / compact["peak_allocated_gb"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
