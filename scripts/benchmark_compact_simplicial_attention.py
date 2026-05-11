#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from matterformer.models import CompactSimplicialBias, compact_simplicial_attention_torch, compact_simplicial_attention_triton
from matterformer.models.triton_compact_simplicial_attention import TRITON_COMPACT_SIMPLICIAL_AVAILABLE


@dataclass
class BenchmarkResult:
    name: str
    backend: str
    dtype: str
    precision: str
    batch_size: int
    heads: int
    tokens: int
    head_dim: int
    neighbors: int
    angle_rank: int
    message_rank: int
    message: bool
    gate_mode: str
    forward_ms: float
    fwd_bwd_ms: float
    forward_peak_delta_mb: float
    fwd_bwd_peak_delta_mb: float
    max_abs_diff: float | None = None
    max_grad_abs_diff: float | None = None


def _dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def _make_neighbor_idx(batch_size: int, tokens: int, neighbors: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(tokens, device=device)[:, None]
    offsets = torch.arange(neighbors, device=device)[None, :]
    return ((base + offsets) % tokens).expand(batch_size, -1, -1).contiguous()


def _make_case(args: argparse.Namespace, *, dtype: torch.dtype) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, CompactSimplicialBias]:
    device = torch.device("cuda")
    shape = (args.batch_size, args.heads, args.tokens, args.head_dim)
    scale = args.head_dim**-0.5
    tensors = [
        (torch.randn(shape, device=device, dtype=dtype) * scale).requires_grad_(True),
        torch.randn(shape, device=device, dtype=dtype).requires_grad_(True),
        torch.randn(shape, device=device, dtype=dtype).requires_grad_(True),
        torch.randn(shape, device=device, dtype=dtype).requires_grad_(True),
        torch.randn(shape, device=device, dtype=dtype).requires_grad_(True),
    ]
    neighbor_idx = _make_neighbor_idx(args.batch_size, args.tokens, args.neighbors, device)
    if args.mask_mode == "all":
        neighbor_mask = torch.ones(args.batch_size, args.tokens, args.neighbors, device=device, dtype=torch.bool)
    else:
        neighbor_mask = (torch.rand(args.batch_size, args.tokens, args.neighbors, device=device) > args.mask_invalid_prob).contiguous()
        neighbor_mask[..., 0] = True

    bias_shape = (args.batch_size, args.heads, args.tokens, args.neighbors)
    rank_shape = (*bias_shape, args.angle_rank)
    gate = torch.zeros(args.batch_size, args.heads, args.tokens, device=device, dtype=dtype)
    angle_gate = torch.zeros_like(gate)
    if args.gate_mode == "random":
        gate = torch.randn_like(gate)
        angle_gate = torch.randn_like(angle_gate)
    elif args.gate_mode == "small":
        gate = torch.randn_like(gate) * 0.1
        angle_gate = torch.randn_like(angle_gate) * 0.1
    elif args.gate_mode != "zero":
        raise ValueError(f"Unsupported gate mode {args.gate_mode!r}")

    message_left = message_right = message_basis = None
    if args.message:
        message_shape = (*bias_shape, args.message_rank)
        message_left = torch.randn(message_shape, device=device, dtype=dtype, requires_grad=True)
        message_right = torch.randn(message_shape, device=device, dtype=dtype, requires_grad=True)
        message_basis = torch.randn(args.heads, args.message_rank, args.head_dim, device=device, dtype=dtype, requires_grad=True)

    bias = CompactSimplicialBias(
        u=torch.randn(bias_shape, device=device, dtype=dtype, requires_grad=True),
        v=torch.randn(bias_shape, device=device, dtype=dtype, requires_grad=True),
        gate=gate.requires_grad_(True),
        angle_left=torch.randn(rank_shape, device=device, dtype=dtype, requires_grad=True),
        angle_right=torch.randn(rank_shape, device=device, dtype=dtype, requires_grad=True),
        angle_gate=angle_gate.requires_grad_(True),
        message_left=message_left,
        message_right=message_right,
        message_basis=message_basis,
    )
    return tensors, neighbor_idx, neighbor_mask, bias


def _bias_tensors(bias: CompactSimplicialBias) -> list[torch.Tensor]:
    return [
        tensor
        for tensor in (
            bias.u,
            bias.v,
            bias.gate,
            bias.angle_left,
            bias.angle_right,
            bias.angle_gate,
            bias.message_left,
            bias.message_right,
            bias.message_basis,
        )
        if tensor is not None
    ]


def _zero_grads(tensors: list[torch.Tensor], bias: CompactSimplicialBias) -> None:
    for tensor in tensors + _bias_tensors(bias):
        tensor.grad = None


def _clone_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def _clone_bias(bias: CompactSimplicialBias) -> CompactSimplicialBias:
    def clone(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.detach().clone().requires_grad_(True) if tensor is not None else None

    return CompactSimplicialBias(
        u=clone(bias.u),
        v=clone(bias.v),
        gate=clone(bias.gate),
        angle_left=clone(bias.angle_left),
        angle_right=clone(bias.angle_right),
        angle_gate=clone(bias.angle_gate),
        message_left=clone(bias.message_left),
        message_right=clone(bias.message_right),
        message_basis=clone(bias.message_basis),
    )


def _call(
    backend: str,
    tensors: list[torch.Tensor],
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: CompactSimplicialBias,
    *,
    precision: str,
) -> torch.Tensor:
    if backend == "torch":
        return compact_simplicial_attention_torch(
            *tensors,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=bias,
        )
    if backend == "triton":
        return compact_simplicial_attention_triton(
            *tensors,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=bias,
            precision=precision,
            strict=True,
        )
    raise ValueError(f"Unsupported backend {backend!r}")


def _time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _peak_delta_mb(fn: Callable[[], None]) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024**2)


def _compare(
    tensors: list[torch.Tensor],
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: CompactSimplicialBias,
    *,
    precision: str,
) -> tuple[float, float]:
    ref_tensors = _clone_tensors(tensors)
    tri_tensors = _clone_tensors(tensors)
    ref_bias = _clone_bias(bias)
    tri_bias = _clone_bias(bias)
    ref = _call("torch", ref_tensors, neighbor_idx, neighbor_mask, ref_bias, precision=precision)
    tri = _call("triton", tri_tensors, neighbor_idx, neighbor_mask, tri_bias, precision=precision)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    tri.backward(grad)
    torch.cuda.synchronize()
    max_abs_diff = float((ref.float() - tri.float()).abs().max().item())
    grad_pairs = [(a.grad, b.grad) for a, b in zip(tri_tensors, ref_tensors)]
    grad_pairs.extend((a.grad, b.grad) for a, b in zip(_bias_tensors(tri_bias), _bias_tensors(ref_bias)))
    max_grad_abs_diff = max(float((a.float() - b.float()).abs().max().item()) for a, b in grad_pairs if a is not None and b is not None)
    return max_abs_diff, max_grad_abs_diff


def _benchmark_backend(
    name: str,
    backend: str,
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    precision: str,
) -> BenchmarkResult:
    tensors, neighbor_idx, neighbor_mask, bias = _make_case(args, dtype=dtype)

    def forward_once() -> None:
        _zero_grads(tensors, bias)
        out = _call(backend, tensors, neighbor_idx, neighbor_mask, bias, precision=precision)
        # Keep a tiny dependent op in the CUDA stream without adding a host sync.
        _ = out.float().mean()

    def fwd_bwd_once() -> None:
        _zero_grads(tensors, bias)
        out = _call(backend, tensors, neighbor_idx, neighbor_mask, bias, precision=precision)
        grad = torch.randn_like(out)
        out.backward(grad)

    forward_ms = _time_cuda(forward_once, args.warmup, args.iters)
    fwd_bwd_ms = _time_cuda(fwd_bwd_once, args.warmup, args.iters)
    forward_peak = _peak_delta_mb(forward_once)
    fwd_bwd_peak = _peak_delta_mb(fwd_bwd_once)
    max_abs_diff = max_grad_abs_diff = None
    if backend == "triton":
        max_abs_diff, max_grad_abs_diff = _compare(tensors, neighbor_idx, neighbor_mask, bias, precision=precision)

    return BenchmarkResult(
        name=name,
        backend=backend,
        dtype=str(dtype).replace("torch.", ""),
        precision=precision,
        batch_size=args.batch_size,
        heads=args.heads,
        tokens=args.tokens,
        head_dim=args.head_dim,
        neighbors=args.neighbors,
        angle_rank=args.angle_rank,
        message_rank=args.message_rank,
        message=args.message,
        gate_mode=args.gate_mode,
        forward_ms=forward_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        forward_peak_delta_mb=forward_peak,
        fwd_bwd_peak_delta_mb=fwd_bwd_peak,
        max_abs_diff=max_abs_diff,
        max_grad_abs_diff=max_grad_abs_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compact kNN simplicial attention Torch vs Triton.")
    parser.add_argument("--name", default="qm9_like")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--angle-rank", type=int, default=32)
    parser.add_argument("--message-rank", type=int, default=16)
    parser.add_argument("--message", action="store_true")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--precision", choices=("bf16_tc", "tf32", "ieee_fp32"), default="tf32")
    parser.add_argument("--gate-mode", choices=("zero", "small", "random"), default="random")
    parser.add_argument("--mask-mode", choices=("all", "random"), default="all")
    parser.add_argument("--mask-invalid-prob", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not TRITON_COMPACT_SIMPLICIAL_AVAILABLE:
        raise RuntimeError("Triton compact simplicial attention is unavailable")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dtype = _dtype(args.dtype)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} triton_available={TRITON_COMPACT_SIMPLICIAL_AVAILABLE}")
    print(
        "shape="
        f"B{args.batch_size} H{args.heads} N{args.tokens} K{args.neighbors} D{args.head_dim} "
        f"R{args.angle_rank} message={args.message} dtype={args.dtype} precision={args.precision}"
    )

    results = [
        _benchmark_backend(args.name, "torch", args, dtype=dtype, precision=args.precision),
        _benchmark_backend(args.name, "triton", args, dtype=dtype, precision=args.precision),
    ]
    torch_result, triton_result = results
    speedup_fwd = torch_result.forward_ms / triton_result.forward_ms
    speedup_bwd = torch_result.fwd_bwd_ms / triton_result.fwd_bwd_ms
    mem_ratio_fwd = torch_result.forward_peak_delta_mb / max(triton_result.forward_peak_delta_mb, 1e-9)
    mem_ratio_bwd = torch_result.fwd_bwd_peak_delta_mb / max(triton_result.fwd_bwd_peak_delta_mb, 1e-9)
    print("\nresults:")
    for result in results:
        print(
            f"{result.backend:>6} forward={result.forward_ms:8.3f} ms "
            f"fwd+bwd={result.fwd_bwd_ms:8.3f} ms "
            f"peak_fwd_delta={result.forward_peak_delta_mb:8.2f} MiB "
            f"peak_fwd_bwd_delta={result.fwd_bwd_peak_delta_mb:8.2f} MiB"
        )
        if result.max_abs_diff is not None:
            print(
                f"       max_abs_diff={result.max_abs_diff:.6g} "
                f"max_grad_abs_diff={result.max_grad_abs_diff:.6g}"
            )
    print(
        f"\nsummary speedup_forward={speedup_fwd:.3f}x speedup_fwd_bwd={speedup_bwd:.3f}x "
        f"mem_ratio_forward={mem_ratio_fwd:.3f}x mem_ratio_fwd_bwd={mem_ratio_bwd:.3f}x"
    )

    if args.json_out:
        payload = {
            "results": [asdict(result) for result in results],
            "summary": {
                "speedup_forward": speedup_fwd,
                "speedup_fwd_bwd": speedup_bwd,
                "mem_ratio_forward": mem_ratio_fwd,
                "mem_ratio_fwd_bwd": mem_ratio_bwd,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
