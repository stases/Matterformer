#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from flash_attn import flash_attn_varlen_func

from matterformer.models.platonic.triton_attention import platonic_attention_flat_triton


def make_lengths(num_segments: int = 202, total: int = 12000, max_len: int = 253, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lengths = np.clip(rng.gamma(2.0, 28.0, size=num_segments).astype(np.int64) + 1, 1, max_len)
    lengths[0] = max_len
    while int(lengths.sum()) < total:
        choices = np.flatnonzero(lengths < max_len)
        idx = int(rng.choice(choices))
        lengths[idx] += min(max_len - int(lengths[idx]), total - int(lengths.sum()), int(rng.integers(1, 16)))
    while int(lengths.sum()) > total:
        choices = np.flatnonzero(lengths > 1)
        idx = int(rng.choice(choices))
        lengths[idx] -= min(int(lengths[idx]) - 1, int(lengths.sum()) - total, int(rng.integers(1, 16)))
    return lengths.astype(np.int32)


def summarize(times: list[float]) -> dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).float().pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-12)).item()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).float().abs().max().item()


def set_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    return old


def restore_env(name: str, value) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark flat Platonic attention backend and radial bias modes.")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--segments", type=int, default=202)
    parser.add_argument("--tokens", type=int, default=12000)
    parser.add_argument("--max-len", type=int, default=253)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(20260516)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = "cuda"
    heads = 60
    head_dim = 32
    heads_per_frame = 5
    num_rbf = 8
    lengths = make_lengths(num_segments=args.segments, total=args.tokens, max_len=args.max_len, seed=args.seed)
    total_tokens = int(lengths.sum())
    max_seqlen = int(lengths.max())
    cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)

    generator = torch.Generator(device=device).manual_seed(1234)
    q0 = torch.randn((total_tokens, heads, head_dim), device=device, generator=generator) / math.sqrt(head_dim)
    k0 = torch.randn((total_tokens, heads, head_dim), device=device, generator=generator) / math.sqrt(head_dim)
    v0 = torch.randn((total_tokens, heads, head_dim), device=device, generator=generator)
    pos = torch.randn((total_tokens, 3), device=device, generator=generator) * 3.0
    dout = torch.randn((total_tokens, heads, head_dim), device=device, generator=generator)
    weights = {
        "radial_rbf": torch.randn((heads_per_frame, num_rbf), device=device, generator=generator) * 0.01,
        "radial_r2": torch.randn((heads_per_frame, 1), device=device, generator=generator) * 0.01,
        "radial_slope": torch.randn((heads_per_frame, 1), device=device, generator=generator) * 0.01,
    }
    gate0 = torch.randn((heads_per_frame,), device=device, generator=generator) * 0.01
    centers_rbf = torch.linspace(0.0, 6.0, num_rbf, device=device)
    gamma_rbf = torch.tensor(1.0 / float((6.0 / max(num_rbf - 1, 1)) ** 2), device=device)
    center_one = torch.zeros(1, device=device)
    gamma_one = torch.ones((), device=device)

    def flash_current(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_attn_varlen_func(
            q.contiguous().to(torch.bfloat16),
            k.contiguous().to(torch.bfloat16),
            v.contiguous().to(torch.bfloat16),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
        ).to(q.dtype)

    def triton_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        precision: str = "tf32x3",
        bias_kind: str | None = None,
        weight: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias_kind is None:
            return platonic_attention_flat_triton(
                q,
                k,
                v,
                cu_seqlens=cu,
                max_seqlen=max_seqlen,
                precision=precision,
                block_m=args.block_m,
                block_n=args.block_n,
                strict=True,
            )
        if bias_kind == "radial_rbf":
            centers = centers_rbf
            gamma = gamma_rbf
        else:
            centers = center_one
            gamma = gamma_one
        return platonic_attention_flat_triton(
            q,
            k,
            v,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            pos=pos,
            heads_per_frame=heads_per_frame,
            rbf_weight=weight,
            gate=gate,
            centers=centers,
            gamma=gamma,
            radial_bias_kind=bias_kind,
            diag_zero=True,
            precision=precision,
            block_m=args.block_m,
            block_n=args.block_n,
            strict=True,
        )

    def bench_forward(name: str, fn: Callable[[], torch.Tensor]) -> dict[str, object]:
        with torch.no_grad():
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(args.repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
        return {"name": name, "pass": "fwd", **summarize(times)}

    def bench_fwd_bwd(
        name: str,
        fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor],
        *,
        radial_kind: str | None = None,
        bwd_mode: str = "auto",
        delta_mode: str = "torch",
    ) -> dict[str, object]:
        q = q0.detach().clone().requires_grad_(True)
        k = k0.detach().clone().requires_grad_(True)
        v = v0.detach().clone().requires_grad_(True)
        if radial_kind is None:
            weight = gate = None
        else:
            weight = weights[radial_kind].detach().clone().requires_grad_(True)
            gate = gate0.detach().clone().requires_grad_(True)
        old_bwd = set_env("MATTERFORMER_PLATONIC_TRITON_BWD", bwd_mode)
        old_delta = set_env("MATTERFORMER_PLATONIC_TRITON_DELTA", delta_mode)

        def step() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            if weight is not None:
                weight.grad = None
                assert gate is not None
                gate.grad = None
            out = fn(q, k, v, weight, gate)
            out.backward(dout)

        try:
            for _ in range(args.warmup):
                step()
            torch.cuda.synchronize()
            times = []
            for _ in range(args.repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                step()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
        finally:
            restore_env("MATTERFORMER_PLATONIC_TRITON_BWD", old_bwd)
            restore_env("MATTERFORMER_PLATONIC_TRITON_DELTA", old_delta)
        return {"name": name, "pass": "fwd_bwd", "bwd_mode": bwd_mode, "delta_mode": delta_mode, **summarize(times)}

    rows: list[dict[str, object]] = []
    rows.append(bench_forward("flash_bf16_no_bias", lambda: flash_current(q0, k0, v0)))
    rows.append(bench_fwd_bwd("flash_bf16_no_bias", lambda q, k, v, w, g: flash_current(q, k, v)))

    fwd_specs = [
        ("triton_tf32x3_no_bias", lambda: triton_attention(q0, k0, v0, precision="tf32x3")),
        ("triton_bf16compat_no_bias", lambda: triton_attention(q0, k0, v0, precision="bf16_flash_compat")),
        (
            "triton_tf32x3_radial_r2",
            lambda: triton_attention(q0, k0, v0, bias_kind="radial_r2", weight=weights["radial_r2"], gate=gate0),
        ),
        (
            "triton_tf32x3_radial_slope",
            lambda: triton_attention(q0, k0, v0, bias_kind="radial_slope", weight=weights["radial_slope"], gate=gate0),
        ),
        (
            "triton_tf32x3_radial_rbf8",
            lambda: triton_attention(q0, k0, v0, bias_kind="radial_rbf", weight=weights["radial_rbf"], gate=gate0),
        ),
    ]
    for name, fn in fwd_specs:
        rows.append(bench_forward(name, fn))

    def no_bias_tf32(q, k, v, w, g):
        return triton_attention(q, k, v, precision="tf32x3")

    def no_bias_bf16(q, k, v, w, g):
        return triton_attention(q, k, v, precision="bf16_flash_compat")

    def radial(kind: str):
        return lambda q, k, v, w, g: triton_attention(q, k, v, bias_kind=kind, weight=w, gate=g)

    bwd_specs = [
        ("triton_tf32x3_no_bias_split_delta_triton", no_bias_tf32, None, "split", "triton"),
        ("triton_tf32x3_no_bias_split_delta_vector", no_bias_tf32, None, "split", "triton_vector"),
        ("triton_tf32x3_no_bias_split_delta_torch", no_bias_tf32, None, "split", "torch"),
        ("triton_bf16compat_no_bias_split_delta_triton", no_bias_bf16, None, "split", "triton"),
        ("triton_bf16compat_no_bias_split_delta_vector", no_bias_bf16, None, "split", "triton_vector"),
        ("triton_bf16compat_no_bias_split_delta_torch", no_bias_bf16, None, "split", "torch"),
        ("triton_tf32x3_radial_r2_atomic_delta_vector", radial("radial_r2"), "radial_r2", "atomic", "triton_vector"),
        ("triton_tf32x3_radial_r2_atomic_delta_torch", radial("radial_r2"), "radial_r2", "atomic", "torch"),
        ("triton_tf32x3_radial_r2_split_delta_vector", radial("radial_r2"), "radial_r2", "split", "triton_vector"),
        ("triton_tf32x3_radial_r2_split_delta_torch", radial("radial_r2"), "radial_r2", "split", "torch"),
        ("triton_tf32x3_radial_slope_atomic_delta_vector", radial("radial_slope"), "radial_slope", "atomic", "triton_vector"),
        ("triton_tf32x3_radial_slope_atomic_delta_torch", radial("radial_slope"), "radial_slope", "atomic", "torch"),
        ("triton_tf32x3_radial_slope_split_delta_vector", radial("radial_slope"), "radial_slope", "split", "triton_vector"),
        ("triton_tf32x3_radial_slope_split_delta_torch", radial("radial_slope"), "radial_slope", "split", "torch"),
        ("triton_tf32x3_radial_rbf8_atomic_delta_triton", radial("radial_rbf"), "radial_rbf", "atomic", "triton"),
        ("triton_tf32x3_radial_rbf8_atomic_delta_vector", radial("radial_rbf"), "radial_rbf", "atomic", "triton_vector"),
        ("triton_tf32x3_radial_rbf8_atomic_delta_torch", radial("radial_rbf"), "radial_rbf", "atomic", "torch"),
        ("triton_tf32x3_radial_rbf8_split_delta_vector", radial("radial_rbf"), "radial_rbf", "split", "triton_vector"),
        ("triton_tf32x3_radial_rbf8_split_delta_torch", radial("radial_rbf"), "radial_rbf", "split", "torch"),
    ]
    for name, fn, kind, bwd_mode, delta_mode in bwd_specs:
        rows.append(bench_fwd_bwd(name, fn, radial_kind=kind, bwd_mode=bwd_mode, delta_mode=delta_mode))

    with torch.no_grad():
        out_flash = flash_current(q0, k0, v0)
        out_tri = triton_attention(q0, k0, v0, precision="tf32x3")
        out_bf16 = triton_attention(q0, k0, v0, precision="bf16_flash_compat")
        parity = {
            "triton_tf32x3_no_bias_vs_flash_bf16": {
                "out_max_abs": max_abs(out_tri, out_flash),
                "out_rel_rms": rel_rms(out_tri, out_flash),
            },
            "triton_bf16compat_no_bias_vs_flash_bf16": {
                "out_max_abs": max_abs(out_bf16, out_flash),
                "out_rel_rms": rel_rms(out_bf16, out_flash),
            },
        }
        for kind, weight in weights.items():
            out_zero = triton_attention(
                q0,
                k0,
                v0,
                bias_kind=kind,
                weight=torch.zeros_like(weight),
                gate=torch.zeros_like(gate0),
            )
            parity[f"{kind}_zero_vs_triton_tf32x3_no_bias"] = {
                "out_max_abs": max_abs(out_zero, out_tri),
                "out_rel_rms": rel_rms(out_zero, out_tri),
            }

    shape = {
        "segments": int(len(lengths)),
        "tokens": total_tokens,
        "heads": heads,
        "head_dim": head_dim,
        "max_seqlen": max_seqlen,
        "mean_len": float(lengths.mean()),
        "median_len": float(np.median(lengths)),
        "sum_n2": int((lengths.astype(np.int64) ** 2).sum()),
        "block_m": args.block_m,
        "block_n": args.block_n,
    }
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shape": shape,
        "rows": rows,
        "parity": parity,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nbackend | pass | median_ms | mean_ms | p10_ms | p90_ms")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for row in rows:
        print(
            f"{row['name']} | {row['pass']} | {row['median_ms']:.3f} | "
            f"{row['mean_ms']:.3f} | {row['p10_ms']:.3f} | {row['p90_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
