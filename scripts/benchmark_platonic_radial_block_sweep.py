#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from matterformer.models.platonic.triton_attention import platonic_attention_flat_triton


def parse_block_sizes(value: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in value.split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        if "x" not in stripped:
            raise ValueError(f"Block size {item!r} must use MxN format")
        left, right = stripped.split("x", 1)
        block_m = int(left)
        block_n = int(right)
        if block_m <= 0 or block_n <= 0:
            raise ValueError("Block sizes must be positive")
        sizes.append((block_m, block_n))
    if not sizes:
        raise ValueError("At least one block size is required")
    return sizes


def make_lengths(*, num_segments: int, total: int, max_len: int, seed: int) -> np.ndarray:
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


def summarize_ms(times: list[float]) -> dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def current_memory() -> dict[str, float]:
    return {
        "allocated_gb": float(torch.cuda.memory_allocated() / 1024**3),
        "reserved_gb": float(torch.cuda.memory_reserved() / 1024**3),
    }


def peak_memory(before_allocated: int, before_reserved: int) -> dict[str, float]:
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "peak_allocated_gb": float(peak_allocated / 1024**3),
        "peak_reserved_gb": float(peak_reserved / 1024**3),
        "peak_allocated_delta_gb": float(max(0, peak_allocated - before_allocated) / 1024**3),
        "peak_reserved_delta_gb": float(max(0, peak_reserved - before_reserved) / 1024**3),
    }


def set_env(name: str, value: str) -> str | None:
    old = os.environ.get(name)
    os.environ[name] = value
    return old


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def cuda_time_ms(fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Triton radial-r2 Platonic attention block sizes.")
    parser.add_argument("--block-sizes", type=str, default="16x32,16x16,8x32,8x16,4x16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=12000)
    parser.add_argument("--segments", type=int, default=218)
    parser.add_argument("--max-len", type=int, default=283)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=160)
    parser.add_argument("--heads-per-frame", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--precision", type=str, default="tf32x3")
    parser.add_argument("--bwd-mode", type=str, default="auto")
    parser.add_argument("--delta-mode", type=str, default="triton_vector")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.heads % args.heads_per_frame != 0:
        raise ValueError("--heads must be divisible by --heads-per-frame")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    old_bwd = set_env("MATTERFORMER_PLATONIC_TRITON_BWD", args.bwd_mode)
    old_delta = set_env("MATTERFORMER_PLATONIC_TRITON_DELTA", args.delta_mode)
    try:
        device = torch.device("cuda")
        lengths = make_lengths(
            num_segments=args.segments,
            total=args.tokens,
            max_len=args.max_len,
            seed=args.seed,
        )
        total_tokens = int(lengths.sum())
        max_seqlen = int(lengths.max())
        cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
        cu[1:] = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)

        generator = torch.Generator(device=device).manual_seed(args.seed + 17)
        scale = 1.0 / math.sqrt(args.head_dim)
        q_base = torch.randn(
            (total_tokens, args.heads, args.head_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ) * scale
        k_base = torch.randn(q_base.shape, device=device, dtype=torch.float32, generator=generator) * scale
        v_base = torch.randn(q_base.shape, device=device, dtype=torch.float32, generator=generator)
        dout = torch.randn(q_base.shape, device=device, dtype=torch.float32, generator=generator)
        pos = torch.randn((total_tokens, 3), device=device, dtype=torch.float32, generator=generator) * 3.0
        rbf_weight_base = torch.randn((args.heads_per_frame, 1), device=device, dtype=torch.float32, generator=generator) * 0.01
        gate_base = torch.randn((args.heads_per_frame,), device=device, dtype=torch.float32, generator=generator) * 0.01
        centers = torch.zeros(1, device=device, dtype=torch.float32)
        gamma = torch.ones((), device=device, dtype=torch.float32)

        shape = {
            "tokens": total_tokens,
            "segments": int(len(lengths)),
            "max_seqlen": max_seqlen,
            "mean_len": float(lengths.mean()),
            "median_len": float(np.median(lengths)),
            "sum_n2": int((lengths.astype(np.int64) ** 2).sum()),
            "heads": int(args.heads),
            "head_dim": int(args.head_dim),
            "heads_per_frame": int(args.heads_per_frame),
            "dtype": "float32",
            "precision": args.precision,
            "bwd_mode": args.bwd_mode,
            "delta_mode": args.delta_mode,
        }

        def attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            rbf_weight: torch.Tensor,
            gate: torch.Tensor,
            *,
            block_m: int,
            block_n: int,
        ) -> torch.Tensor:
            return platonic_attention_flat_triton(
                q,
                k,
                v,
                cu_seqlens=cu,
                max_seqlen=max_seqlen,
                pos=pos,
                heads_per_frame=args.heads_per_frame,
                rbf_weight=rbf_weight,
                gate=gate,
                centers=centers,
                gamma=gamma,
                radial_bias_kind="radial_r2",
                diag_zero=True,
                precision=args.precision,
                block_m=block_m,
                block_n=block_n,
                strict=True,
            )

        rows: list[dict[str, object]] = []
        for block_m, block_n in parse_block_sizes(args.block_sizes):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row: dict[str, object] = {
                "block_m": int(block_m),
                "block_n": int(block_n),
                "status": "ok",
                **shape,
            }
            print(f"[block {block_m}x{block_n}] starting", flush=True)
            try:
                with torch.no_grad():
                    for _ in range(args.warmup):
                        attention(q_base, k_base, v_base, rbf_weight_base, gate_base, block_m=block_m, block_n=block_n)
                    torch.cuda.synchronize()
                    before_allocated = int(torch.cuda.memory_allocated())
                    before_reserved = int(torch.cuda.memory_reserved())
                    torch.cuda.reset_peak_memory_stats()
                    fwd_times = [
                        cuda_time_ms(
                            lambda: attention(
                                q_base,
                                k_base,
                                v_base,
                                rbf_weight_base,
                                gate_base,
                                block_m=block_m,
                                block_n=block_n,
                            )
                        )
                        for _ in range(args.repeat)
                    ]
                    torch.cuda.synchronize()
                row.update({f"fwd_{key}": value for key, value in summarize_ms(fwd_times).items()})
                row.update({f"fwd_{key}": value for key, value in peak_memory(before_allocated, before_reserved).items()})

                for _ in range(args.warmup):
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    w = rbf_weight_base.detach().clone().requires_grad_(True)
                    g = gate_base.detach().clone().requires_grad_(True)
                    out = attention(q, k, v, w, g, block_m=block_m, block_n=block_n)
                    out.backward(dout)
                torch.cuda.synchronize()

                bwd_times = []
                for _ in range(args.repeat):
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    w = rbf_weight_base.detach().clone().requires_grad_(True)
                    g = gate_base.detach().clone().requires_grad_(True)
                    out = attention(q, k, v, w, g, block_m=block_m, block_n=block_n)
                    torch.cuda.synchronize()
                    before_allocated = int(torch.cuda.memory_allocated())
                    before_reserved = int(torch.cuda.memory_reserved())
                    torch.cuda.reset_peak_memory_stats()
                    bwd_times.append(cuda_time_ms(lambda: out.backward(dout)))
                    row.update({f"bwd_{key}": value for key, value in peak_memory(before_allocated, before_reserved).items()})
                    del q, k, v, w, g, out
                torch.cuda.synchronize()
                row.update({f"bwd_{key}": value for key, value in summarize_ms(bwd_times).items()})

                step_times = []
                for _ in range(args.repeat):
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    w = rbf_weight_base.detach().clone().requires_grad_(True)
                    g = gate_base.detach().clone().requires_grad_(True)

                    def step() -> None:
                        out = attention(q, k, v, w, g, block_m=block_m, block_n=block_n)
                        out.backward(dout)

                    step_times.append(cuda_time_ms(step))
                    del q, k, v, w, g
                torch.cuda.synchronize()
                row.update({f"step_{key}": value for key, value in summarize_ms(step_times).items()})
            except Exception as exc:  # noqa: BLE001 - benchmarking should record launch failures and keep sweeping.
                row["status"] = "failed"
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc).splitlines()[0]
                row["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-8:])
                torch.cuda.synchronize()
            row.update({f"final_{key}": value for key, value in current_memory().items()})
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

        result = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "shape": shape,
            "rows": rows,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        restore_env("MATTERFORMER_PLATONIC_TRITON_BWD", old_bwd)
        restore_env("MATTERFORMER_PLATONIC_TRITON_DELTA", old_delta)


if __name__ == "__main__":
    main()
