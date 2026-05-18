#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from matterformer.models.platonic import triton_attention as current_triton_attention


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def patched_env(values: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def summarize_ms(times: list[float]) -> dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.float().pow(2).mean().sqrt().clamp_min(1e-12)
    return float((a.float() - b.float()).pow(2).mean().sqrt() / denom)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def cuda_time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current and patched Platonic radial-r2 Triton attention kernels.")
    parser.add_argument("--patched-file", type=Path, default=Path("patches/triton_attention_patched.py"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=12000)
    parser.add_argument("--segments", type=int, default=218)
    parser.add_argument("--max-len", type=int, default=283)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=160)
    parser.add_argument("--heads-per-frame", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.heads % args.heads_per_frame != 0:
        raise ValueError("--heads must be divisible by --heads-per-frame")

    patched = load_module(args.patched_file.resolve(), "patched_triton_attention_for_benchmark")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    lengths = make_lengths(num_segments=args.segments, total=args.tokens, max_len=args.max_len, seed=args.seed)
    total_tokens = int(lengths.sum())
    max_seqlen = int(lengths.max())
    cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)

    generator = torch.Generator(device=device).manual_seed(args.seed + 101)
    scale = 1.0 / math.sqrt(args.head_dim)
    q_base = torch.randn((total_tokens, args.heads, args.head_dim), device=device, dtype=torch.float32, generator=generator) * scale
    k_base = torch.randn(q_base.shape, device=device, dtype=torch.float32, generator=generator) * scale
    v_base = torch.randn(q_base.shape, device=device, dtype=torch.float32, generator=generator)
    dout = torch.randn(q_base.shape, device=device, dtype=torch.float32, generator=generator)
    pos = torch.randn((total_tokens, 3), device=device, dtype=torch.float32, generator=generator) * 3.0
    weight_base = torch.randn((args.heads_per_frame, 1), device=device, dtype=torch.float32, generator=generator) * 0.01
    gate_base = torch.randn((args.heads_per_frame,), device=device, dtype=torch.float32, generator=generator) * 0.01
    centers = torch.zeros(1, device=device, dtype=torch.float32)
    gamma = torch.ones((), device=device, dtype=torch.float32)

    def call(module, q, k, v, weight, gate):
        return module.platonic_attention_flat_triton(
            q,
            k,
            v,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            pos=pos,
            heads_per_frame=args.heads_per_frame,
            rbf_weight=weight,
            gate=gate,
            centers=centers,
            gamma=gamma,
            radial_bias_kind="radial_r2",
            diag_zero=True,
            precision="tf32x3",
            block_m=args.block_m,
            block_n=args.block_n,
            strict=True,
        )

    variants = [
        {
            "name": "current_split",
            "module": current_triton_attention,
            "env": {
                "MATTERFORMER_PLATONIC_TRITON_BWD": "split",
                "MATTERFORMER_PLATONIC_TRITON_DELTA": "triton_vector",
                "MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM": "0",
            },
        },
        {
            "name": "patched_split_no_splitD",
            "module": patched,
            "env": {
                "MATTERFORMER_PLATONIC_TRITON_BWD": "split",
                "MATTERFORMER_PLATONIC_TRITON_DELTA": "triton_vector",
                "MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM": "0",
                "MATTERFORMER_PLATONIC_TRITON_BWD_NUM_STAGES": "1",
            },
        },
        {
            "name": "patched_auto_splitD_stage1",
            "module": patched,
            "env": {
                "MATTERFORMER_PLATONIC_TRITON_BWD": "auto",
                "MATTERFORMER_PLATONIC_TRITON_DELTA": "triton_vector",
                "MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM": "auto",
                "MATTERFORMER_PLATONIC_TRITON_BWD_NUM_STAGES": "1",
            },
        },
        {
            "name": "patched_split_splitD_stage1",
            "module": patched,
            "env": {
                "MATTERFORMER_PLATONIC_TRITON_BWD": "split",
                "MATTERFORMER_PLATONIC_TRITON_DELTA": "triton_vector",
                "MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM": "auto",
                "MATTERFORMER_PLATONIC_TRITON_BWD_NUM_STAGES": "1",
            },
        },
        {
            "name": "patched_auto_splitD_stage2",
            "module": patched,
            "env": {
                "MATTERFORMER_PLATONIC_TRITON_BWD": "auto",
                "MATTERFORMER_PLATONIC_TRITON_DELTA": "triton_vector",
                "MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM": "auto",
                "MATTERFORMER_PLATONIC_TRITON_BWD_NUM_STAGES": "2",
            },
        },
    ]

    def run_once(module, env: dict[str, str | None]) -> dict[str, torch.Tensor]:
        with patched_env(env):
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            v = v_base.detach().clone().requires_grad_(True)
            w = weight_base.detach().clone().requires_grad_(True)
            g = gate_base.detach().clone().requires_grad_(True)
            out = call(module, q, k, v, w, g)
            out.backward(dout)
            torch.cuda.synchronize()
            return {
                "out": out.detach(),
                "dq": q.grad.detach(),
                "dk": k.grad.detach(),
                "dv": v.grad.detach(),
                "dw": w.grad.detach(),
                "dg": g.grad.detach(),
            }

    baseline = run_once(current_triton_attention, variants[0]["env"])
    rows: list[dict[str, object]] = []
    for variant in variants:
        name = str(variant["name"])
        module = variant["module"]
        env = variant["env"]
        row: dict[str, object] = {
            "name": name,
            "status": "ok",
            "block_m": int(args.block_m),
            "block_n": int(args.block_n),
            "heads": int(args.heads),
            "head_dim": int(args.head_dim),
            "tokens": total_tokens,
            "segments": int(len(lengths)),
            "max_seqlen": max_seqlen,
            "sum_n2": int((lengths.astype(np.int64) ** 2).sum()),
            **{f"env/{key}": value for key, value in env.items()},
        }
        print(f"[{name}] starting", flush=True)
        try:
            once = run_once(module, env)
            for key, value in once.items():
                row[f"{key}_max_abs_vs_current"] = max_abs(value, baseline[key])
                row[f"{key}_rel_rms_vs_current"] = rel_rms(value, baseline[key])

            with patched_env(env):
                with torch.no_grad():
                    for _ in range(args.warmup):
                        call(module, q_base, k_base, v_base, weight_base, gate_base)
                    torch.cuda.synchronize()
                    fwd_times = []
                    torch.cuda.reset_peak_memory_stats()
                    for _ in range(args.repeat):
                        fwd_times.append(cuda_time_ms(lambda: call(module, q_base, k_base, v_base, weight_base, gate_base)))
                    fwd_peak = float(torch.cuda.max_memory_allocated() / 1024**3)
                row.update({f"fwd_{key}": value for key, value in summarize_ms(fwd_times).items()})
                row["fwd_peak_allocated_gb"] = fwd_peak

                for _ in range(args.warmup):
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    w = weight_base.detach().clone().requires_grad_(True)
                    g = gate_base.detach().clone().requires_grad_(True)
                    out = call(module, q, k, v, w, g)
                    out.backward(dout)
                torch.cuda.synchronize()

                bwd_times = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.repeat):
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    w = weight_base.detach().clone().requires_grad_(True)
                    g = gate_base.detach().clone().requires_grad_(True)
                    out = call(module, q, k, v, w, g)
                    torch.cuda.synchronize()
                    bwd_times.append(cuda_time_ms(lambda: out.backward(dout)))
                    del q, k, v, w, g, out
                row.update({f"bwd_{key}": value for key, value in summarize_ms(bwd_times).items()})
                row["bwd_peak_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)

                step_times = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.repeat):
                    q = q_base.detach().clone().requires_grad_(True)
                    k = k_base.detach().clone().requires_grad_(True)
                    v = v_base.detach().clone().requires_grad_(True)
                    w = weight_base.detach().clone().requires_grad_(True)
                    g = gate_base.detach().clone().requires_grad_(True)

                    def step() -> None:
                        out = call(module, q, k, v, w, g)
                        out.backward(dout)

                    step_times.append(cuda_time_ms(step))
                    del q, k, v, w, g
                row.update({f"step_{key}": value for key, value in summarize_ms(step_times).items()})
                row["step_peak_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc).splitlines()[0]
            row["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-12:])
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "patched_file": str(args.patched_file.resolve()),
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


if __name__ == "__main__":
    main()
