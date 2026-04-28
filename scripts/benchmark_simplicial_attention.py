#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from matterformer.models import (
    SimplicialAttentionMask,
    SimplicialFactorizedBias,
    SimplicialLowRankAngleResidual,
    TwoSimplicialAttention,
)


BENCHMARK_GRID = (
    (512, 8, 24, 256),
    (512, 8, 32, 128),
    (512, 8, 48, 64),
    (512, 8, 64, 32),
    (512, 8, 96, 16),
    (512, 8, 128, 8),
    (512, 8, 192, 4),
    (768, 12, 24, 256),
    (768, 12, 32, 128),
    (768, 12, 48, 64),
    (768, 12, 64, 32),
    (768, 12, 96, 16),
    (768, 12, 128, 8),
    (768, 12, 192, 4),
)


@dataclass
class BenchmarkResult:
    impl: str
    mode: str
    precision: str
    pass_type: str
    d_model: int
    n_heads: int
    tokens: int
    batch_size: int
    dtype: str
    compile_ms: float
    steady_ms: float
    peak_mem_mib: float


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _make_factorized_bias(
    batch_size: int,
    n_heads: int,
    num_tokens: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> SimplicialFactorizedBias:
    kwargs = {
        "device": device,
        "dtype": dtype,
        "requires_grad": requires_grad,
    }
    return SimplicialFactorizedBias(
        u=torch.randn(batch_size, n_heads, num_tokens, num_tokens, **kwargs),
        v=torch.randn(batch_size, n_heads, num_tokens, num_tokens, **kwargs),
        w=torch.randn(batch_size, n_heads, num_tokens, num_tokens, **kwargs),
        gate=torch.randn(batch_size, n_heads, num_tokens, **kwargs),
    )


def _make_low_rank_angle_residual(
    batch_size: int,
    n_heads: int,
    num_tokens: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> SimplicialLowRankAngleResidual:
    kwargs = {
        "device": device,
        "dtype": dtype,
        "requires_grad": requires_grad,
    }
    return SimplicialLowRankAngleResidual(
        left=torch.randn(batch_size, n_heads, num_tokens, num_tokens, rank, **kwargs),
        right=torch.randn(batch_size, n_heads, num_tokens, num_tokens, rank, **kwargs),
        gate=torch.randn(batch_size, n_heads, num_tokens, **kwargs),
    )


def _make_inputs(
    *,
    batch_size: int,
    tokens: int,
    d_model: int,
    n_heads: int,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[
    torch.Tensor,
    SimplicialFactorizedBias | None,
    SimplicialLowRankAngleResidual | None,
    SimplicialAttentionMask,
]:
    x = torch.randn(batch_size, tokens, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    factorized_bias = None
    angle_residual = None
    if mode == "factorized":
        factorized_bias = _make_factorized_bias(
            batch_size,
            n_heads,
            tokens,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
    elif mode == "angle_low_rank":
        angle_residual = _make_low_rank_angle_residual(
            batch_size,
            n_heads,
            tokens,
            16,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        None,
        batch_size=batch_size,
        num_tokens=tokens,
        device=device,
    )
    return x, factorized_bias, angle_residual, attention_mask


def _run_single_case(
    *,
    impl: str,
    mode: str,
    precision: str,
    pass_type: str,
    d_model: int,
    n_heads: int,
    tokens: int,
    batch_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    device: torch.device,
) -> BenchmarkResult:
    attn = TwoSimplicialAttention(
        dim=d_model,
        num_heads=n_heads,
        chunk_size=min(tokens, 128),
        dropout=0.0,
        impl=impl,
        precision=precision,
    ).to(device=device, dtype=dtype)
    if pass_type == "forward":
        attn.eval()
    else:
        attn.train()

    def run_once() -> None:
        requires_grad = pass_type == "train_step"
        x, factorized_bias, angle_residual, attention_mask = _make_inputs(
            batch_size=batch_size,
            tokens=tokens,
            d_model=d_model,
            n_heads=n_heads,
            mode=mode,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        attn.zero_grad(set_to_none=True)
        if pass_type == "forward":
            with torch.no_grad():
                _ = attn(
                    x,
                    attention_mask=attention_mask,
                    factorized_bias=factorized_bias,
                    angle_residual=angle_residual,
                )
            return
        out = attn(
            x,
            attention_mask=attention_mask,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
        )
        loss = out.square().mean()
        loss.backward()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    run_once()
    torch.cuda.synchronize(device)
    compile_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize(device)
    steady_ms = ((time.perf_counter() - t0) * 1000.0) / max(iters, 1)
    peak_mem_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    return BenchmarkResult(
        impl=impl,
        mode=mode,
        precision=precision,
        pass_type=pass_type,
        d_model=d_model,
        n_heads=n_heads,
        tokens=tokens,
        batch_size=batch_size,
        dtype=str(dtype).replace("torch.", ""),
        compile_ms=compile_ms,
        steady_ms=steady_ms,
        peak_mem_mib=peak_mem_mib,
    )


def _format_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| impl | mode | precision | pass | d_model | n_heads | T | B | dtype | compile_ms | steady_ms | peak_mem_mib |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.impl} | {result.mode} | {result.precision} | {result.pass_type} | "
            f"{result.d_model} | {result.n_heads} | {result.tokens} | {result.batch_size} | "
            f"{result.dtype} | {result.compile_ms:.2f} | {result.steady_ms:.2f} | {result.peak_mem_mib:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense simplicial attention backends")
    parser.add_argument("--impls", nargs="+", default=["torch", "triton"], choices=["auto", "torch", "triton"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "factorized"],
        choices=["none", "factorized", "angle_low_rank"],
    )
    parser.add_argument("--precisions", nargs="+", default=["bf16_tc"], choices=["bf16_tc", "tf32", "ieee_fp32"])
    parser.add_argument("--passes", nargs="+", default=["forward", "train_step"], choices=["forward", "train_step"])
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_simplicial_attention.py")
    device = torch.device("cuda")
    dtype = _parse_dtype(args.dtype)

    results: list[BenchmarkResult] = []
    for impl in args.impls:
        for mode in args.modes:
            for precision in args.precisions:
                for pass_type in args.passes:
                    for d_model, n_heads, tokens, batch_size in BENCHMARK_GRID:
                        print(
                            f"[bench] impl={impl} mode={mode} precision={precision} pass={pass_type} "
                            f"d_model={d_model} n_heads={n_heads} T={tokens} B={batch_size} dtype={dtype}"
                        )
                        results.append(
                            _run_single_case(
                                impl=impl,
                                mode=mode,
                                precision=precision,
                                pass_type=pass_type,
                                d_model=d_model,
                                n_heads=n_heads,
                                tokens=tokens,
                                batch_size=batch_size,
                                dtype=dtype,
                                warmup=args.warmup,
                                iters=args.iters,
                                device=device,
                            )
                        )

    table = _format_table(results)
    print(table)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
    if args.json_output is not None:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": {
                "device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "dtype": str(dtype).replace("torch.", ""),
            "warmup": args.warmup,
            "iters": args.iters,
            "passes": args.passes,
            "precisions": args.precisions,
            "grid": [
                {
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "tokens": tokens,
                    "batch_size": batch_size,
                }
                for d_model, n_heads, tokens, batch_size in BENCHMARK_GRID
            ],
            "results": [asdict(result) for result in results],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
