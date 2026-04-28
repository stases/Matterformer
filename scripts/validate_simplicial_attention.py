#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from matterformer.models import SimplicialAttentionMask, SimplicialFactorizedBias, SimplicialLowRankAngleResidual
from matterformer.models.attention import (
    _TritonTwoSimplicialAttentionFunction,
    simplicial_attention_torch_from_projected,
)
from matterformer.models.attention_triton import TRITON_AVAILABLE


PARITY_CASES = (
    {
        "name": "ieee_none_padding",
        "dtype": torch.float32,
        "precision": "ieee_fp32",
        "mode": "none",
        "mask_mode": "padding",
        "batch_size": 2,
        "num_heads": 4,
        "tokens": 5,
        "head_dim": 8,
        "chunk_size": 2,
        "atol": 1e-4,
        "rtol": 1e-4,
    },
    {
        "name": "ieee_factorized_padding",
        "dtype": torch.float32,
        "precision": "ieee_fp32",
        "mode": "factorized",
        "mask_mode": "padding",
        "batch_size": 2,
        "num_heads": 4,
        "tokens": 5,
        "head_dim": 8,
        "chunk_size": 2,
        "atol": 1e-4,
        "rtol": 1e-4,
    },
    {
        "name": "ieee_angle_low_rank_padding",
        "dtype": torch.float32,
        "precision": "ieee_fp32",
        "mode": "angle_low_rank",
        "mask_mode": "padding",
        "batch_size": 2,
        "num_heads": 4,
        "tokens": 5,
        "head_dim": 8,
        "angle_rank": 4,
        "chunk_size": 2,
        "atol": 1e-4,
        "rtol": 1e-4,
    },
    {
        "name": "bf16_none_padding",
        "dtype": torch.bfloat16,
        "precision": "bf16_tc",
        "mode": "none",
        "mask_mode": "padding",
        "batch_size": 2,
        "num_heads": 4,
        "tokens": 16,
        "head_dim": 16,
        "chunk_size": 8,
        "atol": 5e-2,
        "rtol": 5e-2,
    },
    {
        "name": "bf16_factorized_padding",
        "dtype": torch.bfloat16,
        "precision": "bf16_tc",
        "mode": "factorized",
        "mask_mode": "padding",
        "batch_size": 2,
        "num_heads": 4,
        "tokens": 16,
        "head_dim": 16,
        "chunk_size": 8,
        "atol": 5e-2,
        "rtol": 5e-2,
    },
    {
        "name": "ieee_factorized_cls_query_atoms_only",
        "dtype": torch.float32,
        "precision": "ieee_fp32",
        "mode": "factorized",
        "mask_mode": "cls_query",
        "batch_size": 2,
        "num_heads": 4,
        "tokens": 5,
        "head_dim": 8,
        "chunk_size": 2,
        "atol": 1e-4,
        "rtol": 1e-4,
    },
    {
        "name": "ieee_none_single_pair",
        "dtype": torch.float32,
        "precision": "ieee_fp32",
        "mode": "none",
        "mask_mode": "single_pair",
        "batch_size": 1,
        "num_heads": 2,
        "tokens": 4,
        "head_dim": 8,
        "chunk_size": 2,
        "atol": 1e-4,
        "rtol": 1e-4,
    },
)


def _make_factorized_bias(
    batch_size: int,
    num_heads: int,
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
        u=torch.randn(batch_size, num_heads, num_tokens, num_tokens, **kwargs),
        v=torch.randn(batch_size, num_heads, num_tokens, num_tokens, **kwargs),
        w=torch.randn(batch_size, num_heads, num_tokens, num_tokens, **kwargs),
        gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
    )


def _make_low_rank_angle_residual(
    batch_size: int,
    num_heads: int,
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
        left=torch.randn(batch_size, num_heads, num_tokens, num_tokens, rank, **kwargs),
        right=torch.randn(batch_size, num_heads, num_tokens, num_tokens, rank, **kwargs),
        gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
    )


def _grad_stats(actual: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float) -> dict[str, float | bool]:
    diff = (actual - reference).abs().float()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose": bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
    }


def _build_attention_mask(case: dict[str, object], device: torch.device) -> SimplicialAttentionMask:
    batch_size = int(case["batch_size"])
    tokens = int(case["tokens"])
    mask_mode = str(case["mask_mode"])
    if mask_mode == "padding":
        key_padding_mask = torch.zeros(batch_size, tokens, device=device, dtype=torch.bool)
        key_padding_mask[0, -1] = True
        if tokens > 4 and batch_size > 1:
            key_padding_mask[1, tokens // 2 :] = True
        return SimplicialAttentionMask.from_key_padding_mask(
            key_padding_mask,
            batch_size=batch_size,
            num_tokens=tokens,
            device=device,
        )
    if mask_mode == "cls_query":
        query_valid = torch.ones(batch_size, tokens, device=device, dtype=torch.bool)
        pair_key_valid = torch.zeros(batch_size, tokens, device=device, dtype=torch.bool)
        pair_key_valid[:, : tokens - 1] = True
        return SimplicialAttentionMask(query_valid=query_valid, pair_key_valid=pair_key_valid)
    if mask_mode == "single_pair":
        query_valid = torch.ones(batch_size, tokens, device=device, dtype=torch.bool)
        pair_key_valid = torch.ones(batch_size, tokens, device=device, dtype=torch.bool)
        pair_valid = torch.zeros(batch_size, tokens, tokens, device=device, dtype=torch.bool)
        pair_valid[:, 0, 0] = True
        return SimplicialAttentionMask(query_valid=query_valid, pair_key_valid=pair_key_valid, pair_valid=pair_valid)
    raise ValueError(f"Unsupported mask mode: {mask_mode}")


def run_parity_case(case: dict[str, object], device: torch.device) -> dict[str, object]:
    torch.manual_seed(0)
    dtype = case["dtype"]
    batch_size = int(case["batch_size"])
    num_heads = int(case["num_heads"])
    tokens = int(case["tokens"])
    head_dim = int(case["head_dim"])
    chunk_size = int(case["chunk_size"])
    precision = str(case["precision"])
    atol = float(case["atol"])
    rtol = float(case["rtol"])
    debug_torch_backward = precision == "ieee_fp32"

    q_ref = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, dtype=dtype, requires_grad=True)
    k1_ref = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, dtype=dtype, requires_grad=True)
    v1_ref = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, dtype=dtype, requires_grad=True)
    k2_ref = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, dtype=dtype, requires_grad=True)
    v2_ref = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, dtype=dtype, requires_grad=True)
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k1_tri = k1_ref.detach().clone().requires_grad_(True)
    v1_tri = v1_ref.detach().clone().requires_grad_(True)
    k2_tri = k2_ref.detach().clone().requires_grad_(True)
    v2_tri = v2_ref.detach().clone().requires_grad_(True)

    attention_mask = _build_attention_mask(case, device)
    factorized_bias_ref = None
    factorized_bias_tri = None
    angle_residual_ref = None
    angle_residual_tri = None
    if case["mode"] == "factorized":
        factorized_bias_ref = _make_factorized_bias(
            batch_size,
            num_heads,
            tokens,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        factorized_bias_tri = SimplicialFactorizedBias(
            u=factorized_bias_ref.u.detach().clone().requires_grad_(True),
            v=factorized_bias_ref.v.detach().clone().requires_grad_(True),
            w=factorized_bias_ref.w.detach().clone().requires_grad_(True),
            gate=factorized_bias_ref.gate.detach().clone().requires_grad_(True),
        )
    if case["mode"] == "angle_low_rank":
        angle_rank = int(case.get("angle_rank", 4))
        angle_residual_ref = _make_low_rank_angle_residual(
            batch_size,
            num_heads,
            tokens,
            angle_rank,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        angle_residual_tri = SimplicialLowRankAngleResidual(
            left=angle_residual_ref.left.detach().clone().requires_grad_(True),
            right=angle_residual_ref.right.detach().clone().requires_grad_(True),
            gate=angle_residual_ref.gate.detach().clone().requires_grad_(True),
        )

    ref_out = simplicial_attention_torch_from_projected(
        q_ref,
        k1_ref,
        v1_ref,
        k2_ref,
        v2_ref,
        attention_mask=attention_mask,
        factorized_bias=factorized_bias_ref,
        angle_residual=angle_residual_ref,
        chunk_size=chunk_size,
        fp32_core=debug_torch_backward,
    )
    tri_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_tri.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16),
        k1_tri.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16),
        v1_tri.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16),
        k2_tri.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16),
        v2_tri.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16),
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        attention_mask.pair_valid if attention_mask.pair_valid is not None else torch.empty(0, device=device, dtype=torch.bool),
        factorized_bias_tri.u.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if factorized_bias_tri is not None else q_tri.new_empty(0),
        factorized_bias_tri.v.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if factorized_bias_tri is not None else q_tri.new_empty(0),
        factorized_bias_tri.w.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if factorized_bias_tri is not None else q_tri.new_empty(0),
        factorized_bias_tri.gate.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if factorized_bias_tri is not None else q_tri.new_empty(0),
        angle_residual_tri.left.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if angle_residual_tri is not None else q_tri.new_empty(0),
        angle_residual_tri.right.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if angle_residual_tri is not None else q_tri.new_empty(0),
        angle_residual_tri.gate.to(torch.float32 if precision != "bf16_tc" else torch.bfloat16) if angle_residual_tri is not None else q_tri.new_empty(0),
        q_tri.new_empty(0),
        q_tri.new_empty(0),
        q_tri.new_empty(0),
        precision,
        chunk_size,
        debug_torch_backward,
    )

    forward_diff = (tri_out.float() - ref_out.float()).abs()
    ref_loss = ref_out.float().square().mean()
    tri_loss = tri_out.float().square().mean()
    ref_loss.backward()
    tri_loss.backward()

    gradients = {
        "q": _grad_stats(q_tri.grad.float(), q_ref.grad.float(), atol=atol, rtol=rtol),
        "k1": _grad_stats(k1_tri.grad.float(), k1_ref.grad.float(), atol=atol, rtol=rtol),
        "v1": _grad_stats(v1_tri.grad.float(), v1_ref.grad.float(), atol=atol, rtol=rtol),
        "k2": _grad_stats(k2_tri.grad.float(), k2_ref.grad.float(), atol=atol, rtol=rtol),
        "v2": _grad_stats(v2_tri.grad.float(), v2_ref.grad.float(), atol=atol, rtol=rtol),
    }
    if factorized_bias_ref is not None and factorized_bias_tri is not None:
        gradients.update(
            {
                "u": _grad_stats(factorized_bias_tri.u.grad.float(), factorized_bias_ref.u.grad.float(), atol=atol, rtol=rtol),
                "v": _grad_stats(factorized_bias_tri.v.grad.float(), factorized_bias_ref.v.grad.float(), atol=atol, rtol=rtol),
                "w": _grad_stats(factorized_bias_tri.w.grad.float(), factorized_bias_ref.w.grad.float(), atol=atol, rtol=rtol),
                "gate": _grad_stats(
                    factorized_bias_tri.gate.grad.float(),
                    factorized_bias_ref.gate.grad.float(),
                    atol=atol,
                    rtol=rtol,
                ),
            }
        )
    if angle_residual_ref is not None and angle_residual_tri is not None:
        gradients.update(
            {
                "angle_left": _grad_stats(
                    angle_residual_tri.left.grad.float(),
                    angle_residual_ref.left.grad.float(),
                    atol=atol,
                    rtol=rtol,
                ),
                "angle_right": _grad_stats(
                    angle_residual_tri.right.grad.float(),
                    angle_residual_ref.right.grad.float(),
                    atol=atol,
                    rtol=rtol,
                ),
                "angle_gate": _grad_stats(
                    angle_residual_tri.gate.grad.float(),
                    angle_residual_ref.gate.grad.float(),
                    atol=atol,
                    rtol=rtol,
                ),
            }
        )

    return {
        "name": str(case["name"]),
        "mode": str(case["mode"]),
        "precision": precision,
        "mask_mode": str(case["mask_mode"]),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "tokens": tokens,
            "head_dim": head_dim,
            "chunk_size": chunk_size,
        },
        "tolerance": {"atol": atol, "rtol": rtol},
        "forward": {
            "max_abs_diff": float(forward_diff.max().item()),
            "mean_abs_diff": float(forward_diff.mean().item()),
            "allclose": bool(torch.allclose(tri_out.float(), ref_out.float(), atol=atol, rtol=rtol)),
        },
        "gradients": gradients,
    }


def run_pytest(command: list[str], env: dict[str, str]) -> dict[str, object]:
    proc = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    summary = stdout_lines[-1] if stdout_lines else ""
    return {
        "command": command,
        "returncode": proc.returncode,
        "summary": summary,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dense simplicial attention and emit a JSON report")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--benchmark-markdown", type=str, default=None)
    parser.add_argument("--benchmark-json", type=str, default=None)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--benchmark-dtype", type=str, default="bfloat16")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validate_simplicial_attention.py")
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for validate_simplicial_attention.py")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_md = Path(args.benchmark_markdown) if args.benchmark_markdown else output_path.with_suffix(".bench.md")
    benchmark_json = Path(args.benchmark_json) if args.benchmark_json else output_path.with_suffix(".bench.json")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")

    full_suite = run_pytest([sys.executable, "-m", "pytest", "-q"], env=env)
    triton_subset = run_pytest(
        [sys.executable, "-m", "pytest", "-q", "tests/test_simplicial_attention.py", "-k", "triton or simplicial"],
        env=env,
    )

    device = torch.device("cuda")
    parity_cases = [run_parity_case(case, device) for case in PARITY_CASES]

    benchmark_payload = None
    if not args.skip_benchmark:
        benchmark_command = [
            sys.executable,
            "scripts/benchmark_simplicial_attention.py",
            "--impls",
            "torch",
            "triton",
            "--modes",
            "none",
            "factorized",
            "--precisions",
            "bf16_tc",
            "--passes",
            "forward",
            "train_step",
            "--dtype",
            args.benchmark_dtype,
            "--warmup",
            str(args.benchmark_warmup),
            "--iters",
            str(args.benchmark_iters),
            "--output",
            str(benchmark_md),
            "--json-output",
            str(benchmark_json),
        ]
        subprocess.run(benchmark_command, cwd=REPO_ROOT, env=env, check=True)
        benchmark_payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    elif benchmark_json.exists():
        benchmark_payload = json.loads(benchmark_json.read_text(encoding="utf-8"))

    payload = {
        "environment": {
            "python": sys.executable,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "triton_version": __import__("triton").__version__,
            "device_name": torch.cuda.get_device_name(device),
            "cuda_available": True,
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "pytest": {
            "full_suite": full_suite,
            "triton_subset": triton_subset,
        },
        "parity_cases": parity_cases,
        "benchmarks": benchmark_payload,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
