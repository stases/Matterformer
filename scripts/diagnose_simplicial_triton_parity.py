#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from matterformer.models import SimplicialAttentionMask, SimplicialFactorizedBias
from matterformer.models.attention import (
    _TritonTwoSimplicialAttentionFunction,
    simplicial_attention_torch_from_projected,
)
from matterformer.models.attention_triton import TRITON_AVAILABLE


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _stats(actual: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    actual_f = actual.detach().float()
    reference_f = reference.detach().float()
    diff = (actual_f - reference_f).abs()
    rel = diff / reference_f.abs().clamp_min(1e-6)
    return {
        "allclose": bool(torch.allclose(actual_f, reference_f, atol=atol, rtol=rtol)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms_abs": float(diff.square().mean().sqrt().item()),
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item()),
        "max_rel": float(rel.max().item()),
        "p99_rel": float(torch.quantile(rel.flatten(), 0.99).item()),
    }


def _attention_mask(batch_size: int, tokens: int, device: torch.device) -> SimplicialAttentionMask:
    pad_mask = torch.zeros(batch_size, tokens, device=device, dtype=torch.bool)
    pad_mask[0, -1] = True
    if batch_size > 1 and tokens > 4:
        pad_mask[1, tokens // 2 :] = True
    return SimplicialAttentionMask.from_key_padding_mask(
        pad_mask,
        batch_size=batch_size,
        num_tokens=tokens,
        device=device,
    )


def _make_factorized_bias(
    batch_size: int,
    num_heads: int,
    tokens: int,
    *,
    device: torch.device,
    kind: str,
) -> SimplicialFactorizedBias:
    shape = (batch_size, num_heads, tokens, tokens)
    if kind == "random":
        u = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)
        w = torch.randn(shape, device=device)
        gate = torch.randn(batch_size, num_heads, tokens, device=device)
    elif kind == "cancellation":
        u = 1000.0 + 0.2 * torch.randn(shape, device=device)
        v = -1000.0 + 0.2 * torch.randn(shape, device=device)
        w = 0.2 * torch.randn(shape, device=device)
        gate = torch.ones(batch_size, num_heads, tokens, device=device)
    else:
        raise ValueError(f"Unsupported bias kind: {kind}")
    return SimplicialFactorizedBias(
        u=u.requires_grad_(True),
        v=v.requires_grad_(True),
        w=w.requires_grad_(True),
        gate=gate.requires_grad_(True),
    )


def _clone_inputs(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        q.detach().clone().requires_grad_(True),
        k1.detach().clone().requires_grad_(True),
        v1.detach().clone().requires_grad_(True),
        k2.detach().clone().requires_grad_(True),
        v2.detach().clone().requires_grad_(True),
    )


def _clone_bias(bias: SimplicialFactorizedBias | None) -> SimplicialFactorizedBias | None:
    if bias is None:
        return None
    return SimplicialFactorizedBias(
        u=bias.u.detach().clone().requires_grad_(True),
        v=bias.v.detach().clone().requires_grad_(True),
        w=bias.w.detach().clone().requires_grad_(True),
        gate=bias.gate.detach().clone().requires_grad_(True),
    )


def _empty(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device)


def _run_torch_reference(
    *,
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    bias: SimplicialFactorizedBias | None,
    attention_mask: SimplicialAttentionMask,
    grad_out: torch.Tensor,
    chunk_size: int,
    fp32_core: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = simplicial_attention_torch_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        factorized_bias=bias,
        chunk_size=chunk_size,
        fp32_core=fp32_core,
    )
    out.float().backward(grad_out)
    grads = {
        "q": q.grad.detach().clone(),
        "k1": k1.grad.detach().clone(),
        "v1": v1.grad.detach().clone(),
        "k2": k2.grad.detach().clone(),
        "v2": v2.grad.detach().clone(),
    }
    if bias is not None:
        grads.update(
            {
                "u": bias.u.grad.detach().clone(),
                "v": bias.v.grad.detach().clone(),
                "w": bias.w.grad.detach().clone(),
                "gate": bias.gate.grad.detach().clone(),
            }
        )
    return out.detach(), grads


def _run_triton_variant(
    *,
    q_base: torch.Tensor,
    k1_base: torch.Tensor,
    v1_base: torch.Tensor,
    k2_base: torch.Tensor,
    v2_base: torch.Tensor,
    bias_base: SimplicialFactorizedBias | None,
    attention_mask: SimplicialAttentionMask,
    grad_out: torch.Tensor,
    precision: str,
    activation_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = q_base.device
    q, k1, v1, k2, v2 = _clone_inputs(q_base, k1_base, v1_base, k2_base, v2_base)
    bias = _clone_bias(bias_base)
    out = _TritonTwoSimplicialAttentionFunction.apply(
        q.to(activation_dtype),
        k1.to(activation_dtype),
        v1.to(activation_dtype),
        k2.to(activation_dtype),
        v2.to(activation_dtype),
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        attention_mask.pair_valid if attention_mask.pair_valid is not None else torch.empty(0, device=device, dtype=torch.bool),
        bias.u.to(bias_dtype) if bias is not None else _empty(device),
        bias.v.to(bias_dtype) if bias is not None else _empty(device),
        bias.w.to(bias_dtype) if bias is not None else _empty(device),
        bias.gate.to(bias_dtype) if bias is not None else _empty(device),
        _empty(device),
        _empty(device),
        _empty(device),
        _empty(device),
        _empty(device),
        _empty(device),
        precision,
        chunk_size,
        False,
    )
    out.float().backward(grad_out)
    grads = {
        "q": q.grad.detach().clone(),
        "k1": k1.grad.detach().clone(),
        "v1": v1.grad.detach().clone(),
        "k2": k2.grad.detach().clone(),
        "v2": v2.grad.detach().clone(),
    }
    if bias is not None:
        grads.update(
            {
                "u": bias.u.grad.detach().clone(),
                "v": bias.v.grad.detach().clone(),
                "w": bias.w.grad.detach().clone(),
                "gate": bias.gate.grad.detach().clone(),
            }
        )
    return out.detach(), grads


def _bias_quantization_stats(bias: SimplicialFactorizedBias | None) -> dict[str, Any] | None:
    if bias is None:
        return None
    b32 = bias.gate[..., None, None] * (
        bias.u[..., :, None] + bias.v[..., None, :] + bias.w[:, :, None, :, :]
    )
    u = bias.u.to(torch.bfloat16).float()
    v = bias.v.to(torch.bfloat16).float()
    w = bias.w.to(torch.bfloat16).float()
    gate = bias.gate.to(torch.bfloat16).float()
    bbf = gate[..., None, None] * (u[..., :, None] + v[..., None, :] + w[:, :, None, :, :])
    diff = (bbf - b32).abs()
    rel = diff / b32.abs().clamp_min(1e-6)
    p32 = torch.softmax(b32.flatten(-2), dim=-1)
    pbf = torch.softmax(bbf.flatten(-2), dim=-1)
    kl = (p32 * (p32.clamp_min(1e-30).log() - pbf.clamp_min(1e-30).log())).sum(dim=-1)
    return {
        "bias_abs_max": float(b32.abs().max().item()),
        "bf16_abs_err_mean": float(diff.mean().item()),
        "bf16_abs_err_p99": float(torch.quantile(diff.flatten(), 0.99).item()),
        "bf16_abs_err_max": float(diff.max().item()),
        "bf16_rel_err_p99": float(torch.quantile(rel.flatten(), 0.99).item()),
        "bias_only_kl_mean": float(kl.mean().item()),
        "bias_only_kl_p99": float(torch.quantile(kl.flatten(), 0.99).item()),
        "top_pair_changed_frac": float((p32.argmax(dim=-1) != pbf.argmax(dim=-1)).float().mean().item()),
    }


def run_case(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    torch.manual_seed(int(case["seed"]))
    batch_size = int(case["batch_size"])
    num_heads = int(case["num_heads"])
    tokens = int(case["tokens"])
    head_dim = int(case["head_dim"])
    chunk_size = int(case["chunk_size"])
    mode = str(case["mode"])
    bias_kind = str(case.get("bias_kind", "random"))

    q = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, requires_grad=True)
    k1 = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, requires_grad=True)
    v1 = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, requires_grad=True)
    k2 = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, requires_grad=True)
    v2 = torch.randn(batch_size, num_heads, tokens, head_dim, device=device, requires_grad=True)
    bias = None
    if mode == "factorized":
        bias = _make_factorized_bias(batch_size, num_heads, tokens, device=device, kind=bias_kind)
    elif mode != "none":
        raise ValueError(f"Unsupported mode: {mode}")

    attention_mask = _attention_mask(batch_size, tokens, device)
    grad_out = torch.randn(batch_size, num_heads, tokens, head_dim, device=device)
    q_ref, k1_ref, v1_ref, k2_ref, v2_ref = _clone_inputs(q, k1, v1, k2, v2)
    bias_ref = _clone_bias(bias)
    ref_out, ref_grads = _run_torch_reference(
        q=q_ref,
        k1=k1_ref,
        v1=v1_ref,
        k2=k2_ref,
        v2=v2_ref,
        bias=bias_ref,
        attention_mask=attention_mask,
        grad_out=grad_out,
        chunk_size=chunk_size,
        fp32_core=True,
    )

    variants = [
        ("triton_ieee_fp32", "ieee_fp32", torch.float32, torch.float32, 1e-4, 1e-4),
        ("triton_tf32", "tf32", torch.float32, torch.float32, 2e-2, 2e-2),
        ("triton_bf16_tc_bias_fp32", "bf16_tc", torch.bfloat16, torch.float32, 6e-2, 6e-2),
    ]
    if bias is not None:
        variants.append(("triton_bf16_tc_bias_bf16_old", "bf16_tc", torch.bfloat16, torch.bfloat16, 6e-2, 6e-2))

    variant_results = []
    for name, precision, activation_dtype, bias_dtype, atol, rtol in variants:
        out, grads = _run_triton_variant(
            q_base=q,
            k1_base=k1,
            v1_base=v1,
            k2_base=k2,
            v2_base=v2,
            bias_base=bias,
            attention_mask=attention_mask,
            grad_out=grad_out,
            precision=precision,
            activation_dtype=activation_dtype,
            bias_dtype=bias_dtype,
            chunk_size=chunk_size,
        )
        variant_results.append(
            {
                "name": name,
                "precision": precision,
                "activation_dtype": _dtype_name(activation_dtype),
                "factorized_bias_dtype": _dtype_name(bias_dtype) if bias is not None else None,
                "tolerance": {"atol": atol, "rtol": rtol},
                "forward": _stats(out, ref_out, atol=atol, rtol=rtol),
                "gradients": {
                    key: _stats(value, ref_grads[key], atol=atol, rtol=rtol)
                    for key, value in grads.items()
                },
            }
        )

    return {
        "name": str(case["name"]),
        "mode": mode,
        "bias_kind": bias_kind if bias is not None else None,
        "shape": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "tokens": tokens,
            "head_dim": head_dim,
            "chunk_size": chunk_size,
        },
        "reference": "torch_fp32_core",
        "bias_bf16_quantization": _bias_quantization_stats(bias),
        "variants": variant_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Triton simplicial parity across precision modes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    cases = [
        {
            "name": "none",
            "mode": "none",
            "seed": 0,
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
            "chunk_size": args.chunk_size,
        },
        {
            "name": "factorized_random",
            "mode": "factorized",
            "bias_kind": "random",
            "seed": 1,
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
            "chunk_size": args.chunk_size,
        },
        {
            "name": "factorized_cancellation",
            "mode": "factorized",
            "bias_kind": "cancellation",
            "seed": 2,
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
            "chunk_size": args.chunk_size,
        },
    ]
    payload = {
        "environment": {
            "python": sys.executable,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "triton_version": __import__("triton").__version__,
            "device_name": torch.cuda.get_device_name(device),
            "tf32_disabled_for_torch_reference": True,
        },
        "cases": [run_case(case, device) for case in cases],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
