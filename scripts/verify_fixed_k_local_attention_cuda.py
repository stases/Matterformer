#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from matterformer.geometry import build_triton_nonperiodic_knn_geometry_cache
from matterformer.models.platonic import (
    ESENEnvelopedRBFTypeFixedKBias,
    NoFixedKLocalBias,
    fixed_k_local_attention_triton,
    fixed_k_local_attention_torch_reference,
    prepare_esen_fixed_k_local_attention_features,
)
from matterformer.models.platonic.local_attention_triton import TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


@torch.no_grad()
def _run_case(
    *,
    device: torch.device,
    with_bias: bool,
    with_type_bias: bool,
    diag_zero: bool,
    envelope_in_score: bool,
    head_dim: int,
    k_neighbors: int,
    qkv_dtype: torch.dtype,
    precision: str,
    use_precomputed: bool = False,
) -> dict[str, float | int | str | bool]:
    seed = (
        1000
        + int(with_bias) * 3
        + int(with_type_bias) * 5
        + int(diag_zero) * 7
        + int(envelope_in_score) * 11
        + int(head_dim)
        + int(k_neighbors)
    )
    torch.manual_seed(seed)
    num_atoms = 9
    coords = torch.randn(1, num_atoms, 3, device=device, dtype=torch.float32) * 0.7
    num_heads = 4
    heads_per_frame = 2
    cutoff = 3.0
    rbf_dim = 6
    q = torch.randn(num_atoms, num_heads, head_dim, device=device, dtype=torch.float32).to(qkv_dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    atom_types = torch.tensor([1, 6, 8, 7, 16, 6, 1, 8, 7], device=device, dtype=torch.long)
    geom = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=None,
        k_neighbors=k_neighbors,
        rbf_dim=rbf_dim,
        cutoff=cutoff,
        seq_len=num_atoms,
        strict=True,
        include_self=True,
        self_as_first_neighbor=True,
        mask_by_cutoff=True,
    )
    bias = None
    esen_features = None
    if with_bias:
        centers = torch.linspace(0.0, cutoff, rbf_dim, device=device)
        delta = cutoff / max(rbf_dim - 1, 1)
        gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), device=device)
        type_bias = None
        if with_type_bias:
            type_bias = torch.randn(20, 20, heads_per_frame, device=device) * 0.03
        bias = ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=torch.randn(heads_per_frame, rbf_dim, device=device) * 0.05,
            type_bias=type_bias,
            centers=centers,
            gamma=gamma,
            cutoff=cutoff,
            heads_per_frame=heads_per_frame,
            diag_zero=diag_zero,
            envelope_in_score=envelope_in_score,
            trainable=False,
        ).to(device)
        if use_precomputed:
            esen_features = prepare_esen_fixed_k_local_attention_features(
                neighbor_idx=geom.neighbor_idx[0],
                neighbor_mask=geom.neighbor_mask[0],
                dist=geom.dist[0],
                centers=bias.centers,
                gamma=bias.gamma,
                cutoff=bias.cutoff,
                heads_per_frame=bias.heads_per_frame,
                atom_types=atom_types,
                max_atomic_number=19,
                diag_zero=bias.diag_zero,
                envelope_in_score=bias.envelope_in_score,
            )
    if precision == "bf16_flash_compat":
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
        q_ref = q
        k_ref = k
        v_ref = v
        expected_dtype = q.dtype
    expected = fixed_k_local_attention_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        rbf=geom.rbf[0],
        atom_types=atom_types,
        bias=bias,
    ).to(expected_dtype).to(q.dtype)
    actual, lse = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        atom_types=atom_types,
        bias=bias,
        esen_features=esen_features,
        precision=precision,
        strict=True,
        return_lse=True,
    )
    diff = (actual - expected).abs()
    tol = 8.0e-3 if qkv_dtype is torch.bfloat16 or precision == "bf16_flash_compat" else 4.0e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    if not torch.isfinite(lse).all():
        raise AssertionError("lse contains non-finite values")
    return {
        "with_bias": bool(with_bias),
        "with_type_bias": bool(with_type_bias),
        "diag_zero": bool(diag_zero),
        "envelope_in_score": bool(envelope_in_score),
        "head_dim": int(head_dim),
        "k_neighbors": int(k_neighbors),
        "qkv_dtype": str(qkv_dtype).replace("torch.", ""),
        "precision": str(precision),
        "use_precomputed": bool(use_precomputed),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _reference_qkv_for_backward(
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    *,
    qkv_dtype: torch.dtype,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    if precision == "bf16_flash_compat":
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        return q.to(torch.bfloat16).float(), k.to(torch.bfloat16).float(), v.to(torch.bfloat16).float(), q, k, v, q.dtype
    if qkv_dtype is torch.bfloat16:
        q = q_base.detach().to(torch.bfloat16).clone().requires_grad_(True)
        k = k_base.detach().to(torch.bfloat16).clone().requires_grad_(True)
        v = v_base.detach().to(torch.bfloat16).clone().requires_grad_(True)
        return q.float(), k.float(), v.float(), q, k, v, q.dtype
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    return q, k, v, q, k, v, q.dtype


def _run_backward_case(
    *,
    device: torch.device,
    geom,
    atom_types: torch.Tensor,
    centers: torch.Tensor,
    rbf_init: torch.Tensor,
    type_init: torch.Tensor,
    head_dim: int,
    qkv_dtype: torch.dtype,
    precision: str,
    case_name: str,
    bias_kind: str,
    use_precomputed: bool = False,
) -> None:
    q_base = torch.randn(5, 2, head_dim, device=device)
    k_base = torch.randn_like(q_base)
    v_base = torch.randn_like(q_base)
    grad_seed = torch.randn_like(q_base).to(torch.bfloat16 if qkv_dtype is torch.bfloat16 else torch.float32)

    q_tri = q_base.detach().to(qkv_dtype).clone().requires_grad_(True)
    k_tri = k_base.detach().to(qkv_dtype).clone().requires_grad_(True)
    v_tri = v_base.detach().to(qkv_dtype).clone().requires_grad_(True)
    q_ref_apply, k_ref_apply, v_ref_apply, q_ref_leaf, k_ref_leaf, v_ref_leaf, expected_input_dtype = _reference_qkv_for_backward(
        q_base,
        k_base,
        v_base,
        qkv_dtype=qkv_dtype,
        precision=precision,
    )

    bias_tri = None
    bias_ref = None
    esen_features = None
    if bias_kind != "none":
        type_tensor = type_init if bias_kind == "esen_type" else None
        bias_tri = ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=rbf_init.clone(),
            type_bias=None if type_tensor is None else type_tensor.clone(),
            centers=centers,
            gamma=torch.tensor(1.0, device=device),
            cutoff=3.0,
            heads_per_frame=1,
            trainable=True,
        ).to(device)
        if use_precomputed:
            esen_features = prepare_esen_fixed_k_local_attention_features(
                neighbor_idx=geom.neighbor_idx[0],
                neighbor_mask=geom.neighbor_mask[0],
                dist=geom.dist[0],
                centers=bias_tri.centers,
                gamma=bias_tri.gamma,
                cutoff=bias_tri.cutoff,
                heads_per_frame=bias_tri.heads_per_frame,
                atom_types=atom_types,
                max_atomic_number=19,
                diag_zero=bias_tri.diag_zero,
                envelope_in_score=bias_tri.envelope_in_score,
            )
        bias_ref = ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=rbf_init.clone(),
            type_bias=None if type_tensor is None else type_tensor.clone(),
            centers=centers,
            gamma=torch.tensor(1.0, device=device),
            cutoff=3.0,
            heads_per_frame=1,
            trainable=True,
        ).to(device)

    out_tri = fixed_k_local_attention_triton(
        q_tri,
        k_tri,
        v_tri,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        atom_types=atom_types,
        bias=bias_tri,
        esen_features=esen_features,
        precision=precision,
        strict=True,
    )
    out_ref = fixed_k_local_attention_torch_reference(
        q_ref_apply,
        k_ref_apply,
        v_ref_apply,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        atom_types=atom_types,
        bias=bias_ref,
    )
    if precision == "bf16_flash_compat":
        out_ref = out_ref.to(torch.bfloat16).to(q_base.dtype)
    elif qkv_dtype is torch.bfloat16:
        out_ref = out_ref.to(torch.bfloat16)

    loose = qkv_dtype is torch.bfloat16 or precision == "bf16_flash_compat"
    out_tol = 2.0e-2 if loose else 4.0e-5
    grad_tol = 5.0e-2 if loose else 3.0e-4
    param_tol = 5.0e-2 if loose else 5.0e-4
    torch.testing.assert_close(out_tri.detach(), out_ref.detach().to(out_tri.dtype), atol=out_tol, rtol=out_tol)
    (out_tri * grad_seed.to(out_tri.dtype)).sum().backward()
    (out_ref * grad_seed.to(out_ref.dtype)).sum().backward()

    torch.testing.assert_close(q_tri.grad.float(), q_ref_leaf.grad.float(), atol=grad_tol, rtol=grad_tol)
    torch.testing.assert_close(k_tri.grad.float(), k_ref_leaf.grad.float(), atol=grad_tol, rtol=grad_tol)
    torch.testing.assert_close(v_tri.grad.float(), v_ref_leaf.grad.float(), atol=grad_tol, rtol=grad_tol)
    if q_tri.grad.dtype != expected_input_dtype:
        raise AssertionError(f"q grad dtype {q_tri.grad.dtype} != expected {expected_input_dtype}")
    if bias_tri is not None and bias_ref is not None:
        torch.testing.assert_close(bias_tri.rbf_weight.grad, bias_ref.rbf_weight.grad, atol=param_tol, rtol=param_tol)
        if bias_tri.type_bias is not None:
            assert bias_ref.type_bias is not None
            torch.testing.assert_close(bias_tri.type_bias.grad, bias_ref.type_bias.grad, atol=param_tol, rtol=param_tol)
    print(
        "backward dtype parity ok",
        {
            "case": case_name,
            "bias": bias_kind,
            "head_dim": head_dim,
            "qkv_dtype": str(qkv_dtype).replace("torch.", ""),
            "precision": precision,
            "use_precomputed": bool(use_precomputed),
            "grad_dtype": str(q_tri.grad.dtype).replace("torch.", ""),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--precisions", nargs="+", default=["ieee", "bf16_flash_compat"])
    parser.add_argument("--head-dims", nargs="+", type=int, default=[64, 128, 160])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 8, 32, 48, 64])
    args = parser.parse_args()

    print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
    print("triton_fixed_k_available", TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE:
        raise SystemExit("fixed-K Triton attention is not available")
    device = torch.device("cuda")
    print("gpu", torch.cuda.get_device_name(0))

    smoke = _run_case(
        device=device,
        with_bias=False,
        with_type_bias=False,
        diag_zero=True,
        envelope_in_score=True,
        head_dim=64,
        k_neighbors=8,
        qkv_dtype=torch.float32,
        precision="ieee",
    )
    print("case ok", smoke)
    precomputed_smoke = _run_case(
        device=device,
        with_bias=True,
        with_type_bias=True,
        diag_zero=True,
        envelope_in_score=True,
        head_dim=64,
        k_neighbors=8,
        qkv_dtype=torch.float32,
        precision="ieee",
        use_precomputed=True,
    )
    print("precomputed case ok", precomputed_smoke)

    no_bias_module = NoFixedKLocalBias().to(device)
    q = torch.randn(5, 2, 16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.arange(5, device=device, dtype=torch.int32).view(5, 1)
    neighbor_mask = torch.ones(5, 1, device=device, dtype=torch.bool)
    expected = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=None,
        strict=True,
    )
    actual = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=no_bias_module,
        strict=True,
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)
    print("NoFixedKLocalBias fast path ok")

    coords = torch.randn(1, 5, 3, device=device, dtype=torch.float32) * 0.4
    geom = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=None,
        k_neighbors=5,
        rbf_dim=4,
        cutoff=3.0,
        seq_len=5,
        strict=True,
        include_self=True,
        self_as_first_neighbor=True,
        mask_by_cutoff=True,
    )
    centers = torch.linspace(0.0, 3.0, 4, device=device)
    atom_types = torch.tensor([1, 6, 8, 7, 6], device=device, dtype=torch.long)
    rbf_init = torch.randn(1, 4, device=device) * 0.01
    type_init = torch.randn(20, 20, 1, device=device) * 0.01
    backward_cases = [
        ("fp32", torch.float32, "ieee"),
        ("direct_bf16", torch.bfloat16, "ieee"),
        ("bf16_flash_compat", torch.float32, "bf16_flash_compat"),
    ]
    for case_name, qkv_dtype, precision in backward_cases:
        for backward_head_dim in (32, 160, 256):
            for bias_kind in ("none", "esen_no_type", "esen_type"):
                _run_backward_case(
                    device=device,
                    geom=geom,
                    atom_types=atom_types,
                    centers=centers,
                    rbf_init=rbf_init,
                    type_init=type_init,
                    head_dim=backward_head_dim,
                    qkv_dtype=qkv_dtype,
                    precision=precision,
                    case_name=case_name,
                    bias_kind=bias_kind,
                )
                if bias_kind == "esen_type":
                    _run_backward_case(
                        device=device,
                        geom=geom,
                        atom_types=atom_types,
                        centers=centers,
                        rbf_init=rbf_init,
                        type_init=type_init,
                        head_dim=backward_head_dim,
                        qkv_dtype=qkv_dtype,
                        precision=precision,
                        case_name=case_name,
                        bias_kind=bias_kind,
                        use_precomputed=True,
                    )

    for dtype_name in args.dtypes:
        dtype = _dtype_from_name(dtype_name)
        for precision in args.precisions:
            for head_dim in args.head_dims:
                for k_neighbors in args.k_values:
                    for with_type_bias in (False, True):
                        result = _run_case(
                            device=device,
                            with_bias=True,
                            with_type_bias=with_type_bias,
                            diag_zero=True,
                            envelope_in_score=True,
                            head_dim=head_dim,
                            k_neighbors=k_neighbors,
                            qkv_dtype=dtype,
                            precision=precision,
                        )
                        print("case ok", result)
    for diag_zero in (False, True):
        for envelope_in_score in (False, True):
            result = _run_case(
                device=device,
                with_bias=True,
                with_type_bias=True,
                diag_zero=diag_zero,
                envelope_in_score=envelope_in_score,
                head_dim=64,
                k_neighbors=8,
                qkv_dtype=torch.float32,
                precision="ieee",
            )
            print("flag case ok", result)
    print("all fixed-K local attention CUDA cases ok")


if __name__ == "__main__":
    main()
