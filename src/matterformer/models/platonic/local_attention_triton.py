from __future__ import annotations

import math

import torch

from matterformer.models.platonic.local_attention import (
    ESENEnvelopedRBFTypeFixedKBias,
    FixedKLocalBias,
    NoFixedKLocalBias,
    fixed_k_local_attention_torch_reference,
)
from matterformer.models.platonic.triton_attention import (
    _kernel_input_precision,
    normalize_platonic_attention_precision,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA nodes with Triton installed.
    triton = None
    tl = None


TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE = triton is not None and tl is not None

_FIXED_K_BIAS_NONE = 0
_FIXED_K_BIAS_ESEN_RBF_TYPE = 1


def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _requires_grad(*values: torch.Tensor | None) -> bool:
    return any(value is not None and bool(value.requires_grad) for value in values)


def _fallback_or_raise(
    reason: str,
    *,
    strict: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: FixedKLocalBias | None,
    dist: torch.Tensor | None,
    rbf: torch.Tensor | None,
    atom_types: torch.Tensor | None,
    return_lse: bool = False,
) -> torch.Tensor:
    if return_lse:
        raise RuntimeError(
            "return_lse=True is only implemented by the fixed-K Triton forward path; "
            f"fallback was required because {reason}"
        )
    if strict:
        raise RuntimeError(f"fixed-K local Triton attention is unavailable: {reason}")
    return fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=bias,
        dist=dist,
        rbf=rbf,
        atom_types=atom_types,
    )


if TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE:

    @triton.jit
    def _fixed_k_local_attention_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        neighbor_idx_ptr,
        neighbor_mask_ptr,
        dist_ptr,
        atom_type_ptr,
        rbf_weight_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        out_ptr,
        lse_ptr,
        num_tokens: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        k_neighbors: tl.constexpr,
        scale: tl.constexpr,
        bias_mode: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        has_type_bias: tl.constexpr,
        diag_zero: tl.constexpr,
        envelope_in_score: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        k_mask = offs_k < k_neighbors
        d_mask = offs_d < head_dim

        n_base = token_idx * k_neighbors + offs_k
        neigh_idx = tl.load(neighbor_idx_ptr + n_base, mask=k_mask, other=0).to(tl.int64)
        neigh_valid = (tl.load(neighbor_mask_ptr + n_base, mask=k_mask, other=0) > 0) & k_mask
        q = tl.load(
            q_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
            mask=neigh_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.sum(k_tile.to(tl.float32) * q[None, :], axis=1) * scale

        if bias_mode == 1:
            dist = tl.load(dist_ptr + n_base, mask=k_mask, other=0.0).to(tl.float32)
            local = neigh_valid & (dist < cutoff)
            same = neigh_idx == token_idx
            local = local | (same & neigh_valid)

            x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
            x2 = x * x
            x3 = x2 * x
            env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
            env = tl.where(dist < cutoff, env, 0.0)
            env = tl.where(neigh_valid, env, 0.0)

            env_bias = env
            if diag_zero:
                env_bias = tl.where(same, 0.0, env_bias)

            subhead = head_idx % heads_per_frame
            gamma = tl.load(gamma_ptr).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                raw_bias += weight * rho

            if has_type_bias:
                zi = tl.load(atom_type_ptr + token_idx).to(tl.int64)
                zj = tl.load(atom_type_ptr + neigh_idx, mask=neigh_valid, other=0).to(tl.int64)
                zi = tl.minimum(tl.maximum(zi, 0), max_atomic_number)
                zj = tl.minimum(tl.maximum(zj, 0), max_atomic_number)
                zdim = max_atomic_number + 1
                type_index = ((zi * zdim + zj) * heads_per_frame + subhead)
                raw_bias += tl.load(type_bias_ptr + type_index, mask=neigh_valid, other=0.0).to(tl.float32)

            scores += env_bias * raw_bias
            if envelope_in_score:
                env_score = tl.where(same & neigh_valid, 1.0, env)
                scores += tl.log(tl.maximum(env_score, 1.0e-20))
            neigh_valid = local

        scores = tl.where(neigh_valid, scores, -float("inf"))
        row_max = tl.max(scores, axis=0)
        has_any = row_max > -float("inf")
        row_max_safe = tl.where(has_any, row_max, 0.0)
        probs = tl.exp(scores - row_max_safe)
        probs = tl.where(neigh_valid, probs, 0.0)
        denom = tl.sum(probs, axis=0)
        probs = probs / tl.maximum(denom, 1.0e-20)

        v_tile = tl.load(
            v_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
            mask=neigh_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        out = tl.sum(probs[:, None] * v_tile.to(tl.float32), axis=0)
        tl.store(
            out_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
            out,
            mask=d_mask,
        )
        lse = tl.where(has_any, row_max + tl.log(tl.maximum(denom, 1.0e-20)), -float("inf"))
        tl.store(lse_ptr + token_idx * num_heads + head_idx, lse)


def fixed_k_local_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: FixedKLocalBias | None = None,
    dist: torch.Tensor | None = None,
    rbf: torch.Tensor | None = None,
    atom_types: torch.Tensor | None = None,
    precision: str = "tf32x3",
    max_atomic_number: int | None = None,
    return_lse: bool = False,
    strict: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward-only Triton implementation of fixed-K local atom attention.

    Arbitrary Python bias modules remain supported by the torch reference.  The
    Triton path currently compiles two known modes: no bias and eSEN-like
    enveloped RBF/type bias.  In the eSEN Triton path, RBF values are
    recomputed from ``dist`` and the bias module's ``centers``/``gamma``;
    the optional ``rbf`` argument is used only by torch fallback/reference code.
    """

    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q/k/v must have identical shape [N, H, D]")
    if neighbor_idx.ndim != 2 or neighbor_mask.shape != neighbor_idx.shape:
        raise ValueError("neighbor_idx and neighbor_mask must have shape [N, K]")
    if neighbor_idx.shape[0] != q.shape[0]:
        raise ValueError("neighbor_idx first dimension must match q/k/v tokens")
    if q.shape[0] == 0:
        out = torch.empty_like(q)
        lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        return (out, lse) if return_lse else out
    if isinstance(bias, NoFixedKLocalBias):
        bias = None

    if not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE:
        return _fallback_or_raise(
            "Triton is not installed",
            strict=strict,
            q=q,
            k=k,
            v=v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=bias,
            dist=dist,
            rbf=rbf,
            atom_types=atom_types,
            return_lse=return_lse,
        )
    if q.device.type != "cuda":
        return _fallback_or_raise(
            "q/k/v are not CUDA tensors",
            strict=strict,
            q=q,
            k=k,
            v=v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=bias,
            dist=dist,
            rbf=rbf,
            atom_types=atom_types,
            return_lse=return_lse,
        )
    if torch.is_grad_enabled() and _requires_grad(q, k, v):
        return _fallback_or_raise(
            "forward-only Triton path does not implement autograd yet",
            strict=strict,
            q=q,
            k=k,
            v=v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=bias,
            dist=dist,
            rbf=rbf,
            atom_types=atom_types,
            return_lse=return_lse,
        )

    bias_mode = _FIXED_K_BIAS_NONE
    rbf_weight = torch.empty((1, 1), device=q.device, dtype=torch.float32)
    centers = torch.empty((1,), device=q.device, dtype=torch.float32)
    gamma = torch.ones((), device=q.device, dtype=torch.float32)
    type_bias = torch.empty((1, 1, 1), device=q.device, dtype=torch.float32)
    heads_per_frame = 1
    num_rbf = 1
    cutoff = 1.0
    has_type_bias = False
    diag_zero = True
    envelope_in_score = True
    if bias is not None:
        if not isinstance(bias, ESENEnvelopedRBFTypeFixedKBias):
            return _fallback_or_raise(
                f"unsupported Triton bias module {type(bias).__name__!r}",
                strict=strict,
                q=q,
                k=k,
                v=v,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                bias=bias,
                dist=dist,
                rbf=rbf,
                atom_types=atom_types,
                return_lse=return_lse,
            )
        if dist is None:
            raise ValueError("ESEN fixed-K Triton bias requires dist")
        if dist.shape != neighbor_idx.shape:
            raise ValueError("dist must have shape [N, K]")
        if q.shape[1] % int(bias.heads_per_frame) != 0:
            raise ValueError("heads_per_frame must divide num_heads")
        if torch.is_grad_enabled() and _requires_grad(bias.rbf_weight, getattr(bias, "type_bias", None)):
            return _fallback_or_raise(
                "forward-only Triton path does not implement bias parameter autograd yet",
                strict=strict,
                q=q,
                k=k,
                v=v,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                bias=bias,
                dist=dist,
                rbf=rbf,
                atom_types=atom_types,
                return_lse=return_lse,
            )
        bias_mode = _FIXED_K_BIAS_ESEN_RBF_TYPE
        rbf_weight = bias.rbf_weight.to(device=q.device, dtype=torch.float32).contiguous()
        centers = bias.centers.to(device=q.device, dtype=torch.float32).contiguous()
        gamma = bias.gamma.to(device=q.device, dtype=torch.float32).reshape(()).contiguous()
        heads_per_frame = int(bias.heads_per_frame)
        num_rbf = int(rbf_weight.shape[-1])
        cutoff = float(bias.cutoff)
        diag_zero = bool(bias.diag_zero)
        envelope_in_score = bool(bias.envelope_in_score)
        if bias.type_bias is not None:
            if atom_types is None:
                raise ValueError("type_bias requires atom_types")
            type_bias = bias.type_bias.to(device=q.device, dtype=torch.float32).contiguous()
            has_type_bias = True
            inferred_max_z = int(type_bias.shape[0]) - 1
            if max_atomic_number is None:
                max_atomic_number = inferred_max_z
            elif int(max_atomic_number) > inferred_max_z:
                raise ValueError(f"max_atomic_number={max_atomic_number} exceeds type_bias capacity {inferred_max_z}")
        if atom_types is None:
            atom_types_apply = torch.empty((1,), device=q.device, dtype=torch.long)
        else:
            atom_types_apply = atom_types.to(device=q.device, dtype=torch.long).contiguous()
    else:
        atom_types_apply = torch.empty((1,), device=q.device, dtype=torch.long)

    if max_atomic_number is None:
        max_atomic_number = 0
    if int(neighbor_idx.shape[-1]) > 128:
        raise ValueError("fixed-K local Triton attention currently supports K <= 128")
    if int(q.shape[-1]) > 256:
        raise ValueError("fixed-K local Triton attention currently supports head_dim <= 256")
    normalized_precision = normalize_platonic_attention_precision(precision)
    input_precision = _kernel_input_precision(normalized_precision)

    q_apply = q.contiguous()
    k_apply = k.contiguous()
    v_apply = v.contiguous()
    neighbor_idx_apply = neighbor_idx.to(device=q.device, dtype=torch.int64).contiguous()
    neighbor_mask_apply = neighbor_mask.to(device=q.device, dtype=torch.bool).contiguous()
    dist_apply = (
        dist.to(device=q.device, dtype=torch.float32).contiguous()
        if dist is not None
        else torch.empty((1, 1), device=q.device, dtype=torch.float32)
    )
    out = torch.empty_like(q_apply)
    lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    block_k = _next_power_of_2(int(neighbor_idx.shape[-1]))
    block_d = _next_power_of_2(int(q.shape[-1]))
    grid = (q.shape[0], q.shape[1])
    _fixed_k_local_attention_fwd_kernel[grid](
        q_apply,
        k_apply,
        v_apply,
        neighbor_idx_apply,
        neighbor_mask_apply,
        dist_apply,
        atom_types_apply,
        rbf_weight,
        type_bias,
        centers,
        gamma,
        out,
        lse,
        num_tokens=q.shape[0],
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        k_neighbors=neighbor_idx.shape[1],
        scale=1.0 / math.sqrt(q.shape[-1]),
        bias_mode=bias_mode,
        heads_per_frame=heads_per_frame,
        num_rbf=num_rbf,
        cutoff=cutoff,
        max_atomic_number=int(max_atomic_number),
        has_type_bias=has_type_bias,
        diag_zero=diag_zero,
        envelope_in_score=envelope_in_score,
        input_precision=input_precision,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 128 else 8,
        num_stages=2,
    )
    return (out, lse) if return_lse else out
