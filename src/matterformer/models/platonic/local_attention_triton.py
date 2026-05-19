from __future__ import annotations

import math

import torch

from matterformer.models.platonic.local_attention import (
    ESENEnvelopedRBFTypeFixedKBias,
    ESENFixedKLocalAttentionFeatures,
    ESENFixedKLocalBiasView,
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

try:
    import torch._dynamo as _dynamo
except Exception:  # pragma: no cover - Dynamo is optional in some torch builds.
    _dynamo = None


TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE = triton is not None and tl is not None

_FIXED_K_BIAS_NONE = 0
_FIXED_K_BIAS_ESEN_RBF_TYPE = 1
_FIXED_K_BIAS_ESEN_PRECOMPUTED = 2
_MAX_FULL_TILE_K = 64
_MAX_FULL_TILE_HEAD_DIM = 128
_MAX_CHUNKED_HEAD_DIM = 256
_CHUNKED_D = 64
_H_BLOCK_HEADS = 2


def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _requires_grad(*values: torch.Tensor | None) -> bool:
    return any(value is not None and bool(value.requires_grad) for value in values)


def _dynamo_disable(fn):
    return _dynamo.disable(fn) if _dynamo is not None else fn


def _fallback_or_raise(
    reason: str,
    *,
    strict: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: FixedKLocalBias | ESENFixedKLocalBiasView | None,
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


def _fixed_k_local_attention_forward_cuda_prepared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    dist: torch.Tensor,
    atom_types: torch.Tensor,
    rbf_weight: torch.Tensor,
    type_bias: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    pre_local_mask: torch.Tensor,
    pre_env_bias: torch.Tensor,
    pre_log_env: torch.Tensor,
    pre_rho_env: torch.Tensor,
    pre_type_base: torch.Tensor,
    bias_mode: int,
    max_atomic_number: int,
    heads_per_frame: int,
    num_rbf: int,
    cutoff: float,
    has_type_bias: bool,
    diag_zero: bool,
    envelope_in_score: bool,
    input_precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    block_k = _next_power_of_2(int(neighbor_idx.shape[-1]))
    use_hblock_forward = (
        int(heads_per_frame) == 1
        and int(q.shape[1]) >= _H_BLOCK_HEADS
        and int(q.shape[-1]) <= _MAX_FULL_TILE_HEAD_DIM
    )
    if use_hblock_forward:
        block_d = _next_power_of_2(int(q.shape[-1]))
        block_h = _H_BLOCK_HEADS
        grid = (q.shape[0], triton.cdiv(q.shape[1], block_h))
        _fixed_k_local_attention_fwd_kernel_hblock[grid](
            q,
            k,
            v,
            neighbor_idx,
            neighbor_mask,
            dist,
            atom_types,
            rbf_weight,
            type_bias,
            centers,
            gamma,
            pre_local_mask,
            pre_env_bias,
            pre_log_env,
            pre_rho_env,
            pre_type_base,
            out,
            lse,
            num_tokens=q.shape[0],
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            k_neighbors=neighbor_idx.shape[1],
            scale=1.0 / math.sqrt(q.shape[-1]),
            bias_mode=int(bias_mode),
            heads_per_frame=int(heads_per_frame),
            num_rbf=int(num_rbf),
            cutoff=float(cutoff),
            max_atomic_number=int(max_atomic_number),
            has_type_bias=bool(has_type_bias),
            diag_zero=bool(diag_zero),
            envelope_in_score=bool(envelope_in_score),
            input_precision=input_precision,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )
    elif int(q.shape[-1]) <= _MAX_FULL_TILE_HEAD_DIM:
        block_d = _next_power_of_2(int(q.shape[-1]))
        grid = (q.shape[0], q.shape[1])
        _fixed_k_local_attention_fwd_kernel[grid](
            q,
            k,
            v,
            neighbor_idx,
            neighbor_mask,
            dist,
            atom_types,
            rbf_weight,
            type_bias,
            centers,
            gamma,
            pre_local_mask,
            pre_env_bias,
            pre_log_env,
            pre_rho_env,
            pre_type_base,
            out,
            lse,
            num_tokens=q.shape[0],
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            k_neighbors=neighbor_idx.shape[1],
            scale=1.0 / math.sqrt(q.shape[-1]),
            bias_mode=int(bias_mode),
            heads_per_frame=int(heads_per_frame),
            num_rbf=int(num_rbf),
            cutoff=float(cutoff),
            max_atomic_number=int(max_atomic_number),
            has_type_bias=bool(has_type_bias),
            diag_zero=bool(diag_zero),
            envelope_in_score=bool(envelope_in_score),
            input_precision=input_precision,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )
    else:
        block_d = _CHUNKED_D
        grid = (q.shape[0], q.shape[1], triton.cdiv(q.shape[2], block_d))
        _fixed_k_local_attention_fwd_kernel_chunked_d[grid](
            q,
            k,
            v,
            neighbor_idx,
            neighbor_mask,
            dist,
            atom_types,
            rbf_weight,
            type_bias,
            centers,
            gamma,
            pre_local_mask,
            pre_env_bias,
            pre_log_env,
            pre_rho_env,
            pre_type_base,
            out,
            lse,
            num_tokens=q.shape[0],
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            k_neighbors=neighbor_idx.shape[1],
            scale=1.0 / math.sqrt(q.shape[-1]),
            bias_mode=int(bias_mode),
            heads_per_frame=int(heads_per_frame),
            num_rbf=int(num_rbf),
            cutoff=float(cutoff),
            max_atomic_number=int(max_atomic_number),
            has_type_bias=bool(has_type_bias),
            diag_zero=bool(diag_zero),
            envelope_in_score=bool(envelope_in_score),
            input_precision=input_precision,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )
    return out, lse


def _fixed_k_local_attention_backward_cuda_prepared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    dist: torch.Tensor,
    atom_types: torch.Tensor,
    rbf_weight: torch.Tensor,
    type_bias: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    pre_local_mask: torch.Tensor,
    pre_env_bias: torch.Tensor,
    pre_log_env: torch.Tensor,
    pre_rho_env: torch.Tensor,
    pre_type_base: torch.Tensor,
    bias_mode: int,
    max_atomic_number: int,
    heads_per_frame: int,
    num_rbf: int,
    cutoff: float,
    has_type_bias: bool,
    diag_zero: bool,
    envelope_in_score: bool,
    input_precision: str,
    need_drbf: bool,
    need_dtype_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    dout = dout.contiguous()
    dq_acc = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    dk_acc = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv_acc = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    is_esen_bias = int(bias_mode) in {_FIXED_K_BIAS_ESEN_RBF_TYPE, _FIXED_K_BIAS_ESEN_PRECOMPUTED}
    d_rbf_weight = (
        torch.zeros(rbf_weight.shape, device=rbf_weight.device, dtype=torch.float32)
        if is_esen_bias and bool(need_drbf)
        else torch.empty((1,), device=q.device, dtype=torch.float32)
    )
    d_type_bias = (
        torch.zeros(type_bias.shape, device=type_bias.device, dtype=torch.float32)
        if is_esen_bias and bool(has_type_bias) and bool(need_dtype_bias)
        else torch.empty((1,), device=q.device, dtype=torch.float32)
    )
    block_k = _next_power_of_2(int(neighbor_idx.shape[-1]))
    block_d = min(_CHUNKED_D, _next_power_of_2(int(q.shape[-1])))
    grid = (q.shape[0], q.shape[1])
    _fixed_k_local_attention_bwd_kernel_streaming_d[grid](
        q,
        k,
        v,
        lse,
        dout,
        neighbor_idx,
        neighbor_mask,
        dist,
        atom_types,
        rbf_weight,
        type_bias,
        centers,
        gamma,
        pre_local_mask,
        pre_env_bias,
        pre_log_env,
        pre_rho_env,
        pre_type_base,
        dq_acc,
        dk_acc,
        dv_acc,
        d_rbf_weight,
        d_type_bias,
        num_tokens=q.shape[0],
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        k_neighbors=neighbor_idx.shape[1],
        scale=1.0 / math.sqrt(q.shape[-1]),
        bias_mode=int(bias_mode),
        heads_per_frame=int(heads_per_frame),
        num_rbf=int(num_rbf),
        cutoff=float(cutoff),
        max_atomic_number=int(max_atomic_number),
        has_type_bias=bool(has_type_bias),
        diag_zero=bool(diag_zero),
        envelope_in_score=bool(envelope_in_score),
        need_drbf=bool(need_drbf),
        need_dtype_bias=bool(need_dtype_bias),
        input_precision=input_precision,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    d_rbf_return = (
        d_rbf_weight.to(dtype=rbf_weight.dtype)
        if is_esen_bias and bool(need_drbf)
        else None
    )
    d_type_return = (
        d_type_bias.to(dtype=type_bias.dtype)
        if is_esen_bias and bool(has_type_bias) and bool(need_dtype_bias)
        else None
    )
    return dq_acc.to(dtype=q.dtype), dk_acc.to(dtype=k.dtype), dv_acc.to(dtype=v.dtype), d_rbf_return, d_type_return


class _FixedKLocalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        dist: torch.Tensor,
        atom_types: torch.Tensor,
        rbf_weight: torch.Tensor,
        type_bias: torch.Tensor,
        centers: torch.Tensor,
        gamma: torch.Tensor,
        pre_local_mask: torch.Tensor,
        pre_env_bias: torch.Tensor,
        pre_log_env: torch.Tensor,
        pre_rho_env: torch.Tensor,
        pre_type_base: torch.Tensor,
        bias_mode: int,
        max_atomic_number: int,
        heads_per_frame: int,
        num_rbf: int,
        cutoff: float,
        has_type_bias: bool,
        diag_zero: bool,
        envelope_in_score: bool,
        input_precision: str,
        normalized_precision: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = _fixed_k_local_attention_forward_cuda_prepared(
            q,
            k,
            v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            dist=dist,
            atom_types=atom_types,
            rbf_weight=rbf_weight,
            type_bias=type_bias,
            centers=centers,
            gamma=gamma,
            pre_local_mask=pre_local_mask,
            pre_env_bias=pre_env_bias,
            pre_log_env=pre_log_env,
            pre_rho_env=pre_rho_env,
            pre_type_base=pre_type_base,
            bias_mode=int(bias_mode),
            max_atomic_number=int(max_atomic_number),
            heads_per_frame=int(heads_per_frame),
            num_rbf=int(num_rbf),
            cutoff=float(cutoff),
            has_type_bias=bool(has_type_bias),
            diag_zero=bool(diag_zero),
            envelope_in_score=bool(envelope_in_score),
            input_precision=input_precision,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            lse,
            neighbor_idx,
            neighbor_mask,
            dist,
            atom_types,
            rbf_weight,
            type_bias,
            centers,
            gamma,
            pre_local_mask,
            pre_env_bias,
            pre_log_env,
            pre_rho_env,
            pre_type_base,
        )
        ctx.mark_non_differentiable(lse)
        ctx.bias_mode = int(bias_mode)
        ctx.max_atomic_number = int(max_atomic_number)
        ctx.heads_per_frame = int(heads_per_frame)
        ctx.num_rbf = int(num_rbf)
        ctx.cutoff = float(cutoff)
        ctx.has_type_bias = bool(has_type_bias)
        ctx.diag_zero = bool(diag_zero)
        ctx.envelope_in_score = bool(envelope_in_score)
        ctx.input_precision = input_precision
        ctx.normalized_precision = normalized_precision
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: torch.Tensor | None = None):
        del dlse
        (
            q,
            k,
            v,
            lse,
            neighbor_idx,
            neighbor_mask,
            dist,
            atom_types,
            rbf_weight,
            type_bias,
            centers,
            gamma,
            pre_local_mask,
            pre_env_bias,
            pre_log_env,
            pre_rho_env,
            pre_type_base,
        ) = ctx.saved_tensors
        need_drbf = bool(ctx.needs_input_grad[7])
        need_dtype_bias = bool(ctx.needs_input_grad[8]) and bool(ctx.has_type_bias)
        dq, dk, dv, d_rbf_weight, d_type_bias = _fixed_k_local_attention_backward_cuda_prepared(
            q,
            k,
            v,
            lse,
            dout,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            dist=dist,
            atom_types=atom_types,
            rbf_weight=rbf_weight,
            type_bias=type_bias,
            centers=centers,
            gamma=gamma,
            pre_local_mask=pre_local_mask,
            pre_env_bias=pre_env_bias,
            pre_log_env=pre_log_env,
            pre_rho_env=pre_rho_env,
            pre_type_base=pre_type_base,
            bias_mode=ctx.bias_mode,
            max_atomic_number=ctx.max_atomic_number,
            heads_per_frame=ctx.heads_per_frame,
            num_rbf=ctx.num_rbf,
            cutoff=ctx.cutoff,
            has_type_bias=ctx.has_type_bias,
            diag_zero=ctx.diag_zero,
            envelope_in_score=ctx.envelope_in_score,
            input_precision=ctx.input_precision,
            need_drbf=need_drbf,
            need_dtype_bias=need_dtype_bias,
        )
        return (
            dq,
            dk,
            dv,
            None,  # neighbor_idx
            None,  # neighbor_mask
            None,  # dist
            None,  # atom_types
            d_rbf_weight,
            d_type_bias,
            None,  # centers
            None,  # gamma
            None,  # pre_local_mask
            None,  # pre_env_bias
            None,  # pre_log_env
            None,  # pre_rho_env
            None,  # pre_type_base
            None,  # bias_mode
            None,  # max_atomic_number
            None,  # heads_per_frame
            None,  # num_rbf
            None,  # cutoff
            None,  # has_type_bias
            None,  # diag_zero
            None,  # envelope_in_score
            None,  # input_precision
            None,  # normalized_precision
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
        pre_local_mask_ptr,
        pre_env_bias_ptr,
        pre_log_env_ptr,
        pre_rho_env_ptr,
        pre_type_base_ptr,
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
        neigh_idx = tl.where(neigh_valid, neigh_idx, 0)
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
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            local = (tl.load(pre_local_mask_ptr + pre_base, mask=k_mask, other=0) > 0) & neigh_valid & k_mask
            env_bias = tl.load(pre_env_bias_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            scores += tl.load(pre_log_env_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            subhead = head_idx % heads_per_frame
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                rho_env = tl.load(
                    pre_rho_env_ptr + (pre_base * num_rbf + rb),
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                raw_bias += weight * rho_env
            scores += raw_bias
            if has_type_bias:
                type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                type_index = type_base + subhead
                scores += env_bias * tl.load(type_bias_ptr + type_index, mask=k_mask, other=0.0).to(tl.float32)
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

    @triton.jit
    def _fixed_k_local_attention_fwd_kernel_hblock(
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
        pre_local_mask_ptr,
        pre_env_bias_ptr,
        pre_log_env_ptr,
        pre_rho_env_ptr,
        pre_type_base_ptr,
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
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_block = tl.program_id(1) * BLOCK_H
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        k_mask = offs_k < k_neighbors
        d_mask = offs_d < head_dim

        n_base = token_idx * k_neighbors + offs_k
        neigh_idx = tl.load(neighbor_idx_ptr + n_base, mask=k_mask, other=0).to(tl.int64)
        neigh_valid = (tl.load(neighbor_mask_ptr + n_base, mask=k_mask, other=0) > 0) & k_mask
        neigh_idx = tl.where(neigh_valid, neigh_idx, 0)

        shared_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
        pair_valid = neigh_valid

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

            gamma = tl.load(gamma_ptr).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                raw_bias += weight * rho

            if has_type_bias:
                zi = tl.load(atom_type_ptr + token_idx).to(tl.int64)
                zj = tl.load(atom_type_ptr + neigh_idx, mask=neigh_valid, other=0).to(tl.int64)
                zi = tl.minimum(tl.maximum(zi, 0), max_atomic_number)
                zj = tl.minimum(tl.maximum(zj, 0), max_atomic_number)
                zdim = max_atomic_number + 1
                type_index = (zi * zdim + zj) * heads_per_frame
                raw_bias += tl.load(type_bias_ptr + type_index, mask=neigh_valid, other=0.0).to(tl.float32)

            shared_bias += env_bias * raw_bias
            if envelope_in_score:
                env_score = tl.where(same & neigh_valid, 1.0, env)
                shared_bias += tl.log(tl.maximum(env_score, 1.0e-20))
            pair_valid = local
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            local = (tl.load(pre_local_mask_ptr + pre_base, mask=k_mask, other=0) > 0) & neigh_valid & k_mask
            env_bias = tl.load(pre_env_bias_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            shared_bias += tl.load(pre_log_env_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                rho_env = tl.load(
                    pre_rho_env_ptr + (pre_base * num_rbf + rb),
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + rb).to(tl.float32)
                raw_bias += weight * rho_env
            shared_bias += raw_bias
            if has_type_bias:
                type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                shared_bias += env_bias * tl.load(type_bias_ptr + type_base, mask=k_mask, other=0.0).to(tl.float32)
            pair_valid = local

        for hh in range(0, BLOCK_H):
            head_idx = head_block + hh
            h_valid = head_idx < num_heads
            q = tl.load(
                q_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
                mask=h_valid & d_mask,
                other=0.0,
            ).to(tl.float32)
            k_tile = tl.load(
                k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
                mask=h_valid & pair_valid[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(k_tile.to(tl.float32) * q[None, :], axis=1) * scale
            scores += shared_bias
            scores = tl.where(pair_valid, scores, -float("inf"))

            row_max = tl.max(scores, axis=0)
            has_any = row_max > -float("inf")
            row_max_safe = tl.where(has_any, row_max, 0.0)
            probs = tl.exp(scores - row_max_safe)
            probs = tl.where(pair_valid, probs, 0.0)
            denom = tl.sum(probs, axis=0)
            probs = probs / tl.maximum(denom, 1.0e-20)

            v_tile = tl.load(
                v_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
                mask=h_valid & pair_valid[:, None] & d_mask[None, :],
                other=0.0,
            )
            out = tl.sum(probs[:, None] * v_tile.to(tl.float32), axis=0)
            tl.store(
                out_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
                out,
                mask=h_valid & d_mask,
            )
            lse = tl.where(has_any, row_max + tl.log(tl.maximum(denom, 1.0e-20)), -float("inf"))
            tl.store(lse_ptr + token_idx * num_heads + head_idx, lse, mask=h_valid)

    @triton.jit
    def _fixed_k_local_attention_fwd_kernel_chunked_d(
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
        pre_local_mask_ptr,
        pre_env_bias_ptr,
        pre_log_env_ptr,
        pre_rho_env_ptr,
        pre_type_base_ptr,
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
        d_chunk_idx = tl.program_id(2)
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        k_mask = offs_k < k_neighbors

        n_base = token_idx * k_neighbors + offs_k
        neigh_idx = tl.load(neighbor_idx_ptr + n_base, mask=k_mask, other=0).to(tl.int64)
        neigh_valid = (tl.load(neighbor_mask_ptr + n_base, mask=k_mask, other=0) > 0) & k_mask
        neigh_idx = tl.where(neigh_valid, neigh_idx, 0)

        scores = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for d_start in range(0, head_dim, BLOCK_D):
            d_abs = d_start + offs_d
            d_mask = d_abs < head_dim
            q = tl.load(
                q_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            k_tile = tl.load(
                k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                mask=neigh_valid[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores += tl.sum(k_tile.to(tl.float32) * q[None, :], axis=1)
        scores *= scale

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
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            local = (tl.load(pre_local_mask_ptr + pre_base, mask=k_mask, other=0) > 0) & neigh_valid & k_mask
            env_bias = tl.load(pre_env_bias_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            scores += tl.load(pre_log_env_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            subhead = head_idx % heads_per_frame
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                rho_env = tl.load(
                    pre_rho_env_ptr + (pre_base * num_rbf + rb),
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                raw_bias += weight * rho_env
            scores += raw_bias
            if has_type_bias:
                type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                type_index = type_base + subhead
                scores += env_bias * tl.load(type_bias_ptr + type_index, mask=k_mask, other=0.0).to(tl.float32)
            neigh_valid = local

        scores = tl.where(neigh_valid, scores, -float("inf"))
        row_max = tl.max(scores, axis=0)
        has_any = row_max > -float("inf")
        row_max_safe = tl.where(has_any, row_max, 0.0)
        probs = tl.exp(scores - row_max_safe)
        probs = tl.where(neigh_valid, probs, 0.0)
        denom = tl.sum(probs, axis=0)
        probs = probs / tl.maximum(denom, 1.0e-20)

        d_abs_out = d_chunk_idx * BLOCK_D + offs_d
        d_out_mask = d_abs_out < head_dim
        v_tile = tl.load(
            v_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs_out[None, :]),
            mask=neigh_valid[:, None] & d_out_mask[None, :],
            other=0.0,
        )
        out = tl.sum(probs[:, None] * v_tile.to(tl.float32), axis=0)
        tl.store(
            out_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs_out),
            out,
            mask=d_out_mask,
        )
        if d_chunk_idx == 0:
            lse = tl.where(has_any, row_max + tl.log(tl.maximum(denom, 1.0e-20)), -float("inf"))
            tl.store(lse_ptr + token_idx * num_heads + head_idx, lse)

    @triton.jit
    def _fixed_k_local_attention_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        lse_ptr,
        dout_ptr,
        neighbor_idx_ptr,
        neighbor_mask_ptr,
        dist_ptr,
        atom_type_ptr,
        rbf_weight_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        pre_local_mask_ptr,
        pre_env_bias_ptr,
        pre_log_env_ptr,
        pre_rho_env_ptr,
        pre_type_base_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_type_bias_ptr,
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
        need_drbf: tl.constexpr,
        need_dtype_bias: tl.constexpr,
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
        raw_neigh_idx = tl.load(neighbor_idx_ptr + n_base, mask=k_mask, other=0).to(tl.int64)
        row_mask = (tl.load(neighbor_mask_ptr + n_base, mask=k_mask, other=0) > 0) & k_mask
        neigh_idx = tl.where(row_mask, raw_neigh_idx, 0)

        q = tl.load(
            q_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        dout = tl.load(
            dout_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            v_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.sum(k_tile.to(tl.float32) * q[None, :], axis=1) * scale
        pair_valid = row_mask
        dist = tl.zeros((BLOCK_K,), dtype=tl.float32)
        env_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
        type_index = tl.zeros((BLOCK_K,), dtype=tl.int64)
        subhead = head_idx % heads_per_frame

        if bias_mode == 1:
            dist = tl.load(dist_ptr + n_base, mask=k_mask, other=0.0).to(tl.float32)
            local = row_mask & (dist < cutoff)
            same = neigh_idx == token_idx
            local = local | (same & row_mask)

            x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
            x2 = x * x
            x3 = x2 * x
            env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
            env = tl.where(dist < cutoff, env, 0.0)
            env = tl.where(row_mask, env, 0.0)

            env_bias = env
            if diag_zero:
                env_bias = tl.where(same, 0.0, env_bias)

            gamma = tl.load(gamma_ptr).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                raw_bias += weight * rho

            if has_type_bias:
                zi = tl.load(atom_type_ptr + token_idx).to(tl.int64)
                zj = tl.load(atom_type_ptr + neigh_idx, mask=row_mask, other=0).to(tl.int64)
                zi = tl.minimum(tl.maximum(zi, 0), max_atomic_number)
                zj = tl.minimum(tl.maximum(zj, 0), max_atomic_number)
                zdim = max_atomic_number + 1
                type_index = ((zi * zdim + zj) * heads_per_frame + subhead)
                raw_bias += tl.load(type_bias_ptr + type_index, mask=row_mask, other=0.0).to(tl.float32)

            scores += env_bias * raw_bias
            if envelope_in_score:
                env_score = tl.where(same & row_mask, 1.0, env)
                scores += tl.log(tl.maximum(env_score, 1.0e-20))
            pair_valid = local
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            local = (tl.load(pre_local_mask_ptr + pre_base, mask=k_mask, other=0) > 0) & row_mask & k_mask
            env_bias = tl.load(pre_env_bias_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            scores += tl.load(pre_log_env_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                rho_env = tl.load(
                    pre_rho_env_ptr + (pre_base * num_rbf + rb),
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                raw_bias += weight * rho_env
            scores += raw_bias
            if has_type_bias:
                type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                type_index = type_base + subhead
                scores += env_bias * tl.load(type_bias_ptr + type_index, mask=k_mask, other=0.0).to(tl.float32)
            pair_valid = local

        scores = tl.where(pair_valid, scores, -float("inf"))
        lse = tl.load(lse_ptr + token_idx * num_heads + head_idx, mask=True, other=-float("inf")).to(tl.float32)
        row_valid = lse > -float("inf")
        lse_safe = tl.where(row_valid, lse, 0.0)
        probs = tl.exp(scores - lse_safe)
        probs = tl.where(pair_valid & row_valid, probs, 0.0)

        dp = tl.sum(v_tile.to(tl.float32) * dout[None, :], axis=1)
        delta = tl.sum(probs * dp, axis=0)
        ds = probs * (dp - delta)
        ds = tl.where(pair_valid & row_valid, ds, 0.0)

        dq = tl.sum(ds[:, None] * k_tile.to(tl.float32), axis=0) * scale
        tl.store(
            dq_ptr + ((token_idx * num_heads + head_idx) * head_dim + offs_d),
            dq,
            mask=d_mask,
        )

        dk_update = ds[:, None] * q[None, :] * scale
        dv_update = probs[:, None] * dout[None, :]
        tl.atomic_add(
            dk_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
            dk_update,
            mask=pair_valid[:, None] & row_valid & d_mask[None, :],
            sem="relaxed",
        )
        tl.atomic_add(
            dv_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + offs_d[None, :]),
            dv_update,
            mask=pair_valid[:, None] & row_valid & d_mask[None, :],
            sem="relaxed",
        )
        if bias_mode == 1:
            if need_drbf:
                gamma = tl.load(gamma_ptr).to(tl.float32)
                for rb in range(0, num_rbf):
                    center = tl.load(centers_ptr + rb).to(tl.float32)
                    rho = tl.exp(-gamma * (dist - center) * (dist - center))
                    grad_w = tl.sum(ds * env_bias * rho, axis=0)
                    tl.atomic_add(
                        d_rbf_weight_ptr + subhead * num_rbf + rb,
                        grad_w,
                        sem="relaxed",
                    )
            if need_dtype_bias:
                if has_type_bias:
                    tl.atomic_add(
                        d_type_bias_ptr + type_index,
                        ds * env_bias,
                        mask=pair_valid & row_valid,
                        sem="relaxed",
                    )
        if bias_mode == 2:
            if need_drbf:
                pre_base = token_idx * k_neighbors + offs_k
                for rb in range(0, num_rbf):
                    rho_env = tl.load(
                        pre_rho_env_ptr + (pre_base * num_rbf + rb),
                        mask=k_mask,
                        other=0.0,
                    ).to(tl.float32)
                    grad_w = tl.sum(ds * rho_env, axis=0)
                    tl.atomic_add(
                        d_rbf_weight_ptr + subhead * num_rbf + rb,
                        grad_w,
                        sem="relaxed",
                    )
            if need_dtype_bias:
                if has_type_bias:
                    tl.atomic_add(
                        d_type_bias_ptr + type_index,
                        ds * env_bias,
                        mask=pair_valid & row_valid,
                        sem="relaxed",
                    )

    @triton.jit
    def _fixed_k_local_attention_bwd_kernel_chunked_d(
        q_ptr,
        k_ptr,
        v_ptr,
        lse_ptr,
        dout_ptr,
        neighbor_idx_ptr,
        neighbor_mask_ptr,
        dist_ptr,
        atom_type_ptr,
        rbf_weight_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        pre_local_mask_ptr,
        pre_env_bias_ptr,
        pre_log_env_ptr,
        pre_rho_env_ptr,
        pre_type_base_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_type_bias_ptr,
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
        need_drbf: tl.constexpr,
        need_dtype_bias: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        d_chunk_idx = tl.program_id(2)
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        k_mask = offs_k < k_neighbors

        n_base = token_idx * k_neighbors + offs_k
        raw_neigh_idx = tl.load(neighbor_idx_ptr + n_base, mask=k_mask, other=0).to(tl.int64)
        row_mask = (tl.load(neighbor_mask_ptr + n_base, mask=k_mask, other=0) > 0) & k_mask
        neigh_idx = tl.where(row_mask, raw_neigh_idx, 0)

        scores = tl.zeros((BLOCK_K,), dtype=tl.float32)
        dp = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for d_start in range(0, head_dim, BLOCK_D):
            d_abs = d_start + offs_d
            d_mask = d_abs < head_dim
            q = tl.load(
                q_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            dout = tl.load(
                dout_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            k_tile = tl.load(
                k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v_tile = tl.load(
                v_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(k_tile * q[None, :], axis=1)
            dp += tl.sum(v_tile * dout[None, :], axis=1)
        scores *= scale

        pair_valid = row_mask
        dist = tl.zeros((BLOCK_K,), dtype=tl.float32)
        env_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
        type_index = tl.zeros((BLOCK_K,), dtype=tl.int64)
        subhead = head_idx % heads_per_frame

        if bias_mode == 1:
            dist = tl.load(dist_ptr + n_base, mask=k_mask, other=0.0).to(tl.float32)
            local = row_mask & (dist < cutoff)
            same = neigh_idx == token_idx
            local = local | (same & row_mask)

            x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
            x2 = x * x
            x3 = x2 * x
            env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
            env = tl.where(dist < cutoff, env, 0.0)
            env = tl.where(row_mask, env, 0.0)

            env_bias = env
            if diag_zero:
                env_bias = tl.where(same, 0.0, env_bias)

            gamma = tl.load(gamma_ptr).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                raw_bias += weight * rho

            if has_type_bias:
                zi = tl.load(atom_type_ptr + token_idx).to(tl.int64)
                zj = tl.load(atom_type_ptr + neigh_idx, mask=row_mask, other=0).to(tl.int64)
                zi = tl.minimum(tl.maximum(zi, 0), max_atomic_number)
                zj = tl.minimum(tl.maximum(zj, 0), max_atomic_number)
                zdim = max_atomic_number + 1
                type_index = ((zi * zdim + zj) * heads_per_frame + subhead)
                raw_bias += tl.load(type_bias_ptr + type_index, mask=row_mask, other=0.0).to(tl.float32)

            scores += env_bias * raw_bias
            if envelope_in_score:
                env_score = tl.where(same & row_mask, 1.0, env)
                scores += tl.log(tl.maximum(env_score, 1.0e-20))
            pair_valid = local
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            local = (tl.load(pre_local_mask_ptr + pre_base, mask=k_mask, other=0) > 0) & row_mask & k_mask
            env_bias = tl.load(pre_env_bias_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            scores += tl.load(pre_log_env_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                rho_env = tl.load(
                    pre_rho_env_ptr + (pre_base * num_rbf + rb),
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                raw_bias += weight * rho_env
            scores += raw_bias
            if has_type_bias:
                type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                type_index = type_base + subhead
                scores += env_bias * tl.load(type_bias_ptr + type_index, mask=k_mask, other=0.0).to(tl.float32)
            pair_valid = local

        scores = tl.where(pair_valid, scores, -float("inf"))
        lse = tl.load(lse_ptr + token_idx * num_heads + head_idx, mask=True, other=-float("inf")).to(tl.float32)
        row_valid = lse > -float("inf")
        lse_safe = tl.where(row_valid, lse, 0.0)
        probs = tl.exp(scores - lse_safe)
        probs = tl.where(pair_valid & row_valid, probs, 0.0)
        delta = tl.sum(probs * dp, axis=0)
        ds = probs * (dp - delta)
        ds = tl.where(pair_valid & row_valid, ds, 0.0)

        d_abs_out = d_chunk_idx * BLOCK_D + offs_d
        d_out_mask = d_abs_out < head_dim
        q_chunk = tl.load(
            q_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs_out),
            mask=d_out_mask,
            other=0.0,
        ).to(tl.float32)
        dout_chunk = tl.load(
            dout_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs_out),
            mask=d_out_mask,
            other=0.0,
        ).to(tl.float32)
        k_chunk = tl.load(
            k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs_out[None, :]),
            mask=row_mask[:, None] & d_out_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        dq_chunk = tl.sum(ds[:, None] * k_chunk, axis=0) * scale
        tl.store(
            dq_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs_out),
            dq_chunk,
            mask=d_out_mask,
        )

        dk_update = ds[:, None] * q_chunk[None, :] * scale
        dv_update = probs[:, None] * dout_chunk[None, :]
        tl.atomic_add(
            dk_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs_out[None, :]),
            dk_update,
            mask=pair_valid[:, None] & row_valid & d_out_mask[None, :],
            sem="relaxed",
        )
        tl.atomic_add(
            dv_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs_out[None, :]),
            dv_update,
            mask=pair_valid[:, None] & row_valid & d_out_mask[None, :],
            sem="relaxed",
        )

        if d_chunk_idx == 0:
            if bias_mode == 1:
                if need_drbf:
                    gamma = tl.load(gamma_ptr).to(tl.float32)
                    for rb in range(0, num_rbf):
                        center = tl.load(centers_ptr + rb).to(tl.float32)
                        rho = tl.exp(-gamma * (dist - center) * (dist - center))
                        grad_w = tl.sum(ds * env_bias * rho, axis=0)
                        tl.atomic_add(
                            d_rbf_weight_ptr + subhead * num_rbf + rb,
                            grad_w,
                            sem="relaxed",
                        )
                if need_dtype_bias:
                    if has_type_bias:
                        tl.atomic_add(
                            d_type_bias_ptr + type_index,
                            ds * env_bias,
                            mask=pair_valid & row_valid,
                            sem="relaxed",
                        )
            if bias_mode == 2:
                pre_base = token_idx * k_neighbors + offs_k
                if need_drbf:
                    for rb in range(0, num_rbf):
                        rho_env = tl.load(
                            pre_rho_env_ptr + (pre_base * num_rbf + rb),
                            mask=k_mask,
                            other=0.0,
                        ).to(tl.float32)
                        grad_w = tl.sum(ds * rho_env, axis=0)
                        tl.atomic_add(
                            d_rbf_weight_ptr + subhead * num_rbf + rb,
                            grad_w,
                            sem="relaxed",
                        )
                if need_dtype_bias:
                    if has_type_bias:
                        type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                        type_index = type_base + subhead
                        tl.atomic_add(
                            d_type_bias_ptr + type_index,
                            ds * env_bias,
                            mask=pair_valid & row_valid,
                            sem="relaxed",
                        )


    @triton.jit
    def _fixed_k_local_attention_bwd_kernel_streaming_d(
        q_ptr,
        k_ptr,
        v_ptr,
        lse_ptr,
        dout_ptr,
        neighbor_idx_ptr,
        neighbor_mask_ptr,
        dist_ptr,
        atom_type_ptr,
        rbf_weight_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        pre_local_mask_ptr,
        pre_env_bias_ptr,
        pre_log_env_ptr,
        pre_rho_env_ptr,
        pre_type_base_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_type_bias_ptr,
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
        need_drbf: tl.constexpr,
        need_dtype_bias: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        k_mask = offs_k < k_neighbors

        n_base = token_idx * k_neighbors + offs_k
        raw_neigh_idx = tl.load(neighbor_idx_ptr + n_base, mask=k_mask, other=0).to(tl.int64)
        row_mask = (tl.load(neighbor_mask_ptr + n_base, mask=k_mask, other=0) > 0) & k_mask
        neigh_idx = tl.where(row_mask, raw_neigh_idx, 0)

        scores = tl.zeros((BLOCK_K,), dtype=tl.float32)
        dp = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for d_start in range(0, head_dim, BLOCK_D):
            d_abs = d_start + offs_d
            d_mask = d_abs < head_dim
            q_chunk = tl.load(
                q_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            dout_chunk = tl.load(
                dout_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            k_chunk = tl.load(
                k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v_chunk = tl.load(
                v_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(k_chunk * q_chunk[None, :], axis=1)
            dp += tl.sum(v_chunk * dout_chunk[None, :], axis=1)
        scores *= scale

        pair_valid = row_mask
        dist = tl.zeros((BLOCK_K,), dtype=tl.float32)
        env_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
        type_index = tl.zeros((BLOCK_K,), dtype=tl.int64)
        subhead = head_idx % heads_per_frame

        if bias_mode == 1:
            dist = tl.load(dist_ptr + n_base, mask=k_mask, other=0.0).to(tl.float32)
            local = row_mask & (dist < cutoff)
            same = neigh_idx == token_idx
            local = local | (same & row_mask)

            x = tl.minimum(tl.maximum(dist / cutoff, 0.0), 1.0)
            x2 = x * x
            x3 = x2 * x
            env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
            env = tl.where(dist < cutoff, env, 0.0)
            env = tl.where(row_mask, env, 0.0)

            env_bias = env
            if diag_zero:
                env_bias = tl.where(same, 0.0, env_bias)

            gamma = tl.load(gamma_ptr).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                raw_bias += weight * rho

            if has_type_bias:
                zi = tl.load(atom_type_ptr + token_idx).to(tl.int64)
                zj = tl.load(atom_type_ptr + neigh_idx, mask=row_mask, other=0).to(tl.int64)
                zi = tl.minimum(tl.maximum(zi, 0), max_atomic_number)
                zj = tl.minimum(tl.maximum(zj, 0), max_atomic_number)
                zdim = max_atomic_number + 1
                type_index = ((zi * zdim + zj) * heads_per_frame + subhead)
                raw_bias += tl.load(type_bias_ptr + type_index, mask=row_mask, other=0.0).to(tl.float32)

            scores += env_bias * raw_bias
            if envelope_in_score:
                env_score = tl.where(same & row_mask, 1.0, env)
                scores += tl.log(tl.maximum(env_score, 1.0e-20))
            pair_valid = local
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            local = (tl.load(pre_local_mask_ptr + pre_base, mask=k_mask, other=0) > 0) & row_mask & k_mask
            env_bias = tl.load(pre_env_bias_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            scores += tl.load(pre_log_env_ptr + pre_base, mask=k_mask, other=0.0).to(tl.float32)
            raw_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for rb in range(0, num_rbf):
                rho_env = tl.load(
                    pre_rho_env_ptr + (pre_base * num_rbf + rb),
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                raw_bias += weight * rho_env
            scores += raw_bias
            if has_type_bias:
                type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                type_index = type_base + subhead
                scores += env_bias * tl.load(type_bias_ptr + type_index, mask=k_mask, other=0.0).to(tl.float32)
            pair_valid = local

        scores = tl.where(pair_valid, scores, -float("inf"))
        lse = tl.load(lse_ptr + token_idx * num_heads + head_idx, mask=True, other=-float("inf")).to(tl.float32)
        row_valid = lse > -float("inf")
        lse_safe = tl.where(row_valid, lse, 0.0)
        probs = tl.exp(scores - lse_safe)
        probs = tl.where(pair_valid & row_valid, probs, 0.0)
        delta = tl.sum(probs * dp, axis=0)
        ds = probs * (dp - delta)
        ds = tl.where(pair_valid & row_valid, ds, 0.0)

        for d_start in range(0, head_dim, BLOCK_D):
            d_abs = d_start + offs_d
            d_mask = d_abs < head_dim
            q_chunk = tl.load(
                q_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            dout_chunk = tl.load(
                dout_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            k_chunk = tl.load(
                k_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            dq_chunk = tl.sum(ds[:, None] * k_chunk, axis=0) * scale
            tl.store(
                dq_ptr + ((token_idx * num_heads + head_idx) * head_dim + d_abs),
                dq_chunk,
                mask=d_mask,
            )

            dk_update = ds[:, None] * q_chunk[None, :] * scale
            dv_update = probs[:, None] * dout_chunk[None, :]
            tl.atomic_add(
                dk_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                dk_update,
                mask=pair_valid[:, None] & row_valid & d_mask[None, :],
                sem="relaxed",
            )
            tl.atomic_add(
                dv_ptr + ((neigh_idx[:, None] * num_heads + head_idx) * head_dim + d_abs[None, :]),
                dv_update,
                mask=pair_valid[:, None] & row_valid & d_mask[None, :],
                sem="relaxed",
            )

        if bias_mode == 1:
            if need_drbf:
                gamma = tl.load(gamma_ptr).to(tl.float32)
                for rb in range(0, num_rbf):
                    center = tl.load(centers_ptr + rb).to(tl.float32)
                    rho = tl.exp(-gamma * (dist - center) * (dist - center))
                    grad_w = tl.sum(ds * env_bias * rho, axis=0)
                    tl.atomic_add(
                        d_rbf_weight_ptr + subhead * num_rbf + rb,
                        grad_w,
                        sem="relaxed",
                    )
            if need_dtype_bias:
                if has_type_bias:
                    tl.atomic_add(
                        d_type_bias_ptr + type_index,
                        ds * env_bias,
                        mask=pair_valid & row_valid,
                        sem="relaxed",
                    )
        if bias_mode == 2:
            pre_base = token_idx * k_neighbors + offs_k
            if need_drbf:
                for rb in range(0, num_rbf):
                    rho_env = tl.load(
                        pre_rho_env_ptr + (pre_base * num_rbf + rb),
                        mask=k_mask,
                        other=0.0,
                    ).to(tl.float32)
                    grad_w = tl.sum(ds * rho_env, axis=0)
                    tl.atomic_add(
                        d_rbf_weight_ptr + subhead * num_rbf + rb,
                        grad_w,
                        sem="relaxed",
                    )
            if need_dtype_bias:
                if has_type_bias:
                    type_base = tl.load(pre_type_base_ptr + pre_base, mask=k_mask, other=0).to(tl.int64)
                    type_index = type_base + subhead
                    tl.atomic_add(
                        d_type_bias_ptr + type_index,
                        ds * env_bias,
                        mask=pair_valid & row_valid,
                        sem="relaxed",
                    )


@_dynamo_disable
def fixed_k_local_attention_triton_prepared_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    dist: torch.Tensor,
    atom_types: torch.Tensor,
    rbf_weight: torch.Tensor,
    type_bias: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    pre_local_mask: torch.Tensor,
    pre_env_bias: torch.Tensor,
    pre_log_env: torch.Tensor,
    pre_rho_env: torch.Tensor,
    pre_type_base: torch.Tensor,
    bias_mode: int,
    max_atomic_number: int,
    heads_per_frame: int,
    num_rbf: int,
    cutoff: float,
    has_type_bias: bool,
    diag_zero: bool,
    envelope_in_score: bool,
    input_precision: str,
    normalized_precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run prepared fixed-K Triton attention outside Dynamo graphs.

    ``fixed_k_local_attention_triton`` remains the validation/preparation
    wrapper. Compiled model layers can trace that ordinary tensor plumbing, then
    graph-break here for the custom Triton/autograd call.
    """

    needs_autograd = torch.is_grad_enabled() and _requires_grad(
        q,
        k,
        v,
        rbf_weight if bias_mode in {_FIXED_K_BIAS_ESEN_RBF_TYPE, _FIXED_K_BIAS_ESEN_PRECOMPUTED} else None,
        type_bias if has_type_bias else None,
    )
    if needs_autograd:
        return _FixedKLocalAttentionFunction.apply(
            q,
            k,
            v,
            neighbor_idx,
            neighbor_mask,
            dist,
            atom_types,
            rbf_weight,
            type_bias,
            centers,
            gamma,
            pre_local_mask,
            pre_env_bias,
            pre_log_env,
            pre_rho_env,
            pre_type_base,
            int(bias_mode),
            int(max_atomic_number),
            int(heads_per_frame),
            int(num_rbf),
            float(cutoff),
            bool(has_type_bias),
            bool(diag_zero),
            bool(envelope_in_score),
            input_precision,
            normalized_precision,
        )
    return _fixed_k_local_attention_forward_cuda_prepared(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        atom_types=atom_types,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        pre_local_mask=pre_local_mask,
        pre_env_bias=pre_env_bias,
        pre_log_env=pre_log_env,
        pre_rho_env=pre_rho_env,
        pre_type_base=pre_type_base,
        bias_mode=int(bias_mode),
        max_atomic_number=int(max_atomic_number),
        heads_per_frame=int(heads_per_frame),
        num_rbf=int(num_rbf),
        cutoff=float(cutoff),
        has_type_bias=bool(has_type_bias),
        diag_zero=bool(diag_zero),
        envelope_in_score=bool(envelope_in_score),
        input_precision=input_precision,
    )


def fixed_k_local_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: FixedKLocalBias | ESENFixedKLocalBiasView | None = None,
    dist: torch.Tensor | None = None,
    rbf: torch.Tensor | None = None,
    esen_features: ESENFixedKLocalAttentionFeatures | None = None,
    atom_types: torch.Tensor | None = None,
    precision: str = "tf32x3",
    max_atomic_number: int | None = None,
    return_lse: bool = False,
    strict: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of fixed-K local atom attention.

    Arbitrary Python bias modules remain supported by the torch reference.  The
    Triton path currently compiles two known modes: no bias and eSEN-like
    enveloped RBF/type bias.  In the eSEN Triton path, RBF values are
    recomputed from ``dist`` and the bias module's ``centers``/``gamma`` unless
    ``esen_features`` is supplied with precomputed geometry terms.  The optional
    ``rbf`` argument is used only by torch fallback/reference code.

    This gathered fixed-K kernel uses explicit fp32 accumulation for q/k dot
    products rather than ``tl.dot``.  Backward currently covers q/k/v and
    eSEN bias-parameter gradients with a streaming-D kernel for supported
    head dimensions.
    ``precision=
    "bf16_flash_compat"`` mirrors the dense Platonic path by internally casting
    q/k/v to bfloat16 and casting the output back to the original q dtype. Other
    precision strings are accepted for API compatibility, but do not change
    Tensor Core precision.
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
    bias_mode = _FIXED_K_BIAS_NONE
    rbf_weight = torch.empty((1, 1), device=q.device, dtype=torch.float32)
    centers = torch.empty((1,), device=q.device, dtype=torch.float32)
    gamma = torch.ones((), device=q.device, dtype=torch.float32)
    type_bias = torch.empty((1, 1, 1), device=q.device, dtype=torch.float32)
    pre_local_mask = torch.empty((1, 1), device=q.device, dtype=torch.bool)
    pre_env_bias = torch.empty((1, 1), device=q.device, dtype=torch.float32)
    pre_log_env = torch.empty((1, 1), device=q.device, dtype=torch.float32)
    pre_rho_env = torch.empty((1, 1, 1), device=q.device, dtype=torch.float32)
    pre_type_base = torch.empty((1, 1), device=q.device, dtype=torch.int32)
    heads_per_frame = 1
    num_rbf = 1
    cutoff = 1.0
    has_type_bias = False
    diag_zero = True
    envelope_in_score = True
    if bias is not None:
        if not isinstance(bias, (ESENEnvelopedRBFTypeFixedKBias, ESENFixedKLocalBiasView)):
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
        if esen_features is None and dist is None:
            raise ValueError("ESEN fixed-K Triton bias requires dist or esen_features")
        if dist is not None and dist.shape != neighbor_idx.shape:
            raise ValueError("dist must have shape [N, K]")
        if dist is not None and dist.requires_grad and torch.is_grad_enabled():
            raise NotImplementedError(
                "fixed-K local Triton attention does not implement gradients with respect to dist/coordinates yet"
            )
        if q.shape[1] % int(bias.heads_per_frame) != 0:
            raise ValueError("heads_per_frame must divide num_heads")
        bias_mode = _FIXED_K_BIAS_ESEN_PRECOMPUTED if esen_features is not None else _FIXED_K_BIAS_ESEN_RBF_TYPE
        rbf_weight = bias.rbf_weight.to(device=q.device, dtype=torch.float32).contiguous()
        centers = bias.centers.to(device=q.device, dtype=torch.float32).contiguous()
        gamma = bias.gamma.to(device=q.device, dtype=torch.float32).reshape(()).contiguous()
        heads_per_frame = int(bias.heads_per_frame)
        num_rbf = int(rbf_weight.shape[-1])
        cutoff = float(bias.cutoff)
        diag_zero = bool(bias.diag_zero)
        envelope_in_score = bool(bias.envelope_in_score)
        if bias.type_bias is not None:
            if atom_types is None and esen_features is None:
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
        if esen_features is not None:
            expected_shape = tuple(neighbor_idx.shape)
            if tuple(esen_features.local_mask.shape) != expected_shape:
                raise ValueError("esen_features.local_mask must have shape [N, K]")
            if tuple(esen_features.env_bias.shape) != expected_shape:
                raise ValueError("esen_features.env_bias must have shape [N, K]")
            if tuple(esen_features.log_env.shape) != expected_shape:
                raise ValueError("esen_features.log_env must have shape [N, K]")
            if tuple(esen_features.type_base.shape) != expected_shape:
                raise ValueError("esen_features.type_base must have shape [N, K]")
            if esen_features.rho_env.shape[:2] != neighbor_idx.shape or int(esen_features.rho_env.shape[-1]) != int(num_rbf):
                raise ValueError("esen_features.rho_env must have shape [N, K, num_rbf]")
            if _requires_grad(
                esen_features.local_mask,
                esen_features.env_bias,
                esen_features.log_env,
                esen_features.rho_env,
                esen_features.type_base,
            ) and torch.is_grad_enabled():
                raise NotImplementedError("fixed-K local Triton attention does not implement gradients through esen_features")
            pre_local_mask = esen_features.local_mask.to(device=q.device, dtype=torch.bool).contiguous()
            pre_env_bias = esen_features.env_bias.to(device=q.device).contiguous()
            pre_log_env = esen_features.log_env.to(device=q.device).contiguous()
            pre_rho_env = esen_features.rho_env.to(device=q.device).contiguous()
            pre_type_base = esen_features.type_base.to(device=q.device, dtype=torch.int32).contiguous()
    else:
        atom_types_apply = torch.empty((1,), device=q.device, dtype=torch.long)

    if max_atomic_number is None:
        max_atomic_number = 0
    if int(neighbor_idx.shape[-1]) > _MAX_FULL_TILE_K:
        raise ValueError(f"fixed-K local Triton attention currently supports K <= {_MAX_FULL_TILE_K}")
    if int(q.shape[-1]) > _MAX_CHUNKED_HEAD_DIM:
        raise ValueError(f"fixed-K local Triton attention currently supports head_dim <= {_MAX_CHUNKED_HEAD_DIM}")
    normalized_precision = normalize_platonic_attention_precision(precision)
    input_precision = _kernel_input_precision(normalized_precision)
    flash_compat_bf16 = normalized_precision == "bf16_flash_compat"
    orig_dtype = q.dtype

    if flash_compat_bf16:
        q_apply = q.to(torch.bfloat16).contiguous()
        k_apply = k.to(torch.bfloat16).contiguous()
        v_apply = v.to(torch.bfloat16).contiguous()
    else:
        q_apply = q.contiguous()
        k_apply = k.contiguous()
        v_apply = v.contiguous()
    neighbor_idx_apply = neighbor_idx.to(device=q.device, dtype=torch.int32).contiguous()
    neighbor_mask_apply = neighbor_mask.to(device=q.device, dtype=torch.bool).contiguous()
    dist_apply = (
        dist.to(device=q.device, dtype=torch.float32).contiguous()
        if dist is not None
        else torch.empty((1, 1), device=q.device, dtype=torch.float32)
    )
    out, lse = fixed_k_local_attention_triton_prepared_eager(
        q_apply,
        k_apply,
        v_apply,
        neighbor_idx=neighbor_idx_apply,
        neighbor_mask=neighbor_mask_apply,
        dist=dist_apply,
        atom_types=atom_types_apply,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        pre_local_mask=pre_local_mask,
        pre_env_bias=pre_env_bias,
        pre_log_env=pre_log_env,
        pre_rho_env=pre_rho_env,
        pre_type_base=pre_type_base,
        bias_mode=int(bias_mode),
        max_atomic_number=int(max_atomic_number),
        heads_per_frame=int(heads_per_frame),
        num_rbf=int(num_rbf),
        cutoff=float(cutoff),
        has_type_bias=bool(has_type_bias),
        diag_zero=bool(diag_zero),
        envelope_in_score=bool(envelope_in_score),
        input_precision=input_precision,
        normalized_precision=normalized_precision,
    )
    if flash_compat_bf16:
        out = out.to(orig_dtype)
    return (out, lse) if return_lse else out
