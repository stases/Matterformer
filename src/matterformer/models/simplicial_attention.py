from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from matterformer.models.triton_simplicial_attention import (
    TRITON_AVAILABLE,
    normalize_simplicial_precision,
    triton_simplicial_attention_backward,
    triton_simplicial_attention_forward,
)


@dataclass(frozen=True)
class SimplicialFactorizedBias:
    """Structured factorized simplicial bias.

    Shapes:
    - ``u`` / ``v`` / ``w``: ``(B, H, T, T)``
    - ``gate``: ``(B, H, T)``, already expanded over query positions.
    """

    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    gate: torch.Tensor

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        expected_pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
        for name, value in (("u", self.u), ("v", self.v), ("w", self.w)):
            if value.shape != expected_pair_shape:
                raise ValueError(f"{name} must have shape {expected_pair_shape}, got {tuple(value.shape)}")
        expected_gate_shape = (batch_size, num_heads, num_tokens)
        if self.gate.shape != expected_gate_shape:
            raise ValueError(f"gate must have shape {expected_gate_shape}, got {tuple(self.gate.shape)}")

    def chunk(self, start: int, end: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        u = self.u[:, :, start:end, :].float()
        v = self.v[:, :, start:end, :].float()
        w = self.w.float()
        gate = self.gate[:, :, start:end].float()
        bias = u[:, :, :, :, None] + v[:, :, :, None, :] + w[:, :, None, :, :]
        return (gate[:, :, :, None, None] * bias).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SimplicialLowRankAngleResidual:
    """Structured low-rank triplet residual for simplicial logits."""

    left: torch.Tensor
    right: torch.Tensor
    gate: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.left.shape[-1])

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        if self.left.ndim != 5:
            raise ValueError(f"left must have shape (B, H, T, T, R), got {tuple(self.left.shape)}")
        expected_prefix = (batch_size, num_heads, num_tokens, num_tokens)
        if self.left.shape[:4] != expected_prefix:
            raise ValueError(f"left must have shape {expected_prefix + (self.left.shape[-1],)}, got {tuple(self.left.shape)}")
        if self.right.shape != self.left.shape:
            raise ValueError(f"right must have shape {tuple(self.left.shape)}, got {tuple(self.right.shape)}")
        if self.rank <= 0:
            raise ValueError("low-rank angle residual rank must be positive")
        expected_gate_shape = (batch_size, num_heads, num_tokens)
        if self.gate.shape != expected_gate_shape:
            raise ValueError(f"gate must have shape {expected_gate_shape}, got {tuple(self.gate.shape)}")

    def chunk(self, start: int, end: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        left = self.left[:, :, start:end, :, :].float()
        right = self.right[:, :, start:end, :, :].float()
        residual = torch.einsum("bhqjr,bhqkr->bhqjk", left, right) * (self.rank**-0.5)
        gate = self.gate[:, :, start:end].float().unsqueeze(-1).unsqueeze(-1)
        return (gate * residual).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SimplicialLowRankMessageResidual:
    """Structured low-rank triplet residual for simplicial values/messages."""

    left: torch.Tensor
    right: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.left.shape[-1])

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        if self.left.ndim != 5:
            raise ValueError(f"left must have shape (B, H, T, T, R), got {tuple(self.left.shape)}")
        expected_prefix = (batch_size, num_heads, num_tokens, num_tokens)
        if self.left.shape[:4] != expected_prefix:
            raise ValueError(f"left must have shape {expected_prefix + (self.left.shape[-1],)}, got {tuple(self.left.shape)}")
        if self.right.shape != self.left.shape:
            raise ValueError(f"right must have shape {tuple(self.left.shape)}, got {tuple(self.right.shape)}")
        if self.rank <= 0:
            raise ValueError("low-rank message residual rank must be positive")


@dataclass(frozen=True)
class SimplicialAttentionMask:
    """Internal mask semantics for simplicial attention."""

    query_valid: torch.Tensor
    pair_key_valid: torch.Tensor
    pair_valid: torch.Tensor | None = None

    def validate(self, *, batch_size: int, num_tokens: int) -> None:
        expected_mask_shape = (batch_size, num_tokens)
        if self.query_valid.shape != expected_mask_shape:
            raise ValueError(f"query_valid must have shape {expected_mask_shape}, got {tuple(self.query_valid.shape)}")
        if self.pair_key_valid.shape != expected_mask_shape:
            raise ValueError(f"pair_key_valid must have shape {expected_mask_shape}, got {tuple(self.pair_key_valid.shape)}")
        if self.pair_valid is not None and self.pair_valid.shape != (batch_size, num_tokens, num_tokens):
            raise ValueError(f"pair_valid must have shape {(batch_size, num_tokens, num_tokens)}, got {tuple(self.pair_valid.shape)}")

    @classmethod
    def from_key_padding_mask(
        cls,
        key_padding_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> "SimplicialAttentionMask":
        if key_padding_mask is None:
            valid = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
        else:
            if key_padding_mask.shape != (batch_size, num_tokens):
                raise ValueError(f"key_padding_mask must have shape {(batch_size, num_tokens)}, got {tuple(key_padding_mask.shape)}")
            valid = ~key_padding_mask.bool().to(device=device)
        return cls(query_valid=valid, pair_key_valid=valid)

    def pair_mask(self) -> torch.Tensor:
        pair_valid = self.pair_key_valid[:, :, None] & self.pair_key_valid[:, None, :]
        if self.pair_valid is not None:
            pair_valid = pair_valid & self.pair_valid.bool()
        return pair_valid


def _canonicalize_simplicial_content_logits(mode: str) -> str:
    mode = str(mode).lower().replace("-", "_")
    if mode in {"disabled", "off", "false", "no"}:
        return "off"
    if mode in {"enabled", "on", "true", "yes"}:
        return "on"
    if mode in {"gated", "learned_scale", "learnable", "learnable_scale"}:
        return "learned"
    if mode not in {"on", "off", "learned"}:
        raise ValueError("simplicial_content_logits must be one of {'on', 'off', 'learned'}")
    return mode


def _materialize_chunk_logits(
    q_chunk: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    *,
    factorized_bias: SimplicialFactorizedBias | None,
    angle_residual: SimplicialLowRankAngleResidual | None,
    logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None,
    start: int,
    end: int,
) -> torch.Tensor:
    qk2 = q_chunk.unsqueeze(-2) * k2.unsqueeze(-3)
    logits = torch.matmul(k1.unsqueeze(-3), qk2.transpose(-1, -2)).float()
    if factorized_bias is not None:
        logits = logits + factorized_bias.chunk(start, end, dtype=logits.dtype, device=logits.device)
    if angle_residual is not None:
        logits = logits + angle_residual.chunk(start, end, dtype=logits.dtype, device=logits.device)
    if logit_bias_fn is not None:
        logits = logits + logit_bias_fn(start, end, logits.dtype, logits.device)
    return logits


def _coerce_attention_mask(
    attention_mask: SimplicialAttentionMask | None,
    *,
    key_padding_mask: torch.Tensor | None,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> SimplicialAttentionMask:
    if attention_mask is None:
        attention_mask = SimplicialAttentionMask.from_key_padding_mask(
            key_padding_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=device,
        )
    attention_mask.validate(batch_size=batch_size, num_tokens=num_tokens)
    return SimplicialAttentionMask(
        query_valid=attention_mask.query_valid.bool().to(device=device),
        pair_key_valid=attention_mask.pair_key_valid.bool().to(device=device),
        pair_valid=attention_mask.pair_valid.bool().to(device=device) if attention_mask.pair_valid is not None else None,
    )


def simplicial_attention_torch_from_projected(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    key_padding_mask: torch.Tensor | None = None,
    attention_mask: SimplicialAttentionMask | None = None,
    factorized_bias: SimplicialFactorizedBias | None = None,
    angle_residual: SimplicialLowRankAngleResidual | None = None,
    message_residual: SimplicialLowRankMessageResidual | None = None,
    message_basis: torch.Tensor | None = None,
    logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    chunk_size: int = 128,
    return_attn: bool = False,
    fp32_core: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, num_tokens, head_dim = q.shape
    if factorized_bias is not None:
        factorized_bias.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
    if angle_residual is not None:
        angle_residual.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
    if message_residual is not None:
        message_residual.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
        if message_basis is None:
            raise ValueError("message_basis must be provided when message_residual is provided")
        expected_basis_shape = (num_heads, message_residual.rank, head_dim)
        if message_basis.shape != expected_basis_shape:
            raise ValueError(f"message_basis must have shape {expected_basis_shape}, got {tuple(message_basis.shape)}")
    elif message_basis is not None:
        raise ValueError("message_basis must be None when message_residual is None")

    attention_mask = _coerce_attention_mask(
        attention_mask,
        key_padding_mask=key_padding_mask,
        batch_size=batch_size,
        num_tokens=num_tokens,
        device=q.device,
    )
    query_valid = attention_mask.query_valid
    pair_valid = attention_mask.pair_mask()

    compute_in_fp32 = fp32_core and q.dtype in (torch.float16, torch.bfloat16)
    if compute_in_fp32:
        q_work = q.float()
        k1_work = k1.float()
        v1_work = v1.float()
        k2_work = k2.float()
        v2_work = v2.float()
        factorized_bias_work = (
            SimplicialFactorizedBias(
                u=factorized_bias.u.float(),
                v=factorized_bias.v.float(),
                w=factorized_bias.w.float(),
                gate=factorized_bias.gate.float(),
            )
            if factorized_bias is not None
            else None
        )
        angle_residual_work = (
            SimplicialLowRankAngleResidual(
                left=angle_residual.left.float(),
                right=angle_residual.right.float(),
                gate=angle_residual.gate.float(),
            )
            if angle_residual is not None
            else None
        )
        message_residual_work = (
            SimplicialLowRankMessageResidual(
                left=message_residual.left.float(),
                right=message_residual.right.float(),
            )
            if message_residual is not None
            else None
        )
        message_basis_work = message_basis.float() if message_basis is not None else None
        out_dtype = torch.float32
    else:
        q_work = q
        k1_work = k1
        v1_work = v1
        k2_work = k2
        v2_work = v2
        factorized_bias_work = factorized_bias
        angle_residual_work = angle_residual
        message_residual_work = message_residual
        message_basis_work = message_basis
        out_dtype = v1.dtype

    out = torch.empty((batch_size, num_heads, num_tokens, head_dim), device=q.device, dtype=out_dtype)
    attn_chunks: list[torch.Tensor] | None = [] if return_attn else None
    pair_mask = pair_valid[:, None, None, :, :]
    flat_mask = pair_mask.flatten(-2)

    for start in range(0, num_tokens, chunk_size):
        end = min(num_tokens, start + chunk_size)
        q_chunk = q_work[:, :, start:end, :]
        logits = _materialize_chunk_logits(
            q_chunk,
            k1_work,
            k2_work,
            factorized_bias=factorized_bias_work,
            angle_residual=angle_residual_work,
            logit_bias_fn=logit_bias_fn,
            start=start,
            end=end,
        )
        flat_logits = logits.flatten(-2).masked_fill(~flat_mask, torch.finfo(logits.dtype).min)
        attn = torch.softmax(flat_logits, dim=-1)
        attn = torch.where(flat_mask, attn, torch.zeros_like(attn)).view_as(logits)

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=training)
        if return_attn and attn_chunks is not None:
            attn_chunks.append(attn.detach())

        attn_float = attn.float()
        attn_value = attn.to(v1_work.dtype)
        tmp = torch.matmul(attn_value.transpose(-2, -1), v1_work.unsqueeze(-3))
        out_chunk = (tmp * v2_work.unsqueeze(-3)).sum(dim=-2)
        if message_residual_work is not None and message_basis_work is not None:
            message_left = message_residual_work.left[:, :, start:end, :, :].float()
            message_right = message_residual_work.right[:, :, start:end, :, :].float()
            message_coeff = torch.einsum("bhqjk,bhqjr,bhqkr->bhqr", attn_float, message_left, message_right) * (
                message_residual_work.rank**-0.5
            )
            message_out = torch.einsum("bhqr,hrd->bhqd", message_coeff, message_basis_work.float())
            out_chunk = out_chunk + message_out.to(dtype=out_chunk.dtype)
        q_valid = query_valid[:, start:end].to(dtype=out_chunk.dtype)
        out_chunk = out_chunk * q_valid[:, None, :, None]
        out[:, :, start:end, :] = out_chunk

    if return_attn and attn_chunks is not None:
        return out.to(v1.dtype), torch.cat(attn_chunks, dim=2)
    return out.to(v1.dtype)


class _TritonTwoSimplicialAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        k2: torch.Tensor,
        v2: torch.Tensor,
        query_valid: torch.Tensor,
        pair_key_valid: torch.Tensor,
        pair_valid: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        gate: torch.Tensor,
        angle_left: torch.Tensor,
        angle_right: torch.Tensor,
        angle_gate: torch.Tensor,
        message_left: torch.Tensor,
        message_right: torch.Tensor,
        message_basis: torch.Tensor,
        precision_mode: str,
        chunk_size: int,
        debug_torch_backward: bool,
    ) -> torch.Tensor:
        has_pair_valid = pair_valid.numel() > 0
        has_factorized_bias = u.numel() > 0
        has_low_rank_angle = angle_left.numel() > 0
        has_low_rank_message = message_left.numel() > 0
        out, lse, out_for_backward = triton_simplicial_attention_forward(
            q,
            k1,
            v1,
            k2,
            v2,
            query_valid=query_valid,
            pair_key_valid=pair_key_valid,
            pair_valid=pair_valid if has_pair_valid else None,
            u=u if has_factorized_bias else None,
            v_bias=v if has_factorized_bias else None,
            w=w if has_factorized_bias else None,
            gate=gate if has_factorized_bias else None,
            angle_left=angle_left if has_low_rank_angle else None,
            angle_right=angle_right if has_low_rank_angle else None,
            angle_gate=angle_gate if has_low_rank_angle else None,
            message_left=message_left if has_low_rank_message else None,
            message_right=message_right if has_low_rank_message else None,
            message_basis=message_basis if has_low_rank_message else None,
            precision=precision_mode,
        )
        ctx.precision_mode = str(precision_mode)
        ctx.chunk_size = int(chunk_size)
        ctx.debug_torch_backward = bool(debug_torch_backward)
        ctx.has_pair_valid = has_pair_valid
        ctx.has_factorized_bias = has_factorized_bias
        ctx.has_low_rank_angle = has_low_rank_angle
        ctx.has_low_rank_message = has_low_rank_message
        ctx.save_for_backward(
            q,
            k1,
            v1,
            k2,
            v2,
            query_valid,
            pair_key_valid,
            pair_valid,
            u,
            v,
            w,
            gate,
            angle_left,
            angle_right,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out_for_backward,
            lse,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (
            q,
            k1,
            v1,
            k2,
            v2,
            query_valid,
            pair_key_valid,
            pair_valid,
            u,
            v,
            w,
            gate,
            angle_left,
            angle_right,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out,
            lse,
        ) = ctx.saved_tensors

        if ctx.debug_torch_backward:
            differentiable_inputs = [
                q.detach().requires_grad_(True),
                k1.detach().requires_grad_(True),
                v1.detach().requires_grad_(True),
                k2.detach().requires_grad_(True),
                v2.detach().requires_grad_(True),
            ]
            if ctx.has_factorized_bias:
                differentiable_inputs.extend(
                    [u.detach().requires_grad_(True), v.detach().requires_grad_(True), w.detach().requires_grad_(True), gate.detach().requires_grad_(True)]
                )
            if ctx.has_low_rank_angle:
                differentiable_inputs.extend(
                    [angle_left.detach().requires_grad_(True), angle_right.detach().requires_grad_(True), angle_gate.detach().requires_grad_(True)]
                )
            if ctx.has_low_rank_message:
                differentiable_inputs.extend(
                    [message_left.detach().requires_grad_(True), message_right.detach().requires_grad_(True), message_basis.detach().requires_grad_(True)]
                )

            with torch.enable_grad():
                q_re, k1_re, v1_re, k2_re, v2_re = differentiable_inputs[:5]
                cursor = 5
                factorized_bias = None
                if ctx.has_factorized_bias:
                    u_re, v_re, w_re, gate_re = differentiable_inputs[cursor : cursor + 4]
                    cursor += 4
                    factorized_bias = SimplicialFactorizedBias(u=u_re, v=v_re, w=w_re, gate=gate_re)
                angle_residual = None
                if ctx.has_low_rank_angle:
                    angle_left_re, angle_right_re, angle_gate_re = differentiable_inputs[cursor : cursor + 3]
                    cursor += 3
                    angle_residual = SimplicialLowRankAngleResidual(left=angle_left_re, right=angle_right_re, gate=angle_gate_re)
                message_residual = None
                message_basis_re = None
                if ctx.has_low_rank_message:
                    message_left_re, message_right_re, message_basis_re = differentiable_inputs[cursor : cursor + 3]
                    message_residual = SimplicialLowRankMessageResidual(left=message_left_re, right=message_right_re)
                attention_mask = SimplicialAttentionMask(
                    query_valid=query_valid,
                    pair_key_valid=pair_key_valid,
                    pair_valid=pair_valid if ctx.has_pair_valid else None,
                )
                reference_out = simplicial_attention_torch_from_projected(
                    q_re,
                    k1_re,
                    v1_re,
                    k2_re,
                    v2_re,
                    attention_mask=attention_mask,
                    factorized_bias=factorized_bias,
                    angle_residual=angle_residual,
                    message_residual=message_residual,
                    message_basis=message_basis_re,
                    dropout_p=0.0,
                    training=False,
                    chunk_size=ctx.chunk_size,
                    return_attn=False,
                    fp32_core=True,
                )
                grads = torch.autograd.grad(reference_out, differentiable_inputs, grad_out, allow_unused=True)

            dq, dk1, dv1, dk2, dv2 = grads[:5]
            du = dv = dw = dgate = None
            dangle_left = dangle_right = dangle_gate = None
            dmessage_left = dmessage_right = dmessage_basis = None
            cursor = 5
            if ctx.has_factorized_bias:
                du, dv, dw, dgate = grads[cursor : cursor + 4]
                cursor += 4
            if ctx.has_low_rank_angle:
                dangle_left, dangle_right, dangle_gate = grads[cursor : cursor + 3]
                cursor += 3
            if ctx.has_low_rank_message:
                dmessage_left, dmessage_right, dmessage_basis = grads[cursor : cursor + 3]
            return (
                dq,
                dk1,
                dv1,
                dk2,
                dv2,
                None,
                None,
                None,
                du,
                dv,
                dw,
                dgate,
                dangle_left,
                dangle_right,
                dangle_gate,
                dmessage_left,
                dmessage_right,
                dmessage_basis,
                None,
                None,
                None,
            )

        need_du = bool(ctx.has_factorized_bias and ctx.needs_input_grad[8])
        need_dv_bias = bool(ctx.has_factorized_bias and ctx.needs_input_grad[9])
        need_dw = bool(ctx.has_factorized_bias and ctx.needs_input_grad[10])
        need_dgate = bool(ctx.has_factorized_bias and ctx.needs_input_grad[11])
        need_dangle_left = bool(ctx.has_low_rank_angle and ctx.needs_input_grad[12])
        need_dangle_right = bool(ctx.has_low_rank_angle and ctx.needs_input_grad[13])
        need_dangle_gate = bool(ctx.has_low_rank_angle and ctx.needs_input_grad[14])
        need_dmessage_left = bool(ctx.has_low_rank_message and ctx.needs_input_grad[15])
        need_dmessage_right = bool(ctx.has_low_rank_message and ctx.needs_input_grad[16])
        need_dmessage_basis = bool(ctx.has_low_rank_message and ctx.needs_input_grad[17])

        (
            dq,
            dk1,
            dv1,
            dk2,
            dv2,
            du,
            dv_bias,
            dw,
            dgate,
            dangle_left,
            dangle_right,
            dangle_gate,
            dmessage_left,
            dmessage_right,
            dmessage_basis,
        ) = triton_simplicial_attention_backward(
            grad_out,
            q,
            k1,
            v1,
            k2,
            v2,
            out,
            lse,
            query_valid=query_valid,
            pair_key_valid=pair_key_valid,
            pair_valid=pair_valid if ctx.has_pair_valid else None,
            u=u if ctx.has_factorized_bias else None,
            v_bias=v if ctx.has_factorized_bias else None,
            w=w if ctx.has_factorized_bias else None,
            gate=gate if ctx.has_factorized_bias else None,
            angle_left=angle_left if ctx.has_low_rank_angle else None,
            angle_right=angle_right if ctx.has_low_rank_angle else None,
            angle_gate=angle_gate if ctx.has_low_rank_angle else None,
            message_left=message_left if ctx.has_low_rank_message else None,
            message_right=message_right if ctx.has_low_rank_message else None,
            message_basis=message_basis if ctx.has_low_rank_message else None,
            precision=ctx.precision_mode,
            need_du=need_du,
            need_dv_bias=need_dv_bias,
            need_dw=need_dw,
            need_dgate=need_dgate,
            need_dangle_left=need_dangle_left,
            need_dangle_right=need_dangle_right,
            need_dangle_gate=need_dangle_gate,
            need_dmessage_left=need_dmessage_left,
            need_dmessage_right=need_dmessage_right,
            need_dmessage_basis=need_dmessage_basis,
        )
        return (
            dq,
            dk1,
            dv1,
            dk2,
            dv2,
            None,
            None,
            None,
            du,
            dv_bias,
            dw,
            dgate,
            dangle_left,
            dangle_right,
            dangle_gate,
            dmessage_left,
            dmessage_right,
            dmessage_basis,
            None,
            None,
            None,
        )


class SimplicialAttention(nn.Module):
    """Dense non-causal 2-simplicial attention with explicit torch/Triton backends."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        chunk_size: int = 128,
        out_proj: bool = True,
        impl: str = "auto",
        precision: str = "bf16_tc",
        message_mode: str = "none",
        message_rank: int = 16,
        content_logits: str = "on",
        debug_torch_backward: bool = False,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}) when head_dim is None.")
            head_dim = dim // num_heads
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        impl = impl.lower()
        if impl not in {"auto", "torch", "triton"}:
            raise ValueError(f"Unsupported simplicial_impl: {impl}")
        message_mode = message_mode.lower()
        if message_mode not in {"none", "low_rank"}:
            raise ValueError(f"Unsupported simplicial_message_mode: {message_mode}")
        message_rank = int(message_rank)
        if message_mode == "low_rank" and message_rank <= 0:
            raise ValueError("simplicial_message_rank must be positive when message_mode='low_rank'")
        content_logits = _canonicalize_simplicial_content_logits(content_logits)

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.chunk_size = int(chunk_size)
        self.impl = impl
        self.precision = normalize_simplicial_precision(precision)
        self.message_mode = message_mode
        self.message_rank = message_rank
        self.content_logits = content_logits
        self.debug_torch_backward = bool(debug_torch_backward)
        self.scale = self.head_dim**-0.5
        self._last_impl_used = "torch"

        self.in_proj = nn.Linear(self.dim, 5 * self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, self.dim, bias=bias) if out_proj else nn.Identity()
        if self.message_mode == "low_rank":
            self.message_basis = nn.Parameter(torch.empty(self.num_heads, self.message_rank, self.head_dim))
            nn.init.normal_(self.message_basis, mean=0.0, std=self.head_dim**-0.5)
        else:
            self.message_basis = None
        self.content_logit_log_scale = (
            nn.Parameter(torch.zeros((1, self.num_heads, 1, 1), dtype=torch.float32))
            if self.content_logits == "learned"
            else None
        )

    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        return x.view(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_tokens, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, num_tokens, num_heads * head_dim)

    def _project_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k1, v1, k2, v2 = self.in_proj(x).chunk(5, dim=-1)
        q = self._split_heads(q, self.num_heads, self.head_dim) * self.scale
        k1 = self._split_heads(k1, self.num_heads, self.head_dim)
        v1 = self._split_heads(v1, self.num_heads, self.head_dim)
        k2 = self._split_heads(k2, self.num_heads, self.head_dim)
        v2 = self._split_heads(v2, self.num_heads, self.head_dim)
        return q, k1, v1, k2, v2

    def _content_query(self, q: torch.Tensor) -> torch.Tensor:
        if self.content_logits == "on":
            return q
        if self.content_logits == "off":
            return torch.zeros_like(q)
        assert self.content_logit_log_scale is not None
        return q * self.content_logit_log_scale.to(device=q.device, dtype=q.dtype).exp()

    def _triton_unavailable_reason(
        self,
        *,
        x: torch.Tensor,
        factorized_bias: SimplicialFactorizedBias | None,
        angle_residual: SimplicialLowRankAngleResidual | None,
        message_residual: SimplicialLowRankMessageResidual | None,
        message_basis: torch.Tensor | None,
        logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None,
        return_attn: bool,
    ) -> str | None:
        if not TRITON_AVAILABLE:
            return "triton is not installed"
        if x.device.type != "cuda":
            return "the Triton backend requires CUDA tensors"
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return f"the Triton backend only supports float16/bfloat16/float32 inputs, got {x.dtype}"
        if self.training and self.dropout > 0.0:
            return "the Triton backend does not support attention dropout in training mode in v1"
        if return_attn:
            return "the Triton backend does not support return_attn=True in v1"
        if logit_bias_fn is not None:
            return "the Triton backend does not support arbitrary Python logit_bias_fn callbacks"
        if self.head_dim > 128:
            return f"the Triton backend only supports head_dim <= 128 in v1, got {self.head_dim}"
        if factorized_bias is not None:
            for name, tensor in (("u", factorized_bias.u), ("v", factorized_bias.v), ("w", factorized_bias.w), ("gate", factorized_bias.gate)):
                if tensor.device != x.device:
                    return f"factorized bias tensor {name} must be on the same CUDA device as the attention input"
        if angle_residual is not None:
            if angle_residual.left.device != x.device or angle_residual.right.device != x.device or angle_residual.gate.device != x.device:
                return "angle residual tensors must be on the same CUDA device as the attention input"
            if angle_residual.rank > 64:
                return f"the Triton backend only supports low-rank angle residual rank <= 64, got {angle_residual.rank}"
        if message_residual is not None:
            if message_basis is None:
                return "message residual tensors require a message_basis"
            if message_residual.left.device != x.device or message_residual.right.device != x.device:
                return "message residual tensors must be on the same CUDA device as the attention input"
            if message_basis.device != x.device:
                return "message basis tensor must be on the same CUDA device as the attention input"
            if message_basis.shape != (self.num_heads, message_residual.rank, self.head_dim):
                return f"message basis tensor must have shape {(self.num_heads, message_residual.rank, self.head_dim)}, got {tuple(message_basis.shape)}"
            if message_residual.rank > 64:
                return f"the Triton backend only supports low-rank message residual rank <= 64, got {message_residual.rank}"
        elif message_basis is not None:
            return "message_basis must be None when message_residual is None"
        return None

    def _triton_compute_dtype(self) -> torch.dtype:
        if self.precision == "bf16_tc":
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _cast_angle_residual_for_triton(
        angle_residual: SimplicialLowRankAngleResidual | None,
        *,
        dtype: torch.dtype,
    ) -> SimplicialLowRankAngleResidual | None:
        if angle_residual is None:
            return None
        return SimplicialLowRankAngleResidual(
            left=angle_residual.left.to(dtype=dtype),
            right=angle_residual.right.to(dtype=dtype),
            gate=angle_residual.gate.to(dtype=dtype),
        )

    @staticmethod
    def _cast_message_residual_for_triton(
        message_residual: SimplicialLowRankMessageResidual | None,
        *,
        dtype: torch.dtype,
    ) -> SimplicialLowRankMessageResidual | None:
        if message_residual is None:
            return None
        return SimplicialLowRankMessageResidual(
            left=message_residual.left.to(dtype=dtype),
            right=message_residual.right.to(dtype=dtype),
        )

    def _forward_triton(
        self,
        q: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        k2: torch.Tensor,
        v2: torch.Tensor,
        *,
        attention_mask: SimplicialAttentionMask,
        factorized_bias: SimplicialFactorizedBias | None,
        angle_residual: SimplicialLowRankAngleResidual | None,
        message_residual: SimplicialLowRankMessageResidual | None,
        message_basis: torch.Tensor | None,
    ) -> torch.Tensor:
        attention_mask = _coerce_attention_mask(
            attention_mask,
            key_padding_mask=None,
            batch_size=q.shape[0],
            num_tokens=q.shape[2],
            device=q.device,
        )
        compute_dtype = self._triton_compute_dtype()
        q = q.to(dtype=compute_dtype)
        k1 = k1.to(dtype=compute_dtype)
        v1 = v1.to(dtype=compute_dtype)
        k2 = k2.to(dtype=compute_dtype)
        v2 = v2.to(dtype=compute_dtype)
        angle_residual = self._cast_angle_residual_for_triton(angle_residual, dtype=torch.float32)
        message_residual = self._cast_message_residual_for_triton(message_residual, dtype=torch.float32)

        empty_float = q.new_empty(0)
        empty_bool = torch.empty(0, device=q.device, dtype=torch.bool)
        if factorized_bias is None:
            u = v_bias = w = gate = empty_float
        else:
            factorized_bias.validate(batch_size=q.shape[0], num_heads=q.shape[1], num_tokens=q.shape[2])
            u, v_bias, w, gate = factorized_bias.u, factorized_bias.v, factorized_bias.w, factorized_bias.gate
        if angle_residual is None:
            angle_left = angle_right = angle_gate = empty_float
        else:
            angle_residual.validate(batch_size=q.shape[0], num_heads=q.shape[1], num_tokens=q.shape[2])
            angle_left = angle_residual.left
            angle_right = angle_residual.right
            angle_gate = angle_residual.gate
        if message_residual is None:
            message_left = message_right = message_basis_tensor = empty_float
        else:
            if message_basis is None:
                raise RuntimeError("message_residual requires message_basis")
            message_residual.validate(batch_size=q.shape[0], num_heads=q.shape[1], num_tokens=q.shape[2])
            expected_basis_shape = (q.shape[1], message_residual.rank, q.shape[-1])
            if message_basis.shape != expected_basis_shape:
                raise ValueError(f"message_basis must have shape {expected_basis_shape}, got {tuple(message_basis.shape)}")
            message_left = message_residual.left
            message_right = message_residual.right
            message_basis_tensor = message_basis.to(device=q.device, dtype=torch.float32)

        return _TritonTwoSimplicialAttentionFunction.apply(
            q.contiguous(),
            k1.contiguous(),
            v1.contiguous(),
            k2.contiguous(),
            v2.contiguous(),
            attention_mask.query_valid.contiguous(),
            attention_mask.pair_key_valid.contiguous(),
            attention_mask.pair_valid.contiguous() if attention_mask.pair_valid is not None else empty_bool,
            u.contiguous(),
            v_bias.contiguous(),
            w.contiguous(),
            gate.contiguous(),
            angle_left.contiguous(),
            angle_right.contiguous(),
            angle_gate.contiguous(),
            message_left.contiguous(),
            message_right.contiguous(),
            message_basis_tensor.contiguous(),
            self.precision,
            self.chunk_size,
            self.debug_torch_backward,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: SimplicialAttentionMask | None = None,
        factorized_bias: SimplicialFactorizedBias | None = None,
        angle_residual: SimplicialLowRankAngleResidual | None = None,
        message_residual: SimplicialLowRankMessageResidual | None = None,
        logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, C), got {tuple(x.shape)}")
        batch_size, num_tokens, _ = x.shape
        attention_mask = _coerce_attention_mask(
            attention_mask,
            key_padding_mask=key_padding_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=x.device,
        )
        if factorized_bias is not None:
            factorized_bias.validate(batch_size=batch_size, num_heads=self.num_heads, num_tokens=num_tokens)
        if angle_residual is not None:
            angle_residual.validate(batch_size=batch_size, num_heads=self.num_heads, num_tokens=num_tokens)
        if message_residual is not None:
            if self.message_basis is None:
                raise ValueError("message_residual requires simplicial_message_mode='low_rank'")
            message_residual.validate(batch_size=batch_size, num_heads=self.num_heads, num_tokens=num_tokens)
            if message_residual.rank != self.message_rank:
                raise ValueError(f"message_residual rank {message_residual.rank} does not match attention message_rank {self.message_rank}")
        message_basis = self.message_basis if message_residual is not None else None

        q, k1, v1, k2, v2 = self._project_inputs(x)
        q_content = self._content_query(q)
        reason = self._triton_unavailable_reason(
            x=x,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
            message_residual=message_residual,
            message_basis=message_basis,
            logit_bias_fn=logit_bias_fn,
            return_attn=return_attn,
        )
        if self.impl == "auto":
            impl_used = "triton" if reason is None else "torch"
        elif self.impl == "triton":
            if reason is not None:
                raise RuntimeError(f"Triton simplicial attention is unavailable: {reason}")
            impl_used = "triton"
        else:
            impl_used = "torch"
        self._last_impl_used = impl_used

        if impl_used == "triton":
            out = self._forward_triton(
                q_content,
                k1,
                v1,
                k2,
                v2,
                attention_mask=attention_mask,
                factorized_bias=factorized_bias,
                angle_residual=angle_residual,
                message_residual=message_residual,
                message_basis=message_basis,
            )
            return self.out_proj(self._merge_heads(out))

        out = simplicial_attention_torch_from_projected(
            q_content,
            k1,
            v1,
            k2,
            v2,
            attention_mask=attention_mask,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
            message_residual=message_residual,
            message_basis=message_basis,
            logit_bias_fn=logit_bias_fn,
            dropout_p=self.dropout,
            training=self.training,
            chunk_size=self.chunk_size,
            return_attn=return_attn,
        )
        if return_attn:
            out_heads, attn = out
            return self.out_proj(self._merge_heads(out_heads)), attn
        return self.out_proj(self._merge_heads(out))


TwoSimplicialAttention = SimplicialAttention
