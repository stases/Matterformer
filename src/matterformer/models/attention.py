from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from matterformer.models.attention_triton import (
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
    - ``gate``: ``(B, H, T)``, already expanded over query positions and with any
      non-atom queries zeroed out.
    """

    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    gate: torch.Tensor

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        expected_pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
        for name, value in (("u", self.u), ("v", self.v), ("w", self.w)):
            if value.shape != expected_pair_shape:
                raise ValueError(
                    f"{name} must have shape {expected_pair_shape}, got {tuple(value.shape)}"
                )
        expected_gate_shape = (batch_size, num_heads, num_tokens)
        if self.gate.shape != expected_gate_shape:
            raise ValueError(
                f"gate must have shape {expected_gate_shape}, got {tuple(self.gate.shape)}"
            )

    def chunk(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        bias = (
            self.u[:, :, start:end, :, None]
            + self.v[:, :, start:end, None, :]
            + self.w[:, :, None, :, :]
        )
        gate = self.gate[:, :, start:end].unsqueeze(-1).unsqueeze(-1)
        return (gate * bias).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SimplicialLowRankAngleResidual:
    """Structured low-rank triplet residual for simplicial logits.

    Shapes:
    - ``left`` / ``right``: ``(B, H, T, T, R)``
    - ``gate``: ``(B, H, T)``, applied per query position.
    """

    left: torch.Tensor
    right: torch.Tensor
    gate: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.left.shape[-1])

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        if self.left.ndim != 5:
            raise ValueError(f"left must have shape (B, H, T, T, R), got {tuple(self.left.shape)}")
        expected_factor_prefix = (batch_size, num_heads, num_tokens, num_tokens)
        if self.left.shape[:4] != expected_factor_prefix:
            raise ValueError(
                f"left must have shape {expected_factor_prefix + (self.left.shape[-1],)}, "
                f"got {tuple(self.left.shape)}"
            )
        if self.right.shape != self.left.shape:
            raise ValueError(f"right must have shape {tuple(self.left.shape)}, got {tuple(self.right.shape)}")
        if self.rank <= 0:
            raise ValueError("low-rank angle residual rank must be positive")
        expected_gate_shape = (batch_size, num_heads, num_tokens)
        if self.gate.shape != expected_gate_shape:
            raise ValueError(
                f"gate must have shape {expected_gate_shape}, got {tuple(self.gate.shape)}"
            )

    def chunk(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        left = self.left[:, :, start:end, :, :].float()
        right = self.right[:, :, start:end, :, :].float()
        residual = torch.einsum("bhqjr,bhqkr->bhqjk", left, right) * (self.rank**-0.5)
        gate = self.gate[:, :, start:end].float().unsqueeze(-1).unsqueeze(-1)
        return (gate * residual).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SimplicialAttentionMask:
    """Internal mask semantics for simplicial attention.

    Shapes:
    - ``query_valid``: ``(B, T)``
    - ``pair_key_valid``: ``(B, T)``
    - ``pair_valid``: optional ``(B, T, T)``
    """

    query_valid: torch.Tensor
    pair_key_valid: torch.Tensor
    pair_valid: torch.Tensor | None = None

    def validate(self, *, batch_size: int, num_tokens: int) -> None:
        expected_mask_shape = (batch_size, num_tokens)
        if self.query_valid.shape != expected_mask_shape:
            raise ValueError(
                f"query_valid must have shape {expected_mask_shape}, got {tuple(self.query_valid.shape)}"
            )
        if self.pair_key_valid.shape != expected_mask_shape:
            raise ValueError(
                f"pair_key_valid must have shape {expected_mask_shape}, got {tuple(self.pair_key_valid.shape)}"
            )
        if self.pair_valid is not None and self.pair_valid.shape != (batch_size, num_tokens, num_tokens):
            raise ValueError(
                "pair_valid must have shape "
                f"{(batch_size, num_tokens, num_tokens)}, got {tuple(self.pair_valid.shape)}"
            )

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
                raise ValueError(
                    f"key_padding_mask must have shape {(batch_size, num_tokens)}, got {tuple(key_padding_mask.shape)}"
                )
            valid = ~key_padding_mask.bool().to(device=device)
        return cls(query_valid=valid, pair_key_valid=valid)

    def pair_mask(self) -> torch.Tensor:
        pair_valid = self.pair_key_valid[:, :, None] & self.pair_key_valid[:, None, :]
        if self.pair_valid is not None:
            pair_valid = pair_valid & self.pair_valid.bool()
        return pair_valid


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
        out_dtype = torch.float32
    else:
        q_work = q
        k1_work = k1
        v1_work = v1
        k2_work = k2
        v2_work = v2
        factorized_bias_work = factorized_bias
        angle_residual_work = angle_residual
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

        flat_logits = logits.flatten(-2).masked_fill(
            ~flat_mask,
            torch.finfo(logits.dtype).min,
        )
        attn = torch.softmax(flat_logits, dim=-1)
        attn = torch.where(flat_mask, attn, torch.zeros_like(attn)).view_as(logits)

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=training)

        if return_attn and attn_chunks is not None:
            attn_chunks.append(attn.detach())

        attn = attn.to(v1_work.dtype)
        tmp = torch.matmul(attn.transpose(-2, -1), v1_work.unsqueeze(-3))
        out_chunk = (tmp * v2_work.unsqueeze(-3)).sum(dim=-2)
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
        precision_mode: str,
        chunk_size: int,
        debug_torch_backward: bool,
    ) -> torch.Tensor:
        has_pair_valid = pair_valid.numel() > 0
        has_factorized_bias = u.numel() > 0
        has_low_rank_angle = angle_left.numel() > 0
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
            precision=precision_mode,
        )
        ctx.precision_mode = str(precision_mode)
        ctx.chunk_size = int(chunk_size)
        ctx.debug_torch_backward = bool(debug_torch_backward)
        ctx.has_pair_valid = has_pair_valid
        ctx.has_factorized_bias = has_factorized_bias
        ctx.has_low_rank_angle = has_low_rank_angle
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
            out_for_backward,
            lse,
        )
        return out

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
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
                    [
                        u.detach().requires_grad_(True),
                        v.detach().requires_grad_(True),
                        w.detach().requires_grad_(True),
                        gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_low_rank_angle:
                differentiable_inputs.extend(
                    [
                        angle_left.detach().requires_grad_(True),
                        angle_right.detach().requires_grad_(True),
                        angle_gate.detach().requires_grad_(True),
                    ]
                )

            with torch.enable_grad():
                q_re, k1_re, v1_re, k2_re, v2_re = differentiable_inputs[:5]
                factorized_bias = None
                cursor = 5
                if ctx.has_factorized_bias:
                    u_re, v_re, w_re, gate_re = differentiable_inputs[cursor : cursor + 4]
                    cursor += 4
                    factorized_bias = SimplicialFactorizedBias(
                        u=u_re,
                        v=v_re,
                        w=w_re,
                        gate=gate_re,
                    )
                angle_residual = None
                if ctx.has_low_rank_angle:
                    angle_left_re, angle_right_re, angle_gate_re = differentiable_inputs[cursor : cursor + 3]
                    angle_residual = SimplicialLowRankAngleResidual(
                        left=angle_left_re,
                        right=angle_right_re,
                        gate=angle_gate_re,
                    )
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
                    logit_bias_fn=None,
                    dropout_p=0.0,
                    training=False,
                    chunk_size=ctx.chunk_size,
                    return_attn=False,
                    fp32_core=True,
                )
                grads = torch.autograd.grad(
                    reference_out,
                    differentiable_inputs,
                    grad_out,
                    allow_unused=True,
                )

            dq, dk1, dv1, dk2, dv2 = grads[:5]
            du = dv = dw = dgate = None
            dangle_left = dangle_right = dangle_gate = None
            cursor = 5
            if ctx.has_factorized_bias:
                du, dv, dw, dgate = grads[cursor : cursor + 4]
                cursor += 4
            if ctx.has_low_rank_angle:
                dangle_left, dangle_right, dangle_gate = grads[cursor : cursor + 3]
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
                None,
                None,
                None,
            )

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
            precision=ctx.precision_mode,
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
            None,
            None,
            None,
        )


class TwoSimplicialAttention(nn.Module):
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
        debug_torch_backward: bool = False,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_heads ({num_heads}) when head_dim is None."
                )
            head_dim = dim // num_heads

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        impl = impl.lower()
        if impl not in {"auto", "torch", "triton"}:
            raise ValueError(f"Unsupported simplicial_impl: {impl}")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.chunk_size = int(chunk_size)
        self.impl = impl
        self.precision = normalize_simplicial_precision(precision)
        self.debug_torch_backward = bool(debug_torch_backward)
        self.scale = self.head_dim**-0.5
        self._last_impl_used = "torch"

        self.in_proj = nn.Linear(self.dim, 5 * self.inner_dim, bias=bias)
        self.out_proj = (
            nn.Linear(self.inner_dim, self.dim, bias=bias) if out_proj else nn.Identity()
        )

    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        return x.view(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_tokens, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, num_tokens, num_heads * head_dim)

    def _project_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k1, v1, k2, v2 = self.in_proj(x).chunk(5, dim=-1)
        q = self._split_heads(q, self.num_heads, self.head_dim) * self.scale
        k1 = self._split_heads(k1, self.num_heads, self.head_dim)
        v1 = self._split_heads(v1, self.num_heads, self.head_dim)
        k2 = self._split_heads(k2, self.num_heads, self.head_dim)
        v2 = self._split_heads(v2, self.num_heads, self.head_dim)
        return q, k1, v1, k2, v2

    def _triton_unavailable_reason(
        self,
        *,
        x: torch.Tensor,
        factorized_bias: SimplicialFactorizedBias | None,
        angle_residual: SimplicialLowRankAngleResidual | None,
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
        if factorized_bias is not None and factorized_bias.u.device != x.device:
            return "factorized bias tensors must be on the same CUDA device as the attention input"
        if angle_residual is not None:
            if (
                angle_residual.left.device != x.device
                or angle_residual.right.device != x.device
                or angle_residual.gate.device != x.device
            ):
                return "angle residual tensors must be on the same CUDA device as the attention input"
            if angle_residual.rank > 64:
                return f"the Triton backend only supports low-rank angle residual rank <= 64, got {angle_residual.rank}"
        return None

    def _triton_compute_dtype(self) -> torch.dtype:
        if self.precision == "bf16_tc":
            return torch.bfloat16
        return torch.float32

    def _cast_factorized_bias_for_triton(
        self,
        factorized_bias: SimplicialFactorizedBias | None,
        *,
        dtype: torch.dtype,
    ) -> SimplicialFactorizedBias | None:
        if factorized_bias is None:
            return None
        return SimplicialFactorizedBias(
            u=factorized_bias.u.to(dtype=dtype),
            v=factorized_bias.v.to(dtype=dtype),
            w=factorized_bias.w.to(dtype=dtype),
            gate=factorized_bias.gate.to(dtype=dtype),
        )

    def _cast_angle_residual_for_triton(
        self,
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
        factorized_bias = self._cast_factorized_bias_for_triton(factorized_bias, dtype=torch.float32)
        angle_residual = self._cast_angle_residual_for_triton(angle_residual, dtype=torch.float32)

        empty_float = q.new_empty(0)
        empty_bool = torch.empty(0, device=q.device, dtype=torch.bool)
        if factorized_bias is None:
            u = v = w = gate = empty_float
        else:
            factorized_bias.validate(
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                num_tokens=q.shape[2],
            )
            u, v, w, gate = factorized_bias.u, factorized_bias.v, factorized_bias.w, factorized_bias.gate
        if angle_residual is None:
            angle_left = angle_right = angle_gate = empty_float
        else:
            angle_residual.validate(
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                num_tokens=q.shape[2],
            )
            angle_left = angle_residual.left
            angle_right = angle_residual.right
            angle_gate = angle_residual.gate

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
            v.contiguous(),
            w.contiguous(),
            gate.contiguous(),
            angle_left.contiguous(),
            angle_right.contiguous(),
            angle_gate.contiguous(),
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
            factorized_bias.validate(
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_tokens=num_tokens,
            )
        if angle_residual is not None:
            angle_residual.validate(
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_tokens=num_tokens,
            )

        reason = self._triton_unavailable_reason(
            x=x,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
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

        q, k1, v1, k2, v2 = self._project_inputs(x)

        if impl_used == "triton":
            out = self._forward_triton(
                q,
                k1,
                v1,
                k2,
                v2,
                attention_mask=attention_mask,
                factorized_bias=factorized_bias,
                angle_residual=angle_residual,
            )
            y = self._merge_heads(out)
            return self.out_proj(y)

        out = simplicial_attention_torch_from_projected(
            q,
            k1,
            v1,
            k2,
            v2,
            attention_mask=attention_mask,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
            logit_bias_fn=logit_bias_fn,
            dropout_p=self.dropout,
            training=self.training,
            chunk_size=self.chunk_size,
            return_attn=return_attn,
        )
        if return_attn:
            out_heads, attn = out
            y = self._merge_heads(out_heads)
            return self.out_proj(y), attn
        y = self._merge_heads(out)
        return self.out_proj(y)
