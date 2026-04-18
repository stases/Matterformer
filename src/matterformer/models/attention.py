from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoSimplicialAttention(nn.Module):
    """Dense non-causal 2-simplicial attention in pure PyTorch."""

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

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.chunk_size = int(chunk_size)
        self.scale = self.head_dim**-0.5

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, C), got {tuple(x.shape)}")

        batch_size, num_tokens, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim

        q, k1, v1, k2, v2 = self.in_proj(x).chunk(5, dim=-1)
        q = self._split_heads(q, num_heads, head_dim) * self.scale
        k1 = self._split_heads(k1, num_heads, head_dim)
        v1 = self._split_heads(v1, num_heads, head_dim)
        k2 = self._split_heads(k2, num_heads, head_dim)
        v2 = self._split_heads(v2, num_heads, head_dim)

        pair_valid = None
        key_valid = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"key_padding_mask must have shape {(batch_size, num_tokens)}, got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.bool()
            key_valid = ~key_padding_mask
            pair_valid = key_valid[:, :, None] & key_valid[:, None, :]

        out = torch.empty((batch_size, num_heads, num_tokens, head_dim), device=x.device, dtype=x.dtype)
        attn_chunks: list[torch.Tensor] | None = [] if return_attn else None

        for start in range(0, num_tokens, self.chunk_size):
            end = min(num_tokens, start + self.chunk_size)
            q_chunk = q[:, :, start:end, :]
            qk2 = q_chunk.unsqueeze(-2) * k2.unsqueeze(-3)
            logits = torch.matmul(k1.unsqueeze(-3), qk2.transpose(-1, -2)).float()

            if logit_bias_fn is not None:
                logits = logits + logit_bias_fn(start, end, logits.dtype, logits.device)

            if pair_valid is not None:
                pair_mask = pair_valid[:, None, None, :, :]
                flat_mask = pair_mask.flatten(-2)
                flat_logits = logits.flatten(-2).masked_fill(
                    ~flat_mask,
                    torch.finfo(logits.dtype).min,
                )
                attn = torch.softmax(flat_logits, dim=-1)
                attn = torch.where(flat_mask, attn, torch.zeros_like(attn)).view_as(logits)
            else:
                attn = torch.softmax(logits.flatten(-2), dim=-1).view_as(logits)

            if self.dropout > 0.0:
                attn = F.dropout(attn, p=self.dropout, training=self.training)

            if return_attn and attn_chunks is not None:
                attn_chunks.append(attn.detach())

            attn = attn.to(v1.dtype)
            tmp = torch.matmul(attn.transpose(-2, -1), v1.unsqueeze(-3))
            out_chunk = (tmp * v2.unsqueeze(-3)).sum(dim=-2)
            if key_valid is not None:
                q_valid = key_valid[:, start:end].to(dtype=out_chunk.dtype)
                out_chunk = out_chunk * q_valid[:, None, :, None]
            out[:, :, start:end, :] = out_chunk

        y = self._merge_heads(out)
        y = self.out_proj(y)
        if return_attn and attn_chunks is not None:
            return y, torch.cat(attn_chunks, dim=2)
        return y
