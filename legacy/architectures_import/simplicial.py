from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoSimplicialAttention(nn.Module):
    """
    Dense non-causal 2-simplicial attention (non-Triton, pure PyTorch).

    This implements the trilinear 2-simplicial attention:

        logits[i, j, k] = <q_i, k1_j, k2_k> / sqrt(D)
                        = sum_d q_i[d] * k1_j[d] * k2_k[d] / sqrt(D)

        attn[i, j, k]   = softmax over the joint (j, k) axes

        out_i[d]        = sum_{j,k} attn[i,j,k] * (v1_j[d] * v2_k[d])

    over all non-padded tokens:
        j, k in {0, ..., T-1}

    Notes:
    - This is a reference / baseline implementation that relies on autograd.
      It is not FlashAttention-fast; the Triton kernel in the paper is much faster.
    - `chunk_size` controls peak memory by processing queries in blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: Optional[int] = None,
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
            raise ValueError("head_dim must be a positive integer.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.chunk_size = int(chunk_size)

        self.scale = self.head_dim ** -0.5
        self.in_proj = nn.Linear(self.dim, 5 * self.inner_dim, bias=bias)
        self.out_proj = (
            nn.Linear(self.inner_dim, self.dim, bias=bias) if out_proj else nn.Identity()
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, head_dim={self.head_dim}, "
            f"dropout={self.dropout}, chunk_size={self.chunk_size}"
        )

    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        logit_bias_fn: Optional[Callable[[int, int, torch.dtype, torch.device], torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, C)
            key_padding_mask: optional (B, T) bool, True = padding/masked token
            logit_bias_fn: optional callback returning an additive logit bias chunk
                with shape (B, H, Q, T, T) for the query block [start:end).
            return_attn: if True, also returns attention weights (B, H, T, T, T).
                         WARNING: this can be extremely memory-heavy.

        Returns:
            y: (B, T, C)
            optionally attn: (B, H, T, T, T)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, T, C), got {tuple(x.shape)}")

        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q, k1, v1, k2, v2 = self.in_proj(x).chunk(5, dim=-1)
        q = self._split_heads(q, H, D) * self.scale
        k1 = self._split_heads(k1, H, D)
        v1 = self._split_heads(v1, H, D)
        k2 = self._split_heads(k2, H, D)
        v2 = self._split_heads(v2, H, D)

        pair_valid = None
        key_valid = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, T):
                raise ValueError(
                    f"key_padding_mask must have shape (B, T)={(B, T)}, got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
            key_valid = ~key_padding_mask
            pair_valid = key_valid[:, :, None] & key_valid[:, None, :]

        out = torch.empty((B, H, T, D), device=x.device, dtype=x.dtype)
        attn_chunks = [] if return_attn else None

        for start in range(0, T, self.chunk_size):
            end = min(T, start + self.chunk_size)
            q_blk = q[:, :, start:end, :]

            # Dense non-causal triplet attention over all non-padded keys.
            qk2 = q_blk.unsqueeze(-2) * k2.unsqueeze(-3)  # (B, H, Q, T, D)
            logits = torch.matmul(
                k1.unsqueeze(-3),
                qk2.transpose(-1, -2),
            ).float()  # (B, H, Q, T, T)

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

            if return_attn:
                attn_chunks.append(attn.detach())

            attn = attn.to(v1.dtype)

            # out_d = v1_d^T @ attn @ v2_d  (per dimension d)
            tmp = torch.matmul(attn.transpose(-2, -1), v1.unsqueeze(-3))  # (B, H, Q, T, D)
            out_blk = (tmp * v2.unsqueeze(-3)).sum(dim=-2)  # (B, H, Q, D)
            if key_valid is not None:
                q_valid = key_valid[:, start:end].to(dtype=out_blk.dtype)
                out_blk = out_blk * q_valid[:, None, :, None]
            out[:, :, start:end, :] = out_blk

        y = self._merge_heads(out)
        y = self.out_proj(y)

        if return_attn:
            attn_full = torch.cat(attn_chunks, dim=2)
            return y, attn_full

        return y
