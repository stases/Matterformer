from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _canonicalize_mha_position_mode(mode: str) -> str:
    mode = mode.lower().replace("-", "_")
    if mode in {"disabled", "off", "false", "no"}:
        return "none"
    if mode in {"rotary", "mha_rope", "rotary_position_embedding"}:
        return "rope"
    if mode not in {"none", "rope"}:
        raise ValueError("mha_position_mode must be one of {'none', 'rope'}")
    return mode


class RotaryPositionEmbedding3D(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        freq_sigma: float = 1.0,
        learned_freqs: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rotary_dim = (self.head_dim // 2) * 2
        if self.rotary_dim == 0:
            raise ValueError("RoPE requires attention head_dim >= 2")
        self.num_pairs = self.rotary_dim // 2
        freqs = self._build_spiral_frequencies(
            num_heads=self.num_heads,
            num_pairs=self.num_pairs,
            freq_sigma=float(freq_sigma),
        )
        if learned_freqs:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)

    @staticmethod
    def _build_spiral_frequencies(
        *,
        num_heads: int,
        num_pairs: int,
        freq_sigma: float,
    ) -> torch.Tensor:
        indices = torch.arange(num_pairs, dtype=torch.float32) + 0.5
        magnitudes = torch.linspace(
            freq_sigma / max(num_pairs, 1),
            freq_sigma,
            num_pairs,
            dtype=torch.float32,
        )
        head_phases = torch.linspace(0.0, 2.0 * math.pi, num_heads + 1, dtype=torch.float32)[:-1]
        head_phases = head_phases[:, None]

        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        y = (1.0 - 2.0 * indices / float(num_pairs)).clamp(min=-1.0, max=1.0)
        radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
        theta = (2.0 * math.pi * indices / golden_ratio)[None, :] + head_phases
        x = radius[None, :] * torch.cos(theta)
        z = radius[None, :] * torch.sin(theta)
        directions = torch.stack([x, y[None, :].expand(num_heads, -1), z], dim=-1)
        return directions * magnitudes.view(1, num_pairs, 1)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"mha_rope positions must have shape (B, T, 3), got {tuple(positions.shape)}")
        if positions.shape[:2] != (x.shape[0], x.shape[2]):
            raise ValueError(
                f"mha_rope positions shape {tuple(positions.shape[:2])} must match attention tokens "
                f"{(x.shape[0], x.shape[2])}"
            )

        freqs = self.freqs.to(device=x.device, dtype=torch.float32)
        angles = torch.einsum("btd,hfd->bhtf", positions.to(device=x.device, dtype=torch.float32), freqs)
        cos = torch.cos(angles).to(dtype=x.dtype)
        sin = torch.sin(angles).to(dtype=x.dtype)
        if inverse:
            sin = -sin

        x_rot = x[..., : self.rotary_dim].view(*x.shape[:-1], self.num_pairs, 2)
        x0, x1 = x_rot.unbind(dim=-1)
        x_rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
        x_rotated = x_rotated.reshape(*x.shape[:-1], self.rotary_dim)
        if self.rotary_dim == self.head_dim:
            return x_rotated
        return torch.cat([x_rotated, x[..., self.rotary_dim :]], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_rotary(q, positions), self._apply_rotary(k, positions)


class RotaryMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        rope_freq_sigma: float = 1.0,
        rope_learned_freqs: bool = False,
        rope_use_key: bool = True,
        rope_on_values: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.rope_use_key = bool(rope_use_key)
        self.rope_on_values = bool(rope_on_values)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.embed_dim, self.embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim)) if bias else None
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.rope = RotaryPositionEmbedding3D(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            freq_sigma=rope_freq_sigma,
            learned_freqs=rope_learned_freqs,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        return x.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embed_dim)

    def _coerce_attn_mask(
        self,
        attn_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attn_mask is None:
            return None
        if attn_mask.ndim == 2:
            if attn_mask.shape != (num_tokens, num_tokens):
                raise ValueError(f"attn_mask must have shape {(num_tokens, num_tokens)}, got {tuple(attn_mask.shape)}")
            attn_mask = attn_mask.view(1, 1, num_tokens, num_tokens)
        elif attn_mask.ndim == 3:
            if attn_mask.shape != (batch_size * self.num_heads, num_tokens, num_tokens):
                raise ValueError(
                    "3D attn_mask must have shape "
                    f"{(batch_size * self.num_heads, num_tokens, num_tokens)}, got {tuple(attn_mask.shape)}"
                )
            attn_mask = attn_mask.view(batch_size, self.num_heads, num_tokens, num_tokens)
        elif attn_mask.ndim == 4:
            if attn_mask.shape != (batch_size, self.num_heads, num_tokens, num_tokens):
                raise ValueError(
                    "4D attn_mask must have shape "
                    f"{(batch_size, self.num_heads, num_tokens, num_tokens)}, got {tuple(attn_mask.shape)}"
                )
        else:
            raise ValueError(f"attn_mask must have 2, 3, or 4 dims, got {attn_mask.ndim}")
        return attn_mask.to(device=device, dtype=dtype)

    def _key_padding_bias(
        self,
        key_padding_mask: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if key_padding_mask is None:
            return None
        if key_padding_mask.ndim != 2:
            raise ValueError(f"key_padding_mask must have shape (B, T), got {tuple(key_padding_mask.shape)}")
        if key_padding_mask.dtype == torch.bool:
            bias = torch.zeros(key_padding_mask.shape, device=device, dtype=dtype)
            return bias.masked_fill(key_padding_mask.to(device=device), torch.finfo(dtype).min)[:, None, None, :]
        return key_padding_mask.to(device=device, dtype=dtype)[:, None, None, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if need_weights:
            raise ValueError("RotaryMultiheadAttention does not support need_weights=True")
        if query is not key or key is not value:
            raise ValueError("RotaryMultiheadAttention only supports self-attention")
        if positions is None:
            raise ValueError("positions must be provided when mha_position_mode='rope'")
        batch_size, num_tokens, _ = query.shape
        q = F.linear(
            query,
            self.in_proj_weight[: self.embed_dim],
            None if self.in_proj_bias is None else self.in_proj_bias[: self.embed_dim],
        )
        if self.rope_use_key:
            k = F.linear(
                query,
                self.in_proj_weight[self.embed_dim : 2 * self.embed_dim],
                None if self.in_proj_bias is None else self.in_proj_bias[self.embed_dim : 2 * self.embed_dim],
            )
        else:
            k = torch.ones_like(q)
        v = F.linear(
            query,
            self.in_proj_weight[2 * self.embed_dim :],
            None if self.in_proj_bias is None else self.in_proj_bias[2 * self.embed_dim :],
        )
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        q, k = self.rope(q, k, positions)
        if self.rope_on_values:
            v = self.rope._apply_rotary(v, positions)

        additive_mask = self._coerce_attn_mask(
            attn_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            dtype=q.dtype,
            device=q.device,
        )
        key_bias = self._key_padding_bias(key_padding_mask, dtype=q.dtype, device=q.device)
        if key_bias is not None:
            additive_mask = key_bias if additive_mask is None else additive_mask + key_bias

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        if self.rope_on_values:
            attn_out = self.rope._apply_rotary(attn_out, positions, inverse=True)
        return self.out_proj(self._merge_heads(attn_out)), None



class RegularAttention(nn.Module):
    """Self-attention wrapper for standard MHA and 3D rotary MHA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        position_mode: str = "none",
        rope_freq_sigma: float = 1.0,
        rope_learned_freqs: bool = False,
        rope_use_key: bool = True,
        rope_on_values: bool = False,
    ) -> None:
        super().__init__()
        self.position_mode = _canonicalize_mha_position_mode(position_mode)
        if self.position_mode == "rope":
            self.attn = RotaryMultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                rope_freq_sigma=rope_freq_sigma,
                rope_learned_freqs=rope_learned_freqs,
                rope_use_key=rope_use_key,
                rope_on_values=rope_on_values,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.position_mode == "rope":
            return self.attn(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                positions=positions,
            )
        return self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
