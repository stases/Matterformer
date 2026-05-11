from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from matterformer.models.platonic.groups import PLATONIC_GROUPS
from matterformer.models.platonic.linear import PlatonicLinear
from matterformer.models.platonic.rope import PlatonicRoPE


class GroupLayerNorm(nn.Module):
    def __init__(self, group_order: int, channels_per_group: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.group_order = int(group_order)
        self.channels_per_group = int(channels_per_group)
        self.norm = nn.LayerNorm(self.channels_per_group, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.group_order, self.channels_per_group)
        x = self.norm(x)
        return x.reshape(shape)


class PlatonicAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        solid_name: str = "tetrahedron",
        *,
        dropout: float = 0.0,
        rope_sigma: float = 0.5,
        learned_freqs: bool = True,
        freq_init: str = "spiral",
        use_key: bool = False,
        rope_on_values: bool = True,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        solid_name = solid_name.lower()
        group = PLATONIC_GROUPS[solid_name]
        self.group_order = group.G
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.d_model % self.group_order != 0:
            raise ValueError("d_model must be divisible by the group order")
        if self.num_heads % self.group_order != 0:
            raise ValueError("num_heads must be divisible by the group order")
        self.heads_per_frame = self.num_heads // self.group_order
        group_dim = self.d_model // self.group_order
        if group_dim % self.heads_per_frame != 0:
            raise ValueError("channels per group must be divisible by heads per frame")
        self.head_dim = group_dim // self.heads_per_frame
        if self.head_dim % 2 != 0:
            raise ValueError("Platonic attention head_dim must be even")
        self.dropout = float(dropout)
        self.use_key = bool(use_key)
        self.rope_on_values = bool(rope_on_values)
        self.attention_backend = str(attention_backend).lower()
        if self.attention_backend == "default":
            self.attention_backend = "sdpa"
        if self.attention_backend not in {"sdpa", "flash"}:
            raise ValueError("attention_backend must be one of {'sdpa', 'flash'}")
        self.q_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name)
        self.v_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name)
        self.k_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name) if self.use_key else None
        self.out_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name)
        self.rope = PlatonicRoPE(
            embed_dim=self.d_model,
            num_heads=self.heads_per_frame,
            solid_name=solid_name,
            head_dim=self.head_dim,
            freq_sigma=rope_sigma,
            learned_freqs=learned_freqs,
            freq_init=freq_init,
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.group_order, self.heads_per_frame, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split(self.q_proj(x))
        v = self._split(self.v_proj(x))
        k = self._split(self.k_proj(x)) if self.k_proj is not None else torch.ones_like(q)
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        if self.rope_on_values:
            v = self.rope(v, pos)
        batch_size, num_tokens = x.shape[:2]
        q = q.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if pad_mask is not None:
            if self.attention_backend == "flash":
                attn_mask = (~pad_mask)[:, None, None, :]
            else:
                attn_mask = torch.zeros_like(pad_mask, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(pad_mask, torch.finfo(q.dtype).min)[:, None, None, :]
        orig_dtype = q.dtype
        if self.attention_backend == "flash":
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.to(orig_dtype)
        out = out.transpose(1, 2).contiguous().view(
            batch_size,
            num_tokens,
            self.group_order,
            self.heads_per_frame,
            self.head_dim,
        )
        if self.rope_on_values:
            out = self.rope(out, pos, inverse=True)
        return self.out_proj(out.reshape(batch_size, num_tokens, self.d_model))


class PlatonicBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        solid_name: str = "tetrahedron",
        *,
        dropout: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        layer_norm_eps: float = 1e-6,
        rope_sigma: float = 0.5,
        learned_freqs: bool = True,
        freq_init: str = "spiral",
        use_key: bool = False,
        rope_on_values: bool = True,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        solid_name = solid_name.lower()
        group = PLATONIC_GROUPS[solid_name]
        if d_model % group.G != 0:
            raise ValueError("d_model must be divisible by group order")
        self.group_order = group.G
        self.channels_per_group = d_model // group.G
        self.norm1 = GroupLayerNorm(group.G, self.channels_per_group, eps=layer_norm_eps)
        self.norm2 = GroupLayerNorm(group.G, self.channels_per_group, eps=layer_norm_eps)
        self.attn = PlatonicAttention(
            d_model,
            nhead,
            solid_name,
            dropout=dropout,
            rope_sigma=rope_sigma,
            learned_freqs=learned_freqs,
            freq_init=freq_init,
            use_key=use_key,
            rope_on_values=rope_on_values,
            attention_backend=attention_backend,
        )
        self.linear1 = PlatonicLinear(d_model, dim_feedforward, solid=solid_name)
        self.linear2 = PlatonicLinear(dim_feedforward, d_model, solid=solid_name)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), pos=pos, pad_mask=pad_mask))
        x = x + self.dropout(self.linear2(self.activation(self.linear1(self.norm2(x)))))
        return x
