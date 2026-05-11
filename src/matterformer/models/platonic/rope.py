from __future__ import annotations

import math

import torch
from torch import nn

from matterformer.models.platonic.groups import PLATONIC_GROUPS


class PlatonicRoPE(nn.Module):
    """Group-aware 3D RoPE for tensors shaped ``[..., G, H, D]``."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
        head_dim: int,
        *,
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        learned_freqs: bool = True,
        freq_init: str = "spiral",
    ) -> None:
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("Matterformer PlatonicRoPE currently supports spatial_dims=3")
        solid_name = solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unknown Platonic group {solid_name!r}")
        group = PLATONIC_GROUPS[solid_name]
        self.G = group.G
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if embed_dim % self.G != 0:
            raise ValueError("embed_dim must be divisible by group order")
        self.num_pairs = self.head_dim // 2
        self.register_buffer("group_elements", group.elements, persistent=False)
        freq_init = str(freq_init).lower()
        if freq_init == "spiral":
            freqs = self._spiral_frequencies(self.num_heads, self.num_pairs, float(freq_sigma))
        elif freq_init == "random":
            freqs = torch.randn(self.num_heads, self.num_pairs, spatial_dims, dtype=torch.float32) * float(freq_sigma)
        else:
            raise ValueError("freq_init must be one of {'spiral', 'random'}")
        if learned_freqs:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)

    @staticmethod
    def _spiral_frequencies(num_heads: int, num_pairs: int, freq_sigma: float) -> torch.Tensor:
        idx = torch.arange(num_pairs, dtype=torch.float32) + 0.5
        magnitudes = torch.linspace(freq_sigma / max(num_pairs, 1), freq_sigma, num_pairs)
        head_phases = torch.linspace(0.0, 2.0 * math.pi, num_heads + 1, dtype=torch.float32)[:-1, None]
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        y = (1.0 - 2.0 * idx / float(num_pairs)).clamp(-1.0, 1.0)
        radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
        theta = 2.0 * math.pi * idx[None, :] / phi + head_phases
        x = radius[None, :] * torch.cos(theta)
        z = radius[None, :] * torch.sin(theta)
        return torch.stack([x, y[None, :].expand(num_heads, -1), z], dim=-1) * magnitudes.view(1, -1, 1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        *leading, group_order, num_heads, head_dim = x.shape
        if (group_order, num_heads, head_dim) != (self.G, self.num_heads, self.head_dim):
            raise ValueError(
                f"Expected x trailing shape {(self.G, self.num_heads, self.head_dim)}, "
                f"got {(group_order, num_heads, head_dim)}"
            )
        if pos.shape[-1] != 3 or tuple(pos.shape[:-1]) != tuple(leading):
            raise ValueError(f"pos shape must be {tuple(leading) + (3,)}, got {tuple(pos.shape)}")
        freqs = self.freqs.to(device=x.device, dtype=torch.float32)
        frames = self.group_elements.to(device=x.device, dtype=torch.float32)
        rotated_freqs = torch.einsum("gde,hfe->ghfd", frames, freqs)
        angles = torch.einsum("...d,ghfd->...ghf", pos.to(dtype=torch.float32), rotated_freqs)
        cos = torch.cos(angles).to(dtype=x.dtype)
        sin = torch.sin(angles).to(dtype=x.dtype)
        if inverse:
            sin = -sin
        paired = x.view(*leading, self.G, self.num_heads, self.num_pairs, 2)
        x0, x1 = paired.unbind(dim=-1)
        out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
        return out.reshape(*leading, self.G, self.num_heads, self.head_dim)
