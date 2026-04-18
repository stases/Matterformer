from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedder(nn.Module):
    def __init__(self, d_model: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim * 2, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def _timestep_embedding(
        self,
        t: torch.Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(f"t must have shape (B,) or (B, 1), got {tuple(t.shape)}")
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        args = 2.0 * math.pi * t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t_global: torch.Tensor, t_atomic: torch.Tensor) -> torch.Tensor:
        global_emb = self._timestep_embedding(t_global, self.freq_dim)
        atomic_emb = self._timestep_embedding(t_atomic, self.freq_dim)
        return self.mlp(torch.cat([global_emb, atomic_emb], dim=-1))


class FourierCoordEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_freqs: int = 32,
        mode: str = "rff",
        rff_dim: int | None = None,
        rff_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"fourier", "rff"}:
            raise ValueError(f"Unsupported coord embed mode: {mode}")

        if self.mode == "fourier":
            self.n_freqs = int(n_freqs)
            self.register_buffer(
                "freqs",
                torch.arange(1, self.n_freqs + 1, dtype=torch.float32),
                persistent=False,
            )
            in_dim = 6 * self.n_freqs
        else:
            self.n_rff = int(rff_dim) if rff_dim is not None else int(n_freqs)
            self.register_buffer(
                "proj",
                torch.randn(3, self.n_rff) * float(rff_sigma),
                persistent=False,
            )
            in_dim = 2 * self.n_rff

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (B, N, 3), got {tuple(coords.shape)}")
        if self.mode == "fourier":
            args = 2.0 * math.pi * coords[..., None] * self.freqs.view(1, 1, 1, -1)
            feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            feats = feats.reshape(coords.shape[0], coords.shape[1], -1)
        else:
            proj = coords @ self.proj
            args = 2.0 * math.pi * proj
            feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(feats)


class LatticeEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        mode: str = "rff",
        rff_dim: int = 256,
        rff_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"mlp", "rff"}:
            raise ValueError(f"Unsupported lattice embed mode: {mode}")

        if self.mode == "mlp":
            in_dim = 6
        else:
            self.n_rff = int(rff_dim)
            self.register_buffer(
                "proj",
                torch.randn(6, self.n_rff) * float(rff_sigma),
                persistent=False,
            )
            in_dim = 2 * self.n_rff

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, lattice_features: torch.Tensor) -> torch.Tensor:
        if lattice_features.ndim != 2 or lattice_features.shape[-1] != 6:
            raise ValueError(
                f"lattice_features must have shape (B, 6), got {tuple(lattice_features.shape)}"
            )
        if self.mode == "mlp":
            feats = lattice_features
        else:
            proj = lattice_features @ self.proj
            args = 2.0 * math.pi * proj
            feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(feats)


class MaskEmbedder(nn.Module):
    def __init__(self, d_model: int, in_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(mask.float())


class TokenEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vz: int,
        n_freqs: int = 32,
        coord_embed_mode: str = "rff",
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        lattice_embed_mode: str = "rff",
        lattice_rff_dim: int = 256,
        lattice_rff_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.type_embed = nn.Embedding(vz + 2, d_model, padding_idx=0)
        self.coord_embed = FourierCoordEmbedder(
            d_model=d_model,
            n_freqs=n_freqs,
            mode=coord_embed_mode,
            rff_dim=coord_rff_dim,
            rff_sigma=coord_rff_sigma,
        )
        self.lattice_embed = LatticeEmbedder(
            d_model=d_model,
            mode=lattice_embed_mode,
            rff_dim=lattice_rff_dim,
            rff_sigma=lattice_rff_sigma,
        )
        self.mask_embed_atom = MaskEmbedder(d_model=d_model, in_dim=2)
        self.mask_embed_lat = MaskEmbedder(d_model=d_model, in_dim=1)
        self.segment_embed = nn.Embedding(2, d_model)

    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        lattice_features: torch.Tensor,
        atom_type_mask: torch.Tensor,
        coord_mask: torch.Tensor,
        lattice_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_atoms = self.type_embed(atom_types) + self.coord_embed(coords)
        atom_mask = torch.stack([atom_type_mask, coord_mask], dim=-1)
        h_atoms = h_atoms + self.mask_embed_atom(atom_mask) + self.segment_embed.weight[0]

        h_lattice = self.lattice_embed(lattice_features)
        h_lattice = h_lattice + self.mask_embed_lat(lattice_mask) + self.segment_embed.weight[1]
        h_lattice = h_lattice[:, None, :]

        tokens = torch.cat([h_atoms, h_lattice], dim=1)
        sequence_pad_mask = torch.cat(
            [
                pad_mask.bool(),
                torch.zeros(
                    (pad_mask.shape[0], 1),
                    device=pad_mask.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        return tokens, sequence_pad_mask
