from __future__ import annotations

import torch
from torch import nn

from matterformer.data.mof_bwdb import MOF_BLOCK_PAD_TOKEN, MOF_NUM_BLOCK_TYPES
from matterformer.geometry.adapters import BaseGeometryAdapter, PeriodicGeometryAdapter
from matterformer.models.embeddings import FourierCoordEmbedder, LatticeEmbedder, TimeEmbedder
from matterformer.models.transformer import (
    GeometryBiasBuilder,
    SimplicialGeometryBias,
    TransformerTrunk,
)


def _masked_mean(features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(dtype=features.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (features * valid.unsqueeze(-1)).sum(dim=1) / denom


def _mod1(coords: torch.Tensor) -> torch.Tensor:
    return coords - torch.floor(coords)


class MOFStage1EDMModel(nn.Module):
    def __init__(
        self,
        *,
        block_feature_dim: int,
        num_block_types: int = MOF_NUM_BLOCK_TYPES,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        attn_type: str = "simplicial",
        simplicial_geom_mode: str = "factorized",
        simplicial_impl: str = "auto",
        simplicial_precision: str = "bf16_tc",
        simplicial_angle_rank: int = 16,
        simplicial_message_mode: str = "none",
        simplicial_message_rank: int = 16,
        geometry_adapter: BaseGeometryAdapter | None = None,
        use_geometry_bias: bool = True,
        lattice_repr: str = "ltri",
        pbc_radius: int = 1,
        coord_embed_mode: str = "rff",
        coord_n_freqs: int = 32,
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        lattice_embed_mode: str = "rff",
        lattice_rff_dim: int = 256,
        lattice_rff_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        geometry_adapter = geometry_adapter or PeriodicGeometryAdapter(
            pbc_radius=pbc_radius,
            lattice_repr=lattice_repr,
        )
        simplicial_geom_mode = simplicial_geom_mode.lower()
        simplicial_message_mode = simplicial_message_mode.lower()
        if simplicial_geom_mode not in {"none", "factorized", "angle_residual", "angle_low_rank"}:
            raise ValueError(
                "simplicial_geom_mode must be one of {'none', 'factorized', 'angle_residual', 'angle_low_rank'}"
            )
        if simplicial_message_mode not in {"none", "low_rank"}:
            raise ValueError("simplicial_message_mode must be one of {'none', 'low_rank'}")
        geometry_bias = None
        simplicial_geometry_bias = None
        effective_message_mode = (
            simplicial_message_mode
            if use_geometry_bias and attn_type.lower() == "simplicial"
            else "none"
        )
        if use_geometry_bias:
            if attn_type.lower() == "simplicial":
                if simplicial_geom_mode != "none" or effective_message_mode != "none":
                    simplicial_geometry_bias = SimplicialGeometryBias(
                        n_heads=n_heads,
                        mode=simplicial_geom_mode,
                        angle_residual_rank=simplicial_angle_rank,
                        message_mode=effective_message_mode,
                        message_rank=simplicial_message_rank,
                        use_periodic_features=True,
                        use_noise_gate=True,
                    )
            else:
                geometry_bias = GeometryBiasBuilder(
                    n_heads=n_heads,
                    use_distance_bias=True,
                    use_edge_bias=True,
                    use_periodic_features=True,
                    use_noise_gate=True,
                )

        self.block_pad_token = MOF_BLOCK_PAD_TOKEN
        self.block_feature_proj = nn.Sequential(
            nn.Linear(block_feature_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.block_type_embedding = nn.Embedding(
            num_block_types + 1,
            d_model,
            padding_idx=self.block_pad_token,
        )
        self.coord_embedding = FourierCoordEmbedder(
            d_model=d_model,
            n_freqs=coord_n_freqs,
            mode=coord_embed_mode,
            rff_dim=coord_rff_dim,
            rff_sigma=coord_rff_sigma,
        )
        self.lattice_embedding = LatticeEmbedder(
            d_model=d_model,
            mode=lattice_embed_mode,
            rff_dim=lattice_rff_dim,
            rff_sigma=lattice_rff_sigma,
        )
        self.segment_embedding = nn.Embedding(2, d_model)
        self.conditioning = TimeEmbedder(d_model)
        self.trunk = TransformerTrunk(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            attn_type=attn_type,
            simplicial_impl=simplicial_impl,
            simplicial_precision=simplicial_precision,
            simplicial_message_mode=effective_message_mode,
            simplicial_message_rank=simplicial_message_rank,
            geometry_adapter=geometry_adapter,
            geometry_bias=geometry_bias,
            simplicial_geometry_bias=simplicial_geometry_bias,
        )
        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )
        self.lattice_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 6),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        block_type_ids: torch.Tensor,
        coords_noisy: torch.Tensor,
        pad_mask: torch.Tensor,
        sigma: torch.Tensor,
        *,
        lattice: torch.Tensor,
        lattice_bias_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.ndim == 2 and sigma.shape[-1] == 1:
            sigma = sigma[:, 0]
        if sigma.ndim != 1:
            raise ValueError(f"sigma must have shape (B,) or (B, 1), got {tuple(sigma.shape)}")

        pad_mask = pad_mask.bool()
        block_features = torch.log1p(block_features.float())
        coords_wrapped = _mod1(coords_noisy.float())
        lattice = lattice.float()
        lattice_bias_latent = lattice.float() if lattice_bias_latent is None else lattice_bias_latent.float()
        sigma_condition = torch.log(sigma.clamp_min(1e-8)) / 4.0
        lattice_features = self.lattice_embedding(lattice)
        cond = self.conditioning(sigma_condition, sigma_condition)

        block_tokens = (
            self.block_feature_proj(block_features)
            + self.block_type_embedding(block_type_ids.clamp(min=0, max=self.block_pad_token))
            + self.coord_embedding(coords_wrapped)
            + self.segment_embedding.weight[0]
        )
        block_tokens = block_tokens.masked_fill(pad_mask[..., None], 0.0)
        lattice_token = lattice_features[:, None, :] + self.segment_embedding.weight[1]
        token_features = torch.cat([block_tokens, lattice_token], dim=1)
        sequence_pad_mask = torch.cat(
            [
                pad_mask,
                torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
        trunk_out = self.trunk(
            token_features,
            cond,
            pad_mask=sequence_pad_mask,
            coords=coords_wrapped,
            lattice=lattice_bias_latent,
            sigma=sigma,
        )
        coord_raw = self.coord_head(trunk_out[:, :-1, :]).masked_fill(pad_mask[..., None], 0.0)
        lattice_raw = self.lattice_head(trunk_out[:, -1, :])
        return coord_raw, lattice_raw
