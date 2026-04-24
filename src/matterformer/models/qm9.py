from __future__ import annotations

import torch
from torch import nn

from matterformer.data.qm9 import QM9_ATOM_PAD_TOKEN, QM9_NUM_ATOM_TYPES
from matterformer.geometry.adapters import BaseGeometryAdapter, NonPeriodicGeometryAdapter
from matterformer.models.embeddings import FourierCoordEmbedder, TimeEmbedder
from matterformer.models.transformer import (
    GeometryBiasBuilder,
    LearnedNullConditioning,
    MhaFactorizedGeometryBias,
    SimplicialGeometryBias,
    TransformerTrunk,
)


def _masked_mean(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(dtype=coords.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)


def _center_coords(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    coords = coords - _masked_mean(coords, pad_mask)
    return coords.masked_fill(pad_mask[..., None], 0.0)


def _masked_rms_radius(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(dtype=coords.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_sq_norm = (coords.square().sum(dim=-1) * valid).sum(dim=1, keepdim=True) / denom
    return torch.sqrt(mean_sq_norm.clamp_min(1e-8)).unsqueeze(-1)


class QM9RegressionModel(nn.Module):
    def __init__(
        self,
        *,
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
        readout_mode: str = "cls",
        mha_geom_bias_mode: str = "standard",
        geometry_adapter: BaseGeometryAdapter | None = None,
        use_geometry_bias: bool = True,
    ) -> None:
        super().__init__()
        geometry_adapter = geometry_adapter or NonPeriodicGeometryAdapter()
        simplicial_geom_mode = simplicial_geom_mode.lower()
        mha_geom_bias_mode = mha_geom_bias_mode.lower()
        readout_mode = readout_mode.lower()
        if simplicial_geom_mode not in {"none", "factorized", "angle_residual"}:
            raise ValueError(
                "simplicial_geom_mode must be one of {'none', 'factorized', 'angle_residual'}"
            )
        if mha_geom_bias_mode not in {"standard", "factorized_marginal"}:
            raise ValueError(
                "mha_geom_bias_mode must be one of {'standard', 'factorized_marginal'}"
            )
        if readout_mode not in {"cls", "sum", "mean"}:
            raise ValueError("readout_mode must be one of {'cls', 'sum', 'mean'}")
        geometry_bias = None
        simplicial_geometry_bias = None
        if use_geometry_bias:
            if attn_type.lower() == "simplicial":
                if simplicial_geom_mode != "none":
                    simplicial_geometry_bias = SimplicialGeometryBias(
                        n_heads=n_heads,
                        mode=simplicial_geom_mode,
                        use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                        use_noise_gate=False,
                    )
            else:
                if mha_geom_bias_mode == "factorized_marginal":
                    geometry_bias = MhaFactorizedGeometryBias(
                        n_heads=n_heads,
                        use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                        use_noise_gate=False,
                    )
                else:
                    geometry_bias = GeometryBiasBuilder(
                        n_heads=n_heads,
                        use_distance_bias=True,
                        use_edge_bias=True,
                        use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                        use_noise_gate=False,
                    )

        self.pad_token = QM9_ATOM_PAD_TOKEN
        self.readout_mode = readout_mode
        self.atom_embedding = nn.Embedding(
            QM9_NUM_ATOM_TYPES + 1,
            d_model,
            padding_idx=self.pad_token,
        )
        self.coord_embedding = FourierCoordEmbedder(d_model=d_model, mode="rff", n_freqs=32)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if readout_mode == "cls" else None
        self.conditioning = LearnedNullConditioning(d_model)
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
            geometry_adapter=geometry_adapter,
            geometry_bias=geometry_bias,
            simplicial_geometry_bias=simplicial_geometry_bias,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        pad_mask: torch.Tensor,
        lattice: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = atom_types.shape[0]
        pad_mask = pad_mask.bool()
        centered_coords = _center_coords(coords, pad_mask)
        coord_scale = _masked_rms_radius(centered_coords, pad_mask)
        coords_for_embedding = centered_coords / coord_scale.clamp_min(1e-8)
        token_features = self.atom_embedding(atom_types.clamp(min=0, max=self.pad_token))
        coord_features = self.coord_embedding(coords_for_embedding).masked_fill(
            pad_mask[..., None], 0.0
        )
        token_features = token_features + coord_features
        if self.readout_mode == "cls":
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            token_features = torch.cat([token_features, cls_token], dim=1)
            full_pad_mask = torch.cat(
                [
                    pad_mask,
                    torch.zeros(batch_size, 1, device=pad_mask.device, dtype=torch.bool),
                ],
                dim=1,
            )
        else:
            full_pad_mask = pad_mask
        trunk_out = self.trunk(
            token_features,
            self.conditioning(batch_size),
            pad_mask=full_pad_mask,
            coords=centered_coords,
            lattice=lattice,
        )
        if self.readout_mode == "cls":
            pooled = trunk_out[:, -1, :]
        else:
            atom_out = trunk_out[:, : centered_coords.shape[1], :]
            atom_out = atom_out.masked_fill(pad_mask[..., None], 0.0)
            if self.readout_mode == "sum":
                pooled = atom_out.sum(dim=1)
            else:
                valid = (~pad_mask).to(dtype=atom_out.dtype)
                pooled = atom_out.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class QM9EDMModel(nn.Module):
    def __init__(
        self,
        *,
        atom_channels: int = QM9_NUM_ATOM_TYPES,
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
        mha_geom_bias_mode: str = "standard",
        geometry_adapter: BaseGeometryAdapter | None = None,
        use_geometry_bias: bool = True,
        pair_hidden_dim: int = 128,
        pair_n_rbf: int = 16,
        pair_rbf_max: float = 4.0,
    ) -> None:
        super().__init__()
        geometry_adapter = geometry_adapter or NonPeriodicGeometryAdapter()
        simplicial_geom_mode = simplicial_geom_mode.lower()
        mha_geom_bias_mode = mha_geom_bias_mode.lower()
        if simplicial_geom_mode not in {"none", "factorized", "angle_residual"}:
            raise ValueError(
                "simplicial_geom_mode must be one of {'none', 'factorized', 'angle_residual'}"
            )
        if mha_geom_bias_mode not in {"standard", "factorized_marginal"}:
            raise ValueError(
                "mha_geom_bias_mode must be one of {'standard', 'factorized_marginal'}"
            )
        geometry_bias = None
        simplicial_geometry_bias = None
        if use_geometry_bias:
            if attn_type.lower() == "simplicial":
                if simplicial_geom_mode != "none":
                    simplicial_geometry_bias = SimplicialGeometryBias(
                        n_heads=n_heads,
                        mode=simplicial_geom_mode,
                        use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                        use_noise_gate=True,
                    )
            else:
                if mha_geom_bias_mode == "factorized_marginal":
                    geometry_bias = MhaFactorizedGeometryBias(
                        n_heads=n_heads,
                        use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                        use_noise_gate=True,
                    )
                else:
                    geometry_bias = GeometryBiasBuilder(
                        n_heads=n_heads,
                        use_distance_bias=True,
                        use_edge_bias=True,
                        use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                        use_noise_gate=True,
                    )

        self.atom_channels = int(atom_channels)
        self.geometry_adapter = geometry_adapter
        self.atom_proj = nn.Linear(atom_channels, d_model)
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
            geometry_adapter=geometry_adapter,
            geometry_bias=geometry_bias,
            simplicial_geometry_bias=simplicial_geometry_bias,
        )
        self.atom_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, atom_channels),
        )
        self.pair_head = nn.Sequential(
            nn.LazyLinear(pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, 1),
        )
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)

        self.register_buffer(
            "_pair_rbf_centers",
            torch.linspace(0.0, pair_rbf_max, pair_n_rbf, dtype=torch.float32),
            persistent=False,
        )
        delta = pair_rbf_max / max(pair_n_rbf - 1, 1)
        self.register_buffer(
            "_pair_rbf_gamma",
            torch.tensor(1.0 / max(delta * delta, 1e-6), dtype=torch.float32),
            persistent=False,
        )

    def _distance_rbf(self, pair_dist_norm: torch.Tensor) -> torch.Tensor:
        centers = self._pair_rbf_centers.to(
            device=pair_dist_norm.device,
            dtype=pair_dist_norm.dtype,
        )
        gamma = self._pair_rbf_gamma.to(
            device=pair_dist_norm.device,
            dtype=pair_dist_norm.dtype,
        )
        return torch.exp(-gamma * (pair_dist_norm.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())

    def forward(
        self,
        atom_noisy: torch.Tensor,
        coords_noisy: torch.Tensor,
        pad_mask: torch.Tensor,
        sigma: torch.Tensor,
        lattice: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.ndim == 2 and sigma.shape[-1] == 1:
            sigma = sigma[:, 0]
        if sigma.ndim != 1:
            raise ValueError(f"sigma must have shape (B,) or (B, 1), got {tuple(sigma.shape)}")
        sigma_condition = torch.log(sigma.clamp_min(1e-8)) / 4.0

        trunk_out = self.trunk(
            self.atom_proj(atom_noisy),
            self.conditioning(sigma_condition, sigma_condition),
            pad_mask=pad_mask,
            coords=coords_noisy,
            lattice=lattice,
            sigma=sigma,
        )
        atom_delta = self.atom_head(trunk_out)
        atom_delta = atom_delta.masked_fill(pad_mask[..., None], 0.0)

        geom_features = self.trunk.compute_geometry_features(
            coords=coords_noisy,
            pad_mask=pad_mask,
            lattice=lattice,
        )
        pair_rbf = self._distance_rbf(geom_features.pair_dist_norm)
        num_atoms = trunk_out.shape[1]
        hi = trunk_out[:, :, None, :].expand(-1, -1, num_atoms, -1)
        hj = trunk_out[:, None, :, :].expand(-1, num_atoms, -1, -1)
        pair_input = torch.cat([hi, hj, pair_rbf], dim=-1)
        pair_weights = self.pair_head(pair_input).squeeze(-1)
        pair_weights = pair_weights.masked_fill(~geom_features.pair_mask, 0.0)
        pair_weights = pair_weights - torch.diag_embed(
            torch.diagonal(pair_weights, dim1=-2, dim2=-1)
        )

        coord_delta = (pair_weights[..., None] * geom_features.pair_delta).sum(dim=2)
        valid = (~pad_mask).to(dtype=coord_delta.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        coord_delta = coord_delta / denom.unsqueeze(-1)
        coord_delta = coord_delta.masked_fill(pad_mask[..., None], 0.0)
        coord_delta = coord_delta - _masked_mean(coord_delta, pad_mask)
        coord_delta = coord_delta.masked_fill(pad_mask[..., None], 0.0)
        return atom_delta, coord_delta
