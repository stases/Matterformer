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


def _canonicalize_noise_conditioning(
    noise_conditioning: object,
    concat_sigma_condition: bool | None,
) -> tuple[str, ...]:
    if noise_conditioning is None:
        modes = ["adaln"]
        if concat_sigma_condition is None or bool(concat_sigma_condition):
            modes.insert(0, "concat")
        return tuple(modes)

    if isinstance(noise_conditioning, str):
        text = noise_conditioning.strip().strip("[]()")
        for separator in ("+", "|", ";"):
            text = text.replace(separator, ",")
        raw_values: list[object] = []
        for chunk in text.split(","):
            raw_values.extend(chunk.split())
    elif isinstance(noise_conditioning, (list, tuple, set)):
        raw_values = list(noise_conditioning)
    else:
        raw_values = [noise_conditioning]

    modes: set[str] = set()
    for value in raw_values:
        token = str(value).strip().strip("'\"").lower().replace("-", "_")
        if not token:
            continue
        if token in {"none", "off", "false", "no", "disabled"}:
            continue
        if token in {"both", "all", "concat_adaln", "adaln_concat"}:
            modes.update({"concat", "adaln"})
            continue
        if token in {"concat", "sigma_concat", "concat_sigma", "c_noise", "cnoise"}:
            modes.add("concat")
            continue
        if token in {"adaln", "ada_ln", "time", "time_embed", "time_embedder", "time_embedding"}:
            modes.add("adaln")
            continue
        raise ValueError(
            "noise_conditioning entries must be drawn from {'concat', 'adaln'} "
            f"or aliases; got {value!r}"
        )
    return tuple(mode for mode in ("concat", "adaln") if mode in modes)


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
        simplicial_angle_rank: int = 16,
        simplicial_message_mode: str = "none",
        simplicial_message_rank: int = 16,
        readout_mode: str = "cls",
        mha_geom_bias_mode: str = "standard",
        mha_position_mode: str = "none",
        mha_rope_freq_sigma: float = 1.0,
        mha_rope_learned_freqs: bool = False,
        mha_rope_use_key: bool = True,
        mha_rope_on_values: bool = False,
        geometry_adapter: BaseGeometryAdapter | None = None,
        use_geometry_bias: bool = True,
    ) -> None:
        super().__init__()
        geometry_adapter = geometry_adapter or NonPeriodicGeometryAdapter()
        simplicial_geom_mode = simplicial_geom_mode.lower()
        simplicial_message_mode = simplicial_message_mode.lower()
        mha_geom_bias_mode = mha_geom_bias_mode.lower()
        readout_mode = readout_mode.lower()
        if simplicial_geom_mode not in {"none", "factorized", "angle_residual", "angle_low_rank"}:
            raise ValueError(
                "simplicial_geom_mode must be one of {'none', 'factorized', 'angle_residual', 'angle_low_rank'}"
            )
        if simplicial_message_mode not in {"none", "low_rank"}:
            raise ValueError("simplicial_message_mode must be one of {'none', 'low_rank'}")
        if mha_geom_bias_mode not in {"standard", "factorized_marginal"}:
            raise ValueError(
                "mha_geom_bias_mode must be one of {'standard', 'factorized_marginal'}"
            )
        if readout_mode not in {"cls", "sum", "mean"}:
            raise ValueError("readout_mode must be one of {'cls', 'sum', 'mean'}")
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
            simplicial_message_mode=effective_message_mode,
            simplicial_message_rank=simplicial_message_rank,
            geometry_adapter=geometry_adapter,
            geometry_bias=geometry_bias,
            simplicial_geometry_bias=simplicial_geometry_bias,
            mha_position_mode=mha_position_mode,
            mha_rope_freq_sigma=mha_rope_freq_sigma,
            mha_rope_learned_freqs=mha_rope_learned_freqs,
            mha_rope_use_key=mha_rope_use_key,
            mha_rope_on_values=mha_rope_on_values,
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
        atom_channels: int = QM9_NUM_ATOM_TYPES + 1,
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
        simplicial_position_mode: str = "none",
        simplicial_rope_key_mode: str = "constant",
        simplicial_rope_n_freqs: int = 16,
        simplicial_rope_freq_sigma: float = 1.0,
        simplicial_rope_learned_freqs: bool = False,
        simplicial_rope_gate: str = "none",
        simplicial_pair_rope_scale_init: float = 1.0,
        simplicial_pair_rope_gate_mode: str = "none",
        simplicial_pair_rope_zero_diag: bool = False,
        simplicial_rope_value_n_freqs: int | None = None,
        simplicial_rope_value_scale_init: float = 1.0,
        simplicial_rope_on_values: str = "none",
        simplicial_content_logits: str = "on",
        mha_geom_bias_mode: str = "standard",
        mha_position_mode: str = "none",
        mha_rope_freq_sigma: float = 1.0,
        mha_rope_learned_freqs: bool = False,
        geometry_adapter: BaseGeometryAdapter | None = None,
        use_geometry_bias: bool = True,
        coord_embed_mode: str = "none",
        coord_n_freqs: int = 32,
        coord_rff_dim: int | None = None,
        coord_rff_sigma: float = 1.0,
        coord_embed_normalize: bool = False,
        coord_head_mode: str = "equivariant",
        noise_conditioning: object = None,
        concat_sigma_condition: bool | None = None,
        mha_rope_use_key: bool = True,
        mha_rope_on_values: bool = False,
        charge_feature_scale: float = 8.0,
        pair_hidden_dim: int = 128,
        pair_n_rbf: int = 16,
        pair_rbf_max: float = 4.0,
        norm_affine_when_no_adaln: bool = False,
        use_final_norm: bool = True,
    ) -> None:
        super().__init__()
        geometry_adapter = geometry_adapter or NonPeriodicGeometryAdapter()
        simplicial_geom_mode = simplicial_geom_mode.lower()
        simplicial_message_mode = simplicial_message_mode.lower()
        simplicial_position_mode = str(simplicial_position_mode).lower().replace("-", "_")
        mha_geom_bias_mode = mha_geom_bias_mode.lower()
        mha_position_mode = mha_position_mode.lower().replace("-", "_")
        coord_embed_mode = coord_embed_mode.lower().replace("-", "_")
        coord_head_mode = coord_head_mode.lower().replace("-", "_")
        if coord_embed_mode in {"disabled", "off", "false", "no"}:
            coord_embed_mode = "none"
        if coord_embed_mode in {"rope", "mha_rope", "rotary"}:
            coord_embed_mode = "none"
            mha_position_mode = "rope"
        if coord_embed_mode in {"coords", "coord", "input", "on", "true", "yes", "learned_rff", "fieldformer_rff"}:
            coord_embed_mode = "learnable_rff"
        if mha_position_mode in {"disabled", "off", "false", "no"}:
            mha_position_mode = "none"
        if mha_position_mode in {"rotary", "mha_rope", "rotary_position_embedding"}:
            mha_position_mode = "rope"
        if simplicial_position_mode in {"disabled", "off", "false", "no"}:
            simplicial_position_mode = "none"
        if simplicial_position_mode in {"center_edge_rope", "closed_simplicial_rope", "cs_rope"}:
            simplicial_position_mode = "closed_rope"
        if simplicial_position_mode in {"pairwise_rope", "pair_rope_bias", "pairwise_rope_bias"}:
            simplicial_position_mode = "pair_rope"
        if coord_head_mode in {"relative", "pair", "pairwise"}:
            coord_head_mode = "equivariant"
        if coord_head_mode in {"non_relative", "non_equivariant", "nonrelative"}:
            coord_head_mode = "direct"
        if simplicial_geom_mode not in {"none", "factorized", "angle_residual", "angle_low_rank"}:
            raise ValueError(
                "simplicial_geom_mode must be one of {'none', 'factorized', 'angle_residual', 'angle_low_rank'}"
            )
        if simplicial_message_mode not in {"none", "low_rank"}:
            raise ValueError("simplicial_message_mode must be one of {'none', 'low_rank'}")
        if simplicial_position_mode not in {"none", "closed_rope", "pair_rope"}:
            raise ValueError("simplicial_position_mode must be one of {'none', 'closed_rope', 'pair_rope'}")
        if mha_geom_bias_mode not in {"standard", "factorized_marginal"}:
            raise ValueError(
                "mha_geom_bias_mode must be one of {'standard', 'factorized_marginal'}"
            )
        if coord_embed_mode not in {"none", "fourier", "rff", "learnable_rff"}:
            raise ValueError(
                "coord_embed_mode must be one of {'none', 'fourier', 'rff', 'learnable_rff'}"
            )
        if mha_position_mode not in {"none", "rope"}:
            raise ValueError("mha_position_mode must be one of {'none', 'rope'}")
        if mha_position_mode == "rope" and attn_type.lower() != "mha":
            raise ValueError("mha_position_mode='rope' requires attn_type='mha'")
        if coord_head_mode not in {"equivariant", "direct"}:
            raise ValueError(
                "coord_head_mode must be one of {'equivariant', 'direct'}"
            )
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
        self.coord_embed_mode = coord_embed_mode
        self.mha_position_mode = mha_position_mode
        self.simplicial_position_mode = simplicial_position_mode
        self.simplicial_content_logits = str(simplicial_content_logits).lower().replace("-", "_")
        self.coord_embed_normalize = bool(coord_embed_normalize)
        self.coord_head_mode = coord_head_mode
        self.noise_conditioning = _canonicalize_noise_conditioning(
            noise_conditioning,
            concat_sigma_condition,
        )
        self.concat_sigma_condition = "concat" in self.noise_conditioning
        self.use_adaln_conditioning = "adaln" in self.noise_conditioning
        self.charge_feature_scale = float(charge_feature_scale)
        atom_input_channels = self.atom_channels + (1 if self.concat_sigma_condition else 0)
        self.atom_proj = nn.Linear(atom_input_channels, d_model)
        self.coord_embedding = (
            FourierCoordEmbedder(
                d_model=d_model,
                n_freqs=coord_n_freqs,
                mode=coord_embed_mode,
                rff_dim=coord_rff_dim,
                rff_sigma=coord_rff_sigma,
            )
            if coord_embed_mode != "none"
            else None
        )
        self.conditioning = TimeEmbedder(d_model) if self.use_adaln_conditioning else None
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
            simplicial_position_mode=simplicial_position_mode,
            simplicial_rope_key_mode=simplicial_rope_key_mode,
            simplicial_rope_n_freqs=simplicial_rope_n_freqs,
            simplicial_rope_freq_sigma=simplicial_rope_freq_sigma,
            simplicial_rope_learned_freqs=simplicial_rope_learned_freqs,
            simplicial_rope_gate=simplicial_rope_gate,
            simplicial_pair_rope_scale_init=simplicial_pair_rope_scale_init,
            simplicial_pair_rope_gate_mode=simplicial_pair_rope_gate_mode,
            simplicial_pair_rope_zero_diag=simplicial_pair_rope_zero_diag,
            simplicial_rope_value_n_freqs=simplicial_rope_value_n_freqs,
            simplicial_rope_value_scale_init=simplicial_rope_value_scale_init,
            simplicial_rope_on_values=simplicial_rope_on_values,
            simplicial_content_logits=simplicial_content_logits,
            geometry_adapter=geometry_adapter,
            geometry_bias=geometry_bias,
            simplicial_geometry_bias=simplicial_geometry_bias,
            mha_position_mode=mha_position_mode,
            mha_rope_freq_sigma=mha_rope_freq_sigma,
            mha_rope_learned_freqs=mha_rope_learned_freqs,
            mha_rope_use_key=mha_rope_use_key,
            mha_rope_on_values=mha_rope_on_values,
            use_adaln_conditioning=self.use_adaln_conditioning,
            norm_affine_when_no_adaln=norm_affine_when_no_adaln,
            use_final_norm=use_final_norm,
        )
        self.atom_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, atom_channels),
        )
        if self.coord_head_mode == "equivariant":
            self.pair_head = nn.Sequential(
                nn.LazyLinear(pair_hidden_dim),
                nn.SiLU(),
                nn.Linear(pair_hidden_dim, 1),
            )
            nn.init.zeros_(self.pair_head[-1].weight)
            nn.init.zeros_(self.pair_head[-1].bias)
        else:
            self.coord_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 3),
            )
            nn.init.zeros_(self.coord_head[-1].weight)
            nn.init.zeros_(self.coord_head[-1].bias)

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

    def _coords_for_embedding(self, coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        coords = _center_coords(coords, pad_mask)
        if self.coord_embed_normalize:
            coord_scale = _masked_rms_radius(coords, pad_mask)
            coords = coords / coord_scale.clamp_min(1e-8)
        return coords.masked_fill(pad_mask[..., None], 0.0)

    def _equivariant_coord_delta(
        self,
        trunk_out: torch.Tensor,
        geom_features,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        return coord_delta.masked_fill(pad_mask[..., None], 0.0)

    def _direct_coord_delta(self, trunk_out: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        coord_delta = self.coord_head(trunk_out)
        coord_delta = coord_delta.masked_fill(pad_mask[..., None], 0.0)
        coord_delta = coord_delta - _masked_mean(coord_delta, pad_mask)
        return coord_delta.masked_fill(pad_mask[..., None], 0.0)

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
        if atom_noisy.shape[-1] != self.atom_channels:
            raise ValueError(
                f"atom_noisy last dim must match atom_channels={self.atom_channels}, "
                f"got {atom_noisy.shape[-1]}"
            )
        sigma_condition = torch.log(sigma.clamp_min(1e-8)) / 4.0

        atom_input = atom_noisy
        if self.concat_sigma_condition:
            sigma_feature = sigma_condition[:, None, None].expand(-1, atom_noisy.shape[1], 1)
            sigma_feature = sigma_feature.masked_fill(pad_mask[..., None], 0.0)
            atom_input = torch.cat([atom_noisy, sigma_feature], dim=-1)
        token_features = self.atom_proj(atom_input)
        if self.coord_embedding is not None:
            coord_features = self.coord_embedding(
                self._coords_for_embedding(coords_noisy, pad_mask)
            ).masked_fill(pad_mask[..., None], 0.0)
            token_features = token_features + coord_features

        cond_emb = (
            self.conditioning(sigma_condition, sigma_condition)
            if self.conditioning is not None
            else None
        )
        trunk_out = self.trunk(
            token_features,
            cond_emb,
            pad_mask=pad_mask,
            coords=coords_noisy,
            lattice=lattice,
            sigma=sigma,
        )
        atom_delta = self.atom_head(trunk_out)
        atom_delta = atom_delta.masked_fill(pad_mask[..., None], 0.0)

        if self.coord_head_mode == "equivariant":
            geom_features = self.trunk.compute_geometry_features(
                coords=coords_noisy,
                pad_mask=pad_mask,
                lattice=lattice,
            )
            coord_delta = self._equivariant_coord_delta(trunk_out, geom_features, pad_mask)
        else:
            coord_delta = self._direct_coord_delta(trunk_out, pad_mask)
        return atom_delta, coord_delta
