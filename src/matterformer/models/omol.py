from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.profiler import record_function

from matterformer.geometry import NonPeriodicGeometryAdapter
from matterformer.models.hybrid import HybridConfig, HybridTransformerTrunk, HybridTrunkOutput
from matterformer.models.platonic import PLATONIC_GROUPS


def _masked_mean(value: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(dtype=value.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (value * valid[..., None]).sum(dim=1, keepdim=True) / denom[..., None]


def _center_coords(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    centered = coords - _masked_mean(coords, pad_mask)
    return centered.masked_fill(pad_mask[..., None], 0.0)


class ScalarFourierEmbedding(nn.Module):
    def __init__(self, embedding_size: int, *, zero_when_input_zero: bool = False, scale: float = 1.0) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.zero_when_input_zero = bool(zero_when_input_zero)
        num_freqs = max(1, math.ceil(self.embedding_size / 2))
        self.weight = nn.Parameter(torch.randn(num_freqs) * float(scale))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.view(-1).to(dtype=self.weight.dtype)
        projected = value[:, None] * self.weight[None, :] * (2.0 * math.pi)
        embedding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        embedding = embedding[:, : self.embedding_size]
        if self.zero_when_input_zero:
            embedding = embedding.masked_fill((value == 0)[:, None], 0.0)
        return embedding


class ChargeSpinConditioning(nn.Module):
    def __init__(self, d_model: int, *, mode: str = "add", embedding_dim: int | None = None) -> None:
        super().__init__()
        mode = str(mode).lower().replace("-", "_")
        if mode in {"none", "false", "off"}:
            mode = "off"
        if mode not in {"off", "add", "concat"}:
            raise ValueError("chgspin_mode must be one of {'off', 'add', 'concat'}")
        self.mode = mode
        self.d_model = int(d_model)
        self.embedding_dim = int(embedding_dim or d_model)
        if mode == "off":
            self.charge_embedding = None
            self.spin_embedding = None
            self.mix = None
            self.concat_project = None
            return
        self.charge_embedding = ScalarFourierEmbedding(self.embedding_dim)
        self.spin_embedding = ScalarFourierEmbedding(self.embedding_dim, zero_when_input_zero=True)
        self.mix = nn.Linear(2 * self.embedding_dim, self.d_model)
        nn.init.normal_(self.mix.weight, std=0.02)
        nn.init.zeros_(self.mix.bias)
        self.concat_project = nn.Linear(2 * self.d_model, self.d_model) if mode == "concat" else None

    def forward(
        self,
        token_features: torch.Tensor,
        *,
        charge: torch.Tensor | None,
        spin: torch.Tensor | None,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "off":
            return token_features
        if self.charge_embedding is None or self.spin_embedding is None or self.mix is None:
            raise RuntimeError("Charge/spin conditioning is not configured")
        batch_size, num_atoms = token_features.shape[:2]
        if charge is None:
            charge = torch.zeros(batch_size, device=token_features.device, dtype=torch.float32)
        if spin is None:
            spin = torch.zeros(batch_size, device=token_features.device, dtype=torch.float32)
        charge_emb = self.charge_embedding(charge.to(device=token_features.device, dtype=torch.float32))
        spin_emb = self.spin_embedding(spin.to(device=token_features.device, dtype=torch.float32))
        mixed = torch.nn.functional.silu(self.mix(torch.cat([charge_emb, spin_emb], dim=-1)))
        mixed = mixed.to(dtype=token_features.dtype)
        mixed_nodes = mixed[:, None, :].expand(batch_size, num_atoms, -1).masked_fill(pad_mask[..., None], 0.0)
        if self.mode == "add":
            return token_features + mixed_nodes
        if self.concat_project is None:
            raise RuntimeError("concat_project is not configured")
        return self.concat_project(torch.cat([token_features, mixed_nodes], dim=-1))


class MatterformerOMolForceField(nn.Module):
    """Matterformer direct energy/force model for OMol-style padded batches."""

    def __init__(
        self,
        *,
        max_atomic_number: int = 118,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        hybrid_config: dict[str, Any] | HybridConfig | None = None,
        chgspin_mode: str = "add",
        chgspin_emb_dim: int | None = None,
        pair_hidden_dim: int = 128,
        pair_n_rbf: int = 16,
        pair_rbf_max: float = 6.0,
        force_head_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.d_model = int(d_model)
        self.pair_n_rbf = int(pair_n_rbf)
        self.pair_rbf_max = float(pair_rbf_max)
        self.hybrid_config = HybridConfig.from_input(hybrid_config, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.stream_type = self.hybrid_config.stream_type
        force_head_mode = str(force_head_mode).lower().replace("-", "_")
        if force_head_mode == "auto":
            force_head_mode = "tetra_vector" if self.stream_type == "tetra" else "pairwise"
        if force_head_mode in {"equivariant", "equivariant_pairwise", "scalar_pairwise"}:
            force_head_mode = "pairwise"
        if force_head_mode in {"direct_3d", "direct_scalar", "mlp3d", "non_equivariant"}:
            force_head_mode = "direct"
        if self.stream_type == "scalar" and force_head_mode not in {"pairwise", "direct"}:
            raise ValueError("Scalar OMol force_head_mode must be one of {'auto', 'pairwise', 'direct'}")
        if self.stream_type == "tetra" and force_head_mode not in {"tetra_vector"}:
            raise ValueError("Tetra OMol force_head_mode currently supports only {'auto', 'tetra_vector'}")
        self.force_head_mode = force_head_mode

        self.atom_embedding = nn.Embedding(self.max_atomic_number + 1, d_model, padding_idx=0)
        nn.init.normal_(self.atom_embedding.weight, std=1.0 / math.sqrt(d_model))
        with torch.no_grad():
            self.atom_embedding.weight[0].zero_()
        self.charge_spin = ChargeSpinConditioning(d_model, mode=chgspin_mode, embedding_dim=chgspin_emb_dim)
        self.trunk = HybridTransformerTrunk(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            input_dim=d_model,
            hybrid_config=self.hybrid_config,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            geometry_adapter=NonPeriodicGeometryAdapter(),
            use_adaln_conditioning=False,
            norm_affine_when_no_adaln=True,
            use_final_norm=True,
        )
        self._materialize_unused_distance_bias_lazy_modules()
        self.energy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        if self.stream_type == "scalar" and self.force_head_mode == "pairwise":
            self.scalar_force_head = nn.Sequential(
                nn.Linear(2 * d_model + self.pair_n_rbf, pair_hidden_dim),
                nn.SiLU(),
                nn.Linear(pair_hidden_dim, 1),
            )
            nn.init.zeros_(self.scalar_force_head[-1].weight)
            nn.init.zeros_(self.scalar_force_head[-1].bias)
        else:
            self.scalar_force_head = None
        if self.stream_type == "scalar" and self.force_head_mode == "direct":
            self.scalar_direct_force_head = nn.Sequential(
                nn.LayerNorm(d_model + 3),
                nn.Linear(d_model + 3, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 3),
            )
            nn.init.normal_(self.scalar_direct_force_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.scalar_direct_force_head[-1].bias)
        else:
            self.scalar_direct_force_head = None

        if self.stream_type == "tetra":
            group = PLATONIC_GROUPS[str(self.hybrid_config.tetra.get("group", "tetrahedron")).lower()]
            dim_per_frame = int(self.hybrid_config.tetra_dim_per_frame or 0)
            self.group_force_head = nn.Sequential(
                nn.LayerNorm(dim_per_frame),
                nn.Linear(dim_per_frame, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 3),
            )
            nn.init.zeros_(self.group_force_head[-1].weight)
            nn.init.zeros_(self.group_force_head[-1].bias)
            self.register_buffer("_group_rotations", group.elements, persistent=False)
        else:
            self.group_force_head = None
            self.register_buffer("_group_rotations", torch.empty(0), persistent=False)

        self.register_buffer("_rbf_centers", torch.linspace(0.0, self.pair_rbf_max, self.pair_n_rbf), persistent=False)
        delta = self.pair_rbf_max / max(self.pair_n_rbf - 1, 1)
        self.register_buffer("_rbf_gamma", torch.tensor(1.0 / max(delta * delta, 1e-8)), persistent=False)

    def _materialize_unused_distance_bias_lazy_modules(self) -> None:
        for block in self.trunk.blocks:
            for layer in getattr(block, "sublayers", []):
                geometry_bias = getattr(layer, "geometry_bias", None)
                if geometry_bias is None or bool(getattr(geometry_bias, "use_edge_bias", False)):
                    continue
                edge_bias_mlp = getattr(geometry_bias, "edge_bias_mlp", None)
                if edge_bias_mlp is None:
                    continue
                in_features = int(getattr(geometry_bias, "edge_bias_n_rbf", 16)) + 3
                with torch.no_grad():
                    _ = edge_bias_mlp(torch.zeros(1, 1, 1, in_features))

    def _distance_rbf(self, pair_dist: torch.Tensor) -> torch.Tensor:
        centers = self._rbf_centers.to(device=pair_dist.device, dtype=pair_dist.dtype)
        gamma = self._rbf_gamma.to(device=pair_dist.device, dtype=pair_dist.dtype)
        return torch.exp(-gamma * (pair_dist.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())

    def _scalar_forces(self, trunk_out: torch.Tensor, coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if self.scalar_force_head is None:
            raise RuntimeError("Pairwise scalar force head is not configured")
        with record_function("omol/force_pairwise_dense_geometry"):
            geom = self.trunk.compute_geometry_features(coords=coords, pad_mask=pad_mask, lattice=None)
            rbf = self._distance_rbf(geom.pair_dist)
        with record_function("omol/force_pairwise_pair_materialization"):
            batch_size, num_atoms = trunk_out.shape[:2]
            hi = trunk_out[:, :, None, :].expand(batch_size, num_atoms, num_atoms, -1)
            hj = trunk_out[:, None, :, :].expand(batch_size, num_atoms, num_atoms, -1)
            pair_input = torch.cat([hi, hj, rbf.to(dtype=trunk_out.dtype)], dim=-1)
        with record_function("omol/force_pairwise_mlp_and_reduce"):
            weights = self.scalar_force_head(pair_input).squeeze(-1)
            weights = weights.masked_fill(~geom.pair_mask, 0.0)
            weights = weights - torch.diag_embed(torch.diagonal(weights, dim1=-2, dim2=-1))
            forces = (weights[..., None].to(dtype=geom.pair_delta.dtype) * geom.pair_delta).sum(dim=2)
            forces = forces.masked_fill(pad_mask[..., None], 0.0)
            forces = forces - _masked_mean(forces, pad_mask)
            return forces.masked_fill(pad_mask[..., None], 0.0)

    def _scalar_direct_forces(self, trunk_out: torch.Tensor, coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if self.scalar_direct_force_head is None:
            raise RuntimeError("Direct scalar force head is not configured")
        with record_function("omol/force_direct_3d_head"):
            head_input = torch.cat([trunk_out, coords.to(dtype=trunk_out.dtype)], dim=-1)
            forces = self.scalar_direct_force_head(head_input)
            forces = forces.masked_fill(pad_mask[..., None], 0.0)
            forces = forces - _masked_mean(forces, pad_mask)
            return forces.masked_fill(pad_mask[..., None], 0.0)

    def _tetra_forces(self, group_out: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if self.group_force_head is None:
            raise RuntimeError("Tetra force head is not configured")
        with record_function("omol/force_tetra_vector_head"):
            local_vectors = self.group_force_head(group_out)
            rotations = self._group_rotations.to(device=local_vectors.device, dtype=local_vectors.dtype)
            vectors = torch.einsum("gij,bngi->bngj", rotations, local_vectors).mean(dim=2)
            vectors = vectors.masked_fill(pad_mask[..., None], 0.0)
            vectors = vectors - _masked_mean(vectors, pad_mask)
            return vectors.masked_fill(pad_mask[..., None], 0.0)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        charge: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with record_function("omol/input_embed_condition"):
            pad_mask = pad_mask.bool()
            centered_coords = _center_coords(coords, pad_mask)
            atomic_numbers = atomic_numbers.clamp(min=0, max=self.max_atomic_number)
            token_features = self.atom_embedding(atomic_numbers)
            token_features = token_features.masked_fill(pad_mask[..., None], 0.0)
            token_features = self.charge_spin(token_features, charge=charge, spin=spin, pad_mask=pad_mask)
            token_features = token_features.masked_fill(pad_mask[..., None], 0.0)
            sigma = torch.ones(atomic_numbers.shape[0], device=atomic_numbers.device, dtype=coords.dtype)

        if self.stream_type == "tetra":
            with record_function("omol/trunk_tetra"):
                trunk_result = self.trunk(
                    token_features,
                    None,
                    pad_mask=pad_mask,
                    coords=centered_coords,
                    sigma=sigma,
                    return_output=True,
                )
            if not isinstance(trunk_result, HybridTrunkOutput) or trunk_result.group is None:
                raise RuntimeError("Tetra OMol model requires group output from the hybrid trunk")
            atom_out = trunk_result.scalar
            forces = self._tetra_forces(trunk_result.group[:, : centered_coords.shape[1]], pad_mask)
        else:
            with record_function("omol/trunk_scalar"):
                atom_out = self.trunk(
                    token_features,
                    None,
                    pad_mask=pad_mask,
                    coords=centered_coords,
                    sigma=sigma,
                )
            atom_features = atom_out[:, : centered_coords.shape[1]]
            if self.force_head_mode == "direct":
                forces = self._scalar_direct_forces(atom_features, centered_coords, pad_mask)
            else:
                forces = self._scalar_forces(atom_features, centered_coords, pad_mask)

        with record_function("omol/energy_head"):
            atom_out = atom_out[:, : centered_coords.shape[1]].masked_fill(pad_mask[..., None], 0.0)
            per_atom_energy = self.energy_head(atom_out).squeeze(-1).masked_fill(pad_mask, 0.0)
            energy = per_atom_energy.double().sum(dim=1).to(dtype=atom_out.dtype)
        return {"energy": energy, "forces": forces.to(dtype=atom_out.dtype)}

    def collect_sg_diagnostics(self) -> dict[str, float]:
        return self.trunk.collect_sg_diagnostics()
