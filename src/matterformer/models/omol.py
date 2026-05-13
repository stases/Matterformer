from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.profiler import record_function

from matterformer.geometry import NonPeriodicGeometryAdapter
from matterformer.geometry.cache import flatten_padded_geometry_cache
from matterformer.models.hybrid import HybridConfig, HybridFlatTrunkOutput, HybridTransformerTrunk, HybridTrunkOutput
from matterformer.models.platonic import PLATONIC_GROUPS, PlatonicLinear


def _masked_mean(value: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(dtype=value.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (value * valid[..., None]).sum(dim=1, keepdim=True) / denom[..., None]


def _center_coords(coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    centered = coords - _masked_mean(coords, pad_mask)
    return centered.masked_fill(pad_mask[..., None], 0.0)


def _segment_mean_flat(
    value: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    *,
    counts: torch.Tensor | None = None,
) -> torch.Tensor:
    out = value.new_zeros((int(num_graphs), *value.shape[1:]))
    out.index_add_(0, batch_index, value)
    if counts is None:
        counts = torch.bincount(batch_index, minlength=int(num_graphs))
    denom_shape = (int(num_graphs),) + (1,) * (value.ndim - 1)
    denom = counts.to(device=value.device, dtype=value.dtype).clamp_min(1).view(denom_shape)
    return out / denom


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

    def forward_flat(
        self,
        token_features: torch.Tensor,
        *,
        charge: torch.Tensor | None,
        spin: torch.Tensor | None,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.mode == "off":
            return token_features
        if self.charge_embedding is None or self.spin_embedding is None or self.mix is None:
            raise RuntimeError("Charge/spin conditioning is not configured")
        if charge is None:
            charge = torch.zeros(int(num_graphs), device=token_features.device, dtype=torch.float32)
        if spin is None:
            spin = torch.zeros(int(num_graphs), device=token_features.device, dtype=torch.float32)
        charge_emb = self.charge_embedding(charge.to(device=token_features.device, dtype=torch.float32))
        spin_emb = self.spin_embedding(spin.to(device=token_features.device, dtype=torch.float32))
        mixed = torch.nn.functional.silu(self.mix(torch.cat([charge_emb, spin_emb], dim=-1)))
        mixed_nodes = mixed.to(dtype=token_features.dtype)[batch_index]
        if self.mode == "add":
            return token_features + mixed_nodes
        if self.concat_project is None:
            raise RuntimeError("concat_project is not configured")
        return self.concat_project(torch.cat([token_features, mixed_nodes], dim=-1))


class _Sin(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)


def _readout_activation(name: str | None) -> nn.Module:
    value = "gelu" if name is None else str(name).lower()
    if value == "gelu":
        return nn.GELU()
    if value == "silu":
        return nn.SiLU()
    if value == "relu":
        return nn.ReLU()
    if value == "mish":
        return nn.Mish()
    if value == "sin":
        return _Sin()
    raise ValueError("readout_activation must be one of {'gelu', 'silu', 'relu', 'mish', 'sin'}")


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
        readout_head_mode: str = "dense",
        readout_activation: str | None = None,
        runtime_mode: str = "padded",
    ) -> None:
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.d_model = int(d_model)
        self.pair_n_rbf = int(pair_n_rbf)
        self.pair_rbf_max = float(pair_rbf_max)
        self.hybrid_config = HybridConfig.from_input(hybrid_config, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.stream_type = self.hybrid_config.stream_type
        self.runtime_mode = str(runtime_mode).lower().replace("-", "_")
        if self.runtime_mode not in {"padded", "internal_flat_tetra", "internal_flat_hybrid"}:
            raise ValueError("runtime_mode must be one of {'padded', 'internal_flat_tetra', 'internal_flat_hybrid'}")
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
        readout_head_mode = str(readout_head_mode).lower().replace("-", "_")
        if readout_head_mode in {"mlp", "legacy"}:
            readout_head_mode = "dense"
        if readout_head_mode in {"platonic", "platonic_ffn", "platonic_readout"}:
            readout_head_mode = "platonic"
        if readout_head_mode not in {"dense", "platonic"}:
            raise ValueError("readout_head_mode must be one of {'dense', 'platonic'}")
        if readout_head_mode == "platonic" and self.stream_type != "tetra":
            raise ValueError("readout_head_mode='platonic' requires stream_type='tetra'")
        self.readout_head_mode = readout_head_mode

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
        if self.runtime_mode == "internal_flat_tetra" and not self.trunk.supports_flat_tetra:
            raise ValueError(
                "runtime_mode='internal_flat_tetra' requires a pure tetra-global trunk "
                "with input_lift.kind='scalar_copy'"
            )
        if self.runtime_mode == "internal_flat_hybrid" and not self.trunk.supports_flat_hybrid:
            raise ValueError(
                "runtime_mode='internal_flat_hybrid' requires a tetra trunk with input_lift.kind='scalar_copy' "
                "and only tetra-global / group-framewise simplicial sublayers"
            )
        self._materialize_unused_distance_bias_lazy_modules()
        self.energy_head = (
            None
            if self.readout_head_mode == "platonic"
            else nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),
            )
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

        self.platonic_scalar_readout = None
        self.platonic_vector_readout = None
        if self.stream_type == "tetra":
            group = PLATONIC_GROUPS[str(self.hybrid_config.tetra.get("group", "tetrahedron")).lower()]
            dim_per_frame = int(self.hybrid_config.tetra_dim_per_frame or 0)
            if self.readout_head_mode == "platonic":
                readout_cfg = dict(self.hybrid_config.readout)
                activation_name = (
                    readout_activation
                    or readout_cfg.get("activation")
                    or readout_cfg.get("readout_activation")
                    or self.hybrid_config.tetra.get("readout_activation")
                    or self.hybrid_config.tetra.get("activation")
                    or "gelu"
                )
                readout_ffn = bool(readout_cfg.get("ffn", True))
                if readout_ffn:
                    self.platonic_scalar_readout = nn.Sequential(
                        PlatonicLinear(d_model, d_model, solid=group.name),
                        _readout_activation(str(activation_name)),
                        PlatonicLinear(d_model, group.G, solid=group.name),
                    )
                    self.platonic_vector_readout = nn.Sequential(
                        PlatonicLinear(d_model, d_model, solid=group.name),
                        _readout_activation(str(activation_name)),
                        PlatonicLinear(d_model, group.G * 3, solid=group.name),
                    )
                else:
                    self.platonic_scalar_readout = PlatonicLinear(d_model, group.G, solid=group.name)
                    self.platonic_vector_readout = PlatonicLinear(d_model, group.G * 3, solid=group.name)
                self.group_force_head = None
                self.register_buffer("_platonic_readout_rotations", group.elements, persistent=False)
                self.register_buffer("_group_rotations", torch.empty(0), persistent=False)
            else:
                self.group_force_head = nn.Sequential(
                    nn.LayerNorm(dim_per_frame),
                    nn.Linear(dim_per_frame, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 3),
                )
                nn.init.zeros_(self.group_force_head[-1].weight)
                nn.init.zeros_(self.group_force_head[-1].bias)
                self.register_buffer("_group_rotations", group.elements, persistent=False)
                self.register_buffer("_platonic_readout_rotations", torch.empty(0), persistent=False)
        else:
            self.group_force_head = None
            self.register_buffer("_group_rotations", torch.empty(0), persistent=False)
            self.register_buffer("_platonic_readout_rotations", torch.empty(0), persistent=False)

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

    def _tetra_forces_flat(
        self,
        group_out: torch.Tensor,
        *,
        batch_index: torch.Tensor,
        num_graphs: int,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        if self.group_force_head is None:
            raise RuntimeError("Tetra force head is not configured")
        with record_function("omol/force_tetra_vector_head_flat"):
            local_vectors = self.group_force_head(group_out)
            rotations = self._group_rotations.to(device=local_vectors.device, dtype=local_vectors.dtype)
            vectors = torch.einsum("gij,ngi->ngj", rotations, local_vectors).mean(dim=1)
            centered = vectors - _segment_mean_flat(vectors, batch_index, num_graphs, counts=counts)[batch_index]
            return centered

    def _platonic_tetra_readout(
        self,
        group_out: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.platonic_scalar_readout is None or self.platonic_vector_readout is None:
            raise RuntimeError("Platonic OMol readout is not configured")
        with record_function("omol/platonic_scalar_vector_readout"):
            batch_size, num_atoms, group_order, channels = group_out.shape
            hidden = group_out.reshape(batch_size, num_atoms, group_order * channels)
            scalar_raw = self.platonic_scalar_readout(hidden).view(batch_size, num_atoms, group_order, 1)
            per_atom_energy = scalar_raw.mean(dim=2).squeeze(-1).masked_fill(pad_mask, 0.0)

            local_vectors = self.platonic_vector_readout(hidden).view(batch_size, num_atoms, group_order, 3)
            rotations = self._platonic_readout_rotations.to(device=local_vectors.device, dtype=local_vectors.dtype)
            vectors = torch.einsum("gij,bngj->bni", rotations, local_vectors) / float(group_order)
            vectors = vectors.masked_fill(pad_mask[..., None], 0.0)
            vectors = vectors - _masked_mean(vectors, pad_mask)
            return per_atom_energy, vectors.masked_fill(pad_mask[..., None], 0.0)

    def _platonic_tetra_readout_flat(
        self,
        group_out: torch.Tensor,
        *,
        batch_index: torch.Tensor,
        num_graphs: int,
        counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.platonic_scalar_readout is None or self.platonic_vector_readout is None:
            raise RuntimeError("Platonic OMol readout is not configured")
        with record_function("omol/platonic_scalar_vector_readout_flat"):
            num_atoms, group_order, channels = group_out.shape
            hidden = group_out.reshape(num_atoms, group_order * channels)
            scalar_raw = self.platonic_scalar_readout(hidden).view(num_atoms, group_order, 1)
            per_atom_energy = scalar_raw.mean(dim=1).squeeze(-1)

            local_vectors = self.platonic_vector_readout(hidden).view(num_atoms, group_order, 3)
            rotations = self._platonic_readout_rotations.to(device=local_vectors.device, dtype=local_vectors.dtype)
            vectors = torch.einsum("gij,ngj->ni", rotations, local_vectors) / float(group_order)
            centered = vectors - _segment_mean_flat(vectors, batch_index, num_graphs, counts=counts)[batch_index]
            return per_atom_energy, centered

    def _forward_flat_tetra_from_padded(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        charge: torch.Tensor | None,
        spin: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        flat_hybrid = self.runtime_mode == "internal_flat_hybrid"
        if self.stream_type != "tetra":
            raise RuntimeError("Internal flat OMol runtime requires a tetra trunk")
        if flat_hybrid:
            if not self.trunk.supports_flat_hybrid:
                raise RuntimeError("internal_flat_hybrid runtime requires a supported flat hybrid trunk")
        elif not self.trunk.supports_flat_tetra:
            raise RuntimeError("internal_flat_tetra runtime requires a supported pure tetra trunk")
        with record_function("omol/input_flatten_condition"):
            valid = ~pad_mask
            batch_size, num_slots = atomic_numbers.shape
            flat_index = valid.nonzero(as_tuple=False)
            if flat_index.numel() == 0:
                return {
                    "energy": coords.new_zeros(batch_size),
                    "forces": coords.new_zeros(batch_size, num_slots, 3),
                }
            batch_index = flat_index[:, 0]
            num_atoms = valid.sum(dim=1)
            counts_i32 = num_atoms.to(device=coords.device, dtype=torch.int32)
            cu_seqlens = torch.zeros(batch_size + 1, device=coords.device, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(counts_i32, dim=0)
            max_seqlen = int(num_atoms.max().item()) if num_atoms.numel() > 0 else 0
            atomic_flat = atomic_numbers[valid].clamp(min=0, max=self.max_atomic_number)
            coords_flat = coords[valid]
            center = _segment_mean_flat(coords_flat, batch_index, batch_size, counts=num_atoms)
            centered_coords = coords_flat - center[batch_index]
            centered_coords_padded = (coords - center[:, None, :]).masked_fill(pad_mask[..., None], 0.0)
            token_features = self.atom_embedding(atomic_flat)
            token_features = self.charge_spin.forward_flat(
                token_features,
                charge=charge,
                spin=spin,
                batch_index=batch_index,
                num_graphs=batch_size,
            )

        flat_geom = None
        if flat_hybrid and self.trunk.needs_compact_geometry:
            with record_function("omol/flat_hybrid_geometry_cache"):
                padded_geom = self.trunk._build_geom_cache(
                    coords=centered_coords_padded,
                    pad_mask=pad_mask,
                    lattice=None,
                    seq_len=num_slots,
                )
                if padded_geom is None:
                    raise RuntimeError("internal_flat_hybrid requires a compact geometry cache")
                flat_geom = flatten_padded_geometry_cache(
                    padded_geom,
                    valid=valid,
                    batch_index=batch_index,
                    cu_seqlens=cu_seqlens,
                )
                del padded_geom

        with record_function("omol/trunk_flat"):
            if flat_hybrid:
                trunk_result = self.trunk.forward_flat_hybrid(
                    token_features,
                    coords=centered_coords,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    flat_geom=flat_geom,
                    return_output=True,
                )
            else:
                trunk_result = self.trunk.forward_flat_tetra(
                    token_features,
                    coords=centered_coords,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    return_output=True,
                )
        if not isinstance(trunk_result, HybridFlatTrunkOutput) or trunk_result.group is None:
            raise RuntimeError("Flat tetra OMol model requires group output from the hybrid trunk")
        if self.readout_head_mode == "platonic":
            per_atom_energy, forces_flat = self._platonic_tetra_readout_flat(
                trunk_result.group,
                batch_index=batch_index,
                num_graphs=batch_size,
                counts=num_atoms,
            )
            atom_dtype = trunk_result.group.dtype
        else:
            atom_out = trunk_result.scalar
            forces_flat = self._tetra_forces_flat(
                trunk_result.group,
                batch_index=batch_index,
                num_graphs=batch_size,
                counts=num_atoms,
            )
            if self.energy_head is None:
                raise RuntimeError("Dense OMol energy head is not configured")
            with record_function("omol/energy_head_flat"):
                per_atom_energy = self.energy_head(atom_out).squeeze(-1)
            atom_dtype = atom_out.dtype
        forces = coords.new_zeros((batch_size, num_slots, 3), dtype=forces_flat.dtype)
        forces[valid] = forces_flat
        with record_function("omol/energy_sum_flat"):
            energy = per_atom_energy.new_zeros(batch_size, dtype=torch.float64)
            energy.index_add_(0, batch_index, per_atom_energy.double())
            energy = energy.to(dtype=atom_dtype)
        return {"energy": energy, "forces": forces.to(dtype=atom_dtype)}

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
            if self.runtime_mode in {"internal_flat_tetra", "internal_flat_hybrid"}:
                return self._forward_flat_tetra_from_padded(
                    atomic_numbers,
                    coords,
                    pad_mask,
                    charge=charge,
                    spin=spin,
                )
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
            group_atoms = trunk_result.group[:, : centered_coords.shape[1]]
            if self.readout_head_mode == "platonic":
                per_atom_energy, forces = self._platonic_tetra_readout(group_atoms, pad_mask)
                with record_function("omol/energy_sum"):
                    energy = per_atom_energy.double().sum(dim=1).to(dtype=per_atom_energy.dtype)
                return {"energy": energy, "forces": forces.to(dtype=per_atom_energy.dtype)}
            atom_out = trunk_result.scalar
            forces = self._tetra_forces(group_atoms, pad_mask)
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
            if self.energy_head is None:
                raise RuntimeError("Dense OMol energy head is not configured")
            atom_out = atom_out[:, : centered_coords.shape[1]].masked_fill(pad_mask[..., None], 0.0)
            per_atom_energy = self.energy_head(atom_out).squeeze(-1).masked_fill(pad_mask, 0.0)
            energy = per_atom_energy.double().sum(dim=1).to(dtype=atom_out.dtype)
        return {"energy": energy, "forces": forces.to(dtype=atom_out.dtype)}

    def collect_sg_diagnostics(self) -> dict[str, float]:
        return self.trunk.collect_sg_diagnostics()
