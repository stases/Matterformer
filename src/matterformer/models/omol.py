from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.profiler import record_function

from matterformer.geometry import NonPeriodicGeometryAdapter
from matterformer.geometry.cache import FlatGeometryCache, GeometryCache, flatten_padded_geometry_cache
from matterformer.geometry.triton_nonperiodic_knn import build_triton_nonperiodic_knn_geometry_cache
from matterformer.models.hybrid import HybridConfig, HybridFlatTrunkOutput, HybridTransformerTrunk, HybridTrunkOutput
from matterformer.models.platonic import PLATONIC_GROUPS, PlatonicLinear, PlatonicRoPE
from matterformer.models.platonic.linear import _tetra_fourier_data


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


def _vector_rms(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    value_f = value.float()
    valid = valid_mask.to(device=value.device, dtype=value_f.dtype)
    denom = valid.sum().clamp_min(1.0)
    return ((value_f.square().sum(dim=-1) * valid).sum() / denom).sqrt()


def _masked_scalar_rms(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    value_f = value.float()
    valid = valid_mask.to(device=value.device, dtype=value_f.dtype)
    denom = valid.sum().clamp_min(1.0)
    return ((value_f.square() * valid).sum() / denom).sqrt()


def _masked_scalar_absmax(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return value.float().abs().masked_fill(~valid_mask, 0.0).amax()


def _gather_padded_nodes(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_atoms, channels = values.shape
    num_neighbors = neighbor_idx.shape[-1]
    idx = neighbor_idx.to(dtype=torch.long)[..., None].expand(batch_size, num_atoms, num_neighbors, channels)
    expanded = values[:, None, :, :].expand(batch_size, num_atoms, num_atoms, channels)
    return torch.gather(expanded, dim=2, index=idx)


class _DisableTf32:
    def __enter__(self) -> None:
        self.prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self.prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = self.prev_cudnn_tf32


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
    def __init__(
        self,
        d_model: int,
        *,
        mode: str = "add",
        embedding_dim: int | None = None,
        film_dim: int | None = None,
        share_film_with_mix: bool = False,
    ) -> None:
        super().__init__()
        mode = str(mode).lower().replace("-", "_")
        if mode in {"none", "false", "off"}:
            mode = "off"
        if mode not in {"off", "add", "concat"}:
            raise ValueError("chgspin_mode must be one of {'off', 'add', 'concat'}")
        self.mode = mode
        self.d_model = int(d_model)
        self.embedding_dim = int(embedding_dim or d_model)
        self.film_dim = int(film_dim) if film_dim is not None else None
        self.share_film_with_mix = bool(share_film_with_mix)
        if self.share_film_with_mix and self.film_dim is not None and self.film_dim != self.d_model:
            raise ValueError("share_film_with_mix=True requires film_dim to equal d_model")
        if mode == "off" and self.film_dim is None:
            self.charge_embedding = None
            self.spin_embedding = None
            self.mix = None
            self.film_mix = None
            self.concat_project = None
            return
        self.charge_embedding = ScalarFourierEmbedding(self.embedding_dim)
        self.spin_embedding = ScalarFourierEmbedding(self.embedding_dim, zero_when_input_zero=True)
        needs_mix = mode != "off" or (self.share_film_with_mix and self.film_dim is not None)
        self.mix = nn.Linear(2 * self.embedding_dim, self.d_model) if needs_mix else None
        if self.mix is not None:
            nn.init.normal_(self.mix.weight, std=0.02)
            nn.init.zeros_(self.mix.bias)
        self.film_mix = (
            None
            if self.share_film_with_mix
            else nn.Linear(2 * self.embedding_dim, self.film_dim)
            if self.film_dim is not None
            else None
        )
        if self.film_mix is not None:
            nn.init.normal_(self.film_mix.weight, std=0.02)
            nn.init.zeros_(self.film_mix.bias)
        self.concat_project = nn.Linear(2 * self.d_model, self.d_model) if mode == "concat" else None

    def _mixed_graph_values(
        self,
        *,
        charge: torch.Tensor | None,
        spin: torch.Tensor | None,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
        film: bool = False,
    ) -> torch.Tensor:
        mix = self.mix if film and self.share_film_with_mix else self.film_mix if film else self.mix
        if self.charge_embedding is None or self.spin_embedding is None or mix is None:
            raise RuntimeError("Charge/spin conditioning is not configured")
        if charge is None:
            charge = torch.zeros(int(num_graphs), device=device, dtype=torch.float32)
        if spin is None:
            spin = torch.zeros(int(num_graphs), device=device, dtype=torch.float32)
        charge_emb = self.charge_embedding(charge.to(device=device, dtype=torch.float32))
        spin_emb = self.spin_embedding(spin.to(device=device, dtype=torch.float32))
        return torch.nn.functional.silu(mix(torch.cat([charge_emb, spin_emb], dim=-1))).to(dtype=dtype)

    def film_tokens(
        self,
        *,
        charge: torch.Tensor | None,
        spin: torch.Tensor | None,
        pad_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.film_dim is None:
            return None
        batch_size, num_atoms = pad_mask.shape[:2]
        mixed = self._mixed_graph_values(
            charge=charge,
            spin=spin,
            num_graphs=batch_size,
            device=pad_mask.device,
            dtype=dtype,
            film=True,
        )
        return mixed[:, None, :].expand(batch_size, num_atoms, -1).masked_fill(pad_mask[..., None], 0.0)

    def film_tokens_flat(
        self,
        *,
        charge: torch.Tensor | None,
        spin: torch.Tensor | None,
        batch_index: torch.Tensor,
        num_graphs: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.film_dim is None:
            return None
        mixed = self._mixed_graph_values(
            charge=charge,
            spin=spin,
            num_graphs=num_graphs,
            device=batch_index.device,
            dtype=dtype,
            film=True,
        )
        return mixed[batch_index]

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
        batch_size, num_atoms = token_features.shape[:2]
        mixed = self._mixed_graph_values(
            charge=charge,
            spin=spin,
            num_graphs=batch_size,
            device=token_features.device,
            dtype=token_features.dtype,
        )
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
        mixed = self._mixed_graph_values(
            charge=charge,
            spin=spin,
            num_graphs=num_graphs,
            device=token_features.device,
            dtype=token_features.dtype,
        )
        mixed_nodes = mixed[batch_index]
        if self.mode == "add":
            return token_features + mixed_nodes
        if self.concat_project is None:
            raise RuntimeError("concat_project is not configured")
        return self.concat_project(torch.cat([token_features, mixed_nodes], dim=-1))


class _Sin(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)


def _readout_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    value = str(name).strip().lower()
    if value in {"", "none", "null", "identity", "linear"}:
        return nn.Identity()
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
    raise ValueError("readout_activation must be one of {'gelu', 'silu', 'relu', 'mish', 'sin', 'none'}")


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
        tetra_pair_force_mode: str = "off",
        tetra_pair_k_neighbors: int = 30,
        tetra_pair_feature_dim: int = 128,
        tetra_pair_element_dim: int = 32,
        tetra_pair_gate_init: float = 0.0,
        tetra_pair_geometry_strict: bool = False,
        force_head_mode: str = "auto",
        readout_head_mode: str = "dense",
        tetra_readout_mode: str = "platonic",
        tetra_irrep_scalar_input: str = "rho1",
        readout_activation: str | None = None,
        runtime_mode: str = "padded",
        platonic_input_conditioning: bool = False,
        force_zero_mean: bool = False,
        rope_fp64: bool = True,
        readout_disable_tf32: bool = True,
    ) -> None:
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.d_model = int(d_model)
        self.pair_n_rbf = int(pair_n_rbf)
        self.pair_rbf_max = float(pair_rbf_max)
        self.hybrid_config = HybridConfig.from_input(hybrid_config, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.stream_type = self.hybrid_config.stream_type
        self.platonic_input_conditioning = bool(platonic_input_conditioning)
        if self.platonic_input_conditioning and self.stream_type != "tetra":
            raise ValueError("platonic_input_conditioning=True requires stream_type='tetra'")
        requested_chgspin_film = bool(self.hybrid_config.tetra.get("chgspin_film", False))
        if requested_chgspin_film and self.stream_type != "tetra":
            raise ValueError("tetra.chgspin_film=True requires stream_type='tetra'")
        self.chgspin_film = requested_chgspin_film
        self.runtime_mode = str(runtime_mode).lower().replace("-", "_")
        if self.runtime_mode not in {"padded", "internal_flat_tetra", "internal_flat_hybrid"}:
            raise ValueError("runtime_mode must be one of {'padded', 'internal_flat_tetra', 'internal_flat_hybrid'}")
        self.force_zero_mean = bool(force_zero_mean)
        self.rope_fp64 = bool(rope_fp64)
        self.readout_disable_tf32 = bool(readout_disable_tf32)
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
        tetra_pair_force_mode = str(tetra_pair_force_mode).lower().replace("-", "_")
        if tetra_pair_force_mode in {"none", "false", "0", "disabled"}:
            tetra_pair_force_mode = "off"
        if tetra_pair_force_mode in {"pairwise", "pairwise_residual", "antisymmetric", "antisymmetric_residual", "knn"}:
            tetra_pair_force_mode = "residual"
        if tetra_pair_force_mode not in {"off", "residual"}:
            raise ValueError("tetra_pair_force_mode must be one of {'off', 'residual'}")
        if tetra_pair_force_mode != "off" and self.stream_type != "tetra":
            raise ValueError("tetra_pair_force_mode='residual' requires stream_type='tetra'")
        if tetra_pair_force_mode != "off" and int(tetra_pair_k_neighbors) <= 0:
            raise ValueError("tetra_pair_k_neighbors must be positive when tetra pair force is enabled")
        self.tetra_pair_force_mode = tetra_pair_force_mode
        self.tetra_pair_k_neighbors = int(tetra_pair_k_neighbors)
        self.tetra_pair_feature_dim = int(tetra_pair_feature_dim)
        self.tetra_pair_element_dim = int(tetra_pair_element_dim)
        self.tetra_pair_geometry_strict = bool(tetra_pair_geometry_strict)
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
        tetra_readout_mode = str(tetra_readout_mode).lower().replace("-", "_")
        if tetra_readout_mode in {"default", "platonic_ffn", "platonic_readout"}:
            tetra_readout_mode = "platonic"
        if tetra_readout_mode in {"fourier", "fourier_irrep", "irreps", "irrep_readout"}:
            tetra_readout_mode = "irrep"
        if tetra_readout_mode not in {"platonic", "irrep"}:
            raise ValueError("tetra_readout_mode must be one of {'platonic', 'irrep'}")
        if tetra_readout_mode == "irrep" and self.stream_type != "tetra":
            raise ValueError("tetra_readout_mode='irrep' requires stream_type='tetra'")
        if tetra_readout_mode == "irrep" and self.readout_head_mode != "platonic":
            raise ValueError("tetra_readout_mode='irrep' requires readout_head_mode='platonic'")
        self.tetra_readout_mode = tetra_readout_mode
        tetra_irrep_scalar_input = str(tetra_irrep_scalar_input).lower().replace("-", "_")
        if tetra_irrep_scalar_input in {"default", "trivial", "scalar", "scalar_only", "rho_1"}:
            tetra_irrep_scalar_input = "rho1"
        if tetra_irrep_scalar_input in {"invariant", "norms", "irrep_norms", "rho_norms", "all_invariants"}:
            tetra_irrep_scalar_input = "invariants"
        if tetra_irrep_scalar_input not in {"rho1", "invariants"}:
            raise ValueError("tetra_irrep_scalar_input must be one of {'rho1', 'invariants'}")
        self.tetra_irrep_scalar_input = tetra_irrep_scalar_input

        self.input_feature_dim = (
            int(self.hybrid_config.tetra_dim_per_frame) if self.platonic_input_conditioning else d_model
        )
        self.atom_embedding = nn.Embedding(
            self.max_atomic_number + 1,
            self.input_feature_dim,
            padding_idx=None if self.platonic_input_conditioning else 0,
        )
        nn.init.normal_(self.atom_embedding.weight, std=1.0 / math.sqrt(self.input_feature_dim))
        if not self.platonic_input_conditioning:
            with torch.no_grad():
                self.atom_embedding.weight[0].zero_()
        chgspin_film_dim = int(self.hybrid_config.tetra_dim_per_frame) if self.chgspin_film else None
        self.charge_spin = ChargeSpinConditioning(
            self.input_feature_dim,
            mode=chgspin_mode,
            embedding_dim=chgspin_emb_dim,
            film_dim=chgspin_film_dim,
            share_film_with_mix=self.platonic_input_conditioning,
        )
        self.trunk = HybridTransformerTrunk(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            input_dim=self.input_feature_dim,
            hybrid_config=self.hybrid_config,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            geometry_adapter=NonPeriodicGeometryAdapter(),
            use_adaln_conditioning=False,
            norm_affine_when_no_adaln=True,
            use_final_norm=True,
        )
        for module in self.trunk.modules():
            if isinstance(module, PlatonicRoPE):
                module.use_fp64_trig = self.rope_fp64
        if self.runtime_mode == "internal_flat_tetra" and not self.trunk.supports_flat_tetra:
            raise ValueError(
                "runtime_mode='internal_flat_tetra' requires a pure tetra-global trunk "
                "with input_lift.kind in {'scalar_copy', 'platonic_linear'}"
            )
        if self.runtime_mode == "internal_flat_hybrid" and not self.trunk.supports_flat_hybrid:
            raise ValueError(
                "runtime_mode='internal_flat_hybrid' requires a tetra trunk with input_lift.kind in "
                "{'scalar_copy', 'platonic_linear'} and only tetra-global / group-framewise simplicial sublayers"
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

        if self.tetra_pair_force_mode == "residual":
            if self.tetra_pair_feature_dim <= 0:
                raise ValueError("tetra_pair_feature_dim must be positive when tetra pair force is enabled")
            if self.tetra_pair_element_dim < 0:
                raise ValueError("tetra_pair_element_dim must be non-negative")
            self.tetra_pair_feature_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.tetra_pair_feature_dim),
                nn.SiLU(),
            )
            self.tetra_pair_element_embedding = (
                nn.Embedding(self.max_atomic_number + 1, self.tetra_pair_element_dim, padding_idx=0)
                if self.tetra_pair_element_dim > 0
                else None
            )
            pair_input_dim = 2 * self.tetra_pair_feature_dim + 2 * self.tetra_pair_element_dim + self.pair_n_rbf
            self.tetra_pair_force_head = nn.Sequential(
                nn.Linear(pair_input_dim, pair_hidden_dim),
                nn.SiLU(),
                nn.Linear(pair_hidden_dim, 1),
            )
            nn.init.normal_(self.tetra_pair_force_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.tetra_pair_force_head[-1].bias)
            self.tetra_pair_force_gate = nn.Parameter(torch.tensor(float(tetra_pair_gate_init)))
        else:
            self.tetra_pair_feature_proj = None
            self.tetra_pair_element_embedding = None
            self.tetra_pair_force_head = None
            self.register_parameter("tetra_pair_force_gate", None)

        self.platonic_scalar_readout = None
        self.platonic_vector_readout = None
        self.irrep_scalar_readout = None
        self.irrep_vector_weight: nn.Parameter | None
        if self.stream_type == "tetra":
            group = PLATONIC_GROUPS[str(self.hybrid_config.tetra.get("group", "tetrahedron")).lower()]
            dim_per_frame = int(self.hybrid_config.tetra_dim_per_frame or 0)
            if self.readout_head_mode == "platonic":
                readout_cfg = dict(self.hybrid_config.readout)
                if readout_activation is not None and str(readout_activation).strip() != "":
                    activation_name = readout_activation
                elif "activation" in readout_cfg:
                    activation_name = readout_cfg.get("activation")
                elif "readout_activation" in readout_cfg:
                    activation_name = readout_cfg.get("readout_activation")
                elif "readout_activation" in self.hybrid_config.tetra:
                    activation_name = self.hybrid_config.tetra.get("readout_activation")
                elif "activation" in self.hybrid_config.tetra:
                    activation_name = self.hybrid_config.tetra.get("activation")
                else:
                    activation_name = "gelu"
                readout_ffn = bool(readout_cfg.get("ffn", True))
                linear_backend = str(
                    readout_cfg.get(
                        "linear_backend",
                        self.hybrid_config.tetra.get("linear_backend", "spatial"),
                    )
                )
                if self.tetra_readout_mode == "irrep":
                    if group.name != "tetrahedron" or group.G != 12:
                        raise ValueError("tetra_readout_mode='irrep' currently supports only the tetrahedron group")
                    scalar_readout_dim = dim_per_frame if self.tetra_irrep_scalar_input == "rho1" else 5 * dim_per_frame
                    if readout_ffn:
                        self.irrep_scalar_readout = nn.Sequential(
                            nn.LayerNorm(scalar_readout_dim),
                            nn.Linear(scalar_readout_dim, d_model),
                            _readout_activation(str(activation_name)),
                            nn.Linear(d_model, 1),
                        )
                    else:
                        self.irrep_scalar_readout = nn.Linear(scalar_readout_dim, 1)
                    self.irrep_vector_weight = nn.Parameter(torch.empty(3, dim_per_frame))
                    nn.init.normal_(self.irrep_vector_weight, std=1.0 / math.sqrt(max(3 * dim_per_frame, 1)))
                    fourier_basis, _, _ = _tetra_fourier_data(group)
                    self.register_buffer("_tetra_readout_fourier_basis", fourier_basis, persistent=False)
                    self.platonic_scalar_readout = None
                    self.platonic_vector_readout = None
                else:
                    self.register_parameter("irrep_vector_weight", None)
                    if readout_ffn:
                        self.platonic_scalar_readout = nn.Sequential(
                            PlatonicLinear(d_model, d_model, solid=group.name, linear_backend=linear_backend),
                            _readout_activation(str(activation_name)),
                            PlatonicLinear(d_model, group.G, solid=group.name, linear_backend=linear_backend),
                        )
                        self.platonic_vector_readout = nn.Sequential(
                            PlatonicLinear(d_model, d_model, solid=group.name, linear_backend=linear_backend),
                            _readout_activation(str(activation_name)),
                            PlatonicLinear(d_model, group.G * 3, solid=group.name, linear_backend=linear_backend),
                        )
                    else:
                        self.platonic_scalar_readout = PlatonicLinear(
                            d_model,
                            group.G,
                            solid=group.name,
                            linear_backend=linear_backend,
                        )
                        self.platonic_vector_readout = PlatonicLinear(
                            d_model,
                            group.G * 3,
                            solid=group.name,
                            linear_backend=linear_backend,
                        )
                    self.register_buffer(
                        "_tetra_readout_fourier_basis",
                        torch.empty(0),
                        persistent=False,
                    )
                self.group_force_head = None
                self.register_buffer("_platonic_readout_rotations", group.elements, persistent=False)
                self.register_buffer("_group_rotations", torch.empty(0), persistent=False)
            else:
                self.register_parameter("irrep_vector_weight", None)
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
                self.register_buffer("_tetra_readout_fourier_basis", torch.empty(0), persistent=False)
        else:
            self.group_force_head = None
            self.register_parameter("irrep_vector_weight", None)
            self.register_buffer("_group_rotations", torch.empty(0), persistent=False)
            self.register_buffer("_platonic_readout_rotations", torch.empty(0), persistent=False)
            self.register_buffer("_tetra_readout_fourier_basis", torch.empty(0), persistent=False)

        self.register_buffer("_rbf_centers", torch.linspace(0.0, self.pair_rbf_max, self.pair_n_rbf), persistent=False)
        delta = self.pair_rbf_max / max(self.pair_n_rbf - 1, 1)
        self.register_buffer("_rbf_gamma", torch.tensor(1.0 / max(delta * delta, 1e-8)), persistent=False)

    def _readout_precision_context(self):
        if not self.readout_disable_tf32 or not torch.cuda.is_available():
            return nullcontext()
        return _DisableTf32()

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

    @torch._dynamo.disable
    def _build_tetra_pair_geometry(self, coords: torch.Tensor, pad_mask: torch.Tensor) -> GeometryCache:
        return build_triton_nonperiodic_knn_geometry_cache(
            coords,
            pad_mask=pad_mask,
            k_neighbors=self.tetra_pair_k_neighbors,
            rbf_dim=self.pair_n_rbf,
            cutoff=self.pair_rbf_max,
            seq_len=coords.shape[1],
            strict=self.tetra_pair_geometry_strict,
        )

    def _tetra_pair_projected_features(self, atom_features: torch.Tensor) -> torch.Tensor:
        if self.tetra_pair_feature_proj is None:
            raise RuntimeError("Tetra pair force feature projection is not configured")
        return self.tetra_pair_feature_proj(atom_features)

    def _tetra_pair_element_features(self, atomic_numbers: torch.Tensor) -> torch.Tensor | None:
        if self.tetra_pair_element_embedding is None:
            return None
        z = atomic_numbers.clamp(min=0, max=self.max_atomic_number)
        return self.tetra_pair_element_embedding(z)

    def _tetra_pair_coefficients(
        self,
        pair_features: torch.Tensor,
        pair_rbf: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.tetra_pair_force_head is None:
            raise RuntimeError("Tetra pair force head is not configured")
        coeff = self.tetra_pair_force_head(torch.cat([pair_features, pair_rbf.to(dtype=pair_features.dtype)], dim=-1))
        coeff = coeff.squeeze(-1).masked_fill(~pair_mask, 0.0)
        return coeff

    def _tetra_pair_diagnostics(
        self,
        *,
        direct_forces: torch.Tensor,
        pair_forces: torch.Tensor,
        coeff: torch.Tensor,
        pair_mask: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        atom_hit_cap: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        direct_rms = _vector_rms(direct_forces.detach(), atom_valid_mask)
        pair_rms = _vector_rms(pair_forces.detach(), atom_valid_mask)
        coeff_rms = _masked_scalar_rms(coeff.detach(), pair_mask)
        coeff_abs_max = _masked_scalar_absmax(coeff.detach(), pair_mask)
        valid_atoms = atom_valid_mask.to(dtype=torch.float32, device=direct_forces.device)
        cap_fraction = (atom_hit_cap & atom_valid_mask).to(dtype=torch.float32, device=direct_forces.device).sum()
        cap_fraction = cap_fraction / valid_atoms.sum().clamp_min(1.0)
        gate = self.tetra_pair_force_gate
        if gate is None:
            gate_value = direct_rms.new_zeros(())
        else:
            gate_value = gate.detach().to(device=direct_forces.device, dtype=torch.float32)
        return {
            "pair_force/gate": gate_value,
            "pair_force/direct_force_rms": direct_rms,
            "pair_force/residual_force_rms": pair_rms,
            "pair_force/residual_to_direct_rms": pair_rms / direct_rms.clamp_min(1.0e-12),
            "pair_force/coeff_rms": coeff_rms,
            "pair_force/coeff_abs_max": coeff_abs_max,
            "pair_force/knn_cap_fraction": cap_fraction,
            "pair_force/knn_cap_percent": 100.0 * cap_fraction,
        }

    def _tetra_pair_forces(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        geom: GeometryCache,
        pad_mask: torch.Tensor,
        *,
        direct_forces: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.tetra_pair_force_gate is None:
            raise RuntimeError("Tetra pair force residual is not configured")
        with record_function("omol/force_tetra_pair_residual"):
            node = self._tetra_pair_projected_features(atom_features)
            node_j = _gather_padded_nodes(node, geom.neighbor_idx)
            node_i = node[:, :, None, :].expand_as(node_j)
            pair_parts = [node_i, node_j]
            element = self._tetra_pair_element_features(atomic_numbers)
            if element is not None:
                elem_j = _gather_padded_nodes(element, geom.neighbor_idx)
                elem_i = element[:, :, None, :].expand_as(elem_j)
                pair_parts.extend([elem_i, elem_j])
            pair_features = torch.cat(pair_parts, dim=-1)
            coeff = self._tetra_pair_coefficients(pair_features, geom.rbf, geom.neighbor_mask)
            contribution = coeff.float()[..., None] * geom.rel.float()
            forces = contribution.sum(dim=2)
            scatter_idx = geom.neighbor_idx.to(dtype=torch.long)[..., None].expand(*geom.neighbor_idx.shape, 3)
            forces.scatter_add_(1, scatter_idx.reshape(scatter_idx.shape[0], -1, 3), -contribution.reshape(contribution.shape[0], -1, 3))
            gate = self.tetra_pair_force_gate.to(device=forces.device, dtype=forces.dtype)
            forces = gate * forces
            forces = forces.masked_fill(pad_mask[..., None], 0.0)
            diagnostics = self._tetra_pair_diagnostics(
                direct_forces=direct_forces,
                pair_forces=forces,
                coeff=coeff,
                pair_mask=geom.neighbor_mask,
                atom_valid_mask=~pad_mask,
                atom_hit_cap=geom.neighbor_mask.sum(dim=-1) >= self.tetra_pair_k_neighbors,
            )
            return forces.to(dtype=atom_features.dtype), diagnostics

    def _tetra_pair_forces_flat(
        self,
        atom_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        geom: FlatGeometryCache,
        *,
        direct_forces: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.tetra_pair_force_gate is None:
            raise RuntimeError("Tetra pair force residual is not configured")
        with record_function("omol/force_tetra_pair_residual_flat"):
            node = self._tetra_pair_projected_features(atom_features)
            neighbor_idx = geom.neighbor_idx.to(dtype=torch.long)
            node_j = node[neighbor_idx]
            node_i = node[:, None, :].expand_as(node_j)
            pair_parts = [node_i, node_j]
            element = self._tetra_pair_element_features(atomic_numbers)
            if element is not None:
                elem_j = element[neighbor_idx]
                elem_i = element[:, None, :].expand_as(elem_j)
                pair_parts.extend([elem_i, elem_j])
            pair_features = torch.cat(pair_parts, dim=-1)
            coeff = self._tetra_pair_coefficients(pair_features, geom.rbf, geom.neighbor_mask)
            contribution = coeff.float()[..., None] * geom.rel.float()
            forces = contribution.sum(dim=1)
            forces.index_add_(0, neighbor_idx.reshape(-1), -contribution.reshape(-1, 3))
            gate = self.tetra_pair_force_gate.to(device=forces.device, dtype=forces.dtype)
            forces = gate * forces
            atom_valid_mask = torch.ones(forces.shape[0], device=forces.device, dtype=torch.bool)
            diagnostics = self._tetra_pair_diagnostics(
                direct_forces=direct_forces,
                pair_forces=forces,
                coeff=coeff,
                pair_mask=geom.neighbor_mask,
                atom_valid_mask=atom_valid_mask,
                atom_hit_cap=geom.neighbor_mask.sum(dim=-1) >= self.tetra_pair_k_neighbors,
            )
            return forces.to(dtype=atom_features.dtype), diagnostics

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
            if self.force_zero_mean:
                forces = forces - _masked_mean(forces, pad_mask)
            return forces.masked_fill(pad_mask[..., None], 0.0)

    def _scalar_direct_forces(self, trunk_out: torch.Tensor, coords: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if self.scalar_direct_force_head is None:
            raise RuntimeError("Direct scalar force head is not configured")
        with record_function("omol/force_direct_3d_head"):
            head_input = torch.cat([trunk_out, coords.to(dtype=trunk_out.dtype)], dim=-1)
            forces = self.scalar_direct_force_head(head_input)
            forces = forces.masked_fill(pad_mask[..., None], 0.0)
            if self.force_zero_mean:
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
            if self.force_zero_mean:
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
            if self.force_zero_mean:
                vectors = vectors - _segment_mean_flat(vectors, batch_index, num_graphs, counts=counts)[batch_index]
            return vectors

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
            with self._readout_precision_context():
                scalar_raw = self.platonic_scalar_readout(hidden).view(batch_size, num_atoms, group_order, 1)
                local_vectors = self.platonic_vector_readout(hidden).view(batch_size, num_atoms, group_order, 3)
            per_atom_energy = scalar_raw.mean(dim=2).squeeze(-1).masked_fill(pad_mask, 0.0)

            rotations = self._platonic_readout_rotations.to(device=local_vectors.device, dtype=local_vectors.dtype)
            vectors = torch.einsum("gij,bngj->bni", rotations, local_vectors) / float(group_order)
            vectors = vectors.masked_fill(pad_mask[..., None], 0.0)
            if self.force_zero_mean:
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
            with self._readout_precision_context():
                scalar_raw = self.platonic_scalar_readout(hidden).view(num_atoms, group_order, 1)
                local_vectors = self.platonic_vector_readout(hidden).view(num_atoms, group_order, 3)
            per_atom_energy = scalar_raw.mean(dim=1).squeeze(-1)

            rotations = self._platonic_readout_rotations.to(device=local_vectors.device, dtype=local_vectors.dtype)
            vectors = torch.einsum("gij,ngj->ni", rotations, local_vectors) / float(group_order)
            if self.force_zero_mean:
                vectors = vectors - _segment_mean_flat(vectors, batch_index, num_graphs, counts=counts)[batch_index]
            return per_atom_energy, vectors

    def _irrep_scalar_features(self, coeff: torch.Tensor) -> torch.Tensor:
        rho1 = coeff[..., 0, :]
        if self.tetra_irrep_scalar_input == "rho1":
            return rho1
        rho2 = coeff[..., 1:3, :]
        rho3 = coeff[..., 3:, :].reshape(*coeff.shape[:-2], 3, 3, coeff.shape[-1])
        inv2 = rho2.square().sum(dim=-2)
        inv3 = rho3.square().sum(dim=-3).reshape(*coeff.shape[:-2], 3 * coeff.shape[-1])
        return torch.cat([rho1, inv2, inv3], dim=-1)

    def _irrep_tetra_readout(
        self,
        group_out: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.irrep_scalar_readout is None or self.irrep_vector_weight is None:
            raise RuntimeError("Irrep OMol readout is not configured")
        with record_function("omol/irrep_scalar_vector_readout"):
            batch_size, num_atoms, group_order, channels = group_out.shape
            basis = self._tetra_readout_fourier_basis.to(device=group_out.device, dtype=group_out.dtype)
            coeff = torch.einsum("bngc,ga->bnac", group_out, basis)

            scalar_coeff = self._irrep_scalar_features(coeff)
            per_atom_energy = self.irrep_scalar_readout(scalar_coeff).squeeze(-1).masked_fill(pad_mask, 0.0)

            vector_coeff = coeff[:, :, 3:, :].reshape(batch_size, num_atoms, 3, 3, channels)
            # The tetra rho3 basis is stored as matrix entries [row, col];
            # keep the row/world axis and mix only the col/channel axes.
            vectors = torch.einsum(
                "bnijc,jc->bni",
                vector_coeff,
                self.irrep_vector_weight.to(device=group_out.device, dtype=group_out.dtype),
            )
            vectors = vectors / math.sqrt(3.0 * float(group_order))
            vectors = vectors.masked_fill(pad_mask[..., None], 0.0)
            if self.force_zero_mean:
                vectors = vectors - _masked_mean(vectors, pad_mask)
            return per_atom_energy, vectors.masked_fill(pad_mask[..., None], 0.0)

    def _irrep_tetra_readout_flat(
        self,
        group_out: torch.Tensor,
        *,
        batch_index: torch.Tensor,
        num_graphs: int,
        counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.irrep_scalar_readout is None or self.irrep_vector_weight is None:
            raise RuntimeError("Irrep OMol readout is not configured")
        with record_function("omol/irrep_scalar_vector_readout_flat"):
            num_atoms, group_order, channels = group_out.shape
            basis = self._tetra_readout_fourier_basis.to(device=group_out.device, dtype=group_out.dtype)
            coeff = torch.einsum("ngc,ga->nac", group_out, basis)

            scalar_coeff = self._irrep_scalar_features(coeff)
            per_atom_energy = self.irrep_scalar_readout(scalar_coeff).squeeze(-1)

            vector_coeff = coeff[:, 3:, :].reshape(num_atoms, 3, 3, channels)
            # The tetra rho3 basis is stored as matrix entries [row, col];
            # keep the row/world axis and mix only the col/channel axes.
            vectors = torch.einsum(
                "nijc,jc->ni",
                vector_coeff,
                self.irrep_vector_weight.to(device=group_out.device, dtype=group_out.dtype),
            )
            vectors = vectors / math.sqrt(3.0 * float(group_order))
            if self.force_zero_mean:
                vectors = vectors - _segment_mean_flat(vectors, batch_index, num_graphs, counts=counts)[batch_index]
            return per_atom_energy, vectors

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
            chgspin_film = self.charge_spin.film_tokens_flat(
                charge=charge,
                spin=spin,
                batch_index=batch_index,
                num_graphs=batch_size,
                dtype=token_features.dtype,
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

        fixed_k_context = None
        fixed_k_cfg = self.trunk._fixed_k_local_config()
        if fixed_k_cfg is not None:
            if flat_hybrid:
                raise RuntimeError("fixed-K local Platonic attention is currently supported only in internal_flat_tetra")
            with record_function("omol/fixed_k_local_geometry_cache"):
                padded_fixed_k_geom = build_triton_nonperiodic_knn_geometry_cache(
                    centered_coords_padded,
                    pad_mask=pad_mask,
                    k_neighbors=int(fixed_k_cfg["k_neighbors"]),
                    rbf_dim=int(fixed_k_cfg["num_rbf"]),
                    cutoff=float(fixed_k_cfg["cutoff"]),
                    seq_len=num_slots,
                    strict=bool(fixed_k_cfg["strict"]),
                    include_self=bool(fixed_k_cfg["include_self"]),
                    self_as_first_neighbor=bool(fixed_k_cfg["self_as_first_neighbor"]),
                    mask_by_cutoff=bool(fixed_k_cfg["mask_by_cutoff"]),
                )
                flat_fixed_k_geom = flatten_padded_geometry_cache(
                    padded_fixed_k_geom,
                    valid=valid,
                    batch_index=batch_index,
                    cu_seqlens=cu_seqlens,
                )
                fixed_k_context = self.trunk.prepare_fixed_k_local_context(
                    flat_fixed_k_geom,
                    atom_types=atomic_flat,
                )
                del padded_fixed_k_geom, flat_fixed_k_geom

        with record_function("omol/trunk_flat"):
            if flat_hybrid:
                trunk_result = self.trunk.forward_flat_hybrid(
                    token_features,
                    coords=centered_coords,
                    atom_types=atomic_flat,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    flat_geom=flat_geom,
                    chgspin_film=chgspin_film,
                    return_output=True,
                )
            else:
                trunk_result = self.trunk.forward_flat_tetra(
                    token_features,
                    coords=centered_coords,
                    atom_types=atomic_flat,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    chgspin_film=chgspin_film,
                    fixed_k_context=fixed_k_context,
                    return_output=True,
                )
        if not isinstance(trunk_result, HybridFlatTrunkOutput) or trunk_result.group is None:
            raise RuntimeError("Flat tetra OMol model requires group output from the hybrid trunk")
        diagnostics: dict[str, torch.Tensor] = {}
        if self.readout_head_mode == "platonic":
            if self.tetra_readout_mode == "irrep":
                per_atom_energy, forces_flat = self._irrep_tetra_readout_flat(
                    trunk_result.group,
                    batch_index=batch_index,
                    num_graphs=batch_size,
                    counts=num_atoms,
                )
            else:
                per_atom_energy, forces_flat = self._platonic_tetra_readout_flat(
                    trunk_result.group,
                    batch_index=batch_index,
                    num_graphs=batch_size,
                    counts=num_atoms,
                )
            atom_dtype = trunk_result.group.dtype
            atom_features = trunk_result.scalar
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
            atom_features = atom_out
        if self.tetra_pair_force_mode == "residual":
            with record_function("omol/tetra_pair_geometry_flat"):
                pair_geom = self._build_tetra_pair_geometry(centered_coords_padded, pad_mask)
                flat_pair_geom = flatten_padded_geometry_cache(
                    pair_geom,
                    valid=valid,
                    batch_index=batch_index,
                    cu_seqlens=cu_seqlens,
                )
            pair_forces_flat, diagnostics = self._tetra_pair_forces_flat(
                atom_features,
                atomic_flat,
                flat_pair_geom,
                direct_forces=forces_flat,
            )
            forces_flat = forces_flat + pair_forces_flat.to(dtype=forces_flat.dtype)
        forces = coords.new_zeros((batch_size, num_slots, 3), dtype=forces_flat.dtype)
        forces[valid] = forces_flat
        with record_function("omol/energy_sum_flat"):
            energy = per_atom_energy.new_zeros(batch_size, dtype=torch.float64)
            energy.index_add_(0, batch_index, per_atom_energy.double())
            energy = energy.to(dtype=atom_dtype)
        return {"energy": energy, "forces": forces.to(dtype=atom_dtype), "diagnostics": diagnostics}

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
            chgspin_film = self.charge_spin.film_tokens(
                charge=charge,
                spin=spin,
                pad_mask=pad_mask,
                dtype=token_features.dtype,
            )
            sigma = torch.ones(atomic_numbers.shape[0], device=atomic_numbers.device, dtype=coords.dtype)
            diagnostics: dict[str, torch.Tensor] = {}

        if self.stream_type == "tetra":
            with record_function("omol/trunk_tetra"):
                trunk_result = self.trunk(
                    token_features,
                    None,
                    pad_mask=pad_mask,
                    coords=centered_coords,
                    sigma=sigma,
                    atom_types=atomic_numbers,
                    chgspin_film=chgspin_film,
                    return_output=True,
                )
            if not isinstance(trunk_result, HybridTrunkOutput) or trunk_result.group is None:
                raise RuntimeError("Tetra OMol model requires group output from the hybrid trunk")
            group_atoms = trunk_result.group[:, : centered_coords.shape[1]]
            diagnostics: dict[str, torch.Tensor] = {}
            if self.readout_head_mode == "platonic":
                if self.tetra_readout_mode == "irrep":
                    per_atom_energy, forces = self._irrep_tetra_readout(group_atoms, pad_mask)
                else:
                    per_atom_energy, forces = self._platonic_tetra_readout(group_atoms, pad_mask)
                if self.tetra_pair_force_mode == "residual":
                    atom_features = trunk_result.scalar[:, : centered_coords.shape[1]]
                    with record_function("omol/tetra_pair_geometry"):
                        pair_geom = self._build_tetra_pair_geometry(centered_coords, pad_mask)
                    pair_forces, diagnostics = self._tetra_pair_forces(
                        atom_features,
                        atomic_numbers,
                        pair_geom,
                        pad_mask,
                        direct_forces=forces,
                    )
                    forces = forces + pair_forces.to(dtype=forces.dtype)
                with record_function("omol/energy_sum"):
                    energy = per_atom_energy.double().sum(dim=1).to(dtype=per_atom_energy.dtype)
                return {"energy": energy, "forces": forces.to(dtype=per_atom_energy.dtype), "diagnostics": diagnostics}
            atom_out = trunk_result.scalar
            forces = self._tetra_forces(group_atoms, pad_mask)
            if self.tetra_pair_force_mode == "residual":
                atom_features = atom_out[:, : centered_coords.shape[1]]
                with record_function("omol/tetra_pair_geometry"):
                    pair_geom = self._build_tetra_pair_geometry(centered_coords, pad_mask)
                pair_forces, diagnostics = self._tetra_pair_forces(
                    atom_features,
                    atomic_numbers,
                    pair_geom,
                    pad_mask,
                    direct_forces=forces,
                )
                forces = forces + pair_forces.to(dtype=forces.dtype)
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
        return {"energy": energy, "forces": forces.to(dtype=atom_out.dtype), "diagnostics": diagnostics}

    def collect_sg_diagnostics(self) -> dict[str, float]:
        return self.trunk.collect_sg_diagnostics()
