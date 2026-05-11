from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from matterformer.geometry.adapters import BaseGeometryAdapter, GeometryFeatures
from matterformer.models.platonic import PLATONIC_GROUPS, PlatonicBlock, PlatonicLinear
from matterformer.models.platonic.layers import GroupLayerNorm
from matterformer.models.transformer import AdaLNBlock, GeometryBiasBuilder, _canonicalize_mha_position_mode
from matterformer.models.triton_compact_simplicial_attention import (
    TRITON_COMPACT_SIMPLICIAL_AVAILABLE,
    triton_compact_simplicial_attention,
)
from matterformer.models.triton_grouped_compact_simplicial_attention import (
    TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE,
    triton_grouped_compact_simplicial_attention,
)

LayerType = Literal["simplicial", "tetra", "trivial"]


def _canonical_layer_type(layer_type: str) -> LayerType:
    value = str(layer_type).lower()
    aliases = {"s": "simplicial", "local": "simplicial", "t": "tetra", "i": "trivial", "mha": "trivial"}
    value = aliases.get(value, value)
    if value not in {"simplicial", "tetra", "trivial"}:
        raise ValueError(f"Unknown hybrid layer type {layer_type!r}")
    return value  # type: ignore[return-value]


def _normalize_mix_entry(entry: Any) -> tuple[int, int, int]:
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        raise ValueError("Each block_mix entry must be a length-3 sequence [simplicial, tetra, trivial]")
    values = tuple(int(x) for x in entry)
    if any(x < 0 for x in values):
        raise ValueError(f"block_mix counts must be non-negative, got {values}")
    return values  # type: ignore[return-value]


def _is_single_mix(block_mix: Any) -> bool:
    return isinstance(block_mix, (list, tuple)) and len(block_mix) == 3 and all(
        not isinstance(x, (list, tuple)) for x in block_mix
    )


def _interleave_counts(num_simplicial: int, num_tetra: int, num_trivial: int) -> list[LayerType]:
    local = ["simplicial"] * num_simplicial
    global_layers: list[LayerType] = ["tetra"] * num_tetra + ["trivial"] * num_trivial
    if not local:
        return global_layers
    out: list[LayerType] = []
    for local_layer, global_layer in zip_longest(local, global_layers):
        if local_layer is not None:
            out.append(local_layer)  # type: ignore[arg-type]
        if global_layer is not None:
            out.append(global_layer)
    return out


def expand_hybrid_schedule(
    num_blocks: int,
    block_mix: tuple[int, int, int] | list[int] | list[tuple[int, int, int]] | list[list[int]],
    order_policy: str = "local_then_global",
    explicit_orders: list[list[str]] | None = None,
) -> list[list[LayerType]]:
    """Expand macro-block mixes into concrete layer orders."""

    num_blocks = int(num_blocks)
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    order_policy = str(order_policy).lower()
    if order_policy not in {"local_then_global", "interleave", "explicit"}:
        raise ValueError("order_policy must be one of {'local_then_global', 'interleave', 'explicit'}")
    if order_policy == "explicit":
        if not explicit_orders:
            raise ValueError("explicit_orders must be provided when order_policy='explicit'")
        return [
            [_canonical_layer_type(layer_type) for layer_type in explicit_orders[idx % len(explicit_orders)]]
            for idx in range(num_blocks)
        ]

    mixes = [_normalize_mix_entry(block_mix)] if _is_single_mix(block_mix) else [_normalize_mix_entry(x) for x in block_mix]
    schedule: list[list[LayerType]] = []
    for block_idx in range(num_blocks):
        num_simplicial, num_tetra, num_trivial = mixes[block_idx % len(mixes)]
        if order_policy == "interleave":
            schedule.append(_interleave_counts(num_simplicial, num_tetra, num_trivial))
        else:
            schedule.append(
                ["simplicial"] * num_simplicial
                + ["tetra"] * num_tetra
                + ["trivial"] * num_trivial
            )
    return schedule


@dataclass
class HybridConfig:
    num_blocks: int = 4
    block_mix: Any = field(default_factory=lambda: [[1, 0, 1]])
    order_policy: str = "local_then_global"
    explicit_orders: list[list[str]] | None = None
    stream_type: str = "scalar"
    scalar_dim: int | None = None
    tetra_dim_per_frame: int | None = None
    d_model_total: int | None = None
    drop_path: float = 0.0
    simplicial: dict[str, Any] = field(default_factory=dict)
    tetra: dict[str, Any] = field(default_factory=dict)
    trivial: dict[str, Any] = field(default_factory=dict)
    input_lift: dict[str, Any] = field(default_factory=dict)
    readout: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(
        cls,
        value: dict[str, Any] | "HybridConfig" | None,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
    ) -> "HybridConfig":
        if isinstance(value, HybridConfig):
            cfg = value
        else:
            raw = dict(value or {})
            legacy_keys = {"branch_mode", "coupling"} & set(raw)
            if legacy_keys:
                raise ValueError(
                    "Dual-stream hybrid configuration has been removed. "
                    f"Unsupported legacy key(s): {sorted(legacy_keys)}. "
                    "Use stream_type='scalar' for S+I or stream_type='tetra' for S_g+T."
                )
            cfg = cls(**raw)
        cfg.stream_type = str(cfg.stream_type).lower().replace("-", "_")
        if cfg.stream_type in {"scalar_only", "trivial", "sit"}:
            cfg.stream_type = "scalar"
        if cfg.stream_type in {"tetra_only", "sgt"}:
            cfg.stream_type = "tetra"
        if cfg.stream_type not in {"scalar", "tetra"}:
            raise ValueError("HybridConfig.stream_type must be one of {'scalar', 'tetra'}")
        cfg.scalar_dim = int(cfg.scalar_dim or d_model)
        if cfg.scalar_dim != d_model:
            raise ValueError(f"Hybrid scalar_dim ({cfg.scalar_dim}) must match d_model ({d_model}) for wrapper compatibility")
        if cfg.tetra_dim_per_frame is None:
            per_frame = max(2, int(round(d_model / 12)))
            if per_frame % 2 == 1:
                per_frame += 1
            cfg.tetra_dim_per_frame = per_frame
        cfg.num_blocks = int(cfg.num_blocks or n_layers)
        cfg.simplicial = {
            "target": "scalar",
            "k_neighbors": 32,
            "num_heads": n_heads,
            "head_dim": None,
            "bias": {"kind": "spherical_low_rank", "angle_rank": 32, "radial_basis_dim": 32},
            "message": {"enabled": False, "rank": 16},
            "kernel": {"backend": "triton_knn"},
            "projection_mode": "group_linear",
            **dict(cfg.simplicial),
        }
        cfg.tetra = {
            "group": "tetrahedron",
            "group_order": 12,
            "heads_per_frame": 1,
            "rope_sigma": 0.5,
            "learned_freqs": True,
            "use_key": False,
            "rope_on_values": True,
            "ffn_mult": 4,
            **dict(cfg.tetra),
        }
        cfg.trivial = {
            "target": "scalar",
            "attention": {"kind": "mha", "num_heads": n_heads, "position_encoding": "edge_delta_bias"},
            "ffn": {"ffn_mult": 4},
            **dict(cfg.trivial),
        }
        cfg.input_lift = {"kind": "scalar_copy", **dict(cfg.input_lift)}
        cfg.readout = {"kind": "group_mean", **dict(cfg.readout)}
        cfg.d_model_total = cfg.d_model_total or (cfg.scalar_dim if cfg.stream_type == "scalar" else 12 * cfg.tetra_dim_per_frame)
        return cfg


@dataclass
class GeometryCache:
    features: GeometryFeatures
    coords_len: int
    seq_len: int
    neighbor_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    rel: torch.Tensor
    dist: torch.Tensor
    unit: torch.Tensor
    rbf: torch.Tensor
    pair_mask: torch.Tensor


@dataclass
class ModelState:
    pos: torch.Tensor
    mask: torch.Tensor | None
    scalar: torch.Tensor | None
    group: torch.Tensor | None
    geom: GeometryCache | None
    cond_emb: torch.Tensor | None = None
    sigma: torch.Tensor | None = None


@dataclass(frozen=True)
class HybridTrunkOutput:
    scalar: torch.Tensor
    group: torch.Tensor | None
    stream_type: Literal["scalar", "tetra"]


def _gather_neighbor_values(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_atoms, _, channels = values.shape
    _, _, num_neighbors = neighbor_idx.shape
    idx = neighbor_idx[..., None].expand(batch_size, num_atoms, num_neighbors, channels)
    return torch.gather(values, dim=2, index=idx)


def _gather_neighbor_heads(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, num_atoms, channels = values.shape
    _, _, num_neighbors = neighbor_idx.shape
    idx = neighbor_idx[:, None, :, :, None].expand(batch_size, num_heads, num_atoms, num_neighbors, channels)
    source = values[:, :, None, :, :].expand(batch_size, num_heads, num_atoms, num_atoms, channels)
    return torch.gather(source, dim=3, index=idx)


def _rbf(dist: torch.Tensor, rbf_dim: int, cutoff: float | None = None) -> torch.Tensor:
    max_dist = float(cutoff) if cutoff is not None else 1.0
    centers = torch.linspace(0.0, max_dist, int(rbf_dim), device=dist.device, dtype=dist.dtype)
    delta = max_dist / max(int(rbf_dim) - 1, 1)
    gamma = 1.0 / max(delta * delta, 1e-6)
    return torch.exp(-gamma * (dist[..., None] - centers.view(*((1,) * dist.ndim), -1)).square())


def _normalize_channels_by_l(rank: int, value: Any = None) -> dict[int, int]:
    if value is not None:
        channels = {int(k): int(v) for k, v in dict(value).items()}
        channels = {degree: channels.get(degree, 0) for degree in (0, 1, 2)}
        if any(count < 0 for count in channels.values()):
            raise ValueError(f"channels_by_l counts must be non-negative, got {channels}")
    elif int(rank) == 32:
        channels = {0: 5, 1: 4, 2: 3}
    else:
        # Greedy l=2/l=1/l=0 allocation; l=0 can fill any leftover rank.
        remaining = int(rank)
        channels = {0: 0, 1: 0, 2: remaining // 5}
        remaining -= 5 * channels[2]
        channels[1] = remaining // 3
        remaining -= 3 * channels[1]
        channels[0] = remaining
    expanded_rank = channels[0] + 3 * channels[1] + 5 * channels[2]
    if expanded_rank != int(rank):
        raise ValueError(
            f"channels_by_l expands to rank {expanded_rank}, but angle_rank is {rank}: {channels}"
        )
    return channels


def _num_spherical_coefficients(channels_by_l: dict[int, int]) -> int:
    return int(channels_by_l.get(0, 0) + channels_by_l.get(1, 0) + channels_by_l.get(2, 0))


def _spherical_basis_lmax2(unit: torch.Tensor) -> dict[int, torch.Tensor]:
    x, y, z = unit.unbind(dim=-1)
    one = torch.ones_like(x).unsqueeze(-1)
    l2 = torch.stack(
        [
            (3.0 ** 0.5) * x * y,
            (3.0 ** 0.5) * y * z,
            (3.0 ** 0.5) * x * z,
            0.5 * (3.0 ** 0.5) * (x.square() - y.square()),
            0.5 * (3.0 * z.square() - 1.0),
        ],
        dim=-1,
    )
    return {0: one, 1: unit, 2: l2}


def _expand_spherical_coefficients(
    coeff: torch.Tensor,
    *,
    basis: dict[int, torch.Tensor],
    channels_by_l: dict[int, int],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    offset = 0
    for degree in (0, 1, 2):
        num_channels = int(channels_by_l.get(degree, 0))
        if num_channels == 0:
            continue
        coeff_l = coeff[..., offset : offset + num_channels]
        offset += num_channels
        basis_l = basis[degree]
        piece = coeff_l.unsqueeze(-1) * basis_l[:, :, :, None, None, :]
        pieces.append(piece.flatten(-2))
    if not pieces:
        raise ValueError("At least one spherical coefficient channel is required")
    return torch.cat(pieces, dim=-1).permute(0, 3, 1, 2, 4).contiguous()


def build_geometry_cache(
    geom_features: GeometryFeatures,
    *,
    coords_len: int,
    seq_len: int,
    k_neighbors: int,
    pad_mask: torch.Tensor | None = None,
    rbf_dim: int = 32,
    cutoff: float | None = None,
) -> GeometryCache:
    pair_mask = geom_features.pair_mask.clone()
    num_atoms = geom_features.pair_dist.shape[1]
    eye = torch.eye(num_atoms, device=pair_mask.device, dtype=torch.bool).view(1, num_atoms, num_atoms)
    pair_mask = pair_mask & ~eye
    if pad_mask is not None:
        atom_pad = pad_mask[:, :coords_len].bool()
        pair_mask = pair_mask & ~atom_pad[:, :, None] & ~atom_pad[:, None, :]
    k_eff = min(int(k_neighbors), max(num_atoms, 1))
    masked_dist = geom_features.pair_dist.masked_fill(~pair_mask, torch.finfo(geom_features.pair_dist.dtype).max)
    dist, neighbor_idx = torch.topk(masked_dist, k=k_eff, dim=-1, largest=False)
    neighbor_mask = torch.gather(pair_mask, dim=-1, index=neighbor_idx)
    if k_eff < int(k_neighbors):
        pad_k = int(k_neighbors) - k_eff
        neighbor_idx = F.pad(neighbor_idx, (0, pad_k), value=0)
        neighbor_mask = F.pad(neighbor_mask, (0, pad_k), value=False)
        dist = F.pad(dist, (0, pad_k), value=0.0)
    rel_all = -geom_features.pair_delta
    rel = _gather_neighbor_values(rel_all, neighbor_idx)
    dist = torch.where(neighbor_mask, dist, torch.zeros_like(dist))
    unit = rel / dist.clamp_min(1e-8).unsqueeze(-1)
    unit = torch.where(neighbor_mask[..., None], unit, torch.zeros_like(unit))
    pair_knn_mask = neighbor_mask[:, :, :, None] & neighbor_mask[:, :, None, :]
    return GeometryCache(
        features=geom_features,
        coords_len=coords_len,
        seq_len=seq_len,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        rel=rel,
        dist=dist,
        unit=unit,
        rbf=_rbf(dist, rbf_dim=rbf_dim, cutoff=cutoff),
        pair_mask=pair_knn_mask,
    )


def _repeat_geometry_cache_for_group(geom: GeometryCache, group_order: int) -> GeometryCache:
    repeats = int(group_order)
    return GeometryCache(
        features=geom.features,
        coords_len=geom.coords_len,
        seq_len=geom.seq_len,
        neighbor_idx=geom.neighbor_idx.repeat_interleave(repeats, dim=0),
        neighbor_mask=geom.neighbor_mask.repeat_interleave(repeats, dim=0),
        rel=geom.rel.repeat_interleave(repeats, dim=0),
        dist=geom.dist.repeat_interleave(repeats, dim=0),
        unit=geom.unit.repeat_interleave(repeats, dim=0),
        rbf=geom.rbf.repeat_interleave(repeats, dim=0),
        pair_mask=geom.pair_mask.repeat_interleave(repeats, dim=0),
    )


def validate_hybrid_schedule(schedule: list[list[LayerType]], stream_type: str) -> None:
    stream_type = str(stream_type)
    allowed = {"simplicial", "trivial"} if stream_type == "scalar" else {"simplicial", "tetra"}
    for block_idx, block in enumerate(schedule):
        invalid = [layer_type for layer_type in block if layer_type not in allowed]
        if invalid:
            raise ValueError(
                f"Hybrid stream_type={stream_type!r} does not allow layer(s) {invalid} "
                f"in block {block_idx}; allowed layers are {sorted(allowed)}"
            )


@dataclass(frozen=True)
class CompactSimplicialBias:
    u: torch.Tensor | None = None
    v: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    angle_left: torch.Tensor | None = None
    angle_right: torch.Tensor | None = None
    angle_gate: torch.Tensor | None = None
    message_left: torch.Tensor | None = None
    message_right: torch.Tensor | None = None
    message_basis: torch.Tensor | None = None


def compact_simplicial_attention_torch(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: CompactSimplicialBias | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    k1_n = _gather_neighbor_heads(k1, neighbor_idx)
    v1_n = _gather_neighbor_heads(v1, neighbor_idx)
    k2_n = _gather_neighbor_heads(k2, neighbor_idx)
    v2_n = _gather_neighbor_heads(v2, neighbor_idx)
    scores = torch.einsum("bhnd,bhnjd,bhnkd->bhnjk", q, k1_n, k2_n).float()
    if bias is not None:
        if bias.u is not None and bias.v is not None and bias.gate is not None:
            scores = scores + bias.gate[:, :, :, None, None].float() * (
                bias.u[:, :, :, :, None].float() + bias.v[:, :, :, None, :].float()
            )
        if bias.angle_left is not None and bias.angle_right is not None:
            if bias.angle_gate is None:
                angle_gate = 1.0
            else:
                angle_gate = bias.angle_gate[:, :, :, None, None].float()
            angle = torch.einsum("bhnjr,bhnkr->bhnjk", bias.angle_left.float(), bias.angle_right.float())
            angle = angle * (bias.angle_left.shape[-1] ** -0.5)
            scores = scores + angle_gate * angle
    valid = neighbor_mask[:, None, :, :, None] & neighbor_mask[:, None, :, None, :]
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores.flatten(-2), dim=-1).view_as(scores)
    attn = torch.where(valid, attn, torch.zeros_like(attn))
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    tmp = torch.einsum("bhnjk,bhnjd->bhnkd", attn.to(v1_n.dtype), v1_n)
    out = (tmp * v2_n).sum(dim=-2)
    if bias is not None and bias.message_left is not None and bias.message_right is not None and bias.message_basis is not None:
        coeff = torch.einsum(
            "bhnjk,bhnjr,bhnkr->bhnr",
            attn.float(),
            bias.message_left.float(),
            bias.message_right.float(),
        ) * (bias.message_left.shape[-1] ** -0.5)
        out = out + torch.einsum("bhnr,hrd->bhnd", coeff, bias.message_basis.float()).to(dtype=out.dtype)
    return out


def compact_simplicial_attention_triton(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: CompactSimplicialBias | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    precision: str = "bf16_tc",
    debug_torch_backward: bool = False,
    strict: bool = False,
) -> torch.Tensor:
    """Compact-kNN Triton backend entrypoint with Torch fallback."""

    reason: str | None = None
    if not TRITON_COMPACT_SIMPLICIAL_AVAILABLE:
        reason = "triton is not installed"
    elif q.device.type != "cuda":
        reason = "compact Triton requires CUDA tensors"
    elif neighbor_idx.device != q.device or neighbor_mask.device != q.device:
        reason = "neighbor_idx and neighbor_mask must be on the same CUDA device as q"
    elif training and dropout_p > 0.0:
        reason = "compact Triton does not support training dropout"
    elif q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        reason = f"compact Triton supports float16/bfloat16/float32 inputs, got {q.dtype}"
    elif q.shape[-1] > 128:
        reason = f"compact Triton supports head_dim <= 128, got {q.shape[-1]}"
    elif neighbor_idx.shape[-1] > 64:
        reason = f"compact Triton supports k_neighbors <= 64, got {neighbor_idx.shape[-1]}"
    elif bias is not None and bias.angle_left is not None and bias.angle_left.shape[-1] > 64:
        reason = f"compact Triton supports angle rank <= 64, got {bias.angle_left.shape[-1]}"
    elif bias is not None and bias.message_left is not None and bias.message_left.shape[-1] > 64:
        reason = f"compact Triton supports message rank <= 64, got {bias.message_left.shape[-1]}"

    if reason is not None:
        if strict:
            raise RuntimeError(f"Compact Triton simplicial attention is unavailable: {reason}")
        return compact_simplicial_attention_torch(
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=bias,
            dropout_p=dropout_p,
            training=training,
        )

    if bias is None:
        return triton_compact_simplicial_attention(
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            dropout_p=dropout_p,
            training=training,
            precision=precision,
            debug_torch_backward=debug_torch_backward,
            strict=strict,
        )
    return triton_compact_simplicial_attention(
        q,
        k1,
        v1,
        k2,
        v2,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=bias.u,
        v_bias=bias.v,
        gate=bias.gate,
        angle_left=bias.angle_left,
        angle_right=bias.angle_right,
        angle_gate=bias.angle_gate,
        message_left=bias.message_left,
        message_right=bias.message_right,
        message_basis=bias.message_basis,
        dropout_p=dropout_p,
        training=training,
        precision=precision,
        debug_torch_backward=debug_torch_backward,
        strict=strict,
    )


class CompactSimplicialGeometryBias(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        rank: int = 32,
        rbf_dim: int = 32,
        channels_by_l: dict[int, int] | None = None,
        hidden_dim: int = 128,
        use_radial_uv: bool = True,
        use_angle: bool = True,
        message_enabled: bool = False,
        message_rank: int = 16,
        message_channels_by_l: dict[int, int] | None = None,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.rank = int(rank)
        self.rbf_dim = int(rbf_dim)
        self.use_radial_uv = bool(use_radial_uv)
        self.use_angle = bool(use_angle)
        self.message_enabled = bool(message_enabled)
        self.message_rank = int(message_rank)
        self.channels_by_l = _normalize_channels_by_l(self.rank, channels_by_l)
        self.num_coefficients = _num_spherical_coefficients(self.channels_by_l)
        edge_dim = self.rbf_dim
        if self.use_radial_uv:
            self.u_mlp = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.num_heads))
            self.v_mlp = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.num_heads))
            self.gate = nn.Parameter(torch.zeros(1, self.num_heads, 1))
        else:
            self.u_mlp = None
            self.v_mlp = None
            self.register_parameter("gate", None)
        if self.use_angle:
            self.left_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.num_heads * self.num_coefficients),
            )
            self.right_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.num_heads * self.num_coefficients),
            )
            self.angle_gate = nn.Parameter(torch.zeros(1, self.num_heads, 1))
        else:
            self.left_mlp = None
            self.right_mlp = None
            self.register_parameter("angle_gate", None)
        if self.message_enabled:
            self.message_channels_by_l = _normalize_channels_by_l(self.message_rank, message_channels_by_l)
            self.num_message_coefficients = _num_spherical_coefficients(self.message_channels_by_l)
            self.message_left_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.num_heads * self.num_message_coefficients),
            )
            self.message_right_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.num_heads * self.num_message_coefficients),
            )
            self.message_basis = nn.Parameter(torch.empty(self.num_heads, self.message_rank, int(head_dim)))
            nn.init.normal_(self.message_basis, mean=0.0, std=float(head_dim) ** -0.5)
        else:
            self.message_channels_by_l = {}
            self.num_message_coefficients = 0
            self.message_left_mlp = None
            self.message_right_mlp = None
            self.register_parameter("message_basis", None)

    def forward(self, geom: GeometryCache, *, dtype: torch.dtype) -> CompactSimplicialBias:
        edge_features = geom.rbf
        batch_size, num_atoms, num_neighbors, _ = edge_features.shape
        basis = _spherical_basis_lmax2(geom.unit) if self.use_angle or self.message_enabled else None
        u = v = gate = None
        if self.use_radial_uv:
            assert self.u_mlp is not None and self.v_mlp is not None and self.gate is not None
            u = self.u_mlp(edge_features).permute(0, 3, 1, 2).to(dtype=dtype)
            v = self.v_mlp(edge_features).permute(0, 3, 1, 2).to(dtype=dtype)
            u = u.masked_fill(~geom.neighbor_mask[:, None, :, :], 0.0)
            v = v.masked_fill(~geom.neighbor_mask[:, None, :, :], 0.0)
            gate = self.gate.to(dtype=dtype).expand(batch_size, -1, num_atoms)
        left = right = angle_gate = None
        edge_mask = geom.neighbor_mask[:, None, :, :, None]
        if self.use_angle:
            assert self.left_mlp is not None and self.right_mlp is not None and self.angle_gate is not None and basis is not None
            left_coeff = self.left_mlp(edge_features).view(
                batch_size,
                num_atoms,
                num_neighbors,
                self.num_heads,
                self.num_coefficients,
            )
            right_coeff = self.right_mlp(edge_features).view(
                batch_size,
                num_atoms,
                num_neighbors,
                self.num_heads,
                self.num_coefficients,
            )
            left = _expand_spherical_coefficients(left_coeff, basis=basis, channels_by_l=self.channels_by_l).to(dtype=dtype)
            right = _expand_spherical_coefficients(right_coeff, basis=basis, channels_by_l=self.channels_by_l).to(dtype=dtype)
            left = left.masked_fill(~edge_mask, 0.0)
            right = right.masked_fill(~edge_mask, 0.0)
            angle_gate = self.angle_gate.to(dtype=dtype).expand(batch_size, -1, num_atoms)
        message_left = message_right = message_basis = None
        if self.message_enabled:
            assert self.message_left_mlp is not None and self.message_right_mlp is not None and self.message_basis is not None and basis is not None
            message_left_coeff = self.message_left_mlp(edge_features).view(
                batch_size,
                num_atoms,
                num_neighbors,
                self.num_heads,
                self.num_message_coefficients,
            )
            message_right_coeff = self.message_right_mlp(edge_features).view(
                batch_size,
                num_atoms,
                num_neighbors,
                self.num_heads,
                self.num_message_coefficients,
            )
            message_left = _expand_spherical_coefficients(
                message_left_coeff,
                basis=basis,
                channels_by_l=self.message_channels_by_l,
            ).to(dtype=dtype)
            message_right = _expand_spherical_coefficients(
                message_right_coeff,
                basis=basis,
                channels_by_l=self.message_channels_by_l,
            ).to(dtype=dtype)
            message_left = message_left.masked_fill(~edge_mask, 0.0)
            message_right = message_right.masked_fill(~edge_mask, 0.0)
            message_basis = self.message_basis.to(dtype=dtype)
        return CompactSimplicialBias(
            u=u,
            v=v,
            gate=gate,
            angle_left=left,
            angle_right=right,
            angle_gate=angle_gate,
            message_left=message_left,
            message_right=message_right,
            message_basis=message_basis,
        )


class CompactSimplicialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        dropout: float = 0.0,
        backend: str = "triton_knn",
        precision: str = "bf16_tc",
        debug_torch_backward: bool = False,
        strict_triton: bool = False,
        bias_config: dict[str, Any] | None = None,
        message_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if head_dim is None:
            if self.dim % self.num_heads != 0:
                raise ValueError("dim must be divisible by num_heads when head_dim is omitted")
            head_dim = self.dim // self.num_heads
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)
        self.backend = str(backend).lower()
        self.precision = str(precision).lower()
        self.debug_torch_backward = bool(debug_torch_backward)
        self.strict_triton = bool(strict_triton)
        self.in_proj = nn.Linear(self.dim, 5 * self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.dim)
        bias_config = dict(bias_config or {})
        message_config = dict(message_config or {})
        bias_kind = str(bias_config.get("kind", "spherical_low_rank")).lower()
        if bias_kind == "feature_gated_spherical_low_rank":
            raise NotImplementedError(
                "feature_gated_spherical_low_rank is not implemented yet; "
                "the current compact local bias is geometry-only spherical_low_rank"
            )
        if bias_kind not in {"spherical_low_rank"}:
            raise ValueError(f"Unsupported compact simplicial bias kind: {bias_kind!r}")
        self.bias = CompactSimplicialGeometryBias(
            num_heads=self.num_heads,
            rank=int(bias_config.get("angle_rank", 32)),
            rbf_dim=int(bias_config.get("radial_basis_dim", 32)),
            channels_by_l=bias_config.get("channels_by_l"),
            use_radial_uv=bool(bias_config.get("use_radial_uv", bias_config.get("use_radial_bias", True))),
            use_angle=bool(bias_config.get("use_angle", bias_config.get("use_angle_bias", True))),
            message_enabled=bool(message_config.get("enabled", False)),
            message_rank=int(message_config.get("rank", 16)),
            message_channels_by_l=message_config.get("channels_by_l"),
            head_dim=self.head_dim,
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.inner_dim)

    def forward(self, x_atoms: torch.Tensor, geom: GeometryCache) -> torch.Tensor:
        q, k1, v1, k2, v2 = self.in_proj(x_atoms).chunk(5, dim=-1)
        q = self._split(q) * self.scale
        k1 = self._split(k1)
        v1 = self._split(v1)
        k2 = self._split(k2)
        v2 = self._split(v2)
        bias = self.bias(geom, dtype=x_atoms.dtype)
        if self.backend in {"triton", "triton_knn"}:
            out = compact_simplicial_attention_triton(
                q,
                k1,
                v1,
                k2,
                v2,
                neighbor_idx=geom.neighbor_idx,
                neighbor_mask=geom.neighbor_mask,
                bias=bias,
                dropout_p=self.dropout,
                training=self.training,
                precision=self.precision,
                debug_torch_backward=self.debug_torch_backward,
                strict=self.strict_triton,
            )
        else:
            out = compact_simplicial_attention_torch(
                q,
                k1,
                v1,
                k2,
                v2,
                neighbor_idx=geom.neighbor_idx,
                neighbor_mask=geom.neighbor_mask,
                bias=bias,
                dropout_p=self.dropout,
                training=self.training,
            )
        return self.out_proj(self._merge(out))


class SimplicialLocalLayer(nn.Module):
    def __init__(self, d_model: int, config: dict[str, Any], *, dropout: float = 0.0, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.target = str(config.get("target", "scalar")).lower()
        num_heads = int(config.get("num_heads", 8))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = CompactSimplicialAttention(
            d_model,
            num_heads,
            head_dim=config.get("head_dim"),
            dropout=dropout,
            backend=dict(config.get("kernel", {})).get("backend", "triton_knn"),
            precision=dict(config.get("kernel", {})).get("precision", "bf16_tc"),
            debug_torch_backward=bool(dict(config.get("kernel", {})).get("debug_torch_backward", False)),
            strict_triton=bool(dict(config.get("kernel", {})).get("strict", False)),
            bias_config=dict(config.get("bias", {})),
            message_config=dict(config.get("message", {})),
        )
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: ModelState) -> ModelState:
        if state.scalar is None:
            return state
        if state.geom is None:
            raise RuntimeError("SimplicialLocalLayer requires geometry cache")
        coords_len = state.geom.coords_len
        scalar = state.scalar
        atoms = scalar[:, :coords_len, :]
        attn_out = self.attn(self.norm1(atoms), state.geom)
        atoms = atoms + self.dropout(attn_out)
        atoms = atoms + self.dropout(self.mlp(self.norm2(atoms)))
        if scalar.shape[1] == coords_len:
            scalar = atoms
        else:
            scalar = torch.cat([atoms, scalar[:, coords_len:, :]], dim=1)
        if state.mask is not None:
            scalar = scalar.masked_fill(state.mask[..., None], 0.0)
        state.scalar = scalar
        return state


class GroupFramewiseSimplicialAttention(nn.Module):
    def __init__(
        self,
        dim_per_frame: int,
        group_order: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        dropout: float = 0.0,
        backend: str = "triton_knn",
        precision: str = "bf16_tc",
        debug_torch_backward: bool = False,
        strict_triton: bool = False,
        projection_mode: str = "group_linear",
        solid: str = "tetrahedron",
        bias_config: dict[str, Any] | None = None,
        message_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dim_per_frame = int(dim_per_frame)
        self.group_order = int(group_order)
        self.num_heads = int(num_heads)
        if head_dim is None:
            if self.dim_per_frame % self.num_heads != 0:
                raise ValueError("dim_per_frame must be divisible by num_heads when head_dim is omitted")
            head_dim = self.dim_per_frame // self.num_heads
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)
        self.backend = str(backend).lower()
        self.precision = str(precision).lower()
        self.debug_torch_backward = bool(debug_torch_backward)
        self.strict_triton = bool(strict_triton)
        self.projection_mode = str(projection_mode).lower()
        if self.projection_mode not in {"group_linear", "shared_frame"}:
            raise ValueError("projection_mode must be one of {'group_linear', 'shared_frame'}")

        if self.projection_mode == "group_linear":
            d_group_in = self.group_order * self.dim_per_frame
            d_group_out = self.group_order * self.inner_dim
            self.in_proj = PlatonicLinear(d_group_in, 5 * d_group_out, solid=solid)
            self.out_proj = PlatonicLinear(d_group_out, d_group_in, solid=solid)
        else:
            self.in_proj = nn.Linear(self.dim_per_frame, 5 * self.inner_dim)
            self.out_proj = nn.Linear(self.inner_dim, self.dim_per_frame)

        bias_config = dict(bias_config or {})
        message_config = dict(message_config or {})
        bias_kind = str(bias_config.get("kind", "spherical_low_rank")).lower()
        if bias_kind == "feature_gated_spherical_low_rank":
            raise NotImplementedError(
                "feature_gated_spherical_low_rank is not implemented for S_g yet; "
                "the current group-framewise local bias is geometry-only spherical_low_rank"
            )
        if bias_kind not in {"spherical_low_rank"}:
            raise ValueError(f"Unsupported group-framewise simplicial bias kind: {bias_kind!r}")
        self.bias = CompactSimplicialGeometryBias(
            num_heads=self.num_heads,
            rank=int(bias_config.get("angle_rank", 32)),
            rbf_dim=int(bias_config.get("radial_basis_dim", 32)),
            channels_by_l=bias_config.get("channels_by_l"),
            use_radial_uv=bool(bias_config.get("use_radial_uv", bias_config.get("use_radial_bias", True))),
            use_angle=bool(bias_config.get("use_angle", bias_config.get("use_angle_bias", True))),
            message_enabled=bool(message_config.get("enabled", False)),
            message_rank=int(message_config.get("rank", 16)),
            message_channels_by_l=message_config.get("channels_by_l"),
            head_dim=self.head_dim,
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.inner_dim)

    def _project_in(self, x_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_atoms, group_order, channels = x_group.shape
        if group_order != self.group_order or channels != self.dim_per_frame:
            raise ValueError(
                f"Expected group tensor [B, N, {self.group_order}, {self.dim_per_frame}], "
                f"got {tuple(x_group.shape)}"
            )
        if self.projection_mode == "group_linear":
            projected = self.in_proj(x_group.reshape(batch_size, num_atoms, self.group_order * self.dim_per_frame))
            projected = projected.view(batch_size, num_atoms, self.group_order, 5 * self.inner_dim)
        else:
            projected = self.in_proj(x_group)
        return tuple(projected.chunk(5, dim=-1))  # type: ignore[return-value]

    def _project_out(self, out_group: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms = out_group.shape[:2]
        if self.projection_mode == "group_linear":
            out = self.out_proj(out_group.reshape(batch_size, num_atoms, self.group_order * self.inner_dim))
            return out.view(batch_size, num_atoms, self.group_order, self.dim_per_frame)
        return self.out_proj(out_group)

    def forward(self, x_group_atoms: torch.Tensor, geom: GeometryCache) -> torch.Tensor:
        batch_size, num_atoms, group_order, _ = x_group_atoms.shape
        q, k1, v1, k2, v2 = self._project_in(x_group_atoms)

        def split_group(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.reshape(batch_size, num_atoms, group_order, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

        q_g = split_group(q) * self.scale
        k1_g = split_group(k1)
        v1_g = split_group(v1)
        k2_g = split_group(k2)
        v2_g = split_group(v2)

        if self.backend in {"triton_grouped", "triton_sg"}:
            # Geometry is invariant across tetra frames.  Compute it once per real
            # batch item and let the grouped Triton kernel map each program id to
            # (batch, group, head, atom), reusing the same neighbor/bias rows for
            # all group frames.  This avoids the old [B * G, H, N, K, R] materialization.
            bias = self.bias(geom, dtype=x_group_atoms.dtype)
            out_g = triton_grouped_compact_simplicial_attention(
                q_g,
                k1_g,
                v1_g,
                k2_g,
                v2_g,
                neighbor_idx=geom.neighbor_idx,
                neighbor_mask=geom.neighbor_mask,
                u=bias.u,
                v_bias=bias.v,
                gate=bias.gate,
                angle_left=bias.angle_left,
                angle_right=bias.angle_right,
                angle_gate=bias.angle_gate,
                message_left=bias.message_left,
                message_right=bias.message_right,
                message_basis=bias.message_basis,
                dropout_p=self.dropout,
                training=self.training,
                precision=self.precision,
                debug_torch_backward=self.debug_torch_backward,
                strict=self.strict_triton,
            )
        elif self.backend in {"triton", "triton_knn"}:
            q_flat = q_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            k1_flat = k1_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            v1_flat = v1_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            k2_flat = k2_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            v2_flat = v2_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            geom_group = _repeat_geometry_cache_for_group(geom, group_order)
            bias = self.bias(geom_group, dtype=x_group_atoms.dtype)
            out_flat = compact_simplicial_attention_triton(
                q_flat,
                k1_flat,
                v1_flat,
                k2_flat,
                v2_flat,
                neighbor_idx=geom_group.neighbor_idx,
                neighbor_mask=geom_group.neighbor_mask,
                bias=bias,
                dropout_p=self.dropout,
                training=self.training,
                precision=self.precision,
                debug_torch_backward=self.debug_torch_backward,
                strict=self.strict_triton,
            )
            out_g = out_flat.view(batch_size, group_order, self.num_heads, num_atoms, self.head_dim)
        else:
            # Reference path keeps the pre-existing fold-B*G behavior.
            q_flat = q_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            k1_flat = k1_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            v1_flat = v1_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            k2_flat = k2_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            v2_flat = v2_g.reshape(batch_size * group_order, self.num_heads, num_atoms, self.head_dim)
            geom_group = _repeat_geometry_cache_for_group(geom, group_order)
            bias = self.bias(geom_group, dtype=x_group_atoms.dtype)
            out_flat = compact_simplicial_attention_torch(
                q_flat,
                k1_flat,
                v1_flat,
                k2_flat,
                v2_flat,
                neighbor_idx=geom_group.neighbor_idx,
                neighbor_mask=geom_group.neighbor_mask,
                bias=bias,
                dropout_p=self.dropout,
                training=self.training,
            )
            out_g = out_flat.view(batch_size, group_order, self.num_heads, num_atoms, self.head_dim)

        out = out_g.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, num_atoms, group_order, self.inner_dim)
        return self._project_out(out)


class GroupFramewiseSimplicialLayer(nn.Module):
    def __init__(
        self,
        *,
        group_order: int,
        dim_per_frame: int,
        config: dict[str, Any],
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        solid: str = "tetrahedron",
    ) -> None:
        super().__init__()
        self.group_order = int(group_order)
        self.dim_per_frame = int(dim_per_frame)
        self.projection_mode = str(config.get("projection_mode", "group_linear")).lower()
        num_heads = int(config.get("num_heads", 1))
        self.norm1 = GroupLayerNorm(self.group_order, self.dim_per_frame, eps=eps)
        self.norm2 = GroupLayerNorm(self.group_order, self.dim_per_frame, eps=eps)
        self.attn = GroupFramewiseSimplicialAttention(
            self.dim_per_frame,
            self.group_order,
            num_heads,
            head_dim=config.get("head_dim"),
            dropout=dropout,
            backend=dict(config.get("kernel", {})).get("backend", "triton_knn"),
            precision=dict(config.get("kernel", {})).get("precision", "bf16_tc"),
            debug_torch_backward=bool(dict(config.get("kernel", {})).get("debug_torch_backward", False)),
            strict_triton=bool(dict(config.get("kernel", {})).get("strict", False)),
            projection_mode=self.projection_mode,
            solid=solid,
            bias_config=dict(config.get("bias", {})),
            message_config=dict(config.get("message", {})),
        )
        d_group = self.group_order * self.dim_per_frame
        hidden = int(d_group * mlp_ratio)
        self.mlp = nn.Sequential(
            PlatonicLinear(d_group, hidden, solid=solid),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            PlatonicLinear(hidden, d_group, solid=solid),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: ModelState) -> ModelState:
        if state.group is None:
            return state
        if state.geom is None:
            raise RuntimeError("GroupFramewiseSimplicialLayer requires geometry cache")
        coords_len = state.geom.coords_len
        group = state.group
        atoms = group[:, :coords_len, :, :]
        atoms_flat = atoms.reshape(atoms.shape[0], atoms.shape[1], self.group_order * self.dim_per_frame)
        attn_in = self.norm1(atoms_flat).view_as(atoms)
        atoms = atoms + self.dropout(self.attn(attn_in, state.geom))
        atoms_flat = atoms.reshape(atoms.shape[0], atoms.shape[1], self.group_order * self.dim_per_frame)
        atoms = atoms + self.dropout(self.mlp(self.norm2(atoms_flat)).view_as(atoms))
        if group.shape[1] == coords_len:
            group = atoms
        else:
            group = torch.cat([atoms, group[:, coords_len:, :, :]], dim=1)
        if state.mask is not None:
            group = group.masked_fill(state.mask[..., None, None], 0.0)
        state.group = group
        return state


class TetraPlatonicGlobalLayer(nn.Module):
    def __init__(
        self,
        *,
        group_order: int,
        dim_per_frame: int,
        config: dict[str, Any],
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        solid = str(config.get("group", "tetrahedron")).lower()
        if PLATONIC_GROUPS[solid].G != int(group_order):
            raise ValueError(f"Group {solid!r} has order {PLATONIC_GROUPS[solid].G}, expected {group_order}")
        heads_per_frame = int(config.get("heads_per_frame", 1))
        d_group = int(group_order) * int(dim_per_frame)
        n_heads = int(group_order) * heads_per_frame
        self.group_order = int(group_order)
        self.dim_per_frame = int(dim_per_frame)
        self.block = PlatonicBlock(
            d_model=d_group,
            nhead=n_heads,
            dim_feedforward=int(d_group * float(config.get("ffn_mult", 4))),
            solid_name=solid,
            dropout=dropout,
            layer_norm_eps=eps,
            rope_sigma=float(config.get("rope_sigma", 0.5)),
            learned_freqs=bool(config.get("learned_freqs", True)),
            use_key=bool(config.get("use_key", False)),
            rope_on_values=bool(config.get("rope_on_values", True)),
        )

    @staticmethod
    def _positions(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
        if pos.shape[1] == seq_len:
            return pos
        pad = torch.zeros(pos.shape[0], seq_len - pos.shape[1], 3, device=pos.device, dtype=pos.dtype)
        return torch.cat([pos, pad], dim=1)

    def forward(self, state: ModelState) -> ModelState:
        if state.group is None:
            return state
        group_shape = state.group.shape
        x = state.group.reshape(group_shape[0], group_shape[1], self.group_order * self.dim_per_frame)
        x = self.block(x, pos=self._positions(state.pos, group_shape[1]), pad_mask=state.mask)
        state.group = x.view(group_shape)
        if state.mask is not None:
            state.group = state.group.masked_fill(state.mask[..., None, None], 0.0)
        return state


class TrivialGlobalLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        eps: float,
        geometry_adapter: BaseGeometryAdapter | None,
        use_adaln_conditioning: bool,
        position_encoding: str = "edge_delta_bias",
        mha_position_mode: str = "none",
        mha_rope_freq_sigma: float = 1.0,
        mha_rope_learned_freqs: bool = False,
        mha_rope_use_key: bool = True,
        mha_rope_on_values: bool = False,
        pair_hidden_dim: int = 128,
        pair_n_rbf: int = 16,
        pair_rbf_max: float = 2.0,
    ) -> None:
        super().__init__()
        position_encoding = str(position_encoding).lower()
        if position_encoding in {"edge_bias", "edge_delta", "delta_bias", "standard"}:
            position_encoding = "edge_delta_bias"
        if position_encoding in {"distance", "radial", "radial_bias"}:
            position_encoding = "distance_bias"
        if position_encoding not in {"edge_delta_bias", "distance_bias", "none"}:
            raise ValueError(
                "trivial attention position_encoding must be one of "
                "{'edge_delta_bias', 'distance_bias', 'none'}"
            )
        self.mha_position_mode = _canonicalize_mha_position_mode(mha_position_mode)
        self.block = AdaLNBlock(
            d_model=d_model,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            eps=eps,
            attn_type="mha",
            mha_position_mode=self.mha_position_mode,
            mha_rope_freq_sigma=mha_rope_freq_sigma,
            mha_rope_learned_freqs=mha_rope_learned_freqs,
            mha_rope_use_key=mha_rope_use_key,
            mha_rope_on_values=mha_rope_on_values,
            use_adaln_conditioning=use_adaln_conditioning,
            norm_affine_when_no_adaln=not use_adaln_conditioning,
        )
        use_distance_bias = position_encoding in {"edge_delta_bias", "distance_bias"}
        use_edge_bias = position_encoding == "edge_delta_bias"
        self.geometry_bias = (
            GeometryBiasBuilder(
                n_heads=n_heads,
                use_distance_bias=use_distance_bias,
                use_edge_bias=use_edge_bias,
                edge_bias_hidden_dim=pair_hidden_dim,
                edge_bias_n_rbf=pair_n_rbf,
                edge_bias_rbf_max=pair_rbf_max,
                use_periodic_features=geometry_adapter.geometry_kind == "periodic",
                use_noise_gate=True,
            )
            if geometry_adapter is not None and position_encoding != "none"
            else None
        )

    @staticmethod
    def _positions(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
        if pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError(f"pos must have shape (B, N, 3) for MHA RoPE, got {tuple(pos.shape)}")
        if seq_len < pos.shape[1]:
            raise ValueError(f"seq_len={seq_len} is smaller than coords_len={pos.shape[1]}")
        positions = pos.float()
        if pos.shape[1] == seq_len:
            return positions
        pad = torch.zeros(pos.shape[0], seq_len - pos.shape[1], 3, device=pos.device, dtype=positions.dtype)
        return torch.cat([positions, pad], dim=1)

    def forward(self, state: ModelState) -> ModelState:
        if state.scalar is None:
            return state
        attn_bias = None
        if self.geometry_bias is not None and state.geom is not None:
            attn_bias = self.geometry_bias.forward_from_features(
                geom_features=state.geom.features,
                coords_len=state.geom.coords_len,
                pad_mask=state.mask,
                seq_len=state.scalar.shape[1],
                sigma=state.sigma,
                out_dtype=state.scalar.dtype,
            )
        state.scalar = self.block(
            state.scalar,
            state.cond_emb,
            pad_mask=state.mask,
            attn_head_bias=attn_bias,
            mha_positions=self._positions(state.pos, state.scalar.shape[1]) if self.mha_position_mode == "rope" else None,
            sigma=state.sigma,
        )
        if state.mask is not None:
            state.scalar = state.scalar.masked_fill(state.mask[..., None], 0.0)
        return state


class HybridBlock(nn.Module):
    def __init__(self, sublayers: list[nn.Module]) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList(sublayers)

    def forward(self, state: ModelState) -> ModelState:
        for layer in self.sublayers:
            state = layer(state)
        return state


class HybridTransformerTrunk(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        input_dim: int | None = None,
        hybrid_config: dict[str, Any] | HybridConfig | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        eps: float = 1e-6,
        geometry_adapter: BaseGeometryAdapter | None = None,
        use_adaln_conditioning: bool = True,
        norm_affine_when_no_adaln: bool = False,
        use_final_norm: bool = True,
    ) -> None:
        super().__init__()
        self.config = HybridConfig.from_input(hybrid_config, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.geometry_adapter = geometry_adapter
        self.use_final_norm = bool(use_final_norm)
        self.d_model = int(d_model)
        self.input_dim = int(input_dim or d_model)
        self.stream_type: Literal["scalar", "tetra"] = self.config.stream_type  # type: ignore[assignment]
        self.group_order = int(self.config.tetra.get("group_order", 12))
        self.group_dim_per_frame = int(self.config.tetra_dim_per_frame or 0)
        self.tetra_solid = str(self.config.tetra.get("group", "tetrahedron")).lower()
        self.input_lift_kind = str(self.config.input_lift.get("kind", "scalar_copy")).lower().replace("-", "_")
        if self.input_lift_kind in {"platonic", "platonic_scalar", "platonic_copy"}:
            self.input_lift_kind = "platonic_linear"
        self.readout_kind = str(self.config.readout.get("kind", "group_mean")).lower().replace("-", "_")
        if self.readout_kind in {"platonic", "platonic_readout"}:
            self.readout_kind = "platonic_ffn"
        schedule = expand_hybrid_schedule(
            self.config.num_blocks,
            self.config.block_mix,
            self.config.order_policy,
            self.config.explicit_orders,
        )
        validate_hybrid_schedule(schedule, self.stream_type)
        if self.stream_type == "tetra" and self.input_lift_kind not in {"scalar_copy", "platonic_linear"}:
            raise NotImplementedError(
                "Only tetra input_lift kinds 'scalar_copy' and 'platonic_linear' are implemented"
            )
        if self.stream_type == "tetra" and self.readout_kind not in {"group_mean", "platonic_ffn"}:
            raise NotImplementedError("Only tetra readout kinds 'group_mean' and 'platonic_ffn' are implemented")
        if self.stream_type == "tetra" and self.input_lift_kind == "platonic_linear":
            self.group_input_proj = PlatonicLinear(
                self.group_order * self.input_dim,
                self.group_order * self.group_dim_per_frame,
                solid=self.tetra_solid,
                bias=False,
            )
        else:
            self.group_input_proj = (
                nn.Linear(self.input_dim, self.group_dim_per_frame) if self.stream_type == "tetra" else None
            )
        self.group_readout_proj = (
            nn.Linear(self.group_dim_per_frame, d_model)
            if self.stream_type == "tetra" and self.readout_kind == "group_mean"
            else None
        )
        self.k_neighbors = int(self.config.simplicial.get("k_neighbors", 32))
        bias_cfg = dict(self.config.simplicial.get("bias", {}))
        self.rbf_dim = int(bias_cfg.get("radial_basis_dim", 32))
        self.rbf_cutoff = bias_cfg.get("radial_cutoff")

        blocks: list[HybridBlock] = []
        for block_order in schedule:
            sublayers: list[nn.Module] = []
            for layer_type in block_order:
                if layer_type == "simplicial":
                    if self.stream_type == "scalar":
                        sublayers.append(SimplicialLocalLayer(d_model, self.config.simplicial, dropout=dropout, mlp_ratio=mlp_ratio))
                    else:
                        sublayers.append(
                            GroupFramewiseSimplicialLayer(
                                group_order=self.group_order,
                                dim_per_frame=self.group_dim_per_frame,
                                config=self.config.simplicial,
                                dropout=dropout,
                                mlp_ratio=mlp_ratio,
                                eps=eps,
                                solid=self.tetra_solid,
                            )
                        )
                elif layer_type == "tetra":
                    sublayers.append(
                        TetraPlatonicGlobalLayer(
                            group_order=self.group_order,
                            dim_per_frame=self.group_dim_per_frame,
                            config=self.config.tetra,
                            dropout=dropout,
                            eps=eps,
                        )
                    )
                elif layer_type == "trivial":
                    trivial_attn = dict(self.config.trivial.get("attention", {}))
                    sublayers.append(
                        TrivialGlobalLayer(
                            d_model=d_model,
                            n_heads=int(trivial_attn.get("num_heads", n_heads)),
                            mlp_ratio=float(dict(self.config.trivial.get("ffn", {})).get("ffn_mult", mlp_ratio)),
                            dropout=dropout,
                            attn_dropout=attn_dropout,
                            eps=eps,
                            geometry_adapter=geometry_adapter,
                            use_adaln_conditioning=use_adaln_conditioning,
                            position_encoding=str(trivial_attn.get("position_encoding", "edge_delta_bias")),
                            mha_position_mode=str(
                                trivial_attn.get(
                                    "mha_position_mode",
                                    trivial_attn.get("position_mode", "none"),
                                )
                            ),
                            mha_rope_freq_sigma=float(
                                trivial_attn.get(
                                    "mha_rope_freq_sigma",
                                    trivial_attn.get("rope_sigma", 1.0),
                                )
                            ),
                            mha_rope_learned_freqs=bool(
                                trivial_attn.get(
                                    "mha_rope_learned_freqs",
                                    trivial_attn.get("learned_freqs", False),
                                )
                            ),
                            mha_rope_use_key=bool(
                                trivial_attn.get(
                                    "mha_rope_use_key",
                                    trivial_attn.get("use_key", True),
                                )
                            ),
                            mha_rope_on_values=bool(
                                trivial_attn.get(
                                    "mha_rope_on_values",
                                    trivial_attn.get("rope_on_values", False),
                                )
                            ),
                            pair_hidden_dim=int(trivial_attn.get("pair_hidden_dim", 128)),
                            pair_n_rbf=int(trivial_attn.get("pair_n_rbf", 16)),
                            pair_rbf_max=float(trivial_attn.get("pair_rbf_max", 2.0)),
                        )
                    )
            blocks.append(HybridBlock(sublayers))
        self.blocks = nn.ModuleList(blocks)
        norm_affine = bool(norm_affine_when_no_adaln) and not bool(use_adaln_conditioning)
        self.norm_out = nn.LayerNorm(d_model, eps=eps, elementwise_affine=norm_affine) if self.use_final_norm else nn.Identity()

    def _lift_tetra_input(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if self.group_input_proj is None:
            raise RuntimeError("group_input_proj is not configured for tetra stream")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected tetra trunk input dim {self.input_dim}, got {x.shape[-1]}")
        if self.input_lift_kind == "platonic_linear":
            lifted = x.unsqueeze(2).expand(-1, -1, self.group_order, -1).reshape(
                *x.shape[:-1],
                self.group_order * self.input_dim,
            )
            group = self.group_input_proj(lifted).view(*x.shape[:-1], self.group_order, self.group_dim_per_frame)
        else:
            group = self.group_input_proj(x).unsqueeze(2).expand(-1, -1, self.group_order, -1).contiguous()
        if pad_mask is not None:
            group = group.masked_fill(pad_mask[..., None, None], 0.0)
        return group

    def compute_geometry_features(
        self,
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
    ) -> GeometryFeatures:
        if self.geometry_adapter is None:
            raise RuntimeError("geometry_adapter is not configured")
        atom_pad_mask = pad_mask[:, : coords.shape[1]] if pad_mask is not None else None
        return self.geometry_adapter(coords=coords, pad_mask=atom_pad_mask, lattice=lattice)

    def _build_geom_cache(
        self,
        *,
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None,
        lattice: torch.Tensor | None,
        seq_len: int,
    ) -> GeometryCache | None:
        if self.geometry_adapter is None:
            return None
        geom = self.compute_geometry_features(coords=coords, pad_mask=pad_mask, lattice=lattice)
        return build_geometry_cache(
            geom,
            coords_len=coords.shape[1],
            seq_len=seq_len,
            k_neighbors=self.k_neighbors,
            pad_mask=pad_mask,
            rbf_dim=self.rbf_dim,
            cutoff=float(self.rbf_cutoff) if self.rbf_cutoff is not None else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_emb: torch.Tensor | None,
        *,
        pad_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | HybridTrunkOutput:
        if pad_mask is not None and pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        if pad_mask is not None and pad_mask.shape[:2] != x.shape[:2]:
            raise ValueError(f"pad_mask shape {tuple(pad_mask.shape)} does not match x {tuple(x.shape)}")
        if self.geometry_adapter is not None and coords is None:
            raise ValueError("coords must be provided when geometry_adapter is configured")
        geom = self._build_geom_cache(coords=coords, pad_mask=pad_mask, lattice=lattice, seq_len=x.shape[1]) if coords is not None else None
        scalar = x if self.stream_type == "scalar" else None
        group = None
        if self.stream_type == "tetra":
            group = self._lift_tetra_input(x, pad_mask)
        state = ModelState(
            pos=coords if coords is not None else x.new_zeros(x.shape[0], x.shape[1], 3),
            mask=pad_mask,
            scalar=scalar,
            group=group,
            geom=geom,
            cond_emb=cond_emb,
            sigma=sigma,
        )
        for block in self.blocks:
            state = block(state)
        if self.stream_type == "scalar":
            if state.scalar is None:
                raise RuntimeError("Scalar hybrid trunk lost scalar state")
            scalar_out = state.scalar
        else:
            if state.group is None:
                raise RuntimeError("Tetra hybrid trunk lost group state")
            if self.readout_kind == "platonic_ffn":
                scalar_out = state.group.reshape(state.group.shape[0], state.group.shape[1], self.d_model)
                out = scalar_out
                if pad_mask is not None:
                    out = out.masked_fill(pad_mask[..., None], 0.0)
                if return_output:
                    return HybridTrunkOutput(scalar=out, group=state.group, stream_type=self.stream_type)
                return out
            if self.group_readout_proj is None:
                raise RuntimeError("group_readout_proj is not configured for group_mean tetra readout")
            scalar_out = self.group_readout_proj(state.group.mean(dim=2))
        out = self.norm_out(scalar_out)
        if pad_mask is not None:
            out = out.masked_fill(pad_mask[..., None], 0.0)
        if return_output:
            return HybridTrunkOutput(scalar=out, group=state.group, stream_type=self.stream_type)
        return out
