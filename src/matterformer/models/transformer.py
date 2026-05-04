from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from matterformer.geometry.adapters import BaseGeometryAdapter, GeometryFeatures
from matterformer.models.regular_attention import (
    RegularAttention,
    RotaryMultiheadAttention,
    RotaryPositionEmbedding3D,
    _canonicalize_mha_position_mode,
)
from matterformer.models.simplicial_attention import (
    SimplicialAttention,
    SimplicialAttentionMask,
    SimplicialFactorizedBias,
    SimplicialLowRankAngleResidual,
    SimplicialLowRankMessageResidual,
    TwoSimplicialAttention,
)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-8)
    return value + torch.log(-torch.expm1(-value))


def _build_simplicial_attention_mask(
    *,
    coords_len: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    pad_mask: torch.Tensor | None = None,
    geom_pair_mask: torch.Tensor | None = None,
    include_global_tokens_as_pair_keys: bool = False,
) -> SimplicialAttentionMask:
    if seq_len < coords_len:
        raise ValueError(f"seq_len={seq_len} is smaller than coords_len={coords_len}")
    if pad_mask is None:
        query_valid = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    else:
        query_valid = ~pad_mask.bool().to(device=device)
    if include_global_tokens_as_pair_keys:
        pair_key_valid = query_valid
        return SimplicialAttentionMask(
            query_valid=query_valid,
            pair_key_valid=pair_key_valid,
            pair_valid=None,
        )

    atom_positions = (torch.arange(seq_len, device=device) < coords_len).view(1, seq_len)
    pair_key_valid = query_valid & atom_positions
    pair_valid = None
    if geom_pair_mask is not None or seq_len != coords_len:
        pair_valid = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
        if geom_pair_mask is None:
            pair_valid[:, :coords_len, :coords_len] = True
        else:
            pair_valid[:, :coords_len, :coords_len] = geom_pair_mask.bool().to(device=device)
    return SimplicialAttentionMask(
        query_valid=query_valid,
        pair_key_valid=pair_key_valid,
        pair_valid=pair_valid,
    )


def _sinusoidal_embedding(
    values: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"Expected values to have shape (B,) or (B, 1), got {tuple(values.shape)}")
    half = dim // 2
    if half == 0:
        return values[:, None]
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=values.device, dtype=values.dtype) / half
    )
    args = values[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ScalarConditionEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        fourier_dim: int | None = None,
    ) -> None:
        super().__init__()
        fourier_dim = int(fourier_dim or d_model)
        hidden_dim = int(hidden_dim or d_model * 4)
        self.d_model = int(d_model)
        self.fourier_dim = fourier_dim
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        return self.net(_sinusoidal_embedding(values, dim=self.fourier_dim))


class LearnedNullConditioning(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.expand(batch_size, -1)


class Mlp(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        eps: float = 1e-6,
        attn_type: str = "mha",
        simplicial_chunk_size: int = 128,
        simplicial_head_dim: int | None = None,
        simplicial_impl: str = "auto",
        simplicial_precision: str = "bf16_tc",
        simplicial_message_mode: str = "none",
        simplicial_message_rank: int = 16,
        simplicial_content_logits: str = "on",
        mha_position_mode: str = "none",
        mha_rope_freq_sigma: float = 1.0,
        mha_rope_learned_freqs: bool = False,
        mha_rope_use_key: bool = True,
        mha_rope_on_values: bool = False,
        use_adaln_conditioning: bool = True,
        norm_affine_when_no_adaln: bool = False,
    ) -> None:
        super().__init__()
        attn_type = attn_type.lower()
        if attn_type not in {"mha", "simplicial"}:
            raise ValueError(f"Unsupported attn_type: {attn_type}")
        self.attn_type = attn_type
        self.mha_position_mode = _canonicalize_mha_position_mode(mha_position_mode)
        if self.mha_position_mode != "none" and self.attn_type != "mha":
            raise ValueError("mha_position_mode='rope' is only supported when attn_type='mha'")
        self.n_heads = int(n_heads)
        self.use_adaln_conditioning = bool(use_adaln_conditioning)
        norm_affine = bool(norm_affine_when_no_adaln) and not self.use_adaln_conditioning

        self.norm1 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=norm_affine)
        self.norm2 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=norm_affine)
        if self.use_adaln_conditioning:
            self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
            nn.init.zeros_(self.ada_ln[-1].weight)
            nn.init.zeros_(self.ada_ln[-1].bias)
        else:
            self.ada_ln = None

        if self.attn_type == "mha":
            self.attn = RegularAttention(
                d_model,
                self.n_heads,
                dropout=attn_dropout,
                position_mode=self.mha_position_mode,
                rope_freq_sigma=mha_rope_freq_sigma,
                rope_learned_freqs=mha_rope_learned_freqs,
                rope_use_key=mha_rope_use_key,
                rope_on_values=mha_rope_on_values,
            )
        else:
            self.attn = SimplicialAttention(
                dim=d_model,
                num_heads=self.n_heads,
                head_dim=simplicial_head_dim,
                dropout=attn_dropout,
                chunk_size=simplicial_chunk_size,
                bias=True,
                out_proj=True,
                impl=simplicial_impl,
                precision=simplicial_precision,
                message_mode=simplicial_message_mode,
                message_rank=simplicial_message_rank,
                content_logits=simplicial_content_logits,
            )
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond_emb: torch.Tensor | None,
        pad_mask: torch.Tensor | None = None,
        attn_head_bias: torch.Tensor | None = None,
        simplicial_attention_mask: SimplicialAttentionMask | None = None,
        simplicial_factorized_bias: SimplicialFactorizedBias | None = None,
        simplicial_angle_residual: SimplicialLowRankAngleResidual | None = None,
        simplicial_message_residual: SimplicialLowRankMessageResidual | None = None,
        simplicial_logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None = None,
        mha_positions: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_adaln_conditioning:
            if cond_emb is None:
                raise ValueError("cond_emb must be provided when use_adaln_conditioning=True")
            if cond_emb.ndim != 2 or cond_emb.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"cond_emb must have shape (B, {x.shape[-1]}), got {tuple(cond_emb.shape)}"
                )
            assert self.ada_ln is not None
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond_emb).chunk(6, dim=-1)
            h = _modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            h = self.norm1(x)
        if self.attn_type == "mha":
            attn_mask = None
            key_padding_mask = pad_mask
            if attn_head_bias is not None:
                if attn_head_bias.shape[:2] != (x.shape[0], self.n_heads):
                    raise ValueError(
                        f"attn_head_bias must have shape (B, H, T, T), got {tuple(attn_head_bias.shape)}"
                    )
                attn_mask = attn_head_bias.reshape(-1, x.shape[1], x.shape[1]).to(dtype=h.dtype)
                if pad_mask is not None:
                    key_padding_mask = torch.zeros_like(pad_mask, dtype=h.dtype)
                    key_padding_mask = key_padding_mask.masked_fill(
                        pad_mask,
                        torch.finfo(h.dtype).min,
                    )
            if simplicial_logit_bias_fn is not None:
                raise ValueError("simplicial_logit_bias_fn must be None for MultiheadAttention")
            if simplicial_factorized_bias is not None:
                raise ValueError("simplicial_factorized_bias is only supported for simplicial attention")
            if simplicial_angle_residual is not None:
                raise ValueError("simplicial_angle_residual is only supported for simplicial attention")
            if simplicial_message_residual is not None:
                raise ValueError("simplicial_message_residual is only supported for simplicial attention")
            if self.mha_position_mode == "rope":
                attn_out, _ = self.attn(
                    h,
                    h,
                    h,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                    positions=mha_positions,
                )
            else:
                attn_out, _ = self.attn(
                    h,
                    h,
                    h,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )
        else:
            if attn_head_bias is not None:
                raise ValueError("attn_head_bias is only supported for MultiheadAttention")
            attn_out = self.attn(
                h,
                key_padding_mask=pad_mask,
                attention_mask=simplicial_attention_mask,
                factorized_bias=simplicial_factorized_bias,
                angle_residual=simplicial_angle_residual,
                message_residual=simplicial_message_residual,
                logit_bias_fn=simplicial_logit_bias_fn,
            )
        if self.use_adaln_conditioning:
            x = x + gate_msa[:, None, :] * attn_out
            h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp[:, None, :] * self.mlp(h)
        else:
            x = x + attn_out
            h = self.norm2(x)
            x = x + self.mlp(h)
        return x


class _GeometryBiasBase(nn.Module):
    def __init__(
        self,
        n_heads: int,
        use_noise_gate: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.use_noise_gate = bool(use_noise_gate)
        if self.use_noise_gate:
            gate_alpha_init = torch.ones((1, self.n_heads, 1, 1), dtype=torch.float32)
            self.noise_gate_alpha_raw = nn.Parameter(_softplus_inverse(gate_alpha_init))
            self.noise_gate_beta = nn.Parameter(
                torch.zeros((1, self.n_heads, 1, 1), dtype=torch.float32)
            )

    @staticmethod
    def _mask_atom_pairwise_bias(
        pair_bias: torch.Tensor,
        atom_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if atom_pad_mask is None:
            return pair_bias
        pair_bias = pair_bias.masked_fill(atom_pad_mask[:, None, :, None], 0.0)
        pair_bias = pair_bias.masked_fill(atom_pad_mask[:, None, None, :], 0.0)
        return pair_bias

    @staticmethod
    def _zero_pairwise_diag(pair_bias: torch.Tensor) -> torch.Tensor:
        diag = torch.diagonal(pair_bias, offset=0, dim1=-2, dim2=-1)
        return pair_bias - torch.diag_embed(diag)

    @staticmethod
    def _expand_atom_head_bias(
        bias_atoms: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size, n_heads, num_atoms, _ = bias_atoms.shape
        if seq_len < num_atoms:
            raise ValueError(f"seq_len={seq_len} is smaller than the geometry length {num_atoms}")
        if seq_len == num_atoms:
            return bias_atoms
        bias_full = torch.zeros(
            batch_size,
            n_heads,
            seq_len,
            seq_len,
            device=bias_atoms.device,
            dtype=bias_atoms.dtype,
        )
        bias_full[:, :, :num_atoms, :num_atoms] = bias_atoms
        return bias_full

    def _sigma_gate(
        self,
        sigma: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_noise_gate:
            return torch.ones((batch_size, self.n_heads, 1, 1), device=device, dtype=dtype)
        if sigma is None:
            raise ValueError("sigma must be provided when use_noise_gate=True")
        if sigma.ndim == 2 and sigma.shape[1] == 1:
            sigma = sigma[:, 0]
        if sigma.ndim != 1:
            raise ValueError(f"sigma must have shape (B,) or (B, 1), got {tuple(sigma.shape)}")
        if sigma.shape[0] == 1 and batch_size > 1:
            sigma = sigma.expand(batch_size)
        if sigma.shape[0] != batch_size:
            raise ValueError(f"sigma batch {sigma.shape[0]} does not match batch size {batch_size}")
        sigma_feat = -torch.log(sigma.to(device=device, dtype=torch.float32).clamp_min(1e-8))
        alpha = F.softplus(self.noise_gate_alpha_raw).to(device=device, dtype=torch.float32)
        beta = self.noise_gate_beta.to(device=device, dtype=torch.float32)
        gate = torch.sigmoid(alpha * sigma_feat[:, None, None, None] + beta)
        return gate.to(dtype=dtype)


class GeometryBiasBuilder(_GeometryBiasBase):
    def __init__(
        self,
        n_heads: int,
        use_distance_bias: bool = True,
        use_edge_bias: bool = True,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        edge_bias_n_freqs: int = 8,
        use_periodic_features: bool = False,
        dist_slope_init: float = -1.0,
        use_noise_gate: bool = True,
    ) -> None:
        super().__init__(n_heads=n_heads, use_noise_gate=use_noise_gate)
        if not (use_distance_bias or use_edge_bias):
            raise ValueError("GeometryBiasBuilder requires use_distance_bias or use_edge_bias")
        self.use_distance_bias = bool(use_distance_bias)
        self.use_edge_bias = bool(use_edge_bias)
        self.use_periodic_features = bool(use_periodic_features)
        self.edge_bias_n_rbf = int(edge_bias_n_rbf)
        if self.edge_bias_n_rbf <= 0:
            raise ValueError("edge_bias_n_rbf must be positive")

        slope_mag = torch.full(
            (1, self.n_heads, 1, 1),
            fill_value=abs(float(dist_slope_init)),
            dtype=torch.float32,
        ).clamp_min(1e-4)
        self.dist_slope_raw = nn.Parameter(_softplus_inverse(slope_mag))

        self.register_buffer(
            "_edge_rbf_centers",
            torch.linspace(0.0, edge_bias_rbf_max, edge_bias_n_rbf, dtype=torch.float32),
            persistent=False,
        )
        delta = edge_bias_rbf_max / max(edge_bias_n_rbf - 1, 1)
        self.register_buffer(
            "_edge_rbf_gamma",
            torch.tensor(1.0 / max(delta * delta, 1e-6), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_edge_freqs",
            torch.arange(1, edge_bias_n_freqs + 1, dtype=torch.float32),
            persistent=False,
        )

        self.edge_bias_mlp = nn.Sequential(
            nn.LazyLinear(edge_bias_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_bias_hidden_dim, self.n_heads),
        )
        nn.init.zeros_(self.edge_bias_mlp[-1].weight)
        nn.init.zeros_(self.edge_bias_mlp[-1].bias)

    def _distance_rbf(self, dist_norm: torch.Tensor) -> torch.Tensor:
        centers = self._edge_rbf_centers.to(device=dist_norm.device, dtype=dist_norm.dtype)
        gamma = self._edge_rbf_gamma.to(device=dist_norm.device, dtype=dist_norm.dtype)
        return torch.exp(-gamma * (dist_norm.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())

    def _build_pair_features(self, geom_features: GeometryFeatures) -> torch.Tensor:
        features = [self._distance_rbf(geom_features.pair_dist_norm), geom_features.pair_delta]
        if self.use_periodic_features and geom_features.kind == "periodic":
            freqs = self._edge_freqs.to(
                device=geom_features.pair_delta.device,
                dtype=geom_features.pair_delta.dtype,
            )
            args = 2.0 * torch.pi * geom_features.pair_delta[..., None] * freqs.view(1, 1, 1, 1, -1)
            fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            batch_size, num_atoms, _, _, _ = fourier.shape
            features.append(fourier.reshape(batch_size, num_atoms, num_atoms, -1))
            features.append(
                geom_features.global_geom[:, None, None, :].expand(-1, num_atoms, num_atoms, -1)
            )
        return torch.cat(features, dim=-1)

    def forward_from_features(
        self,
        geom_features: GeometryFeatures,
        coords_len: int,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        sigma: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        pair_bias = None
        if self.use_distance_bias:
            slope = -F.softplus(self.dist_slope_raw)
            pair_bias = slope * geom_features.pair_dist_norm.unsqueeze(1)

        if self.use_edge_bias:
            edge_feat = self._build_pair_features(geom_features)
            edge_bias = self.edge_bias_mlp(edge_feat).permute(0, 3, 1, 2).contiguous()
            edge_bias = self._zero_pairwise_diag(edge_bias)
            pair_bias = edge_bias if pair_bias is None else pair_bias + edge_bias

        if pair_bias is None:
            return None

        atom_pad_mask = pad_mask[:, :coords_len].bool() if pad_mask is not None else None
        pair_bias = self._mask_atom_pairwise_bias(pair_bias, atom_pad_mask)
        pair_bias = pair_bias * geom_features.pair_mask[:, None, :, :].to(dtype=pair_bias.dtype)
        pair_bias = self._sigma_gate(
            sigma=sigma,
            batch_size=pair_bias.shape[0],
            dtype=pair_bias.dtype,
            device=pair_bias.device,
        ) * pair_bias
        pair_bias = self._expand_atom_head_bias(
            pair_bias,
            seq_len=seq_len or coords_len,
        )
        if out_dtype is not None:
            pair_bias = pair_bias.to(dtype=out_dtype)
        return pair_bias


class SimplicialGeometryBias(_GeometryBiasBase):
    def __init__(
        self,
        n_heads: int,
        mode: str = "factorized",
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        edge_bias_n_freqs: int = 8,
        angle_residual_rank: int = 16,
        message_mode: str = "none",
        message_rank: int = 16,
        use_periodic_features: bool = False,
        use_noise_gate: bool = True,
    ) -> None:
        super().__init__(n_heads=n_heads, use_noise_gate=use_noise_gate)
        mode = mode.lower()
        if mode not in {"none", "factorized", "angle_residual", "angle_low_rank"}:
            raise ValueError(f"Unsupported simplicial geometry mode: {mode}")
        message_mode = message_mode.lower()
        if message_mode not in {"none", "low_rank"}:
            raise ValueError(f"Unsupported simplicial message mode: {message_mode}")
        self.mode = mode
        self.message_mode = message_mode
        self.use_periodic_features = bool(use_periodic_features)
        self.edge_bias_n_rbf = int(edge_bias_n_rbf)
        self.angle_residual_rank = int(angle_residual_rank)
        self.message_rank = int(message_rank)
        if self.angle_residual_rank <= 0:
            raise ValueError("angle_residual_rank must be positive")
        if self.message_mode == "low_rank" and self.message_rank <= 0:
            raise ValueError("message_rank must be positive when message_mode='low_rank'")

        self.register_buffer(
            "_edge_rbf_centers",
            torch.linspace(0.0, edge_bias_rbf_max, edge_bias_n_rbf, dtype=torch.float32),
            persistent=False,
        )
        delta = edge_bias_rbf_max / max(edge_bias_n_rbf - 1, 1)
        self.register_buffer(
            "_edge_rbf_gamma",
            torch.tensor(1.0 / max(delta * delta, 1e-6), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_edge_freqs",
            torch.arange(1, edge_bias_n_freqs + 1, dtype=torch.float32),
            persistent=False,
        )

        if self.mode != "none":
            self.spoke_bias_u_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads),
            )
            self.spoke_bias_v_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads),
            )
            self.pair_bias_w_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads),
            )
            for mlp in (self.spoke_bias_u_mlp, self.spoke_bias_v_mlp, self.pair_bias_w_mlp):
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
        else:
            self.spoke_bias_u_mlp = None
            self.spoke_bias_v_mlp = None
            self.pair_bias_w_mlp = None

        if self.mode == "angle_residual":
            self.angle_residual_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads),
            )
            nn.init.zeros_(self.angle_residual_mlp[-1].weight)
            nn.init.zeros_(self.angle_residual_mlp[-1].bias)
        else:
            self.angle_residual_mlp = None
        if self.mode == "angle_low_rank":
            self.angle_left_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads * self.angle_residual_rank),
            )
            self.angle_right_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads * self.angle_residual_rank),
            )
            # Zero only one side: zeroing both factors would make the product branch have zero gradients.
            nn.init.zeros_(self.angle_left_mlp[-1].weight)
            nn.init.zeros_(self.angle_left_mlp[-1].bias)
        else:
            self.angle_left_mlp = None
            self.angle_right_mlp = None
        if self.message_mode == "low_rank":
            self.message_left_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads * self.message_rank),
            )
            self.message_right_mlp = nn.Sequential(
                nn.LazyLinear(edge_bias_hidden_dim),
                nn.SiLU(),
                nn.Linear(edge_bias_hidden_dim, self.n_heads * self.message_rank),
            )
            nn.init.zeros_(self.message_left_mlp[-1].weight)
            nn.init.zeros_(self.message_left_mlp[-1].bias)
        else:
            self.message_left_mlp = None
            self.message_right_mlp = None

    def _distance_rbf(self, dist_norm: torch.Tensor) -> torch.Tensor:
        centers = self._edge_rbf_centers.to(device=dist_norm.device, dtype=dist_norm.dtype)
        gamma = self._edge_rbf_gamma.to(device=dist_norm.device, dtype=dist_norm.dtype)
        return torch.exp(-gamma * (dist_norm.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())

    def _build_spoke_features(self, geom_features: GeometryFeatures) -> torch.Tensor:
        features = [geom_features.pair_delta, self._distance_rbf(geom_features.pair_dist_norm)]
        if self.use_periodic_features and geom_features.kind == "periodic":
            freqs = self._edge_freqs.to(
                device=geom_features.pair_delta.device,
                dtype=geom_features.pair_delta.dtype,
            )
            args = 2.0 * torch.pi * geom_features.pair_delta[..., None] * freqs.view(1, 1, 1, 1, -1)
            fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            batch_size, num_atoms, _, _, _ = fourier.shape
            features.append(fourier.reshape(batch_size, num_atoms, num_atoms, -1))
            features.append(
                geom_features.global_geom[:, None, None, :].expand(-1, num_atoms, num_atoms, -1)
            )
        return torch.cat(features, dim=-1)

    def _build_angle_spoke_features(self, geom_features: GeometryFeatures) -> torch.Tensor:
        unit_delta = geom_features.pair_delta / geom_features.pair_dist.clamp_min(1e-8).unsqueeze(-1)
        unit_delta = torch.where(
            geom_features.pair_mask.unsqueeze(-1),
            unit_delta,
            torch.zeros_like(unit_delta),
        )
        metric = geom_features.extras["metric"].to(
            device=geom_features.pair_delta.device,
            dtype=geom_features.pair_delta.dtype,
        )
        metric_unit_delta = torch.einsum("bnmd,bdc->bnmc", geom_features.pair_delta, metric)
        metric_unit_delta = metric_unit_delta / geom_features.pair_dist.clamp_min(1e-8).unsqueeze(-1)
        metric_unit_delta = torch.where(
            geom_features.pair_mask.unsqueeze(-1),
            metric_unit_delta,
            torch.zeros_like(metric_unit_delta),
        )
        features = [
            unit_delta,
            metric_unit_delta,
            geom_features.pair_delta,
            self._distance_rbf(geom_features.pair_dist_norm),
        ]
        if self.use_periodic_features and geom_features.kind == "periodic":
            freqs = self._edge_freqs.to(
                device=geom_features.pair_delta.device,
                dtype=geom_features.pair_delta.dtype,
            )
            args = 2.0 * torch.pi * geom_features.pair_delta[..., None] * freqs.view(1, 1, 1, 1, -1)
            fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            batch_size, num_atoms, _, _, _ = fourier.shape
            features.append(fourier.reshape(batch_size, num_atoms, num_atoms, -1))
            features.append(
                geom_features.global_geom[:, None, None, :].expand(-1, num_atoms, num_atoms, -1)
            )
        return torch.cat(features, dim=-1)

    def _build_pair_features(self, geom_features: GeometryFeatures) -> torch.Tensor:
        features = [self._distance_rbf(geom_features.pair_dist_norm)]
        if self.use_periodic_features and geom_features.kind == "periodic":
            batch_size, num_atoms, _, _ = geom_features.pair_dist_norm.unsqueeze(-1).shape
            features.append(
                geom_features.global_geom[:, None, None, :].expand(-1, num_atoms, num_atoms, -1)
            )
        return torch.cat(features, dim=-1)

    def _compute_factorized_terms(
        self,
        geom_features: GeometryFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.spoke_bias_u_mlp is None or self.spoke_bias_v_mlp is None or self.pair_bias_w_mlp is None:
            raise RuntimeError("Factorized simplicial geometry requested without factorized geometry MLPs")
        spoke_feat = self._build_spoke_features(geom_features)
        pair_feat = self._build_pair_features(geom_features)
        u_bias = self.spoke_bias_u_mlp(spoke_feat).permute(0, 3, 1, 2).contiguous()
        v_bias = self.spoke_bias_v_mlp(spoke_feat).permute(0, 3, 1, 2).contiguous()
        w_bias = self.pair_bias_w_mlp(pair_feat).permute(0, 3, 1, 2).contiguous()
        u_bias = self._zero_pairwise_diag(u_bias)
        v_bias = self._zero_pairwise_diag(v_bias)
        w_bias = self._zero_pairwise_diag(w_bias)
        mask = geom_features.pair_mask[:, None, :, :].to(dtype=u_bias.dtype)
        return u_bias * mask, v_bias * mask, w_bias * mask

    def _compute_low_rank_angle_residual(
        self,
        geom_features: GeometryFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.angle_left_mlp is None or self.angle_right_mlp is None:
            raise RuntimeError("Low-rank angle residual requested without low-rank angle MLPs")
        features = self._build_angle_spoke_features(geom_features)
        batch_size, num_atoms, _, _ = geom_features.pair_delta.shape
        factor_shape = (batch_size, num_atoms, num_atoms, self.n_heads, self.angle_residual_rank)
        left = self.angle_left_mlp(features).view(factor_shape).permute(0, 3, 1, 2, 4).contiguous()
        right = self.angle_right_mlp(features).view(factor_shape).permute(0, 3, 1, 2, 4).contiguous()
        pair_mask = geom_features.pair_mask[:, None, :, :, None].to(dtype=left.dtype)
        eye = torch.eye(num_atoms, device=left.device, dtype=torch.bool).view(1, 1, num_atoms, num_atoms, 1)
        left = left.masked_fill(eye, 0.0) * pair_mask
        right = right.masked_fill(eye, 0.0) * pair_mask
        return left, right

    def _compute_low_rank_message_residual(
        self,
        geom_features: GeometryFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.message_left_mlp is None or self.message_right_mlp is None:
            raise RuntimeError("Low-rank message residual requested without message MLPs")
        features = self._build_angle_spoke_features(geom_features)
        batch_size, num_atoms, _, _ = geom_features.pair_delta.shape
        factor_shape = (batch_size, num_atoms, num_atoms, self.n_heads, self.message_rank)
        left = self.message_left_mlp(features).view(factor_shape).permute(0, 3, 1, 2, 4).contiguous()
        right = self.message_right_mlp(features).view(factor_shape).permute(0, 3, 1, 2, 4).contiguous()
        pair_mask = geom_features.pair_mask[:, None, :, :, None].to(dtype=left.dtype)
        eye = torch.eye(num_atoms, device=left.device, dtype=torch.bool).view(1, 1, num_atoms, num_atoms, 1)
        left = left.masked_fill(eye, 0.0) * pair_mask
        right = right.masked_fill(eye, 0.0) * pair_mask
        return left, right

    @staticmethod
    def _mask_atom_pairwise_rank_factor(
        factor: torch.Tensor,
        atom_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if atom_pad_mask is None:
            return factor
        factor = factor.masked_fill(atom_pad_mask[:, None, :, None, None], 0.0)
        factor = factor.masked_fill(atom_pad_mask[:, None, None, :, None], 0.0)
        return factor

    @staticmethod
    def _expand_atom_head_rank_factor(
        factor_atoms: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size, n_heads, num_atoms, _, rank = factor_atoms.shape
        if seq_len < num_atoms:
            raise ValueError(f"seq_len={seq_len} is smaller than the geometry length {num_atoms}")
        if seq_len == num_atoms:
            return factor_atoms
        factor_full = torch.zeros(
            batch_size,
            n_heads,
            seq_len,
            seq_len,
            rank,
            device=factor_atoms.device,
            dtype=factor_atoms.dtype,
        )
        factor_full[:, :, :num_atoms, :num_atoms, :] = factor_atoms
        return factor_full

    def _compute_angle_residual_chunk(
        self,
        geom_features: GeometryFeatures,
        q_start: int,
        q_end: int,
    ) -> torch.Tensor:
        if self.angle_residual_mlp is None:
            raise RuntimeError("Angle residual requested without angle_residual_mlp")
        delta_q = geom_features.pair_delta[:, q_start:q_end, :, :]
        dist_q = geom_features.pair_dist[:, q_start:q_end, :]
        dist_q_norm = geom_features.pair_dist_norm[:, q_start:q_end, :]
        metric = geom_features.extras["metric"].to(
            device=delta_q.device,
            dtype=delta_q.dtype,
        )
        delta_q_metric = torch.einsum("bqnd,bdc->bqnc", delta_q, metric)
        numerator = torch.einsum("bqjc,bqkc->bqjk", delta_q_metric, delta_q)
        denom = dist_q[:, :, :, None] * dist_q[:, :, None, :]
        valid = denom > 1e-8
        pair_mask = geom_features.pair_mask[:, q_start:q_end, :]
        valid = valid & pair_mask[:, :, :, None] & pair_mask[:, :, None, :]
        cos_theta = torch.zeros_like(numerator)
        cos_theta = torch.where(valid, numerator / denom.clamp_min(1e-8), cos_theta)
        cos_theta = cos_theta.clamp(min=-1.0, max=1.0)

        dist_rbf = self._distance_rbf(dist_q_norm)
        num_atoms = dist_q_norm.shape[-1]
        dij_feat = dist_rbf.unsqueeze(-2).expand(-1, -1, -1, num_atoms, -1)
        dik_feat = dist_rbf.unsqueeze(-3).expand(-1, -1, num_atoms, -1, -1)
        tri_feat = torch.cat([dij_feat, dik_feat, cos_theta.unsqueeze(-1)], dim=-1)
        tri_bias = self.angle_residual_mlp(tri_feat).permute(0, 4, 1, 2, 3).contiguous()
        return tri_bias * valid[:, None, :, :, :].to(dtype=tri_bias.dtype)

    def build_structured_bias_inputs(
        self,
        geom_features: GeometryFeatures,
        coords_len: int,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        sigma: torch.Tensor | None = None,
    ) -> tuple[
        SimplicialFactorizedBias | None,
        Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None,
        SimplicialLowRankAngleResidual | None,
        SimplicialLowRankMessageResidual | None,
        SimplicialAttentionMask,
    ]:
        atom_pad_mask = pad_mask[:, :coords_len].bool() if pad_mask is not None else None
        target_seq_len = seq_len or coords_len
        gate_source = geom_features.pair_dist_norm
        gate = self._sigma_gate(
            sigma=sigma,
            batch_size=gate_source.shape[0],
            dtype=gate_source.dtype,
            device=gate_source.device,
        )
        gate = gate.expand(-1, -1, target_seq_len, -1).squeeze(-1)
        if target_seq_len > coords_len:
            q_is_atom = (
                torch.arange(target_seq_len, device=gate.device) < coords_len
            ).to(dtype=gate.dtype)
            gate = gate * q_is_atom.view(1, 1, target_seq_len)

        attention_mask = _build_simplicial_attention_mask(
            coords_len=coords_len,
            seq_len=target_seq_len,
            batch_size=gate.shape[0],
            device=gate.device,
            pad_mask=pad_mask,
            geom_pair_mask=geom_features.pair_mask,
        )

        factorized_bias = None
        if self.mode != "none":
            u_atoms, v_atoms, w_atoms = self._compute_factorized_terms(geom_features)
            u_atoms = self._mask_atom_pairwise_bias(u_atoms, atom_pad_mask)
            v_atoms = self._mask_atom_pairwise_bias(v_atoms, atom_pad_mask)
            w_atoms = self._mask_atom_pairwise_bias(w_atoms, atom_pad_mask)
            factorized_bias = SimplicialFactorizedBias(
                u=self._expand_atom_head_bias(u_atoms, seq_len=target_seq_len),
                v=self._expand_atom_head_bias(v_atoms, seq_len=target_seq_len),
                w=self._expand_atom_head_bias(w_atoms, seq_len=target_seq_len),
                gate=gate,
            )

        message_residual = None
        if self.message_mode == "low_rank":
            message_left_atoms, message_right_atoms = self._compute_low_rank_message_residual(geom_features)
            message_left_atoms = self._mask_atom_pairwise_rank_factor(message_left_atoms, atom_pad_mask)
            message_right_atoms = self._mask_atom_pairwise_rank_factor(message_right_atoms, atom_pad_mask)
            message_residual = SimplicialLowRankMessageResidual(
                left=self._expand_atom_head_rank_factor(message_left_atoms, seq_len=target_seq_len),
                right=self._expand_atom_head_rank_factor(message_right_atoms, seq_len=target_seq_len),
            )

        if self.mode == "none":
            return None, None, None, message_residual, attention_mask

        if factorized_bias is None:
            raise RuntimeError("factorized_bias was not built for a non-none simplicial geometry mode")

        if self.mode == "factorized":
            return factorized_bias, None, None, message_residual, attention_mask

        if self.mode == "angle_low_rank":
            left_atoms, right_atoms = self._compute_low_rank_angle_residual(geom_features)
            left_atoms = self._mask_atom_pairwise_rank_factor(left_atoms, atom_pad_mask)
            right_atoms = self._mask_atom_pairwise_rank_factor(right_atoms, atom_pad_mask)
            left_full = self._expand_atom_head_rank_factor(left_atoms, seq_len=target_seq_len)
            right_full = self._expand_atom_head_rank_factor(right_atoms, seq_len=target_seq_len)
            angle_residual = SimplicialLowRankAngleResidual(
                left=left_full,
                right=right_full,
                gate=gate,
            )
            return factorized_bias, None, angle_residual, message_residual, attention_mask

        def _residual_fn(
            start: int,
            end: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            residual = torch.zeros(
                gate.shape[0],
                self.n_heads,
                end - start,
                target_seq_len,
                target_seq_len,
                device=gate.device,
                dtype=gate.dtype,
            )
            atom_q_end = min(end, coords_len)
            if start < atom_q_end:
                angle_chunk = self._compute_angle_residual_chunk(
                    geom_features=geom_features,
                    q_start=start,
                    q_end=atom_q_end,
                )
                residual[:, :, : atom_q_end - start, :coords_len, :coords_len] = angle_chunk
            gate_chunk = gate[:, :, start:end].unsqueeze(-1).unsqueeze(-1)
            return (gate_chunk * residual).to(device=device, dtype=dtype)

        return factorized_bias, _residual_fn, None, message_residual, attention_mask

    def build_bias_inputs(
        self,
        geom_features: GeometryFeatures,
        coords_len: int,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        sigma: torch.Tensor | None = None,
    ) -> tuple[
        SimplicialFactorizedBias | None,
        Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None,
        SimplicialAttentionMask,
    ]:
        factorized_bias, residual_fn, angle_residual, _, attention_mask = self.build_structured_bias_inputs(
            geom_features=geom_features,
            coords_len=coords_len,
            pad_mask=pad_mask,
            seq_len=seq_len,
            sigma=sigma,
        )
        if residual_fn is None and angle_residual is not None:

            def _low_rank_residual_fn(
                start: int,
                end: int,
                dtype: torch.dtype,
                device: torch.device,
            ) -> torch.Tensor:
                return angle_residual.chunk(start, end, dtype=dtype, device=device)

            residual_fn = _low_rank_residual_fn
        return factorized_bias, residual_fn, attention_mask

    def build_logit_bias_fn(
        self,
        geom_features: GeometryFeatures,
        coords_len: int,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        sigma: torch.Tensor | None = None,
    ) -> Callable[[int, int, torch.dtype, torch.device], torch.Tensor]:
        factorized_bias, residual_fn, angle_residual, _, _ = self.build_structured_bias_inputs(
            geom_features=geom_features,
            coords_len=coords_len,
            pad_mask=pad_mask,
            seq_len=seq_len,
            sigma=sigma,
        )

        def _bias_fn(
            start: int,
            end: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            if factorized_bias is None:
                raise RuntimeError("build_logit_bias_fn requires a non-none simplicial geometry mode")
            bias = factorized_bias.chunk(start, end, dtype=dtype, device=device)
            if residual_fn is None:
                if angle_residual is None:
                    return bias
                return bias + angle_residual.chunk(start, end, dtype=dtype, device=device)
            return bias + residual_fn(start, end, dtype, device)

        return _bias_fn


class MhaFactorizedGeometryBias(SimplicialGeometryBias):
    """MHA attention-logit bias from the same factorized geometry terms as simplicial attention."""

    def __init__(
        self,
        n_heads: int,
        edge_bias_hidden_dim: int = 128,
        edge_bias_n_rbf: int = 16,
        edge_bias_rbf_max: float = 2.0,
        edge_bias_n_freqs: int = 8,
        use_periodic_features: bool = False,
        use_noise_gate: bool = True,
        marginal_chunk_size: int = 8,
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            mode="factorized",
            edge_bias_hidden_dim=edge_bias_hidden_dim,
            edge_bias_n_rbf=edge_bias_n_rbf,
            edge_bias_rbf_max=edge_bias_rbf_max,
            edge_bias_n_freqs=edge_bias_n_freqs,
            use_periodic_features=use_periodic_features,
            use_noise_gate=use_noise_gate,
        )
        if marginal_chunk_size <= 0:
            raise ValueError("marginal_chunk_size must be positive")
        self.marginal_chunk_size = int(marginal_chunk_size)

    def _collapse_factorized_terms(
        self,
        u_atoms: torch.Tensor,
        v_atoms: torch.Tensor,
        w_atoms: torch.Tensor,
        pair_mask: torch.Tensor,
        gate_atoms: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_heads, num_atoms, _ = u_atoms.shape
        pair_bias = torch.zeros(
            batch_size,
            n_heads,
            num_atoms,
            num_atoms,
            device=u_atoms.device,
            dtype=torch.float32,
        )
        valid_pair_keys = pair_mask[:, None, None, :, :]
        valid_counts = pair_mask.sum(dim=-1).clamp_min(1).to(device=u_atoms.device)
        log_valid_counts = valid_counts.to(dtype=torch.float32).log()[:, None, None, :]
        has_valid_keys = pair_mask.any(dim=-1)[:, None, None, :]

        for start in range(0, num_atoms, self.marginal_chunk_size):
            end = min(num_atoms, start + self.marginal_chunk_size)
            gate_chunk = gate_atoms[:, :, start:end].float()
            vk = v_atoms[:, :, start:end, None, :].float()
            wjk = w_atoms[:, :, None, :, :].float()
            logits = gate_chunk[:, :, :, None, None] * (vk + wjk)
            logits = logits.masked_fill(~valid_pair_keys, torch.finfo(logits.dtype).min)
            marginal = torch.logsumexp(logits, dim=-1) - log_valid_counts
            marginal = torch.where(has_valid_keys, marginal, torch.zeros_like(marginal))
            pair_bias[:, :, start:end, :] = (
                gate_chunk[:, :, :, None] * u_atoms[:, :, start:end, :].float() + marginal
            )

        return pair_bias

    def forward_from_features(
        self,
        geom_features: GeometryFeatures,
        coords_len: int,
        pad_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        sigma: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        u_atoms, v_atoms, w_atoms = self._compute_factorized_terms(geom_features)
        atom_pad_mask = pad_mask[:, :coords_len].bool() if pad_mask is not None else None
        u_atoms = self._mask_atom_pairwise_bias(u_atoms, atom_pad_mask)
        v_atoms = self._mask_atom_pairwise_bias(v_atoms, atom_pad_mask)
        w_atoms = self._mask_atom_pairwise_bias(w_atoms, atom_pad_mask)

        gate = self._sigma_gate(
            sigma=sigma,
            batch_size=u_atoms.shape[0],
            dtype=u_atoms.dtype,
            device=u_atoms.device,
        )
        gate_atoms = gate.expand(-1, -1, coords_len, -1).squeeze(-1)
        pair_bias = self._collapse_factorized_terms(
            u_atoms=u_atoms,
            v_atoms=v_atoms,
            w_atoms=w_atoms,
            pair_mask=geom_features.pair_mask,
            gate_atoms=gate_atoms,
        )
        pair_bias = self._mask_atom_pairwise_bias(pair_bias, atom_pad_mask)
        pair_bias = pair_bias * geom_features.pair_mask[:, None, :, :].to(dtype=pair_bias.dtype)
        pair_bias = self._expand_atom_head_bias(
            pair_bias,
            seq_len=seq_len or coords_len,
        )
        if out_dtype is not None:
            pair_bias = pair_bias.to(dtype=out_dtype)
        return pair_bias


class TransformerTrunk(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        eps: float = 1e-6,
        attn_type: str = "mha",
        simplicial_chunk_size: int = 128,
        simplicial_head_dim: int | None = None,
        simplicial_impl: str = "auto",
        simplicial_precision: str = "bf16_tc",
        simplicial_message_mode: str = "none",
        simplicial_message_rank: int = 16,
        simplicial_content_logits: str = "on",
        geometry_adapter: BaseGeometryAdapter | None = None,
        geometry_bias: GeometryBiasBuilder | None = None,
        simplicial_geometry_bias: SimplicialGeometryBias | None = None,
        mha_position_mode: str = "none",
        mha_rope_freq_sigma: float = 1.0,
        mha_rope_learned_freqs: bool = False,
        mha_rope_use_key: bool = True,
        mha_rope_on_values: bool = False,
        use_adaln_conditioning: bool = True,
        norm_affine_when_no_adaln: bool = False,
        use_final_norm: bool = True,
    ) -> None:
        super().__init__()
        attn_type = attn_type.lower()
        self.attn_type = attn_type
        self.mha_position_mode = _canonicalize_mha_position_mode(mha_position_mode)
        self.geometry_adapter = geometry_adapter
        self.geometry_bias = geometry_bias
        self.simplicial_geometry_bias = simplicial_geometry_bias
        self.use_adaln_conditioning = bool(use_adaln_conditioning)
        self.use_final_norm = bool(use_final_norm)
        norm_affine = bool(norm_affine_when_no_adaln) and not self.use_adaln_conditioning
        if self.mha_position_mode != "none" and self.attn_type != "mha":
            raise ValueError("mha_position_mode='rope' is only supported with attn_type='mha'")
        if self.geometry_bias is not None and self.attn_type != "mha":
            raise ValueError("geometry_bias is only supported with attn_type='mha'")
        if self.simplicial_geometry_bias is not None and self.attn_type != "simplicial":
            raise ValueError("simplicial_geometry_bias requires attn_type='simplicial'")

        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    eps=eps,
                    attn_type=attn_type,
                    simplicial_chunk_size=simplicial_chunk_size,
                    simplicial_head_dim=simplicial_head_dim,
                    simplicial_impl=simplicial_impl,
                    simplicial_precision=simplicial_precision,
                    simplicial_message_mode=simplicial_message_mode,
                    simplicial_message_rank=simplicial_message_rank,
                    simplicial_content_logits=simplicial_content_logits,
                    mha_position_mode=self.mha_position_mode,
                    mha_rope_freq_sigma=mha_rope_freq_sigma,
                    mha_rope_learned_freqs=mha_rope_learned_freqs,
                    mha_rope_use_key=mha_rope_use_key,
                    mha_rope_on_values=mha_rope_on_values,
                    use_adaln_conditioning=self.use_adaln_conditioning,
                    norm_affine_when_no_adaln=norm_affine_when_no_adaln,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = (
            nn.LayerNorm(d_model, eps=eps, elementwise_affine=norm_affine)
            if self.use_final_norm
            else nn.Identity()
        )

    def compute_geometry_features(
        self,
        coords: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
    ) -> GeometryFeatures:
        if self.geometry_adapter is None:
            raise RuntimeError("geometry_adapter is not configured")
        atom_pad_mask = None
        if pad_mask is not None:
            atom_pad_mask = pad_mask[:, : coords.shape[1]]
        return self.geometry_adapter(coords=coords, pad_mask=atom_pad_mask, lattice=lattice)

    @staticmethod
    def _build_mha_positions(
        coords: torch.Tensor,
        *,
        seq_len: int,
    ) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (B, N, 3) for MHA RoPE, got {tuple(coords.shape)}")
        coords_len = coords.shape[1]
        if seq_len < coords_len:
            raise ValueError(f"seq_len={seq_len} is smaller than coords_len={coords_len}")
        positions = coords.float()
        if seq_len == coords_len:
            return positions
        pad = torch.zeros(
            coords.shape[0],
            seq_len - coords_len,
            3,
            device=coords.device,
            dtype=positions.dtype,
        )
        return torch.cat([positions, pad], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cond_emb: torch.Tensor | None,
        *,
        pad_mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        lattice: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pad_mask is not None and pad_mask.dtype != torch.bool:
            pad_mask = pad_mask.bool()
        if pad_mask is not None and pad_mask.shape[:2] != x.shape[:2]:
            raise ValueError(f"pad_mask shape {tuple(pad_mask.shape)} does not match x {tuple(x.shape)}")

        geom_features = None
        attn_head_bias = None
        mha_positions = None
        simplicial_attention_mask = None
        simplicial_factorized_bias = None
        simplicial_angle_residual = None
        simplicial_message_residual = None
        simplicial_logit_bias_fn = None
        if self.mha_position_mode == "rope":
            if coords is None:
                raise ValueError("coords must be provided when mha_position_mode='rope'")
            mha_positions = self._build_mha_positions(coords, seq_len=x.shape[1])
        if self.geometry_adapter is not None:
            if coords is None:
                raise ValueError("coords must be provided when geometry_adapter is configured")
            geom_features = self.compute_geometry_features(coords=coords, pad_mask=pad_mask, lattice=lattice)

        if self.geometry_bias is not None and geom_features is not None:
            attn_head_bias = self.geometry_bias.forward_from_features(
                geom_features=geom_features,
                coords_len=coords.shape[1],
                pad_mask=pad_mask,
                seq_len=x.shape[1],
                sigma=sigma,
                out_dtype=x.dtype,
            )
        if self.simplicial_geometry_bias is not None and geom_features is not None:
            (
                simplicial_factorized_bias,
                simplicial_logit_bias_fn,
                simplicial_angle_residual,
                simplicial_message_residual,
                simplicial_attention_mask,
            ) = self.simplicial_geometry_bias.build_structured_bias_inputs(
                geom_features=geom_features,
                coords_len=coords.shape[1],
                pad_mask=pad_mask,
                seq_len=x.shape[1],
                sigma=sigma,
            )
        elif self.attn_type == "simplicial":
            if coords is not None:
                simplicial_attention_mask = _build_simplicial_attention_mask(
                    coords_len=coords.shape[1],
                    seq_len=x.shape[1],
                    batch_size=x.shape[0],
                    device=x.device,
                    pad_mask=pad_mask,
                    geom_pair_mask=geom_features.pair_mask if geom_features is not None else None,
                    include_global_tokens_as_pair_keys=(
                        x.shape[1] > coords.shape[1]
                        and geom_features is not None
                        and geom_features.kind == "periodic"
                    ),
                )
            else:
                simplicial_attention_mask = SimplicialAttentionMask.from_key_padding_mask(
                    pad_mask,
                    batch_size=x.shape[0],
                    num_tokens=x.shape[1],
                    device=x.device,
                )

        for block in self.blocks:
            if pad_mask is not None:
                x = x.masked_fill(pad_mask[..., None], 0.0)
            x = block(
                x,
                cond_emb,
                pad_mask=pad_mask,
                attn_head_bias=attn_head_bias,
                simplicial_attention_mask=simplicial_attention_mask,
                simplicial_factorized_bias=simplicial_factorized_bias,
                simplicial_angle_residual=simplicial_angle_residual,
                simplicial_message_residual=simplicial_message_residual,
                simplicial_logit_bias_fn=simplicial_logit_bias_fn,
                mha_positions=mha_positions,
                sigma=sigma,
            )
        x = self.norm_out(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask[..., None], 0.0)
        return x
