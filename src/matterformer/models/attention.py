from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from matterformer.models.attention_triton import (
    TRITON_AVAILABLE,
    normalize_simplicial_precision,
    triton_simplicial_attention_backward,
    triton_simplicial_attention_forward,
)


@dataclass(frozen=True)
class SimplicialFactorizedBias:
    """Structured factorized simplicial bias.

    Shapes:
    - ``u`` / ``v`` / ``w``: ``(B, H, T, T)``
    - ``gate``: ``(B, H, T)``, already expanded over query positions and with any
      non-atom queries zeroed out.
    """

    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    gate: torch.Tensor

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        expected_pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
        for name, value in (("u", self.u), ("v", self.v), ("w", self.w)):
            if value.shape != expected_pair_shape:
                raise ValueError(
                    f"{name} must have shape {expected_pair_shape}, got {tuple(value.shape)}"
                )
        expected_gate_shape = (batch_size, num_heads, num_tokens)
        if self.gate.shape != expected_gate_shape:
            raise ValueError(
                f"gate must have shape {expected_gate_shape}, got {tuple(self.gate.shape)}"
            )

    def chunk(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        u = self.u[:, :, start:end, :].float()
        v = self.v[:, :, start:end, :].float()
        w = self.w.float()
        gate = self.gate[:, :, start:end].float()
        bias = (
            u[:, :, :, :, None]
            + v[:, :, :, None, :]
            + w[:, :, None, :, :]
        )
        return (gate[:, :, :, None, None] * bias).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SimplicialLowRankAngleResidual:
    """Structured low-rank triplet residual for simplicial logits.

    Shapes:
    - ``left`` / ``right``: ``(B, H, T, T, R)``
    - ``gate``: ``(B, H, T)``, applied per query position.
    """

    left: torch.Tensor
    right: torch.Tensor
    gate: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.left.shape[-1])

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        if self.left.ndim != 5:
            raise ValueError(f"left must have shape (B, H, T, T, R), got {tuple(self.left.shape)}")
        expected_factor_prefix = (batch_size, num_heads, num_tokens, num_tokens)
        if self.left.shape[:4] != expected_factor_prefix:
            raise ValueError(
                f"left must have shape {expected_factor_prefix + (self.left.shape[-1],)}, "
                f"got {tuple(self.left.shape)}"
            )
        if self.right.shape != self.left.shape:
            raise ValueError(f"right must have shape {tuple(self.left.shape)}, got {tuple(self.right.shape)}")
        if self.rank <= 0:
            raise ValueError("low-rank angle residual rank must be positive")
        expected_gate_shape = (batch_size, num_heads, num_tokens)
        if self.gate.shape != expected_gate_shape:
            raise ValueError(
                f"gate must have shape {expected_gate_shape}, got {tuple(self.gate.shape)}"
            )

    def chunk(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        left = self.left[:, :, start:end, :, :].float()
        right = self.right[:, :, start:end, :, :].float()
        residual = torch.einsum("bhqjr,bhqkr->bhqjk", left, right) * (self.rank**-0.5)
        gate = self.gate[:, :, start:end].float().unsqueeze(-1).unsqueeze(-1)
        return (gate * residual).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SimplicialLowRankMessageResidual:
    """Structured low-rank triplet residual for simplicial values/messages.

    Shapes:
    - ``left`` / ``right``: ``(B, H, T, T, R)``
    """

    left: torch.Tensor
    right: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.left.shape[-1])

    def validate(self, *, batch_size: int, num_heads: int, num_tokens: int) -> None:
        if self.left.ndim != 5:
            raise ValueError(f"left must have shape (B, H, T, T, R), got {tuple(self.left.shape)}")
        expected_factor_prefix = (batch_size, num_heads, num_tokens, num_tokens)
        if self.left.shape[:4] != expected_factor_prefix:
            raise ValueError(
                f"left must have shape {expected_factor_prefix + (self.left.shape[-1],)}, "
                f"got {tuple(self.left.shape)}"
            )
        if self.right.shape != self.left.shape:
            raise ValueError(f"right must have shape {tuple(self.left.shape)}, got {tuple(self.right.shape)}")
        if self.rank <= 0:
            raise ValueError("low-rank message residual rank must be positive")


@dataclass(frozen=True)
class SimplicialAttentionMask:
    """Internal mask semantics for simplicial attention.

    Shapes:
    - ``query_valid``: ``(B, T)``
    - ``pair_key_valid``: ``(B, T)``
    - ``pair_valid``: optional ``(B, T, T)``
    """

    query_valid: torch.Tensor
    pair_key_valid: torch.Tensor
    pair_valid: torch.Tensor | None = None

    def validate(self, *, batch_size: int, num_tokens: int) -> None:
        expected_mask_shape = (batch_size, num_tokens)
        if self.query_valid.shape != expected_mask_shape:
            raise ValueError(
                f"query_valid must have shape {expected_mask_shape}, got {tuple(self.query_valid.shape)}"
            )
        if self.pair_key_valid.shape != expected_mask_shape:
            raise ValueError(
                f"pair_key_valid must have shape {expected_mask_shape}, got {tuple(self.pair_key_valid.shape)}"
            )
        if self.pair_valid is not None and self.pair_valid.shape != (batch_size, num_tokens, num_tokens):
            raise ValueError(
                "pair_valid must have shape "
                f"{(batch_size, num_tokens, num_tokens)}, got {tuple(self.pair_valid.shape)}"
            )

    @classmethod
    def from_key_padding_mask(
        cls,
        key_padding_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> "SimplicialAttentionMask":
        if key_padding_mask is None:
            valid = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
        else:
            if key_padding_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"key_padding_mask must have shape {(batch_size, num_tokens)}, got {tuple(key_padding_mask.shape)}"
                )
            valid = ~key_padding_mask.bool().to(device=device)
        return cls(query_valid=valid, pair_key_valid=valid)

    def pair_mask(self) -> torch.Tensor:
        pair_valid = self.pair_key_valid[:, :, None] & self.pair_key_valid[:, None, :]
        if self.pair_valid is not None:
            pair_valid = pair_valid & self.pair_valid.bool()
        return pair_valid


def _canonicalize_simplicial_position_mode(mode: str) -> str:
    mode = str(mode).lower().replace("-", "_")
    if mode in {"disabled", "off", "false", "no"}:
        return "none"
    if mode in {"center_edge_rope", "closed_simplicial_rope", "cs_rope"}:
        return "closed_rope"
    if mode not in {"none", "closed_rope"}:
        raise ValueError("simplicial_position_mode must be one of {'none', 'closed_rope'}")
    return mode


def _canonicalize_simplicial_rope_gate(mode: str) -> str:
    mode = str(mode).lower().replace("-", "_")
    if mode in {"disabled", "off", "false", "no"}:
        return "none"
    if mode not in {"none", "learned", "sigma"}:
        raise ValueError("simplicial_rope_gate must be one of {'none', 'learned', 'sigma'}")
    return mode


def _canonicalize_simplicial_content_logits(mode: str) -> str:
    mode = str(mode).lower().replace("-", "_")
    if mode in {"disabled", "off", "false", "no"}:
        return "off"
    if mode in {"enabled", "on", "true", "yes"}:
        return "on"
    if mode in {"gated", "learned_scale", "learnable", "learnable_scale"}:
        return "learned"
    if mode not in {"on", "off", "learned"}:
        raise ValueError("simplicial_content_logits must be one of {'on', 'off', 'learned'}")
    return mode


def _canonicalize_simplicial_rope_on_values(mode: str) -> str:
    mode = str(mode).lower().replace("-", "_")
    if mode in {"disabled", "off", "false", "no"}:
        return "none"
    if mode not in {"none", "carrier"}:
        raise ValueError("simplicial_rope_on_values must be one of {'none', 'carrier'}")
    return mode


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


class SimplicialClosedRoPE(nn.Module):
    """Constant-key closed simplicial RoPE as a low-rank logit residual."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        n_freqs: int = 16,
        freq_sigma: float = 1.0,
        learned_freqs: bool = False,
        gate: str = "none",
        gate_init: float = 0.0,
        logit_scale_init: float | None = None,
        value_n_freqs: int | None = None,
        value_scale_init: float = 1.0,
        enable_value_carrier: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.n_freqs = int(n_freqs)
        self.value_n_freqs = int(value_n_freqs) if value_n_freqs is not None else min(self.n_freqs, 16)
        self.value_carrier_enabled = bool(enable_value_carrier)
        self.freq_sigma = float(freq_sigma)
        self.gate = _canonicalize_simplicial_rope_gate(gate)
        if self.n_freqs <= 0:
            raise ValueError("simplicial_rope_n_freqs must be positive")
        if 2 * self.n_freqs > self.head_dim:
            raise ValueError(
                "closed simplicial RoPE uses two real query channels per frequency; "
                f"got n_freqs={self.n_freqs} and head_dim={self.head_dim}"
            )
        if self.value_carrier_enabled:
            if self.value_n_freqs <= 0:
                raise ValueError("simplicial_rope_value_n_freqs must be positive")
            if self.value_n_freqs > self.n_freqs:
                raise ValueError(
                    "simplicial_rope_value_n_freqs must be <= simplicial_rope_n_freqs; "
                    f"got {self.value_n_freqs} > {self.n_freqs}"
                )
            if 2 * self.value_n_freqs > self.head_dim:
                raise ValueError(
                    "closed simplicial RoPE value carrier uses two real value channels per frequency; "
                    f"got value_n_freqs={self.value_n_freqs} and head_dim={self.head_dim}"
                )

        mu = self._build_spiral_frequencies(
            num_heads=self.num_heads,
            num_freqs=self.n_freqs,
            freq_sigma=0.5 * self.freq_sigma,
            phase_offset=0.0,
        )
        nu = self._build_spiral_frequencies(
            num_heads=self.num_heads,
            num_freqs=self.n_freqs,
            freq_sigma=self.freq_sigma,
            phase_offset=math.pi / max(self.num_heads, 1),
        )
        if learned_freqs:
            self.mu = nn.Parameter(mu)
            self.nu = nn.Parameter(nu)
        else:
            self.register_buffer("mu", mu, persistent=False)
            self.register_buffer("nu", nu, persistent=False)

        if self.gate == "learned":
            self.gate_logit = nn.Parameter(torch.full((1, self.num_heads, 1), float(gate_init)))
        else:
            self.gate_logit = None
        if self.gate == "sigma":
            alpha_init = torch.ones((1, self.num_heads, 1), dtype=torch.float32)
            self.sigma_gate_alpha_raw = nn.Parameter(_softplus_inverse(alpha_init))
            self.sigma_gate_beta = nn.Parameter(torch.full((1, self.num_heads, 1), float(gate_init)))
        else:
            self.sigma_gate_alpha_raw = None
            self.sigma_gate_beta = None
        if logit_scale_init is None:
            logit_scale_init = math.sqrt(self.head_dim)
        if logit_scale_init <= 0:
            raise ValueError("simplicial_rope_logit_scale_init must be positive")
        self.logit_scale = nn.Parameter(
            torch.full((1, self.num_heads, 1, 1, 1), math.log(float(logit_scale_init)), dtype=torch.float32)
        )
        if self.value_carrier_enabled:
            if value_scale_init <= 0:
                raise ValueError("simplicial_rope_value_scale_init must be positive")
            self.value_log_scale = nn.Parameter(
                torch.full((self.num_heads, 1, 1), math.log(float(value_scale_init)), dtype=torch.float32)
            )
            self.register_buffer("value_carrier_basis", self._build_value_carrier_basis(), persistent=False)
        else:
            self.value_log_scale = None
            self.register_buffer("value_carrier_basis", torch.empty(0, dtype=torch.float32), persistent=False)

    def _build_value_carrier_basis(self) -> torch.Tensor:
        value_rank = 4 * self.value_n_freqs
        basis = torch.zeros(self.num_heads, value_rank, self.head_dim, dtype=torch.float32)
        scale = math.sqrt(value_rank)
        for freq_idx in range(self.value_n_freqs):
            base = 4 * freq_idx
            real_dim = 2 * freq_idx
            imag_dim = real_dim + 1
            basis[:, base + 0, real_dim] = scale
            basis[:, base + 1, real_dim] = -scale
            basis[:, base + 2, imag_dim] = scale
            basis[:, base + 3, imag_dim] = scale
        return basis

    @staticmethod
    def _build_spiral_frequencies(
        *,
        num_heads: int,
        num_freqs: int,
        freq_sigma: float,
        phase_offset: float,
    ) -> torch.Tensor:
        indices = torch.arange(num_freqs, dtype=torch.float32) + 0.5
        magnitudes = torch.linspace(
            freq_sigma / max(num_freqs, 1),
            freq_sigma,
            num_freqs,
            dtype=torch.float32,
        )
        head_phases = torch.linspace(0.0, 2.0 * math.pi, num_heads + 1, dtype=torch.float32)[:-1]
        head_phases = head_phases[:, None] + float(phase_offset)

        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        y = (1.0 - 2.0 * indices / float(num_freqs)).clamp(min=-1.0, max=1.0)
        radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
        theta = (2.0 * math.pi * indices / golden_ratio)[None, :] + head_phases
        x = radius[None, :] * torch.cos(theta)
        z = radius[None, :] * torch.sin(theta)
        directions = torch.stack([x, y[None, :].expand(num_heads, -1), z], dim=-1)
        return directions * magnitudes.view(1, num_freqs, 1)

    def _gate(
        self,
        *,
        batch_size: int,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        sigma: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gate == "none":
            return torch.ones((batch_size, self.num_heads, num_tokens), device=device, dtype=dtype)
        if self.gate == "learned":
            assert self.gate_logit is not None
            return torch.sigmoid(self.gate_logit).to(device=device, dtype=dtype).expand(batch_size, -1, num_tokens)
        if sigma is None:
            raise ValueError("sigma must be provided when simplicial_rope_gate='sigma'")
        if sigma.ndim == 2 and sigma.shape[-1] == 1:
            sigma = sigma[:, 0]
        if sigma.ndim != 1:
            raise ValueError(f"sigma must have shape (B,) or (B, 1), got {tuple(sigma.shape)}")
        if sigma.shape[0] == 1 and batch_size > 1:
            sigma = sigma.expand(batch_size)
        if sigma.shape[0] != batch_size:
            raise ValueError(f"sigma batch {sigma.shape[0]} does not match batch size {batch_size}")
        assert self.sigma_gate_alpha_raw is not None
        assert self.sigma_gate_beta is not None
        sigma_feat = -torch.log(sigma.to(device=device, dtype=torch.float32).clamp_min(1e-8))
        alpha = F.softplus(self.sigma_gate_alpha_raw).to(device=device, dtype=torch.float32)
        beta = self.sigma_gate_beta.to(device=device, dtype=torch.float32)
        gate = torch.sigmoid(alpha * sigma_feat[:, None, None] + beta)
        return gate.to(dtype=dtype).expand(-1, -1, num_tokens)

    def forward(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        *,
        sigma: torch.Tensor | None = None,
        position_query_mask: torch.Tensor | None = None,
    ) -> SimplicialLowRankAngleResidual:
        if q.ndim != 4:
            raise ValueError(f"q must have shape (B, H, T, D), got {tuple(q.shape)}")
        batch_size, num_heads, num_tokens, head_dim = q.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"q shape {(num_heads, head_dim)} does not match closed RoPE "
                f"{(self.num_heads, self.head_dim)}"
            )
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"simplicial RoPE positions must have shape (B, T, 3), got {tuple(positions.shape)}")
        if positions.shape[:2] != (batch_size, num_tokens):
            raise ValueError(
                f"simplicial RoPE positions shape {tuple(positions.shape[:2])} must match "
                f"{(batch_size, num_tokens)}"
            )
        if position_query_mask is not None and position_query_mask.shape != (batch_size, num_tokens):
            raise ValueError(
                f"position_query_mask must have shape {(batch_size, num_tokens)}, "
                f"got {tuple(position_query_mask.shape)}"
            )

        work_dtype = torch.float32
        q_pairs = q[..., : 2 * self.n_freqs].to(dtype=work_dtype).view(
            batch_size,
            num_heads,
            num_tokens,
            self.n_freqs,
            2,
        )
        q_r, q_i = q_pairs.unbind(dim=-1)

        mu = self.mu.to(device=q.device, dtype=work_dtype)
        nu = self.nu.to(device=q.device, dtype=work_dtype)
        a = 0.5 * (mu + nu)
        b = 0.5 * (mu - nu)

        positions_work = positions.to(device=q.device, dtype=work_dtype)
        rel = positions_work[:, None, :, :] - positions_work[:, :, None, :]
        phi = torch.einsum("bqjd,hmd->bhqjm", rel, a)
        psi = torch.einsum("bqkd,hmd->bhqkm", rel, b)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        q_r = q_r.unsqueeze(3)
        q_i = q_i.unsqueeze(3)
        left_even = q_r * cos_phi + q_i * sin_phi
        left_odd = -q_r * sin_phi + q_i * cos_phi
        right_even = cos_psi
        right_odd = sin_psi

        rank = 2 * self.n_freqs
        left = math.sqrt(2.0) * torch.stack((left_even, left_odd), dim=-1).reshape(
            batch_size,
            num_heads,
            num_tokens,
            num_tokens,
            rank,
        )
        left = left * self.logit_scale.to(device=q.device, dtype=left.dtype).exp()
        right = torch.stack((right_even, right_odd), dim=-1).reshape(
            batch_size,
            num_heads,
            num_tokens,
            num_tokens,
            rank,
        )
        gate = self._gate(
            batch_size=batch_size,
            num_tokens=num_tokens,
            dtype=left.dtype,
            device=q.device,
            sigma=sigma,
        )
        if position_query_mask is not None:
            gate = gate * position_query_mask.to(device=q.device, dtype=gate.dtype)[:, None, :]
        return SimplicialLowRankAngleResidual(
            left=left.contiguous(),
            right=right.contiguous(),
            gate=gate.contiguous(),
        )

    def build_value_carrier(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        positions: torch.Tensor,
        *,
        position_query_mask: torch.Tensor | None = None,
    ) -> tuple[SimplicialLowRankMessageResidual, torch.Tensor]:
        if not self.value_carrier_enabled or self.value_log_scale is None:
            raise ValueError("simplicial RoPE value carrier is not enabled")
        if v1.shape != v2.shape:
            raise ValueError(f"v1 and v2 must have the same shape, got {tuple(v1.shape)} and {tuple(v2.shape)}")
        if v1.ndim != 4:
            raise ValueError(f"v1/v2 must have shape (B, H, T, D), got {tuple(v1.shape)}")
        batch_size, num_heads, num_tokens, head_dim = v1.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"value shape {(num_heads, head_dim)} does not match closed RoPE "
                f"{(self.num_heads, self.head_dim)}"
            )
        if positions.ndim != 3 or positions.shape != (batch_size, num_tokens, 3):
            raise ValueError(f"positions must have shape {(batch_size, num_tokens, 3)}, got {tuple(positions.shape)}")
        if position_query_mask is not None and position_query_mask.shape != (batch_size, num_tokens):
            raise ValueError(
                f"position_query_mask must have shape {(batch_size, num_tokens)}, "
                f"got {tuple(position_query_mask.shape)}"
            )

        work_dtype = torch.float32
        value_n_freqs = self.value_n_freqs
        value_rank = 4 * value_n_freqs
        v1_pairs = v1[..., : 2 * value_n_freqs].to(dtype=work_dtype).view(
            batch_size,
            num_heads,
            num_tokens,
            value_n_freqs,
            2,
        )
        v2_pairs = v2[..., : 2 * value_n_freqs].to(dtype=work_dtype).view(
            batch_size,
            num_heads,
            num_tokens,
            value_n_freqs,
            2,
        )
        v1_r, v1_i = v1_pairs.unbind(dim=-1)
        v2_r, v2_i = v2_pairs.unbind(dim=-1)

        mu = self.mu[:, :value_n_freqs].to(device=v1.device, dtype=work_dtype)
        nu = self.nu[:, :value_n_freqs].to(device=v1.device, dtype=work_dtype)
        a = 0.5 * (mu + nu)
        b = 0.5 * (mu - nu)

        positions_work = positions.to(device=v1.device, dtype=work_dtype)
        rel = positions_work[:, None, :, :] - positions_work[:, :, None, :]
        phi = torch.einsum("bqjd,hmd->bhqjm", rel, a)
        psi = torch.einsum("bqkd,hmd->bhqkm", rel, b)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        v1_r = v1_r.unsqueeze(2)
        v1_i = v1_i.unsqueeze(2)
        v2_r = v2_r.unsqueeze(2)
        v2_i = v2_i.unsqueeze(2)
        a_r = v1_r * cos_phi - v1_i * sin_phi
        a_i = v1_r * sin_phi + v1_i * cos_phi
        b_r = v2_r * cos_psi - v2_i * sin_psi
        b_i = v2_r * sin_psi + v2_i * cos_psi

        left = torch.empty(batch_size, num_heads, num_tokens, num_tokens, value_rank, device=v1.device, dtype=work_dtype)
        right = torch.empty_like(left)
        left[..., 0::4] = a_r
        right[..., 0::4] = b_r
        left[..., 1::4] = a_i
        right[..., 1::4] = b_i
        left[..., 2::4] = a_r
        right[..., 2::4] = b_i
        left[..., 3::4] = a_i
        right[..., 3::4] = b_r
        if position_query_mask is not None:
            left = left * position_query_mask.to(device=v1.device, dtype=work_dtype)[:, None, :, None, None]

        basis = self.value_carrier_basis.to(device=v1.device, dtype=work_dtype)
        basis = basis * self.value_log_scale.to(device=v1.device, dtype=work_dtype).exp()
        return (
            SimplicialLowRankMessageResidual(left=left.contiguous(), right=right.contiguous()),
            basis.contiguous(),
        )


def _merge_angle_residuals(
    *residuals: SimplicialLowRankAngleResidual | None,
) -> SimplicialLowRankAngleResidual | None:
    active = [residual for residual in residuals if residual is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    total_rank = sum(residual.rank for residual in active)
    common_dtype = active[0].left.dtype
    for residual in active:
        common_dtype = torch.promote_types(common_dtype, residual.left.dtype)
        common_dtype = torch.promote_types(common_dtype, residual.right.dtype)
        common_dtype = torch.promote_types(common_dtype, residual.gate.dtype)

    left_parts = []
    right_parts = []
    for residual in active:
        scale = math.sqrt(total_rank / residual.rank)
        left = residual.left.to(dtype=common_dtype)
        right = residual.right.to(dtype=common_dtype)
        gate = residual.gate.to(dtype=common_dtype)
        left_parts.append(
            left
            * gate.unsqueeze(-1).unsqueeze(-1)
            * scale
        )
        right_parts.append(right)
    gate = torch.ones_like(active[0].gate, dtype=common_dtype)
    return SimplicialLowRankAngleResidual(
        left=torch.cat(left_parts, dim=-1).contiguous(),
        right=torch.cat(right_parts, dim=-1).contiguous(),
        gate=gate,
    )


def _merge_message_residuals(
    *branches: tuple[SimplicialLowRankMessageResidual, torch.Tensor] | None,
) -> tuple[SimplicialLowRankMessageResidual | None, torch.Tensor | None]:
    active = [branch for branch in branches if branch is not None]
    if not active:
        return None, None
    if len(active) == 1:
        return active[0]

    batch_size, num_heads, num_tokens, _, _ = active[0][0].left.shape
    head_dim = active[0][1].shape[-1]
    total_rank = sum(residual.rank for residual, _ in active)
    common_dtype = active[0][0].left.dtype
    for residual, basis in active:
        residual.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
        if basis.shape != (num_heads, residual.rank, head_dim):
            raise ValueError(
                f"message basis must have shape {(num_heads, residual.rank, head_dim)}, "
                f"got {tuple(basis.shape)}"
            )
        common_dtype = torch.promote_types(common_dtype, residual.left.dtype)
        common_dtype = torch.promote_types(common_dtype, residual.right.dtype)
        common_dtype = torch.promote_types(common_dtype, basis.dtype)

    left_parts = []
    right_parts = []
    basis_parts = []
    for residual, basis in active:
        scale = math.sqrt(total_rank / residual.rank)
        left_parts.append(residual.left.to(dtype=common_dtype) * scale)
        right_parts.append(residual.right.to(dtype=common_dtype))
        basis_parts.append(basis.to(dtype=common_dtype))
    return (
        SimplicialLowRankMessageResidual(
            left=torch.cat(left_parts, dim=-1).contiguous(),
            right=torch.cat(right_parts, dim=-1).contiguous(),
        ),
        torch.cat(basis_parts, dim=1).contiguous(),
    )


def _materialize_chunk_logits(
    q_chunk: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    *,
    factorized_bias: SimplicialFactorizedBias | None,
    angle_residual: SimplicialLowRankAngleResidual | None,
    logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None,
    start: int,
    end: int,
) -> torch.Tensor:
    qk2 = q_chunk.unsqueeze(-2) * k2.unsqueeze(-3)
    logits = torch.matmul(k1.unsqueeze(-3), qk2.transpose(-1, -2)).float()
    if factorized_bias is not None:
        logits = logits + factorized_bias.chunk(start, end, dtype=logits.dtype, device=logits.device)
    if angle_residual is not None:
        logits = logits + angle_residual.chunk(start, end, dtype=logits.dtype, device=logits.device)
    if logit_bias_fn is not None:
        logits = logits + logit_bias_fn(start, end, logits.dtype, logits.device)
    return logits


def _coerce_attention_mask(
    attention_mask: SimplicialAttentionMask | None,
    *,
    key_padding_mask: torch.Tensor | None,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> SimplicialAttentionMask:
    if attention_mask is None:
        attention_mask = SimplicialAttentionMask.from_key_padding_mask(
            key_padding_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=device,
        )
    attention_mask.validate(batch_size=batch_size, num_tokens=num_tokens)
    return SimplicialAttentionMask(
        query_valid=attention_mask.query_valid.bool().to(device=device),
        pair_key_valid=attention_mask.pair_key_valid.bool().to(device=device),
        pair_valid=attention_mask.pair_valid.bool().to(device=device) if attention_mask.pair_valid is not None else None,
    )


def simplicial_attention_torch_from_projected(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    key_padding_mask: torch.Tensor | None = None,
    attention_mask: SimplicialAttentionMask | None = None,
    factorized_bias: SimplicialFactorizedBias | None = None,
    angle_residual: SimplicialLowRankAngleResidual | None = None,
    message_residual: SimplicialLowRankMessageResidual | None = None,
    message_basis: torch.Tensor | None = None,
    logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    chunk_size: int = 128,
    return_attn: bool = False,
    fp32_core: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, num_tokens, head_dim = q.shape
    if factorized_bias is not None:
        factorized_bias.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
    if angle_residual is not None:
        angle_residual.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
    if message_residual is not None:
        message_residual.validate(batch_size=batch_size, num_heads=num_heads, num_tokens=num_tokens)
        if message_basis is None:
            raise ValueError("message_basis must be provided when message_residual is provided")
        expected_basis_shape = (num_heads, message_residual.rank, head_dim)
        if message_basis.shape != expected_basis_shape:
            raise ValueError(
                f"message_basis must have shape {expected_basis_shape}, got {tuple(message_basis.shape)}"
            )
    elif message_basis is not None:
        raise ValueError("message_basis must be None when message_residual is None")

    attention_mask = _coerce_attention_mask(
        attention_mask,
        key_padding_mask=key_padding_mask,
        batch_size=batch_size,
        num_tokens=num_tokens,
        device=q.device,
    )
    query_valid = attention_mask.query_valid
    pair_valid = attention_mask.pair_mask()

    compute_in_fp32 = fp32_core and q.dtype in (torch.float16, torch.bfloat16)
    if compute_in_fp32:
        q_work = q.float()
        k1_work = k1.float()
        v1_work = v1.float()
        k2_work = k2.float()
        v2_work = v2.float()
        factorized_bias_work = (
            SimplicialFactorizedBias(
                u=factorized_bias.u.float(),
                v=factorized_bias.v.float(),
                w=factorized_bias.w.float(),
                gate=factorized_bias.gate.float(),
            )
            if factorized_bias is not None
            else None
        )
        angle_residual_work = (
            SimplicialLowRankAngleResidual(
                left=angle_residual.left.float(),
                right=angle_residual.right.float(),
                gate=angle_residual.gate.float(),
            )
            if angle_residual is not None
            else None
        )
        message_residual_work = (
            SimplicialLowRankMessageResidual(
                left=message_residual.left.float(),
                right=message_residual.right.float(),
            )
            if message_residual is not None
            else None
        )
        message_basis_work = message_basis.float() if message_basis is not None else None
        out_dtype = torch.float32
    else:
        q_work = q
        k1_work = k1
        v1_work = v1
        k2_work = k2
        v2_work = v2
        factorized_bias_work = factorized_bias
        angle_residual_work = angle_residual
        message_residual_work = message_residual
        message_basis_work = message_basis
        out_dtype = v1.dtype

    out = torch.empty((batch_size, num_heads, num_tokens, head_dim), device=q.device, dtype=out_dtype)
    attn_chunks: list[torch.Tensor] | None = [] if return_attn else None
    pair_mask = pair_valid[:, None, None, :, :]
    flat_mask = pair_mask.flatten(-2)

    for start in range(0, num_tokens, chunk_size):
        end = min(num_tokens, start + chunk_size)
        q_chunk = q_work[:, :, start:end, :]
        logits = _materialize_chunk_logits(
            q_chunk,
            k1_work,
            k2_work,
            factorized_bias=factorized_bias_work,
            angle_residual=angle_residual_work,
            logit_bias_fn=logit_bias_fn,
            start=start,
            end=end,
        )

        flat_logits = logits.flatten(-2).masked_fill(
            ~flat_mask,
            torch.finfo(logits.dtype).min,
        )
        attn = torch.softmax(flat_logits, dim=-1)
        attn = torch.where(flat_mask, attn, torch.zeros_like(attn)).view_as(logits)

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=training)

        if return_attn and attn_chunks is not None:
            attn_chunks.append(attn.detach())

        attn_float = attn.float()
        attn_value = attn.to(v1_work.dtype)
        tmp = torch.matmul(attn_value.transpose(-2, -1), v1_work.unsqueeze(-3))
        out_chunk = (tmp * v2_work.unsqueeze(-3)).sum(dim=-2)
        if message_residual_work is not None and message_basis_work is not None:
            message_left = message_residual_work.left[:, :, start:end, :, :].float()
            message_right = message_residual_work.right[:, :, start:end, :, :].float()
            message_coeff = torch.einsum(
                "bhqjk,bhqjr,bhqkr->bhqr",
                attn_float,
                message_left,
                message_right,
            ) * (message_residual_work.rank**-0.5)
            message_out = torch.einsum("bhqr,hrd->bhqd", message_coeff, message_basis_work.float())
            out_chunk = out_chunk + message_out.to(dtype=out_chunk.dtype)
        q_valid = query_valid[:, start:end].to(dtype=out_chunk.dtype)
        out_chunk = out_chunk * q_valid[:, None, :, None]
        out[:, :, start:end, :] = out_chunk

    if return_attn and attn_chunks is not None:
        return out.to(v1.dtype), torch.cat(attn_chunks, dim=2)
    return out.to(v1.dtype)


class _TritonTwoSimplicialAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        k2: torch.Tensor,
        v2: torch.Tensor,
        query_valid: torch.Tensor,
        pair_key_valid: torch.Tensor,
        pair_valid: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        gate: torch.Tensor,
        angle_left: torch.Tensor,
        angle_right: torch.Tensor,
        angle_gate: torch.Tensor,
        message_left: torch.Tensor,
        message_right: torch.Tensor,
        message_basis: torch.Tensor,
        precision_mode: str,
        chunk_size: int,
        debug_torch_backward: bool,
    ) -> torch.Tensor:
        has_pair_valid = pair_valid.numel() > 0
        has_factorized_bias = u.numel() > 0
        has_low_rank_angle = angle_left.numel() > 0
        has_low_rank_message = message_left.numel() > 0
        out, lse, out_for_backward = triton_simplicial_attention_forward(
            q,
            k1,
            v1,
            k2,
            v2,
            query_valid=query_valid,
            pair_key_valid=pair_key_valid,
            pair_valid=pair_valid if has_pair_valid else None,
            u=u if has_factorized_bias else None,
            v_bias=v if has_factorized_bias else None,
            w=w if has_factorized_bias else None,
            gate=gate if has_factorized_bias else None,
            angle_left=angle_left if has_low_rank_angle else None,
            angle_right=angle_right if has_low_rank_angle else None,
            angle_gate=angle_gate if has_low_rank_angle else None,
            message_left=message_left if has_low_rank_message else None,
            message_right=message_right if has_low_rank_message else None,
            message_basis=message_basis if has_low_rank_message else None,
            precision=precision_mode,
        )
        ctx.precision_mode = str(precision_mode)
        ctx.chunk_size = int(chunk_size)
        ctx.debug_torch_backward = bool(debug_torch_backward)
        ctx.has_pair_valid = has_pair_valid
        ctx.has_factorized_bias = has_factorized_bias
        ctx.has_low_rank_angle = has_low_rank_angle
        ctx.has_low_rank_message = has_low_rank_message
        ctx.save_for_backward(
            q,
            k1,
            v1,
            k2,
            v2,
            query_valid,
            pair_key_valid,
            pair_valid,
            u,
            v,
            w,
            gate,
            angle_left,
            angle_right,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out_for_backward,
            lse,
        )
        return out

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            q,
            k1,
            v1,
            k2,
            v2,
            query_valid,
            pair_key_valid,
            pair_valid,
            u,
            v,
            w,
            gate,
            angle_left,
            angle_right,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out,
            lse,
        ) = ctx.saved_tensors

        if ctx.debug_torch_backward:
            differentiable_inputs = [
                q.detach().requires_grad_(True),
                k1.detach().requires_grad_(True),
                v1.detach().requires_grad_(True),
                k2.detach().requires_grad_(True),
                v2.detach().requires_grad_(True),
            ]
            if ctx.has_factorized_bias:
                differentiable_inputs.extend(
                    [
                        u.detach().requires_grad_(True),
                        v.detach().requires_grad_(True),
                        w.detach().requires_grad_(True),
                        gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_low_rank_angle:
                differentiable_inputs.extend(
                    [
                        angle_left.detach().requires_grad_(True),
                        angle_right.detach().requires_grad_(True),
                        angle_gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_low_rank_message:
                differentiable_inputs.extend(
                    [
                        message_left.detach().requires_grad_(True),
                        message_right.detach().requires_grad_(True),
                        message_basis.detach().requires_grad_(True),
                    ]
                )

            with torch.enable_grad():
                q_re, k1_re, v1_re, k2_re, v2_re = differentiable_inputs[:5]
                factorized_bias = None
                cursor = 5
                if ctx.has_factorized_bias:
                    u_re, v_re, w_re, gate_re = differentiable_inputs[cursor : cursor + 4]
                    cursor += 4
                    factorized_bias = SimplicialFactorizedBias(
                        u=u_re,
                        v=v_re,
                        w=w_re,
                        gate=gate_re,
                    )
                angle_residual = None
                if ctx.has_low_rank_angle:
                    angle_left_re, angle_right_re, angle_gate_re = differentiable_inputs[cursor : cursor + 3]
                    cursor += 3
                    angle_residual = SimplicialLowRankAngleResidual(
                        left=angle_left_re,
                        right=angle_right_re,
                        gate=angle_gate_re,
                    )
                message_residual = None
                message_basis_re = None
                if ctx.has_low_rank_message:
                    message_left_re, message_right_re, message_basis_re = differentiable_inputs[cursor : cursor + 3]
                    message_residual = SimplicialLowRankMessageResidual(
                        left=message_left_re,
                        right=message_right_re,
                    )
                attention_mask = SimplicialAttentionMask(
                    query_valid=query_valid,
                    pair_key_valid=pair_key_valid,
                    pair_valid=pair_valid if ctx.has_pair_valid else None,
                )
                reference_out = simplicial_attention_torch_from_projected(
                    q_re,
                    k1_re,
                    v1_re,
                    k2_re,
                    v2_re,
                    attention_mask=attention_mask,
                    factorized_bias=factorized_bias,
                    angle_residual=angle_residual,
                    message_residual=message_residual,
                    message_basis=message_basis_re,
                    logit_bias_fn=None,
                    dropout_p=0.0,
                    training=False,
                    chunk_size=ctx.chunk_size,
                    return_attn=False,
                    fp32_core=True,
                )
                grads = torch.autograd.grad(
                    reference_out,
                    differentiable_inputs,
                    grad_out,
                    allow_unused=True,
                )

            dq, dk1, dv1, dk2, dv2 = grads[:5]
            du = dv = dw = dgate = None
            dangle_left = dangle_right = dangle_gate = None
            dmessage_left = dmessage_right = dmessage_basis = None
            cursor = 5
            if ctx.has_factorized_bias:
                du, dv, dw, dgate = grads[cursor : cursor + 4]
                cursor += 4
            if ctx.has_low_rank_angle:
                dangle_left, dangle_right, dangle_gate = grads[cursor : cursor + 3]
                cursor += 3
            if ctx.has_low_rank_message:
                dmessage_left, dmessage_right, dmessage_basis = grads[cursor : cursor + 3]
            return (
                dq,
                dk1,
                dv1,
                dk2,
                dv2,
                None,
                None,
                None,
                du,
                dv,
                dw,
                dgate,
                dangle_left,
                dangle_right,
                dangle_gate,
                dmessage_left,
                dmessage_right,
                dmessage_basis,
                None,
                None,
                None,
            )

        need_du = bool(ctx.has_factorized_bias and ctx.needs_input_grad[8])
        need_dv_bias = bool(ctx.has_factorized_bias and ctx.needs_input_grad[9])
        need_dw = bool(ctx.has_factorized_bias and ctx.needs_input_grad[10])
        need_dgate = bool(ctx.has_factorized_bias and ctx.needs_input_grad[11])
        need_dangle_left = bool(ctx.has_low_rank_angle and ctx.needs_input_grad[12])
        need_dangle_right = bool(ctx.has_low_rank_angle and ctx.needs_input_grad[13])
        need_dangle_gate = bool(ctx.has_low_rank_angle and ctx.needs_input_grad[14])
        need_dmessage_left = bool(ctx.has_low_rank_message and ctx.needs_input_grad[15])
        need_dmessage_right = bool(ctx.has_low_rank_message and ctx.needs_input_grad[16])
        need_dmessage_basis = bool(ctx.has_low_rank_message and ctx.needs_input_grad[17])

        (
            dq,
            dk1,
            dv1,
            dk2,
            dv2,
            du,
            dv_bias,
            dw,
            dgate,
            dangle_left,
            dangle_right,
            dangle_gate,
            dmessage_left,
            dmessage_right,
            dmessage_basis,
        ) = triton_simplicial_attention_backward(
            grad_out,
            q,
            k1,
            v1,
            k2,
            v2,
            out,
            lse,
            query_valid=query_valid,
            pair_key_valid=pair_key_valid,
            pair_valid=pair_valid if ctx.has_pair_valid else None,
            u=u if ctx.has_factorized_bias else None,
            v_bias=v if ctx.has_factorized_bias else None,
            w=w if ctx.has_factorized_bias else None,
            gate=gate if ctx.has_factorized_bias else None,
            angle_left=angle_left if ctx.has_low_rank_angle else None,
            angle_right=angle_right if ctx.has_low_rank_angle else None,
            angle_gate=angle_gate if ctx.has_low_rank_angle else None,
            message_left=message_left if ctx.has_low_rank_message else None,
            message_right=message_right if ctx.has_low_rank_message else None,
            message_basis=message_basis if ctx.has_low_rank_message else None,
            precision=ctx.precision_mode,
            need_du=need_du,
            need_dv_bias=need_dv_bias,
            need_dw=need_dw,
            need_dgate=need_dgate,
            need_dangle_left=need_dangle_left,
            need_dangle_right=need_dangle_right,
            need_dangle_gate=need_dangle_gate,
            need_dmessage_left=need_dmessage_left,
            need_dmessage_right=need_dmessage_right,
            need_dmessage_basis=need_dmessage_basis,
        )
        return (
            dq,
            dk1,
            dv1,
            dk2,
            dv2,
            None,
            None,
            None,
            du,
            dv_bias,
            dw,
            dgate,
            dangle_left,
            dangle_right,
            dangle_gate,
            dmessage_left,
            dmessage_right,
            dmessage_basis,
            None,
            None,
            None,
        )


class TwoSimplicialAttention(nn.Module):
    """Dense non-causal 2-simplicial attention with explicit torch/Triton backends."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        chunk_size: int = 128,
        out_proj: bool = True,
        impl: str = "auto",
        precision: str = "bf16_tc",
        message_mode: str = "none",
        message_rank: int = 16,
        position_mode: str = "none",
        rope_key_mode: str = "constant",
        rope_n_freqs: int = 16,
        rope_freq_sigma: float = 1.0,
        rope_learned_freqs: bool = False,
        rope_gate: str = "none",
        rope_value_n_freqs: int | None = None,
        rope_value_scale_init: float = 1.0,
        rope_on_values: str = "none",
        content_logits: str = "on",
        debug_torch_backward: bool = False,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_heads ({num_heads}) when head_dim is None."
                )
            head_dim = dim // num_heads

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        impl = impl.lower()
        if impl not in {"auto", "torch", "triton"}:
            raise ValueError(f"Unsupported simplicial_impl: {impl}")
        message_mode = message_mode.lower()
        if message_mode not in {"none", "low_rank"}:
            raise ValueError(f"Unsupported simplicial_message_mode: {message_mode}")
        message_rank = int(message_rank)
        if message_mode == "low_rank" and message_rank <= 0:
            raise ValueError("simplicial_message_rank must be positive when message_mode='low_rank'")
        position_mode = _canonicalize_simplicial_position_mode(position_mode)
        rope_key_mode = str(rope_key_mode).lower().replace("-", "_")
        if rope_key_mode != "constant":
            raise ValueError("Only simplicial_rope_key_mode='constant' is implemented in the first closed-RoPE version")
        rope_on_values = _canonicalize_simplicial_rope_on_values(rope_on_values)
        if rope_on_values == "carrier" and position_mode != "closed_rope":
            raise ValueError("simplicial_rope_on_values='carrier' requires simplicial_position_mode='closed_rope'")
        content_logits = _canonicalize_simplicial_content_logits(content_logits)

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.chunk_size = int(chunk_size)
        self.impl = impl
        self.precision = normalize_simplicial_precision(precision)
        self.message_mode = message_mode
        self.message_rank = message_rank
        self.position_mode = position_mode
        self.rope_key_mode = rope_key_mode
        self.rope_on_values = rope_on_values
        self.content_logits = content_logits
        self.debug_torch_backward = bool(debug_torch_backward)
        self.scale = self.head_dim**-0.5
        self._last_impl_used = "torch"

        self.in_proj = nn.Linear(self.dim, 5 * self.inner_dim, bias=bias)
        self.out_proj = (
            nn.Linear(self.inner_dim, self.dim, bias=bias) if out_proj else nn.Identity()
        )
        if self.message_mode == "low_rank":
            self.message_basis = nn.Parameter(
                torch.empty(self.num_heads, self.message_rank, self.head_dim)
            )
            nn.init.normal_(self.message_basis, mean=0.0, std=self.head_dim**-0.5)
        else:
            self.message_basis = None
        self.content_logit_log_scale = (
            nn.Parameter(torch.zeros((1, self.num_heads, 1, 1), dtype=torch.float32))
            if self.content_logits == "learned"
            else None
        )
        self.closed_rope = (
            SimplicialClosedRoPE(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                n_freqs=rope_n_freqs,
                freq_sigma=rope_freq_sigma,
                learned_freqs=rope_learned_freqs,
                gate=rope_gate,
                value_n_freqs=rope_value_n_freqs,
                value_scale_init=rope_value_scale_init,
                enable_value_carrier=rope_on_values == "carrier",
            )
            if self.position_mode == "closed_rope"
            else None
        )

    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        return x.view(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_tokens, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, num_tokens, num_heads * head_dim)

    def _project_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k1, v1, k2, v2 = self.in_proj(x).chunk(5, dim=-1)
        q = self._split_heads(q, self.num_heads, self.head_dim) * self.scale
        k1 = self._split_heads(k1, self.num_heads, self.head_dim)
        v1 = self._split_heads(v1, self.num_heads, self.head_dim)
        k2 = self._split_heads(k2, self.num_heads, self.head_dim)
        v2 = self._split_heads(v2, self.num_heads, self.head_dim)
        return q, k1, v1, k2, v2

    def _content_query(self, q: torch.Tensor) -> torch.Tensor:
        if self.content_logits == "on":
            return q
        if self.content_logits == "off":
            return torch.zeros_like(q)
        assert self.content_logit_log_scale is not None
        return q * self.content_logit_log_scale.to(device=q.device, dtype=q.dtype).exp()

    def _triton_unavailable_reason(
        self,
        *,
        x: torch.Tensor,
        factorized_bias: SimplicialFactorizedBias | None,
        angle_residual: SimplicialLowRankAngleResidual | None,
        message_residual: SimplicialLowRankMessageResidual | None,
        message_basis: torch.Tensor | None,
        logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None,
        return_attn: bool,
    ) -> str | None:
        if not TRITON_AVAILABLE:
            return "triton is not installed"
        if x.device.type != "cuda":
            return "the Triton backend requires CUDA tensors"
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return f"the Triton backend only supports float16/bfloat16/float32 inputs, got {x.dtype}"
        if self.training and self.dropout > 0.0:
            return "the Triton backend does not support attention dropout in training mode in v1"
        if return_attn:
            return "the Triton backend does not support return_attn=True in v1"
        if logit_bias_fn is not None:
            return "the Triton backend does not support arbitrary Python logit_bias_fn callbacks"
        if self.head_dim > 128:
            return f"the Triton backend only supports head_dim <= 128 in v1, got {self.head_dim}"
        if factorized_bias is not None:
            for name, tensor in (
                ("u", factorized_bias.u),
                ("v", factorized_bias.v),
                ("w", factorized_bias.w),
                ("gate", factorized_bias.gate),
            ):
                if tensor.device != x.device:
                    return f"factorized bias tensor {name} must be on the same CUDA device as the attention input"
        if angle_residual is not None:
            if (
                angle_residual.left.device != x.device
                or angle_residual.right.device != x.device
                or angle_residual.gate.device != x.device
            ):
                return "angle residual tensors must be on the same CUDA device as the attention input"
            if angle_residual.rank > 64:
                return f"the Triton backend only supports low-rank angle residual rank <= 64, got {angle_residual.rank}"
        if message_residual is not None:
            if message_basis is None:
                return "message residual tensors require a message_basis"
            if message_residual.left.device != x.device or message_residual.right.device != x.device:
                return "message residual tensors must be on the same CUDA device as the attention input"
            if message_basis.device != x.device:
                return "message basis tensor must be on the same CUDA device as the attention input"
            if message_basis.shape != (self.num_heads, message_residual.rank, self.head_dim):
                return (
                    "message basis tensor must have shape "
                    f"{(self.num_heads, message_residual.rank, self.head_dim)}, got {tuple(message_basis.shape)}"
                )
            if message_residual.rank > 64:
                return f"the Triton backend only supports low-rank message residual rank <= 64, got {message_residual.rank}"
        elif message_basis is not None:
            return "message_basis must be None when message_residual is None"
        return None

    def _triton_compute_dtype(self) -> torch.dtype:
        if self.precision == "bf16_tc":
            return torch.bfloat16
        return torch.float32

    def _cast_angle_residual_for_triton(
        self,
        angle_residual: SimplicialLowRankAngleResidual | None,
        *,
        dtype: torch.dtype,
    ) -> SimplicialLowRankAngleResidual | None:
        if angle_residual is None:
            return None
        return SimplicialLowRankAngleResidual(
            left=angle_residual.left.to(dtype=dtype),
            right=angle_residual.right.to(dtype=dtype),
            gate=angle_residual.gate.to(dtype=dtype),
        )

    def _cast_message_residual_for_triton(
        self,
        message_residual: SimplicialLowRankMessageResidual | None,
        *,
        dtype: torch.dtype,
    ) -> SimplicialLowRankMessageResidual | None:
        if message_residual is None:
            return None
        return SimplicialLowRankMessageResidual(
            left=message_residual.left.to(dtype=dtype),
            right=message_residual.right.to(dtype=dtype),
        )

    def _forward_triton(
        self,
        q: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        k2: torch.Tensor,
        v2: torch.Tensor,
        *,
        attention_mask: SimplicialAttentionMask,
        factorized_bias: SimplicialFactorizedBias | None,
        angle_residual: SimplicialLowRankAngleResidual | None,
        message_residual: SimplicialLowRankMessageResidual | None,
        message_basis: torch.Tensor | None,
    ) -> torch.Tensor:
        attention_mask = _coerce_attention_mask(
            attention_mask,
            key_padding_mask=None,
            batch_size=q.shape[0],
            num_tokens=q.shape[2],
            device=q.device,
        )

        compute_dtype = self._triton_compute_dtype()
        q = q.to(dtype=compute_dtype)
        k1 = k1.to(dtype=compute_dtype)
        v1 = v1.to(dtype=compute_dtype)
        k2 = k2.to(dtype=compute_dtype)
        v2 = v2.to(dtype=compute_dtype)
        angle_residual = self._cast_angle_residual_for_triton(angle_residual, dtype=torch.float32)
        message_residual = self._cast_message_residual_for_triton(message_residual, dtype=torch.float32)

        empty_float = q.new_empty(0)
        empty_bool = torch.empty(0, device=q.device, dtype=torch.bool)
        if factorized_bias is None:
            u = v = w = gate = empty_float
        else:
            factorized_bias.validate(
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                num_tokens=q.shape[2],
            )
            u, v, w, gate = factorized_bias.u, factorized_bias.v, factorized_bias.w, factorized_bias.gate
        if angle_residual is None:
            angle_left = angle_right = angle_gate = empty_float
        else:
            angle_residual.validate(
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                num_tokens=q.shape[2],
            )
            angle_left = angle_residual.left
            angle_right = angle_residual.right
            angle_gate = angle_residual.gate
        if message_residual is None:
            message_left = message_right = message_basis = empty_float
        else:
            if message_basis is None:
                raise RuntimeError("message_residual requires message_basis")
            message_residual.validate(
                batch_size=q.shape[0],
                num_heads=q.shape[1],
                num_tokens=q.shape[2],
            )
            expected_basis_shape = (q.shape[1], message_residual.rank, q.shape[-1])
            if message_basis.shape != expected_basis_shape:
                raise ValueError(
                    f"message_basis must have shape {expected_basis_shape}, got {tuple(message_basis.shape)}"
                )
            message_left = message_residual.left
            message_right = message_residual.right
            message_basis = message_basis.to(device=q.device, dtype=torch.float32)

        return _TritonTwoSimplicialAttentionFunction.apply(
            q.contiguous(),
            k1.contiguous(),
            v1.contiguous(),
            k2.contiguous(),
            v2.contiguous(),
            attention_mask.query_valid.contiguous(),
            attention_mask.pair_key_valid.contiguous(),
            attention_mask.pair_valid.contiguous() if attention_mask.pair_valid is not None else empty_bool,
            u.contiguous(),
            v.contiguous(),
            w.contiguous(),
            gate.contiguous(),
            angle_left.contiguous(),
            angle_right.contiguous(),
            angle_gate.contiguous(),
            message_left.contiguous(),
            message_right.contiguous(),
            message_basis.contiguous(),
            self.precision,
            self.chunk_size,
            self.debug_torch_backward,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: SimplicialAttentionMask | None = None,
        factorized_bias: SimplicialFactorizedBias | None = None,
        angle_residual: SimplicialLowRankAngleResidual | None = None,
        message_residual: SimplicialLowRankMessageResidual | None = None,
        logit_bias_fn: Callable[[int, int, torch.dtype, torch.device], torch.Tensor] | None = None,
        positions: torch.Tensor | None = None,
        position_query_mask: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, C), got {tuple(x.shape)}")

        batch_size, num_tokens, _ = x.shape
        attention_mask = _coerce_attention_mask(
            attention_mask,
            key_padding_mask=key_padding_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=x.device,
        )
        if factorized_bias is not None:
            factorized_bias.validate(
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_tokens=num_tokens,
            )
        if angle_residual is not None:
            angle_residual.validate(
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_tokens=num_tokens,
            )
        if message_residual is not None:
            if self.message_basis is None:
                raise ValueError("message_residual requires simplicial_message_mode='low_rank'")
            message_residual.validate(
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_tokens=num_tokens,
            )
            if message_residual.rank != self.message_rank:
                raise ValueError(
                    f"message_residual rank {message_residual.rank} does not match attention message_rank {self.message_rank}"
                )
        external_message_residual = message_residual
        external_message_basis = self.message_basis if external_message_residual is not None else None

        q, k1, v1, k2, v2 = self._project_inputs(x)
        rope_angle_residual = None
        carrier_message_residual = None
        carrier_message_basis = None
        if self.closed_rope is not None:
            if positions is None:
                raise ValueError("positions must be provided when simplicial_position_mode='closed_rope'")
            rope_angle_residual = self.closed_rope(
                q,
                positions,
                sigma=sigma,
                position_query_mask=position_query_mask,
            )
            if self.rope_on_values == "carrier":
                carrier_message_residual, carrier_message_basis = self.closed_rope.build_value_carrier(
                    v1,
                    v2,
                    positions,
                    position_query_mask=position_query_mask,
                )
                carrier_dim = 2 * self.closed_rope.value_n_freqs
                v1 = v1.clone()
                v2 = v2.clone()
                v1[..., :carrier_dim] = 0.0
                v2[..., :carrier_dim] = 0.0
        angle_residual = _merge_angle_residuals(angle_residual, rope_angle_residual)
        message_residual, message_basis = _merge_message_residuals(
            (external_message_residual, external_message_basis)
            if external_message_residual is not None and external_message_basis is not None
            else None,
            (carrier_message_residual, carrier_message_basis)
            if carrier_message_residual is not None and carrier_message_basis is not None
            else None,
        )
        q_content = self._content_query(q)

        reason = self._triton_unavailable_reason(
            x=x,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
            message_residual=message_residual,
            message_basis=message_basis,
            logit_bias_fn=logit_bias_fn,
            return_attn=return_attn,
        )
        if self.impl == "auto":
            impl_used = "triton" if reason is None else "torch"
        elif self.impl == "triton":
            if reason is not None:
                raise RuntimeError(f"Triton simplicial attention is unavailable: {reason}")
            impl_used = "triton"
        else:
            impl_used = "torch"
        self._last_impl_used = impl_used

        if impl_used == "triton":
            out = self._forward_triton(
                q_content,
                k1,
                v1,
                k2,
                v2,
                attention_mask=attention_mask,
                factorized_bias=factorized_bias,
                angle_residual=angle_residual,
                message_residual=message_residual,
                message_basis=message_basis,
            )
            y = self._merge_heads(out)
            return self.out_proj(y)

        out = simplicial_attention_torch_from_projected(
            q_content,
            k1,
            v1,
            k2,
            v2,
            attention_mask=attention_mask,
            factorized_bias=factorized_bias,
            angle_residual=angle_residual,
            message_residual=message_residual,
            message_basis=message_basis,
            logit_bias_fn=logit_bias_fn,
            dropout_p=self.dropout,
            training=self.training,
            chunk_size=self.chunk_size,
            return_attn=return_attn,
        )
        if return_attn:
            out_heads, attn = out
            y = self._merge_heads(out_heads)
            return self.out_proj(y), attn
        y = self._merge_heads(out)
        return self.out_proj(y)
