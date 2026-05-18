from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedKLocalBiasResult:
    """Bias contribution for fixed-K local atom attention.

    ``bias`` is added to attention logits and must have shape ``[N, H, K]``.
    ``mask`` is optional extra validity mask with shape ``[N, K]``; it is
    intersected with the neighbor mask supplied to the attention function.
    """

    bias: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ESENFixedKLocalAttentionFeatures:
    """Precomputed eSEN fixed-K edge features shared across heads/layers.

    These tensors are geometry-derived and intentionally detached.  They are a
    fast path for direct-force settings where fixed-K attention does not need
    gradients through distances/coordinates.
    """

    local_mask: torch.Tensor
    env_bias: torch.Tensor
    log_env: torch.Tensor
    rho_env: torch.Tensor
    type_base: torch.Tensor


@dataclass(frozen=True)
class FixedKLocalAttentionContext:
    """Prepared fixed-K neighbor tensors for one flat atom batch."""

    neighbor_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    dist: torch.Tensor
    rbf: torch.Tensor | None = None
    esen_features: ESENFixedKLocalAttentionFeatures | None = None


@dataclass(frozen=True)
class ESENFixedKLocalBiasView:
    """Non-owning eSEN fixed-K bias view over PlatonicAttention parameters."""

    rbf_weight: torch.Tensor
    centers: torch.Tensor
    gamma: torch.Tensor
    cutoff: float
    heads_per_frame: int
    type_bias: torch.Tensor | None = None
    diag_zero: bool = True
    envelope_in_score: bool = True

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        dist: torch.Tensor | None,
        rbf: torch.Tensor | None,
        atom_types: torch.Tensor | None,
    ) -> FixedKLocalBiasResult:
        del k, v
        if dist is None:
            raise ValueError("ESENFixedKLocalBiasView requires dist")
        dist = dist.to(device=q.device, dtype=q.dtype)
        neighbor_mask = neighbor_mask.to(device=q.device, dtype=torch.bool)
        neighbor_idx = neighbor_idx.to(device=q.device, dtype=torch.long).masked_fill(~neighbor_mask, 0)
        if rbf is None:
            centers = self.centers.to(device=q.device, dtype=q.dtype)
            gamma = self.gamma.to(device=q.device, dtype=q.dtype)
            rbf = torch.exp(-gamma * (dist[..., None] - centers.view(1, 1, -1)).square())
        else:
            rbf = rbf.to(device=q.device, dtype=q.dtype)
        if q.shape[1] % int(self.heads_per_frame) != 0:
            raise ValueError("heads_per_frame must divide the number of attention heads")

        same = neighbor_idx == torch.arange(q.shape[0], device=q.device)[:, None]
        local_mask = neighbor_mask & (dist < float(self.cutoff))
        local_mask = local_mask | (same & neighbor_mask)
        env = _c2_quintic_envelope(dist, float(self.cutoff))
        env = torch.where(neighbor_mask, env, torch.zeros_like(env))
        env_bias = env.masked_fill(same, 0.0) if bool(self.diag_zero) else env

        raw_subhead = torch.einsum(
            "nkr,sr->nks",
            rbf,
            self.rbf_weight.to(device=q.device, dtype=q.dtype),
        )
        if self.type_bias is not None:
            if atom_types is None:
                raise ValueError("type_bias requires atom_types")
            zmax = int(self.type_bias.shape[0]) - 1
            zi = atom_types.to(device=q.device).long().clamp(min=0, max=zmax)
            zj = zi.index_select(0, neighbor_idx.reshape(-1)).reshape_as(neighbor_idx)
            raw_subhead = raw_subhead + self.type_bias.to(device=q.device, dtype=q.dtype)[zi[:, None], zj]

        subhead_bias = env_bias[..., None] * raw_subhead
        head_subidx = torch.arange(q.shape[1], device=q.device) % int(self.heads_per_frame)
        bias = subhead_bias.index_select(dim=-1, index=head_subidx).permute(0, 2, 1).contiguous()
        if self.envelope_in_score:
            env_score = env.masked_fill(same & neighbor_mask, 1.0)
            bias = bias + env_score.clamp_min(1.0e-20).log().unsqueeze(1).to(dtype=bias.dtype)
        return FixedKLocalBiasResult(bias=bias, mask=local_mask)


class FixedKLocalBias(nn.Module):
    """Base class for replaceable fixed-K local attention biases."""

    def forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        dist: torch.Tensor | None,
        rbf: torch.Tensor | None,
        atom_types: torch.Tensor | None,
    ) -> FixedKLocalBiasResult:
        raise NotImplementedError


class NoFixedKLocalBias(FixedKLocalBias):
    """No-op bias module for fixed-K local attention."""

    def forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        dist: torch.Tensor | None,
        rbf: torch.Tensor | None,
        atom_types: torch.Tensor | None,
    ) -> FixedKLocalBiasResult:
        del k, v, neighbor_idx, dist, rbf, atom_types
        mask = neighbor_mask.to(device=q.device, dtype=torch.bool)
        return FixedKLocalBiasResult(
            bias=torch.zeros(
                (q.shape[0], q.shape[1], mask.shape[-1]),
                device=q.device,
                dtype=q.dtype,
            ),
            mask=mask,
        )


def _safe_masked_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    valid = torch.isfinite(scores)
    row_has_neighbor = valid.any(dim=dim, keepdim=True)
    scores_safe = torch.where(row_has_neighbor, scores, torch.zeros_like(scores))
    probs = torch.softmax(scores_safe, dim=dim)
    return torch.where(row_has_neighbor & valid, probs, torch.zeros_like(probs))


def _c2_quintic_envelope(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    x = (dist / float(cutoff)).clamp(min=0.0, max=1.0)
    x2 = x.square()
    x3 = x2 * x
    env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
    return torch.where(dist < float(cutoff), env, torch.zeros_like(env))


@torch.no_grad()
def prepare_esen_fixed_k_local_attention_features(
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    dist: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor | float,
    cutoff: float,
    heads_per_frame: int,
    atom_types: torch.Tensor | None = None,
    max_atomic_number: int | None = None,
    diag_zero: bool = True,
    envelope_in_score: bool = True,
) -> ESENFixedKLocalAttentionFeatures:
    """Precompute fixed-K eSEN bias features for the Triton fast path."""

    if neighbor_idx.ndim != 2 or neighbor_mask.shape != neighbor_idx.shape or dist.shape != neighbor_idx.shape:
        raise ValueError("neighbor_idx, neighbor_mask, and dist must have shape [N, K]")
    if int(heads_per_frame) <= 0:
        raise ValueError("heads_per_frame must be positive")
    if float(cutoff) <= 0.0:
        raise ValueError("cutoff must be positive")
    device = dist.device
    dtype = torch.float32
    mask = neighbor_mask.to(device=device, dtype=torch.bool)
    idx = neighbor_idx.to(device=device, dtype=torch.long).masked_fill(~mask, 0)
    dist_f = dist.to(device=device, dtype=dtype)
    centers_f = centers.to(device=device, dtype=dtype)
    gamma_f = torch.as_tensor(gamma, device=device, dtype=dtype).reshape(())
    if centers_f.ndim != 1:
        raise ValueError("centers must have shape [R]")

    same = idx == torch.arange(idx.shape[0], device=device, dtype=torch.long)[:, None]
    local_mask = mask & (dist_f < float(cutoff))
    local_mask = local_mask | (same & mask)

    env = _c2_quintic_envelope(dist_f, float(cutoff))
    env = torch.where(mask, env, torch.zeros_like(env))
    env_bias = env.masked_fill(same, 0.0) if bool(diag_zero) else env
    env_score = env.masked_fill(same & mask, 1.0)
    log_env = env_score.clamp_min(1.0e-20).log() if bool(envelope_in_score) else torch.zeros_like(env_score)
    log_env = torch.where(local_mask, log_env, torch.zeros_like(log_env))

    rho = torch.exp(-gamma_f * (dist_f[..., None] - centers_f.view(1, 1, -1)).square())
    rho_env = env_bias[..., None] * rho

    type_base = torch.zeros_like(idx, dtype=torch.long)
    if atom_types is not None:
        zmax = int(max_atomic_number) if max_atomic_number is not None else int(atom_types.max().item())
        zi = atom_types.to(device=device, dtype=torch.long).clamp(min=0, max=zmax)
        zj = zi.index_select(0, idx.reshape(-1)).reshape_as(idx)
        zdim = int(zmax) + 1
        type_base = (zi[:, None] * zdim + zj) * int(heads_per_frame)

    return ESENFixedKLocalAttentionFeatures(
        local_mask=local_mask.contiguous(),
        env_bias=env_bias.contiguous(),
        log_env=log_env.contiguous(),
        rho_env=rho_env.contiguous(),
        type_base=type_base.contiguous(),
    )


def _as_trainable_or_buffer(module: nn.Module, name: str, value: torch.Tensor, *, trainable: bool) -> None:
    tensor = torch.as_tensor(value).detach().clone()
    if trainable:
        module.register_parameter(name, nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)


class ESENEnvelopedRBFTypeFixedKBias(FixedKLocalBias):
    """eSEN-like smooth local RBF/type bias for fixed-K atom attention.

    The attention mask is still supplied by ``neighbor_mask``.  This module only
    computes the smooth local score terms:

    ``log(env_ij) + env_ij * (RBF_ij @ w_h + type_bias[Zi, Zj, h])``.
    """

    def __init__(
        self,
        *,
        rbf_weight: torch.Tensor,
        centers: torch.Tensor,
        gamma: torch.Tensor | float,
        cutoff: float,
        heads_per_frame: int,
        type_bias: torch.Tensor | None = None,
        diag_zero: bool = True,
        envelope_in_score: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if int(heads_per_frame) <= 0:
            raise ValueError("heads_per_frame must be positive")
        if float(cutoff) <= 0.0:
            raise ValueError("cutoff must be positive")
        rbf_weight = torch.as_tensor(rbf_weight)
        if rbf_weight.ndim != 2:
            raise ValueError(f"rbf_weight must have shape [heads_per_frame, R], got {tuple(rbf_weight.shape)}")
        if int(rbf_weight.shape[0]) != int(heads_per_frame):
            raise ValueError("rbf_weight first dimension must equal heads_per_frame")
        centers = torch.as_tensor(centers)
        if centers.ndim != 1 or int(centers.shape[0]) != int(rbf_weight.shape[1]):
            raise ValueError("centers must have shape [R] matching rbf_weight")
        gamma = torch.as_tensor(gamma, dtype=rbf_weight.dtype)
        if gamma.numel() != 1:
            raise ValueError("gamma must be scalar")
        if type_bias is not None:
            type_bias = torch.as_tensor(type_bias)
            if type_bias.ndim != 3 or int(type_bias.shape[-1]) != int(heads_per_frame):
                raise ValueError("type_bias must have shape [Z, Z, heads_per_frame]")
            if int(type_bias.shape[0]) != int(type_bias.shape[1]):
                raise ValueError("type_bias must be square in the atomic-number dimensions")

        _as_trainable_or_buffer(self, "rbf_weight", rbf_weight, trainable=trainable)
        _as_trainable_or_buffer(self, "centers", centers, trainable=False)
        _as_trainable_or_buffer(self, "gamma", gamma.reshape(()), trainable=False)
        if type_bias is None:
            self.type_bias = None
        else:
            _as_trainable_or_buffer(self, "type_bias", type_bias, trainable=trainable)
        self.cutoff = float(cutoff)
        self.heads_per_frame = int(heads_per_frame)
        self.diag_zero = bool(diag_zero)
        self.envelope_in_score = bool(envelope_in_score)

    def forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        dist: torch.Tensor | None,
        rbf: torch.Tensor | None,
        atom_types: torch.Tensor | None,
    ) -> FixedKLocalBiasResult:
        del k, v
        if dist is None:
            raise ValueError("ESENEnvelopedRBFTypeFixedKBias requires dist")
        dist = dist.to(device=q.device, dtype=q.dtype)
        neighbor_mask = neighbor_mask.to(device=q.device, dtype=torch.bool)
        neighbor_idx = neighbor_idx.to(device=q.device, dtype=torch.long).masked_fill(~neighbor_mask, 0)
        if rbf is None:
            centers = self.centers.to(device=q.device, dtype=q.dtype)
            gamma = self.gamma.to(device=q.device, dtype=q.dtype)
            rbf = torch.exp(-gamma * (dist[..., None] - centers.view(1, 1, -1)).square())
        else:
            rbf = rbf.to(device=q.device, dtype=q.dtype)
        if q.shape[1] % self.heads_per_frame != 0:
            raise ValueError("heads_per_frame must divide the number of attention heads")
        if rbf.shape[-1] != self.rbf_weight.shape[-1]:
            raise ValueError(f"rbf last dimension {rbf.shape[-1]} does not match rbf_weight {self.rbf_weight.shape[-1]}")

        same = neighbor_idx == torch.arange(q.shape[0], device=q.device)[:, None]
        local_mask = neighbor_mask & (dist < self.cutoff)
        local_mask = local_mask | (same & neighbor_mask)
        env = _c2_quintic_envelope(dist, self.cutoff)
        env = torch.where(neighbor_mask, env, torch.zeros_like(env))

        env_bias = env
        if self.diag_zero:
            env_bias = env_bias.masked_fill(same, 0.0)

        raw_subhead = torch.einsum(
            "nkr,sr->nks",
            rbf,
            self.rbf_weight.to(device=q.device, dtype=q.dtype),
        )
        if self.type_bias is not None:
            if atom_types is None:
                raise ValueError("type_bias requires atom_types")
            zmax = self.type_bias.shape[0] - 1
            zi = atom_types.to(device=q.device).long().clamp(min=0, max=zmax)
            zj = zi.index_select(0, neighbor_idx.reshape(-1)).reshape_as(neighbor_idx)
            pair_type = self.type_bias.to(device=q.device, dtype=q.dtype)[zi[:, None], zj]
            raw_subhead = raw_subhead + pair_type

        subhead_bias = env_bias[..., None] * raw_subhead
        head_subidx = torch.arange(q.shape[1], device=q.device) % self.heads_per_frame
        bias = subhead_bias.index_select(dim=-1, index=head_subidx).permute(0, 2, 1).contiguous()

        if self.envelope_in_score:
            env_score = env.masked_fill(same & neighbor_mask, 1.0)
            bias = bias + env_score.clamp_min(1.0e-20).log().unsqueeze(1).to(dtype=bias.dtype)
        return FixedKLocalBiasResult(bias=bias, mask=local_mask)


def _gather_neighbor_tokens(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    num_tokens, num_heads, head_dim = values.shape
    safe_idx = neighbor_idx.to(device=values.device, dtype=torch.long).reshape(-1)
    gathered = values.index_select(0, safe_idx)
    return gathered.reshape(num_tokens, neighbor_idx.shape[-1], num_heads, head_dim)


def fixed_k_local_attention_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    bias: FixedKLocalBias | None = None,
    dist: torch.Tensor | None = None,
    rbf: torch.Tensor | None = None,
    atom_types: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Reference fixed-K local atom attention with pluggable local bias.

    This is intentionally a simple PyTorch parity target for the future Triton
    kernel.  Inputs are flat atom-token tensors; padded batches should first be
    flattened through ``GeometryCache``/``FlatGeometryCache``.
    """

    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q/k/v must have identical shape [N, H, D]")
    if neighbor_idx.ndim != 2 or neighbor_mask.shape != neighbor_idx.shape:
        raise ValueError("neighbor_idx and neighbor_mask must have shape [N, K]")
    if neighbor_idx.shape[0] != q.shape[0]:
        raise ValueError("neighbor_idx first dimension must match q/k/v tokens")
    if dist is not None and dist.shape != neighbor_idx.shape:
        raise ValueError("dist must have shape [N, K]")
    if rbf is not None and rbf.shape[:2] != neighbor_idx.shape:
        raise ValueError("rbf must have shape [N, K, R]")
    if atom_types is not None and atom_types.shape[0] != q.shape[0]:
        raise ValueError("atom_types must have shape [N]")
    if dropout_p != 0.0 and not training:
        raise ValueError("dropout_p is only supported with training=True")

    neighbor_mask = neighbor_mask.to(device=q.device, dtype=torch.bool)
    neighbor_idx = neighbor_idx.to(device=q.device, dtype=torch.long).masked_fill(~neighbor_mask, 0)
    k_neighbors = _gather_neighbor_tokens(k, neighbor_idx)
    v_neighbors = _gather_neighbor_tokens(v, neighbor_idx)

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("nhd,nkhd->nhk", q, k_neighbors) * scale
    if bias is not None:
        bias_result = bias(
            q=q,
            k=k,
            v=v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            dist=dist,
            rbf=rbf,
            atom_types=atom_types,
        )
        if bias_result.bias.shape != scores.shape:
            raise ValueError(f"bias must have shape {tuple(scores.shape)}, got {tuple(bias_result.bias.shape)}")
        scores = scores + bias_result.bias.to(device=q.device, dtype=scores.dtype)
        if bias_result.mask is not None:
            neighbor_mask = neighbor_mask & bias_result.mask.to(device=q.device, dtype=torch.bool)
    scores = scores.masked_fill(~neighbor_mask[:, None, :], -torch.inf)
    probs = _safe_masked_softmax(scores, dim=-1)
    if training and dropout_p > 0.0:
        probs = F.dropout(probs, p=float(dropout_p), training=True)
    return torch.einsum("nhk,nkhd->nhd", probs, v_neighbors)
