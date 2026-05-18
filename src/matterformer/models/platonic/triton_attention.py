from __future__ import annotations

import math
import os

import torch

from matterformer.models.platonic.radius_sparse import RadiusBlockSparseLayout

try:
    import torch._dynamo as _torch_dynamo
except ImportError:  # pragma: no cover - older PyTorch without Dynamo.
    _torch_dynamo = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA nodes with Triton installed.
    triton = None
    tl = None


TRITON_PLATONIC_ATTENTION_AVAILABLE = triton is not None and tl is not None

SUPPORTED_PLATONIC_ATTENTION_PRECISIONS = {"tf32", "tf32x3", "ieee", "bf16_flash_compat"}
PLATONIC_ATTENTION_BIAS_MODES = {
    "none": 0,
    "radial_rbf": 1,
    "radial_r2": 2,
    "radial_slope": 3,
    "rbf_type_enveloped": 4,
    "radius_rbf_type_enveloped": 5,
}


def normalize_platonic_attention_precision(value: str | None) -> str:
    precision = str(value or "tf32").lower()
    if precision not in SUPPORTED_PLATONIC_ATTENTION_PRECISIONS:
        raise ValueError(
            "platonic Triton attention precision must be one of "
            f"{sorted(SUPPORTED_PLATONIC_ATTENTION_PRECISIONS)}, got {value!r}"
        )
    return precision


def _kernel_input_precision(precision: str) -> str:
    return "tf32" if precision == "bf16_flash_compat" else precision


def normalize_platonic_attention_bias_mode(value: str | int | None, *, has_bias: bool = False) -> int:
    if value is None:
        return PLATONIC_ATTENTION_BIAS_MODES["radial_rbf"] if has_bias else PLATONIC_ATTENTION_BIAS_MODES["none"]
    if isinstance(value, int):
        if value not in set(PLATONIC_ATTENTION_BIAS_MODES.values()):
            raise ValueError(f"unknown Platonic attention bias mode id: {value}")
        return int(value)
    key = str(value).lower().replace("-", "_")
    if key in {"rbf", "radial"}:
        key = "radial_rbf"
    elif key in {"r2", "radial_square", "radial_squared"}:
        key = "radial_r2"
    elif key in {"slope", "radial_linear"}:
        key = "radial_slope"
    elif key in {"rbf_type", "type_rbf", "rbf_type_bias", "smooth_local", "smooth_local_mod"}:
        key = "rbf_type_enveloped"
    elif key in {
        "radius_rbf_type",
        "radius_rbf_type_bias",
        "radius_rbf_type_enveloped",
        "radius_sparse_rbf_type",
        "radius_sparse_rbf_type_enveloped",
        "sparse_rbf_type",
        "sparse_rbf_type_enveloped",
        "esen_local",
        "esen_like",
    }:
        key = "radius_rbf_type_enveloped"
    if key not in PLATONIC_ATTENTION_BIAS_MODES:
        raise ValueError(f"unknown Platonic attention bias mode: {value!r}")
    return PLATONIC_ATTENTION_BIAS_MODES[key]


def _is_rbf_type_bias_mode(bias_mode: int) -> bool:
    return int(bias_mode) in {
        PLATONIC_ATTENTION_BIAS_MODES["rbf_type_enveloped"],
        PLATONIC_ATTENTION_BIAS_MODES["radius_rbf_type_enveloped"],
    }


def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def platonic_attention_block_d_for_head_dim(head_dim: int) -> int:
    return max(16, _next_power_of_2(int(head_dim)))


def platonic_attention_split_head_dim_tail(head_dim: int) -> int | None:
    """Return the power-of-two tail width for the 128+tail head-dim path.

    Triton currently requires ``tl.arange(0, BLOCK)`` bounds to be powers of two,
    so the old generic path rounded head_dim=160 all the way up to BLOCK_D=256.
    The h1920 / 12-head OMol recipe has head_dim=160, where a 128+32 split
    avoids 96 padded channels in every q/k/v/dout tile.  Keep this deliberately
    narrow until more head dimensions have been benchmarked.
    """

    head_dim = int(head_dim)
    mode = os.environ.get("MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM", "auto").strip().lower()
    if mode in {"0", "false", "off", "no", "disable", "disabled"}:
        return None
    if head_dim != 160:
        return None
    return 32


def _triton_meta_int_env(name: str, default: int, *, min_value: int = 1, max_value: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError:
            value = int(default)
    value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def _disable_dynamo_if_available(fn):
    if _torch_dynamo is None:
        return fn
    return _torch_dynamo.disable(fn)


def _validate_flat_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q/k/v must have identical shape [N, H, D]")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a 1D tensor")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least start and end offsets")


def _radial_rbf_bias_reference(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    *,
    num_heads: int,
    heads_per_frame: int,
    rbf_weight: torch.Tensor,
    gate: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    diag_zero: bool,
    bias_mode: int = PLATONIC_ATTENTION_BIAS_MODES["radial_rbf"],
) -> torch.Tensor:
    if heads_per_frame <= 0 or num_heads % int(heads_per_frame) != 0:
        raise ValueError("heads_per_frame must divide num_heads for Platonic radial attention bias")
    delta = pos_j[None, :, :] - pos_i[:, None, :]
    dist2 = delta.square().sum(dim=-1).clamp_min(0.0)
    weights = rbf_weight.to(device=dist2.device, dtype=dist2.dtype)
    if bias_mode == PLATONIC_ATTENTION_BIAS_MODES["radial_rbf"]:
        dist = dist2.sqrt()
        basis = torch.exp(
            -gamma.to(dtype=dist.dtype, device=dist.device)
            * (dist[..., None] - centers.to(dist.device, dist.dtype)).square()
        )
        subhead_bias = torch.einsum("ijm,hm->ijh", basis, weights)
    elif bias_mode == PLATONIC_ATTENTION_BIAS_MODES["radial_r2"]:
        subhead_bias = dist2[..., None] * weights[:, 0].view(1, 1, -1)
    elif bias_mode == PLATONIC_ATTENTION_BIAS_MODES["radial_slope"]:
        subhead_bias = dist2.sqrt()[..., None] * weights[:, 0].view(1, 1, -1)
    else:
        raise ValueError(f"unsupported radial bias mode id: {bias_mode}")
    subhead_bias = subhead_bias * (1.0 + gate.to(device=dist2.device, dtype=dist2.dtype)).view(1, 1, -1)
    if diag_zero and pos_i.shape[0] == pos_j.shape[0]:
        diagonal = torch.eye(pos_i.shape[0], device=pos_i.device, dtype=torch.bool)
        subhead_bias = subhead_bias.masked_fill(diagonal[..., None], 0.0)
    head_subidx = torch.arange(num_heads, device=pos_i.device) % int(heads_per_frame)
    return subhead_bias.index_select(dim=-1, index=head_subidx).permute(2, 0, 1).contiguous()


def _c2_quintic_envelope(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    x = (dist / float(cutoff)).clamp(min=0.0, max=1.0)
    x2 = x.square()
    x3 = x2 * x
    env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
    return torch.where(dist < float(cutoff), env, torch.zeros_like(env))


def _rbf_type_bias_reference(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    atom_i: torch.Tensor | None,
    atom_j: torch.Tensor | None,
    *,
    num_heads: int,
    heads_per_frame: int,
    rbf_weight: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    cutoff: float,
    type_bias: torch.Tensor | None = None,
    diag_zero: bool,
) -> torch.Tensor:
    if heads_per_frame <= 0 or num_heads % int(heads_per_frame) != 0:
        raise ValueError("heads_per_frame must divide num_heads for Platonic RBF/type attention bias")
    delta = pos_j[None, :, :] - pos_i[:, None, :]
    dist = delta.square().sum(dim=-1).clamp_min(0.0).sqrt()
    env = _c2_quintic_envelope(dist, cutoff).to(dtype=dist.dtype)
    if diag_zero and pos_i.shape[0] == pos_j.shape[0]:
        diagonal = torch.eye(pos_i.shape[0], device=pos_i.device, dtype=torch.bool)
        env = env.masked_fill(diagonal, 0.0)

    centers = centers.to(device=dist.device, dtype=dist.dtype)
    gamma = gamma.to(device=dist.device, dtype=dist.dtype)
    rbf = torch.exp(-gamma * (dist[..., None] - centers.view(1, 1, -1)).square())
    rbf = rbf * env[..., None]
    weights = rbf_weight.to(device=dist.device, dtype=dist.dtype)
    subhead_bias = torch.einsum("ijm,hm->ijh", rbf, weights)

    if type_bias is not None:
        if atom_i is None or atom_j is None:
            raise ValueError("type_bias requires atom_i and atom_j")
        zmax = type_bias.shape[0] - 1
        zi = atom_i.to(device=dist.device).long().clamp(min=0, max=zmax)
        zj = atom_j.to(device=dist.device).long().clamp(min=0, max=zmax)
        pair_type = type_bias.to(device=dist.device, dtype=dist.dtype)[zi[:, None], zj[None, :]]
        subhead_bias = subhead_bias + env[..., None] * pair_type

    head_subidx = torch.arange(num_heads, device=pos_i.device) % int(heads_per_frame)
    return subhead_bias.index_select(dim=-1, index=head_subidx).permute(2, 0, 1).contiguous()


def _radius_rbf_type_sparse_score_reference(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    atom_i: torch.Tensor | None,
    atom_j: torch.Tensor | None,
    *,
    num_heads: int,
    heads_per_frame: int,
    rbf_weight: torch.Tensor,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    cutoff: float,
    type_bias: torch.Tensor | None = None,
    diag_zero: bool,
    include_self: bool,
    envelope_in_score: bool,
    abs_i: torch.Tensor | None = None,
    abs_j: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if heads_per_frame <= 0 or num_heads % int(heads_per_frame) != 0:
        raise ValueError("heads_per_frame must divide num_heads for Platonic radius-sparse attention")
    delta = pos_j[None, :, :] - pos_i[:, None, :]
    dist = delta.square().sum(dim=-1).clamp_min(0.0).sqrt()
    env = _c2_quintic_envelope(dist, cutoff).to(dtype=dist.dtype)
    local_mask = dist < float(cutoff)
    diagonal = None
    if abs_i is not None or abs_j is not None:
        if abs_i is None or abs_j is None:
            raise ValueError("abs_i and abs_j must be provided together")
        diagonal = abs_i.to(device=dist.device)[:, None] == abs_j.to(device=dist.device)[None, :]
    elif pos_i.shape[0] == pos_j.shape[0]:
        diagonal = torch.eye(pos_i.shape[0], device=pos_i.device, dtype=torch.bool)
    if diagonal is not None and include_self:
        local_mask = local_mask | diagonal

    env_for_bias = env
    if diag_zero and diagonal is not None:
        env_for_bias = env_for_bias.masked_fill(diagonal, 0.0)

    centers = centers.to(device=dist.device, dtype=dist.dtype)
    gamma = gamma.to(device=dist.device, dtype=dist.dtype)
    rbf = torch.exp(-gamma * (dist[..., None] - centers.view(1, 1, -1)).square())
    rbf = rbf * env_for_bias[..., None]
    weights = rbf_weight.to(device=dist.device, dtype=dist.dtype)
    subhead_bias = torch.einsum("ijm,hm->ijh", rbf, weights)

    if type_bias is not None:
        if atom_i is None or atom_j is None:
            raise ValueError("type_bias requires atom_i and atom_j")
        zmax = type_bias.shape[0] - 1
        zi = atom_i.to(device=dist.device).long().clamp(min=0, max=zmax)
        zj = atom_j.to(device=dist.device).long().clamp(min=0, max=zmax)
        pair_type = type_bias.to(device=dist.device, dtype=dist.dtype)[zi[:, None], zj[None, :]]
        subhead_bias = subhead_bias + env_for_bias[..., None] * pair_type

    head_subidx = torch.arange(num_heads, device=pos_i.device) % int(heads_per_frame)
    bias = subhead_bias.index_select(dim=-1, index=head_subidx).permute(2, 0, 1).contiguous()

    if envelope_in_score:
        env_for_score = env
        if diagonal is not None and include_self:
            env_for_score = env_for_score.masked_fill(diagonal, 1.0)
        bias = bias + env_for_score.clamp_min(1.0e-20).log().unsqueeze(0).to(dtype=bias.dtype)
    return bias, local_mask


def platonic_attention_flat_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,
    pos: torch.Tensor | None = None,
    atom_types: torch.Tensor | None = None,
    heads_per_frame: int | None = None,
    rbf_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    type_bias: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    cutoff: float | None = None,
    radial_bias_kind: str | int | None = None,
    diag_zero: bool = True,
    include_self: bool = True,
    envelope_in_score: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Reference flat varlen attention with optional group-shared radial bias.

    This intentionally materializes per-segment scores and is only for tests,
    CPU fallback, and parity debugging. Production flat OMol runs should use
    the Triton path on CUDA.
    """

    _validate_flat_inputs(q, k, v, cu_seqlens)
    if dropout_p != 0.0:
        raise ValueError("Platonic flat reference attention only supports dropout_p=0")
    use_rbf = rbf_weight is not None
    bias_mode = normalize_platonic_attention_bias_mode(radial_bias_kind, has_bias=use_rbf)
    use_rbf_type = use_rbf and _is_rbf_type_bias_mode(bias_mode)
    use_radius_sparse = use_rbf and bias_mode == PLATONIC_ATTENTION_BIAS_MODES["radius_rbf_type_enveloped"]
    if use_rbf:
        if pos is None or heads_per_frame is None or centers is None or gamma is None:
            raise ValueError("Radial bias requires pos, heads_per_frame, centers, and gamma")
        if not use_rbf_type and gate is None:
            raise ValueError("Legacy radial bias requires gate")
        if use_rbf_type and cutoff is None:
            raise ValueError("RBF/type enveloped bias requires cutoff")
        if use_rbf_type and type_bias is not None and atom_types is None:
            raise ValueError("RBF/type enveloped bias with type_bias requires atom_types")
        if pos.ndim != 2 or pos.shape != (q.shape[0], 3):
            raise ValueError(f"pos must have shape [{q.shape[0]}, 3], got {tuple(pos.shape)}")
    outputs: list[torch.Tensor] = []
    starts = cu_seqlens[:-1].detach().cpu().tolist()
    ends = cu_seqlens[1:].detach().cpu().tolist()
    scale = 1.0 / math.sqrt(q.shape[-1])
    for start, end in zip(starts, ends):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        q_seg = q[start_i:end_i].transpose(0, 1)
        k_seg = k[start_i:end_i].transpose(0, 1)
        v_seg = v[start_i:end_i].transpose(0, 1)
        scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale
        if use_radius_sparse:
            assert pos is not None
            bias, local_mask = _radius_rbf_type_sparse_score_reference(
                pos[start_i:end_i],
                pos[start_i:end_i],
                None if atom_types is None else atom_types[start_i:end_i],
                None if atom_types is None else atom_types[start_i:end_i],
                num_heads=q.shape[1],
                heads_per_frame=int(heads_per_frame),
                rbf_weight=rbf_weight,
                type_bias=type_bias,
                centers=centers,
                gamma=gamma,
                cutoff=float(cutoff),
                diag_zero=diag_zero,
                include_self=include_self,
                envelope_in_score=envelope_in_score,
            )
            scores = scores + bias.to(dtype=scores.dtype)
            scores = scores.masked_fill(~local_mask.unsqueeze(0), -torch.inf)
        elif use_rbf_type:
            assert pos is not None
            bias = _rbf_type_bias_reference(
                pos[start_i:end_i],
                pos[start_i:end_i],
                None if atom_types is None else atom_types[start_i:end_i],
                None if atom_types is None else atom_types[start_i:end_i],
                num_heads=q.shape[1],
                heads_per_frame=int(heads_per_frame),
                rbf_weight=rbf_weight,
                type_bias=type_bias,
                centers=centers,
                gamma=gamma,
                cutoff=float(cutoff),
                diag_zero=diag_zero,
            )
            scores = scores + bias.to(dtype=scores.dtype)
        elif use_rbf:
            assert pos is not None
            bias = _radial_rbf_bias_reference(
                pos[start_i:end_i],
                pos[start_i:end_i],
                num_heads=q.shape[1],
                heads_per_frame=int(heads_per_frame),
                rbf_weight=rbf_weight,
                gate=gate,  # type: ignore[arg-type]
                centers=centers,
                gamma=gamma,
                diag_zero=diag_zero,
                bias_mode=bias_mode,
            )
            scores = scores + bias.to(dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        if use_radius_sparse:
            row_has_neighbor = torch.isfinite(scores).any(dim=-1, keepdim=True)
            probs = torch.where(row_has_neighbor, probs, torch.zeros_like(probs))
        if training and dropout_p > 0.0:
            probs = torch.nn.functional.dropout(probs, p=float(dropout_p), training=True)
        outputs.append(torch.matmul(probs, v_seg).transpose(0, 1).contiguous())
    if not outputs:
        return torch.zeros_like(q)
    return torch.cat(outputs, dim=0)


def platonic_radius_block_sparse_attention_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    pos: torch.Tensor,
    atom_types: torch.Tensor | None,
    heads_per_frame: int,
    rbf_weight: torch.Tensor,
    type_bias: torch.Tensor | None,
    centers: torch.Tensor,
    gamma: torch.Tensor,
    cutoff: float,
    diag_zero: bool,
    include_self: bool,
    envelope_in_score: bool,
    radius_layout: RadiusBlockSparseLayout,
) -> torch.Tensor:
    """Block-sparse torch reference for radius-local RBF/type attention.

    This keeps the same eSEN-like score semantics as the dense reference but
    iterates only over live radius block pairs from ``RadiusBlockSparseLayout``.
    It is intended for correctness/prototyping and autograd parity; the fast
    CUDA path should use Triton kernels that consume the same layout.
    """

    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q/k/v must have identical shape [N, H, D]")
    if pos.ndim != 2 or pos.shape != (q.shape[0], 3):
        raise ValueError(f"pos must have shape [{q.shape[0]}, 3], got {tuple(pos.shape)}")
    if type_bias is not None and atom_types is None:
        raise ValueError("type_bias requires atom_types")
    if radius_layout.q_block_start.device != q.device:
        radius_layout = radius_layout.to(q.device)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.zeros_like(q)
    block_ptr = radius_layout.block_ptr.detach().cpu().tolist()
    block_col = radius_layout.block_col.detach().cpu().tolist()
    q_starts = radius_layout.q_block_start.detach().cpu().tolist()
    q_ends = radius_layout.q_block_end.detach().cpu().tolist()
    k_starts = radius_layout.k_block_start.detach().cpu().tolist()
    k_ends = radius_layout.k_block_end.detach().cpu().tolist()

    for q_block, (m_start, m_end) in enumerate(zip(q_starts, q_ends)):
        m_start_i = int(m_start)
        m_end_i = int(m_end)
        if m_end_i <= m_start_i:
            continue
        score_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        q_tile = q[m_start_i:m_end_i]
        pos_i = pos[m_start_i:m_end_i]
        atom_i = None if atom_types is None else atom_types[m_start_i:m_end_i]
        for ptr in range(int(block_ptr[q_block]), int(block_ptr[q_block + 1])):
            k_block = int(block_col[ptr])
            n_start_i = int(k_starts[k_block])
            n_end_i = int(k_ends[k_block])
            if n_end_i <= n_start_i:
                continue
            k_tile = k[n_start_i:n_end_i]
            scores = torch.einsum("mhd,nhd->hmn", q_tile, k_tile) * scale
            bias, local_mask = _radius_rbf_type_sparse_score_reference(
                pos_i,
                pos[n_start_i:n_end_i],
                atom_i,
                None if atom_types is None else atom_types[n_start_i:n_end_i],
                num_heads=q.shape[1],
                heads_per_frame=int(heads_per_frame),
                rbf_weight=rbf_weight,
                type_bias=type_bias,
                centers=centers,
                gamma=gamma,
                cutoff=float(cutoff),
                diag_zero=diag_zero,
                include_self=include_self,
                envelope_in_score=envelope_in_score,
                abs_i=torch.arange(m_start_i, m_end_i, device=pos.device),
                abs_j=torch.arange(n_start_i, n_end_i, device=pos.device),
            )
            scores = scores + bias.to(dtype=scores.dtype)
            scores = scores.masked_fill(~local_mask.unsqueeze(0), -torch.inf)
            score_chunks.append(scores)
            value_chunks.append(v[n_start_i:n_end_i])
        if not score_chunks:
            continue
        scores_full = torch.cat(score_chunks, dim=-1)
        values_full = torch.cat(value_chunks, dim=0)
        probs = torch.softmax(scores_full, dim=-1)
        row_has_neighbor = torch.isfinite(scores_full).any(dim=-1, keepdim=True)
        probs = torch.where(row_has_neighbor, probs, torch.zeros_like(probs))
        out[m_start_i:m_end_i] = torch.einsum("hmn,nhd->mhd", probs, values_full)
    return out


if TRITON_PLATONIC_ATTENTION_AVAILABLE:

    @triton.jit
    def _radial_bias_tile(
        pos_ptr,
        atom_type_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        start: tl.tensor,
        offs_m,
        offs_n,
        m_mask,
        n_mask,
        head_idx: tl.tensor,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pos_i_base = (start + offs_m) * 3
        pos_j_base = (start + offs_n) * 3
        xi = tl.load(pos_ptr + pos_i_base + 0, mask=m_mask, other=0.0)[:, None]
        yi = tl.load(pos_ptr + pos_i_base + 1, mask=m_mask, other=0.0)[:, None]
        zi = tl.load(pos_ptr + pos_i_base + 2, mask=m_mask, other=0.0)[:, None]
        xj = tl.load(pos_ptr + pos_j_base + 0, mask=n_mask, other=0.0)[None, :]
        yj = tl.load(pos_ptr + pos_j_base + 1, mask=n_mask, other=0.0)[None, :]
        zj = tl.load(pos_ptr + pos_j_base + 2, mask=n_mask, other=0.0)[None, :]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        dist2 = tl.maximum(dx * dx + dy * dy + dz * dz, 0.0)
        subhead = head_idx % heads_per_frame
        bias = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if bias_mode == 1:
            dist = tl.sqrt(dist2)
            gamma = tl.load(gamma_ptr).to(tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                bias += weight * rho
            gate = tl.load(gate_ptr + subhead).to(tl.float32)
            bias *= 1.0 + gate
        elif bias_mode == 2:
            weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
            bias = weight * dist2
            gate = tl.load(gate_ptr + subhead).to(tl.float32)
            bias *= 1.0 + gate
        elif bias_mode == 3:
            weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
            bias = weight * tl.sqrt(dist2)
            gate = tl.load(gate_ptr + subhead).to(tl.float32)
            bias *= 1.0 + gate
        elif bias_mode == 4:
            dist = tl.sqrt(dist2)
            x = dist / cutoff
            x = tl.minimum(tl.maximum(x, 0.0), 1.0)
            x2 = x * x
            x3 = x2 * x
            env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
            env = tl.where(dist < cutoff, env, 0.0)
            gamma = tl.load(gamma_ptr).to(tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                bias += weight * rho
            zi_atom = tl.load(atom_type_ptr + start + offs_m, mask=m_mask, other=0)[:, None]
            zj_atom = tl.load(atom_type_ptr + start + offs_n, mask=n_mask, other=0)[None, :]
            zi_atom = tl.minimum(tl.maximum(zi_atom, 0), max_atomic_number)
            zj_atom = tl.minimum(tl.maximum(zj_atom, 0), max_atomic_number)
            zdim = max_atomic_number + 1
            type_index = ((zi_atom * zdim + zj_atom) * heads_per_frame + subhead)
            type_term = tl.load(
                type_bias_ptr + type_index,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            bias += env * type_term
        if diag_zero:
            bias = tl.where(offs_m[:, None] == offs_n[None, :], 0.0, bias)
        return bias

    @triton.jit
    def _platonic_flat_attention_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        out_ptr,
        lse_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        m_start = pid_m * BLOCK_M
        if m_start >= seqlen:
            return
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < seqlen
        d_mask = offs_d < head_dim

        q = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for block_n in range(0, tl.cdiv(max_seqlen, BLOCK_N)):
            n_start = block_n * BLOCK_N
            if n_start < seqlen:
                n = n_start + offs_n
                n_mask = n < seqlen
                k = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, tl.trans(k), input_precision=input_precision).to(tl.float32) * scale
                if has_rbf:
                    scores += _radial_bias_tile(
                        pos_ptr,
                        atom_type_ptr,
                        rbf_weight_ptr,
                        gate_ptr,
                        type_bias_ptr,
                        centers_ptr,
                        gamma_ptr,
                        start,
                        offs_m,
                        n,
                        m_mask,
                        n_mask,
                        head_idx,
                        heads_per_frame,
                        num_rbf,
                        bias_mode,
                        cutoff,
                        max_atomic_number,
                        diag_zero,
                        BLOCK_M,
                        BLOCK_N,
                    )
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=input_precision)
                m_i = m_ij
        acc = acc / tl.maximum(l_i[:, None], 1.0e-20)
        tl.store(
            out_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            acc,
            mask=m_mask[:, None] & d_mask[None, :],
        )
        tl.store(lse_ptr + (start + offs_m) * num_heads + head_idx, m_i + tl.log(tl.maximum(l_i, 1.0e-20)), mask=m_mask)

    @triton.jit
    def _platonic_flat_attention_fwd_kernel_d128_tail(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        out_ptr,
        lse_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D_TAIL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        m_start = pid_m * BLOCK_M
        if m_start >= seqlen:
            return
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d0 = tl.arange(0, 128)
        offs_dt = tl.arange(0, BLOCK_D_TAIL)
        offs_d1 = 128 + offs_dt
        m_mask = offs_m < seqlen
        d0_mask = offs_d0 < head_dim
        d1_mask = offs_d1 < head_dim

        q0 = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            mask=m_mask[:, None] & d0_mask[None, :],
            other=0.0,
        )
        q1 = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            mask=m_mask[:, None] & d1_mask[None, :],
            other=0.0,
        )
        acc0 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_D_TAIL), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for block_n in range(0, tl.cdiv(max_seqlen, BLOCK_N)):
            n_start = block_n * BLOCK_N
            if n_start < seqlen:
                n = n_start + offs_n
                n_mask = n < seqlen
                k0 = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
                    mask=n_mask[:, None] & d0_mask[None, :],
                    other=0.0,
                )
                k1 = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
                    mask=n_mask[:, None] & d1_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(q0, tl.trans(k0), input_precision=input_precision).to(tl.float32)
                scores += tl.dot(q1, tl.trans(k1), input_precision=input_precision).to(tl.float32)
                scores *= scale
                if has_rbf:
                    scores += _radial_bias_tile(
                        pos_ptr,
                        atom_type_ptr,
                        rbf_weight_ptr,
                        gate_ptr,
                        type_bias_ptr,
                        centers_ptr,
                        gamma_ptr,
                        start,
                        offs_m,
                        n,
                        m_mask,
                        n_mask,
                        head_idx,
                        heads_per_frame,
                        num_rbf,
                        bias_mode,
                        cutoff,
                        max_atomic_number,
                        diag_zero,
                        BLOCK_M,
                        BLOCK_N,
                    )
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                v0 = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
                    mask=n_mask[:, None] & d0_mask[None, :],
                    other=0.0,
                )
                v1 = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
                    mask=n_mask[:, None] & d1_mask[None, :],
                    other=0.0,
                )
                acc0 = acc0 * alpha[:, None] + tl.dot(p.to(v0.dtype), v0, input_precision=input_precision)
                acc1 = acc1 * alpha[:, None] + tl.dot(p.to(v1.dtype), v1, input_precision=input_precision)
                m_i = m_ij
        inv_l = 1.0 / tl.maximum(l_i, 1.0e-20)
        acc0 = acc0 * inv_l[:, None]
        acc1 = acc1 * inv_l[:, None]
        tl.store(
            out_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            acc0,
            mask=m_mask[:, None] & d0_mask[None, :],
        )
        tl.store(
            out_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            acc1,
            mask=m_mask[:, None] & d1_mask[None, :],
        )
        tl.store(lse_ptr + (start + offs_m) * num_heads + head_idx, m_i + tl.log(tl.maximum(l_i, 1.0e-20)), mask=m_mask)

    @triton.jit
    def _platonic_flat_attention_bwd_preprocess_kernel(
        out_ptr,
        dout_ptr,
        delta_ptr,
        total_tokens: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        base = (token_idx * num_heads + head_idx) * head_dim
        out = tl.load(out_ptr + base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        tl.store(delta_ptr + token_idx * num_heads + head_idx, tl.sum(out * dout, axis=0))

    @triton.jit
    def _platonic_flat_attention_bwd_preprocess_vector_kernel(
        out_ptr,
        dout_ptr,
        delta_ptr,
        total_pairs: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
        offs_d = tl.arange(0, BLOCK_D)
        p_mask = offs_p < total_pairs
        d_mask = offs_d < head_dim
        base = offs_p[:, None] * head_dim + offs_d[None, :]
        out = tl.load(out_ptr + base, mask=p_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + base, mask=p_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        tl.store(delta_ptr + offs_p, tl.sum(out * dout, axis=1), mask=p_mask)

    @triton.jit
    def _platonic_flat_attention_bwd_preprocess_kernel_d128_tail(
        out_ptr,
        dout_ptr,
        delta_ptr,
        total_tokens: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D_TAIL: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_d0 = tl.arange(0, 128)
        offs_dt = tl.arange(0, BLOCK_D_TAIL)
        offs_d1 = 128 + offs_dt
        d0_mask = offs_d0 < head_dim
        d1_mask = offs_d1 < head_dim
        base = (token_idx * num_heads + head_idx) * head_dim
        out0 = tl.load(out_ptr + base + offs_d0, mask=d0_mask, other=0.0).to(tl.float32)
        dout0 = tl.load(dout_ptr + base + offs_d0, mask=d0_mask, other=0.0).to(tl.float32)
        out1 = tl.load(out_ptr + base + offs_d1, mask=d1_mask, other=0.0).to(tl.float32)
        dout1 = tl.load(dout_ptr + base + offs_d1, mask=d1_mask, other=0.0).to(tl.float32)
        delta = tl.sum(out0 * dout0, axis=0) + tl.sum(out1 * dout1, axis=0)
        tl.store(delta_ptr + token_idx * num_heads + head_idx, delta)

    @triton.jit
    def _platonic_flat_attention_bwd_preprocess_vector_kernel_d128_tail(
        out_ptr,
        dout_ptr,
        delta_ptr,
        total_pairs: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D_TAIL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
        offs_d0 = tl.arange(0, 128)
        offs_dt = tl.arange(0, BLOCK_D_TAIL)
        offs_d1 = 128 + offs_dt
        p_mask = offs_p < total_pairs
        d0_mask = offs_d0 < head_dim
        d1_mask = offs_d1 < head_dim
        base0 = offs_p[:, None] * head_dim + offs_d0[None, :]
        base1 = offs_p[:, None] * head_dim + offs_d1[None, :]
        out0 = tl.load(out_ptr + base0, mask=p_mask[:, None] & d0_mask[None, :], other=0.0).to(tl.float32)
        dout0 = tl.load(dout_ptr + base0, mask=p_mask[:, None] & d0_mask[None, :], other=0.0).to(tl.float32)
        out1 = tl.load(out_ptr + base1, mask=p_mask[:, None] & d1_mask[None, :], other=0.0).to(tl.float32)
        dout1 = tl.load(dout_ptr + base1, mask=p_mask[:, None] & d1_mask[None, :], other=0.0).to(tl.float32)
        delta = tl.sum(out0 * dout0, axis=1) + tl.sum(out1 * dout1, axis=1)
        tl.store(delta_ptr + offs_p, delta, mask=p_mask)

    @triton.jit
    def _platonic_flat_attention_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_gate_ptr,
        d_type_bias_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        m_start = pid_m * BLOCK_M
        if m_start >= seqlen:
            return
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < seqlen
        d_mask = offs_d < head_dim

        q = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        dout = tl.load(
            dout_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        lse = tl.load(lse_ptr + (start + offs_m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + (start + offs_m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        subhead = head_idx % heads_per_frame
        gate = tl.load(gate_ptr + subhead).to(tl.float32) if has_rbf and bias_mode != 4 else 0.0
        gate_factor = 1.0 + gate
        for block_n in range(0, tl.cdiv(max_seqlen, BLOCK_N)):
            n_start = block_n * BLOCK_N
            if n_start < seqlen:
                n = n_start + offs_n
                n_mask = n < seqlen
                k = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, tl.trans(k), input_precision=input_precision).to(tl.float32) * scale
                base_bias = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                if has_rbf:
                    pos_i_base = (start + offs_m) * 3
                    pos_j_base = (start + n) * 3
                    xi = tl.load(pos_ptr + pos_i_base + 0, mask=m_mask, other=0.0)[:, None]
                    yi = tl.load(pos_ptr + pos_i_base + 1, mask=m_mask, other=0.0)[:, None]
                    zi = tl.load(pos_ptr + pos_i_base + 2, mask=m_mask, other=0.0)[:, None]
                    xj = tl.load(pos_ptr + pos_j_base + 0, mask=n_mask, other=0.0)[None, :]
                    yj = tl.load(pos_ptr + pos_j_base + 1, mask=n_mask, other=0.0)[None, :]
                    zj = tl.load(pos_ptr + pos_j_base + 2, mask=n_mask, other=0.0)[None, :]
                    dx = xj - xi
                    dy = yj - yi
                    dz = zj - zi
                    dist2 = tl.maximum(dx * dx + dy * dy + dz * dz, 0.0)
                    if bias_mode == 1:
                        dist = tl.sqrt(dist2)
                        gamma = tl.load(gamma_ptr).to(tl.float32)
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center))
                            base_bias += weight * rho
                    elif bias_mode == 2:
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
                        base_bias = weight * dist2
                    elif bias_mode == 3:
                        dist = tl.sqrt(dist2)
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
                        base_bias = weight * dist
                    elif bias_mode == 4:
                        dist = tl.sqrt(dist2)
                        x = dist / cutoff
                        x = tl.minimum(tl.maximum(x, 0.0), 1.0)
                        x2 = x * x
                        x3 = x2 * x
                        env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
                        env = tl.where(dist < cutoff, env, 0.0)
                        gamma = tl.load(gamma_ptr).to(tl.float32)
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                            base_bias += weight * rho
                        zi_atom = tl.load(atom_type_ptr + start + offs_m, mask=m_mask, other=0)[:, None]
                        zj_atom = tl.load(atom_type_ptr + start + n, mask=n_mask, other=0)[None, :]
                        zi_atom = tl.minimum(tl.maximum(zi_atom, 0), max_atomic_number)
                        zj_atom = tl.minimum(tl.maximum(zj_atom, 0), max_atomic_number)
                        zdim = max_atomic_number + 1
                        type_index = ((zi_atom * zdim + zj_atom) * heads_per_frame + subhead)
                        type_term = tl.load(
                            type_bias_ptr + type_index,
                            mask=m_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        base_bias += env * type_term
                    bias = base_bias if bias_mode == 4 else base_bias * gate_factor
                    if diag_zero:
                        bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                    scores += bias
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
                dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
                dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
                dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
                tl.atomic_add(
                    dk_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    dk,
                    sem="relaxed",
                    mask=n_mask[:, None] & d_mask[None, :],
                )
                tl.atomic_add(
                    dv_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    dv,
                    sem="relaxed",
                    mask=n_mask[:, None] & d_mask[None, :],
                )
                if has_rbf:
                    if diag_zero:
                        ds_bias = tl.where(offs_m[:, None] == n[None, :], 0.0, ds)
                    else:
                        ds_bias = ds
                    if bias_mode != 4:
                        tl.atomic_add(d_gate_ptr + subhead, tl.sum(tl.sum(ds_bias * base_bias, axis=0), axis=0), sem="relaxed")
                    if bias_mode == 1:
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center))
                            grad_w = tl.sum(tl.sum(ds_bias * rho * gate_factor, axis=0), axis=0)
                            tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
                    elif bias_mode == 2:
                        grad_w = tl.sum(tl.sum(ds_bias * dist2 * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf, grad_w, sem="relaxed")
                    elif bias_mode == 3:
                        grad_w = tl.sum(tl.sum(ds_bias * dist * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf, grad_w, sem="relaxed")
                    elif bias_mode == 4:
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                            grad_w = tl.sum(tl.sum(ds_bias * rho, axis=0), axis=0)
                            tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
                        tl.atomic_add(
                            d_type_bias_ptr + type_index,
                            ds_bias * env,
                            sem="relaxed",
                            mask=m_mask[:, None] & n_mask[None, :],
                        )
        tl.store(
            dq_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            dq,
            mask=m_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _platonic_flat_attention_bwd_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        dq_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        m_start = pid_m * BLOCK_M
        if m_start >= seqlen:
            return
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < seqlen
        d_mask = offs_d < head_dim

        q = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        dout = tl.load(
            dout_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        lse = tl.load(lse_ptr + (start + offs_m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + (start + offs_m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        subhead = head_idx % heads_per_frame
        gate = tl.load(gate_ptr + subhead).to(tl.float32) if has_rbf else 0.0
        gate_factor = 1.0 + gate
        for block_n in range(0, tl.cdiv(max_seqlen, BLOCK_N)):
            n_start = block_n * BLOCK_N
            if n_start < seqlen:
                n = n_start + offs_n
                n_mask = n < seqlen
                k = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, tl.trans(k), input_precision=input_precision).to(tl.float32) * scale
                if has_rbf:
                    scores += _radial_bias_tile(
                        pos_ptr,
                        atom_type_ptr,
                        rbf_weight_ptr,
                        gate_ptr,
                        type_bias_ptr,
                        centers_ptr,
                        gamma_ptr,
                        start,
                        offs_m,
                        n,
                        m_mask,
                        n_mask,
                        head_idx,
                        heads_per_frame,
                        num_rbf,
                        bias_mode,
                        cutoff,
                        max_atomic_number,
                        diag_zero,
                        BLOCK_M,
                        BLOCK_N,
                    )
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
                dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
        tl.store(
            dq_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            dq,
            mask=m_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _platonic_flat_attention_bwd_dkv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_gate_ptr,
        d_type_bias_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        n_start = pid_n * BLOCK_N
        if n_start >= seqlen:
            return
        offs_m = tl.arange(0, BLOCK_M)
        n = n_start + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        n_mask = n < seqlen
        d_mask = offs_d < head_dim

        k = tl.load(
            k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        subhead = head_idx % heads_per_frame
        gate = tl.load(gate_ptr + subhead).to(tl.float32) if has_rbf and bias_mode != 4 else 0.0
        gate_factor = 1.0 + gate
        for block_m in range(0, tl.cdiv(max_seqlen, BLOCK_M)):
            m_start = block_m * BLOCK_M
            if m_start < seqlen:
                m = m_start + offs_m
                m_mask = m < seqlen
                q = tl.load(
                    q_ptr + ((start + m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=m_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                dout = tl.load(
                    dout_ptr + ((start + m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
                    mask=m_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                lse = tl.load(lse_ptr + (start + m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
                delta = tl.load(delta_ptr + (start + m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
                scores = tl.dot(q, tl.trans(k), input_precision=input_precision).to(tl.float32) * scale
                base_bias = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                if has_rbf:
                    pos_i_base = (start + m) * 3
                    pos_j_base = (start + n) * 3
                    xi = tl.load(pos_ptr + pos_i_base + 0, mask=m_mask, other=0.0)[:, None]
                    yi = tl.load(pos_ptr + pos_i_base + 1, mask=m_mask, other=0.0)[:, None]
                    zi = tl.load(pos_ptr + pos_i_base + 2, mask=m_mask, other=0.0)[:, None]
                    xj = tl.load(pos_ptr + pos_j_base + 0, mask=n_mask, other=0.0)[None, :]
                    yj = tl.load(pos_ptr + pos_j_base + 1, mask=n_mask, other=0.0)[None, :]
                    zj = tl.load(pos_ptr + pos_j_base + 2, mask=n_mask, other=0.0)[None, :]
                    dx = xj - xi
                    dy = yj - yi
                    dz = zj - zi
                    dist2 = tl.maximum(dx * dx + dy * dy + dz * dz, 0.0)
                    if bias_mode == 1:
                        dist = tl.sqrt(dist2)
                        gamma = tl.load(gamma_ptr).to(tl.float32)
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center))
                            base_bias += weight * rho
                    elif bias_mode == 2:
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
                        base_bias = weight * dist2
                    elif bias_mode == 3:
                        dist = tl.sqrt(dist2)
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
                        base_bias = weight * dist
                    elif bias_mode == 4:
                        dist = tl.sqrt(dist2)
                        x = dist / cutoff
                        x = tl.minimum(tl.maximum(x, 0.0), 1.0)
                        x2 = x * x
                        x3 = x2 * x
                        env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
                        env = tl.where(dist < cutoff, env, 0.0)
                        gamma = tl.load(gamma_ptr).to(tl.float32)
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                            base_bias += weight * rho
                        zi_atom = tl.load(atom_type_ptr + start + m, mask=m_mask, other=0)[:, None]
                        zj_atom = tl.load(atom_type_ptr + start + n, mask=n_mask, other=0)[None, :]
                        zi_atom = tl.minimum(tl.maximum(zi_atom, 0), max_atomic_number)
                        zj_atom = tl.minimum(tl.maximum(zj_atom, 0), max_atomic_number)
                        zdim = max_atomic_number + 1
                        type_index = ((zi_atom * zdim + zj_atom) * heads_per_frame + subhead)
                        type_term = tl.load(
                            type_bias_ptr + type_index,
                            mask=m_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        base_bias += env * type_term
                    bias = base_bias if bias_mode == 4 else base_bias * gate_factor
                    if diag_zero:
                        bias = tl.where(m[:, None] == n[None, :], 0.0, bias)
                    scores += bias
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
                dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
                dv += tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
                if has_rbf:
                    if diag_zero:
                        ds_bias = tl.where(m[:, None] == n[None, :], 0.0, ds)
                    else:
                        ds_bias = ds
                    if bias_mode != 4:
                        tl.atomic_add(d_gate_ptr + subhead, tl.sum(tl.sum(ds_bias * base_bias, axis=0), axis=0), sem="relaxed")
                    if bias_mode == 1:
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center))
                            grad_w = tl.sum(tl.sum(ds_bias * rho * gate_factor, axis=0), axis=0)
                            tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
                    elif bias_mode == 2:
                        grad_w = tl.sum(tl.sum(ds_bias * dist2 * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf, grad_w, sem="relaxed")
                    elif bias_mode == 3:
                        grad_w = tl.sum(tl.sum(ds_bias * dist * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf, grad_w, sem="relaxed")
                    elif bias_mode == 4:
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                            grad_w = tl.sum(tl.sum(ds_bias * rho, axis=0), axis=0)
                            tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
                        tl.atomic_add(
                            d_type_bias_ptr + type_index,
                            ds_bias * env,
                            sem="relaxed",
                            mask=m_mask[:, None] & n_mask[None, :],
                        )
        tl.store(
            dk_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            dk,
            mask=n_mask[:, None] & d_mask[None, :],
        )
        tl.store(
            dv_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            dv,
            mask=n_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _platonic_flat_attention_bwd_dq_kernel_d128_tail(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        dq_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D_TAIL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        m_start = pid_m * BLOCK_M
        if m_start >= seqlen:
            return
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d0 = tl.arange(0, 128)
        offs_dt = tl.arange(0, BLOCK_D_TAIL)
        offs_d1 = 128 + offs_dt
        m_mask = offs_m < seqlen
        d0_mask = offs_d0 < head_dim
        d1_mask = offs_d1 < head_dim

        q0 = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            mask=m_mask[:, None] & d0_mask[None, :],
            other=0.0,
        )
        q1 = tl.load(
            q_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            mask=m_mask[:, None] & d1_mask[None, :],
            other=0.0,
        )
        dout0 = tl.load(
            dout_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            mask=m_mask[:, None] & d0_mask[None, :],
            other=0.0,
        )
        dout1 = tl.load(
            dout_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            mask=m_mask[:, None] & d1_mask[None, :],
            other=0.0,
        )
        lse = tl.load(lse_ptr + (start + offs_m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + (start + offs_m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
        dq0 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        dq1 = tl.zeros((BLOCK_M, BLOCK_D_TAIL), dtype=tl.float32)
        for block_n in range(0, tl.cdiv(max_seqlen, BLOCK_N)):
            n_start = block_n * BLOCK_N
            if n_start < seqlen:
                n = n_start + offs_n
                n_mask = n < seqlen
                k0 = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
                    mask=n_mask[:, None] & d0_mask[None, :],
                    other=0.0,
                )
                k1 = tl.load(
                    k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
                    mask=n_mask[:, None] & d1_mask[None, :],
                    other=0.0,
                )
                v0 = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
                    mask=n_mask[:, None] & d0_mask[None, :],
                    other=0.0,
                )
                v1 = tl.load(
                    v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
                    mask=n_mask[:, None] & d1_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(q0, tl.trans(k0), input_precision=input_precision).to(tl.float32)
                scores += tl.dot(q1, tl.trans(k1), input_precision=input_precision).to(tl.float32)
                scores *= scale
                if has_rbf:
                    scores += _radial_bias_tile(
                        pos_ptr,
                        atom_type_ptr,
                        rbf_weight_ptr,
                        gate_ptr,
                        type_bias_ptr,
                        centers_ptr,
                        gamma_ptr,
                        start,
                        offs_m,
                        n,
                        m_mask,
                        n_mask,
                        head_idx,
                        heads_per_frame,
                        num_rbf,
                        bias_mode,
                        cutoff,
                        max_atomic_number,
                        diag_zero,
                        BLOCK_M,
                        BLOCK_N,
                    )
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
                dp = tl.dot(dout0, tl.trans(v0), input_precision=input_precision).to(tl.float32)
                dp += tl.dot(dout1, tl.trans(v1), input_precision=input_precision).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dq0 += tl.dot(ds.to(k0.dtype), k0, input_precision=input_precision) * scale
                dq1 += tl.dot(ds.to(k1.dtype), k1, input_precision=input_precision) * scale
        tl.store(
            dq_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            dq0,
            mask=m_mask[:, None] & d0_mask[None, :],
        )
        tl.store(
            dq_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            dq1,
            mask=m_mask[:, None] & d1_mask[None, :],
        )

    @triton.jit
    def _platonic_flat_attention_bwd_dkv_kernel_d128_tail(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        atom_type_ptr,
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        type_bias_ptr,
        centers_ptr,
        gamma_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_gate_ptr,
        d_type_bias_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
        cutoff: tl.constexpr,
        max_atomic_number: tl.constexpr,
        diag_zero: tl.constexpr,
        input_precision: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D_TAIL: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)
        start = tl.load(cu_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_ptr + batch_idx + 1).to(tl.int64)
        seqlen = end - start
        n_start = pid_n * BLOCK_N
        if n_start >= seqlen:
            return
        offs_m = tl.arange(0, BLOCK_M)
        n = n_start + tl.arange(0, BLOCK_N)
        offs_d0 = tl.arange(0, 128)
        offs_dt = tl.arange(0, BLOCK_D_TAIL)
        offs_d1 = 128 + offs_dt
        n_mask = n < seqlen
        d0_mask = offs_d0 < head_dim
        d1_mask = offs_d1 < head_dim

        k0 = tl.load(
            k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            mask=n_mask[:, None] & d0_mask[None, :],
            other=0.0,
        )
        k1 = tl.load(
            k_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            mask=n_mask[:, None] & d1_mask[None, :],
            other=0.0,
        )
        v0 = tl.load(
            v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            mask=n_mask[:, None] & d0_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            mask=n_mask[:, None] & d1_mask[None, :],
            other=0.0,
        )
        dk0 = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
        dk1 = tl.zeros((BLOCK_N, BLOCK_D_TAIL), dtype=tl.float32)
        dv0 = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
        dv1 = tl.zeros((BLOCK_N, BLOCK_D_TAIL), dtype=tl.float32)
        subhead = head_idx % heads_per_frame
        gate = tl.load(gate_ptr + subhead).to(tl.float32) if has_rbf and bias_mode != 4 else 0.0
        gate_factor = 1.0 + gate
        for block_m in range(0, tl.cdiv(max_seqlen, BLOCK_M)):
            m_start = block_m * BLOCK_M
            if m_start < seqlen:
                m = m_start + offs_m
                m_mask = m < seqlen
                q0 = tl.load(
                    q_ptr + ((start + m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
                    mask=m_mask[:, None] & d0_mask[None, :],
                    other=0.0,
                )
                q1 = tl.load(
                    q_ptr + ((start + m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
                    mask=m_mask[:, None] & d1_mask[None, :],
                    other=0.0,
                )
                dout0 = tl.load(
                    dout_ptr + ((start + m[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
                    mask=m_mask[:, None] & d0_mask[None, :],
                    other=0.0,
                )
                dout1 = tl.load(
                    dout_ptr + ((start + m[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
                    mask=m_mask[:, None] & d1_mask[None, :],
                    other=0.0,
                )
                lse = tl.load(lse_ptr + (start + m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
                delta = tl.load(delta_ptr + (start + m) * num_heads + head_idx, mask=m_mask, other=0.0).to(tl.float32)
                scores = tl.dot(q0, tl.trans(k0), input_precision=input_precision).to(tl.float32)
                scores += tl.dot(q1, tl.trans(k1), input_precision=input_precision).to(tl.float32)
                scores *= scale
                base_bias = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                if has_rbf:
                    pos_i_base = (start + m) * 3
                    pos_j_base = (start + n) * 3
                    xi = tl.load(pos_ptr + pos_i_base + 0, mask=m_mask, other=0.0)[:, None]
                    yi = tl.load(pos_ptr + pos_i_base + 1, mask=m_mask, other=0.0)[:, None]
                    zi = tl.load(pos_ptr + pos_i_base + 2, mask=m_mask, other=0.0)[:, None]
                    xj = tl.load(pos_ptr + pos_j_base + 0, mask=n_mask, other=0.0)[None, :]
                    yj = tl.load(pos_ptr + pos_j_base + 1, mask=n_mask, other=0.0)[None, :]
                    zj = tl.load(pos_ptr + pos_j_base + 2, mask=n_mask, other=0.0)[None, :]
                    dx = xj - xi
                    dy = yj - yi
                    dz = zj - zi
                    dist2 = tl.maximum(dx * dx + dy * dy + dz * dz, 0.0)
                    if bias_mode == 1:
                        dist = tl.sqrt(dist2)
                        gamma = tl.load(gamma_ptr).to(tl.float32)
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center))
                            base_bias += weight * rho
                    elif bias_mode == 2:
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
                        base_bias = weight * dist2
                    elif bias_mode == 3:
                        dist = tl.sqrt(dist2)
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
                        base_bias = weight * dist
                    elif bias_mode == 4:
                        dist = tl.sqrt(dist2)
                        x = dist / cutoff
                        x = tl.minimum(tl.maximum(x, 0.0), 1.0)
                        x2 = x * x
                        x3 = x2 * x
                        env = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
                        env = tl.where(dist < cutoff, env, 0.0)
                        gamma = tl.load(gamma_ptr).to(tl.float32)
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                            base_bias += weight * rho
                        zi_atom = tl.load(atom_type_ptr + start + m, mask=m_mask, other=0)[:, None]
                        zj_atom = tl.load(atom_type_ptr + start + n, mask=n_mask, other=0)[None, :]
                        zi_atom = tl.minimum(tl.maximum(zi_atom, 0), max_atomic_number)
                        zj_atom = tl.minimum(tl.maximum(zj_atom, 0), max_atomic_number)
                        zdim = max_atomic_number + 1
                        type_index = ((zi_atom * zdim + zj_atom) * heads_per_frame + subhead)
                        type_term = tl.load(
                            type_bias_ptr + type_index,
                            mask=m_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        base_bias += env * type_term
                    bias = base_bias if bias_mode == 4 else base_bias * gate_factor
                    if diag_zero:
                        bias = tl.where(m[:, None] == n[None, :], 0.0, bias)
                    scores += bias
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
                dp = tl.dot(dout0, tl.trans(v0), input_precision=input_precision).to(tl.float32)
                dp += tl.dot(dout1, tl.trans(v1), input_precision=input_precision).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dk0 += tl.dot(tl.trans(ds.to(q0.dtype)), q0, input_precision=input_precision) * scale
                dk1 += tl.dot(tl.trans(ds.to(q1.dtype)), q1, input_precision=input_precision) * scale
                dv0 += tl.dot(tl.trans(p.to(dout0.dtype)), dout0, input_precision=input_precision)
                dv1 += tl.dot(tl.trans(p.to(dout1.dtype)), dout1, input_precision=input_precision)
                if has_rbf:
                    if diag_zero:
                        ds_bias = tl.where(m[:, None] == n[None, :], 0.0, ds)
                    else:
                        ds_bias = ds
                    if bias_mode != 4:
                        tl.atomic_add(d_gate_ptr + subhead, tl.sum(tl.sum(ds_bias * base_bias, axis=0), axis=0), sem="relaxed")
                    if bias_mode == 1:
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center))
                            grad_w = tl.sum(tl.sum(ds_bias * rho * gate_factor, axis=0), axis=0)
                            tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
                    elif bias_mode == 2:
                        grad_w = tl.sum(tl.sum(ds_bias * dist2 * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf, grad_w, sem="relaxed")
                    elif bias_mode == 3:
                        grad_w = tl.sum(tl.sum(ds_bias * dist * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf, grad_w, sem="relaxed")
                    elif bias_mode == 4:
                        for rb in range(0, num_rbf):
                            center = tl.load(centers_ptr + rb).to(tl.float32)
                            rho = tl.exp(-gamma * (dist - center) * (dist - center)) * env
                            grad_w = tl.sum(tl.sum(ds_bias * rho, axis=0), axis=0)
                            tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
                        tl.atomic_add(
                            d_type_bias_ptr + type_index,
                            ds_bias * env,
                            sem="relaxed",
                            mask=m_mask[:, None] & n_mask[None, :],
                        )
        tl.store(
            dk_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            dk0,
            mask=n_mask[:, None] & d0_mask[None, :],
        )
        tl.store(
            dk_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            dk1,
            mask=n_mask[:, None] & d1_mask[None, :],
        )
        tl.store(
            dv_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d0[None, :],
            dv0,
            mask=n_mask[:, None] & d0_mask[None, :],
        )
        tl.store(
            dv_ptr + ((start + n[:, None]) * num_heads + head_idx) * head_dim + offs_d1[None, :],
            dv1,
            mask=n_mask[:, None] & d1_mask[None, :],
        )


def _platonic_flat_attention_backward_return(
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    d_rbf_weight: torch.Tensor | None = None,
    d_gate: torch.Tensor | None = None,
    d_type_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, ...]:
    return (
        dq,
        dk,
        dv,
        None,
        None,
        None,
        None,
        d_rbf_weight,
        d_gate,
        d_type_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


class _PlatonicFlatAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rbf_weight: torch.Tensor,
        gate: torch.Tensor,
        type_bias: torch.Tensor,
        centers: torch.Tensor,
        gamma: torch.Tensor,
        cutoff: float,
        max_atomic_number: int,
        heads_per_frame: int,
        diag_zero: bool,
        precision: str,
        block_m: int,
        block_n: int,
        has_rbf: bool,
        bias_mode: int,
    ) -> torch.Tensor:
        precision = normalize_platonic_attention_precision(precision)
        input_precision = _kernel_input_precision(precision)
        bias_mode = normalize_platonic_attention_bias_mode(bias_mode, has_bias=has_rbf)
        _validate_flat_inputs(q, k, v, cu_seqlens)
        if not TRITON_PLATONIC_ATTENTION_AVAILABLE or not q.is_cuda:
            return platonic_attention_flat_torch_reference(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                pos=pos if has_rbf else None,
                atom_types=atom_types if has_rbf else None,
                heads_per_frame=heads_per_frame if has_rbf else None,
                rbf_weight=rbf_weight if has_rbf else None,
                gate=gate if has_rbf else None,
                type_bias=type_bias if has_rbf and _is_rbf_type_bias_mode(bias_mode) else None,
                centers=centers if has_rbf else None,
                gamma=gamma if has_rbf else None,
                cutoff=cutoff if has_rbf else None,
                radial_bias_kind=bias_mode,
                diag_zero=diag_zero,
            )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pos = pos.contiguous()
        atom_types = atom_types.contiguous()
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        rbf_weight = rbf_weight.contiguous()
        gate = gate.contiguous()
        type_bias = type_bias.contiguous()
        centers = centers.contiguous()
        gamma = gamma.contiguous()
        total_tokens, num_heads, head_dim = q.shape
        batch_size = int(cu.numel() - 1)
        max_seqlen = int(max_seqlen)
        out = torch.empty_like(q)
        lse = torch.empty((total_tokens, num_heads), device=q.device, dtype=torch.float32)
        block_d = platonic_attention_block_d_for_head_dim(head_dim)
        split_tail = platonic_attention_split_head_dim_tail(head_dim)
        fwd_num_stages = _triton_meta_int_env("MATTERFORMER_PLATONIC_TRITON_FWD_NUM_STAGES", 3, min_value=1, max_value=8)
        grid = (triton.cdiv(max(max_seqlen, 1), int(block_m)), num_heads, batch_size)
        if split_tail is not None:
            _platonic_flat_attention_fwd_kernel_d128_tail[grid](
                q,
                k,
                v,
                pos,
                atom_types,
                cu,
                rbf_weight,
                gate,
                type_bias,
                centers,
                gamma,
                out,
                lse,
                num_heads,
                head_dim,
                max_seqlen,
                1.0 / math.sqrt(head_dim),
                int(heads_per_frame),
                int(rbf_weight.shape[-1]) if has_rbf else 1,
                bool(has_rbf),
                int(bias_mode),
                float(cutoff),
                int(max_atomic_number),
                bool(diag_zero),
                input_precision,
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                BLOCK_D_TAIL=int(split_tail),
                num_warps=4,
                num_stages=fwd_num_stages,
            )
        else:
            _platonic_flat_attention_fwd_kernel[grid](
                q,
                k,
                v,
                pos,
                atom_types,
                cu,
                rbf_weight,
                gate,
                type_bias,
                centers,
                gamma,
                out,
                lse,
                num_heads,
                head_dim,
                max_seqlen,
                1.0 / math.sqrt(head_dim),
                int(heads_per_frame),
                int(rbf_weight.shape[-1]) if has_rbf else 1,
                bool(has_rbf),
                int(bias_mode),
                float(cutoff),
                int(max_atomic_number),
                bool(diag_zero),
                input_precision,
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                BLOCK_D=block_d,
                num_warps=4,
                num_stages=fwd_num_stages,
            )
        ctx.save_for_backward(q, k, v, pos, atom_types, cu, out, lse, rbf_weight, gate, type_bias, centers, gamma)
        ctx.heads_per_frame = int(heads_per_frame)
        ctx.cutoff = float(cutoff)
        ctx.max_atomic_number = int(max_atomic_number)
        ctx.diag_zero = bool(diag_zero)
        ctx.precision = input_precision
        ctx.bias_mode = int(bias_mode)
        ctx.block_m = int(block_m)
        ctx.block_n = int(block_n)
        ctx.max_seqlen = max_seqlen
        ctx.has_rbf = bool(has_rbf)
        ctx.split_head_dim_tail = int(split_tail) if split_tail is not None else None
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, pos, atom_types, cu, out, lse, rbf_weight, gate, type_bias, centers, gamma = ctx.saved_tensors
        if not TRITON_PLATONIC_ATTENTION_AVAILABLE or not q.is_cuda:
            with torch.enable_grad():
                q_ref = q.detach().requires_grad_(True)
                k_ref = k.detach().requires_grad_(True)
                v_ref = v.detach().requires_grad_(True)
                weight_ref = rbf_weight.detach().requires_grad_(ctx.has_rbf)
                is_rbf_type = _is_rbf_type_bias_mode(ctx.bias_mode)
                gate_ref = gate.detach().requires_grad_(ctx.has_rbf and not is_rbf_type)
                type_ref = type_bias.detach().requires_grad_(ctx.has_rbf and is_rbf_type)
                ref_out = platonic_attention_flat_torch_reference(
                    q_ref,
                    k_ref,
                    v_ref,
                    cu_seqlens=cu,
                    pos=pos if ctx.has_rbf else None,
                    atom_types=atom_types if ctx.has_rbf else None,
                    heads_per_frame=ctx.heads_per_frame if ctx.has_rbf else None,
                    rbf_weight=weight_ref if ctx.has_rbf else None,
                    gate=gate_ref if ctx.has_rbf else None,
                    type_bias=type_ref if ctx.has_rbf and is_rbf_type else None,
                    centers=centers if ctx.has_rbf else None,
                    gamma=gamma if ctx.has_rbf else None,
                    cutoff=ctx.cutoff if ctx.has_rbf else None,
                    radial_bias_kind=ctx.bias_mode,
                    diag_zero=ctx.diag_zero,
                )
                grad_inputs = (
                    (q_ref, k_ref, v_ref, weight_ref, type_ref)
                    if ctx.has_rbf and is_rbf_type
                    else (q_ref, k_ref, v_ref, weight_ref, gate_ref)
                    if ctx.has_rbf
                    else (q_ref, k_ref, v_ref)
                )
                grads = torch.autograd.grad(
                    ref_out,
                    grad_inputs,
                    dout,
                    allow_unused=True,
                )
            if ctx.has_rbf and is_rbf_type:
                dq, dk, dv, dw, dt = grads
                dg = None
            elif ctx.has_rbf:
                dq, dk, dv, dw, dg = grads
                dt = None
            else:
                dq, dk, dv = grads
                dw = dg = dt = None
            return _platonic_flat_attention_backward_return(dq, dk, dv, dw, dg, dt)

        dout = dout.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        d_rbf_weight = torch.zeros_like(rbf_weight) if ctx.has_rbf else torch.zeros_like(rbf_weight)
        d_gate = torch.zeros_like(gate) if ctx.has_rbf else torch.zeros_like(gate)
        d_type_bias = torch.zeros_like(type_bias) if ctx.has_rbf else torch.zeros_like(type_bias)
        block_d = platonic_attention_block_d_for_head_dim(q.shape[-1])
        split_tail = getattr(ctx, "split_head_dim_tail", None)
        default_bwd_stages = 1 if block_d >= 256 else 2
        bwd_num_stages = _triton_meta_int_env(
            "MATTERFORMER_PLATONIC_TRITON_BWD_NUM_STAGES",
            default_bwd_stages,
            min_value=1,
            max_value=8,
        )
        delta_mode = os.environ.get("MATTERFORMER_PLATONIC_TRITON_DELTA", "triton_vector").lower()
        if delta_mode == "triton":
            delta = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
            if split_tail is not None:
                _platonic_flat_attention_bwd_preprocess_kernel_d128_tail[(q.shape[0], q.shape[1])](
                    out,
                    dout,
                    delta,
                    q.shape[0],
                    q.shape[1],
                    q.shape[2],
                    BLOCK_D_TAIL=int(split_tail),
                    num_warps=1,
                )
            else:
                _platonic_flat_attention_bwd_preprocess_kernel[(q.shape[0], q.shape[1])](
                    out,
                    dout,
                    delta,
                    q.shape[0],
                    q.shape[1],
                    q.shape[2],
                    BLOCK_D=block_d,
                    num_warps=1,
                )
        elif delta_mode == "torch":
            delta = torch.sum(out.float() * dout.float(), dim=-1).contiguous()
        else:
            delta = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
            block_pairs = int(os.environ.get("MATTERFORMER_PLATONIC_TRITON_DELTA_BLOCK_PAIRS", "128"))
            total_pairs = int(q.shape[0] * q.shape[1])
            if split_tail is not None:
                _platonic_flat_attention_bwd_preprocess_vector_kernel_d128_tail[(triton.cdiv(max(total_pairs, 1), block_pairs),)](
                    out,
                    dout,
                    delta,
                    total_pairs,
                    q.shape[2],
                    BLOCK_P=block_pairs,
                    BLOCK_D_TAIL=int(split_tail),
                    num_warps=4,
                )
            else:
                _platonic_flat_attention_bwd_preprocess_vector_kernel[(triton.cdiv(max(total_pairs, 1), block_pairs),)](
                    out,
                    dout,
                    delta,
                    total_pairs,
                    q.shape[2],
                    BLOCK_P=block_pairs,
                    BLOCK_D=block_d,
                    num_warps=4,
                )
        batch_size = int(cu.numel() - 1)
        max_seqlen = int(ctx.max_seqlen)
        bwd_mode = os.environ.get("MATTERFORMER_PLATONIC_TRITON_BWD", "auto").lower()
        if bwd_mode not in {"auto", "atomic", "split"}:
            bwd_mode = "auto"
        auto_atomic_allowed = block_d <= 128 and split_tail is None
        use_atomic_bwd = bwd_mode == "atomic" or (
            bwd_mode == "auto"
            and auto_atomic_allowed
            and ctx.has_rbf
            and ctx.bias_mode
            not in {
                PLATONIC_ATTENTION_BIAS_MODES["radial_rbf"],
                PLATONIC_ATTENTION_BIAS_MODES["rbf_type_enveloped"],
                PLATONIC_ATTENTION_BIAS_MODES["radius_rbf_type_enveloped"],
            }
        )
        if use_atomic_bwd:
            dk.zero_()
            dv.zero_()
            grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_m), q.shape[1], batch_size)
            _platonic_flat_attention_bwd_kernel[grid](
                q,
                k,
                v,
                pos,
                atom_types,
                cu,
                dout,
                lse,
                delta,
                rbf_weight,
                gate,
                type_bias,
                centers,
                gamma,
                dq,
                dk,
                dv,
                d_rbf_weight,
                d_gate,
                d_type_bias,
                q.shape[1],
                q.shape[2],
                max_seqlen,
                1.0 / math.sqrt(q.shape[2]),
                ctx.heads_per_frame,
                int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
                ctx.has_rbf,
                ctx.bias_mode,
                ctx.cutoff,
                ctx.max_atomic_number,
                ctx.diag_zero,
                ctx.precision,
                BLOCK_M=ctx.block_m,
                BLOCK_N=ctx.block_n,
                BLOCK_D=block_d,
                num_warps=4,
                num_stages=bwd_num_stages,
            )
            return _platonic_flat_attention_backward_return(
                dq,
                dk,
                dv,
                d_rbf_weight if ctx.has_rbf else None,
                d_gate if ctx.has_rbf else None,
                d_type_bias if ctx.has_rbf and _is_rbf_type_bias_mode(ctx.bias_mode) else None,
            )
        if split_tail is not None:
            dq_grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_m), q.shape[1], batch_size)
            _platonic_flat_attention_bwd_dq_kernel_d128_tail[dq_grid](
                q,
                k,
                v,
                pos,
                atom_types,
                cu,
                dout,
                lse,
                delta,
                rbf_weight,
                gate,
                type_bias,
                centers,
                gamma,
                dq,
                q.shape[1],
                q.shape[2],
                max_seqlen,
                1.0 / math.sqrt(q.shape[2]),
                ctx.heads_per_frame,
                int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
                ctx.has_rbf,
                ctx.bias_mode,
                ctx.cutoff,
                ctx.max_atomic_number,
                ctx.diag_zero,
                ctx.precision,
                BLOCK_M=ctx.block_m,
                BLOCK_N=ctx.block_n,
                BLOCK_D_TAIL=int(split_tail),
                num_warps=4,
                num_stages=bwd_num_stages,
            )
            dkv_grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_n), q.shape[1], batch_size)
            _platonic_flat_attention_bwd_dkv_kernel_d128_tail[dkv_grid](
                q,
                k,
                v,
                pos,
                atom_types,
                cu,
                dout,
                lse,
                delta,
                rbf_weight,
                gate,
                type_bias,
                centers,
                gamma,
                dk,
                dv,
                d_rbf_weight,
                d_gate,
                d_type_bias,
                q.shape[1],
                q.shape[2],
                max_seqlen,
                1.0 / math.sqrt(q.shape[2]),
                ctx.heads_per_frame,
                int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
                ctx.has_rbf,
                ctx.bias_mode,
                ctx.cutoff,
                ctx.max_atomic_number,
                ctx.diag_zero,
                ctx.precision,
                BLOCK_M=ctx.block_m,
                BLOCK_N=ctx.block_n,
                BLOCK_D_TAIL=int(split_tail),
                num_warps=4,
                num_stages=bwd_num_stages,
            )
            return _platonic_flat_attention_backward_return(
                dq,
                dk,
                dv,
                d_rbf_weight if ctx.has_rbf else None,
                d_gate if ctx.has_rbf else None,
                d_type_bias if ctx.has_rbf and _is_rbf_type_bias_mode(ctx.bias_mode) else None,
            )
        dq_grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_m), q.shape[1], batch_size)
        _platonic_flat_attention_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            pos,
            atom_types,
            cu,
            dout,
            lse,
            delta,
            rbf_weight,
            gate,
            type_bias,
            centers,
            gamma,
            dq,
            q.shape[1],
            q.shape[2],
            max_seqlen,
            1.0 / math.sqrt(q.shape[2]),
            ctx.heads_per_frame,
            int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
            ctx.has_rbf,
            ctx.bias_mode,
            ctx.cutoff,
            ctx.max_atomic_number,
            ctx.diag_zero,
            ctx.precision,
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=bwd_num_stages,
        )
        dkv_grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_n), q.shape[1], batch_size)
        _platonic_flat_attention_bwd_dkv_kernel[dkv_grid](
            q,
            k,
            v,
            pos,
            atom_types,
            cu,
            dout,
            lse,
            delta,
            rbf_weight,
            gate,
            type_bias,
            centers,
            gamma,
            dk,
            dv,
            d_rbf_weight,
            d_gate,
            d_type_bias,
            q.shape[1],
            q.shape[2],
            max_seqlen,
            1.0 / math.sqrt(q.shape[2]),
            ctx.heads_per_frame,
            int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
            ctx.has_rbf,
            ctx.bias_mode,
            ctx.cutoff,
            ctx.max_atomic_number,
            ctx.diag_zero,
            ctx.precision,
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=bwd_num_stages,
        )
        return _platonic_flat_attention_backward_return(
            dq,
            dk,
            dv,
            d_rbf_weight if ctx.has_rbf else None,
            d_gate if ctx.has_rbf else None,
            d_type_bias if ctx.has_rbf and _is_rbf_type_bias_mode(ctx.bias_mode) else None,
        )


@_disable_dynamo_if_available
def platonic_attention_flat_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    pos: torch.Tensor | None = None,
    atom_types: torch.Tensor | None = None,
    heads_per_frame: int | None = None,
    rbf_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    type_bias: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    cutoff: float | None = None,
    max_atomic_number: int | None = None,
    diag_zero: bool = True,
    radial_bias_kind: str | int | None = None,
    precision: str = "tf32",
    block_m: int = 16,
    block_n: int = 32,
    strict: bool = False,
) -> torch.Tensor:
    _validate_flat_inputs(q, k, v, cu_seqlens)
    if max_seqlen == 0 or q.shape[0] == 0:
        return torch.zeros_like(q)
    normalized_precision = normalize_platonic_attention_precision(precision)
    flash_compat_bf16 = normalized_precision == "bf16_flash_compat"
    kernel_precision = _kernel_input_precision(normalized_precision)
    use_rbf = rbf_weight is not None
    bias_mode = normalize_platonic_attention_bias_mode(radial_bias_kind, has_bias=use_rbf)
    use_rbf_type = use_rbf and _is_rbf_type_bias_mode(bias_mode)
    if use_rbf and bias_mode == PLATONIC_ATTENTION_BIAS_MODES["radius_rbf_type_enveloped"] and q.is_cuda:
        raise ValueError(
            "radius_rbf_type_enveloped is a torch reference backend today; "
            "the Triton radius-sparse kernel is not wired yet"
        )
    if use_rbf:
        if pos is None or heads_per_frame is None or centers is None or gamma is None:
            raise ValueError("Platonic radial Triton attention requires pos, heads_per_frame, centers, and gamma")
        if not use_rbf_type and gate is None:
            raise ValueError("Platonic legacy radial Triton attention requires gate")
        if use_rbf_type:
            if cutoff is None:
                raise ValueError("Platonic RBF/type Triton attention requires cutoff")
            if atom_types is None:
                if type_bias is not None:
                    raise ValueError("Platonic RBF/type Triton attention with type_bias requires atom_types")
                atom_types = torch.zeros((q.shape[0],), device=q.device, dtype=torch.long)
            elif atom_types.ndim != 1 or atom_types.shape[0] != q.shape[0]:
                raise ValueError(f"atom_types must have shape [{q.shape[0]}], got {tuple(atom_types.shape)}")
            if type_bias is None:
                if heads_per_frame is None:
                    raise ValueError("heads_per_frame is required when type_bias is omitted")
                zdim = int(max_atomic_number if max_atomic_number is not None else int(atom_types.detach().max().item())) + 1
                type_bias = torch.zeros(
                    (zdim, zdim, int(heads_per_frame)),
                    device=q.device,
                    dtype=torch.float32,
                )
            if type_bias.ndim != 3 or type_bias.shape[-1] != int(heads_per_frame):
                raise ValueError("type_bias must have shape [max_z + 1, max_z + 1, heads_per_frame]")
        if q.shape[1] % int(heads_per_frame) != 0:
            raise ValueError("heads_per_frame must divide num_heads for Platonic radial Triton attention")
        if rbf_weight.ndim != 2 or rbf_weight.shape[0] != int(heads_per_frame):
            raise ValueError("rbf_weight must have shape [heads_per_frame, num_basis]")
        if gate is not None and gate.shape != (int(heads_per_frame),):
            raise ValueError("gate must have shape [heads_per_frame]")
        if pos.ndim != 2 or pos.shape != (q.shape[0], 3):
            raise ValueError(f"pos must have shape [{q.shape[0]}, 3], got {tuple(pos.shape)}")
        if centers.ndim != 1:
            raise ValueError("centers must be a 1D tensor")
        if centers.numel() != rbf_weight.shape[1]:
            raise ValueError("centers length must match rbf_weight.shape[1]")
    if strict and q.is_cuda and not TRITON_PLATONIC_ATTENTION_AVAILABLE:
        raise RuntimeError("attention_backend='triton' requested, but Triton is not available")
    if not q.is_cuda or not TRITON_PLATONIC_ATTENTION_AVAILABLE:
        return platonic_attention_flat_torch_reference(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            pos=pos if use_rbf else None,
            atom_types=atom_types if use_rbf_type else None,
            heads_per_frame=heads_per_frame if use_rbf else None,
            rbf_weight=rbf_weight,
            gate=gate,
            type_bias=type_bias if use_rbf_type else None,
            centers=centers,
            gamma=gamma,
            cutoff=cutoff if use_rbf_type else None,
            radial_bias_kind=bias_mode,
            diag_zero=diag_zero,
        )
    if int(block_m) < 16 or int(block_n) < 16:
        raise ValueError(
            "Platonic Triton attention uses tl.dot tiles; block_m and block_n must be >= 16. "
            "Use 16x16 or 16x32 for the head_dim=160 radial path."
        )
    q_apply = q.to(torch.bfloat16) if flash_compat_bf16 else q
    k_apply = k.to(torch.bfloat16) if flash_compat_bf16 else k
    v_apply = v.to(torch.bfloat16) if flash_compat_bf16 else v
    orig_dtype = q.dtype
    dummy_weight = rbf_weight if rbf_weight is not None else torch.empty((1, 1), device=q.device, dtype=torch.float32)
    dummy_gate = gate if gate is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
    dummy_type_bias = type_bias if type_bias is not None else torch.empty((1, 1, 1), device=q.device, dtype=torch.float32)
    dummy_centers = centers if centers is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
    dummy_gamma = gamma if gamma is not None else torch.ones((), device=q.device, dtype=torch.float32)
    dummy_pos = pos if pos is not None else torch.empty((q.shape[0], 3), device=q.device, dtype=torch.float32)
    dummy_atom_types = atom_types if atom_types is not None else torch.empty((q.shape[0],), device=q.device, dtype=torch.int64)
    if (
        q.requires_grad
        or k.requires_grad
        or v.requires_grad
        or (rbf_weight is not None and rbf_weight.requires_grad)
        or (type_bias is not None and type_bias.requires_grad)
    ):
        out = _PlatonicFlatAttentionFunction.apply(
            q_apply,
            k_apply,
            v_apply,
            dummy_pos,
            dummy_atom_types,
            cu_seqlens,
            int(max_seqlen),
            dummy_weight,
            dummy_gate,
            dummy_type_bias,
            dummy_centers,
            dummy_gamma,
            float(cutoff or 1.0),
            int(max_atomic_number if max_atomic_number is not None else (dummy_type_bias.shape[0] - 1)),
            int(heads_per_frame or q.shape[1]),
            bool(diag_zero),
            kernel_precision,
            int(block_m),
            int(block_n),
            bool(use_rbf),
            int(bias_mode),
        )
        return out.to(orig_dtype) if flash_compat_bf16 else out
    out = _PlatonicFlatAttentionFunction.apply(
        q_apply,
        k_apply,
        v_apply,
        dummy_pos,
        dummy_atom_types,
        cu_seqlens,
        int(max_seqlen),
        dummy_weight,
        dummy_gate,
        dummy_type_bias,
        dummy_centers,
        dummy_gamma,
        float(cutoff or 1.0),
        int(max_atomic_number if max_atomic_number is not None else (dummy_type_bias.shape[0] - 1)),
        int(heads_per_frame or q.shape[1]),
        bool(diag_zero),
        kernel_precision,
        int(block_m),
        int(block_n),
        bool(use_rbf),
        int(bias_mode),
    )
    return out.to(orig_dtype) if flash_compat_bf16 else out
