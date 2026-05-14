from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA nodes with Triton installed.
    triton = None
    tl = None

TRITON_COMPACT_SIMPLICIAL_AVAILABLE = triton is not None and tl is not None
SUPPORTED_COMPACT_SIMPLICIAL_PRECISIONS = ("bf16_tc", "tf32", "ieee_fp32")


def normalize_compact_simplicial_precision(precision: str) -> str:
    normalized = str(precision).lower()
    if normalized not in SUPPORTED_COMPACT_SIMPLICIAL_PRECISIONS:
        raise ValueError(
            f"Unsupported compact simplicial precision: {precision}. "
            f"Expected one of {SUPPORTED_COMPACT_SIMPLICIAL_PRECISIONS}."
        )
    return normalized


def compact_block_d_for_head_dim(head_dim: int) -> int:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if head_dim <= 16:
        return 16
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    raise ValueError(f"Compact Triton simplicial attention only supports head_dim <= 128, got {head_dim}")


def compact_block_k_for_neighbors(k_neighbors: int) -> int:
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if k_neighbors <= 16:
        return 16
    if k_neighbors <= 32:
        return 32
    if k_neighbors <= 64:
        return 64
    raise ValueError(f"Compact Triton simplicial attention only supports k_neighbors <= 64, got {k_neighbors}")


def compact_block_r_for_rank(rank: int) -> int:
    if rank <= 0:
        return 1
    if rank <= 16:
        return 16
    if rank <= 32:
        return 32
    if rank <= 64:
        return 64
    raise ValueError(f"Compact Triton simplicial attention only supports low-rank features <= 64, got {rank}")


def _precision_kernel_config(precision: str) -> str:
    normalized = normalize_compact_simplicial_precision(precision)
    if normalized == "ieee_fp32":
        return "ieee"
    return "tf32"


def _gather_neighbor_heads(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, num_atoms, channels = values.shape
    _, _, num_neighbors = neighbor_idx.shape
    idx = neighbor_idx[:, None, :, :, None].expand(batch_size, num_heads, num_atoms, num_neighbors, channels)
    source = values[:, :, None, :, :].expand(batch_size, num_heads, num_atoms, num_atoms, channels)
    return torch.gather(source, dim=3, index=idx)


def _gate_for_scores_torch(gate: torch.Tensor, *, num_heads: int) -> torch.Tensor:
    if gate.ndim == 1:
        if gate.shape[0] != num_heads:
            raise ValueError(f"per-head gate has {gate.shape[0]} heads, expected {num_heads}")
        return gate.view(1, num_heads, 1, 1, 1).float()
    if gate.ndim == 3:
        if gate.shape[1] != num_heads:
            raise ValueError(f"gate has {gate.shape[1]} heads, expected {num_heads}")
        return gate[:, :, :, None, None].float()
    raise ValueError(f"gate must be [H] or [B,H,N], got shape {tuple(gate.shape)}")


def _compact_angle_coefficients_as_bhnkc(
    *,
    unit: torch.Tensor,
    angle_left_coeff: torch.Tensor,
    angle_right_coeff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_atoms, k_neighbors, _ = unit.shape
    if angle_left_coeff.shape != angle_right_coeff.shape:
        raise ValueError(
            "compact angle coefficient tensors must have matching shapes, got "
            f"{tuple(angle_left_coeff.shape)} and {tuple(angle_right_coeff.shape)}"
        )
    if angle_left_coeff.ndim != 5:
        raise ValueError(f"compact angle coefficients must be rank-5, got {tuple(angle_left_coeff.shape)}")
    if angle_left_coeff.shape[0] != batch_size:
        raise ValueError(
            f"compact angle coefficient batch dim {angle_left_coeff.shape[0]} does not match {batch_size}"
        )
    if angle_left_coeff.shape[2] == num_atoms and angle_left_coeff.shape[3] == k_neighbors:
        return angle_left_coeff, angle_right_coeff
    if angle_left_coeff.shape[1] == num_atoms and angle_left_coeff.shape[2] == k_neighbors:
        return angle_left_coeff.permute(0, 3, 1, 2, 4), angle_right_coeff.permute(0, 3, 1, 2, 4)
    raise ValueError(
        "compact angle coefficients must be [B,H,N,K,C] or [B,N,K,H,C], got "
        f"{tuple(angle_left_coeff.shape)} for unit shape {tuple(unit.shape)}"
    )


def _compact_angle_coeff_canonical_strides(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    num_atoms: int,
    k_neighbors: int,
) -> tuple[int, int, int, int, int]:
    if tensor.numel() == 0:
        return (0, 0, 0, 0, 0)
    if tensor.ndim != 5:
        raise ValueError(f"compact angle coefficients must be rank-5, got {tuple(tensor.shape)}")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"compact angle coefficient batch dim {tensor.shape[0]} does not match {batch_size}")
    if tensor.shape[1] == num_heads and tensor.shape[2] == num_atoms and tensor.shape[3] == k_neighbors:
        return (tensor.stride(0), tensor.stride(2), tensor.stride(3), tensor.stride(1), tensor.stride(4))
    if tensor.shape[1] == num_atoms and tensor.shape[2] == k_neighbors and tensor.shape[3] == num_heads:
        return (tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3), tensor.stride(4))
    raise ValueError(
        "compact angle coefficients must be [B,H,N,K,C] or [B,N,K,H,C], got "
        f"{tuple(tensor.shape)} for B={batch_size}, H={num_heads}, N={num_atoms}, K={k_neighbors}"
    )


def _compact_angle_from_coefficients_torch(
    *,
    unit: torch.Tensor,
    angle_left_coeff: torch.Tensor,
    angle_right_coeff: torch.Tensor,
    angle_channels_by_l: tuple[int, int, int],
    angle_rank: int,
) -> torch.Tensor:
    angle_left_coeff, angle_right_coeff = _compact_angle_coefficients_as_bhnkc(
        unit=unit,
        angle_left_coeff=angle_left_coeff,
        angle_right_coeff=angle_right_coeff,
    )
    c0, c1, c2 = (int(v) for v in angle_channels_by_l)
    if c0 + c1 + c2 != int(angle_left_coeff.shape[-1]):
        raise ValueError(
            "angle_channels_by_l does not match compact coefficient width: "
            f"{(c0, c1, c2)} vs {angle_left_coeff.shape[-1]}"
        )
    cos = torch.einsum("bnjc,bnkc->bnjk", unit.float(), unit.float())
    p2 = 0.5 * (3.0 * cos.square() - 1.0)
    pieces: list[torch.Tensor] = []
    offset = 0
    if c0 > 0:
        left = angle_left_coeff[..., offset : offset + c0].float()
        right = angle_right_coeff[..., offset : offset + c0].float()
        pieces.append(torch.einsum("bhnjc,bhnkc->bhnjk", left, right))
        offset += c0
    if c1 > 0:
        left = angle_left_coeff[..., offset : offset + c1].float()
        right = angle_right_coeff[..., offset : offset + c1].float()
        pieces.append(torch.einsum("bhnjc,bhnkc->bhnjk", left, right) * cos[:, None])
        offset += c1
    if c2 > 0:
        left = angle_left_coeff[..., offset : offset + c2].float()
        right = angle_right_coeff[..., offset : offset + c2].float()
        pieces.append(torch.einsum("bhnjc,bhnkc->bhnjk", left, right) * p2[:, None])
    if not pieces:
        return angle_left_coeff.new_zeros(*angle_left_coeff.shape[:3], angle_left_coeff.shape[3], angle_left_coeff.shape[3])
    return sum(pieces) * (int(angle_rank) ** -0.5)


def compact_simplicial_attention_torch_reference(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    u: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    angle_left: torch.Tensor | None = None,
    angle_right: torch.Tensor | None = None,
    unit: torch.Tensor | None = None,
    angle_left_coeff: torch.Tensor | None = None,
    angle_right_coeff: torch.Tensor | None = None,
    angle_channels_by_l: tuple[int, int, int] | None = None,
    angle_rank: int | None = None,
    angle_gate: torch.Tensor | None = None,
    message_left: torch.Tensor | None = None,
    message_right: torch.Tensor | None = None,
    message_basis: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    q_scale: float = 1.0,
) -> torch.Tensor:
    k1_n = _gather_neighbor_heads(k1, neighbor_idx)
    v1_n = _gather_neighbor_heads(v1, neighbor_idx)
    k2_n = _gather_neighbor_heads(k2, neighbor_idx)
    v2_n = _gather_neighbor_heads(v2, neighbor_idx)
    q_eff = q * float(q_scale)
    scores = torch.einsum("bhnd,bhnjd,bhnkd->bhnjk", q_eff, k1_n, k2_n).float()
    if u is not None and v_bias is not None and gate is not None:
        scores = scores + _gate_for_scores_torch(gate, num_heads=q.shape[1]) * (
            u[:, :, :, :, None].float() + v_bias[:, :, :, None, :].float()
        )
    if angle_left is not None and angle_right is not None:
        local_angle_gate = 1.0 if angle_gate is None else _gate_for_scores_torch(angle_gate, num_heads=q.shape[1])
        angle = torch.einsum("bhnjr,bhnkr->bhnjk", angle_left.float(), angle_right.float())
        scores = scores + local_angle_gate * angle * (angle_left.shape[-1] ** -0.5)
    if angle_left_coeff is not None or angle_right_coeff is not None:
        if unit is None or angle_left_coeff is None or angle_right_coeff is None or angle_channels_by_l is None:
            raise ValueError("compact angle coefficients require unit, both coefficient tensors, and angle_channels_by_l")
        local_rank = int(angle_rank or (angle_channels_by_l[0] + 3 * angle_channels_by_l[1] + 5 * angle_channels_by_l[2]))
        local_angle_gate = 1.0 if angle_gate is None else _gate_for_scores_torch(angle_gate, num_heads=q.shape[1])
        angle = _compact_angle_from_coefficients_torch(
            unit=unit,
            angle_left_coeff=angle_left_coeff,
            angle_right_coeff=angle_right_coeff,
            angle_channels_by_l=angle_channels_by_l,
            angle_rank=local_rank,
        )
        scores = scores + local_angle_gate * angle
    valid = neighbor_mask[:, None, :, :, None] & neighbor_mask[:, None, :, None, :]
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores.flatten(-2), dim=-1).view_as(scores)
    attn = torch.where(valid, attn, torch.zeros_like(attn))
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p, training=training)
    tmp = torch.einsum("bhnjk,bhnjd->bhnkd", attn.to(v1_n.dtype), v1_n)
    out = (tmp * v2_n).sum(dim=-2)
    if message_left is not None and message_right is not None and message_basis is not None:
        coeff = torch.einsum(
            "bhnjk,bhnjr,bhnkr->bhnr",
            attn.float(),
            message_left.float(),
            message_right.float(),
        ) * (message_left.shape[-1] ** -0.5)
        out = out + torch.einsum("bhnr,hrd->bhnd", coeff, message_basis.float()).to(dtype=out.dtype)
    return out


if TRITON_COMPACT_SIMPLICIAL_AVAILABLE:

    @triton.jit
    def _compact_forward_kernel(
        q_ptr,
        k1_ptr,
        v1_ptr,
        k2_ptr,
        v2_ptr,
        neighbor_idx_ptr,
        neighbor_mask_ptr,
        unit_ptr,
        u_ptr,
        v_bias_ptr,
        gate_ptr,
        angle_left_ptr,
        angle_right_ptr,
        angle_left_coeff_ptr,
        angle_right_coeff_ptr,
        angle_gate_ptr,
        message_left_ptr,
        message_right_ptr,
        message_basis_ptr,
        out_ptr,
        lse_ptr,
        num_heads,
        num_atoms,
        q_scale,
        q_stride_b,
        q_stride_h,
        q_stride_n,
        q_stride_d,
        k1_stride_b,
        k1_stride_h,
        k1_stride_n,
        k1_stride_d,
        v1_stride_b,
        v1_stride_h,
        v1_stride_n,
        v1_stride_d,
        k2_stride_b,
        k2_stride_h,
        k2_stride_n,
        k2_stride_d,
        v2_stride_b,
        v2_stride_h,
        v2_stride_n,
        v2_stride_d,
        angle_left_coeff_stride_b,
        angle_left_coeff_stride_n,
        angle_left_coeff_stride_k,
        angle_left_coeff_stride_h,
        angle_left_coeff_stride_c,
        angle_right_coeff_stride_b,
        angle_right_coeff_stride_n,
        angle_right_coeff_stride_k,
        angle_right_coeff_stride_h,
        angle_right_coeff_stride_c,
        head_dim: tl.constexpr,
        k_neighbors: tl.constexpr,
        angle_rank: tl.constexpr,
        message_rank: tl.constexpr,
        HAS_RADIAL_BIAS: tl.constexpr,
        HAS_ANGLE: tl.constexpr,
        HAS_COMPACT_ANGLE: tl.constexpr,
        HAS_MESSAGE: tl.constexpr,
        GATE_HEAD_ONLY: tl.constexpr,
        ANGLE_GATE_HEAD_ONLY: tl.constexpr,
        OUTPUT_LAYOUT_BNHD: tl.constexpr,
        ANGLE_C0: tl.constexpr,
        ANGLE_C1: tl.constexpr,
        ANGLE_C2: tl.constexpr,
        ANGLE_COEFFS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        atom_idx = pid % num_atoms
        bh_idx = pid // num_atoms
        batch_idx = bh_idx // num_heads
        head_idx = bh_idx - batch_idx * num_heads

        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, BLOCK_R)
        offs_c = tl.arange(0, BLOCK_C)
        k_mask = offs_k < k_neighbors
        d_mask = offs_d < head_dim

        n_base = (batch_idx * num_atoms + atom_idx) * k_neighbors
        raw_idx = tl.load(neighbor_idx_ptr + n_base + offs_k, mask=k_mask, other=0)
        valid = (tl.load(neighbor_mask_ptr + n_base + offs_k, mask=k_mask, other=0) > 0) & k_mask
        safe_idx = tl.where(valid, raw_idx, 0)

        q = tl.load(
            q_ptr
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + atom_idx * q_stride_n
            + offs_d * q_stride_d,
            mask=d_mask,
            other=0.0,
        )
        q = q * q_scale
        k1 = tl.load(
            k1_ptr
            + batch_idx * k1_stride_b
            + head_idx * k1_stride_h
            + safe_idx[:, None] * k1_stride_n
            + offs_d[None, :] * k1_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            k2_ptr
            + batch_idx * k2_stride_b
            + head_idx * k2_stride_h
            + safe_idx[:, None] * k2_stride_n
            + offs_d[None, :] * k2_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v1_ptr
            + batch_idx * v1_stride_b
            + head_idx * v1_stride_h
            + safe_idx[:, None] * v1_stride_n
            + offs_d[None, :] * v1_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v2 = tl.load(
            v2_ptr
            + batch_idx * v2_stride_b
            + head_idx * v2_stride_h
            + safe_idx[:, None] * v2_stride_n
            + offs_d[None, :] * v2_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        q_dot = q.to(k1.dtype)
        score = tl.dot(k1 * q_dot[None, :], tl.trans(k2), input_precision=INPUT_PRECISION).to(tl.float32)

        if HAS_RADIAL_BIAS:
            bias_base = (bh_idx * num_atoms + atom_idx) * k_neighbors
            u = tl.load(u_ptr + bias_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            vb = tl.load(v_bias_ptr + bias_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            if GATE_HEAD_ONLY:
                gate = tl.load(gate_ptr + head_idx).to(tl.float32)
            else:
                gate = tl.load(gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += gate * (u[:, None] + vb[None, :])

        if HAS_ANGLE:
            r_mask = offs_r < angle_rank
            angle_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * angle_rank
            left = tl.load(
                angle_left_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            right = tl.load(
                angle_right_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            angle = tl.dot(left, tl.trans(right), input_precision=INPUT_PRECISION) * tl.rsqrt(angle_rank + 0.0)
            if ANGLE_GATE_HEAD_ONLY:
                agate = tl.load(angle_gate_ptr + head_idx).to(tl.float32)
            else:
                agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle

        if HAS_COMPACT_ANGLE:
            unit_base = ((batch_idx * num_atoms + atom_idx) * k_neighbors + offs_k) * 3
            ux = tl.load(unit_ptr + unit_base + 0, mask=k_mask, other=0.0).to(tl.float32)
            uy = tl.load(unit_ptr + unit_base + 1, mask=k_mask, other=0.0).to(tl.float32)
            uz = tl.load(unit_ptr + unit_base + 2, mask=k_mask, other=0.0).to(tl.float32)
            cos = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :] + uz[:, None] * uz[None, :]
            p2 = 0.5 * (3.0 * cos * cos - 1.0)
            left_coeff_base = (
                batch_idx * angle_left_coeff_stride_b
                + atom_idx * angle_left_coeff_stride_n
                + head_idx * angle_left_coeff_stride_h
            )
            right_coeff_base = (
                batch_idx * angle_right_coeff_stride_b
                + atom_idx * angle_right_coeff_stride_n
                + head_idx * angle_right_coeff_stride_h
            )
            angle_unscaled = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
            if ANGLE_C0 > 0:
                c_mask = offs_c < ANGLE_C0
                left0 = tl.load(
                    angle_left_coeff_ptr
                    + left_coeff_base
                    + offs_k[:, None] * angle_left_coeff_stride_k
                    + offs_c[None, :] * angle_left_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right0 = tl.load(
                    angle_right_coeff_ptr
                    + right_coeff_base
                    + offs_k[:, None] * angle_right_coeff_stride_k
                    + offs_c[None, :] * angle_right_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left0, tl.trans(right0), input_precision=INPUT_PRECISION)
            if ANGLE_C1 > 0:
                c_mask = offs_c < ANGLE_C1
                c_offs = ANGLE_C0 + offs_c
                left1 = tl.load(
                    angle_left_coeff_ptr
                    + left_coeff_base
                    + offs_k[:, None] * angle_left_coeff_stride_k
                    + c_offs[None, :] * angle_left_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right1 = tl.load(
                    angle_right_coeff_ptr
                    + right_coeff_base
                    + offs_k[:, None] * angle_right_coeff_stride_k
                    + c_offs[None, :] * angle_right_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left1, tl.trans(right1), input_precision=INPUT_PRECISION) * cos
            if ANGLE_C2 > 0:
                c_mask = offs_c < ANGLE_C2
                c_offs = ANGLE_C0 + ANGLE_C1 + offs_c
                left2 = tl.load(
                    angle_left_coeff_ptr
                    + left_coeff_base
                    + offs_k[:, None] * angle_left_coeff_stride_k
                    + c_offs[None, :] * angle_left_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right2 = tl.load(
                    angle_right_coeff_ptr
                    + right_coeff_base
                    + offs_k[:, None] * angle_right_coeff_stride_k
                    + c_offs[None, :] * angle_right_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left2, tl.trans(right2), input_precision=INPUT_PRECISION) * p2
            angle = angle_unscaled * tl.rsqrt(angle_rank + 0.0)
            if ANGLE_GATE_HEAD_ONLY:
                agate = tl.load(angle_gate_ptr + head_idx).to(tl.float32)
            else:
                agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle

        pair_valid = valid[:, None] & valid[None, :]
        valid_count = tl.sum(tl.sum(pair_valid.to(tl.int32), axis=0), axis=0)
        has_valid = valid_count > 0
        neg_large = -3.4028234663852886e38
        score = tl.where(pair_valid, score, neg_large)
        max_cols = tl.max(score, axis=0)
        row_max = tl.max(max_cols, axis=0)
        row_max = tl.where(has_valid, row_max, 0.0)
        exp_score = tl.where(pair_valid, tl.exp(score - row_max), 0.0)
        denom_cols = tl.sum(exp_score, axis=0)
        denom = tl.sum(denom_cols, axis=0)
        denom_safe = tl.where(has_valid, denom, 1.0)
        probs = tl.where(has_valid, exp_score / denom_safe, 0.0)

        tmp = tl.dot(tl.trans(probs), v1, input_precision=INPUT_PRECISION)
        out = tl.sum(tmp * v2, axis=0).to(tl.float32)

        if HAS_MESSAGE:
            r_mask = offs_r < message_rank
            msg_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * message_rank
            ml = tl.load(
                message_left_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            mr = tl.load(
                message_right_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            msg_tmp = tl.dot(tl.trans(probs), ml, input_precision=INPUT_PRECISION)
            coeff = tl.sum(msg_tmp * mr, axis=0) * tl.rsqrt(message_rank + 0.0)
            basis = tl.load(
                message_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                mask=r_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            out += tl.sum(coeff[:, None] * basis, axis=0)

        out = tl.where(has_valid, out, 0.0)
        if OUTPUT_LAYOUT_BNHD:
            out_offset = ((batch_idx * num_atoms + atom_idx) * num_heads + head_idx) * head_dim
        else:
            out_offset = (bh_idx * num_atoms + atom_idx) * head_dim
        tl.store(out_ptr + out_offset + offs_d, out, mask=d_mask)
        lse = tl.where(has_valid, row_max + tl.log(denom), 0.0)
        tl.store(lse_ptr + bh_idx * num_atoms + atom_idx, lse)

    @triton.jit
    def _compact_backward_kernel(
        grad_out_ptr,
        q_ptr,
        k1_ptr,
        v1_ptr,
        k2_ptr,
        v2_ptr,
        neighbor_idx_ptr,
        neighbor_mask_ptr,
        unit_ptr,
        u_ptr,
        v_bias_ptr,
        gate_ptr,
        angle_left_ptr,
        angle_right_ptr,
        angle_left_coeff_ptr,
        angle_right_coeff_ptr,
        angle_gate_ptr,
        message_left_ptr,
        message_right_ptr,
        message_basis_ptr,
        out_ptr,
        lse_ptr,
        dq_ptr,
        dk1_ptr,
        dv1_ptr,
        dk2_ptr,
        dv2_ptr,
        du_ptr,
        dv_bias_ptr,
        dgate_ptr,
        dangle_left_ptr,
        dangle_right_ptr,
        dangle_left_coeff_ptr,
        dangle_right_coeff_ptr,
        dangle_gate_ptr,
        dmessage_left_ptr,
        dmessage_right_ptr,
        dmessage_basis_ptr,
        num_heads,
        num_atoms,
        q_scale,
        q_stride_b,
        q_stride_h,
        q_stride_n,
        q_stride_d,
        k1_stride_b,
        k1_stride_h,
        k1_stride_n,
        k1_stride_d,
        v1_stride_b,
        v1_stride_h,
        v1_stride_n,
        v1_stride_d,
        k2_stride_b,
        k2_stride_h,
        k2_stride_n,
        k2_stride_d,
        v2_stride_b,
        v2_stride_h,
        v2_stride_n,
        v2_stride_d,
        angle_left_coeff_stride_b,
        angle_left_coeff_stride_n,
        angle_left_coeff_stride_k,
        angle_left_coeff_stride_h,
        angle_left_coeff_stride_c,
        angle_right_coeff_stride_b,
        angle_right_coeff_stride_n,
        angle_right_coeff_stride_k,
        angle_right_coeff_stride_h,
        angle_right_coeff_stride_c,
        dangle_left_coeff_stride_b,
        dangle_left_coeff_stride_n,
        dangle_left_coeff_stride_k,
        dangle_left_coeff_stride_h,
        dangle_left_coeff_stride_c,
        dangle_right_coeff_stride_b,
        dangle_right_coeff_stride_n,
        dangle_right_coeff_stride_k,
        dangle_right_coeff_stride_h,
        dangle_right_coeff_stride_c,
        head_dim: tl.constexpr,
        k_neighbors: tl.constexpr,
        angle_rank: tl.constexpr,
        message_rank: tl.constexpr,
        HAS_RADIAL_BIAS: tl.constexpr,
        HAS_ANGLE: tl.constexpr,
        HAS_COMPACT_ANGLE: tl.constexpr,
        HAS_MESSAGE: tl.constexpr,
        GATE_HEAD_ONLY: tl.constexpr,
        ANGLE_GATE_HEAD_ONLY: tl.constexpr,
        OUTPUT_LAYOUT_BNHD: tl.constexpr,
        ANGLE_C0: tl.constexpr,
        ANGLE_C1: tl.constexpr,
        ANGLE_C2: tl.constexpr,
        ANGLE_COEFFS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        atom_idx = pid % num_atoms
        bh_idx = pid // num_atoms
        batch_idx = bh_idx // num_heads
        head_idx = bh_idx - batch_idx * num_heads

        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, BLOCK_R)
        offs_c = tl.arange(0, BLOCK_C)
        k_mask = offs_k < k_neighbors
        d_mask = offs_d < head_dim

        n_base = (batch_idx * num_atoms + atom_idx) * k_neighbors
        raw_idx = tl.load(neighbor_idx_ptr + n_base + offs_k, mask=k_mask, other=0)
        valid = (tl.load(neighbor_mask_ptr + n_base + offs_k, mask=k_mask, other=0) > 0) & k_mask
        safe_idx = tl.where(valid, raw_idx, 0)

        q = tl.load(
            q_ptr
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + atom_idx * q_stride_n
            + offs_d * q_stride_d,
            mask=d_mask,
            other=0.0,
        )
        q = q * q_scale
        if OUTPUT_LAYOUT_BNHD:
            out_offset = ((batch_idx * num_atoms + atom_idx) * num_heads + head_idx) * head_dim
        else:
            out_offset = (bh_idx * num_atoms + atom_idx) * head_dim
        go = tl.load(grad_out_ptr + out_offset + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + out_offset + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)

        k1 = tl.load(
            k1_ptr
            + batch_idx * k1_stride_b
            + head_idx * k1_stride_h
            + safe_idx[:, None] * k1_stride_n
            + offs_d[None, :] * k1_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            k2_ptr
            + batch_idx * k2_stride_b
            + head_idx * k2_stride_h
            + safe_idx[:, None] * k2_stride_n
            + offs_d[None, :] * k2_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v1_ptr
            + batch_idx * v1_stride_b
            + head_idx * v1_stride_h
            + safe_idx[:, None] * v1_stride_n
            + offs_d[None, :] * v1_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v2 = tl.load(
            v2_ptr
            + batch_idx * v2_stride_b
            + head_idx * v2_stride_h
            + safe_idx[:, None] * v2_stride_n
            + offs_d[None, :] * v2_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        q_dot = q.to(k1.dtype)
        score = tl.dot(k1 * q_dot[None, :], tl.trans(k2), input_precision=INPUT_PRECISION).to(tl.float32)
        q = q.to(tl.float32)
        k1 = k1.to(tl.float32)
        k2 = k2.to(tl.float32)
        v1 = v1.to(tl.float32)
        v2 = v2.to(tl.float32)
        gate = 0.0
        u = tl.zeros((BLOCK_K,), dtype=tl.float32)
        vb = tl.zeros((BLOCK_K,), dtype=tl.float32)
        if HAS_RADIAL_BIAS:
            bias_base = (bh_idx * num_atoms + atom_idx) * k_neighbors
            u = tl.load(u_ptr + bias_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            vb = tl.load(v_bias_ptr + bias_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            if GATE_HEAD_ONLY:
                gate = tl.load(gate_ptr + head_idx).to(tl.float32)
            else:
                gate = tl.load(gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += gate * (u[:, None] + vb[None, :])

        angle = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        agate = 0.0
        left = tl.zeros((BLOCK_K, BLOCK_R), dtype=tl.float32)
        right = tl.zeros((BLOCK_K, BLOCK_R), dtype=tl.float32)
        compact_cos = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        compact_p2 = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        left0 = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)
        right0 = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)
        left1 = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)
        right1 = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)
        left2 = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)
        right2 = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)
        if HAS_ANGLE:
            r_mask = offs_r < angle_rank
            angle_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * angle_rank
            left = tl.load(
                angle_left_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            right = tl.load(
                angle_right_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            angle = tl.dot(left, tl.trans(right), input_precision=INPUT_PRECISION) * tl.rsqrt(angle_rank + 0.0)
            if ANGLE_GATE_HEAD_ONLY:
                agate = tl.load(angle_gate_ptr + head_idx).to(tl.float32)
            else:
                agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle

        if HAS_COMPACT_ANGLE:
            unit_base = ((batch_idx * num_atoms + atom_idx) * k_neighbors + offs_k) * 3
            ux = tl.load(unit_ptr + unit_base + 0, mask=k_mask, other=0.0).to(tl.float32)
            uy = tl.load(unit_ptr + unit_base + 1, mask=k_mask, other=0.0).to(tl.float32)
            uz = tl.load(unit_ptr + unit_base + 2, mask=k_mask, other=0.0).to(tl.float32)
            compact_cos = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :] + uz[:, None] * uz[None, :]
            compact_p2 = 0.5 * (3.0 * compact_cos * compact_cos - 1.0)
            left_coeff_base = (
                batch_idx * angle_left_coeff_stride_b
                + atom_idx * angle_left_coeff_stride_n
                + head_idx * angle_left_coeff_stride_h
            )
            right_coeff_base = (
                batch_idx * angle_right_coeff_stride_b
                + atom_idx * angle_right_coeff_stride_n
                + head_idx * angle_right_coeff_stride_h
            )
            angle_unscaled = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
            if ANGLE_C0 > 0:
                c_mask = offs_c < ANGLE_C0
                left0 = tl.load(
                    angle_left_coeff_ptr
                    + left_coeff_base
                    + offs_k[:, None] * angle_left_coeff_stride_k
                    + offs_c[None, :] * angle_left_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right0 = tl.load(
                    angle_right_coeff_ptr
                    + right_coeff_base
                    + offs_k[:, None] * angle_right_coeff_stride_k
                    + offs_c[None, :] * angle_right_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left0, tl.trans(right0), input_precision=INPUT_PRECISION)
            if ANGLE_C1 > 0:
                c_mask = offs_c < ANGLE_C1
                c_offs = ANGLE_C0 + offs_c
                left1 = tl.load(
                    angle_left_coeff_ptr
                    + left_coeff_base
                    + offs_k[:, None] * angle_left_coeff_stride_k
                    + c_offs[None, :] * angle_left_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right1 = tl.load(
                    angle_right_coeff_ptr
                    + right_coeff_base
                    + offs_k[:, None] * angle_right_coeff_stride_k
                    + c_offs[None, :] * angle_right_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left1, tl.trans(right1), input_precision=INPUT_PRECISION) * compact_cos
            if ANGLE_C2 > 0:
                c_mask = offs_c < ANGLE_C2
                c_offs = ANGLE_C0 + ANGLE_C1 + offs_c
                left2 = tl.load(
                    angle_left_coeff_ptr
                    + left_coeff_base
                    + offs_k[:, None] * angle_left_coeff_stride_k
                    + c_offs[None, :] * angle_left_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right2 = tl.load(
                    angle_right_coeff_ptr
                    + right_coeff_base
                    + offs_k[:, None] * angle_right_coeff_stride_k
                    + c_offs[None, :] * angle_right_coeff_stride_c,
                    mask=valid[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left2, tl.trans(right2), input_precision=INPUT_PRECISION) * compact_p2
            angle = angle_unscaled * tl.rsqrt(angle_rank + 0.0)
            if ANGLE_GATE_HEAD_ONLY:
                agate = tl.load(angle_gate_ptr + head_idx).to(tl.float32)
            else:
                agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle

        pair_valid = valid[:, None] & valid[None, :]
        neg_large = -3.4028234663852886e38
        score = tl.where(pair_valid, score, neg_large)
        probs = tl.where(pair_valid, tl.exp(score - lse), 0.0)

        d_pair = tl.dot(v1 * go[None, :], tl.trans(v2), input_precision=INPUT_PRECISION).to(tl.float32)
        ml = tl.zeros((BLOCK_K, BLOCK_R), dtype=tl.float32)
        mr = tl.zeros((BLOCK_K, BLOCK_R), dtype=tl.float32)
        basis_go = tl.zeros((BLOCK_R,), dtype=tl.float32)
        if HAS_MESSAGE:
            r_mask = offs_r < message_rank
            msg_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * message_rank
            ml = tl.load(
                message_left_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            mr = tl.load(
                message_right_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                mask=k_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            basis = tl.load(
                message_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                mask=r_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            basis_go = tl.sum(basis * go[None, :], axis=1)
            weighted_ml = ml * basis_go[None, :]
            d_pair += tl.dot(weighted_ml, tl.trans(mr), input_precision=INPUT_PRECISION) * tl.rsqrt(message_rank + 0.0)

            msg_tmp = tl.dot(tl.trans(probs), ml, input_precision=INPUT_PRECISION)
            coeff = tl.sum(msg_tmp * mr, axis=0) * tl.rsqrt(message_rank + 0.0)
            d_basis = coeff[:, None] * go[None, :]
            tl.atomic_add(
                dmessage_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                d_basis,
                sem="relaxed",
                mask=r_mask[:, None] & d_mask[None, :],
            )
            dml = tl.dot(probs, mr, input_precision=INPUT_PRECISION) * (basis_go[None, :] * tl.rsqrt(message_rank + 0.0))
            dmr = tl.dot(tl.trans(probs), ml, input_precision=INPUT_PRECISION) * (basis_go[None, :] * tl.rsqrt(message_rank + 0.0))
            tl.store(
                dmessage_left_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                dml,
                mask=k_mask[:, None] & r_mask[None, :],
            )
            tl.store(
                dmessage_right_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                dmr,
                mask=k_mask[:, None] & r_mask[None, :],
            )

        center = tl.sum(go * out, axis=0)
        ds = probs * (d_pair - center)
        ds = tl.where(pair_valid, ds, 0.0)

        tmp_q = tl.dot(ds, k2, input_precision=INPUT_PRECISION)
        dq = tl.sum(tmp_q * k1, axis=0) * q_scale
        tl.store(dq_ptr + (bh_idx * num_atoms + atom_idx) * head_dim + offs_d, dq, mask=d_mask)

        dk1 = tl.dot(ds, k2, input_precision=INPUT_PRECISION) * q[None, :]
        dk2 = tl.dot(tl.trans(ds), k1, input_precision=INPUT_PRECISION) * q[None, :]
        dv1 = tl.dot(probs, v2, input_precision=INPUT_PRECISION) * go[None, :]
        dv2 = tl.dot(tl.trans(probs), v1, input_precision=INPUT_PRECISION) * go[None, :]

        tl.atomic_add(
            dk1_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dk1,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            dk2_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dk2,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            dv1_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dv1,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            dv2_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dv2,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )

        if HAS_RADIAL_BIAS:
            bias_base = (bh_idx * num_atoms + atom_idx) * k_neighbors
            du = gate * tl.sum(ds, axis=1)
            dvb = gate * tl.sum(ds, axis=0)
            dgate = tl.sum(tl.sum(ds * (u[:, None] + vb[None, :]), axis=0), axis=0)
            tl.store(du_ptr + bias_base + offs_k, du, mask=k_mask)
            tl.store(dv_bias_ptr + bias_base + offs_k, dvb, mask=k_mask)
            if GATE_HEAD_ONLY:
                tl.atomic_add(dgate_ptr + head_idx, dgate, sem="relaxed")
            else:
                tl.store(dgate_ptr + bh_idx * num_atoms + atom_idx, dgate)

        if HAS_ANGLE:
            r_mask = offs_r < angle_rank
            angle_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * angle_rank
            scale = agate * tl.rsqrt(angle_rank + 0.0)
            dleft = tl.dot(ds, right, input_precision=INPUT_PRECISION) * scale
            dright = tl.dot(tl.trans(ds), left, input_precision=INPUT_PRECISION) * scale
            dangle_gate = tl.sum(tl.sum(ds * angle, axis=0), axis=0)
            tl.store(
                dangle_left_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                dleft,
                mask=k_mask[:, None] & r_mask[None, :],
            )
            tl.store(
                dangle_right_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                dright,
                mask=k_mask[:, None] & r_mask[None, :],
            )
            if ANGLE_GATE_HEAD_ONLY:
                tl.atomic_add(dangle_gate_ptr + head_idx, dangle_gate, sem="relaxed")
            else:
                tl.store(dangle_gate_ptr + bh_idx * num_atoms + atom_idx, dangle_gate)

        if HAS_COMPACT_ANGLE:
            dleft_coeff_base = (
                batch_idx * dangle_left_coeff_stride_b
                + atom_idx * dangle_left_coeff_stride_n
                + head_idx * dangle_left_coeff_stride_h
            )
            dright_coeff_base = (
                batch_idx * dangle_right_coeff_stride_b
                + atom_idx * dangle_right_coeff_stride_n
                + head_idx * dangle_right_coeff_stride_h
            )
            scale = agate * tl.rsqrt(angle_rank + 0.0)
            if ANGLE_C0 > 0:
                c_mask = offs_c < ANGLE_C0
                weighted = ds * scale
                dleft0 = tl.dot(weighted, right0, input_precision=INPUT_PRECISION)
                dright0 = tl.dot(tl.trans(weighted), left0, input_precision=INPUT_PRECISION)
                tl.store(
                    dangle_left_coeff_ptr
                    + dleft_coeff_base
                    + offs_k[:, None] * dangle_left_coeff_stride_k
                    + offs_c[None, :] * dangle_left_coeff_stride_c,
                    dleft0,
                    mask=k_mask[:, None] & c_mask[None, :],
                )
                tl.store(
                    dangle_right_coeff_ptr
                    + dright_coeff_base
                    + offs_k[:, None] * dangle_right_coeff_stride_k
                    + offs_c[None, :] * dangle_right_coeff_stride_c,
                    dright0,
                    mask=k_mask[:, None] & c_mask[None, :],
                )
            if ANGLE_C1 > 0:
                c_mask = offs_c < ANGLE_C1
                c_offs = ANGLE_C0 + offs_c
                weighted = ds * (scale * compact_cos)
                dleft1 = tl.dot(weighted, right1, input_precision=INPUT_PRECISION)
                dright1 = tl.dot(tl.trans(weighted), left1, input_precision=INPUT_PRECISION)
                tl.store(
                    dangle_left_coeff_ptr
                    + dleft_coeff_base
                    + offs_k[:, None] * dangle_left_coeff_stride_k
                    + c_offs[None, :] * dangle_left_coeff_stride_c,
                    dleft1,
                    mask=k_mask[:, None] & c_mask[None, :],
                )
                tl.store(
                    dangle_right_coeff_ptr
                    + dright_coeff_base
                    + offs_k[:, None] * dangle_right_coeff_stride_k
                    + c_offs[None, :] * dangle_right_coeff_stride_c,
                    dright1,
                    mask=k_mask[:, None] & c_mask[None, :],
                )
            if ANGLE_C2 > 0:
                c_mask = offs_c < ANGLE_C2
                c_offs = ANGLE_C0 + ANGLE_C1 + offs_c
                weighted = ds * (scale * compact_p2)
                dleft2 = tl.dot(weighted, right2, input_precision=INPUT_PRECISION)
                dright2 = tl.dot(tl.trans(weighted), left2, input_precision=INPUT_PRECISION)
                tl.store(
                    dangle_left_coeff_ptr
                    + dleft_coeff_base
                    + offs_k[:, None] * dangle_left_coeff_stride_k
                    + c_offs[None, :] * dangle_left_coeff_stride_c,
                    dleft2,
                    mask=k_mask[:, None] & c_mask[None, :],
                )
                tl.store(
                    dangle_right_coeff_ptr
                    + dright_coeff_base
                    + offs_k[:, None] * dangle_right_coeff_stride_k
                    + c_offs[None, :] * dangle_right_coeff_stride_c,
                    dright2,
                    mask=k_mask[:, None] & c_mask[None, :],
                )
            dangle_gate = tl.sum(tl.sum(ds * angle, axis=0), axis=0)
            if ANGLE_GATE_HEAD_ONLY:
                tl.atomic_add(dangle_gate_ptr + head_idx, dangle_gate, sem="relaxed")
            else:
                tl.store(dangle_gate_ptr + bh_idx * num_atoms + atom_idx, dangle_gate)


class _TritonCompactSimplicialAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k1: torch.Tensor,
        v1: torch.Tensor,
        k2: torch.Tensor,
        v2: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        unit: torch.Tensor,
        u: torch.Tensor,
        v_bias: torch.Tensor,
        gate: torch.Tensor,
        angle_left: torch.Tensor,
        angle_right: torch.Tensor,
        angle_left_coeff: torch.Tensor,
        angle_right_coeff: torch.Tensor,
        angle_gate: torch.Tensor,
        message_left: torch.Tensor,
        message_right: torch.Tensor,
        message_basis: torch.Tensor,
        precision: str,
        debug_torch_backward: bool,
        q_scale: float,
        angle_c0: int,
        angle_c1: int,
        angle_c2: int,
        angle_coeffs: int,
        compact_angle_rank: int,
        output_layout_bnhd: bool,
    ) -> torch.Tensor:
        if not TRITON_COMPACT_SIMPLICIAL_AVAILABLE:  # pragma: no cover - guarded before dispatch.
            raise RuntimeError("triton is not installed")
        batch_size, num_heads, num_atoms, head_dim = q.shape
        k_neighbors = int(neighbor_idx.shape[-1])
        has_radial_bias = u.numel() > 0 and v_bias.numel() > 0 and gate.numel() > 0
        has_expanded_angle = angle_left.numel() > 0 and angle_right.numel() > 0
        has_compact_angle = angle_left_coeff.numel() > 0 and angle_right_coeff.numel() > 0
        has_message = message_left.numel() > 0 and message_right.numel() > 0 and message_basis.numel() > 0
        gate_head_only = has_radial_bias and gate.ndim == 1
        angle_gate_head_only = (has_expanded_angle or has_compact_angle) and angle_gate.ndim == 1
        angle_rank = (
            int(angle_left.shape[-1])
            if has_expanded_angle
            else int(compact_angle_rank)
            if has_compact_angle
            else 1
        )
        message_rank = int(message_left.shape[-1]) if has_message else 1
        angle_c0 = int(angle_c0)
        angle_c1 = int(angle_c1)
        angle_c2 = int(angle_c2)
        angle_coeffs = int(angle_coeffs)
        block_k = compact_block_k_for_neighbors(k_neighbors)
        block_d = compact_block_d_for_head_dim(head_dim)
        block_r = compact_block_r_for_rank(max(angle_rank if has_expanded_angle else 1, message_rank if has_message else 1))
        block_c = compact_block_r_for_rank(max(angle_c0, angle_c1, angle_c2, 1))
        input_precision = _precision_kernel_config(precision)
        left_coeff_strides = _compact_angle_coeff_canonical_strides(
            angle_left_coeff,
            batch_size=batch_size,
            num_heads=num_heads,
            num_atoms=num_atoms,
            k_neighbors=k_neighbors,
        )
        if has_compact_angle:
            right_coeff_strides = _compact_angle_coeff_canonical_strides(
                angle_right_coeff,
                batch_size=batch_size,
                num_heads=num_heads,
                num_atoms=num_atoms,
                k_neighbors=k_neighbors,
            )
        else:
            right_coeff_strides = (0, 0, 0, 0, 0)

        output_layout_bnhd = bool(output_layout_bnhd)
        out_shape = (
            (batch_size, num_atoms, num_heads, head_dim)
            if output_layout_bnhd
            else (batch_size, num_heads, num_atoms, head_dim)
        )
        out_fp32 = torch.empty(out_shape, device=q.device, dtype=torch.float32)
        lse = torch.empty((batch_size, num_heads, num_atoms), device=q.device, dtype=torch.float32)
        grid = (batch_size * num_heads * num_atoms,)
        _compact_forward_kernel[grid](
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx,
            neighbor_mask,
            unit,
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
            angle_left_coeff,
            angle_right_coeff,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out_fp32,
            lse,
            num_heads,
            num_atoms,
            float(q_scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k1.stride(0),
            k1.stride(1),
            k1.stride(2),
            k1.stride(3),
            v1.stride(0),
            v1.stride(1),
            v1.stride(2),
            v1.stride(3),
            k2.stride(0),
            k2.stride(1),
            k2.stride(2),
            k2.stride(3),
            v2.stride(0),
            v2.stride(1),
            v2.stride(2),
            v2.stride(3),
            left_coeff_strides[0],
            left_coeff_strides[1],
            left_coeff_strides[2],
            left_coeff_strides[3],
            left_coeff_strides[4],
            right_coeff_strides[0],
            right_coeff_strides[1],
            right_coeff_strides[2],
            right_coeff_strides[3],
            right_coeff_strides[4],
            head_dim=head_dim,
            k_neighbors=k_neighbors,
            angle_rank=angle_rank,
            message_rank=message_rank,
            HAS_RADIAL_BIAS=has_radial_bias,
            HAS_ANGLE=has_expanded_angle,
            HAS_COMPACT_ANGLE=has_compact_angle,
            HAS_MESSAGE=has_message,
            GATE_HEAD_ONLY=gate_head_only,
            ANGLE_GATE_HEAD_ONLY=angle_gate_head_only,
            OUTPUT_LAYOUT_BNHD=output_layout_bnhd,
            ANGLE_C0=angle_c0,
            ANGLE_C1=angle_c1,
            ANGLE_C2=angle_c2,
            ANGLE_COEFFS=angle_coeffs,
            INPUT_PRECISION=input_precision,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=4 if block_d <= 64 else 8,
        )
        ctx.save_for_backward(
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx,
            neighbor_mask,
            unit,
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
            angle_left_coeff,
            angle_right_coeff,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out_fp32,
            lse,
        )
        ctx.precision = str(precision)
        ctx.debug_torch_backward = bool(debug_torch_backward)
        ctx.q_scale = float(q_scale)
        ctx.has_radial_bias = has_radial_bias
        ctx.has_expanded_angle = has_expanded_angle
        ctx.has_compact_angle = has_compact_angle
        ctx.has_message = has_message
        ctx.gate_head_only = gate_head_only
        ctx.angle_gate_head_only = angle_gate_head_only
        ctx.output_layout_bnhd = output_layout_bnhd
        ctx.angle_rank = angle_rank
        ctx.message_rank = message_rank
        ctx.angle_c0 = angle_c0
        ctx.angle_c1 = angle_c1
        ctx.angle_c2 = angle_c2
        ctx.angle_coeffs = angle_coeffs
        return out_fp32.to(dtype=q.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx,
            neighbor_mask,
            unit,
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
            angle_left_coeff,
            angle_right_coeff,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out_fp32,
            lse,
        ) = ctx.saved_tensors
        if ctx.debug_torch_backward:
            inputs: list[torch.Tensor] = [
                q.detach().requires_grad_(True),
                k1.detach().requires_grad_(True),
                v1.detach().requires_grad_(True),
                k2.detach().requires_grad_(True),
                v2.detach().requires_grad_(True),
            ]
            cursor = 5
            if ctx.has_radial_bias:
                inputs.extend(
                    [
                        u.detach().requires_grad_(True),
                        v_bias.detach().requires_grad_(True),
                        gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_expanded_angle:
                inputs.extend(
                    [
                        angle_left.detach().requires_grad_(True),
                        angle_right.detach().requires_grad_(True),
                        angle_gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_compact_angle:
                inputs.extend(
                    [
                        angle_left_coeff.detach().requires_grad_(True),
                        angle_right_coeff.detach().requires_grad_(True),
                        angle_gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_message:
                inputs.extend(
                    [
                        message_left.detach().requires_grad_(True),
                        message_right.detach().requires_grad_(True),
                        message_basis.detach().requires_grad_(True),
                    ]
                )
            with torch.enable_grad():
                q_re, k1_re, v1_re, k2_re, v2_re = inputs[:5]
                local_u = local_v = local_gate = None
                local_left = local_right = local_angle_gate = None
                local_left_coeff = local_right_coeff = None
                local_compact_angle_gate = None
                local_msg_left = local_msg_right = local_msg_basis = None
                cursor = 5
                if ctx.has_radial_bias:
                    local_u, local_v, local_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_expanded_angle:
                    local_left, local_right, local_angle_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_compact_angle:
                    local_left_coeff, local_right_coeff, local_compact_angle_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_message:
                    local_msg_left, local_msg_right, local_msg_basis = inputs[cursor : cursor + 3]
                ref = compact_simplicial_attention_torch_reference(
                    q_re,
                    k1_re,
                    v1_re,
                    k2_re,
                    v2_re,
                    neighbor_idx=neighbor_idx,
                    neighbor_mask=neighbor_mask,
                    u=local_u,
                    v_bias=local_v,
                    gate=local_gate,
                    angle_left=local_left,
                    angle_right=local_right,
                    unit=unit,
                    angle_left_coeff=local_left_coeff,
                    angle_right_coeff=local_right_coeff,
                    angle_channels_by_l=(ctx.angle_c0, ctx.angle_c1, ctx.angle_c2) if ctx.has_compact_angle else None,
                    angle_rank=ctx.angle_rank if ctx.has_compact_angle else None,
                    angle_gate=local_angle_gate if ctx.has_expanded_angle else local_compact_angle_gate,
                    message_left=local_msg_left,
                    message_right=local_msg_right,
                    message_basis=local_msg_basis,
                    q_scale=ctx.q_scale,
                )
                if ctx.output_layout_bnhd:
                    ref = ref.transpose(1, 2).contiguous()
                grads = torch.autograd.grad(ref, inputs, grad_out, allow_unused=True)
            dq = dk1 = dv1 = dk2 = dv2 = du = dvb = dgate = dleft = dright = dleft_coeff = dright_coeff = dagate = dml = dmr = dmb = None
            dq, dk1, dv1, dk2, dv2 = grads[:5]
            cursor = 5
            if ctx.has_radial_bias:
                du, dvb, dgate = grads[cursor : cursor + 3]
                cursor += 3
            if ctx.has_expanded_angle:
                dleft, dright, dagate = grads[cursor : cursor + 3]
                cursor += 3
            if ctx.has_compact_angle:
                dleft_coeff, dright_coeff, dagate = grads[cursor : cursor + 3]
                cursor += 3
            if ctx.has_message:
                dml, dmr, dmb = grads[cursor : cursor + 3]
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
                dvb,
                dgate,
                dleft,
                dright,
                dleft_coeff,
                dright_coeff,
                dagate,
                dml,
                dmr,
                dmb,
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

        batch_size, num_heads, num_atoms, head_dim = q.shape
        k_neighbors = int(neighbor_idx.shape[-1])
        block_k = compact_block_k_for_neighbors(k_neighbors)
        block_d = compact_block_d_for_head_dim(head_dim)
        block_r = compact_block_r_for_rank(max(ctx.angle_rank if ctx.has_expanded_angle else 1, ctx.message_rank if ctx.has_message else 1))
        block_c = compact_block_r_for_rank(max(ctx.angle_c0, ctx.angle_c1, ctx.angle_c2, 1))
        input_precision = _precision_kernel_config(ctx.precision)
        grad_out = grad_out.contiguous()
        grad_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        dq = torch.empty(q.shape, device=q.device, dtype=grad_dtype)
        dk1 = torch.zeros(k1.shape, device=k1.device, dtype=grad_dtype)
        dv1 = torch.zeros(v1.shape, device=v1.device, dtype=grad_dtype)
        dk2 = torch.zeros(k2.shape, device=k2.device, dtype=grad_dtype)
        dv2 = torch.zeros(v2.shape, device=v2.device, dtype=grad_dtype)
        du = torch.empty_like(u, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(u)
        dvb = torch.empty_like(v_bias, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(v_bias)
        dgate = (
            torch.zeros_like(gate, dtype=grad_dtype)
            if ctx.has_radial_bias and ctx.gate_head_only
            else torch.empty_like(gate, dtype=grad_dtype)
            if ctx.has_radial_bias
            else torch.empty_like(gate)
        )
        dleft = torch.empty_like(angle_left, dtype=grad_dtype) if ctx.has_expanded_angle else torch.empty_like(angle_left)
        dright = torch.empty_like(angle_right, dtype=grad_dtype) if ctx.has_expanded_angle else torch.empty_like(angle_right)
        dleft_coeff = (
            torch.empty_like(angle_left_coeff, dtype=grad_dtype)
            if ctx.has_compact_angle
            else torch.empty_like(angle_left_coeff)
        )
        dright_coeff = (
            torch.empty_like(angle_right_coeff, dtype=grad_dtype)
            if ctx.has_compact_angle
            else torch.empty_like(angle_right_coeff)
        )
        dagate = (
            torch.zeros_like(angle_gate, dtype=grad_dtype)
            if (ctx.has_expanded_angle or ctx.has_compact_angle) and ctx.angle_gate_head_only
            else torch.empty_like(angle_gate, dtype=grad_dtype)
            if ctx.has_expanded_angle or ctx.has_compact_angle
            else torch.empty_like(angle_gate)
        )
        dml = torch.empty_like(message_left, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_left)
        dmr = torch.empty_like(message_right, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_right)
        dmb = torch.zeros_like(message_basis, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_basis)
        left_coeff_strides = _compact_angle_coeff_canonical_strides(
            angle_left_coeff,
            batch_size=batch_size,
            num_heads=num_heads,
            num_atoms=num_atoms,
            k_neighbors=k_neighbors,
        )
        right_coeff_strides = (
            _compact_angle_coeff_canonical_strides(
                angle_right_coeff,
                batch_size=batch_size,
                num_heads=num_heads,
                num_atoms=num_atoms,
                k_neighbors=k_neighbors,
            )
            if ctx.has_compact_angle
            else (0, 0, 0, 0, 0)
        )
        dleft_coeff_strides = (
            _compact_angle_coeff_canonical_strides(
                dleft_coeff,
                batch_size=batch_size,
                num_heads=num_heads,
                num_atoms=num_atoms,
                k_neighbors=k_neighbors,
            )
            if ctx.has_compact_angle
            else (0, 0, 0, 0, 0)
        )
        dright_coeff_strides = (
            _compact_angle_coeff_canonical_strides(
                dright_coeff,
                batch_size=batch_size,
                num_heads=num_heads,
                num_atoms=num_atoms,
                k_neighbors=k_neighbors,
            )
            if ctx.has_compact_angle
            else (0, 0, 0, 0, 0)
        )
        grid = (batch_size * num_heads * num_atoms,)
        _compact_backward_kernel[grid](
            grad_out,
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx,
            neighbor_mask,
            unit,
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
            angle_left_coeff,
            angle_right_coeff,
            angle_gate,
            message_left,
            message_right,
            message_basis,
            out_fp32,
            lse,
            dq,
            dk1,
            dv1,
            dk2,
            dv2,
            du,
            dvb,
            dgate,
            dleft,
            dright,
            dleft_coeff,
            dright_coeff,
            dagate,
            dml,
            dmr,
            dmb,
            num_heads,
            num_atoms,
            ctx.q_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k1.stride(0),
            k1.stride(1),
            k1.stride(2),
            k1.stride(3),
            v1.stride(0),
            v1.stride(1),
            v1.stride(2),
            v1.stride(3),
            k2.stride(0),
            k2.stride(1),
            k2.stride(2),
            k2.stride(3),
            v2.stride(0),
            v2.stride(1),
            v2.stride(2),
            v2.stride(3),
            left_coeff_strides[0],
            left_coeff_strides[1],
            left_coeff_strides[2],
            left_coeff_strides[3],
            left_coeff_strides[4],
            right_coeff_strides[0],
            right_coeff_strides[1],
            right_coeff_strides[2],
            right_coeff_strides[3],
            right_coeff_strides[4],
            dleft_coeff_strides[0],
            dleft_coeff_strides[1],
            dleft_coeff_strides[2],
            dleft_coeff_strides[3],
            dleft_coeff_strides[4],
            dright_coeff_strides[0],
            dright_coeff_strides[1],
            dright_coeff_strides[2],
            dright_coeff_strides[3],
            dright_coeff_strides[4],
            head_dim=head_dim,
            k_neighbors=k_neighbors,
            angle_rank=ctx.angle_rank,
            message_rank=ctx.message_rank,
            HAS_RADIAL_BIAS=ctx.has_radial_bias,
            HAS_ANGLE=ctx.has_expanded_angle,
            HAS_COMPACT_ANGLE=ctx.has_compact_angle,
            HAS_MESSAGE=ctx.has_message,
            GATE_HEAD_ONLY=ctx.gate_head_only,
            ANGLE_GATE_HEAD_ONLY=ctx.angle_gate_head_only,
            OUTPUT_LAYOUT_BNHD=ctx.output_layout_bnhd,
            ANGLE_C0=ctx.angle_c0,
            ANGLE_C1=ctx.angle_c1,
            ANGLE_C2=ctx.angle_c2,
            ANGLE_COEFFS=ctx.angle_coeffs,
            INPUT_PRECISION=input_precision,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=4 if block_d <= 64 else 8,
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
            du if ctx.has_radial_bias else None,
            dvb if ctx.has_radial_bias else None,
            dgate if ctx.has_radial_bias else None,
            dleft if ctx.has_expanded_angle else None,
            dright if ctx.has_expanded_angle else None,
            dleft_coeff if ctx.has_compact_angle else None,
            dright_coeff if ctx.has_compact_angle else None,
            dagate if (ctx.has_expanded_angle or ctx.has_compact_angle) else None,
            dml if ctx.has_message else None,
            dmr if ctx.has_message else None,
            dmb if ctx.has_message else None,
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


def _unavailable_reason(
    q: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    *,
    angle_left: torch.Tensor | None,
    angle_right: torch.Tensor | None,
    unit: torch.Tensor | None,
    angle_left_coeff: torch.Tensor | None,
    angle_right_coeff: torch.Tensor | None,
    angle_channels_by_l: tuple[int, int, int] | None,
    message_left: torch.Tensor | None,
    message_right: torch.Tensor | None,
    message_basis: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> str | None:
    if not TRITON_COMPACT_SIMPLICIAL_AVAILABLE:
        return "triton is not installed"
    if q.device.type != "cuda":
        return "the compact Triton backend requires CUDA tensors"
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"the compact Triton backend supports float16/bfloat16/float32 inputs, got {q.dtype}"
    if neighbor_idx.device != q.device or neighbor_mask.device != q.device:
        return "neighbor_idx and neighbor_mask must be on the same CUDA device as q"
    if training and dropout_p > 0.0:
        return "the compact Triton backend does not support training dropout in v1"
    if q.shape[-1] > 128:
        return f"the compact Triton backend only supports head_dim <= 128, got {q.shape[-1]}"
    if neighbor_idx.shape[-1] > 64:
        return f"the compact Triton backend only supports k_neighbors <= 64, got {neighbor_idx.shape[-1]}"
    if angle_left is not None and angle_right is None:
        return "angle_left requires angle_right"
    if angle_right is not None and angle_left is None:
        return "angle_right requires angle_left"
    if angle_left_coeff is not None and angle_right_coeff is None:
        return "angle_left_coeff requires angle_right_coeff"
    if angle_right_coeff is not None and angle_left_coeff is None:
        return "angle_right_coeff requires angle_left_coeff"
    if angle_left is not None and angle_left_coeff is not None:
        return "use either expanded angle tensors or compact angle coefficients, not both"
    if angle_left is not None and angle_left.shape[-1] > 64:
        return f"the compact Triton backend only supports angle rank <= 64, got {angle_left.shape[-1]}"
    if angle_left_coeff is not None:
        if unit is None:
            return "compact angle coefficients require unit vectors"
        if unit.requires_grad:
            return "compact angle coefficients do not implement gradients through unit vectors"
        if angle_channels_by_l is None:
            return "compact angle coefficients require angle_channels_by_l"
        c0, c1, c2 = (int(v) for v in angle_channels_by_l)
        if c0 + c1 + c2 != int(angle_left_coeff.shape[-1]):
            return (
                "angle_channels_by_l does not match compact coefficient width: "
                f"{(c0, c1, c2)} vs {angle_left_coeff.shape[-1]}"
            )
        if angle_left_coeff.shape[-1] > 64:
            return (
                "the compact Triton backend only supports compact angle coeff count <= 64, "
                f"got {angle_left_coeff.shape[-1]}"
            )
    if message_left is not None and message_left.shape[-1] > 64:
        return f"the compact Triton backend only supports message rank <= 64, got {message_left.shape[-1]}"
    if message_left is not None and message_right is None:
        return "message_left requires message_right"
    if message_right is not None and message_left is None:
        return "message_right requires message_left"
    if message_left is not None and message_basis is None:
        return "message_left/message_right require message_basis"
    return None


def triton_compact_simplicial_attention(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    u: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    angle_left: torch.Tensor | None = None,
    angle_right: torch.Tensor | None = None,
    unit: torch.Tensor | None = None,
    angle_left_coeff: torch.Tensor | None = None,
    angle_right_coeff: torch.Tensor | None = None,
    angle_channels_by_l: tuple[int, int, int] | None = None,
    angle_rank: int | None = None,
    angle_gate: torch.Tensor | None = None,
    message_left: torch.Tensor | None = None,
    message_right: torch.Tensor | None = None,
    message_basis: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    precision: str = "bf16_tc",
    debug_torch_backward: bool = False,
    strict: bool = False,
    q_scale: float = 1.0,
    output_layout: str = "bhnd",
) -> torch.Tensor:
    precision = normalize_compact_simplicial_precision(precision)
    output_layout = str(output_layout).lower()
    if output_layout not in {"bhnd", "bnhd"}:
        raise ValueError("output_layout must be one of {'bhnd', 'bnhd'}")
    reason = _unavailable_reason(
        q,
        neighbor_idx,
        neighbor_mask,
        angle_left=angle_left,
        angle_right=angle_right,
        unit=unit,
        angle_left_coeff=angle_left_coeff,
        angle_right_coeff=angle_right_coeff,
        angle_channels_by_l=angle_channels_by_l,
        message_left=message_left,
        message_right=message_right,
        message_basis=message_basis,
        dropout_p=dropout_p,
        training=training,
    )
    if reason is not None:
        if strict:
            raise RuntimeError(f"Compact Triton simplicial attention is unavailable: {reason}")
        out = compact_simplicial_attention_torch_reference(
            q,
            k1,
            v1,
            k2,
            v2,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            u=u,
            v_bias=v_bias,
            gate=gate,
            angle_left=angle_left,
            angle_right=angle_right,
            unit=unit,
            angle_left_coeff=angle_left_coeff,
            angle_right_coeff=angle_right_coeff,
            angle_channels_by_l=angle_channels_by_l,
            angle_rank=angle_rank,
            angle_gate=angle_gate,
            message_left=message_left,
            message_right=message_right,
            message_basis=message_basis,
            dropout_p=dropout_p,
            training=training,
            q_scale=q_scale,
        )
        return out.transpose(1, 2).contiguous() if output_layout == "bnhd" else out
    empty_float = q.new_empty(0)
    u_tensor = u.contiguous() if u is not None and v_bias is not None and gate is not None else empty_float
    v_tensor = v_bias.contiguous() if u is not None and v_bias is not None and gate is not None else empty_float
    if u is not None and v_bias is not None and gate is not None:
        if gate.ndim == 1:
            gate_tensor = gate
        elif gate.ndim == 3 and gate.shape[0] == 1 and gate.shape[2] == 1:
            gate_tensor = gate.reshape(-1)
        elif gate.ndim == 3:
            gate_tensor = gate.contiguous()
        else:
            raise ValueError(f"gate must be [H] or [B,H,N], got {tuple(gate.shape)}")
    else:
        gate_tensor = empty_float
    has_expanded_angle = angle_left is not None and angle_right is not None
    has_compact_angle = angle_left_coeff is not None and angle_right_coeff is not None
    angle_left_tensor = angle_left.contiguous() if has_expanded_angle else empty_float
    angle_right_tensor = angle_right.contiguous() if has_expanded_angle else empty_float
    unit_tensor = unit.contiguous() if has_compact_angle and unit is not None else empty_float
    angle_left_coeff_tensor = angle_left_coeff if has_compact_angle else empty_float
    angle_right_coeff_tensor = angle_right_coeff if has_compact_angle else empty_float
    if (has_expanded_angle or has_compact_angle) and angle_gate is not None:
        if angle_gate.ndim == 1:
            angle_gate_tensor = angle_gate
        elif angle_gate.ndim == 3 and angle_gate.shape[0] == 1 and angle_gate.shape[2] == 1:
            angle_gate_tensor = angle_gate.reshape(-1)
        elif angle_gate.ndim == 3:
            angle_gate_tensor = angle_gate.contiguous()
        else:
            raise ValueError(f"angle_gate must be [H] or [B,H,N], got {tuple(angle_gate.shape)}")
    else:
        angle_gate_tensor = empty_float
    if (has_expanded_angle or has_compact_angle) and angle_gate is None:
        angle_gate_tensor = torch.ones((q.shape[1],), device=q.device, dtype=q.dtype)
    if has_compact_angle:
        if angle_channels_by_l is None:
            raise ValueError("compact angle coefficients require angle_channels_by_l")
        angle_c0, angle_c1, angle_c2 = (int(v) for v in angle_channels_by_l)
        angle_coeffs = int(angle_left_coeff_tensor.shape[-1])
        compact_angle_rank = int(angle_rank or (angle_c0 + 3 * angle_c1 + 5 * angle_c2))
    else:
        angle_c0 = angle_c1 = angle_c2 = 0
        angle_coeffs = 0
        compact_angle_rank = 1
    message_left_tensor = (
        message_left.contiguous()
        if message_left is not None and message_right is not None and message_basis is not None
        else empty_float
    )
    message_right_tensor = (
        message_right.contiguous()
        if message_left is not None and message_right is not None and message_basis is not None
        else empty_float
    )
    message_basis_tensor = (
        message_basis.contiguous()
        if message_left is not None and message_right is not None and message_basis is not None
        else empty_float
    )
    return _TritonCompactSimplicialAttentionFunction.apply(
        q,
        k1,
        v1,
        k2,
        v2,
        neighbor_idx.contiguous(),
        neighbor_mask.contiguous(),
        unit_tensor,
        u_tensor,
        v_tensor,
        gate_tensor,
        angle_left_tensor,
        angle_right_tensor,
        angle_left_coeff_tensor,
        angle_right_coeff_tensor,
        angle_gate_tensor,
        message_left_tensor,
        message_right_tensor,
        message_basis_tensor,
        precision,
        bool(debug_torch_backward),
        float(q_scale),
        angle_c0,
        angle_c1,
        angle_c2,
        angle_coeffs,
        compact_angle_rank,
        output_layout == "bnhd",
    )
