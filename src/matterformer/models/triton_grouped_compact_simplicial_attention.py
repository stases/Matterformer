from __future__ import annotations

import torch

from matterformer.models.triton_compact_simplicial_attention import (
    TRITON_COMPACT_SIMPLICIAL_AVAILABLE,
    compact_block_d_for_head_dim,
    compact_block_k_for_neighbors,
    compact_block_r_for_rank,
    compact_simplicial_attention_torch_reference,
    normalize_compact_simplicial_precision,
    _precision_kernel_config,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA nodes with Triton installed.
    triton = None
    tl = None

TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE = TRITON_COMPACT_SIMPLICIAL_AVAILABLE and triton is not None and tl is not None


def _repeat_for_group(tensor: torch.Tensor | None, group_order: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.numel() == 0:
        return tensor
    # [B, ...] -> [B * G, ...], sharing storage in the forward expression when possible.
    return tensor[:, None].expand(-1, int(group_order), *tensor.shape[1:]).reshape(
        tensor.shape[0] * int(group_order), *tensor.shape[1:]
    )


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


def _expand_compact_spherical_coefficients(
    coeff: torch.Tensor,
    *,
    basis: dict[int, torch.Tensor],
    channels_by_l: tuple[int, int, int],
) -> torch.Tensor:
    # coeff: [B, H, N, K, C] -> expanded [B, H, N, K, R]
    coeff = coeff.permute(0, 2, 3, 1, 4).contiguous()
    pieces: list[torch.Tensor] = []
    offset = 0
    for degree, num_channels in enumerate(channels_by_l):
        num_channels = int(num_channels)
        if num_channels == 0:
            continue
        coeff_l = coeff[..., offset : offset + num_channels]
        offset += num_channels
        piece = coeff_l.unsqueeze(-1) * basis[degree][:, :, :, None, None, :]
        pieces.append(piece.flatten(-2))
    if not pieces:
        raise ValueError("At least one compact spherical channel is required")
    return torch.cat(pieces, dim=-1).permute(0, 3, 1, 2, 4).contiguous()


def grouped_compact_simplicial_attention_torch_reference(
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
) -> torch.Tensor:
    """Reference for grouped S_g: Q/K/V have [B, G, H, N, D], geometry has [B, ...]."""

    batch_size, group_order, num_heads, num_atoms, head_dim = q.shape
    if angle_left is not None and angle_left_coeff is not None:
        raise ValueError("Use either expanded angle tensors or compact angle coefficients, not both")
    if angle_left_coeff is not None or angle_right_coeff is not None:
        if unit is None or angle_left_coeff is None or angle_right_coeff is None or angle_channels_by_l is None:
            raise ValueError("Compact angle coefficients require unit, left/right coeffs, and channels_by_l")
        basis = _spherical_basis_lmax2(unit)
        angle_left = _expand_compact_spherical_coefficients(
            angle_left_coeff,
            basis=basis,
            channels_by_l=angle_channels_by_l,
        )
        angle_right = _expand_compact_spherical_coefficients(
            angle_right_coeff,
            basis=basis,
            channels_by_l=angle_channels_by_l,
        )
        if angle_rank is not None and angle_left.shape[-1] != int(angle_rank):
            raise ValueError(f"Compact angle coefficients expanded to {angle_left.shape[-1]}, expected {angle_rank}")
    flat_shape = (batch_size * group_order, num_heads, num_atoms, head_dim)
    neighbor_idx_g = _repeat_for_group(neighbor_idx, group_order)
    neighbor_mask_g = _repeat_for_group(neighbor_mask, group_order)
    out = compact_simplicial_attention_torch_reference(
        q.reshape(flat_shape),
        k1.reshape(flat_shape),
        v1.reshape(flat_shape),
        k2.reshape(flat_shape),
        v2.reshape(flat_shape),
        neighbor_idx=neighbor_idx_g,
        neighbor_mask=neighbor_mask_g,
        u=_repeat_for_group(u, group_order),
        v_bias=_repeat_for_group(v_bias, group_order),
        gate=_repeat_for_group(gate, group_order),
        angle_left=_repeat_for_group(angle_left, group_order),
        angle_right=_repeat_for_group(angle_right, group_order),
        angle_gate=_repeat_for_group(angle_gate, group_order),
        message_left=_repeat_for_group(message_left, group_order),
        message_right=_repeat_for_group(message_right, group_order),
        message_basis=message_basis,
        dropout_p=dropout_p,
        training=training,
    )
    return out.view(batch_size, group_order, num_heads, num_atoms, head_dim)


if TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE:

    @triton.jit
    def _grouped_compact_forward_kernel(
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
        batch_size: tl.constexpr,
        group_order: tl.constexpr,
        num_heads: tl.constexpr,
        num_atoms: tl.constexpr,
        head_dim: tl.constexpr,
        k_neighbors: tl.constexpr,
        angle_rank: tl.constexpr,
        message_rank: tl.constexpr,
        HAS_RADIAL_BIAS: tl.constexpr,
        HAS_ANGLE: tl.constexpr,
        HAS_COMPACT_ANGLE: tl.constexpr,
        HAS_MESSAGE: tl.constexpr,
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
        tmp = pid // num_atoms
        head_idx = tmp % num_heads
        tmp = tmp // num_heads
        group_idx = tmp % group_order
        batch_idx = tmp // group_order

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

        bgh_idx = ((batch_idx * group_order + group_idx) * num_heads + head_idx)
        bh_idx = batch_idx * num_heads + head_idx
        q_base = (bgh_idx * num_atoms + atom_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0)
        k1 = tl.load(
            k1_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            k2_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v1_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v2 = tl.load(
            v2_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        score = tl.dot(k1 * q[None, :], tl.trans(k2), input_precision=INPUT_PRECISION).to(tl.float32)

        if HAS_RADIAL_BIAS:
            bias_base = (bh_idx * num_atoms + atom_idx) * k_neighbors
            u = tl.load(u_ptr + bias_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            vb = tl.load(v_bias_ptr + bias_base + offs_k, mask=k_mask, other=0.0).to(tl.float32)
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
            agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle

        if HAS_COMPACT_ANGLE:
            unit_base = ((batch_idx * num_atoms + atom_idx) * k_neighbors + offs_k) * 3
            ux = tl.load(unit_ptr + unit_base + 0, mask=k_mask, other=0.0).to(tl.float32)
            uy = tl.load(unit_ptr + unit_base + 1, mask=k_mask, other=0.0).to(tl.float32)
            uz = tl.load(unit_ptr + unit_base + 2, mask=k_mask, other=0.0).to(tl.float32)
            cos = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :] + uz[:, None] * uz[None, :]
            p2 = 0.5 * (3.0 * cos * cos - 1.0)
            coeff_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * ANGLE_COEFFS
            angle_unscaled = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
            if ANGLE_C0 > 0:
                c_mask = offs_c < ANGLE_C0
                left0 = tl.load(
                    angle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + offs_c[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right0 = tl.load(
                    angle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + offs_c[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left0, tl.trans(right0), input_precision=INPUT_PRECISION)
            if ANGLE_C1 > 0:
                c_mask = offs_c < ANGLE_C1
                c_offs = ANGLE_C0 + offs_c
                left1 = tl.load(
                    angle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right1 = tl.load(
                    angle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left1, tl.trans(right1), input_precision=INPUT_PRECISION) * cos
            if ANGLE_C2 > 0:
                c_mask = offs_c < ANGLE_C2
                c_offs = ANGLE_C0 + ANGLE_C1 + offs_c
                left2 = tl.load(
                    angle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right2 = tl.load(
                    angle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left2, tl.trans(right2), input_precision=INPUT_PRECISION) * p2
            angle = angle_unscaled * tl.rsqrt(angle_rank + 0.0)
            agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle

        pair_valid = valid[:, None] & valid[None, :]
        valid_count = tl.sum(tl.sum(pair_valid.to(tl.int32), axis=0), axis=0)
        has_valid = valid_count > 0
        neg_large = -3.4028234663852886e38
        score = tl.where(pair_valid, score, neg_large)
        row_max = tl.max(tl.max(score, axis=0), axis=0)
        row_max = tl.where(has_valid, row_max, 0.0)
        exp_score = tl.where(pair_valid, tl.exp(score - row_max), 0.0)
        denom = tl.sum(tl.sum(exp_score, axis=0), axis=0)
        denom_safe = tl.where(has_valid, denom, 1.0)
        probs = tl.where(has_valid, exp_score / denom_safe, 0.0)

        tmp_v = tl.dot(tl.trans(probs), v1, input_precision=INPUT_PRECISION)
        out = tl.sum(tmp_v * v2, axis=0).to(tl.float32)

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
        tl.store(out_ptr + q_base + offs_d, out, mask=d_mask)
        lse = tl.where(has_valid, row_max + tl.log(denom), 0.0)
        tl.store(lse_ptr + (bgh_idx * num_atoms + atom_idx), lse)

    @triton.jit
    def _grouped_compact_backward_kernel(
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
        batch_size: tl.constexpr,
        group_order: tl.constexpr,
        num_heads: tl.constexpr,
        num_atoms: tl.constexpr,
        head_dim: tl.constexpr,
        k_neighbors: tl.constexpr,
        angle_rank: tl.constexpr,
        message_rank: tl.constexpr,
        HAS_RADIAL_BIAS: tl.constexpr,
        HAS_ANGLE: tl.constexpr,
        HAS_COMPACT_ANGLE: tl.constexpr,
        HAS_MESSAGE: tl.constexpr,
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
        tmp = pid // num_atoms
        head_idx = tmp % num_heads
        tmp = tmp // num_heads
        group_idx = tmp % group_order
        batch_idx = tmp // group_order

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

        bgh_idx = ((batch_idx * group_order + group_idx) * num_heads + head_idx)
        bh_idx = batch_idx * num_heads + head_idx
        q_base = (bgh_idx * num_atoms + atom_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0)
        go = tl.load(grad_out_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + bgh_idx * num_atoms + atom_idx).to(tl.float32)

        k1 = tl.load(
            k1_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            k2_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v1_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v2 = tl.load(
            v2_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        score = tl.dot(k1 * q[None, :], tl.trans(k2), input_precision=INPUT_PRECISION).to(tl.float32)
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
            agate = tl.load(angle_gate_ptr + bh_idx * num_atoms + atom_idx).to(tl.float32)
            score += agate * angle
        if HAS_COMPACT_ANGLE:
            unit_base = ((batch_idx * num_atoms + atom_idx) * k_neighbors + offs_k) * 3
            ux = tl.load(unit_ptr + unit_base + 0, mask=k_mask, other=0.0).to(tl.float32)
            uy = tl.load(unit_ptr + unit_base + 1, mask=k_mask, other=0.0).to(tl.float32)
            uz = tl.load(unit_ptr + unit_base + 2, mask=k_mask, other=0.0).to(tl.float32)
            compact_cos = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :] + uz[:, None] * uz[None, :]
            compact_p2 = 0.5 * (3.0 * compact_cos * compact_cos - 1.0)
            coeff_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * ANGLE_COEFFS
            angle_unscaled = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
            if ANGLE_C0 > 0:
                c_mask = offs_c < ANGLE_C0
                left0 = tl.load(
                    angle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + offs_c[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right0 = tl.load(
                    angle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + offs_c[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left0, tl.trans(right0), input_precision=INPUT_PRECISION)
            if ANGLE_C1 > 0:
                c_mask = offs_c < ANGLE_C1
                c_offs = ANGLE_C0 + offs_c
                left1 = tl.load(
                    angle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right1 = tl.load(
                    angle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left1, tl.trans(right1), input_precision=INPUT_PRECISION) * compact_cos
            if ANGLE_C2 > 0:
                c_mask = offs_c < ANGLE_C2
                c_offs = ANGLE_C0 + ANGLE_C1 + offs_c
                left2 = tl.load(
                    angle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                right2 = tl.load(
                    angle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                angle_unscaled += tl.dot(left2, tl.trans(right2), input_precision=INPUT_PRECISION) * compact_p2
            angle = angle_unscaled * tl.rsqrt(angle_rank + 0.0)
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
            inv_sqrt_message = tl.rsqrt(message_rank + 0.0)
            basis_go = tl.sum(basis * go[None, :], axis=1)
            d_pair += tl.dot(ml * basis_go[None, :], tl.trans(mr), input_precision=INPUT_PRECISION) * inv_sqrt_message

            msg_tmp = tl.dot(tl.trans(probs), ml, input_precision=INPUT_PRECISION)
            coeff = tl.sum(msg_tmp * mr, axis=0) * inv_sqrt_message
            d_basis = coeff[:, None] * go[None, :]
            tl.atomic_add(
                dmessage_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                d_basis,
                sem="relaxed",
                mask=r_mask[:, None] & d_mask[None, :],
            )
            dml = tl.dot(probs, mr, input_precision=INPUT_PRECISION) * (basis_go[None, :] * inv_sqrt_message)
            dmr = tl.dot(tl.trans(probs), ml, input_precision=INPUT_PRECISION) * (basis_go[None, :] * inv_sqrt_message)
            tl.atomic_add(
                dmessage_left_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                dml,
                sem="relaxed",
                mask=valid[:, None] & r_mask[None, :],
            )
            tl.atomic_add(
                dmessage_right_ptr + msg_base + offs_k[:, None] * message_rank + offs_r[None, :],
                dmr,
                sem="relaxed",
                mask=valid[:, None] & r_mask[None, :],
            )

        center = tl.sum(go * out, axis=0)
        ds = probs * (d_pair - center)
        ds = tl.where(pair_valid, ds, 0.0)

        tmp_q = tl.dot(ds, k2, input_precision=INPUT_PRECISION)
        dq = tl.sum(tmp_q * k1, axis=0)
        tl.store(dq_ptr + q_base + offs_d, dq, mask=d_mask)

        dk1 = tl.dot(ds, k2, input_precision=INPUT_PRECISION) * q[None, :]
        dk2 = tl.dot(tl.trans(ds), k1, input_precision=INPUT_PRECISION) * q[None, :]
        dv1 = tl.dot(probs, v2, input_precision=INPUT_PRECISION) * go[None, :]
        dv2 = tl.dot(tl.trans(probs), v1, input_precision=INPUT_PRECISION) * go[None, :]

        tl.atomic_add(
            dk1_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dk1,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            dk2_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dk2,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            dv1_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dv1,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            dv2_ptr + (bgh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            dv2,
            sem="relaxed",
            mask=valid[:, None] & d_mask[None, :],
        )

        if HAS_RADIAL_BIAS:
            bias_base = (bh_idx * num_atoms + atom_idx) * k_neighbors
            du = gate * tl.sum(ds, axis=1)
            dvb = gate * tl.sum(ds, axis=0)
            dgate = tl.sum(tl.sum(ds * (u[:, None] + vb[None, :]), axis=0), axis=0)
            tl.atomic_add(du_ptr + bias_base + offs_k, du, sem="relaxed", mask=valid)
            tl.atomic_add(dv_bias_ptr + bias_base + offs_k, dvb, sem="relaxed", mask=valid)
            tl.atomic_add(dgate_ptr + bh_idx * num_atoms + atom_idx, dgate, sem="relaxed")

        if HAS_ANGLE:
            r_mask = offs_r < angle_rank
            angle_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * angle_rank
            scale = agate * tl.rsqrt(angle_rank + 0.0)
            dleft = tl.dot(ds, right, input_precision=INPUT_PRECISION) * scale
            dright = tl.dot(tl.trans(ds), left, input_precision=INPUT_PRECISION) * scale
            dangle_gate = tl.sum(tl.sum(ds * angle, axis=0), axis=0)
            tl.atomic_add(
                dangle_left_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                dleft,
                sem="relaxed",
                mask=valid[:, None] & r_mask[None, :],
            )
            tl.atomic_add(
                dangle_right_ptr + angle_base + offs_k[:, None] * angle_rank + offs_r[None, :],
                dright,
                sem="relaxed",
                mask=valid[:, None] & r_mask[None, :],
            )
            tl.atomic_add(dangle_gate_ptr + bh_idx * num_atoms + atom_idx, dangle_gate, sem="relaxed")

        if HAS_COMPACT_ANGLE:
            coeff_base = ((bh_idx * num_atoms + atom_idx) * k_neighbors) * ANGLE_COEFFS
            scale = agate * tl.rsqrt(angle_rank + 0.0)
            if ANGLE_C0 > 0:
                c_mask = offs_c < ANGLE_C0
                weighted = ds * scale
                dleft0 = tl.dot(weighted, right0, input_precision=INPUT_PRECISION)
                dright0 = tl.dot(tl.trans(weighted), left0, input_precision=INPUT_PRECISION)
                tl.atomic_add(
                    dangle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + offs_c[None, :],
                    dleft0,
                    sem="relaxed",
                    mask=valid[:, None] & c_mask[None, :],
                )
                tl.atomic_add(
                    dangle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + offs_c[None, :],
                    dright0,
                    sem="relaxed",
                    mask=valid[:, None] & c_mask[None, :],
                )
            if ANGLE_C1 > 0:
                c_mask = offs_c < ANGLE_C1
                c_offs = ANGLE_C0 + offs_c
                weighted = ds * (scale * compact_cos)
                dleft1 = tl.dot(weighted, right1, input_precision=INPUT_PRECISION)
                dright1 = tl.dot(tl.trans(weighted), left1, input_precision=INPUT_PRECISION)
                tl.atomic_add(
                    dangle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    dleft1,
                    sem="relaxed",
                    mask=valid[:, None] & c_mask[None, :],
                )
                tl.atomic_add(
                    dangle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    dright1,
                    sem="relaxed",
                    mask=valid[:, None] & c_mask[None, :],
                )
            if ANGLE_C2 > 0:
                c_mask = offs_c < ANGLE_C2
                c_offs = ANGLE_C0 + ANGLE_C1 + offs_c
                weighted = ds * (scale * compact_p2)
                dleft2 = tl.dot(weighted, right2, input_precision=INPUT_PRECISION)
                dright2 = tl.dot(tl.trans(weighted), left2, input_precision=INPUT_PRECISION)
                tl.atomic_add(
                    dangle_left_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    dleft2,
                    sem="relaxed",
                    mask=valid[:, None] & c_mask[None, :],
                )
                tl.atomic_add(
                    dangle_right_coeff_ptr + coeff_base + offs_k[:, None] * ANGLE_COEFFS + c_offs[None, :],
                    dright2,
                    sem="relaxed",
                    mask=valid[:, None] & c_mask[None, :],
                )
            dangle_gate = tl.sum(tl.sum(ds * angle, axis=0), axis=0)
            tl.atomic_add(dangle_gate_ptr + bh_idx * num_atoms + atom_idx, dangle_gate, sem="relaxed")


class _TritonGroupedCompactSimplicialAttentionFunction(torch.autograd.Function):
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
        angle_c0: int,
        angle_c1: int,
        angle_c2: int,
        angle_coeffs: int,
        compact_angle_rank: int,
    ) -> torch.Tensor:
        if not TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE:  # pragma: no cover - guarded before dispatch.
            raise RuntimeError("triton is not installed")
        batch_size, group_order, num_heads, num_atoms, head_dim = q.shape
        k_neighbors = int(neighbor_idx.shape[-1])
        has_radial_bias = u.numel() > 0 and v_bias.numel() > 0 and gate.numel() > 0
        has_expanded_angle = angle_left.numel() > 0 and angle_right.numel() > 0
        has_compact_angle = angle_left_coeff.numel() > 0 and angle_right_coeff.numel() > 0
        has_message = message_left.numel() > 0 and message_right.numel() > 0 and message_basis.numel() > 0
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

        out_fp32 = torch.empty((batch_size, group_order, num_heads, num_atoms, head_dim), device=q.device, dtype=torch.float32)
        lse = torch.empty((batch_size, group_order, num_heads, num_atoms), device=q.device, dtype=torch.float32)
        grid = (batch_size * group_order * num_heads * num_atoms,)
        _grouped_compact_forward_kernel[grid](
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
            batch_size=batch_size,
            group_order=group_order,
            num_heads=num_heads,
            num_atoms=num_atoms,
            head_dim=head_dim,
            k_neighbors=k_neighbors,
            angle_rank=angle_rank,
            message_rank=message_rank,
            HAS_RADIAL_BIAS=has_radial_bias,
            HAS_ANGLE=has_expanded_angle,
            HAS_COMPACT_ANGLE=has_compact_angle,
            HAS_MESSAGE=has_message,
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
        ctx.has_radial_bias = has_radial_bias
        ctx.has_expanded_angle = has_expanded_angle
        ctx.has_compact_angle = has_compact_angle
        ctx.has_message = has_message
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
            if ctx.has_radial_bias:
                inputs.extend([u.detach().requires_grad_(True), v_bias.detach().requires_grad_(True), gate.detach().requires_grad_(True)])
            if ctx.has_expanded_angle:
                inputs.extend([angle_left.detach().requires_grad_(True), angle_right.detach().requires_grad_(True), angle_gate.detach().requires_grad_(True)])
            if ctx.has_compact_angle:
                inputs.extend(
                    [
                        angle_left_coeff.detach().requires_grad_(True),
                        angle_right_coeff.detach().requires_grad_(True),
                        angle_gate.detach().requires_grad_(True),
                    ]
                )
            if ctx.has_message:
                inputs.extend([message_left.detach().requires_grad_(True), message_right.detach().requires_grad_(True), message_basis.detach().requires_grad_(True)])
            with torch.enable_grad():
                q_re, k1_re, v1_re, k2_re, v2_re = inputs[:5]
                cursor = 5
                local_u = local_v = local_gate = None
                local_left = local_right = local_angle_gate = None
                local_left_coeff = local_right_coeff = None
                local_msg_left = local_msg_right = local_msg_basis = None
                if ctx.has_radial_bias:
                    local_u, local_v, local_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_expanded_angle:
                    local_left, local_right, local_angle_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_compact_angle:
                    local_left_coeff, local_right_coeff, local_angle_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_message:
                    local_msg_left, local_msg_right, local_msg_basis = inputs[cursor : cursor + 3]
                ref = grouped_compact_simplicial_attention_torch_reference(
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
                    unit=unit if ctx.has_compact_angle else None,
                    angle_left_coeff=local_left_coeff,
                    angle_right_coeff=local_right_coeff,
                    angle_channels_by_l=(ctx.angle_c0, ctx.angle_c1, ctx.angle_c2) if ctx.has_compact_angle else None,
                    angle_rank=ctx.angle_rank if ctx.has_compact_angle else None,
                    angle_gate=local_angle_gate,
                    message_left=local_msg_left,
                    message_right=local_msg_right,
                    message_basis=local_msg_basis,
                )
                grads = torch.autograd.grad(ref, inputs, grad_out, allow_unused=True)
            dq = dk1 = dv1 = dk2 = dv2 = du = dvb = dgate = None
            dleft = dright = dleft_coeff = dright_coeff = dagate = None
            dml = dmr = dmb = None
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
            )

        batch_size, group_order, num_heads, num_atoms, head_dim = q.shape
        k_neighbors = int(neighbor_idx.shape[-1])
        block_k = compact_block_k_for_neighbors(k_neighbors)
        block_d = compact_block_d_for_head_dim(head_dim)
        block_r = compact_block_r_for_rank(max(ctx.angle_rank if ctx.has_expanded_angle else 1, ctx.message_rank if ctx.has_message else 1))
        block_c = compact_block_r_for_rank(max(ctx.angle_c0, ctx.angle_c1, ctx.angle_c2, 1))
        input_precision = _precision_kernel_config(ctx.precision)
        grad_out = grad_out.contiguous()
        grad_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        dq = torch.empty_like(q, dtype=grad_dtype)
        dk1 = torch.zeros_like(k1, dtype=grad_dtype)
        dv1 = torch.zeros_like(v1, dtype=grad_dtype)
        dk2 = torch.zeros_like(k2, dtype=grad_dtype)
        dv2 = torch.zeros_like(v2, dtype=grad_dtype)
        du = torch.zeros_like(u, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(u)
        dvb = torch.zeros_like(v_bias, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(v_bias)
        dgate = torch.zeros_like(gate, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(gate)
        dleft = torch.zeros_like(angle_left, dtype=grad_dtype) if ctx.has_expanded_angle else torch.empty_like(angle_left)
        dright = torch.zeros_like(angle_right, dtype=grad_dtype) if ctx.has_expanded_angle else torch.empty_like(angle_right)
        dleft_coeff = (
            torch.zeros_like(angle_left_coeff, dtype=grad_dtype) if ctx.has_compact_angle else torch.empty_like(angle_left_coeff)
        )
        dright_coeff = (
            torch.zeros_like(angle_right_coeff, dtype=grad_dtype) if ctx.has_compact_angle else torch.empty_like(angle_right_coeff)
        )
        has_any_angle = ctx.has_expanded_angle or ctx.has_compact_angle
        dagate = torch.zeros_like(angle_gate, dtype=grad_dtype) if has_any_angle else torch.empty_like(angle_gate)
        dml = torch.zeros_like(message_left, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_left)
        dmr = torch.zeros_like(message_right, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_right)
        dmb = torch.zeros_like(message_basis, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_basis)
        grid = (batch_size * group_order * num_heads * num_atoms,)
        _grouped_compact_backward_kernel[grid](
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
            batch_size=batch_size,
            group_order=group_order,
            num_heads=num_heads,
            num_atoms=num_atoms,
            head_dim=head_dim,
            k_neighbors=k_neighbors,
            angle_rank=ctx.angle_rank,
            message_rank=ctx.message_rank,
            HAS_RADIAL_BIAS=ctx.has_radial_bias,
            HAS_ANGLE=ctx.has_expanded_angle,
            HAS_COMPACT_ANGLE=ctx.has_compact_angle,
            HAS_MESSAGE=ctx.has_message,
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
            dagate if has_any_angle else None,
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
        )


def _grouped_unavailable_reason(
    q: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    *,
    angle_left: torch.Tensor | None,
    angle_right: torch.Tensor | None,
    unit: torch.Tensor | None,
    angle_left_coeff: torch.Tensor | None,
    angle_right_coeff: torch.Tensor | None,
    message_left: torch.Tensor | None,
    message_basis: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> str | None:
    if not TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE:
        return "triton is not installed"
    if q.device.type != "cuda":
        return "the grouped compact Triton backend requires CUDA tensors"
    if q.ndim != 5:
        return f"grouped compact Triton expects q/k/v as [B, G, H, N, D], got shape {tuple(q.shape)}"
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"the grouped compact Triton backend supports float16/bfloat16/float32 inputs, got {q.dtype}"
    if neighbor_idx.device != q.device or neighbor_mask.device != q.device:
        return "neighbor_idx and neighbor_mask must be on the same CUDA device as q"
    if training and dropout_p > 0.0:
        return "the grouped compact Triton backend does not support training dropout in v1"
    if q.shape[-1] > 128:
        return f"the grouped compact Triton backend only supports head_dim <= 128, got {q.shape[-1]}"
    if neighbor_idx.shape[-1] > 64:
        return f"the grouped compact Triton backend only supports k_neighbors <= 64, got {neighbor_idx.shape[-1]}"
    if angle_left is not None and angle_left.shape[-1] > 64:
        return f"the grouped compact Triton backend only supports angle rank <= 64, got {angle_left.shape[-1]}"
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
    if angle_left_coeff is not None:
        if unit is None:
            return "compact angle coefficients require unit vectors"
        if unit.requires_grad:
            return "compact angle Triton does not implement gradients through unit vectors"
        if unit.device != q.device:
            return "unit must be on the same CUDA device as q"
        if angle_left_coeff.shape[-1] > 64:
            return f"the grouped compact Triton backend only supports compact angle coeff count <= 64, got {angle_left_coeff.shape[-1]}"
    if message_left is not None and message_left.shape[-1] > 64:
        return f"the grouped compact Triton backend only supports message rank <= 64, got {message_left.shape[-1]}"
    if message_left is not None and message_basis is None:
        return "message_left/message_right require message_basis"
    return None


def triton_grouped_compact_simplicial_attention(
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
) -> torch.Tensor:
    """Grouped kNN 2-simplicial attention for S_g.

    Q/K/V are [B, G, H, N, D]. Neighbor lists and geometry bias tensors are [B, ...]
    and are reused by all G group frames. This is mathematically equivalent to folding G
    into the batch and repeating the geometry, but avoids materializing the repeated
    GeometryCache and repeated [B * G, H, N, K, R] bias/message tensors.
    """

    precision = normalize_compact_simplicial_precision(precision)
    has_expanded_angle = angle_left is not None or angle_right is not None
    has_compact_angle = angle_left_coeff is not None or angle_right_coeff is not None
    if has_expanded_angle and (angle_left is None or angle_right is None):
        raise ValueError("Expanded angle bias requires both angle_left and angle_right")
    if has_compact_angle:
        if angle_left_coeff is None or angle_right_coeff is None:
            raise ValueError("Compact angle bias requires both angle_left_coeff and angle_right_coeff")
        if unit is None:
            raise ValueError("Compact angle bias requires unit vectors")
        if angle_channels_by_l is None:
            raise ValueError("Compact angle bias requires angle_channels_by_l")
    if has_expanded_angle and has_compact_angle:
        raise ValueError("Use either expanded angle tensors or compact angle coefficients, not both")

    reason = _grouped_unavailable_reason(
        q,
        neighbor_idx,
        neighbor_mask,
        angle_left=angle_left,
        angle_right=angle_right,
        unit=unit,
        angle_left_coeff=angle_left_coeff,
        angle_right_coeff=angle_right_coeff,
        message_left=message_left,
        message_basis=message_basis,
        dropout_p=dropout_p,
        training=training,
    )
    if reason is not None:
        if strict:
            raise RuntimeError(f"Grouped compact Triton simplicial attention is unavailable: {reason}")
        return grouped_compact_simplicial_attention_torch_reference(
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
        )

    empty_float = q.new_empty(0)
    u_tensor = u.contiguous() if u is not None and v_bias is not None and gate is not None else empty_float
    v_tensor = v_bias.contiguous() if u is not None and v_bias is not None and gate is not None else empty_float
    gate_tensor = gate.contiguous() if u is not None and v_bias is not None and gate is not None else empty_float
    angle_left_tensor = angle_left.contiguous() if has_expanded_angle else empty_float
    angle_right_tensor = angle_right.contiguous() if has_expanded_angle else empty_float
    unit_tensor = unit.contiguous() if has_compact_angle else empty_float
    angle_left_coeff_tensor = angle_left_coeff.contiguous() if has_compact_angle else empty_float
    angle_right_coeff_tensor = angle_right_coeff.contiguous() if has_compact_angle else empty_float
    if angle_channels_by_l is not None:
        angle_c0, angle_c1, angle_c2 = (int(angle_channels_by_l[0]), int(angle_channels_by_l[1]), int(angle_channels_by_l[2]))
    else:
        angle_c0 = angle_c1 = angle_c2 = 0
    angle_coeffs = int(angle_c0 + angle_c1 + angle_c2)
    expected_compact_angle_rank = int(angle_c0 + 3 * angle_c1 + 5 * angle_c2)
    compact_angle_rank = int(angle_rank) if angle_rank is not None else expected_compact_angle_rank
    if has_compact_angle and compact_angle_rank != expected_compact_angle_rank:
        raise ValueError(
            f"Compact angle coefficients expand to rank {expected_compact_angle_rank}, got angle_rank={compact_angle_rank}"
        )
    has_any_angle = has_expanded_angle or has_compact_angle
    angle_gate_tensor = angle_gate.contiguous() if has_any_angle and angle_gate is not None else empty_float
    if has_any_angle and angle_gate is None:
        angle_gate_tensor = torch.ones((q.shape[0], q.shape[2], q.shape[3]), device=q.device, dtype=q.dtype).contiguous()
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
    return _TritonGroupedCompactSimplicialAttentionFunction.apply(
        q.contiguous(),
        k1.contiguous(),
        v1.contiguous(),
        k2.contiguous(),
        v2.contiguous(),
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
        angle_c0,
        angle_c1,
        angle_c2,
        angle_coeffs,
        compact_angle_rank,
    )
