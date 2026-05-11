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
    angle_gate: torch.Tensor | None = None,
    message_left: torch.Tensor | None = None,
    message_right: torch.Tensor | None = None,
    message_basis: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Reference for grouped S_g: Q/K/V have [B, G, H, N, D], geometry has [B, ...]."""

    batch_size, group_order, num_heads, num_atoms, head_dim = q.shape
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
        u_ptr,
        v_bias_ptr,
        gate_ptr,
        angle_left_ptr,
        angle_right_ptr,
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
        HAS_MESSAGE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
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
        u_ptr,
        v_bias_ptr,
        gate_ptr,
        angle_left_ptr,
        angle_right_ptr,
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
        HAS_MESSAGE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
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
        u: torch.Tensor,
        v_bias: torch.Tensor,
        gate: torch.Tensor,
        angle_left: torch.Tensor,
        angle_right: torch.Tensor,
        angle_gate: torch.Tensor,
        message_left: torch.Tensor,
        message_right: torch.Tensor,
        message_basis: torch.Tensor,
        precision: str,
        debug_torch_backward: bool,
    ) -> torch.Tensor:
        if not TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE:  # pragma: no cover - guarded before dispatch.
            raise RuntimeError("triton is not installed")
        batch_size, group_order, num_heads, num_atoms, head_dim = q.shape
        k_neighbors = int(neighbor_idx.shape[-1])
        has_radial_bias = u.numel() > 0 and v_bias.numel() > 0 and gate.numel() > 0
        has_angle = angle_left.numel() > 0 and angle_right.numel() > 0
        has_message = message_left.numel() > 0 and message_right.numel() > 0 and message_basis.numel() > 0
        angle_rank = int(angle_left.shape[-1]) if has_angle else 1
        message_rank = int(message_left.shape[-1]) if has_message else 1
        block_k = compact_block_k_for_neighbors(k_neighbors)
        block_d = compact_block_d_for_head_dim(head_dim)
        block_r = compact_block_r_for_rank(max(angle_rank if has_angle else 1, message_rank if has_message else 1))
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
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
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
            HAS_ANGLE=has_angle,
            HAS_MESSAGE=has_message,
            INPUT_PRECISION=input_precision,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
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
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
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
        ctx.has_angle = has_angle
        ctx.has_message = has_message
        ctx.angle_rank = angle_rank
        ctx.message_rank = message_rank
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
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
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
            if ctx.has_angle:
                inputs.extend([angle_left.detach().requires_grad_(True), angle_right.detach().requires_grad_(True), angle_gate.detach().requires_grad_(True)])
            if ctx.has_message:
                inputs.extend([message_left.detach().requires_grad_(True), message_right.detach().requires_grad_(True), message_basis.detach().requires_grad_(True)])
            with torch.enable_grad():
                q_re, k1_re, v1_re, k2_re, v2_re = inputs[:5]
                cursor = 5
                local_u = local_v = local_gate = None
                local_left = local_right = local_angle_gate = None
                local_msg_left = local_msg_right = local_msg_basis = None
                if ctx.has_radial_bias:
                    local_u, local_v, local_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_angle:
                    local_left, local_right, local_angle_gate = inputs[cursor : cursor + 3]
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
                    angle_gate=local_angle_gate,
                    message_left=local_msg_left,
                    message_right=local_msg_right,
                    message_basis=local_msg_basis,
                )
                grads = torch.autograd.grad(ref, inputs, grad_out, allow_unused=True)
            dq = dk1 = dv1 = dk2 = dv2 = du = dvb = dgate = dleft = dright = dagate = dml = dmr = dmb = None
            dq, dk1, dv1, dk2, dv2 = grads[:5]
            cursor = 5
            if ctx.has_radial_bias:
                du, dvb, dgate = grads[cursor : cursor + 3]
                cursor += 3
            if ctx.has_angle:
                dleft, dright, dagate = grads[cursor : cursor + 3]
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
                du,
                dvb,
                dgate,
                dleft,
                dright,
                dagate,
                dml,
                dmr,
                dmb,
                None,
                None,
            )

        batch_size, group_order, num_heads, num_atoms, head_dim = q.shape
        k_neighbors = int(neighbor_idx.shape[-1])
        block_k = compact_block_k_for_neighbors(k_neighbors)
        block_d = compact_block_d_for_head_dim(head_dim)
        block_r = compact_block_r_for_rank(max(ctx.angle_rank if ctx.has_angle else 1, ctx.message_rank if ctx.has_message else 1))
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
        dleft = torch.zeros_like(angle_left, dtype=grad_dtype) if ctx.has_angle else torch.empty_like(angle_left)
        dright = torch.zeros_like(angle_right, dtype=grad_dtype) if ctx.has_angle else torch.empty_like(angle_right)
        dagate = torch.zeros_like(angle_gate, dtype=grad_dtype) if ctx.has_angle else torch.empty_like(angle_gate)
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
            u,
            v_bias,
            gate,
            angle_left,
            angle_right,
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
            HAS_ANGLE=ctx.has_angle,
            HAS_MESSAGE=ctx.has_message,
            INPUT_PRECISION=input_precision,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
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
            du if ctx.has_radial_bias else None,
            dvb if ctx.has_radial_bias else None,
            dgate if ctx.has_radial_bias else None,
            dleft if ctx.has_angle else None,
            dright if ctx.has_angle else None,
            dagate if ctx.has_angle else None,
            dml if ctx.has_message else None,
            dmr if ctx.has_message else None,
            dmb if ctx.has_message else None,
            None,
            None,
        )


def _grouped_unavailable_reason(
    q: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    *,
    angle_left: torch.Tensor | None,
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
    reason = _grouped_unavailable_reason(
        q,
        neighbor_idx,
        neighbor_mask,
        angle_left=angle_left,
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
    angle_left_tensor = angle_left.contiguous() if angle_left is not None and angle_right is not None else empty_float
    angle_right_tensor = angle_right.contiguous() if angle_left is not None and angle_right is not None else empty_float
    angle_gate_tensor = (
        angle_gate.contiguous()
        if angle_left is not None and angle_right is not None and angle_gate is not None
        else empty_float
    )
    if angle_left is not None and angle_right is not None and angle_gate is None:
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
        u_tensor,
        v_tensor,
        gate_tensor,
        angle_left_tensor,
        angle_right_tensor,
        angle_gate_tensor,
        message_left_tensor,
        message_right_tensor,
        message_basis_tensor,
        precision,
        bool(debug_torch_backward),
    )
