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
    angle_gate: torch.Tensor | None = None,
    message_left: torch.Tensor | None = None,
    message_right: torch.Tensor | None = None,
    message_basis: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    k1_n = _gather_neighbor_heads(k1, neighbor_idx)
    v1_n = _gather_neighbor_heads(v1, neighbor_idx)
    k2_n = _gather_neighbor_heads(k2, neighbor_idx)
    v2_n = _gather_neighbor_heads(v2, neighbor_idx)
    scores = torch.einsum("bhnd,bhnjd,bhnkd->bhnjk", q, k1_n, k2_n).float()
    if u is not None and v_bias is not None and gate is not None:
        scores = scores + gate[:, :, :, None, None].float() * (
            u[:, :, :, :, None].float() + v_bias[:, :, :, None, :].float()
        )
    if angle_left is not None and angle_right is not None:
        local_angle_gate = 1.0 if angle_gate is None else angle_gate[:, :, :, None, None].float()
        angle = torch.einsum("bhnjr,bhnkr->bhnjk", angle_left.float(), angle_right.float())
        scores = scores + local_angle_gate * angle * (angle_left.shape[-1] ** -0.5)
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
        bh_idx = pid // num_atoms
        batch_idx = bh_idx // num_heads
        head_idx = bh_idx - batch_idx * num_heads

        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, BLOCK_R)
        k_mask = offs_k < k_neighbors
        d_mask = offs_d < head_dim

        n_base = (batch_idx * num_atoms + atom_idx) * k_neighbors
        raw_idx = tl.load(neighbor_idx_ptr + n_base + offs_k, mask=k_mask, other=0)
        valid = (tl.load(neighbor_mask_ptr + n_base + offs_k, mask=k_mask, other=0) > 0) & k_mask
        safe_idx = tl.where(valid, raw_idx, 0)

        q = tl.load(q_ptr + (bh_idx * num_atoms + atom_idx) * head_dim + offs_d, mask=d_mask, other=0.0)
        k1 = tl.load(
            k1_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            k2_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v1_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v2 = tl.load(
            v2_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
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
        score = tl.where(pair_valid, score, -float("inf"))
        max_cols = tl.max(score, axis=0)
        row_max = tl.max(max_cols, axis=0)
        exp_score = tl.where(pair_valid, tl.exp(score - row_max), 0.0)
        denom_cols = tl.sum(exp_score, axis=0)
        denom = tl.sum(denom_cols, axis=0)
        has_valid = denom > 0.0
        probs = tl.where(has_valid, exp_score / denom, 0.0)

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
        tl.store(out_ptr + (bh_idx * num_atoms + atom_idx) * head_dim + offs_d, out, mask=d_mask)
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
        bh_idx = pid // num_atoms
        batch_idx = bh_idx // num_heads
        head_idx = bh_idx - batch_idx * num_heads

        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, BLOCK_R)
        k_mask = offs_k < k_neighbors
        d_mask = offs_d < head_dim

        n_base = (batch_idx * num_atoms + atom_idx) * k_neighbors
        raw_idx = tl.load(neighbor_idx_ptr + n_base + offs_k, mask=k_mask, other=0)
        valid = (tl.load(neighbor_mask_ptr + n_base + offs_k, mask=k_mask, other=0) > 0) & k_mask
        safe_idx = tl.where(valid, raw_idx, 0)

        q = tl.load(q_ptr + (bh_idx * num_atoms + atom_idx) * head_dim + offs_d, mask=d_mask, other=0.0)
        go = tl.load(grad_out_ptr + (bh_idx * num_atoms + atom_idx) * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + (bh_idx * num_atoms + atom_idx) * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        k1 = tl.load(
            k1_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        k2 = tl.load(
            k2_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v1 = tl.load(
            v1_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v2 = tl.load(
            v2_ptr + (bh_idx * num_atoms + safe_idx[:, None]) * head_dim + offs_d[None, :],
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
        score = tl.where(pair_valid, score, -float("inf"))
        max_cols = tl.max(score, axis=0)
        row_max = tl.max(max_cols, axis=0)
        exp_score = tl.where(pair_valid, tl.exp(score - row_max), 0.0)
        denom_cols = tl.sum(exp_score, axis=0)
        denom = tl.sum(denom_cols, axis=0)
        has_valid = denom > 0.0
        probs = tl.where(has_valid, exp_score / denom, 0.0)

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
        dq = tl.sum(tmp_q * k1, axis=0)
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
        if not TRITON_COMPACT_SIMPLICIAL_AVAILABLE:  # pragma: no cover - guarded before dispatch.
            raise RuntimeError("triton is not installed")
        batch_size, num_heads, num_atoms, head_dim = q.shape
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

        out_fp32 = torch.empty((batch_size, num_heads, num_atoms, head_dim), device=q.device, dtype=torch.float32)
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
            if ctx.has_angle:
                inputs.extend(
                    [
                        angle_left.detach().requires_grad_(True),
                        angle_right.detach().requires_grad_(True),
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
                local_msg_left = local_msg_right = local_msg_basis = None
                cursor = 5
                if ctx.has_radial_bias:
                    local_u, local_v, local_gate = inputs[cursor : cursor + 3]
                    cursor += 3
                if ctx.has_angle:
                    local_left, local_right, local_angle_gate = inputs[cursor : cursor + 3]
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

        batch_size, num_heads, num_atoms, head_dim = q.shape
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
        du = torch.empty_like(u, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(u)
        dvb = torch.empty_like(v_bias, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(v_bias)
        dgate = torch.empty_like(gate, dtype=grad_dtype) if ctx.has_radial_bias else torch.empty_like(gate)
        dleft = torch.empty_like(angle_left, dtype=grad_dtype) if ctx.has_angle else torch.empty_like(angle_left)
        dright = torch.empty_like(angle_right, dtype=grad_dtype) if ctx.has_angle else torch.empty_like(angle_right)
        dagate = torch.empty_like(angle_gate, dtype=grad_dtype) if ctx.has_angle else torch.empty_like(angle_gate)
        dml = torch.empty_like(message_left, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_left)
        dmr = torch.empty_like(message_right, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_right)
        dmb = torch.zeros_like(message_basis, dtype=grad_dtype) if ctx.has_message else torch.empty_like(message_basis)
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


def _unavailable_reason(
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
    if angle_left is not None and angle_left.shape[-1] > 64:
        return f"the compact Triton backend only supports angle rank <= 64, got {angle_left.shape[-1]}"
    if message_left is not None and message_left.shape[-1] > 64:
        return f"the compact Triton backend only supports message rank <= 64, got {message_left.shape[-1]}"
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
    precision = normalize_compact_simplicial_precision(precision)
    reason = _unavailable_reason(
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
            raise RuntimeError(f"Compact Triton simplicial attention is unavailable: {reason}")
        return compact_simplicial_attention_torch_reference(
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
        angle_gate_tensor = torch.ones(q.shape[:3], device=q.device, dtype=q.dtype).contiguous()
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
