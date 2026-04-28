from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on GPU nodes with Triton installed.
    triton = None
    tl = None

TRITON_AVAILABLE = triton is not None and tl is not None
SUPPORTED_SIMPLICIAL_PRECISIONS = ("bf16_tc", "tf32", "ieee_fp32")

_BLOCK_J = 32
_BLOCK_K = 32


def normalize_simplicial_precision(precision: str) -> str:
    normalized = str(precision).lower()
    if normalized not in SUPPORTED_SIMPLICIAL_PRECISIONS:
        raise ValueError(
            f"Unsupported simplicial_precision: {precision}. Expected one of {SUPPORTED_SIMPLICIAL_PRECISIONS}."
        )
    return normalized


def triton_block_d_for_head_dim(head_dim: int) -> int:
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
    raise ValueError(f"Triton simplicial attention only supports head_dim <= 128, got {head_dim}")


def _num_warps_for_block_d(block_d: int) -> int:
    if block_d <= 32:
        return 2
    if block_d <= 64:
        return 4
    return 8


def _block_r_for_rank(rank: int) -> int:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank <= 16:
        return 16
    if rank <= 32:
        return 32
    if rank <= 64:
        return 64
    raise ValueError(f"Triton simplicial attention only supports low-rank features <= 64, got {rank}")


def _precision_kernel_config(precision: str) -> tuple[bool, str]:
    normalized = normalize_simplicial_precision(precision)
    if normalized == "bf16_tc":
        return False, "tf32"
    if normalized == "tf32":
        return True, "tf32"
    return True, "ieee"


if TRITON_AVAILABLE:

    @triton.jit
    def _streaming_forward_kernel(
        q_ptr,
        k1_ptr,
        v1_ptr,
        k2_ptr,
        v2_ptr,
        query_valid_ptr,
        pair_key_valid_ptr,
        pair_valid_ptr,
        u_ptr,
        v_bias_ptr,
        w_ptr,
        gate_ptr,
        angle_left_ptr,
        angle_right_ptr,
        angle_gate_ptr,
        message_left_ptr,
        message_right_ptr,
        message_basis_ptr,
        out_ptr,
        lse_ptr,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_LOW_RANK_ANGLE: tl.constexpr,
        HAS_LOW_RANK_MESSAGE: tl.constexpr,
        USE_FP32_INPUT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid % num_tokens
        bh_idx = pid // num_tokens
        batch_idx = bh_idx // num_heads

        q_is_valid = tl.load(query_valid_ptr + batch_idx * num_tokens + q_idx) > 0
        offs_d = tl.arange(0, BLOCK_D)
        out_row_ptr = out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        lse_ptr_row = lse_ptr + bh_idx * num_tokens + q_idx
        if not q_is_valid:
            tl.store(out_row_ptr, 0.0, mask=offs_d < head_dim)
            tl.store(lse_ptr_row, 0.0)
            return

        q_raw = tl.load(
            q_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        )
        offs_r = tl.arange(0, BLOCK_R)
        if USE_FP32_INPUT:
            q_score = q_raw.to(tl.float32)
        else:
            q_score = q_raw

        m = -float("inf")
        l = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if HAS_LOW_RANK_MESSAGE:
            message_acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        for tile_j in range(0, num_tiles_j):
            offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)
            valid_j = (offs_j < num_tokens) & (
                tl.load(
                    pair_key_valid_ptr + batch_idx * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0,
                )
                > 0
            )
            k1_raw = tl.load(
                k1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            v1_raw = tl.load(
                v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            if USE_FP32_INPUT:
                k1_score = k1_raw.to(tl.float32)
            else:
                k1_score = k1_raw

            if HAS_BIAS:
                u = tl.load(
                    u_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
            if HAS_LOW_RANK_ANGLE:
                angle_left_raw = tl.load(
                    angle_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * angle_rank + offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                    other=0.0,
                )
                angle_left_32 = angle_left_raw.to(tl.float32)
                if USE_FP32_INPUT:
                    angle_left_score = angle_left_32
                else:
                    angle_left_score = angle_left_raw
            if HAS_LOW_RANK_MESSAGE:
                message_left_32 = tl.load(
                    message_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * message_rank + offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                    other=0.0,
                ).to(tl.float32)

            for tile_k in range(0, num_tiles_k):
                offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
                valid_k = (offs_k < num_tokens) & (
                    tl.load(
                        pair_key_valid_ptr + batch_idx * num_tokens + offs_k,
                        mask=offs_k < num_tokens,
                        other=0,
                    )
                    > 0
                )
                valid = valid_j[:, None] & valid_k[None, :]
                if HAS_PAIR_VALID:
                    dense_valid = (
                        tl.load(
                            pair_valid_ptr
                            + ((batch_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            other=0,
                        )
                        > 0
                    )
                    valid = valid & dense_valid
                tile_has_any = tl.sum(tl.sum(valid.to(tl.int32), axis=1), axis=0) > 0
                if tile_has_any:
                    k2_raw = tl.load(
                        k2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                        other=0.0,
                    )
                    v2_raw = tl.load(
                        v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                        other=0.0,
                    ).to(tl.float32)
                    if USE_FP32_INPUT:
                        k2_score = k2_raw.to(tl.float32)
                        qk2 = k2_score * q_score[None, :]
                    else:
                        k2_score = k2_raw
                        qk2 = (k2_score * q_score[None, :]).to(k2_score.dtype)

                    scores = tl.dot(
                        k1_score,
                        tl.trans(qk2),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    )

                    if HAS_BIAS:
                        v_bias = tl.load(
                            v_bias_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        w = tl.load(
                            w_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            other=0.0,
                        ).to(tl.float32)
                        gate = tl.load(gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
                        scores = scores + gate * (u[:, None] + v_bias[None, :] + w)

                    if HAS_LOW_RANK_ANGLE:
                        angle_right_raw = tl.load(
                            angle_right_ptr
                            + (
                                ((bh_idx * num_tokens + q_idx) * num_tokens + offs_k[:, None]) * angle_rank
                                + offs_r[None, :]
                            ),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                            other=0.0,
                        )
                        if USE_FP32_INPUT:
                            angle_right_score = angle_right_raw.to(tl.float32)
                        else:
                            angle_right_score = angle_right_raw
                        angle = tl.dot(
                            angle_left_score,
                            tl.trans(angle_right_score),
                            out_dtype=tl.float32,
                            input_precision=INPUT_PRECISION,
                        )
                        angle_gate = tl.load(angle_gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
                        scores = scores + angle_gate * angle * angle_scale
                    if HAS_LOW_RANK_MESSAGE:
                        message_right_32 = tl.load(
                            message_right_ptr
                            + (
                                ((bh_idx * num_tokens + q_idx) * num_tokens + offs_k[:, None]) * message_rank
                                + offs_r[None, :]
                            ),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                            other=0.0,
                        ).to(tl.float32)

                    scores = tl.where(valid, scores, -float("inf"))
                    tile_rowmax = tl.max(scores, axis=1)
                    m_tile = tl.max(tile_rowmax, axis=0)
                    m_new = tl.maximum(m, m_tile)
                    alpha = tl.where(l > 0.0, tl.exp(m - m_new), 0.0)
                    weights = tl.where(valid, tl.exp(scores - m_new), 0.0)
                    l = l * alpha + tl.sum(tl.sum(weights, axis=1), axis=0)
                    tmp_weighted = tl.dot(
                        tl.trans(weights),
                        v1_raw,
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    tile_out = tl.sum(tmp_weighted * v2_raw, axis=0)
                    acc = acc * alpha + tile_out
                    if HAS_LOW_RANK_MESSAGE:
                        tmp_message = tl.dot(
                            tl.trans(weights),
                            message_left_32,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        tile_message = tl.sum(tmp_message * message_right_32, axis=0)
                        message_acc = message_acc * alpha + tile_message
                    m = m_new

        out = tl.where(l > 0.0, acc / l, 0.0)
        if HAS_LOW_RANK_MESSAGE:
            head_idx = bh_idx - batch_idx * num_heads
            message_basis = tl.load(
                message_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                mask=(offs_r[:, None] < message_rank) & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            message_coeff = tl.where(l > 0.0, message_acc * message_scale / l, 0.0)
            message_out = tl.sum(message_coeff[:, None] * message_basis, axis=0)
            out = out + message_out
        lse = tl.where(l > 0.0, m + tl.log(l), 0.0)
        tl.store(out_row_ptr, out, mask=offs_d < head_dim)
        tl.store(lse_ptr_row, lse)


    @triton.jit
    def _streaming_backward_kernel(
        grad_out_ptr,
        q_ptr,
        k1_ptr,
        v1_ptr,
        k2_ptr,
        v2_ptr,
        out_ptr,
        lse_ptr,
        query_valid_ptr,
        pair_key_valid_ptr,
        pair_valid_ptr,
        u_ptr,
        v_bias_ptr,
        w_ptr,
        gate_ptr,
        angle_left_ptr,
        angle_right_ptr,
        angle_gate_ptr,
        message_left_ptr,
        message_right_ptr,
        message_basis_ptr,
        dq_ptr,
        dk1_ptr,
        dv1_ptr,
        dk2_ptr,
        dv2_ptr,
        du_ptr,
        dv_bias_ptr,
        dw_ptr,
        dgate_ptr,
        dangle_left_ptr,
        dangle_right_ptr,
        dangle_gate_ptr,
        dmessage_left_ptr,
        dmessage_right_ptr,
        dmessage_basis_ptr,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_LOW_RANK_ANGLE: tl.constexpr,
        HAS_LOW_RANK_MESSAGE: tl.constexpr,
        USE_FP32_INPUT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid % num_tokens
        bh_idx = pid // num_tokens
        batch_idx = bh_idx // num_heads

        q_is_valid = tl.load(query_valid_ptr + batch_idx * num_tokens + q_idx) > 0
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, BLOCK_R)
        dq_row_ptr = dq_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        if not q_is_valid:
            tl.store(dq_row_ptr, 0.0, mask=offs_d < head_dim)
            if HAS_BIAS:
                tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, 0.0)
            if HAS_LOW_RANK_ANGLE:
                tl.store(dangle_gate_ptr + bh_idx * num_tokens + q_idx, 0.0)
            return

        q_raw = tl.load(
            q_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        )
        if USE_FP32_INPUT:
            q_score = q_raw.to(tl.float32)
        else:
            q_score = q_raw
        q32 = q_raw.to(tl.float32)

        go = tl.load(
            grad_out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)
        out_vec = tl.load(
            out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(lse_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
        go_dot_out = tl.sum(go * out_vec, axis=0)
        if HAS_LOW_RANK_MESSAGE:
            head_idx = bh_idx - batch_idx * num_heads
            message_basis = tl.load(
                message_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                mask=(offs_r[:, None] < message_rank) & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            message_basis_go = tl.sum(message_basis * go[None, :], axis=1)

        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dgate_acc = 0.0
        dangle_gate_acc = 0.0

        for tile_j in range(0, num_tiles_j):
            offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)
            valid_j = (offs_j < num_tokens) & (
                tl.load(
                    pair_key_valid_ptr + batch_idx * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0,
                )
                > 0
            )
            k1_raw = tl.load(
                k1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            k1_32 = k1_raw.to(tl.float32)
            v1_32 = tl.load(
                v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            if USE_FP32_INPUT:
                k1_score = k1_raw.to(tl.float32)
            else:
                k1_score = k1_raw

            du_running = tl.zeros((BLOCK_J,), dtype=tl.float32)
            if HAS_BIAS:
                u = tl.load(
                    u_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                gate = tl.load(gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
            if HAS_LOW_RANK_ANGLE:
                angle_left_raw = tl.load(
                    angle_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * angle_rank + offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                    other=0.0,
                )
                angle_left_32 = angle_left_raw.to(tl.float32)
                if USE_FP32_INPUT:
                    angle_left_score = angle_left_32
                else:
                    angle_left_score = angle_left_raw
                dangle_left_running = tl.zeros((BLOCK_J, BLOCK_R), dtype=tl.float32)
                angle_gate = tl.load(angle_gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
            if HAS_LOW_RANK_MESSAGE:
                message_left_32 = tl.load(
                    message_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * message_rank + offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                    other=0.0,
                ).to(tl.float32)
                dmessage_left_running = tl.zeros((BLOCK_J, BLOCK_R), dtype=tl.float32)

            for tile_k in range(0, num_tiles_k):
                offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
                valid_k = (offs_k < num_tokens) & (
                    tl.load(
                        pair_key_valid_ptr + batch_idx * num_tokens + offs_k,
                        mask=offs_k < num_tokens,
                        other=0,
                    )
                    > 0
                )
                valid = valid_j[:, None] & valid_k[None, :]
                if HAS_PAIR_VALID:
                    dense_valid = (
                        tl.load(
                            pair_valid_ptr
                            + ((batch_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            other=0,
                        )
                        > 0
                    )
                    valid = valid & dense_valid
                tile_has_any = tl.sum(tl.sum(valid.to(tl.int32), axis=1), axis=0) > 0
                if tile_has_any:
                    k2_raw = tl.load(
                        k2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                        other=0.0,
                    )
                    k2_32 = k2_raw.to(tl.float32)
                    v2_32 = tl.load(
                        v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                        other=0.0,
                    ).to(tl.float32)
                    if USE_FP32_INPUT:
                        k2_score = k2_raw.to(tl.float32)
                        qk2 = k2_score * q_score[None, :]
                    else:
                        k2_score = k2_raw
                        qk2 = (k2_score * q_score[None, :]).to(k2_score.dtype)

                    scores = tl.dot(
                        k1_score,
                        tl.trans(qk2),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    )

                    bias_base = tl.zeros((BLOCK_J, BLOCK_K), dtype=tl.float32)
                    if HAS_BIAS:
                        v_bias = tl.load(
                            v_bias_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        w = tl.load(
                            w_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            other=0.0,
                        ).to(tl.float32)
                        bias_base = u[:, None] + v_bias[None, :] + w
                        scores = scores + gate * bias_base

                    angle = tl.zeros((BLOCK_J, BLOCK_K), dtype=tl.float32)
                    if HAS_LOW_RANK_ANGLE:
                        angle_right_raw = tl.load(
                            angle_right_ptr
                            + (
                                ((bh_idx * num_tokens + q_idx) * num_tokens + offs_k[:, None]) * angle_rank
                                + offs_r[None, :]
                            ),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                            other=0.0,
                        )
                        angle_right_32 = angle_right_raw.to(tl.float32)
                        if USE_FP32_INPUT:
                            angle_right_score = angle_right_32
                        else:
                            angle_right_score = angle_right_raw
                        angle = tl.dot(
                            angle_left_score,
                            tl.trans(angle_right_score),
                            out_dtype=tl.float32,
                            input_precision=INPUT_PRECISION,
                        )
                        scores = scores + angle_gate * angle * angle_scale
                    if HAS_LOW_RANK_MESSAGE:
                        message_right_32 = tl.load(
                            message_right_ptr
                            + (
                                ((bh_idx * num_tokens + q_idx) * num_tokens + offs_k[:, None]) * message_rank
                                + offs_r[None, :]
                            ),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                            other=0.0,
                        ).to(tl.float32)

                    scores = tl.where(valid, scores, -float("inf"))
                    probs = tl.where(valid, tl.exp(scores - lse), 0.0)

                    value_dot = tl.dot(
                        v1_32 * go[None, :],
                        tl.trans(v2_32),
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    if HAS_LOW_RANK_MESSAGE:
                        value_dot += message_scale * tl.dot(
                            message_left_32 * message_basis_go[None, :],
                            tl.trans(message_right_32),
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                    dscores = probs * (value_dot - go_dot_out)

                    tmp_k2 = tl.dot(dscores, k2_32, out_dtype=tl.float32, input_precision="ieee")
                    tmp_k1 = tl.dot(tl.trans(dscores), k1_32, out_dtype=tl.float32, input_precision="ieee")
                    dq_acc += tl.sum(tmp_k2 * k1_32, axis=0)

                    dk1_tile = tmp_k2 * q32[None, :]
                    dk2_tile = tmp_k1 * q32[None, :]

                    tl.atomic_add(
                        dk1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                        dk1_tile,
                        mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                    )
                    tl.atomic_add(
                        dk2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        dk2_tile,
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                    )

                    tmp_v2 = tl.dot(probs, v2_32, out_dtype=tl.float32, input_precision="ieee")
                    tmp_v1 = tl.dot(tl.trans(probs), v1_32, out_dtype=tl.float32, input_precision="ieee")
                    dv1_tile = tmp_v2 * go[None, :]
                    dv2_tile = tmp_v1 * go[None, :]

                    tl.atomic_add(
                        dv1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                        dv1_tile,
                        mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                    )
                    tl.atomic_add(
                        dv2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        dv2_tile,
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                    )

                    if HAS_BIAS:
                        dsum_k = tl.sum(dscores, axis=1)
                        dsum_j = tl.sum(dscores, axis=0)
                        du_running += gate * dsum_k

                        dv_bias_row_ptr = dv_bias_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k
                        dv_bias_prev = tl.load(dv_bias_row_ptr, mask=offs_k < num_tokens, other=0.0)
                        tl.store(
                            dv_bias_row_ptr,
                            dv_bias_prev + gate * dsum_j,
                            mask=offs_k < num_tokens,
                        )

                        tl.atomic_add(
                            dw_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            gate * dscores,
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                        )
                        dgate_acc += tl.sum(tl.sum(dscores * bias_base, axis=1), axis=0)

                    if HAS_LOW_RANK_ANGLE:
                        angle_grad_scale = angle_gate * angle_scale
                        dangle_left_running += angle_grad_scale * tl.dot(
                            dscores,
                            angle_right_32,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        dangle_right_tile = angle_grad_scale * tl.dot(
                            tl.trans(dscores),
                            angle_left_32,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        dangle_right_tile_ptr = (
                            dangle_right_ptr
                            + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_k[:, None]) * angle_rank + offs_r[None, :])
                        )
                        dangle_right_prev = tl.load(
                            dangle_right_tile_ptr,
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                            other=0.0,
                        )
                        tl.store(
                            dangle_right_tile_ptr,
                            dangle_right_prev + dangle_right_tile,
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                        )
                        dangle_gate_acc += tl.sum(tl.sum(dscores * angle * angle_scale, axis=1), axis=0)

                    if HAS_LOW_RANK_MESSAGE:
                        message_tmp_right = tl.dot(
                            probs,
                            message_right_32,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        message_tmp_left = tl.dot(
                            tl.trans(probs),
                            message_left_32,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        dmessage_left_running += message_scale * message_tmp_right * message_basis_go[None, :]
                        dmessage_right_tile = message_scale * message_tmp_left * message_basis_go[None, :]
                        dmessage_right_tile_ptr = (
                            dmessage_right_ptr
                            + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_k[:, None]) * message_rank + offs_r[None, :])
                        )
                        dmessage_right_prev = tl.load(
                            dmessage_right_tile_ptr,
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                            other=0.0,
                        )
                        tl.store(
                            dmessage_right_tile_ptr,
                            dmessage_right_prev + dmessage_right_tile,
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                        )
                        message_coeff = tl.sum(message_tmp_left * message_right_32, axis=0)
                        tl.atomic_add(
                            dmessage_basis_ptr
                            + (head_idx * message_rank + offs_r[:, None]) * head_dim
                            + offs_d[None, :],
                            message_scale * message_coeff[:, None] * go[None, :],
                            mask=(offs_r[:, None] < message_rank) & (offs_d[None, :] < head_dim),
                        )

            if HAS_BIAS:
                tl.store(
                    du_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    du_running,
                    mask=offs_j < num_tokens,
                )
            if HAS_LOW_RANK_ANGLE:
                tl.store(
                    dangle_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * angle_rank + offs_r[None, :]),
                    dangle_left_running,
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                )
            if HAS_LOW_RANK_MESSAGE:
                tl.store(
                    dmessage_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * message_rank + offs_r[None, :]),
                    dmessage_left_running,
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                )

        tl.store(dq_row_ptr, dq_acc, mask=offs_d < head_dim)
        if HAS_BIAS:
            tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, dgate_acc)
        if HAS_LOW_RANK_ANGLE:
            tl.store(dangle_gate_ptr + bh_idx * num_tokens + q_idx, dangle_gate_acc)


def _reshape_batch_heads(tensor: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, num_tokens = tensor.shape[:3]
    return tensor.contiguous().view(batch_size * num_heads, num_tokens, *tensor.shape[3:])


def triton_simplicial_attention_forward(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    pair_key_valid: torch.Tensor,
    pair_valid: torch.Tensor | None,
    u: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    w: torch.Tensor | None,
    gate: torch.Tensor | None,
    angle_left: torch.Tensor | None,
    angle_right: torch.Tensor | None,
    angle_gate: torch.Tensor | None,
    message_left: torch.Tensor | None,
    message_right: torch.Tensor | None,
    message_basis: torch.Tensor | None,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by dispatch logic.
        raise RuntimeError("triton is not installed")
    if q.device.type != "cuda":
        raise RuntimeError("triton_simplicial_attention_forward requires CUDA tensors")

    batch_size, num_heads, num_tokens, head_dim = q.shape
    block_d = triton_block_d_for_head_dim(head_dim)
    use_fp32_input, input_precision = _precision_kernel_config(precision)

    q_bh = _reshape_batch_heads(q)
    k1_bh = _reshape_batch_heads(k1)
    v1_bh = _reshape_batch_heads(v1)
    k2_bh = _reshape_batch_heads(k2)
    v2_bh = _reshape_batch_heads(v2)

    has_pair_valid = pair_valid is not None
    pair_valid_tensor = pair_valid.contiguous() if has_pair_valid else torch.empty(0, device=q.device, dtype=torch.bool)
    has_bias = u is not None and v_bias is not None and w is not None and gate is not None
    empty = q_bh.new_empty(0, dtype=torch.float32)
    if has_bias:
        u_bh = _reshape_batch_heads(u)
        v_bias_bh = _reshape_batch_heads(v_bias)
        w_bh = _reshape_batch_heads(w)
        gate_bh = gate.contiguous().view(batch_size * num_heads, num_tokens)
    else:
        u_bh = v_bias_bh = w_bh = gate_bh = empty

    has_low_rank_angle = angle_left is not None and angle_right is not None and angle_gate is not None
    if has_low_rank_angle:
        if angle_left.shape != angle_right.shape:
            raise ValueError(f"angle_left and angle_right must have the same shape, got {angle_left.shape} and {angle_right.shape}")
        angle_rank = int(angle_left.shape[-1])
        angle_scale = angle_rank**-0.5
        angle_left_bh = _reshape_batch_heads(angle_left)
        angle_right_bh = _reshape_batch_heads(angle_right)
        angle_gate_bh = angle_gate.contiguous().view(batch_size * num_heads, num_tokens)
    else:
        angle_rank = 1
        angle_scale = 1.0
        angle_left_bh = angle_right_bh = angle_gate_bh = empty

    has_low_rank_message = message_left is not None and message_right is not None and message_basis is not None
    if has_low_rank_message:
        if message_left.shape != message_right.shape:
            raise ValueError(
                f"message_left and message_right must have the same shape, got {message_left.shape} and {message_right.shape}"
            )
        message_rank = int(message_left.shape[-1])
        if message_basis.shape != (num_heads, message_rank, head_dim):
            raise ValueError(
                f"message_basis must have shape {(num_heads, message_rank, head_dim)}, got {tuple(message_basis.shape)}"
            )
        message_scale = message_rank**-0.5
        message_left_bh = _reshape_batch_heads(message_left)
        message_right_bh = _reshape_batch_heads(message_right)
        message_basis_h = message_basis.contiguous()
    else:
        message_rank = 1
        message_scale = 1.0
        message_left_bh = message_right_bh = message_basis_h = empty

    block_r = _block_r_for_rank(max(angle_rank if has_low_rank_angle else 1, message_rank if has_low_rank_message else 1))

    batch_heads = batch_size * num_heads
    out = torch.empty((batch_heads, num_tokens, head_dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch_heads, num_tokens), device=q.device, dtype=torch.float32)
    num_tiles_j = triton.cdiv(num_tokens, _BLOCK_J)
    num_tiles_k = triton.cdiv(num_tokens, _BLOCK_K)
    grid = (batch_heads * num_tokens,)
    _streaming_forward_kernel[grid](
        q_bh,
        k1_bh,
        v1_bh,
        k2_bh,
        v2_bh,
        query_valid.contiguous(),
        pair_key_valid.contiguous(),
        pair_valid_tensor,
        u_bh,
        v_bias_bh,
        w_bh,
        gate_bh,
        angle_left_bh,
        angle_right_bh,
        angle_gate_bh,
        message_left_bh,
        message_right_bh,
        message_basis_h,
        out,
        lse,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_BIAS=has_bias,
        HAS_LOW_RANK_ANGLE=has_low_rank_angle,
        HAS_LOW_RANK_MESSAGE=has_low_rank_message,
        USE_FP32_INPUT=use_fp32_input,
        INPUT_PRECISION=input_precision,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=_num_warps_for_block_d(block_d),
    )
    out_for_backward = out.view(batch_size, num_heads, num_tokens, head_dim)
    out = out_for_backward.to(q.dtype)
    lse = lse.view(batch_size, num_heads, num_tokens)
    return out, lse, out_for_backward


def triton_simplicial_attention_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    pair_key_valid: torch.Tensor,
    pair_valid: torch.Tensor | None,
    u: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    w: torch.Tensor | None,
    gate: torch.Tensor | None,
    angle_left: torch.Tensor | None,
    angle_right: torch.Tensor | None,
    angle_gate: torch.Tensor | None,
    message_left: torch.Tensor | None,
    message_right: torch.Tensor | None,
    message_basis: torch.Tensor | None,
    precision: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by dispatch logic.
        raise RuntimeError("triton is not installed")
    if q.device.type != "cuda":
        raise RuntimeError("triton_simplicial_attention_backward requires CUDA tensors")

    batch_size, num_heads, num_tokens, head_dim = q.shape
    block_d = triton_block_d_for_head_dim(head_dim)
    use_fp32_input, input_precision = _precision_kernel_config(precision)

    grad_out_bh = _reshape_batch_heads(grad_out)
    q_bh = _reshape_batch_heads(q)
    k1_bh = _reshape_batch_heads(k1)
    v1_bh = _reshape_batch_heads(v1)
    k2_bh = _reshape_batch_heads(k2)
    v2_bh = _reshape_batch_heads(v2)
    out_bh = _reshape_batch_heads(out)
    lse_bh = lse.contiguous().view(batch_size * num_heads, num_tokens)

    dq = torch.zeros_like(q_bh, dtype=torch.float32)
    dk1 = torch.zeros_like(k1_bh, dtype=torch.float32)
    dv1 = torch.zeros_like(v1_bh, dtype=torch.float32)
    dk2 = torch.zeros_like(k2_bh, dtype=torch.float32)
    dv2 = torch.zeros_like(v2_bh, dtype=torch.float32)

    has_pair_valid = pair_valid is not None
    pair_valid_tensor = pair_valid.contiguous() if has_pair_valid else torch.empty(0, device=q.device, dtype=torch.bool)
    has_bias = u is not None and v_bias is not None and w is not None and gate is not None
    empty = q_bh.new_empty(0, dtype=torch.float32)
    if has_bias:
        u_bh = _reshape_batch_heads(u)
        v_bias_bh = _reshape_batch_heads(v_bias)
        w_bh = _reshape_batch_heads(w)
        gate_bh = gate.contiguous().view(batch_size * num_heads, num_tokens)
        du = torch.zeros_like(u_bh, dtype=torch.float32)
        dv_bias = torch.zeros_like(v_bias_bh, dtype=torch.float32)
        dw = torch.zeros_like(w_bh, dtype=torch.float32)
        dgate = torch.zeros_like(gate_bh, dtype=torch.float32)
    else:
        u_bh = v_bias_bh = w_bh = gate_bh = empty
        du = dv_bias = dw = dgate = None

    has_low_rank_angle = angle_left is not None and angle_right is not None and angle_gate is not None
    if has_low_rank_angle:
        if angle_left.shape != angle_right.shape:
            raise ValueError(f"angle_left and angle_right must have the same shape, got {angle_left.shape} and {angle_right.shape}")
        angle_rank = int(angle_left.shape[-1])
        angle_scale = angle_rank**-0.5
        angle_left_bh = _reshape_batch_heads(angle_left)
        angle_right_bh = _reshape_batch_heads(angle_right)
        angle_gate_bh = angle_gate.contiguous().view(batch_size * num_heads, num_tokens)
        dangle_left = torch.zeros_like(angle_left_bh, dtype=torch.float32)
        dangle_right = torch.zeros_like(angle_right_bh, dtype=torch.float32)
        dangle_gate = torch.zeros_like(angle_gate_bh, dtype=torch.float32)
    else:
        angle_rank = 1
        angle_scale = 1.0
        angle_left_bh = angle_right_bh = angle_gate_bh = empty
        dangle_left = dangle_right = dangle_gate = None

    has_low_rank_message = message_left is not None and message_right is not None and message_basis is not None
    if has_low_rank_message:
        if message_left.shape != message_right.shape:
            raise ValueError(
                f"message_left and message_right must have the same shape, got {message_left.shape} and {message_right.shape}"
            )
        message_rank = int(message_left.shape[-1])
        if message_basis.shape != (num_heads, message_rank, head_dim):
            raise ValueError(
                f"message_basis must have shape {(num_heads, message_rank, head_dim)}, got {tuple(message_basis.shape)}"
            )
        message_scale = message_rank**-0.5
        message_left_bh = _reshape_batch_heads(message_left)
        message_right_bh = _reshape_batch_heads(message_right)
        message_basis_h = message_basis.contiguous()
        dmessage_left = torch.zeros_like(message_left_bh, dtype=torch.float32)
        dmessage_right = torch.zeros_like(message_right_bh, dtype=torch.float32)
        dmessage_basis = torch.zeros_like(message_basis_h, dtype=torch.float32)
    else:
        message_rank = 1
        message_scale = 1.0
        message_left_bh = message_right_bh = message_basis_h = empty
        dmessage_left = dmessage_right = dmessage_basis = None

    block_r = _block_r_for_rank(max(angle_rank if has_low_rank_angle else 1, message_rank if has_low_rank_message else 1))

    batch_heads = batch_size * num_heads
    num_tiles_j = triton.cdiv(num_tokens, _BLOCK_J)
    num_tiles_k = triton.cdiv(num_tokens, _BLOCK_K)
    grid = (batch_heads * num_tokens,)
    _streaming_backward_kernel[grid](
        grad_out_bh,
        q_bh,
        k1_bh,
        v1_bh,
        k2_bh,
        v2_bh,
        out_bh,
        lse_bh,
        query_valid.contiguous(),
        pair_key_valid.contiguous(),
        pair_valid_tensor,
        u_bh,
        v_bias_bh,
        w_bh,
        gate_bh,
        angle_left_bh,
        angle_right_bh,
        angle_gate_bh,
        message_left_bh,
        message_right_bh,
        message_basis_h,
        dq,
        dk1,
        dv1,
        dk2,
        dv2,
        du if du is not None else empty,
        dv_bias if dv_bias is not None else empty,
        dw if dw is not None else empty,
        dgate if dgate is not None else empty,
        dangle_left if dangle_left is not None else empty,
        dangle_right if dangle_right is not None else empty,
        dangle_gate if dangle_gate is not None else empty,
        dmessage_left if dmessage_left is not None else empty,
        dmessage_right if dmessage_right is not None else empty,
        dmessage_basis if dmessage_basis is not None else empty,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_BIAS=has_bias,
        HAS_LOW_RANK_ANGLE=has_low_rank_angle,
        HAS_LOW_RANK_MESSAGE=has_low_rank_message,
        USE_FP32_INPUT=use_fp32_input,
        INPUT_PRECISION=input_precision,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=_num_warps_for_block_d(block_d),
    )

    reshape = (batch_size, num_heads, num_tokens, head_dim)
    dq_out = dq.view(reshape).to(q.dtype)
    dk1_out = dk1.view(reshape).to(k1.dtype)
    dv1_out = dv1.view(reshape).to(v1.dtype)
    dk2_out = dk2.view(reshape).to(k2.dtype)
    dv2_out = dv2.view(reshape).to(v2.dtype)

    if has_bias:
        pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
        du_out = du.view(pair_shape).to(u.dtype)
        dv_bias_out = dv_bias.view(pair_shape).to(v_bias.dtype)
        dw_out = dw.view(pair_shape).to(w.dtype)
        dgate_out = dgate.view(batch_size, num_heads, num_tokens).to(gate.dtype)
    else:
        du_out = dv_bias_out = dw_out = dgate_out = None

    if has_low_rank_angle:
        angle_shape = (batch_size, num_heads, num_tokens, num_tokens, angle_rank)
        dangle_left_out = dangle_left.view(angle_shape).to(angle_left.dtype)
        dangle_right_out = dangle_right.view(angle_shape).to(angle_right.dtype)
        dangle_gate_out = dangle_gate.view(batch_size, num_heads, num_tokens).to(angle_gate.dtype)
    else:
        dangle_left_out = dangle_right_out = dangle_gate_out = None

    if has_low_rank_message:
        message_shape = (batch_size, num_heads, num_tokens, num_tokens, message_rank)
        dmessage_left_out = dmessage_left.view(message_shape).to(message_left.dtype)
        dmessage_right_out = dmessage_right.view(message_shape).to(message_right.dtype)
        dmessage_basis_out = dmessage_basis.to(message_basis.dtype)
    else:
        dmessage_left_out = dmessage_right_out = dmessage_basis_out = None

    return (
        dq_out,
        dk1_out,
        dv1_out,
        dk2_out,
        dv2_out,
        du_out,
        dv_bias_out,
        dw_out,
        dgate_out,
        dangle_left_out,
        dangle_right_out,
        dangle_gate_out,
        dmessage_left_out,
        dmessage_right_out,
        dmessage_basis_out,
    )
