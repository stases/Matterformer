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
        u_extra_ptr,
        v_bias_extra_ptr,
        w_extra_ptr,
        gate_extra_ptr,
        angle_left_ptr,
        angle_right_ptr,
        angle_gate_ptr,
        message_left_ptr,
        message_right_ptr,
        message_basis_ptr,
        pair_value_positions_ptr,
        pair_value_freqs_ptr,
        pair_value_scale_ptr,
        pair_value_query_mask_ptr,
        out_ptr,
        lse_ptr,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        pair_value_n_freqs,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_BIAS_EXTRA: tl.constexpr,
        HAS_LOW_RANK_ANGLE: tl.constexpr,
        HAS_LOW_RANK_MESSAGE: tl.constexpr,
        HAS_PAIR_VALUE_MARGINAL: tl.constexpr,
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
        head_idx = bh_idx - batch_idx * num_heads

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
        if HAS_PAIR_VALUE_MARGINAL:
            pair_value_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            pair_value_query_gate = tl.load(pair_value_query_mask_ptr + batch_idx * num_tokens + q_idx).to(tl.float32)
            pair_value_scale = tl.load(pair_value_scale_ptr + head_idx).to(tl.float32) * pair_value_query_gate
            pos_i_x = tl.load(pair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 0).to(tl.float32)
            pos_i_y = tl.load(pair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 1).to(tl.float32)
            pos_i_z = tl.load(pair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 2).to(tl.float32)
            freq_x = tl.load(
                pair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 0,
                mask=offs_r < pair_value_n_freqs,
                other=0.0,
            ).to(tl.float32)
            freq_y = tl.load(
                pair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 1,
                mask=offs_r < pair_value_n_freqs,
                other=0.0,
            ).to(tl.float32)
            freq_z = tl.load(
                pair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 2,
                mask=offs_r < pair_value_n_freqs,
                other=0.0,
            ).to(tl.float32)

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
            if HAS_BIAS_EXTRA:
                u_extra = tl.load(
                    u_extra_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
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
            if HAS_PAIR_VALUE_MARGINAL:
                pos_j_x = tl.load(
                    pair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 0,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                pos_j_y = tl.load(
                    pair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 1,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                pos_j_z = tl.load(
                    pair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 2,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                theta_j = (
                    (pos_j_x - pos_i_x)[:, None] * freq_x[None, :]
                    + (pos_j_y - pos_i_y)[:, None] * freq_y[None, :]
                    + (pos_j_z - pos_i_z)[:, None] * freq_z[None, :]
                )
                cos_j = tl.cos(theta_j)
                sin_j = tl.sin(theta_j)
                v1_r = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                v1_i = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_r[None, :] + 1),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                rot1_r = v1_r * cos_j - v1_i * sin_j
                rot1_i = v1_r * sin_j + v1_i * cos_j
                rot1_r = tl.where((offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs), rot1_r, 0.0)
                rot1_i = tl.where((offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs), rot1_i, 0.0)

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

                    if HAS_BIAS_EXTRA:
                        v_bias_extra = tl.load(
                            v_bias_extra_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        w_extra = tl.load(
                            w_extra_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            other=0.0,
                        ).to(tl.float32)
                        gate_extra = tl.load(gate_extra_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
                        scores = scores + gate_extra * (u_extra[:, None] + v_bias_extra[None, :] + w_extra)

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
                    if HAS_PAIR_VALUE_MARGINAL:
                        pos_k_x = tl.load(
                            pair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 0,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        pos_k_y = tl.load(
                            pair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 1,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        pos_k_z = tl.load(
                            pair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 2,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        theta_k = (
                            (pos_k_x - pos_i_x)[:, None] * freq_x[None, :]
                            + (pos_k_y - pos_i_y)[:, None] * freq_y[None, :]
                            + (pos_k_z - pos_i_z)[:, None] * freq_z[None, :]
                        )
                        cos_k = tl.cos(theta_k)
                        sin_k = tl.sin(theta_k)
                        v2_r = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_r[None, :]),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        v2_i = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_r[None, :] + 1),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        rot2_r = v2_r * cos_k - v2_i * sin_k
                        rot2_i = v2_r * sin_k + v2_i * cos_k
                        rot2_r = tl.where(
                            (offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            rot2_r,
                            0.0,
                        )
                        rot2_i = tl.where(
                            (offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            rot2_i,
                            0.0,
                        )

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
                    if HAS_PAIR_VALUE_MARGINAL:
                        row_marginal = tl.sum(weights, axis=1)
                        col_marginal = tl.sum(weights, axis=0)
                        transport_r = tl.sum(row_marginal[:, None] * rot1_r, axis=0) + tl.sum(
                            col_marginal[:, None] * rot2_r,
                            axis=0,
                        )
                        transport_i = tl.sum(row_marginal[:, None] * rot1_i, axis=0) + tl.sum(
                            col_marginal[:, None] * rot2_i,
                            axis=0,
                        )
                        dim_to_freq_real = offs_d[:, None] == 2 * offs_r[None, :]
                        dim_to_freq_imag = offs_d[:, None] == (2 * offs_r[None, :] + 1)
                        transport_matrix = tl.where(dim_to_freq_real, transport_r[None, :], 0.0) + tl.where(
                            dim_to_freq_imag,
                            transport_i[None, :],
                            0.0,
                        )
                        transport_matrix = tl.where(offs_r[None, :] < pair_value_n_freqs, transport_matrix, 0.0)
                        pair_value_acc = pair_value_acc * alpha + pair_value_scale * tl.sum(transport_matrix, axis=1)
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
        if HAS_PAIR_VALUE_MARGINAL:
            out = out + tl.where(l > 0.0, pair_value_acc / l, 0.0)
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
    def _streaming_forward_native_rope_kernel(
        q_content_ptr,
        q_rope_ptr,
        k1_ptr,
        v1_ptr,
        k2_ptr,
        v2_ptr,
        query_valid_ptr,
        pair_key_valid_ptr,
        pair_valid_ptr,
        position_query_valid_ptr,
        u_ptr,
        v_bias_ptr,
        w_ptr,
        gate_ptr,
        positions_ptr,
        mu_ptr,
        nu_ptr,
        rope_logit_scale_ptr,
        rope_value_scale_ptr,
        out_ptr,
        lse_ptr,
        num_tokens,
        head_dim,
        rope_n_freqs,
        rope_value_n_freqs,
        rope_logit_norm,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_POSITION_QUERY_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_ROPE_VALUE_CARRIER: tl.constexpr,
        USE_FP32_INPUT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid % num_tokens
        bh_idx = pid // num_tokens
        batch_idx = bh_idx // num_heads
        head_idx = bh_idx - batch_idx * num_heads

        q_is_valid = tl.load(query_valid_ptr + batch_idx * num_tokens + q_idx) > 0
        offs_d = tl.arange(0, BLOCK_D)
        offs_f = tl.arange(0, BLOCK_F)
        out_row_ptr = out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        lse_ptr_row = lse_ptr + bh_idx * num_tokens + q_idx
        if not q_is_valid:
            tl.store(out_row_ptr, 0.0, mask=offs_d < head_dim)
            tl.store(lse_ptr_row, 0.0)
            return

        position_query_gate = 1.0
        if HAS_POSITION_QUERY_VALID:
            position_query_gate = tl.load(position_query_valid_ptr + batch_idx * num_tokens + q_idx).to(tl.float32)

        q_raw = tl.load(
            q_content_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        )
        if USE_FP32_INPUT:
            q_score = q_raw.to(tl.float32)
        else:
            q_score = q_raw

        q_rope_r = tl.load(
            q_rope_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_f,
            mask=(offs_f < rope_n_freqs) & ((2 * offs_f) < head_dim),
            other=0.0,
        ).to(tl.float32)
        q_rope_i = tl.load(
            q_rope_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_f + 1,
            mask=(offs_f < rope_n_freqs) & ((2 * offs_f + 1) < head_dim),
            other=0.0,
        ).to(tl.float32)

        mu_x = tl.load(
            mu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        mu_y = tl.load(
            mu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        mu_z = tl.load(
            mu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        nu_x = tl.load(
            nu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        nu_y = tl.load(
            nu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        nu_z = tl.load(
            nu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        a_x = 0.5 * (mu_x + nu_x)
        a_y = 0.5 * (mu_y + nu_y)
        a_z = 0.5 * (mu_z + nu_z)
        b_x = 0.5 * (mu_x - nu_x)
        b_y = 0.5 * (mu_y - nu_y)
        b_z = 0.5 * (mu_z - nu_z)
        carrier_dim = 2 * rope_value_n_freqs

        pos_i_x = tl.load(positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 0).to(tl.float32)
        pos_i_y = tl.load(positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 1).to(tl.float32)
        pos_i_z = tl.load(positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 2).to(tl.float32)
        rope_logit_scale = (
            tl.load(rope_logit_scale_ptr + head_idx).to(tl.float32)
            * rope_logit_norm
            * position_query_gate
        )
        rope_value_scale = 0.0
        if HAS_ROPE_VALUE_CARRIER:
            rope_value_scale = tl.load(rope_value_scale_ptr + head_idx).to(tl.float32) * position_query_gate

        m = -float("inf")
        l = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

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
            if HAS_ROPE_VALUE_CARRIER:
                carrier_dim = 2 * rope_value_n_freqs
                v1_base = tl.where(offs_d[None, :] < carrier_dim, 0.0, v1_raw)
            else:
                v1_base = v1_raw
            if USE_FP32_INPUT:
                k1_score = k1_raw.to(tl.float32)
            else:
                k1_score = k1_raw

            pos_j_x = tl.load(
                positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 0,
                mask=offs_j < num_tokens,
                other=0.0,
            ).to(tl.float32)
            pos_j_y = tl.load(
                positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 1,
                mask=offs_j < num_tokens,
                other=0.0,
            ).to(tl.float32)
            pos_j_z = tl.load(
                positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 2,
                mask=offs_j < num_tokens,
                other=0.0,
            ).to(tl.float32)
            rel_j_x = pos_j_x - pos_i_x
            rel_j_y = pos_j_y - pos_i_y
            rel_j_z = pos_j_z - pos_i_z
            phi = (
                rel_j_x[:, None] * a_x[None, :]
                + rel_j_y[:, None] * a_y[None, :]
                + rel_j_z[:, None] * a_z[None, :]
            )
            cos_phi = tl.cos(phi)
            sin_phi = tl.sin(phi)
            left_even = q_rope_r[None, :] * cos_phi + q_rope_i[None, :] * sin_phi
            left_odd = -q_rope_r[None, :] * sin_phi + q_rope_i[None, :] * cos_phi
            left_even = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs), left_even, 0.0)
            left_odd = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs), left_odd, 0.0)

            if HAS_BIAS:
                u = tl.load(
                    u_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)

            if HAS_ROPE_VALUE_CARRIER:
                v1_r = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_f[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                v1_i = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_f[None, :] + 1),
                    mask=(offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                a_r = v1_r * cos_phi - v1_i * sin_phi
                a_i = v1_r * sin_phi + v1_i * cos_phi
                a_r = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs), a_r, 0.0)
                a_i = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs), a_i, 0.0)

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
                    if HAS_ROPE_VALUE_CARRIER:
                        v2_base = tl.where(offs_d[None, :] < carrier_dim, 0.0, v2_raw)
                    else:
                        v2_base = v2_raw
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

                    pos_k_x = tl.load(
                        positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 0,
                        mask=offs_k < num_tokens,
                        other=0.0,
                    ).to(tl.float32)
                    pos_k_y = tl.load(
                        positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 1,
                        mask=offs_k < num_tokens,
                        other=0.0,
                    ).to(tl.float32)
                    pos_k_z = tl.load(
                        positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 2,
                        mask=offs_k < num_tokens,
                        other=0.0,
                    ).to(tl.float32)
                    rel_k_x = pos_k_x - pos_i_x
                    rel_k_y = pos_k_y - pos_i_y
                    rel_k_z = pos_k_z - pos_i_z
                    psi = (
                        rel_k_x[:, None] * b_x[None, :]
                        + rel_k_y[:, None] * b_y[None, :]
                        + rel_k_z[:, None] * b_z[None, :]
                    )
                    cos_psi = tl.cos(psi)
                    sin_psi = tl.sin(psi)
                    right_even = tl.where(
                        (offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs),
                        cos_psi,
                        0.0,
                    )
                    right_odd = tl.where(
                        (offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs),
                        sin_psi,
                        0.0,
                    )
                    rope_scores = tl.dot(
                        left_even,
                        tl.trans(right_even),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    ) + tl.dot(
                        left_odd,
                        tl.trans(right_odd),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    )
                    scores = scores + rope_logit_scale * rope_scores

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

                    scores = tl.where(valid, scores, -float("inf"))
                    tile_rowmax = tl.max(scores, axis=1)
                    m_tile = tl.max(tile_rowmax, axis=0)
                    m_new = tl.maximum(m, m_tile)
                    alpha = tl.where(l > 0.0, tl.exp(m - m_new), 0.0)
                    weights = tl.where(valid, tl.exp(scores - m_new), 0.0)
                    l = l * alpha + tl.sum(tl.sum(weights, axis=1), axis=0)

                    tmp_weighted = tl.dot(
                        tl.trans(weights),
                        v1_base,
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    tile_out = tl.sum(tmp_weighted * v2_base, axis=0)

                    if HAS_ROPE_VALUE_CARRIER:
                        v2_r = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_f[None, :]),
                            mask=(offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        v2_i = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_f[None, :] + 1),
                            mask=(offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        b_r = v2_r * cos_psi - v2_i * sin_psi
                        b_i = v2_r * sin_psi + v2_i * cos_psi
                        b_r = tl.where(
                            (offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                            b_r,
                            0.0,
                        )
                        b_i = tl.where(
                            (offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                            b_i,
                            0.0,
                        )
                        tmp_a_r = tl.dot(
                            tl.trans(weights),
                            a_r,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        tmp_a_i = tl.dot(
                            tl.trans(weights),
                            a_i,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        carrier_real = rope_value_scale * tl.sum(tmp_a_r * b_r - tmp_a_i * b_i, axis=0)
                        carrier_imag = rope_value_scale * tl.sum(tmp_a_r * b_i + tmp_a_i * b_r, axis=0)
                        dim_to_freq = offs_d[:, None] == 2 * offs_f[None, :]
                        dim_to_freq_imag = offs_d[:, None] == (2 * offs_f[None, :] + 1)
                        carrier_matrix = tl.where(dim_to_freq, carrier_real[None, :], 0.0) + tl.where(
                            dim_to_freq_imag,
                            carrier_imag[None, :],
                            0.0,
                        )
                        carrier_matrix = tl.where(offs_f[None, :] < rope_value_n_freqs, carrier_matrix, 0.0)
                        tile_out = tile_out + tl.sum(carrier_matrix, axis=1)

                    acc = acc * alpha + tile_out
                    m = m_new

        out = tl.where(l > 0.0, acc / l, 0.0)
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
        u_extra_ptr,
        v_bias_extra_ptr,
        w_extra_ptr,
        gate_extra_ptr,
        angle_left_ptr,
        angle_right_ptr,
        angle_gate_ptr,
        message_left_ptr,
        message_right_ptr,
        message_basis_ptr,
        pair_value_positions_ptr,
        pair_value_freqs_ptr,
        pair_value_scale_ptr,
        pair_value_query_mask_ptr,
        dq_ptr,
        dk1_ptr,
        dv1_ptr,
        dk2_ptr,
        dv2_ptr,
        du_ptr,
        dv_bias_ptr,
        dw_ptr,
        dgate_ptr,
        du_extra_ptr,
        dv_bias_extra_ptr,
        dw_extra_ptr,
        dgate_extra_ptr,
        dangle_left_ptr,
        dangle_right_ptr,
        dangle_gate_ptr,
        dmessage_left_ptr,
        dmessage_right_ptr,
        dmessage_basis_ptr,
        dpair_value_positions_ptr,
        dpair_value_freqs_ptr,
        dpair_value_scale_ptr,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        pair_value_n_freqs,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_BIAS_EXTRA: tl.constexpr,
        HAS_LOW_RANK_ANGLE: tl.constexpr,
        HAS_LOW_RANK_MESSAGE: tl.constexpr,
        HAS_PAIR_VALUE_MARGINAL: tl.constexpr,
        COMPUTE_DU: tl.constexpr,
        COMPUTE_DV_BIAS: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
        COMPUTE_DGATE: tl.constexpr,
        COMPUTE_DU_EXTRA: tl.constexpr,
        COMPUTE_DV_BIAS_EXTRA: tl.constexpr,
        COMPUTE_DW_EXTRA: tl.constexpr,
        COMPUTE_DGATE_EXTRA: tl.constexpr,
        COMPUTE_DANGLE_LEFT: tl.constexpr,
        COMPUTE_DANGLE_RIGHT: tl.constexpr,
        COMPUTE_DANGLE_GATE: tl.constexpr,
        COMPUTE_DMESSAGE_LEFT: tl.constexpr,
        COMPUTE_DMESSAGE_RIGHT: tl.constexpr,
        COMPUTE_DMESSAGE_BASIS: tl.constexpr,
        COMPUTE_DPAIR_VALUE_POSITIONS: tl.constexpr,
        COMPUTE_DPAIR_VALUE_FREQS: tl.constexpr,
        COMPUTE_DPAIR_VALUE_SCALE: tl.constexpr,
        COMPUTE_DPAIR_VALUE_GEOMETRY: tl.constexpr,
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
        head_idx = bh_idx - batch_idx * num_heads

        q_is_valid = tl.load(query_valid_ptr + batch_idx * num_tokens + q_idx) > 0
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, BLOCK_R)
        dq_row_ptr = dq_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        if not q_is_valid:
            tl.store(dq_row_ptr, 0.0, mask=offs_d < head_dim)
            if COMPUTE_DGATE:
                tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, 0.0)
            if COMPUTE_DGATE_EXTRA:
                tl.store(dgate_extra_ptr + bh_idx * num_tokens + q_idx, 0.0)
            if COMPUTE_DANGLE_GATE:
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
            message_basis = tl.load(
                message_basis_ptr + (head_idx * message_rank + offs_r[:, None]) * head_dim + offs_d[None, :],
                mask=(offs_r[:, None] < message_rank) & (offs_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            message_basis_go = tl.sum(message_basis * go[None, :], axis=1)

        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dgate_acc = 0.0
        dgate_extra_acc = 0.0
        dangle_gate_acc = 0.0
        if HAS_PAIR_VALUE_MARGINAL:
            pair_value_query_gate = tl.load(pair_value_query_mask_ptr + batch_idx * num_tokens + q_idx).to(tl.float32)
            pair_value_scale_raw = tl.load(pair_value_scale_ptr + head_idx).to(tl.float32)
            pair_value_scale = pair_value_scale_raw * pair_value_query_gate
            pos_i_x = tl.load(pair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 0).to(tl.float32)
            pos_i_y = tl.load(pair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 1).to(tl.float32)
            pos_i_z = tl.load(pair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 2).to(tl.float32)
            freq_x = tl.load(
                pair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 0,
                mask=offs_r < pair_value_n_freqs,
                other=0.0,
            ).to(tl.float32)
            freq_y = tl.load(
                pair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 1,
                mask=offs_r < pair_value_n_freqs,
                other=0.0,
            ).to(tl.float32)
            freq_z = tl.load(
                pair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 2,
                mask=offs_r < pair_value_n_freqs,
                other=0.0,
            ).to(tl.float32)
            go_r = tl.load(
                grad_out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_r,
                mask=(offs_r < pair_value_n_freqs) & ((2 * offs_r) < head_dim),
                other=0.0,
            ).to(tl.float32)
            go_i = tl.load(
                grad_out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_r + 1,
                mask=(offs_r < pair_value_n_freqs) & ((2 * offs_r + 1) < head_dim),
                other=0.0,
            ).to(tl.float32)
            dpos_i_x = 0.0
            dpos_i_y = 0.0
            dpos_i_z = 0.0
            dpair_value_scale_acc = 0.0

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

            if COMPUTE_DU:
                du_running = tl.zeros((BLOCK_J,), dtype=tl.float32)
            if HAS_BIAS:
                u = tl.load(
                    u_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                gate = tl.load(gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
            if COMPUTE_DU_EXTRA:
                du_extra_running = tl.zeros((BLOCK_J,), dtype=tl.float32)
            if HAS_BIAS_EXTRA:
                u_extra = tl.load(
                    u_extra_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                gate_extra = tl.load(gate_extra_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
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
                if COMPUTE_DANGLE_LEFT:
                    dangle_left_running = tl.zeros((BLOCK_J, BLOCK_R), dtype=tl.float32)
                angle_gate = tl.load(angle_gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)
            if HAS_LOW_RANK_MESSAGE:
                message_left_32 = tl.load(
                    message_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * message_rank + offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                    other=0.0,
                ).to(tl.float32)
                if COMPUTE_DMESSAGE_LEFT:
                    dmessage_left_running = tl.zeros((BLOCK_J, BLOCK_R), dtype=tl.float32)
            if HAS_PAIR_VALUE_MARGINAL:
                pos_j_x = tl.load(
                    pair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 0,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                pos_j_y = tl.load(
                    pair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 1,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                pos_j_z = tl.load(
                    pair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 2,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                rel_j_x = pos_j_x - pos_i_x
                rel_j_y = pos_j_y - pos_i_y
                rel_j_z = pos_j_z - pos_i_z
                theta_j = rel_j_x[:, None] * freq_x[None, :] + rel_j_y[:, None] * freq_y[None, :] + rel_j_z[:, None] * freq_z[None, :]
                cos_j = tl.cos(theta_j)
                sin_j = tl.sin(theta_j)
                v1_r = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_r[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                v1_i = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_r[None, :] + 1),
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                rot1_r = v1_r * cos_j - v1_i * sin_j
                rot1_i = v1_r * sin_j + v1_i * cos_j
                rot1_r = tl.where((offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs), rot1_r, 0.0)
                rot1_i = tl.where((offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs), rot1_i, 0.0)
                transport_dot_j_raw = tl.sum(go_r[None, :] * rot1_r + go_i[None, :] * rot1_i, axis=1)
                if COMPUTE_DPAIR_VALUE_GEOMETRY:
                    dtheta_j_running = tl.zeros((BLOCK_J, BLOCK_R), dtype=tl.float32)

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

                    bias_extra_base = tl.zeros((BLOCK_J, BLOCK_K), dtype=tl.float32)
                    if HAS_BIAS_EXTRA:
                        v_bias_extra = tl.load(
                            v_bias_extra_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        w_extra = tl.load(
                            w_extra_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                            mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            other=0.0,
                        ).to(tl.float32)
                        bias_extra_base = u_extra[:, None] + v_bias_extra[None, :] + w_extra
                        scores = scores + gate_extra * bias_extra_base

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
                    if HAS_PAIR_VALUE_MARGINAL:
                        pos_k_x = tl.load(
                            pair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 0,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        pos_k_y = tl.load(
                            pair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 1,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        pos_k_z = tl.load(
                            pair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 2,
                            mask=offs_k < num_tokens,
                            other=0.0,
                        ).to(tl.float32)
                        rel_k_x = pos_k_x - pos_i_x
                        rel_k_y = pos_k_y - pos_i_y
                        rel_k_z = pos_k_z - pos_i_z
                        theta_k = (
                            rel_k_x[:, None] * freq_x[None, :]
                            + rel_k_y[:, None] * freq_y[None, :]
                            + rel_k_z[:, None] * freq_z[None, :]
                        )
                        cos_k = tl.cos(theta_k)
                        sin_k = tl.sin(theta_k)
                        v2_r = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_r[None, :]),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        v2_i = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_r[None, :] + 1),
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        rot2_r = v2_r * cos_k - v2_i * sin_k
                        rot2_i = v2_r * sin_k + v2_i * cos_k
                        rot2_r = tl.where(
                            (offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            rot2_r,
                            0.0,
                        )
                        rot2_i = tl.where(
                            (offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                            rot2_i,
                            0.0,
                        )
                        transport_dot_k_raw = tl.sum(go_r[None, :] * rot2_r + go_i[None, :] * rot2_i, axis=1)

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
                    if HAS_PAIR_VALUE_MARGINAL:
                        value_dot += pair_value_scale * (transport_dot_j_raw[:, None] + transport_dot_k_raw[None, :])
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

                    if HAS_PAIR_VALUE_MARGINAL:
                        row_marginal = tl.sum(probs, axis=1)
                        col_marginal = tl.sum(probs, axis=0)
                        drot1_r = pair_value_scale * row_marginal[:, None] * go_r[None, :]
                        drot1_i = pair_value_scale * row_marginal[:, None] * go_i[None, :]
                        dv1_r = drot1_r * cos_j + drot1_i * sin_j
                        dv1_i = -drot1_r * sin_j + drot1_i * cos_j
                        tl.atomic_add(
                            dv1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_r[None, :]),
                            dv1_r,
                            mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                        )
                        tl.atomic_add(
                            dv1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_r[None, :] + 1),
                            dv1_i,
                            mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                        )

                        drot2_r = pair_value_scale * col_marginal[:, None] * go_r[None, :]
                        drot2_i = pair_value_scale * col_marginal[:, None] * go_i[None, :]
                        dv2_r = drot2_r * cos_k + drot2_i * sin_k
                        dv2_i = -drot2_r * sin_k + drot2_i * cos_k
                        tl.atomic_add(
                            dv2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_r[None, :]),
                            dv2_r,
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                        )
                        tl.atomic_add(
                            dv2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_r[None, :] + 1),
                            dv2_i,
                            mask=(offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                        )

                        if COMPUTE_DPAIR_VALUE_GEOMETRY:
                            dtheta_j_running += (
                                drot1_r * (-v1_r * sin_j - v1_i * cos_j)
                                + drot1_i * (v1_r * cos_j - v1_i * sin_j)
                            )
                            dtheta_k = (
                                drot2_r * (-v2_r * sin_k - v2_i * cos_k)
                                + drot2_i * (v2_r * cos_k - v2_i * sin_k)
                            )
                            dtheta_k = tl.where(
                                (offs_k[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                                dtheta_k,
                                0.0,
                            )
                            dpos_k_x = tl.sum(dtheta_k * freq_x[None, :], axis=1)
                            dpos_k_y = tl.sum(dtheta_k * freq_y[None, :], axis=1)
                            dpos_k_z = tl.sum(dtheta_k * freq_z[None, :], axis=1)
                            if COMPUTE_DPAIR_VALUE_POSITIONS:
                                tl.atomic_add(
                                    dpair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 0,
                                    dpos_k_x,
                                    mask=offs_k < num_tokens,
                                )
                                tl.atomic_add(
                                    dpair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 1,
                                    dpos_k_y,
                                    mask=offs_k < num_tokens,
                                )
                                tl.atomic_add(
                                    dpair_value_positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 2,
                                    dpos_k_z,
                                    mask=offs_k < num_tokens,
                                )
                                dpos_i_x -= tl.sum(dpos_k_x, axis=0)
                                dpos_i_y -= tl.sum(dpos_k_y, axis=0)
                                dpos_i_z -= tl.sum(dpos_k_z, axis=0)
                            if COMPUTE_DPAIR_VALUE_FREQS:
                                tl.atomic_add(
                                    dpair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 0,
                                    tl.sum(dtheta_k * rel_k_x[:, None], axis=0),
                                    mask=offs_r < pair_value_n_freqs,
                                )
                                tl.atomic_add(
                                    dpair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 1,
                                    tl.sum(dtheta_k * rel_k_y[:, None], axis=0),
                                    mask=offs_r < pair_value_n_freqs,
                                )
                                tl.atomic_add(
                                    dpair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 2,
                                    tl.sum(dtheta_k * rel_k_z[:, None], axis=0),
                                    mask=offs_r < pair_value_n_freqs,
                                )
                        if COMPUTE_DPAIR_VALUE_SCALE:
                            dpair_value_scale_acc += (
                                tl.sum(
                                    tl.sum(probs * (transport_dot_j_raw[:, None] + transport_dot_k_raw[None, :]), axis=1),
                                    axis=0,
                                )
                                * pair_value_query_gate
                            )

                    if HAS_BIAS:
                        dsum_k = tl.sum(dscores, axis=1)
                        dsum_j = tl.sum(dscores, axis=0)
                        if COMPUTE_DU:
                            du_running += gate * dsum_k

                        if COMPUTE_DV_BIAS:
                            dv_bias_row_ptr = dv_bias_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k
                            dv_bias_prev = tl.load(dv_bias_row_ptr, mask=offs_k < num_tokens, other=0.0)
                            tl.store(
                                dv_bias_row_ptr,
                                dv_bias_prev + gate * dsum_j,
                                mask=offs_k < num_tokens,
                            )

                        if COMPUTE_DW:
                            tl.atomic_add(
                                dw_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                                gate * dscores,
                                mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            )
                        if COMPUTE_DGATE:
                            dgate_acc += tl.sum(tl.sum(dscores * bias_base, axis=1), axis=0)

                    if HAS_BIAS_EXTRA:
                        dsum_k_extra = tl.sum(dscores, axis=1)
                        dsum_j_extra = tl.sum(dscores, axis=0)
                        if COMPUTE_DU_EXTRA:
                            du_extra_running += gate_extra * dsum_k_extra

                        if COMPUTE_DV_BIAS_EXTRA:
                            dv_bias_extra_row_ptr = dv_bias_extra_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k
                            dv_bias_extra_prev = tl.load(dv_bias_extra_row_ptr, mask=offs_k < num_tokens, other=0.0)
                            tl.store(
                                dv_bias_extra_row_ptr,
                                dv_bias_extra_prev + gate_extra * dsum_j_extra,
                                mask=offs_k < num_tokens,
                            )

                        if COMPUTE_DW_EXTRA:
                            tl.atomic_add(
                                dw_extra_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                                gate_extra * dscores,
                                mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            )
                        if COMPUTE_DGATE_EXTRA:
                            dgate_extra_acc += tl.sum(tl.sum(dscores * bias_extra_base, axis=1), axis=0)

                    if HAS_LOW_RANK_ANGLE:
                        angle_grad_scale = angle_gate * angle_scale
                        if COMPUTE_DANGLE_LEFT:
                            dangle_left_running += angle_grad_scale * tl.dot(
                                dscores,
                                angle_right_32,
                                out_dtype=tl.float32,
                                input_precision="ieee",
                            )
                        if COMPUTE_DANGLE_RIGHT:
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
                        if COMPUTE_DANGLE_GATE:
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
                        if COMPUTE_DMESSAGE_LEFT:
                            dmessage_left_running += message_scale * message_tmp_right * message_basis_go[None, :]
                        if COMPUTE_DMESSAGE_RIGHT:
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
                        if COMPUTE_DMESSAGE_BASIS:
                            message_coeff = tl.sum(message_tmp_left * message_right_32, axis=0)
                            tl.atomic_add(
                                dmessage_basis_ptr
                                + (head_idx * message_rank + offs_r[:, None]) * head_dim
                                + offs_d[None, :],
                                message_scale * message_coeff[:, None] * go[None, :],
                                mask=(offs_r[:, None] < message_rank) & (offs_d[None, :] < head_dim),
                            )

            if COMPUTE_DU:
                tl.store(
                    du_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    du_running,
                    mask=offs_j < num_tokens,
                )
            if COMPUTE_DU_EXTRA:
                tl.store(
                    du_extra_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    du_extra_running,
                    mask=offs_j < num_tokens,
                )
            if COMPUTE_DANGLE_LEFT:
                tl.store(
                    dangle_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * angle_rank + offs_r[None, :]),
                    dangle_left_running,
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < angle_rank),
                )
            if COMPUTE_DMESSAGE_LEFT:
                tl.store(
                    dmessage_left_ptr
                    + (((bh_idx * num_tokens + q_idx) * num_tokens + offs_j[:, None]) * message_rank + offs_r[None, :]),
                    dmessage_left_running,
                    mask=(offs_j[:, None] < num_tokens) & (offs_r[None, :] < message_rank),
                )
            if HAS_PAIR_VALUE_MARGINAL:
                if COMPUTE_DPAIR_VALUE_GEOMETRY:
                    dtheta_j_running = tl.where(
                        (offs_j[:, None] < num_tokens) & (offs_r[None, :] < pair_value_n_freqs),
                        dtheta_j_running,
                        0.0,
                    )
                    dpos_j_x = tl.sum(dtheta_j_running * freq_x[None, :], axis=1)
                    dpos_j_y = tl.sum(dtheta_j_running * freq_y[None, :], axis=1)
                    dpos_j_z = tl.sum(dtheta_j_running * freq_z[None, :], axis=1)
                    if COMPUTE_DPAIR_VALUE_POSITIONS:
                        tl.atomic_add(
                            dpair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 0,
                            dpos_j_x,
                            mask=offs_j < num_tokens,
                        )
                        tl.atomic_add(
                            dpair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 1,
                            dpos_j_y,
                            mask=offs_j < num_tokens,
                        )
                        tl.atomic_add(
                            dpair_value_positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 2,
                            dpos_j_z,
                            mask=offs_j < num_tokens,
                        )
                        dpos_i_x -= tl.sum(dpos_j_x, axis=0)
                        dpos_i_y -= tl.sum(dpos_j_y, axis=0)
                        dpos_i_z -= tl.sum(dpos_j_z, axis=0)
                    if COMPUTE_DPAIR_VALUE_FREQS:
                        tl.atomic_add(
                            dpair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 0,
                            tl.sum(dtheta_j_running * rel_j_x[:, None], axis=0),
                            mask=offs_r < pair_value_n_freqs,
                        )
                        tl.atomic_add(
                            dpair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 1,
                            tl.sum(dtheta_j_running * rel_j_y[:, None], axis=0),
                            mask=offs_r < pair_value_n_freqs,
                        )
                        tl.atomic_add(
                            dpair_value_freqs_ptr + (head_idx * pair_value_n_freqs + offs_r) * 3 + 2,
                            tl.sum(dtheta_j_running * rel_j_z[:, None], axis=0),
                            mask=offs_r < pair_value_n_freqs,
                        )

        tl.store(dq_row_ptr, dq_acc, mask=offs_d < head_dim)
        if COMPUTE_DGATE:
            tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, dgate_acc)
        if COMPUTE_DGATE_EXTRA:
            tl.store(dgate_extra_ptr + bh_idx * num_tokens + q_idx, dgate_extra_acc)
        if COMPUTE_DANGLE_GATE:
            tl.store(dangle_gate_ptr + bh_idx * num_tokens + q_idx, dangle_gate_acc)
        if HAS_PAIR_VALUE_MARGINAL:
            if COMPUTE_DPAIR_VALUE_POSITIONS:
                tl.atomic_add(dpair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 0, dpos_i_x)
                tl.atomic_add(dpair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 1, dpos_i_y)
                tl.atomic_add(dpair_value_positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 2, dpos_i_z)
            if COMPUTE_DPAIR_VALUE_SCALE:
                tl.atomic_add(dpair_value_scale_ptr + head_idx, dpair_value_scale_acc)


    @triton.jit
    def _streaming_backward_native_rope_no_carrier_kernel(
        grad_out_ptr,
        q_content_ptr,
        q_rope_ptr,
        k1_ptr,
        v1_ptr,
        k2_ptr,
        v2_ptr,
        out_ptr,
        lse_ptr,
        positions_ptr,
        mu_ptr,
        nu_ptr,
        rope_logit_scale_ptr,
        rope_value_scale_ptr,
        query_valid_ptr,
        pair_key_valid_ptr,
        pair_valid_ptr,
        position_query_valid_ptr,
        u_ptr,
        v_bias_ptr,
        w_ptr,
        gate_ptr,
        dq_content_ptr,
        dq_rope_ptr,
        dk1_ptr,
        dv1_ptr,
        dk2_ptr,
        dv2_ptr,
        dpositions_ptr,
        dmu_ptr,
        dnu_ptr,
        d_rope_logit_scale_ptr,
        d_rope_value_scale_ptr,
        du_ptr,
        dv_bias_ptr,
        dw_ptr,
        dgate_ptr,
        num_tokens,
        head_dim,
        rope_n_freqs,
        rope_value_n_freqs,
        rope_logit_norm,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_POSITION_QUERY_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_ROPE_VALUE_CARRIER: tl.constexpr,
        COMPUTE_DU: tl.constexpr,
        COMPUTE_DV_BIAS: tl.constexpr,
        COMPUTE_DW: tl.constexpr,
        COMPUTE_DGATE: tl.constexpr,
        COMPUTE_DVALUE_SCALE: tl.constexpr,
        USE_FP32_INPUT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid % num_tokens
        bh_idx = pid // num_tokens
        batch_idx = bh_idx // num_heads
        head_idx = bh_idx - batch_idx * num_heads

        offs_d = tl.arange(0, BLOCK_D)
        offs_f = tl.arange(0, BLOCK_F)
        q_is_valid = tl.load(query_valid_ptr + batch_idx * num_tokens + q_idx) > 0
        dq_content_row_ptr = dq_content_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        dq_rope_row_ptr = dq_rope_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        if not q_is_valid:
            tl.store(dq_content_row_ptr, 0.0, mask=offs_d < head_dim)
            tl.store(dq_rope_row_ptr, 0.0, mask=offs_d < head_dim)
            if COMPUTE_DGATE:
                tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, 0.0)
            return

        position_query_gate = 1.0
        if HAS_POSITION_QUERY_VALID:
            position_query_gate = tl.load(position_query_valid_ptr + batch_idx * num_tokens + q_idx).to(tl.float32)

        q_content_raw = tl.load(
            q_content_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        )
        if USE_FP32_INPUT:
            q_content_score = q_content_raw.to(tl.float32)
        else:
            q_content_score = q_content_raw
        q_content_32 = q_content_raw.to(tl.float32)

        q_rope_r = tl.load(
            q_rope_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_f,
            mask=(offs_f < rope_n_freqs) & ((2 * offs_f) < head_dim),
            other=0.0,
        ).to(tl.float32)
        q_rope_i = tl.load(
            q_rope_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_f + 1,
            mask=(offs_f < rope_n_freqs) & ((2 * offs_f + 1) < head_dim),
            other=0.0,
        ).to(tl.float32)

        mu_x = tl.load(
            mu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        mu_y = tl.load(
            mu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        mu_z = tl.load(
            mu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        nu_x = tl.load(
            nu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        nu_y = tl.load(
            nu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        nu_z = tl.load(
            nu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
            mask=offs_f < rope_n_freqs,
            other=0.0,
        ).to(tl.float32)
        a_x = 0.5 * (mu_x + nu_x)
        a_y = 0.5 * (mu_y + nu_y)
        a_z = 0.5 * (mu_z + nu_z)
        b_x = 0.5 * (mu_x - nu_x)
        b_y = 0.5 * (mu_y - nu_y)
        b_z = 0.5 * (mu_z - nu_z)
        carrier_dim = 2 * rope_value_n_freqs

        pos_i_x = tl.load(positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 0).to(tl.float32)
        pos_i_y = tl.load(positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 1).to(tl.float32)
        pos_i_z = tl.load(positions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 2).to(tl.float32)
        rope_logit_scale = (
            tl.load(rope_logit_scale_ptr + head_idx).to(tl.float32)
            * rope_logit_norm
            * position_query_gate
        )
        rope_value_scale = 0.0
        if HAS_ROPE_VALUE_CARRIER:
            rope_value_scale = tl.load(rope_value_scale_ptr + head_idx).to(tl.float32) * position_query_gate

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

        dq_content_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dq_rope_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dpos_i_x = 0.0
        dpos_i_y = 0.0
        dpos_i_z = 0.0
        dscale_acc = 0.0
        dvalue_scale_acc = 0.0
        dgate_acc = 0.0
        go_r = tl.load(
            grad_out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_f,
            mask=(offs_f < rope_value_n_freqs) & ((2 * offs_f) < head_dim),
            other=0.0,
        ).to(tl.float32)
        go_i = tl.load(
            grad_out_ptr + (bh_idx * num_tokens + q_idx) * head_dim + 2 * offs_f + 1,
            mask=(offs_f < rope_value_n_freqs) & ((2 * offs_f + 1) < head_dim),
            other=0.0,
        ).to(tl.float32)

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
            if HAS_ROPE_VALUE_CARRIER:
                v1_base_32 = tl.where(offs_d[None, :] < carrier_dim, 0.0, v1_32)
            else:
                v1_base_32 = v1_32
            if USE_FP32_INPUT:
                k1_score = k1_raw.to(tl.float32)
            else:
                k1_score = k1_raw

            pos_j_x = tl.load(
                positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 0,
                mask=offs_j < num_tokens,
                other=0.0,
            ).to(tl.float32)
            pos_j_y = tl.load(
                positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 1,
                mask=offs_j < num_tokens,
                other=0.0,
            ).to(tl.float32)
            pos_j_z = tl.load(
                positions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 2,
                mask=offs_j < num_tokens,
                other=0.0,
            ).to(tl.float32)
            rel_j_x = pos_j_x - pos_i_x
            rel_j_y = pos_j_y - pos_i_y
            rel_j_z = pos_j_z - pos_i_z
            phi = (
                rel_j_x[:, None] * a_x[None, :]
                + rel_j_y[:, None] * a_y[None, :]
                + rel_j_z[:, None] * a_z[None, :]
            )
            cos_phi = tl.cos(phi)
            sin_phi = tl.sin(phi)
            left_even = q_rope_r[None, :] * cos_phi + q_rope_i[None, :] * sin_phi
            left_odd = -q_rope_r[None, :] * sin_phi + q_rope_i[None, :] * cos_phi
            left_even = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs), left_even, 0.0)
            left_odd = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs), left_odd, 0.0)

            if HAS_ROPE_VALUE_CARRIER:
                v1_r = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_f[None, :]),
                    mask=(offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                v1_i = tl.load(
                    v1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_f[None, :] + 1),
                    mask=(offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                    other=0.0,
                ).to(tl.float32)
                a_r = v1_r * cos_phi - v1_i * sin_phi
                a_i = v1_r * sin_phi + v1_i * cos_phi
                a_r = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs), a_r, 0.0)
                a_i = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs), a_i, 0.0)

            if COMPUTE_DU:
                du_running = tl.zeros((BLOCK_J,), dtype=tl.float32)
            if HAS_BIAS:
                u = tl.load(
                    u_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    mask=offs_j < num_tokens,
                    other=0.0,
                ).to(tl.float32)
                gate = tl.load(gate_ptr + bh_idx * num_tokens + q_idx).to(tl.float32)

            dleft_even_running = tl.zeros((BLOCK_J, BLOCK_F), dtype=tl.float32)
            dleft_odd_running = tl.zeros((BLOCK_J, BLOCK_F), dtype=tl.float32)
            if HAS_ROPE_VALUE_CARRIER:
                dphi_value_running = tl.zeros((BLOCK_J, BLOCK_F), dtype=tl.float32)

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
                    if HAS_ROPE_VALUE_CARRIER:
                        v2_base_32 = tl.where(offs_d[None, :] < carrier_dim, 0.0, v2_32)
                    else:
                        v2_base_32 = v2_32
                    if USE_FP32_INPUT:
                        k2_score = k2_raw.to(tl.float32)
                        qk2 = k2_score * q_content_score[None, :]
                    else:
                        k2_score = k2_raw
                        qk2 = (k2_score * q_content_score[None, :]).to(k2_score.dtype)
                    scores = tl.dot(
                        k1_score,
                        tl.trans(qk2),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    )

                    pos_k_x = tl.load(
                        positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 0,
                        mask=offs_k < num_tokens,
                        other=0.0,
                    ).to(tl.float32)
                    pos_k_y = tl.load(
                        positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 1,
                        mask=offs_k < num_tokens,
                        other=0.0,
                    ).to(tl.float32)
                    pos_k_z = tl.load(
                        positions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 2,
                        mask=offs_k < num_tokens,
                        other=0.0,
                    ).to(tl.float32)
                    rel_k_x = pos_k_x - pos_i_x
                    rel_k_y = pos_k_y - pos_i_y
                    rel_k_z = pos_k_z - pos_i_z
                    psi = (
                        rel_k_x[:, None] * b_x[None, :]
                        + rel_k_y[:, None] * b_y[None, :]
                        + rel_k_z[:, None] * b_z[None, :]
                    )
                    cos_psi = tl.cos(psi)
                    sin_psi = tl.sin(psi)
                    right_even = tl.where(
                        (offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs),
                        cos_psi,
                        0.0,
                    )
                    right_odd = tl.where(
                        (offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs),
                        sin_psi,
                        0.0,
                    )
                    if HAS_ROPE_VALUE_CARRIER:
                        v2_r = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_f[None, :]),
                            mask=(offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        v2_i = tl.load(
                            v2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_f[None, :] + 1),
                            mask=(offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                            other=0.0,
                        ).to(tl.float32)
                        b_r = v2_r * cos_psi - v2_i * sin_psi
                        b_i = v2_r * sin_psi + v2_i * cos_psi
                        b_r = tl.where((offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs), b_r, 0.0)
                        b_i = tl.where((offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs), b_i, 0.0)
                    rope_scores_raw = tl.dot(
                        left_even,
                        tl.trans(right_even),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    ) + tl.dot(
                        left_odd,
                        tl.trans(right_odd),
                        out_dtype=tl.float32,
                        input_precision=INPUT_PRECISION,
                    )
                    scores = scores + rope_logit_scale * rope_scores_raw

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

                    scores = tl.where(valid, scores, -float("inf"))
                    probs = tl.where(valid, tl.exp(scores - lse), 0.0)
                    value_dot = tl.dot(
                        v1_base_32 * go[None, :],
                        tl.trans(v2_base_32),
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    if HAS_ROPE_VALUE_CARRIER:
                        carrier_left_r = go_r[None, :] * a_r + go_i[None, :] * a_i
                        carrier_left_i = go_i[None, :] * a_r - go_r[None, :] * a_i
                        carrier_value_dot_raw = tl.dot(
                            carrier_left_r,
                            tl.trans(b_r),
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        ) + tl.dot(
                            carrier_left_i,
                            tl.trans(b_i),
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        value_dot = value_dot + rope_value_scale * carrier_value_dot_raw
                    dscores = probs * (value_dot - go_dot_out)

                    tmp_k2 = tl.dot(dscores, k2_32, out_dtype=tl.float32, input_precision="ieee")
                    tmp_k1 = tl.dot(tl.trans(dscores), k1_32, out_dtype=tl.float32, input_precision="ieee")
                    dq_content_acc += tl.sum(tmp_k2 * k1_32, axis=0)
                    dk1_tile = tmp_k2 * q_content_32[None, :]
                    dk2_tile = tmp_k1 * q_content_32[None, :]
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

                    tmp_v2 = tl.dot(probs, v2_base_32, out_dtype=tl.float32, input_precision="ieee")
                    tmp_v1 = tl.dot(tl.trans(probs), v1_base_32, out_dtype=tl.float32, input_precision="ieee")
                    tl.atomic_add(
                        dv1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + offs_d[None, :]),
                        tmp_v2 * go[None, :],
                        mask=(offs_j[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                    )
                    tl.atomic_add(
                        dv2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + offs_d[None, :]),
                        tmp_v1 * go[None, :],
                        mask=(offs_k[:, None] < num_tokens) & (offs_d[None, :] < head_dim),
                    )

                    if HAS_ROPE_VALUE_CARRIER:
                        dvalue_scale_acc += (
                            tl.sum(tl.sum(probs * carrier_value_dot_raw, axis=1), axis=0)
                            * position_query_gate
                        )

                        bcomb_r = go_r[None, :] * b_r + go_i[None, :] * b_i
                        bcomb_i = -go_r[None, :] * b_i + go_i[None, :] * b_r
                        da_r_value = rope_value_scale * tl.dot(
                            probs,
                            bcomb_r,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        da_i_value = rope_value_scale * tl.dot(
                            probs,
                            bcomb_i,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        acomp_r = go_r[None, :] * a_r + go_i[None, :] * a_i
                        acomp_i = -go_r[None, :] * a_i + go_i[None, :] * a_r
                        db_r_value = rope_value_scale * tl.dot(
                            tl.trans(probs),
                            acomp_r,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )
                        db_i_value = rope_value_scale * tl.dot(
                            tl.trans(probs),
                            acomp_i,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )

                        dv1_r = da_r_value * cos_phi + da_i_value * sin_phi
                        dv1_i = -da_r_value * sin_phi + da_i_value * cos_phi
                        dv2_r = db_r_value * cos_psi + db_i_value * sin_psi
                        dv2_i = -db_r_value * sin_psi + db_i_value * cos_psi
                        tl.atomic_add(
                            dv1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_f[None, :]),
                            dv1_r,
                            mask=(offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                        )
                        tl.atomic_add(
                            dv1_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * head_dim + 2 * offs_f[None, :] + 1),
                            dv1_i,
                            mask=(offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                        )
                        tl.atomic_add(
                            dv2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_f[None, :]),
                            dv2_r,
                            mask=(offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                        )
                        tl.atomic_add(
                            dv2_ptr + ((bh_idx * num_tokens + offs_k[:, None]) * head_dim + 2 * offs_f[None, :] + 1),
                            dv2_i,
                            mask=(offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_value_n_freqs),
                        )

                        dphi_value_running += -da_r_value * a_i + da_i_value * a_r
                        dpsi_value = -db_r_value * b_i + db_i_value * b_r
                    else:
                        dpsi_value = tl.zeros((BLOCK_K, BLOCK_F), dtype=tl.float32)

                    dleft_even = rope_logit_scale * tl.dot(
                        dscores,
                        right_even,
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    dleft_odd = rope_logit_scale * tl.dot(
                        dscores,
                        right_odd,
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    dright_even = rope_logit_scale * tl.dot(
                        tl.trans(dscores),
                        left_even,
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    dright_odd = rope_logit_scale * tl.dot(
                        tl.trans(dscores),
                        left_odd,
                        out_dtype=tl.float32,
                        input_precision="ieee",
                    )
                    dleft_even_running += dleft_even
                    dleft_odd_running += dleft_odd
                    dpsi = -dright_even * sin_psi + dright_odd * cos_psi
                    if HAS_ROPE_VALUE_CARRIER:
                        dpsi += dpsi_value
                    dpsi = tl.where((offs_k[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs), dpsi, 0.0)

                    dpos_k_x = tl.sum(dpsi * b_x[None, :], axis=1)
                    dpos_k_y = tl.sum(dpsi * b_y[None, :], axis=1)
                    dpos_k_z = tl.sum(dpsi * b_z[None, :], axis=1)
                    tl.atomic_add(
                        dpositions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 0,
                        dpos_k_x,
                        mask=offs_k < num_tokens,
                    )
                    tl.atomic_add(
                        dpositions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 1,
                        dpos_k_y,
                        mask=offs_k < num_tokens,
                    )
                    tl.atomic_add(
                        dpositions_ptr + (batch_idx * num_tokens + offs_k) * 3 + 2,
                        dpos_k_z,
                        mask=offs_k < num_tokens,
                    )
                    dpos_i_x -= tl.sum(dpos_k_x, axis=0)
                    dpos_i_y -= tl.sum(dpos_k_y, axis=0)
                    dpos_i_z -= tl.sum(dpos_k_z, axis=0)

                    db_x = tl.sum(dpsi * rel_k_x[:, None], axis=0)
                    db_y = tl.sum(dpsi * rel_k_y[:, None], axis=0)
                    db_z = tl.sum(dpsi * rel_k_z[:, None], axis=0)
                    tl.atomic_add(
                        dmu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
                        0.5 * db_x,
                        mask=offs_f < rope_n_freqs,
                    )
                    tl.atomic_add(
                        dmu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
                        0.5 * db_y,
                        mask=offs_f < rope_n_freqs,
                    )
                    tl.atomic_add(
                        dmu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
                        0.5 * db_z,
                        mask=offs_f < rope_n_freqs,
                    )
                    tl.atomic_add(
                        dnu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
                        -0.5 * db_x,
                        mask=offs_f < rope_n_freqs,
                    )
                    tl.atomic_add(
                        dnu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
                        -0.5 * db_y,
                        mask=offs_f < rope_n_freqs,
                    )
                    tl.atomic_add(
                        dnu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
                        -0.5 * db_z,
                        mask=offs_f < rope_n_freqs,
                    )

                    dscale_acc += tl.sum(tl.sum(dscores * rope_scores_raw, axis=1), axis=0) * rope_logit_norm * position_query_gate

                    if HAS_BIAS:
                        dsum_k = tl.sum(dscores, axis=1)
                        dsum_j = tl.sum(dscores, axis=0)
                        if COMPUTE_DU:
                            du_running += gate * dsum_k
                        if COMPUTE_DV_BIAS:
                            dv_bias_row_ptr = dv_bias_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_k
                            dv_bias_prev = tl.load(dv_bias_row_ptr, mask=offs_k < num_tokens, other=0.0)
                            tl.store(
                                dv_bias_row_ptr,
                                dv_bias_prev + gate * dsum_j,
                                mask=offs_k < num_tokens,
                            )
                        if COMPUTE_DW:
                            tl.atomic_add(
                                dw_ptr + ((bh_idx * num_tokens + offs_j[:, None]) * num_tokens + offs_k[None, :]),
                                gate * dscores,
                                mask=(offs_j[:, None] < num_tokens) & (offs_k[None, :] < num_tokens),
                            )
                        if COMPUTE_DGATE:
                            dgate_acc += tl.sum(tl.sum(dscores * bias_base, axis=1), axis=0)

            dphi = (
                dleft_even_running * (-q_rope_r[None, :] * sin_phi + q_rope_i[None, :] * cos_phi)
                + dleft_odd_running * (-q_rope_r[None, :] * cos_phi - q_rope_i[None, :] * sin_phi)
            )
            if HAS_ROPE_VALUE_CARRIER:
                dphi += dphi_value_running
            dphi = tl.where((offs_j[:, None] < num_tokens) & (offs_f[None, :] < rope_n_freqs), dphi, 0.0)
            dq_r = tl.sum(dleft_even_running * cos_phi - dleft_odd_running * sin_phi, axis=0)
            dq_i = tl.sum(dleft_even_running * sin_phi + dleft_odd_running * cos_phi, axis=0)
            dim_to_freq_real = offs_d[:, None] == 2 * offs_f[None, :]
            dim_to_freq_imag = offs_d[:, None] == (2 * offs_f[None, :] + 1)
            dq_rope_matrix = tl.where(dim_to_freq_real, dq_r[None, :], 0.0) + tl.where(
                dim_to_freq_imag,
                dq_i[None, :],
                0.0,
            )
            dq_rope_matrix = tl.where(offs_f[None, :] < rope_n_freqs, dq_rope_matrix, 0.0)
            dq_rope_acc += tl.sum(dq_rope_matrix, axis=1)

            dpos_j_x = tl.sum(dphi * a_x[None, :], axis=1)
            dpos_j_y = tl.sum(dphi * a_y[None, :], axis=1)
            dpos_j_z = tl.sum(dphi * a_z[None, :], axis=1)
            tl.atomic_add(
                dpositions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 0,
                dpos_j_x,
                mask=offs_j < num_tokens,
            )
            tl.atomic_add(
                dpositions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 1,
                dpos_j_y,
                mask=offs_j < num_tokens,
            )
            tl.atomic_add(
                dpositions_ptr + (batch_idx * num_tokens + offs_j) * 3 + 2,
                dpos_j_z,
                mask=offs_j < num_tokens,
            )
            dpos_i_x -= tl.sum(dpos_j_x, axis=0)
            dpos_i_y -= tl.sum(dpos_j_y, axis=0)
            dpos_i_z -= tl.sum(dpos_j_z, axis=0)

            da_x = tl.sum(dphi * rel_j_x[:, None], axis=0)
            da_y = tl.sum(dphi * rel_j_y[:, None], axis=0)
            da_z = tl.sum(dphi * rel_j_z[:, None], axis=0)
            tl.atomic_add(
                dmu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
                0.5 * da_x,
                mask=offs_f < rope_n_freqs,
            )
            tl.atomic_add(
                dmu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
                0.5 * da_y,
                mask=offs_f < rope_n_freqs,
            )
            tl.atomic_add(
                dmu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
                0.5 * da_z,
                mask=offs_f < rope_n_freqs,
            )
            tl.atomic_add(
                dnu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 0,
                0.5 * da_x,
                mask=offs_f < rope_n_freqs,
            )
            tl.atomic_add(
                dnu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 1,
                0.5 * da_y,
                mask=offs_f < rope_n_freqs,
            )
            tl.atomic_add(
                dnu_ptr + (head_idx * rope_n_freqs + offs_f) * 3 + 2,
                0.5 * da_z,
                mask=offs_f < rope_n_freqs,
            )

            if COMPUTE_DU:
                tl.store(
                    du_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    du_running,
                    mask=offs_j < num_tokens,
                )

        tl.store(dq_content_row_ptr, dq_content_acc, mask=offs_d < head_dim)
        tl.store(dq_rope_row_ptr, dq_rope_acc, mask=offs_d < head_dim)
        tl.atomic_add(dpositions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 0, dpos_i_x)
        tl.atomic_add(dpositions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 1, dpos_i_y)
        tl.atomic_add(dpositions_ptr + (batch_idx * num_tokens + q_idx) * 3 + 2, dpos_i_z)
        tl.atomic_add(d_rope_logit_scale_ptr + head_idx, dscale_acc)
        if COMPUTE_DVALUE_SCALE:
            tl.atomic_add(d_rope_value_scale_ptr + head_idx, dvalue_scale_acc)
        if COMPUTE_DGATE:
            tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, dgate_acc)


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
    u_extra: torch.Tensor | None,
    v_bias_extra: torch.Tensor | None,
    w_extra: torch.Tensor | None,
    gate_extra: torch.Tensor | None,
    angle_left: torch.Tensor | None,
    angle_right: torch.Tensor | None,
    angle_gate: torch.Tensor | None,
    message_left: torch.Tensor | None,
    message_right: torch.Tensor | None,
    message_basis: torch.Tensor | None,
    pair_value_positions: torch.Tensor | None,
    pair_value_freqs: torch.Tensor | None,
    pair_value_scale: torch.Tensor | None,
    pair_value_query_mask: torch.Tensor | None,
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
    has_bias_extra = u_extra is not None and v_bias_extra is not None and w_extra is not None and gate_extra is not None
    if has_bias_extra:
        u_extra_bh = _reshape_batch_heads(u_extra)
        v_bias_extra_bh = _reshape_batch_heads(v_bias_extra)
        w_extra_bh = _reshape_batch_heads(w_extra)
        gate_extra_bh = gate_extra.contiguous().view(batch_size * num_heads, num_tokens)
    else:
        u_extra_bh = v_bias_extra_bh = w_extra_bh = gate_extra_bh = empty

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

    has_pair_value_marginal = pair_value_positions is not None and pair_value_freqs is not None and pair_value_scale is not None
    if has_pair_value_marginal:
        if pair_value_positions.shape != (batch_size, num_tokens, 3):
            raise ValueError(
                f"pair_value_positions must have shape {(batch_size, num_tokens, 3)}, got {tuple(pair_value_positions.shape)}"
            )
        if pair_value_freqs.ndim != 3 or pair_value_freqs.shape[0] != num_heads or pair_value_freqs.shape[2] != 3:
            raise ValueError(f"pair_value_freqs must have shape {(num_heads, 'F', 3)}, got {tuple(pair_value_freqs.shape)}")
        pair_value_n_freqs = int(pair_value_freqs.shape[1])
        if pair_value_n_freqs <= 0:
            raise ValueError("pair_value_freqs must contain at least one frequency")
        if 2 * pair_value_n_freqs > head_dim:
            raise ValueError(
                f"pair-RoPE marginal value transport needs 2 * n_freqs <= head_dim, got {pair_value_n_freqs} and {head_dim}"
            )
        if pair_value_scale.shape != (num_heads,):
            raise ValueError(f"pair_value_scale must have shape {(num_heads,)}, got {tuple(pair_value_scale.shape)}")
        if pair_value_query_mask is None:
            pair_value_query_mask_tensor = torch.ones((batch_size, num_tokens), device=q.device, dtype=torch.bool)
        else:
            if pair_value_query_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"pair_value_query_mask must have shape {(batch_size, num_tokens)}, got {tuple(pair_value_query_mask.shape)}"
                )
            pair_value_query_mask_tensor = pair_value_query_mask.contiguous()
        pair_value_positions_tensor = pair_value_positions.contiguous().to(device=q.device, dtype=torch.float32)
        pair_value_freqs_tensor = pair_value_freqs.contiguous().to(device=q.device, dtype=torch.float32)
        pair_value_scale_tensor = pair_value_scale.contiguous().to(device=q.device, dtype=torch.float32)
    else:
        pair_value_n_freqs = 1
        pair_value_positions_tensor = pair_value_freqs_tensor = pair_value_scale_tensor = empty
        pair_value_query_mask_tensor = torch.empty(0, device=q.device, dtype=torch.bool)

    block_r = _block_r_for_rank(
        max(
            angle_rank if has_low_rank_angle else 1,
            message_rank if has_low_rank_message else 1,
            pair_value_n_freqs if has_pair_value_marginal else 1,
        )
    )

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
        u_extra_bh,
        v_bias_extra_bh,
        w_extra_bh,
        gate_extra_bh,
        angle_left_bh,
        angle_right_bh,
        angle_gate_bh,
        message_left_bh,
        message_right_bh,
        message_basis_h,
        pair_value_positions_tensor,
        pair_value_freqs_tensor,
        pair_value_scale_tensor,
        pair_value_query_mask_tensor,
        out,
        lse,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        pair_value_n_freqs,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_BIAS=has_bias,
        HAS_BIAS_EXTRA=has_bias_extra,
        HAS_LOW_RANK_ANGLE=has_low_rank_angle,
        HAS_LOW_RANK_MESSAGE=has_low_rank_message,
        HAS_PAIR_VALUE_MARGINAL=has_pair_value_marginal,
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


def triton_simplicial_attention_native_rope_forward(
    q_content: torch.Tensor,
    q_rope: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    pair_key_valid: torch.Tensor,
    pair_valid: torch.Tensor | None,
    position_query_valid: torch.Tensor | None,
    u: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    w: torch.Tensor | None,
    gate: torch.Tensor | None,
    positions: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    rope_logit_scale: torch.Tensor,
    rope_value_scale: torch.Tensor | None,
    rope_value_n_freqs: int,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by dispatch logic.
        raise RuntimeError("triton is not installed")
    if q_content.device.type != "cuda":
        raise RuntimeError("triton_simplicial_attention_native_rope_forward requires CUDA tensors")
    if q_content.shape != q_rope.shape or q_content.shape != k1.shape or q_content.shape != v1.shape:
        raise ValueError("q_content, q_rope, k1, and v1 must have the same shape")
    if q_content.shape != k2.shape or q_content.shape != v2.shape:
        raise ValueError("k2 and v2 must have the same shape as q_content")

    batch_size, num_heads, num_tokens, head_dim = q_content.shape
    rope_n_freqs = int(mu.shape[1])
    if mu.shape != nu.shape or mu.shape != (num_heads, rope_n_freqs, 3):
        raise ValueError(
            f"mu and nu must have shape {(num_heads, rope_n_freqs, 3)}, got {tuple(mu.shape)} and {tuple(nu.shape)}"
        )
    if positions.shape != (batch_size, num_tokens, 3):
        raise ValueError(f"positions must have shape {(batch_size, num_tokens, 3)}, got {tuple(positions.shape)}")
    if 2 * rope_n_freqs > head_dim:
        raise ValueError(f"closed RoPE needs 2 * n_freqs <= head_dim, got {rope_n_freqs} and {head_dim}")
    if rope_value_n_freqs < 0:
        raise ValueError("rope_value_n_freqs must be non-negative")
    if rope_value_n_freqs > rope_n_freqs:
        raise ValueError("rope_value_n_freqs must be <= rope_n_freqs")
    if rope_value_n_freqs > 0 and 2 * rope_value_n_freqs > head_dim:
        raise ValueError(
            f"closed RoPE value carrier needs 2 * value_n_freqs <= head_dim, got {rope_value_n_freqs} and {head_dim}"
        )

    block_d = triton_block_d_for_head_dim(head_dim)
    block_f = _block_r_for_rank(max(rope_n_freqs, rope_value_n_freqs, 1))
    use_fp32_input, input_precision = _precision_kernel_config(precision)

    q_content_bh = _reshape_batch_heads(q_content)
    q_rope_bh = _reshape_batch_heads(q_rope)
    k1_bh = _reshape_batch_heads(k1)
    v1_bh = _reshape_batch_heads(v1)
    k2_bh = _reshape_batch_heads(k2)
    v2_bh = _reshape_batch_heads(v2)

    has_pair_valid = pair_valid is not None
    pair_valid_tensor = pair_valid.contiguous() if has_pair_valid else torch.empty(0, device=q_content.device, dtype=torch.bool)
    has_position_query_valid = position_query_valid is not None
    position_query_valid_tensor = (
        position_query_valid.contiguous()
        if has_position_query_valid
        else torch.empty(0, device=q_content.device, dtype=torch.bool)
    )
    has_bias = u is not None and v_bias is not None and w is not None and gate is not None
    empty = q_content_bh.new_empty(0, dtype=torch.float32)
    if has_bias:
        u_bh = _reshape_batch_heads(u)
        v_bias_bh = _reshape_batch_heads(v_bias)
        w_bh = _reshape_batch_heads(w)
        gate_bh = gate.contiguous().view(batch_size * num_heads, num_tokens)
    else:
        u_bh = v_bias_bh = w_bh = gate_bh = empty

    rope_logit_scale_h = rope_logit_scale.contiguous().view(num_heads).to(device=q_content.device, dtype=torch.float32)
    if rope_value_n_freqs > 0:
        if rope_value_scale is None:
            raise ValueError("rope_value_scale must be provided when rope_value_n_freqs > 0")
        rope_value_scale_h = rope_value_scale.contiguous().view(num_heads).to(device=q_content.device, dtype=torch.float32)
    else:
        rope_value_scale_h = empty

    batch_heads = batch_size * num_heads
    out = torch.empty((batch_heads, num_tokens, head_dim), device=q_content.device, dtype=torch.float32)
    lse = torch.empty((batch_heads, num_tokens), device=q_content.device, dtype=torch.float32)
    num_tiles_j = triton.cdiv(num_tokens, _BLOCK_J)
    num_tiles_k = triton.cdiv(num_tokens, _BLOCK_K)
    grid = (batch_heads * num_tokens,)
    _streaming_forward_native_rope_kernel[grid](
        q_content_bh,
        q_rope_bh,
        k1_bh,
        v1_bh,
        k2_bh,
        v2_bh,
        query_valid.contiguous(),
        pair_key_valid.contiguous(),
        pair_valid_tensor,
        position_query_valid_tensor,
        u_bh,
        v_bias_bh,
        w_bh,
        gate_bh,
        positions.contiguous().to(device=q_content.device, dtype=torch.float32),
        mu.contiguous().to(device=q_content.device, dtype=torch.float32),
        nu.contiguous().to(device=q_content.device, dtype=torch.float32),
        rope_logit_scale_h,
        rope_value_scale_h,
        out,
        lse,
        num_tokens,
        head_dim,
        rope_n_freqs,
        rope_value_n_freqs,
        rope_n_freqs**-0.5,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_POSITION_QUERY_VALID=has_position_query_valid,
        HAS_BIAS=has_bias,
        HAS_ROPE_VALUE_CARRIER=rope_value_n_freqs > 0,
        USE_FP32_INPUT=use_fp32_input,
        INPUT_PRECISION=input_precision,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=block_d,
        BLOCK_F=block_f,
        num_warps=_num_warps_for_block_d(block_d),
    )
    out_for_backward = out.view(batch_size, num_heads, num_tokens, head_dim)
    out = out_for_backward.to(q_content.dtype)
    lse = lse.view(batch_size, num_heads, num_tokens)
    return out, lse, out_for_backward


def triton_simplicial_attention_native_rope_backward(
    grad_out: torch.Tensor,
    q_content: torch.Tensor,
    q_rope: torch.Tensor,
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
    position_query_valid: torch.Tensor | None,
    u: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    w: torch.Tensor | None,
    gate: torch.Tensor | None,
    positions: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    rope_logit_scale: torch.Tensor,
    rope_value_scale: torch.Tensor | None,
    rope_value_n_freqs: int,
    precision: str,
    need_du: bool = True,
    need_dv_bias: bool = True,
    need_dw: bool = True,
    need_dgate: bool = True,
    need_dvalue_scale: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
]:
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by dispatch logic.
        raise RuntimeError("triton is not installed")
    if q_content.device.type != "cuda":
        raise RuntimeError("triton_simplicial_attention_native_rope_backward requires CUDA tensors")
    if q_content.shape != q_rope.shape or q_content.shape != k1.shape or q_content.shape != v1.shape:
        raise ValueError("q_content, q_rope, k1, and v1 must have the same shape")
    if q_content.shape != k2.shape or q_content.shape != v2.shape or q_content.shape != grad_out.shape:
        raise ValueError("k2, v2, and grad_out must have the same shape as q_content")

    batch_size, num_heads, num_tokens, head_dim = q_content.shape
    rope_n_freqs = int(mu.shape[1])
    if mu.shape != nu.shape or mu.shape != (num_heads, rope_n_freqs, 3):
        raise ValueError(
            f"mu and nu must have shape {(num_heads, rope_n_freqs, 3)}, got {tuple(mu.shape)} and {tuple(nu.shape)}"
        )
    if positions.shape != (batch_size, num_tokens, 3):
        raise ValueError(f"positions must have shape {(batch_size, num_tokens, 3)}, got {tuple(positions.shape)}")
    if 2 * rope_n_freqs > head_dim:
        raise ValueError(f"closed RoPE needs 2 * n_freqs <= head_dim, got {rope_n_freqs} and {head_dim}")
    if rope_value_n_freqs < 0:
        raise ValueError("rope_value_n_freqs must be non-negative")
    if rope_value_n_freqs > rope_n_freqs:
        raise ValueError("rope_value_n_freqs must be <= rope_n_freqs")
    if rope_value_n_freqs > 0 and 2 * rope_value_n_freqs > head_dim:
        raise ValueError(
            f"closed RoPE value carrier needs 2 * value_n_freqs <= head_dim, got {rope_value_n_freqs} and {head_dim}"
        )
    if rope_value_n_freqs > 0 and rope_value_scale is None:
        raise ValueError("rope_value_scale must be provided when rope_value_n_freqs > 0")

    block_d = triton_block_d_for_head_dim(head_dim)
    block_f = _block_r_for_rank(max(rope_n_freqs, rope_value_n_freqs, 1))
    use_fp32_input, input_precision = _precision_kernel_config(precision)

    grad_out_bh = _reshape_batch_heads(grad_out)
    q_content_bh = _reshape_batch_heads(q_content)
    q_rope_bh = _reshape_batch_heads(q_rope)
    k1_bh = _reshape_batch_heads(k1)
    v1_bh = _reshape_batch_heads(v1)
    k2_bh = _reshape_batch_heads(k2)
    v2_bh = _reshape_batch_heads(v2)
    out_bh = _reshape_batch_heads(out)
    lse_bh = lse.contiguous().view(batch_size * num_heads, num_tokens)

    dq_content = torch.zeros_like(q_content_bh, dtype=torch.float32)
    dq_rope = torch.zeros_like(q_rope_bh, dtype=torch.float32)
    dk1 = torch.zeros_like(k1_bh, dtype=torch.float32)
    dv1 = torch.zeros_like(v1_bh, dtype=torch.float32)
    dk2 = torch.zeros_like(k2_bh, dtype=torch.float32)
    dv2 = torch.zeros_like(v2_bh, dtype=torch.float32)
    dpositions = torch.zeros((batch_size, num_tokens, 3), device=q_content.device, dtype=torch.float32)
    dmu = torch.zeros((num_heads, rope_n_freqs, 3), device=q_content.device, dtype=torch.float32)
    dnu = torch.zeros_like(dmu)
    d_rope_logit_scale = torch.zeros((num_heads,), device=q_content.device, dtype=torch.float32)
    has_value_carrier = rope_value_n_freqs > 0
    compute_dvalue_scale = bool(has_value_carrier and need_dvalue_scale)
    if has_value_carrier:
        assert rope_value_scale is not None
        rope_value_scale_h = rope_value_scale.contiguous().view(num_heads).to(device=q_content.device, dtype=torch.float32)
        d_rope_value_scale = torch.zeros((num_heads,), device=q_content.device, dtype=torch.float32) if compute_dvalue_scale else None
    else:
        rope_value_scale_h = q_content_bh.new_empty(0, dtype=torch.float32)
        d_rope_value_scale = None

    has_pair_valid = pair_valid is not None
    pair_valid_tensor = (
        pair_valid.contiguous()
        if has_pair_valid
        else torch.empty(0, device=q_content.device, dtype=torch.bool)
    )
    has_position_query_valid = position_query_valid is not None
    position_query_valid_tensor = (
        position_query_valid.contiguous()
        if has_position_query_valid
        else torch.empty(0, device=q_content.device, dtype=torch.bool)
    )
    has_bias = u is not None and v_bias is not None and w is not None and gate is not None
    empty = q_content_bh.new_empty(0, dtype=torch.float32)
    compute_du = bool(has_bias and need_du)
    compute_dv_bias = bool(has_bias and need_dv_bias)
    compute_dw = bool(has_bias and need_dw)
    compute_dgate = bool(has_bias and need_dgate)
    if has_bias:
        u_bh = _reshape_batch_heads(u)
        v_bias_bh = _reshape_batch_heads(v_bias)
        w_bh = _reshape_batch_heads(w)
        gate_bh = gate.contiguous().view(batch_size * num_heads, num_tokens)
        du = torch.zeros_like(u_bh, dtype=torch.float32) if compute_du else None
        dv_bias = torch.zeros_like(v_bias_bh, dtype=torch.float32) if compute_dv_bias else None
        dw = torch.zeros_like(w_bh, dtype=torch.float32) if compute_dw else None
        dgate = torch.zeros_like(gate_bh, dtype=torch.float32) if compute_dgate else None
    else:
        u_bh = v_bias_bh = w_bh = gate_bh = empty
        du = dv_bias = dw = dgate = None

    batch_heads = batch_size * num_heads
    num_tiles_j = triton.cdiv(num_tokens, _BLOCK_J)
    num_tiles_k = triton.cdiv(num_tokens, _BLOCK_K)
    grid = (batch_heads * num_tokens,)
    _streaming_backward_native_rope_no_carrier_kernel[grid](
        grad_out_bh,
        q_content_bh,
        q_rope_bh,
        k1_bh,
        v1_bh,
        k2_bh,
        v2_bh,
        out_bh,
        lse_bh,
        positions.contiguous().to(device=q_content.device, dtype=torch.float32),
        mu.contiguous().to(device=q_content.device, dtype=torch.float32),
        nu.contiguous().to(device=q_content.device, dtype=torch.float32),
        rope_logit_scale.contiguous().view(num_heads).to(device=q_content.device, dtype=torch.float32),
        rope_value_scale_h,
        query_valid.contiguous(),
        pair_key_valid.contiguous(),
        pair_valid_tensor,
        position_query_valid_tensor,
        u_bh,
        v_bias_bh,
        w_bh,
        gate_bh,
        dq_content,
        dq_rope,
        dk1,
        dv1,
        dk2,
        dv2,
        dpositions,
        dmu,
        dnu,
        d_rope_logit_scale,
        d_rope_value_scale if d_rope_value_scale is not None else empty,
        du if du is not None else empty,
        dv_bias if dv_bias is not None else empty,
        dw if dw is not None else empty,
        dgate if dgate is not None else empty,
        num_tokens,
        head_dim,
        rope_n_freqs,
        rope_value_n_freqs,
        rope_n_freqs**-0.5,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_POSITION_QUERY_VALID=has_position_query_valid,
        HAS_BIAS=has_bias,
        HAS_ROPE_VALUE_CARRIER=has_value_carrier,
        COMPUTE_DU=compute_du,
        COMPUTE_DV_BIAS=compute_dv_bias,
        COMPUTE_DW=compute_dw,
        COMPUTE_DGATE=compute_dgate,
        COMPUTE_DVALUE_SCALE=compute_dvalue_scale,
        USE_FP32_INPUT=use_fp32_input,
        INPUT_PRECISION=input_precision,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=block_d,
        BLOCK_F=block_f,
        num_warps=_num_warps_for_block_d(block_d),
    )

    reshape = (batch_size, num_heads, num_tokens, head_dim)
    dq_content_out = dq_content.view(reshape).to(q_content.dtype)
    dq_rope_out = dq_rope.view(reshape).to(q_rope.dtype)
    dk1_out = dk1.view(reshape).to(k1.dtype)
    dv1_out = dv1.view(reshape).to(v1.dtype)
    dk2_out = dk2.view(reshape).to(k2.dtype)
    dv2_out = dv2.view(reshape).to(v2.dtype)
    dpositions_out = dpositions.to(positions.dtype)
    dmu_out = dmu.to(mu.dtype)
    dnu_out = dnu.to(nu.dtype)
    d_rope_logit_scale_out = d_rope_logit_scale.to(rope_logit_scale.dtype)
    d_rope_value_scale_out = (
        d_rope_value_scale.to(rope_value_scale.dtype)
        if d_rope_value_scale is not None and rope_value_scale is not None
        else None
    )

    if has_bias:
        pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
        du_out = du.view(pair_shape).to(u.dtype) if du is not None else None
        dv_bias_out = dv_bias.view(pair_shape).to(v_bias.dtype) if dv_bias is not None else None
        dw_out = dw.view(pair_shape).to(w.dtype) if dw is not None else None
        dgate_out = dgate.view(batch_size, num_heads, num_tokens).to(gate.dtype) if dgate is not None else None
    else:
        du_out = dv_bias_out = dw_out = dgate_out = None

    return (
        dq_content_out,
        dq_rope_out,
        dk1_out,
        dv1_out,
        dk2_out,
        dv2_out,
        dpositions_out,
        dmu_out,
        dnu_out,
        d_rope_logit_scale_out,
        d_rope_value_scale_out,
        du_out,
        dv_bias_out,
        dw_out,
        dgate_out,
    )


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
    u_extra: torch.Tensor | None,
    v_bias_extra: torch.Tensor | None,
    w_extra: torch.Tensor | None,
    gate_extra: torch.Tensor | None,
    angle_left: torch.Tensor | None,
    angle_right: torch.Tensor | None,
    angle_gate: torch.Tensor | None,
    message_left: torch.Tensor | None,
    message_right: torch.Tensor | None,
    message_basis: torch.Tensor | None,
    pair_value_positions: torch.Tensor | None,
    pair_value_freqs: torch.Tensor | None,
    pair_value_scale: torch.Tensor | None,
    pair_value_query_mask: torch.Tensor | None,
    precision: str,
    need_du: bool = True,
    need_dv_bias: bool = True,
    need_dw: bool = True,
    need_dgate: bool = True,
    need_du_extra: bool = True,
    need_dv_bias_extra: bool = True,
    need_dw_extra: bool = True,
    need_dgate_extra: bool = True,
    need_dangle_left: bool = True,
    need_dangle_right: bool = True,
    need_dangle_gate: bool = True,
    need_dmessage_left: bool = True,
    need_dmessage_right: bool = True,
    need_dmessage_basis: bool = True,
    need_dpair_value_positions: bool = True,
    need_dpair_value_freqs: bool = True,
    need_dpair_value_scale: bool = True,
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
    compute_du = bool(has_bias and need_du)
    compute_dv_bias = bool(has_bias and need_dv_bias)
    compute_dw = bool(has_bias and need_dw)
    compute_dgate = bool(has_bias and need_dgate)
    if has_bias:
        u_bh = _reshape_batch_heads(u)
        v_bias_bh = _reshape_batch_heads(v_bias)
        w_bh = _reshape_batch_heads(w)
        gate_bh = gate.contiguous().view(batch_size * num_heads, num_tokens)
        du = torch.zeros_like(u_bh, dtype=torch.float32) if compute_du else None
        dv_bias = torch.zeros_like(v_bias_bh, dtype=torch.float32) if compute_dv_bias else None
        dw = torch.zeros_like(w_bh, dtype=torch.float32) if compute_dw else None
        dgate = torch.zeros_like(gate_bh, dtype=torch.float32) if compute_dgate else None
    else:
        u_bh = v_bias_bh = w_bh = gate_bh = empty
        du = dv_bias = dw = dgate = None
    has_bias_extra = u_extra is not None and v_bias_extra is not None and w_extra is not None and gate_extra is not None
    compute_du_extra = bool(has_bias_extra and need_du_extra)
    compute_dv_bias_extra = bool(has_bias_extra and need_dv_bias_extra)
    compute_dw_extra = bool(has_bias_extra and need_dw_extra)
    compute_dgate_extra = bool(has_bias_extra and need_dgate_extra)
    if has_bias_extra:
        u_extra_bh = _reshape_batch_heads(u_extra)
        v_bias_extra_bh = _reshape_batch_heads(v_bias_extra)
        w_extra_bh = _reshape_batch_heads(w_extra)
        gate_extra_bh = gate_extra.contiguous().view(batch_size * num_heads, num_tokens)
        du_extra = torch.zeros_like(u_extra_bh, dtype=torch.float32) if compute_du_extra else None
        dv_bias_extra_out = torch.zeros_like(v_bias_extra_bh, dtype=torch.float32) if compute_dv_bias_extra else None
        dw_extra = torch.zeros_like(w_extra_bh, dtype=torch.float32) if compute_dw_extra else None
        dgate_extra = torch.zeros_like(gate_extra_bh, dtype=torch.float32) if compute_dgate_extra else None
    else:
        u_extra_bh = v_bias_extra_bh = w_extra_bh = gate_extra_bh = empty
        du_extra = dv_bias_extra_out = dw_extra = dgate_extra = None

    has_low_rank_angle = angle_left is not None and angle_right is not None and angle_gate is not None
    compute_dangle_left = bool(has_low_rank_angle and need_dangle_left)
    compute_dangle_right = bool(has_low_rank_angle and need_dangle_right)
    compute_dangle_gate = bool(has_low_rank_angle and need_dangle_gate)
    if has_low_rank_angle:
        if angle_left.shape != angle_right.shape:
            raise ValueError(f"angle_left and angle_right must have the same shape, got {angle_left.shape} and {angle_right.shape}")
        angle_rank = int(angle_left.shape[-1])
        angle_scale = angle_rank**-0.5
        angle_left_bh = _reshape_batch_heads(angle_left)
        angle_right_bh = _reshape_batch_heads(angle_right)
        angle_gate_bh = angle_gate.contiguous().view(batch_size * num_heads, num_tokens)
        dangle_left = torch.zeros_like(angle_left_bh, dtype=torch.float32) if compute_dangle_left else None
        dangle_right = torch.zeros_like(angle_right_bh, dtype=torch.float32) if compute_dangle_right else None
        dangle_gate = torch.zeros_like(angle_gate_bh, dtype=torch.float32) if compute_dangle_gate else None
    else:
        angle_rank = 1
        angle_scale = 1.0
        angle_left_bh = angle_right_bh = angle_gate_bh = empty
        dangle_left = dangle_right = dangle_gate = None

    has_low_rank_message = message_left is not None and message_right is not None and message_basis is not None
    compute_dmessage_left = bool(has_low_rank_message and need_dmessage_left)
    compute_dmessage_right = bool(has_low_rank_message and need_dmessage_right)
    compute_dmessage_basis = bool(has_low_rank_message and need_dmessage_basis)
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
        dmessage_left = torch.zeros_like(message_left_bh, dtype=torch.float32) if compute_dmessage_left else None
        dmessage_right = torch.zeros_like(message_right_bh, dtype=torch.float32) if compute_dmessage_right else None
        dmessage_basis = torch.zeros_like(message_basis_h, dtype=torch.float32) if compute_dmessage_basis else None
    else:
        message_rank = 1
        message_scale = 1.0
        message_left_bh = message_right_bh = message_basis_h = empty
        dmessage_left = dmessage_right = dmessage_basis = None

    has_pair_value_marginal = pair_value_positions is not None and pair_value_freqs is not None and pair_value_scale is not None
    compute_dpair_value_positions = bool(has_pair_value_marginal and need_dpair_value_positions)
    compute_dpair_value_freqs = bool(has_pair_value_marginal and need_dpair_value_freqs)
    compute_dpair_value_scale = bool(has_pair_value_marginal and need_dpair_value_scale)
    compute_dpair_value_geometry = bool(compute_dpair_value_positions or compute_dpair_value_freqs)
    if has_pair_value_marginal:
        if pair_value_positions.shape != (batch_size, num_tokens, 3):
            raise ValueError(
                f"pair_value_positions must have shape {(batch_size, num_tokens, 3)}, got {tuple(pair_value_positions.shape)}"
            )
        if pair_value_freqs.ndim != 3 or pair_value_freqs.shape[0] != num_heads or pair_value_freqs.shape[2] != 3:
            raise ValueError(f"pair_value_freqs must have shape {(num_heads, 'F', 3)}, got {tuple(pair_value_freqs.shape)}")
        pair_value_n_freqs = int(pair_value_freqs.shape[1])
        if pair_value_n_freqs <= 0:
            raise ValueError("pair_value_freqs must contain at least one frequency")
        if 2 * pair_value_n_freqs > head_dim:
            raise ValueError(
                f"pair-RoPE marginal value transport needs 2 * n_freqs <= head_dim, got {pair_value_n_freqs} and {head_dim}"
            )
        if pair_value_scale.shape != (num_heads,):
            raise ValueError(f"pair_value_scale must have shape {(num_heads,)}, got {tuple(pair_value_scale.shape)}")
        if pair_value_query_mask is None:
            pair_value_query_mask_tensor = torch.ones((batch_size, num_tokens), device=q.device, dtype=torch.bool)
        else:
            if pair_value_query_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    f"pair_value_query_mask must have shape {(batch_size, num_tokens)}, got {tuple(pair_value_query_mask.shape)}"
                )
            pair_value_query_mask_tensor = pair_value_query_mask.contiguous()
        pair_value_positions_tensor = pair_value_positions.contiguous().to(device=q.device, dtype=torch.float32)
        pair_value_freqs_tensor = pair_value_freqs.contiguous().to(device=q.device, dtype=torch.float32)
        pair_value_scale_tensor = pair_value_scale.contiguous().to(device=q.device, dtype=torch.float32)
        dpair_value_positions = (
            torch.zeros((batch_size, num_tokens, 3), device=q.device, dtype=torch.float32)
            if compute_dpair_value_positions
            else None
        )
        dpair_value_freqs = (
            torch.zeros((num_heads, pair_value_n_freqs, 3), device=q.device, dtype=torch.float32)
            if compute_dpair_value_freqs
            else None
        )
        dpair_value_scale = (
            torch.zeros((num_heads,), device=q.device, dtype=torch.float32)
            if compute_dpair_value_scale
            else None
        )
    else:
        pair_value_n_freqs = 1
        pair_value_positions_tensor = pair_value_freqs_tensor = pair_value_scale_tensor = empty
        pair_value_query_mask_tensor = torch.empty(0, device=q.device, dtype=torch.bool)
        dpair_value_positions = dpair_value_freqs = dpair_value_scale = None

    block_r = _block_r_for_rank(
        max(
            angle_rank if has_low_rank_angle else 1,
            message_rank if has_low_rank_message else 1,
            pair_value_n_freqs if has_pair_value_marginal else 1,
        )
    )

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
        u_extra_bh,
        v_bias_extra_bh,
        w_extra_bh,
        gate_extra_bh,
        angle_left_bh,
        angle_right_bh,
        angle_gate_bh,
        message_left_bh,
        message_right_bh,
        message_basis_h,
        pair_value_positions_tensor,
        pair_value_freqs_tensor,
        pair_value_scale_tensor,
        pair_value_query_mask_tensor,
        dq,
        dk1,
        dv1,
        dk2,
        dv2,
        du if du is not None else empty,
        dv_bias if dv_bias is not None else empty,
        dw if dw is not None else empty,
        dgate if dgate is not None else empty,
        du_extra if du_extra is not None else empty,
        dv_bias_extra_out if dv_bias_extra_out is not None else empty,
        dw_extra if dw_extra is not None else empty,
        dgate_extra if dgate_extra is not None else empty,
        dangle_left if dangle_left is not None else empty,
        dangle_right if dangle_right is not None else empty,
        dangle_gate if dangle_gate is not None else empty,
        dmessage_left if dmessage_left is not None else empty,
        dmessage_right if dmessage_right is not None else empty,
        dmessage_basis if dmessage_basis is not None else empty,
        dpair_value_positions if dpair_value_positions is not None else empty,
        dpair_value_freqs if dpair_value_freqs is not None else empty,
        dpair_value_scale if dpair_value_scale is not None else empty,
        num_tokens,
        head_dim,
        angle_rank,
        angle_scale,
        message_rank,
        message_scale,
        pair_value_n_freqs,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_BIAS=has_bias,
        HAS_BIAS_EXTRA=has_bias_extra,
        HAS_LOW_RANK_ANGLE=has_low_rank_angle,
        HAS_LOW_RANK_MESSAGE=has_low_rank_message,
        HAS_PAIR_VALUE_MARGINAL=has_pair_value_marginal,
        COMPUTE_DU=compute_du,
        COMPUTE_DV_BIAS=compute_dv_bias,
        COMPUTE_DW=compute_dw,
        COMPUTE_DGATE=compute_dgate,
        COMPUTE_DU_EXTRA=compute_du_extra,
        COMPUTE_DV_BIAS_EXTRA=compute_dv_bias_extra,
        COMPUTE_DW_EXTRA=compute_dw_extra,
        COMPUTE_DGATE_EXTRA=compute_dgate_extra,
        COMPUTE_DANGLE_LEFT=compute_dangle_left,
        COMPUTE_DANGLE_RIGHT=compute_dangle_right,
        COMPUTE_DANGLE_GATE=compute_dangle_gate,
        COMPUTE_DMESSAGE_LEFT=compute_dmessage_left,
        COMPUTE_DMESSAGE_RIGHT=compute_dmessage_right,
        COMPUTE_DMESSAGE_BASIS=compute_dmessage_basis,
        COMPUTE_DPAIR_VALUE_POSITIONS=compute_dpair_value_positions,
        COMPUTE_DPAIR_VALUE_FREQS=compute_dpair_value_freqs,
        COMPUTE_DPAIR_VALUE_SCALE=compute_dpair_value_scale,
        COMPUTE_DPAIR_VALUE_GEOMETRY=compute_dpair_value_geometry,
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
        du_out = du.view(pair_shape).to(u.dtype) if du is not None else None
        dv_bias_out = dv_bias.view(pair_shape).to(v_bias.dtype) if dv_bias is not None else None
        dw_out = dw.view(pair_shape).to(w.dtype) if dw is not None else None
        dgate_out = dgate.view(batch_size, num_heads, num_tokens).to(gate.dtype) if dgate is not None else None
    else:
        du_out = dv_bias_out = dw_out = dgate_out = None
    if has_bias_extra:
        pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
        du_extra_out = du_extra.view(pair_shape).to(u_extra.dtype) if du_extra is not None else None
        dv_bias_extra_result = (
            dv_bias_extra_out.view(pair_shape).to(v_bias_extra.dtype)
            if dv_bias_extra_out is not None
            else None
        )
        dw_extra_out = dw_extra.view(pair_shape).to(w_extra.dtype) if dw_extra is not None else None
        dgate_extra_out = (
            dgate_extra.view(batch_size, num_heads, num_tokens).to(gate_extra.dtype)
            if dgate_extra is not None
            else None
        )
    else:
        du_extra_out = dv_bias_extra_result = dw_extra_out = dgate_extra_out = None

    if has_low_rank_angle:
        angle_shape = (batch_size, num_heads, num_tokens, num_tokens, angle_rank)
        dangle_left_out = dangle_left.view(angle_shape).to(angle_left.dtype) if dangle_left is not None else None
        dangle_right_out = dangle_right.view(angle_shape).to(angle_right.dtype) if dangle_right is not None else None
        dangle_gate_out = (
            dangle_gate.view(batch_size, num_heads, num_tokens).to(angle_gate.dtype)
            if dangle_gate is not None
            else None
        )
    else:
        dangle_left_out = dangle_right_out = dangle_gate_out = None

    if has_low_rank_message:
        message_shape = (batch_size, num_heads, num_tokens, num_tokens, message_rank)
        dmessage_left_out = dmessage_left.view(message_shape).to(message_left.dtype) if dmessage_left is not None else None
        dmessage_right_out = (
            dmessage_right.view(message_shape).to(message_right.dtype)
            if dmessage_right is not None
            else None
        )
        dmessage_basis_out = dmessage_basis.to(message_basis.dtype) if dmessage_basis is not None else None
    else:
        dmessage_left_out = dmessage_right_out = dmessage_basis_out = None
    if has_pair_value_marginal:
        dpair_value_positions_out = (
            dpair_value_positions.to(pair_value_positions.dtype)
            if dpair_value_positions is not None and pair_value_positions is not None
            else None
        )
        dpair_value_freqs_out = (
            dpair_value_freqs.to(pair_value_freqs.dtype)
            if dpair_value_freqs is not None and pair_value_freqs is not None
            else None
        )
        dpair_value_scale_out = (
            dpair_value_scale.to(pair_value_scale.dtype)
            if dpair_value_scale is not None and pair_value_scale is not None
            else None
        )
    else:
        dpair_value_positions_out = dpair_value_freqs_out = dpair_value_scale_out = None

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
        du_extra_out,
        dv_bias_extra_result,
        dw_extra_out,
        dgate_extra_out,
        dangle_left_out,
        dangle_right_out,
        dangle_gate_out,
        dmessage_left_out,
        dmessage_right_out,
        dmessage_basis_out,
        dpair_value_positions_out,
        dpair_value_freqs_out,
        dpair_value_scale_out,
    )
