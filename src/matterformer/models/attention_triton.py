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
        out_ptr,
        lse_ptr,
        num_tokens,
        head_dim,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        USE_FP32_INPUT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
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
        if USE_FP32_INPUT:
            q_score = q_raw.to(tl.float32)
        else:
            q_score = q_raw

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
                        qk1 = k1_score * q_score[None, :]
                    else:
                        k2_score = k2_raw
                        qk1 = (k1_score * q_score[None, :]).to(k1_score.dtype)

                    scores = tl.dot(
                        qk1,
                        tl.trans(k2_score),
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
        dq_ptr,
        dk1_ptr,
        dv1_ptr,
        dk2_ptr,
        dv2_ptr,
        du_ptr,
        dv_bias_ptr,
        dw_ptr,
        dgate_ptr,
        num_tokens,
        head_dim,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        USE_FP32_INPUT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_idx = pid % num_tokens
        bh_idx = pid // num_tokens
        batch_idx = bh_idx // num_heads

        q_is_valid = tl.load(query_valid_ptr + batch_idx * num_tokens + q_idx) > 0
        offs_d = tl.arange(0, BLOCK_D)
        dq_row_ptr = dq_ptr + (bh_idx * num_tokens + q_idx) * head_dim + offs_d
        if not q_is_valid:
            tl.store(dq_row_ptr, 0.0, mask=offs_d < head_dim)
            if HAS_BIAS:
                tl.store(dgate_ptr + bh_idx * num_tokens + q_idx, 0.0)
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

        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dgate_acc = 0.0

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
                        qk1 = k1_score * q_score[None, :]
                    else:
                        k2_score = k2_raw
                        qk1 = (k1_score * q_score[None, :]).to(k1_score.dtype)

                    scores = tl.dot(
                        qk1,
                        tl.trans(k2_score),
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

                    scores = tl.where(valid, scores, -float("inf"))
                    probs = tl.where(valid, tl.exp(scores - lse), 0.0)

                    value_dot = tl.dot(
                        v1_32 * go[None, :],
                        tl.trans(v2_32),
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

            if HAS_BIAS:
                tl.store(
                    du_ptr + (bh_idx * num_tokens + q_idx) * num_tokens + offs_j,
                    du_running,
                    mask=offs_j < num_tokens,
                )

        tl.store(dq_row_ptr, dq_acc, mask=offs_d < head_dim)
        if HAS_BIAS:
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
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if has_bias:
        u_bh = _reshape_batch_heads(u)
        v_bias_bh = _reshape_batch_heads(v_bias)
        w_bh = _reshape_batch_heads(w)
        gate_bh = gate.contiguous().view(batch_size * num_heads, num_tokens)
    else:
        empty = q_bh.new_empty(0)
        u_bh = v_bias_bh = w_bh = gate_bh = empty

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
        out,
        lse,
        num_tokens,
        head_dim,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_BIAS=has_bias,
        USE_FP32_INPUT=use_fp32_input,
        INPUT_PRECISION=input_precision,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=block_d,
        num_warps=_num_warps_for_block_d(block_d),
    )
    out = out.view(batch_size, num_heads, num_tokens, head_dim).to(q.dtype)
    lse = lse.view(batch_size, num_heads, num_tokens)
    return out, lse


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
        empty = q_bh.new_empty(0, dtype=torch.float32)
        u_bh = v_bias_bh = w_bh = gate_bh = empty
        du = dv_bias = dw = dgate = None

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
        dq,
        dk1,
        dv1,
        dk2,
        dv2,
        du if du is not None else empty,
        dv_bias if dv_bias is not None else empty,
        dw if dw is not None else empty,
        dgate if dgate is not None else empty,
        num_tokens,
        head_dim,
        num_heads,
        num_tiles_j,
        num_tiles_k,
        HAS_PAIR_VALID=has_pair_valid,
        HAS_BIAS=has_bias,
        USE_FP32_INPUT=use_fp32_input,
        INPUT_PRECISION=input_precision,
        BLOCK_J=_BLOCK_J,
        BLOCK_K=_BLOCK_K,
        BLOCK_D=block_d,
        num_warps=_num_warps_for_block_d(block_d),
    )

    reshape = (batch_size, num_heads, num_tokens, head_dim)
    dq_out = dq.view(reshape).to(q.dtype)
    dk1_out = dk1.view(reshape).to(k1.dtype)
    dv1_out = dv1.view(reshape).to(v1.dtype)
    dk2_out = dk2.view(reshape).to(k2.dtype)
    dv2_out = dv2.view(reshape).to(v2.dtype)

    if not has_bias:
        return dq_out, dk1_out, dv1_out, dk2_out, dv2_out, None, None, None, None

    pair_shape = (batch_size, num_heads, num_tokens, num_tokens)
    du_out = du.view(pair_shape).to(u.dtype)
    dv_bias_out = dv_bias.view(pair_shape).to(v_bias.dtype)
    dw_out = dw.view(pair_shape).to(w.dtype)
    dgate_out = dgate.view(batch_size, num_heads, num_tokens).to(gate.dtype)
    return dq_out, dk1_out, dv1_out, dk2_out, dv2_out, du_out, dv_bias_out, dw_out, dgate_out
