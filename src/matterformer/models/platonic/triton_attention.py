from __future__ import annotations

import math
import os

import torch

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
    if key not in PLATONIC_ATTENTION_BIAS_MODES:
        raise ValueError(f"unknown Platonic attention bias mode: {value!r}")
    return PLATONIC_ATTENTION_BIAS_MODES[key]


def _next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def platonic_attention_block_d_for_head_dim(head_dim: int) -> int:
    return max(16, _next_power_of_2(int(head_dim)))


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


def platonic_attention_flat_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,
    pos: torch.Tensor | None = None,
    heads_per_frame: int | None = None,
    rbf_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    radial_bias_kind: str | int | None = None,
    diag_zero: bool = True,
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
    if use_rbf:
        if pos is None or heads_per_frame is None or gate is None or centers is None or gamma is None:
            raise ValueError("Radial bias requires pos, heads_per_frame, gate, centers, and gamma")
        if pos.ndim != 2 or pos.shape != (q.shape[0], 3):
            raise ValueError(f"pos must have shape [{q.shape[0]}, 3], got {tuple(pos.shape)}")
    bias_mode = normalize_platonic_attention_bias_mode(radial_bias_kind, has_bias=use_rbf)
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
        if use_rbf:
            assert pos is not None
            bias = _radial_rbf_bias_reference(
                pos[start_i:end_i],
                pos[start_i:end_i],
                num_heads=q.shape[1],
                heads_per_frame=int(heads_per_frame),
                rbf_weight=rbf_weight,
                gate=gate,
                centers=centers,
                gamma=gamma,
                diag_zero=diag_zero,
                bias_mode=bias_mode,
            )
            scores = scores + bias.to(dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        if training and dropout_p > 0.0:
            probs = torch.nn.functional.dropout(probs, p=float(dropout_p), training=True)
        outputs.append(torch.matmul(probs, v_seg).transpose(0, 1).contiguous())
    if not outputs:
        return torch.zeros_like(q)
    return torch.cat(outputs, dim=0)


if TRITON_PLATONIC_ATTENTION_AVAILABLE:

    @triton.jit
    def _radial_bias_tile(
        pos_ptr,
        rbf_weight_ptr,
        gate_ptr,
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
        gate = tl.load(gate_ptr + subhead).to(tl.float32)
        bias = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if bias_mode == 1:
            dist = tl.sqrt(dist2)
            gamma = tl.load(gamma_ptr).to(tl.float32)
            for rb in range(0, num_rbf):
                center = tl.load(centers_ptr + rb).to(tl.float32)
                weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                rho = tl.exp(-gamma * (dist - center) * (dist - center))
                bias += weight * rho
        elif bias_mode == 2:
            weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
            bias = weight * dist2
        elif bias_mode == 3:
            weight = tl.load(rbf_weight_ptr + subhead * num_rbf).to(tl.float32)
            bias = weight * tl.sqrt(dist2)
        bias *= 1.0 + gate
        if diag_zero:
            bias = tl.where(offs_m[:, None] == offs_n[None, :], 0.0, bias)
        return bias

    @triton.jit
    def _platonic_flat_attention_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        cu_ptr,
        rbf_weight_ptr,
        gate_ptr,
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
                        rbf_weight_ptr,
                        gate_ptr,
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
    def _platonic_flat_attention_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pos_ptr,
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        centers_ptr,
        gamma_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_gate_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
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
                    bias = base_bias * gate_factor
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
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
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
                        rbf_weight_ptr,
                        gate_ptr,
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
        cu_ptr,
        dout_ptr,
        lse_ptr,
        delta_ptr,
        rbf_weight_ptr,
        gate_ptr,
        centers_ptr,
        gamma_ptr,
        dk_ptr,
        dv_ptr,
        d_rbf_weight_ptr,
        d_gate_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_seqlen: tl.constexpr,
        scale: tl.constexpr,
        heads_per_frame: tl.constexpr,
        num_rbf: tl.constexpr,
        has_rbf: tl.constexpr,
        bias_mode: tl.constexpr,
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
        gate = tl.load(gate_ptr + subhead).to(tl.float32) if has_rbf else 0.0
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
                    bias = base_bias * gate_factor
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


class _PlatonicFlatAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rbf_weight: torch.Tensor,
        gate: torch.Tensor,
        centers: torch.Tensor,
        gamma: torch.Tensor,
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
                heads_per_frame=heads_per_frame if has_rbf else None,
                rbf_weight=rbf_weight if has_rbf else None,
                gate=gate if has_rbf else None,
                centers=centers if has_rbf else None,
                gamma=gamma if has_rbf else None,
                radial_bias_kind=bias_mode,
                diag_zero=diag_zero,
            )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pos = pos.contiguous()
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        rbf_weight = rbf_weight.contiguous()
        gate = gate.contiguous()
        centers = centers.contiguous()
        gamma = gamma.contiguous()
        total_tokens, num_heads, head_dim = q.shape
        batch_size = int(cu.numel() - 1)
        max_seqlen = int(max_seqlen)
        out = torch.empty_like(q)
        lse = torch.empty((total_tokens, num_heads), device=q.device, dtype=torch.float32)
        block_d = platonic_attention_block_d_for_head_dim(head_dim)
        grid = (triton.cdiv(max(max_seqlen, 1), int(block_m)), num_heads, batch_size)
        _platonic_flat_attention_fwd_kernel[grid](
            q,
            k,
            v,
            pos,
            cu,
            rbf_weight,
            gate,
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
            bool(diag_zero),
            input_precision,
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, pos, cu, out, lse, rbf_weight, gate, centers, gamma)
        ctx.heads_per_frame = int(heads_per_frame)
        ctx.diag_zero = bool(diag_zero)
        ctx.precision = input_precision
        ctx.bias_mode = int(bias_mode)
        ctx.block_m = int(block_m)
        ctx.block_n = int(block_n)
        ctx.max_seqlen = max_seqlen
        ctx.has_rbf = bool(has_rbf)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, pos, cu, out, lse, rbf_weight, gate, centers, gamma = ctx.saved_tensors
        if not TRITON_PLATONIC_ATTENTION_AVAILABLE or not q.is_cuda:
            with torch.enable_grad():
                q_ref = q.detach().requires_grad_(True)
                k_ref = k.detach().requires_grad_(True)
                v_ref = v.detach().requires_grad_(True)
                weight_ref = rbf_weight.detach().requires_grad_(ctx.has_rbf)
                gate_ref = gate.detach().requires_grad_(ctx.has_rbf)
                ref_out = platonic_attention_flat_torch_reference(
                    q_ref,
                    k_ref,
                    v_ref,
                    cu_seqlens=cu,
                    pos=pos if ctx.has_rbf else None,
                    heads_per_frame=ctx.heads_per_frame if ctx.has_rbf else None,
                    rbf_weight=weight_ref if ctx.has_rbf else None,
                    gate=gate_ref if ctx.has_rbf else None,
                    centers=centers if ctx.has_rbf else None,
                    gamma=gamma if ctx.has_rbf else None,
                    radial_bias_kind=ctx.bias_mode,
                    diag_zero=ctx.diag_zero,
                )
                grads = torch.autograd.grad(
                    ref_out,
                    (q_ref, k_ref, v_ref, weight_ref, gate_ref) if ctx.has_rbf else (q_ref, k_ref, v_ref),
                    dout,
                    allow_unused=True,
                )
            if ctx.has_rbf:
                dq, dk, dv, dw, dg = grads
            else:
                dq, dk, dv = grads
                dw = dg = None
            return dq, dk, dv, None, None, None, dw, dg, None, None, None, None, None, None, None, None, None

        dout = dout.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        d_rbf_weight = torch.zeros_like(rbf_weight) if ctx.has_rbf else torch.zeros_like(rbf_weight)
        d_gate = torch.zeros_like(gate) if ctx.has_rbf else torch.zeros_like(gate)
        block_d = platonic_attention_block_d_for_head_dim(q.shape[-1])
        delta_mode = os.environ.get("MATTERFORMER_PLATONIC_TRITON_DELTA", "triton_vector").lower()
        if delta_mode == "triton":
            delta = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
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
        use_atomic_bwd = bwd_mode == "atomic" or (
            bwd_mode == "auto" and ctx.has_rbf and ctx.bias_mode != PLATONIC_ATTENTION_BIAS_MODES["radial_rbf"]
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
                cu,
                dout,
                lse,
                delta,
                rbf_weight,
                gate,
                centers,
                gamma,
                dq,
                dk,
                dv,
                d_rbf_weight,
                d_gate,
                q.shape[1],
                q.shape[2],
                max_seqlen,
                1.0 / math.sqrt(q.shape[2]),
                ctx.heads_per_frame,
                int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
                ctx.has_rbf,
                ctx.bias_mode,
                ctx.diag_zero,
                ctx.precision,
                BLOCK_M=ctx.block_m,
                BLOCK_N=ctx.block_n,
                BLOCK_D=block_d,
                num_warps=4,
            )
            return (
                dq,
                dk,
                dv,
                None,
                None,
                None,
                d_rbf_weight if ctx.has_rbf else None,
                d_gate if ctx.has_rbf else None,
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
        dq_grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_m), q.shape[1], batch_size)
        _platonic_flat_attention_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            pos,
            cu,
            dout,
            lse,
            delta,
            rbf_weight,
            gate,
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
            ctx.diag_zero,
            ctx.precision,
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        dkv_grid = (triton.cdiv(max(max_seqlen, 1), ctx.block_n), q.shape[1], batch_size)
        _platonic_flat_attention_bwd_dkv_kernel[dkv_grid](
            q,
            k,
            v,
            pos,
            cu,
            dout,
            lse,
            delta,
            rbf_weight,
            gate,
            centers,
            gamma,
            dk,
            dv,
            d_rbf_weight,
            d_gate,
            q.shape[1],
            q.shape[2],
            max_seqlen,
            1.0 / math.sqrt(q.shape[2]),
            ctx.heads_per_frame,
            int(rbf_weight.shape[-1]) if ctx.has_rbf else 1,
            ctx.has_rbf,
            ctx.bias_mode,
            ctx.diag_zero,
            ctx.precision,
            BLOCK_M=ctx.block_m,
            BLOCK_N=ctx.block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            d_rbf_weight if ctx.has_rbf else None,
            d_gate if ctx.has_rbf else None,
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


@_disable_dynamo_if_available
def platonic_attention_flat_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    pos: torch.Tensor | None = None,
    heads_per_frame: int | None = None,
    rbf_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
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
    if use_rbf:
        if pos is None or heads_per_frame is None or gate is None or centers is None or gamma is None:
            raise ValueError("Platonic radial Triton attention requires pos, heads_per_frame, gate, centers, and gamma")
        if q.shape[1] % int(heads_per_frame) != 0:
            raise ValueError("heads_per_frame must divide num_heads for Platonic radial Triton attention")
        if rbf_weight.ndim != 2 or rbf_weight.shape[0] != int(heads_per_frame):
            raise ValueError("rbf_weight must have shape [heads_per_frame, num_basis]")
        if gate.shape != (int(heads_per_frame),):
            raise ValueError("gate must have shape [heads_per_frame]")
        if pos.ndim != 2 or pos.shape != (q.shape[0], 3):
            raise ValueError(f"pos must have shape [{q.shape[0]}, 3], got {tuple(pos.shape)}")
        if centers.ndim != 1:
            raise ValueError("centers must be a 1D tensor")
        if centers.numel() != rbf_weight.shape[1]:
            raise ValueError("centers length must match rbf_weight.shape[1]")
    bias_mode = normalize_platonic_attention_bias_mode(radial_bias_kind, has_bias=use_rbf)
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
            heads_per_frame=heads_per_frame if use_rbf else None,
            rbf_weight=rbf_weight,
            gate=gate,
            centers=centers,
            gamma=gamma,
            radial_bias_kind=bias_mode,
            diag_zero=diag_zero,
        )
    q_apply = q.to(torch.bfloat16) if flash_compat_bf16 else q
    k_apply = k.to(torch.bfloat16) if flash_compat_bf16 else k
    v_apply = v.to(torch.bfloat16) if flash_compat_bf16 else v
    orig_dtype = q.dtype
    dummy_weight = rbf_weight if rbf_weight is not None else torch.empty((1, 1), device=q.device, dtype=torch.float32)
    dummy_gate = gate if gate is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
    dummy_centers = centers if centers is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
    dummy_gamma = gamma if gamma is not None else torch.ones((), device=q.device, dtype=torch.float32)
    dummy_pos = pos if pos is not None else torch.empty((q.shape[0], 3), device=q.device, dtype=torch.float32)
    if q.requires_grad or k.requires_grad or v.requires_grad or (rbf_weight is not None and rbf_weight.requires_grad):
        out = _PlatonicFlatAttentionFunction.apply(
            q_apply,
            k_apply,
            v_apply,
            dummy_pos,
            cu_seqlens,
            int(max_seqlen),
            dummy_weight,
            dummy_gate,
            dummy_centers,
            dummy_gamma,
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
        cu_seqlens,
        int(max_seqlen),
        dummy_weight,
        dummy_gate,
        dummy_centers,
        dummy_gamma,
        int(heads_per_frame or q.shape[1]),
        bool(diag_zero),
        kernel_precision,
        int(block_m),
        int(block_n),
        bool(use_rbf),
        int(bias_mode),
    )
    return out.to(orig_dtype) if flash_compat_bf16 else out
