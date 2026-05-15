from __future__ import annotations

import math

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

SUPPORTED_PLATONIC_ATTENTION_PRECISIONS = {"tf32", "tf32x3", "ieee"}


def normalize_platonic_attention_precision(value: str | None) -> str:
    precision = str(value or "tf32").lower()
    if precision not in SUPPORTED_PLATONIC_ATTENTION_PRECISIONS:
        raise ValueError(
            "platonic Triton attention precision must be one of "
            f"{sorted(SUPPORTED_PLATONIC_ATTENTION_PRECISIONS)}, got {value!r}"
        )
    return precision


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
) -> torch.Tensor:
    if heads_per_frame <= 0 or num_heads % int(heads_per_frame) != 0:
        raise ValueError("heads_per_frame must divide num_heads for radial RBF bias")
    delta = pos_j[None, :, :] - pos_i[:, None, :]
    dist = delta.square().sum(dim=-1).clamp_min(0.0).sqrt()
    basis = torch.exp(-gamma.to(dtype=dist.dtype, device=dist.device) * (dist[..., None] - centers.to(dist.device, dist.dtype)).square())
    subhead_bias = torch.einsum("ijm,hm->ijh", basis, rbf_weight.to(device=dist.device, dtype=dist.dtype))
    subhead_bias = subhead_bias * (1.0 + gate.to(device=dist.device, dtype=dist.dtype)).view(1, 1, -1)
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
    diag_zero: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Reference flat varlen attention with optional group-shared radial RBF bias.

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
            raise ValueError("Radial RBF bias requires pos, heads_per_frame, gate, centers, and gamma")
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
        dist = tl.sqrt(tl.maximum(dx * dx + dy * dy + dz * dz, 0.0))
        subhead = head_idx % heads_per_frame
        gamma = tl.load(gamma_ptr).to(tl.float32)
        gate = tl.load(gate_ptr + subhead).to(tl.float32)
        bias = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for rb in range(0, num_rbf):
            center = tl.load(centers_ptr + rb).to(tl.float32)
            weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
            rho = tl.exp(-gamma * (dist - center) * (dist - center))
            bias += weight * rho
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
                        diag_zero,
                        BLOCK_M,
                        BLOCK_N,
                    )
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=input_precision)
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
                    dist = tl.sqrt(tl.maximum(dx * dx + dy * dy + dz * dz, 0.0))
                    gamma = tl.load(gamma_ptr).to(tl.float32)
                    for rb in range(0, num_rbf):
                        center = tl.load(centers_ptr + rb).to(tl.float32)
                        weight = tl.load(rbf_weight_ptr + subhead * num_rbf + rb).to(tl.float32)
                        rho = tl.exp(-gamma * (dist - center) * (dist - center))
                        base_bias += weight * rho
                    bias = base_bias * gate_factor
                    if diag_zero:
                        bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                    scores += bias
                scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
                dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dq += tl.dot(ds, k, input_precision=input_precision) * scale
                dk = tl.dot(tl.trans(ds), q, input_precision=input_precision) * scale
                dv = tl.dot(tl.trans(p), dout, input_precision=input_precision)
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
                    for rb in range(0, num_rbf):
                        center = tl.load(centers_ptr + rb).to(tl.float32)
                        rho = tl.exp(-gamma * (dist - center) * (dist - center))
                        grad_w = tl.sum(tl.sum(ds_bias * rho * gate_factor, axis=0), axis=0)
                        tl.atomic_add(d_rbf_weight_ptr + subhead * num_rbf + rb, grad_w, sem="relaxed")
        tl.store(
            dq_ptr + ((start + offs_m[:, None]) * num_heads + head_idx) * head_dim + offs_d[None, :],
            dq,
            mask=m_mask[:, None] & d_mask[None, :],
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
    ) -> torch.Tensor:
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
            bool(diag_zero),
            normalize_platonic_attention_precision(precision),
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, pos, cu, out, lse, rbf_weight, gate, centers, gamma)
        ctx.heads_per_frame = int(heads_per_frame)
        ctx.diag_zero = bool(diag_zero)
        ctx.precision = normalize_platonic_attention_precision(precision)
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
            return dq, dk, dv, None, None, None, dw, dg, None, None, None, None, None, None, None, None

        dout = dout.contiguous()
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        d_rbf_weight = torch.zeros_like(rbf_weight) if ctx.has_rbf else torch.zeros_like(rbf_weight)
        d_gate = torch.zeros_like(gate) if ctx.has_rbf else torch.zeros_like(gate)
        delta = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        block_d = platonic_attention_block_d_for_head_dim(q.shape[-1])
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
        batch_size = int(cu.numel() - 1)
        max_seqlen = int(ctx.max_seqlen)
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
    precision: str = "tf32",
    block_m: int = 16,
    block_n: int = 32,
    strict: bool = False,
) -> torch.Tensor:
    _validate_flat_inputs(q, k, v, cu_seqlens)
    if max_seqlen == 0 or q.shape[0] == 0:
        return torch.zeros_like(q)
    use_rbf = rbf_weight is not None
    if use_rbf:
        if pos is None or heads_per_frame is None or gate is None or centers is None or gamma is None:
            raise ValueError("Radial RBF Triton attention requires pos, heads_per_frame, gate, centers, and gamma")
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
            diag_zero=diag_zero,
        )
    if q.requires_grad or k.requires_grad or v.requires_grad or (rbf_weight is not None and rbf_weight.requires_grad):
        dummy_weight = rbf_weight if rbf_weight is not None else torch.empty((1, 1), device=q.device, dtype=torch.float32)
        dummy_gate = gate if gate is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
        dummy_centers = centers if centers is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
        dummy_gamma = gamma if gamma is not None else torch.ones((), device=q.device, dtype=torch.float32)
        dummy_pos = pos if pos is not None else torch.empty((q.shape[0], 3), device=q.device, dtype=q.dtype)
        return _PlatonicFlatAttentionFunction.apply(
            q,
            k,
            v,
            dummy_pos,
            cu_seqlens,
            int(max_seqlen),
            dummy_weight,
            dummy_gate,
            dummy_centers,
            dummy_gamma,
            int(heads_per_frame or q.shape[1]),
            bool(diag_zero),
            normalize_platonic_attention_precision(precision),
            int(block_m),
            int(block_n),
            bool(use_rbf),
        )
    dummy_weight = rbf_weight if rbf_weight is not None else torch.empty((1, 1), device=q.device, dtype=torch.float32)
    dummy_gate = gate if gate is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
    dummy_centers = centers if centers is not None else torch.empty((1,), device=q.device, dtype=torch.float32)
    dummy_gamma = gamma if gamma is not None else torch.ones((), device=q.device, dtype=torch.float32)
    dummy_pos = pos if pos is not None else torch.empty((q.shape[0], 3), device=q.device, dtype=q.dtype)
    return _PlatonicFlatAttentionFunction.apply(
        q,
        k,
        v,
        dummy_pos,
        cu_seqlens,
        int(max_seqlen),
        dummy_weight,
        dummy_gate,
        dummy_centers,
        dummy_gamma,
        int(heads_per_frame or q.shape[1]),
        bool(diag_zero),
        normalize_platonic_attention_precision(precision),
        int(block_m),
        int(block_n),
        bool(use_rbf),
    )
