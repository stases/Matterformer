from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from matterformer.models.platonic.groups import PLATONIC_GROUPS
from matterformer.models.platonic.linear import PlatonicLinear
from matterformer.models.platonic.radius_sparse import RadiusBlockSparseLayout
from matterformer.models.platonic.rope import PlatonicRoPE
from matterformer.models.platonic.triton_attention import (
    platonic_radius_block_sparse_attention_torch_reference,
    platonic_radius_sparse_attention_flat_triton,
    platonic_attention_flat_torch_reference,
    platonic_attention_flat_triton,
)

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
except (ImportError, OSError):
    flash_attn_varlen_func = None


TRITON_BIAS_BACKEND_ALIASES = {
    "triton_radial_rbf": "radial_rbf",
    "triton_radial_r2": "radial_r2",
    "triton_radial_slope": "radial_slope",
    "triton_rbf_type_bias": "rbf_type_enveloped",
    "triton_radius_sparse": "radius_rbf_type_enveloped",
    "triton_esen_local": "radius_rbf_type_enveloped",
}
TORCH_BIAS_BACKEND_ALIASES = {
    "torch_rbf_type_bias": "rbf_type_enveloped",
    "torch_smooth_local": "rbf_type_enveloped",
    "torch_radius_sparse": "radius_rbf_type_enveloped",
    "torch_radius_rbf_type_bias": "radius_rbf_type_enveloped",
    "torch_esen_local": "radius_rbf_type_enveloped",
}
TRITON_RADIAL_BACKENDS = TRITON_BIAS_BACKEND_ALIASES
TORCH_LOCAL_BIAS_BACKENDS = TORCH_BIAS_BACKEND_ALIASES
LOCAL_BIAS_BACKENDS = {**TRITON_BIAS_BACKEND_ALIASES, **TORCH_BIAS_BACKEND_ALIASES}
CANONICAL_ATTENTION_BACKENDS = {"sdpa", "flash", "triton", "torch_reference"}
CANONICAL_ATTENTION_BIAS_KINDS = {
    "radial_rbf",
    "radial_r2",
    "radial_slope",
    "rbf_type_enveloped",
    "radius_rbf_type_enveloped",
}


def _normalize_attention_bias_kind(value: str | None) -> str | None:
    if value is None:
        return None
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
    if key not in CANONICAL_ATTENTION_BIAS_KINDS:
        raise ValueError(f"attention_bias.kind must be one of {sorted(CANONICAL_ATTENTION_BIAS_KINDS)}, got {value!r}")
    return key


def _normalize_attention_backend_and_bias(
    attention_backend: str,
    attention_bias: dict,
) -> tuple[str, dict]:
    backend = str(attention_backend).lower().replace("-", "_")
    if backend == "default":
        backend = "sdpa"
    bias = dict(attention_bias)
    alias_kind: str | None = None
    if backend in TRITON_BIAS_BACKEND_ALIASES:
        alias_kind = TRITON_BIAS_BACKEND_ALIASES[backend]
        backend = "triton"
    elif backend in TORCH_BIAS_BACKEND_ALIASES:
        alias_kind = TORCH_BIAS_BACKEND_ALIASES[backend]
        backend = "torch_reference"
    elif backend == "torch":
        backend = "torch_reference"
    if backend not in CANONICAL_ATTENTION_BACKENDS:
        raise ValueError(f"attention_backend must be one of {sorted(CANONICAL_ATTENTION_BACKENDS)}")
    raw_kind = bias.get("kind")
    bias_kind = _normalize_attention_bias_kind(raw_kind) if raw_kind is not None else None
    if alias_kind is not None:
        if bias_kind is not None and bias_kind != alias_kind:
            raise ValueError(
                f"legacy attention_backend alias requires attention_bias.kind={alias_kind!r}, got {raw_kind!r}"
            )
        bias_kind = alias_kind
    if bias_kind is not None:
        bias["kind"] = bias_kind
    elif "kind" in bias:
        bias.pop("kind")
    return backend, bias


class GroupLayerNorm(nn.Module):
    def __init__(self, group_order: int, channels_per_group: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.group_order = int(group_order)
        self.channels_per_group = int(channels_per_group)
        self.norm = nn.LayerNorm(self.channels_per_group, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], self.group_order, self.channels_per_group)
        x = self.norm(x)
        return x.reshape(shape)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x_float * rms).to(dtype=x.dtype)
        return x_norm * self.weight.to(device=x.device, dtype=x.dtype)


class PlatonicAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        solid_name: str = "tetrahedron",
        *,
        dropout: float = 0.0,
        rope_sigma: float = 0.5,
        learned_freqs: bool = True,
        freq_init: str = "spiral",
        use_key: bool = False,
        rope_on_values: bool = True,
        attention_backend: str = "sdpa",
        attention_bias: dict | None = None,
        linear_backend: str = "spatial",
        rope_cache: bool = True,
        constant_key_fastpath: bool = True,
        fused_qv: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        solid_name = solid_name.lower()
        group = PLATONIC_GROUPS[solid_name]
        self.group_order = group.G
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.d_model % self.group_order != 0:
            raise ValueError("d_model must be divisible by the group order")
        if self.num_heads % self.group_order != 0:
            raise ValueError("num_heads must be divisible by the group order")
        self.heads_per_frame = self.num_heads // self.group_order
        group_dim = self.d_model // self.group_order
        if group_dim % self.heads_per_frame != 0:
            raise ValueError("channels per group must be divisible by heads per frame")
        self.head_dim = group_dim // self.heads_per_frame
        if self.head_dim % 2 != 0:
            raise ValueError("Platonic attention head_dim must be even")
        self.dropout = float(dropout)
        self.use_key = bool(use_key)
        self.rope_on_values = bool(rope_on_values)
        self.rope_cache = bool(rope_cache)
        self.constant_key_fastpath = bool(constant_key_fastpath)
        self.fused_qv = bool(fused_qv)
        self.qk_norm = bool(qk_norm)
        if self.fused_qv and self.use_key:
            raise ValueError("fused_qv=True is only supported when use_key=False")
        self.attention_bias_config = dict(attention_bias or {})
        self.attention_backend, self.attention_bias_config = _normalize_attention_backend_and_bias(
            attention_backend,
            self.attention_bias_config,
        )
        if self.fused_qv:
            self.qv_proj = PlatonicLinear(self.d_model, 2 * self.d_model, solid=solid_name, linear_backend=linear_backend)
            self.q_proj = None
            self.v_proj = None
        else:
            self.qv_proj = None
            self.q_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name, linear_backend=linear_backend)
            self.v_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name, linear_backend=linear_backend)
        self.k_proj = (
            PlatonicLinear(self.d_model, self.d_model, solid=solid_name, linear_backend=linear_backend)
            if self.use_key
            else None
        )
        self.q_norm = RMSNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if self.qk_norm else nn.Identity()
        self.out_proj = PlatonicLinear(self.d_model, self.d_model, solid=solid_name, linear_backend=linear_backend)
        self.triton_block_m = int(self.attention_bias_config.get("block_m", 16))
        self.triton_block_n = int(self.attention_bias_config.get("block_n", 32))
        self.triton_strict = bool(self.attention_bias_config.get("strict", False))
        self.triton_precision = str(self.attention_bias_config.get("precision", "tf32"))
        self.rbf_weight: nn.Parameter | None
        self.rbf_gate: nn.Parameter | None
        self.rbf_type_bias: nn.Parameter | None
        self.radial_bias_kind: str | None
        bias_kind = self.attention_bias_config.get("kind")
        if bias_kind is not None:
            bias_kind = _normalize_attention_bias_kind(str(bias_kind))
            if self.attention_backend not in {"triton", "torch_reference"}:
                raise ValueError(
                    f"attention_bias.kind={bias_kind!r} requires attention_backend='triton' or 'torch_reference'"
                )
            if self.attention_backend == "torch_reference" and bias_kind not in {
                "rbf_type_enveloped",
                "radius_rbf_type_enveloped",
            }:
                raise ValueError(
                    "attention_backend='torch_reference' supports attention_bias.kind="
                    "{'rbf_type_enveloped', 'radius_rbf_type_enveloped'}"
                )
            self.radial_bias_kind = bias_kind
            if bias_kind in {"rbf_type_enveloped", "radius_rbf_type_enveloped"}:
                default_num_rbf = 4
            else:
                default_num_rbf = 8 if bias_kind == "radial_rbf" else 1
            num_rbf = int(self.attention_bias_config.get("num_rbf", default_num_rbf))
            if num_rbf <= 0:
                raise ValueError("attention_bias.num_rbf/num_basis must be positive")
            if bias_kind in {"radial_rbf", "rbf_type_enveloped", "radius_rbf_type_enveloped"}:
                rbf_min = float(self.attention_bias_config.get("rbf_min", 0.0))
                rbf_max = float(self.attention_bias_config.get("rbf_max", self.attention_bias_config.get("cutoff", 6.0)))
                centers = torch.linspace(rbf_min, rbf_max, num_rbf)
                delta = (rbf_max - rbf_min) / max(num_rbf - 1, 1)
                gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), dtype=torch.float32)
            else:
                if num_rbf != 1:
                    raise ValueError(f"{bias_kind} uses a single coefficient; set attention_bias.num_rbf=1 or omit it")
                centers = torch.zeros(1, dtype=torch.float32)
                gamma = torch.ones((), dtype=torch.float32)
            self.register_buffer("rbf_centers", centers, persistent=True)
            self.register_buffer("rbf_gamma", gamma, persistent=True)
            zero_init = bool(self.attention_bias_config.get("zero_init", True))
            self.rbf_weight = nn.Parameter(torch.zeros(self.heads_per_frame, num_rbf))
            if not zero_init:
                nn.init.normal_(self.rbf_weight, mean=0.0, std=1.0e-3)
            if bias_kind in {"rbf_type_enveloped", "radius_rbf_type_enveloped"}:
                self.rbf_gate = None
                self.local_cutoff = float(self.attention_bias_config.get("cutoff", 6.0))
                self.local_max_atomic_number = int(self.attention_bias_config.get("max_atomic_number", 118))
                if self.local_max_atomic_number < 0:
                    raise ValueError("attention_bias.max_atomic_number must be non-negative")
                if bool(self.attention_bias_config.get("include_type_pair", True)):
                    self.rbf_type_bias = nn.Parameter(
                        torch.zeros(self.local_max_atomic_number + 1, self.local_max_atomic_number + 1, self.heads_per_frame)
                    )
                    if not zero_init:
                        nn.init.normal_(self.rbf_type_bias, mean=0.0, std=1.0e-3)
                else:
                    self.register_parameter("rbf_type_bias", None)
                self.local_include_self = bool(self.attention_bias_config.get("include_self", True))
                self.local_envelope_in_score = bool(
                    self.attention_bias_config.get(
                        "envelope_in_score",
                        bias_kind == "radius_rbf_type_enveloped",
                    )
                )
            else:
                self.rbf_gate = nn.Parameter(
                    torch.full((self.heads_per_frame,), float(self.attention_bias_config.get("gate_init", 0.0)))
                )
                self.register_parameter("rbf_type_bias", None)
                self.local_cutoff = 0.0
                self.local_max_atomic_number = 0
                self.local_include_self = True
                self.local_envelope_in_score = False
            self.rbf_diag_zero = bool(self.attention_bias_config.get("diag_zero", True))
        else:
            self.radial_bias_kind = None
            self.rbf_weight = None
            self.rbf_gate = None
            self.rbf_type_bias = None
            self.register_buffer("rbf_centers", torch.empty(0), persistent=False)
            self.register_buffer("rbf_gamma", torch.ones((), dtype=torch.float32), persistent=False)
            self.rbf_diag_zero = True
            self.local_cutoff = 0.0
            self.local_max_atomic_number = 0
            self.local_include_self = True
            self.local_envelope_in_score = False
        self.rope = PlatonicRoPE(
            embed_dim=self.d_model,
            num_heads=self.heads_per_frame,
            solid_name=solid_name,
            head_dim=self.head_dim,
            freq_sigma=rope_sigma,
            learned_freqs=learned_freqs,
            freq_init=freq_init,
        )

    def radius_sparse_layout_config(self) -> dict | None:
        if self.radial_bias_kind != "radius_rbf_type_enveloped":
            return None
        return {
            "cutoff": float(self.local_cutoff),
            "block_m": int(self.triton_block_m),
            "block_n": int(self.triton_block_n),
            "sort": str(self.attention_bias_config.get("sort", "cell")),
            "include_self": bool(self.local_include_self),
        }

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.group_order, self.heads_per_frame, self.head_dim)

    def _project_qv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qv_proj is not None:
            qv = self.qv_proj(x).view(
                *x.shape[:-1],
                self.group_order,
                2,
                self.heads_per_frame,
                self.head_dim,
            )
            q, v = qv.unbind(dim=-3)
            return q, v
        if self.q_proj is None or self.v_proj is None:
            raise RuntimeError("Separate q/v projections are not configured")
        return self._split(self.q_proj(x)), self._split(self.v_proj(x))

    @torch.no_grad()
    def set_fused_qv_from_separate_(self, q_proj: PlatonicLinear, v_proj: PlatonicLinear) -> None:
        """Initialize a fused q/v projection from equivalent separate projections."""
        if self.qv_proj is None:
            raise RuntimeError("set_fused_qv_from_separate_ requires fused_qv=True")
        if q_proj.linear_backend != v_proj.linear_backend or q_proj.linear_backend != self.qv_proj.linear_backend:
            raise ValueError("q, v, and qv projections must use the same linear backend")
        if q_proj.in_channels != self.qv_proj.in_channels or v_proj.in_channels != self.qv_proj.in_channels:
            raise ValueError("q, v, and qv projections must have matching input channels")
        if q_proj.out_channels + v_proj.out_channels != self.qv_proj.out_channels:
            raise ValueError("qv output channels must equal q output channels plus v output channels")

        if self.qv_proj.linear_backend == "fourier_direct":
            self.qv_proj.w1.copy_(torch.cat([q_proj.w1, v_proj.w1], dim=0))
            self.qv_proj.w2_re.copy_(torch.cat([q_proj.w2_re, v_proj.w2_re], dim=0))
            self.qv_proj.w2_im.copy_(torch.cat([q_proj.w2_im, v_proj.w2_im], dim=0))
            q_w3 = q_proj.w3.view(3, q_proj.out_channels, 3 * q_proj.in_channels)
            v_w3 = v_proj.w3.view(3, v_proj.out_channels, 3 * v_proj.in_channels)
            qv_w3 = torch.cat([q_w3, v_w3], dim=1).reshape(3 * self.qv_proj.out_channels, 3 * self.qv_proj.in_channels)
            self.qv_proj.w3.copy_(qv_w3)
        else:
            if q_proj.kernel is None or v_proj.kernel is None or self.qv_proj.kernel is None:
                raise RuntimeError("Spatial-kernel projections are not configured")
            self.qv_proj.kernel.copy_(torch.cat([q_proj.kernel, v_proj.kernel], dim=1))

        if self.qv_proj.bias is not None:
            if q_proj.bias is None or v_proj.bias is None:
                raise ValueError("Cannot initialize fused qv bias from bias-free q/v projections")
            self.qv_proj.bias.copy_(torch.cat([q_proj.bias, v_proj.bias], dim=0))

    def _project_qkv_with_rope(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        q, v = self._project_qv(x)
        q = self.q_norm(q)
        if self.k_proj is not None:
            k_unrotated = self.k_norm(self._split(self.k_proj(x)))
        elif self.qk_norm:
            k_unrotated = self.k_norm(torch.ones_like(q))
        else:
            k_unrotated = None
        if not self.rope_cache:
            k = k_unrotated if k_unrotated is not None else torch.ones_like(q)
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            if self.rope_on_values:
                v = self.rope(v, pos)
            return q, k, v, None

        cos, sin = self.rope.cos_sin(pos, dtype=q.dtype, device=q.device)
        q = self.rope.apply_from_cos_sin(q, cos, sin)
        if k_unrotated is not None:
            k = self.rope.apply_from_cos_sin(k_unrotated, cos, sin)
        elif self.constant_key_fastpath:
            k = self.rope.constant_key_from_cos_sin(cos, sin)
        else:
            k = self.rope.apply_from_cos_sin(torch.ones_like(q), cos, sin)
        if self.rope_on_values:
            v = self.rope.apply_from_cos_sin(v, cos, sin)
        return q, k, v, (cos, sin)

    def _inverse_value_rope(
        self,
        out: torch.Tensor,
        pos: torch.Tensor,
        rope_factors: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.rope_on_values:
            return out
        if rope_factors is None:
            return self.rope(out, pos, inverse=True)
        cos, sin = rope_factors
        return self.rope.apply_from_cos_sin(out, cos, sin, inverse=True)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_mask = None
        if pad_mask is not None:
            attn_mask = torch.zeros_like(pad_mask, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(pad_mask, torch.finfo(q.dtype).min)[:, None, None, :]
        if q.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
                enable_cudnn=False,
            ):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if flash_attn_varlen_func is None or not q.is_cuda:
            return self._sdpa_attention(q, k, v, pad_mask)

        batch_size, _, num_tokens, _ = q.shape
        orig_dtype = q.dtype
        q_bnhd = q.transpose(1, 2).contiguous()
        k_bnhd = k.transpose(1, 2).contiguous()
        v_bnhd = v.transpose(1, 2).contiguous()

        if pad_mask is None:
            counts = torch.full(
                (batch_size,),
                num_tokens,
                dtype=torch.int32,
                device=q.device,
            )
            q_unpad = q_bnhd.reshape(batch_size * num_tokens, self.num_heads, self.head_dim)
            k_unpad = k_bnhd.reshape(batch_size * num_tokens, self.num_heads, self.head_dim)
            v_unpad = v_bnhd.reshape(batch_size * num_tokens, self.num_heads, self.head_dim)
            valid = None
        else:
            valid = ~pad_mask
            counts = valid.sum(dim=1, dtype=torch.int32)
            q_unpad = q_bnhd[valid]
            k_unpad = k_bnhd[valid]
            v_unpad = v_bnhd[valid]

        max_seqlen = int(counts.max().item()) if counts.numel() > 0 else 0
        if max_seqlen == 0:
            return torch.zeros_like(q)

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
        cu_seqlens[1:] = torch.cumsum(counts, dim=0)
        out_unpad = flash_attn_varlen_func(
            q_unpad.to(torch.bfloat16),
            k_unpad.to(torch.bfloat16),
            v_unpad.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        ).to(orig_dtype)

        if valid is None:
            out_bnhd = out_unpad.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        else:
            out_bnhd = torch.zeros_like(q_bnhd)
            out_bnhd[valid] = out_unpad
        return out_bnhd.transpose(1, 2).contiguous()

    def _sdpa_attention_flat(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        starts = cu_seqlens[:-1].detach().cpu().tolist()
        ends = cu_seqlens[1:].detach().cpu().tolist()
        for start, end in zip(starts, ends):
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i:
                continue
            out = F.scaled_dot_product_attention(
                q[start_i:end_i].transpose(0, 1).unsqueeze(0),
                k[start_i:end_i].transpose(0, 1).unsqueeze(0),
                v[start_i:end_i].transpose(0, 1).unsqueeze(0),
                dropout_p=self.dropout if self.training else 0.0,
            )
            outputs.append(out.squeeze(0).transpose(0, 1).contiguous())
        if not outputs:
            return torch.zeros_like(q)
        return torch.cat(outputs, dim=0)

    def _flash_attention_flat(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if int(max_seqlen) == 0 or q.shape[0] == 0:
            return torch.zeros_like(q)
        if flash_attn_varlen_func is None or not q.is_cuda:
            return self._sdpa_attention_flat(q, k, v, cu_seqlens)
        orig_dtype = q.dtype
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)
        return flash_attn_varlen_func(
            q.contiguous().to(torch.bfloat16),
            k.contiguous().to(torch.bfloat16),
            v.contiguous().to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=int(max_seqlen),
            max_seqlen_k=int(max_seqlen),
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        ).to(orig_dtype)

    def _triton_attention_flat(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pos: torch.Tensor,
        atom_types: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        radius_layout: RadiusBlockSparseLayout | None = None,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if dropout_p != 0.0:
            raise ValueError("Platonic Triton flat attention currently requires dropout=0")
        if self.radial_bias_kind is not None:
            if self.rbf_weight is None:
                raise RuntimeError(f"{self.attention_backend} backend was initialized without radial parameters")
            if self.radial_bias_kind in {"rbf_type_enveloped", "radius_rbf_type_enveloped"}:
                if self.rbf_type_bias is not None and atom_types is None:
                    raise ValueError(
                        f"attention_backend='triton' with attention_bias.kind={self.radial_bias_kind!r} "
                        "requires atom_types in forward_flat"
                    )
                if self.radial_bias_kind == "radius_rbf_type_enveloped":
                    if radius_layout is None:
                        raise ValueError("radius sparse Triton attention requires radius_layout")
                    return platonic_radius_sparse_attention_flat_triton(
                        q,
                        k,
                        v,
                        pos=pos,
                        atom_types=atom_types,
                        heads_per_frame=self.heads_per_frame,
                        rbf_weight=self.rbf_weight,
                        type_bias=self.rbf_type_bias,
                        centers=self.rbf_centers,
                        gamma=self.rbf_gamma,
                        cutoff=self.local_cutoff,
                        max_atomic_number=self.local_max_atomic_number,
                        diag_zero=self.rbf_diag_zero,
                        include_self=self.local_include_self,
                        envelope_in_score=self.local_envelope_in_score,
                        radius_layout=radius_layout,
                        precision=self.triton_precision,
                        strict=self.triton_strict,
                    )
                return platonic_attention_flat_triton(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=int(max_seqlen),
                    pos=pos,
                    atom_types=atom_types,
                    heads_per_frame=self.heads_per_frame,
                    rbf_weight=self.rbf_weight,
                    type_bias=self.rbf_type_bias,
                    centers=self.rbf_centers,
                    gamma=self.rbf_gamma,
                    cutoff=self.local_cutoff,
                    max_atomic_number=self.local_max_atomic_number,
                    diag_zero=self.rbf_diag_zero,
                    radial_bias_kind=self.radial_bias_kind,
                    precision=self.triton_precision,
                    block_m=self.triton_block_m,
                    block_n=self.triton_block_n,
                    strict=self.triton_strict,
                )
            if self.rbf_gate is None:
                raise RuntimeError(f"{self.attention_backend} backend was initialized without radial gate parameters")
            return platonic_attention_flat_triton(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=int(max_seqlen),
                pos=pos,
                heads_per_frame=self.heads_per_frame,
                rbf_weight=self.rbf_weight,
                gate=self.rbf_gate,
                centers=self.rbf_centers,
                gamma=self.rbf_gamma,
                diag_zero=self.rbf_diag_zero,
                radial_bias_kind=self.radial_bias_kind,
                precision=self.triton_precision,
                block_m=self.triton_block_m,
                block_n=self.triton_block_n,
                strict=self.triton_strict,
            )
        return platonic_attention_flat_triton(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
            precision=self.triton_precision,
            block_m=self.triton_block_m,
            block_n=self.triton_block_n,
            strict=self.triton_strict,
        )

    def _torch_local_bias_attention_flat(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pos: torch.Tensor,
        atom_types: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        radius_layout: RadiusBlockSparseLayout | None = None,
    ) -> torch.Tensor:
        if self.rbf_weight is None:
            raise RuntimeError(f"{self.attention_backend} backend was initialized without local bias parameters")
        if self.rbf_type_bias is not None and atom_types is None:
            raise ValueError(f"attention_backend={self.attention_backend!r} requires atom_types in forward_flat")
        if self.radial_bias_kind == "radius_rbf_type_enveloped" and radius_layout is not None:
            return platonic_radius_block_sparse_attention_torch_reference(
                q,
                k,
                v,
                pos=pos,
                atom_types=atom_types,
                heads_per_frame=self.heads_per_frame,
                rbf_weight=self.rbf_weight,
                type_bias=self.rbf_type_bias,
                centers=self.rbf_centers,
                gamma=self.rbf_gamma,
                cutoff=self.local_cutoff,
                diag_zero=self.rbf_diag_zero,
                include_self=self.local_include_self,
                envelope_in_score=self.local_envelope_in_score,
                radius_layout=radius_layout,
            )
        return platonic_attention_flat_torch_reference(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
            pos=pos,
            atom_types=atom_types,
            heads_per_frame=self.heads_per_frame,
            rbf_weight=self.rbf_weight,
            type_bias=self.rbf_type_bias,
            centers=self.rbf_centers,
            gamma=self.rbf_gamma,
            cutoff=self.local_cutoff,
            radial_bias_kind=self.radial_bias_kind,
            diag_zero=self.rbf_diag_zero,
            include_self=self.local_include_self,
            envelope_in_score=self.local_envelope_in_score,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        atom_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v, rope_factors = self._project_qkv_with_rope(x, pos)
        batch_size, num_tokens = x.shape[:2]
        q = q.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        if self.attention_backend in {"triton", "torch_reference"}:
            raise ValueError(
                "Platonic Triton/local-bias attention is implemented for forward_flat only; "
                "use omol_runtime_mode='internal_flat_tetra' or switch attention_backend to 'flash'/'sdpa'."
            )
        if self.attention_backend == "flash":
            out = self._flash_attention(q, k, v, pad_mask)
        else:
            out = self._sdpa_attention(q, k, v, pad_mask)
        out = out.transpose(1, 2).contiguous().view(
            batch_size,
            num_tokens,
            self.group_order,
            self.heads_per_frame,
            self.head_dim,
        )
        out = self._inverse_value_rope(out, pos, rope_factors)
        return self.out_proj(out.reshape(batch_size, num_tokens, self.d_model))

    def forward_flat(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        atom_types: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        radius_layout: RadiusBlockSparseLayout | None = None,
    ) -> torch.Tensor:
        q, k, v, rope_factors = self._project_qkv_with_rope(x, pos)
        num_tokens = x.shape[0]
        q = q.reshape(num_tokens, self.num_heads, self.head_dim)
        k = k.reshape(num_tokens, self.num_heads, self.head_dim)
        v = v.reshape(num_tokens, self.num_heads, self.head_dim)
        if self.attention_backend == "flash":
            out = self._flash_attention_flat(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))
        elif self.attention_backend == "triton":
            out = self._triton_attention_flat(
                q,
                k,
                v,
                pos=pos,
                atom_types=atom_types,
                cu_seqlens=cu_seqlens,
                max_seqlen=int(max_seqlen),
                radius_layout=radius_layout,
            )
        elif self.attention_backend == "torch_reference":
            out = self._torch_local_bias_attention_flat(
                q,
                k,
                v,
                pos=pos,
                atom_types=atom_types,
                cu_seqlens=cu_seqlens,
                max_seqlen=int(max_seqlen),
                radius_layout=radius_layout,
            )
        else:
            out = self._sdpa_attention_flat(q, k, v, cu_seqlens)
        out = out.contiguous().view(num_tokens, self.group_order, self.heads_per_frame, self.head_dim)
        out = self._inverse_value_rope(out, pos, rope_factors)
        return self.out_proj(out.reshape(num_tokens, self.d_model))


class PlatonicBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        solid_name: str = "tetrahedron",
        *,
        dropout: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        layer_norm_eps: float = 1e-6,
        rope_sigma: float = 0.5,
        learned_freqs: bool = True,
        freq_init: str = "spiral",
        use_key: bool = False,
        rope_on_values: bool = True,
        attention_backend: str = "sdpa",
        attention_bias: dict | None = None,
        layer_scale_init_value: float | None = None,
        linear_backend: str = "spatial",
        attention_linear_backend: str | None = None,
        ffn_linear_backend: str | None = None,
        rope_cache: bool = True,
        constant_key_fastpath: bool = True,
        fused_qv: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        solid_name = solid_name.lower()
        group = PLATONIC_GROUPS[solid_name]
        if d_model % group.G != 0:
            raise ValueError("d_model must be divisible by group order")
        self.group_order = group.G
        self.channels_per_group = d_model // group.G
        self.norm1 = GroupLayerNorm(group.G, self.channels_per_group, eps=layer_norm_eps)
        self.norm2 = GroupLayerNorm(group.G, self.channels_per_group, eps=layer_norm_eps)
        attn_linear_backend = str(attention_linear_backend or linear_backend)
        mlp_linear_backend = str(ffn_linear_backend or linear_backend)
        self.attn = PlatonicAttention(
            d_model,
            nhead,
            solid_name,
            dropout=dropout,
            rope_sigma=rope_sigma,
            learned_freqs=learned_freqs,
            freq_init=freq_init,
            use_key=use_key,
            rope_on_values=rope_on_values,
            attention_backend=attention_backend,
            attention_bias=attention_bias,
            linear_backend=attn_linear_backend,
            rope_cache=rope_cache,
            constant_key_fastpath=constant_key_fastpath,
            fused_qv=fused_qv,
            qk_norm=qk_norm,
        )
        self.linear1 = PlatonicLinear(d_model, dim_feedforward, solid=solid_name, linear_backend=mlp_linear_backend)
        self.linear2 = PlatonicLinear(dim_feedforward, d_model, solid=solid_name, linear_backend=mlp_linear_backend)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        if layer_scale_init_value is None:
            self.gamma_1 = None
            self.gamma_2 = None
        else:
            init = float(layer_scale_init_value)
            self.gamma_1 = nn.Parameter(init * torch.ones(self.channels_per_group))
            self.gamma_2 = nn.Parameter(init * torch.ones(self.channels_per_group))

    def radius_sparse_layout_config(self) -> dict | None:
        return self.attn.radius_sparse_layout_config()

    def _apply_layer_scale(self, x: torch.Tensor, gamma: torch.Tensor | None) -> torch.Tensor:
        if gamma is None:
            return x
        shape = x.shape
        x = x.view(*shape[:-1], self.group_order, self.channels_per_group)
        x = x * gamma.to(device=x.device, dtype=x.dtype)
        return x.reshape(shape)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        atom_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.dropout(self.attn(self.norm1(x), pos=pos, pad_mask=pad_mask, atom_types=atom_types))
        x = x + self._apply_layer_scale(attn_out, self.gamma_1)
        ffn_out = self.dropout(self.linear2(self.activation(self.linear1(self.norm2(x)))))
        x = x + self._apply_layer_scale(ffn_out, self.gamma_2)
        return x

    def forward_flat(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        atom_types: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        radius_layout: RadiusBlockSparseLayout | None = None,
    ) -> torch.Tensor:
        attn_out = self.dropout(
            self.attn.forward_flat(
                self.norm1(x),
                pos=pos,
                atom_types=atom_types,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                radius_layout=radius_layout,
            )
        )
        x = x + self._apply_layer_scale(attn_out, self.gamma_1)
        ffn_out = self.dropout(self.linear2(self.activation(self.linear1(self.norm2(x)))))
        x = x + self._apply_layer_scale(ffn_out, self.gamma_2)
        return x
