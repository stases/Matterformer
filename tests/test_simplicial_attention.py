import pytest
import torch

import matterformer.models.attention as attention_module
from matterformer.geometry import NonPeriodicGeometryAdapter
from matterformer.models import QM9EDMModel, SimplicialGeometryBias, TwoSimplicialAttention
from matterformer.models.attention import (
    SimplicialAttentionMask,
    SimplicialFactorizedBias,
    SimplicialLowRankAngleResidual,
    SimplicialLowRankMessageResidual,
    _TritonTwoSimplicialAttentionFunction,
    simplicial_attention_torch_from_projected,
)
from matterformer.models.attention_triton import TRITON_AVAILABLE


def _reference_from_projected(
    q: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    k2: torch.Tensor,
    v2: torch.Tensor,
    *,
    attention_mask: SimplicialAttentionMask,
    factorized_bias: SimplicialFactorizedBias | None = None,
    angle_residual: SimplicialLowRankAngleResidual | None = None,
    message_residual: SimplicialLowRankMessageResidual | None = None,
    message_basis: torch.Tensor | None = None,
    logit_bias_fn=None,
    dropout_p: float = 0.0,
    training: bool = False,
    chunk_size: int = 128,
):
    batch_size, num_heads, num_tokens, head_dim = q.shape
    query_valid = attention_mask.query_valid.bool()
    pair_valid = attention_mask.pair_mask()

    out = torch.empty((batch_size, num_heads, num_tokens, head_dim), device=q.device, dtype=v1.dtype)
    pair_mask = pair_valid[:, None, None, :, :]
    flat_mask = pair_mask.flatten(-2)
    for start in range(0, num_tokens, chunk_size):
        end = min(num_tokens, start + chunk_size)
        q_chunk = q[:, :, start:end, :]
        qk2 = q_chunk.unsqueeze(-2) * k2.unsqueeze(-3)
        logits = torch.matmul(k1.unsqueeze(-3), qk2.transpose(-1, -2)).float()
        if factorized_bias is not None:
            logits = logits + factorized_bias.chunk(start, end, dtype=logits.dtype, device=logits.device)
        if angle_residual is not None:
            logits = logits + angle_residual.chunk(start, end, dtype=logits.dtype, device=logits.device)
        if logit_bias_fn is not None:
            logits = logits + logit_bias_fn(start, end, logits.dtype, logits.device)

        flat_logits = logits.flatten(-2).masked_fill(
            ~flat_mask,
            torch.finfo(logits.dtype).min,
        )
        attn = torch.softmax(flat_logits, dim=-1)
        attn = torch.where(flat_mask, attn, torch.zeros_like(attn)).view_as(logits)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p, training=training)

        attn_float = attn.float()
        attn_value = attn.to(v1.dtype)
        tmp = torch.matmul(attn_value.transpose(-2, -1), v1.unsqueeze(-3))
        out_chunk = (tmp * v2.unsqueeze(-3)).sum(dim=-2)
        if message_residual is not None and message_basis is not None:
            message_coeff = torch.einsum(
                "bhqjk,bhqjr,bhqkr->bhqr",
                attn_float,
                message_residual.left[:, :, start:end, :, :].float(),
                message_residual.right[:, :, start:end, :, :].float(),
            ) * (message_residual.rank**-0.5)
            out_chunk = out_chunk + torch.einsum(
                "bhqr,hrd->bhqd",
                message_coeff,
                message_basis.float(),
            ).to(dtype=out_chunk.dtype)
        out_chunk = out_chunk * query_valid[:, start:end].to(dtype=out_chunk.dtype)[:, None, :, None]
        out[:, :, start:end, :] = out_chunk
    return out


def _make_factorized_bias(
    batch_size: int,
    num_heads: int,
    num_tokens: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> SimplicialFactorizedBias:
    factory = lambda: torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        num_tokens,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    gate = torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return SimplicialFactorizedBias(u=factory(), v=factory(), w=factory(), gate=gate)


def _make_low_rank_angle_residual(
    batch_size: int,
    num_heads: int,
    num_tokens: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> SimplicialLowRankAngleResidual:
    factory = lambda: torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        num_tokens,
        rank,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    gate = torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return SimplicialLowRankAngleResidual(left=factory(), right=factory(), gate=gate)


def _make_low_rank_message_residual(
    batch_size: int,
    num_heads: int,
    num_tokens: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> SimplicialLowRankMessageResidual:
    factory = lambda: torch.randn(
        batch_size,
        num_heads,
        num_tokens,
        num_tokens,
        rank,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return SimplicialLowRankMessageResidual(left=factory(), right=factory())


def _mask_from_padding(
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> SimplicialAttentionMask:
    return SimplicialAttentionMask.from_key_padding_mask(
        key_padding_mask,
        batch_size=batch_size,
        num_tokens=num_tokens,
        device=device,
    )


def test_torch_backend_matches_reference_without_bias():
    torch.manual_seed(0)
    attn = TwoSimplicialAttention(dim=32, num_heads=4, impl="torch", chunk_size=2)
    x = torch.randn(2, 5, 32)
    key_padding_mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, True, True, True],
        ]
    )
    q, k1, v1, k2, v2 = attn._project_inputs(x)
    attention_mask = _mask_from_padding(
        key_padding_mask,
        batch_size=x.shape[0],
        num_tokens=x.shape[1],
        device=x.device,
    )
    expected = _reference_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        chunk_size=attn.chunk_size,
    )
    actual = simplicial_attention_torch_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        chunk_size=attn.chunk_size,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_torch_backend_matches_reference_with_factorized_bias():
    torch.manual_seed(0)
    attn = TwoSimplicialAttention(dim=32, num_heads=4, impl="torch", chunk_size=3)
    x = torch.randn(2, 5, 32)
    factorized_bias = _make_factorized_bias(
        2,
        4,
        5,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = _mask_from_padding(
        None,
        batch_size=x.shape[0],
        num_tokens=x.shape[1],
        device=x.device,
    )
    q, k1, v1, k2, v2 = attn._project_inputs(x)
    expected = _reference_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        factorized_bias=factorized_bias,
        chunk_size=attn.chunk_size,
    )
    actual = simplicial_attention_torch_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        factorized_bias=factorized_bias,
        chunk_size=attn.chunk_size,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_factorized_bias_chunk_sums_in_fp32():
    u = torch.full((1, 1, 2, 2), 1.0, dtype=torch.bfloat16)
    v = torch.full((1, 1, 2, 2), 0.00390625, dtype=torch.bfloat16)
    w = torch.full((1, 1, 2, 2), 0.00390625, dtype=torch.bfloat16)
    gate = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    bias = SimplicialFactorizedBias(u=u, v=v, w=w, gate=gate)

    chunk = bias.chunk(0, 2, dtype=torch.float32, device=u.device)
    native_bf16_sum = (u[:, :, :, :, None] + v[:, :, :, None, :] + w[:, :, None, :, :]).float()
    fp32_sum = u.float()[:, :, :, :, None] + v.float()[:, :, :, None, :] + w.float()[:, :, None, :, :]

    assert torch.allclose(chunk, fp32_sum)
    assert not torch.allclose(chunk, native_bf16_sum)


def test_torch_backend_matches_reference_with_low_rank_angle_residual():
    torch.manual_seed(0)
    attn = TwoSimplicialAttention(dim=32, num_heads=4, impl="torch", chunk_size=3)
    x = torch.randn(2, 5, 32)
    angle_residual = _make_low_rank_angle_residual(
        2,
        4,
        5,
        3,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = _mask_from_padding(
        None,
        batch_size=x.shape[0],
        num_tokens=x.shape[1],
        device=x.device,
    )
    q, k1, v1, k2, v2 = attn._project_inputs(x)
    expected = _reference_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        angle_residual=angle_residual,
        chunk_size=attn.chunk_size,
    )
    actual = simplicial_attention_torch_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        angle_residual=angle_residual,
        chunk_size=attn.chunk_size,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_torch_backend_matches_reference_with_low_rank_message_residual():
    torch.manual_seed(0)
    attn = TwoSimplicialAttention(
        dim=32,
        num_heads=4,
        impl="torch",
        chunk_size=3,
        message_mode="low_rank",
        message_rank=3,
    )
    x = torch.randn(2, 5, 32)
    message_residual = _make_low_rank_message_residual(
        2,
        4,
        5,
        3,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = _mask_from_padding(
        None,
        batch_size=x.shape[0],
        num_tokens=x.shape[1],
        device=x.device,
    )
    q, k1, v1, k2, v2 = attn._project_inputs(x)
    expected = _reference_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        message_residual=message_residual,
        message_basis=attn.message_basis,
        chunk_size=attn.chunk_size,
    )
    actual = simplicial_attention_torch_from_projected(
        q,
        k1,
        v1,
        k2,
        v2,
        attention_mask=attention_mask,
        message_residual=message_residual,
        message_basis=attn.message_basis,
        chunk_size=attn.chunk_size,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_message_mode_none_creates_no_message_basis():
    attn = TwoSimplicialAttention(dim=32, num_heads=4, message_mode="none")
    assert attn.message_basis is None
    assert "message_basis" not in dict(attn.named_parameters())


def test_pair_key_mask_excludes_non_atom_pair_members():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 3, 4)
    k1 = torch.randn(1, 1, 3, 4)
    v1 = torch.randn(1, 1, 3, 4)
    k2 = torch.randn(1, 1, 3, 4)
    v2 = torch.randn(1, 1, 3, 4)
    attention_mask = SimplicialAttentionMask(
        query_valid=torch.tensor([[True, True, True]]),
        pair_key_valid=torch.tensor([[True, True, False]]),
    )
    out = simplicial_attention_torch_from_projected(q, k1, v1, k2, v2, attention_mask=attention_mask, chunk_size=3)

    k1_alt = k1.clone()
    v1_alt = v1.clone()
    k2_alt = k2.clone()
    v2_alt = v2.clone()
    k1_alt[:, :, 2, :] = 1_000.0
    v1_alt[:, :, 2, :] = -1_000.0
    k2_alt[:, :, 2, :] = 2_000.0
    v2_alt[:, :, 2, :] = 3_000.0
    out_alt = simplicial_attention_torch_from_projected(
        q,
        k1_alt,
        v1_alt,
        k2_alt,
        v2_alt,
        attention_mask=attention_mask,
        chunk_size=3,
    )
    assert torch.allclose(out, out_alt, atol=1e-6, rtol=1e-5)


def test_pair_mask_excludes_invalid_pairs_with_negative_infinity():
    q = torch.zeros(1, 1, 3, 2)
    k1 = torch.zeros(1, 1, 3, 2)
    v1 = torch.tensor([[[[1.0, 2.0], [5.0, 7.0], [11.0, 13.0]]]])
    k2 = torch.zeros(1, 1, 3, 2)
    v2 = torch.tensor([[[[3.0, 5.0], [17.0, 19.0], [23.0, 29.0]]]])
    pair_valid = torch.zeros(1, 3, 3, dtype=torch.bool)
    pair_valid[:, 0, 0] = True
    attention_mask = SimplicialAttentionMask(
        query_valid=torch.tensor([[True, True, True]]),
        pair_key_valid=torch.tensor([[True, True, True]]),
        pair_valid=pair_valid,
    )
    out = simplicial_attention_torch_from_projected(q, k1, v1, k2, v2, attention_mask=attention_mask, chunk_size=3)
    expected = torch.tensor([3.0, 10.0]).view(1, 1, 1, 2).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_auto_dispatch_falls_back_to_torch_for_cpu_and_unsupported_modes():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32)
    key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
        ]
    )

    auto_attn = TwoSimplicialAttention(dim=32, num_heads=4, impl="auto")
    _ = auto_attn(x, key_padding_mask=key_padding_mask)
    assert auto_attn._last_impl_used == "torch"

    auto_attn_dropout = TwoSimplicialAttention(dim=32, num_heads=4, impl="auto", dropout=0.1)
    auto_attn_dropout.train()
    _ = auto_attn_dropout(x, key_padding_mask=key_padding_mask)
    assert auto_attn_dropout._last_impl_used == "torch"

    auto_attn_with_residual = TwoSimplicialAttention(dim=32, num_heads=4, impl="auto")
    residual = lambda start, end, dtype, device: torch.zeros(
        x.shape[0],
        auto_attn_with_residual.num_heads,
        end - start,
        x.shape[1],
        x.shape[1],
        dtype=dtype,
        device=device,
    )
    _ = auto_attn_with_residual(x, key_padding_mask=key_padding_mask, logit_bias_fn=residual)
    assert auto_attn_with_residual._last_impl_used == "torch"

    _, _ = auto_attn(x, key_padding_mask=key_padding_mask, return_attn=True)
    assert auto_attn._last_impl_used == "torch"

    explicit_triton = TwoSimplicialAttention(dim=32, num_heads=4, impl="triton")
    with pytest.raises(RuntimeError):
        explicit_triton(x, key_padding_mask=key_padding_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton dtype tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_explicit_triton_rejects_unsupported_dtype():
    device = torch.device("cuda")
    attn = TwoSimplicialAttention(dim=32, num_heads=4, impl="triton").to(device=device, dtype=torch.float32)
    x = torch.randn(2, 4, 32, device=device, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="only supports float16/bfloat16/float32"):
        attn(x)


def test_factorized_bias_builder_zeros_cls_queries_and_builds_mask():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    coords = torch.randn(2, 4, 3)
    pad_mask = torch.tensor(
        [
            [False, False, False, True, False],
            [False, False, True, True, False],
        ]
    )
    geom_features = adapter(coords, pad_mask=pad_mask[:, : coords.shape[1]])
    simplicial_bias = SimplicialGeometryBias(n_heads=4, mode="factorized", use_noise_gate=False)
    factorized_bias, residual_fn, attention_mask = simplicial_bias.build_bias_inputs(
        geom_features=geom_features,
        coords_len=coords.shape[1],
        pad_mask=pad_mask,
        seq_len=pad_mask.shape[1],
        sigma=None,
    )
    assert residual_fn is None
    assert factorized_bias.gate.shape == (2, 4, pad_mask.shape[1])
    assert torch.count_nonzero(factorized_bias.gate[:, :, coords.shape[1] :]) == 0
    assert attention_mask.query_valid.shape == pad_mask.shape
    assert attention_mask.pair_key_valid.shape == pad_mask.shape
    assert attention_mask.pair_valid is not None
    assert torch.count_nonzero(attention_mask.pair_key_valid[:, coords.shape[1] :]) == 0


def test_low_rank_angle_bias_builder_returns_structured_residual():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    coords = torch.randn(2, 4, 3)
    pad_mask = torch.tensor(
        [
            [False, False, False, True, False],
            [False, False, True, True, False],
        ]
    )
    geom_features = adapter(coords, pad_mask=pad_mask[:, : coords.shape[1]])
    simplicial_bias = SimplicialGeometryBias(
        n_heads=4,
        mode="angle_low_rank",
        angle_residual_rank=3,
        use_noise_gate=False,
    )
    factorized_bias, residual_fn, angle_residual, message_residual, attention_mask = simplicial_bias.build_structured_bias_inputs(
        geom_features=geom_features,
        coords_len=coords.shape[1],
        pad_mask=pad_mask,
        seq_len=pad_mask.shape[1],
        sigma=None,
    )
    assert residual_fn is None
    assert angle_residual is not None
    assert message_residual is None
    assert angle_residual.left.shape == (2, 4, pad_mask.shape[1], pad_mask.shape[1], 3)
    assert angle_residual.right.shape == angle_residual.left.shape
    assert angle_residual.gate.shape == factorized_bias.gate.shape
    assert torch.count_nonzero(angle_residual.left) == 0
    assert torch.count_nonzero(angle_residual.right) > 0
    assert torch.count_nonzero(angle_residual.gate[:, :, coords.shape[1] :]) == 0
    assert attention_mask.pair_valid is not None


def test_low_rank_message_builder_returns_structured_residual():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    coords = torch.randn(2, 4, 3)
    pad_mask = torch.tensor(
        [
            [False, False, False, True, False],
            [False, False, True, True, False],
        ]
    )
    geom_features = adapter(coords, pad_mask=pad_mask[:, : coords.shape[1]])
    simplicial_bias = SimplicialGeometryBias(
        n_heads=4,
        mode="factorized",
        message_mode="low_rank",
        message_rank=3,
        use_noise_gate=False,
    )
    factorized_bias, residual_fn, angle_residual, message_residual, attention_mask = (
        simplicial_bias.build_structured_bias_inputs(
            geom_features=geom_features,
            coords_len=coords.shape[1],
            pad_mask=pad_mask,
            seq_len=pad_mask.shape[1],
            sigma=None,
        )
    )
    assert factorized_bias is not None
    assert residual_fn is None
    assert angle_residual is None
    assert message_residual is not None
    assert message_residual.left.shape == (2, 4, pad_mask.shape[1], pad_mask.shape[1], 3)
    assert message_residual.right.shape == message_residual.left.shape
    assert torch.count_nonzero(message_residual.left) == 0
    assert torch.count_nonzero(message_residual.right) > 0
    assert torch.count_nonzero(message_residual.left[:, :, coords.shape[1] :, :, :]) == 0
    assert attention_mask.pair_valid is not None


def test_low_rank_message_builder_can_disable_logit_geometry_params():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    coords = torch.randn(1, 4, 3)
    geom_features = adapter(coords)
    simplicial_bias = SimplicialGeometryBias(
        n_heads=2,
        mode="none",
        message_mode="low_rank",
        message_rank=3,
        use_noise_gate=False,
    )
    assert not any(
        name.startswith(("spoke_bias_", "pair_bias_w_"))
        for name, _ in simplicial_bias.named_parameters()
    )
    factorized_bias, residual_fn, angle_residual, message_residual, attention_mask = (
        simplicial_bias.build_structured_bias_inputs(
            geom_features=geom_features,
            coords_len=coords.shape[1],
            pad_mask=None,
        )
    )
    assert factorized_bias is None
    assert residual_fn is None
    assert angle_residual is None
    assert message_residual is not None
    assert message_residual.left.shape == (1, 2, coords.shape[1], coords.shape[1], 3)
    assert attention_mask.pair_valid is not None


def test_checkpoint_roundtrip_preserves_simplicial_impl_and_precision():
    torch.manual_seed(0)
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="simplicial",
        simplicial_geom_mode="factorized",
        simplicial_impl="torch",
        simplicial_precision="tf32",
    )
    checkpoint = {
        "args": {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "attn_type": "simplicial",
            "simplicial_geom_mode": "factorized",
            "simplicial_impl": "torch",
            "simplicial_precision": "tf32",
            "disable_geometry_bias": False,
        },
        "model_state": model.state_dict(),
    }
    reloaded = QM9EDMModel(
        d_model=int(checkpoint["args"]["d_model"]),
        n_heads=int(checkpoint["args"]["n_heads"]),
        n_layers=int(checkpoint["args"]["n_layers"]),
        mlp_ratio=float(checkpoint["args"]["mlp_ratio"]),
        dropout=float(checkpoint["args"]["dropout"]),
        attn_dropout=float(checkpoint["args"]["attn_dropout"]),
        attn_type=str(checkpoint["args"]["attn_type"]),
        simplicial_geom_mode=str(checkpoint["args"]["simplicial_geom_mode"]),
        simplicial_impl=str(checkpoint["args"]["simplicial_impl"]),
        simplicial_precision=str(checkpoint["args"]["simplicial_precision"]),
        use_geometry_bias=not bool(checkpoint["args"]["disable_geometry_bias"]),
    )
    reloaded.load_state_dict(checkpoint["model_state"])
    attn = reloaded.trunk.blocks[0].attn
    assert attn.impl == "torch"
    assert attn.precision == "tf32"


def test_legacy_checkpoint_defaults_to_ieee_fp32():
    checkpoint = {
        "args": {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "attn_type": "simplicial",
            "simplicial_geom_mode": "factorized",
            "simplicial_impl": "auto",
            "disable_geometry_bias": False,
        }
    }
    reloaded = QM9EDMModel(
        d_model=int(checkpoint["args"]["d_model"]),
        n_heads=int(checkpoint["args"]["n_heads"]),
        n_layers=int(checkpoint["args"]["n_layers"]),
        mlp_ratio=float(checkpoint["args"]["mlp_ratio"]),
        dropout=float(checkpoint["args"]["dropout"]),
        attn_dropout=float(checkpoint["args"]["attn_dropout"]),
        attn_type=str(checkpoint["args"]["attn_type"]),
        simplicial_geom_mode=str(checkpoint["args"]["simplicial_geom_mode"]),
        simplicial_impl=str(checkpoint["args"]["simplicial_impl"]),
        simplicial_precision=str(checkpoint["args"].get("simplicial_precision", "ieee_fp32")),
        use_geometry_bias=not bool(checkpoint["args"]["disable_geometry_bias"]),
    )
    assert reloaded.trunk.blocks[0].attn.precision == "ieee_fp32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_eval_mode_dropout_allows_triton():
    torch.manual_seed(0)
    device = torch.device("cuda")
    attn = TwoSimplicialAttention(dim=32, num_heads=4, impl="auto", dropout=0.1, precision="ieee_fp32").to(device)
    attn.eval()
    x = torch.randn(2, 5, 32, device=device, dtype=torch.float32)
    _ = attn(x)
    assert attn._last_impl_used == "triton"


def test_triton_autograd_passes_optional_gradient_flags(monkeypatch):
    captured: dict[str, bool] = {}

    def fake_forward(q, k1, v1, k2, v2, **kwargs):
        del k1, v1, k2, v2, kwargs
        out = torch.zeros_like(q)
        lse = q.new_zeros(q.shape[:3], dtype=torch.float32)
        return out, lse, out.float()

    def fake_backward(grad_out, q, k1, v1, k2, v2, *args, **kwargs):
        del grad_out, args
        for name in (
            "need_du",
            "need_dv_bias",
            "need_dw",
            "need_dgate",
            "need_dangle_left",
            "need_dangle_right",
            "need_dangle_gate",
            "need_dmessage_left",
            "need_dmessage_right",
            "need_dmessage_basis",
        ):
            captured[name] = bool(kwargs[name])
        zeros = [torch.zeros_like(tensor) for tensor in (q, k1, v1, k2, v2)]
        angle_left = kwargs["angle_left"]
        angle_gate = kwargs["angle_gate"]
        message_right = kwargs["message_right"]
        return (
            *zeros,
            None,
            None,
            None,
            None,
            torch.zeros_like(angle_left),
            None,
            torch.zeros_like(angle_gate),
            None,
            torch.zeros_like(message_right),
            None,
        )

    monkeypatch.setattr(attention_module, "triton_simplicial_attention_forward", fake_forward)
    monkeypatch.setattr(attention_module, "triton_simplicial_attention_backward", fake_backward)

    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    k1 = torch.randn_like(q, requires_grad=True)
    v1 = torch.randn_like(q, requires_grad=True)
    k2 = torch.randn_like(q, requires_grad=True)
    v2 = torch.randn_like(q, requires_grad=True)
    query_valid = torch.ones(1, 3, dtype=torch.bool)
    pair_key_valid = torch.ones(1, 3, dtype=torch.bool)
    pair_valid = torch.empty(0, dtype=torch.bool)
    u = torch.randn(1, 2, 3, 3)
    v = torch.randn_like(u)
    w = torch.randn_like(u)
    gate = torch.randn(1, 2, 3)
    angle_left = torch.randn(1, 2, 3, 3, 2, requires_grad=True)
    angle_right = torch.randn(1, 2, 3, 3, 2)
    angle_gate = torch.randn(1, 2, 3, requires_grad=True)
    message_left = torch.randn(1, 2, 3, 3, 2)
    message_right = torch.randn(1, 2, 3, 3, 2, requires_grad=True)
    message_basis = torch.randn(2, 2, 4)

    out = _TritonTwoSimplicialAttentionFunction.apply(
        q,
        k1,
        v1,
        k2,
        v2,
        query_valid,
        pair_key_valid,
        pair_valid,
        u,
        v,
        w,
        gate,
        angle_left,
        angle_right,
        angle_gate,
        message_left,
        message_right,
        message_basis,
        "ieee_fp32",
        128,
        False,
    )
    out.sum().backward()

    assert captured == {
        "need_du": False,
        "need_dv_bias": False,
        "need_dw": False,
        "need_dgate": False,
        "need_dangle_left": True,
        "need_dangle_right": False,
        "need_dangle_gate": True,
        "need_dmessage_left": False,
        "need_dmessage_right": True,
        "need_dmessage_basis": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_triton_no_bias_forward_backward_matches_torch_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_triton = q_torch.detach().clone().requires_grad_(True)
    k1_triton = k1_torch.detach().clone().requires_grad_(True)
    v1_triton = v1_torch.detach().clone().requires_grad_(True)
    k2_triton = k2_torch.detach().clone().requires_grad_(True)
    v2_triton = v2_torch.detach().clone().requires_grad_(True)
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        torch.tensor(
            [
                [False, False, False, False, True],
                [False, False, True, True, True],
            ],
            device=device,
        ),
        batch_size=2,
        num_tokens=5,
        device=device,
    )
    torch_out = simplicial_attention_torch_from_projected(
        q_torch,
        k1_torch,
        v1_torch,
        k2_torch,
        v2_torch,
        attention_mask=attention_mask,
        chunk_size=2,
        fp32_core=True,
    )
    triton_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_triton,
        k1_triton,
        v1_triton,
        k2_triton,
        v2_triton,
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        torch.empty(0, device=device, dtype=torch.bool),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        "ieee_fp32",
        2,
        False,
    )
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4)
    torch_loss = torch_out.square().mean()
    triton_loss = triton_out.square().mean()
    torch_loss.backward()
    triton_loss.backward()
    for actual, expected in (
        (q_triton.grad, q_torch.grad),
        (k1_triton.grad, k1_torch.grad),
        (v1_triton.grad, v1_torch.grad),
        (k2_triton.grad, k2_torch.grad),
        (v2_triton.grad, v2_torch.grad),
    ):
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_triton_forward_backward_ieee_matches_torch_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_triton = q_torch.detach().clone().requires_grad_(True)
    k1_triton = k1_torch.detach().clone().requires_grad_(True)
    v1_triton = v1_torch.detach().clone().requires_grad_(True)
    k2_triton = k2_torch.detach().clone().requires_grad_(True)
    v2_triton = v2_torch.detach().clone().requires_grad_(True)
    factorized_bias_torch = _make_factorized_bias(2, 4, 5, device=device, dtype=torch.float32, requires_grad=True)
    factorized_bias_triton = SimplicialFactorizedBias(
        u=factorized_bias_torch.u.detach().clone().requires_grad_(True),
        v=factorized_bias_torch.v.detach().clone().requires_grad_(True),
        w=factorized_bias_torch.w.detach().clone().requires_grad_(True),
        gate=factorized_bias_torch.gate.detach().clone().requires_grad_(True),
    )
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        torch.tensor(
            [
                [False, False, False, False, True],
                [False, False, True, True, True],
            ],
            device=device,
        ),
        batch_size=2,
        num_tokens=5,
        device=device,
    )
    torch_out = simplicial_attention_torch_from_projected(
        q_torch,
        k1_torch,
        v1_torch,
        k2_torch,
        v2_torch,
        attention_mask=attention_mask,
        factorized_bias=factorized_bias_torch,
        chunk_size=2,
        fp32_core=True,
    )
    triton_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_triton,
        k1_triton,
        v1_triton,
        k2_triton,
        v2_triton,
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        torch.empty(0, device=device, dtype=torch.bool),
        factorized_bias_triton.u,
        factorized_bias_triton.v,
        factorized_bias_triton.w,
        factorized_bias_triton.gate,
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        "ieee_fp32",
        2,
        True,
    )
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4)
    torch_loss = torch_out.square().mean()
    triton_loss = triton_out.square().mean()
    torch_loss.backward()
    triton_loss.backward()
    for actual, expected in (
        (q_triton.grad, q_torch.grad),
        (k1_triton.grad, k1_torch.grad),
        (v1_triton.grad, v1_torch.grad),
        (k2_triton.grad, k2_torch.grad),
        (v2_triton.grad, v2_torch.grad),
        (factorized_bias_triton.u.grad, factorized_bias_torch.u.grad),
        (factorized_bias_triton.v.grad, factorized_bias_torch.v.grad),
        (factorized_bias_triton.w.grad, factorized_bias_torch.w.grad),
        (factorized_bias_triton.gate.grad, factorized_bias_torch.gate.grad),
    ):
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_triton_low_rank_angle_forward_backward_matches_torch_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_triton = q_torch.detach().clone().requires_grad_(True)
    k1_triton = k1_torch.detach().clone().requires_grad_(True)
    v1_triton = v1_torch.detach().clone().requires_grad_(True)
    k2_triton = k2_torch.detach().clone().requires_grad_(True)
    v2_triton = v2_torch.detach().clone().requires_grad_(True)
    angle_torch = _make_low_rank_angle_residual(2, 4, 5, 4, device=device, dtype=torch.float32, requires_grad=True)
    angle_triton = SimplicialLowRankAngleResidual(
        left=angle_torch.left.detach().clone().requires_grad_(True),
        right=angle_torch.right.detach().clone().requires_grad_(True),
        gate=angle_torch.gate.detach().clone().requires_grad_(True),
    )
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        torch.tensor(
            [
                [False, False, False, False, True],
                [False, False, True, True, True],
            ],
            device=device,
        ),
        batch_size=2,
        num_tokens=5,
        device=device,
    )
    torch_out = simplicial_attention_torch_from_projected(
        q_torch,
        k1_torch,
        v1_torch,
        k2_torch,
        v2_torch,
        attention_mask=attention_mask,
        angle_residual=angle_torch,
        chunk_size=2,
        fp32_core=True,
    )
    triton_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_triton,
        k1_triton,
        v1_triton,
        k2_triton,
        v2_triton,
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        torch.empty(0, device=device, dtype=torch.bool),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        angle_triton.left,
        angle_triton.right,
        angle_triton.gate,
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        "ieee_fp32",
        2,
        False,
    )
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4)
    torch_loss = torch_out.square().mean()
    triton_loss = triton_out.square().mean()
    torch_loss.backward()
    triton_loss.backward()
    for actual, expected in (
        (q_triton.grad, q_torch.grad),
        (k1_triton.grad, k1_torch.grad),
        (v1_triton.grad, v1_torch.grad),
        (k2_triton.grad, k2_torch.grad),
        (v2_triton.grad, v2_torch.grad),
        (angle_triton.left.grad, angle_torch.left.grad),
        (angle_triton.right.grad, angle_torch.right.grad),
        (angle_triton.gate.grad, angle_torch.gate.grad),
    ):
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_triton_low_rank_message_forward_backward_matches_torch_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    q_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v1_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    k2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    v2_torch = torch.randn(2, 4, 5, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_triton = q_torch.detach().clone().requires_grad_(True)
    k1_triton = k1_torch.detach().clone().requires_grad_(True)
    v1_triton = v1_torch.detach().clone().requires_grad_(True)
    k2_triton = k2_torch.detach().clone().requires_grad_(True)
    v2_triton = v2_torch.detach().clone().requires_grad_(True)
    message_torch = _make_low_rank_message_residual(
        2,
        4,
        5,
        3,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    message_triton = SimplicialLowRankMessageResidual(
        left=message_torch.left.detach().clone().requires_grad_(True),
        right=message_torch.right.detach().clone().requires_grad_(True),
    )
    message_basis_torch = torch.randn(4, 3, 8, device=device, dtype=torch.float32, requires_grad=True)
    message_basis_triton = message_basis_torch.detach().clone().requires_grad_(True)
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        torch.tensor(
            [
                [False, False, False, False, True],
                [False, False, True, True, True],
            ],
            device=device,
        ),
        batch_size=2,
        num_tokens=5,
        device=device,
    )
    torch_out = simplicial_attention_torch_from_projected(
        q_torch,
        k1_torch,
        v1_torch,
        k2_torch,
        v2_torch,
        attention_mask=attention_mask,
        message_residual=message_torch,
        message_basis=message_basis_torch,
        chunk_size=2,
        fp32_core=True,
    )
    triton_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_triton,
        k1_triton,
        v1_triton,
        k2_triton,
        v2_triton,
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        torch.empty(0, device=device, dtype=torch.bool),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        message_triton.left,
        message_triton.right,
        message_basis_triton,
        "ieee_fp32",
        2,
        False,
    )
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4)
    torch_loss = torch_out.square().mean()
    triton_loss = triton_out.square().mean()
    torch_loss.backward()
    triton_loss.backward()
    for actual, expected in (
        (q_triton.grad, q_torch.grad),
        (k1_triton.grad, k1_torch.grad),
        (v1_triton.grad, v1_torch.grad),
        (k2_triton.grad, k2_torch.grad),
        (v2_triton.grad, v2_torch.grad),
        (message_triton.left.grad, message_torch.left.grad),
        (message_triton.right.grad, message_torch.right.grad),
        (message_basis_triton.grad, message_basis_torch.grad),
    ):
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_triton_factorized_angle_and_message_mismatched_ranks_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size, num_heads, num_tokens, head_dim = 1, 2, 7, 8
    q_torch = torch.randn(batch_size, num_heads, num_tokens, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    k1_torch = torch.randn_like(q_torch, requires_grad=True)
    v1_torch = torch.randn_like(q_torch, requires_grad=True)
    k2_torch = torch.randn_like(q_torch, requires_grad=True)
    v2_torch = torch.randn_like(q_torch, requires_grad=True)
    q_triton = q_torch.detach().clone().requires_grad_(True)
    k1_triton = k1_torch.detach().clone().requires_grad_(True)
    v1_triton = v1_torch.detach().clone().requires_grad_(True)
    k2_triton = k2_torch.detach().clone().requires_grad_(True)
    v2_triton = v2_torch.detach().clone().requires_grad_(True)
    factorized_torch = _make_factorized_bias(
        batch_size,
        num_heads,
        num_tokens,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    factorized_triton = SimplicialFactorizedBias(
        u=factorized_torch.u.detach().clone().requires_grad_(True),
        v=factorized_torch.v.detach().clone().requires_grad_(True),
        w=factorized_torch.w.detach().clone().requires_grad_(True),
        gate=factorized_torch.gate.detach().clone().requires_grad_(True),
    )
    angle_torch = _make_low_rank_angle_residual(
        batch_size,
        num_heads,
        num_tokens,
        5,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    angle_triton = SimplicialLowRankAngleResidual(
        left=angle_torch.left.detach().clone().requires_grad_(True),
        right=angle_torch.right.detach().clone().requires_grad_(True),
        gate=angle_torch.gate.detach().clone().requires_grad_(True),
    )
    message_torch = _make_low_rank_message_residual(
        batch_size,
        num_heads,
        num_tokens,
        3,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    message_triton = SimplicialLowRankMessageResidual(
        left=message_torch.left.detach().clone().requires_grad_(True),
        right=message_torch.right.detach().clone().requires_grad_(True),
    )
    message_basis_torch = torch.randn(num_heads, 3, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    message_basis_triton = message_basis_torch.detach().clone().requires_grad_(True)
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        torch.tensor([[False, False, False, False, False, True, True]], device=device),
        batch_size=batch_size,
        num_tokens=num_tokens,
        device=device,
    )
    pair_valid = torch.ones(batch_size, num_tokens, num_tokens, device=device, dtype=torch.bool)
    pair_valid[:, 1, 4] = False
    pair_valid[:, 4, 1] = False
    attention_mask = SimplicialAttentionMask(
        query_valid=attention_mask.query_valid,
        pair_key_valid=attention_mask.pair_key_valid,
        pair_valid=pair_valid,
    )
    torch_out = simplicial_attention_torch_from_projected(
        q_torch,
        k1_torch,
        v1_torch,
        k2_torch,
        v2_torch,
        attention_mask=attention_mask,
        factorized_bias=factorized_torch,
        angle_residual=angle_torch,
        message_residual=message_torch,
        message_basis=message_basis_torch,
        chunk_size=3,
        fp32_core=True,
    )
    triton_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_triton,
        k1_triton,
        v1_triton,
        k2_triton,
        v2_triton,
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        attention_mask.pair_valid,
        factorized_triton.u,
        factorized_triton.v,
        factorized_triton.w,
        factorized_triton.gate,
        angle_triton.left,
        angle_triton.right,
        angle_triton.gate,
        message_triton.left,
        message_triton.right,
        message_basis_triton,
        "ieee_fp32",
        3,
        False,
    )
    assert torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4)
    torch_loss = torch_out.square().mean()
    triton_loss = triton_out.square().mean()
    torch_loss.backward()
    triton_loss.backward()
    for actual, expected in (
        (q_triton.grad, q_torch.grad),
        (k1_triton.grad, k1_torch.grad),
        (v1_triton.grad, v1_torch.grad),
        (k2_triton.grad, k2_torch.grad),
        (v2_triton.grad, v2_torch.grad),
        (factorized_triton.u.grad, factorized_torch.u.grad),
        (factorized_triton.v.grad, factorized_torch.v.grad),
        (factorized_triton.w.grad, factorized_torch.w.grad),
        (factorized_triton.gate.grad, factorized_torch.gate.grad),
        (angle_triton.left.grad, angle_torch.left.grad),
        (angle_triton.right.grad, angle_torch.right.grad),
        (angle_triton.gate.grad, angle_torch.gate.grad),
        (message_triton.left.grad, message_torch.left.grad),
        (message_triton.right.grad, message_torch.right.grad),
        (message_basis_triton.grad, message_basis_torch.grad),
    ):
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity tests")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
def test_triton_bf16_forward_backward_within_tolerance():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_torch = torch.randn(2, 4, 8, 16, device=device, dtype=dtype, requires_grad=True)
    k1_torch = torch.randn(2, 4, 8, 16, device=device, dtype=dtype, requires_grad=True)
    v1_torch = torch.randn(2, 4, 8, 16, device=device, dtype=dtype, requires_grad=True)
    k2_torch = torch.randn(2, 4, 8, 16, device=device, dtype=dtype, requires_grad=True)
    v2_torch = torch.randn(2, 4, 8, 16, device=device, dtype=dtype, requires_grad=True)
    q_triton = q_torch.detach().clone().requires_grad_(True)
    k1_triton = k1_torch.detach().clone().requires_grad_(True)
    v1_triton = v1_torch.detach().clone().requires_grad_(True)
    k2_triton = k2_torch.detach().clone().requires_grad_(True)
    v2_triton = v2_torch.detach().clone().requires_grad_(True)
    attention_mask = SimplicialAttentionMask.from_key_padding_mask(
        torch.tensor(
            [
                [False, False, False, False, False, True, True, True],
                [False, False, False, False, True, True, True, True],
            ],
            device=device,
        ),
        batch_size=2,
        num_tokens=8,
        device=device,
    )
    factorized_bias_torch = _make_factorized_bias(2, 4, 8, device=device, dtype=dtype, requires_grad=True)
    factorized_bias_triton = SimplicialFactorizedBias(
        u=factorized_bias_torch.u.detach().clone().requires_grad_(True),
        v=factorized_bias_torch.v.detach().clone().requires_grad_(True),
        w=factorized_bias_torch.w.detach().clone().requires_grad_(True),
        gate=factorized_bias_torch.gate.detach().clone().requires_grad_(True),
    )
    torch_out = simplicial_attention_torch_from_projected(
        q_torch,
        k1_torch,
        v1_torch,
        k2_torch,
        v2_torch,
        attention_mask=attention_mask,
        factorized_bias=factorized_bias_torch,
        chunk_size=4,
    )
    triton_out = _TritonTwoSimplicialAttentionFunction.apply(
        q_triton.to(torch.bfloat16),
        k1_triton.to(torch.bfloat16),
        v1_triton.to(torch.bfloat16),
        k2_triton.to(torch.bfloat16),
        v2_triton.to(torch.bfloat16),
        attention_mask.query_valid,
        attention_mask.pair_key_valid,
        torch.empty(0, device=device, dtype=torch.bool),
        factorized_bias_triton.u.to(torch.bfloat16),
        factorized_bias_triton.v.to(torch.bfloat16),
        factorized_bias_triton.w.to(torch.bfloat16),
        factorized_bias_triton.gate.to(torch.bfloat16),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        q_triton.new_empty(0),
        "bf16_tc",
        4,
        False,
    )
    assert torch.allclose(triton_out.float(), torch_out.float(), atol=5e-2, rtol=5e-2)
    torch_loss = torch_out.float().square().mean()
    triton_loss = triton_out.float().square().mean()
    torch_loss.backward()
    triton_loss.backward()
    for actual, expected in (
        (q_triton.grad.float(), q_torch.grad.float()),
        (k1_triton.grad.float(), k1_torch.grad.float()),
        (v1_triton.grad.float(), v1_torch.grad.float()),
        (k2_triton.grad.float(), k2_torch.grad.float()),
        (v2_triton.grad.float(), v2_torch.grad.float()),
    ):
        assert torch.allclose(actual, expected, atol=8e-2, rtol=8e-2)
