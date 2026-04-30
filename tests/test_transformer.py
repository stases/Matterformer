import torch
import pytest

from matterformer.geometry import NonPeriodicGeometryAdapter, PeriodicGeometryAdapter
from matterformer.models.transformer import (
    GeometryBiasBuilder,
    MhaFactorizedGeometryBias,
    RotaryMultiheadAttention,
    SimplicialGeometryBias,
    TransformerTrunk,
    _build_simplicial_attention_mask,
)


def test_transformer_trunk_nonperiodic_forward_masks_padding():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    bias = GeometryBiasBuilder(n_heads=4, use_noise_gate=False)
    trunk = TransformerTrunk(
        d_model=32,
        n_heads=4,
        n_layers=2,
        geometry_adapter=adapter,
        geometry_bias=bias,
        attn_type="mha",
    )

    x = torch.randn(2, 5, 32)
    cond = torch.randn(2, 32)
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, True, True, True],
        ]
    )
    output = trunk(x, cond, pad_mask=pad_mask, coords=coords)
    assert output.shape == (2, 5, 32)
    assert torch.allclose(output[pad_mask], torch.zeros_like(output[pad_mask]))


def test_transformer_trunk_default_mha_keeps_torch_multihead_attention():
    trunk = TransformerTrunk(
        d_model=32,
        n_heads=4,
        n_layers=1,
        attn_type="mha",
    )
    assert trunk.mha_position_mode == "none"
    assert isinstance(trunk.blocks[0].attn, torch.nn.MultiheadAttention)


def test_transformer_trunk_mha_rope_forward_with_geometry_bias_and_extra_token():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    bias = GeometryBiasBuilder(n_heads=4, use_noise_gate=False)
    trunk = TransformerTrunk(
        d_model=32,
        n_heads=4,
        n_layers=2,
        geometry_adapter=adapter,
        geometry_bias=bias,
        attn_type="mha",
        mha_position_mode="rope",
    )

    x = torch.randn(2, 6, 32, requires_grad=True)
    cond = torch.randn(2, 32)
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.tensor(
        [
            [False, False, False, False, True, False],
            [False, False, True, True, True, False],
        ]
    )
    output = trunk(x, cond, pad_mask=pad_mask, coords=coords)
    assert output.shape == (2, 6, 32)
    assert torch.isfinite(output).all()
    assert torch.allclose(output[pad_mask], torch.zeros_like(output[pad_mask]))
    output.square().mean().backward()
    assert x.grad is not None


def test_rotary_mha_supports_platonic_style_keys_and_values():
    torch.manual_seed(0)
    attn = RotaryMultiheadAttention(
        embed_dim=32,
        num_heads=4,
        rope_use_key=False,
        rope_on_values=True,
    )
    x = torch.randn(2, 5, 32, requires_grad=True)
    positions = torch.randn(2, 5, 3)
    output, _ = attn(x, x, x, positions=positions)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert x.grad is not None
    key_grad = attn.in_proj_weight.grad[32:64]
    assert torch.allclose(key_grad, torch.zeros_like(key_grad))


def test_transformer_trunk_mha_rope_requires_coords():
    trunk = TransformerTrunk(
        d_model=32,
        n_heads=4,
        n_layers=1,
        attn_type="mha",
        mha_position_mode="rope",
    )
    with pytest.raises(ValueError, match="coords must be provided"):
        trunk(torch.randn(1, 4, 32), torch.randn(1, 32))


def test_transformer_trunk_mha_factorized_geometry_bias_forward():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    bias = MhaFactorizedGeometryBias(n_heads=4, use_noise_gate=True)
    trunk = TransformerTrunk(
        d_model=32,
        n_heads=4,
        n_layers=2,
        geometry_adapter=adapter,
        geometry_bias=bias,
        attn_type="mha",
    )

    x = torch.randn(2, 5, 32)
    cond = torch.randn(2, 32)
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, True, True, True],
        ]
    )
    sigma = torch.tensor([0.1, 1.0])
    output = trunk(x, cond, pad_mask=pad_mask, coords=coords, sigma=sigma)
    assert output.shape == (2, 5, 32)
    assert torch.isfinite(output).all()
    assert torch.allclose(output[pad_mask], torch.zeros_like(output[pad_mask]))


def test_transformer_trunk_periodic_simplicial_forward():
    torch.manual_seed(0)
    adapter = PeriodicGeometryAdapter(pbc_radius=1)
    simplicial_bias = SimplicialGeometryBias(n_heads=4, mode="factorized", use_noise_gate=False)
    trunk = TransformerTrunk(
        d_model=32,
        n_heads=4,
        n_layers=2,
        geometry_adapter=adapter,
        simplicial_geometry_bias=simplicial_bias,
        attn_type="simplicial",
    )
    x = torch.randn(1, 4, 32)
    cond = torch.randn(1, 32)
    coords = torch.rand(1, 4, 3)
    lattice = torch.zeros(1, 6)
    output = trunk(x, cond, coords=coords, lattice=lattice)
    assert output.shape == (1, 4, 32)
    assert torch.isfinite(output).all()


def test_simplicial_mask_can_include_periodic_global_tokens_as_pair_keys():
    pad_mask = torch.tensor([[False, False, False, False, False]])
    attention_mask = _build_simplicial_attention_mask(
        coords_len=4,
        seq_len=5,
        batch_size=1,
        device=torch.device("cpu"),
        pad_mask=pad_mask,
        include_global_tokens_as_pair_keys=True,
    )
    assert attention_mask.pair_valid is None
    assert torch.equal(
        attention_mask.pair_key_valid,
        torch.tensor([[True, True, True, True, True]]),
    )
