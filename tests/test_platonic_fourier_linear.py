import pytest
import torch
import torch.nn.functional as F

from matterformer.data import SyntheticOMolDataset, collate_omol
from matterformer.models import MatterformerOMolForceField
from matterformer.models.platonic import PLATONIC_GROUPS, PlatonicBlock, PlatonicLinear
from matterformer.models.platonic.layers import PlatonicAttention
from matterformer.models.platonic.rope import PlatonicRoPE


def test_platonic_rope_cached_helpers_match_forward_and_constant_key():
    torch.manual_seed(20)
    rope = PlatonicRoPE(
        embed_dim=12 * 4,
        num_heads=1,
        solid_name="tetrahedron",
        head_dim=4,
        freq_sigma=1.3,
        learned_freqs=True,
        freq_init="random",
    )
    x = torch.randn(2, 5, 12, 1, 4)
    pos = torch.randn(2, 5, 3)

    cos, sin = rope.cos_sin(pos, dtype=x.dtype, device=x.device)
    torch.testing.assert_close(rope.apply_from_cos_sin(x, cos, sin), rope(x, pos), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        rope.apply_from_cos_sin(x, cos, sin, inverse=True),
        rope(x, pos, inverse=True),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        rope.constant_key_from_cos_sin(cos, sin),
        rope(torch.ones_like(x), pos),
        atol=1e-6,
        rtol=1e-6,
    )


def test_platonic_attention_rope_cache_matches_recompute_forward_backward():
    torch.manual_seed(21)
    kwargs = dict(
        d_model=12 * 4,
        num_heads=12,
        solid_name="tetrahedron",
        dropout=0.0,
        rope_sigma=1.0,
        learned_freqs=True,
        freq_init="random",
        use_key=False,
        rope_on_values=True,
        attention_backend="sdpa",
    )
    cached = PlatonicAttention(**kwargs, rope_cache=True, constant_key_fastpath=True)
    recompute = PlatonicAttention(**kwargs, rope_cache=False, constant_key_fastpath=False)
    recompute.load_state_dict(cached.state_dict())

    x0 = torch.randn(2, 5, 12 * 4)
    pos = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    x_cached = x0.clone().requires_grad_(True)
    x_recompute = x0.clone().requires_grad_(True)
    y_cached = cached(x_cached, pos=pos, pad_mask=pad_mask)
    y_recompute = recompute(x_recompute, pos=pos, pad_mask=pad_mask)
    torch.testing.assert_close(y_cached, y_recompute, atol=2e-5, rtol=2e-5)

    loss_cached = y_cached.square().mean() + 0.01 * y_cached.sum()
    loss_recompute = y_recompute.square().mean() + 0.01 * y_recompute.sum()
    loss_cached.backward()
    loss_recompute.backward()
    torch.testing.assert_close(x_cached.grad, x_recompute.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(cached.rope.freqs.grad, recompute.rope.freqs.grad, atol=5e-5, rtol=5e-5)


def test_platonic_attention_flat_rope_cache_matches_recompute():
    torch.manual_seed(22)
    kwargs = dict(
        d_model=12 * 4,
        num_heads=12,
        solid_name="tetrahedron",
        dropout=0.0,
        rope_sigma=0.7,
        learned_freqs=True,
        freq_init="random",
        use_key=False,
        rope_on_values=True,
        attention_backend="sdpa",
    )
    cached = PlatonicAttention(**kwargs, rope_cache=True, constant_key_fastpath=True)
    recompute = PlatonicAttention(**kwargs, rope_cache=False, constant_key_fastpath=False)
    recompute.load_state_dict(cached.state_dict())
    x = torch.randn(7, 12 * 4)
    pos = torch.randn(7, 3)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    y_cached = cached.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=4)
    y_recompute = recompute.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=4)
    torch.testing.assert_close(y_cached, y_recompute, atol=2e-5, rtol=2e-5)


def test_fourier_platonic_linear_matches_spatial_tetra_forward_backward():
    torch.manual_seed(0)
    spatial = PlatonicLinear(12 * 4, 12 * 7, solid="tetrahedron", linear_backend="spatial")
    fourier = PlatonicLinear(12 * 4, 12 * 7, solid="tetrahedron", linear_backend="fourier")
    fourier.load_state_dict(spatial.state_dict())

    x0 = torch.randn(3, 5, 12 * 4)
    x_spatial = x0.clone().requires_grad_(True)
    x_fourier = x0.clone().requires_grad_(True)
    y_spatial = spatial(x_spatial)
    y_fourier = fourier(x_fourier)
    torch.testing.assert_close(y_fourier, y_spatial, atol=2e-5, rtol=2e-5)

    loss_spatial = y_spatial.square().mean() + 0.01 * y_spatial.sum()
    loss_fourier = y_fourier.square().mean() + 0.01 * y_fourier.sum()
    loss_spatial.backward()
    loss_fourier.backward()

    torch.testing.assert_close(x_fourier.grad, x_spatial.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(fourier.kernel.grad, spatial.kernel.grad, atol=5e-5, rtol=5e-5)
    assert fourier.bias is not None
    assert spatial.bias is not None
    torch.testing.assert_close(fourier.bias.grad, spatial.bias.grad, atol=5e-5, rtol=5e-5)


def test_fourier_direct_platonic_linear_matches_spatial_tetra_forward_input_grad():
    torch.manual_seed(10)
    spatial = PlatonicLinear(12 * 4, 12 * 7, solid="tetrahedron", linear_backend="spatial")
    direct = PlatonicLinear(12 * 4, 12 * 7, solid="tetrahedron", linear_backend="fourier_direct")
    assert spatial.bias is not None
    direct.set_spatial_parameters_(spatial.kernel, spatial.bias)

    x0 = torch.randn(3, 5, 12 * 4)
    x_spatial = x0.clone().requires_grad_(True)
    x_direct = x0.clone().requires_grad_(True)
    y_spatial = spatial(x_spatial)
    y_direct = direct(x_direct)
    torch.testing.assert_close(y_direct, y_spatial, atol=2e-5, rtol=2e-5)

    loss_spatial = y_spatial.square().mean() + 0.01 * y_spatial.sum()
    loss_direct = y_direct.square().mean() + 0.01 * y_direct.sum()
    loss_spatial.backward()
    loss_direct.backward()
    torch.testing.assert_close(x_direct.grad, x_spatial.grad, atol=5e-5, rtol=5e-5)
    assert direct.w1.grad is not None
    assert direct.w2_re.grad is not None
    assert direct.w2_im.grad is not None
    assert direct.w3.grad is not None
    assert direct.bias is not None and direct.bias.grad is not None


def test_fourier_platonic_linear_equivariance_tetra():
    torch.manual_seed(1)
    group = PLATONIC_GROUPS["tetrahedron"]
    layer = PlatonicLinear(12 * 3, 12 * 5, solid="tetrahedron", linear_backend="fourier")
    x = torch.randn(2, 12, 3)
    y = layer(x.reshape(2, -1)).view(2, 12, 5)
    for group_idx in (0, 3, 7):
        permutation = group.cayley_table[group_idx]
        lhs = layer(x[:, permutation].reshape(2, -1)).view(2, 12, 5)
        rhs = y[:, permutation]
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_fourier_direct_platonic_linear_equivariance_tetra():
    torch.manual_seed(11)
    group = PLATONIC_GROUPS["tetrahedron"]
    layer = PlatonicLinear(12 * 3, 12 * 5, solid="tetrahedron", linear_backend="fourier_direct")
    x = torch.randn(2, 12, 3)
    y = layer(x.reshape(2, -1)).view(2, 12, 5)
    for group_idx in (0, 3, 7):
        permutation = group.cayley_table[group_idx]
        lhs = layer(x[:, permutation].reshape(2, -1)).view(2, 12, 5)
        rhs = y[:, permutation]
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_tetra_fourier_backend_matches_dense_weight_for_random_kernel():
    torch.manual_seed(3)
    layer = PlatonicLinear(12 * 2, 12 * 3, solid="tetrahedron", linear_backend="fourier")
    x = torch.randn(4, 12 * 2)
    dense_y = F.linear(x, layer.get_weight(), None).view(4, 12, 3)
    assert layer.bias is not None
    dense_y = dense_y + layer.bias
    fourier_y = layer(x).view(4, 12, 3)
    torch.testing.assert_close(fourier_y, dense_y, atol=2e-5, rtol=2e-5)


def test_tetra_fourier_direct_backend_matches_dense_weight_from_spatial_kernel():
    torch.manual_seed(12)
    spatial = PlatonicLinear(12 * 2, 12 * 3, solid="tetrahedron", linear_backend="spatial")
    direct = PlatonicLinear(12 * 2, 12 * 3, solid="tetrahedron", linear_backend="fourier_direct")
    assert spatial.bias is not None
    direct.set_spatial_parameters_(spatial.kernel, spatial.bias)
    x = torch.randn(4, 12 * 2)
    dense_y = F.linear(x, spatial.get_weight(), None).view(4, 12, 3) + spatial.bias
    direct_y = direct(x).view(4, 12, 3)
    torch.testing.assert_close(direct_y, dense_y, atol=2e-5, rtol=2e-5)


def test_platonic_block_fourier_backend_shape_backward():
    torch.manual_seed(2)
    block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        linear_backend="fourier",
    )
    x = torch.randn(2, 5, 12 * 4, requires_grad=True)
    pos = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    out = block(x, pos=pos, pad_mask=pad_mask)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_platonic_block_fourier_direct_backend_shape_backward():
    torch.manual_seed(13)
    block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        linear_backend="fourier_direct",
    )
    x = torch.randn(2, 5, 12 * 4, requires_grad=True)
    pos = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    out = block(x, pos=pos, pad_mask=pad_mask)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_platonic_block_can_split_attention_and_ffn_linear_backends():
    block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        linear_backend="spatial",
        attention_linear_backend="spatial",
        ffn_linear_backend="fourier_direct",
    )
    assert block.attn.q_proj.linear_backend == "spatial"
    assert block.attn.v_proj.linear_backend == "spatial"
    assert block.attn.out_proj.linear_backend == "spatial"
    assert block.linear1.linear_backend == "fourier_direct"
    assert block.linear2.linear_backend == "fourier_direct"


def test_omol_tetra_fourier_linear_backend_forward_backward():
    torch.manual_seed(4)
    dataset = SyntheticOMolDataset(num_samples=2, seed=7, min_atoms=3, max_atoms=4)
    batch = collate_omol([dataset[0], dataset[1]])
    hybrid_config = {
        "num_blocks": 1,
        "stream_type": "tetra",
        "block_mix": [0, 1, 0],
        "scalar_dim": 24,
        "tetra_dim_per_frame": 2,
        "tetra": {
            "group": "tetrahedron",
            "group_order": 12,
            "heads_per_frame": 1,
            "rope_sigma": 1.0,
            "learned_freqs": True,
            "use_key": False,
            "rope_on_values": True,
            "ffn_mult": 2,
            "linear_backend": "fourier",
        },
        "readout": {"kind": "platonic_ffn"},
    }
    model = MatterformerOMolForceField(
        d_model=24,
        n_heads=4,
        n_layers=1,
        hybrid_config=hybrid_config,
        chgspin_mode="off",
        pair_hidden_dim=24,
        pair_n_rbf=8,
        readout_head_mode="platonic",
    )
    outputs = model(batch.atomic_numbers, batch.coords, batch.pad_mask, charge=batch.charge, spin=batch.spin)
    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == batch.forces.shape
    loss = outputs["energy"].sum() + outputs["forces"].square().mean()
    loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark requires a CUDA device")
def test_fourier_platonic_linear_benchmark_smoke():
    torch.manual_seed(5)
    device = torch.device("cuda")
    spatial = PlatonicLinear(12 * 160, 12 * 160, solid="tetrahedron", linear_backend="spatial").to(device)
    fourier = PlatonicLinear(12 * 160, 12 * 160, solid="tetrahedron", linear_backend="fourier").to(device)
    fourier.load_state_dict(spatial.state_dict())
    x = torch.randn(512, 12 * 160, device=device)
    with torch.no_grad():
        y_spatial = spatial(x)
        y_fourier = fourier(x)
    torch.testing.assert_close(y_fourier, y_spatial, atol=1e-3, rtol=1e-3)
