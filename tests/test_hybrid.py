import pytest
import torch

from matterformer.geometry import NonPeriodicGeometryAdapter
from matterformer.models import (
    CompactSimplicialBias,
    CompactSimplicialAttention,
    CompactSimplicialGeometryBias,
    GeomDrugsEDMModel,
    MOFStage1EDMModel,
    ModelState,
    QM9EDMModel,
    TrivialGlobalLayer,
    build_geometry_cache,
    compact_simplicial_attention_torch,
    compact_simplicial_attention_triton,
    expand_hybrid_schedule,
)
from matterformer.models.hybrid import HybridCoupling
from matterformer.models.platonic import PLATONIC_GROUPS, PlatonicBlock, PlatonicLinear
from matterformer.models.triton_compact_simplicial_attention import TRITON_COMPACT_SIMPLICIAL_AVAILABLE


@pytest.mark.parametrize(
    ("block_mix", "expected"),
    [
        ([1, 0, 0], [["simplicial"], ["simplicial"]]),
        ([0, 1, 0], [["tetra"], ["tetra"]]),
        ([0, 0, 1], [["trivial"], ["trivial"]]),
        ([2, 1, 0], [["simplicial", "simplicial", "tetra"], ["simplicial", "simplicial", "tetra"]]),
        ([2, 0, 1], [["simplicial", "simplicial", "trivial"], ["simplicial", "simplicial", "trivial"]]),
        ([2, 1, 1], [["simplicial", "simplicial", "tetra", "trivial"], ["simplicial", "simplicial", "tetra", "trivial"]]),
    ],
)
def test_expand_hybrid_schedule_single_mix(block_mix, expected):
    assert expand_hybrid_schedule(2, block_mix) == expected


def test_expand_hybrid_schedule_round_robin_and_explicit():
    assert expand_hybrid_schedule(4, [[2, 1, 0], [2, 0, 1]]) == [
        ["simplicial", "simplicial", "tetra"],
        ["simplicial", "simplicial", "trivial"],
        ["simplicial", "simplicial", "tetra"],
        ["simplicial", "simplicial", "trivial"],
    ]
    assert expand_hybrid_schedule(
        3,
        [1, 0, 0],
        order_policy="explicit",
        explicit_orders=[["S", "T"], ["S", "I"]],
    ) == [["simplicial", "tetra"], ["simplicial", "trivial"], ["simplicial", "tetra"]]


def test_platonic_group_and_linear_equivariance():
    group = PLATONIC_GROUPS["tetrahedron"]
    assert group.G == 12
    layer = PlatonicLinear(12 * 3, 12 * 5, solid="tetrahedron")
    x = torch.randn(2, 12, 3)
    y = layer(x.reshape(2, -1)).view(2, 12, 5)
    for group_idx in (0, 3, 7):
        permutation = group.cayley_table[group_idx]
        lhs = layer(x[:, permutation].reshape(2, -1)).view(2, 12, 5)
        rhs = y[:, permutation]
        assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_platonic_block_dense_shape():
    torch.manual_seed(0)
    block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
    )
    x = torch.randn(2, 5, 12 * 4, requires_grad=True)
    pos = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    out = block(x, pos=pos, pad_mask=pad_mask)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert x.grad is not None


def test_trivial_global_layer_position_encoding_modes():
    kwargs = dict(
        d_model=8,
        n_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        eps=1e-6,
        geometry_adapter=NonPeriodicGeometryAdapter(),
        use_adaln_conditioning=False,
    )
    edge_delta = TrivialGlobalLayer(**kwargs, position_encoding="edge_delta_bias")
    assert edge_delta.geometry_bias is not None
    assert edge_delta.geometry_bias.use_distance_bias
    assert edge_delta.geometry_bias.use_edge_bias

    distance = TrivialGlobalLayer(**kwargs, position_encoding="distance_bias")
    assert distance.geometry_bias is not None
    assert distance.geometry_bias.use_distance_bias
    assert not distance.geometry_bias.use_edge_bias

    none = TrivialGlobalLayer(**kwargs, position_encoding="none")
    assert none.geometry_bias is None


def test_trivial_global_layer_can_use_mha_rope_and_pair_rbf_config():
    torch.manual_seed(0)
    adapter = NonPeriodicGeometryAdapter()
    layer = TrivialGlobalLayer(
        d_model=12,
        n_heads=3,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        eps=1e-6,
        geometry_adapter=adapter,
        use_adaln_conditioning=False,
        position_encoding="edge_delta_bias",
        mha_position_mode="rope",
        mha_rope_freq_sigma=4.0,
        mha_rope_learned_freqs=True,
        mha_rope_use_key=False,
        mha_rope_on_values=True,
        pair_hidden_dim=32,
        pair_n_rbf=7,
        pair_rbf_max=4.0,
    )
    assert layer.block.mha_position_mode == "rope"
    assert layer.geometry_bias is not None
    assert layer.geometry_bias._edge_rbf_centers.numel() == 7
    assert torch.isclose(layer.geometry_bias._edge_rbf_centers[-1], torch.tensor(4.0))

    scalar = torch.randn(2, 4, 12)
    coords = torch.randn(2, 4, 3)
    mask = torch.tensor([[False, False, False, True], [False, False, False, False]])
    geom_features = adapter(coords=coords, pad_mask=mask)
    state = ModelState(
        pos=coords,
        mask=mask,
        scalar=scalar,
        group=None,
        geom=build_geometry_cache(geom_features, coords_len=4, seq_len=4, k_neighbors=3, pad_mask=mask),
        sigma=torch.tensor([0.1, 1.0]),
    )
    out = layer(state).scalar
    assert out is not None
    assert out.shape == scalar.shape
    assert torch.all(out[mask] == 0)


def test_hybrid_coupling_pre_tetra_refreshes_group_without_scalar_injection_at_init():
    torch.manual_seed(0)
    coupling = HybridCoupling(
        scalar_dim=4,
        group_order=2,
        group_dim_per_frame=3,
        config={
            "scalar_to_group": {"enabled": True, "pre_gate_init": 1.0, "gate_init": 0.0},
            "group_to_scalar": {"enabled": True, "gate_init": 0.0},
        },
    )
    scalar = torch.randn(1, 5, 4)
    state = ModelState(
        pos=torch.zeros(1, 5, 3),
        mask=None,
        scalar=scalar.clone(),
        group=torch.zeros(1, 5, 2, 3),
        geom=None,
    )
    state = coupling.scalar_to_group_pre(state)
    assert state.group is not None
    assert not torch.allclose(state.group, torch.zeros_like(state.group))
    state = coupling.group_to_scalar_post(state)
    assert torch.allclose(state.scalar, scalar)


def _make_neighbor_idx(num_tokens: int, k_neighbors: int) -> torch.Tensor:
    idx = torch.arange(k_neighbors) % num_tokens
    return idx.view(1, 1, k_neighbors).expand(1, num_tokens, k_neighbors).clone()


@pytest.mark.parametrize("k_neighbors", [8, 16, 32])
@pytest.mark.parametrize("rank", [16, 32])
@pytest.mark.parametrize("with_message", [False, True])
def test_compact_simplicial_triton_entrypoint_matches_reference(k_neighbors, rank, with_message):
    torch.manual_seed(0)
    batch_size, num_heads, num_tokens, head_dim = 1, 2, k_neighbors, 4
    kwargs = {"requires_grad": True}
    q = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    k1 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    v1 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    k2 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    v2 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    neighbor_idx = _make_neighbor_idx(num_tokens, k_neighbors)
    neighbor_mask = torch.ones(batch_size, num_tokens, k_neighbors, dtype=torch.bool)
    bias = CompactSimplicialBias(
        u=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        v=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        angle_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        angle_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        angle_gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        message_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs) if with_message else None,
        message_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs) if with_message else None,
        message_basis=torch.randn(num_heads, rank, head_dim, **kwargs) if with_message else None,
    )

    ref = compact_simplicial_attention_torch(
        q,
        k1,
        v1,
        k2,
        v2,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=bias,
    )
    actual = compact_simplicial_attention_triton(
        q,
        k1,
        v1,
        k2,
        v2,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=bias,
    )
    assert torch.allclose(actual, ref, atol=0.0, rtol=0.0)
    actual.square().mean().backward()
    assert q.grad is not None


def _clone_leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(tensor.requires_grad)


def _clone_bias(bias: CompactSimplicialBias) -> CompactSimplicialBias:
    return CompactSimplicialBias(
        u=_clone_leaf(bias.u) if bias.u is not None else None,
        v=_clone_leaf(bias.v) if bias.v is not None else None,
        gate=_clone_leaf(bias.gate) if bias.gate is not None else None,
        angle_left=_clone_leaf(bias.angle_left) if bias.angle_left is not None else None,
        angle_right=_clone_leaf(bias.angle_right) if bias.angle_right is not None else None,
        angle_gate=_clone_leaf(bias.angle_gate) if bias.angle_gate is not None else None,
        message_left=_clone_leaf(bias.message_left) if bias.message_left is not None else None,
        message_right=_clone_leaf(bias.message_right) if bias.message_right is not None else None,
        message_basis=_clone_leaf(bias.message_basis) if bias.message_basis is not None else None,
    )


def _bias_grad_tensors(bias: CompactSimplicialBias) -> list[torch.Tensor]:
    tensors = [
        bias.u,
        bias.v,
        bias.gate,
        bias.angle_left,
        bias.angle_right,
        bias.angle_gate,
        bias.message_left,
        bias.message_right,
        bias.message_basis,
    ]
    return [tensor for tensor in tensors if tensor is not None]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="compact Triton simplicial parity requires CUDA and Triton",
)
@pytest.mark.parametrize("with_message", [False, True])
def test_compact_simplicial_triton_cuda_forward_backward_matches_reference(with_message):
    torch.manual_seed(123)
    device = torch.device("cuda")
    batch_size, num_heads, num_tokens, k_neighbors, head_dim, rank = 2, 3, 9, 8, 16, 16
    kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
    q = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    k1 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    v1 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    k2 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)
    v2 = torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs)

    base_idx = torch.arange(num_tokens, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :]
    neighbor_idx = ((base_idx + offsets) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.15).contiguous()
    neighbor_mask[..., 0] = True
    neighbor_mask[0, 0, :] = False

    bias = CompactSimplicialBias(
        u=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        v=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        angle_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        angle_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        angle_gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        message_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs) if with_message else None,
        message_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs) if with_message else None,
        message_basis=torch.randn(num_heads, rank, head_dim, **kwargs) if with_message else None,
    )

    ref_tensors = [_clone_leaf(t) for t in (q, k1, v1, k2, v2)]
    tri_tensors = [_clone_leaf(t) for t in (q, k1, v1, k2, v2)]
    ref_bias = _clone_bias(bias)
    tri_bias = _clone_bias(bias)

    ref = compact_simplicial_attention_torch(
        *ref_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=ref_bias,
    )
    actual = compact_simplicial_attention_triton(
        *tri_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=tri_bias,
        precision="ieee_fp32",
        strict=True,
    )
    assert torch.allclose(actual, ref, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    actual.backward(grad)
    for actual_tensor, ref_tensor in zip(tri_tensors, ref_tensors):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)
    for actual_tensor, ref_tensor in zip(_bias_grad_tensors(tri_bias), _bias_grad_tensors(ref_bias)):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)


def _random_rotation() -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_compact_simplicial_geometry_bias_is_rotation_invariant():
    torch.manual_seed(0)
    coords = torch.randn(1, 8, 3)
    pad_mask = torch.zeros(1, 8, dtype=torch.bool)
    rotation = _random_rotation()
    rotated_coords = torch.einsum("ij,bnj->bni", rotation, coords)
    adapter = NonPeriodicGeometryAdapter()
    geom = build_geometry_cache(
        adapter(coords=coords, pad_mask=pad_mask),
        coords_len=coords.shape[1],
        seq_len=coords.shape[1],
        k_neighbors=4,
        pad_mask=pad_mask,
        rbf_dim=8,
    )
    rotated_geom = build_geometry_cache(
        adapter(coords=rotated_coords, pad_mask=pad_mask),
        coords_len=coords.shape[1],
        seq_len=coords.shape[1],
        k_neighbors=4,
        pad_mask=pad_mask,
        rbf_dim=8,
    )
    assert torch.equal(geom.neighbor_idx, rotated_geom.neighbor_idx)

    bias = CompactSimplicialGeometryBias(num_heads=2, rank=8, rbf_dim=8)
    original = bias(geom, dtype=torch.float32)
    rotated = bias(rotated_geom, dtype=torch.float32)
    angle_original = torch.einsum("bhnjr,bhnkr->bhnjk", original.angle_left, original.angle_right)
    angle_rotated = torch.einsum("bhnjr,bhnkr->bhnjk", rotated.angle_left, rotated.angle_right)
    assert torch.allclose(original.u, rotated.u, atol=1e-5, rtol=1e-5)
    assert torch.allclose(original.v, rotated.v, atol=1e-5, rtol=1e-5)
    assert torch.allclose(angle_original, angle_rotated, atol=1e-5, rtol=1e-5)


def test_feature_gated_spherical_low_rank_is_not_silently_accepted():
    with pytest.raises(NotImplementedError, match="feature_gated_spherical_low_rank"):
        CompactSimplicialAttention(
            16,
            4,
            bias_config={"kind": "feature_gated_spherical_low_rank", "angle_rank": 8, "radial_basis_dim": 8},
        )


def test_compact_simplicial_geometry_bias_can_disable_radial_or_angle():
    torch.manual_seed(0)
    coords = torch.randn(1, 6, 3)
    pad_mask = torch.zeros(1, 6, dtype=torch.bool)
    adapter = NonPeriodicGeometryAdapter()
    geom = build_geometry_cache(
        adapter(coords=coords, pad_mask=pad_mask),
        coords_len=coords.shape[1],
        seq_len=coords.shape[1],
        k_neighbors=4,
        pad_mask=pad_mask,
        rbf_dim=8,
    )

    angle_only = CompactSimplicialGeometryBias(num_heads=2, rank=8, rbf_dim=8, use_radial_uv=False, use_angle=True)
    angle_bias = angle_only(geom, dtype=torch.float32)
    assert angle_bias.u is None
    assert angle_bias.v is None
    assert angle_bias.gate is None
    assert angle_bias.angle_left is not None
    assert angle_bias.angle_right is not None
    assert angle_bias.angle_gate is not None

    radial_only = CompactSimplicialGeometryBias(num_heads=2, rank=8, rbf_dim=8, use_radial_uv=True, use_angle=False)
    radial_bias = radial_only(geom, dtype=torch.float32)
    assert radial_bias.u is not None
    assert radial_bias.v is not None
    assert radial_bias.gate is not None
    assert radial_bias.angle_left is None
    assert radial_bias.angle_right is None
    assert radial_bias.angle_gate is None


def _hybrid_config() -> dict:
    return {
        "num_blocks": 1,
        "block_mix": [1, 1, 1],
        "tetra_dim_per_frame": 4,
        "simplicial": {"k_neighbors": 3, "bias": {"angle_rank": 8, "radial_basis_dim": 8}},
    }


def test_qm9_hybrid_edm_forward_backward():
    torch.manual_seed(0)
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="hybrid",
        hybrid_config=_hybrid_config(),
        geometry_adapter=NonPeriodicGeometryAdapter(),
    )
    atom = torch.randn(2, 4, 6)
    coords = torch.randn(2, 4, 3)
    pad_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    atom_delta, coord_delta = model(atom, coords, pad_mask, torch.tensor([0.1, 1.0]))
    assert atom_delta.shape == atom.shape
    assert coord_delta.shape == coords.shape
    (atom_delta.square().mean() + coord_delta.square().mean()).backward()


def test_geom_drugs_hybrid_edm_forward():
    torch.manual_seed(0)
    model = GeomDrugsEDMModel(d_model=32, n_heads=4, n_layers=2, attn_type="hybrid", hybrid_config=_hybrid_config())
    atom = torch.randn(2, 5, model.atom_channels)
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    atom_delta, coord_delta = model(atom, coords, pad_mask, torch.tensor([0.1, 1.0]))
    assert atom_delta.shape == atom.shape
    assert coord_delta.shape == coords.shape


def test_mof_stage1_hybrid_edm_forward_backward():
    torch.manual_seed(0)
    model = MOFStage1EDMModel(
        block_feature_dim=5,
        d_model=32,
        n_heads=4,
        n_layers=2,
        attn_type="hybrid",
        hybrid_config={**_hybrid_config(), "simplicial": {"k_neighbors": 2, "bias": {"angle_rank": 8, "radial_basis_dim": 8}}},
    )
    block_features = torch.rand(2, 3, 5)
    block_types = torch.zeros(2, 3, dtype=torch.long)
    coords = torch.rand(2, 3, 3)
    pad_mask = torch.tensor([[False, False, True], [False, False, False]])
    coord_delta, lattice_delta = model(
        block_features,
        block_types,
        coords,
        pad_mask,
        torch.tensor([0.1, 1.0]),
        lattice=torch.zeros(2, 6),
    )
    assert coord_delta.shape == coords.shape
    assert lattice_delta.shape == (2, 6)
    (coord_delta.square().mean() + lattice_delta.square().mean()).backward()
