import pytest
import torch
import torch.nn.functional as F

from matterformer.geometry import NonPeriodicGeometryAdapter
from matterformer.models import (
    CompactSimplicialBias,
    CompactSimplicialAttention,
    CompactSimplicialGeometryBias,
    GeomDrugsEDMModel,
    GroupFramewiseSimplicialLayer,
    MOFStage1EDMModel,
    ModelState,
    QM9EDMModel,
    HybridConfig,
    HybridTransformerTrunk,
    TrivialGlobalLayer,
    build_geometry_cache,
    compact_simplicial_attention_torch,
    compact_simplicial_attention_triton,
    expand_hybrid_schedule,
)
from matterformer.models.platonic import PLATONIC_GROUPS, PlatonicBlock, PlatonicLinear
from matterformer.models.platonic.triton_attention import (
    TRITON_PLATONIC_ATTENTION_AVAILABLE,
    platonic_attention_flat_torch_reference,
    platonic_attention_flat_triton,
)
from matterformer.models.platonic.layers import flash_attn_varlen_func
from matterformer.models.triton_compact_simplicial_attention import TRITON_COMPACT_SIMPLICIAL_AVAILABLE
from matterformer.models.triton_grouped_compact_simplicial_attention import (
    TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE,
    _expand_compact_spherical_coefficients,
    _spherical_basis_lmax2,
    grouped_compact_simplicial_attention_torch_reference,
    triton_grouped_compact_simplicial_attention,
)


class CountingNonPeriodicGeometryAdapter(NonPeriodicGeometryAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return super().forward(*args, **kwargs)


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


def _manual_flat_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    outputs = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        q_seg = q[start:end].transpose(0, 1).unsqueeze(0)
        k_seg = k[start:end].transpose(0, 1).unsqueeze(0)
        v_seg = v[start:end].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_seg, k_seg, v_seg, dropout_p=0.0)
        outputs.append(out.squeeze(0).transpose(0, 1).contiguous())
    return torch.cat(outputs, dim=0)


def test_platonic_flat_attention_reference_matches_sdpa():
    torch.manual_seed(101)
    q = torch.randn(5, 3, 4, requires_grad=True)
    k = torch.randn(5, 3, 4, requires_grad=True)
    v = torch.randn(5, 3, 4, requires_grad=True)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    ref = platonic_attention_flat_torch_reference(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=3)
    manual = _manual_flat_sdpa(q, k, v, cu_seqlens)
    torch.testing.assert_close(ref, manual, atol=1e-6, rtol=1e-6)

    ref.square().sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_platonic_radial_rbf_zero_init_is_noop_and_weight_gets_grad():
    torch.manual_seed(102)
    q = torch.randn(6, 4, 4, requires_grad=True)
    k = torch.randn(6, 4, 4, requires_grad=True)
    v = torch.randn(6, 4, 4, requires_grad=True)
    pos = torch.randn(6, 3)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    centers = torch.linspace(0.0, 6.0, 8)
    gamma = torch.tensor(1.0)
    rbf_weight = torch.zeros(2, 8, requires_grad=True)
    gate = torch.zeros(2, requires_grad=True)

    unbiased = platonic_attention_flat_torch_reference(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=3)
    biased = platonic_attention_flat_torch_reference(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        pos=pos,
        heads_per_frame=2,
        rbf_weight=rbf_weight,
        gate=gate,
        centers=centers,
        gamma=gamma,
        diag_zero=True,
    )
    torch.testing.assert_close(biased, unbiased, atol=1e-7, rtol=1e-7)
    biased.square().sum().backward()
    assert rbf_weight.grad is not None
    assert torch.count_nonzero(rbf_weight.grad.abs() > 0) > 0


@pytest.mark.parametrize("radial_bias_kind", ["radial_r2", "radial_slope"])
def test_platonic_cheap_radial_zero_init_is_noop_and_weight_gets_grad(radial_bias_kind):
    torch.manual_seed(109)
    q = torch.randn(6, 4, 4, requires_grad=True)
    k = torch.randn(6, 4, 4, requires_grad=True)
    v = torch.randn(6, 4, 4, requires_grad=True)
    pos = torch.randn(6, 3)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    centers = torch.zeros(1)
    gamma = torch.tensor(1.0)
    rbf_weight = torch.zeros(2, 1, requires_grad=True)
    gate = torch.zeros(2, requires_grad=True)

    unbiased = platonic_attention_flat_torch_reference(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=3)
    biased = platonic_attention_flat_torch_reference(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        pos=pos,
        heads_per_frame=2,
        rbf_weight=rbf_weight,
        gate=gate,
        centers=centers,
        gamma=gamma,
        radial_bias_kind=radial_bias_kind,
        diag_zero=True,
    )
    torch.testing.assert_close(biased, unbiased, atol=1e-7, rtol=1e-7)
    biased.square().sum().backward()
    assert rbf_weight.grad is not None
    assert torch.count_nonzero(rbf_weight.grad.abs() > 0) > 0


def test_platonic_rbf_type_enveloped_zero_init_is_noop_and_weights_get_grad():
    torch.manual_seed(112)
    q = torch.randn(6, 4, 4, requires_grad=True)
    k = torch.randn(6, 4, 4, requires_grad=True)
    v = torch.randn(6, 4, 4, requires_grad=True)
    pos = torch.randn(6, 3)
    atom_types = torch.tensor([1, 6, 8, 1, 7, 6])
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    centers = torch.linspace(0.0, 6.0, 4)
    gamma = torch.tensor(0.75)
    rbf_weight = torch.zeros(2, 4, requires_grad=True)
    type_bias = torch.zeros(9, 9, 2, requires_grad=True)

    unbiased = platonic_attention_flat_torch_reference(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=3)
    biased = platonic_attention_flat_torch_reference(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        pos=pos,
        atom_types=atom_types,
        heads_per_frame=2,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=6.0,
        radial_bias_kind="rbf_type_enveloped",
        diag_zero=True,
    )
    torch.testing.assert_close(biased, unbiased, atol=1e-7, rtol=1e-7)
    biased.square().sum().backward()
    assert rbf_weight.grad is not None
    assert type_bias.grad is not None
    assert torch.count_nonzero(rbf_weight.grad.abs() > 0) > 0
    assert torch.count_nonzero(type_bias.grad.abs() > 0) > 0


def test_platonic_radial_rbf_group_shared_bias_commutes_with_group_permutation():
    torch.manual_seed(103)
    group = PLATONIC_GROUPS["tetrahedron"]
    num_tokens = 5
    group_order = group.G
    heads_per_frame = 2
    head_dim = 4
    q = torch.randn(num_tokens, group_order, heads_per_frame, head_dim)
    k = torch.randn(num_tokens, group_order, heads_per_frame, head_dim)
    v = torch.randn(num_tokens, group_order, heads_per_frame, head_dim)
    pos = torch.randn(num_tokens, 3)
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)
    rbf_weight = torch.randn(heads_per_frame, 4) * 0.01
    gate = torch.zeros(heads_per_frame)
    centers = torch.linspace(0.0, 6.0, 4)
    gamma = torch.tensor(0.75)

    out = platonic_attention_flat_torch_reference(
        q.reshape(num_tokens, group_order * heads_per_frame, head_dim),
        k.reshape(num_tokens, group_order * heads_per_frame, head_dim),
        v.reshape(num_tokens, group_order * heads_per_frame, head_dim),
        cu_seqlens=cu_seqlens,
        max_seqlen=num_tokens,
        pos=pos,
        heads_per_frame=heads_per_frame,
        rbf_weight=rbf_weight,
        gate=gate,
        centers=centers,
        gamma=gamma,
    ).view(num_tokens, group_order, heads_per_frame, head_dim)
    perm = group.cayley_table[3]
    out_perm = platonic_attention_flat_torch_reference(
        q[:, perm].reshape(num_tokens, group_order * heads_per_frame, head_dim),
        k[:, perm].reshape(num_tokens, group_order * heads_per_frame, head_dim),
        v[:, perm].reshape(num_tokens, group_order * heads_per_frame, head_dim),
        cu_seqlens=cu_seqlens,
        max_seqlen=num_tokens,
        pos=pos,
        heads_per_frame=heads_per_frame,
        rbf_weight=rbf_weight,
        gate=gate,
        centers=centers,
        gamma=gamma,
    ).view(num_tokens, group_order, heads_per_frame, head_dim)
    torch.testing.assert_close(out_perm, out[:, perm], atol=1e-6, rtol=1e-6)


def test_platonic_rbf_type_enveloped_group_shared_bias_commutes_with_group_permutation():
    torch.manual_seed(113)
    group = PLATONIC_GROUPS["tetrahedron"]
    num_tokens = 5
    group_order = group.G
    heads_per_frame = 2
    head_dim = 4
    q = torch.randn(num_tokens, group_order, heads_per_frame, head_dim)
    k = torch.randn(num_tokens, group_order, heads_per_frame, head_dim)
    v = torch.randn(num_tokens, group_order, heads_per_frame, head_dim)
    pos = torch.randn(num_tokens, 3)
    atom_types = torch.tensor([1, 6, 8, 7, 1])
    cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32)
    rbf_weight = torch.randn(heads_per_frame, 4) * 0.01
    type_bias = torch.randn(9, 9, heads_per_frame) * 0.01
    centers = torch.linspace(0.0, 6.0, 4)
    gamma = torch.tensor(0.75)

    out = platonic_attention_flat_torch_reference(
        q.reshape(num_tokens, group_order * heads_per_frame, head_dim),
        k.reshape(num_tokens, group_order * heads_per_frame, head_dim),
        v.reshape(num_tokens, group_order * heads_per_frame, head_dim),
        cu_seqlens=cu_seqlens,
        max_seqlen=num_tokens,
        pos=pos,
        atom_types=atom_types,
        heads_per_frame=heads_per_frame,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=6.0,
        radial_bias_kind="rbf_type_enveloped",
    ).view(num_tokens, group_order, heads_per_frame, head_dim)
    perm = group.cayley_table[3]
    out_perm = platonic_attention_flat_torch_reference(
        q[:, perm].reshape(num_tokens, group_order * heads_per_frame, head_dim),
        k[:, perm].reshape(num_tokens, group_order * heads_per_frame, head_dim),
        v[:, perm].reshape(num_tokens, group_order * heads_per_frame, head_dim),
        cu_seqlens=cu_seqlens,
        max_seqlen=num_tokens,
        pos=pos,
        atom_types=atom_types,
        heads_per_frame=heads_per_frame,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=6.0,
        radial_bias_kind="rbf_type_enveloped",
    ).view(num_tokens, group_order, heads_per_frame, head_dim)
    torch.testing.assert_close(out_perm, out[:, perm], atol=1e-6, rtol=1e-6)


def test_platonic_block_flat_triton_backends_match_flash_zero_init():
    torch.manual_seed(104)
    flash = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="flash",
    )
    triton_block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="triton",
        attention_bias={"precision": "tf32x3", "strict": True},
    )
    radial = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="triton",
        attention_bias={"kind": "radial_rbf", "num_rbf": 8, "zero_init": True},
    )
    triton_block.load_state_dict(flash.state_dict())
    radial.load_state_dict(flash.state_dict(), strict=False)
    x = torch.randn(5, 12 * 4)
    pos = torch.randn(5, 3)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    flash.eval()
    triton_block.eval()
    radial.eval()
    with torch.no_grad():
        flash_out = flash.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=3)
        triton_out = triton_block.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=3)
        radial_out = radial.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=3)
    torch.testing.assert_close(triton_out, flash_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(radial_out, triton_out, atol=1e-6, rtol=1e-6)


def test_platonic_block_flat_rbf_type_bias_torch_reference_matches_triton_zero_init():
    torch.manual_seed(114)
    triton_block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="triton",
        attention_bias={"precision": "tf32x3", "strict": True},
    )
    local = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="torch_reference",
        attention_bias={
            "kind": "rbf_type_enveloped",
            "num_rbf": 4,
            "cutoff": 6.0,
            "max_atomic_number": 8,
            "zero_init": True,
        },
    )
    local_triton = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="triton",
        attention_bias={
            "kind": "rbf_type_enveloped",
            "num_rbf": 4,
            "cutoff": 6.0,
            "max_atomic_number": 8,
            "zero_init": True,
        },
    )
    local.load_state_dict(triton_block.state_dict(), strict=False)
    local_triton.load_state_dict(triton_block.state_dict(), strict=False)
    x = torch.randn(5, 12 * 4)
    pos = torch.randn(5, 3)
    atom_types = torch.tensor([1, 6, 8, 1, 7])
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    triton_block.eval()
    local.eval()
    local_triton.eval()
    with torch.no_grad():
        triton_out = triton_block.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=3)
        local_out = local.forward_flat(
            x,
            pos=pos,
            atom_types=atom_types,
            cu_seqlens=cu_seqlens,
            max_seqlen=3,
        )
        local_triton_out = local_triton.forward_flat(
            x,
            pos=pos,
            atom_types=atom_types,
            cu_seqlens=cu_seqlens,
            max_seqlen=3,
        )
    torch.testing.assert_close(local_out, triton_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local_triton_out, triton_out, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("backend", "bias", "expected_backend", "expected_kind"),
    [
        ("triton", {"kind": "rbf_type_enveloped"}, "triton", "rbf_type_enveloped"),
        ("triton_rbf_type_bias", {}, "triton", "rbf_type_enveloped"),
        ("triton_radial_r2", {}, "triton", "radial_r2"),
        ("torch_reference", {"kind": "rbf_type_enveloped"}, "torch_reference", "rbf_type_enveloped"),
        ("torch_rbf_type_bias", {}, "torch_reference", "rbf_type_enveloped"),
    ],
)
def test_platonic_attention_backend_bias_names_normalize(backend, bias, expected_backend, expected_kind):
    block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend=backend,
        attention_bias=bias,
    )
    assert block.attn.attention_backend == expected_backend
    assert block.attn.attention_bias_config["kind"] == expected_kind
    assert block.attn.radial_bias_kind == expected_kind


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE, reason="requires CUDA and Triton")
def test_platonic_flat_triton_cuda_matches_reference_forward_backward():
    torch.manual_seed(105)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)

    out = platonic_attention_flat_triton(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=4, strict=True, precision="tf32x3")
    ref = platonic_attention_flat_torch_reference(q_ref, k_ref, v_ref, cu_seqlens=cu_seqlens, max_seqlen=4)
    torch.testing.assert_close(out, ref, atol=5e-5, rtol=5e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE or flash_attn_varlen_func is None,
    reason="requires CUDA, Triton, and FlashAttention",
)
def test_platonic_block_flat_triton_cuda_matches_flash_backend():
    torch.manual_seed(106)
    device = torch.device("cuda")
    flash = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="flash",
    ).to(device)
    triton_block = PlatonicBlock(
        d_model=12 * 4,
        nhead=12,
        dim_feedforward=12 * 8,
        solid_name="tetrahedron",
        dropout=0.0,
        attention_backend="triton",
        attention_bias={"precision": "tf32x3", "strict": True},
    ).to(device)
    triton_block.load_state_dict(flash.state_dict())
    x = torch.randn(7, 12 * 4, device=device)
    pos = torch.randn(7, 3, device=device)
    cu_seqlens = torch.tensor([0, 3, 7], device=device, dtype=torch.int32)
    flash.eval()
    triton_block.eval()
    with torch.no_grad():
        flash_out = flash.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=4)
        triton_out = triton_block.forward_flat(x, pos=pos, cu_seqlens=cu_seqlens, max_seqlen=4)
    torch.testing.assert_close(triton_out, flash_out, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE, reason="requires CUDA and Triton")
def test_platonic_flat_triton_radial_cuda_matches_reference_forward_backward():
    torch.manual_seed(107)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pos = torch.randn(6, 3, device=device)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    centers = torch.linspace(0.0, 6.0, 4, device=device)
    gamma = torch.tensor(0.75, device=device)
    weight = (torch.randn(2, 4, device=device) * 0.01).requires_grad_(True)
    gate = torch.zeros(2, device=device, requires_grad=True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    gate_ref = gate.detach().clone().requires_grad_(True)

    out = platonic_attention_flat_triton(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        pos=pos,
        heads_per_frame=2,
        rbf_weight=weight,
        gate=gate,
        centers=centers,
        gamma=gamma,
        strict=True,
        precision="tf32x3",
    )
    ref = platonic_attention_flat_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        pos=pos,
        heads_per_frame=2,
        rbf_weight=weight_ref,
        gate=gate_ref,
        centers=centers,
        gamma=gamma,
    )
    torch.testing.assert_close(out, ref, atol=5e-5, rtol=5e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(gate.grad, gate_ref.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE, reason="requires CUDA and Triton")
def test_platonic_flat_triton_rbf_type_cuda_matches_reference_forward_backward():
    torch.manual_seed(115)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pos = torch.randn(6, 3, device=device)
    atom_types = torch.tensor([1, 6, 8, 1, 7, 6], device=device)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    centers = torch.linspace(0.0, 6.0, 4, device=device)
    gamma = torch.tensor(0.75, device=device)
    weight = (torch.randn(2, 4, device=device) * 0.01).requires_grad_(True)
    type_bias = (torch.randn(9, 9, 2, device=device) * 0.01).requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    type_bias_ref = type_bias.detach().clone().requires_grad_(True)

    out = platonic_attention_flat_triton(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        pos=pos,
        atom_types=atom_types,
        heads_per_frame=2,
        rbf_weight=weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=6.0,
        max_atomic_number=8,
        radial_bias_kind="rbf_type_enveloped",
        strict=True,
        precision="ieee",
    )
    ref = platonic_attention_flat_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        pos=pos,
        atom_types=atom_types,
        heads_per_frame=2,
        rbf_weight=weight_ref,
        type_bias=type_bias_ref,
        centers=centers,
        gamma=gamma,
        cutoff=6.0,
        radial_bias_kind="rbf_type_enveloped",
    )
    torch.testing.assert_close(out, ref, atol=5e-5, rtol=5e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-5, rtol=5e-5)
    # The DKV kernel accumulates over tiled softmax probabilities, so dK/dV
    # have slightly larger absolute drift than the forward pass and parameter
    # gradients even with IEEE dot precision.
    torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(type_bias.grad, type_bias_ref.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE, reason="requires CUDA and Triton")
@pytest.mark.parametrize("radial_bias_kind", ["radial_r2", "radial_slope"])
def test_platonic_flat_triton_cheap_radial_cuda_matches_reference_forward_backward(radial_bias_kind):
    torch.manual_seed(110)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    pos = torch.randn(6, 3, device=device)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    centers = torch.zeros(1, device=device)
    gamma = torch.ones((), device=device)
    weight = (torch.randn(2, 1, device=device) * 0.01).requires_grad_(True)
    gate = torch.zeros(2, device=device, requires_grad=True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    gate_ref = gate.detach().clone().requires_grad_(True)

    out = platonic_attention_flat_triton(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        pos=pos,
        heads_per_frame=2,
        rbf_weight=weight,
        gate=gate,
        centers=centers,
        gamma=gamma,
        radial_bias_kind=radial_bias_kind,
        strict=True,
        precision="tf32x3",
    )
    ref = platonic_attention_flat_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        pos=pos,
        heads_per_frame=2,
        rbf_weight=weight_ref,
        gate=gate_ref,
        centers=centers,
        gamma=gamma,
        radial_bias_kind=radial_bias_kind,
    )
    torch.testing.assert_close(out, ref, atol=5e-5, rtol=5e-5)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(gate.grad, gate_ref.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE, reason="requires CUDA and Triton")
def test_platonic_flat_triton_bf16_flash_compat_runs_with_fp32_output():
    torch.manual_seed(111)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    out = platonic_attention_flat_triton(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        strict=True,
        precision="bf16_flash_compat",
    )
    assert out.dtype == torch.float32
    assert out.shape == q.shape
    out.square().mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_PLATONIC_ATTENTION_AVAILABLE, reason="requires CUDA and Triton")
def test_platonic_flat_triton_radial_cuda_allows_outer_torch_compile():
    torch.manual_seed(108)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(6, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    pos = torch.randn(6, 3, device=device)
    cu_seqlens = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    centers = torch.linspace(0.0, 6.0, 4, device=device)
    gamma = torch.tensor(0.75, device=device)
    weight = (torch.randn(2, 4, device=device) * 0.01).requires_grad_(True)
    gate = torch.zeros(2, device=device, requires_grad=True)

    def run_attention(q_in, k_in, v_in, weight_in, gate_in):
        return platonic_attention_flat_triton(
            q_in,
            k_in,
            v_in,
            cu_seqlens=cu_seqlens,
            max_seqlen=4,
            pos=pos,
            heads_per_frame=2,
            rbf_weight=weight_in,
            gate=gate_in,
            centers=centers,
            gamma=gamma,
            strict=True,
            precision="tf32x3",
        )

    compiled = torch.compile(run_attention, mode="default")
    out = compiled(q, k, v, weight, gate)
    assert out.shape == q.shape
    out.square().mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert weight.grad is not None


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
    assert not none.requires_geometry_cache


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


def test_hybrid_config_rejects_legacy_dual_stream_keys():
    with pytest.raises(ValueError, match="Dual-stream"):
        HybridConfig.from_input({"branch_mode": "two_stream"}, d_model=16, n_heads=4, n_layers=1)
    with pytest.raises(ValueError, match="Dual-stream"):
        HybridConfig.from_input({"coupling": {}}, d_model=16, n_heads=4, n_layers=1)


def test_hybrid_stream_schedule_validation():
    HybridTransformerTrunk(
        d_model=16,
        n_heads=4,
        n_layers=1,
        hybrid_config={"stream_type": "scalar", "num_blocks": 1, "block_mix": [1, 0, 1]},
        geometry_adapter=NonPeriodicGeometryAdapter(),
    )
    HybridTransformerTrunk(
        d_model=16,
        n_heads=4,
        n_layers=1,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 1,
            "block_mix": [1, 1, 0],
            "tetra_dim_per_frame": 4,
            "simplicial": {"num_heads": 1, "bias": {"angle_rank": 8, "radial_basis_dim": 8}},
        },
        geometry_adapter=NonPeriodicGeometryAdapter(),
    )
    with pytest.raises(ValueError, match="does not allow"):
        HybridTransformerTrunk(
            d_model=16,
            n_heads=4,
            n_layers=1,
            hybrid_config={"stream_type": "scalar", "num_blocks": 1, "block_mix": [0, 1, 0]},
            geometry_adapter=NonPeriodicGeometryAdapter(),
        )


def test_hybrid_trunk_skips_geometry_cache_for_pure_tetra_global():
    torch.manual_seed(0)
    adapter = CountingNonPeriodicGeometryAdapter()
    trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 1,
            "block_mix": [0, 1, 0],
            "tetra_dim_per_frame": 2,
            "tetra": {"heads_per_frame": 1, "rope_sigma": 1.0},
        },
        geometry_adapter=adapter,
    )
    assert not trunk.requires_geometry_cache

    x = torch.randn(2, 5, 24)
    coords = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = trunk(x, None, pad_mask=mask, coords=coords)

    assert out.shape == x.shape
    assert adapter.calls == 0


def test_hybrid_tetra_scheduled_local_attention_mod_selects_requested_layers():
    trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=4,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 4,
            "block_mix": [0, 1, 0],
            "tetra_dim_per_frame": 2,
            "tetra": {
                "heads_per_frame": 1,
                "rope_sigma": 1.0,
                "attention_backend": "flash",
                "local_attention_mod": {
                    "enabled": True,
                    "backend": "torch_reference",
                    "kind": "rbf_type_enveloped",
                    "every": 2,
                    "offset": 1,
                    "num_rbf": 4,
                    "cutoff": 6.0,
                    "max_atomic_number": 8,
                },
            },
        },
        geometry_adapter=NonPeriodicGeometryAdapter(),
    )
    backends = [
        layer.block.attn.attention_backend
        for block in trunk.blocks
        for layer in block.sublayers
        if hasattr(layer, "block")
    ]
    assert backends == ["flash", "torch_reference", "flash", "torch_reference"]


def test_hybrid_trunk_builds_geometry_cache_when_required():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 24)
    coords = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.bool)

    sg_adapter = CountingNonPeriodicGeometryAdapter()
    sg_trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 1,
            "block_mix": [1, 1, 0],
            "tetra_dim_per_frame": 2,
            "simplicial": {
                "k_neighbors": 2,
                "num_heads": 1,
                "head_dim": 2,
                "bias": {"angle_rank": 4, "radial_basis_dim": 4, "hidden_dim": 8},
                "kernel": {"backend": "torch"},
            },
            "tetra": {"heads_per_frame": 1, "rope_sigma": 1.0},
        },
        geometry_adapter=sg_adapter,
    )
    assert sg_trunk.requires_geometry_cache
    assert sg_trunk(x, None, pad_mask=mask, coords=coords).shape == x.shape
    assert sg_adapter.calls == 1

    moment_adapter = CountingNonPeriodicGeometryAdapter()
    moment_trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 1,
            "block_mix": [0, 1, 0],
            "tetra_dim_per_frame": 2,
            "input_lift": {"kind": "local_moment_lift", "hidden_dim": 8},
            "simplicial": {"k_neighbors": 2, "bias": {"radial_basis_dim": 4}},
            "tetra": {"heads_per_frame": 1, "rope_sigma": 1.0},
        },
        geometry_adapter=moment_adapter,
    )
    assert moment_trunk.requires_geometry_cache
    assert moment_trunk(x, None, pad_mask=mask, coords=coords).shape == x.shape
    assert moment_adapter.calls == 1

    trivial_adapter = CountingNonPeriodicGeometryAdapter()
    trivial_trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=6,
        n_layers=1,
        hybrid_config={
            "stream_type": "scalar",
            "num_blocks": 1,
            "block_mix": [0, 0, 1],
            "trivial": {"attention": {"num_heads": 6, "position_encoding": "distance_bias"}},
        },
        geometry_adapter=trivial_adapter,
        use_adaln_conditioning=False,
    )
    assert trivial_trunk.requires_geometry_cache
    assert trivial_trunk(x, None, pad_mask=mask, coords=coords, sigma=torch.ones(2)).shape == x.shape
    assert trivial_adapter.calls == 1
    with pytest.raises(ValueError, match="does not allow"):
        HybridTransformerTrunk(
            d_model=16,
            n_heads=4,
            n_layers=1,
            hybrid_config={"stream_type": "tetra", "num_blocks": 1, "block_mix": [0, 0, 1]},
            geometry_adapter=NonPeriodicGeometryAdapter(),
        )


def test_tetra_platonic_linear_input_and_ffn_readout_path_shape():
    torch.manual_seed(0)
    trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        input_dim=7,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 1,
            "block_mix": [0, 1, 0],
            "tetra_dim_per_frame": 2,
            "input_lift": {"kind": "platonic_linear"},
            "readout": {"kind": "platonic_ffn"},
            "tetra": {"heads_per_frame": 1, "rope_sigma": 4.0},
        },
        geometry_adapter=NonPeriodicGeometryAdapter(),
    )
    x = torch.randn(2, 4, 7)
    coords = torch.randn(2, 4, 3)
    mask = torch.tensor([[False, False, False, True], [False, False, False, False]])
    out = trunk(x, None, pad_mask=mask, coords=coords, return_output=True)
    assert out.stream_type == "tetra"
    assert out.group is not None
    assert out.group.shape == (2, 4, 12, 2)
    assert out.scalar.shape == (2, 4, 24)
    assert torch.all(out.group[mask] == 0)


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
        angle_left_coeff=_clone_leaf(bias.angle_left_coeff) if bias.angle_left_coeff is not None else None,
        angle_right_coeff=_clone_leaf(bias.angle_right_coeff) if bias.angle_right_coeff is not None else None,
        angle_channels_by_l=bias.angle_channels_by_l,
        angle_rank=bias.angle_rank,
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
        bias.angle_left_coeff,
        bias.angle_right_coeff,
        bias.angle_gate,
        bias.message_left,
        bias.message_right,
        bias.message_basis,
    ]
    return [tensor for tensor in tensors if tensor is not None]


def _repeat_group_for_test(tensor: torch.Tensor, group_order: int) -> torch.Tensor:
    return tensor[:, None].expand(-1, group_order, *tensor.shape[1:]).reshape(
        tensor.shape[0] * group_order, *tensor.shape[1:]
    )


def test_grouped_compact_simplicial_reference_matches_folded_reference():
    torch.manual_seed(7)
    batch_size, group_order, num_heads, num_tokens, k_neighbors, head_dim, rank = 2, 3, 2, 5, 3, 4, 8
    kwargs = {"requires_grad": True}
    q = torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs)
    k1 = torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs)
    v1 = torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs)
    k2 = torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs)
    v2 = torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs)
    base_idx = torch.arange(num_tokens)[:, None]
    offsets = torch.arange(k_neighbors)[None, :]
    neighbor_idx = ((base_idx + offsets) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = torch.ones(batch_size, num_tokens, k_neighbors, dtype=torch.bool)
    u = torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs)
    v_bias = torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs)
    gate = torch.randn(batch_size, num_heads, num_tokens, **kwargs)
    angle_left = torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs)
    angle_right = torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs)
    angle_gate = torch.randn(batch_size, num_heads, num_tokens, **kwargs)
    message_left = torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs)
    message_right = torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs)
    message_basis = torch.randn(num_heads, rank, head_dim, **kwargs)

    grouped = grouped_compact_simplicial_attention_torch_reference(
        q,
        k1,
        v1,
        k2,
        v2,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=u,
        v_bias=v_bias,
        gate=gate,
        angle_left=angle_left,
        angle_right=angle_right,
        angle_gate=angle_gate,
        message_left=message_left,
        message_right=message_right,
        message_basis=message_basis,
    )
    folded = compact_simplicial_attention_torch(
        q.reshape(batch_size * group_order, num_heads, num_tokens, head_dim),
        k1.reshape(batch_size * group_order, num_heads, num_tokens, head_dim),
        v1.reshape(batch_size * group_order, num_heads, num_tokens, head_dim),
        k2.reshape(batch_size * group_order, num_heads, num_tokens, head_dim),
        v2.reshape(batch_size * group_order, num_heads, num_tokens, head_dim),
        neighbor_idx=_repeat_group_for_test(neighbor_idx, group_order),
        neighbor_mask=_repeat_group_for_test(neighbor_mask, group_order),
        bias=CompactSimplicialBias(
            u=_repeat_group_for_test(u, group_order),
            v=_repeat_group_for_test(v_bias, group_order),
            gate=_repeat_group_for_test(gate, group_order),
            angle_left=_repeat_group_for_test(angle_left, group_order),
            angle_right=_repeat_group_for_test(angle_right, group_order),
            angle_gate=_repeat_group_for_test(angle_gate, group_order),
            message_left=_repeat_group_for_test(message_left, group_order),
            message_right=_repeat_group_for_test(message_right, group_order),
            message_basis=message_basis,
        ),
    ).view(batch_size, group_order, num_heads, num_tokens, head_dim)
    assert torch.allclose(grouped, folded, atol=1e-6, rtol=1e-6)


def test_grouped_compact_spherical_coefficients_match_expanded_reference():
    torch.manual_seed(11)
    batch_size, group_order, num_heads, num_tokens, k_neighbors, head_dim = 2, 3, 2, 7, 4, 8
    channels_by_l = (2, 3, 1)
    angle_rank = channels_by_l[0] + 3 * channels_by_l[1] + 5 * channels_by_l[2]
    coeff_dim = sum(channels_by_l)
    kwargs = {"dtype": torch.float64, "requires_grad": True}
    tensors = [torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens)[:, None]
    offsets = torch.arange(k_neighbors)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors) > 0.2).contiguous()
    neighbor_mask[..., 0] = True
    unit = torch.randn(batch_size, num_tokens, k_neighbors, 3, dtype=torch.float64)
    unit = torch.nn.functional.normalize(unit, dim=-1)
    unit = unit.masked_fill(~neighbor_mask[..., None], 0.0)
    bias_tensors = [
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
    ]
    exp_tensors = [_clone_leaf(t) for t in tensors]
    cmp_tensors = [_clone_leaf(t) for t in tensors]
    exp_bias = [_clone_leaf(t) for t in bias_tensors]
    cmp_bias = [_clone_leaf(t) for t in bias_tensors]
    basis = _spherical_basis_lmax2(unit)
    left_expanded = _expand_compact_spherical_coefficients(exp_bias[3], basis=basis, channels_by_l=channels_by_l)
    right_expanded = _expand_compact_spherical_coefficients(exp_bias[4], basis=basis, channels_by_l=channels_by_l)

    expanded = grouped_compact_simplicial_attention_torch_reference(
        *exp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=exp_bias[0],
        v_bias=exp_bias[1],
        gate=exp_bias[2],
        angle_left=left_expanded,
        angle_right=right_expanded,
        angle_gate=exp_bias[5],
    )
    compact = grouped_compact_simplicial_attention_torch_reference(
        *cmp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=cmp_bias[0],
        v_bias=cmp_bias[1],
        gate=cmp_bias[2],
        unit=unit,
        angle_left_coeff=cmp_bias[3],
        angle_right_coeff=cmp_bias[4],
        angle_channels_by_l=channels_by_l,
        angle_rank=angle_rank,
        angle_gate=cmp_bias[5],
    )
    assert torch.allclose(compact, expanded, atol=1e-6, rtol=1e-6)

    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    compact.backward(grad)
    for actual_tensor, ref_tensor in zip(cmp_tensors + cmp_bias, exp_tensors + exp_bias):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-5, rtol=1e-5)


def test_scalar_compact_spherical_coefficients_match_expanded_reference():
    torch.manual_seed(13)
    batch_size, num_heads, num_tokens, k_neighbors, head_dim = 2, 2, 7, 4, 8
    channels_by_l = (2, 3, 1)
    angle_rank = channels_by_l[0] + 3 * channels_by_l[1] + 5 * channels_by_l[2]
    coeff_dim = sum(channels_by_l)
    kwargs = {"dtype": torch.float64, "requires_grad": True}
    tensors = [torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens)[:, None]
    offsets = torch.arange(k_neighbors)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors) > 0.2).contiguous()
    neighbor_mask[..., 0] = True
    neighbor_mask[0, 0, :] = False
    unit = torch.randn(batch_size, num_tokens, k_neighbors, 3, dtype=torch.float64)
    unit = torch.nn.functional.normalize(unit, dim=-1)
    unit = unit.masked_fill(~neighbor_mask[..., None], 0.0)
    bias_tensors = [
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
    ]
    exp_tensors = [_clone_leaf(t) for t in tensors]
    cmp_tensors = [_clone_leaf(t) for t in tensors]
    exp_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    cmp_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    basis = _spherical_basis_lmax2(unit)
    left_expanded = _expand_compact_spherical_coefficients(
        exp_bias_tensors[3], basis=basis, channels_by_l=channels_by_l
    )
    right_expanded = _expand_compact_spherical_coefficients(
        exp_bias_tensors[4], basis=basis, channels_by_l=channels_by_l
    )
    expanded_bias = CompactSimplicialBias(
        u=exp_bias_tensors[0],
        v=exp_bias_tensors[1],
        gate=exp_bias_tensors[2],
        angle_left=left_expanded,
        angle_right=right_expanded,
        angle_gate=exp_bias_tensors[5],
    )
    compact_bias = CompactSimplicialBias(
        u=cmp_bias_tensors[0],
        v=cmp_bias_tensors[1],
        gate=cmp_bias_tensors[2],
        angle_left_coeff=cmp_bias_tensors[3],
        angle_right_coeff=cmp_bias_tensors[4],
        angle_channels_by_l=channels_by_l,
        angle_rank=angle_rank,
        angle_gate=cmp_bias_tensors[5],
    )

    expanded = compact_simplicial_attention_torch(
        *exp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=expanded_bias,
    )
    compact = compact_simplicial_attention_torch(
        *cmp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=compact_bias,
        unit=unit,
    )
    assert torch.allclose(compact, expanded, atol=1e-6, rtol=1e-6)

    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    compact.backward(grad)
    for actual_tensor, ref_tensor in zip(cmp_tensors + cmp_bias_tensors, exp_tensors + exp_bias_tensors):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-5, rtol=1e-5)


def test_compact_simplicial_attention_ignores_invalid_bias_entries():
    torch.manual_seed(15)
    batch_size, num_heads, num_tokens, k_neighbors, head_dim, rank = 2, 2, 6, 4, 8, 8
    kwargs = {"dtype": torch.float64, "requires_grad": True}
    tensors = [torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens)[:, None]
    offsets = torch.arange(k_neighbors)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors) > 0.35).contiguous()
    neighbor_mask[..., 0] = True
    neighbor_mask[1, 2, :] = False
    raw_bias = CompactSimplicialBias(
        u=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        v=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        angle_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        angle_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        angle_gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        message_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        message_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        message_basis=torch.randn(num_heads, rank, head_dim, **kwargs),
    )
    edge_mask = neighbor_mask[:, None, :, :]
    edge_rank_mask = edge_mask[..., None]
    masked_bias = CompactSimplicialBias(
        u=raw_bias.u.masked_fill(~edge_mask, 0.0),
        v=raw_bias.v.masked_fill(~edge_mask, 0.0),
        gate=raw_bias.gate,
        angle_left=raw_bias.angle_left.masked_fill(~edge_rank_mask, 0.0),
        angle_right=raw_bias.angle_right.masked_fill(~edge_rank_mask, 0.0),
        angle_gate=raw_bias.angle_gate,
        message_left=raw_bias.message_left.masked_fill(~edge_rank_mask, 0.0),
        message_right=raw_bias.message_right.masked_fill(~edge_rank_mask, 0.0),
        message_basis=raw_bias.message_basis,
    )
    raw = compact_simplicial_attention_torch(
        *tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=raw_bias,
    )
    masked = compact_simplicial_attention_torch(
        *tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=masked_bias,
    )
    assert torch.allclose(raw, masked, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="grouped compact Triton simplicial parity requires CUDA and Triton",
)
def test_grouped_compact_simplicial_triton_cuda_forward_backward_matches_reference():
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch_size, group_order, num_heads, num_tokens, k_neighbors, head_dim, rank = 2, 3, 2, 7, 4, 32, 16
    kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
    tensors = [torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :]
    neighbor_idx = ((base_idx + offsets) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.1).contiguous()
    neighbor_mask[..., 0] = True
    bias_tensors = [
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
        torch.randn(num_heads, rank, head_dim, **kwargs),
    ]
    ref_tensors = [_clone_leaf(t) for t in tensors]
    tri_tensors = [_clone_leaf(t) for t in tensors]
    ref_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    tri_bias_tensors = [_clone_leaf(t) for t in bias_tensors]

    ref = grouped_compact_simplicial_attention_torch_reference(
        *ref_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=ref_bias_tensors[0],
        v_bias=ref_bias_tensors[1],
        gate=ref_bias_tensors[2],
        angle_left=ref_bias_tensors[3],
        angle_right=ref_bias_tensors[4],
        angle_gate=ref_bias_tensors[5],
        message_left=ref_bias_tensors[6],
        message_right=ref_bias_tensors[7],
        message_basis=ref_bias_tensors[8],
    )
    actual = triton_grouped_compact_simplicial_attention(
        *tri_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=tri_bias_tensors[0],
        v_bias=tri_bias_tensors[1],
        gate=tri_bias_tensors[2],
        angle_left=tri_bias_tensors[3],
        angle_right=tri_bias_tensors[4],
        angle_gate=tri_bias_tensors[5],
        message_left=tri_bias_tensors[6],
        message_right=tri_bias_tensors[7],
        message_basis=tri_bias_tensors[8],
        precision="ieee_fp32",
        strict=True,
    )
    assert torch.allclose(actual, ref, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    actual.backward(grad)
    for actual_tensor, ref_tensor in zip(tri_tensors + tri_bias_tensors, ref_tensors + ref_bias_tensors):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_GROUPED_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="grouped compact spherical coefficient Triton parity requires CUDA and Triton",
)
def test_grouped_compact_spherical_coefficients_triton_match_expanded_cuda():
    torch.manual_seed(19)
    device = torch.device("cuda")
    batch_size, group_order, num_heads, num_tokens, k_neighbors, head_dim = 2, 3, 2, 7, 4, 32
    channels_by_l = (2, 3, 1)
    angle_rank = channels_by_l[0] + 3 * channels_by_l[1] + 5 * channels_by_l[2]
    coeff_dim = sum(channels_by_l)
    kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
    tensors = [torch.randn(batch_size, group_order, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.2).contiguous()
    neighbor_mask[..., 0] = True
    unit = torch.randn(batch_size, num_tokens, k_neighbors, 3, device=device, dtype=torch.float32)
    unit = torch.nn.functional.normalize(unit, dim=-1)
    unit = unit.masked_fill(~neighbor_mask[..., None], 0.0)
    bias_tensors = [
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
    ]
    exp_tensors = [_clone_leaf(t) for t in tensors]
    cmp_tensors = [_clone_leaf(t) for t in tensors]
    exp_bias = [_clone_leaf(t) for t in bias_tensors]
    cmp_bias = [_clone_leaf(t) for t in bias_tensors]
    basis = _spherical_basis_lmax2(unit)
    left_expanded = _expand_compact_spherical_coefficients(exp_bias[3], basis=basis, channels_by_l=channels_by_l)
    right_expanded = _expand_compact_spherical_coefficients(exp_bias[4], basis=basis, channels_by_l=channels_by_l)

    expanded = triton_grouped_compact_simplicial_attention(
        *exp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=exp_bias[0],
        v_bias=exp_bias[1],
        gate=exp_bias[2],
        angle_left=left_expanded,
        angle_right=right_expanded,
        angle_gate=exp_bias[5],
        precision="ieee_fp32",
        strict=True,
    )
    compact = triton_grouped_compact_simplicial_attention(
        *cmp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        u=cmp_bias[0],
        v_bias=cmp_bias[1],
        gate=cmp_bias[2],
        unit=unit,
        angle_left_coeff=cmp_bias[3],
        angle_right_coeff=cmp_bias[4],
        angle_channels_by_l=channels_by_l,
        angle_rank=angle_rank,
        angle_gate=cmp_bias[5],
        precision="ieee_fp32",
        strict=True,
    )
    assert torch.allclose(compact, expanded, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    compact.backward(grad)
    for actual_tensor, ref_tensor in zip(cmp_tensors + cmp_bias, exp_tensors + exp_bias):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="scalar compact spherical coefficient Triton parity requires CUDA and Triton",
)
def test_scalar_compact_spherical_coefficients_triton_match_expanded_cuda():
    torch.manual_seed(23)
    device = torch.device("cuda")
    batch_size, num_heads, num_tokens, k_neighbors, head_dim = 2, 2, 7, 4, 32
    channels_by_l = (2, 3, 1)
    angle_rank = channels_by_l[0] + 3 * channels_by_l[1] + 5 * channels_by_l[2]
    coeff_dim = sum(channels_by_l)
    kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
    tensors = [torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.2).contiguous()
    neighbor_mask[..., 0] = True
    neighbor_mask[0, 0, :] = False
    neighbor_mask[1, 3, 1:] = False
    unit = torch.randn(batch_size, num_tokens, k_neighbors, 3, device=device, dtype=torch.float32)
    unit = torch.nn.functional.normalize(unit, dim=-1)
    unit = unit.masked_fill(~neighbor_mask[..., None], 0.0)
    bias_tensors = [
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, coeff_dim, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, **kwargs),
    ]
    exp_tensors = [_clone_leaf(t) for t in tensors]
    cmp_tensors = [_clone_leaf(t) for t in tensors]
    exp_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    cmp_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    basis = _spherical_basis_lmax2(unit)
    left_expanded = _expand_compact_spherical_coefficients(
        exp_bias_tensors[3], basis=basis, channels_by_l=channels_by_l
    )
    right_expanded = _expand_compact_spherical_coefficients(
        exp_bias_tensors[4], basis=basis, channels_by_l=channels_by_l
    )
    expanded_bias = CompactSimplicialBias(
        u=exp_bias_tensors[0],
        v=exp_bias_tensors[1],
        gate=exp_bias_tensors[2],
        angle_left=left_expanded,
        angle_right=right_expanded,
        angle_gate=exp_bias_tensors[5],
    )
    compact_bias = CompactSimplicialBias(
        u=cmp_bias_tensors[0],
        v=cmp_bias_tensors[1],
        gate=cmp_bias_tensors[2],
        angle_left_coeff=cmp_bias_tensors[3],
        angle_right_coeff=cmp_bias_tensors[4],
        angle_channels_by_l=channels_by_l,
        angle_rank=angle_rank,
        angle_gate=cmp_bias_tensors[5],
    )

    expanded = compact_simplicial_attention_triton(
        *exp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=expanded_bias,
        precision="ieee_fp32",
        strict=True,
    )
    compact = compact_simplicial_attention_triton(
        *cmp_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=compact_bias,
        unit=unit,
        precision="ieee_fp32",
        strict=True,
    )
    assert torch.allclose(compact, expanded, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(expanded)
    expanded.backward(grad)
    compact.backward(grad)
    for actual_tensor, ref_tensor in zip(cmp_tensors + cmp_bias_tensors, exp_tensors + exp_bias_tensors):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)
    invalid_coeff_mask = ~neighbor_mask[:, None, :, :, None]
    left_invalid_grad = cmp_bias_tensors[3].grad.masked_select(invalid_coeff_mask.expand_as(cmp_bias_tensors[3]))
    right_invalid_grad = cmp_bias_tensors[4].grad.masked_select(invalid_coeff_mask.expand_as(cmp_bias_tensors[4]))
    assert torch.allclose(
        left_invalid_grad,
        torch.zeros_like(left_invalid_grad),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        right_invalid_grad,
        torch.zeros_like(right_invalid_grad),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="scalar compact Triton natural coefficient layout parity requires CUDA and Triton",
)
def test_scalar_compact_simplicial_triton_natural_coeffs_head_gates_bnhd_cuda():
    torch.manual_seed(27)
    device = torch.device("cuda")
    batch_size, num_heads, num_tokens, k_neighbors, head_dim = 2, 3, 8, 4, 16
    channels_by_l = (2, 3, 1)
    angle_rank = channels_by_l[0] + 3 * channels_by_l[1] + 5 * channels_by_l[2]
    coeff_dim = sum(channels_by_l)
    kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
    tensors = [torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
    base_idx = torch.arange(num_tokens, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.25).contiguous()
    neighbor_mask[..., 0] = True
    neighbor_mask[0, 0, :] = False
    unit = torch.randn(batch_size, num_tokens, k_neighbors, 3, device=device, dtype=torch.float32)
    unit = torch.nn.functional.normalize(unit, dim=-1)
    unit = unit.masked_fill(~neighbor_mask[..., None], 0.0)
    bias_tensors = [
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(batch_size, num_heads, num_tokens, k_neighbors, **kwargs),
        torch.randn(num_heads, **kwargs),
        torch.randn(batch_size, num_tokens, k_neighbors, num_heads, coeff_dim, **kwargs),
        torch.randn(batch_size, num_tokens, k_neighbors, num_heads, coeff_dim, **kwargs),
        torch.randn(num_heads, **kwargs),
    ]
    ref_tensors = [_clone_leaf(t) for t in tensors]
    tri_tensors = [_clone_leaf(t) for t in tensors]
    ref_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    tri_bias_tensors = [_clone_leaf(t) for t in bias_tensors]
    ref_bias = CompactSimplicialBias(
        u=ref_bias_tensors[0],
        v=ref_bias_tensors[1],
        gate=ref_bias_tensors[2],
        angle_left_coeff=ref_bias_tensors[3],
        angle_right_coeff=ref_bias_tensors[4],
        angle_channels_by_l=channels_by_l,
        angle_rank=angle_rank,
        angle_gate=ref_bias_tensors[5],
    )
    tri_bias = CompactSimplicialBias(
        u=tri_bias_tensors[0],
        v=tri_bias_tensors[1],
        gate=tri_bias_tensors[2],
        angle_left_coeff=tri_bias_tensors[3],
        angle_right_coeff=tri_bias_tensors[4],
        angle_channels_by_l=channels_by_l,
        angle_rank=angle_rank,
        angle_gate=tri_bias_tensors[5],
    )
    ref = compact_simplicial_attention_torch(
        *ref_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=ref_bias,
        unit=unit,
    )
    actual_bnhd = compact_simplicial_attention_triton(
        *tri_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=tri_bias,
        unit=unit,
        precision="ieee_fp32",
        strict=True,
        output_layout="bnhd",
    )
    assert actual_bnhd.shape == (batch_size, num_tokens, num_heads, head_dim)
    actual = actual_bnhd.transpose(1, 2)
    assert torch.allclose(actual, ref, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    actual_bnhd.backward(grad.transpose(1, 2).contiguous())
    for actual_tensor, ref_tensor in zip(tri_tensors + tri_bias_tensors, ref_tensors + ref_bias_tensors):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)
    invalid_coeff_mask = ~neighbor_mask[:, :, :, None, None]
    left_invalid_grad = tri_bias_tensors[3].grad.masked_select(invalid_coeff_mask.expand_as(tri_bias_tensors[3]))
    right_invalid_grad = tri_bias_tensors[4].grad.masked_select(invalid_coeff_mask.expand_as(tri_bias_tensors[4]))
    assert torch.allclose(left_invalid_grad, torch.zeros_like(left_invalid_grad), atol=1e-6, rtol=0.0)
    assert torch.allclose(right_invalid_grad, torch.zeros_like(right_invalid_grad), atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="scalar compact Triton variable-N parity requires CUDA and Triton",
)
def test_scalar_compact_simplicial_triton_accepts_variable_num_atoms_cuda():
    torch.manual_seed(29)
    device = torch.device("cuda")
    for num_tokens in (6, 9, 7):
        batch_size, num_heads, k_neighbors, head_dim, rank = 2, 2, 4, 16, 16
        kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
        tensors = [torch.randn(batch_size, num_heads, num_tokens, head_dim, **kwargs) for _ in range(5)]
        base_idx = torch.arange(num_tokens, device=device)[:, None]
        offsets = torch.arange(k_neighbors, device=device)[None, :]
        neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
        neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.2).contiguous()
        neighbor_mask[..., 0] = True
        bias = CompactSimplicialBias(
            angle_left=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
            angle_right=torch.randn(batch_size, num_heads, num_tokens, k_neighbors, rank, **kwargs),
            angle_gate=torch.randn(batch_size, num_heads, num_tokens, **kwargs),
        )
        ref_tensors = [_clone_leaf(t) for t in tensors]
        tri_tensors = [_clone_leaf(t) for t in tensors]
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


@pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_COMPACT_SIMPLICIAL_AVAILABLE),
    reason="scalar compact Triton stride parity requires CUDA and Triton",
)
def test_scalar_compact_simplicial_triton_accepts_noncontiguous_qkv_cuda():
    torch.manual_seed(31)
    device = torch.device("cuda")
    batch_size, num_heads, num_tokens, k_neighbors, head_dim = 2, 3, 8, 4, 16
    q_scale = head_dim**-0.5
    base_kwargs = {"device": device, "dtype": torch.float32}
    base_tensors = [torch.randn(batch_size, num_tokens, num_heads, head_dim, **base_kwargs) for _ in range(5)]

    def make_leaf(base: torch.Tensor) -> torch.Tensor:
        leaf = base.clone().transpose(1, 2).detach().requires_grad_(True)
        assert leaf.shape == (batch_size, num_heads, num_tokens, head_dim)
        assert not leaf.is_contiguous()
        return leaf

    ref_tensors = [make_leaf(tensor) for tensor in base_tensors]
    tri_tensors = [make_leaf(tensor) for tensor in base_tensors]
    base_idx = torch.arange(num_tokens, device=device)[:, None]
    offsets = torch.arange(k_neighbors, device=device)[None, :]
    neighbor_idx = ((base_idx + offsets + 1) % num_tokens).expand(batch_size, -1, -1).contiguous()
    neighbor_mask = (torch.rand(batch_size, num_tokens, k_neighbors, device=device) > 0.25).contiguous()
    neighbor_mask[..., 0] = True
    neighbor_mask[0, 2, :] = False

    ref = compact_simplicial_attention_torch(
        *ref_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        q_scale=q_scale,
    )
    actual = compact_simplicial_attention_triton(
        *tri_tensors,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        precision="ieee_fp32",
        strict=True,
        q_scale=q_scale,
    )
    assert torch.allclose(actual, ref, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    actual.backward(grad)
    for actual_tensor, ref_tensor in zip(tri_tensors, ref_tensors):
        assert actual_tensor.grad is not None
        assert ref_tensor.grad is not None
        assert torch.allclose(actual_tensor.grad, ref_tensor.grad, atol=1e-3, rtol=1e-3)


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


def _small_geom(coords: torch.Tensor, pad_mask: torch.Tensor, k_neighbors: int = 3):
    adapter = NonPeriodicGeometryAdapter()
    return build_geometry_cache(
        adapter(coords=coords, pad_mask=pad_mask),
        coords_len=coords.shape[1],
        seq_len=coords.shape[1],
        k_neighbors=k_neighbors,
        pad_mask=pad_mask,
        rbf_dim=8,
    )


def test_compact_simplicial_geometry_bias_gate_init_and_message_gate():
    torch.manual_seed(0)
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.zeros(2, 5, dtype=torch.bool)
    geom = _small_geom(coords, pad_mask, k_neighbors=3)
    bias_module = CompactSimplicialGeometryBias(
        num_heads=2,
        rank=8,
        rbf_dim=8,
        gate_init=0.05,
        message_enabled=True,
        message_rank=8,
        head_dim=4,
    )
    bias = bias_module(geom, dtype=torch.float32)
    assert bias.gate is not None
    assert bias.angle_gate is not None
    assert bias.message_left is not None
    assert bias.message_right is not None
    assert torch.allclose(bias_module.gate, torch.full_like(bias_module.gate, 0.05))
    assert torch.allclose(bias_module.angle_gate, torch.full_like(bias_module.angle_gate, 0.05))
    assert torch.allclose(bias_module.message_gate, torch.full_like(bias_module.message_gate, 0.05))
    assert bias.message_left.abs().max() > 0


def test_group_framewise_simplicial_layer_shape_mask_and_equivariance():
    torch.manual_seed(0)
    group = PLATONIC_GROUPS["tetrahedron"]
    layer = GroupFramewiseSimplicialLayer(
        group_order=group.G,
        dim_per_frame=4,
        config={
            "k_neighbors": 3,
            "num_heads": 1,
            "projection_mode": "group_linear",
            "bias": {"angle_rank": 8, "radial_basis_dim": 8},
            "kernel": {"backend": "torch"},
        },
        dropout=0.0,
        mlp_ratio=2.0,
    ).eval()
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    geom = _small_geom(coords, pad_mask, k_neighbors=3)
    group_features = torch.randn(2, 5, group.G, 4)
    state = ModelState(pos=coords, mask=pad_mask, scalar=None, group=group_features.clone(), geom=geom)
    out = layer(state).group
    assert out is not None
    assert out.shape == group_features.shape
    assert torch.all(out[pad_mask] == 0)
    assert "attn_delta_rms" in layer.last_diagnostics
    assert "radial_gate_mean" in layer.last_diagnostics

    permutation = group.cayley_table[3]
    state_perm = ModelState(
        pos=coords,
        mask=pad_mask,
        scalar=None,
        group=group_features[:, :, permutation].clone(),
        geom=geom,
    )
    out_perm = layer(state_perm).group
    assert out_perm is not None
    assert torch.allclose(out_perm, out[:, :, permutation], atol=1e-5, rtol=1e-5)


def test_hybrid_tetra_local_moment_lift_and_sg_diagnostics():
    torch.manual_seed(0)
    trunk = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        input_dim=5,
        hybrid_config={
            "stream_type": "tetra",
            "num_blocks": 1,
            "block_mix": [1, 0, 0],
            "tetra_dim_per_frame": 2,
            "simplicial": {
                "k_neighbors": 3,
                "num_heads": 1,
                "head_dim": 4,
                "bias": {"angle_rank": 8, "radial_basis_dim": 8, "gate_init": 0.05},
                "message": {"enabled": True, "rank": 8, "gate_init": 0.05},
                "kernel": {"backend": "torch"},
            },
            "input_lift": {"kind": "local_moment_lift", "hidden_dim": 16, "scale_init": 0.1},
            "readout": {"kind": "platonic_ffn"},
        },
        geometry_adapter=NonPeriodicGeometryAdapter(),
        use_final_norm=False,
    )
    x = torch.randn(2, 5, 5)
    coords = torch.randn(2, 5, 3)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    out = trunk(x, None, pad_mask=pad_mask, coords=coords, return_output=True)
    assert isinstance(out, tuple) is False
    assert out.group is not None
    assert out.group.shape == (2, 5, 12, 2)
    diagnostics = trunk.collect_sg_diagnostics()
    assert diagnostics["sg/layers_logged"] == 1.0
    assert "sg/mean/message_gate_mean" in diagnostics


def test_group_vector_readout_is_tetra_equivariant():
    torch.manual_seed(0)
    model = QM9EDMModel(
        d_model=24,
        n_heads=4,
        n_layers=1,
        attn_type="hybrid",
        hybrid_config=_tetra_hybrid_config([0, 1, 0]),
        coord_head_mode="group_vector",
    )
    model.group_vector_scale.data.fill_(1.0)
    group = PLATONIC_GROUPS["tetrahedron"]
    group_features = torch.randn(2, 4, group.G, 4)
    pad_mask = torch.zeros(2, 4, dtype=torch.bool)
    coord_delta = model._group_vector_coord_delta(group_features, pad_mask)
    permutation = group.cayley_table[:, 5]
    rotated = model._group_vector_coord_delta(group_features[:, :, permutation], pad_mask)
    expected = torch.einsum("ij,bnj->bni", group.elements[5], coord_delta)
    assert torch.allclose(rotated, expected, atol=1e-5, rtol=1e-5)


def _hybrid_config() -> dict:
    return {
        "num_blocks": 1,
        "stream_type": "scalar",
        "block_mix": [1, 0, 1],
        "tetra_dim_per_frame": 4,
        "simplicial": {"k_neighbors": 3, "bias": {"angle_rank": 8, "radial_basis_dim": 8}},
    }


def _tetra_hybrid_config(block_mix=None) -> dict:
    return {
        "num_blocks": 1,
        "stream_type": "tetra",
        "block_mix": [1, 1, 0] if block_mix is None else block_mix,
        "tetra_dim_per_frame": 4,
        "simplicial": {
            "k_neighbors": 3,
            "num_heads": 1,
            "projection_mode": "group_linear",
            "bias": {"angle_rank": 8, "radial_basis_dim": 8},
        },
        "tetra": {"heads_per_frame": 1},
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


@pytest.mark.parametrize("block_mix", [[0, 1, 0], [1, 1, 0]])
def test_qm9_tetra_hybrid_edm_forward_backward(block_mix):
    torch.manual_seed(0)
    model = QM9EDMModel(
        d_model=32,
        n_heads=4,
        n_layers=1,
        attn_type="hybrid",
        hybrid_config=_tetra_hybrid_config(block_mix),
        coord_head_mode="group_vector",
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


def test_geom_drugs_tetra_hybrid_edm_forward():
    torch.manual_seed(0)
    model = GeomDrugsEDMModel(
        d_model=32,
        n_heads=4,
        n_layers=1,
        attn_type="hybrid",
        hybrid_config=_tetra_hybrid_config([1, 1, 0]),
        coord_head_mode="group_vector",
    )
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


def test_mof_stage1_tetra_hybrid_raises_until_lattice_readout_exists():
    with pytest.raises(NotImplementedError, match="lattice/cell-aware readout"):
        MOFStage1EDMModel(
            block_feature_dim=5,
            d_model=32,
            n_heads=4,
            n_layers=1,
            attn_type="hybrid",
            hybrid_config=_tetra_hybrid_config([0, 1, 0]),
        )
