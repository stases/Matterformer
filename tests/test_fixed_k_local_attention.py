import math

import pytest
import torch

from matterformer.geometry import build_triton_nonperiodic_knn_geometry_cache
from matterformer.models.platonic import (
    ESENEnvelopedRBFTypeFixedKBias,
    FixedKLocalBias,
    FixedKLocalBiasResult,
    NoFixedKLocalBias,
    fixed_k_local_attention_triton,
    fixed_k_local_attention_torch_reference,
)
from matterformer.models.platonic.local_attention_triton import TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE
from matterformer.models.platonic.triton_attention import platonic_attention_flat_torch_reference


def test_fixed_k_local_attention_no_bias_matches_manual_reference():
    torch.manual_seed(0)
    q = torch.randn(4, 2, 3)
    k = torch.randn(4, 2, 3)
    v = torch.randn(4, 2, 3)
    neighbor_idx = torch.tensor(
        [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
            [3, 2, 1],
        ],
        dtype=torch.long,
    )
    neighbor_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )

    actual = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
    )

    expected = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for i in range(q.shape[0]):
        js = neighbor_idx[i, neighbor_mask[i]]
        scores = torch.einsum("hd,jhd->hj", q[i], k[js]) * scale
        probs = torch.softmax(scores, dim=-1)
        expected[i] = torch.einsum("hj,jhd->hd", probs, v[js])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fixed_k_local_attention_accepts_replaceable_bias_module():
    class OnlySecondSlotBias(FixedKLocalBias):
        def forward(
            self,
            *,
            q,
            k,
            v,
            neighbor_idx,
            neighbor_mask,
            dist,
            rbf,
            atom_types,
        ):
            del k, v, neighbor_idx, neighbor_mask, dist, rbf, atom_types
            mask = torch.zeros((q.shape[0], 2), device=q.device, dtype=torch.bool)
            mask[:, 1] = True
            return FixedKLocalBiasResult(
                bias=torch.zeros((q.shape[0], q.shape[1], 2), device=q.device, dtype=q.dtype),
                mask=mask,
            )

    q = torch.zeros(3, 2, 4)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    neighbor_mask = torch.ones(3, 2, dtype=torch.bool)

    actual = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=OnlySecondSlotBias(),
    )
    expected = v.index_select(0, neighbor_idx[:, 1])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fixed_k_no_bias_module_matches_none_reference():
    torch.manual_seed(7)
    q = torch.randn(4, 2, 3)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long)
    neighbor_mask = torch.ones(4, 2, dtype=torch.bool)

    expected = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=None,
    )
    actual = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        bias=NoFixedKLocalBias(),
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("envelope_in_score", [False, True])
def test_fixed_k_esen_bias_masks_neighbors_outside_cutoff(envelope_in_score):
    q = torch.zeros(2, 1, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]])
    neighbor_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    neighbor_mask = torch.ones(2, 2, dtype=torch.bool)
    dist = torch.tensor([[0.0, 4.0], [0.0, 4.0]])
    cutoff = 1.0
    centers = torch.tensor([0.0, cutoff])
    bias = ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=torch.zeros(1, 2),
        centers=centers,
        gamma=torch.tensor(1.0),
        cutoff=cutoff,
        heads_per_frame=1,
        diag_zero=True,
        envelope_in_score=envelope_in_score,
        trainable=False,
    )

    actual = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        atom_types=None,
        bias=bias,
    )
    assert torch.allclose(actual, v, atol=1e-6, rtol=1e-6)


def test_fixed_k_esen_type_bias_requires_square_atomic_dims():
    with pytest.raises(ValueError, match="square"):
        ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=torch.zeros(1, 2),
            type_bias=torch.zeros(4, 5, 1),
            centers=torch.tensor([0.0, 1.0]),
            gamma=torch.tensor(1.0),
            cutoff=1.0,
            heads_per_frame=1,
        )


def test_fixed_k_esen_bias_matches_dense_radius_reference_when_k_covers_radius():
    torch.manual_seed(1)
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.0],
                [1.6, 0.0, 0.2],
                [0.1, 1.1, 0.0],
                [1.0, 1.0, 0.3],
            ]
        ],
        dtype=torch.float32,
    )
    num_atoms = coords.shape[1]
    num_heads = 4
    head_dim = 5
    heads_per_frame = 2
    cutoff = 4.0
    rbf_dim = 6
    q = torch.randn(num_atoms, num_heads, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    atom_types = torch.tensor([1, 6, 8, 7, 16], dtype=torch.long)
    centers = torch.linspace(0.0, cutoff, rbf_dim)
    delta = cutoff / max(rbf_dim - 1, 1)
    gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6))
    rbf_weight = torch.randn(heads_per_frame, rbf_dim) * 0.05
    type_bias = torch.randn(20, 20, heads_per_frame) * 0.03

    geom = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=None,
        k_neighbors=num_atoms,
        rbf_dim=rbf_dim,
        cutoff=cutoff,
        seq_len=num_atoms,
        include_self=True,
        self_as_first_neighbor=True,
        mask_by_cutoff=True,
    )
    bias = ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=cutoff,
        heads_per_frame=heads_per_frame,
        diag_zero=True,
        envelope_in_score=True,
        trainable=False,
    )

    actual = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        rbf=geom.rbf[0],
        atom_types=atom_types,
        bias=bias,
    )
    expected = platonic_attention_flat_torch_reference(
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, num_atoms], dtype=torch.int32),
        pos=coords[0],
        atom_types=atom_types,
        heads_per_frame=heads_per_frame,
        rbf_weight=rbf_weight,
        type_bias=type_bias,
        centers=centers,
        gamma=gamma,
        cutoff=cutoff,
        radial_bias_kind="radius_rbf_type_enveloped",
        diag_zero=True,
        include_self=True,
        envelope_in_score=True,
    )
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_fixed_k_esen_bias_backpropagates_to_attention_and_bias_parameters():
    torch.manual_seed(2)
    num_tokens = 5
    num_heads = 4
    head_dim = 3
    heads_per_frame = 2
    rbf_dim = 4
    q = torch.randn(num_tokens, num_heads, head_dim, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    neighbor_idx = torch.tensor(
        [
            [0, 1, 2],
            [1, 0, 3],
            [2, 4, 0],
            [3, 1, 4],
            [4, 2, 3],
        ],
        dtype=torch.long,
    )
    neighbor_mask = torch.ones(num_tokens, 3, dtype=torch.bool)
    dist = torch.tensor(
        [
            [0.0, 0.9, 1.4],
            [0.0, 0.9, 1.2],
            [0.0, 1.0, 1.4],
            [0.0, 1.2, 0.8],
            [0.0, 1.0, 0.8],
        ]
    )
    centers = torch.linspace(0.0, 3.0, rbf_dim)
    gamma = torch.tensor(1.0)
    rbf = torch.exp(-gamma * (dist[..., None] - centers.view(1, 1, -1)).square())
    atom_types = torch.tensor([1, 6, 8, 7, 6], dtype=torch.long)
    bias = ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=torch.randn(heads_per_frame, rbf_dim) * 0.01,
        type_bias=torch.randn(10, 10, heads_per_frame) * 0.01,
        centers=centers,
        gamma=gamma,
        cutoff=3.0,
        heads_per_frame=heads_per_frame,
        trainable=True,
    )

    out = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        rbf=rbf,
        atom_types=atom_types,
        bias=bias,
    )
    out.square().mean().backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert bias.rbf_weight.grad is not None and torch.isfinite(bias.rbf_weight.grad).all()
    assert bias.type_bias is not None
    assert bias.type_bias.grad is not None and torch.isfinite(bias.type_bias.grad).all()


def test_fixed_k_local_attention_triton_cpu_falls_back_to_reference():
    torch.manual_seed(4)
    q = torch.randn(5, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.tensor(
        [
            [0, 1, 2],
            [1, 0, 3],
            [2, 4, 0],
            [3, 1, 4],
            [4, 2, 3],
        ],
        dtype=torch.long,
    )
    neighbor_mask = torch.ones(5, 3, dtype=torch.bool)

    actual = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        strict=False,
    )
    expected = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fixed_k_local_attention_triton_return_lse_requires_triton_path():
    q = torch.randn(3, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.tensor([[0], [1], [2]], dtype=torch.long)
    neighbor_mask = torch.ones(3, 1, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="return_lse=True"):
        fixed_k_local_attention_triton(
            q,
            k,
            v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            return_lse=True,
            strict=False,
        )


@torch.no_grad()
def _run_cuda_triton_forward_parity(*, with_bias: bool) -> None:
    torch.manual_seed(5 if with_bias else 6)
    device = torch.device("cuda")
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.0],
                [1.6, 0.0, 0.2],
                [0.1, 1.1, 0.0],
                [1.0, 1.0, 0.3],
                [2.0, 1.0, 0.1],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    num_atoms = coords.shape[1]
    num_heads = 4
    head_dim = 16
    heads_per_frame = 2
    cutoff = 4.0
    rbf_dim = 6
    q = torch.randn(num_atoms, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    atom_types = torch.tensor([1, 6, 8, 7, 16, 6], device=device, dtype=torch.long)
    centers = torch.linspace(0.0, cutoff, rbf_dim, device=device)
    delta = cutoff / max(rbf_dim - 1, 1)
    gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), device=device)
    geom = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=None,
        k_neighbors=num_atoms,
        rbf_dim=rbf_dim,
        cutoff=cutoff,
        seq_len=num_atoms,
        strict=True,
        include_self=True,
        self_as_first_neighbor=True,
        mask_by_cutoff=True,
    )
    bias = None
    if with_bias:
        bias = ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=torch.randn(heads_per_frame, rbf_dim, device=device) * 0.05,
            type_bias=torch.randn(20, 20, heads_per_frame, device=device) * 0.03,
            centers=centers,
            gamma=gamma,
            cutoff=cutoff,
            heads_per_frame=heads_per_frame,
            trainable=False,
        ).to(device)
    expected = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        rbf=geom.rbf[0],
        atom_types=atom_types,
        bias=bias,
    )
    actual, lse = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        atom_types=atom_types,
        bias=bias,
        strict=True,
        return_lse=True,
        precision="ieee",
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert lse.shape == (num_atoms, num_heads)
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(
    not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE or not torch.cuda.is_available(),
    reason="fixed-K local Triton forward parity requires CUDA and Triton",
)
def test_fixed_k_local_attention_triton_cuda_no_bias_matches_reference():
    _run_cuda_triton_forward_parity(with_bias=False)


@pytest.mark.skipif(
    not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE or not torch.cuda.is_available(),
    reason="fixed-K local Triton eSEN forward parity requires CUDA and Triton",
)
def test_fixed_k_local_attention_triton_cuda_esen_bias_matches_reference():
    _run_cuda_triton_forward_parity(with_bias=True)
