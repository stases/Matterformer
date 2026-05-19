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
    prepare_esen_fixed_k_local_attention_features,
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


def test_fixed_k_local_attention_handles_empty_and_self_only_rows():
    q = torch.randn(3, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.tensor([[0, 1], [1, 0], [2, 1]], dtype=torch.long)
    neighbor_mask = torch.tensor([[False, False], [True, False], [True, True]], dtype=torch.bool)
    dist = torch.tensor([[9.0, 9.0], [0.0, 9.0], [0.0, 9.0]])
    bias = ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=torch.zeros(1, 2),
        centers=torch.tensor([0.0, 1.0]),
        gamma=torch.tensor(1.0),
        cutoff=1.0,
        heads_per_frame=1,
        diag_zero=True,
        envelope_in_score=True,
        trainable=False,
    )

    actual = fixed_k_local_attention_torch_reference(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        bias=bias,
    )
    assert torch.allclose(actual[0], torch.zeros_like(actual[0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual[1], v[1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual[2], v[2], atol=1e-6, rtol=1e-6)


def test_prepare_esen_fixed_k_features_reconstructs_bias_module_output():
    torch.manual_seed(12)
    q = torch.randn(4, 3, 5)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
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
            [True, True, True],
            [True, True, False],
            [True, True, True],
            [True, False, True],
        ],
        dtype=torch.bool,
    )
    dist = torch.tensor(
        [
            [0.0, 0.8, 2.8],
            [0.0, 0.8, 9.0],
            [0.0, 1.1, 2.9],
            [0.0, 9.0, 1.2],
        ]
    )
    atom_types = torch.tensor([1, 6, 8, 7], dtype=torch.long)
    type_bias = torch.randn(10, 10, 1) * 0.01
    bias = ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=torch.randn(1, 4) * 0.01,
        type_bias=type_bias,
        centers=torch.linspace(0.0, 3.0, 4),
        gamma=torch.tensor(1.0),
        cutoff=3.0,
        heads_per_frame=1,
        diag_zero=True,
        envelope_in_score=True,
        trainable=False,
    )

    expected = bias(
        q=q,
        k=k,
        v=v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        rbf=None,
        atom_types=atom_types,
    )
    features = prepare_esen_fixed_k_local_attention_features(
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        centers=bias.centers,
        gamma=bias.gamma,
        cutoff=bias.cutoff,
        heads_per_frame=bias.heads_per_frame,
        atom_types=atom_types,
        max_atomic_number=9,
        diag_zero=bias.diag_zero,
        envelope_in_score=bias.envelope_in_score,
        feature_dtype=torch.float32,
    )
    assert features.type_base.dtype == torch.int32
    subhead = torch.arange(q.shape[1]) % int(bias.heads_per_frame)
    rbf_term = torch.einsum("nkr,hr->nhk", features.rho_env, bias.rbf_weight[subhead])
    type_index = features.type_base.long()[:, None, :] + subhead.view(1, -1, 1)
    type_term = features.env_bias[:, None, :] * bias.type_bias.reshape(-1)[type_index]
    reconstructed = features.log_env[:, None, :] + rbf_term + type_term

    assert torch.equal(features.local_mask, expected.mask)
    mask = expected.mask[:, None, :].expand_as(expected.bias)
    torch.testing.assert_close(reconstructed[mask], expected.bias[mask], atol=1e-6, rtol=1e-6)

    default_features = prepare_esen_fixed_k_local_attention_features(
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        dist=dist,
        centers=bias.centers,
        gamma=bias.gamma,
        cutoff=bias.cutoff,
        heads_per_frame=bias.heads_per_frame,
        atom_types=atom_types,
        max_atomic_number=9,
        diag_zero=bias.diag_zero,
        envelope_in_score=bias.envelope_in_score,
    )
    assert default_features.env_bias.dtype == torch.bfloat16
    assert default_features.log_env.dtype == torch.bfloat16
    assert default_features.rho_env.dtype == torch.bfloat16
    assert default_features.type_base.dtype == torch.int32


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


def test_fixed_k_local_attention_triton_strict_rejects_unsupported_fallback():
    class UnsupportedBias(FixedKLocalBias):
        def forward(self, *, q, k, v, neighbor_idx, neighbor_mask, dist, rbf, atom_types):
            del k, v, neighbor_idx, neighbor_mask, dist, rbf, atom_types
            return FixedKLocalBiasResult(
                bias=torch.zeros((q.shape[0], q.shape[1], 1), device=q.device, dtype=q.dtype),
                mask=torch.ones((q.shape[0], 1), device=q.device, dtype=torch.bool),
            )

    q = torch.randn(3, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    neighbor_idx = torch.tensor([[0], [1], [2]], dtype=torch.long)
    neighbor_mask = torch.ones(3, 1, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="unavailable"):
        fixed_k_local_attention_triton(
            q,
            k,
            v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            bias=UnsupportedBias(),
            strict=True,
        )


@torch.no_grad()
def _run_cuda_triton_forward_parity(
    *,
    with_bias: bool,
    head_dim: int = 16,
    k_neighbors: int | None = None,
    qkv_dtype: torch.dtype = torch.float32,
    precision: str = "ieee",
) -> None:
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
    heads_per_frame = 2
    cutoff = 4.0
    rbf_dim = 6
    q = torch.randn(num_atoms, num_heads, head_dim, device=device, dtype=qkv_dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    atom_types = torch.tensor([1, 6, 8, 7, 16, 6], device=device, dtype=torch.long)
    centers = torch.linspace(0.0, cutoff, rbf_dim, device=device)
    delta = cutoff / max(rbf_dim - 1, 1)
    gamma = torch.tensor(1.0 / max(delta * delta, 1.0e-6), device=device)
    geom = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=None,
        k_neighbors=k_neighbors or num_atoms,
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
    if precision == "bf16_flash_compat":
        q_ref = q.to(torch.bfloat16).float()
        k_ref = k.to(torch.bfloat16).float()
        v_ref = v.to(torch.bfloat16).float()
        expected_dtype = torch.bfloat16
    elif qkv_dtype is torch.bfloat16:
        q_ref = q.float()
        k_ref = k.float()
        v_ref = v.float()
        expected_dtype = torch.bfloat16
    else:
        q_ref = q
        k_ref = k
        v_ref = v
        expected_dtype = q.dtype
    expected = fixed_k_local_attention_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        rbf=geom.rbf[0],
        atom_types=atom_types,
        bias=bias,
    ).to(expected_dtype).to(q.dtype)
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
        precision=precision,
    )
    atol = 8e-3 if qkv_dtype is torch.bfloat16 or precision == "bf16_flash_compat" else 3e-5
    rtol = 8e-3 if qkv_dtype is torch.bfloat16 or precision == "bf16_flash_compat" else 3e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
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


@pytest.mark.skipif(
    not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE or not torch.cuda.is_available(),
    reason="fixed-K local Triton dtype/chunked-D parity requires CUDA and Triton",
)
@pytest.mark.parametrize("head_dim", [64, 128, 160])
@pytest.mark.parametrize("precision,qkv_dtype", [("ieee", torch.float32), ("ieee", torch.bfloat16), ("bf16_flash_compat", torch.float32)])
def test_fixed_k_local_attention_triton_cuda_dtype_and_chunked_d(head_dim, precision, qkv_dtype):
    _run_cuda_triton_forward_parity(
        with_bias=True,
        head_dim=head_dim,
        k_neighbors=6,
        qkv_dtype=qkv_dtype,
        precision=precision,
    )


@pytest.mark.skipif(
    not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE or not torch.cuda.is_available(),
    reason="fixed-K local Triton backward requires CUDA and Triton",
)
@pytest.mark.parametrize("bias_case", ["none", "esen_no_type", "esen_type"])
@pytest.mark.parametrize("head_dim", [32, 160, 256])
@pytest.mark.parametrize("precision,qkv_dtype", [("ieee", torch.float32), ("ieee", torch.bfloat16), ("bf16_flash_compat", torch.float32)])
def test_fixed_k_local_attention_triton_cuda_backward_matches_reference(bias_case, head_dim, precision, qkv_dtype):
    torch.manual_seed(9)
    device = torch.device("cuda")
    num_atoms = 6
    num_heads = 2
    rbf_dim = 4
    cutoff = 3.0
    coords = torch.randn(1, num_atoms, 3, device=device) * 0.4
    q_base = torch.randn(num_atoms, num_heads, head_dim, device=device)
    q = q_base.to(qkv_dtype).detach().clone().requires_grad_(True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    atom_types = torch.tensor([1, 6, 8, 7, 16, 6], device=device, dtype=torch.long)
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
    centers = torch.linspace(0.0, cutoff, rbf_dim, device=device)
    gamma = torch.tensor(1.0, device=device)
    bias = None
    bias_ref = None
    if bias_case != "none":
        rbf_weight = torch.randn(1, rbf_dim, device=device) * 0.01
        type_bias = torch.randn(20, 20, 1, device=device) * 0.01 if bias_case == "esen_type" else None
        bias = ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=rbf_weight.clone(),
            type_bias=None if type_bias is None else type_bias.clone(),
            centers=centers,
            gamma=gamma,
            cutoff=cutoff,
            heads_per_frame=1,
            trainable=True,
        ).to(device)
        bias_ref = ESENEnvelopedRBFTypeFixedKBias(
            rbf_weight=rbf_weight.clone(),
            type_bias=None if type_bias is None else type_bias.clone(),
            centers=centers,
            gamma=gamma,
            cutoff=cutoff,
            heads_per_frame=1,
            trainable=True,
        ).to(device)
    if precision == "bf16_flash_compat":
        q_ref_leaf = q.detach().float().clone().requires_grad_(True)
        k_ref_leaf = k.detach().float().clone().requires_grad_(True)
        v_ref_leaf = v.detach().float().clone().requires_grad_(True)
        q_ref = q_ref_leaf.to(torch.bfloat16).float()
        k_ref = k_ref_leaf.to(torch.bfloat16).float()
        v_ref = v_ref_leaf.to(torch.bfloat16).float()
    elif qkv_dtype is torch.bfloat16:
        q_ref_leaf = q.detach().clone().requires_grad_(True)
        k_ref_leaf = k.detach().clone().requires_grad_(True)
        v_ref_leaf = v.detach().clone().requires_grad_(True)
        q_ref = q_ref_leaf.float()
        k_ref = k_ref_leaf.float()
        v_ref = v_ref_leaf.float()
    else:
        q_ref_leaf = q.detach().clone().requires_grad_(True)
        k_ref_leaf = k.detach().clone().requires_grad_(True)
        v_ref_leaf = v.detach().clone().requires_grad_(True)
        q_ref = q_ref_leaf
        k_ref = k_ref_leaf
        v_ref = v_ref_leaf
    grad_seed = torch.randn_like(q.float()).to(q.dtype)

    actual = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        atom_types=atom_types,
        bias=bias,
        precision=precision,
        strict=True,
    )
    expected = fixed_k_local_attention_torch_reference(
        q_ref,
        k_ref,
        v_ref,
        neighbor_idx=geom.neighbor_idx[0],
        neighbor_mask=geom.neighbor_mask[0],
        dist=geom.dist[0],
        atom_types=atom_types,
        bias=bias_ref,
    )
    if precision == "bf16_flash_compat":
        expected = expected.to(torch.bfloat16).to(torch.float32)
    elif qkv_dtype is torch.bfloat16:
        expected = expected.to(torch.bfloat16)
    loose = qkv_dtype is torch.bfloat16 or precision == "bf16_flash_compat"
    out_tol = 2e-2 if loose else 3e-5
    grad_tol = 5e-2 if loose else 2e-4
    param_tol = 5e-2 if loose else 5e-4
    torch.testing.assert_close(actual.detach(), expected.detach().to(actual.dtype), atol=out_tol, rtol=out_tol)

    (actual * grad_seed).sum().backward()
    (expected * grad_seed.to(expected.dtype)).sum().backward()
    torch.testing.assert_close(q.grad.float(), q_ref_leaf.grad.float(), atol=grad_tol, rtol=grad_tol)
    torch.testing.assert_close(k.grad.float(), k_ref_leaf.grad.float(), atol=grad_tol, rtol=grad_tol)
    torch.testing.assert_close(v.grad.float(), v_ref_leaf.grad.float(), atol=grad_tol, rtol=grad_tol)
    if bias is not None and bias_ref is not None:
        torch.testing.assert_close(bias.rbf_weight.grad, bias_ref.rbf_weight.grad, atol=param_tol, rtol=param_tol)
        if bias.type_bias is not None:
            assert bias_ref.type_bias is not None
            torch.testing.assert_close(bias.type_bias.grad, bias_ref.type_bias.grad, atol=param_tol, rtol=param_tol)


@pytest.mark.skipif(
    not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE or not torch.cuda.is_available(),
    reason="fixed-K local Triton lse autograd semantics require CUDA and Triton",
)
def test_fixed_k_local_attention_triton_cuda_lse_is_nondifferentiable():
    torch.manual_seed(10)
    device = torch.device("cuda")
    q = torch.randn(4, 2, 16, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    neighbor_idx = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], device=device, dtype=torch.int32)
    neighbor_mask = torch.ones(4, 2, device=device, dtype=torch.bool)

    out, lse = fixed_k_local_attention_triton(
        q,
        k,
        v,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        return_lse=True,
        strict=True,
    )

    assert not lse.requires_grad
    assert lse.grad_fn is None
    out.square().mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.skipif(
    not TRITON_FIXED_K_LOCAL_ATTENTION_AVAILABLE or not torch.cuda.is_available(),
    reason="fixed-K local Triton dist-gradient guard requires CUDA and Triton",
)
def test_fixed_k_local_attention_triton_cuda_rejects_dist_gradients():
    torch.manual_seed(11)
    device = torch.device("cuda")
    q = torch.randn(4, 2, 16, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    neighbor_idx = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], device=device, dtype=torch.int32)
    neighbor_mask = torch.ones(4, 2, device=device, dtype=torch.bool)
    dist = torch.rand(4, 2, device=device, requires_grad=True)
    centers = torch.linspace(0.0, 3.0, 4, device=device)
    bias = ESENEnvelopedRBFTypeFixedKBias(
        rbf_weight=torch.randn(1, 4, device=device) * 0.01,
        type_bias=None,
        centers=centers,
        gamma=torch.tensor(1.0, device=device),
        cutoff=3.0,
        heads_per_frame=1,
        trainable=True,
    ).to(device)

    with pytest.raises(NotImplementedError, match="dist/coordinates"):
        fixed_k_local_attention_triton(
            q,
            k,
            v,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            dist=dist,
            bias=bias,
            strict=True,
        )
