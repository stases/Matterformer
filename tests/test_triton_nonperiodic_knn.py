import pytest
import torch

from matterformer.geometry import (
    NonPeriodicGeometryAdapter,
    TRITON_NONPERIODIC_KNN_AVAILABLE,
    build_triton_nonperiodic_knn_geometry_cache,
)
from matterformer.models import HybridTransformerTrunk, build_geometry_cache


def _assert_knn_cache_close(actual, expected, *, atol=1e-5, rtol=1e-5):
    assert torch.equal(actual.neighbor_mask.cpu(), expected.neighbor_mask.cpu())
    valid = expected.neighbor_mask
    assert torch.equal(actual.neighbor_idx[valid].cpu(), expected.neighbor_idx[valid].cpu())
    assert torch.allclose(actual.dist, expected.dist, atol=atol, rtol=rtol)
    assert torch.allclose(actual.rel, expected.rel, atol=atol, rtol=rtol)
    assert torch.allclose(actual.unit, expected.unit, atol=atol, rtol=rtol)
    assert torch.allclose(actual.rbf, expected.rbf, atol=atol, rtol=rtol)
    assert torch.equal(actual.pair_mask.cpu(), expected.pair_mask.cpu())


def _dense_cache(coords, pad_mask, *, k_neighbors, rbf_dim, cutoff, seq_len):
    features = NonPeriodicGeometryAdapter()(coords, pad_mask=pad_mask)
    return build_geometry_cache(
        features,
        coords_len=coords.shape[1],
        seq_len=seq_len,
        k_neighbors=k_neighbors,
        pad_mask=pad_mask,
        rbf_dim=rbf_dim,
        cutoff=cutoff,
    )


def test_triton_nonperiodic_knn_cpu_fallback_matches_dense_cache():
    torch.manual_seed(0)
    coords = torch.randn(3, 7, 3)
    pad_mask = torch.zeros(3, 7, dtype=torch.bool)
    pad_mask[1, -2:] = True
    pad_mask[2, -4:] = True

    expected = _dense_cache(coords, pad_mask, k_neighbors=5, rbf_dim=8, cutoff=6.0, seq_len=7)
    actual = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=pad_mask,
        k_neighbors=5,
        rbf_dim=8,
        cutoff=6.0,
        seq_len=7,
        strict=False,
    )

    assert actual.features is not None
    _assert_knn_cache_close(actual, expected)


def test_triton_nonperiodic_knn_strict_requires_available_cuda_path():
    if TRITON_NONPERIODIC_KNN_AVAILABLE and torch.cuda.is_available():
        pytest.skip("strict CUDA path is available in this environment")
    coords = torch.randn(1, 4, 3)
    with pytest.raises(RuntimeError, match="triton_nonperiodic"):
        build_triton_nonperiodic_knn_geometry_cache(
            coords,
            pad_mask=None,
            k_neighbors=2,
            rbf_dim=4,
            cutoff=None,
            seq_len=4,
            strict=True,
        )


@pytest.mark.skipif(
    not TRITON_NONPERIODIC_KNN_AVAILABLE or not torch.cuda.is_available(),
    reason="compact nonperiodic Triton kNN parity requires CUDA and Triton",
)
def test_triton_nonperiodic_knn_cuda_matches_dense_cache():
    torch.manual_seed(1)
    coords = torch.randn(3, 11, 3, device="cuda")
    pad_mask = torch.zeros(3, 11, dtype=torch.bool, device="cuda")
    pad_mask[1, -2:] = True
    pad_mask[2, -5:] = True

    expected = _dense_cache(coords, pad_mask, k_neighbors=4, rbf_dim=8, cutoff=6.0, seq_len=11)
    actual = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=pad_mask,
        k_neighbors=4,
        rbf_dim=8,
        cutoff=6.0,
        seq_len=11,
        strict=True,
    )

    assert actual.features is None
    _assert_knn_cache_close(actual, expected)


@pytest.mark.skipif(
    not TRITON_NONPERIODIC_KNN_AVAILABLE or not torch.cuda.is_available(),
    reason="compact nonperiodic Triton kNN fallback parity requires CUDA and Triton",
)
def test_triton_nonperiodic_knn_requires_grad_falls_back_to_dense_cache():
    coords = torch.randn(2, 6, 3, device="cuda", requires_grad=True)
    actual = build_triton_nonperiodic_knn_geometry_cache(
        coords,
        pad_mask=None,
        k_neighbors=3,
        rbf_dim=5,
        cutoff=None,
        seq_len=6,
        strict=False,
    )
    expected = _dense_cache(coords, None, k_neighbors=3, rbf_dim=5, cutoff=None, seq_len=6)

    assert actual.features is not None
    _assert_knn_cache_close(actual, expected)


@pytest.mark.skipif(
    not TRITON_NONPERIODIC_KNN_AVAILABLE or not torch.cuda.is_available(),
    reason="S_g+T dense-vs-compact geometry parity requires CUDA and Triton",
)
def test_sg_t_trunk_dense_and_triton_knn_geometry_outputs_match():
    torch.manual_seed(2)
    base_config = {
        "stream_type": "tetra",
        "num_blocks": 1,
        "block_mix": [1, 1, 0],
        "tetra_dim_per_frame": 2,
        "simplicial": {
            "k_neighbors": 3,
            "num_heads": 1,
            "head_dim": 2,
            "bias": {
                "angle_rank": 4,
                "channels_by_l": {"0": 1, "1": 1, "2": 0},
                "radial_basis_dim": 6,
                "hidden_dim": 8,
                "gate_init": 0.05,
            },
            "kernel": {"backend": "torch"},
            "geometry": {"builder": "dense"},
        },
        "tetra": {"heads_per_frame": 1, "rope_sigma": 1.0},
    }
    dense = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        hybrid_config=base_config,
        geometry_adapter=NonPeriodicGeometryAdapter(),
        use_adaln_conditioning=False,
    ).cuda()
    compact_config = {
        **base_config,
        "simplicial": {
            **base_config["simplicial"],
            "geometry": {"builder": "triton_nonperiodic", "strict": True},
        },
    }
    compact = HybridTransformerTrunk(
        d_model=24,
        n_heads=12,
        n_layers=1,
        hybrid_config=compact_config,
        geometry_adapter=NonPeriodicGeometryAdapter(),
        use_adaln_conditioning=False,
    ).cuda()
    compact.load_state_dict(dense.state_dict())
    dense.eval()
    compact.eval()

    x = torch.randn(2, 7, 24, device="cuda")
    coords = torch.randn(2, 7, 3, device="cuda")
    pad_mask = torch.zeros(2, 7, dtype=torch.bool, device="cuda")
    pad_mask[1, -2:] = True

    with torch.no_grad():
        expected = dense(x, None, pad_mask=pad_mask, coords=coords)
        actual = compact(x, None, pad_mask=pad_mask, coords=coords)

    assert torch.allclose(actual, expected, atol=5e-5, rtol=5e-5)
