import torch

from matterformer.geometry import NonPeriodicGeometryAdapter, PeriodicGeometryAdapter


def test_nonperiodic_geometry_adapter_masks_and_normalizes():
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    pad_mask = torch.tensor([[False, False, True]])
    adapter = NonPeriodicGeometryAdapter()
    features = adapter(coords, pad_mask=pad_mask)

    assert features.pair_delta.shape == (1, 3, 3, 3)
    assert features.pair_dist.shape == (1, 3, 3)
    assert features.pair_mask.shape == (1, 3, 3)
    assert torch.allclose(features.pair_dist[0, 0, 1], torch.tensor(1.0))
    assert features.pair_mask[0, 0, 2].item() is False
    assert torch.all(features.pair_dist[0, 2] == 0.0)
    assert features.global_geom.shape == (1, 1)
    assert torch.isfinite(features.pair_dist_norm).all()


def test_periodic_geometry_adapter_uses_minimum_image():
    coords = torch.tensor([[[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]], dtype=torch.float32)
    lattice = torch.zeros(1, 6, dtype=torch.float32)
    adapter = PeriodicGeometryAdapter(pbc_radius=1, lattice_repr="y1")
    features = adapter(coords, lattice=lattice)

    assert torch.allclose(features.pair_dist[0, 0, 1], torch.tensor(0.2), atol=1e-6)
    assert torch.allclose(features.pair_delta[0, 0, 1], torch.tensor([0.2, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(features.pair_delta[0, 1, 0], torch.tensor([-0.2, 0.0, 0.0]), atol=1e-6)
    assert features.global_geom.shape == (1, 6)
