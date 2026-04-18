import torch

from matterformer.geometry import NonPeriodicGeometryAdapter, PeriodicGeometryAdapter
from matterformer.models.transformer import GeometryBiasBuilder, SimplicialGeometryBias, TransformerTrunk


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
