from __future__ import annotations

from dataclasses import dataclass

import torch

from matterformer.data.mof_bwdb import collate_mof_stage1, structure_to_mof_sample
from matterformer.models import MOFStage1EDMModel
from matterformer.tasks import (
    MOFStage1EDMLoss,
    MOFStage1EDMPreconditioner,
    lattice_latent_to_lattice_params,
    mof_stage1_edm_sampler,
    y1_to_lattice_params,
)


@dataclass
class _Block:
    block_type: str
    cart_coords: torch.Tensor
    atomic_numbers: torch.Tensor
    bond_indices: torch.Tensor
    bond_types: torch.Tensor
    formal_charges: torch.Tensor
    chirality_tags: torch.Tensor


@dataclass
class _Structure:
    blocks: list[_Block]
    cart_coords: torch.Tensor
    frac_coords: torch.Tensor
    lattice: torch.Tensor
    cell: torch.Tensor


def _make_block(block_type: str, frac_coords: list[list[float]], atomic_numbers: list[int]) -> _Block:
    frac = torch.tensor(frac_coords, dtype=torch.float32)
    return _Block(
        block_type=block_type,
        cart_coords=frac.clone(),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        bond_indices=torch.zeros(2, 0, dtype=torch.long),
        bond_types=torch.zeros(0, dtype=torch.long),
        formal_charges=torch.zeros(len(atomic_numbers), dtype=torch.float32),
        chirality_tags=torch.zeros(len(atomic_numbers), dtype=torch.long),
    )


def _make_synthetic_structure() -> _Structure:
    block_a = _make_block(
        "NODE",
        frac_coords=[[0.00, 0.00, 0.00], [0.10, 0.00, 0.00]],
        atomic_numbers=[29, 8],
    )
    block_b = _make_block(
        "LINKER",
        frac_coords=[[0.70, 0.20, 0.00], [0.80, 0.20, 0.00]],
        atomic_numbers=[6, 6],
    )
    block_c = _make_block(
        "LINKER",
        frac_coords=[[1.20, 0.40, 0.10], [1.30, 0.40, 0.10]],
        atomic_numbers=[6, 6],
    )
    blocks = [block_a, block_b, block_c]
    cart_coords = torch.cat([block.cart_coords for block in blocks], dim=0)
    frac_coords = cart_coords.clone()
    lattice = torch.tensor([1.0, 1.0, 1.0, 90.0, 90.0, 90.0], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)
    return _Structure(
        blocks=blocks,
        cart_coords=cart_coords,
        frac_coords=frac_coords,
        lattice=lattice,
        cell=cell,
    )


def _make_stage1_batch():
    sample = structure_to_mof_sample(_make_synthetic_structure(), key="synthetic")
    return collate_mof_stage1([sample, sample])


def test_stage1_public_exports_resolve():
    assert MOFStage1EDMModel is not None
    assert MOFStage1EDMLoss is not None
    assert MOFStage1EDMPreconditioner is not None
    assert callable(mof_stage1_edm_sampler)
    assert callable(y1_to_lattice_params)


def test_mof_stage1_loss_and_sampler_smoke():
    torch.manual_seed(0)
    batch = _make_stage1_batch()
    model = MOFStage1EDMModel(
        block_feature_dim=batch.block_features.shape[-1],
        d_model=32,
        n_heads=4,
        n_layers=2,
    )
    net = MOFStage1EDMPreconditioner(model)
    criterion = MOFStage1EDMLoss()

    loss, diagnostics = criterion(net, batch)
    assert loss.ndim == 0
    assert diagnostics["sigma"].shape == (2,)
    assert torch.isfinite(loss)
    for key in ("coord_loss", "lattice_loss", "coord_frac_rmse", "length_mae", "angle_mae"):
        assert diagnostics[key].ndim == 0
        assert torch.isfinite(diagnostics[key])
    loss.backward()

    coords, lattice_latent, pad_mask = mof_stage1_edm_sampler(
        net,
        batch.block_features,
        batch.block_type_ids,
        batch.num_blocks,
        num_steps=3,
        sigma_min=0.01,
        sigma_max=1.0,
        s_churn=0.0,
        s_noise=1.0,
    )
    lattice_params = lattice_latent_to_lattice_params(lattice_latent, lattice_repr="ltri")
    assert coords.shape == batch.block_com_frac.shape
    assert lattice_latent.shape == batch.lattice.shape
    assert lattice_params.shape == batch.lattice.shape
    assert pad_mask.shape == batch.block_pad_mask.shape
    assert torch.isfinite(coords).all()
    assert torch.isfinite(lattice_latent).all()
    assert torch.isfinite(lattice_params).all()


def test_y1_decode_helper_smoke():
    lattice_y1 = torch.tensor([[0.0, 0.1, -0.2, 0.0, 0.1, -0.1]], dtype=torch.float32)
    lattice_params = y1_to_lattice_params(lattice_y1)
    assert lattice_params.shape == (1, 6)
    assert torch.isfinite(lattice_params).all()
