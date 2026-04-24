from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from matterformer.data.mof_bwdb import (
    BWDBDataset,
    MOF_MAX_ATOMIC_NUMBER,
    MOF_BLOCK_TYPE_TO_ID,
    cartesian_to_fractional,
    collate_mof_stage1,
    collate_mof_stage2,
    compute_block_com_cart,
    structure_to_mof_sample,
    unwrap_block_fractional_coords,
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


def _make_block(
    block_type: str,
    frac_coords: list[list[float]],
    atomic_numbers: list[int],
    bond_indices: list[list[int]] | None = None,
) -> _Block:
    frac = torch.tensor(frac_coords, dtype=torch.float32)
    if bond_indices is None:
        bond_index_tensor = torch.zeros(2, 0, dtype=torch.long)
    else:
        bond_index_tensor = torch.tensor(bond_indices, dtype=torch.long)
    return _Block(
        block_type=block_type,
        cart_coords=frac.clone(),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        bond_indices=bond_index_tensor,
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
    lattice = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)
    return _Structure(
        blocks=blocks,
        cart_coords=cart_coords,
        frac_coords=frac_coords,
        lattice=lattice,
        cell=cell,
    )


def test_pbc_safe_block_com_uses_unwrapped_cartesian_coordinates():
    cell = torch.eye(3, dtype=torch.float32)
    frac_unwrapped = torch.tensor([[0.95, 0.0, 0.0], [1.05, 0.0, 0.0]], dtype=torch.float32)
    frac_wrapped = frac_unwrapped.remainder(1.0)
    atom_types = torch.tensor([1, 1], dtype=torch.long)

    com_cart = compute_block_com_cart(frac_unwrapped, atom_types)
    com_frac = cartesian_to_fractional(com_cart[None, :], cell)[0]
    naive_wrapped_mean = frac_wrapped.mean(dim=0)

    assert torch.allclose(com_frac, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(naive_wrapped_mean, torch.tensor([0.5, 0.0, 0.0]), atol=1e-6)
    assert not torch.allclose(com_frac, naive_wrapped_mean)


def test_unwrap_block_fractional_coords_tracks_bonded_linker_across_boundary():
    frac_wrapped = torch.tensor([[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=torch.float32)
    atomic_numbers = torch.tensor([6, 6], dtype=torch.long)
    bond_indices = torch.tensor([[0], [1]], dtype=torch.long)

    frac_unwrapped = unwrap_block_fractional_coords(frac_wrapped, bond_indices, atomic_numbers)

    assert torch.allclose(frac_unwrapped[:, 0], torch.tensor([0.95, 1.05]), atol=1e-6)


def test_structure_to_mof_sample_preserves_block_and_atom_invariants():
    sample = structure_to_mof_sample(_make_synthetic_structure(), key="synthetic")

    assert sample.key == "synthetic"
    assert sample.num_atoms == 6
    assert sample.num_blocks == 3
    assert sample.block_sizes.sum().item() == sample.num_atoms
    assert sample.block_index.min().item() == 0
    assert sample.block_index.max().item() == sample.num_blocks - 1
    assert torch.equal(sample.block_type_ids, torch.tensor([0, 1, 1], dtype=torch.long))

    structure_hist = torch.bincount(sample.atom_types, minlength=MOF_MAX_ATOMIC_NUMBER + 1)[1:].float()
    block_hist = sample.block_element_count_vecs.sum(dim=0)
    assert torch.equal(block_hist, structure_hist)


def test_structure_to_mof_sample_uses_pbc_aware_block_centers_for_cross_boundary_linker():
    block = _make_block(
        "LINKER",
        frac_coords=[[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]],
        atomic_numbers=[6, 6],
        bond_indices=[[0], [1]],
    )
    structure = _Structure(
        blocks=[block],
        cart_coords=block.cart_coords.clone(),
        frac_coords=block.cart_coords.clone(),
        lattice=torch.zeros(6, dtype=torch.float32),
        cell=torch.eye(3, dtype=torch.float32),
    )

    sample = structure_to_mof_sample(structure, key="cross_boundary")

    assert torch.allclose(sample.block_com_frac[0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(
        sample.atom_local_frac[:, 0],
        torch.tensor([-0.05, 0.05]),
        atol=1e-6,
    )


def test_collate_mof_stage1_pads_blocks_without_reordering_repeated_compositions():
    sample = structure_to_mof_sample(_make_synthetic_structure(), key="synthetic")
    batch = collate_mof_stage1([sample, sample])

    assert batch.block_features.shape == (2, 3, MOF_MAX_ATOMIC_NUMBER)
    assert batch.block_pad_mask.shape == (2, 3)
    assert batch.num_blocks.tolist() == [3, 3]
    assert not batch.block_pad_mask.any()

    linker_id = MOF_BLOCK_TYPE_TO_ID["LINKER"]
    assert batch.block_type_ids[0, 1].item() == linker_id
    assert batch.block_type_ids[0, 2].item() == linker_id
    assert torch.equal(batch.block_features[0, 1], batch.block_features[0, 2])
    assert torch.allclose(batch.block_com_frac[0, : sample.num_blocks], sample.block_com_frac)


def test_collate_mof_stage2_builds_atom_priors_from_parent_block_coms():
    sample = structure_to_mof_sample(_make_synthetic_structure(), key="synthetic")
    batch = collate_mof_stage2([sample])

    assert batch.atom_types.shape == (1, sample.num_atoms)
    assert batch.atom_pad_mask.shape == (1, sample.num_atoms)
    assert not batch.atom_pad_mask.any()
    assert torch.equal(batch.atom_types[0], sample.atom_types)
    assert torch.equal(batch.atom_block_index[0], sample.block_index)

    expected_prior = sample.block_com_frac[sample.block_index]
    assert torch.allclose(batch.atom_prior_mu_frac[0], expected_prior)
    assert torch.isfinite(batch.atom_local_frac).all()
    assert torch.isfinite(batch.atom_local_cart).all()


def test_bwdb_dataset_smoke_loads_real_samples_and_collates():
    data_root = Path(__file__).resolve().parents[1] / "data" / "mofs" / "bwdb"
    if not data_root.exists():
        pytest.skip(f"BWDB data not available at {data_root}")

    dataset = BWDBDataset(data_root, split="train", sample_limit=2)
    assert len(dataset) == 2

    samples = [dataset[0], dataset[1]]
    for sample in samples:
        assert sample.block_sizes.sum().item() == sample.num_atoms
        assert sample.block_index.min().item() == 0
        assert sample.block_index.max().item() == sample.num_blocks - 1
        structure_hist = torch.bincount(sample.atom_types, minlength=MOF_MAX_ATOMIC_NUMBER + 1)[1:].float()
        block_hist = sample.block_element_count_vecs.sum(dim=0)
        assert torch.equal(block_hist, structure_hist)

    stage1 = collate_mof_stage1(samples)
    stage2 = collate_mof_stage2(samples)
    assert stage1.block_features.shape[0] == 2
    assert stage1.block_com_frac.shape[-1] == 3
    assert stage2.atom_types.shape[0] == 2
    assert stage2.atom_prior_mu_frac.shape == stage2.atom_frac.shape
