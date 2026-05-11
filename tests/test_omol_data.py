import pytest
import torch

from matterformer.data import (
    FairChemOMolDataset,
    OMolDynamicBatchSampler,
    SyntheticOMolDataset,
    collate_omol,
)


def test_collate_omol_variable_atoms_and_masks():
    samples = [
        {
            "atomic_numbers": torch.tensor([6, 1, 1]),
            "pos": torch.randn(3, 3),
            "forces": torch.ones(3, 3),
            "energy": torch.tensor([-10.0]),
            "charge": torch.tensor([1]),
            "spin": torch.tensor([2]),
            "fixed": torch.tensor([False, True, False]),
            "idx": 4,
        },
        {
            "atomic_numbers": torch.tensor([8, 1]),
            "pos": torch.randn(2, 3),
            "forces": torch.zeros(2, 3),
            "energy": torch.tensor([-5.0]),
        },
    ]
    batch = collate_omol(samples)
    assert batch.atomic_numbers.shape == (2, 3)
    assert batch.coords.shape == (2, 3, 3)
    assert batch.forces.shape == (2, 3, 3)
    assert torch.equal(batch.pad_mask, torch.tensor([[False, False, False], [False, False, True]]))
    assert torch.equal(batch.free_atom_mask[0], torch.tensor([True, False, True]))
    assert torch.equal(batch.free_atom_mask[1], torch.tensor([True, True, False]))
    assert batch.charge.tolist() == [1, 0]
    assert batch.spin.tolist() == [2, 0]
    assert batch.indices.tolist() == [4, 0]


def test_dynamic_sampler_respects_max_atoms():
    dataset = SyntheticOMolDataset(num_samples=8, seed=1, min_atoms=2, max_atoms=4)
    sampler = OMolDynamicBatchSampler(dataset, max_batch_size=10, max_atoms=6, shuffle=False)
    batches = list(sampler)
    assert batches
    for batch in batches:
        assert sum(dataset.get_num_atoms(idx) for idx in batch) <= 6 or len(batch) == 1


def test_fairchem_dataset_does_not_require_fairchem(tmp_path):
    with pytest.raises(ValueError, match="No .aselmdb files found"):
        FairChemOMolDataset(tmp_path)
