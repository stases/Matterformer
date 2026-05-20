import pytest
import torch

from matterformer.data import (
    FairChemOMolDataset,
    OMolDynamicBatchSampler,
    SyntheticOMolDataset,
    collate_omol,
)


class FixedAtomDataset:
    def __init__(self, counts):
        self.counts = list(counts)

    def __len__(self):
        return len(self.counts)

    def get_num_atoms(self, idx):
        return self.counts[int(idx)]


class LengthOnlyDataset:
    def __init__(self, length):
        self.length = int(length)

    def __len__(self):
        return self.length


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


def test_dynamic_sampler_bucketed_mode_is_deterministic_and_valid():
    dataset = SyntheticOMolDataset(num_samples=24, seed=3, min_atoms=2, max_atoms=9)
    sampler = OMolDynamicBatchSampler(
        dataset,
        max_batch_size=6,
        max_atoms=18,
        shuffle=True,
        seed=7,
        batching_mode="bucketed",
        bucket_window_size=12,
        bucket_shuffle_groups=3,
    )
    first = list(sampler)
    sampler.set_epoch(0)
    second = list(sampler)
    assert first == second
    assert sorted(idx for batch in first for idx in batch) == list(range(len(dataset)))
    for batch in first:
        assert len(batch) <= 6
        assert sum(dataset.get_num_atoms(idx) for idx in batch) <= 18 or len(batch) == 1


def test_dynamic_sampler_rank_slices_indices_before_packing():
    dataset = FixedAtomDataset([6, 6, 6, 6, 4, 4, 4, 4])
    samplers = [
        OMolDynamicBatchSampler(
            dataset,
            max_batch_size=10,
            max_atoms=10,
            shuffle=False,
            num_replicas=2,
            rank=rank,
        )
        for rank in (0, 1)
    ]

    rank_batches = [list(sampler) for sampler in samplers]
    assert [idx for batch in rank_batches[0] for idx in batch] == [0, 2, 4, 6]
    assert [idx for batch in rank_batches[1] for idx in batch] == [1, 3, 5, 7]
    for batches in rank_batches:
        for batch in batches:
            assert sum(dataset.get_num_atoms(idx) for idx in batch) <= 10 or len(batch) == 1


def test_dynamic_sampler_pads_indices_before_ddp_rank_slice():
    dataset = FixedAtomDataset([1, 1, 1, 1, 1])
    samplers = [
        OMolDynamicBatchSampler(
            dataset,
            max_batch_size=10,
            max_atoms=10,
            shuffle=False,
            num_replicas=2,
            rank=rank,
            pad_distributed_batches=False,
        )
        for rank in (0, 1)
    ]

    assert [[idx for batch in sampler for idx in batch] for sampler in samplers] == [[0, 2, 4], [1, 3, 0]]


def test_dynamic_sampler_len_matches_platonic_ddp_estimate():
    dataset = LengthOnlyDataset(3_986_754)
    sampler = OMolDynamicBatchSampler(dataset, max_batch_size=10_000, max_atoms=3000, num_replicas=4, rank=0)
    assert len(sampler) == 16612


def test_fairchem_dataset_does_not_require_fairchem(tmp_path):
    with pytest.raises(ValueError, match="No .aselmdb files found"):
        FairChemOMolDataset(tmp_path)
