import pickle

import numpy as np
import torch

from matterformer.data import QM9Dataset, collate_qm9, compute_target_stats


def _dummy_entry(index: int, num_atoms: int) -> dict[str, object]:
    coords = np.random.randn(num_atoms, 3).astype(np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    return {
        "atom_types": np.arange(num_atoms, dtype=np.int64) % 5,
        "charges": np.zeros(num_atoms, dtype=np.int64),
        "coords": coords,
        "targets": np.arange(19, dtype=np.float32) + index,
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 4), dtype=np.int64),
        "smiles": f"mol-{index}",
        "name": f"mol-{index}",
        "idx": index,
        "num_atoms": num_atoms,
    }


def test_qm9_dataset_uses_cached_processed_file(tmp_path):
    processed_path = tmp_path / "processed_qm9_data.pkl"
    data = [_dummy_entry(0, 3), _dummy_entry(1, 5), _dummy_entry(2, 4)]
    with processed_path.open("wb") as handle:
        pickle.dump(data, handle)

    dataset = QM9Dataset(tmp_path, split=None, target="homo", download=False)
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["targets"].shape == (1,)
    batch = collate_qm9([dataset[0], dataset[1]])
    assert batch.atom_types.shape == (2, 5)
    assert batch.charges.shape == (2, 5)
    assert batch.pad_mask.shape == (2, 5)
    mean, std = compute_target_stats(dataset)
    assert mean.shape == (1,)
    assert std.shape == (1,)
    sampler = dataset.make_num_atoms_sampler()
    sampled = sampler(8)
    assert sampled.shape == (8,)
    assert sampled.min().item() >= 1
    assert sampled.max().item() <= 5


def test_qm9_dataset_accepts_legacy_processed_schema(tmp_path):
    processed_path = tmp_path / "processed_qm9_data.pkl"
    legacy_entry = {
        "pos": np.array([[0.0, 0.1, 0.2], [1.0, -0.2, 0.3]], dtype=np.float32),
        "x": np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.float32),
        "y": np.arange(19, dtype=np.float32),
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "edge_attr": np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        "smiles": "CH",
        "name": "legacy-mol",
        "idx": 17,
        "num_atoms": 2,
        "charges": np.array([6, 1], dtype=np.int64),
    }
    with processed_path.open("wb") as handle:
        pickle.dump([legacy_entry], handle)

    dataset = QM9Dataset(tmp_path, split=None, target="homo", download=False)
    sample = dataset[0]
    assert torch.equal(sample["atom_types"], torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(sample["charges"], torch.tensor([6, 1], dtype=torch.long))
    assert sample["coords"].shape == (2, 3)
    assert sample["targets"].shape == (1,)
    assert sample["idx"] == 17
