from __future__ import annotations

import pytest
import torch

from matterformer.data import (
    SYNTHMOLFORCE_LEVELS,
    SynthMolForceConfig,
    SynthMolForceDataset,
    collate_synthmolforce,
    compute_synthmolforce_energy,
    compute_synthmolforce_energy_batch,
    compute_synthmolforce_labels,
    compute_synthmolforce_labels_batch,
    generate_synthmolforce_batch,
    pack_synthmolforce_samples,
)


def _assert_samples_close(left: dict[str, object], right: dict[str, object]) -> None:
    assert torch.equal(left["atom_types"], right["atom_types"])
    assert torch.allclose(left["coords"], right["coords"])
    assert torch.allclose(left["energy"], right["energy"])
    assert torch.allclose(left["forces"], right["forces"])
    for key in ("pair", "coord", "angle", "chiral"):
        assert torch.allclose(left["component_energies"][key], right["component_energies"][key])


def test_online_generation_is_deterministic_and_epoch_dependent(tmp_path):
    config = SynthMolForceConfig(level="v2", pair_mode="cutoff", num_atoms=8, length=4, seed=123)
    dataset_a = SynthMolForceDataset(tmp_path, split="train", config=config, mode="online")
    dataset_b = SynthMolForceDataset(tmp_path, split="train", config=config, mode="online")

    _assert_samples_close(dataset_a[0], dataset_b[0])

    first_epoch_sample = dataset_a[0]
    dataset_a.set_epoch(1)
    second_epoch_sample = dataset_a[0]
    assert not torch.allclose(first_epoch_sample["coords"], second_epoch_sample["coords"])


def test_fixed_cache_roundtrip_loads_packed_split(tmp_path):
    config = SynthMolForceConfig(level="v3", pair_mode="cutoff", num_atoms=6, length=3, seed=7)
    online = SynthMolForceDataset(tmp_path, split="train", config=config, mode="online")
    samples = [online[index] for index in range(len(online))]
    packed = pack_synthmolforce_samples(samples, config=config, split="train")

    cache_dir = tmp_path / config.cache_name()
    cache_dir.mkdir()
    torch.save(packed, cache_dir / "train.pt")

    fixed = SynthMolForceDataset(tmp_path, split="train", config=config, mode="fixed")
    assert len(fixed) == 3
    _assert_samples_close(fixed[1], samples[1])


def test_collate_pads_masks_and_node_features(tmp_path):
    small_config = SynthMolForceConfig(level="v1", pair_mode="complete", num_atoms=5, length=1, seed=0)
    large_config = SynthMolForceConfig(level="v1", pair_mode="complete", num_atoms=7, length=1, seed=1)
    small = SynthMolForceDataset(tmp_path, config=small_config, mode="online")[0]
    large = SynthMolForceDataset(tmp_path, config=large_config, mode="online")[0]

    batch = collate_synthmolforce([small, large])

    assert batch.atom_types.shape == (2, 7)
    assert batch.coords.shape == (2, 7, 3)
    assert batch.forces.shape == (2, 7, 3)
    assert batch.energy.shape == (2,)
    assert batch.component_energies["pair"].shape == (2,)
    assert batch.pad_mask[0, 5:].all()
    assert not batch.pad_mask[1].any()
    assert torch.allclose(batch.coords[0, 5:], torch.zeros(2, 3))
    assert torch.allclose(batch.forces[0, 5:], torch.zeros(2, 3))
    assert batch.node_features().shape == (2, 7, 10)
    assert torch.allclose(batch.node_features()[0, 5:], torch.zeros(2, 10))
    assert batch.to("cpu").coords.device.type == "cpu"


def test_variable_num_atoms_and_levels_have_finite_labels(tmp_path):
    variable_config = SynthMolForceConfig(level="v2", pair_mode="cutoff", num_atoms=(4, 7), length=6, seed=11)
    variable_dataset = SynthMolForceDataset(tmp_path, config=variable_config, mode="online")
    counts = [int(variable_dataset[index]["num_atoms"]) for index in range(len(variable_dataset))]
    assert all(4 <= count <= 7 for count in counts)

    for level in SYNTHMOLFORCE_LEVELS:
        config = SynthMolForceConfig(level=level, pair_mode="cutoff", num_atoms=6, length=1, seed=13)
        sample = SynthMolForceDataset(tmp_path, config=config, mode="online")[0]
        assert sample["atom_types"].shape == (6,)
        assert sample["coords"].shape == (6, 3)
        assert sample["forces"].shape == (6, 3)
        assert torch.isfinite(sample["energy"])
        assert torch.isfinite(sample["forces"]).all()
        for value in sample["component_energies"].values():
            assert torch.isfinite(value)


def test_energy_is_rotation_invariant_and_forces_are_equivariant(tmp_path):
    config = SynthMolForceConfig(level="v3", pair_mode="cutoff", num_atoms=7, length=1, seed=19)
    sample = SynthMolForceDataset(tmp_path, config=config, mode="online")[0]
    atom_types = sample["atom_types"]
    coords = sample["coords"].double()
    energy, forces, _ = compute_synthmolforce_labels(atom_types, coords, config)

    theta = torch.tensor(0.37, dtype=torch.float64)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, torch.tensor(0.0, dtype=torch.float64)]),
            torch.stack([sin_theta, cos_theta, torch.tensor(0.0, dtype=torch.float64)]),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        ],
    )
    rotated_coords = coords @ rotation.T
    rotated_energy, rotated_forces, _ = compute_synthmolforce_labels(atom_types, rotated_coords, config)

    assert torch.allclose(rotated_energy, energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(rotated_forces, forces @ rotation.T, atol=1e-7, rtol=1e-6)


def test_chiral_level_is_reflection_sensitive():
    atom_types = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [-0.2, 0.8, 0.4],
            [0.1, -0.3, 1.0],
            [-0.7, -0.4, -0.2],
            [0.4, -0.8, 0.5],
        ],
        dtype=torch.float64,
    )
    coords = coords - coords.mean(dim=0, keepdim=True)
    reflected = coords.clone()
    reflected[:, 0] = -reflected[:, 0]
    config = SynthMolForceConfig(level="v3", pair_mode="cutoff", num_atoms=6)

    _, components = compute_synthmolforce_energy(atom_types, coords, config)
    _, reflected_components = compute_synthmolforce_energy(atom_types, reflected, config)

    assert not torch.allclose(components["chiral"], reflected_components["chiral"], atol=1e-8, rtol=1e-8)


def test_batched_energy_and_labels_match_sequential_on_padded_batch(tmp_path):
    config = SynthMolForceConfig(level="v3", pair_mode="cutoff", num_atoms=(4, 7), length=4, seed=29)
    dataset = SynthMolForceDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(len(dataset))]
    batch = collate_synthmolforce(samples)

    energy, components = compute_synthmolforce_energy_batch(
        batch.atom_types,
        batch.coords.double(),
        config,
        pad_mask=batch.pad_mask,
    )
    label_energy, forces, label_components = compute_synthmolforce_labels_batch(
        batch.atom_types,
        batch.coords.double(),
        config,
        pad_mask=batch.pad_mask,
    )

    assert torch.allclose(energy.float(), batch.energy, atol=2e-6, rtol=2e-6)
    assert torch.allclose(label_energy.float(), batch.energy, atol=2e-6, rtol=2e-6)
    assert torch.allclose(forces.float(), batch.forces, atol=2e-6, rtol=2e-6)
    for name in ("pair", "coord", "angle", "chiral"):
        assert torch.allclose(components[name].float(), batch.component_energies[name], atol=2e-6, rtol=2e-6)
        assert torch.allclose(label_components[name].float(), batch.component_energies[name], atol=2e-6, rtol=2e-6)


def test_batched_generation_matches_sequential_generation(tmp_path):
    config = SynthMolForceConfig(level="v3", pair_mode="cutoff", num_atoms=(4, 7), length=6, seed=31)
    dataset = SynthMolForceDataset(tmp_path, split="train", config=config, mode="online")
    indices = [0, 2, 5]

    batch = generate_synthmolforce_batch(indices, split="train", epoch=0, config=config, device="cpu")

    assert batch.coords.device.type == "cpu"
    for row_idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        num_atoms = int(sample["num_atoms"])
        assert torch.equal(batch.atom_types[row_idx, :num_atoms], sample["atom_types"])
        assert torch.allclose(batch.coords[row_idx, :num_atoms], sample["coords"], atol=2e-6, rtol=2e-6)
        assert torch.allclose(batch.forces[row_idx, :num_atoms], sample["forces"], atol=2e-6, rtol=2e-6)
        assert torch.allclose(batch.energy[row_idx], sample["energy"], atol=2e-6, rtol=2e-6)
        for name in ("pair", "coord", "angle", "chiral"):
            assert torch.allclose(
                batch.component_energies[name][row_idx],
                sample["component_energies"][name],
                atol=2e-6,
                rtol=2e-6,
            )


def test_batched_generation_respects_requested_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SynthMolForceConfig(level="v2", pair_mode="cutoff", num_atoms=5, length=3, seed=37)

    batch = generate_synthmolforce_batch([0, 1, 2], split="test", epoch=4, config=config, device=device)

    assert batch.atom_types.device == device
    assert batch.coords.device == device
    assert batch.forces.device == device
    assert batch.energy.device == device
    assert batch.pad_mask.device == device
    assert torch.isfinite(batch.energy).all()
    assert torch.isfinite(batch.forces).all()


def test_invalid_config_errors(tmp_path):
    with pytest.raises(ValueError, match="level"):
        SynthMolForceConfig(level="v4")
    with pytest.raises(ValueError, match="pair_mode"):
        SynthMolForceConfig(pair_mode="bad")
    with pytest.raises(ValueError, match="num_atoms"):
        SynthMolForceConfig(num_atoms=(8, 4))
    with pytest.raises(ValueError, match="mode"):
        SynthMolForceDataset(tmp_path, mode="bad")
