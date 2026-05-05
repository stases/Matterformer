from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from matterformer.data import (
    SYNOMOL_TRANSFER_ALIAS_PAIRS,
    SYNOMOL_TRANSFER_COMPONENT_NAMES,
    SYNOMOL_TRANSFER_NUM_ATOM_TYPES,
    SYNOMOL_TRANSFER_OOD_MOTIFS,
    SYNOMOL_TRANSFER_SPLITS,
    SynOMolTransferConfig,
    SynOMolTransferDataset,
    calibrate_synomol_transfer_component_scales,
    collate_synomol_transfer,
    compute_synomol_transfer_energy,
    compute_synomol_transfer_energy_batch_dense,
    compute_synomol_transfer_energy_batch_local,
    compute_synomol_transfer_labels,
    compute_synomol_transfer_labels_batch_dense,
    compute_synomol_transfer_labels_batch_local,
    compute_synomol_transfer_multilevel_labels_batch_local,
    langevin_coords_batch,
    materialize_synomol_transfer_split,
    pack_synomol_transfer_samples,
    summarize_synomol_transfer_samples,
    synomol_transfer_atom_type_table,
)
from matterformer.data.synomol_transfer import _active_type_triples, _motif_directions

_PREPARE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_synomol_transfer_data.py"
_PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_synomol_transfer_data_for_tests", _PREPARE_SCRIPT_PATH)
assert _PREPARE_SPEC is not None and _PREPARE_SPEC.loader is not None
_PREPARE_MODULE = importlib.util.module_from_spec(_PREPARE_SPEC)
_PREPARE_SPEC.loader.exec_module(_PREPARE_MODULE)
_cleanup_stale_split_outputs = _PREPARE_MODULE._cleanup_stale_split_outputs
_enforce_label_cap_policy = _PREPARE_MODULE._enforce_label_cap_policy
_generate_range_gpu_full = _PREPARE_MODULE._generate_range_gpu_full


def _small_config(length: int = 4) -> SynOMolTransferConfig:
    return SynOMolTransferConfig(
        num_atoms=(5, 7),
        size_ood_num_atoms=(8, 9),
        length=length,
        seed=123,
        radial_basis_size=4,
        angle_lmax=4,
        relax_steps=2,
        relaxation_snapshot_steps=1,
        langevin_steps=1,
    )


def _assert_samples_close(left: dict[str, object], right: dict[str, object]) -> None:
    assert torch.equal(left["atom_types"], right["atom_types"])
    assert torch.allclose(left["coords"], right["coords"])
    assert torch.allclose(left["energy"], right["energy"])
    assert torch.allclose(left["forces"], right["forces"])
    assert torch.equal(left["motif_labels"], right["motif_labels"])
    assert left["sample_kind"] == right["sample_kind"]
    assert left["primary_motif"] == right["primary_motif"]
    assert bool(left["has_heldout_type_combo"]) == bool(right["has_heldout_type_combo"])
    for key in SYNOMOL_TRANSFER_COMPONENT_NAMES:
        assert torch.allclose(left["component_energies"][key], right["component_energies"][key])


def test_online_generation_is_deterministic_and_epoch_dependent(tmp_path):
    config = _small_config(length=3)
    dataset_a = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    dataset_b = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")

    _assert_samples_close(dataset_a[0], dataset_b[0])

    first_epoch_sample = dataset_a[0]
    dataset_a.set_epoch(1)
    second_epoch_sample = dataset_a[0]
    assert not torch.allclose(first_epoch_sample["coords"], second_epoch_sample["coords"])


def test_fixed_cache_roundtrip_loads_packed_split(tmp_path):
    config = _small_config(length=3)
    online = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    samples = [online[index] for index in range(len(online))]
    packed = pack_synomol_transfer_samples(samples, config=config, split="train")

    cache_dir = tmp_path / config.cache_name()
    cache_dir.mkdir()
    torch.save(packed, cache_dir / "train.pt")

    fixed = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="fixed")
    assert len(fixed) == 3
    _assert_samples_close(fixed[1], samples[1])


def test_fixed_cache_resolves_manifest_alias(tmp_path):
    config = _small_config(length=2)
    online = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    samples = [online[index] for index in range(len(online))]
    packed = pack_synomol_transfer_samples(samples, config=config, split="train")
    cache_dir = tmp_path / "synomol_transfer_v1_default"
    cache_dir.mkdir()
    torch.save(packed, cache_dir / "train.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"aliases": {"default": "synomol_transfer_v1_default"}}),
    )

    fixed = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="fixed", config_name="default")
    _assert_samples_close(fixed[0], samples[0])


def test_fixed_cache_loads_sharded_split(tmp_path):
    config = _small_config(length=4)
    online = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    samples = [online[index] for index in range(len(online))]
    cache_dir = tmp_path / config.cache_name()
    cache_dir.mkdir()
    shard_a = pack_synomol_transfer_samples(samples[:2], config=config, split="train")
    shard_b = pack_synomol_transfer_samples(samples[2:], config=config, split="train")
    torch.save(shard_a, cache_dir / "train_00000.pt")
    torch.save(shard_b, cache_dir / "train_00001.pt")
    (cache_dir / "train_manifest.json").write_text(
        json.dumps(
            {
                "format": "synomol_transfer_sharded_v1",
                "split": "train",
                "num_samples": 4,
                "shards": [
                    {"path": "train_00000.pt", "num_samples": 2},
                    {"path": "train_00001.pt", "num_samples": 2},
                ],
            }
        )
    )

    fixed = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="fixed")
    assert len(fixed) == 4
    _assert_samples_close(fixed[3], samples[3])


def test_fixed_cache_prefers_sharded_manifest_over_stale_single_file(tmp_path):
    config = _small_config(length=4)
    online = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    samples = [online[index] for index in range(len(online))]
    cache_dir = tmp_path / config.cache_name()
    cache_dir.mkdir()
    torch.save(pack_synomol_transfer_samples(samples[:2], config=config, split="train"), cache_dir / "train.pt")
    torch.save(pack_synomol_transfer_samples(samples[2:], config=config, split="train"), cache_dir / "train_00000.pt")
    (cache_dir / "train_manifest.json").write_text(
        json.dumps(
            {
                "format": "synomol_transfer_sharded_v1",
                "split": "train",
                "num_samples": 2,
                "shards": [{"path": "train_00000.pt", "num_samples": 2}],
            }
        )
    )

    fixed = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="fixed")
    assert len(fixed) == 2
    _assert_samples_close(fixed[0], samples[2])


def test_cache_cleanup_removes_stale_alternate_representations(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train.pt").write_bytes(b"stale")
    _cleanup_stale_split_outputs(cache_dir, "train", writing_sharded=True)
    assert not (cache_dir / "train.pt").exists()

    (cache_dir / "train_00000.pt").write_bytes(b"stale")
    (cache_dir / "train_manifest.json").write_text(
        json.dumps({"shards": [{"path": "train_00000.pt", "num_samples": 1}]})
    )
    _cleanup_stale_split_outputs(cache_dir, "train", writing_sharded=False)
    assert not (cache_dir / "train_manifest.json").exists()
    assert not (cache_dir / "train_00000.pt").exists()


def test_cache_name_hash_ignores_length_but_tracks_generation_fields():
    base = _small_config(length=2)
    same_distribution = SynOMolTransferConfig.from_dict({**base.__dict__, "length": 99})
    changed_distribution = SynOMolTransferConfig.from_dict({**base.__dict__, "relax_steps": base.relax_steps + 1})

    assert base.cache_name() == same_distribution.cache_name()
    assert base.cache_name() != changed_distribution.cache_name()


def test_collate_pads_masks_and_node_features(tmp_path):
    small_config = _small_config(length=1)
    large_config = SynOMolTransferConfig.from_dict({**small_config.__dict__, "num_atoms": 7, "length": 1})
    small = SynOMolTransferDataset(tmp_path, config=small_config, mode="online")[0]
    large = SynOMolTransferDataset(tmp_path, config=large_config, mode="online")[0]

    batch = collate_synomol_transfer([small, large])

    assert batch.atom_types.shape == (2, 7)
    assert batch.coords.shape == (2, 7, 3)
    assert batch.forces.shape == (2, 7, 3)
    assert batch.energy.shape == (2,)
    assert batch.component_energies["pair"].shape == (2,)
    assert batch.pad_mask[0, int(small["num_atoms"]):].all()
    assert not batch.pad_mask[1].any()
    assert torch.allclose(batch.coords[0][batch.pad_mask[0]], torch.zeros_like(batch.coords[0][batch.pad_mask[0]]))
    assert torch.allclose(batch.forces[0][batch.pad_mask[0]], torch.zeros_like(batch.forces[0][batch.pad_mask[0]]))
    assert torch.equal(batch.motif_labels[0][batch.pad_mask[0]], torch.full_like(batch.motif_labels[0][batch.pad_mask[0]], -1))
    assert batch.node_features().shape == (2, 7, 16)
    assert torch.allclose(batch.node_features()[0][batch.pad_mask[0]], torch.zeros_like(batch.node_features()[0][batch.pad_mask[0]]))
    assert batch.to("cpu").coords.device.type == "cpu"


def test_all_splits_have_finite_labels_and_expected_metadata(tmp_path):
    config = _small_config(length=2)
    for split in SYNOMOL_TRANSFER_SPLITS:
        dataset = SynOMolTransferDataset(tmp_path, split=split, config=config, mode="online")
        sample = dataset[0]
        assert sample["atom_types"].ndim == 1
        assert sample["coords"].shape == (sample["num_atoms"], 3)
        assert sample["forces"].shape == (sample["num_atoms"], 3)
        assert sample["motif_labels"].shape == (sample["num_atoms"],)
        assert torch.isfinite(sample["energy"])
        assert torch.isfinite(sample["forces"]).all()
        for value in sample["component_energies"].values():
            assert torch.isfinite(value)


def test_forces_match_negative_energy_gradient(tmp_path):
    config = _small_config(length=1)
    sample = SynOMolTransferDataset(tmp_path, config=config, mode="online")[0]
    atom_types = sample["atom_types"]
    coords = sample["coords"].double().detach().requires_grad_(True)
    energy, _ = compute_synomol_transfer_energy(atom_types, coords, config)
    (grad_coords,) = torch.autograd.grad(energy, coords)

    _, forces, _ = compute_synomol_transfer_labels(atom_types, sample["coords"].double(), config)

    assert torch.allclose(forces, -grad_coords.detach(), atol=1e-7, rtol=1e-6)


def test_batched_dense_oracle_matches_single_sample_oracle(tmp_path):
    config = _small_config(length=2)
    dataset = SynOMolTransferDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(2)]
    batch = collate_synomol_transfer(samples)
    mask = ~batch.pad_mask

    energy, components = compute_synomol_transfer_energy_batch_dense(
        batch.atom_types,
        batch.coords.double(),
        mask,
        config,
    )

    for index, sample in enumerate(samples):
        single_energy, single_components = compute_synomol_transfer_energy(
            sample["atom_types"],
            sample["coords"].double(),
            config,
        )
        assert torch.allclose(energy[index], single_energy, atol=1e-8, rtol=1e-7)
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
            assert torch.allclose(components[name][index], single_components[name], atol=1e-8, rtol=1e-7)


def test_batched_dense_forces_match_single_sample_forces(tmp_path):
    config = _small_config(length=2)
    dataset = SynOMolTransferDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(2)]
    batch = collate_synomol_transfer(samples)
    energy, forces, _ = compute_synomol_transfer_labels_batch_dense(
        batch.atom_types,
        batch.coords.double(),
        ~batch.pad_mask,
        config,
    )

    assert energy.shape == (2,)
    for index, sample in enumerate(samples):
        single_energy, single_forces, _ = compute_synomol_transfer_labels(
            sample["atom_types"],
            sample["coords"].double(),
            config,
        )
        count = int(sample["num_atoms"])
        assert torch.allclose(energy[index], single_energy, atol=1e-8, rtol=1e-7)
        assert torch.allclose(forces[index, :count], single_forces, atol=1e-7, rtol=1e-6)
        assert torch.allclose(forces[index, count:], torch.zeros_like(forces[index, count:]))


def test_batched_local_oracle_matches_single_when_cap_covers_neighbors(tmp_path):
    config = _small_config(length=2)
    dataset = SynOMolTransferDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(2)]
    batch = collate_synomol_transfer(samples)
    mask = ~batch.pad_mask
    energy, components, stats = compute_synomol_transfer_energy_batch_local(
        batch.atom_types,
        batch.coords.double(),
        mask,
        config,
        k_label=batch.atom_types.shape[1],
        return_stats=True,
    )

    assert stats["neighbor_cap_hit_fraction"] == 0.0
    for index, sample in enumerate(samples):
        single_energy, single_components = compute_synomol_transfer_energy(
            sample["atom_types"],
            sample["coords"].double(),
            config,
        )
        assert torch.allclose(energy[index], single_energy, atol=1e-7, rtol=1e-6)
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
            assert torch.allclose(components[name][index], single_components[name], atol=1e-7, rtol=1e-6)


def test_batched_local_sparse_centers_match_single_sample_oracle():
    config = _small_config(length=1)
    atom_types = torch.tensor([0, 1], dtype=torch.long)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
    energy, components = compute_synomol_transfer_energy(atom_types, coords, config)
    batch_energy, batch_components, stats = compute_synomol_transfer_energy_batch_local(
        atom_types[None],
        coords[None],
        torch.ones(1, 2, dtype=torch.bool),
        config,
        k_label=2,
        return_stats=True,
    )

    assert stats["neighbor_cap_hit_fraction"] == 0.0
    assert torch.allclose(batch_energy[0], energy, atol=1e-8, rtol=1e-7)
    for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
        assert torch.allclose(batch_components[name][0], components[name], atol=1e-8, rtol=1e-7)


def test_batched_local_forces_and_cap_stats_are_finite(tmp_path):
    config = _small_config(length=2)
    dataset = SynOMolTransferDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(2)]
    batch = collate_synomol_transfer(samples)
    energy, forces, components, stats = compute_synomol_transfer_labels_batch_local(
        batch.atom_types,
        batch.coords.double(),
        ~batch.pad_mask,
        config,
        k_label=1,
        return_stats=True,
    )

    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    assert all(torch.isfinite(value).all() for value in components.values())
    assert 0.0 <= stats["neighbor_cap_hit_fraction"] <= 1.0


def test_batched_local_rejects_invalid_valid_atom_types_but_ignores_padding():
    config = _small_config(length=1)
    atom_types = torch.tensor([[0, 1, SYNOMOL_TRANSFER_NUM_ATOM_TYPES]], dtype=torch.long)
    coords = torch.zeros(1, 3, 3, dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])

    energy, components = compute_synomol_transfer_energy_batch_local(atom_types, coords, mask, config, k_label=2)
    assert torch.isfinite(energy).all()
    assert all(torch.isfinite(value).all() for value in components.values())

    bad_mask = torch.tensor([[True, True, True]])
    with pytest.raises(ValueError, match="valid atom_types"):
        compute_synomol_transfer_energy_batch_local(atom_types, coords, bad_mask, config, k_label=2)


def test_multilevel_labels_match_full_local_and_component_force_sum(tmp_path):
    config = _small_config(length=2)
    dataset = SynOMolTransferDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(2)]
    batch = collate_synomol_transfer(samples)
    mask = ~batch.pad_mask
    energy, forces, _, stats = compute_synomol_transfer_labels_batch_local(
        batch.atom_types,
        batch.coords.double(),
        mask,
        config,
        k_label=batch.atom_types.shape[1],
        return_stats=True,
    )
    level_energies, level_forces, component_energies, component_forces, level_stats = (
        compute_synomol_transfer_multilevel_labels_batch_local(
            batch.atom_types,
            batch.coords.double(),
            mask,
            config,
            k_label=batch.atom_types.shape[1],
            return_stats=True,
        )
    )

    assert stats["neighbor_cap_hit_fraction"] == 0.0
    assert level_stats["neighbor_cap_hit_fraction"] == 0.0
    assert torch.allclose(level_energies["full_local"], energy, atol=1e-7, rtol=1e-6)
    assert torch.allclose(level_forces["full_local"], forces, atol=1e-7, rtol=1e-6)
    summed_energy = sum(component_energies[name] for name in SYNOMOL_TRANSFER_COMPONENT_NAMES)
    summed_forces = sum(component_forces[name] for name in SYNOMOL_TRANSFER_COMPONENT_NAMES)
    assert torch.allclose(level_energies["full_local"], summed_energy, atol=1e-7, rtol=1e-6)
    assert torch.allclose(level_forces["full_local"], summed_forces, atol=1e-7, rtol=1e-6)


def test_langevin_batch_noise_is_per_sample_seeded_independent_of_batch_composition():
    config = _small_config(length=1)
    atom_types = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
            [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.2, 0.0]],
        ],
        dtype=torch.float64,
    )
    mask = torch.ones(2, 3, dtype=torch.bool)
    seeds = torch.tensor([1234, 5678], dtype=torch.long)
    batch_out = langevin_coords_batch(
        atom_types,
        coords,
        mask,
        config,
        steps=2,
        k_label=3,
        sample_seeds=seeds,
    )
    single_out = langevin_coords_batch(
        atom_types[1:],
        coords[1:],
        mask[1:],
        config,
        steps=2,
        k_label=3,
        sample_seeds=seeds[1:],
    )

    assert torch.allclose(batch_out[1], single_out[0], atol=1e-10, rtol=1e-10)


def test_label_cap_policy_fails_by_default_and_allows_override():
    stats = {
        "label_neighbor_stats": {
            "neighbor_cap_hit_fraction": {"mean": 0.25, "max": 0.25},
        }
    }
    with pytest.raises(RuntimeError, match="cap hit fraction"):
        _enforce_label_cap_policy(
            stats,
            split="train",
            max_label_cap_hit_fraction=0.0,
            allow_label_cap_hits=False,
        )
    _enforce_label_cap_policy(
        stats,
        split="train",
        max_label_cap_hit_fraction=0.0,
        allow_label_cap_hits=True,
    )
    assert stats["label_cap_policy"]["allow_label_cap_hits"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_local_cuda_parity_when_available(tmp_path):
    config = _small_config(length=2)
    dataset = SynOMolTransferDataset(tmp_path, config=config, mode="online")
    samples = [dataset[index] for index in range(2)]
    batch = collate_synomol_transfer(samples)
    mask = ~batch.pad_mask
    cpu_energy, cpu_components = compute_synomol_transfer_energy_batch_local(
        batch.atom_types,
        batch.coords.double(),
        mask,
        config,
        k_label=batch.atom_types.shape[1],
    )
    cuda_energy, cuda_components = compute_synomol_transfer_energy_batch_local(
        batch.atom_types.cuda(),
        batch.coords.double().cuda(),
        mask.cuda(),
        config,
        k_label=batch.atom_types.shape[1],
    )

    assert torch.allclose(cuda_energy.cpu(), cpu_energy, atol=1e-7, rtol=1e-6)
    for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
        assert torch.allclose(cuda_components[name].cpu(), cpu_components[name], atol=1e-7, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_full_backend_smoke_when_cuda_available():
    config = SynOMolTransferConfig.from_dict(
        {
            **_small_config(length=1).__dict__,
            "relax_steps": 0,
            "relaxation_snapshot_steps": 0,
            "langevin_steps": 0,
        }
    )
    generated = _generate_range_gpu_full(
        start=0,
        end=1,
        split="train",
        split_config=config,
        generation_device="cuda",
        generation_batch_size=1,
        k_label=8,
        max_gpu_work=1_000_000,
    )

    sample, stats = generated[0]
    assert torch.isfinite(sample["energy"])
    assert torch.isfinite(sample["forces"]).all()
    assert stats["backend"] == "gpu-full"


def test_oracle_uses_total_energy_for_force_derivatives():
    config = _small_config(length=1)
    atom_types = torch.tensor([2, 0, 1, 6], dtype=torch.long)
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [-0.4, 1.2, 0.0],
            [0.1, -0.2, 1.3],
        ],
        dtype=torch.float64,
    )
    energy, forces, _ = compute_synomol_transfer_labels(atom_types, coords, config)
    offset = torch.tensor([25.0, 0.0, 0.0], dtype=torch.float64)
    duplicated_types = torch.cat([atom_types, atom_types], dim=0)
    duplicated_coords = torch.cat([coords, coords + offset], dim=0)
    duplicated_energy, duplicated_forces, _ = compute_synomol_transfer_labels(duplicated_types, duplicated_coords, config)

    assert torch.allclose(duplicated_energy, 2.0 * energy, atol=1e-8, rtol=1e-7)
    assert torch.allclose(duplicated_forces[: atom_types.shape[0]], forces, atol=1e-8, rtol=1e-7)


def test_energy_is_rotation_invariant_and_forces_are_equivariant(tmp_path):
    config = _small_config(length=1)
    sample = SynOMolTransferDataset(tmp_path, config=config, mode="online")[0]
    atom_types = sample["atom_types"]
    coords = sample["coords"].double()
    energy, forces, _ = compute_synomol_transfer_labels(atom_types, coords, config)

    theta = torch.tensor(0.41, dtype=torch.float64)
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
    rotated_energy, rotated_forces, _ = compute_synomol_transfer_labels(atom_types, rotated_coords, config)

    assert torch.allclose(rotated_energy, energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(rotated_forces, forces @ rotation.T, atol=1e-7, rtol=1e-6)


def test_translation_invariance_zero_net_force_and_zero_torque(tmp_path):
    config = _small_config(length=1)
    sample = SynOMolTransferDataset(tmp_path, config=config, mode="online")[0]
    atom_types = sample["atom_types"]
    coords = sample["coords"].double()
    energy, forces, _ = compute_synomol_transfer_labels(atom_types, coords, config)

    shift = torch.tensor([3.0, -2.0, 0.7], dtype=torch.float64)
    shifted_energy, shifted_forces, _ = compute_synomol_transfer_labels(atom_types, coords + shift, config)
    torque = torch.cross(coords, forces, dim=-1).sum(dim=0)

    assert torch.allclose(shifted_energy, energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(shifted_forces, forces, atol=1e-7, rtol=1e-6)
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(torque, torch.zeros_like(torque), atol=1e-5)


def test_permutation_invariance_and_force_equivariance(tmp_path):
    config = _small_config(length=1)
    sample = SynOMolTransferDataset(tmp_path, config=config, mode="online")[0]
    atom_types = sample["atom_types"]
    coords = sample["coords"].double()
    energy, forces, _ = compute_synomol_transfer_labels(atom_types, coords, config)

    perm = torch.arange(atom_types.shape[0] - 1, -1, -1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    perm_energy, perm_forces, _ = compute_synomol_transfer_labels(atom_types[perm], coords[perm], config)

    assert torch.allclose(perm_energy, energy, atol=1e-8, rtol=1e-8)
    assert torch.allclose(perm_forces[inv], forces, atol=1e-7, rtol=1e-6)


def test_alias_types_share_radial_parameters_but_differ_in_angular_preferences():
    table = synomol_transfer_atom_type_table()
    for left, right in SYNOMOL_TRANSFER_ALIAS_PAIRS:
        assert torch.allclose(table["size"][left], table["size"][right])
        assert torch.allclose(table["pair_strength"][left], table["pair_strength"][right])
        assert not torch.allclose(table["preferred_cos"][left], table["preferred_cos"][right])


def test_alias_types_have_same_pair_energy_but_different_angular_or_motif_energy():
    config = _small_config(length=1)
    table = synomol_transfer_atom_type_table()
    left_alias, right_alias = SYNOMOL_TRANSFER_ALIAS_PAIRS[0]
    neighbor_types = [0, 1, 6, 7]
    directions = _motif_directions("tetrahedral").double()
    radius = float(table["size"][left_alias] + table["size"][0] + 0.05)
    coords = torch.cat([torch.zeros(1, 3, dtype=torch.float64), directions * radius], dim=0)
    atom_types_left = torch.tensor([left_alias, *neighbor_types], dtype=torch.long)
    atom_types_right = torch.tensor([right_alias, *neighbor_types], dtype=torch.long)

    _, components_left = compute_synomol_transfer_energy(atom_types_left, coords, config)
    _, components_right = compute_synomol_transfer_energy(atom_types_right, coords, config)

    assert torch.allclose(components_left["pair"], components_right["pair"], atol=1e-8, rtol=1e-7)
    angular_delta = abs(float((components_left["angle"] - components_right["angle"]).item()))
    motif_delta = abs(float((components_left["motif"] - components_right["motif"]).item()))
    assert angular_delta + motif_delta > 1.0e-4


def test_ood_split_constraints_are_reflected_in_metadata(tmp_path):
    config = _small_config(length=4)
    train = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    type_combo = SynOMolTransferDataset(tmp_path, split="test_type_combo", config=config, mode="online")
    motif = SynOMolTransferDataset(tmp_path, split="test_motif", config=config, mode="online")
    size = SynOMolTransferDataset(tmp_path, split="test_size", config=config, mode="online")

    assert all(not bool(train[index]["has_heldout_type_combo"]) for index in range(len(train)))
    assert all(train[index]["primary_motif"] not in SYNOMOL_TRANSFER_OOD_MOTIFS for index in range(len(train)))
    assert all(bool(type_combo[index]["has_heldout_type_combo"]) for index in range(len(type_combo)))
    assert all(motif[index]["primary_motif"] in SYNOMOL_TRANSFER_OOD_MOTIFS for index in range(len(motif)))
    assert all(8 <= int(size[index]["num_atoms"]) <= 9 for index in range(len(size)))


def test_active_heldout_type_triples_are_enforced_globally(tmp_path):
    config = _small_config(length=6)
    train = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    type_combo = SynOMolTransferDataset(tmp_path, split="test_type_combo", config=config, mode="online")

    for index in range(len(train)):
        sample = train[index]
        triples = _active_type_triples(sample["atom_types"], sample["coords"], config)
        assert not any(
            (center, min(left, right), max(left, right)) in triples
            for center, left, right in ((2, 0, 1), (3, 4, 5), (10, 6, 7), (11, 8, 9), (12, 1, 3), (13, 5, 6))
        )
    for index in range(len(type_combo)):
        sample = type_combo[index]
        triples = _active_type_triples(sample["atom_types"], sample["coords"], config)
        assert any(
            (center, min(left, right), max(left, right)) in triples
            for center, left, right in ((2, 0, 1), (3, 4, 5), (10, 6, 7), (11, 8, 9), (12, 1, 3), (13, 5, 6))
        )


def test_materialize_helper_writes_loadable_split(tmp_path):
    config = _small_config(length=2)
    output_path = tmp_path / config.cache_name() / "train.pt"

    packed = materialize_synomol_transfer_split(output_path, config=config, split="train", size=2)

    assert output_path.is_file()
    assert packed["energy"].shape == (2,)
    fixed = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="fixed")
    assert len(fixed) == 2
    assert fixed[0]["coords"].shape[-1] == 3


def test_sample_summary_and_calibration_metadata_are_finite(tmp_path):
    config = _small_config(length=3)
    dataset = SynOMolTransferDataset(tmp_path, split="train", config=config, mode="online")
    samples = [dataset[index] for index in range(len(dataset))]
    summary = summarize_synomol_transfer_samples(samples)

    assert summary["accepted"] == 3
    assert summary["sample_kind_counts"]
    assert summary["force_rms"]["max"] >= summary["force_rms"]["min"]
    assert "component_energies_per_atom" in summary
    assert "heldout_repair_count" in summary

    summary_with_label_stats = summarize_synomol_transfer_samples(
        samples,
        generation_stats=[
            {
                "attempts": 1,
                "label_neighbor_stats": {
                    "mean_label_neighbors": 2.0,
                    "q95_label_neighbors": 3.0,
                    "max_label_neighbors": 4.0,
                    "neighbor_cap_hit_fraction": 0.0,
                },
            }
            for _ in samples
        ],
    )
    assert "label_neighbor_stats" in summary_with_label_stats
    assert summary_with_label_stats["label_neighbor_stats"]["neighbor_cap_hit_fraction"]["max"] == 0.0

    calibrated, metadata = calibrate_synomol_transfer_component_scales(config, calibration_size=4)
    assert isinstance(calibrated, SynOMolTransferConfig)
    assert all(value > 0.0 for value in metadata["component_scales"].values())
