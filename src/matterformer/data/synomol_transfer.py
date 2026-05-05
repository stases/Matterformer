from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from matterformer.data.qm9 import build_pad_mask


SYNOMOL_TRANSFER_NUM_ATOM_TYPES = 16
SYNOMOL_TRANSFER_CACHE_VERSION = "v1local"
SYNOMOL_TRANSFER_ATOM_PAD_TOKEN = SYNOMOL_TRANSFER_NUM_ATOM_TYPES
SYNOMOL_TRANSFER_SPLITS = (
    "train",
    "val",
    "test_iid",
    "test_type_combo",
    "test_motif",
    "test_size",
    "test_perturb",
)
SYNOMOL_TRANSFER_COMPONENT_NAMES = ("pair", "angle", "motif", "many_body")
SYNOMOL_TRANSFER_GEOMETRY_LEVELS: dict[str, tuple[str, ...]] = {
    "pair_only": ("pair",),
    "pair_angle": ("pair", "angle"),
    "pair_angle_motif": ("pair", "angle", "motif"),
    "full_local": SYNOMOL_TRANSFER_COMPONENT_NAMES,
}
SYNOMOL_TRANSFER_GEOMETRY_LEVEL_NAMES = tuple(SYNOMOL_TRANSFER_GEOMETRY_LEVELS)
SYNOMOL_TRANSFER_SAMPLE_KINDS = (
    "relaxed",
    "relaxation",
    "langevin",
    "bond_distortion",
    "angle_distortion",
    "shell_distortion",
    "collision",
)
SYNOMOL_TRANSFER_MOTIFS = (
    "linear",
    "bent",
    "trigonal",
    "tetrahedral",
    "square_planar",
    "octahedral",
    "chain",
    "ring",
)
SYNOMOL_TRANSFER_TRAIN_MOTIFS = ("linear", "bent", "trigonal", "tetrahedral", "chain", "ring")
SYNOMOL_TRANSFER_OOD_MOTIFS = ("square_planar", "octahedral")
SYNOMOL_TRANSFER_ALIAS_PAIRS = ((2, 10), (3, 11), (4, 12), (5, 13))
SYNOMOL_TRANSFER_HELDOUT_TYPE_COMBOS = (
    (2, 0, 1),
    (3, 4, 5),
    (10, 6, 7),
    (11, 8, 9),
    (12, 1, 3),
    (13, 5, 6),
)
SYNOMOL_TRANSFER_DATASET_INFO = {
    "name": "synomol_transfer",
    "num_atom_types": SYNOMOL_TRANSFER_NUM_ATOM_TYPES,
    "atom_pad_token": SYNOMOL_TRANSFER_ATOM_PAD_TOKEN,
    "splits": SYNOMOL_TRANSFER_SPLITS,
    "components": SYNOMOL_TRANSFER_COMPONENT_NAMES,
    "geometry_levels": SYNOMOL_TRANSFER_GEOMETRY_LEVEL_NAMES,
    "sample_kinds": SYNOMOL_TRANSFER_SAMPLE_KINDS,
    "motifs": SYNOMOL_TRANSFER_MOTIFS,
    "alias_pairs": SYNOMOL_TRANSFER_ALIAS_PAIRS,
    "heldout_type_combos": SYNOMOL_TRANSFER_HELDOUT_TYPE_COMBOS,
}

_SPLIT_OFFSETS = {
    "train": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "test": 2,
    "test_iid": 2,
    "test_type_combo": 3,
    "test_motif": 4,
    "test_size": 5,
    "test_perturb": 6,
}
_MOTIF_TO_ID = {name: idx for idx, name in enumerate(SYNOMOL_TRANSFER_MOTIFS)}
_ID_TO_MOTIF = {idx: name for name, idx in _MOTIF_TO_ID.items()}
_KIND_TO_ID = {name: idx for idx, name in enumerate(SYNOMOL_TRANSFER_SAMPLE_KINDS)}
_HELDOUT_COMBO_SET = {
    (center, min(left, right), max(left, right))
    for center, left, right in SYNOMOL_TRANSFER_HELDOUT_TYPE_COMBOS
}
_DEFAULT_FORCE_BUDGET = {
    "pair": 0.25,
    "angle": 0.30,
    "motif": 0.25,
    "many_body": 0.20,
}
_DEFAULT_KIND_WEIGHTS = (
    ("relaxed", 0.35),
    ("relaxation", 0.15),
    ("langevin", 0.25),
    ("bond_distortion", 0.08),
    ("angle_distortion", 0.08),
    ("shell_distortion", 0.04),
    ("collision", 0.05),
)
_TRAIN_SHELL_DISTORTION_MOTIFS = ("linear", "bent", "trigonal", "tetrahedral")


@dataclass(frozen=True)
class SynOMolTransferConfig:
    num_atoms: int | tuple[int, int] = (16, 96)
    size_ood_num_atoms: int | tuple[int, int] = (128, 256)
    length: int = 10_000
    seed: int = 0
    density: float = 0.18
    d_min: float = 0.45
    max_resample_attempts: int = 32
    cutoff: float = 4.2
    cutoff_delta: float = 0.7
    radial_basis_size: int = 8
    radial_min: float = 0.7
    radial_max: float = 4.0
    radial_width: float = 0.45
    angle_lmax: int = 4
    motif_sigma: float = 0.18
    distance_eps: float = 1.0e-6
    relax_steps: int = 8
    relaxation_snapshot_steps: int = 3
    relax_step_size: float = 0.025
    max_relax_displacement: float = 0.18
    langevin_steps: int = 5
    langevin_step_size: float = 0.012
    langevin_temperature: float = 0.025
    distortion_scale: float = 0.28
    pair_scale: float = 0.04
    angle_scale: float = 0.006
    motif_scale: float = 0.08
    many_body_scale: float = 0.10
    generation_pair_scale: float = 0.08
    generation_angle_scale: float = 0.012
    generation_motif_scale: float = 0.08
    generation_many_body_scale: float = 0.0
    force_rms_min: float = 1.0e-5
    force_rms_max: float = 80.0
    max_abs_energy: float = 1.0e5
    max_abs_energy_per_atom: float = 1.0e3

    def __post_init__(self) -> None:
        _validate_num_atoms(self.num_atoms, name="num_atoms")
        _validate_num_atoms(self.size_ood_num_atoms, name="size_ood_num_atoms")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.d_min < 0.0:
            raise ValueError("d_min must be non-negative")
        if self.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")
        if self.cutoff <= 0.0 or self.cutoff_delta <= 0.0 or self.cutoff_delta >= self.cutoff:
            raise ValueError("cutoff must be positive and cutoff_delta must be in (0, cutoff)")
        if self.radial_basis_size <= 0 or self.radial_basis_size > 8:
            raise ValueError("radial_basis_size must be in [1, 8]")
        if self.radial_min < 0.0 or self.radial_max <= self.radial_min:
            raise ValueError("radial basis bounds must satisfy 0 <= radial_min < radial_max")
        if self.radial_width <= 0.0 or self.motif_sigma <= 0.0 or self.distance_eps <= 0.0:
            raise ValueError("radial_width, motif_sigma, and distance_eps must be positive")
        if self.angle_lmax < 0 or self.angle_lmax > 6:
            raise ValueError("angle_lmax must be in [0, 6]")
        if self.relax_steps < 0 or self.relaxation_snapshot_steps < 0 or self.langevin_steps < 0:
            raise ValueError("relaxation and Langevin step counts must be non-negative")
        if self.relax_step_size <= 0.0 or self.langevin_step_size <= 0.0:
            raise ValueError("relax_step_size and langevin_step_size must be positive")
        if self.max_relax_displacement <= 0.0 or self.langevin_temperature < 0.0 or self.distortion_scale < 0.0:
            raise ValueError("relax displacement, Langevin temperature, and distortion scale must be valid")
        for name in (
            "pair_scale",
            "angle_scale",
            "motif_scale",
            "many_body_scale",
            "generation_pair_scale",
            "generation_angle_scale",
            "generation_motif_scale",
            "generation_many_body_scale",
        ):
            if not torch.isfinite(torch.tensor(float(getattr(self, name)))):
                raise ValueError(f"{name} must be finite")
        if self.force_rms_min < 0.0 or self.force_rms_max <= self.force_rms_min:
            raise ValueError("force_rms_min and force_rms_max must satisfy 0 <= min < max")
        if self.max_abs_energy <= 0.0 or self.max_abs_energy_per_atom <= 0.0:
            raise ValueError("max_abs_energy and max_abs_energy_per_atom must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SynOMolTransferConfig":
        values = dict(values)
        for key in ("num_atoms", "size_ood_num_atoms"):
            if isinstance(values.get(key), list):
                values[key] = tuple(int(item) for item in values[key])
        return cls(**values)

    def cache_name(self) -> str:
        return f"synomol_{SYNOMOL_TRANSFER_CACHE_VERSION}_{_config_digest(self)}"


@dataclass
class SynOMolTransferBatch:
    atom_types: torch.Tensor
    coords: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    component_energies: dict[str, torch.Tensor]
    pad_mask: torch.Tensor
    num_atoms: torch.Tensor
    indices: torch.Tensor
    system_ids: torch.Tensor
    motif_labels: torch.Tensor
    sample_kinds: list[str]
    primary_motifs: list[str]
    primary_triples: torch.Tensor
    has_heldout_type_combo: torch.Tensor
    split_labels: list[str] | None = None
    lattice: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "SynOMolTransferBatch":
        return SynOMolTransferBatch(
            atom_types=self.atom_types.to(device),
            coords=self.coords.to(device),
            forces=self.forces.to(device),
            energy=self.energy.to(device),
            component_energies={key: value.to(device) for key, value in self.component_energies.items()},
            pad_mask=self.pad_mask.to(device),
            num_atoms=self.num_atoms.to(device),
            indices=self.indices.to(device),
            system_ids=self.system_ids.to(device),
            motif_labels=self.motif_labels.to(device),
            sample_kinds=self.sample_kinds,
            primary_motifs=self.primary_motifs,
            primary_triples=self.primary_triples.to(device),
            has_heldout_type_combo=self.has_heldout_type_combo.to(device),
            split_labels=self.split_labels,
            lattice=None if self.lattice is None else self.lattice.to(device),
        )

    def atom_onehot(self, num_classes: int = SYNOMOL_TRANSFER_NUM_ATOM_TYPES) -> torch.Tensor:
        atom_types = self.atom_types.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(atom_types, num_classes=num_classes).float()
        return one_hot.masked_fill(self.pad_mask[..., None], 0.0)

    def node_features(self) -> torch.Tensor:
        return self.atom_onehot()


def collate_synomol_transfer(samples: list[dict[str, object]]) -> SynOMolTransferBatch:
    if not samples:
        raise ValueError("samples must not be empty")

    num_atoms = torch.tensor([int(sample["num_atoms"]) for sample in samples], dtype=torch.long)
    max_atoms = int(num_atoms.max().item())
    batch_size = len(samples)

    atom_types = torch.full((batch_size, max_atoms), SYNOMOL_TRANSFER_ATOM_PAD_TOKEN, dtype=torch.long)
    coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    forces = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    motif_labels = torch.full((batch_size, max_atoms), -1, dtype=torch.long)
    energy = torch.zeros(batch_size, dtype=torch.float32)
    component_energies = {
        name: torch.zeros(batch_size, dtype=torch.float32)
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
    }
    indices = torch.zeros(batch_size, dtype=torch.long)
    system_ids = torch.zeros(batch_size, dtype=torch.long)
    primary_triples = torch.zeros(batch_size, 3, dtype=torch.long)
    heldout = torch.zeros(batch_size, dtype=torch.bool)
    sample_kinds: list[str] = []
    primary_motifs: list[str] = []
    split_labels: list[str] = []

    for batch_idx, sample in enumerate(samples):
        count = int(sample["num_atoms"])
        atom_types[batch_idx, :count] = torch.as_tensor(sample["atom_types"], dtype=torch.long)
        coords[batch_idx, :count] = torch.as_tensor(sample["coords"], dtype=torch.float32)
        forces[batch_idx, :count] = torch.as_tensor(sample["forces"], dtype=torch.float32)
        motif_labels[batch_idx, :count] = torch.as_tensor(sample["motif_labels"], dtype=torch.long)
        energy[batch_idx] = torch.as_tensor(sample["energy"], dtype=torch.float32).reshape(())
        sample_components = sample.get("component_energies", {})
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
            if isinstance(sample_components, dict) and name in sample_components:
                component_energies[name][batch_idx] = torch.as_tensor(
                    sample_components[name],
                    dtype=torch.float32,
                ).reshape(())
        indices[batch_idx] = int(sample["idx"])
        system_ids[batch_idx] = int(sample["system_id"])
        primary_triples[batch_idx] = torch.as_tensor(sample["primary_triple"], dtype=torch.long)
        heldout[batch_idx] = bool(sample["has_heldout_type_combo"])
        sample_kinds.append(str(sample["sample_kind"]))
        primary_motifs.append(str(sample["primary_motif"]))
        split_labels.append(str(sample.get("split", "")))

    pad_mask = build_pad_mask(num_atoms, max_atoms=max_atoms)
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    forces = forces.masked_fill(pad_mask[..., None], 0.0)
    motif_labels = motif_labels.masked_fill(pad_mask, -1)
    return SynOMolTransferBatch(
        atom_types=atom_types,
        coords=coords,
        forces=forces,
        energy=energy,
        component_energies=component_energies,
        pad_mask=pad_mask,
        num_atoms=num_atoms,
        indices=indices,
        system_ids=system_ids,
        motif_labels=motif_labels,
        sample_kinds=sample_kinds,
        primary_motifs=primary_motifs,
        primary_triples=primary_triples,
        has_heldout_type_combo=heldout,
        split_labels=split_labels,
    )


class SynOMolTransferDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "./data/synomol_transfer",
        *,
        split: str = "train",
        config: SynOMolTransferConfig | None = None,
        mode: str = "online",
        config_name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = _normalize_split(split)
        self.mode = mode.lower()
        self.epoch = 0
        if self.mode not in {"online", "fixed"}:
            raise ValueError("mode must be one of {'online', 'fixed'}")

        self.config = config or SynOMolTransferConfig()
        self.config_name = _resolve_fixed_config_name(self.root, config_name or self.config.cache_name())
        self._fixed: dict[str, Any] | None = None
        if self.mode == "fixed":
            loaded = _load_fixed_split(self.root / self.config_name, self.split)
            if "config" in loaded:
                self.config = SynOMolTransferConfig.from_dict(loaded["config"])
            self._fixed = _normalize_fixed_split(loaded)

    @property
    def dataset_info(self) -> dict[str, object]:
        return SYNOMOL_TRANSFER_DATASET_INFO

    @property
    def num_atoms(self) -> torch.Tensor:
        if self._fixed is not None:
            return self._fixed["num_atoms"].clone()
        values = [
            _sample_num_atoms_for_split(self.config, self.split, _make_generator(_sample_seed(self.config.seed, self.split, self.epoch, index)))
            for index in range(len(self))
        ]
        return torch.tensor(values, dtype=torch.long)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self._fixed is not None:
            return int(self._fixed["energy"].shape[0])
        return int(self.config.length)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._fixed is not None:
            return self._get_fixed(index)
        return generate_synomol_transfer_sample(
            index=index,
            split=self.split,
            epoch=self.epoch,
            config=self.config,
        )

    def _get_fixed(self, index: int) -> dict[str, object]:
        if self._fixed is None:
            raise RuntimeError("fixed data is not loaded")
        ptr = self._fixed["ptr"]
        start = int(ptr[index].item())
        end = int(ptr[index + 1].item())
        components = {
            name: self._fixed["component_energies"][name][index].clone()
            for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
        }
        return {
            "atom_types": self._fixed["atom_types"][start:end].clone(),
            "coords": self._fixed["coords"][start:end].clone(),
            "forces": self._fixed["forces"][start:end].clone(),
            "energy": self._fixed["energy"][index].clone(),
            "component_energies": components,
            "num_atoms": int(self._fixed["num_atoms"][index].item()),
            "idx": int(self._fixed["indices"][index].item()),
            "system_id": int(self._fixed["system_ids"][index].item()),
            "sample_kind": str(self._fixed["sample_kinds"][index]),
            "primary_motif": str(self._fixed["primary_motifs"][index]),
            "primary_triple": self._fixed["primary_triples"][index].clone(),
            "original_primary_triple": self._fixed["original_primary_triples"][index].clone(),
            "has_heldout_type_combo": bool(self._fixed["has_heldout_type_combo"][index].item()),
            "heldout_repair_count": int(self._fixed["heldout_repair_counts"][index].item()),
            "proposal_seed": int(self._fixed["proposal_seeds"][index].item()),
            "accepted_attempts": int(self._fixed["accepted_attempts"][index].item()),
            "global_sample_id": int(self._fixed["global_sample_ids"][index].item()),
            "shard_id": int(self._fixed["shard_ids"][index].item()),
            "motif_labels": self._fixed["motif_labels"][start:end].clone(),
            "split": str(self._fixed["splits"][index]),
            **(
                {
                    "component_forces": {
                        name: self._fixed["component_forces"][name][start:end].clone()
                        for name in self._fixed["component_forces"]
                    },
                    "level_energies": {
                        name: self._fixed["level_energies"][name][index].clone()
                        for name in self._fixed["level_energies"]
                    },
                    "level_forces": {
                        name: self._fixed["level_forces"][name][start:end].clone()
                        for name in self._fixed["level_forces"]
                    },
                }
                if "component_forces" in self._fixed
                else {}
            ),
        }


def generate_synomol_transfer_sample(
    *,
    index: int,
    split: str,
    epoch: int,
    config: SynOMolTransferConfig,
) -> dict[str, object]:
    sample, _ = generate_synomol_transfer_sample_with_stats(
        index=index,
        split=split,
        epoch=epoch,
        config=config,
    )
    return sample


def generate_synomol_transfer_sample_with_stats(
    *,
    index: int,
    split: str,
    epoch: int,
    config: SynOMolTransferConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    split = _normalize_split(split)
    last_error: str | None = None
    rejection_reasons = {
        "heldout_type_combo": 0,
        "sample_filter": 0,
    }
    for attempt in range(config.max_resample_attempts):
        atom_types, coords, metadata = _generate_synomol_transfer_inputs(
            index=index,
            split=split,
            epoch=epoch,
            config=config,
            attempt=attempt,
        )
        active_triples = _active_type_triples(atom_types, coords, config)
        has_heldout = bool(active_triples & _HELDOUT_COMBO_SET)
        if not _split_allows_active_heldout(split, has_heldout):
            last_error = "active held-out type combo constraint was not satisfied"
            rejection_reasons["heldout_type_combo"] += 1
            continue
        energy, forces, components = compute_synomol_transfer_labels(atom_types, coords, config)
        if not _sample_is_acceptable(atom_types, coords, energy, forces, components, config):
            last_error = "sample failed finite/min-distance/force filters"
            rejection_reasons["sample_filter"] += 1
            continue
        metadata = dict(metadata)
        metadata["has_heldout_type_combo"] = has_heldout
        break
    else:
        raise RuntimeError(
            f"failed to generate an acceptable SynOMol-Transfer sample for split={split!r}, "
            f"index={index}, epoch={epoch}: {last_error or 'unknown rejection'}"
        )
    sample = {
        "atom_types": atom_types,
        "coords": coords,
        "forces": forces,
        "energy": energy,
        "component_energies": components,
        "num_atoms": int(atom_types.shape[0]),
        "idx": int(index),
        "system_id": int(metadata["system_id"]),
        "sample_kind": str(metadata["sample_kind"]),
        "primary_motif": str(metadata["primary_motif"]),
        "primary_triple": torch.as_tensor(metadata["primary_triple"], dtype=torch.long),
        "original_primary_triple": torch.as_tensor(metadata["original_primary_triple"], dtype=torch.long),
        "has_heldout_type_combo": bool(metadata["has_heldout_type_combo"]),
        "heldout_repair_count": int(metadata["heldout_repair_count"]),
        "proposal_seed": int(metadata["proposal_seed"]),
        "accepted_attempts": int(attempt + 1),
        "global_sample_id": int(index),
        "shard_id": -1,
        "motif_labels": torch.as_tensor(metadata["motif_labels"], dtype=torch.long),
        "split": split,
    }
    stats = {
        "attempts": int(attempt + 1),
        "rejection_reasons": rejection_reasons,
        "accepted_sample_kind": str(metadata["sample_kind"]),
        "accepted_primary_motif": str(metadata["primary_motif"]),
        "heldout_repair_count": int(metadata["heldout_repair_count"]),
    }
    return sample, stats


def compute_synomol_transfer_labels(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    coords_for_grad = coords.detach().clone().requires_grad_(True)
    energy, components = compute_synomol_transfer_energy(atom_types, coords_for_grad, config)
    (grad_coords,) = torch.autograd.grad(energy, coords_for_grad, create_graph=False)
    forces = -grad_coords
    return (
        energy.detach(),
        forces.detach(),
        {name: value.detach() for name, value in components.items()},
    )


def compute_synomol_transfer_labels_batch_dense(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    coords_for_grad = coords.detach().clone().requires_grad_(True)
    energy, components = compute_synomol_transfer_energy_batch_dense(atom_types, coords_for_grad, mask, config)
    (grad_coords,) = torch.autograd.grad(energy.sum(), coords_for_grad, create_graph=False)
    forces = -grad_coords
    forces = forces.masked_fill(~mask.to(device=forces.device, dtype=torch.bool)[..., None], 0.0)
    return (
        energy.detach(),
        forces.detach(),
        {name: value.detach() for name, value in components.items()},
    )


def compute_synomol_transfer_labels_batch_local(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    k_label: int = 64,
    return_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, float],
]:
    coords_for_grad = coords.detach().clone().requires_grad_(True)
    output = compute_synomol_transfer_energy_batch_local(
        atom_types,
        coords_for_grad,
        mask,
        config,
        k_label=k_label,
        return_stats=return_stats,
    )
    if return_stats:
        energy, components, stats = output
    else:
        energy, components = output
        stats = {}
    (grad_coords,) = torch.autograd.grad(energy.sum(), coords_for_grad, create_graph=False)
    forces = -grad_coords
    forces = forces.masked_fill(~mask.to(device=forces.device, dtype=torch.bool)[..., None], 0.0)
    result = (
        energy.detach(),
        forces.detach(),
        {name: value.detach() for name, value in components.items()},
    )
    if return_stats:
        return (*result, stats)
    return result


def compute_synomol_transfer_multilevel_labels_batch_local(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    levels: tuple[str, ...] | list[str] | None = None,
    k_label: int = 64,
    return_stats: bool = False,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
] | tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, float],
]:
    """Compute component and cumulative geometry-level labels in one graph.

    Returns ``level_energies``, ``level_forces``, ``component_energies``, and
    ``component_forces``. The default levels are cumulative local-geometry
    targets sharing the same input coordinates.
    """
    requested_levels = tuple(levels or SYNOMOL_TRANSFER_GEOMETRY_LEVEL_NAMES)
    unknown = sorted(set(requested_levels) - set(SYNOMOL_TRANSFER_GEOMETRY_LEVELS))
    if unknown:
        raise ValueError(f"unknown SynOMol-Transfer geometry levels: {unknown}")
    coords_for_grad = coords.detach().clone().requires_grad_(True)
    output = compute_synomol_transfer_energy_batch_local(
        atom_types,
        coords_for_grad,
        mask,
        config,
        k_label=k_label,
        return_stats=return_stats,
    )
    if return_stats:
        _, components, stats = output
    else:
        _, components = output
        stats = {}
    mask = mask.to(device=coords_for_grad.device, dtype=torch.bool)
    component_forces: dict[str, torch.Tensor] = {}
    differentiable_names = [name for name in SYNOMOL_TRANSFER_COMPONENT_NAMES if components[name].requires_grad]
    for grad_index, name in enumerate(differentiable_names):
        (grad_coords,) = torch.autograd.grad(
            components[name].sum(),
            coords_for_grad,
            retain_graph=grad_index < len(differentiable_names) - 1,
            create_graph=False,
            allow_unused=True,
        )
        if grad_coords is None:
            grad_coords = torch.zeros_like(coords_for_grad)
        component_forces[name] = -grad_coords.masked_fill(~mask[..., None], 0.0)
    for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
        if name not in component_forces:
            component_forces[name] = torch.zeros_like(coords_for_grad)

    level_energies: dict[str, torch.Tensor] = {}
    level_forces: dict[str, torch.Tensor] = {}
    for level in requested_levels:
        component_names = SYNOMOL_TRANSFER_GEOMETRY_LEVELS[level]
        level_energies[level] = sum(components[name] for name in component_names)
        level_forces[level] = sum(component_forces[name] for name in component_names)

    result = (
        {name: value.detach() for name, value in level_energies.items()},
        {name: value.detach() for name, value in level_forces.items()},
        {name: value.detach() for name, value in components.items()},
        {name: value.detach() for name, value in component_forces.items()},
    )
    if return_stats:
        return (*result, stats)
    return result


def compute_synomol_transfer_energy_batch_dense(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_batched_inputs(atom_types, coords, mask)
    mask = mask.to(device=coords.device, dtype=torch.bool)
    energies: list[torch.Tensor] = []
    component_values = {name: [] for name in SYNOMOL_TRANSFER_COMPONENT_NAMES}
    for batch_idx in range(coords.shape[0]):
        valid = mask[batch_idx]
        if int(valid.sum().item()) == 0:
            raise ValueError("each batched SynOMol-Transfer sample must contain at least one valid atom")
        sample_energy, sample_components = compute_synomol_transfer_energy(
            atom_types[batch_idx, valid],
            coords[batch_idx, valid],
            config,
        )
        energies.append(sample_energy)
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES:
            component_values[name].append(sample_components[name])
    return (
        torch.stack(energies),
        {name: torch.stack(values) for name, values in component_values.items()},
    )


def compute_synomol_transfer_energy_batch_local(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    k_label: int = 64,
    return_stats: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    _validate_batched_inputs(atom_types, coords, mask)
    if k_label <= 0:
        raise ValueError("k_label must be positive")
    mask = mask.to(device=coords.device, dtype=torch.bool)
    if (mask.sum(dim=1) == 0).any().item():
        raise ValueError("each batched SynOMol-Transfer sample must contain at least one valid atom")
    atom_types = atom_types.to(device=coords.device, dtype=torch.long)
    valid_atom_types = atom_types[mask]
    if int(valid_atom_types.min().item()) < 0 or int(valid_atom_types.max().item()) >= SYNOMOL_TRANSFER_NUM_ATOM_TYPES:
        raise ValueError(f"valid atom_types must be in [0, {SYNOMOL_TRANSFER_NUM_ATOM_TYPES})")
    safe_atom_types = atom_types.clamp(min=0, max=SYNOMOL_TRANSFER_NUM_ATOM_TYPES - 1)
    dtype = coords.dtype
    device = coords.device
    batch_size, max_atoms, _ = coords.shape
    params = synomol_transfer_atom_type_table(device=device, dtype=dtype)

    deltas = coords[:, :, None, :] - coords[:, None, :, :]
    distances = torch.sqrt((deltas * deltas).sum(dim=-1) + config.distance_eps**2)
    eye = torch.eye(max_atoms, device=device, dtype=torch.bool)[None, :, :]
    valid_pair = mask[:, :, None] & mask[:, None, :] & ~eye
    upper_mask = torch.triu(torch.ones(max_atoms, max_atoms, device=device, dtype=torch.bool), diagonal=1)[None, :, :]
    pair_weight = _smooth_cutoff(distances, config.cutoff, config.cutoff_delta)

    zi = safe_atom_types[:, :, None]
    zj = safe_atom_types[:, None, :]
    size = params["size"]
    strength = params["pair_strength"]
    r0 = size[zi] + size[zj] + 0.08 * torch.abs(size[zi] - size[zj])
    eps = torch.sqrt(strength[zi] * strength[zj])
    pair_terms = (
        0.08 * eps * torch.exp(-2.4 * (distances - 0.65 * r0))
        - 0.10 * eps * torch.exp(-((distances - r0) ** 2) / 0.18)
        + 0.012 * eps / torch.sqrt(distances * distances + 0.25)
    )
    pair_energy = config.pair_scale * (pair_terms * pair_weight * valid_pair.to(dtype) * upper_mask.to(dtype)).sum(dim=(1, 2))

    active_pair = valid_pair & (pair_weight > 1.0e-8)
    active_counts = active_pair.sum(dim=-1)
    has_two_neighbors = (active_counts >= 2) & mask
    has_two_f = has_two_neighbors.to(dtype)
    k = min(int(k_label), max(max_atoms - 1, 1))
    large = torch.full_like(distances, 1.0e6)
    neighbor_distances, neighbor_idx = torch.topk(
        torch.where(active_pair, distances, large),
        k=k,
        dim=-1,
        largest=False,
    )
    neighbor_valid = neighbor_distances < 1.0e5
    neighbor_idx = neighbor_idx.clamp(min=0, max=max_atoms - 1)

    neighbor_coords = _batch_gather_nodes(coords, neighbor_idx)
    neighbor_types = _batch_gather_nodes(safe_atom_types, neighbor_idx)
    center_coords = coords[:, :, None, :]
    rel = neighbor_coords - center_coords
    neighbor_distances = torch.sqrt((rel * rel).sum(dim=-1) + config.distance_eps**2)
    directions = rel / neighbor_distances.clamp_min(config.distance_eps)[..., None]
    weights = _smooth_cutoff(neighbor_distances, config.cutoff, config.cutoff_delta)
    weights = weights * neighbor_valid.to(dtype) * mask[:, :, None].to(dtype)
    cos = torch.einsum("bnkd,bnld->bnkl", directions, directions).clamp(min=-1.0, max=1.0)

    jk_upper = torch.triu(torch.ones(k, k, device=device, dtype=torch.bool), diagonal=1)
    jk_mask = (
        jk_upper[None, None, :, :]
        & neighbor_valid[:, :, :, None]
        & neighbor_valid[:, :, None, :]
        & mask[:, :, None, None]
    )
    jk_mask_f = jk_mask.to(dtype)
    pair_weights = jk_mask_f * weights[:, :, :, None] * weights[:, :, None, :]
    pair_weight_sum = pair_weights.sum(dim=(2, 3)).clamp_min(1.0)

    radial_centers = torch.linspace(
        config.radial_min,
        config.radial_max,
        config.radial_basis_size,
        device=device,
        dtype=dtype,
    )
    rbf = torch.exp(-((neighbor_distances[..., None] - radial_centers) ** 2) / (config.radial_width**2))
    rbf = rbf * weights[..., None]
    legendre = _legendre_stack(cos, config.angle_lmax)

    center_types = safe_atom_types
    left_coeff = params["angle_left"][neighbor_types, : config.angle_lmax + 1, : config.radial_basis_size]
    right_coeff = params["angle_right"][neighbor_types, : config.angle_lmax + 1, : config.radial_basis_size]
    left = torch.einsum("bnkp,bnklp->bnkl", rbf, left_coeff)
    right = torch.einsum("bnkp,bnklp->bnkl", rbf, right_coeff)
    center_angle = params["angle_center"][center_types, : config.angle_lmax + 1]
    angle_bilinear = coords.new_zeros(batch_size)
    for ell in range(config.angle_lmax + 1):
        angle_matrix = center_angle[:, :, ell, None, None] * (
            left[:, :, :, ell, None] * right[:, :, None, :, ell]
            + right[:, :, :, ell, None] * left[:, :, None, :, ell]
        ) * legendre[ell]
        angle_bilinear = angle_bilinear + (angle_matrix * jk_mask_f).sum(dim=(1, 2, 3))
    target_cos = params["preferred_cos"][center_types]
    angle_preference = (((cos - target_cos[:, :, None, None]) ** 2) * pair_weights).sum(dim=(2, 3)) / pair_weight_sum
    angle_pref = (params["angle_pref_strength"][center_types] * angle_preference * has_two_f).sum(dim=1)
    angle_energy_raw = angle_bilinear + angle_pref

    motif_cosines = torch.tensor([-1.0, -0.5, -1.0 / 3.0, 0.0, 0.5], device=device, dtype=dtype)
    target_logits = -((motif_cosines[None, None, :] - target_cos[:, :, None]) ** 2) / (
        2.0 * (1.6 * config.motif_sigma) ** 2
    )
    target_logits = target_logits + 0.75 * torch.nn.functional.one_hot(
        params["motif_well_index"][center_types],
        num_classes=motif_cosines.shape[0],
    ).to(dtype=dtype, device=device)
    motif_probs = torch.softmax(target_logits, dim=-1)
    motif_wells = torch.exp(-((cos[..., None] - motif_cosines) ** 2) / (2.0 * config.motif_sigma**2))
    motif_match = (motif_probs[:, :, None, None, :] * motif_wells).sum(dim=-1)
    motif_neighbor = params["motif_neighbor"][neighbor_types]
    neighbor_factor = 1.0 + 0.04 * (motif_neighbor[:, :, :, None] + motif_neighbor[:, :, None, :])
    motif_penalty = (1.0 - motif_match).clamp_min(0.0) * neighbor_factor
    motif_energy_raw = (
        params["motif_strength"][center_types]
        * (motif_penalty * pair_weights).sum(dim=(2, 3))
        / pair_weight_sum
        * has_two_f
    ).sum(dim=1)

    coord_norm = weights.sum(dim=2).clamp_min(1.0)
    mb_pair = params["mb_pair"][center_types[:, :, None], neighbor_types, : config.radial_basis_size]
    radial_density = (rbf * mb_pair).sum(dim=2) / coord_norm[:, :, None]
    mb_neighbor = params["mb_neighbor"][neighbor_types, : config.angle_lmax + 1]
    angular_density_values: list[torch.Tensor] = []
    coord_norm_sq = (coord_norm * coord_norm).clamp_min(1.0)
    for ell in range(config.angle_lmax + 1):
        angular_density_values.append(
            (
                jk_mask_f
                * legendre[ell]
                * (mb_neighbor[:, :, :, ell, None] + mb_neighbor[:, :, None, :, ell])
                * weights[:, :, :, None]
                * weights[:, :, None, :]
            ).sum(dim=(2, 3))
            / coord_norm_sq
        )
    angular_density = torch.stack(angular_density_values, dim=-1)
    mb_u = params["mb_u"][center_types, : config.radial_basis_size]
    mb_v = params["mb_v"][center_types, : config.angle_lmax + 1]
    mb_cross = params["mb_cross"][center_types, : config.radial_basis_size, : config.angle_lmax + 1]
    many_body_per_center = (
        0.04 * torch.sum(mb_u * radial_density * radial_density, dim=-1)
        + 0.015 * torch.sum(mb_v * angular_density * angular_density, dim=-1)
        + 0.006 * torch.sum(mb_cross * radial_density[:, :, :, None] * angular_density[:, :, None, :], dim=(2, 3))
    )
    many_body_energy_raw = (many_body_per_center * has_two_f).sum(dim=1)

    components = {
        "pair": pair_energy,
        "angle": config.angle_scale * angle_energy_raw,
        "motif": config.motif_scale * motif_energy_raw,
        "many_body": config.many_body_scale * many_body_energy_raw,
    }
    total = sum(components.values())
    if not return_stats:
        return total, components
    active_counts_f = active_counts[mask].to(torch.float32)
    if active_counts_f.numel() == 0:
        stats = {
            "mean_label_neighbors": 0.0,
            "q95_label_neighbors": 0.0,
            "max_label_neighbors": 0.0,
            "neighbor_cap_hit_fraction": 0.0,
        }
    else:
        cap_hits = (active_counts[mask] > k).to(torch.float32)
        stats = {
            "mean_label_neighbors": float(active_counts_f.mean().item()),
            "q95_label_neighbors": float(torch.quantile(active_counts_f, 0.95).item()),
            "max_label_neighbors": float(active_counts_f.max().item()),
            "neighbor_cap_hit_fraction": float(cap_hits.mean().item()),
        }
    return total, components, stats


def compute_generation_energy_batch_local(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    k_label: int = 64,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    raw_config = replace(
        config,
        pair_scale=config.generation_pair_scale,
        angle_scale=config.generation_angle_scale,
        motif_scale=config.generation_motif_scale,
        many_body_scale=config.generation_many_body_scale,
    )
    output = compute_synomol_transfer_energy_batch_local(
        atom_types,
        coords,
        mask,
        raw_config,
        k_label=k_label,
        return_stats=return_stats,
    )
    if return_stats:
        energy, components, stats = output
    else:
        energy, components = output
        stats = {}
    mask = mask.to(device=coords.device, dtype=torch.bool)
    atom_types = atom_types.to(device=coords.device, dtype=torch.long)
    safe_atom_types = atom_types.clamp(min=0, max=SYNOMOL_TRANSFER_NUM_ATOM_TYPES - 1)
    table = synomol_transfer_atom_type_table(device=coords.device, dtype=coords.dtype)
    soft_bonds = _soft_bond_weights_batch(safe_atom_types, coords, mask, table, config)
    coordination = soft_bonds.sum(dim=2)
    preferred = table["preferred_coordination"][safe_atom_types]
    coord_error = ((coordination - preferred) ** 2) * mask.to(coords.dtype)
    denom = mask.sum(dim=1).clamp_min(1).to(coords.dtype)
    coord_energy = 0.02 * coord_error.sum(dim=1) / denom
    generation_energy = energy + coord_energy + 0.3 * components["motif"]
    if return_stats:
        return generation_energy, stats
    return generation_energy


def center_coords_batch(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=coords.device, dtype=torch.bool)
    mask_f = mask.to(dtype=coords.dtype)[..., None]
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (coords * mask_f).sum(dim=1, keepdim=True) / denom
    centered = coords - mean
    return torch.where(mask[..., None], centered, torch.zeros_like(centered))


def relax_coords_batch(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    steps: int,
    k_label: int = 64,
) -> torch.Tensor:
    x = center_coords_batch(coords.detach().clone(), mask)
    mask = mask.to(device=x.device, dtype=torch.bool)
    for _ in range(int(steps)):
        x = x.detach().requires_grad_(True)
        energy = compute_generation_energy_batch_local(atom_types, x, mask, config, k_label=k_label)
        (grad_x,) = torch.autograd.grad(energy.sum(), x, create_graph=False)
        step = (-config.relax_step_size * grad_x).clamp(
            min=-config.max_relax_displacement,
            max=config.max_relax_displacement,
        )
        x = torch.where(mask[..., None], x + step, torch.zeros_like(x)).detach()
        x = center_coords_batch(x, mask)
    return x.detach()


def langevin_coords_batch(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    steps: int | None = None,
    k_label: int = 64,
    generator: torch.Generator | None = None,
    sample_seeds: torch.Tensor | None = None,
) -> torch.Tensor:
    x = center_coords_batch(coords.detach().clone(), mask)
    mask = mask.to(device=x.device, dtype=torch.bool)
    noise_scale = (2.0 * config.langevin_step_size * config.langevin_temperature) ** 0.5
    if sample_seeds is not None:
        sample_seeds = sample_seeds.to(device=x.device, dtype=torch.long)
        if sample_seeds.ndim != 1 or sample_seeds.shape[0] != x.shape[0]:
            raise ValueError("sample_seeds must have shape (B,)")
    for step_idx in range(int(config.langevin_steps if steps is None else steps)):
        x = x.detach().requires_grad_(True)
        energy, _ = compute_synomol_transfer_energy_batch_local(atom_types, x, mask, config, k_label=k_label)
        (grad_x,) = torch.autograd.grad(energy.sum(), x, create_graph=False)
        if sample_seeds is None:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        else:
            noise = torch.empty_like(x)
            for batch_idx in range(x.shape[0]):
                sample_generator = torch.Generator(device=x.device).manual_seed(
                    int(sample_seeds[batch_idx].item()) + 10_000_019 * int(step_idx)
                )
                noise[batch_idx] = torch.randn(
                    x[batch_idx].shape,
                    device=x.device,
                    dtype=x.dtype,
                    generator=sample_generator,
                )
        noise = noise_scale * noise
        noise = noise.masked_fill(~mask[..., None], 0.0)
        x = torch.where(mask[..., None], x - config.langevin_step_size * grad_x + noise, torch.zeros_like(x)).detach()
        x = center_coords_batch(x, mask)
    return x.detach()


def compute_synomol_transfer_energy(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if atom_types.ndim != 1:
        raise ValueError(f"atom_types must have shape (N,), got {tuple(atom_types.shape)}")
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {tuple(coords.shape)}")
    if coords.shape[0] != atom_types.shape[0]:
        raise ValueError("atom_types and coords must describe the same number of atoms")
    if atom_types.numel() == 0:
        raise ValueError("SynOMol-Transfer samples must contain at least one atom")
    if int(atom_types.min().item()) < 0 or int(atom_types.max().item()) >= SYNOMOL_TRANSFER_NUM_ATOM_TYPES:
        raise ValueError(f"atom_types must be in [0, {SYNOMOL_TRANSFER_NUM_ATOM_TYPES})")

    atom_types = atom_types.to(device=coords.device, dtype=torch.long)
    dtype = coords.dtype
    device = coords.device
    num_atoms = int(coords.shape[0])
    params = synomol_transfer_atom_type_table(device=device, dtype=dtype)

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = torch.sqrt((deltas * deltas).sum(dim=-1) + config.distance_eps**2)
    zi = atom_types[:, None]
    zj = atom_types[None, :]
    upper_mask = torch.triu(torch.ones(num_atoms, num_atoms, device=device, dtype=torch.bool), diagonal=1)
    pair_weight = _smooth_cutoff(distances, config.cutoff, config.cutoff_delta)

    size = params["size"]
    strength = params["pair_strength"]
    r0 = size[zi] + size[zj] + 0.08 * torch.abs(size[zi] - size[zj])
    eps = torch.sqrt(strength[zi] * strength[zj])
    pair_terms = (
        0.08 * eps * torch.exp(-2.4 * (distances - 0.65 * r0))
        - 0.10 * eps * torch.exp(-((distances - r0) ** 2) / 0.18)
        + 0.012 * eps / torch.sqrt(distances * distances + 0.25)
    )
    pair_energy = config.pair_scale * (pair_terms * pair_weight).masked_select(upper_mask).sum()

    angle_energy_raw = coords.new_tensor(0.0)
    motif_energy_raw = coords.new_tensor(0.0)
    many_body_energy_raw = coords.new_tensor(0.0)
    radial_centers = torch.linspace(
        config.radial_min,
        config.radial_max,
        config.radial_basis_size,
        device=device,
        dtype=dtype,
    )
    neighbor_not_self = ~torch.eye(num_atoms, device=device, dtype=torch.bool)
    motif_cosines = torch.tensor([-1.0, -0.5, -1.0 / 3.0, 0.0, 0.5], device=device, dtype=dtype)

    for center_idx in range(num_atoms):
        center_type = atom_types[center_idx]
        rel = coords - coords[center_idx]
        dist = distances[center_idx]
        valid = neighbor_not_self[center_idx]
        weights = _smooth_cutoff(dist, config.cutoff, config.cutoff_delta) * valid.to(dtype=dtype)
        active = weights > 1.0e-8
        active_idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        active_count = int(active_idx.numel())
        if active_count < 2:
            continue
        active_types = atom_types[active_idx]
        active_weights = weights[active_idx]
        active_dist = dist[active_idx]
        active_rel = rel[active_idx]
        directions = active_rel / active_dist.clamp_min(config.distance_eps)[:, None]
        rbf = _radial_basis(active_dist, radial_centers, config.radial_width) * active_weights[:, None]
        cos = (directions @ directions.T).clamp(min=-1.0, max=1.0)
        upper_jk = torch.triu(torch.ones(active_count, active_count, device=device, dtype=torch.bool), diagonal=1)
        pair_mask = upper_jk
        pair_mask_f = pair_mask.to(dtype=dtype)
        pair_weights = pair_mask_f * active_weights[:, None] * active_weights[None, :]
        pair_weight_sum = pair_weights.sum().clamp_min(1.0)
        legendre = _legendre_stack(cos, config.angle_lmax)

        left_coeff = params["angle_left"][active_types, : config.angle_lmax + 1, : config.radial_basis_size]
        right_coeff = params["angle_right"][active_types, : config.angle_lmax + 1, : config.radial_basis_size]
        left = torch.einsum("np,nlp->nl", rbf, left_coeff)
        right = torch.einsum("np,nlp->nl", rbf, right_coeff)
        center_angle = params["angle_center"][center_type]
        angle_matrix = torch.zeros_like(cos)
        for ell in range(config.angle_lmax + 1):
            angle_matrix = angle_matrix + center_angle[ell] * (
                left[:, ell, None] * right[None, :, ell]
                + right[:, ell, None] * left[None, :, ell]
            ) * legendre[ell]
        target_cos = params["preferred_cos"][center_type]
        angle_preference = ((cos - target_cos) ** 2 * pair_weights).sum() / pair_weight_sum
        angle_energy_raw = angle_energy_raw + (angle_matrix * pair_mask_f).sum() + params["angle_pref_strength"][center_type] * angle_preference

        target_logits = -((motif_cosines - target_cos) ** 2) / (2.0 * (1.6 * config.motif_sigma) ** 2)
        target_logits = target_logits + 0.75 * torch.nn.functional.one_hot(
            params["motif_well_index"][center_type],
            num_classes=motif_cosines.shape[0],
        ).to(dtype=dtype, device=device)
        motif_probs = torch.softmax(target_logits, dim=0)
        motif_wells = torch.exp(-((cos[None, :, :] - motif_cosines[:, None, None]) ** 2) / (2.0 * config.motif_sigma**2))
        neighbor_factor = 1.0 + 0.04 * (
            params["motif_neighbor"][active_types][:, None] + params["motif_neighbor"][active_types][None, :]
        )
        motif_match = (motif_probs[:, None, None] * motif_wells).sum(dim=0)
        motif_penalty = (1.0 - motif_match).clamp_min(0.0) * neighbor_factor
        motif_energy_raw = motif_energy_raw + params["motif_strength"][center_type] * (
            motif_penalty * pair_weights
        ).sum() / pair_weight_sum

        coord_norm = weights.sum().clamp_min(1.0)
        radial_density = torch.einsum(
            "np,np->p",
            rbf,
            params["mb_pair"][center_type, active_types, : config.radial_basis_size],
        ) / coord_norm
        angular_density = torch.zeros(config.angle_lmax + 1, device=device, dtype=dtype)
        mb_neighbor = params["mb_neighbor"][active_types]
        for ell in range(config.angle_lmax + 1):
                angular_density[ell] = (
                    pair_mask_f
                    * legendre[ell]
                    * (mb_neighbor[:, ell, None] + mb_neighbor[None, :, ell])
                    * active_weights[:, None]
                    * active_weights[None, :]
                ).sum() / (coord_norm * coord_norm)
        mb_u = params["mb_u"][center_type, : config.radial_basis_size]
        mb_v = params["mb_v"][center_type, : config.angle_lmax + 1]
        mb_cross = params["mb_cross"][center_type, : config.radial_basis_size, : config.angle_lmax + 1]
        many_body_energy_raw = many_body_energy_raw + (
            0.04 * torch.sum(mb_u * radial_density * radial_density)
            + 0.015 * torch.sum(mb_v * angular_density * angular_density)
            + 0.006 * torch.sum(mb_cross * radial_density[:, None] * angular_density[None, :])
        )

    angle_energy = config.angle_scale * angle_energy_raw
    motif_energy = config.motif_scale * motif_energy_raw
    many_body_energy = config.many_body_scale * many_body_energy_raw
    components = {
        "pair": pair_energy,
        "angle": angle_energy,
        "motif": motif_energy,
        "many_body": many_body_energy,
    }
    total = pair_energy + angle_energy + motif_energy + many_body_energy
    return total, components


def pack_synomol_transfer_samples(
    samples: list[dict[str, object]],
    *,
    config: SynOMolTransferConfig,
    split: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if not samples:
        raise ValueError("samples must not be empty")
    num_atoms = torch.tensor([int(sample["num_atoms"]) for sample in samples], dtype=torch.long)
    ptr = torch.zeros(len(samples) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(num_atoms, dim=0)
    packed = {
        "config": asdict(config),
        "split": _normalize_split(split),
        "metadata": dict(metadata or {}),
        "atom_types": torch.cat([torch.as_tensor(sample["atom_types"], dtype=torch.long) for sample in samples], dim=0),
        "coords": torch.cat([torch.as_tensor(sample["coords"], dtype=torch.float32) for sample in samples], dim=0),
        "forces": torch.cat([torch.as_tensor(sample["forces"], dtype=torch.float32) for sample in samples], dim=0),
        "energy": torch.stack([torch.as_tensor(sample["energy"], dtype=torch.float32).reshape(()) for sample in samples]),
        "component_energies": {
            name: torch.stack(
                [
                    torch.as_tensor(sample["component_energies"][name], dtype=torch.float32).reshape(())
                    for sample in samples
                ]
            )
            for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
        },
        "num_atoms": num_atoms,
        "ptr": ptr,
        "indices": torch.tensor([int(sample["idx"]) for sample in samples], dtype=torch.long),
        "system_ids": torch.tensor([int(sample["system_id"]) for sample in samples], dtype=torch.long),
        "sample_kinds": [str(sample["sample_kind"]) for sample in samples],
        "primary_motifs": [str(sample["primary_motif"]) for sample in samples],
        "primary_triples": torch.stack([torch.as_tensor(sample["primary_triple"], dtype=torch.long) for sample in samples]),
        "original_primary_triples": torch.stack(
            [
                torch.as_tensor(sample.get("original_primary_triple", sample["primary_triple"]), dtype=torch.long)
                for sample in samples
            ]
        ),
        "heldout_repair_counts": torch.tensor(
            [int(sample.get("heldout_repair_count", 0)) for sample in samples],
            dtype=torch.long,
        ),
        "proposal_seeds": torch.tensor(
            [int(sample.get("proposal_seed", -1)) for sample in samples],
            dtype=torch.long,
        ),
        "accepted_attempts": torch.tensor(
            [int(sample.get("accepted_attempts", 1)) for sample in samples],
            dtype=torch.long,
        ),
        "global_sample_ids": torch.tensor(
            [int(sample.get("global_sample_id", sample["idx"])) for sample in samples],
            dtype=torch.long,
        ),
        "shard_ids": torch.tensor(
            [int(sample.get("shard_id", -1)) for sample in samples],
            dtype=torch.long,
        ),
        "has_heldout_type_combo": torch.tensor(
            [bool(sample["has_heldout_type_combo"]) for sample in samples],
            dtype=torch.bool,
        ),
        "motif_labels": torch.cat([torch.as_tensor(sample["motif_labels"], dtype=torch.long) for sample in samples], dim=0),
        "splits": [str(sample.get("split", split)) for sample in samples],
    }
    if all("component_forces" in sample for sample in samples):
        packed["component_forces"] = {
            name: torch.cat(
                [
                    torch.as_tensor(sample["component_forces"][name], dtype=torch.float32)
                    for sample in samples
                ],
                dim=0,
            )
            for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
        }
    if all("level_energies" in sample and "level_forces" in sample for sample in samples):
        level_names = tuple(samples[0]["level_energies"].keys())
        packed["geometry_levels"] = list(level_names)
        packed["level_energies"] = {
            name: torch.stack(
                [
                    torch.as_tensor(sample["level_energies"][name], dtype=torch.float32).reshape(())
                    for sample in samples
                ]
            )
            for name in level_names
        }
        packed["level_forces"] = {
            name: torch.cat(
                [
                    torch.as_tensor(sample["level_forces"][name], dtype=torch.float32)
                    for sample in samples
                ],
                dim=0,
            )
            for name in level_names
        }
    return packed


def summarize_synomol_transfer_samples(
    samples: list[dict[str, object]],
    *,
    generation_stats: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if not samples:
        raise ValueError("samples must not be empty")
    force_rms = [
        float(torch.sqrt((torch.as_tensor(sample["forces"], dtype=torch.float32) ** 2).sum(dim=-1).mean()).item())
        for sample in samples
    ]
    energy_per_atom = [
        float(torch.as_tensor(sample["energy"], dtype=torch.float32).item()) / float(max(int(sample["num_atoms"]), 1))
        for sample in samples
    ]
    num_atoms = [int(sample["num_atoms"]) for sample in samples]
    component_energies = {
        name: [
            float(torch.as_tensor(sample["component_energies"][name], dtype=torch.float32).item())
            for sample in samples
        ]
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
    }
    component_energies_per_atom = {
        name: [
            float(torch.as_tensor(sample["component_energies"][name], dtype=torch.float32).item())
            / float(max(int(sample["num_atoms"]), 1))
            for sample in samples
        ]
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
    }
    stats = {
        "requested": len(samples),
        "accepted": len(samples),
        "num_atoms": _numeric_summary(num_atoms),
        "force_rms": _numeric_summary(force_rms),
        "energy_per_atom": _numeric_summary(energy_per_atom),
        "component_energies": {
            name: _numeric_summary(values)
            for name, values in component_energies.items()
        },
        "component_energies_per_atom": {
            name: _numeric_summary(values)
            for name, values in component_energies_per_atom.items()
        },
        "sample_kind_counts": _count_strings(str(sample["sample_kind"]) for sample in samples),
        "primary_motif_counts": _count_strings(str(sample["primary_motif"]) for sample in samples),
        "heldout_type_combo_count": int(sum(bool(sample["has_heldout_type_combo"]) for sample in samples)),
        "heldout_repair_count": int(sum(int(sample.get("heldout_repair_count", 0)) for sample in samples)),
    }
    if generation_stats is not None:
        attempts = [int(stat.get("attempts", 1)) for stat in generation_stats]
        rejection_totals: dict[str, int] = {}
        label_neighbor_values: dict[str, list[float]] = {
            "mean_label_neighbors": [],
            "q95_label_neighbors": [],
            "max_label_neighbors": [],
            "neighbor_cap_hit_fraction": [],
        }
        for stat in generation_stats:
            for reason, count in dict(stat.get("rejection_reasons", {})).items():
                rejection_totals[str(reason)] = rejection_totals.get(str(reason), 0) + int(count)
            for name, value in dict(stat.get("label_neighbor_stats", {})).items():
                if name in label_neighbor_values:
                    label_neighbor_values[name].append(float(value))
        total_attempts = int(sum(attempts))
        stats.update(
            {
                "attempts": total_attempts,
                "mean_attempts_per_accepted": float(total_attempts) / float(len(samples)),
                "max_attempts": int(max(attempts) if attempts else 0),
                "rejections": rejection_totals,
                "rejection_rate": float(sum(rejection_totals.values())) / float(max(total_attempts, 1)),
                "heldout_repair_count_from_generation": int(
                    sum(int(stat.get("heldout_repair_count", 0)) for stat in generation_stats)
                ),
            }
        )
        if any(label_neighbor_values.values()):
            stats["label_neighbor_stats"] = {
                name: _numeric_summary(values)
                for name, values in label_neighbor_values.items()
                if values
            }
    return stats


def materialize_synomol_transfer_split(
    output_path: str | Path,
    *,
    config: SynOMolTransferConfig,
    split: str,
    size: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if size <= 0:
        raise ValueError("size must be positive")
    split_config = SynOMolTransferConfig.from_dict({**asdict(config), "length": int(size)})
    dataset = SynOMolTransferDataset(
        Path(output_path).parent.parent,
        split=split,
        config=split_config,
        mode="online",
        config_name=split_config.cache_name(),
    )
    samples = [dataset[index] for index in range(size)]
    packed = pack_synomol_transfer_samples(samples, config=split_config, split=split, metadata=metadata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output_path)
    return packed


def calibrate_synomol_transfer_component_scales(
    config: SynOMolTransferConfig,
    *,
    split: str = "train",
    calibration_size: int = 32,
    force_budget: dict[str, float] | None = None,
    eps: float = 1.0e-8,
) -> tuple[SynOMolTransferConfig, dict[str, float]]:
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    force_budget = dict(force_budget or _DEFAULT_FORCE_BUDGET)
    raw_label_config = replace(
        config,
        pair_scale=1.0,
        angle_scale=1.0,
        motif_scale=1.0,
        many_body_scale=1.0,
    )
    values = {name: [] for name in SYNOMOL_TRANSFER_COMPONENT_NAMES}
    for index in range(calibration_size):
        for attempt in range(config.max_resample_attempts):
            atom_types, coords, _ = _generate_synomol_transfer_inputs(
                index=index,
                split=split,
                epoch=0,
                config=config,
                attempt=attempt,
            )
            has_heldout = bool(_active_type_triples(atom_types, coords, config) & _HELDOUT_COMBO_SET)
            if not _split_allows_active_heldout(_normalize_split(split), has_heldout):
                continue
            energy, forces, components = compute_synomol_transfer_labels(atom_types, coords, config)
            if _sample_is_acceptable(atom_types, coords, energy, forces, components, config):
                break
        else:
            raise RuntimeError(f"failed to sample calibration structure {index} for split={split!r}")
        coords_for_grad = coords.detach().clone().requires_grad_(True)
        _, components = compute_synomol_transfer_energy(atom_types, coords_for_grad, raw_label_config)
        for component_name, component_energy in components.items():
            (grad_coords,) = torch.autograd.grad(
                component_energy,
                coords_for_grad,
                retain_graph=True,
                create_graph=False,
            )
            force_rms = torch.sqrt((grad_coords * grad_coords).sum(dim=-1).mean()).detach()
            values[component_name].append(float(force_rms.item()))
    rms = {name: _robust_mean(values[name]) for name in SYNOMOL_TRANSFER_COMPONENT_NAMES}
    scales = {
        name: float(force_budget.get(name, 1.0)) / (rms[name] + eps)
        for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
    }
    calibrated = replace(
        config,
        pair_scale=scales["pair"],
        angle_scale=scales["angle"],
        motif_scale=scales["motif"],
        many_body_scale=scales["many_body"],
    )
    metadata = {
        "component_force_rms": rms,
        "component_force_budget": {name: float(force_budget.get(name, 1.0)) for name in SYNOMOL_TRANSFER_COMPONENT_NAMES},
        "component_scales": scales,
    }
    return calibrated, metadata


def synomol_transfer_atom_type_table(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    size = torch.tensor(
        [0.52, 0.64, 0.72, 0.80, 0.90, 1.02, 0.58, 0.68, 0.78, 0.88, 0.72, 0.80, 0.90, 1.02, 0.62, 0.96],
        device=device,
        dtype=dtype,
    )
    pair_strength = torch.tensor(
        [0.55, 0.65, 0.82, 0.75, 0.95, 1.05, 0.70, 0.86, 0.92, 1.12, 0.82, 0.75, 0.95, 1.05, 0.78, 1.18],
        device=device,
        dtype=dtype,
    )
    preferred_coordination = torch.tensor(
        [1.0, 2.0, 4.0, 4.0, 6.0, 3.0, 1.5, 2.5, 3.0, 4.5, 4.0, 4.0, 6.0, 3.0, 2.0, 5.0],
        device=device,
        dtype=dtype,
    )
    preferred_cos = torch.tensor(
        [-1.0, -0.5, -1.0 / 3.0, 0.0, -0.5, 0.5, -0.2, -0.6, -1.0 / 3.0, 0.2, 0.0, -1.0 / 3.0, 0.0, -0.5, -1.0, 0.35],
        device=device,
        dtype=dtype,
    )
    motif_class = torch.tensor(
        [0, 1, 3, 4, 5, 2, 6, 7, 3, 2, 4, 3, 5, 2, 0, 5],
        device=device,
        dtype=torch.long,
    )
    motif_cosines = torch.tensor([-1.0, -0.5, -1.0 / 3.0, 0.0, 0.5], device=device, dtype=dtype)
    motif_well_index = torch.argmin(torch.abs(preferred_cos[:, None] - motif_cosines[None, :]), dim=1)
    type_id = torch.arange(SYNOMOL_TRANSFER_NUM_ATOM_TYPES, device=device, dtype=dtype)
    radial_ids = torch.arange(8, device=device, dtype=dtype)
    ell_ids = torch.arange(7, device=device, dtype=dtype)
    motif_ids = torch.arange(5, device=device, dtype=dtype)

    angle_left = torch.sin(0.37 * (type_id[:, None, None] + 1.0) * (ell_ids[None, :, None] + 1.0) + 0.29 * radial_ids[None, None, :])
    angle_right = torch.cos(0.31 * (type_id[:, None, None] + 1.0) * (ell_ids[None, :, None] + 1.0) - 0.23 * radial_ids[None, None, :])
    angle_center = 0.6 + 0.25 * torch.sin(0.41 * (type_id[:, None] + 1.0) * (ell_ids[None, :] + 1.0))
    angle_pref_strength = 0.35 + 0.25 * torch.cos(0.31 * (motif_class.to(dtype=dtype) + 1.0))
    motif_logits = 0.45 + 0.35 * torch.cos(0.53 * (type_id[:, None] + 1.0) + 0.79 * motif_ids[None, :])
    motif_neighbor = torch.sin(0.7 * (type_id + 1.0))
    motif_strength = 0.8 + 0.25 * torch.sin(0.37 * (motif_class.to(dtype=dtype) + 1.0))
    mb_pair = 0.8 + 0.3 * torch.sin(0.19 * (type_id[:, None, None] + 1.0) * (type_id[None, :, None] + 1.0) + 0.43 * radial_ids[None, None, :])
    mb_neighbor = 0.7 + 0.25 * torch.cos(0.47 * (type_id[:, None] + 1.0) * (ell_ids[None, :] + 1.0))
    mb_u = 0.5 + 0.2 * torch.sin(0.17 * (type_id[:, None] + 1.0) * (radial_ids[None, :] + 1.0))
    mb_v = 0.5 + 0.2 * torch.cos(0.21 * (type_id[:, None] + 1.0) * (ell_ids[None, :] + 1.0))
    mb_cross = 0.15 * torch.sin(
        0.11 * (type_id[:, None, None] + 1.0)
        * (radial_ids[None, :, None] + 1.0)
        * (ell_ids[None, None, :] + 1.0)
    )

    return {
        "size": size,
        "pair_strength": pair_strength,
        "preferred_coordination": preferred_coordination,
        "preferred_cos": preferred_cos,
        "motif_class": motif_class,
        "motif_well_index": motif_well_index,
        "angle_left": angle_left,
        "angle_right": angle_right,
        "angle_center": angle_center,
        "angle_pref_strength": angle_pref_strength,
        "motif_logits": motif_logits,
        "motif_neighbor": motif_neighbor,
        "motif_strength": motif_strength,
        "mb_pair": mb_pair,
        "mb_neighbor": mb_neighbor,
        "mb_u": mb_u,
        "mb_v": mb_v,
        "mb_cross": mb_cross,
    }


def _load_fixed_split(cache_dir: Path, split: str) -> dict[str, Any]:
    fixed_path = cache_dir / f"{split}.pt"
    manifest_path = cache_dir / f"{split}_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        shard_entries = manifest.get("shards", [])
        if not shard_entries:
            raise ValueError(f"SynOMol-Transfer shard manifest has no shards: {manifest_path}")
        loaded = []
        for entry in shard_entries:
            shard_name = entry["path"] if isinstance(entry, dict) else str(entry)
            loaded.append(torch.load(cache_dir / shard_name, map_location="cpu", weights_only=False))
        return _merge_packed_splits(loaded)
    if fixed_path.is_file():
        return torch.load(fixed_path, map_location="cpu", weights_only=False)
    raise FileNotFoundError(
        f"SynOMol-Transfer fixed split not found: {fixed_path} or {manifest_path}"
    )


def _merge_packed_splits(shards: list[dict[str, Any]]) -> dict[str, Any]:
    if not shards:
        raise ValueError("shards must not be empty")
    if len(shards) == 1:
        return shards[0]
    ptr_values = [torch.zeros(1, dtype=torch.long)]
    total_atoms = 0
    for shard in shards:
        shard_num_atoms = torch.as_tensor(shard["num_atoms"], dtype=torch.long)
        total_atoms += int(shard_num_atoms.sum().item())
        ptr_values.append(total_atoms - int(shard_num_atoms.sum().item()) + torch.cumsum(shard_num_atoms, dim=0))
    merged = {
        "config": shards[0].get("config", {}),
        "split": shards[0].get("split", ""),
        "metadata": shards[0].get("metadata", {}),
        "atom_types": torch.cat([torch.as_tensor(shard["atom_types"], dtype=torch.long) for shard in shards], dim=0),
        "coords": torch.cat([torch.as_tensor(shard["coords"], dtype=torch.float32) for shard in shards], dim=0),
        "forces": torch.cat([torch.as_tensor(shard["forces"], dtype=torch.float32) for shard in shards], dim=0),
        "energy": torch.cat([torch.as_tensor(shard["energy"], dtype=torch.float32) for shard in shards], dim=0),
        "component_energies": {
            name: torch.cat(
                [
                    torch.as_tensor(
                        shard["component_energies"].get(name, torch.zeros_like(torch.as_tensor(shard["energy"]))),
                        dtype=torch.float32,
                    )
                    for shard in shards
                ],
                dim=0,
            )
            for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
        },
        "num_atoms": torch.cat([torch.as_tensor(shard["num_atoms"], dtype=torch.long) for shard in shards], dim=0),
        "ptr": torch.cat(ptr_values, dim=0),
        "indices": torch.cat([torch.as_tensor(shard.get("indices", torch.arange(len(shard["energy"]))), dtype=torch.long) for shard in shards], dim=0),
        "system_ids": torch.cat([torch.as_tensor(shard.get("system_ids", torch.arange(len(shard["energy"]))), dtype=torch.long) for shard in shards], dim=0),
        "sample_kinds": [value for shard in shards for value in list(shard.get("sample_kinds", [""] * len(shard["energy"])))],
        "primary_motifs": [value for shard in shards for value in list(shard.get("primary_motifs", [""] * len(shard["energy"])))],
        "primary_triples": torch.cat([torch.as_tensor(shard.get("primary_triples", torch.zeros(len(shard["energy"]), 3)), dtype=torch.long) for shard in shards], dim=0),
        "original_primary_triples": torch.cat([torch.as_tensor(shard.get("original_primary_triples", shard.get("primary_triples", torch.zeros(len(shard["energy"]), 3))), dtype=torch.long) for shard in shards], dim=0),
        "heldout_repair_counts": torch.cat([torch.as_tensor(shard.get("heldout_repair_counts", torch.zeros(len(shard["energy"]))), dtype=torch.long) for shard in shards], dim=0),
        "proposal_seeds": torch.cat([torch.as_tensor(shard.get("proposal_seeds", torch.full((len(shard["energy"]),), -1)), dtype=torch.long) for shard in shards], dim=0),
        "accepted_attempts": torch.cat([torch.as_tensor(shard.get("accepted_attempts", torch.ones(len(shard["energy"]))), dtype=torch.long) for shard in shards], dim=0),
        "global_sample_ids": torch.cat([torch.as_tensor(shard.get("global_sample_ids", shard.get("indices", torch.arange(len(shard["energy"])))), dtype=torch.long) for shard in shards], dim=0),
        "shard_ids": torch.cat([torch.as_tensor(shard.get("shard_ids", torch.full((len(shard["energy"]),), -1)), dtype=torch.long) for shard in shards], dim=0),
        "has_heldout_type_combo": torch.cat([torch.as_tensor(shard.get("has_heldout_type_combo", torch.zeros(len(shard["energy"]))), dtype=torch.bool) for shard in shards], dim=0),
        "motif_labels": torch.cat([torch.as_tensor(shard["motif_labels"], dtype=torch.long) for shard in shards], dim=0),
        "splits": [value for shard in shards for value in list(shard.get("splits", [shard.get("split", "")] * len(shard["energy"])))],
    }
    if all("component_forces" in shard for shard in shards):
        merged["component_forces"] = {
            name: torch.cat(
                [
                    torch.as_tensor(shard["component_forces"][name], dtype=torch.float32)
                    for shard in shards
                ],
                dim=0,
            )
            for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
        }
    if all("level_energies" in shard and "level_forces" in shard for shard in shards):
        level_names = tuple(shards[0].get("geometry_levels", shards[0]["level_energies"].keys()))
        merged["geometry_levels"] = list(level_names)
        merged["level_energies"] = {
            name: torch.cat(
                [
                    torch.as_tensor(shard["level_energies"][name], dtype=torch.float32)
                    for shard in shards
                ],
                dim=0,
            )
            for name in level_names
        }
        merged["level_forces"] = {
            name: torch.cat(
                [
                    torch.as_tensor(shard["level_forces"][name], dtype=torch.float32)
                    for shard in shards
                ],
                dim=0,
            )
            for name in level_names
        }
    return merged


def _normalize_fixed_split(loaded: dict[str, Any]) -> dict[str, Any]:
    required = {
        "atom_types",
        "coords",
        "forces",
        "energy",
        "component_energies",
        "num_atoms",
        "ptr",
        "motif_labels",
    }
    missing = sorted(required - set(loaded))
    if missing:
        raise KeyError(f"SynOMol-Transfer fixed split is missing keys: {missing}")
    size = int(torch.as_tensor(loaded["energy"]).shape[0])
    normalized = {
        "atom_types": torch.as_tensor(loaded["atom_types"], dtype=torch.long),
        "coords": torch.as_tensor(loaded["coords"], dtype=torch.float32),
        "forces": torch.as_tensor(loaded["forces"], dtype=torch.float32),
        "energy": torch.as_tensor(loaded["energy"], dtype=torch.float32),
        "component_energies": {
            name: torch.as_tensor(loaded["component_energies"].get(name, torch.zeros(size)), dtype=torch.float32)
            for name in SYNOMOL_TRANSFER_COMPONENT_NAMES
        },
        "num_atoms": torch.as_tensor(loaded["num_atoms"], dtype=torch.long),
        "ptr": torch.as_tensor(loaded["ptr"], dtype=torch.long),
        "indices": torch.as_tensor(loaded.get("indices", torch.arange(size)), dtype=torch.long),
        "system_ids": torch.as_tensor(loaded.get("system_ids", torch.arange(size)), dtype=torch.long),
        "sample_kinds": list(loaded.get("sample_kinds", [""] * size)),
        "primary_motifs": list(loaded.get("primary_motifs", [""] * size)),
        "primary_triples": torch.as_tensor(loaded.get("primary_triples", torch.zeros(size, 3)), dtype=torch.long),
        "original_primary_triples": torch.as_tensor(
            loaded.get("original_primary_triples", loaded.get("primary_triples", torch.zeros(size, 3))),
            dtype=torch.long,
        ),
        "heldout_repair_counts": torch.as_tensor(loaded.get("heldout_repair_counts", torch.zeros(size)), dtype=torch.long),
        "proposal_seeds": torch.as_tensor(loaded.get("proposal_seeds", torch.full((size,), -1)), dtype=torch.long),
        "accepted_attempts": torch.as_tensor(loaded.get("accepted_attempts", torch.ones(size)), dtype=torch.long),
        "global_sample_ids": torch.as_tensor(loaded.get("global_sample_ids", loaded.get("indices", torch.arange(size))), dtype=torch.long),
        "shard_ids": torch.as_tensor(loaded.get("shard_ids", torch.full((size,), -1)), dtype=torch.long),
        "has_heldout_type_combo": torch.as_tensor(loaded.get("has_heldout_type_combo", torch.zeros(size)), dtype=torch.bool),
        "motif_labels": torch.as_tensor(loaded["motif_labels"], dtype=torch.long),
        "splits": list(loaded.get("splits", [loaded.get("split", "")] * size)),
    }
    if "component_forces" in loaded:
        normalized["component_forces"] = {
            name: torch.as_tensor(loaded["component_forces"][name], dtype=torch.float32)
            for name in loaded["component_forces"]
        }
    if "level_energies" in loaded and "level_forces" in loaded:
        level_names = tuple(loaded.get("geometry_levels", loaded["level_energies"].keys()))
        normalized["geometry_levels"] = list(level_names)
        normalized["level_energies"] = {
            name: torch.as_tensor(loaded["level_energies"][name], dtype=torch.float32)
            for name in level_names
        }
        normalized["level_forces"] = {
            name: torch.as_tensor(loaded["level_forces"][name], dtype=torch.float32)
            for name in level_names
        }
    return normalized


def propose_synomol_transfer_inputs(
    *,
    index: int,
    split: str,
    epoch: int,
    config: SynOMolTransferConfig,
    attempt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    split = _normalize_split(split)
    seed = (_sample_seed(config.seed, split, epoch, index) + 7_919 * int(attempt)) % (2**63 - 1)
    generator = _make_generator(seed)
    table = synomol_transfer_atom_type_table()
    num_atoms = _sample_num_atoms_for_split(config, split, generator)
    sample_kind = _sample_kind_for_split(split, index, generator)
    primary_motif = _sample_primary_motif(split, generator)
    primary_triple = _sample_primary_triple(split, generator)
    atom_types, coords, motif_labels = _build_template_geometry(
        num_atoms=num_atoms,
        primary_motif=primary_motif,
        primary_triple=primary_triple,
        table=table,
        config=config,
        generator=generator,
    )
    coords = _random_rotate_translate(coords, generator)

    if sample_kind == "collision":
        coords = _sample_collision_coords(num_atoms, config, generator)

    metadata = {
        "system_id": _system_id(split, epoch, index),
        "sample_kind": sample_kind,
        "primary_motif": primary_motif,
        "primary_triple": primary_triple,
        "original_primary_triple": primary_triple,
        "has_heldout_type_combo": False,
        "heldout_repair_count": 0,
        "motif_labels": motif_labels,
        "proposal_seed": int(seed),
    }
    return atom_types, coords.to(dtype=torch.float32), metadata


def _postprocess_synomol_transfer_inputs_cpu(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    metadata: dict[str, object],
    *,
    split: str,
    config: SynOMolTransferConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    table = synomol_transfer_atom_type_table()
    sample_kind = str(metadata["sample_kind"])
    primary_motif = str(metadata["primary_motif"])
    if sample_kind != "collision":
        relax_steps = config.relax_steps if sample_kind != "relaxation" else config.relaxation_snapshot_steps
        coords = _relax_coords(atom_types, coords, config, steps=relax_steps)
        if sample_kind == "langevin":
            coords = _langevin_coords(atom_types, coords, config, generator)
        elif sample_kind in {"bond_distortion", "angle_distortion", "shell_distortion"}:
            coords = _distort_coords(coords, atom_types, primary_motif, sample_kind, split, table, config, generator)
        coords = coords - coords.mean(dim=0, keepdim=True)
    metadata = dict(metadata)
    repair_count = 0
    if split != "test_type_combo":
        atom_types, repair_count = _repair_heldout_type_combos(atom_types, coords, config, generator)
    metadata["heldout_repair_count"] = int(repair_count)
    if atom_types.shape[0] >= 3:
        metadata["primary_triple"] = (int(atom_types[0].item()), int(atom_types[1].item()), int(atom_types[2].item()))
    return atom_types, coords.to(dtype=torch.float32), metadata


def _generate_synomol_transfer_inputs(
    *,
    index: int,
    split: str,
    epoch: int,
    config: SynOMolTransferConfig,
    attempt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    split = _normalize_split(split)
    atom_types, coords, metadata = propose_synomol_transfer_inputs(
        index=index,
        split=split,
        epoch=epoch,
        config=config,
        attempt=attempt,
    )
    generator = _make_generator(int(metadata["proposal_seed"]))
    return _postprocess_synomol_transfer_inputs_cpu(
        atom_types,
        coords,
        metadata,
        split=split,
        config=config,
        generator=generator,
    )


def _build_template_geometry(
    *,
    num_atoms: int,
    primary_motif: str,
    primary_triple: tuple[int, int, int],
    table: dict[str, torch.Tensor],
    config: SynOMolTransferConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center_type, left_type, right_type = primary_triple
    motif_id = _MOTIF_TO_ID[primary_motif]
    atom_types = torch.empty(num_atoms, dtype=torch.long)
    motif_labels = torch.full((num_atoms,), -1, dtype=torch.long)
    coords = torch.zeros(num_atoms, 3, dtype=torch.float32)
    atom_types[0] = center_type
    motif_labels[0] = motif_id

    directions = _motif_directions(primary_motif)
    neighbor_count = min(len(directions), max(num_atoms - 1, 0))
    if neighbor_count >= 1:
        atom_types[1] = left_type
    if neighbor_count >= 2:
        atom_types[2] = right_type
    for local_idx in range(3, neighbor_count + 1):
        atom_types[local_idx] = _sample_type_for_split("train", generator)

    for atom_idx in range(1, neighbor_count + 1):
        parent_type = int(atom_types[0].item())
        child_type = int(atom_types[atom_idx].item())
        distance = _preferred_distance(parent_type, child_type, table)
        noise = 0.04 * torch.randn(3, generator=generator)
        coords[atom_idx] = directions[atom_idx - 1] * distance + noise
        motif_labels[atom_idx] = motif_id

    fragment_size = 24
    fragment_spacing = config.cutoff + 2.5
    fragment_start = 0
    fragment_index = 0
    next_fragment_start = max(fragment_size, neighbor_count + 1)
    for atom_idx in range(neighbor_count + 1, num_atoms):
        if atom_idx >= next_fragment_start and (atom_idx - next_fragment_start) % fragment_size == 0:
            fragment_index += 1
            fragment_start = atom_idx
            child_type = _sample_type_for_split("train", generator)
            atom_types[atom_idx] = child_type
            lateral = torch.tensor(
                [
                    fragment_spacing * fragment_index,
                    fragment_spacing * 0.37 * (fragment_index % 3),
                    fragment_spacing * 0.19 * (fragment_index % 5),
                ],
                dtype=torch.float32,
            )
            coords[atom_idx] = lateral + 0.15 * torch.randn(3, generator=generator)
            motif_labels[atom_idx] = -1
            continue
        parent = int(torch.randint(low=fragment_start, high=atom_idx, size=(), generator=generator).item())
        child_type = _sample_type_for_split("train", generator)
        atom_types[atom_idx] = child_type
        parent_type = int(atom_types[parent].item())
        direction = _random_unit_vector(generator)
        distance = _preferred_distance(parent_type, child_type, table) * (0.92 + 0.16 * torch.rand((), generator=generator).item())
        coords[atom_idx] = coords[parent] + direction * distance + 0.06 * torch.randn(3, generator=generator)
        motif_labels[atom_idx] = motif_labels[parent] if torch.rand((), generator=generator).item() < 0.35 else -1

    coords = _resolve_initial_clashes(coords, atom_types, table, config, generator)
    return atom_types, coords - coords.mean(dim=0, keepdim=True), motif_labels


def _resolve_initial_clashes(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    table: dict[str, torch.Tensor],
    config: SynOMolTransferConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    coords = coords.clone()
    for _ in range(config.max_resample_attempts):
        distances = torch.cdist(coords, coords)
        distances = distances.masked_fill(torch.eye(coords.shape[0], dtype=torch.bool), float("inf"))
        if float(distances.min().item()) >= config.d_min:
            return coords
        pair = torch.nonzero(distances == distances.min(), as_tuple=False)[0]
        move_idx = int(pair[1].item())
        parent_idx = int(pair[0].item())
        direction = _random_unit_vector(generator)
        distance = _preferred_distance(int(atom_types[parent_idx].item()), int(atom_types[move_idx].item()), table)
        coords[move_idx] = coords[parent_idx] + direction * distance
    return coords


def _relax_coords(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    steps: int,
) -> torch.Tensor:
    x = coords.detach().clone()
    for _ in range(int(steps)):
        x = x.detach().requires_grad_(True)
        energy = _generation_energy(atom_types, x, config)
        (grad_x,) = torch.autograd.grad(energy, x, create_graph=False)
        step = (-config.relax_step_size * grad_x).clamp(
            min=-config.max_relax_displacement,
            max=config.max_relax_displacement,
        )
        x = (x + step).detach()
        x = x - x.mean(dim=0, keepdim=True)
    return x


def _langevin_coords(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    x = coords.detach().clone()
    noise_scale = (2.0 * config.langevin_step_size * config.langevin_temperature) ** 0.5
    for _ in range(int(config.langevin_steps)):
        x = x.detach().requires_grad_(True)
        energy, _ = compute_synomol_transfer_energy(atom_types, x, config)
        (grad_x,) = torch.autograd.grad(energy, x, create_graph=False)
        noise = noise_scale * torch.randn(x.shape, generator=generator, dtype=x.dtype)
        x = (x - config.langevin_step_size * grad_x + noise).detach()
        x = x - x.mean(dim=0, keepdim=True)
    return x


def _generation_energy(atom_types: torch.Tensor, coords: torch.Tensor, config: SynOMolTransferConfig) -> torch.Tensor:
    raw_config = replace(
        config,
        pair_scale=config.generation_pair_scale,
        angle_scale=config.generation_angle_scale,
        motif_scale=config.generation_motif_scale,
        many_body_scale=config.generation_many_body_scale,
    )
    energy, components = compute_synomol_transfer_energy(atom_types, coords, raw_config)
    table = synomol_transfer_atom_type_table(device=coords.device, dtype=coords.dtype)
    soft_bonds = _soft_bond_weights(atom_types, coords, table, config)
    coordination = soft_bonds.sum(dim=1)
    preferred = table["preferred_coordination"][atom_types]
    coord_energy = 0.02 * ((coordination - preferred) ** 2).mean()
    return energy + coord_energy + 0.3 * components["motif"]


def _distort_coords(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    primary_motif: str,
    sample_kind: str,
    split: str,
    table: dict[str, torch.Tensor],
    config: SynOMolTransferConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    x = coords.clone()
    if x.shape[0] < 3:
        return x
    if sample_kind == "bond_distortion":
        target = 1
        direction = x[target] - x[0]
        direction = direction / direction.norm().clamp_min(config.distance_eps)
        delta = config.distortion_scale * (0.5 + torch.rand((), generator=generator).item())
        sign = -1.0 if torch.rand((), generator=generator).item() < 0.35 else 1.0
        x[target] = x[target] + sign * delta * direction
    elif sample_kind == "angle_distortion":
        target = 2
        axis = _random_unit_vector(generator)
        direction = x[target] - x[0]
        perp = torch.cross(axis, direction, dim=0)
        if float(perp.norm().item()) < 1.0e-6:
            perp = _random_unit_vector(generator)
        perp = perp / perp.norm().clamp_min(config.distance_eps)
        x[target] = x[target] + config.distortion_scale * perp
    elif sample_kind == "shell_distortion":
        target_motif = _sample_shell_distortion_motif(split, primary_motif, generator)
        directions = _motif_directions(target_motif)
        count = min(len(directions), x.shape[0] - 1)
        for idx in range(count):
            child = idx + 1
            radius = _preferred_distance(int(atom_types[0].item()), int(atom_types[child].item()), table)
            x[child] = x[0] + directions[idx] * radius
    return x - x.mean(dim=0, keepdim=True)


def _sample_collision_coords(num_atoms: int, config: SynOMolTransferConfig, generator: torch.Generator) -> torch.Tensor:
    box = float((num_atoms / max(config.density * 6.0, 1.0e-6)) ** (1.0 / 3.0))
    coords = (torch.rand(num_atoms, 3, generator=generator) - 0.5) * box
    return coords - coords.mean(dim=0, keepdim=True)


def _sample_shell_distortion_motif(split: str, primary_motif: str, generator: torch.Generator) -> str:
    if split == "test_motif":
        choices = SYNOMOL_TRANSFER_OOD_MOTIFS
    else:
        choices = _TRAIN_SHELL_DISTORTION_MOTIFS
    available = [motif for motif in choices if motif != primary_motif] or list(choices)
    return available[int(torch.randint(len(available), size=(), generator=generator).item())]


def _sample_primary_triple(split: str, generator: torch.Generator) -> tuple[int, int, int]:
    if split == "test_type_combo":
        combo = SYNOMOL_TRANSFER_HELDOUT_TYPE_COMBOS[
            int(torch.randint(len(SYNOMOL_TRANSFER_HELDOUT_TYPE_COMBOS), size=(), generator=generator).item())
        ]
        return int(combo[0]), int(combo[1]), int(combo[2])

    for _ in range(256):
        center = _sample_type_for_split(split, generator)
        left = _sample_type_for_split(split, generator)
        right = _sample_type_for_split(split, generator)
        triple = (center, min(left, right), max(left, right))
        if triple not in _HELDOUT_COMBO_SET:
            return center, left, right
    raise RuntimeError("failed to sample non-held-out SynOMol-Transfer type triple")


def _sample_type_for_split(split: str, generator: torch.Generator) -> int:
    del split
    return int(torch.randint(SYNOMOL_TRANSFER_NUM_ATOM_TYPES, size=(), generator=generator).item())


def _sample_primary_motif(split: str, generator: torch.Generator) -> str:
    if split == "test_motif":
        choices = SYNOMOL_TRANSFER_OOD_MOTIFS
    else:
        choices = SYNOMOL_TRANSFER_TRAIN_MOTIFS
    return choices[int(torch.randint(len(choices), size=(), generator=generator).item())]


def _sample_kind_for_split(split: str, index: int, generator: torch.Generator) -> str:
    if split == "test_perturb":
        return ("bond_distortion", "angle_distortion", "shell_distortion")[index % 3]
    if split == "test_size":
        return ("relaxed", "langevin", "relaxation")[index % 3]
    value = float(torch.rand((), generator=generator).item())
    cumulative = 0.0
    for kind, weight in _DEFAULT_KIND_WEIGHTS:
        cumulative += float(weight)
        if value <= cumulative:
            return kind
    return _DEFAULT_KIND_WEIGHTS[-1][0]


def _sample_num_atoms_for_split(
    config: SynOMolTransferConfig,
    split: str,
    generator: torch.Generator,
) -> int:
    if split == "test_size":
        return _sample_num_atoms(config.size_ood_num_atoms, generator)
    return _sample_num_atoms(config.num_atoms, generator)


def _sample_num_atoms(num_atoms: int | tuple[int, int], generator: torch.Generator) -> int:
    if isinstance(num_atoms, int):
        return int(num_atoms)
    low, high = int(num_atoms[0]), int(num_atoms[1])
    return int(torch.randint(low=low, high=high + 1, size=(), generator=generator).item())


def _motif_directions(motif: str) -> torch.Tensor:
    sqrt3 = 3.0**0.5
    if motif == "linear":
        values = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    elif motif == "bent":
        values = [[1.0, 0.0, 0.0], [-0.5, 0.8660254, 0.0]]
    elif motif == "trigonal":
        values = [[1.0, 0.0, 0.0], [-0.5, 0.8660254, 0.0], [-0.5, -0.8660254, 0.0]]
    elif motif == "tetrahedral":
        values = [
            [1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3],
            [1.0 / sqrt3, -1.0 / sqrt3, -1.0 / sqrt3],
            [-1.0 / sqrt3, 1.0 / sqrt3, -1.0 / sqrt3],
            [-1.0 / sqrt3, -1.0 / sqrt3, 1.0 / sqrt3],
        ]
    elif motif == "square_planar":
        values = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    elif motif == "octahedral":
        values = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    elif motif == "ring":
        values = [
            [1.0, 0.0, 0.0],
            [0.5, 0.8660254, 0.0],
            [-0.5, 0.8660254, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.8660254, 0.0],
            [0.5, -0.8660254, 0.0],
        ]
    else:
        values = [[1.0, 0.0, 0.0], [0.25, 0.95, 0.2], [-0.65, 0.25, -0.72]]
    directions = torch.tensor(values, dtype=torch.float32)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _preferred_distance(type_a: int, type_b: int, table: dict[str, torch.Tensor]) -> float:
    size = table["size"]
    return float(size[type_a].item() + size[type_b].item() + 0.05)


def _random_rotate_translate(coords: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    rotation = _random_rotation(generator)
    translation = 0.25 * torch.randn(3, generator=generator)
    return coords @ rotation.T + translation


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs[None, :]
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _random_unit_vector(generator: torch.Generator) -> torch.Tensor:
    vector = torch.randn(3, generator=generator)
    return vector / vector.norm().clamp_min(1.0e-8)


def _radial_basis(distances: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    return torch.exp(-((distances[:, None] - centers[None, :]) ** 2) / (width**2))


def _validate_batched_inputs(atom_types: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> None:
    if atom_types.ndim != 2:
        raise ValueError(f"atom_types must have shape (B, N), got {tuple(atom_types.shape)}")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (B, N, 3), got {tuple(coords.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (B, N), got {tuple(mask.shape)}")
    if atom_types.shape != mask.shape or coords.shape[:2] != atom_types.shape:
        raise ValueError("atom_types, coords, and mask batch dimensions must match")


def _batch_gather_nodes(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch_idx, indices]


def _soft_bond_weights(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    table: dict[str, torch.Tensor],
    config: SynOMolTransferConfig,
) -> torch.Tensor:
    distances = torch.cdist(coords, coords).clamp_min(config.distance_eps)
    sizes = table["size"][atom_types]
    preferred = sizes[:, None] + sizes[None, :] + 0.05
    gaussian = torch.exp(-((distances - preferred) ** 2) / (0.24**2))
    cutoff = _smooth_cutoff(distances, min(config.cutoff, 2.8), min(config.cutoff_delta, 0.45))
    not_self = ~torch.eye(coords.shape[0], device=coords.device, dtype=torch.bool)
    return gaussian * cutoff * not_self.to(dtype=coords.dtype)


def _soft_bond_weights_batch(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    table: dict[str, torch.Tensor],
    config: SynOMolTransferConfig,
) -> torch.Tensor:
    distances = torch.cdist(coords, coords).clamp_min(config.distance_eps)
    sizes = table["size"][atom_types]
    preferred = sizes[:, :, None] + sizes[:, None, :] + 0.05
    gaussian = torch.exp(-((distances - preferred) ** 2) / (0.24**2))
    cutoff = _smooth_cutoff(distances, min(config.cutoff, 2.8), min(config.cutoff_delta, 0.45))
    max_atoms = coords.shape[1]
    not_self = ~torch.eye(max_atoms, device=coords.device, dtype=torch.bool)[None, :, :]
    valid_pair = mask[:, :, None] & mask[:, None, :] & not_self
    return gaussian * cutoff * valid_pair.to(dtype=coords.dtype)


def _smooth_cutoff(distances: torch.Tensor, cutoff: float, delta: float) -> torch.Tensor:
    start = cutoff - delta
    plateau = distances < start
    taper = (distances >= start) & (distances < cutoff)
    values = torch.zeros_like(distances)
    values = torch.where(plateau, torch.ones_like(values), values)
    taper_arg = torch.pi * (distances - start) / delta
    taper_values = 0.5 * (torch.cos(taper_arg) + 1.0)
    return torch.where(taper, taper_values, values)


def _legendre_stack(values: torch.Tensor, max_l: int) -> torch.Tensor:
    polys = [torch.ones_like(values)]
    if max_l >= 1:
        polys.append(values)
    if max_l >= 2:
        polys.append(0.5 * (3.0 * values.pow(2) - 1.0))
    if max_l >= 3:
        polys.append(0.5 * (5.0 * values.pow(3) - 3.0 * values))
    if max_l >= 4:
        polys.append((35.0 * values.pow(4) - 30.0 * values.pow(2) + 3.0) / 8.0)
    if max_l >= 5:
        polys.append((63.0 * values.pow(5) - 70.0 * values.pow(3) + 15.0 * values) / 8.0)
    if max_l >= 6:
        polys.append((231.0 * values.pow(6) - 315.0 * values.pow(4) + 105.0 * values.pow(2) - 5.0) / 16.0)
    return torch.stack(polys[: max_l + 1], dim=0)


def _active_type_triples(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    threshold: float = 5.0e-2,
) -> set[tuple[int, int, int]]:
    if atom_types.numel() < 3:
        return set()
    table = synomol_transfer_atom_type_table(device=coords.device, dtype=coords.dtype)
    weights = _soft_bond_weights(atom_types.to(device=coords.device, dtype=torch.long), coords, table, config)
    triples: set[tuple[int, int, int]] = set()
    atom_list = [int(value) for value in atom_types.detach().cpu().reshape(-1).tolist()]
    for center_idx in range(coords.shape[0]):
        active = torch.nonzero(weights[center_idx] > threshold**0.5, as_tuple=False).reshape(-1).tolist()
        if len(active) < 2:
            continue
        center_type = atom_list[center_idx]
        for left_pos, left_idx in enumerate(active):
            for right_idx in active[left_pos + 1:]:
                if float((weights[center_idx, left_idx] * weights[center_idx, right_idx]).item()) <= threshold:
                    continue
                left_type = atom_list[int(left_idx)]
                right_type = atom_list[int(right_idx)]
                triples.add((center_type, min(left_type, right_type), max(left_type, right_type)))
    return triples


def _find_active_heldout_triplet_indices(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
    *,
    threshold: float = 5.0e-2,
) -> tuple[int, int, int] | None:
    if atom_types.numel() < 3:
        return None
    table = synomol_transfer_atom_type_table(device=coords.device, dtype=coords.dtype)
    weights = _soft_bond_weights(atom_types.to(device=coords.device, dtype=torch.long), coords, table, config)
    atom_list = [int(value) for value in atom_types.detach().cpu().reshape(-1).tolist()]
    for center_idx in range(coords.shape[0]):
        active = torch.nonzero(weights[center_idx] > threshold**0.5, as_tuple=False).reshape(-1).tolist()
        if len(active) < 2:
            continue
        center_type = atom_list[center_idx]
        for left_pos, left_idx in enumerate(active):
            for right_idx in active[left_pos + 1:]:
                if float((weights[center_idx, left_idx] * weights[center_idx, right_idx]).item()) <= threshold:
                    continue
                left_type = atom_list[int(left_idx)]
                right_type = atom_list[int(right_idx)]
                if (center_type, min(left_type, right_type), max(left_type, right_type)) in _HELDOUT_COMBO_SET:
                    return center_idx, int(left_idx), int(right_idx)
    return None


def _repair_heldout_type_combos(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynOMolTransferConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    repaired = atom_types.clone()
    repair_count = 0
    for _ in range(4 * config.max_resample_attempts):
        hit = _find_active_heldout_triplet_indices(repaired, coords, config)
        if hit is None:
            return repaired, repair_count
        center_idx, left_idx, right_idx = hit
        center_type = int(repaired[center_idx].item())
        left_type = int(repaired[left_idx].item())
        change_idx = right_idx
        other_type = left_type
        if right_idx == 0 and left_idx != 0:
            change_idx = left_idx
            other_type = int(repaired[right_idx].item())
        for _ in range(64):
            candidate = int(torch.randint(SYNOMOL_TRANSFER_NUM_ATOM_TYPES, size=(), generator=generator).item())
            if candidate == int(repaired[change_idx].item()):
                continue
            triple = (center_type, min(other_type, candidate), max(other_type, candidate))
            if triple not in _HELDOUT_COMBO_SET:
                repaired[change_idx] = candidate
                repair_count += 1
                break
        else:
            repaired[change_idx] = (int(repaired[change_idx].item()) + 1) % SYNOMOL_TRANSFER_NUM_ATOM_TYPES
            repair_count += 1
    return repaired, repair_count


def _split_allows_active_heldout(split: str, has_heldout: bool) -> bool:
    if split == "test_type_combo":
        return has_heldout
    return not has_heldout


def _sample_is_acceptable(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    energy: torch.Tensor,
    forces: torch.Tensor,
    components: dict[str, torch.Tensor],
    config: SynOMolTransferConfig,
) -> bool:
    if not torch.isfinite(energy).item() or abs(float(energy.detach().item())) > config.max_abs_energy:
        return False
    energy_per_atom = abs(float(energy.detach().item())) / float(max(int(atom_types.numel()), 1))
    if energy_per_atom > config.max_abs_energy_per_atom:
        return False
    if not torch.isfinite(forces).all().item():
        return False
    if any(not torch.isfinite(value).item() for value in components.values()):
        return False
    if atom_types.numel() > 1:
        distances = torch.cdist(coords, coords)
        distances = distances.masked_fill(torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device), float("inf"))
        if float(distances.min().item()) < config.d_min:
            return False
    force_rms = torch.sqrt((forces * forces).sum(dim=-1).mean())
    if not torch.isfinite(force_rms).item():
        return False
    force_rms_value = float(force_rms.item())
    return config.force_rms_min <= force_rms_value <= config.force_rms_max


def _robust_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() < 8:
        return float(tensor.median().item())
    sorted_values = torch.sort(tensor).values
    trim = max(1, int(0.1 * sorted_values.numel()))
    trimmed = sorted_values[trim:-trim] if 2 * trim < sorted_values.numel() else sorted_values
    return float(trimmed.mean().item())


def _normalize_split(split: str) -> str:
    normalized = split.lower()
    if normalized == "test":
        normalized = "test_iid"
    if normalized in {"valid", "validation"}:
        normalized = "val"
    if normalized not in SYNOMOL_TRANSFER_SPLITS:
        raise ValueError(f"split must be one of {SYNOMOL_TRANSFER_SPLITS}, got {split!r}")
    return normalized


def _validate_num_atoms(num_atoms: int | tuple[int, int], *, name: str) -> None:
    if isinstance(num_atoms, int):
        if num_atoms <= 0:
            raise ValueError(f"{name} must be positive")
        return
    if not isinstance(num_atoms, tuple) or len(num_atoms) != 2:
        raise ValueError(f"{name} must be an int or a (min, max) tuple")
    if int(num_atoms[0]) <= 0 or int(num_atoms[1]) <= 0:
        raise ValueError(f"{name} bounds must be positive")
    if int(num_atoms[0]) > int(num_atoms[1]):
        raise ValueError(f"{name} lower bound must be <= upper bound")


def _format_num_atoms(num_atoms: int | tuple[int, int]) -> str:
    if isinstance(num_atoms, int):
        return str(num_atoms)
    return f"{num_atoms[0]}-{num_atoms[1]}"


def _format_cache_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _config_digest(config: SynOMolTransferConfig) -> str:
    payload = asdict(config)
    payload.pop("length", None)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _resolve_fixed_config_name(root: Path, config_name: str) -> str:
    direct_path = root / config_name
    if direct_path.exists():
        return config_name
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return config_name
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return config_name
    aliases = manifest.get("aliases", {})
    resolved = aliases.get(config_name)
    if isinstance(resolved, str):
        return resolved
    return config_name


def _numeric_summary(values: list[float] | list[int]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {"min": 0.0, "q05": 0.0, "median": 0.0, "mean": 0.0, "q95": 0.0, "max": 0.0}
    return {
        "min": float(tensor.min().item()),
        "q05": float(torch.quantile(tensor, 0.05).item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "mean": float(tensor.mean().item()),
        "q95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def _count_strings(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _split_offset(split: str) -> int:
    normalized = _normalize_split(split)
    return _SPLIT_OFFSETS[normalized]


def _sample_seed(seed: int, split: str, epoch: int, index: int) -> int:
    modulus = 2**63 - 1
    value = (
        int(seed)
        + 1_000_003 * int(index)
        + 9_176_467 * int(epoch)
        + 104_729 * _split_offset(split)
    )
    return int(value % modulus)


def _system_id(split: str, epoch: int, index: int) -> int:
    return int(_sample_seed(17, split, epoch, index) % (2**31 - 1))


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _is_heldout_combo(triple: tuple[int, int, int] | torch.Tensor) -> bool:
    if isinstance(triple, torch.Tensor):
        center, left, right = [int(value) for value in triple.reshape(-1).tolist()]
    else:
        center, left, right = triple
    return (int(center), min(int(left), int(right)), max(int(left), int(right))) in _HELDOUT_COMBO_SET
