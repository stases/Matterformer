from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from matterformer.data.qm9 import build_pad_mask


SYNTHMOLFORCE_NUM_ATOM_TYPES = 10
SYNTHMOLFORCE_ATOM_PAD_TOKEN = SYNTHMOLFORCE_NUM_ATOM_TYPES
SYNTHMOLFORCE_LEVELS = ("v0", "v1", "v2", "v3")
SYNTHMOLFORCE_PAIR_MODES = ("complete", "cutoff")
SYNTHMOLFORCE_COMPONENT_NAMES = ("pair", "coord", "angle", "chiral")
SYNTHMOLFORCE_DATASET_INFO = {
    "name": "synthmolforce",
    "num_atom_types": SYNTHMOLFORCE_NUM_ATOM_TYPES,
    "atom_pad_token": SYNTHMOLFORCE_ATOM_PAD_TOKEN,
    "levels": SYNTHMOLFORCE_LEVELS,
    "pair_modes": SYNTHMOLFORCE_PAIR_MODES,
    "components": SYNTHMOLFORCE_COMPONENT_NAMES,
}

_SPLIT_OFFSETS = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}


@dataclass(frozen=True)
class SynthMolForceConfig:
    level: str = "v2"
    pair_mode: str = "cutoff"
    num_atoms: int | tuple[int, int] = 16
    length: int = 10_000
    seed: int = 0
    density: float = 0.12
    d_min: float = 0.25
    max_resample_attempts: int = 64
    cutoff: float = 3.0
    radial_width: float = 0.55
    pair_sigma: float = 1.1
    pair_tau: float = 0.45
    distance_eps: float = 1.0e-6
    pair_scale: float = 1.0
    coord_scale: float = 0.08
    angle_scale: float = 0.025
    chiral_scale: float = 0.02

    def __post_init__(self) -> None:
        level = self.level.lower()
        pair_mode = self.pair_mode.lower()
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "pair_mode", pair_mode)
        if level not in SYNTHMOLFORCE_LEVELS:
            raise ValueError(f"level must be one of {SYNTHMOLFORCE_LEVELS}, got {self.level!r}")
        if pair_mode not in SYNTHMOLFORCE_PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {SYNTHMOLFORCE_PAIR_MODES}, got {self.pair_mode!r}")
        _validate_num_atoms(self.num_atoms)
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.d_min < 0.0:
            raise ValueError("d_min must be non-negative")
        if self.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.radial_width <= 0.0 or self.pair_sigma <= 0.0 or self.pair_tau <= 0.0:
            raise ValueError("radial_width, pair_sigma, and pair_tau must be positive")
        if self.distance_eps <= 0.0:
            raise ValueError("distance_eps must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SynthMolForceConfig":
        values = dict(values)
        num_atoms = values.get("num_atoms", 16)
        if isinstance(num_atoms, list):
            values["num_atoms"] = tuple(int(item) for item in num_atoms)
        return cls(**values)

    def cache_name(self) -> str:
        if isinstance(self.num_atoms, int):
            atoms = f"n{self.num_atoms}"
        else:
            atoms = f"n{self.num_atoms[0]}-{self.num_atoms[1]}"
        density = _format_cache_float(self.density)
        d_min = _format_cache_float(self.d_min)
        cutoff = _format_cache_float(self.cutoff)
        return f"{self.level}_{self.pair_mode}_{atoms}_rho{density}_dmin{d_min}_cut{cutoff}_seed{self.seed}"


@dataclass
class SynthMolForceBatch:
    atom_types: torch.Tensor
    coords: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    component_energies: dict[str, torch.Tensor]
    pad_mask: torch.Tensor
    num_atoms: torch.Tensor
    indices: torch.Tensor
    levels: list[str] | None = None
    pair_modes: list[str] | None = None
    lattice: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "SynthMolForceBatch":
        return SynthMolForceBatch(
            atom_types=self.atom_types.to(device),
            coords=self.coords.to(device),
            forces=self.forces.to(device),
            energy=self.energy.to(device),
            component_energies={key: value.to(device) for key, value in self.component_energies.items()},
            pad_mask=self.pad_mask.to(device),
            num_atoms=self.num_atoms.to(device),
            indices=self.indices.to(device),
            levels=self.levels,
            pair_modes=self.pair_modes,
            lattice=None if self.lattice is None else self.lattice.to(device),
        )

    def atom_onehot(self, num_classes: int = SYNTHMOLFORCE_NUM_ATOM_TYPES) -> torch.Tensor:
        atom_types = self.atom_types.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(atom_types, num_classes=num_classes).float()
        return one_hot.masked_fill(self.pad_mask[..., None], 0.0)

    def node_features(self) -> torch.Tensor:
        return self.atom_onehot()


def collate_synthmolforce(samples: list[dict[str, object]]) -> SynthMolForceBatch:
    if not samples:
        raise ValueError("samples must not be empty")

    num_atoms = torch.tensor([int(sample["num_atoms"]) for sample in samples], dtype=torch.long)
    max_atoms = int(num_atoms.max().item())
    batch_size = len(samples)

    atom_types = torch.full((batch_size, max_atoms), SYNTHMOLFORCE_ATOM_PAD_TOKEN, dtype=torch.long)
    coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    forces = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    energy = torch.zeros(batch_size, dtype=torch.float32)
    component_energies = {
        name: torch.zeros(batch_size, dtype=torch.float32)
        for name in SYNTHMOLFORCE_COMPONENT_NAMES
    }
    indices = torch.zeros(batch_size, dtype=torch.long)
    levels: list[str] = []
    pair_modes: list[str] = []

    for batch_idx, sample in enumerate(samples):
        count = int(sample["num_atoms"])
        atom_types[batch_idx, :count] = torch.as_tensor(sample["atom_types"], dtype=torch.long)
        coords[batch_idx, :count] = torch.as_tensor(sample["coords"], dtype=torch.float32)
        forces[batch_idx, :count] = torch.as_tensor(sample["forces"], dtype=torch.float32)
        energy[batch_idx] = torch.as_tensor(sample["energy"], dtype=torch.float32).reshape(())
        sample_components = sample.get("component_energies", {})
        for name in SYNTHMOLFORCE_COMPONENT_NAMES:
            if isinstance(sample_components, dict) and name in sample_components:
                component_energies[name][batch_idx] = torch.as_tensor(
                    sample_components[name],
                    dtype=torch.float32,
                ).reshape(())
        indices[batch_idx] = int(sample["idx"])
        levels.append(str(sample.get("level", "")))
        pair_modes.append(str(sample.get("pair_mode", "")))

    pad_mask = build_pad_mask(num_atoms, max_atoms=max_atoms)
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    forces = forces.masked_fill(pad_mask[..., None], 0.0)
    return SynthMolForceBatch(
        atom_types=atom_types,
        coords=coords,
        forces=forces,
        energy=energy,
        component_energies=component_energies,
        pad_mask=pad_mask,
        num_atoms=num_atoms,
        indices=indices,
        levels=levels,
        pair_modes=pair_modes,
    )


class SynthMolForceDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "./data/synthmolforce",
        *,
        split: str = "train",
        config: SynthMolForceConfig | None = None,
        mode: str = "online",
        config_name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.mode = mode.lower()
        self.epoch = 0
        if self.mode not in {"online", "fixed"}:
            raise ValueError("mode must be one of {'online', 'fixed'}")

        self.config = config or SynthMolForceConfig()
        self.config_name = config_name or self.config.cache_name()
        self._fixed: dict[str, Any] | None = None
        if self.mode == "fixed":
            fixed_path = self.root / self.config_name / f"{self.split}.pt"
            if not fixed_path.is_file():
                raise FileNotFoundError(f"SynthMolForce fixed split not found: {fixed_path}")
            loaded = torch.load(fixed_path, map_location="cpu", weights_only=False)
            if "config" in loaded:
                self.config = SynthMolForceConfig.from_dict(loaded["config"])
            self._fixed = _normalize_fixed_split(loaded)

    @property
    def dataset_info(self) -> dict[str, object]:
        return SYNTHMOLFORCE_DATASET_INFO

    @property
    def num_atoms(self) -> torch.Tensor:
        if self._fixed is not None:
            return self._fixed["num_atoms"].clone()
        if isinstance(self.config.num_atoms, int):
            return torch.full((len(self),), int(self.config.num_atoms), dtype=torch.long)
        values = [
            _sample_num_atoms(self.config, _make_generator(_sample_seed(self.config.seed, self.split, self.epoch, index)))
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
        return generate_synthmolforce_sample(
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
            for name in SYNTHMOLFORCE_COMPONENT_NAMES
        }
        return {
            "atom_types": self._fixed["atom_types"][start:end].clone(),
            "coords": self._fixed["coords"][start:end].clone(),
            "forces": self._fixed["forces"][start:end].clone(),
            "energy": self._fixed["energy"][index].clone(),
            "component_energies": components,
            "num_atoms": int(self._fixed["num_atoms"][index].item()),
            "idx": int(self._fixed["indices"][index].item()),
            "level": str(self._fixed["levels"][index]),
            "pair_mode": str(self._fixed["pair_modes"][index]),
        }


def generate_synthmolforce_sample(
    *,
    index: int,
    split: str,
    epoch: int,
    config: SynthMolForceConfig,
) -> dict[str, object]:
    atom_types, coords, num_atoms = _generate_synthmolforce_inputs(
        index=index,
        split=split,
        epoch=epoch,
        config=config,
    )
    energy, forces, components = compute_synthmolforce_labels(atom_types, coords, config)
    return {
        "atom_types": atom_types,
        "coords": coords,
        "forces": forces,
        "energy": energy,
        "component_energies": components,
        "num_atoms": num_atoms,
        "idx": int(index),
        "level": config.level,
        "pair_mode": config.pair_mode,
    }


def generate_synthmolforce_batch(
    indices: torch.Tensor | list[int] | tuple[int, ...],
    *,
    split: str,
    epoch: int,
    config: SynthMolForceConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SynthMolForceBatch:
    if isinstance(indices, torch.Tensor):
        index_list = [int(index) for index in indices.detach().cpu().reshape(-1).tolist()]
    else:
        index_list = [int(index) for index in indices]
    if not index_list:
        raise ValueError("indices must not be empty")

    inputs = [
        _generate_synthmolforce_inputs(
            index=index,
            split=split,
            epoch=epoch,
            config=config,
        )
        for index in index_list
    ]
    num_atoms_cpu = torch.tensor([num_atoms for _, _, num_atoms in inputs], dtype=torch.long)
    max_atoms = int(num_atoms_cpu.max().item())
    batch_size = len(inputs)

    atom_types_cpu = torch.full((batch_size, max_atoms), SYNTHMOLFORCE_ATOM_PAD_TOKEN, dtype=torch.long)
    coords_cpu = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    for batch_idx, (atom_types, coords, num_atoms) in enumerate(inputs):
        atom_types_cpu[batch_idx, :num_atoms] = atom_types
        coords_cpu[batch_idx, :num_atoms] = coords

    device = torch.device(device)
    atom_types = atom_types_cpu.to(device=device)
    coords = coords_cpu.to(device=device, dtype=dtype)
    num_atoms = num_atoms_cpu.to(device=device)
    pad_mask = build_pad_mask(num_atoms, max_atoms=max_atoms)
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    energy, forces, component_energies = compute_synthmolforce_labels_batch(
        atom_types,
        coords,
        config,
        pad_mask=pad_mask,
    )
    return SynthMolForceBatch(
        atom_types=atom_types,
        coords=coords,
        forces=forces,
        energy=energy,
        component_energies=component_energies,
        pad_mask=pad_mask,
        num_atoms=num_atoms,
        indices=torch.tensor(index_list, device=device, dtype=torch.long),
        levels=[config.level for _ in index_list],
        pair_modes=[config.pair_mode for _ in index_list],
    )


def compute_synthmolforce_labels(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynthMolForceConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    coords_for_grad = coords.detach().clone().requires_grad_(True)
    energy, components = compute_synthmolforce_energy(atom_types, coords_for_grad, config)
    (grad_coords,) = torch.autograd.grad(energy, coords_for_grad, create_graph=False)
    forces = -grad_coords
    detached_components = {
        name: components[name].detach()
        for name in SYNTHMOLFORCE_COMPONENT_NAMES
    }
    return energy.detach(), forces.detach(), detached_components


def compute_synthmolforce_labels_batch(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynthMolForceConfig,
    *,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    coords_for_grad = coords.detach().clone().requires_grad_(True)
    energy, components = compute_synthmolforce_energy_batch(
        atom_types,
        coords_for_grad,
        config,
        pad_mask=pad_mask,
    )
    (grad_coords,) = torch.autograd.grad(energy.sum(), coords_for_grad, create_graph=False)
    forces = -grad_coords
    if pad_mask is not None:
        forces = forces.masked_fill(pad_mask.to(device=forces.device)[..., None], 0.0)
    detached_components = {
        name: components[name].detach()
        for name in SYNTHMOLFORCE_COMPONENT_NAMES
    }
    return energy.detach(), forces.detach(), detached_components


def compute_synthmolforce_energy(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynthMolForceConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if atom_types.ndim != 1:
        raise ValueError(f"atom_types must have shape (N,), got {tuple(atom_types.shape)}")
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {tuple(coords.shape)}")
    if coords.shape[0] != atom_types.shape[0]:
        raise ValueError("atom_types and coords must describe the same number of atoms")
    if atom_types.numel() == 0:
        raise ValueError("SynthMolForce samples must contain at least one atom")
    if int(atom_types.min().item()) < 0 or int(atom_types.max().item()) >= SYNTHMOLFORCE_NUM_ATOM_TYPES:
        raise ValueError(f"atom_types must be in [0, {SYNTHMOLFORCE_NUM_ATOM_TYPES})")

    atom_types = atom_types.to(device=coords.device, dtype=torch.long)
    coords = coords.to(dtype=coords.dtype)
    dtype = coords.dtype
    device = coords.device
    num_atoms = int(coords.shape[0])
    denom_atoms = coords.new_tensor(float(max(num_atoms, 1)))

    params = _potential_parameters(device=device, dtype=dtype)
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = torch.sqrt((deltas * deltas).sum(dim=-1) + config.distance_eps**2)
    upper_mask = torch.triu(torch.ones(num_atoms, num_atoms, device=device, dtype=torch.bool), diagonal=1)
    not_self = ~torch.eye(num_atoms, device=device, dtype=torch.bool)
    zi = atom_types[:, None]
    zj = atom_types[None, :]

    a_pair = params["pair_a"][zi, zj]
    b_pair = params["pair_b"][zi, zj]
    c_pair = params["pair_c"][zi, zj]
    r0 = params["r0"][zi, zj]
    pair_terms = (
        a_pair * torch.exp(-distances / config.pair_sigma)
        + b_pair * torch.exp(-((distances - r0) ** 2) / (config.pair_tau**2))
        + c_pair / torch.sqrt(distances * distances + config.d_min**2)
    )
    pair_weight = torch.ones_like(pair_terms)
    if config.pair_mode == "cutoff":
        pair_weight = _smooth_cutoff(distances, config.cutoff)
        pair_denominator = denom_atoms
    else:
        pair_denominator = coords.new_tensor(float(max(num_atoms * (num_atoms - 1) // 2, 1)))
    pair_energy = config.pair_scale * (pair_terms * pair_weight).masked_select(upper_mask).sum() / pair_denominator

    zero = coords.new_tensor(0.0)
    coord_energy = zero
    angle_energy = zero
    chiral_energy = zero
    level_index = SYNTHMOLFORCE_LEVELS.index(config.level)

    if level_index >= 1:
        neighbor_weight = _neighbor_weights(distances, atom_types, params, config)
        s_i = neighbor_weight.sum(dim=1)
        valence = params["valence"][atom_types]
        coord_lambda = params["coord_lambda"][atom_types]
        coord_energy = config.coord_scale * (coord_lambda * (s_i - valence).pow(2)).sum() / denom_atoms

        if level_index >= 2:
            directions = (coords[None, :, :] - coords[:, None, :]) / distances.clamp_min(config.distance_eps)[..., None]
            directions = directions * not_self.to(dtype=dtype)[..., None]
            m_i = torch.einsum("ij,ija->ia", neighbor_weight, directions)
            q_i = torch.einsum("ij,ija,ijb->iab", neighbor_weight, directions, directions)
            q_norm_sq = (q_i * q_i).sum(dim=(-2, -1))
            m_norm_sq = (m_i * m_i).sum(dim=-1)
            c_angle = params["angle_cos"][atom_types]
            angle_lambda = params["angle_lambda"][atom_types]
            angle_raw = q_norm_sq - 2.0 * c_angle * m_norm_sq + c_angle.pow(2) * s_i.pow(2)
            angle_energy = config.angle_scale * (angle_lambda * angle_raw).sum() / denom_atoms

            if level_index >= 3:
                type_a = params["chiral_a"][atom_types][None, :]
                type_b = params["chiral_b"][atom_types][None, :]
                type_c = params["chiral_c"][atom_types][None, :]
                a_i = torch.einsum("ij,ij,ija->ia", neighbor_weight, type_a.expand_as(neighbor_weight), directions)
                b_i = torch.einsum("ij,ij,ija->ia", neighbor_weight, type_b.expand_as(neighbor_weight), directions)
                c_i = torch.einsum("ij,ij,ija->ia", neighbor_weight, type_c.expand_as(neighbor_weight), directions)
                chi = (a_i * torch.cross(b_i, c_i, dim=-1)).sum(dim=-1)
                chi0 = params["chi0"][atom_types]
                chiral_lambda = params["chiral_lambda"][atom_types]
                chiral_energy = config.chiral_scale * (chiral_lambda * (chi - chi0).pow(2)).sum() / denom_atoms

    components = {
        "pair": pair_energy,
        "coord": coord_energy,
        "angle": angle_energy,
        "chiral": chiral_energy,
    }
    total = pair_energy + coord_energy + angle_energy + chiral_energy
    return total, components


def compute_synthmolforce_energy_batch(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    config: SynthMolForceConfig,
    *,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if atom_types.ndim != 2:
        raise ValueError(f"atom_types must have shape (B, N), got {tuple(atom_types.shape)}")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (B, N, 3), got {tuple(coords.shape)}")
    if coords.shape[:2] != atom_types.shape:
        raise ValueError("atom_types and coords must describe the same batch and token dimensions")

    device = coords.device
    dtype = coords.dtype
    atom_types = atom_types.to(device=device, dtype=torch.long)
    if pad_mask is None:
        pad_mask = torch.zeros(atom_types.shape, device=device, dtype=torch.bool)
    else:
        if pad_mask.shape != atom_types.shape:
            raise ValueError(f"pad_mask must have shape {tuple(atom_types.shape)}, got {tuple(pad_mask.shape)}")
        pad_mask = pad_mask.to(device=device, dtype=torch.bool)
    valid_atoms = ~pad_mask
    if (valid_atoms.sum(dim=1) <= 0).any():
        raise ValueError("each SynthMolForce batch item must contain at least one atom")

    atom_types_index = atom_types.masked_fill(pad_mask, 0)
    valid_type_values = atom_types[valid_atoms]
    if valid_type_values.numel() > 0:
        if int(valid_type_values.min().item()) < 0 or int(valid_type_values.max().item()) >= SYNTHMOLFORCE_NUM_ATOM_TYPES:
            raise ValueError(f"atom_types must be in [0, {SYNTHMOLFORCE_NUM_ATOM_TYPES})")

    batch_size, max_atoms = atom_types.shape
    num_atoms = valid_atoms.sum(dim=1).to(dtype=dtype)
    denom_atoms = num_atoms.clamp_min(1.0)
    params = _potential_parameters(device=device, dtype=dtype)

    deltas = coords[:, :, None, :] - coords[:, None, :, :]
    distances = torch.sqrt((deltas * deltas).sum(dim=-1) + config.distance_eps**2)
    pair_valid = valid_atoms[:, :, None] & valid_atoms[:, None, :]
    upper_mask = torch.triu(torch.ones(max_atoms, max_atoms, device=device, dtype=torch.bool), diagonal=1)
    not_self = ~torch.eye(max_atoms, device=device, dtype=torch.bool)

    zi = atom_types_index[:, :, None]
    zj = atom_types_index[:, None, :]
    a_pair = params["pair_a"][zi, zj]
    b_pair = params["pair_b"][zi, zj]
    c_pair = params["pair_c"][zi, zj]
    r0 = params["r0"][zi, zj]
    pair_terms = (
        a_pair * torch.exp(-distances / config.pair_sigma)
        + b_pair * torch.exp(-((distances - r0) ** 2) / (config.pair_tau**2))
        + c_pair / torch.sqrt(distances * distances + config.d_min**2)
    )
    if config.pair_mode == "cutoff":
        pair_weight = _smooth_cutoff(distances, config.cutoff)
        pair_denominator = denom_atoms
    else:
        pair_weight = torch.ones_like(pair_terms)
        pair_denominator = (num_atoms * (num_atoms - 1.0) / 2.0).clamp_min(1.0)
    pair_mask = pair_valid & upper_mask[None, :, :]
    pair_energy = config.pair_scale * (pair_terms * pair_weight * pair_mask.to(dtype=dtype)).sum(dim=(1, 2)) / pair_denominator

    zero = coords.new_zeros(batch_size)
    coord_energy = zero
    angle_energy = zero
    chiral_energy = zero
    level_index = SYNTHMOLFORCE_LEVELS.index(config.level)

    if level_index >= 1:
        neighbor_weight = _neighbor_weights_batch(distances, atom_types_index, valid_atoms, params, config)
        s_i = neighbor_weight.sum(dim=2)
        valence = params["valence"][atom_types_index]
        coord_lambda = params["coord_lambda"][atom_types_index]
        coord_raw = coord_lambda * (s_i - valence).pow(2)
        coord_energy = config.coord_scale * (coord_raw * valid_atoms.to(dtype=dtype)).sum(dim=1) / denom_atoms

        if level_index >= 2:
            directions = (coords[:, None, :, :] - coords[:, :, None, :]) / distances.clamp_min(config.distance_eps)[..., None]
            directions = directions * (pair_valid & not_self[None, :, :]).to(dtype=dtype)[..., None]
            m_i = torch.einsum("bij,bija->bia", neighbor_weight, directions)
            q_i = torch.einsum("bij,bija,bijc->biac", neighbor_weight, directions, directions)
            q_norm_sq = (q_i * q_i).sum(dim=(-2, -1))
            m_norm_sq = (m_i * m_i).sum(dim=-1)
            c_angle = params["angle_cos"][atom_types_index]
            angle_lambda = params["angle_lambda"][atom_types_index]
            angle_raw = q_norm_sq - 2.0 * c_angle * m_norm_sq + c_angle.pow(2) * s_i.pow(2)
            angle_energy = (
                config.angle_scale
                * (angle_lambda * angle_raw * valid_atoms.to(dtype=dtype)).sum(dim=1)
                / denom_atoms
            )

            if level_index >= 3:
                type_a = params["chiral_a"][atom_types_index][:, None, :]
                type_b = params["chiral_b"][atom_types_index][:, None, :]
                type_c = params["chiral_c"][atom_types_index][:, None, :]
                a_i = torch.einsum("bij,bij,bija->bia", neighbor_weight, type_a.expand_as(neighbor_weight), directions)
                b_i = torch.einsum("bij,bij,bija->bia", neighbor_weight, type_b.expand_as(neighbor_weight), directions)
                c_i = torch.einsum("bij,bij,bija->bia", neighbor_weight, type_c.expand_as(neighbor_weight), directions)
                chi = (a_i * torch.cross(b_i, c_i, dim=-1)).sum(dim=-1)
                chi0 = params["chi0"][atom_types_index]
                chiral_lambda = params["chiral_lambda"][atom_types_index]
                chiral_raw = chiral_lambda * (chi - chi0).pow(2)
                chiral_energy = config.chiral_scale * (chiral_raw * valid_atoms.to(dtype=dtype)).sum(dim=1) / denom_atoms

    components = {
        "pair": pair_energy,
        "coord": coord_energy,
        "angle": angle_energy,
        "chiral": chiral_energy,
    }
    total = pair_energy + coord_energy + angle_energy + chiral_energy
    return total, components


def pack_synthmolforce_samples(
    samples: list[dict[str, object]],
    *,
    config: SynthMolForceConfig,
    split: str,
) -> dict[str, object]:
    if not samples:
        raise ValueError("samples must not be empty")
    num_atoms = torch.tensor([int(sample["num_atoms"]) for sample in samples], dtype=torch.long)
    ptr = torch.zeros(len(samples) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(num_atoms, dim=0)
    return {
        "config": asdict(config),
        "split": split,
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
            for name in SYNTHMOLFORCE_COMPONENT_NAMES
        },
        "num_atoms": num_atoms,
        "ptr": ptr,
        "indices": torch.tensor([int(sample["idx"]) for sample in samples], dtype=torch.long),
        "levels": [str(sample.get("level", config.level)) for sample in samples],
        "pair_modes": [str(sample.get("pair_mode", config.pair_mode)) for sample in samples],
    }


def materialize_synthmolforce_split(
    output_path: str | Path,
    *,
    config: SynthMolForceConfig,
    split: str,
    size: int,
) -> dict[str, object]:
    if size <= 0:
        raise ValueError("size must be positive")
    split_config = SynthMolForceConfig.from_dict({**asdict(config), "length": int(size)})
    dataset = SynthMolForceDataset(
        Path(output_path).parent.parent,
        split=split,
        config=split_config,
        mode="online",
        config_name=split_config.cache_name(),
    )
    samples = [dataset[index] for index in range(size)]
    packed = pack_synthmolforce_samples(samples, config=split_config, split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output_path)
    return packed


def _normalize_fixed_split(loaded: dict[str, Any]) -> dict[str, Any]:
    required = {"atom_types", "coords", "forces", "energy", "component_energies", "num_atoms", "ptr"}
    missing = sorted(required - set(loaded))
    if missing:
        raise KeyError(f"SynthMolForce fixed split is missing keys: {missing}")
    component_energies = {
        name: torch.as_tensor(loaded["component_energies"].get(name, torch.zeros_like(loaded["energy"])), dtype=torch.float32)
        for name in SYNTHMOLFORCE_COMPONENT_NAMES
    }
    size = int(torch.as_tensor(loaded["energy"]).shape[0])
    return {
        "atom_types": torch.as_tensor(loaded["atom_types"], dtype=torch.long),
        "coords": torch.as_tensor(loaded["coords"], dtype=torch.float32),
        "forces": torch.as_tensor(loaded["forces"], dtype=torch.float32),
        "energy": torch.as_tensor(loaded["energy"], dtype=torch.float32),
        "component_energies": component_energies,
        "num_atoms": torch.as_tensor(loaded["num_atoms"], dtype=torch.long),
        "ptr": torch.as_tensor(loaded["ptr"], dtype=torch.long),
        "indices": torch.as_tensor(loaded.get("indices", torch.arange(size)), dtype=torch.long),
        "levels": list(loaded.get("levels", [""] * size)),
        "pair_modes": list(loaded.get("pair_modes", [""] * size)),
    }


def _validate_num_atoms(num_atoms: int | tuple[int, int]) -> None:
    if isinstance(num_atoms, int):
        if num_atoms <= 0:
            raise ValueError("num_atoms must be positive")
        return
    if not isinstance(num_atoms, tuple) or len(num_atoms) != 2:
        raise ValueError("num_atoms must be an int or a (min, max) tuple")
    if int(num_atoms[0]) <= 0 or int(num_atoms[1]) <= 0:
        raise ValueError("num_atoms bounds must be positive")
    if int(num_atoms[0]) > int(num_atoms[1]):
        raise ValueError("num_atoms lower bound must be <= upper bound")


def _format_cache_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _split_offset(split: str) -> int:
    normalized = split.lower()
    if normalized in _SPLIT_OFFSETS:
        return _SPLIT_OFFSETS[normalized]
    return sum((idx + 1) * ord(char) for idx, char in enumerate(normalized))


def _sample_seed(seed: int, split: str, epoch: int, index: int) -> int:
    modulus = 2**63 - 1
    value = (
        int(seed)
        + 1_000_003 * int(index)
        + 9_176_467 * int(epoch)
        + 104_729 * _split_offset(split)
    )
    return int(value % modulus)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _sample_num_atoms(config: SynthMolForceConfig, generator: torch.Generator) -> int:
    if isinstance(config.num_atoms, int):
        return int(config.num_atoms)
    low, high = int(config.num_atoms[0]), int(config.num_atoms[1])
    return int(torch.randint(low=low, high=high + 1, size=(), generator=generator).item())


def _generate_synthmolforce_inputs(
    *,
    index: int,
    split: str,
    epoch: int,
    config: SynthMolForceConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    generator = _make_generator(_sample_seed(config.seed, split, epoch, index))
    num_atoms = _sample_num_atoms(config, generator)
    atom_types = torch.randint(
        low=0,
        high=SYNTHMOLFORCE_NUM_ATOM_TYPES,
        size=(num_atoms,),
        generator=generator,
        dtype=torch.long,
    )
    coords = _sample_coords(num_atoms, config, generator)
    return atom_types, coords, num_atoms


def _sample_coords(num_atoms: int, config: SynthMolForceConfig, generator: torch.Generator) -> torch.Tensor:
    box_length = float((num_atoms / config.density) ** (1.0 / 3.0))
    best_coords: torch.Tensor | None = None
    best_min_dist = -1.0
    for _ in range(config.max_resample_attempts):
        coords = (torch.rand(num_atoms, 3, generator=generator, dtype=torch.float32) - 0.5) * box_length
        coords = coords - coords.mean(dim=0, keepdim=True)
        min_dist = _min_pair_distance(coords)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_coords = coords
        if min_dist >= config.d_min:
            return coords
    if best_coords is None:
        raise RuntimeError("failed to sample coordinates")
    return best_coords


def _min_pair_distance(coords: torch.Tensor) -> float:
    if coords.shape[0] < 2:
        return float("inf")
    distances = torch.cdist(coords, coords)
    distances = distances.masked_fill(torch.eye(coords.shape[0], dtype=torch.bool), float("inf"))
    return float(distances.min().item())


def _potential_parameters(*, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    z = torch.arange(SYNTHMOLFORCE_NUM_ATOM_TYPES, device=device, dtype=dtype)
    zi = z[:, None]
    zj = z[None, :]
    zsum = zi + zj
    zdiff = torch.abs(zi - zj)
    zprod = zi * zj
    type_id = torch.arange(SYNTHMOLFORCE_NUM_ATOM_TYPES, device=device, dtype=dtype)
    parity = torch.where((type_id.remainder(2) == 0), torch.ones_like(type_id), -torch.ones_like(type_id))
    return {
        "pair_a": 0.04 + 0.015 * ((zsum.remainder(5.0) + 1.0) / 5.0),
        "pair_b": 0.03 + 0.012 * (((zprod + zsum).remainder(7.0) + 1.0) / 7.0),
        "pair_c": 0.012 + 0.008 * (((zdiff + 2.0 * zsum).remainder(6.0) + 1.0) / 6.0),
        "r0": 0.85 + 0.06 * zsum.remainder(6.0) + 0.035 * zdiff,
        "strength": 0.55 + 0.08 * ((zprod + zsum).remainder(5.0) + 1.0),
        "valence": torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0, 1.5, 2.5, 3.5, 4.5, 5.5], device=device, dtype=dtype),
        "coord_lambda": torch.tensor([0.9, 1.0, 1.1, 1.25, 0.85, 0.95, 1.05, 1.2, 1.1, 0.9], device=device, dtype=dtype),
        "angle_cos": torch.tensor([-1.0, -1.0, -0.5, -1.0 / 3.0, 0.0, -0.2, 0.2, -0.6, -1.0 / 3.0, 0.4], device=device, dtype=dtype),
        "angle_lambda": torch.tensor([0.0, 0.9, 1.0, 1.25, 0.8, 0.5, 0.7, 1.0, 1.2, 0.65], device=device, dtype=dtype),
        "chiral_a": torch.sin(0.8 * (type_id + 1.0)),
        "chiral_b": torch.cos(1.1 * (type_id + 1.0)),
        "chiral_c": torch.sin(1.7 * (type_id + 1.0) + 0.3),
        "chi0": 0.03 * parity * (0.5 + type_id / SYNTHMOLFORCE_NUM_ATOM_TYPES),
        "chiral_lambda": 0.8 + 0.1 * type_id.remainder(3.0),
    }


def _neighbor_weights(
    distances: torch.Tensor,
    atom_types: torch.Tensor,
    params: dict[str, torch.Tensor],
    config: SynthMolForceConfig,
) -> torch.Tensor:
    zi = atom_types[:, None]
    zj = atom_types[None, :]
    radial = torch.exp(-((distances - params["r0"][zi, zj]) ** 2) / (config.radial_width**2))
    cutoff = _smooth_cutoff(distances, config.cutoff)
    not_self = ~torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    return params["strength"][zi, zj] * radial * cutoff * not_self.to(dtype=distances.dtype)


def _neighbor_weights_batch(
    distances: torch.Tensor,
    atom_types: torch.Tensor,
    valid_atoms: torch.Tensor,
    params: dict[str, torch.Tensor],
    config: SynthMolForceConfig,
) -> torch.Tensor:
    zi = atom_types[:, :, None]
    zj = atom_types[:, None, :]
    radial = torch.exp(-((distances - params["r0"][zi, zj]) ** 2) / (config.radial_width**2))
    cutoff = _smooth_cutoff(distances, config.cutoff)
    not_self = ~torch.eye(distances.shape[1], device=distances.device, dtype=torch.bool)
    pair_valid = valid_atoms[:, :, None] & valid_atoms[:, None, :] & not_self[None, :, :]
    return params["strength"][zi, zj] * radial * cutoff * pair_valid.to(dtype=distances.dtype)


def _smooth_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    scaled = (distances / cutoff).clamp(min=0.0, max=1.0)
    values = 0.5 * (torch.cos(torch.pi * scaled) + 1.0)
    return torch.where(distances < cutoff, values, torch.zeros_like(values))
