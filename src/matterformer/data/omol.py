from __future__ import annotations

import bisect
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset


OMOL_MAX_ATOMIC_NUMBER = 118
OMOL_ATOM_PAD_TOKEN = 0


@dataclass
class OMolBatch:
    atomic_numbers: torch.Tensor
    coords: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    pad_mask: torch.Tensor
    num_atoms: torch.Tensor
    charge: torch.Tensor
    spin: torch.Tensor
    free_atom_mask: torch.Tensor
    indices: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "OMolBatch":
        return OMolBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            coords=self.coords.to(device),
            forces=self.forces.to(device),
            energy=self.energy.to(device),
            pad_mask=self.pad_mask.to(device),
            num_atoms=self.num_atoms.to(device),
            charge=self.charge.to(device),
            spin=self.spin.to(device),
            free_atom_mask=self.free_atom_mask.to(device),
            indices=None if self.indices is None else self.indices.to(device),
        )


def _get_value(sample: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(sample, dict) and name in sample:
            return sample[name]
        if hasattr(sample, name):
            return getattr(sample, name)
    return default


def _as_1d_value(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).view(-1)
    if tensor.numel() == 0:
        raise ValueError("Expected a scalar-like value, got an empty tensor")
    return tensor[:1]


def build_omol_pad_mask(num_atoms: torch.Tensor, max_atoms: int | None = None) -> torch.Tensor:
    if num_atoms.ndim != 1:
        raise ValueError(f"num_atoms must have shape (B,), got {tuple(num_atoms.shape)}")
    max_atoms = int(max_atoms or int(num_atoms.max().item()))
    ids = torch.arange(max_atoms, device=num_atoms.device)
    return ids[None, :] >= num_atoms[:, None]


def collate_omol(samples: list[Any]) -> OMolBatch:
    if not samples:
        raise ValueError("samples must not be empty")

    counts = []
    for sample in samples:
        atomic_numbers = _get_value(sample, "atomic_numbers", "z")
        if atomic_numbers is None:
            raise KeyError("OMol samples must contain atomic_numbers or z")
        counts.append(int(torch.as_tensor(atomic_numbers).numel()))
    num_atoms = torch.tensor(counts, dtype=torch.long)
    max_atoms = int(num_atoms.max().item())
    batch_size = len(samples)

    atomic_numbers = torch.full((batch_size, max_atoms), OMOL_ATOM_PAD_TOKEN, dtype=torch.long)
    coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    forces = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    energy = torch.zeros(batch_size, dtype=torch.float32)
    charge = torch.zeros(batch_size, dtype=torch.long)
    spin = torch.zeros(batch_size, dtype=torch.long)
    free_atom_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    indices = torch.zeros(batch_size, dtype=torch.long)
    have_indices = False

    for batch_idx, sample in enumerate(samples):
        count = int(num_atoms[batch_idx].item())
        atomic_tensor = torch.as_tensor(
            _get_value(sample, "atomic_numbers", "z"),
            dtype=torch.long,
        ).view(-1)
        pos_tensor = torch.as_tensor(
            _get_value(sample, "pos", "coords", "positions"),
            dtype=torch.float32,
        ).view(count, 3)
        force_tensor = torch.as_tensor(
            _get_value(sample, "forces", "force"),
            dtype=torch.float32,
        ).view(count, 3)
        energy_value = _get_value(sample, "energy", "y")
        if energy_value is None:
            raise KeyError("OMol samples must contain energy or y")

        atomic_numbers[batch_idx, :count] = atomic_tensor.clamp(min=1, max=OMOL_MAX_ATOMIC_NUMBER)
        coords[batch_idx, :count] = pos_tensor
        forces[batch_idx, :count] = force_tensor
        energy[batch_idx] = _as_1d_value(energy_value, dtype=torch.float32)[0]
        charge[batch_idx] = _as_1d_value(_get_value(sample, "charge", default=0), dtype=torch.long)[0]
        spin[batch_idx] = _as_1d_value(_get_value(sample, "spin", default=0), dtype=torch.long)[0]

        fixed = _get_value(sample, "fixed", default=None)
        free = _get_value(sample, "free_atom_mask", "free_atoms", default=None)
        if free is not None:
            free_tensor = torch.as_tensor(free, dtype=torch.bool).view(-1)
        elif fixed is not None:
            free_tensor = ~torch.as_tensor(fixed, dtype=torch.bool).view(-1)
        else:
            free_tensor = torch.ones(count, dtype=torch.bool)
        free_atom_mask[batch_idx, :count] = free_tensor[:count]

        idx_value = _get_value(sample, "idx", "index", "id", default=None)
        if idx_value is not None:
            indices[batch_idx] = int(_as_1d_value(idx_value, dtype=torch.long)[0].item())
            have_indices = True

    pad_mask = build_omol_pad_mask(num_atoms, max_atoms=max_atoms)
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    forces = forces.masked_fill(pad_mask[..., None], 0.0)
    free_atom_mask = free_atom_mask & ~pad_mask

    return OMolBatch(
        atomic_numbers=atomic_numbers,
        coords=coords,
        forces=forces,
        energy=energy,
        pad_mask=pad_mask,
        num_atoms=num_atoms,
        charge=charge,
        spin=spin,
        free_atom_mask=free_atom_mask,
        indices=indices if have_indices else None,
    )


class SyntheticOMolDataset(Dataset):
    """Tiny deterministic OMol-like dataset for smoke tests and CI."""

    def __init__(self, num_samples: int = 32, *, seed: int = 0, min_atoms: int = 2, max_atoms: int = 8) -> None:
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.min_atoms = int(min_atoms)
        self.max_atoms = int(max_atoms)
        if self.min_atoms <= 0 or self.max_atoms < self.min_atoms:
            raise ValueError("Expected 0 < min_atoms <= max_atoms")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(idx))
        count = int(torch.randint(self.min_atoms, self.max_atoms + 1, (1,), generator=generator).item())
        atomic_numbers = torch.randint(1, 10, (count,), generator=generator, dtype=torch.long)
        coords = torch.randn(count, 3, generator=generator)
        coords = coords - coords.mean(dim=0, keepdim=True)
        spring = coords.square().sum(dim=-1)
        atom_term = atomic_numbers.float() * 0.01
        energy = spring.sum() + atom_term.sum()
        forces = -2.0 * coords
        return {
            "atomic_numbers": atomic_numbers,
            "pos": coords.float(),
            "forces": forces.float(),
            "energy": energy.float().view(1),
            "charge": torch.tensor([0], dtype=torch.long),
            "spin": torch.tensor([0], dtype=torch.long),
            "idx": int(idx),
        }

    def get_num_atoms(self, idx: int) -> int:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(idx))
        return int(torch.randint(self.min_atoms, self.max_atoms + 1, (1,), generator=generator).item())


class FairChemOMolDataset(Dataset):
    """OMol ASE DB wrapper with graph-level charge/spin injection.

    OMol is distributed in FairChem-compatible ASE DB shards.  Reading the
    shards only needs ASE plus ase-db-backends; the full fairchem-core runtime is
    intentionally not required here.
    """

    def __init__(self, src: str | os.PathLike[str], *, keep_in_memory: bool = False) -> None:
        try:
            import ase.db
        except ModuleNotFoundError as exc:
            raise ImportError(
                "OMol AseDB loading requires ase and ase-db-backends. Install them in this environment, "
                "for example: pip install 'ase>=3.26.0' 'ase-db-backends>=0.10.0'"
            ) from exc

        self.src = str(src)
        self.keep_in_memory = bool(keep_in_memory)
        self._cache: dict[int, dict[str, Any]] | None = {} if self.keep_in_memory else None
        self._pid = os.getpid()
        self._connect = ase.db.connect
        self.paths = self._resolve_paths()
        if not self.paths:
            raise ValueError(f"No .aselmdb files found under OMol source: {self.src}")
        self._idlen_cumulative = self._load_shard_lengths()
        self.dbs: list[Any | None] = [None for _ in self.paths]
        self._natoms_cache = self._load_natoms_cache(Path(self.src), len(self))

    @staticmethod
    def _load_natoms_cache(src: Path, expected_len: int) -> np.ndarray | None:
        metadata_path = src / "metadata.npz"
        if not metadata_path.is_file():
            return None
        with np.load(metadata_path, allow_pickle=False) as metadata:
            if "natoms" not in metadata.files:
                return None
            natoms = np.asarray(metadata["natoms"], dtype=np.int64)
        return natoms if len(natoms) == expected_len else None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["dbs"] = [None for _ in self.paths]
        state["_connect"] = None
        state["_cache"] = None if self._cache is None else {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _resolve_paths(self) -> list[Path]:
        src = Path(self.src)
        if src.is_file():
            return [src]
        if src.is_dir():
            return sorted(path for path in src.iterdir() if path.name.endswith(".aselmdb"))
        return sorted(Path(path) for path in glob.glob(self.src))

    def _load_shard_lengths(self) -> list[int]:
        counts: list[int] = []
        for path in self.paths:
            db = self._connect(str(path), readonly=True, use_lock_file=False)
            try:
                counts.append(int(db.count()))
            finally:
                db.close()
        return np.cumsum(counts).tolist()

    def _ensure_process_local_state(self) -> None:
        current_pid = os.getpid()
        if self._pid != current_pid:
            for db in self.dbs:
                if db is not None:
                    try:
                        db.close = lambda: None
                    except Exception:
                        pass
            self.dbs = [None for _ in self.paths]
            self._connect = None
            self._pid = current_pid
        if self._connect is None:
            import ase.db

            self._connect = ase.db.connect

    def _db_for_index(self, db_idx: int):
        self._ensure_process_local_state()
        db = self.dbs[db_idx]
        if db is None:
            db = self._connect(str(self.paths[db_idx]), readonly=True, use_lock_file=False)
            self.dbs[db_idx] = db
        return db

    def _row_for_index(self, idx: int):
        db_idx = bisect.bisect(self._idlen_cumulative, int(idx))
        local_idx = int(idx) if db_idx == 0 else int(idx) - self._idlen_cumulative[db_idx - 1]
        return self._db_for_index(db_idx)._get_row(local_idx + 1)

    def __len__(self) -> int:
        return int(self._idlen_cumulative[-1])

    def close(self) -> None:
        for idx, db in enumerate(self.dbs):
            if db is not None:
                try:
                    db.close()
                finally:
                    self.dbs[idx] = None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None and int(idx) in self._cache:
            return self._cache[int(idx)]
        row = self._row_for_index(int(idx))
        atoms = row.toatoms()
        if isinstance(row.data, dict):
            atoms.info.update(row.data)
        sample = {
            "atomic_numbers": torch.as_tensor(atoms.numbers, dtype=torch.long),
            "pos": torch.as_tensor(atoms.positions, dtype=torch.float32),
            "forces": torch.as_tensor(row.forces, dtype=torch.float32),
            "energy": torch.tensor([float(row.energy)], dtype=torch.float32),
            "charge": torch.tensor([int(atoms.info.get("charge", 0))], dtype=torch.long),
            "spin": torch.tensor([int(atoms.info.get("spin", 0))], dtype=torch.long),
            "idx": int(idx),
        }
        if "fixed" in atoms.arrays:
            sample["fixed"] = torch.as_tensor(atoms.arrays["fixed"], dtype=torch.bool)
        if self._cache is not None:
            self._cache[int(idx)] = sample
        return sample

    def get_atoms(self, idx: int):
        row = self._row_for_index(int(idx))
        atoms = row.toatoms()
        if isinstance(row.data, dict):
            atoms.info.update(row.data)
        return atoms

    def get_num_atoms(self, idx: int) -> int:
        if self._natoms_cache is not None:
            return int(self._natoms_cache[int(idx)])
        return int(self._row_for_index(int(idx)).natoms)


class OMolDynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Dataset,
        *,
        max_batch_size: int,
        max_atoms: int | None = None,
        max_edges: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.dataset = dataset
        self.max_batch_size = int(max_batch_size)
        self.max_atoms = None if max_atoms is None else int(max_atoms)
        self.max_edges = None if max_edges is None else int(max_edges)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self._atom_count_cache: dict[int, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_dataset_index(self, idx: int) -> tuple[Dataset, int]:
        dataset: Dataset = self.dataset
        actual_idx = int(idx)
        while isinstance(dataset, Subset):
            actual_idx = int(dataset.indices[actual_idx])
            dataset = dataset.dataset
        return dataset, actual_idx

    def _get_num_atoms(self, idx: int) -> int:
        if idx in self._atom_count_cache:
            return self._atom_count_cache[idx]
        dataset, actual_idx = self._resolve_dataset_index(idx)
        if hasattr(dataset, "get_num_atoms"):
            count = int(dataset.get_num_atoms(actual_idx))  # type: ignore[attr-defined]
        else:
            sample = dataset[actual_idx]
            atomic_numbers = _get_value(sample, "atomic_numbers", "z")
            if atomic_numbers is None:
                raise TypeError("Dataset samples must expose atomic_numbers/z or dataset.get_num_atoms(idx)")
            count = int(torch.as_tensor(atomic_numbers).numel())
        self._atom_count_cache[idx] = count
        return count

    def __len__(self) -> int:
        if self.max_atoms is None:
            return max(1, math.ceil(len(self.dataset) / self.max_batch_size))
        avg_atoms = max(1, min(self.max_atoms, 64))
        return max(1, math.ceil(len(self.dataset) / max(1, self.max_atoms // avg_atoms)))

    def __iter__(self):
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))

        batch: list[int] = []
        atom_total = 0
        edge_total = 0
        for idx in indices:
            count = self._get_num_atoms(int(idx))
            edges = count * count
            should_flush = batch and (
                len(batch) >= self.max_batch_size
                or (self.max_atoms is not None and atom_total + count > self.max_atoms)
                or (self.max_edges is not None and edge_total + edges > self.max_edges)
            )
            if should_flush:
                yield batch
                batch = []
                atom_total = 0
                edge_total = 0

            if (self.max_atoms is not None and count > self.max_atoms) or (
                self.max_edges is not None and edges > self.max_edges
            ):
                if self.max_edges is None or edges <= self.max_edges:
                    yield [int(idx)]
                continue

            batch.append(int(idx))
            atom_total += count
            edge_total += edges

        if batch and (not self.drop_last or len(batch) == self.max_batch_size):
            yield batch


def split_train_val(dataset: Dataset, *, train_size: float | int, seed: int) -> tuple[Subset, Subset]:
    total = len(dataset)
    if total < 2:
        raise ValueError("Need at least two samples for train/val split")
    if isinstance(train_size, float) and train_size <= 1.0:
        train_count = int(math.floor(total * train_size))
    else:
        train_count = int(train_size)
    train_count = max(1, min(train_count, total - 1))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(total, generator=generator).tolist()
    return Subset(dataset, indices[:train_count]), Subset(dataset, indices[train_count:])


def apply_debug_subset(dataset: Dataset, subset: int | float | str | None) -> Dataset:
    if subset is None:
        return dataset
    if isinstance(subset, str):
        value: int | float = float(subset) if "." in subset else int(subset)
    else:
        value = subset
    if isinstance(value, float) and value <= 1.0:
        count = max(1, int(math.floor(len(dataset) * value)))
    else:
        count = int(value)
    count = max(1, min(count, len(dataset)))
    return Subset(dataset, list(range(count)))
