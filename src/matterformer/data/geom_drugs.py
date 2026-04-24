from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, Sampler

from matterformer.data.qm9 import build_pad_mask


GEOM_DRUGS_ATOM_DECODER = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "As",
    "Br",
    "I",
    "Hg",
    "Bi",
]
GEOM_DRUGS_ATOM_ENCODER = {symbol: idx for idx, symbol in enumerate(GEOM_DRUGS_ATOM_DECODER)}
GEOM_DRUGS_NUM_ATOM_TYPES = len(GEOM_DRUGS_ATOM_DECODER)
GEOM_DRUGS_ATOM_PAD_TOKEN = GEOM_DRUGS_NUM_ATOM_TYPES

GEOM_DRUGS_CHARGE_VOCAB = [0, 1, 2, 3, -1, -2, -3]
GEOM_DRUGS_CHARGE_TO_INDEX = {charge: idx for idx, charge in enumerate(GEOM_DRUGS_CHARGE_VOCAB)}
GEOM_DRUGS_INDEX_TO_CHARGE = {idx: charge for idx, charge in enumerate(GEOM_DRUGS_CHARGE_VOCAB)}
GEOM_DRUGS_NUM_CHARGE_TYPES = len(GEOM_DRUGS_CHARGE_VOCAB)
GEOM_DRUGS_CHARGE_PAD_TOKEN = GEOM_DRUGS_NUM_CHARGE_TYPES

GEOM_DRUGS_RAW_FILENAMES = {
    "train": "train_data.pickle",
    "val": "val_data.pickle",
    "test": "test_data.pickle",
}
GEOM_DRUGS_RAW_URLS = {
    "train": "https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle",
    "val": "https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle",
    "test": "https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle",
}
GEOM_DRUGS_DATASET_INFO = {
    "name": "geom_drugs",
    "atom_encoder": GEOM_DRUGS_ATOM_ENCODER,
    "atom_decoder": GEOM_DRUGS_ATOM_DECODER,
    "charge_vocab": GEOM_DRUGS_CHARGE_VOCAB,
}


@dataclass
class GeomDrugsBatch:
    atom_types: torch.Tensor
    charges: torch.Tensor
    coords: torch.Tensor
    pad_mask: torch.Tensor
    num_atoms: torch.Tensor
    smiles: list[str] | None = None
    indices: torch.Tensor | None = None
    lattice: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "GeomDrugsBatch":
        return GeomDrugsBatch(
            atom_types=self.atom_types.to(device),
            charges=self.charges.to(device),
            coords=self.coords.to(device),
            pad_mask=self.pad_mask.to(device),
            num_atoms=self.num_atoms.to(device),
            smiles=self.smiles,
            indices=None if self.indices is None else self.indices.to(device),
            lattice=None if self.lattice is None else self.lattice.to(device),
        )

    def atom_onehot(self, num_classes: int = GEOM_DRUGS_NUM_ATOM_TYPES) -> torch.Tensor:
        atom_types = self.atom_types.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(atom_types, num_classes=num_classes).float()
        return one_hot.masked_fill(self.pad_mask[..., None], 0.0)

    def charge_indices(self) -> torch.Tensor:
        charge_indices = self.charges.clamp(min=0, max=GEOM_DRUGS_NUM_CHARGE_TYPES - 1)
        return charge_indices.masked_fill(self.pad_mask, GEOM_DRUGS_CHARGE_PAD_TOKEN)

    def charge_onehot(self, num_classes: int = GEOM_DRUGS_NUM_CHARGE_TYPES) -> torch.Tensor:
        charge_indices = self.charges.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(charge_indices, num_classes=num_classes).float()
        return one_hot.masked_fill(self.pad_mask[..., None], 0.0)

    def node_features(self) -> torch.Tensor:
        return torch.cat([self.atom_onehot(), self.charge_onehot()], dim=-1)


class GeomDrugsNumAtomsSampler(torch.nn.Module):
    def __init__(self, num_atoms: torch.Tensor | list[int]) -> None:
        super().__init__()
        num_atoms = torch.as_tensor(num_atoms, dtype=torch.long)
        if num_atoms.numel() == 0:
            raise ValueError("num_atoms must not be empty")
        hist = torch.bincount(num_atoms, minlength=int(num_atoms.max().item()) + 1).float()
        hist = hist[1:]
        self.register_buffer("probabilities", hist / hist.sum())

    def forward(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        sampled = torch.multinomial(self.probabilities, num_samples, replacement=True) + 1
        if device is not None:
            sampled = sampled.to(device)
        return sampled


class GeomDrugsPaddedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        num_atoms: torch.Tensor | list[int],
        *,
        max_examples_per_batch: int,
        max_padded_atoms_per_batch: int | None = None,
        max_attention_cost_per_batch: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 32,
    ) -> None:
        super().__init__()
        num_atoms = torch.as_tensor(num_atoms, dtype=torch.long)
        if num_atoms.numel() == 0:
            raise ValueError("num_atoms must not be empty")
        if max_examples_per_batch <= 0:
            raise ValueError("max_examples_per_batch must be positive")
        if max_padded_atoms_per_batch is not None and max_padded_atoms_per_batch <= 0:
            raise ValueError("max_padded_atoms_per_batch must be positive when set")
        if max_attention_cost_per_batch is not None and max_attention_cost_per_batch <= 0:
            raise ValueError("max_attention_cost_per_batch must be positive when set")
        if max_padded_atoms_per_batch is None and max_attention_cost_per_batch is None:
            raise ValueError("At least one batch budget must be set")
        self.num_atoms = num_atoms
        self.max_examples_per_batch = int(max_examples_per_batch)
        self.max_padded_atoms_per_batch = (
            None if max_padded_atoms_per_batch is None else int(max_padded_atoms_per_batch)
        )
        self.max_attention_cost_per_batch = (
            None if max_attention_cost_per_batch is None else int(max_attention_cost_per_batch)
        )
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size = max(self.max_examples_per_batch * int(bucket_size_multiplier), self.max_examples_per_batch)
        self._cached_length: int | None = None

    def _ordered_indices(self) -> list[int]:
        indices = list(range(self.num_atoms.numel()))
        if not self.shuffle:
            indices.sort(key=lambda idx: (int(self.num_atoms[idx]), idx))
            return indices

        random.shuffle(indices)
        ordered: list[int] = []
        for start in range(0, len(indices), self.bucket_size):
            chunk = indices[start : start + self.bucket_size]
            chunk.sort(key=lambda idx: (int(self.num_atoms[idx]), idx))
            ordered.extend(chunk)
        return ordered

    def _build_batches(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_max_atoms = 0

        for idx in indices:
            atom_count = int(self.num_atoms[idx])
            proposed_max_atoms = max(current_max_atoms, atom_count)
            proposed_batch_size = len(current) + 1
            exceeds_example_cap = proposed_batch_size > self.max_examples_per_batch
            exceeds_padded_atom_cap = (
                self.max_padded_atoms_per_batch is not None
                and proposed_max_atoms * proposed_batch_size > self.max_padded_atoms_per_batch
            )
            exceeds_attention_cap = (
                self.max_attention_cost_per_batch is not None
                and proposed_max_atoms * proposed_max_atoms * proposed_batch_size
                > self.max_attention_cost_per_batch
            )

            if current and (exceeds_example_cap or exceeds_padded_atom_cap or exceeds_attention_cap):
                if not self.drop_last or len(current) == self.max_examples_per_batch:
                    batches.append(current)
                current = [idx]
                current_max_atoms = atom_count
            else:
                current.append(idx)
                current_max_atoms = proposed_max_atoms

        if current and not self.drop_last:
            batches.append(current)
        return batches

    def __iter__(self):
        batches = self._build_batches(self._ordered_indices())
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self._cached_length is None:
            sorted_indices = list(range(self.num_atoms.numel()))
            sorted_indices.sort(key=lambda idx: (int(self.num_atoms[idx]), idx))
            self._cached_length = len(self._build_batches(sorted_indices))
        return self._cached_length


def collate_geom_drugs(samples: list[dict[str, object]]) -> GeomDrugsBatch:
    if not samples:
        raise ValueError("samples must not be empty")

    num_atoms = torch.tensor([int(sample["num_atoms"]) for sample in samples], dtype=torch.long)
    max_atoms = int(num_atoms.max().item())
    batch_size = len(samples)

    atom_types = torch.full((batch_size, max_atoms), GEOM_DRUGS_ATOM_PAD_TOKEN, dtype=torch.long)
    charges = torch.full((batch_size, max_atoms), GEOM_DRUGS_CHARGE_PAD_TOKEN, dtype=torch.long)
    coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    smiles: list[str] = []
    indices = torch.zeros(batch_size, dtype=torch.long)

    for batch_idx, sample in enumerate(samples):
        count = int(sample["num_atoms"])
        atom_tensor = torch.as_tensor(sample["atom_types"], dtype=torch.long)
        charge_tensor = torch.as_tensor(sample["charges"], dtype=torch.long)
        coord_tensor = torch.as_tensor(sample["coords"], dtype=torch.float32)
        atom_types[batch_idx, :count] = atom_tensor
        charges[batch_idx, :count] = charge_tensor
        coords[batch_idx, :count] = coord_tensor
        smiles.append(str(sample["smiles"]))
        indices[batch_idx] = int(sample["idx"])

    pad_mask = build_pad_mask(num_atoms, max_atoms=max_atoms)
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    return GeomDrugsBatch(
        atom_types=atom_types,
        charges=charges,
        coords=coords,
        pad_mask=pad_mask,
        num_atoms=num_atoms,
        smiles=smiles,
        indices=indices,
    )


def _default_processed_filename(split: str) -> str:
    split = split.lower()
    if split not in GEOM_DRUGS_RAW_FILENAMES:
        raise ValueError("split must be one of {'train', 'val', 'test'}")
    return f"{split}.pt"


class GeomDrugsDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "./data/geom_drugs",
        *,
        split: str = "train",
        processed_dir: str = "processed",
        processed_filename: str | None = None,
        download: bool = False,
    ) -> None:
        del download
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split.lower()
        self.processed_file = self.root / processed_dir / (processed_filename or _default_processed_filename(self.split))

        if not self.processed_file.is_file():
            raise FileNotFoundError(
                f"Processed GEOM-Drugs file not found: {self.processed_file}. "
                "Run scripts/prepare_geom_drugs_data.py first."
            )

        loaded = torch.load(self.processed_file, map_location="cpu", weights_only=False)
        self.coords = loaded["coords"].float()
        self.atom_types = loaded["atom_types"].long()
        self.charges = loaded["charges"].long()
        self.ptr = loaded["ptr"].long()
        self.smiles = [str(smiles) for smiles in loaded["smiles"]]
        if len(self.smiles) != (self.ptr.numel() - 1):
            raise ValueError("Processed GEOM-Drugs file has mismatched smiles and pointer lengths")

    @property
    def dataset_info(self) -> dict[str, object]:
        return GEOM_DRUGS_DATASET_INFO

    @property
    def smiles_list(self) -> list[str]:
        return list(self.smiles)

    def make_num_atoms_sampler(self) -> GeomDrugsNumAtomsSampler:
        return GeomDrugsNumAtomsSampler(self.num_atoms)

    def make_padded_batch_sampler(
        self,
        *,
        max_examples_per_batch: int,
        max_padded_atoms_per_batch: int | None = None,
        max_attention_cost_per_batch: int | None = None,
        shuffle: bool,
        drop_last: bool = False,
        bucket_size_multiplier: int = 32,
    ) -> GeomDrugsPaddedBatchSampler:
        return GeomDrugsPaddedBatchSampler(
            self.num_atoms,
            max_examples_per_batch=max_examples_per_batch,
            max_padded_atoms_per_batch=max_padded_atoms_per_batch,
            max_attention_cost_per_batch=max_attention_cost_per_batch,
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_size_multiplier=bucket_size_multiplier,
        )

    @property
    def num_atoms(self) -> torch.Tensor:
        return self.ptr[1:] - self.ptr[:-1]

    def __len__(self) -> int:
        return self.ptr.numel() - 1

    def __getitem__(self, index: int) -> dict[str, object]:
        start = int(self.ptr[index].item())
        end = int(self.ptr[index + 1].item())
        return {
            "atom_types": self.atom_types[start:end],
            "charges": self.charges[start:end],
            "coords": self.coords[start:end],
            "num_atoms": end - start,
            "smiles": self.smiles[index],
            "idx": index,
        }


def load_geom_drugs_reference_smiles(cache_path: str | Path) -> set[str]:
    with Path(cache_path).open("rb") as handle:
        return set(pickle.load(handle))
