from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

QM9_ATOM_ENCODER = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
QM9_ATOM_DECODER = ["H", "C", "N", "O", "F"]
QM9_NUM_ATOM_TYPES = len(QM9_ATOM_DECODER)
QM9_ATOM_PAD_TOKEN = QM9_NUM_ATOM_TYPES

QM9_TARGETS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]
QM9_DATASET_INFO = {
    "name": "qm9",
    "atom_encoder": QM9_ATOM_ENCODER,
    "atom_decoder": QM9_ATOM_DECODER,
}


@dataclass
class QM9Batch:
    atom_types: torch.Tensor
    coords: torch.Tensor
    pad_mask: torch.Tensor
    num_atoms: torch.Tensor
    targets: torch.Tensor | None = None
    smiles: list[str] | None = None
    indices: torch.Tensor | None = None
    target_name: str | None = None
    lattice: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "QM9Batch":
        return QM9Batch(
            atom_types=self.atom_types.to(device),
            coords=self.coords.to(device),
            pad_mask=self.pad_mask.to(device),
            num_atoms=self.num_atoms.to(device),
            targets=None if self.targets is None else self.targets.to(device),
            smiles=self.smiles,
            indices=None if self.indices is None else self.indices.to(device),
            target_name=self.target_name,
            lattice=None if self.lattice is None else self.lattice.to(device),
        )

    def atom_onehot(self, num_classes: int = QM9_NUM_ATOM_TYPES) -> torch.Tensor:
        atom_types = self.atom_types.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(atom_types, num_classes=num_classes).float()
        return one_hot.masked_fill(self.pad_mask[..., None], 0.0)


def build_pad_mask(num_atoms: torch.Tensor, max_atoms: int | None = None) -> torch.Tensor:
    if num_atoms.ndim != 1:
        raise ValueError(f"num_atoms must have shape (B,), got {tuple(num_atoms.shape)}")
    max_atoms = int(max_atoms or int(num_atoms.max().item()))
    token_ids = torch.arange(max_atoms, device=num_atoms.device)
    return token_ids[None, :] >= num_atoms[:, None]


class QM9NumAtomsSampler(torch.nn.Module):
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


def collate_qm9(samples: list[dict[str, object]]) -> QM9Batch:
    if not samples:
        raise ValueError("samples must not be empty")

    num_atoms = torch.tensor([int(sample["num_atoms"]) for sample in samples], dtype=torch.long)
    max_atoms = int(num_atoms.max().item())
    batch_size = len(samples)

    atom_types = torch.full((batch_size, max_atoms), QM9_ATOM_PAD_TOKEN, dtype=torch.long)
    coords = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    targets = None
    target_name = str(samples[0].get("target_name")) if samples[0].get("target_name") is not None else None
    if samples[0].get("targets") is not None:
        target_dim = int(torch.as_tensor(samples[0]["targets"]).numel())
        targets = torch.zeros(batch_size, target_dim, dtype=torch.float32)

    smiles: list[str] = []
    indices = torch.zeros(batch_size, dtype=torch.long)
    for batch_idx, sample in enumerate(samples):
        count = int(sample["num_atoms"])
        atom_tensor = torch.as_tensor(sample["atom_types"], dtype=torch.long)
        coord_tensor = torch.as_tensor(sample["coords"], dtype=torch.float32)
        atom_types[batch_idx, :count] = atom_tensor
        coords[batch_idx, :count] = coord_tensor
        if targets is not None:
            targets[batch_idx] = torch.as_tensor(sample["targets"], dtype=torch.float32).reshape(-1)
        smiles.append(str(sample["smiles"]))
        indices[batch_idx] = int(sample["idx"])

    pad_mask = build_pad_mask(num_atoms, max_atoms=max_atoms)
    coords = coords.masked_fill(pad_mask[..., None], 0.0)
    return QM9Batch(
        atom_types=atom_types,
        coords=coords,
        pad_mask=pad_mask,
        num_atoms=num_atoms,
        targets=targets,
        smiles=smiles,
        indices=indices,
        target_name=target_name,
    )


def compute_target_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for sample in dataset:
        target = sample.get("targets")
        if target is None:
            raise ValueError("Dataset samples do not contain targets")
        target_tensor = torch.as_tensor(target, dtype=torch.float32).reshape(-1)
        values.append(target_tensor)
    stacked = torch.stack(values, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp_min(1e-8)
    return mean, std


class QM9Dataset(Dataset):
    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414
    TOTAL_SIZE = 130_831
    TRAIN_SIZE = 110_000
    VAL_SIZE = 10_000
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE
    QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"

    def __init__(
        self,
        root: str | Path = "./data/qm9",
        *,
        split: str | None = None,
        target: str | None = None,
        sdf_file: str = "gdb9.sdf",
        csv_file: str = "gdb9.sdf.csv",
        processed_filename: str = "processed_qm9_data.pkl",
        download: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sdf_file = self.root / sdf_file
        self.csv_file = self.root / csv_file
        self.processed_file = self.root / processed_filename
        self.uncharacterized_file = self.root / "uncharacterized.txt"
        self.split = split
        self.target = target
        self.target_index = None if target is None else QM9_TARGETS.index(target)

        if self.processed_file.is_file():
            with self.processed_file.open("rb") as handle:
                loaded_data = pickle.load(handle)
        else:
            if not download:
                raise FileNotFoundError(f"Processed QM9 file not found: {self.processed_file}")
            self._download_uncharacterized()
            self._ensure_raw_data_downloaded()
            loaded_data = self._process_raw_data()
            with self.processed_file.open("wb") as handle:
                pickle.dump(loaded_data, handle)

        self._data = [self._normalize_entry(entry, index) for index, entry in enumerate(loaded_data)]

        if split is not None:
            self._data = self._apply_split(self._data, split)

    @property
    def dataset_info(self) -> dict[str, object]:
        return QM9_DATASET_INFO

    @property
    def smiles_list(self) -> list[str]:
        return [str(entry["smiles"]) for entry in self._data]

    def make_num_atoms_sampler(self) -> QM9NumAtomsSampler:
        return QM9NumAtomsSampler([int(entry["num_atoms"]) for entry in self._data])

    def _apply_split(self, data: list[dict[str, object]], split: str) -> list[dict[str, object]]:
        split = split.lower()
        random_state = np.random.RandomState(seed=42)
        perm = random_state.permutation(np.arange(len(data)))
        if len(data) == self.TOTAL_SIZE:
            train_end = self.TRAIN_SIZE
            val_end = self.TRAIN_SIZE + self.VAL_SIZE
        else:
            train_end = int(0.8 * len(data))
            val_end = int(0.9 * len(data))

        if split == "train":
            indices = perm[:train_end]
        elif split == "val":
            indices = perm[train_end:val_end]
        elif split == "test":
            indices = perm[val_end:]
        else:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        return [data[int(index)] for index in indices]

    def _normalize_entry(self, item: dict[str, object], fallback_index: int) -> dict[str, object]:
        if not isinstance(item, dict):
            raise TypeError(f"QM9 processed entries must be dictionaries, got {type(item)!r}")

        if {"atom_types", "coords", "targets"}.issubset(item):
            atom_types = np.asarray(item["atom_types"], dtype=np.int64).reshape(-1)
            coords = np.asarray(item["coords"], dtype=np.float32)
            targets = np.asarray(item["targets"], dtype=np.float32).reshape(-1)
        elif {"pos", "x", "y"}.issubset(item):
            coords = np.asarray(item["pos"], dtype=np.float32)
            node_features = np.asarray(item["x"])
            if node_features.ndim == 2:
                atom_types = node_features.argmax(axis=-1).astype(np.int64)
            else:
                atom_types = node_features.astype(np.int64).reshape(-1)
            targets = np.asarray(item["y"], dtype=np.float32).reshape(-1)
        else:
            raise KeyError(
                "Unsupported QM9 processed entry schema; expected either "
                "{'atom_types', 'coords', 'targets'} or {'pos', 'x', 'y'} keys"
            )

        num_atoms = int(item.get("num_atoms", atom_types.shape[0]))
        return {
            "atom_types": atom_types,
            "coords": coords,
            "targets": targets,
            "edge_index": np.asarray(item.get("edge_index", np.zeros((2, 0), dtype=np.int64)), dtype=np.int64),
            "edge_attr": np.asarray(item.get("edge_attr", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32),
            "smiles": str(item.get("smiles", "")),
            "name": str(item.get("name", "")),
            "idx": int(item.get("idx", fallback_index)),
            "num_atoms": num_atoms,
        }

    def _download_uncharacterized(self) -> None:
        if self.uncharacterized_file.is_file():
            return
        response = requests.get(self.UNCHARACTERIZED_URL, timeout=60)
        response.raise_for_status()
        self.uncharacterized_file.write_bytes(response.content)

    def _ensure_raw_data_downloaded(self) -> None:
        if self.sdf_file.is_file() and self.csv_file.is_file():
            return
        archive_path = self.root / "qm9.zip"
        with requests.get(self.QM9_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(self.root)

    def _read_uncharacterized_indices(self) -> set[int]:
        with self.uncharacterized_file.open("r") as handle:
            entries = [line for line in handle.read().splitlines()[9:-2] if line.strip()]
        return {int(line.split()[0]) - 1 for line in entries}

    def _process_raw_data(self) -> list[dict[str, object]]:
        try:
            import pandas as pd
            from rdkit import Chem
            from rdkit.Chem import rdchem
        except ImportError as exc:
            raise RuntimeError("QM9 processing requires pandas and rdkit to be installed") from exc

        supplier = Chem.SDMolSupplier(str(self.sdf_file), removeHs=False, sanitize=False)
        dataframe = pd.read_csv(self.csv_file)
        raw_targets = dataframe.iloc[:, 1:].values.astype(np.float32)
        reordered_targets = np.concatenate([raw_targets[:, 3:], raw_targets[:, :3]], axis=1)
        conversion_factors = np.array(
            [
                1.0,
                1.0,
                self.HAR2EV,
                self.HAR2EV,
                self.HAR2EV,
                1.0,
                self.HAR2EV,
                self.HAR2EV,
                self.HAR2EV,
                self.HAR2EV,
                self.HAR2EV,
                1.0,
                self.KCALMOL2EV,
                self.KCALMOL2EV,
                self.KCALMOL2EV,
                self.KCALMOL2EV,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        targets = reordered_targets * conversion_factors
        skip_indices = self._read_uncharacterized_indices()
        atomic_number_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        data_list: list[dict[str, object]] = []
        for index, molecule in enumerate(tqdm(supplier, desc="Processing QM9 molecules")):
            if molecule is None or index in skip_indices:
                continue
            num_atoms = molecule.GetNumAtoms()
            atom_types = np.zeros(num_atoms, dtype=np.int64)
            coords = np.array(
                [molecule.GetConformer().GetAtomPosition(i) for i in range(num_atoms)],
                dtype=np.float32,
            )

            for atom_index in range(num_atoms):
                atom = molecule.GetAtomWithIdx(atom_index)
                atom_types[atom_index] = atomic_number_to_index[atom.GetAtomicNum()]

            edge_indices: list[tuple[int, int]] = []
            edge_attrs: list[list[int]] = []
            for bond in molecule.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = [0, 0, 0, 0]
                if bond.GetBondType() == rdchem.BondType.SINGLE:
                    bond_type[0] = 1
                elif bond.GetBondType() == rdchem.BondType.DOUBLE:
                    bond_type[1] = 1
                elif bond.GetBondType() == rdchem.BondType.TRIPLE:
                    bond_type[2] = 1
                elif bond.GetBondType() == rdchem.BondType.AROMATIC:
                    bond_type[3] = 1
                edge_indices.extend([(start, end), (end, start)])
                edge_attrs.extend([bond_type, bond_type])

            if edge_indices:
                edge_index = np.array(edge_indices, dtype=np.int64).T
                edge_attr = np.array(edge_attrs, dtype=np.int64)
                sort_indices = np.lexsort((edge_index[0], edge_index[1]))
                edge_index = edge_index[:, sort_indices]
                edge_attr = edge_attr[sort_indices]
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_attr = np.zeros((0, 4), dtype=np.int64)

            data_list.append(
                {
                    "atom_types": atom_types,
                    "coords": coords.astype(np.float32),
                    "targets": targets[index].astype(np.float32),
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "smiles": Chem.MolToSmiles(molecule),
                    "name": molecule.GetProp("_Name"),
                    "idx": index,
                    "num_atoms": num_atoms,
                }
            )
        return data_list

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self._data[index]
        targets = np.asarray(item["targets"], dtype=np.float32)
        if self.target_index is not None:
            targets = targets[self.target_index : self.target_index + 1]
        return {
            "atom_types": torch.as_tensor(item["atom_types"], dtype=torch.long),
            "coords": torch.as_tensor(item["coords"], dtype=torch.float32),
            "targets": torch.as_tensor(targets, dtype=torch.float32),
            "edge_index": torch.as_tensor(item.get("edge_index", np.zeros((2, 0))), dtype=torch.long),
            "edge_attr": torch.as_tensor(item.get("edge_attr", np.zeros((0, 4))), dtype=torch.float32),
            "smiles": str(item["smiles"]),
            "name": str(item.get("name", "")),
            "idx": int(item["idx"]),
            "num_atoms": int(item["num_atoms"]),
            "target_name": self.target,
        }
