from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import requests
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import SanitizeFlags
from tqdm import tqdm

from matterformer.data.geom_drugs import (
    GEOM_DRUGS_ATOM_ENCODER,
    GEOM_DRUGS_CHARGE_TO_INDEX,
    GEOM_DRUGS_RAW_FILENAMES,
    GEOM_DRUGS_RAW_URLS,
)


covalent_radii = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}

_STREAMING_PICKLE_MEMO_PRESERVE_COUNT = 6


class _StreamingListSink(list):
    def __init__(self, owner, callback) -> None:
        super().__init__()
        self._owner = owner
        self._callback = callback

    def append(self, item) -> None:
        self._callback(item)
        self._prune_owner_memo()

    def extend(self, items) -> None:
        for item in items:
            self.append(item)

    def _prune_owner_memo(self) -> None:
        memo = self._owner.memo
        preserve_count = _STREAMING_PICKLE_MEMO_PRESERVE_COUNT
        if len(memo) <= preserve_count:
            return
        # The GEOM raw pickles reuse a tiny fixed memo prefix for the top-level list
        # and RDKit Mol globals. Everything else is per-entry garbage that can be dropped.
        preserved = {
            idx: memo[idx]
            for idx in range(preserve_count)
            if idx in memo
        }
        memo.clear()
        memo.update(preserved)


class _StreamingTopLevelListUnpickler(pickle._Unpickler):
    def __init__(self, file, callback) -> None:
        super().__init__(file)
        self._callback = callback
        self._used_top_level_list = False

    def load_empty_list(self):
        if not self._used_top_level_list and not self.stack and not self.metastack:
            self._used_top_level_list = True
            self.append(_StreamingListSink(self, self._callback))
        else:
            self.append([])


_StreamingTopLevelListUnpickler.dispatch = pickle._Unpickler.dispatch.copy()
_StreamingTopLevelListUnpickler.dispatch[pickle.EMPTY_LIST[0]] = (
    _StreamingTopLevelListUnpickler.load_empty_list
)


def download_file(url: str, destination: Path, *, force: bool = False, chunk_size: int = 1024 * 1024) -> None:
    if destination.exists() and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        )
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
        progress.close()


def ensure_raw_splits(raw_dir: str | Path, *, force: bool = False) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in GEOM_DRUGS_RAW_FILENAMES.items():
        destination = raw_dir / filename
        download_file(GEOM_DRUGS_RAW_URLS[split], destination, force=force)
        paths[split] = destination
    return paths


def get_reference_distance_matrix(adjacency_matrix, numbers):
    adjacency_mask = (adjacency_matrix > 0).astype(int)
    radii = np.array([covalent_radii.get(i, 1.5) for i in numbers])
    return (radii[:, np.newaxis] + radii[np.newaxis, :]) * adjacency_mask


def calculate_distance_map(coordinates):
    diff = coordinates[:, :, np.newaxis, :] - coordinates[:, np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def check_topology(adjacency_matrix, numbers, coordinates, tolerance: float = 0.4):
    adjacency_mask = (adjacency_matrix > 0).astype(int)
    ref_dist = get_reference_distance_matrix(adjacency_matrix, numbers)
    data_dist = calculate_distance_map(coordinates) * adjacency_mask
    diffs = np.abs(data_dist - ref_dist[np.newaxis, :, :]) <= (ref_dist[np.newaxis, :, :] * tolerance)
    return diffs.all(axis=(1, 2))


def process_molecule(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) > 1:
        return None
    return mol


def validate_topology(mol: Chem.Mol) -> bool:
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    conformers = mol.GetConformers()
    if not conformers:
        return False
    coordinates = np.array([conf.GetPositions() for conf in conformers])
    return check_topology(adjacency_matrix, numbers, coordinates).all()


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _stream_top_level_pickle(path: Path, callback) -> None:
    with path.open("rb") as handle:
        unpickler = _StreamingTopLevelListUnpickler(handle, callback)
        unpickler.load()


def _merge_valency_dicts(
    target: dict[str, dict[str, list[int]]],
    source: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, list[int]]]:
    for element, charge_map in source.items():
        element_target = target.setdefault(element, {})
        for charge, valences in charge_map.items():
            current = set(element_target.setdefault(charge, []))
            current.update(int(valence) for valence in valences)
            element_target[charge] = sorted(current)
    return target


def _collect_selected_molecule(mols) -> Chem.Mol | None:
    for mol in mols:
        sanitized_mol = process_molecule(mol)
        if sanitized_mol is None:
            continue
        if not validate_topology(sanitized_mol):
            continue
        return sanitized_mol
    return None


def _get_explicit_valence(atom) -> int:
    try:
        return int(atom.GetValence(Chem.ValenceType.EXPLICIT))
    except AttributeError:
        return int(atom.GetExplicitValence())


def build_processed_split_from_raw(raw_pickle: str | Path, output_path: str | Path) -> tuple[dict[str, int], dict[str, dict[str, list[int]]]]:
    raw_pickle = Path(raw_pickle)
    output_path = Path(output_path)

    valency_dict: dict[str, dict[str, list[int]]] = {}
    stats = {
        "initial_molecules": 0,
        "saved_molecules": 0,
        "removed_molecules": 0,
        "initial_conformers": 0,
        "saved_conformers": 0,
        "removed_conformers": 0,
    }

    coords_chunks: list[torch.Tensor] = []
    atom_types_chunks: list[torch.Tensor] = []
    charges_chunks: list[torch.Tensor] = []
    coords_buffer: list[torch.Tensor] = []
    atom_types_buffer: list[torch.Tensor] = []
    charges_buffer: list[torch.Tensor] = []
    smiles_list: list[str] = []
    ptr = [0]
    chunk_atom_budget = 250_000
    chunk_atom_count = 0

    def flush_buffers() -> None:
        nonlocal chunk_atom_count
        if not coords_buffer:
            return
        coords_chunks.append(torch.cat(coords_buffer, dim=0))
        atom_types_chunks.append(torch.cat(atom_types_buffer, dim=0))
        charges_chunks.append(torch.cat(charges_buffer, dim=0))
        coords_buffer.clear()
        atom_types_buffer.clear()
        charges_buffer.clear()
        chunk_atom_count = 0

    def handle_entry(entry) -> None:
        nonlocal chunk_atom_count
        smiles, mols = entry
        stats["initial_molecules"] += 1
        stats["initial_conformers"] += len(mols)

        if Chem.MolFromSmiles(smiles) is None:
            return

        selected_mol = _collect_selected_molecule(mols)
        if selected_mol is None:
            return

        for atom in selected_mol.GetAtoms():
            element = atom.GetSymbol()
            charge = str(atom.GetFormalCharge())
            valence = _get_explicit_valence(atom)
            valency_dict.setdefault(element, {}).setdefault(charge, [])
            if valence not in valency_dict[element][charge]:
                valency_dict[element][charge].append(valence)

        coords, atom_types, charges = _extract_processed_fields(selected_mol)
        normalized_smiles = mol_to_largest_fragment_smiles(selected_mol) or str(smiles)

        coords_buffer.append(coords)
        atom_types_buffer.append(atom_types)
        charges_buffer.append(charges)
        smiles_list.append(normalized_smiles)
        ptr.append(ptr[-1] + coords.shape[0])
        chunk_atom_count += int(coords.shape[0])

        stats["saved_molecules"] += 1
        stats["saved_conformers"] += 1

        if chunk_atom_count >= chunk_atom_budget:
            flush_buffers()

    _stream_top_level_pickle(raw_pickle, handle_entry)
    flush_buffers()

    stats["removed_molecules"] = stats["initial_molecules"] - stats["saved_molecules"]
    stats["removed_conformers"] = stats["initial_conformers"] - stats["saved_conformers"]
    stats["atoms"] = int(ptr[-1])
    if stats["saved_molecules"] > 0:
        stats["max_conformers_kept"] = 1
        stats["min_conformers_kept"] = 1

    packed = {
        "coords": torch.cat(coords_chunks, dim=0) if coords_chunks else torch.zeros((0, 3), dtype=torch.float32),
        "atom_types": torch.cat(atom_types_chunks, dim=0) if atom_types_chunks else torch.zeros((0,), dtype=torch.long),
        "charges": torch.cat(charges_chunks, dim=0) if charges_chunks else torch.zeros((0,), dtype=torch.long),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "smiles": smiles_list,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output_path)
    return stats, valency_dict


def process_geom_drugs(raw_dir: str | Path, cleaned_dir: str | Path, *, force: bool = False) -> dict[str, dict[str, int]]:
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    valency_dict: dict[str, dict[str, list[int]]] = {}
    split_stats: dict[str, dict[str, int]] = {}

    for split, filename in GEOM_DRUGS_RAW_FILENAMES.items():
        input_path = raw_dir / filename
        output_path = cleaned_dir / filename
        if output_path.exists() and not force:
            continue
        data = _load_pickle(input_path)
        initial_size = len(data)
        initial_conformer_count = 0
        conformer_count = 0

        cleaned_data = []
        for smiles, mols in data:
            reference_mol = Chem.MolFromSmiles(smiles)
            if reference_mol is None:
                continue

            initial_conformer_count += len(mols)
            selected_mol: Chem.Mol | None = None
            for mol in mols:
                sanitized_mol = process_molecule(mol)
                if sanitized_mol is None:
                    continue
                if not validate_topology(sanitized_mol):
                    continue
                selected_mol = sanitized_mol
                break

            if selected_mol is None:
                continue

            for atom in selected_mol.GetAtoms():
                element = atom.GetSymbol()
                charge = str(atom.GetFormalCharge())
                valence = _get_explicit_valence(atom)
                valency_dict.setdefault(element, {}).setdefault(charge, [])
                if valence not in valency_dict[element][charge]:
                    valency_dict[element][charge].append(valence)

            conformer_count += 1
            cleaned_data.append((smiles, [selected_mol]))

        _save_pickle(output_path, cleaned_data)
        split_stats[split] = {
            "initial_molecules": initial_size,
            "saved_molecules": len(cleaned_data),
            "removed_molecules": initial_size - len(cleaned_data),
            "initial_conformers": initial_conformer_count,
            "saved_conformers": conformer_count,
            "removed_conformers": initial_conformer_count - conformer_count,
        }
        if cleaned_data:
            split_stats[split]["max_conformers_kept"] = 1
            split_stats[split]["min_conformers_kept"] = 1

    valency_path = cleaned_dir / "valency_dict.json"
    with valency_path.open("w") as handle:
        json.dump(valency_dict, handle, indent=2, sort_keys=True)
    return split_stats


def mol_to_largest_fragment_smiles(mol: Chem.Mol) -> str | None:
    mol_copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol_copy)
    except Exception:
        return None
    fragments = Chem.GetMolFrags(mol_copy, asMols=True)
    if not fragments:
        return None
    largest = max(fragments, key=lambda frag: frag.GetNumAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(largest, canonical=True)
    except Exception:
        return None


def _extract_processed_fields(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conf = mol.GetConformer()
    coords = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    atom_types = []
    charges = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = int(atom.GetFormalCharge())
        if symbol not in GEOM_DRUGS_ATOM_ENCODER:
            raise ValueError(f"Unsupported GEOM-Drugs atom symbol in cleaned split: {symbol}")
        if charge not in GEOM_DRUGS_CHARGE_TO_INDEX:
            raise ValueError(f"Unsupported GEOM-Drugs formal charge in cleaned split: {charge}")
        atom_types.append(GEOM_DRUGS_ATOM_ENCODER[symbol])
        charges.append(GEOM_DRUGS_CHARGE_TO_INDEX[charge])
    return (
        coords,
        torch.tensor(atom_types, dtype=torch.long),
        torch.tensor(charges, dtype=torch.long),
    )


def build_processed_split(cleaned_pickle: str | Path, output_path: str | Path) -> dict[str, int]:
    cleaned_pickle = Path(cleaned_pickle)
    output_path = Path(output_path)
    data = _load_pickle(cleaned_pickle)

    coords_list: list[torch.Tensor] = []
    atom_types_list: list[torch.Tensor] = []
    charges_list: list[torch.Tensor] = []
    smiles_list: list[str] = []
    ptr = [0]

    for entry_smiles, mols in data:
        if not mols:
            continue
        mol = mols[0]
        coords, atom_types, charges = _extract_processed_fields(mol)
        smiles = mol_to_largest_fragment_smiles(mol) or str(entry_smiles)
        coords_list.append(coords)
        atom_types_list.append(atom_types)
        charges_list.append(charges)
        smiles_list.append(smiles)
        ptr.append(ptr[-1] + coords.shape[0])

    packed = {
        "coords": torch.cat(coords_list, dim=0) if coords_list else torch.zeros((0, 3), dtype=torch.float32),
        "atom_types": torch.cat(atom_types_list, dim=0) if atom_types_list else torch.zeros((0,), dtype=torch.long),
        "charges": torch.cat(charges_list, dim=0) if charges_list else torch.zeros((0,), dtype=torch.long),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "smiles": smiles_list,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output_path)
    return {
        "molecules": len(smiles_list),
        "atoms": int(ptr[-1]),
    }


def build_train_reference_smiles(processed_train_path: str | Path, cache_path: str | Path) -> set[str]:
    processed_train_path = Path(processed_train_path)
    cache_path = Path(cache_path)
    packed = torch.load(processed_train_path, map_location="cpu", weights_only=False)
    smiles = set(str(smiles) for smiles in packed["smiles"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(sorted(smiles), handle)
    return smiles


def prepare_geom_drugs_dataset(
    data_root: str | Path = "./data/geom_drugs",
    *,
    force_download: bool = False,
    force_clean: bool = False,
    force_processed: bool = False,
) -> dict[str, object]:
    data_root = Path(data_root)
    raw_dir = data_root / "raw"
    cleaned_dir = data_root / "cleaned"
    processed_dir = data_root / "processed"
    cache_dir = data_root / "cache"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    ensure_raw_splits(raw_dir, force=force_download)
    split_stats: dict[str, dict[str, int]] = {}
    processed_stats: dict[str, dict[str, int]] = {}
    merged_valency_dict: dict[str, dict[str, list[int]]] = {}
    valency_path = cleaned_dir / "valency_dict.json"

    for split, filename in GEOM_DRUGS_RAW_FILENAMES.items():
        processed_path = processed_dir / f"{split}.pt"
        split_stats_path = cleaned_dir / f"{split}_stats.json"
        if force_clean or force_processed or not processed_path.exists():
            split_stats[split], split_valency_dict = build_processed_split_from_raw(raw_dir / filename, processed_path)
            processed_stats[split] = {
                "molecules": split_stats[split]["saved_molecules"],
                "atoms": split_stats[split]["atoms"],
            }
            _merge_valency_dicts(merged_valency_dict, split_valency_dict)
            with split_stats_path.open("w") as handle:
                json.dump(split_stats[split], handle, indent=2, sort_keys=True)
        elif split_stats_path.exists():
            with split_stats_path.open("r") as handle:
                split_stats[split] = json.load(handle)

    if merged_valency_dict:
        with valency_path.open("w") as handle:
            json.dump(merged_valency_dict, handle, indent=2, sort_keys=True)
    elif valency_path.exists():
        with valency_path.open("r") as handle:
            merged_valency_dict = json.load(handle)

    train_reference = build_train_reference_smiles(
        processed_dir / "train.pt",
        cache_dir / "train_reference_smiles.pkl",
    )
    return {
        "data_root": str(data_root),
        "split_stats": split_stats,
        "processed_stats": processed_stats,
        "train_reference_smiles": len(train_reference),
    }
