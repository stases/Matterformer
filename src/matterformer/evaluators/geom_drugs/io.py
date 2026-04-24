from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from matterformer.data.geom_drugs import (
    GEOM_DRUGS_ATOM_DECODER,
    GEOM_DRUGS_ATOM_ENCODER,
    GEOM_DRUGS_CHARGE_TO_INDEX,
    GEOM_DRUGS_DATASET_INFO,
    GEOM_DRUGS_INDEX_TO_CHARGE,
    GEOM_DRUGS_NUM_ATOM_TYPES,
    GEOM_DRUGS_NUM_CHARGE_TYPES,
)


ATOMIC_NUM_TO_INDEX = {
    1: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    13: 6,
    14: 7,
    15: 8,
    16: 9,
    17: 10,
    33: 11,
    35: 12,
    53: 13,
    80: 14,
    83: 15,
}


def to_cpu_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def normalize_positions(value) -> torch.Tensor:
    tensor = to_cpu_tensor(value).to(torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"Positions must have shape (N, 3), got {tuple(tensor.shape)}")
    return tensor


def normalize_atom_types(value) -> torch.Tensor:
    tensor = to_cpu_tensor(value)
    if tensor.ndim == 2:
        return tensor.argmax(dim=-1).to(torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"Unsupported atom type shape: {tuple(tensor.shape)}")

    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        ints = tensor.to(torch.long)
        unique = set(int(v) for v in ints.tolist())
        if unique.issubset(set(range(GEOM_DRUGS_NUM_ATOM_TYPES))):
            return ints
        if unique.issubset(set(ATOMIC_NUM_TO_INDEX.keys())):
            return torch.tensor([ATOMIC_NUM_TO_INDEX[int(v)] for v in ints.tolist()], dtype=torch.long)
        raise ValueError(f"Unsupported integer atom type encoding: {sorted(unique)[:10]}")

    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        rounded = tensor.round().to(torch.long)
        unique = set(int(v) for v in rounded.tolist())
        if unique.issubset(set(range(GEOM_DRUGS_NUM_ATOM_TYPES))):
            return rounded
        if unique.issubset(set(ATOMIC_NUM_TO_INDEX.keys())):
            return torch.tensor([ATOMIC_NUM_TO_INDEX[int(v)] for v in rounded.tolist()], dtype=torch.long)
        raise ValueError(f"Unsupported floating atom type encoding: {sorted(unique)[:10]}")

    values = value.tolist()
    if values and isinstance(values[0], str):
        return torch.tensor([GEOM_DRUGS_ATOM_ENCODER[symbol] for symbol in values], dtype=torch.long)
    raise ValueError("Unsupported atom type format")


def normalize_charges(value, n_atoms: int) -> torch.Tensor:
    tensor = to_cpu_tensor(value)
    if tensor.ndim == 2:
        indices = tensor.argmax(dim=-1).to(torch.long).view(-1)
        if indices.numel() != n_atoms:
            raise ValueError(f"Charges must have {n_atoms} entries, got {indices.numel()}")
        return torch.tensor([GEOM_DRUGS_INDEX_TO_CHARGE[int(idx)] for idx in indices.tolist()], dtype=torch.long)

    tensor = tensor.to(torch.long).view(-1)
    if tensor.numel() != n_atoms:
        raise ValueError(f"Charges must have {n_atoms} entries, got {tensor.numel()}")
    unique = set(int(v) for v in tensor.tolist())
    if any(v < 0 for v in unique):
        if not unique.issubset(set(GEOM_DRUGS_CHARGE_TO_INDEX.keys())):
            raise ValueError(f"Unsupported formal charge values: {sorted(unique)}")
        return tensor
    if unique.issubset({0, 1, 2, 3}):
        return tensor
    if unique.issubset(set(range(GEOM_DRUGS_NUM_CHARGE_TYPES))):
        return torch.tensor([GEOM_DRUGS_INDEX_TO_CHARGE[int(idx)] for idx in tensor.tolist()], dtype=torch.long)
    if unique.issubset(set(GEOM_DRUGS_CHARGE_TO_INDEX.keys())):
        return tensor
    raise ValueError(f"Unsupported charge encoding: {sorted(unique)}")


def extract_record_fields(record):
    if isinstance(record, dict):
        positions = (
            record.get("positions")
            or record.get("pos")
            or record.get("coords")
            or record.get("coordinates")
        )
        atom_types = (
            record.get("atom_types")
            or record.get("types")
            or record.get("atom_type")
            or record.get("x")
        )
        charges = (
            record.get("charges")
            or record.get("formal_charges")
            or record.get("charge")
        )
        if positions is None or atom_types is None or charges is None:
            raise ValueError(f"Record is missing required keys: {sorted(record.keys())}")
        return positions, atom_types, charges

    if isinstance(record, (list, tuple)) and len(record) >= 3:
        return record[0], record[1], record[2]

    raise ValueError(f"Unsupported record type: {type(record).__name__}")


def normalize_record(record):
    positions_raw, atom_types_raw, charges_raw = extract_record_fields(record)
    positions = normalize_positions(positions_raw)
    atom_types = normalize_atom_types(atom_types_raw)
    charges = normalize_charges(charges_raw, positions.shape[0])
    if atom_types.numel() != positions.shape[0]:
        raise ValueError("Positions and atom types have mismatched lengths")
    return positions, atom_types, charges


def unwrap_loaded_object(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("molecules", "samples", "generated", "data", "outputs"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        if {"positions", "atom_types", "charges"}.issubset(set(obj.keys())):
            return [obj]
        if {"pos", "x", "charges"}.issubset(set(obj.keys())):
            return [obj]
    raise ValueError("Could not find a list of generated molecules in the input file")


def load_generated_samples(path: str | Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    elif suffix == ".json":
        with path.open("r") as handle:
            obj = json.load(handle)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return [normalize_record(record) for record in unwrap_loaded_object(obj)]


def load_or_build_train_reference_smiles(
    *,
    data_root: str | Path = "./data/geom_drugs",
    cache_filename: str = "train_reference_smiles.pkl",
    processed_filename: str = "train.pt",
) -> set[str]:
    data_root = Path(data_root)
    cache_path = data_root / "cache" / cache_filename
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return set(pickle.load(handle))

    processed_path = data_root / "processed" / processed_filename
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Train reference cache not found: {cache_path}. "
            f"Processed train split also missing: {processed_path}."
        )

    packed = torch.load(processed_path, map_location="cpu", weights_only=False)
    smiles = set(str(smiles) for smiles in packed["smiles"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(sorted(smiles), handle)
    return smiles
