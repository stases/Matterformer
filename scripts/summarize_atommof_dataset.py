#!/usr/bin/env python3
"""Summarize AtomMOF-format datasets stored as pickled LMDB records."""

from __future__ import annotations

import argparse
import io
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np


BLOCK_TYPES = ("NODE", "LINKER", "SOLVENT", "ADSORBATE")
HISTOGRAM_BINS = (0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000)


@dataclass
class Block:
    block_type: str
    cart_coords: np.ndarray
    atomic_numbers: np.ndarray
    bond_indices: np.ndarray
    bond_types: np.ndarray
    formal_charges: np.ndarray
    chirality_tags: np.ndarray


@dataclass
class Structure:
    blocks: list[Block]
    cart_coords: np.ndarray
    frac_coords: np.ndarray
    lattice: np.ndarray
    cell: np.ndarray
    info: Any = None


class AtomMOFUnpickler(pickle.Unpickler):
    """Redirect AtomMOF pickle classes to local compatibility dataclasses."""

    def find_class(self, module: str, name: str):
        if module == "src.data.types" and name == "Block":
            return Block
        if module == "src.data.types" and name == "Structure":
            return Structure
        return super().find_class(module, name)


def loads_atommof(blob: bytes) -> Structure:
    return AtomMOFUnpickler(io.BytesIO(blob)).load()


ATOMIC_SYMBOLS = [
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def symbol_for_atomic_number(z: int) -> str:
    z = int(z)
    if 0 < z < len(ATOMIC_SYMBOLS):
        return ATOMIC_SYMBOLS[z]
    return f"Z{z}"


def round_float(value: float) -> float:
    return round(float(value), 4)


def summarize_numeric(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round_float(array.mean()),
        "std": round_float(array.std(ddof=0)),
        "median": round_float(np.median(array)),
        "p05": round_float(np.percentile(array, 5)),
        "p25": round_float(np.percentile(array, 25)),
        "p75": round_float(np.percentile(array, 75)),
        "p95": round_float(np.percentile(array, 95)),
    }


def histogram(values: list[int], bin_starts: tuple[int, ...] = HISTOGRAM_BINS) -> list[dict[str, float | int | str]]:
    if not values:
        return []

    counts = Counter()
    last = bin_starts[-1]
    for value in values:
        bucket = None
        for start, end in zip(bin_starts[:-1], bin_starts[1:]):
            if start <= value < end:
                bucket = f"{start}-{end - 1}"
                break
        if bucket is None:
            bucket = f"{last}+"
        counts[bucket] += 1

    total = len(values)
    ordered_labels = [f"{start}-{end - 1}" for start, end in zip(bin_starts[:-1], bin_starts[1:])]
    ordered_labels.append(f"{last}+")
    return [
        {
            "range": label,
            "count": int(counts[label]),
            "fraction": round_float(counts[label] / total),
        }
        for label in ordered_labels
        if counts[label] > 0
    ]


class DatasetAccumulator:
    def __init__(self, name: str):
        self.name = name
        self.sample_count = 0
        self.total_atoms = 0
        self.total_blocks = 0
        self.total_intra_block_bonds = 0

        self.atoms_per_mof: list[int] = []
        self.unique_elements_per_mof: list[int] = []
        self.blocks_per_mof: list[int] = []
        self.intra_block_bonds_per_mof: list[int] = []
        self.block_counts_per_type: dict[str, list[int]] = {block_type: [] for block_type in BLOCK_TYPES}
        self.block_atoms_per_type: dict[str, list[int]] = {block_type: [] for block_type in BLOCK_TYPES}
        self.block_bonds_per_type: dict[str, list[int]] = {block_type: [] for block_type in BLOCK_TYPES}

        self.element_atom_counts: Counter[int] = Counter()
        self.element_structure_counts: Counter[int] = Counter()
        self.block_type_counts: Counter[str] = Counter()
        self.block_type_atom_counts: Counter[str] = Counter()
        self.block_type_bond_counts: Counter[str] = Counter()
        self.component_signatures: Counter[tuple[int, int, int, int]] = Counter()

        self.example: dict[str, Any] | None = None

    def update(self, key: str, structure: Structure) -> None:
        num_atoms = int(structure.cart_coords.shape[0])
        num_blocks = len(structure.blocks)
        block_type_counter: Counter[str] = Counter()
        total_bonds = 0
        structure_elements: set[int] = set()

        self.sample_count += 1
        self.total_atoms += num_atoms
        self.total_blocks += num_blocks
        self.atoms_per_mof.append(num_atoms)
        self.blocks_per_mof.append(num_blocks)

        for block in structure.blocks:
            block_type = block.block_type
            atom_count = int(block.atomic_numbers.shape[0])
            bond_count = int(block.bond_indices.shape[1]) if block.bond_indices.ndim == 2 else 0

            block_type_counter[block_type] += 1
            self.block_type_counts[block_type] += 1
            self.block_type_atom_counts[block_type] += atom_count
            self.block_type_bond_counts[block_type] += bond_count
            self.block_atoms_per_type.setdefault(block_type, []).append(atom_count)
            self.block_bonds_per_type.setdefault(block_type, []).append(bond_count)

            total_bonds += bond_count

            atom_counts = Counter(int(z) for z in block.atomic_numbers.tolist())
            self.element_atom_counts.update(atom_counts)
            structure_elements.update(atom_counts.keys())

        for block_type in BLOCK_TYPES:
            self.block_counts_per_type[block_type].append(block_type_counter.get(block_type, 0))

        self.total_intra_block_bonds += total_bonds
        self.intra_block_bonds_per_mof.append(total_bonds)
        self.unique_elements_per_mof.append(len(structure_elements))
        self.element_structure_counts.update(structure_elements)
        self.component_signatures.update(
            [
                tuple(block_type_counter.get(block_type, 0) for block_type in BLOCK_TYPES),
            ]
        )

        if self.example is None:
            element_counts = Counter(int(z) for z in np.concatenate([block.atomic_numbers for block in structure.blocks]).tolist())
            self.example = {
                "key": key,
                "num_atoms": num_atoms,
                "num_blocks": num_blocks,
                "num_unique_elements": len(structure_elements),
                "block_type_counts": dict(block_type_counter),
                "elements": {
                    symbol_for_atomic_number(z): int(count)
                    for z, count in sorted(element_counts.items())
                },
                "lattice_shape": list(structure.lattice.shape),
                "cell_shape": list(structure.cell.shape),
            }

    def finalize(self) -> dict[str, Any]:
        unique_elements = sorted(self.element_atom_counts)

        top_signatures = []
        for signature, count in self.component_signatures.most_common(10):
            signature_dict = {block_type: int(signature[idx]) for idx, block_type in enumerate(BLOCK_TYPES)}
            signature_dict["count"] = int(count)
            top_signatures.append(signature_dict)

        return {
            "name": self.name,
            "samples": int(self.sample_count),
            "total_atoms": int(self.total_atoms),
            "total_blocks": int(self.total_blocks),
            "total_intra_block_bonds": int(self.total_intra_block_bonds),
            "unique_element_count": len(unique_elements),
            "unique_elements": [symbol_for_atomic_number(z) for z in unique_elements],
            "example": self.example,
            "atoms_per_mof": summarize_numeric(self.atoms_per_mof),
            "atoms_per_mof_histogram": histogram(self.atoms_per_mof),
            "unique_elements_per_mof": summarize_numeric(self.unique_elements_per_mof),
            "blocks_per_mof": summarize_numeric(self.blocks_per_mof),
            "intra_block_bonds_per_mof": summarize_numeric(self.intra_block_bonds_per_mof),
            "block_counts_per_mof": {
                block_type: summarize_numeric(values)
                for block_type, values in self.block_counts_per_type.items()
                if values
            },
            "block_sizes": {
                block_type: summarize_numeric(values)
                for block_type, values in self.block_atoms_per_type.items()
                if values
            },
            "block_bonds": {
                block_type: summarize_numeric(values)
                for block_type, values in self.block_bonds_per_type.items()
                if values
            },
            "block_type_totals": [
                {
                    "block_type": block_type,
                    "blocks": int(self.block_type_counts.get(block_type, 0)),
                    "atoms": int(self.block_type_atom_counts.get(block_type, 0)),
                    "bonds": int(self.block_type_bond_counts.get(block_type, 0)),
                }
                for block_type in BLOCK_TYPES
                if self.block_type_counts.get(block_type, 0) > 0
            ],
            "most_common_component_signatures": top_signatures,
            "elements_by_atom_count": [
                {
                    "symbol": symbol_for_atomic_number(z),
                    "atomic_number": int(z),
                    "atom_count": int(count),
                    "structure_count": int(self.element_structure_counts[z]),
                }
                for z, count in self.element_atom_counts.most_common()
            ],
        }


def scan_split(lmdb_path: Path, split_accumulator: DatasetAccumulator, overall_accumulator: DatasetAccumulator) -> None:
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for idx, (key, value) in enumerate(cursor, start=1):
                structure = loads_atommof(bytes(value))
                decoded_key = key.decode("ascii")
                split_accumulator.update(decoded_key, structure)
                overall_accumulator.update(decoded_key, structure)
                if idx % 1000 == 0:
                    print(f"[{split_accumulator.name}] processed {idx} samples")
    finally:
        env.close()


def aggregate_dataset(dataset_root: Path, splits: list[str]) -> dict[str, Any]:
    split_reports: dict[str, dict[str, Any]] = {}
    overall = DatasetAccumulator(name="overall")

    for split in splits:
        lmdb_path = dataset_root / f"{dataset_root.name}_{split}.lmdb"
        if not lmdb_path.exists():
            raise FileNotFoundError(f"Missing split LMDB: {lmdb_path}")

        split_accumulator = DatasetAccumulator(name=split)
        scan_split(lmdb_path=lmdb_path, split_accumulator=split_accumulator, overall_accumulator=overall)
        split_reports[split] = split_accumulator.finalize()

    return {
        "dataset_root": str(dataset_root.resolve()),
        "dataset_name": dataset_root.name,
        "schema": {
            "record_type": "AtomMOF Structure",
            "structure_fields": {
                "blocks": "list[Block]",
                "cart_coords": "(num_atoms, 3)",
                "frac_coords": "(num_atoms, 3)",
                "lattice": "(6,)",
                "cell": "(3, 3)",
            },
            "block_fields": {
                "block_type": "string",
                "cart_coords": "(block_atoms, 3)",
                "atomic_numbers": "(block_atoms,)",
                "bond_indices": "(2, num_bonds)",
                "bond_types": "(num_bonds,)",
                "formal_charges": "(block_atoms,)",
                "chirality_tags": "(block_atoms,)",
            },
        },
        "splits": split_reports,
        "overall": overall.finalize(),
    }


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Dataset: {report['dataset_name']}")
    lines.append(f"Root: {report['dataset_root']}")
    lines.append("")
    lines.append("Schema")
    for key, value in report["schema"]["structure_fields"].items():
        lines.append(f"  Structure.{key}: {value}")
    for key, value in report["schema"]["block_fields"].items():
        lines.append(f"  Block.{key}: {value}")

    for split_name, split_report in [("overall", report["overall"]), *report["splits"].items()]:
        lines.append("")
        lines.append(f"{split_name.upper()}")
        lines.append(
            f"  samples={split_report['samples']} total_atoms={split_report['total_atoms']} "
            f"total_blocks={split_report['total_blocks']} unique_elements={split_report['unique_element_count']}"
        )

        atoms = split_report["atoms_per_mof"]
        blocks = split_report["blocks_per_mof"]
        unique_elements = split_report["unique_elements_per_mof"]
        lines.append(
            "  atoms_per_mof: "
            f"mean={atoms.get('mean')} median={atoms.get('median')} "
            f"p05={atoms.get('p05')} p95={atoms.get('p95')} min={atoms.get('min')} max={atoms.get('max')}"
        )
        lines.append(
            "  blocks_per_mof: "
            f"mean={blocks.get('mean')} median={blocks.get('median')} "
            f"p05={blocks.get('p05')} p95={blocks.get('p95')} min={blocks.get('min')} max={blocks.get('max')}"
        )
        lines.append(
            "  unique_elements_per_mof: "
            f"mean={unique_elements.get('mean')} median={unique_elements.get('median')} "
            f"min={unique_elements.get('min')} max={unique_elements.get('max')}"
        )

        block_lines = []
        for block_type, stats in split_report["block_counts_per_mof"].items():
            if stats.get("max", 0) > 0:
                block_lines.append(f"{block_type.lower()} mean={stats.get('mean')} max={stats.get('max')}")
        if block_lines:
            lines.append("  component_counts: " + ", ".join(block_lines))

        top_elements = split_report["elements_by_atom_count"][:10]
        lines.append(
            "  top_elements_by_atom_count: "
            + ", ".join(f"{item['symbol']}={item['atom_count']}" for item in top_elements)
        )

        top_signatures = split_report["most_common_component_signatures"][:5]
        signature_parts = []
        for item in top_signatures:
            signature_parts.append(
                f"{item['count']}x("
                f"N={item['NODE']},L={item['LINKER']},S={item['SOLVENT']},A={item['ADSORBATE']})"
            )
        if signature_parts:
            lines.append("  common_component_signatures: " + ", ".join(signature_parts))

        example = split_report.get("example")
        if example:
            lines.append(
                f"  example[{example['key']}]: atoms={example['num_atoms']} "
                f"blocks={example['num_blocks']} block_types={example['block_type_counts']} "
                f"elements={example['elements']}"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an AtomMOF-format dataset in Matterformer.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data") / "mofs" / "bwdb",
        help="Path to the AtomMOF dataset directory containing <name>_{split}.lmdb files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to summarize.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=None,
        help="Optional path for a human-readable text report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report = aggregate_dataset(dataset_root=dataset_root, splits=args.splits)
    text = format_report(report)
    print(text, end="")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.output_text is not None:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
