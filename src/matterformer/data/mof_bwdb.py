from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from matterformer.data.qm9 import build_pad_mask

try:
    import lmdb
except ImportError:  # pragma: no cover - handled at runtime where BWDB is used
    lmdb = None


MOF_MAX_ATOMIC_NUMBER = 118
MOF_NUM_ATOM_TYPES = MOF_MAX_ATOMIC_NUMBER + 1
MOF_ATOM_PAD_TOKEN = 0

MOF_BLOCK_TYPE_DECODER = ["NODE", "LINKER", "SOLVENT", "ADSORBATE"]
MOF_BLOCK_TYPE_TO_ID = {name: index for index, name in enumerate(MOF_BLOCK_TYPE_DECODER)}
MOF_BLOCK_PAD_TOKEN = len(MOF_BLOCK_TYPE_DECODER)
MOF_NUM_BLOCK_TYPES = len(MOF_BLOCK_TYPE_DECODER)

BWDB_DATASET_INFO = {
    "name": "bwdb",
    "max_atomic_number": MOF_MAX_ATOMIC_NUMBER,
    "num_atom_types": MOF_NUM_ATOM_TYPES,
    "atom_pad_token": MOF_ATOM_PAD_TOKEN,
    "block_type_decoder": MOF_BLOCK_TYPE_DECODER,
    "block_type_to_id": MOF_BLOCK_TYPE_TO_ID,
}

# Approximate standard atomic weights indexed by atomic number.
ATOMIC_MASSES = torch.tensor(
    [
        0.0,
        1.008,
        4.002602,
        6.94,
        9.0121831,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998403163,
        20.1797,
        22.98976928,
        24.305,
        26.9815385,
        28.085,
        30.973761998,
        32.06,
        35.45,
        39.948,
        39.0983,
        40.078,
        44.955908,
        47.867,
        50.9415,
        51.9961,
        54.938044,
        55.845,
        58.933194,
        58.6934,
        63.546,
        65.38,
        69.723,
        72.63,
        74.921595,
        78.971,
        79.904,
        83.798,
        85.4678,
        87.62,
        88.90584,
        91.224,
        92.90637,
        95.95,
        98.0,
        101.07,
        102.9055,
        106.42,
        107.8682,
        112.414,
        114.818,
        118.71,
        121.76,
        127.6,
        126.90447,
        131.293,
        132.90545196,
        137.327,
        138.90547,
        140.116,
        140.90766,
        144.242,
        145.0,
        150.36,
        151.964,
        157.25,
        158.92535,
        162.5,
        164.93033,
        167.259,
        168.93422,
        173.045,
        174.9668,
        178.49,
        180.94788,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.966569,
        200.592,
        204.38,
        207.2,
        208.9804,
        209.0,
        210.0,
        222.0,
        223.0,
        226.0,
        227.0,
        232.0377,
        231.03588,
        238.02891,
        237.0,
        244.0,
        243.0,
        247.0,
        247.0,
        251.0,
        252.0,
        257.0,
        258.0,
        259.0,
        266.0,
        267.0,
        268.0,
        269.0,
        270.0,
        269.0,
        278.0,
        281.0,
        282.0,
        285.0,
        286.0,
        289.0,
        290.0,
        293.0,
        294.0,
        294.0,
    ],
    dtype=torch.float32,
)


@dataclass
class _LegacyBlock:
    block_type: str
    cart_coords: Any
    atomic_numbers: Any
    bond_indices: Any
    bond_types: Any
    formal_charges: Any
    chirality_tags: Any


@dataclass
class _LegacyStructure:
    blocks: list[_LegacyBlock]
    cart_coords: Any
    frac_coords: Any
    lattice: Any
    cell: Any
    info: Any = None


class _BWDBUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "src.data.types" and name == "Block":
            return _LegacyBlock
        if module == "src.data.types" and name in {"Structure", "MOFStructure"}:
            return _LegacyStructure
        return super().find_class(module, name)


@dataclass
class MOFSample:
    key: str
    atom_types: torch.Tensor
    cart_coords: torch.Tensor
    frac_coords: torch.Tensor
    block_index: torch.Tensor
    block_type: torch.Tensor
    atom_local_cart: torch.Tensor
    atom_local_frac: torch.Tensor
    block_sizes: torch.Tensor
    block_type_ids: torch.Tensor
    block_element_count_vecs: torch.Tensor
    block_com_cart: torch.Tensor
    block_com_frac: torch.Tensor
    lattice: torch.Tensor
    cell: torch.Tensor
    num_atoms: int
    num_blocks: int


@dataclass
class MOFStage1Batch:
    block_features: torch.Tensor
    block_type_ids: torch.Tensor
    block_sizes: torch.Tensor
    block_com_frac: torch.Tensor
    block_com_cart: torch.Tensor
    block_pad_mask: torch.Tensor
    num_blocks: torch.Tensor
    num_atoms: torch.Tensor
    lattice: torch.Tensor
    cell: torch.Tensor
    keys: list[str] | None = None

    def to(self, device: torch.device | str) -> "MOFStage1Batch":
        return MOFStage1Batch(
            block_features=self.block_features.to(device),
            block_type_ids=self.block_type_ids.to(device),
            block_sizes=self.block_sizes.to(device),
            block_com_frac=self.block_com_frac.to(device),
            block_com_cart=self.block_com_cart.to(device),
            block_pad_mask=self.block_pad_mask.to(device),
            num_blocks=self.num_blocks.to(device),
            num_atoms=self.num_atoms.to(device),
            lattice=self.lattice.to(device),
            cell=self.cell.to(device),
            keys=self.keys,
        )


@dataclass
class MOFStage2Batch:
    atom_types: torch.Tensor
    atom_frac: torch.Tensor
    atom_cart: torch.Tensor
    atom_local_frac: torch.Tensor
    atom_local_cart: torch.Tensor
    atom_prior_mu_frac: torch.Tensor
    atom_block_index: torch.Tensor
    atom_block_type: torch.Tensor
    atom_pad_mask: torch.Tensor
    num_atoms: torch.Tensor
    block_features: torch.Tensor
    block_type_ids: torch.Tensor
    block_sizes: torch.Tensor
    block_com_frac: torch.Tensor
    block_com_cart: torch.Tensor
    block_pad_mask: torch.Tensor
    num_blocks: torch.Tensor
    lattice: torch.Tensor
    cell: torch.Tensor
    keys: list[str] | None = None

    def to(self, device: torch.device | str) -> "MOFStage2Batch":
        return MOFStage2Batch(
            atom_types=self.atom_types.to(device),
            atom_frac=self.atom_frac.to(device),
            atom_cart=self.atom_cart.to(device),
            atom_local_frac=self.atom_local_frac.to(device),
            atom_local_cart=self.atom_local_cart.to(device),
            atom_prior_mu_frac=self.atom_prior_mu_frac.to(device),
            atom_block_index=self.atom_block_index.to(device),
            atom_block_type=self.atom_block_type.to(device),
            atom_pad_mask=self.atom_pad_mask.to(device),
            num_atoms=self.num_atoms.to(device),
            block_features=self.block_features.to(device),
            block_type_ids=self.block_type_ids.to(device),
            block_sizes=self.block_sizes.to(device),
            block_com_frac=self.block_com_frac.to(device),
            block_com_cart=self.block_com_cart.to(device),
            block_pad_mask=self.block_pad_mask.to(device),
            num_blocks=self.num_blocks.to(device),
            lattice=self.lattice.to(device),
            cell=self.cell.to(device),
            keys=self.keys,
        )

    def atom_onehot(self, num_classes: int = MOF_NUM_ATOM_TYPES) -> torch.Tensor:
        atom_types = self.atom_types.clamp(min=0, max=num_classes - 1)
        one_hot = torch.nn.functional.one_hot(atom_types, num_classes=num_classes).float()
        return one_hot.masked_fill(self.atom_pad_mask[..., None], 0.0)


def _ensure_lmdb_available() -> None:
    if lmdb is None:
        raise ImportError("lmdb is required to use BWDBDataset.")


def _open_lmdb_readonly(path: str | Path, *, max_readers: int = 32):
    _ensure_lmdb_available()
    return lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=max_readers,
    )


def _load_bwdb_object(value: bytes) -> _LegacyStructure:
    return _BWDBUnpickler(io.BytesIO(value)).load()


def _decode_lmdb_key(key: bytes | str) -> str:
    if isinstance(key, str):
        return key
    try:
        return key.decode("utf-8")
    except UnicodeDecodeError:
        return key.hex()


def _as_lmdb_key(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        return key
    return key.encode("utf-8")


def fractional_to_cartesian(frac_coords: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    frac_coords = torch.as_tensor(frac_coords, dtype=torch.float32)
    cell = torch.as_tensor(cell, dtype=torch.float32)
    return frac_coords @ cell


def cartesian_to_fractional(cart_coords: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    cart_coords = torch.as_tensor(cart_coords, dtype=torch.float32)
    cell = torch.as_tensor(cell, dtype=torch.float32)
    return torch.linalg.solve(cell.transpose(-1, -2), cart_coords.transpose(-1, -2)).transpose(-1, -2)


def wrap_fractional_delta(delta: torch.Tensor) -> torch.Tensor:
    delta = torch.as_tensor(delta, dtype=torch.float32)
    return delta - torch.round(delta)


def compute_block_com_cart(cart_coords: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
    cart_coords = torch.as_tensor(cart_coords, dtype=torch.float32)
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1)
    if cart_coords.ndim != 2 or cart_coords.shape[-1] != 3:
        raise ValueError(f"cart_coords must have shape (N, 3), got {tuple(cart_coords.shape)}")
    if cart_coords.shape[0] != atomic_numbers.shape[0]:
        raise ValueError("cart_coords and atomic_numbers must describe the same number of atoms")
    masses = ATOMIC_MASSES.to(device=cart_coords.device)[atomic_numbers.clamp(min=0, max=MOF_MAX_ATOMIC_NUMBER)]
    masses = masses.clamp_min(1e-8)
    return (cart_coords * masses[:, None]).sum(dim=0) / masses.sum()


def _normalize_bond_indices(bond_indices: Any, num_atoms: int) -> torch.Tensor:
    edge_index = torch.as_tensor(bond_indices, dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)
    if edge_index.ndim != 2:
        raise ValueError(f"bond_indices must have shape (2, E) or (E, 2), got {tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:
        pass
    elif edge_index.shape[1] == 2:
        edge_index = edge_index.transpose(0, 1).contiguous()
    else:
        raise ValueError(f"bond_indices must have shape (2, E) or (E, 2), got {tuple(edge_index.shape)}")
    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(num_atoms):
            raise ValueError("bond_indices contain atom indices outside the block range")
    return edge_index


def unwrap_block_fractional_coords(
    frac_coords: torch.Tensor,
    bond_indices: Any,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    frac_coords = torch.as_tensor(frac_coords, dtype=torch.float32)
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1)
    if frac_coords.ndim != 2 or frac_coords.shape[-1] != 3:
        raise ValueError(f"frac_coords must have shape (N, 3), got {tuple(frac_coords.shape)}")
    if frac_coords.shape[0] != atomic_numbers.shape[0]:
        raise ValueError("frac_coords and atomic_numbers must describe the same number of atoms")
    num_atoms = int(frac_coords.shape[0])
    if num_atoms == 0:
        return frac_coords.clone()

    edge_index = _normalize_bond_indices(bond_indices, num_atoms)
    neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
    for src, dst in edge_index.transpose(0, 1).tolist():
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        neighbors[src_i].append(dst_i)
        neighbors[dst_i].append(src_i)

    masses = ATOMIC_MASSES.to(device=frac_coords.device)[atomic_numbers.clamp(min=0, max=MOF_MAX_ATOMIC_NUMBER)]
    masses = masses.clamp_min(1e-8)
    frac_wrapped = torch.remainder(frac_coords, 1.0)
    frac_unwrapped = frac_wrapped.clone()
    visited = torch.zeros(num_atoms, device=frac_coords.device, dtype=torch.bool)

    for root in range(num_atoms):
        if bool(visited[root]):
            continue
        if bool(visited.any()):
            ref = (frac_unwrapped[visited] * masses[visited, None]).sum(dim=0) / masses[visited].sum()
            frac_unwrapped[root] = ref + wrap_fractional_delta(frac_wrapped[root] - ref)
        else:
            frac_unwrapped[root] = frac_wrapped[root]
        visited[root] = True
        queue = [root]
        while queue:
            src = queue.pop(0)
            src_pos = frac_unwrapped[src]
            for dst in neighbors[src]:
                if bool(visited[dst]):
                    continue
                frac_unwrapped[dst] = src_pos + wrap_fractional_delta(frac_wrapped[dst] - frac_wrapped[src])
                visited[dst] = True
                queue.append(dst)

    return frac_unwrapped


def build_element_count_vector(atomic_numbers: torch.Tensor) -> torch.Tensor:
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long).reshape(-1)
    counts = torch.bincount(
        atomic_numbers.clamp(min=0, max=MOF_MAX_ATOMIC_NUMBER),
        minlength=MOF_NUM_ATOM_TYPES,
    )
    return counts[1:].to(dtype=torch.float32)


def structure_to_mof_sample(structure: Any, *, key: str = "") -> MOFSample:
    cart_coords = torch.as_tensor(structure.cart_coords, dtype=torch.float32)
    frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32)
    lattice = torch.as_tensor(structure.lattice, dtype=torch.float32).reshape(6)
    cell = torch.as_tensor(structure.cell, dtype=torch.float32).reshape(3, 3)

    if cart_coords.ndim != 2 or cart_coords.shape[-1] != 3:
        raise ValueError(f"cart_coords must have shape (N, 3), got {tuple(cart_coords.shape)}")
    if frac_coords.shape != cart_coords.shape:
        raise ValueError("frac_coords must have the same shape as cart_coords")

    num_atoms = int(cart_coords.shape[0])
    num_blocks = len(structure.blocks)

    atom_types = torch.zeros(num_atoms, dtype=torch.long)
    block_index = torch.zeros(num_atoms, dtype=torch.long)
    block_type = torch.full((num_atoms,), MOF_BLOCK_PAD_TOKEN, dtype=torch.long)
    atom_local_cart = torch.zeros(num_atoms, 3, dtype=torch.float32)
    atom_local_frac = torch.zeros(num_atoms, 3, dtype=torch.float32)

    block_sizes = torch.zeros(num_blocks, dtype=torch.long)
    block_type_ids = torch.full((num_blocks,), MOF_BLOCK_PAD_TOKEN, dtype=torch.long)
    block_features = torch.zeros(num_blocks, MOF_MAX_ATOMIC_NUMBER, dtype=torch.float32)
    block_com_cart = torch.zeros(num_blocks, 3, dtype=torch.float32)
    block_com_frac = torch.zeros(num_blocks, 3, dtype=torch.float32)

    cursor = 0
    for block_id, block in enumerate(structure.blocks):
        atomic_numbers = torch.as_tensor(block.atomic_numbers, dtype=torch.long).reshape(-1)
        num_block_atoms = int(atomic_numbers.shape[0])
        next_cursor = cursor + num_block_atoms
        if next_cursor > num_atoms:
            raise ValueError("Block atom counts exceed structure atom count")

        block_cart = cart_coords[cursor:next_cursor]
        block_frac = frac_coords[cursor:next_cursor]
        block_type_name = str(block.block_type).upper()
        if block_type_name not in MOF_BLOCK_TYPE_TO_ID:
            raise KeyError(f"Unsupported block type: {block.block_type!r}")

        atom_types[cursor:next_cursor] = atomic_numbers
        block_index[cursor:next_cursor] = block_id
        block_type[cursor:next_cursor] = MOF_BLOCK_TYPE_TO_ID[block_type_name]
        block_sizes[block_id] = num_block_atoms
        block_type_ids[block_id] = MOF_BLOCK_TYPE_TO_ID[block_type_name]
        block_features[block_id] = build_element_count_vector(atomic_numbers)

        block_frac_unwrapped = unwrap_block_fractional_coords(
            block_frac,
            block.bond_indices,
            atomic_numbers,
        )
        block_cart_unwrapped = fractional_to_cartesian(block_frac_unwrapped, cell)
        com_cart_unwrapped = compute_block_com_cart(block_cart_unwrapped, atomic_numbers)
        com_frac_unwrapped = cartesian_to_fractional(com_cart_unwrapped[None, :], cell)[0]
        com_frac = torch.remainder(com_frac_unwrapped, 1.0)
        com_cart = fractional_to_cartesian(com_frac[None, :], cell)[0]
        block_com_cart[block_id] = com_cart
        block_com_frac[block_id] = com_frac
        atom_local_cart[cursor:next_cursor] = block_cart_unwrapped - com_cart_unwrapped
        atom_local_frac[cursor:next_cursor] = block_frac_unwrapped - com_frac_unwrapped
        cursor = next_cursor

    if cursor != num_atoms:
        raise ValueError(f"Structure has {num_atoms} atoms but blocks account for {cursor}")

    return MOFSample(
        key=key,
        atom_types=atom_types,
        cart_coords=cart_coords,
        frac_coords=frac_coords,
        block_index=block_index,
        block_type=block_type,
        atom_local_cart=atom_local_cart,
        atom_local_frac=atom_local_frac,
        block_sizes=block_sizes,
        block_type_ids=block_type_ids,
        block_element_count_vecs=block_features,
        block_com_cart=block_com_cart,
        block_com_frac=block_com_frac,
        lattice=lattice,
        cell=cell,
        num_atoms=num_atoms,
        num_blocks=num_blocks,
    )


class BWDBDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "./data/mofs/bwdb",
        *,
        split: str = "train",
        sample_limit: int | None = None,
        max_num_atoms: int | None = None,
    ) -> None:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")

        self.root = Path(root)
        self.split = split
        self.lmdb_path = self.root / f"bwdb_{split}.lmdb"
        self.metadata_path = self.root / f"bwdb_{split}_metadata.pkl"
        self.sample_limit = sample_limit
        self.max_num_atoms = max_num_atoms

        if not self.lmdb_path.is_file():
            raise FileNotFoundError(f"BWDB LMDB not found: {self.lmdb_path}")

        self.keys, self.num_atoms_list = self._load_index()
        self.env = None
        self.txn = None

    @property
    def dataset_info(self) -> dict[str, object]:
        return BWDB_DATASET_INFO

    def _load_index(self) -> tuple[list[bytes], list[int]]:
        if self.metadata_path.is_file():
            with self.metadata_path.open("rb") as handle:
                metadata = pickle.load(handle)
            keys = list(metadata.keys())
            num_atoms_list = [int(item["num_atoms"]) for item in metadata.values()]
        else:
            keys, num_atoms_list = self._scan_lmdb_index()

        if self.max_num_atoms is not None:
            filtered = [
                (key, count)
                for key, count in zip(keys, num_atoms_list, strict=True)
                if int(count) <= self.max_num_atoms
            ]
            keys = [key for key, _ in filtered]
            num_atoms_list = [count for _, count in filtered]

        if self.sample_limit is not None:
            keys = keys[: self.sample_limit]
            num_atoms_list = num_atoms_list[: self.sample_limit]

        return list(keys), list(num_atoms_list)

    def _scan_lmdb_index(self) -> tuple[list[bytes], list[int]]:
        env = _open_lmdb_readonly(self.lmdb_path)
        try:
            keys: list[bytes] = []
            num_atoms_list: list[int] = []
            with env.begin(write=False) as txn:
                for key, value in txn.cursor():
                    structure = _load_bwdb_object(bytes(value))
                    keys.append(bytes(key))
                    num_atoms_list.append(int(len(structure.cart_coords)))
            return keys, num_atoms_list
        finally:
            env.close()

    def _get_txn(self):
        if self.txn is None:
            self.env = _open_lmdb_readonly(self.lmdb_path)
            self.txn = self.env.begin(buffers=True, write=False)
        return self.txn

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> MOFSample:
        key = self.keys[idx]
        txn = self._get_txn()
        value = txn.get(_as_lmdb_key(key))
        if value is None:
            raise KeyError(f"Missing LMDB key: {_decode_lmdb_key(key)}")
        structure = _load_bwdb_object(bytes(value))
        return structure_to_mof_sample(structure, key=_decode_lmdb_key(key))

    def get_raw(self, idx: int) -> Any:
        key = self.keys[idx]
        txn = self._get_txn()
        value = txn.get(_as_lmdb_key(key))
        if value is None:
            raise KeyError(f"Missing LMDB key: {_decode_lmdb_key(key)}")
        return _load_bwdb_object(bytes(value))

    def _close(self) -> None:
        if self.txn is not None:
            self.txn.abort()
            self.txn = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self._close()


def collate_mof_stage1(samples: Sequence[MOFSample]) -> MOFStage1Batch:
    if not samples:
        raise ValueError("samples must not be empty")

    num_blocks = torch.tensor([sample.num_blocks for sample in samples], dtype=torch.long)
    num_atoms = torch.tensor([sample.num_atoms for sample in samples], dtype=torch.long)
    max_blocks = int(num_blocks.max().item())
    batch_size = len(samples)

    block_features = torch.zeros(batch_size, max_blocks, MOF_MAX_ATOMIC_NUMBER, dtype=torch.float32)
    block_type_ids = torch.full((batch_size, max_blocks), MOF_BLOCK_PAD_TOKEN, dtype=torch.long)
    block_sizes = torch.zeros(batch_size, max_blocks, dtype=torch.long)
    block_com_frac = torch.zeros(batch_size, max_blocks, 3, dtype=torch.float32)
    block_com_cart = torch.zeros(batch_size, max_blocks, 3, dtype=torch.float32)
    lattice = torch.zeros(batch_size, 6, dtype=torch.float32)
    cell = torch.zeros(batch_size, 3, 3, dtype=torch.float32)
    keys: list[str] = []

    for batch_idx, sample in enumerate(samples):
        count = sample.num_blocks
        block_features[batch_idx, :count] = sample.block_element_count_vecs
        block_type_ids[batch_idx, :count] = sample.block_type_ids
        block_sizes[batch_idx, :count] = sample.block_sizes
        block_com_frac[batch_idx, :count] = sample.block_com_frac
        block_com_cart[batch_idx, :count] = sample.block_com_cart
        lattice[batch_idx] = sample.lattice
        cell[batch_idx] = sample.cell
        keys.append(sample.key)

    block_pad_mask = build_pad_mask(num_blocks, max_atoms=max_blocks)
    block_com_frac = block_com_frac.masked_fill(block_pad_mask[..., None], 0.0)
    block_com_cart = block_com_cart.masked_fill(block_pad_mask[..., None], 0.0)
    block_features = block_features.masked_fill(block_pad_mask[..., None], 0.0)

    return MOFStage1Batch(
        block_features=block_features,
        block_type_ids=block_type_ids,
        block_sizes=block_sizes,
        block_com_frac=block_com_frac,
        block_com_cart=block_com_cart,
        block_pad_mask=block_pad_mask,
        num_blocks=num_blocks,
        num_atoms=num_atoms,
        lattice=lattice,
        cell=cell,
        keys=keys,
    )


def collate_mof_stage2(samples: Sequence[MOFSample]) -> MOFStage2Batch:
    if not samples:
        raise ValueError("samples must not be empty")

    num_atoms = torch.tensor([sample.num_atoms for sample in samples], dtype=torch.long)
    num_blocks = torch.tensor([sample.num_blocks for sample in samples], dtype=torch.long)
    max_atoms = int(num_atoms.max().item())
    max_blocks = int(num_blocks.max().item())
    batch_size = len(samples)

    atom_types = torch.full((batch_size, max_atoms), MOF_ATOM_PAD_TOKEN, dtype=torch.long)
    atom_frac = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    atom_cart = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    atom_local_frac = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    atom_local_cart = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    atom_prior_mu_frac = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    atom_block_index = torch.full((batch_size, max_atoms), -1, dtype=torch.long)
    atom_block_type = torch.full((batch_size, max_atoms), MOF_BLOCK_PAD_TOKEN, dtype=torch.long)

    block_features = torch.zeros(batch_size, max_blocks, MOF_MAX_ATOMIC_NUMBER, dtype=torch.float32)
    block_type_ids = torch.full((batch_size, max_blocks), MOF_BLOCK_PAD_TOKEN, dtype=torch.long)
    block_sizes = torch.zeros(batch_size, max_blocks, dtype=torch.long)
    block_com_frac = torch.zeros(batch_size, max_blocks, 3, dtype=torch.float32)
    block_com_cart = torch.zeros(batch_size, max_blocks, 3, dtype=torch.float32)
    lattice = torch.zeros(batch_size, 6, dtype=torch.float32)
    cell = torch.zeros(batch_size, 3, 3, dtype=torch.float32)
    keys: list[str] = []

    for batch_idx, sample in enumerate(samples):
        atom_count = sample.num_atoms
        block_count = sample.num_blocks
        atom_types[batch_idx, :atom_count] = sample.atom_types
        atom_frac[batch_idx, :atom_count] = sample.frac_coords
        atom_cart[batch_idx, :atom_count] = sample.cart_coords
        atom_local_frac[batch_idx, :atom_count] = sample.atom_local_frac
        atom_local_cart[batch_idx, :atom_count] = sample.atom_local_cart
        atom_block_index[batch_idx, :atom_count] = sample.block_index
        atom_block_type[batch_idx, :atom_count] = sample.block_type
        atom_prior_mu_frac[batch_idx, :atom_count] = sample.block_com_frac[sample.block_index]

        block_features[batch_idx, :block_count] = sample.block_element_count_vecs
        block_type_ids[batch_idx, :block_count] = sample.block_type_ids
        block_sizes[batch_idx, :block_count] = sample.block_sizes
        block_com_frac[batch_idx, :block_count] = sample.block_com_frac
        block_com_cart[batch_idx, :block_count] = sample.block_com_cart
        lattice[batch_idx] = sample.lattice
        cell[batch_idx] = sample.cell
        keys.append(sample.key)

    atom_pad_mask = build_pad_mask(num_atoms, max_atoms=max_atoms)
    block_pad_mask = build_pad_mask(num_blocks, max_atoms=max_blocks)
    atom_frac = atom_frac.masked_fill(atom_pad_mask[..., None], 0.0)
    atom_cart = atom_cart.masked_fill(atom_pad_mask[..., None], 0.0)
    atom_local_frac = atom_local_frac.masked_fill(atom_pad_mask[..., None], 0.0)
    atom_local_cart = atom_local_cart.masked_fill(atom_pad_mask[..., None], 0.0)
    atom_prior_mu_frac = atom_prior_mu_frac.masked_fill(atom_pad_mask[..., None], 0.0)
    block_features = block_features.masked_fill(block_pad_mask[..., None], 0.0)
    block_com_frac = block_com_frac.masked_fill(block_pad_mask[..., None], 0.0)
    block_com_cart = block_com_cart.masked_fill(block_pad_mask[..., None], 0.0)

    return MOFStage2Batch(
        atom_types=atom_types,
        atom_frac=atom_frac,
        atom_cart=atom_cart,
        atom_local_frac=atom_local_frac,
        atom_local_cart=atom_local_cart,
        atom_prior_mu_frac=atom_prior_mu_frac,
        atom_block_index=atom_block_index,
        atom_block_type=atom_block_type,
        atom_pad_mask=atom_pad_mask,
        num_atoms=num_atoms,
        block_features=block_features,
        block_type_ids=block_type_ids,
        block_sizes=block_sizes,
        block_com_frac=block_com_frac,
        block_com_cart=block_com_cart,
        block_pad_mask=block_pad_mask,
        num_blocks=num_blocks,
        lattice=lattice,
        cell=cell,
        keys=keys,
    )
