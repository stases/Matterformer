from __future__ import annotations

from typing import Any

import torch


def fairchem_atomic_data_available() -> bool:
    try:
        from fairchem.core.datasets.atomic_data import AtomicData as _AtomicData  # noqa: F401
    except ImportError:
        return False
    return True


def _require_fairchem_atomic_data() -> tuple[Any, Any]:
    try:
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    except ImportError as exc:
        raise RuntimeError(
            "The AllScAIP OMol backend requires FairChem. Install fairchem-core, or run with "
            "FAIRCHEM_SRC=/path/to/fairchem so the launcher can add it to PYTHONPATH."
        ) from exc
    return AtomicData, atomicdata_list_to_batch


def omol_tensors_to_atomic_data(
    atomic_numbers: torch.Tensor,
    coords: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    charge: torch.Tensor | None = None,
    spin: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
    forces: torch.Tensor | None = None,
    dataset: str | None = "omol",
) -> Any:
    """Convert padded Matterformer OMol tensors into FairChem AtomicData."""

    AtomicData, atomicdata_list_to_batch = _require_fairchem_atomic_data()
    if atomic_numbers.ndim != 2:
        raise ValueError(f"atomic_numbers must have shape [B, N], got {tuple(atomic_numbers.shape)}")
    if coords.shape != (*atomic_numbers.shape, 3):
        raise ValueError(f"coords must have shape [B, N, 3], got {tuple(coords.shape)}")
    if pad_mask.shape != atomic_numbers.shape:
        raise ValueError(f"pad_mask must have shape [B, N], got {tuple(pad_mask.shape)}")
    if coords.dtype not in (torch.float32, torch.float64):
        coords = coords.float()
    if charge is None:
        charge = torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=atomic_numbers.device)
    if spin is None:
        spin = torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=atomic_numbers.device)

    graphs = []
    for batch_idx in range(atomic_numbers.shape[0]):
        atom_mask = ~pad_mask[batch_idx].bool()
        num_atoms = int(atom_mask.sum().item())
        if num_atoms <= 0:
            raise ValueError("AllScAIP AtomicData conversion requires every graph to contain at least one atom")
        graph_pos = coords[batch_idx, atom_mask].contiguous()
        graph_z = atomic_numbers[batch_idx, atom_mask].long().contiguous()
        graph_kwargs: dict[str, Any] = {
            "pos": graph_pos,
            "atomic_numbers": graph_z,
            "cell": coords.new_zeros(1, 3, 3),
            "pbc": torch.zeros(1, 3, dtype=torch.bool, device=coords.device),
            "natoms": torch.tensor([num_atoms], dtype=torch.long, device=coords.device),
            "edge_index": torch.empty(2, 0, dtype=torch.long, device=coords.device),
            "cell_offsets": coords.new_zeros(0, 3),
            "nedges": torch.zeros(1, dtype=torch.long, device=coords.device),
            "charge": charge[batch_idx : batch_idx + 1].long().contiguous(),
            "spin": spin[batch_idx : batch_idx + 1].long().contiguous(),
            "fixed": torch.zeros(num_atoms, dtype=torch.long, device=coords.device),
            "tags": torch.zeros(num_atoms, dtype=torch.long, device=coords.device),
            "sid": [str(batch_idx)],
        }
        if dataset is not None:
            graph_kwargs["dataset"] = dataset
        if energy is not None:
            graph_kwargs["energy"] = energy[batch_idx : batch_idx + 1].to(dtype=coords.dtype).contiguous()
        if forces is not None:
            graph_kwargs["forces"] = forces[batch_idx, atom_mask].to(dtype=coords.dtype).contiguous()
        graphs.append(AtomicData(**graph_kwargs))
    return atomicdata_list_to_batch(graphs)


def repad_flat_vectors(flat_vectors: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    if flat_vectors.ndim != 2:
        raise ValueError(f"flat_vectors must have shape [sum_N, C], got {tuple(flat_vectors.shape)}")
    valid_mask = ~pad_mask.bool()
    expected = int(valid_mask.sum().item())
    if flat_vectors.shape[0] != expected:
        raise ValueError(f"Expected {expected} flat vectors for pad_mask, got {flat_vectors.shape[0]}")
    out = flat_vectors.new_zeros((*pad_mask.shape, flat_vectors.shape[-1]))
    out[valid_mask] = flat_vectors
    return out
