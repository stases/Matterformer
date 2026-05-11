from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlatonicSolidGroup:
    elements: torch.Tensor
    name: str
    inverse_indices: torch.Tensor
    cayley_table: torch.Tensor

    @property
    def G(self) -> int:
        return int(self.elements.shape[0])

    @property
    def dim(self) -> int:
        return int(self.elements.shape[-1])


def _compute_inverse_indices(elements: torch.Tensor) -> torch.Tensor:
    inverses = elements.transpose(-1, -2)
    inverse_indices = torch.empty(elements.shape[0], dtype=torch.long)
    for idx, inverse in enumerate(inverses):
        diffs = (elements - inverse.unsqueeze(0)).square().sum(dim=(1, 2))
        inverse_indices[idx] = int(diffs.argmin().item())
    return inverse_indices


def _compute_cayley_table(elements: torch.Tensor) -> torch.Tensor:
    group_order = elements.shape[0]
    cayley = torch.empty(group_order, group_order, dtype=torch.long)
    for left in range(group_order):
        for right in range(group_order):
            product = elements[left] @ elements[right]
            diffs = (elements - product.unsqueeze(0)).square().sum(dim=(1, 2))
            cayley[left, right] = int(diffs.argmin().item())
    return cayley


def _make_group(elements: torch.Tensor, name: str) -> PlatonicSolidGroup:
    elements = elements.to(dtype=torch.float32)
    return PlatonicSolidGroup(
        elements=elements,
        name=name,
        inverse_indices=_compute_inverse_indices(elements),
        cayley_table=_compute_cayley_table(elements),
    )


def _trivial_elements() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float32).view(1, 3, 3)


def _tetrahedral_elements() -> torch.Tensor:
    return torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        ],
        dtype=torch.float32,
    )


TRIVIAL_GROUP = _make_group(_trivial_elements(), "trivial")
TETRAHEDRAL_GROUP = _make_group(_tetrahedral_elements(), "tetrahedron")

PLATONIC_GROUPS: dict[str, PlatonicSolidGroup] = {
    "trivial": TRIVIAL_GROUP,
    "tetrahedron": TETRAHEDRAL_GROUP,
}


def tetrahedral_permutation(group_element_index: int) -> torch.Tensor:
    group = TETRAHEDRAL_GROUP
    idx = int(group_element_index)
    if idx < 0 or idx >= group.G:
        raise ValueError(f"group_element_index must be in [0, {group.G}), got {idx}")
    return group.cayley_table[idx]
