from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from matterformer.models.platonic.groups import PLATONIC_GROUPS


class PlatonicLinear(nn.Module):
    """Group-constrained linear map over a finite Platonic group axis.

    Inputs and outputs are stored flattened as ``[..., G * C]``. The learned
    kernel is a finite group convolution, so the transform commutes with group
    permutations induced by the Cayley table.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        solid: str = "tetrahedron",
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        solid = solid.lower()
        if solid not in PLATONIC_GROUPS:
            raise ValueError(f"Unknown Platonic group {solid!r}")
        group = PLATONIC_GROUPS[solid]
        self.solid = solid
        self.G = group.G
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features % self.G != 0:
            raise ValueError(f"in_features={self.in_features} must be divisible by group order {self.G}")
        if self.out_features % self.G != 0:
            raise ValueError(f"out_features={self.out_features} must be divisible by group order {self.G}")
        self.in_channels = self.in_features // self.G
        self.out_channels = self.out_features // self.G

        self.kernel = nn.Parameter(torch.empty(self.G, self.out_channels, self.in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("cayley_table", group.cayley_table, persistent=False)
        self.register_buffer("inverse_indices", group.inverse_indices, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.G * self.in_channels
        nn.init.normal_(self.kernel, mean=0.0, std=1.0 / math.sqrt(fan_in))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_weight(self) -> torch.Tensor:
        device = self.kernel.device
        out_group = torch.arange(self.G, device=device).view(self.G, 1)
        in_group = torch.arange(self.G, device=device).view(1, self.G)
        inv_in = self.inverse_indices[in_group]
        kernel_group = self.cayley_table[inv_in, out_group]
        expanded = self.kernel[kernel_group]
        return expanded.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.get_weight(), None)
        if self.bias is None:
            return output
        output_shape = output.shape
        output = output.view(*output_shape[:-1], self.G, self.out_channels)
        output = output + self.bias
        return output.view(output_shape)
