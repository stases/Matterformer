from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from matterformer.models.platonic.groups import PLATONIC_GROUPS, PlatonicSolidGroup


def _tetra_quotient_exponents(group: PlatonicSolidGroup) -> torch.Tensor:
    """Return exponents for the quotient A4 / V4 ~= C3."""
    if group.G != 12 or group.dim != 3:
        raise ValueError("tetra quotient exponents require a 12-element 3D tetrahedral group")
    elements = group.elements.float()
    trace = elements.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    v4 = torch.nonzero((trace > 2.5) | (trace < -0.5), as_tuple=False).flatten()
    if int(v4.numel()) != 4:
        raise RuntimeError(f"Expected V4 subgroup to have 4 elements, found {int(v4.numel())}")

    v4_set = set(int(index) for index in v4.tolist())
    generator = next(index for index in range(group.G) if index not in v4_set)
    generator_sq = int(group.cayley_table[generator, generator].item())
    class1 = torch.unique(group.cayley_table[v4, generator])
    class2 = torch.unique(group.cayley_table[v4, generator_sq])

    exponents = torch.full((group.G,), -1, dtype=torch.long)
    exponents[v4] = 0
    exponents[class1] = 1
    exponents[class2] = 2
    if bool((exponents < 0).any()):
        raise RuntimeError(f"Failed to classify all tetra elements into A4/V4 cosets: {exponents.tolist()}")

    lhs = exponents[group.cayley_table]
    rhs = (exponents[:, None] + exponents[None, :]) % 3
    if not bool(torch.equal(lhs, rhs)):
        raise RuntimeError("Computed tetra quotient exponents are not a group homomorphism")
    return exponents


def _tetra_fourier_data(group: PlatonicSolidGroup) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the real tetra Fourier basis and irreps used by the Fourier backend."""
    if group.G != 12 or group.dim != 3:
        raise ValueError("Fourier backend currently supports only tetrahedron")

    elements = group.elements.float()
    exponents = _tetra_quotient_exponents(group).float()
    theta = (2.0 * math.pi / 3.0) * exponents
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rho2 = torch.zeros(group.G, 2, 2, dtype=torch.float32)
    rho2[:, 0, 0] = cos_theta
    rho2[:, 0, 1] = -sin_theta
    rho2[:, 1, 0] = sin_theta
    rho2[:, 1, 1] = cos_theta

    columns: list[torch.Tensor] = [
        torch.ones(group.G, dtype=torch.float32) / math.sqrt(float(group.G)),
        math.sqrt(2.0 / float(group.G)) * cos_theta,
        math.sqrt(2.0 / float(group.G)) * sin_theta,
    ]
    scale3 = math.sqrt(3.0 / float(group.G))
    for row in range(3):
        for col in range(3):
            columns.append(scale3 * elements[:, row, col])
    basis = torch.stack(columns, dim=1).contiguous()

    orthogonality_error = (basis.T @ basis - torch.eye(group.G, dtype=torch.float32)).abs().max().item()
    if orthogonality_error > 1.0e-5:
        raise RuntimeError(f"Tetra Fourier basis is not orthonormal; max error={orthogonality_error:.3e}")
    return basis, rho2.contiguous(), elements.contiguous()


class PlatonicLinear(nn.Module):
    """Group-constrained linear map over a finite Platonic group axis.

    Inputs and outputs are stored flattened as ``[..., G * C]``. The learned
    kernel is a finite group convolution, so the transform commutes with group
    permutations induced by the Cayley table. The default ``linear_backend`` is
    the original dense spatial implementation. ``linear_backend="fourier"`` is
    an equivalent tetra-only finite-group Fourier implementation that keeps the
    same spatial ``kernel`` and ``bias`` parameters. ``linear_backend="fourier_direct"``
    uses direct Fourier parameters for performance experiments.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        solid: str = "tetrahedron",
        *,
        bias: bool = True,
        linear_backend: str = "spatial",
    ) -> None:
        super().__init__()
        solid = solid.lower()
        if solid not in PLATONIC_GROUPS:
            raise ValueError(f"Unknown Platonic group {solid!r}")
        group = PLATONIC_GROUPS[solid]
        backend = str(linear_backend).lower().replace("-", "_")
        if backend in {"default", "dense"}:
            backend = "spatial"
        if backend not in {"spatial", "fourier", "fourier_direct"}:
            raise ValueError("linear_backend must be one of {'spatial', 'fourier', 'fourier_direct'}")
        if backend in {"fourier", "fourier_direct"} and solid != "tetrahedron":
            raise ValueError(f"linear_backend={backend!r} is currently implemented only for solid='tetrahedron'")
        self.solid = solid
        self.G = group.G
        self.linear_backend = backend
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features % self.G != 0:
            raise ValueError(f"in_features={self.in_features} must be divisible by group order {self.G}")
        if self.out_features % self.G != 0:
            raise ValueError(f"out_features={self.out_features} must be divisible by group order {self.G}")
        self.in_channels = self.in_features // self.G
        self.out_channels = self.out_features // self.G

        if self.linear_backend == "fourier_direct":
            self.register_parameter("kernel", None)
            self.w1 = nn.Parameter(torch.empty(self.out_channels, self.in_channels))
            self.w2_re = nn.Parameter(torch.empty(self.out_channels, self.in_channels))
            self.w2_im = nn.Parameter(torch.empty(self.out_channels, self.in_channels))
            self.w3 = nn.Parameter(torch.empty(3 * self.out_channels, 3 * self.in_channels))
        else:
            self.kernel = nn.Parameter(torch.empty(self.G, self.out_channels, self.in_channels))
            self.register_parameter("w1", None)
            self.register_parameter("w2_re", None)
            self.register_parameter("w2_im", None)
            self.register_parameter("w3", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("cayley_table", group.cayley_table, persistent=False)
        self.register_buffer("inverse_indices", group.inverse_indices, persistent=False)
        out_group = torch.arange(self.G).view(self.G, 1)
        in_group = torch.arange(self.G).view(1, self.G)
        inv_in = group.inverse_indices[in_group]
        kernel_group = group.cayley_table[inv_in, out_group]
        self.register_buffer("kernel_group_indices", kernel_group.contiguous(), persistent=False)
        if self.linear_backend in {"fourier", "fourier_direct"}:
            fourier_basis, rho2, rho3 = _tetra_fourier_data(group)
            self.register_buffer("fourier_basis", fourier_basis, persistent=False)
            self.register_buffer("tetra_rho2", rho2, persistent=False)
            self.register_buffer("tetra_rho3", rho3, persistent=False)
        else:
            self.register_buffer("fourier_basis", torch.empty(0), persistent=False)
            self.register_buffer("tetra_rho2", torch.empty(0), persistent=False)
            self.register_buffer("tetra_rho3", torch.empty(0), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.G * self.in_channels
        if self.linear_backend == "fourier_direct":
            spatial_kernel = torch.empty(self.G, self.out_channels, self.in_channels, device=self.w1.device, dtype=self.w1.dtype)
            nn.init.normal_(spatial_kernel, mean=0.0, std=1.0 / math.sqrt(fan_in))
            low_flat, w3 = self._fourier_weights_from_kernel_tensor(spatial_kernel)
            low = low_flat.view(3, self.out_channels, 3, self.in_channels).permute(0, 2, 1, 3)
            with torch.no_grad():
                self.w1.copy_(low[0, 0])
                self.w2_re.copy_(low[1, 1])
                self.w2_im.copy_(low[2, 1])
                self.w3.copy_(w3)
        else:
            nn.init.normal_(self.kernel, mean=0.0, std=1.0 / math.sqrt(fan_in))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_weight(self) -> torch.Tensor:
        if self.kernel is None:
            raise NotImplementedError("get_weight() is only available for spatial-kernel PlatonicLinear backends")
        expanded = self.kernel[self.kernel_group_indices]
        return expanded.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def _fourier_weights_from_kernel_tensor(self, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho2 = self.tetra_rho2.to(device=kernel.device, dtype=kernel.dtype)
        rho3 = self.tetra_rho3.to(device=kernel.device, dtype=kernel.dtype)

        low = kernel.new_zeros(3, 3, self.out_channels, self.in_channels)
        low[0, 0] = kernel.sum(dim=0)
        low[1:3, 1:3] = torch.einsum("gab,goi->aboi", rho2, kernel)
        w3 = torch.einsum("gqn,goi->nqoi", rho3, kernel)

        w_low_flat = low.permute(0, 2, 1, 3).reshape(3 * self.out_channels, 3 * self.in_channels)
        w3_flat = w3.permute(0, 2, 1, 3).reshape(3 * self.out_channels, 3 * self.in_channels)
        return w_low_flat, w3_flat

    def _fourier_weights_from_spatial_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kernel is None:
            raise RuntimeError("Spatial kernel is not configured for this PlatonicLinear backend")
        return self._fourier_weights_from_kernel_tensor(self.kernel)

    def _fourier_weights_from_direct_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.w1.new_zeros(3, 3, self.out_channels, self.in_channels)
        low[0, 0] = self.w1
        low[1, 1] = self.w2_re
        low[1, 2] = -self.w2_im
        low[2, 1] = self.w2_im
        low[2, 2] = self.w2_re
        w_low_flat = low.permute(0, 2, 1, 3).reshape(3 * self.out_channels, 3 * self.in_channels)
        return w_low_flat, self.w3

    @torch.no_grad()
    def set_spatial_parameters_(self, kernel: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        if tuple(kernel.shape) != (self.G, self.out_channels, self.in_channels):
            raise ValueError(
                f"Expected kernel shape {(self.G, self.out_channels, self.in_channels)}, got {tuple(kernel.shape)}"
            )
        if self.linear_backend == "fourier_direct":
            low_flat, w3 = self._fourier_weights_from_kernel_tensor(kernel.to(device=self.w1.device, dtype=self.w1.dtype))
            low = low_flat.view(3, self.out_channels, 3, self.in_channels).permute(0, 2, 1, 3)
            self.w1.copy_(low[0, 0])
            self.w2_re.copy_(low[1, 1])
            self.w2_im.copy_(low[2, 1])
            self.w3.copy_(w3)
        else:
            if self.kernel is None:
                raise RuntimeError("Spatial kernel is not configured for this PlatonicLinear backend")
            self.kernel.copy_(kernel.to(device=self.kernel.device, dtype=self.kernel.dtype))
        if bias is not None:
            if self.bias is None:
                raise ValueError("Cannot set bias on a bias-free PlatonicLinear")
            if tuple(bias.shape) != (self.out_channels,):
                raise ValueError(f"Expected bias shape {(self.out_channels,)}, got {tuple(bias.shape)}")
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))

    def _fourier_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.linear_backend == "fourier_direct":
            return self._fourier_weights_from_direct_parameters()
        return self._fourier_weights_from_spatial_kernel()

    def _forward_fourier_tetra(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim {self.in_features}, got {x.shape[-1]}")
        leading_shape = x.shape[:-1]
        flat_tokens = x.numel() // self.in_features
        x_group = x.reshape(flat_tokens, self.G, self.in_channels)
        basis = self.fourier_basis.to(device=x.device, dtype=x.dtype)
        x_fourier = torch.matmul(x_group.transpose(1, 2), basis).transpose(1, 2)

        w_low, w3 = self._fourier_weights()
        w_low = w_low.to(dtype=x_fourier.dtype)
        w3 = w3.to(dtype=x_fourier.dtype)

        x_low = x_fourier[:, :3, :].contiguous().view(flat_tokens, 3 * self.in_channels)
        y_low = F.linear(x_low, w_low).view(flat_tokens, 3, self.out_channels)

        x3 = x_fourier[:, 3:, :].contiguous().view(flat_tokens * 3, 3 * self.in_channels)
        y3 = F.linear(x3, w3).view(flat_tokens, 9, self.out_channels)

        y_fourier = torch.cat([y_low, y3], dim=1)
        y_group = torch.matmul(y_fourier.transpose(1, 2), basis.T).transpose(1, 2)
        if self.bias is not None:
            y_group = y_group + self.bias.to(device=y_group.device, dtype=y_group.dtype)
        return y_group.reshape(*leading_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear_backend in {"fourier", "fourier_direct"}:
            return self._forward_fourier_tetra(x)
        output = F.linear(x, self.get_weight(), None)
        if self.bias is None:
            return output
        output_shape = output.shape
        output = output.view(*output_shape[:-1], self.G, self.out_channels)
        output = output + self.bias
        return output.view(output_shape)
