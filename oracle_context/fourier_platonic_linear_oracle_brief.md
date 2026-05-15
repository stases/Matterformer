# Oracle Brief: Fourier-Domain PlatonicLinear for Matterformer

## Repository Snapshot

- Repository: `/home/thadziv/GitHub/Matterformer`
- Current commit at bundle creation: `f489533`
- Python package: `matterformer`
- Main target module: `src/matterformer/models/platonic/linear.py`
- Current date: 2026-05-15

## User Goal

Implement a Fourier-domain version of the finite-group equivariant linear layer used by the tetra/Platonic trunk.

The current code uses a spatial finite-group convolution implemented by expanding a small group kernel into a dense flattened matrix and calling `torch.nn.functional.linear`. For the tetrahedral group, this is correct but likely too expensive at large hidden sizes such as:

- `G = 12`
- `C_per_frame = 160`
- `d_model = 1920`
- `heads_per_frame = 5`
- `total_heads = 60`
- `head_dim = 32`
- `use_key = false`
- `ffn_mult = 2`

The desired implementation should be mathematically equivalent to the existing spatial `PlatonicLinear` at first, behind a config/API switch, then optimized.

## Current State

`PlatonicLinear` stores:

```python
self.kernel = nn.Parameter(torch.empty(self.G, self.out_channels, self.in_channels))
```

Inputs and outputs are flattened as `[..., G * C]`.

The dense weight is built by:

```python
out_group = torch.arange(self.G).view(self.G, 1)
in_group = torch.arange(self.G).view(1, self.G)
inv_in = self.inverse_indices[in_group]
kernel_group = self.cayley_table[inv_in, out_group]
expanded = self.kernel[kernel_group]
weight = expanded.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
```

So the current relative group pose index is:

```text
kernel_group = inverse(in_group) * out_group
```

`forward()` then does:

```python
output = F.linear(x, self.get_weight(), None)
```

and adds one shared bias per output channel, broadcast across group frames.

## Important Distinction

The repo contains coordinate Fourier/RFF embeddings. Those are unrelated. The missing piece is finite-group Fourier transforms for `PlatonicLinear` / group convolution over the tetrahedral group axis.

The repo also contains Triton attention backends. Those only affect attention. They do not optimize `PlatonicLinear`.

## Required Implementation Shape

Please propose a clear implementation that can be integrated later. Prefer a conservative drop-in path:

1. Keep the existing spatial `PlatonicLinear` as default.
2. Add a Fourier backend or a new `FourierPlatonicLinear` that is mathematically equivalent.
3. Expose it through a small config option, for example:

```json
"tetra": {
  "linear_backend": "spatial"
}
```

or:

```json
"tetra": {
  "linear_backend": "fourier"
}
```

4. Thread that option into all `PlatonicLinear` call sites in:
   - global tetra attention projections
   - global tetra FFN
   - group-framewise simplicial projections/MLP when `projection_mode="group_linear"`
   - tetra input lift and readout if applicable
   - OMol/QM9 Platonic readouts if applicable
5. Add parity tests proving equivalence to the current spatial implementation.
6. Add equivariance tests for the Fourier backend.
7. Keep CPU fallback working. GPU optimization can come after correctness.

## Relevant Current Call Sites

Main global tetra block:

- `src/matterformer/models/platonic/layers.py`
  - `PlatonicAttention.q_proj`
  - `PlatonicAttention.v_proj`
  - optional `PlatonicAttention.k_proj`
  - `PlatonicAttention.out_proj`
  - `PlatonicBlock.linear1`
  - `PlatonicBlock.linear2`

Hybrid/tetra trunk:

- `src/matterformer/models/hybrid.py`
  - `GroupFramewiseSimplicialAttention.in_proj`
  - `GroupFramewiseSimplicialAttention.out_proj`
  - `GroupFramewiseSimplicialLayer.mlp`
  - `HybridTrunk.group_input_proj`
  - `HybridTrunk.group_readout_proj`
  - `TetraPlatonicGlobalLayer` construction of `PlatonicBlock`

Model readouts:

- `src/matterformer/models/omol.py`
- `src/matterformer/models/qm9.py`

## Correctness Expectations

The Fourier implementation must match the spatial implementation to numerical tolerance when initialized from the same spatial kernel and bias.

Required behavior:

- Input shape remains `[..., G * Cin]`.
- Output shape remains `[..., G * Cout]`.
- Bias semantics remain one bias vector of shape `[Cout]`, shared over group frames.
- State dict compatibility should be considered. Ideally existing checkpoints using `.kernel` and `.bias` remain loadable, or there is a documented conversion path.
- The implementation must preserve equivariance under the same group-axis permutations used in current tests.
- Autograd must work.
- Dtypes should follow existing PyTorch behavior. Avoid hard-coding float64 in model forward.

## Tetra Fourier Math Hints

For the tetrahedral rotation group `A4`, order 12:

- Irreps over the complex numbers: three 1D irreps and one 3D irrep.
- Over the real representation used by the model, the two nontrivial complex 1D irreps combine into a real 2D block.
- The regular representation decomposes as:

```text
regular = 1 + 1' + 1'' + 3 + 3 + 3
```

or in real terms:

```text
regular = scalar 1D + real 2D block + three copies of vector 3D block
```

The paper/comment motivating this work says a tetra `12C -> 12C` equivariant linear can reduce from roughly `144 C^2` spatial multiply terms to about `36 C^2` using Fourier block structure. That suggests the useful real block sizes are `1`, `2`, and `3`, with the `3D` irrep appearing with multiplicity 3.

Please verify the exact orientation/convention against the current spatial kernel indexing:

```text
kernel_group = inverse(in_group) * out_group
```

Do not silently change the equivariance convention.

## Suggested Test Cases

Add focused tests, likely in `tests/test_hybrid.py` or a new `tests/test_platonic_linear.py`:

1. `test_fourier_platonic_linear_matches_spatial_tetra`
   - create spatial and Fourier layers
   - copy/convert the same kernel and bias
   - compare outputs for random `x`
   - include backward comparison if feasible

2. `test_fourier_platonic_linear_equivariance_tetra`
   - same group permutation test as existing `test_platonic_group_and_linear_equivariance`

3. `test_platonic_block_fourier_backend_shape_backward`
   - instantiate `PlatonicBlock(..., linear_backend="fourier")`
   - forward/backward smoke test

4. Config construction smoke test:
   - instantiate tetra hybrid config with `tetra.linear_backend="fourier"`

5. Optional benchmark:
   - compare spatial vs Fourier forward/backward for `G=12`, `C=160`, batch/token count representative of OMol.

## Acceptance Criteria

- Existing tests continue passing with default spatial backend.
- New Fourier backend tests pass.
- Fourier backend is opt-in by config.
- No dense `[N, N, ...]` attention/bias materialization is introduced.
- No unrelated refactor.
- Implementation is readable and isolated enough that a later Triton/cuBLAS optimization can replace the internal Fourier block matmuls without changing model-level APIs.

## Files Included In This Bundle

The zip includes the source/config/test files needed to understand current behavior and implement the Fourier backend later. See `MANIFEST.txt` in the zip for the exact file list.
