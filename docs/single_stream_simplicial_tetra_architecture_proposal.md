# Single-Stream Simplicial + Global Architecture Proposal

Date: 2026-05-11  
Repo: `Matterformer`  
Branch context: `modular-matterformer`

## Request For Review

Please review the included Matterformer hybrid implementation and advise on a
cleaner architecture direction.  The current dual-stream scalar/tetra sidecar
design has been hard to reason about empirically.  The proposed replacement is
a single-stream scheduler with two separate model families:

```text
Scalar family:
    S + I

Tetra family:
    S_g + T
```

where:

- `S` is local compact kNN simplicial attention.
- `I` is scalar global Matterformer MHA / invariant global attention.
- `T` is tetrahedral Platonic global attention.
- `S_g` is a proposed group-framewise local simplicial layer operating on
  tetra group features.

The key question: should we remove the scalar/tetra dual trunk coupling as the
default and instead build either a scalar single stream (`S+I`) or a tetra
single stream (`S_g+T`)?

## Why We Are Reconsidering The Dual Stream

The original hybrid stack kept:

```text
scalar stream: h_i in R^C
tetra stream:  F_i(g) in R^C_g, g in G_tet, |G_tet|=12
```

The streams were coupled by scalar-to-group and group-to-scalar gates.  This
created useful ablations, but it also added several failure modes:

- Tetra blocks could replace active scalar global communication while their
  feedback to scalar was zero-gated.
- The tetra branch could see stale scalar information unless explicitly
  pre-refreshed.
- Group-to-scalar mean pooling is weak for coordinate denoising.
- Attribution became unclear: better results could come from extra parameters,
  scalar global layers, tetra layers, or coupling dynamics.

Empirically, the scalar SIT run worked best among early hybrid experiments:

```text
S,I repeated 8 times
```

The replacement tetra sidecar variants were much worse, especially on coordinate
loss.  See `docs/qm9_hybrid_sit_run_findings.md` for details.

## Proposed Architecture

Keep the macro-block scheduler, but make the stream type explicit:

```python
stream_type = "scalar"  # permits S and I
stream_type = "tetra"   # permits S_g and T
```

The same count-style scheduler can still be used:

```python
# scalar local/global
block_mix = [[1, 0, 1]]

# tetra local/global
block_mix = [[1, 1, 0]]
```

But the meaning of `S` depends on `stream_type`.

## Scalar Family: S + I

The scalar stream is:

```text
h_i in R^C
```

Layer types:

```text
S = compact local scalar simplicial attention
I = scalar global Matterformer MHA / invariant layer
```

This is closest to the successful SIT baseline and to the AllScAIP local +
global lesson.

Recommended scalar schedules:

```text
S,I repeated
```

or, for a pure global baseline:

```text
I repeated
```

The current scalar `S` uses compact kNN attention over neighbor pairs:

```text
(j,k) in N_K(i) x N_K(i)
```

with a spherical low-rank local geometry bias.  The current bias is
geometry-only, not feature-gated:

```text
coefficients = MLP(RBF(distance))
fixed basis  = Phi_l(unit direction)
angle score  = <L_ij, R_ik>
```

This preserves scalar rotation-invariant local geometry in the logit bias.

## Tetra Family: S_g + T

The tetra stream is:

```text
F_i(g) in R^C_g
g in G_tet
|G_tet| = 12
```

Layer types:

```text
S_g = group-framewise local compact simplicial attention
T   = tetra Platonic global attention
```

The proposed `S_g` layer should apply the same local simplicial operation
independently to every tetra frame:

```text
for each g:
    F_i(g) <- F_i(g) + S_local(F_*(g), invariant geometry)
```

The weights must be shared across `g`.  The geometry bias must be invariant
under continuous rotations, for example using only distances and spherical
basis dot products.  Under these conditions, `S_g` commutes with tetra frame
permutations and should preserve tetra equivariance:

```text
S_g(rho(a) F) = rho(a) S_g(F)
```

This is different from the old dual trunk.  There is no scalar backbone and no
sidecar coupling between every layer.  The stream remains group-valued.

Recommended tetra schedules:

```text
S_g,T repeated
```

or:

```text
S_g,S_g,T
```

The user preference is currently one simplicial layer per block where possible,
because two local simplicial layers per block did not look as effective in early
QM9 EDM experiments.

## Implementation Sketch For S_g

Current compact scalar simplicial tensors:

```text
q, k1, v1, k2, v2: [B, H, N, D]
neighbor_idx:       [B, N, K]
neighbor_mask:      [B, N, K]
geometry bias:      [B, H, N, K, ...]
```

For tetra mode, one simple implementation is to fold the group axis into the
batch for the local layer:

```python
# F: [B, N, G, Cg]
x = F.permute(0, 2, 1, 3).reshape(B * G, N, Cg)

# Run one shared CompactSimplicialAttention module on x.
# Repeat/broadcast neighbor_idx, neighbor_mask, and invariant geometry bias
# across G.

y = local_simplicial(x)
F = y.reshape(B, G, N, Cg).permute(0, 2, 1, 3)
```

This should preserve tetra equivariance because:

1. The same module is applied to every frame.
2. There is no frame-specific parameter.
3. The local geometry bias is invariant and independent of frame index.
4. The tetra group action is a permutation of the frame axis.

Please verify if this is sufficient, or whether `S_g` should use explicit
group-constrained linear maps instead of shared per-frame linears.

## Input And Readout Questions

For tetra single-stream QM9 EDM, we still need scalar atom inputs and atom /
coordinate outputs.

Possible input lift:

```text
F_i(g) = W_in atom_i, copied to every g
```

This is a valid trivial representation inside the group stream.

Atom output:

```text
atom_delta_i = W_atom mean_g F_i(g)
```

Coordinate denoising output is more subtle.  A mean-pooled scalar readout may
not expose orientation information.  A group-to-vector readout may be needed:

```text
Delta x_i = sum_g R_g a_i(g)
```

where `a_i(g)` is predicted from `F_i(g)`.  Please advise whether this is the
right readout for a tetra-only QM9 EDM model, and how to initialize/stabilize it.

## Current Triton Status

The package includes the new compact kNN Triton implementation:

- `src/matterformer/models/triton_compact_simplicial_attention.py`
- `scripts/benchmark_compact_simplicial_attention.py`
- `scripts/benchmark_compact_simplicial_delta.sh`
- `docs/compact_simplicial_triton_delta_benchmark.md`

Benchmark summary on an RTX 6000 Ada for QM9-like tensors:

```text
B = 64, H = 12, N = 32, K = 32, D = 64, angle_rank = 32
```

Observed speedups:

```text
forward:        about 16x to 24x faster than Torch reference
forward+backward: about 4.6x to 7.8x faster
memory:         about 4x to 8x lower peak fwd+bwd delta
```

Current constraints:

- `head_dim <= 128`
- `K <= 64`
- low-rank bias/message rank `<= 64`
- precision modes: `bf16_tc`, `tf32`, `ieee_fp32`
- benchmark excludes kNN construction and bias MLPs

The old dense all-token simplicial Triton implementation is also included as
reference:

- `src/matterformer/models/triton_simplicial_attention.py`
- `src/matterformer/models/simplicial_attention.py`

## Files Included In This Package

Core Matterformer files:

- `src/matterformer/models/hybrid.py`
- `src/matterformer/models/qm9.py`
- `src/matterformer/models/transformer.py`
- `src/matterformer/models/regular_attention.py`
- `src/matterformer/models/simplicial_attention.py`
- `src/matterformer/models/triton_compact_simplicial_attention.py`
- `src/matterformer/models/triton_simplicial_attention.py`
- `src/matterformer/models/platonic/*.py`

Configs and launchers:

- `configs/qm9_hybrid/*.json`
- `scripts/qm9_edm_hybrid*.sh`
- `scripts/train_qm9_edm.py`

Tests and benchmarks:

- `tests/test_hybrid.py`
- `tests/test_simplicial_attention.py`
- `scripts/benchmark_compact_simplicial_attention.py`
- `scripts/diagnose_simplicial_triton_parity.py`
- `scripts/validate_simplicial_attention.py`

Reference docs:

- `docs/qm9_hybrid_sit_run_findings.md`
- `docs/compact_simplicial_triton_delta_benchmark.md`
- `docs/modular_matterformer_hybrid_architecture.tex`
- `references/allscaip/README.md`

## Specific Questions

1. Do you agree with replacing the default dual-stream scalar/tetra sidecar
   architecture with two single-stream families: scalar `S+I` and tetra `S_g+T`?

2. Does the proposed group-framewise `S_g` local simplicial layer preserve tetra
   equivariance if all frame weights are shared and the geometry bias is
   invariant?

3. Should `S_g` be implemented as independent shared per-frame attention, or
   should its Q/K/V/O projections be group-constrained linears that mix frames?

4. What is the right tetra-only coordinate denoising readout for QM9 EDM?
   Is `Delta x_i = sum_g R_g a_i(g)` sufficient?

5. Should tetra-only input lifting copy scalar atom features to every frame, or
   should there be a more expressive equivariant lift?

6. What ablation set should be run first?
   Candidate set:

   ```text
   A. I-only MHA RoPE baseline
   B. scalar S+I
   C. T-only Platonic baseline
   D. tetra S_g+T with copied-frame input and group-to-vector readout
   E. tetra S_g+T without group-to-vector readout
   ```

7. Can the current compact Triton kernel be reused for `S_g` by folding the group
   axis into the batch, or are there hidden numerical/performance issues?

8. What exact code-level changes would you make first in `hybrid.py` to support
   `stream_type="tetra"` cleanly without reintroducing sidecar coupling?

