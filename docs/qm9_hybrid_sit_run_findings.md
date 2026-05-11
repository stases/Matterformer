# QM9 Hybrid SIT Run Findings

Date: 2026-05-09  
Branch: `modular-matterformer`  
Project: `Matterformer_QM9_hybrid`

## Short Conclusion

The good run is the scalar SIT baseline:

- Job `165321`: `matterformer_qm9_scalar_sit_d768_l8_165321`
- Config: `configs/qm9_hybrid/scalar_sit_d768_l8.json`
- Schedule: `S,I` repeated 8 times
- Status after about 26k steps: `val/loss = 1.9909`, `val/coord_loss = 0.7615`

The two bad runs were the tetra-replacement sidecar variants:

- Job `165322`: `matterformer_qm9_sit_sidecar_sparse_1152eff_165322`
- Job `165323`: `matterformer_qm9_sit_sidecar_balanced_1152eff_165323`
- Both were cancelled with `scancel 165322 165323`
- Both stayed near `val/loss ~= 6.96-6.97`, with `val/coord_loss ~= 1.32-1.33`

The likely failure mode is that tetra blocks replaced scalar global blocks. Since scalar-to-group and group-to-scalar gates start at zero, those tetra blocks do not help the scalar coordinate head early in training. The bad runs effectively removed long-range scalar communication from the coordinate path.

## Run Comparison

| Job | Run | Config | Schedule counts | Status | Key metrics |
| --- | --- | --- | --- | --- | --- |
| `165321` | `matterformer_qm9_scalar_sit_d768_l8_165321` | `configs/qm9_hybrid/scalar_sit_d768_l8.json` | `8S / 8I / 0T` | running | `val/loss=1.9909`, `val/coord_loss=0.7615`, `train/loss=1.8557` |
| `165322` | `matterformer_qm9_sit_sidecar_sparse_1152eff_165322` | `configs/qm9_hybrid/sit_sidecar_sparse_1152eff.json` | `8S / 6I / 2T` | cancelled | `val/loss=6.9722`, `val/coord_loss=1.3208`, `train/loss=7.2151` |
| `165323` | `matterformer_qm9_sit_sidecar_balanced_1152eff_165323` | `configs/qm9_hybrid/sit_sidecar_balanced_1152eff.json` | `8S / 4I / 4T` | cancelled | `val/loss=6.9603`, `val/coord_loss=1.3331`, `train/loss=6.5596` |
| `165333` | `matterformer_qm9_sit_sidecar_additive_narrow_960eff_165333` | `configs/qm9_hybrid/sit_sidecar_additive_narrow_960eff.json` | `8S / 8I / 2T` | running | new replacement run |

Legend:

- `S`: compact local kNN simplicial layer.
- `I`: global scalar Transformer/MHA layer with Matterformer distance bias.
- `T`: global tetrahedral Platonic layer.

## Why The Two Tetra Runs Were Bad

The bad configs were not additive sidecars. They replaced scalar global attention with tetra layers:

```text
sparse:   S,I  S,I  S,T  S,I  S,I  S,I  S,T  S,I
balanced: S,I  S,T  S,I  S,T  S,I  S,T  S,I  S,T
```

With zero-initialized coupling gates:

```text
scalar -> group gate = 0
group -> scalar gate = 0
```

the tetra branch starts as an isolated adapter. That is normally desirable for stability, but it also means a tetra layer cannot replace scalar long-range communication at initialization. The existing EDM coordinate head reads the scalar stream, so removing scalar global layers hurts the main coordinate denoising path.

The metric pattern supports this:

- Atom losses are similar across runs, around `0.015-0.017`.
- Coordinate losses separate sharply.
- The bad runs are stuck around `val/coord_loss ~= 1.32`, while the scalar SIT run is around `0.76`.

So the failure is not a total training crash. It is specifically a coordinate/geometry communication failure.

## Replacement Run

The new run keeps scalar global attention in every block and adds a narrow tetra sidecar after scalar global attention in only two blocks:

```text
S,I
S,I
S,I,T
S,I
S,I
S,I
S,I,T
S,I
```

That gives:

```text
8 local simplicial layers
8 scalar global layers
2 tetra global sidecar layers
```

The new Slurm job is:

```bash
sbatch scripts/qm9_edm_hybrid_sidecar_additive_narrow_960eff_delta.sh
```

Current job:

```text
165333  qm9_sit_addT  RUNNING
```

The config is:

```json
{
  "num_blocks": 8,
  "block_mix": [1, 0, 1],
  "order_policy": "explicit",
  "explicit_orders": [
    ["simplicial", "trivial"],
    ["simplicial", "trivial"],
    ["simplicial", "trivial", "tetra"],
    ["simplicial", "trivial"]
  ],
  "branch_mode": "two_stream",
  "scalar_dim": 768,
  "tetra_dim_per_frame": 16,
  "d_model_total": 960
}
```

The important differences from the bad runs:

- The scalar path still has `8` global all-to-all layers.
- Tetra is additive, not a replacement.
- Tetra effective width is narrow: `12 * 16 = 192`.
- Coupling remains gated and zero-initialized.
- Coupling happens after tetra layers, not after every macro-block.

## Architecture Notes

The current recommended QM9 EDM architecture is:

```text
main path:
  scalar local simplicial
  scalar all-to-all global MHA with distance bias

side path:
  narrow tetra Platonic attention
  zero-init gated scalar/group coupling
```

The scalar stream is:

```text
h: [B, T, C_s]
```

with `C_s = d_model = 768` for these runs.

The tetra stream is:

```text
F: [B, T, 12, C_g]
```

with `C_g = 16` for the new additive run.

## Compact kNN Simplicial Status

The local simplicial layer uses compact neighbor tensors:

```text
neighbor_idx:  [B, N, K]
neighbor_mask: [B, N, K]
rel:           [B, N, K, 3]
dist:          [B, N, K]
unit:          [B, N, K, 3]
pair_mask:     [B, N, K, K]
```

Current `K = 32`.

Important implementation detail:

- Config value: `"kernel": {"backend": "triton_knn"}`
- Current behavior: the `triton_knn` entry point delegates to the differentiable PyTorch reference implementation.
- Meaning: the runs are using compact kNN simplicial attention, but not yet a custom Triton kernel body.

This is correct for parity and first experiments, but it is not the final performance kernel.

## Rotation-Invariant Local Bias Fix

The local simplicial bias was changed to avoid an arbitrary MLP over raw unit-vector coordinates.

Unsafe version:

```python
edge_features = torch.cat([rbf, unit], dim=-1)
angle_left = mlp(edge_features)
angle_right = mlp(edge_features)
```

Safe version now used:

```text
edge scalar features = RBF(distance)
MLPs produce scalar coefficients a_{ij,l,t}, b_{ij,l,t}
fixed spherical/STF basis blocks produce Phi_l(unit)
angle_left/right = coefficients * Phi_l(unit)
angle score = <angle_left_ij, angle_right_ik>
```

This makes the scalar local angle term a learned spherical/Legendre filter rather than a global-coordinate MLP.

Files to inspect:

- `src/matterformer/models/hybrid.py`
  - `CompactSimplicialGeometryBias`
  - `_spherical_basis_lmax2`
  - `_expand_spherical_coefficients`
  - `compact_simplicial_attention_torch`
  - `compact_simplicial_attention_triton`
- `tests/test_hybrid.py`
  - `test_compact_simplicial_geometry_bias_is_rotation_invariant`

## Code Map

Core architecture:

- `src/matterformer/models/hybrid.py`
  - `HybridConfig`
  - `ModelState`
  - `GeometryCache`
  - `HybridTransformerTrunk`
  - `HybridBlock`
  - `expand_hybrid_schedule`
  - `SimplicialLocalLayer`
  - `TrivialGlobalLayer`
  - `TetraPlatonicGlobalLayer`
  - `StreamCoupling`

Platonic components:

- `src/matterformer/models/platonic/groups.py`
- `src/matterformer/models/platonic/linear.py`
- `src/matterformer/models/platonic/rope.py`
- `src/matterformer/models/platonic/block.py`

QM9 integration:

- `src/matterformer/models/qm9.py`
- `scripts/train_qm9_edm.py`
- `scripts/qm9_edm_hybrid_delta.sh`

Experiment configs:

- Good scalar baseline: `configs/qm9_hybrid/scalar_sit_d768_l8.json`
- Bad sparse tetra replacement: `configs/qm9_hybrid/sit_sidecar_sparse_1152eff.json`
- Bad balanced tetra replacement: `configs/qm9_hybrid/sit_sidecar_balanced_1152eff.json`
- New additive tetra sidecar: `configs/qm9_hybrid/sit_sidecar_additive_narrow_960eff.json`

Launchers:

- `scripts/qm9_edm_hybrid_scalar_sit_d768_delta.sh`
- `scripts/qm9_edm_hybrid_sidecar_sparse_1152eff_delta.sh`
- `scripts/qm9_edm_hybrid_sidecar_balanced_1152eff_delta.sh`
- `scripts/qm9_edm_hybrid_sidecar_additive_narrow_960eff_delta.sh`

## Commands Run

Cancelled the two bad jobs:

```bash
scancel 165322 165323
```

Submitted the replacement:

```bash
sbatch scripts/qm9_edm_hybrid_sidecar_additive_narrow_960eff_delta.sh
```

Validated the hybrid test subset:

```bash
PYTHONPATH=src pytest -q tests/test_hybrid.py
```

Result:

```text
25 passed, 3 warnings
```

## Next Things To Watch

Watch the new job `165333` early. If it is behaving like the scalar baseline, the coordinate loss should move toward the scalar run rather than staying near `1.3`.

Useful check:

```bash
squeue -j 165321,165333
```

W&B run names:

```text
matterformer_qm9_scalar_sit_d768_l8_165321
matterformer_qm9_sit_sidecar_additive_narrow_960eff_165333
```

If the additive run is still bad, the next suspect is the tetra-to-scalar coupling itself. In that case, run an ablation with `group_to_scalar.enabled=false` while keeping tetra active, to test whether the sidecar is disrupting scalar features once the gate begins learning.

## Follow-Up Implementation Update

After review, three semantic fixes were made:

1. Tetra coupling now runs as:

```text
scalar_to_group_pre(current scalar) -> tetra -> group_to_scalar_post
```

The scalar-to-group pre gate defaults to `1.0`, while the group-to-scalar post
gate remains zero-initialized.

2. Trivial global geometry modes are explicit:

```text
edge_delta_bias = distance bias + raw pair_delta edge MLP
distance_bias   = distance-only scalar bias
none            = no scalar global geometry bias
```

The QM9 EDM configs use `edge_delta_bias` because these are direct-coordinate
performance runs, not strict invariant energy runs.

3. The current local compact simplicial bias is named honestly:

```text
spherical_low_rank
```

`feature_gated_spherical_low_rank` now raises `NotImplementedError` until it
actually consumes scalar hidden features.
