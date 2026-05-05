# SynOMol-Transfer

SynOMol-Transfer is a synthetic molecular energy-and-force dataset for controlled geometry benchmarks. It is not a quantum chemistry dataset. Structures are generated from hidden motif templates and synthetic atom-type tables, then labels are computed from a smooth differentiable oracle that depends only on:

```text
coords, atom_types
```

Forces are conservative:

```text
F_i = -dE_total / dx_i
```

The oracle returns total extensive energies. Normalize energy errors per atom in
training or evaluation code, not inside the oracle before force differentiation.

## Oracle Components

- `pair`: smooth radial pair potential.
- `angle`: explicit triplet term using radial bases and Legendre angular bases.
- `motif`: normalized local preference penalties around linear, trigonal, tetrahedral, square-planar, and acute angular cosines.
- `many_body`: local radial and angular density polynomial.

QEq/global charge and torsion-lite are intentionally not included in V1.
V1 should be treated as a local geometry benchmark. Global charge transfer,
fragment separation, and torsion diagnostics are planned V2 concerns.

The hidden atom table contains alias type pairs with identical radial pair
parameters and different angular/motif preferences. This is intentional: pair
models should not be able to solve every split by memorizing radial behavior.

## Splits

- `train`
- `val`
- `test_iid`
- `test_type_combo`
- `test_motif`
- `test_size`
- `test_perturb`

`test_type_combo` uses held-out center-neighbor type triples. The constraint is
checked against all active soft-bond local triples in the generated structure,
not only the primary hidden motif. Train/val/IID and the other OOD splits reject
samples that contain active held-out type triples.

`test_motif` uses square-planar and octahedral motifs held out from train/val/IID.
Training shell distortions are restricted to training motifs, so square-planar
and octahedral shell geometry does not leak into the IID distribution.

`test_size` uses larger atom counts than the default train/IID range. Large
systems are generated as separated local fragments so force RMS does not grow
just because more atoms fit inside one dense cutoff ball.

## Dataset Usage

```python
from torch.utils.data import DataLoader

from matterformer.data import (
    SynOMolTransferConfig,
    SynOMolTransferDataset,
    collate_synomol_transfer,
)

config = SynOMolTransferConfig(num_atoms=(16, 96), length=10_000)
dataset = SynOMolTransferDataset(
    "data/synomol_transfer",
    split="train",
    config=config,
    mode="online",
)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_synomol_transfer)

batch = next(iter(loader))
print(batch.atom_types.shape, batch.coords.shape, batch.energy.shape, batch.forces.shape)
```

For online data, call `dataset.set_epoch(epoch)` from the training loop if you want fresh deterministic samples each epoch for the same index order.

## Fixed Cache

Materialize all fixed splits with:

```bash
python scripts/prepare_synomol_transfer_data.py --root data/synomol_transfer
```

For quick local checks:

```bash
python scripts/prepare_synomol_transfer_data.py --preset smoke
```

The default materialization preset uses longer relaxation/Langevin trajectories
and larger atom-count ranges than the smoke preset. Component calibration is
enabled by default and uses a robust force-RMS statistic; pass
`--no-calibrate-components` to keep the raw component scales. You can raise
`--relax-steps`, `--relaxation-snapshot-steps`, and `--langevin-steps` for a
heavier offline benchmark.

Available presets:

- `smoke`: small atom-count ranges and shallow dynamics for local checks.
- `default`: fast architecture-iteration scale.
- `full`: larger split sizes and longer relaxation/Langevin trajectories for a heavier offline benchmark.

Use CPU workers during materialization with:

```bash
python scripts/prepare_synomol_transfer_data.py --backend multiprocess-cpu --num-workers 8
```

Use sharded output for large fixed splits:

```bash
python scripts/prepare_synomol_transfer_data.py --preset full --shard-size 10000
```

An experimental offline GPU backend is available for batched label/force
evaluation:

```bash
python scripts/prepare_synomol_transfer_data.py \
  --backend gpu-batched \
  --generation-device cuda \
  --generation-batch-size 32 \
  --k-label 64
```

In `gpu-batched`, proposal generation, held-out repair, filtering, and
trajectory proposal steps remain CPU-side; the expensive final oracle
label/force evaluation is batched on CUDA. A more experimental `gpu-full`
backend moves relaxation and Langevin trajectory steps onto CUDA as well:

```bash
python scripts/prepare_synomol_transfer_data.py \
  --backend gpu-full \
  --generation-device cuda \
  --generation-batch-size 64 \
  --k-label 64
```

GPU local labels are exact by default relative to the scalar CPU oracle:
materialization fails if any center has more active cutoff neighbors than
`--k-label`. Increase `--k-label` or pass `--allow-label-cap-hits` only for an
explicitly capped-oracle experiment. Use `--max-gpu-work` to limit dynamic CUDA
batch size for large atom-count buckets. `gpu-full` uses CUDA relaxation and
Langevin as a backend-specific generator; its cache names include the backend
and `k_label` unless you pass `--config-name`. The online dataset path remains
CPU/per-sample and is intended for debugging or small dynamic runs.

Files are written to:

```text
data/synomol_transfer/<config_name>/<split>.pt
data/synomol_transfer/<config_name>/<split>_stats.json
```

The stats JSON contains rejection counts, attempts per accepted sample,
sample-kind counts, motif counts, atom-count summaries, energy-per-atom
quantiles, force RMS quantiles, component-energy quantiles, per-atom component
energy quantiles, held-out repair counts, and timing.
GPU runs also record label-neighbor statistics and cap-policy metadata.

Load fixed splits with:

```python
dataset = SynOMolTransferDataset(
    "data/synomol_transfer",
    split="test_motif",
    mode="fixed",
    config_name="default",
)
```

Materialization writes `data/synomol_transfer/manifest.json`, so aliases such as
`"smoke"`, `"default"`, and `"full"` resolve to the matching cache directory.
The packed `.pt` file stores the exact calibrated config and overrides the
loader config after it is loaded.

Actual cache directories are hash-based by default. This avoids accidentally
reusing a cache after changing generation fields such as relaxation steps,
atom-count ranges, component scales, or force filters.

## Batch Fields

`collate_synomol_transfer` returns `SynOMolTransferBatch` with:

- `atom_types`: `[B, max_N]`
- `coords`: `[B, max_N, 3]`
- `forces`: `[B, max_N, 3]`
- `energy`: `[B]`
- `component_energies`: `pair`, `angle`, `motif`, `many_body`
- `pad_mask`, `num_atoms`, `indices`
- diagnostics: `system_ids`, `sample_kinds`, `primary_motifs`, `primary_triples`, `has_heldout_type_combo`, `motif_labels`

`node_features()` returns padded one-hot atom-type features only. Diagnostic fields are never used as model inputs by the batch helper.

## Evaluation Convention

Models should predict total energy. A typical loss normalizes only the energy
error:

```text
L_E = |E_pred - E| / N
L_F = mean_i |F_pred_i - F_i|
```

Forces should be computed from the model total energy:

```text
F_pred_i = -dE_pred / dx_i
```
