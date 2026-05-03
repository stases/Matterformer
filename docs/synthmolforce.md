# SynthMolForce

SynthMolForce is a synthetic energy-and-force dataset for quick atomistic sanity checks. It does not use quantum chemistry labels. Each sample is generated from deterministic fake atom-type tables and a differentiable PyTorch potential, then forces are computed as:

```text
F_i = -dE / dr_i
```

## Levels

- `v0`: smooth pair radial potential only.
- `v1`: `v0` plus local soft coordination preferences.
- `v2`: `v1` plus local angle-moment energy.
- `v3`: `v2` plus a rotation-invariant, reflection-sensitive chirality term.

The pair term supports:

- `pair_mode="complete"`: all atom pairs contribute.
- `pair_mode="cutoff"`: pair contributions are multiplied by a smooth cutoff.

## Dataset Usage

```python
from torch.utils.data import DataLoader

from matterformer.data import (
    SynthMolForceConfig,
    SynthMolForceDataset,
    collate_synthmolforce,
)

config = SynthMolForceConfig(level="v2", pair_mode="cutoff", num_atoms=16, length=10_000)
dataset = SynthMolForceDataset("data/synthmolforce", split="train", config=config, mode="online")
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_synthmolforce)

batch = next(iter(loader))
print(batch.atom_types.shape, batch.coords.shape, batch.energy.shape, batch.forces.shape)
```

For online data, call `dataset.set_epoch(epoch)` from the training loop if you want fresh deterministic samples each epoch for the same index order.

## Batched Generation

For faster label generation, especially on larger fixed-size systems, use the batched helper. It samples deterministic structures for the requested indices, pads them if needed, and computes all energies and forces in one autograd pass on the requested device:

```python
from matterformer.data import generate_synthmolforce_batch

batch = generate_synthmolforce_batch(
    [0, 1, 2, 3],
    split="train",
    epoch=0,
    config=config,
    device="cuda",  # or "cpu"
)
```

The lower-level functions `compute_synthmolforce_energy_batch` and `compute_synthmolforce_labels_batch` accept padded tensors plus `pad_mask`, so they can be used with existing collated batches.

## Fixed Cache

Materialize fixed train/val/test splits with:

```bash
python scripts/prepare_synthmolforce_data.py \
  --root data/synthmolforce \
  --level v2 \
  --pair-mode cutoff \
  --num-atoms 16
```

The default split sizes are:

- train: `10_000`
- val: `2_000`
- test: `2_000`

Files are written to:

```text
data/synthmolforce/<config_name>/train.pt
data/synthmolforce/<config_name>/val.pt
data/synthmolforce/<config_name>/test.pt
```

Load fixed splits with:

```python
config = SynthMolForceConfig(level="v2", pair_mode="cutoff", num_atoms=16)
dataset = SynthMolForceDataset(
    "data/synthmolforce",
    split="train",
    config=config,
    mode="fixed",
)
```

For variable-size samples, pass a range:

```bash
python scripts/prepare_synthmolforce_data.py --num-atoms 8:20
```

## Batch Fields

`collate_synthmolforce` returns `SynthMolForceBatch` with:

- `atom_types`: `[B, max_N]`
- `coords`: `[B, max_N, 3]`
- `forces`: `[B, max_N, 3]`
- `energy`: `[B]`
- `component_energies`: `pair`, `coord`, `angle`, `chiral`, each `[B]`
- `pad_mask`, `num_atoms`, `indices`

Padded coordinates and forces are zeroed, and `node_features()` returns padded one-hot atom-type features.
