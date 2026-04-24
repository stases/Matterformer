# Matterformer

Minimal active repo for shared periodic and non-periodic atomistic transformers.

## Layout

- `src/matterformer`: active package
- `scripts`: thin training and sampling entrypoints
- `tests`: unit and smoke tests for adapters, trunk, QM9, and GEOM-Drugs paths
- `legacy`: frozen imported research code kept for reference only

## Active Scope

- Shared dense-token transformer trunk with standard multi-head or 2-simplicial attention
- Geometry adapters for periodic and non-periodic inputs
- Shared QM9 data pipeline
- Native GEOM-Drugs data preparation, evaluation, and EDM generation
- QM9 regression
- QM9 unconditional EDM generation

## Data Layout

Managed local datasets live under `data/` and should stay namespaced by dataset family:

- `data/qm9`: QM9 raw and processed files
- `data/geom_drugs`: GEOM-Drugs raw, cleaned, processed, and cache files
- `data/mofs/<dataset_name>`: MOF datasets and related manifests

For preprocessed AtomMOF datasets, use:

- `python scripts/download_atommof_data.py --dataset bwdb`
- `python scripts/download_atommof_data.py --dataset odac25`

## GEOM-Drugs

Managed local GEOM-Drugs data lives under `data/geom_drugs`:

- `raw/`: downloaded MiDi split pickles
- `cleaned/`: Isayevlab-style cleaned split pickles
- `processed/`: Matterformer-native processed tensors
- `cache/`: novelty-reference SMILES caches

Main entrypoints:

- `python scripts/prepare_geom_drugs_data.py`
- `python scripts/evaluate_geom_drugs.py --input /abs/path/to/generated_samples.pkl`
- `python scripts/train_geom_drugs_edm.py`
