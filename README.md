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

## OMol Platonic Matterformer Runs

The platonic Matterformer OMol architecture is implemented in this repository.
For `MODEL_BACKEND=matterformer`, the model code, platonic/tetra trunk, OMol
force-field head, run wrapper, and JSON config live here:

- `src/matterformer/models/hybrid.py`
- `src/matterformer/models/omol.py`
- `src/matterformer/models/platonic/`
- `scripts/train_omol_forcefield.py`
- `scripts/omol_forcefield_delta.sh`
- `scripts/omol_matterformer_pt2_l16_delta.sh`
- `configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale.json`
- `configs/omol/element_refs.json`

On a new server, assuming OMol data is already available, clone this repo and
install a CUDA-capable Python environment:

```bash
git clone git@github.com:stases/Matterformer.git
cd Matterformer

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e .
python -m pip install "ase>=3.26.0" "ase-db-backends>=0.10.0" wandb triton
```

The Matterformer backend does not need the FairChem source repository. FairChem
is only needed for `MODEL_BACKEND=allscaip_direct` or AllScAIP baseline runs.
If you use the optional Muon optimizer, install Muon separately and run with
`OPTIMIZER=muon`; the default optimizer is AdamW.

The OMol launcher expects FairChem-style ASE DB shards under directories like:

```text
/path/to/omol/open_mol/train_4M
/path/to/omol/open_mol/val
```

Those directories should contain `.aselmdb` shards readable by ASE plus
`ase-db-backends`.

Submit the L16 platonic run by overriding machine-specific paths:

```bash
REPO_ROOT=/path/to/Matterformer \
PYTHON_BIN=/path/to/env/bin/python \
OMOL_DATA_ROOT=/path/to/omol/open_mol \
OUTPUT_ROOT=/path/to/matterformer_runs/omol_4m \
sbatch scripts/omol_matterformer_pt2_l16_delta.sh
```

For an interactive smoke run without SLURM, use the same environment variables
and call the script with `bash` instead of `sbatch`.
