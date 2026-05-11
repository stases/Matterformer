#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/open_mol}"
HYBRID_CONFIG="${HYBRID_CONFIG:-configs/omol/scalar_sit_d768_l8.json}"

PYTHONPATH=src python scripts/train_omol_forcefield.py \
  --train-data-path "${DATA_ROOT}/train_4M" \
  --val-data-path "${DATA_ROOT}/val" \
  --validation-mode heldout \
  --hybrid-config-json "${HYBRID_CONFIG}" \
  --element-refs-json configs/omol/element_refs.json \
  --d-model 768 \
  --n-heads 12 \
  --n-layers 8 \
  --batch-size 32 \
  --max-atoms-per-batch 12000 \
  --max-edges-per-batch 2000000 \
  --normalizer-rmsd 1.433569 \
  --energy-weight 10 \
  --force-weight 10 \
  --train-augmentation o3
