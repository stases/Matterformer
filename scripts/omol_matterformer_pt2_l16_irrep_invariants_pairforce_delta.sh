#!/bin/bash
# Matterformer PT2 exact-style OMol run with tetra irrep invariant scalar readout and pair-force residual.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=mf_l16_irinv_pair
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_l16_irinv_pair.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"

export TETRA_IRREP_SCALAR_INPUT="${TETRA_IRREP_SCALAR_INPUT:-invariants}"
export TETRA_PAIR_FORCE_MODE="${TETRA_PAIR_FORCE_MODE:-residual}"
export TETRA_PAIR_K_NEIGHBORS="${TETRA_PAIR_K_NEIGHBORS:-30}"
export TETRA_PAIR_FEATURE_DIM="${TETRA_PAIR_FEATURE_DIM:-128}"
export TETRA_PAIR_ELEMENT_DIM="${TETRA_PAIR_ELEMENT_DIM:-32}"
export TETRA_PAIR_GATE_INIT="${TETRA_PAIR_GATE_INIT:-0.0}"
export TETRA_PAIR_GEOMETRY_STRICT="${TETRA_PAIR_GEOMETRY_STRICT:-0}"

export RUN_SLUG="${RUN_SLUG:-matterformer-pt2-h1920-l16-ls1e-4-ffn2-actsin-ractsin-rs2.0-ema0.99-wd1e-5-muonffnconv0.01-mwd0-warm5k-auxadamw-40ep-n12000-flat-maxg999999-irrepinv-pairforce30-skip1000}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_pt2_l16_irrepinv_pairforce30_muonffnconv0p01_mwd0_warm5k_auxadamw_skip1000}"

exec "$REPO_ROOT/scripts/omol_matterformer_pt2_l16_irrep_delta.sh"
