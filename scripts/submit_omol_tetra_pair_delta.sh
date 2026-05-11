#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

job_t="$(sbatch "$SCRIPT_DIR/omol_tetra_t_only_h1008_l8_delta.sh")"
job_sgt="$(sbatch "$SCRIPT_DIR/omol_tetra_sgt_h1008_l8_delta.sh")"

echo "$job_t"
echo "$job_sgt"
