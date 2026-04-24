#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
PYTHON_BIN_DEFAULT="${PYTHON_BIN_DEFAULT:-python3}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

if [ "$FORCE_RECREATE" = "1" ] && [ -d "$VENV_PATH" ]; then
  rm -rf "$VENV_PATH"
fi

if [ ! -x "$VENV_PATH/bin/python" ]; then
  "$PYTHON_BIN_DEFAULT" -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_PATH/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0"
"$VENV_PATH/bin/python" -m pip install "triton==3.4.0"
"$VENV_PATH/bin/python" -m pip install -e "$REPO_ROOT[chem,dev]"

echo "[info] Triton environment ready at $VENV_PATH"
echo "[info] Python: $("$VENV_PATH/bin/python" -c 'import sys; print(sys.executable)')"
echo "[info] Torch/Triton: $("$VENV_PATH/bin/python" -c 'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)')"
