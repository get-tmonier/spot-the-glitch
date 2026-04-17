#!/usr/bin/env bash
set -euo pipefail

GDRIVE_FILE_ID="<GDRIVE_FILE_ID>"
CHECKPOINT_PATH="checkpoints/pusht_lewm.pt"

echo "[setup] 1/4 verifying uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] ERROR: uv not found. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "[setup] 2/4 syncing environment with uv"
uv sync

echo "[setup] 3/4 downloading pretrained checkpoint"
mkdir -p checkpoints
if [[ -f "$CHECKPOINT_PATH" ]]; then
  echo "[setup] checkpoint already present at $CHECKPOINT_PATH — skipping"
else
  if [[ "$GDRIVE_FILE_ID" == "<GDRIVE_FILE_ID>" ]]; then
    echo "[setup] ERROR: GDRIVE_FILE_ID not set in setup.sh." >&2
    echo "[setup] See https://github.com/lucas-maes/le-wm README for the Push-T checkpoint link." >&2
    exit 1
  fi
  uv run gdown --id "$GDRIVE_FILE_ID" -O "$CHECKPOINT_PATH" || {
    echo "[setup] ERROR: checkpoint download failed." >&2
    echo "[setup] Verify the GDRIVE_FILE_ID and upstream availability." >&2
    exit 1
  }
fi

echo "[setup] 4/4 running smoke test"
uv run python -c "
from pipeline.lewm_loader import smoke_test
import sys
sys.exit(0 if smoke_test() else 1)
" || {
  echo "[setup] WARNING: smoke test failed on MPS."
  echo "[setup] The pipeline will fall back to CPU. If you want MPS, check which op failed"
  echo "[setup] and open an issue at https://github.com/lucas-maes/le-wm"
}

echo "[setup] done. Next: uv run poe all"
