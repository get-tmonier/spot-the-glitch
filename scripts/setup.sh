#!/usr/bin/env bash
set -euo pipefail

# Upstream moved from Google Drive to HuggingFace. The repo ships weights.pt
# (state dict) alongside config.json; we combine them into the dict layout
# lewm_loader.load() expects.
HF_REPO="quentinll/lewm-pusht"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"
CHECKPOINT_PATH="checkpoints/pusht_lewm.pt"

echo "[setup] 1/4 verifying uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] ERROR: uv not found. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "[setup] 2/4 syncing environment with uv"
uv sync

echo "[setup] 3/4 downloading pretrained checkpoint from ${HF_REPO}"
mkdir -p checkpoints
if [[ -f "$CHECKPOINT_PATH" ]]; then
  echo "[setup] checkpoint already present at $CHECKPOINT_PATH — skipping"
else
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  echo "[setup]   fetching weights.pt (72 MB)"
  curl -fL --retry 3 -o "$TMP_DIR/weights.pt" "$HF_BASE/weights.pt"
  echo "[setup]   fetching config.json"
  curl -fL --retry 3 -o "$TMP_DIR/config.json" "$HF_BASE/config.json"
  uv run python - "$TMP_DIR/weights.pt" "$TMP_DIR/config.json" "$CHECKPOINT_PATH" <<'PY'
import json
import sys
import torch

weights_path, config_path, out_path = sys.argv[1:]
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
with open(config_path) as f:
    config = json.load(f)
torch.save({"state_dict": state_dict, "config": config}, out_path)
PY
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
