"""Shared constants for the pipeline. Change values here, never hardcode elsewhere."""

from __future__ import annotations

from pathlib import Path

# Filesystem layout
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = SCRIPTS_DIR / "data"
TRAJ_DIR = DATA_DIR / "trajectories"
GLITCHED_DIR = DATA_DIR / "glitched"
SURPRISE_DIR = DATA_DIR / "surprise"
DEBUG_DIR = DATA_DIR / "debug"
CHECKPOINTS_DIR = SCRIPTS_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINTS_DIR / "pusht_lewm.pt"
SCHEMA_FILE = SCRIPTS_DIR / "schema" / "quiz-data.schema.json"
PUBLIC_CLIPS_DIR = REPO_ROOT / "public" / "clips"
PUBLIC_QUIZ_JSON = REPO_ROOT / "public" / "quiz-data.json"

# Environment / rollouts
ENV_NAME = "pusht"
ENV_FPS = 10  # Push-T step rate
CLIP_DURATION_SEC = 3
CLIP_STEPS = ENV_FPS * CLIP_DURATION_SEC  # T = 30
DEFAULT_N_ROLLOUTS = 80
QUICK_N_ROLLOUTS = 5
BASE_SEED = 20260417

# Rendering
RENDER_SIZE = 96  # native env render
UPSCALE_SIZE = 384  # nearest-neighbour target

# Glitch families
GLITCH_TELEPORT_SHARE = 0.60
GLITCH_TIMEREV_SHARE = 0.40
TELEPORT_MIN_OFFSET_PX = 6
TELEPORT_MAX_OFFSET_PX = 24
TIMEREV_SEGMENT_MIN = 3
TIMEREV_SEGMENT_MAX = 5
GLITCH_INJECTION_FRAME_MIN = 5  # don't inject too early
GLITCH_INJECTION_FRAME_MAX = CLIP_STEPS - 5  # don't inject in the last steps

# Curation (applied to surprise peak ratio = max / median)
TIER_HARD_RATIO_MAX = 3.5
TIER_MEDIUM_RATIO_MIN = 3.5
TIER_MEDIUM_RATIO_MAX = 8.0
TIER_EASY_RATIO_MIN = 8.0
TARGET_PAIRS = {"easy": 9, "medium": 12, "hard": 6, "gotcha": 3}
GOTCHA_MAX_RATIO = 1.5  # gotcha clips must not spike

# Acceptance thresholds (referenced in export.py self-check)
PEAK_OVER_MEDIAN_MIN = 3.0
NON_GOTCHA_PEAK_PASS_RATE_MIN = 0.80
CLIP_MAX_BYTES = 250_000
