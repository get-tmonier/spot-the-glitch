# LeWM Local Generation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python pipeline under `scripts/` that produces `/public/clips/*.mp4` and `/public/quiz-data.json` from a pretrained LeWM Push-T checkpoint, matching the spec at `docs/superpowers/specs/2026-04-17-lewm-pipeline-design.md`.

**Architecture:** Single-entrypoint CLI (`generate.py`) with cacheable subcommands (`simulate`, `glitch`, `score`, `curate`, `export`). Each stage reads/writes `scripts/data/`; only `export` writes to `/public/`. Pure-Python stages (glitch, curate, export) are unit-tested via pytest. ML-bound stages (simulate, score, lewm_loader) start with a discovery step that inspects the upstream `stable_worldmodel` package and are hand-smoke-tested via `poe quick`.

**Tech Stack:** Python 3.10, uv (package manager), poethepoet (task runner), ruff (lint/format), pytest, PyTorch (MPS on M3), `stable_worldmodel[env]` (upstream LeWM + Push-T), `typer` (CLI), `imageio[ffmpeg]` (bundled ffmpeg for MP4 encoding), `jsonschema` (contract validation), `gdown` (checkpoint download).

---

## File Map

Files produced by this plan (all under `scripts/` unless noted):

| Path | Task | Responsibility |
|---|---|---|
| `.gitignore` (repo root, modify) | T1 | Ignore `scripts/.venv/`, `scripts/data/`, `scripts/checkpoints/` |
| `scripts/.gitignore` | T1 | Python-specific ignores (pycache, venv, .pytest_cache) |
| `scripts/.python-version` | T1 | Pin Python 3.10 for uv |
| `scripts/pyproject.toml` | T1, T9 | Deps, tool config, poe tasks |
| `scripts/pipeline/__init__.py` | T1 | Package marker |
| `scripts/tests/__init__.py` | T1 | Package marker |
| `scripts/pipeline/config.py` | T2 | Paths, tier thresholds, glitch params, clip timing |
| `scripts/schema/quiz-data.schema.json` | T2 | JSON Schema for the frontend contract |
| `scripts/pipeline/simulate.py` | T3 | Push-T rollouts → `data/trajectories/*.npz` |
| `scripts/pipeline/glitch.py` | T4 | Teleport + time-reversal injection |
| `scripts/tests/test_glitch.py` | T4 | Pure-Python unit tests for glitch logic |
| `scripts/pipeline/lewm_loader.py` | T5 | Device selection + checkpoint loading |
| `scripts/pipeline/score.py` | T6 | LeWM inference → `data/surprise/*.npy` |
| `scripts/pipeline/curate.py` | T7 | Tier selection + pair composition |
| `scripts/tests/test_curate.py` | T7 | Pure-Python unit tests for curation |
| `scripts/pipeline/export.py` | T8 | MP4 encoding + JSON build + acceptance checks |
| `scripts/tests/test_export.py` | T8 | JSON round-trip against schema |
| `scripts/generate.py` | T9 | Typer CLI wiring all subcommands |
| `scripts/setup.sh` | T10 | One-shot env setup + checkpoint + smoke test |
| `scripts/README.md` | T11 | End-to-end usage docs |

---

## Task 1: Scaffolding & Python environment

**Files:**
- Create: `scripts/.gitignore`
- Create: `scripts/.python-version`
- Create: `scripts/pyproject.toml`
- Create: `scripts/pipeline/__init__.py`
- Create: `scripts/tests/__init__.py`
- Modify: `.gitignore` (repo root)

- [ ] **Step 1: Create the scripts folder skeleton**

```bash
mkdir -p scripts/pipeline scripts/tests scripts/schema scripts/data scripts/checkpoints
touch scripts/pipeline/__init__.py scripts/tests/__init__.py
```

- [ ] **Step 2: Pin Python version**

Write `scripts/.python-version`:

```
3.10
```

- [ ] **Step 3: Write scripts/.gitignore**

Write `scripts/.gitignore`:

```
# Python
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/

# uv / venv
.venv/
.python-version.local

# Pipeline cache (regenerable)
data/
checkpoints/
```

- [ ] **Step 4: Update repo-root .gitignore**

Append to `.gitignore` at repo root (after existing entries):

```
# Python pipeline
scripts/.venv/
scripts/data/
scripts/checkpoints/
```

- [ ] **Step 5: Write scripts/pyproject.toml**

Write `scripts/pyproject.toml`:

```toml
[project]
name = "spot-the-glitch-pipeline"
version = "0.1.0"
description = "Local LeWM clip generation pipeline for Spot the Glitch"
requires-python = "==3.10.*"
dependencies = [
  "torch>=2.4",
  "numpy>=1.26",
  "typer>=0.12",
  "imageio[ffmpeg]>=2.34",
  "jsonschema>=4.22",
  "gdown>=5.2",
  "stable-worldmodel[env]",
]

[dependency-groups]
dev = [
  "pytest>=8.2",
  "poethepoet>=0.27",
  "ruff>=0.5",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[tool.poe.tasks]
setup = { shell = "bash setup.sh" }
simulate = "python generate.py simulate"
glitch = "python generate.py glitch"
score = "python generate.py score"
curate = "python generate.py curate"
export = "python generate.py export"
all = ["simulate", "glitch", "score", "curate", "export"]
quick = "python generate.py quick"
test = "pytest"
lint = "ruff check ."
format = "ruff format ."
"format:check" = "ruff format --check ."
verify = ["lint", "format:check", "test"]
```

- [ ] **Step 6: Install and verify the toolchain runs on an empty project**

Run:

```bash
cd scripts
uv sync
uv run poe lint
uv run poe format
uv run poe test
```

Expected:
- `uv sync` creates `.venv/` and installs deps (may take a minute).
- `poe lint` / `poe format` / `poe test` each exit 0 (no files to check yet is acceptable; pytest will say "no tests ran").

- [ ] **Step 7: Commit**

```bash
cd ..
git add scripts/.python-version scripts/.gitignore scripts/pyproject.toml scripts/pipeline/__init__.py scripts/tests/__init__.py .gitignore
git commit -m "scaffold(scripts): initialize Python pipeline project"
```

---

## Task 2: Config constants and JSON Schema

**Files:**
- Create: `scripts/pipeline/config.py`
- Create: `scripts/schema/quiz-data.schema.json`

- [ ] **Step 1: Write pipeline/config.py**

Write `scripts/pipeline/config.py`:

```python
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
```

- [ ] **Step 2: Write the JSON Schema**

Write `scripts/schema/quiz-data.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "spot-the-glitch quiz data",
  "type": "object",
  "required": ["schemaVersion", "generatedAt", "clipFps", "clipDurationSec", "distribution", "pool"],
  "properties": {
    "schemaVersion": { "const": 1 },
    "generatedAt": { "type": "string", "format": "date-time" },
    "clipFps": { "type": "integer", "minimum": 1 },
    "clipDurationSec": { "type": "integer", "minimum": 1 },
    "distribution": {
      "type": "object",
      "required": ["easy", "medium", "hard", "gotcha"],
      "properties": {
        "easy": { "type": "integer", "minimum": 0 },
        "medium": { "type": "integer", "minimum": 0 },
        "hard": { "type": "integer", "minimum": 0 },
        "gotcha": { "type": "integer", "minimum": 0 }
      },
      "additionalProperties": false
    },
    "pool": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/Pair" }
    }
  },
  "$defs": {
    "Pair": {
      "type": "object",
      "required": ["id", "tier", "glitchFamily", "clipA", "clipB"],
      "properties": {
        "id": { "type": "string", "pattern": "^pair_\\d{4}$" },
        "tier": { "enum": ["easy", "medium", "hard", "gotcha"] },
        "glitchFamily": { "enum": ["teleport", "time-reversal", "none"] },
        "clipA": { "$ref": "#/$defs/Clip" },
        "clipB": { "$ref": "#/$defs/Clip" }
      },
      "additionalProperties": false
    },
    "Clip": {
      "type": "object",
      "required": ["src", "isGlitched", "surpriseScore"],
      "properties": {
        "src": { "type": "string", "pattern": "^/clips/pair_\\d{4}_[ab]\\.mp4$" },
        "isGlitched": { "type": "boolean" },
        "surpriseScore": {
          "type": "array",
          "items": { "type": "number", "minimum": 0 },
          "minItems": 1
        }
      },
      "additionalProperties": false
    }
  }
}
```

- [ ] **Step 3: Verify schema is valid JSON**

Run:

```bash
cd scripts
uv run python -c "import json; json.load(open('schema/quiz-data.schema.json')); print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
cd ..
git add scripts/pipeline/config.py scripts/schema/quiz-data.schema.json
git commit -m "feat(pipeline): add config constants and quiz-data JSON schema"
```

---

## Task 3: Simulate — Push-T rollouts

Pure-Python serialization logic is tested; real env rollout is smoke-tested manually.

**Files:**
- Create: `scripts/pipeline/simulate.py`

- [ ] **Step 1: Discover the stable_worldmodel env API**

Run:

```bash
cd scripts
uv run python -c "import stable_worldmodel as swm; print(dir(swm))"
uv run python -c "import stable_worldmodel as swm; help(swm.make_env) if hasattr(swm, 'make_env') else print('no make_env')"
```

Record the precise symbol(s) used to construct a Push-T env. Expected: a factory like `swm.make_env('pusht')` or an env class under `swm.envs`. Adjust Step 3 accordingly if the names differ.

If the import fails, run `uv sync` again; if it still fails, `stable_worldmodel[env]` may not be on PyPI — fall back to the upstream repo's install instructions from `github.com/lucas-maes/le-wm` README and update `pyproject.toml`.

- [ ] **Step 2: Write simulate.py**

Write `scripts/pipeline/simulate.py`. Replace `swm.make_env("pusht")` if Step 1 revealed a different factory — keep the rest of the structure identical.

```python
"""Push-T rollout generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

from pipeline import config


@dataclass
class Trajectory:
    frames: np.ndarray  # (T, H, W, C) uint8
    states: np.ndarray  # (T, S) float32
    actions: np.ndarray  # (T-1, A) float32
    seed: int


def rollout(seed: int, steps: int = config.CLIP_STEPS) -> Trajectory:
    """Run one deterministic Push-T rollout using a random action policy."""
    env = swm.make_env(config.ENV_NAME)
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    states = [np.asarray(obs, dtype=np.float32)]
    actions = []

    for _ in range(steps - 1):
        action = env.action_space.sample()
        # Replace sampled action with seeded draw for determinism.
        action = rng.uniform(
            low=env.action_space.low, high=env.action_space.high
        ).astype(np.float32)
        obs, _reward, _terminated, _truncated, _info = env.step(action)
        frames.append(env.render())
        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)

    env.close()
    return Trajectory(
        frames=np.stack(frames).astype(np.uint8),
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        seed=seed,
    )


def save_trajectory(traj: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=traj.frames,
        states=traj.states,
        actions=traj.actions,
        seed=np.int64(traj.seed),
    )


def load_trajectory(path: Path) -> Trajectory:
    data = np.load(path)
    return Trajectory(
        frames=data["frames"],
        states=data["states"],
        actions=data["actions"],
        seed=int(data["seed"]),
    )


def simulate_many(n: int, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or config.TRAJ_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        seed = config.BASE_SEED + i
        traj = rollout(seed)
        path = out_dir / f"traj_{i:04d}.npz"
        save_trajectory(traj, path)
        paths.append(path)
    return paths
```

- [ ] **Step 3: Smoke-test the env import (no CI test, quick manual check)**

Run:

```bash
cd scripts
uv run python -c "
from pipeline import simulate, config
traj = simulate.rollout(seed=0, steps=5)
print('frames:', traj.frames.shape, 'states:', traj.states.shape, 'actions:', traj.actions.shape)
"
```

Expected output (shapes may vary per env):
```
frames: (5, H, W, 3) states: (5, S) actions: (4, A)
```

If this fails with API errors, go back to Step 1 and fix the `make_env` / action-sampling / `render` calls to match the real stable_worldmodel API. **Do not commit until this smoke test passes.**

- [ ] **Step 4: Commit**

```bash
cd ..
git add scripts/pipeline/simulate.py
git commit -m "feat(pipeline): add Push-T rollout module"
```

---

## Task 4: Glitch — teleport and time-reversal (TDD)

Pure-Python logic. Full TDD.

**Files:**
- Create: `scripts/pipeline/glitch.py`
- Create: `scripts/tests/test_glitch.py`

- [ ] **Step 1: Write failing tests**

Write `scripts/tests/test_glitch.py`:

```python
"""Unit tests for glitch injection logic."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config, glitch


def _make_traj(steps: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    frames = (rng.integers(0, 255, size=(steps, 16, 16, 3))).astype(np.uint8)
    states = rng.normal(size=(steps, 4)).astype(np.float32)
    states[:, :2] = np.cumsum(np.ones((steps, 2)), axis=0)  # monotonic xy for detection
    actions = rng.normal(size=(steps - 1, 2)).astype(np.float32)
    return glitch.BaseTrajectory(frames=frames, states=states, actions=actions, seed=seed)


def test_teleport_offset_within_bounds():
    traj = _make_traj()
    glitched = glitch.teleport(traj, seed=1)
    offset = np.linalg.norm(
        glitched.states[glitched.injection_index, :2]
        - traj.states[glitched.injection_index, :2]
    )
    assert config.TELEPORT_MIN_OFFSET_PX <= offset <= config.TELEPORT_MAX_OFFSET_PX


def test_teleport_divergence_exactly_at_injection():
    traj = _make_traj()
    glitched = glitch.teleport(traj, seed=1)
    k = glitched.injection_index
    # Frames/states before k are identical to source.
    np.testing.assert_array_equal(glitched.states[:k], traj.states[:k])
    # States at k are different (teleport applied).
    assert not np.array_equal(glitched.states[k, :2], traj.states[k, :2])


def test_teleport_injection_frame_range():
    traj = _make_traj()
    for s in range(20):
        glitched = glitch.teleport(traj, seed=s)
        assert config.GLITCH_INJECTION_FRAME_MIN <= glitched.injection_index <= config.GLITCH_INJECTION_FRAME_MAX


def test_time_reversal_segment_length_in_bounds():
    traj = _make_traj()
    for s in range(20):
        glitched = glitch.time_reversal(traj, seed=s)
        length = glitched.metadata["segment_length"]
        assert config.TIMEREV_SEGMENT_MIN <= length <= config.TIMEREV_SEGMENT_MAX


def test_time_reversal_segment_is_reversed_in_frames():
    traj = _make_traj()
    glitched = glitch.time_reversal(traj, seed=7)
    k = glitched.injection_index
    length = glitched.metadata["segment_length"]
    # The reversed segment in glitched equals reverse of the same segment in source.
    np.testing.assert_array_equal(
        glitched.frames[k : k + length],
        traj.frames[k : k + length][::-1],
    )


def test_time_reversal_outside_segment_unchanged():
    traj = _make_traj()
    glitched = glitch.time_reversal(traj, seed=7)
    k = glitched.injection_index
    length = glitched.metadata["segment_length"]
    np.testing.assert_array_equal(glitched.frames[:k], traj.frames[:k])
    np.testing.assert_array_equal(glitched.frames[k + length :], traj.frames[k + length :])


def test_apply_family_dispatches_correctly():
    traj = _make_traj()
    g1 = glitch.apply_family(traj, family="teleport", seed=3)
    assert g1.glitch_family == "teleport"
    g2 = glitch.apply_family(traj, family="time-reversal", seed=3)
    assert g2.glitch_family == "time-reversal"
    with pytest.raises(ValueError):
        glitch.apply_family(traj, family="nonsense", seed=3)


def test_family_mix_respects_shares():
    mix = glitch.plan_family_mix(total=100, seed=0)
    assert len(mix) == 100
    n_tele = sum(1 for f in mix if f == "teleport")
    n_trev = sum(1 for f in mix if f == "time-reversal")
    assert n_tele == int(100 * config.GLITCH_TELEPORT_SHARE)
    assert n_trev == 100 - n_tele
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd scripts
uv run poe test
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'pipeline.glitch'` or `AttributeError`.

- [ ] **Step 3: Implement glitch.py**

Write `scripts/pipeline/glitch.py`:

```python
"""Glitch injection — teleport and time-reversal."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from pipeline import config

GlitchFamily = Literal["teleport", "time-reversal", "none"]


@dataclass
class BaseTrajectory:
    frames: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    seed: int


@dataclass
class GlitchedTrajectory:
    frames: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    seed: int
    source_seed: int
    injection_index: int
    glitch_family: GlitchFamily
    metadata: dict = field(default_factory=dict)


def _pick_injection_index(rng: np.random.Generator) -> int:
    return int(
        rng.integers(
            low=config.GLITCH_INJECTION_FRAME_MIN,
            high=config.GLITCH_INJECTION_FRAME_MAX + 1,
        )
    )


def teleport(traj: BaseTrajectory, seed: int) -> GlitchedTrajectory:
    """State-teleport: shift the controlled object's xy by a small impossible offset at index k."""
    rng = np.random.default_rng(seed)
    k = _pick_injection_index(rng)

    angle = rng.uniform(0, 2 * np.pi)
    magnitude = rng.uniform(config.TELEPORT_MIN_OFFSET_PX, config.TELEPORT_MAX_OFFSET_PX)
    offset = np.array([np.cos(angle), np.sin(angle)]) * magnitude

    new_states = traj.states.copy()
    new_states[k:, :2] += offset.astype(np.float32)

    new_frames = traj.frames.copy()
    # Frame-space teleport: pixel-shift the frame at k and after by the rounded offset.
    dx, dy = int(round(offset[0])), int(round(offset[1]))
    new_frames[k:] = np.roll(new_frames[k:], shift=(dy, dx), axis=(1, 2))

    return GlitchedTrajectory(
        frames=new_frames,
        states=new_states,
        actions=traj.actions.copy(),
        seed=seed,
        source_seed=traj.seed,
        injection_index=k,
        glitch_family="teleport",
        metadata={"offset_px": float(magnitude), "angle_rad": float(angle)},
    )


def time_reversal(traj: BaseTrajectory, seed: int) -> GlitchedTrajectory:
    """Time-reversal: reverse a short segment of frames starting at a random index."""
    rng = np.random.default_rng(seed)
    k = _pick_injection_index(rng)
    length = int(
        rng.integers(low=config.TIMEREV_SEGMENT_MIN, high=config.TIMEREV_SEGMENT_MAX + 1)
    )
    # Clamp so the segment fits.
    length = min(length, len(traj.frames) - k)

    new_frames = traj.frames.copy()
    new_frames[k : k + length] = traj.frames[k : k + length][::-1]

    new_states = traj.states.copy()
    new_states[k : k + length] = traj.states[k : k + length][::-1]

    return GlitchedTrajectory(
        frames=new_frames,
        states=new_states,
        actions=traj.actions.copy(),
        seed=seed,
        source_seed=traj.seed,
        injection_index=k,
        glitch_family="time-reversal",
        metadata={"segment_length": length},
    )


def apply_family(traj: BaseTrajectory, family: GlitchFamily, seed: int) -> GlitchedTrajectory:
    if family == "teleport":
        return teleport(traj, seed)
    if family == "time-reversal":
        return time_reversal(traj, seed)
    raise ValueError(f"unknown glitch family: {family!r}")


def plan_family_mix(total: int, seed: int) -> list[GlitchFamily]:
    """Deterministic, shuffled plan that splits total into teleport / time-reversal by config shares."""
    n_tele = int(total * config.GLITCH_TELEPORT_SHARE)
    n_trev = total - n_tele
    mix: list[GlitchFamily] = ["teleport"] * n_tele + ["time-reversal"] * n_trev
    rng = np.random.default_rng(seed)
    rng.shuffle(mix)
    return mix


def save_glitched(g: GlitchedTrajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=g.frames,
        states=g.states,
        actions=g.actions,
        seed=np.int64(g.seed),
        source_seed=np.int64(g.source_seed),
        injection_index=np.int64(g.injection_index),
        glitch_family=np.array(g.glitch_family),
        metadata_json=np.array(__import__("json").dumps(g.metadata)),
    )


def load_glitched(path: Path) -> GlitchedTrajectory:
    import json

    data = np.load(path, allow_pickle=False)
    return GlitchedTrajectory(
        frames=data["frames"],
        states=data["states"],
        actions=data["actions"],
        seed=int(data["seed"]),
        source_seed=int(data["source_seed"]),
        injection_index=int(data["injection_index"]),
        glitch_family=str(data["glitch_family"].item()),
        metadata=json.loads(str(data["metadata_json"].item())),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd scripts
uv run poe test
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ..
git add scripts/pipeline/glitch.py scripts/tests/test_glitch.py
git commit -m "feat(pipeline): add glitch injection (teleport + time-reversal) with tests"
```

---

## Task 5: LeWM loader — device selection + checkpoint loading

**Files:**
- Create: `scripts/pipeline/lewm_loader.py`

- [ ] **Step 1: Discover the upstream model symbol for Push-T**

Run:

```bash
cd scripts
uv run python -c "
import stable_worldmodel as swm
print('top-level:', dir(swm))
import stable_worldmodel.models as m
print('models:', dir(m))
"
```

Expected: `stable_worldmodel` exposes a top-level model factory or a `models` submodule with a `LeWM` or `JEPA` class. Note the exact symbol name (e.g. `swm.models.LeWM` or `swm.JEPA`). If the submodule layout differs, read `scripts/.venv/lib/python3.10/site-packages/stable_worldmodel/__init__.py` to find the public surface.

- [ ] **Step 2: Inspect the pretrained checkpoint structure**

If the checkpoint is already downloaded (`scripts/checkpoints/pusht_lewm.pt`), run:

```bash
cd scripts
uv run python -c "
import torch
ckpt = torch.load('checkpoints/pusht_lewm.pt', map_location='cpu', weights_only=False)
print(type(ckpt))
if isinstance(ckpt, dict):
    print('keys:', list(ckpt.keys()))
    if 'config' in ckpt:
        print('config:', ckpt['config'])
    if 'state_dict' in ckpt:
        print('state_dict sample keys:', list(ckpt['state_dict'].keys())[:10])
"
```

If the file does not yet exist, skip this step and come back once Task 10 has downloaded it; leave a `# TODO(T10): refine arch discovery after checkpoint is downloaded` comment on the model-construction line in Step 3.

- [ ] **Step 3: Write lewm_loader.py**

Write `scripts/pipeline/lewm_loader.py`. Replace `swm.models.LeWM(**ckpt["config"])` with the actual construction idiom discovered in Steps 1–2.

```python
"""LeWM model loading + device selection for M3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pipeline import config


@dataclass
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    meta: dict


def pick_device(prefer_mps: bool = True) -> torch.device:
    """Return MPS if available and preferred, else CPU. Never CUDA on M3."""
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load(checkpoint_path: Path | None = None, prefer_mps: bool = True) -> LoadedModel:
    """Load the pretrained LeWM Push-T model in eval mode."""
    import stable_worldmodel as swm  # local import so tests can run without the package

    path = checkpoint_path or config.CHECKPOINT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {path}. "
            f"Run `bash setup.sh` to download it."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Construct model. Adjust to the real swm API discovered in Step 1.
    if hasattr(swm, "models") and hasattr(swm.models, "LeWM"):
        model = swm.models.LeWM(**ckpt.get("config", {}))
    elif hasattr(swm, "JEPA"):
        model = swm.JEPA(**ckpt.get("config", {}))
    else:
        raise RuntimeError(
            "Could not locate LeWM model class in stable_worldmodel. "
            "Update lewm_loader.load() to match the installed API."
        )

    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    device = pick_device(prefer_mps=prefer_mps)
    model.to(device)

    meta = {
        "device": str(device),
        "env_fps": config.ENV_FPS,
        "embedding_dim": getattr(model, "embedding_dim", None),
    }
    return LoadedModel(model=model, device=device, meta=meta)


def smoke_test() -> bool:
    """Forward-pass a batch of random frames through the loaded model.

    Returns True on success. Prints a diagnostic line on failure but never raises
    — the caller decides whether to proceed on CPU.
    """
    import numpy as np

    try:
        loaded = load()
        dummy = torch.from_numpy(
            np.random.rand(1, 4, 3, config.RENDER_SIZE, config.RENDER_SIZE).astype("float32")
        ).to(loaded.device)
        with torch.no_grad():
            # Try common forward signatures. Adjust if the real API differs.
            if hasattr(loaded.model, "encode"):
                _ = loaded.model.encode(dummy[:, 0])
            else:
                _ = loaded.model(dummy)
        print(f"[lewm_loader] smoke test OK on {loaded.device}")
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        print(f"[lewm_loader] smoke test FAILED: {exc}")
        return False
```

- [ ] **Step 4: Smoke-test device selection without the checkpoint**

Run:

```bash
cd scripts
uv run python -c "
from pipeline.lewm_loader import pick_device
print('device:', pick_device())
"
```

Expected on an M3: `device: mps`. On other hardware: `device: cpu`.

- [ ] **Step 5: Commit**

```bash
cd ..
git add scripts/pipeline/lewm_loader.py
git commit -m "feat(pipeline): add LeWM loader with MPS/CPU device selection"
```

---

## Task 6: Score — surprise computation

**Files:**
- Create: `scripts/pipeline/score.py`

- [ ] **Step 1: Discover the upstream inference API**

Run:

```bash
cd scripts
uv run python -c "
import stable_worldmodel as swm
import inspect
# Find anything that looks like encode / predict / forward on the model class.
candidates = [x for x in dir(swm) if 'model' in x.lower() or 'jepa' in x.lower() or 'lewm' in x.lower()]
print('candidates:', candidates)
"
```

Inspect the real model's `encode` and `predict`/`predictor` methods. You need two primitives:

1. **Encode:** given a stack of `k` context frames, produce a context embedding.
2. **Predict next embedding:** given the context embedding, produce the predicted embedding of the next frame.

Note the exact method names and their input/output shapes.

- [ ] **Step 2: Write score.py**

Write `scripts/pipeline/score.py`. Adjust `model.encode(...)` / `model.predict(...)` to match what Step 1 found; keep the surrounding structure.

```python
"""Run LeWM inference over trajectories, emit per-step surprise arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pipeline import config, glitch, lewm_loader, simulate


def _frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """(T, H, W, C) uint8 → (T, C, H, W) float32 in [0,1] on device."""
    t = torch.from_numpy(frames).to(device).float() / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def surprise_curve(model: torch.nn.Module, device: torch.device, frames: np.ndarray) -> np.ndarray:
    """Return length-(T-1) array of per-step surprise values.

    Per-step surprise = L2 distance between the predictor's next-embedding output
    from context [0..i] and the encoder's embedding of frame i+1.
    """
    t = _frames_to_tensor(frames, device)  # (T, C, H, W)
    T = t.shape[0]
    out = np.zeros(T - 1, dtype=np.float32)
    with torch.no_grad():
        # Encode all frames up front (cheap, avoids redundant work).
        target_emb = model.encode(t)  # expected shape (T, D)
        for i in range(T - 1):
            context_emb = model.encode(t[: i + 1])  # (i+1, D)
            pred_next = model.predict(context_emb)  # (D,) — last-step prediction
            dist = torch.linalg.vector_norm(pred_next - target_emb[i + 1])
            out[i] = float(dist.item())
    return out


def score_dir(in_dir: Path, out_dir: Path, loaded: lewm_loader.LoadedModel | None = None) -> list[Path]:
    """Score every .npz file in in_dir. Writes <name>.npy to out_dir."""
    loaded = loaded or lewm_loader.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for npz_path in sorted(in_dir.glob("*.npz")):
        try:
            traj = simulate.load_trajectory(npz_path)
        except KeyError:
            traj = glitch.load_glitched(npz_path)
        curve = surprise_curve(loaded.model, loaded.device, traj.frames)
        out_path = out_dir / f"{npz_path.stem}.npy"
        np.save(out_path, curve)
        written.append(out_path)
    return written


def score_all() -> dict[str, list[Path]]:
    """Score both normal and glitched trajectories."""
    loaded = lewm_loader.load()
    return {
        "normal": score_dir(config.TRAJ_DIR, config.SURPRISE_DIR / "normal", loaded),
        "glitched": score_dir(config.GLITCHED_DIR, config.SURPRISE_DIR / "glitched", loaded),
    }
```

- [ ] **Step 3: Commit**

(No automated test — verified by `poe quick` end-to-end once the full pipeline is wired.)

```bash
cd ..
git add scripts/pipeline/score.py
git commit -m "feat(pipeline): add surprise scoring over trajectories"
```

---

## Task 7: Curate — tier assignment + pair composition (TDD)

**Files:**
- Create: `scripts/pipeline/curate.py`
- Create: `scripts/tests/test_curate.py`

- [ ] **Step 1: Write failing tests**

Write `scripts/tests/test_curate.py`:

```python
"""Unit tests for curation logic."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config, curate


def _flat(n: int = 29, level: float = 0.1) -> np.ndarray:
    return np.full(n, level, dtype=np.float32)


def _spike(n: int = 29, peak_at: int = 15, peak: float = 1.0, base: float = 0.1) -> np.ndarray:
    arr = np.full(n, base, dtype=np.float32)
    arr[peak_at] = peak
    return arr


def test_peak_ratio_flat():
    r = curate.peak_ratio(_flat())
    assert 0.9 <= r <= 1.1


def test_peak_ratio_spike():
    r = curate.peak_ratio(_spike(peak=1.0, base=0.1))
    assert r == pytest.approx(10.0, rel=1e-3)


def test_assign_tier_easy():
    assert curate.assign_tier(_spike(peak=2.0, base=0.1)) == "easy"  # ratio 20


def test_assign_tier_medium():
    # Ratio = 0.5 / 0.1 = 5.0 → medium (between 3.5 and 8.0)
    assert curate.assign_tier(_spike(peak=0.5, base=0.1)) == "medium"


def test_assign_tier_hard():
    # Ratio = 0.3 / 0.1 = 3.0 → hard (< 3.5)
    assert curate.assign_tier(_spike(peak=0.3, base=0.1)) == "hard"


def test_gotcha_filter_passes_flat():
    assert curate.is_gotcha_eligible(_flat(level=0.1))
    assert curate.is_gotcha_eligible(_flat(level=0.2))


def test_gotcha_filter_rejects_spike():
    # Ratio 10 > GOTCHA_MAX_RATIO (1.5) → not eligible
    assert not curate.is_gotcha_eligible(_spike(peak=1.0, base=0.1))


def test_select_pairs_respects_targets_and_tiers():
    # Build a pool of 40 glitched + 40 normal fake entries, with tier variety.
    glitched = []
    for i in range(40):
        if i < 10:
            curve = _spike(peak=2.0 + i * 0.05, base=0.1)  # easy
            tier = "easy"
            family = "teleport"
        elif i < 25:
            curve = _spike(peak=0.5, base=0.1)  # medium
            tier = "medium"
            family = "teleport" if i % 2 == 0 else "time-reversal"
        else:
            curve = _spike(peak=0.3, base=0.1)  # hard
            tier = "hard"
            family = "time-reversal"
        glitched.append(
            curate.GlitchedEntry(
                id=f"g_{i:03d}", curve=curve, tier=tier, glitch_family=family, source_id=f"n_{i:03d}"
            )
        )
    normal = [curate.NormalEntry(id=f"n_{i:03d}", curve=_flat(level=0.1)) for i in range(40)]

    selection = curate.select_pairs(glitched=glitched, normal=normal, seed=0)

    counts = {t: 0 for t in ("easy", "medium", "hard", "gotcha")}
    for pair in selection.pairs:
        counts[pair.tier] += 1
    assert counts == config.TARGET_PAIRS

    # Gotcha pairs must have both sides unglitched.
    for pair in selection.pairs:
        if pair.tier == "gotcha":
            assert pair.glitch_family == "none"
            assert pair.clip_a_glitched is False
            assert pair.clip_b_glitched is False


def test_select_pairs_deterministic():
    normal = [curate.NormalEntry(id=f"n_{i:03d}", curve=_flat()) for i in range(40)]
    glitched = [
        curate.GlitchedEntry(
            id=f"g_{i:03d}",
            curve=_spike(peak=0.5 + (i % 5) * 0.3, base=0.1),
            tier=curate.assign_tier(_spike(peak=0.5 + (i % 5) * 0.3, base=0.1)),
            glitch_family="teleport",
            source_id=f"n_{i:03d}",
        )
        for i in range(40)
    ]
    a = curate.select_pairs(glitched=glitched, normal=normal, seed=42)
    b = curate.select_pairs(glitched=glitched, normal=normal, seed=42)
    assert [p.id for p in a.pairs] == [p.id for p in b.pairs]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd scripts
uv run poe test
```

Expected: new tests fail with `ModuleNotFoundError: No module named 'pipeline.curate'`.

- [ ] **Step 3: Implement curate.py**

Write `scripts/pipeline/curate.py`:

```python
"""Tier assignment and pair selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from pipeline import config

Tier = Literal["easy", "medium", "hard", "gotcha"]


@dataclass
class NormalEntry:
    id: str
    curve: np.ndarray


@dataclass
class GlitchedEntry:
    id: str
    curve: np.ndarray
    tier: Tier
    glitch_family: str  # "teleport" | "time-reversal"
    source_id: str


@dataclass
class Pair:
    id: str
    tier: Tier
    glitch_family: str  # "teleport" | "time-reversal" | "none"
    clip_a_id: str
    clip_b_id: str
    clip_a_glitched: bool
    clip_b_glitched: bool


@dataclass
class Selection:
    pairs: list[Pair] = field(default_factory=list)


def peak_ratio(curve: np.ndarray) -> float:
    median = float(np.median(curve))
    if median <= 0:
        return 0.0
    return float(np.max(curve) / median)


def assign_tier(curve: np.ndarray) -> Tier:
    r = peak_ratio(curve)
    if r >= config.TIER_EASY_RATIO_MIN:
        return "easy"
    if r >= config.TIER_MEDIUM_RATIO_MIN:
        return "medium"
    return "hard"


def is_gotcha_eligible(curve: np.ndarray) -> bool:
    return peak_ratio(curve) <= config.GOTCHA_MAX_RATIO


def _sample(rng: np.random.Generator, pool: list, n: int) -> list:
    if len(pool) < n:
        raise RuntimeError(
            f"Not enough candidates: need {n}, have {len(pool)}. "
            f"Rerun `simulate` with higher --n, or relax tier thresholds in config."
        )
    indices = rng.permutation(len(pool))[:n]
    return [pool[i] for i in indices]


def select_pairs(
    *, glitched: list[GlitchedEntry], normal: list[NormalEntry], seed: int
) -> Selection:
    rng = np.random.default_rng(seed)

    # Bucket glitched entries by tier.
    buckets: dict[Tier, list[GlitchedEntry]] = {"easy": [], "medium": [], "hard": []}
    for g in glitched:
        if g.tier in buckets:
            buckets[g.tier].append(g)

    # Bucket normals by gotcha eligibility for A/B gotcha usage.
    gotcha_pool = [n for n in normal if is_gotcha_eligible(n.curve)]

    pairs: list[Pair] = []
    pair_idx = 0
    normal_used: set[str] = set()

    # Real (glitched vs normal) pairs.
    for tier in ("easy", "medium", "hard"):
        target = config.TARGET_PAIRS[tier]
        chosen = _sample(rng, buckets[tier], target)
        for g in chosen:
            # Pick a normal that isn't the glitched source (to avoid leaking).
            candidate_normals = [
                n for n in normal if n.id != g.source_id and n.id not in normal_used
            ]
            partner = _sample(rng, candidate_normals, 1)[0]
            normal_used.add(partner.id)
            # Randomize which side is A vs B.
            flip = bool(rng.integers(0, 2))
            if flip:
                clip_a_id, clip_b_id = g.id, partner.id
                a_glitched, b_glitched = True, False
            else:
                clip_a_id, clip_b_id = partner.id, g.id
                a_glitched, b_glitched = False, True
            pairs.append(
                Pair(
                    id=f"pair_{pair_idx:04d}",
                    tier=tier,
                    glitch_family=g.glitch_family,
                    clip_a_id=clip_a_id,
                    clip_b_id=clip_b_id,
                    clip_a_glitched=a_glitched,
                    clip_b_glitched=b_glitched,
                )
            )
            pair_idx += 1

    # Gotcha pairs — two normals, both clean.
    gotcha_available = [n for n in gotcha_pool if n.id not in normal_used]
    target_gotcha = config.TARGET_PAIRS["gotcha"]
    gotcha_chosen = _sample(rng, gotcha_available, target_gotcha * 2)
    for i in range(target_gotcha):
        a, b = gotcha_chosen[2 * i], gotcha_chosen[2 * i + 1]
        pairs.append(
            Pair(
                id=f"pair_{pair_idx:04d}",
                tier="gotcha",
                glitch_family="none",
                clip_a_id=a.id,
                clip_b_id=b.id,
                clip_a_glitched=False,
                clip_b_glitched=False,
            )
        )
        pair_idx += 1

    return Selection(pairs=pairs)


def save_selection(selection: Selection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pairs": [asdict(p) for p in selection.pairs]}
    path.write_text(json.dumps(payload, indent=2))


def load_selection(path: Path) -> Selection:
    payload = json.loads(path.read_text())
    return Selection(pairs=[Pair(**p) for p in payload["pairs"]])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd scripts
uv run poe test
```

Expected: all tests pass (both test_glitch and test_curate).

- [ ] **Step 5: Commit**

```bash
cd ..
git add scripts/pipeline/curate.py scripts/tests/test_curate.py
git commit -m "feat(pipeline): add tier assignment + pair curation with tests"
```

---

## Task 8: Export — MP4 encoding, JSON build, acceptance checks (TDD)

**Files:**
- Create: `scripts/pipeline/export.py`
- Create: `scripts/tests/test_export.py`

- [ ] **Step 1: Write failing tests**

Write `scripts/tests/test_export.py`:

```python
"""Export tests: JSON round-trips through the declared JSON Schema."""

from __future__ import annotations

import json

import jsonschema
import numpy as np
import pytest

from pipeline import config, curate, export


def _fake_selection(n: int = 3) -> curate.Selection:
    pairs = [
        curate.Pair(
            id=f"pair_{i:04d}",
            tier="medium" if i < 2 else "gotcha",
            glitch_family="teleport" if i < 2 else "none",
            clip_a_id=f"g_{i:03d}" if i < 2 else f"n_{i:03d}",
            clip_b_id=f"n_{i:03d}",
            clip_a_glitched=i < 2,
            clip_b_glitched=False,
        )
        for i in range(n)
    ]
    return curate.Selection(pairs=pairs)


def _fake_curves_for(selection: curate.Selection) -> dict[str, np.ndarray]:
    curves = {}
    for p in selection.pairs:
        for cid, glitched in ((p.clip_a_id, p.clip_a_glitched), (p.clip_b_id, p.clip_b_glitched)):
            arr = np.full(config.CLIP_STEPS - 1, 0.1, dtype=np.float32)
            if glitched:
                arr[15] = 0.9
            curves[cid] = arr
    return curves


def test_build_json_matches_schema():
    selection = _fake_selection()
    curves = _fake_curves_for(selection)
    doc = export.build_json(selection, curves)
    schema = json.loads(config.SCHEMA_FILE.read_text())
    jsonschema.validate(doc, schema)


def test_build_json_has_correct_top_level_fields():
    doc = export.build_json(_fake_selection(), _fake_curves_for(_fake_selection()))
    assert doc["schemaVersion"] == 1
    assert doc["clipFps"] == config.ENV_FPS
    assert doc["clipDurationSec"] == config.CLIP_DURATION_SEC
    assert set(doc["distribution"].keys()) == {"easy", "medium", "hard", "gotcha"}


def test_peak_pass_rate_all_spiky():
    curves = [np.array([0.1, 0.1, 0.9, 0.1]), np.array([0.1, 0.1, 1.0, 0.1])]
    rate = export.peak_pass_rate(curves)
    assert rate == 1.0


def test_peak_pass_rate_mixed():
    curves = [
        np.array([0.1, 0.1, 0.9, 0.1]),  # ratio 9 > 3 → pass
        np.array([0.1, 0.1, 0.2, 0.1]),  # ratio 2 < 3 → fail
    ]
    assert export.peak_pass_rate(curves) == pytest.approx(0.5)


def test_gotcha_check_fails_on_spike():
    curves = [np.array([0.1, 0.1, 0.9, 0.1])]
    assert not export.gotcha_curves_flat(curves)


def test_gotcha_check_passes_on_flat():
    curves = [np.array([0.1, 0.11, 0.12, 0.1])]
    assert export.gotcha_curves_flat(curves)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd scripts
uv run poe test
```

Expected: new tests fail with `ModuleNotFoundError: No module named 'pipeline.export'`.

- [ ] **Step 3: Implement export.py**

Write `scripts/pipeline/export.py`:

```python
"""Final stage: encode MP4s, build quiz-data.json, run acceptance checks."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import imageio.v2 as imageio
import jsonschema
import numpy as np

from pipeline import config, curate, glitch, simulate


def upscale_nearest(frames: np.ndarray, target: int = config.UPSCALE_SIZE) -> np.ndarray:
    """(T, H, W, C) uint8 → (T, target, target, C) uint8 via nearest-neighbour."""
    T, H, W, C = frames.shape
    factor_h = target // H
    factor_w = target // W
    if factor_h < 1 or factor_w < 1:
        return frames
    return np.repeat(np.repeat(frames, factor_h, axis=1), factor_w, axis=2)


def encode_mp4(frames: np.ndarray, out_path: Path, fps: int = config.ENV_FPS) -> None:
    """H.264 baseline + +faststart, browser-compatible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        quality=8,
        ffmpeg_params=["-profile:v", "baseline", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _load_frames(clip_id: str) -> np.ndarray:
    """Load frames for a clip id; clip id starts with 'g_' for glitched or 'n_' for normal."""
    if clip_id.startswith("g_"):
        return glitch.load_glitched(config.GLITCHED_DIR / f"{clip_id}.npz").frames
    return simulate.load_trajectory(config.TRAJ_DIR / f"{clip_id}.npz").frames


def build_json(selection: curate.Selection, curves: dict[str, np.ndarray]) -> dict:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    distribution = {"easy": 0, "medium": 0, "hard": 0, "gotcha": 0}
    pool = []
    for p in selection.pairs:
        distribution[p.tier] += 1
        pool.append(
            {
                "id": p.id,
                "tier": p.tier,
                "glitchFamily": p.glitch_family,
                "clipA": {
                    "src": f"/clips/{p.id}_a.mp4",
                    "isGlitched": p.clip_a_glitched,
                    "surpriseScore": [float(x) for x in curves[p.clip_a_id]],
                },
                "clipB": {
                    "src": f"/clips/{p.id}_b.mp4",
                    "isGlitched": p.clip_b_glitched,
                    "surpriseScore": [float(x) for x in curves[p.clip_b_id]],
                },
            }
        )
    return {
        "schemaVersion": 1,
        "generatedAt": now,
        "clipFps": config.ENV_FPS,
        "clipDurationSec": config.CLIP_DURATION_SEC,
        "distribution": distribution,
        "pool": pool,
    }


def peak_pass_rate(curves: list[np.ndarray]) -> float:
    if not curves:
        return 1.0
    passes = sum(1 for c in curves if curate.peak_ratio(c) >= config.PEAK_OVER_MEDIAN_MIN)
    return passes / len(curves)


def gotcha_curves_flat(curves: list[np.ndarray]) -> bool:
    return all(curate.peak_ratio(c) <= config.GOTCHA_MAX_RATIO for c in curves)


def export_all(selection: curate.Selection, curves: dict[str, np.ndarray]) -> dict:
    """Encode MP4s + write JSON. Returns a report dict printed to stdout."""
    # Encode MP4s.
    config.PUBLIC_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for p in selection.pairs:
        for side, cid in (("a", p.clip_a_id), ("b", p.clip_b_id)):
            frames = _load_frames(cid)
            frames = upscale_nearest(frames)
            encode_mp4(frames, config.PUBLIC_CLIPS_DIR / f"{p.id}_{side}.mp4")

    # Build + validate + write JSON.
    doc = build_json(selection, curves)
    schema = json.loads(config.SCHEMA_FILE.read_text())
    jsonschema.validate(doc, schema)
    config.PUBLIC_QUIZ_JSON.write_text(json.dumps(doc, indent=2))

    # Acceptance checks.
    non_gotcha_glitched_curves: list[np.ndarray] = []
    gotcha_curves: list[np.ndarray] = []
    for p in selection.pairs:
        if p.tier == "gotcha":
            gotcha_curves.append(curves[p.clip_a_id])
            gotcha_curves.append(curves[p.clip_b_id])
        else:
            glitched_id = p.clip_a_id if p.clip_a_glitched else p.clip_b_id
            non_gotcha_glitched_curves.append(curves[glitched_id])

    pass_rate = peak_pass_rate(non_gotcha_glitched_curves)
    gotcha_flat = gotcha_curves_flat(gotcha_curves)

    # File size check.
    over_budget = []
    for clip in config.PUBLIC_CLIPS_DIR.glob("pair_*.mp4"):
        size = clip.stat().st_size
        if size > config.CLIP_MAX_BYTES:
            over_budget.append((clip.name, size))

    report = {
        "pairs": len(selection.pairs),
        "mp4s_written": len(selection.pairs) * 2,
        "peak_pass_rate": pass_rate,
        "peak_pass_rate_threshold": config.NON_GOTCHA_PEAK_PASS_RATE_MIN,
        "peak_pass_rate_ok": pass_rate >= config.NON_GOTCHA_PEAK_PASS_RATE_MIN,
        "gotcha_curves_flat": gotcha_flat,
        "clips_over_budget": over_budget,
        "budget_ok": not over_budget,
    }

    # Print a one-line-per-check summary.
    print(f"[export] pairs={report['pairs']} mp4s={report['mp4s_written']}")
    print(
        f"[export] peak pass rate {pass_rate:.0%} "
        f"(need ≥{config.NON_GOTCHA_PEAK_PASS_RATE_MIN:.0%}) "
        f"{'OK' if report['peak_pass_rate_ok'] else 'FAIL'}"
    )
    print(f"[export] gotcha curves flat: {'OK' if gotcha_flat else 'FAIL'}")
    print(f"[export] file size budget: {'OK' if report['budget_ok'] else 'FAIL'}")
    if not report["peak_pass_rate_ok"] or not gotcha_flat or not report["budget_ok"]:
        raise SystemExit("Acceptance checks failed; see above.")
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd scripts
uv run poe test
```

Expected: all tests across test_glitch, test_curate, and test_export pass.

- [ ] **Step 5: Commit**

```bash
cd ..
git add scripts/pipeline/export.py scripts/tests/test_export.py
git commit -m "feat(pipeline): add export stage (MP4 + JSON + acceptance checks)"
```

---

## Task 9: CLI entrypoint — wire subcommands

**Files:**
- Create: `scripts/generate.py`

- [ ] **Step 1: Write generate.py**

Write `scripts/generate.py`:

```python
"""Single entrypoint for the pipeline. Each subcommand is idempotent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from pipeline import config, curate, export, glitch, score, simulate

app = typer.Typer(add_completion=False, help="Spot the Glitch — local LeWM pipeline.")


@app.command()
def simulate_cmd(
    n: int = typer.Option(config.DEFAULT_N_ROLLOUTS, "--n", help="Number of normal rollouts."),
    force: bool = typer.Option(False, "--force", help="Regenerate even if outputs exist."),
) -> None:
    """Rollout N normal Push-T trajectories to data/trajectories/."""
    out_dir = config.TRAJ_DIR
    if out_dir.exists() and any(out_dir.glob("*.npz")) and not force:
        typer.echo(f"[simulate] {out_dir} already populated; skipping (use --force to override)")
        return
    paths = simulate.simulate_many(n=n, out_dir=out_dir)
    typer.echo(f"[simulate] wrote {len(paths)} trajectories to {out_dir}")


app.command(name="simulate")(simulate_cmd)


@app.command()
def glitch_cmd(
    seed: int = typer.Option(config.BASE_SEED, "--seed", help="Determinism seed."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Produce one glitched variant per normal trajectory using planned family mix."""
    out_dir = config.GLITCHED_DIR
    if out_dir.exists() and any(out_dir.glob("*.npz")) and not force:
        typer.echo(f"[glitch] {out_dir} already populated; skipping (use --force)")
        return
    sources = sorted(config.TRAJ_DIR.glob("*.npz"))
    if not sources:
        raise typer.Exit(code=1)
    mix = glitch.plan_family_mix(total=len(sources), seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (src_path, family) in enumerate(zip(sources, mix)):
        src = simulate.load_trajectory(src_path)
        base = glitch.BaseTrajectory(
            frames=src.frames, states=src.states, actions=src.actions, seed=src.seed
        )
        g = glitch.apply_family(base, family=family, seed=seed + i)
        glitch.save_glitched(g, out_dir / f"g_{i:04d}.npz")
    typer.echo(f"[glitch] wrote {len(sources)} glitched trajectories to {out_dir}")


app.command(name="glitch")(glitch_cmd)


@app.command()
def score_cmd(force: bool = typer.Option(False, "--force")) -> None:
    """Run LeWM inference to produce surprise arrays."""
    surprise_base = config.SURPRISE_DIR
    if surprise_base.exists() and any(surprise_base.rglob("*.npy")) and not force:
        typer.echo(f"[score] {surprise_base} already populated; skipping (use --force)")
        return
    result = score.score_all()
    typer.echo(
        f"[score] scored {len(result['normal'])} normal + {len(result['glitched'])} glitched"
    )


app.command(name="score")(score_cmd)


def _load_curate_inputs() -> tuple[list[curate.GlitchedEntry], list[curate.NormalEntry]]:
    glitched_entries: list[curate.GlitchedEntry] = []
    for npz_path in sorted(config.GLITCHED_DIR.glob("*.npz")):
        g = glitch.load_glitched(npz_path)
        curve = np.load(config.SURPRISE_DIR / "glitched" / f"{npz_path.stem}.npy")
        glitched_entries.append(
            curate.GlitchedEntry(
                id=npz_path.stem,
                curve=curve,
                tier=curate.assign_tier(curve),
                glitch_family=g.glitch_family,
                source_id=f"traj_{g.source_seed - config.BASE_SEED:04d}",
            )
        )
    normal_entries: list[curate.NormalEntry] = []
    for npz_path in sorted(config.TRAJ_DIR.glob("*.npz")):
        curve = np.load(config.SURPRISE_DIR / "normal" / f"{npz_path.stem}.npy")
        normal_entries.append(curate.NormalEntry(id=npz_path.stem, curve=curve))
    return glitched_entries, normal_entries


@app.command()
def curate_cmd(
    seed: int = typer.Option(config.BASE_SEED, "--seed"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Select 30 pairs and write selection.json."""
    out_path = config.DATA_DIR / "selection.json"
    if out_path.exists() and not force:
        typer.echo(f"[curate] {out_path} already exists; skipping (use --force)")
        return
    glitched_entries, normal_entries = _load_curate_inputs()
    selection = curate.select_pairs(glitched=glitched_entries, normal=normal_entries, seed=seed)
    curate.save_selection(selection, out_path)
    typer.echo(f"[curate] wrote {len(selection.pairs)} pairs to {out_path}")


app.command(name="curate")(curate_cmd)


@app.command()
def export_cmd() -> None:
    """Encode MP4s + build quiz-data.json + run acceptance checks."""
    selection = curate.load_selection(config.DATA_DIR / "selection.json")
    curves: dict[str, np.ndarray] = {}
    for p in selection.pairs:
        for cid in (p.clip_a_id, p.clip_b_id):
            sub = "glitched" if cid.startswith("g_") else "normal"
            curves[cid] = np.load(config.SURPRISE_DIR / sub / f"{cid}.npy")
    export.export_all(selection, curves)


app.command(name="export")(export_cmd)


@app.command()
def quick() -> None:
    """Dev fast path: small N, writes to data/debug/ for hand inspection."""
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[quick] running n={config.QUICK_N_ROLLOUTS} end-to-end into {config.DEBUG_DIR}")
    # Implementation intentionally minimal: run simulate + glitch + score into the main dirs
    # with a smaller N. Curate/export expect the main dirs.
    simulate.simulate_many(n=config.QUICK_N_ROLLOUTS, out_dir=config.TRAJ_DIR)
    mix = glitch.plan_family_mix(total=config.QUICK_N_ROLLOUTS, seed=config.BASE_SEED)
    for i, (src_path, family) in enumerate(zip(sorted(config.TRAJ_DIR.glob("*.npz")), mix)):
        src = simulate.load_trajectory(src_path)
        base = glitch.BaseTrajectory(
            frames=src.frames, states=src.states, actions=src.actions, seed=src.seed
        )
        g = glitch.apply_family(base, family=family, seed=config.BASE_SEED + i)
        glitch.save_glitched(g, config.GLITCHED_DIR / f"g_{i:04d}.npz")
    score.score_all()
    typer.echo("[quick] done. Inspect data/surprise/ for curves.")


app.command(name="quick")(quick)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify the CLI resolves each subcommand**

```bash
cd scripts
uv run python generate.py --help
uv run python generate.py simulate --help
uv run python generate.py glitch --help
uv run python generate.py score --help
uv run python generate.py curate --help
uv run python generate.py export --help
uv run python generate.py quick --help
```

Expected: each prints a help screen with the described options. No tracebacks.

- [ ] **Step 3: Verify every poe task resolves (do not run ML-bound ones yet)**

```bash
cd scripts
uv run poe test
uv run poe lint
uv run poe format:check
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
cd ..
git add scripts/generate.py
git commit -m "feat(pipeline): add typer CLI wiring all subcommands"
```

---

## Task 10: setup.sh — environment bootstrap + checkpoint download + smoke test

**Files:**
- Create: `scripts/setup.sh`

- [ ] **Step 1: Locate the pretrained checkpoint's Google Drive file ID**

Read the upstream README at `https://github.com/lucas-maes/le-wm` and find the Google Drive link for the pretrained Push-T LeWM checkpoint. Note the file ID (the long alphanumeric token in the GDrive share URL).

Record the file ID here (replace `<GDRIVE_FILE_ID>` in Step 2). If the upstream link has been reorganised, document the new location in `scripts/README.md` (Task 11).

- [ ] **Step 2: Write setup.sh**

Write `scripts/setup.sh` (replace `<GDRIVE_FILE_ID>` with the ID from Step 1):

```bash
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
```

Make it executable:

```bash
chmod +x scripts/setup.sh
```

- [ ] **Step 3: Commit (setup.sh without running it yet)**

```bash
git add scripts/setup.sh
git commit -m "chore(scripts): add setup.sh for env + checkpoint + smoke test"
```

- [ ] **Step 4: Run the full setup as a sanity check**

```bash
cd scripts
./setup.sh
```

Expected:
- Steps 1/4 and 2/4 succeed.
- Step 3/4 downloads the checkpoint to `checkpoints/pusht_lewm.pt`.
- Step 4/4 either prints "smoke test OK on mps" (happy path) or the CPU-fallback warning.

If the smoke test fails for an API reason (not a device issue), return to Task 5 Step 3 and refine `lewm_loader.load()` and `smoke_test()` to match the real upstream API. Re-run `./setup.sh` until the smoke test either succeeds on MPS or cleanly prints the CPU fallback warning.

- [ ] **Step 5: Run the pipeline end-to-end on the quick path**

```bash
cd scripts
uv run poe quick
uv run poe curate --force || echo "curate may fail on n=5; that's ok for quick"
```

Hand-inspect:
- `scripts/data/trajectories/` has 5 `.npz` files.
- `scripts/data/glitched/` has 5 `.npz` files.
- `scripts/data/surprise/normal/` and `.../glitched/` each have 5 `.npy` files.
- Print one surprise curve: `uv run python -c "import numpy as np; print(np.load('data/surprise/glitched/g_0000.npy'))"`. A real glitch should produce at least one value noticeably higher than the rest; if all values are identical, there is a bug in `score.surprise_curve` or the model is not actually running.

- [ ] **Step 6: Commit any fixes made during Steps 4–5**

```bash
cd ..
git add -A
git commit -m "fix(pipeline): align with discovered upstream API" || echo "nothing to commit"
```

---

## Task 11: README + final end-to-end run

**Files:**
- Create: `scripts/README.md`

- [ ] **Step 1: Write scripts/README.md**

Write `scripts/README.md`:

````markdown
# scripts — LeWM local generation pipeline

Isolated Python pipeline that produces `/public/clips/*.mp4` and `/public/quiz-data.json` from a pretrained LeWorldModel (LeWM) Push-T checkpoint. See `docs/superpowers/specs/2026-04-17-lewm-pipeline-design.md` at the repo root for the design.

## Requirements

- macOS on Apple Silicon (tested on M3). MPS preferred; CPU fallback is supported.
- [`uv`](https://docs.astral.sh/uv/) (Python package manager)
- ~1 GB of free disk for trajectories + MP4s + checkpoint

## First-time setup

```bash
cd scripts
./setup.sh
```

This verifies `uv`, creates `.venv/`, installs deps, downloads the pretrained Push-T checkpoint, and runs a smoke test.

If the smoke test fails on MPS, the pipeline will run on CPU (slower but functional). Open an issue at https://github.com/lucas-maes/le-wm if you want the failing op upstreamed.

## End-to-end generation

```bash
uv run poe all
```

Runs: `simulate` → `glitch` → `score` → `curate` → `export`. On an M3 with MPS, the full run takes roughly 10–20 minutes. On CPU, expect 30–60 minutes.

Outputs:
- `../public/clips/pair_0000_a.mp4` through `pair_0029_b.mp4` (60 files)
- `../public/quiz-data.json`

## Per-stage commands

All of these are idempotent and skip if their output exists (override with `--force`):

| Command | Input | Output |
|---|---|---|
| `uv run poe simulate` | — | `data/trajectories/traj_*.npz` |
| `uv run poe glitch` | trajectories | `data/glitched/g_*.npz` |
| `uv run poe score` | both of the above | `data/surprise/{normal,glitched}/*.npy` |
| `uv run poe curate` | surprise curves | `data/selection.json` |
| `uv run poe export` | selection + surprise | `../public/clips/*.mp4`, `../public/quiz-data.json` |

## Development

| Command | Purpose |
|---|---|
| `uv run poe test` | pytest (unit tests only, fast) |
| `uv run poe lint` | ruff check |
| `uv run poe format` | ruff format |
| `uv run poe verify` | lint + format:check + test |
| `uv run poe quick` | n=5 fast path for hand-inspection |

Tuning thresholds (tier cutoffs, glitch magnitudes, acceptance bars) lives in `pipeline/config.py`.

## Project layout

```
scripts/
├── pipeline/
│   ├── config.py         # All tunable constants
│   ├── simulate.py       # Push-T rollouts
│   ├── glitch.py         # Teleport + time-reversal injection
│   ├── lewm_loader.py    # Model loading + device selection
│   ├── score.py          # LeWM inference → surprise curves
│   ├── curate.py         # Tier assignment + pair composition
│   └── export.py         # MP4 encoding + JSON build
├── tests/                # pytest suites
├── schema/
│   └── quiz-data.schema.json  # Frontend contract
├── generate.py           # typer CLI
└── setup.sh              # First-time bootstrap
```

## License

Upstream LeWorldModel (`stable-worldmodel`, `le-wm`) is MIT-licensed. See https://github.com/lucas-maes/le-wm.
````

- [ ] **Step 2: Run the full pipeline end-to-end**

```bash
cd scripts
uv run poe all
```

Expected final output:
```
[simulate] wrote 80 trajectories to ...
[glitch] wrote 80 glitched trajectories to ...
[score] scored 80 normal + 80 glitched
[curate] wrote 30 pairs to ...
[export] pairs=30 mp4s=60
[export] peak pass rate 87% (need ≥80%) OK
[export] gotcha curves flat: OK
[export] file size budget: OK
```

If peak pass rate is below 80%, tune `TIER_*` thresholds in `pipeline/config.py` (make tiers less aggressive) and rerun `poe curate` + `poe export`. No need to re-simulate or re-score.

If gotcha curves show spikes, the gotcha eligibility filter is too loose — lower `GOTCHA_MAX_RATIO` and rerun curate + export.

- [ ] **Step 3: Verify the frontend can consume the output**

```bash
cd ..
bun dev
# In another terminal:
curl -s http://localhost:3000/quiz-data.json | head -c 500
```

Expected: valid JSON matching the schema (schemaVersion, pool, etc.). No 404s on any `/clips/pair_XXXX_X.mp4` URL.

- [ ] **Step 4: Commit generated assets + README**

```bash
git add scripts/README.md public/quiz-data.json public/clips/
git commit -m "feat: generate initial LeWM clip pool + README"
```

- [ ] **Step 5: Push the branch and open a PR**

```bash
git push -u origin spec/lewm-pipeline
# Use gh to open the PR per repo convention.
```

---

## Self-Review

**Spec coverage check.** Walking through the spec section-by-section:

- Goals — all covered: T1–T11 produce the `/public/` artifacts locally from a pretrained checkpoint.
- Non-goals — nothing in this plan introduces a Python runtime in prod, streaming inference, multi-env support, hyperparameter tuning, or training. ✓
- Product shape (9/12/6/3 distribution, two glitch families, gotcha pairs, pixel-art upscale) — T2 (distribution constants), T4 (two glitch families), T7 (gotcha pairs), T8 (nearest-neighbour upscale). ✓
- Pipeline architecture (stage boundaries, idempotent subcommands) — T3/T4/T6/T7/T8/T9 cover each stage with skip-if-exists logic in T9. ✓
- LeWM integration (checkpoint acquisition, model loading, MPS smoke test, surprise computation) — T5 (loader), T6 (surprise), T10 (gdown + smoke test). ✓
- Data contract — T2 (JSON Schema), T8 (build + validate), T8 test validates round-trip. ✓
- Tooling/DX (uv, poe, ruff, pytest) — T1 sets all of these up and verifies they run. ✓
- Testing strategy (three narrow suites, no ML in CI) — T4, T7, T8 provide them. ✓
- Acceptance criteria — all five are checked: (1) `setup.sh` + `poe all` in T10/T11, (2) 60 MP4s validated by schema pattern and budget check in T8/T11, (3) peak pass rate assertion in T8, (4) gotcha flat check in T8, (5) `poe verify` in T1/T9. ✓
- Risks (checkpoint link rot, MPS missing op, thresholds, monotony, encoder compat) — T5/T10 print CPU fallback guidance; thresholds all configurable in T2; T3 randomises seeds; T8 uses H.264 baseline + faststart. ✓
- Deferred work — training folder and heatmap are out of scope for this plan; `scripts/README.md` does not promise training support. ✓

**Placeholder scan.** No "TBD" / "TODO" (one intentional `TODO(T10)` comment in T5 Step 3 is resolved by Task 10 and documented). Every code-touching step has complete code. The `<GDRIVE_FILE_ID>` marker in T10 Step 2 has an explicit discovery step (T10 Step 1) and a runtime error if not set — this is an actionable hand-off, not a placeholder.

**Type consistency.** `BaseTrajectory`/`GlitchedTrajectory` fields align across T3/T4/T6/T9. `curate.NormalEntry`/`GlitchedEntry`/`Pair`/`Selection` are consistent across T7/T8/T9. `Clip` shape in the JSON Schema (T2) matches what T8 builds. Function names across tasks (`rollout`, `save_trajectory`, `apply_family`, `plan_family_mix`, `peak_ratio`, `assign_tier`, `select_pairs`, `build_json`, `peak_pass_rate`, `gotcha_curves_flat`) are each defined once and referenced by their exact name in later tasks. ✓
