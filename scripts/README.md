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

This verifies `uv`, creates `.venv/`, installs deps, downloads the pretrained Push-T checkpoint from HuggingFace (`quentinll/lewm-pusht`), and runs a smoke test.

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

Upstream `stable-worldmodel` is MIT-licensed. LeWM paper: arxiv:2603.19312.
