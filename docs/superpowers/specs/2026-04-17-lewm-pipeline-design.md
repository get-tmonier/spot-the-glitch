# LeWM Local Generation Pipeline — Design

**Date:** 2026-04-17
**Status:** Design approved; ready for implementation plan
**Scope:** Local Python pipeline that produces the static video + JSON assets the game consumes. No Python runtime in production.

## Context

Spot the Glitch is a viral web game: players are shown pairs of 3-second clips of a simulated 2D physics environment (Push-T) and have to identify the one containing a subtle physics violation. After each answer, the game reveals the surprise curve of a pretrained JEPA-style world model (LeWM), which spikes precisely where the violation occurs — the pedagogical payoff.

The frontend (Next.js 16, FSD, TS strict) is already scaffolded. The outstanding piece is the pipeline that generates the clips and the surprise data.

The upstream model lives at `github.com/lucas-maes/le-wm` (MIT). It ships pretrained Push-T checkpoints on Google Drive and is installable via `uv pip install stable-worldmodel[train,env]`.

## Goals

- A reproducible local pipeline on a MacBook Pro M3 that produces `/public/clips/*.mp4` and `/public/quiz-data.json`, using a pretrained LeWM checkpoint.
- Output is static; no Python runtime required for Vercel deployment.
- Pipeline is legible, cacheable, and introspectable. Re-curating or re-encoding doesn't re-run inference.
- DX on the Python side mirrors the TS side's `bun run verify` ergonomics.

## Non-goals (explicit guardrails)

- No Python runtime in production. `/public/` is the contract boundary.
- No live/streaming inference. Everything is static files.
- No multi-environment support. Push-T only.
- No hyperparameter tuning. The published checkpoint is used as-is.
- No training in this spec. Training is deferred (see Deferred Work).

## Product shape

Each quiz draws 10 questions from a curated pool of 30 clip pairs, distributed across difficulty tiers:

- **Easy (9 pairs):** large state-teleport, detectable by attentive humans.
- **Medium (12 pairs):** smaller teleport, or teleport at a visually busy moment.
- **Hard (6 pairs):** subtle teleport, or time-reversal of a short 3–5 frame segment.
- **Gotcha (3 pairs):** neither clip is glitched. The frontend offers a "neither" answer on these. Humans often guess anyway; the AI's flat surprise curves on both sides correctly indicate no violation — a signature "the model really gets this" moment.

Two glitch families are used, split roughly 60% teleport / 40% time-reversal among the non-gotcha pairs. Teleport is the signature physics violation; time-reversal proves the model catches temporal anomalies, not just spatial ones.

Clips are rendered at Push-T's native low resolution and nearest-neighbour upscaled to ~384×384 for a pixel-art aesthetic. File size target: ≤250KB per MP4.

## Pipeline architecture

Single repo folder `scripts/`, isolated from the Next.js project. One CLI entrypoint with cacheable subcommands.

```
scripts/
├── .gitignore              # /.venv/, /data/, /checkpoints/, __pycache__/, .pytest_cache/
├── .python-version         # "3.10" — pins interpreter for uv
├── README.md               # Setup + end-to-end run instructions
├── pyproject.toml          # uv-managed; deps, tool config, poe tasks
├── setup.sh                # one-shot: verify uv, create .venv, install, download checkpoint, smoke test
├── generate.py             # typer-based CLI, one entrypoint per subcommand
├── pipeline/
│   ├── __init__.py
│   ├── config.py           # constants: paths, tier thresholds, glitch params, clip timing
│   ├── simulate.py         # Push-T rollouts → data/trajectories/*.npz
│   ├── glitch.py           # teleport + time-reversal → data/glitched/*.npz
│   ├── score.py            # LeWM inference → data/surprise/*.npy
│   ├── curate.py           # tier selection → data/selection.json
│   ├── export.py           # MP4 encoding + /public/quiz-data.json
│   └── lewm_loader.py      # thin wrapper around upstream model
├── tests/
│   ├── test_glitch.py
│   ├── test_curate.py
│   └── test_export.py
├── data/                   # gitignored; stage-to-stage cache
└── checkpoints/            # gitignored; holds downloaded pretrained weights
```

### Stage boundaries

Each stage reads/writes files in `scripts/data/`; only `export` writes to `/public/`. Stages are idempotent and skip work if their output already exists (override with `--force`).

1. **`simulate`** — rolls Push-T N times with randomised initial and goal poses. Saves per-rollout `.npz` containing states, actions, rendered frames, and the seed. Default N=80.
2. **`glitch`** — reads every normal trajectory and emits one glitched variant per trajectory, writing a new `.npz` with the injection index and glitch family recorded. Family mix across all variants is configured in `config.py` (default 60% teleport / 40% time-reversal). Gotcha pairs are not produced here; they are assembled at curate time by pairing two unglitched trajectories.
3. **`score`** — loads the LeWM model once, runs inference over every trajectory (normal and glitched), writes a per-trajectory `.npy` of shape `(T-1,)` containing per-step surprise (L2 distance between predicted and actual target embedding).
4. **`curate`** — selects 30 pairs across tiers using surprise-peak and baseline-ratio thresholds defined in `config.py`. Writes `data/selection.json` declaring which trajectories pair up and which tier each pair belongs to.
5. **`export`** — reads selection + trajectories + surprise arrays; encodes MP4s with ffmpeg (H.264 baseline + `+faststart`); builds `quiz-data.json` matching the schema below; validates against the schema; prints the pass/fail checks listed under "Acceptance" before exiting.

## LeWM integration

**Checkpoint acquisition.** `setup.sh` invokes `gdown` to pull the Push-T checkpoint from upstream's published Google Drive link into `scripts/checkpoints/pusht_lewm.pt`. On 404 or hash mismatch, the script fails loudly with a pointer to upstream's README.

**Model loading (`pipeline/lewm_loader.py`, ~50 lines).**

1. Import `stable_worldmodel` from the venv.
2. Instantiate the LeWM model architecture matching the Push-T checkpoint.
3. `torch.load(...)` weights, move to device.
4. Device selection: MPS if available, else CPU with a loud warning. No CUDA path.
5. Returns `(model_in_eval_mode, metadata_dict)` where `metadata_dict` holds env step rate and embedding dim.

**MPS compatibility risk.** Upstream does not declare MPS support. `setup.sh` runs a minimal smoke test (encode 4 random frames, forward-pass the predictor once) and reports pass/fail. On failure, it prints two remediation paths: (a) fall back to CPU — tolerable at ~20–40 min for 160 trajectories × 30 steps — or (b) open an upstream issue referencing the failing op.

**Surprise computation.** Per-trajectory output is an array of `T-1` floats, where `T` is the number of env steps in the rollout. No per-frame upsampling in Python. Frontend interpolates if the chart needs it.

## Data contract

`/public/quiz-data.json` — schema-versioned, single file. Example below is abbreviated; real `surpriseScore` arrays have `T-1` entries (~29 for a 3s clip at 10Hz):

```json
{
  "schemaVersion": 1,
  "generatedAt": "2026-04-17T14:19:00Z",
  "clipFps": 10,
  "clipDurationSec": 3,
  "distribution": { "easy": 9, "medium": 12, "hard": 6, "gotcha": 3 },
  "pool": [
    {
      "id": "pair_0001",
      "tier": "medium",
      "glitchFamily": "teleport",
      "clipA": {
        "src": "/clips/pair_0001_a.mp4",
        "isGlitched": true,
        "surpriseScore": [0.08, 0.09, 0.11, 0.94, 0.21, 0.10]
      },
      "clipB": {
        "src": "/clips/pair_0001_b.mp4",
        "isGlitched": false,
        "surpriseScore": [0.09, 0.10, 0.08, 0.11, 0.09, 0.10]
      }
    }
  ]
}
```

Schema notes:

- `glitchFamily`: `"teleport" | "time-reversal" | "none"`. `"none"` is only used when `tier === "gotcha"`.
- `tier: "gotcha"` implies both `clipA.isGlitched` and `clipB.isGlitched` are `false`. Frontend uses this as the signal to offer a "neither" answer on those questions.
- `surpriseScore` length is `T-1` where `T` is the env step count — ~29 for a 3s clip at 10Hz. Documented here so the frontend does not assume 90.
- `clipFps` and `clipDurationSec` at the top level avoid frontend hardcoding.

## Tooling and DX

**Package manager:** `uv`. Matches upstream and is the fastest Python installer with good M3 support.

**Task runner:** `poethepoet`, declared in `pyproject.toml` under `[tool.poe.tasks]`. Gives the Python folder the same command ergonomics as the TS repo's `bun run`:

```bash
uv run poe setup        # = bash setup.sh
uv run poe simulate
uv run poe glitch
uv run poe score
uv run poe curate
uv run poe export
uv run poe all          # sequence: simulate → glitch → score → curate → export
uv run poe quick        # dev fast path: n=5, writes to data/debug/
uv run poe test         # pytest
uv run poe lint         # ruff check
uv run poe format       # ruff format
uv run poe verify       # lint + test (mirrors parent repo's `bun run verify`)
```

**Linting/formatting:** ruff (lint + format). No mypy; types are hints only.

**Testing:** pytest. Unit tests only; no ML inference in CI. See Testing section below.

## Testing strategy

Three narrow suites, all pure-Python, fast, deterministic:

- **`test_glitch.py`** — teleport offset falls within configured bounds; time-reversal indices are legal; glitched trajectory diverges from its source at exactly the recorded injection index.
- **`test_curate.py`** — with synthetic surprise distributions, tier assignments land in the expected buckets; gotcha pairs never contain an `isGlitched: true` clip; sampling is deterministic under a seed.
- **`test_export.py`** — emitted JSON validates against the schema; schema round-trips through a shared JSON Schema file that the TS frontend can also consume.

No integration test against the real model. `poe quick` (n=5) is the manual-run smoke test during development.

## Acceptance

Binary and measurable criteria. Items (2)–(4) are printed as one-line pass/fail at the end of `poe export`.

1. **Fresh clone works end-to-end.** `poe setup` then `poe all` completes on an M3 without manual intervention.
2. **Outputs match the contract.** 60 MP4s (≤250KB each, playable in current Safari and Chrome) + one `quiz-data.json` matching the schema above.
3. **The reveal lands.** For ≥80% of non-gotcha glitched clips, peak surprise is ≥3× that clip's median surprise.
4. **Gotchas stay flat.** For every gotcha clip (A and B), every surprise value is ≤1.5× that clip's median — no false peaks.
5. **Verification passes.** `poe verify` is green.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Upstream Google Drive checkpoint link rots | Medium | `setup.sh` catches 404 and prints upstream README URL. |
| MPS missing an op LeWM uses | Medium | Pre-flight smoke test in `setup.sh`. Documented CPU fallback. |
| Surprise peak criterion (3× baseline) too strict | Medium | Threshold is a constant in `config.py`; tune during curation if needed. |
| Push-T trajectories feel visually monotonous | Low | `simulate.py` randomises initial T-block pose and goal pose per rollout. |
| MP4 encoding inconsistency across browsers | Low | `export.py` uses H.264 baseline + `+faststart`; tested on Safari + Chrome. |

## Deferred work (named, not in scope)

- **Training (`scripts/train/`).** Future folder with a `README.md` stub only. No empty `train.py` placeholder. Will require MPS compatibility triage of the upstream training loop and is unrelated to shipping the game.
- **Spatial attribution heatmap.** Worth ~1h of exploration during implementation to see if `stable_worldmodel` exposes latent gradients cleanly. If yes, add an optional `heatmapFrames` field to the JSON and ship. If no, drop silently. Do not land a partial implementation.
- **Frontend delta.** This pipeline implies three small changes on the TS side: `QuizQuestion` gains optional `tier` and `glitchFamily`; `UserAnswer.chosenClip` becomes `"A" | "B" | "neither"`; the home→play handoff samples 10 from the pool per the `distribution` hint. These are their own brainstorm → spec → plan cycle, triggered after this pipeline lands.

## Open questions

None at design time. All decisions above were taken in the brainstorm.
