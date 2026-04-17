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
    _, h, w, _ = frames.shape
    factor_h = target // h
    factor_w = target // w
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
    """Load frames for a clip id; 'g_' prefix → glitched, otherwise normal trajectory."""
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

    # File size check — audit only clips this run produced, not stale files.
    over_budget = []
    for p in selection.pairs:
        for side in ("a", "b"):
            clip = config.PUBLIC_CLIPS_DIR / f"{p.id}_{side}.mp4"
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
