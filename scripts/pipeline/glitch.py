"""Glitch injection — teleport and time-reversal."""

from __future__ import annotations

import json
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
    metadata: dict[str, object] = field(default_factory=dict)


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
    length = int(rng.integers(low=config.TIMEREV_SEGMENT_MIN, high=config.TIMEREV_SEGMENT_MAX + 1))
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
    """Shuffled plan splitting total into teleport / time-reversal by config shares."""
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
        metadata_json=np.array(json.dumps(g.metadata)),
    )


def load_glitched(path: Path) -> GlitchedTrajectory:
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
