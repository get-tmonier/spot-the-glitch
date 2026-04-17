"""Push-T rollout generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  -- registers `swm/*` envs with gymnasium

from pipeline import config


@dataclass
class Trajectory:
    frames: np.ndarray  # (T, H, W, C) uint8
    states: np.ndarray  # (T, S) float32
    actions: np.ndarray  # (T-1, A) float32
    seed: int


def _extract_state(obs: dict) -> np.ndarray:
    """PushT returns Dict({'proprio': (4,), 'state': (7,)}); we keep the full state."""
    return np.asarray(obs["state"], dtype=np.float32)


def rollout(seed: int, steps: int = config.CLIP_STEPS) -> Trajectory:
    """Run one deterministic Push-T rollout using a seeded random action policy."""
    env = gym.make(
        config.ENV_NAME,
        render_mode="rgb_array",
        resolution=config.RENDER_SIZE,
    )
    rng = np.random.default_rng(seed)

    try:
        obs, _info = env.reset(seed=seed)
        frames = [env.render()]
        states = [_extract_state(obs)]
        actions: list[np.ndarray] = []

        low = env.action_space.low
        high = env.action_space.high
        for _ in range(steps - 1):
            action = rng.uniform(low=low, high=high).astype(np.float32)
            obs, _reward, _terminated, _truncated, _info = env.step(action)
            frames.append(env.render())
            states.append(_extract_state(obs))
            actions.append(action)
    finally:
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
    with np.load(path) as data:
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
