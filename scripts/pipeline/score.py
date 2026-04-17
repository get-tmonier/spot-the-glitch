"""Run LeWM inference over trajectories, emit per-step surprise arrays.

The model was trained with frameskip=5 (each world-model step covers 5 raw env
steps) and stacked 2D actions (5 × 2 = 10-D per step).  We apply the same
preprocessing here so inference matches the training distribution.

Surprise definition for a frameskipped sequence of length T_skip (typically 6):
    surprise[i] = MSE( predict(embed[ctx], act_emb[ctx])[:, -1],  embed[i+1] )
                  for i in [0, T_skip-2]

where:
    embed  — (T_skip, d) CLS-token projected embeddings from LeWM.encode()
    ctx    — up to history_size (=3) most recent frames / actions ending at i
    act_emb — (T_skip-1, d) action embeddings from LeWM.action_encoder()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torchvision.transforms import v2 as T

from pipeline import config, glitch, lewm_loader, simulate

# Training hyperparameters that define the model's input format
_FRAMESKIP = 5        # every 5th raw frame is one world-model step
_RAW_ACT_DIM = 2      # Push-T 2D action space
_ACT_INPUT_DIM = _FRAMESKIP * _RAW_ACT_DIM  # 10-D stacked actions per step
_IMG_SIZE = 224       # ViT input size used during training

# ImageNet normalization applied during training (from stable_pretraining)
_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)

_FRAME_TRANSFORM = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    T.Resize(_IMG_SIZE, antialias=True),
])


def _frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, C, H, W) normalized float32 at 224×224 on device."""
    tensors = [_FRAME_TRANSFORM(frame) for frame in frames]
    return torch.stack(tensors).to(device)


def _apply_frameskip(
    frames: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample frames at _FRAMESKIP and stack consecutive raw actions.

    Args:
        frames:  (T_raw, H, W, C) uint8
        actions: (T_raw-1, raw_act_dim) float32

    Returns:
        skip_frames:  (T_skip, H, W, C)  where T_skip = ceil(T_raw / _FRAMESKIP)
        skip_actions: (T_skip-1, _ACT_INPUT_DIM) float32
    """
    t_raw = frames.shape[0]
    frame_idx = list(range(0, t_raw, _FRAMESKIP))
    skip_frames = frames[frame_idx]

    n_transitions = len(frame_idx) - 1
    skip_actions = np.zeros((n_transitions, _ACT_INPUT_DIM), dtype=np.float32)
    for i in range(n_transitions):
        start = frame_idx[i]
        end = frame_idx[i + 1]
        chunk = actions[start:end]  # (<=FRAMESKIP, raw_act_dim)
        # pad to FRAMESKIP if clip ends early
        if len(chunk) < _FRAMESKIP:
            pad = np.zeros((_FRAMESKIP - len(chunk), actions.shape[-1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        skip_actions[i] = chunk.reshape(_ACT_INPUT_DIM)

    return skip_frames, skip_actions


def surprise_curve(
    model: torch.nn.Module,
    device: torch.device,
    frames: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Return float32 surprise array after applying frameskip.

    Length is (T_skip - 1) where T_skip = ceil(T_raw / _FRAMESKIP).
    For a 30-frame clip with _FRAMESKIP=5 this gives 5 values.

    Args:
        model:   LeWM in eval mode.
        device:  device the model lives on.
        frames:  (T_raw, H, W, C) uint8 trajectory frames.
        actions: (T_raw-1, raw_act_dim) float32 raw actions.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected frames (T,H,W,3), got {frames.shape}")
    if len(frames) < 2:
        raise ValueError(f"need at least 2 frames, got {len(frames)}")

    skip_frames, skip_actions = _apply_frameskip(frames, actions)
    t_skip = len(skip_frames)  # e.g. 6
    history_size = int(getattr(model, "history_size", 3))

    with torch.no_grad():
        # Encode all frameskipped frames in one pass: (1, T_skip, C, H, W)
        pixels = _frames_to_tensor(skip_frames, device).unsqueeze(0)
        info = model.encode({"pixels": pixels})
        embed = info["embed"].squeeze(0)  # (T_skip, d)

        # Encode all stacked actions: (1, T_skip-1, 10) → (T_skip-1, d)
        act_t = torch.from_numpy(skip_actions).to(device).unsqueeze(0)  # (1, T_skip-1, 10)
        act_emb = model.action_encoder(act_t).squeeze(0)  # (T_skip-1, d)

        n_windows = t_skip - 1
        dim = embed.shape[-1]
        act_dim = act_emb.shape[-1]

        # Build sliding windows of length history_size, left-padded at the start.
        # Window i predicts embed[i+1] using context frames/actions [max(0,i+1-H)..i].
        emb_windows = torch.empty((n_windows, history_size, dim), device=device)
        act_windows = torch.empty((n_windows, history_size, act_dim), device=device)

        for i in range(n_windows):
            # Embeddings: context ending at frame i
            start = max(0, i + 1 - history_size)
            ctx_e = embed[start : i + 1]   # (<=H, d)
            ctx_a = act_emb[start : i + 1] # (<=H, d)
            pad = history_size - ctx_e.shape[0]
            if pad > 0:
                ctx_e = torch.cat([ctx_e[:1].expand(pad, -1), ctx_e], dim=0)
                ctx_a = torch.cat([ctx_a[:1].expand(pad, -1), ctx_a], dim=0)
            emb_windows[i] = ctx_e
            act_windows[i] = ctx_a

        # Batched prediction: (N, H, d) → take last-step output as next-frame estimate
        preds = model.predict(emb_windows, act_windows)  # (N, H, d)
        preds_next = preds[:, -1]   # (N, d)
        targets = embed[1:]         # (N, d)

        diff = preds_next - targets
        per_step = (diff * diff).mean(dim=1)  # (N,)

    return per_step.detach().to("cpu", dtype=torch.float32).numpy()


def score_dir(
    in_dir: Path,
    out_dir: Path,
    loaded: lewm_loader.LoadedModel | None = None,
) -> list[Path]:
    """Score every .npz file in in_dir. Writes <name>.npy to out_dir."""
    loaded = loaded or lewm_loader.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for npz_path in sorted(in_dir.glob("*.npz")):
        try:
            traj = simulate.load_trajectory(npz_path)
        except KeyError:
            traj = glitch.load_glitched(npz_path)
        curve = surprise_curve(loaded.model, loaded.device, traj.frames, traj.actions)
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
