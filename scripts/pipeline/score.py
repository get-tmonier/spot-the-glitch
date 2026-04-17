"""Run LeWM inference over trajectories, emit per-step surprise arrays.

Surprise definition (per trajectory of length T):
    surprise[i] = MSE( predict(embed[ctx_start(i):i+1]),  embed[i+1:i+2] )
                  for i in [0, T-2]

where:
    * ``embed = PreJEPA.encode({"pixels": frames})`` is shape (1, T, P, d)
      with P patches per frame and d feature dim.
    * ``ctx_start(i) = max(0, i + 1 - history_size)`` — up to history_size
      frames of causal context ending at frame i.
    * ``predict(window)`` returns a per-frame prediction; we take ``[:, -1:]``
      which is the prediction of the frame immediately after the window.
    * The MSE is averaged over (P, d) dimensions.

Adaptation notes vs. the plan template:
    The upstream ``stable_worldmodel.wm.PreJEPA`` API exposes
    ``encode(info_dict, ...)`` (not ``encode(tensor)``) and returns per-patch
    embeddings ``(B, T, P, d)``. We pool patches via mean-MSE rather than
    flattening, which is equivalent and matches upstream's ``criterion`` style.

    Warm-up: the plan asks for exactly ``T-1`` values. PreJEPA's predictor uses
    causal attention, so a window of length 1 (single frame of context) still
    produces a well-defined prediction — just less informed. We therefore
    produce a real surprise value at every step ``i >= 0``; no NaN/zero
    sentinel is needed. Early values will tend to be higher because the model
    has less context, but this is the true signal — we do not mask it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pipeline import config, glitch, lewm_loader, simulate


def _frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, C, H, W) float32 in [0,1] on device."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).to(device).float() / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def _encode_all(
    model: torch.nn.Module,
    device: torch.device,
    frames: np.ndarray,
) -> torch.Tensor:
    """Return per-frame per-patch embeddings of shape (T, P, d) on device.

    Uses a single ``encode`` call over the whole trajectory: PreJEPA's
    ``_encode_image`` is agnostic to T (it flattens B*T before the backbone),
    so batching all T frames together is cheap and avoids redundant GPU
    round-trips.
    """
    pixels = _frames_to_tensor(frames, device)  # (T, C, H, W)
    # PreJEPA expects (B, T, C, H, W). We have one trajectory -> B=1.
    pixels = pixels.unsqueeze(0)  # (1, T, C, H, W)

    info: dict[str, torch.Tensor] = {"pixels": pixels}
    info = model.encode(info, pixels_key="pixels", emb_keys=[])
    embed = info["embed"]  # (1, T, P, d)
    return embed.squeeze(0)  # (T, P, d)


def surprise_curve(
    model: torch.nn.Module,
    device: torch.device,
    frames: np.ndarray,
) -> np.ndarray:
    """Return length-(T-1) float32 array of per-step surprise values.

    One forward pass through ``encode`` + one batched forward through
    ``predict`` over all T-1 sliding windows.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected frames of shape (T,H,W,3), got {frames.shape}")
    t_frames = int(frames.shape[0])
    if t_frames < 2:
        raise ValueError(f"need at least 2 frames, got {t_frames}")

    history_size = int(getattr(model, "history_size", 3))

    with torch.no_grad():
        # (T, P, d)
        embed = _encode_all(model, device, frames)
        t_total, n_patches, dim = embed.shape

        # Build a batch of sliding windows, each of length `history_size`,
        # left-padded (repeat frame 0) so every window has length history_size.
        # Window i (for i in [0, T-2]) ends at frame i; target is frame i+1.
        n_windows = t_total - 1
        windows = torch.empty(
            (n_windows, history_size, n_patches, dim),
            dtype=embed.dtype,
            device=device,
        )
        for i in range(n_windows):
            start = max(0, i + 1 - history_size)
            ctx = embed[start : i + 1]  # (<=history_size, P, d)
            pad = history_size - ctx.shape[0]
            if pad > 0:
                # Left-pad by repeating the earliest available frame, matching
                # upstream's padding convention in _encode_video.
                pad_block = ctx[:1].expand(pad, -1, -1)
                ctx = torch.cat([pad_block, ctx], dim=0)
            windows[i] = ctx

        # Batched prediction: (N, history_size, P, d) -> (N, history_size, P, d)
        preds = model.predict(windows)
        assert preds.ndim == windows.ndim, (
            f"model.predict returned unexpected rank {preds.ndim}; "
            f"expected {windows.ndim} to match input windows {tuple(windows.shape)!r}"
        )
        assert preds.shape[0] == windows.shape[0] and preds.shape[-2:] == windows.shape[-2:], (
            f"model.predict returned shape {tuple(preds.shape)!r}; "
            f"expected batch/patch/dim to match input {tuple(windows.shape)!r}"
        )
        # Take the last-timestep prediction of each window — that's the
        # model's estimate of the frame immediately following the context.
        preds_next = preds[:, -1]  # (N, P, d)

        targets = embed[1:]  # (N, P, d), the "ground-truth" next embeddings

        # Mean squared error, averaged over (P, d) per step.
        diff = preds_next - targets
        per_step = (diff * diff).mean(dim=(1, 2))  # (N,)

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
