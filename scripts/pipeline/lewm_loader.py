"""LeWM model loading + device selection for M3.

Discovery notes (swm 0.0.6):
    * No ``stable_worldmodel.models`` module exists. The world-model class lives at
      ``stable_worldmodel.wm.PreJEPA`` (LeWM-style JEPA).
    * Upstream does NOT ship ``state_dict`` + ``config`` style checkpoints. Instead
      ``torch.save(module, path)`` is used and ``torch.load(..., weights_only=False)``
      rehydrates the full ``torch.nn.Module`` object. See
      ``stable_worldmodel.policy._load_model_with_attribute`` / ``AutoCostModel``.
    * PreJEPA exposes ``encode(info_dict, ...)``, ``predict(embedding)``,
      ``rollout(info, action_sequence)`` and ``get_cost(info_dict, actions)``.
      There is no plain ``forward(pixels)`` — call ``encode`` through an info dict.

We adopt the upstream loading convention but stay defensive: if the checkpoint
turns out to be a state-dict style file (older plan template) we fall back to
reconstructing via ``swm.wm.PreJEPA(**config)``.
"""

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


def _find_module_with(root: torch.nn.Module, attr: str) -> torch.nn.Module | None:
    """Recursively search ``root`` for a submodule exposing ``attr``."""
    if hasattr(root, attr):
        return root
    for child in root.children():
        found = _find_module_with(child, attr)
        if found is not None:
            return found
    return None


def load(checkpoint_path: Path | None = None, prefer_mps: bool = True) -> LoadedModel:
    """Load the pretrained LeWM Push-T model in eval mode.

    The Push-T LeWM checkpoint shipped by the upstream project is a pickled
    ``torch.nn.Module`` (not a ``state_dict``). We load the full object, then
    scan for a submodule with ``get_cost`` — that's the PreJEPA world model.

    Raises ``FileNotFoundError`` if the checkpoint is missing (T10 downloads it).
    Raises ``RuntimeError`` if the loaded object doesn't match any known layout.
    """
    import stable_worldmodel as swm  # local import so tests can run without package

    path = checkpoint_path or config.CHECKPOINT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {path}. Run `bash setup.sh` to download it."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Case 1: upstream convention — ckpt is a full torch.nn.Module pickle.
    if isinstance(ckpt, torch.nn.Module):
        # Prefer the submodule that implements the LeWM planning interface.
        model = _find_module_with(ckpt, "get_cost") or ckpt
    # Case 2: legacy / plan-template layout — dict with state_dict + config.
    elif isinstance(ckpt, dict):
        cfg = ckpt.get("config", {}) or {}
        state_dict = ckpt.get("state_dict", ckpt)

        if hasattr(swm, "wm") and hasattr(swm.wm, "PreJEPA"):
            try:
                model = swm.wm.PreJEPA(**cfg)
            except TypeError as exc:
                raise RuntimeError(
                    "lewm_loader.load(): failed to construct swm.wm.PreJEPA "
                    f"from checkpoint config={cfg!r}. Upstream API may have "
                    "changed; update lewm_loader.load()."
                ) from exc
        else:
            raise RuntimeError(
                "lewm_loader.load(): could not locate a LeWM model class in "
                "stable_worldmodel. Expected `swm.wm.PreJEPA`. "
                "Update lewm_loader.load() to match the installed API."
            )

        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
    else:
        raise RuntimeError(
            f"lewm_loader.load(): unexpected checkpoint type {type(ckpt)!r}. "
            "Expected torch.nn.Module (upstream) or dict (legacy)."
        )

    model.eval()

    device = pick_device(prefer_mps=prefer_mps)
    model.to(device)

    meta = {
        "device": str(device),
        "env_fps": config.ENV_FPS,
        "embedding_dim": getattr(model, "embedding_dim", None),
        "model_class": type(model).__name__,
    }
    return LoadedModel(model=model, device=device, meta=meta)


def smoke_test() -> bool:
    """Forward-pass a batch of random frames through the loaded model.

    Returns True on success. Prints a diagnostic line on failure but never raises
    — the caller decides whether to proceed on CPU.

    PreJEPA doesn't have a plain ``forward(pixels)``; it wants an info dict.
    We try a few signatures in order of preference.
    """
    import numpy as np

    try:
        loaded = load()
        dummy_pixels = torch.from_numpy(
            np.random.rand(1, 4, 3, config.RENDER_SIZE, config.RENDER_SIZE).astype("float32")
        ).to(loaded.device)

        with torch.no_grad():
            if hasattr(loaded.model, "encode"):
                # PreJEPA.encode takes an info-dict keyed by ``pixels``.
                try:
                    loaded.model.encode({"pixels": dummy_pixels})
                except TypeError:
                    # Legacy tensor-in/tensor-out encoder signature.
                    loaded.model.encode(dummy_pixels[:, 0])
            else:
                loaded.model(dummy_pixels)

        print(f"[lewm_loader] smoke test OK on {loaded.device}")
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic path
        print(f"[lewm_loader] smoke test FAILED: {exc}")
        return False
