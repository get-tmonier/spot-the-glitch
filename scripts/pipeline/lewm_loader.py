"""LeWM model loading + device selection for M3.

The upstream ``quentinll/lewm-pusht`` checkpoint ships a state_dict + Hydra config.
The config references ``stable_worldmodel.wm.lewm.LeWM`` and sub-classes from
``stable_worldmodel.wm.lewm.module``, which don't exist in the publicly released
``stable-worldmodel==0.0.6``.

We reconstruct the architecture from the state dict keys:

  encoder.*          — ViT-tiny (HuggingFace format, via stable_pretraining)
  predictor.*        — temporal DiT-style causal transformer (T=3 timesteps)
  action_encoder.*   — 1-D conv + MLP action embedder (unused at inference)
  projector.*        — MLP projector with BatchNorm (unused at inference)
  pred_proj.*        — MLP prediction projector (unused at inference)

score.py only needs model.encode() and model.predict().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn

from pipeline import config

# ---------------------------------------------------------------------------
# LeWM sub-modules (state-dict-compatible implementations)
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim))

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        if scale is not None:
            h = h * (1.0 + scale) + shift
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.net[0](x)
        if scale is not None:
            h = h * (1.0 + scale) + shift
        h = self.net[1](h)
        h = self.net[2](h)
        h = self.net[3](h)
        h = self.net[4](h)
        return h


class _AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = _Attention(dim, heads, dim_head, dropout)
        self.mlp = _FeedForward(dim, mlp_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mods = self.adaLN_modulation(c)  # (B, T, 6*d)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mods.chunk(6, dim=-1)
        x = x + gate_a * self.attn(x, scale=scale_a, shift=shift_a)
        x = x + gate_m * self.mlp(x, scale=scale_m, shift=shift_m)
        return x


class _PredTransformer(nn.Module):
    def __init__(
        self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [_AdaLNBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class _Predictor(nn.Module):
    def __init__(
        self,
        num_frames: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,  # noqa: ARG002
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,  # noqa: ARG002
        **_: object,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.transformer = _PredTransformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) — one token per timestep (patch-pooled)
        t_steps = x.shape[1]
        pos = self.pos_embedding[:, :t_steps].expand(x.shape[0], -1, -1)
        x = x + pos
        return self.transformer(x, pos)


class _Embedder(nn.Module):
    """Action encoder: Conv1d patch embed + 2-layer MLP."""

    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.GELU(),
            nn.Linear(768, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class _MLP(nn.Module):
    """Projector / pred_proj: Linear → BatchNorm1d → GELU → Linear."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeWM(nn.Module):
    """Le World Model — reconstructed from checkpoint state dict.

    Only encode() and predict() are called at inference by score.py.
    action_encoder / projector / pred_proj are present for load_state_dict
    compatibility (strict=True) but are not invoked during scoring.
    """

    history_size: int = 3

    def __init__(
        self,
        encoder: nn.Module,
        predictor: _Predictor,
        action_encoder: _Embedder,
        projector: _MLP,
        pred_proj: _MLP,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj

    def encode(
        self,
        info: dict[str, torch.Tensor],
        pixels_key: str = "pixels",
        emb_keys: list[str] | None = None,  # noqa: ARG002
        target: str = "embed",
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Encode frame sequences into per-patch embeddings.

        Args:
            info: dict with pixels key of shape (B, T, C, H, W).
            pixels_key: key name for pixels in info.
            emb_keys: extra encoder keys; pass [] to skip action encoding.
            target: output key in info for shape (B, T, P, d).
        """
        pixels = info[pixels_key].float()  # (B, T, C, H, W)
        b_size, t_steps = pixels.shape[:2]
        frames = rearrange(pixels, "b t c h w -> (b t) c h w")
        raw = self.encoder(frames, interpolate_pos_encoding=True)
        if hasattr(raw, "last_hidden_state"):
            embed = raw.last_hidden_state[:, 1:]  # drop CLS: (B*T, P, d)
        else:
            embed = raw.logits.unsqueeze(1)
        embed = rearrange(embed.detach(), "(b t) p d -> b t p d", b=b_size)
        info[target] = embed
        return info

    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        """Predict next-frame patch embeddings from a context window.

        Args:
            windows: (N, T, P, d) — batch of sliding context windows.

        Returns:
            preds: (N, T, P, d) — broadcast predicted embeddings.
        """
        _n, _t, n_patches, _d = windows.shape
        x = windows.mean(dim=2)  # pool patches → (N, T, d)
        preds = self.predictor(x)  # (N, T, d)
        return preds.unsqueeze(2).expand(-1, -1, n_patches, -1)  # (N, T, P, d)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_lewm(cfg: dict, state_dict: dict) -> LeWM:
    from stable_pretraining.backbone.utils import vit_hf  # type: ignore[import-untyped]

    enc = cfg["encoder"]
    encoder = vit_hf(
        size=enc["size"],
        patch_size=enc["patch_size"],
        image_size=enc["image_size"],
        pretrained=enc.get("pretrained", False),
        use_mask_token=enc.get("use_mask_token", False),
    )

    p = cfg["predictor"]
    predictor = _Predictor(
        num_frames=p["num_frames"],
        input_dim=p["input_dim"],
        hidden_dim=p["hidden_dim"],
        output_dim=p["output_dim"],
        depth=p["depth"],
        heads=p["heads"],
        mlp_dim=p["mlp_dim"],
        dim_head=p["dim_head"],
        dropout=p.get("dropout", 0.0),
        emb_dropout=p.get("emb_dropout", 0.0),
    )

    ae = cfg["action_encoder"]
    action_encoder = _Embedder(input_dim=ae["input_dim"], emb_dim=ae["emb_dim"])

    pr = cfg["projector"]
    projector = _MLP(
        input_dim=pr["input_dim"], output_dim=pr["output_dim"], hidden_dim=pr["hidden_dim"]
    )

    pp = cfg["pred_proj"]
    pred_proj = _MLP(
        input_dim=pp["input_dim"], output_dim=pp["output_dim"], hidden_dim=pp["hidden_dim"]
    )

    model = LeWM(encoder, predictor, action_encoder, projector, pred_proj)
    model.load_state_dict(state_dict, strict=True)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    """Load the pretrained LeWM Push-T model in eval mode.

    Raises FileNotFoundError if the checkpoint is missing.
    Raises RuntimeError if architecture build or weight load fails.
    """
    path = checkpoint_path or config.CHECKPOINT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {path}. Run `bash setup.sh` to download it."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        model: nn.Module = ckpt
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and "config" in ckpt:
        try:
            model = _build_lewm(ckpt["config"], ckpt["state_dict"])
        except Exception as exc:
            raise RuntimeError(f"lewm_loader.load(): failed to build/load LeWM — {exc}") from exc
    else:
        raise RuntimeError(
            f"lewm_loader.load(): unexpected checkpoint type {type(ckpt)!r}. "
            "Expected dict with 'state_dict' and 'config' keys."
        )

    model.eval()
    device = pick_device(prefer_mps=prefer_mps)
    model.to(device)

    return LoadedModel(
        model=model,
        device=device,
        meta={
            "device": str(device),
            "env_fps": config.ENV_FPS,
            "model_class": type(model).__name__,
        },
    )


def smoke_test() -> bool:
    """Forward-pass random frames through encode+predict. Returns True on success."""
    import numpy as np

    try:
        loaded = load()
        t_frames = 4
        dummy = torch.from_numpy(
            np.random.rand(1, t_frames, 3, config.RENDER_SIZE, config.RENDER_SIZE).astype("float32")
        ).to(loaded.device)

        with torch.no_grad():
            info = loaded.model.encode({"pixels": dummy})
            embed = info["embed"]  # (1, T, P, d)
            assert embed.ndim == 4, f"encode returned wrong rank: {embed.shape}"
            window = embed[:, :3]  # (1, 3, P, d)
            preds = loaded.model.predict(window)
            assert preds.shape == window.shape, (
                f"predict shape mismatch: {preds.shape} vs {window.shape}"
            )

        print(f"[lewm_loader] smoke test OK on {loaded.device}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[lewm_loader] smoke test FAILED: {exc}")
        return False
