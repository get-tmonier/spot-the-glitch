"""LeWM model loading + device selection for M3.

The upstream ``quentinll/lewm-pusht`` checkpoint ships a state_dict + Hydra config.
The config references ``stable_worldmodel.wm.lewm.LeWM`` and sub-classes from
``stable_worldmodel.wm.lewm.module``, which don't exist in the publicly released
``stable-worldmodel==0.0.6``.

We reconstruct the architecture from the real source (lucas-maes/le-wm):
  encoder.*          — ViT-tiny (HuggingFace format, via stable_pretraining)
  predictor.*        — ARPredictor: causal transformer conditioned on action embeddings
  action_encoder.*   — Embedder: Conv1d + 2-layer MLP mapping 10-D actions → 192-D
  projector.*        — MLP with BatchNorm1d projecting CLS token into predictor space
  pred_proj.*        — MLP with BatchNorm1d projecting predictor output back to embed space

score.py calls model.encode() and model.predict(emb, act_emb).
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
# Helpers
# ---------------------------------------------------------------------------


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# LeWM sub-modules (state-dict-compatible implementations)
# Faithfully reconstructed from lucas-maes/le-wm (module.py / jepa.py).
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),        # 0 — weights at net.0.*
            nn.Linear(dim, mlp_dim),  # 1 — weights at net.1.*
            nn.GELU(),                # 2
            nn.Dropout(dropout),      # 3
            nn.Linear(mlp_dim, dim),  # 4 — weights at net.4.*
            nn.Dropout(dropout),      # 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConditionalBlock(nn.Module):
    """ConditionalBlock from module.py — AdaLN-zero conditioned on action embedding."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = _Attention(dim, heads, dim_head, dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout)
        # elementwise_affine=False: no learned affine params → not in state_dict
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _PredTransformer(nn.Module):
    def __init__(
        self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [_ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class _Predictor(nn.Module):
    """ARPredictor from module.py: pos embedding + causal transformer conditioned on actions."""

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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d)  c: (N, T, act_emb_dim=d)
        t_steps = x.size(1)
        x = x + self.pos_embedding[:, :t_steps]
        return self.transformer(x, c)


class _Embedder(nn.Module):
    """Embedder from module.py — Conv1d patch embed + SiLU MLP → action embedding."""

    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        # mlp_scale=4 from Embedder defaults → hidden = 4 * emb_dim = 768 for emb_dim=192
        hidden = 4 * emb_dim
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden),  # 0
            nn.SiLU(),                     # 1
            nn.Linear(hidden, emb_dim),    # 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class _MLP(nn.Module):
    """MLP from module.py — Linear → BatchNorm1d → GELU → Linear."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 0
            nn.BatchNorm1d(hidden_dim),         # 1
            nn.GELU(),                          # 2
            nn.Linear(hidden_dim, output_dim),  # 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeWM(nn.Module):
    """Le World Model — reconstructed from checkpoint state dict.

    encode(info) → info["embed"] of shape (B, T, d)
    predict(emb, act_emb) → (N, T, d)  predicted embeddings
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
        target: str = "embed",
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Encode frame sequences into CLS-token projected embeddings.

        Args:
            info: dict with pixels of shape (B, T, C, H, W).
            pixels_key: key for pixel tensor in info.
            target: output key in info; result shape (B, T, d).
        """
        pixels = info[pixels_key].float()  # (B, T, C, H, W)
        b_size, t_steps = pixels.shape[:2]
        frames = rearrange(pixels, "b t c h w -> (b t) c h w")
        raw = self.encoder(frames, interpolate_pos_encoding=True)
        # CLS token is index 0 in last_hidden_state
        cls_emb = raw.last_hidden_state[:, 0]  # (B*T, d)
        proj = self.projector(cls_emb)  # (B*T, d)
        info[target] = rearrange(proj, "(b t) d -> b t d", b=b_size)
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next-frame embeddings from context embeddings + action embeddings.

        Args:
            emb: (N, T, d) — context frame embeddings (already projected).
            act_emb: (N, T, act_emb_dim) — action embeddings from action_encoder.

        Returns:
            preds: (N, T, d) — predicted embeddings via pred_proj.
        """
        preds = self.predictor(emb, act_emb)  # (N, T, d)
        n_batch, t_steps, dim = preds.shape
        preds = self.pred_proj(preds.reshape(n_batch * t_steps, dim))
        return preds.reshape(n_batch, t_steps, dim)


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
    """Load the pretrained LeWM Push-T model in eval mode."""
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
    """Forward-pass random frames+actions through encode+predict. Returns True on success."""
    import numpy as np

    try:
        loaded = load()
        model = loaded.model
        device = loaded.device

        # Frameskip=5: 6 frames, 5 action windows of 10D each
        t_frames = 6
        act_input_dim = 10  # 5 raw actions × 2D = 10
        dummy_pixels = torch.from_numpy(
            np.random.rand(1, t_frames, 3, config.RENDER_SIZE, config.RENDER_SIZE).astype("float32")
        ).to(device)
        dummy_actions = torch.zeros(1, t_frames - 1, act_input_dim, device=device)

        with torch.no_grad():
            info = model.encode({"pixels": dummy_pixels})
            embed = info["embed"]  # (1, T, d)
            assert embed.ndim == 3, f"encode returned wrong rank: {embed.shape}"

            act_emb = model.action_encoder(dummy_actions)  # (1, T-1, d)
            # Use first 3 frames as context window
            window_emb = embed[:, :3]   # (1, 3, d)
            window_act = act_emb[:, :3]  # (1, 3, d) — pad: use what we have
            preds = model.predict(window_emb, window_act)
            assert preds.shape == window_emb.shape, (
                f"predict shape mismatch: {preds.shape} vs {window_emb.shape}"
            )

        print(f"[lewm_loader] smoke test OK on {device}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[lewm_loader] smoke test FAILED: {exc}")
        return False
