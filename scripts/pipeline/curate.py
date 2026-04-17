"""Tier assignment and pair selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from pipeline import config

Tier = Literal["easy", "medium", "hard", "gotcha"]


@dataclass
class NormalEntry:
    id: str
    curve: np.ndarray


@dataclass
class GlitchedEntry:
    id: str
    curve: np.ndarray
    tier: Tier
    glitch_family: str  # "teleport" | "time-reversal"
    source_id: str


@dataclass
class Pair:
    id: str
    tier: Tier
    glitch_family: str  # "teleport" | "time-reversal" | "none"
    clip_a_id: str
    clip_b_id: str
    clip_a_glitched: bool
    clip_b_glitched: bool


@dataclass
class Selection:
    pairs: list[Pair] = field(default_factory=list)


def peak_ratio(curve: np.ndarray) -> float:
    median = float(np.median(curve))
    if median <= 0:
        return 0.0
    return float(np.max(curve) / median)


def assign_tier(curve: np.ndarray) -> Tier:
    r = peak_ratio(curve)
    if r >= config.TIER_EASY_RATIO_MIN:
        return "easy"
    if r >= config.TIER_MEDIUM_RATIO_MIN:
        return "medium"
    return "hard"


def is_gotcha_eligible(curve: np.ndarray) -> bool:
    return peak_ratio(curve) <= config.GOTCHA_MAX_RATIO


def _sample(rng: np.random.Generator, pool: list, n: int) -> list:
    if len(pool) < n:
        raise RuntimeError(
            f"Not enough candidates: need {n}, have {len(pool)}. "
            f"Rerun `simulate` with higher --n, or relax tier thresholds in config."
        )
    indices = rng.permutation(len(pool))[:n]
    return [pool[i] for i in indices]


def select_pairs(
    *, glitched: list[GlitchedEntry], normal: list[NormalEntry], seed: int
) -> Selection:
    rng = np.random.default_rng(seed)

    # Bucket glitched entries by tier.
    buckets: dict[Tier, list[GlitchedEntry]] = {"easy": [], "medium": [], "hard": []}
    for g in glitched:
        if g.tier in buckets:
            buckets[g.tier].append(g)

    # Bucket normals by gotcha eligibility for A/B gotcha usage.
    gotcha_pool = [n for n in normal if is_gotcha_eligible(n.curve)]

    pairs: list[Pair] = []
    pair_idx = 0
    normal_used: set[str] = set()

    # Real (glitched vs normal) pairs.
    for tier in ("easy", "medium", "hard"):
        target = config.TARGET_PAIRS[tier]
        chosen = _sample(rng, buckets[tier], target)
        for g in chosen:
            # Pick a normal that isn't the glitched source (to avoid leaking).
            candidate_normals = [
                n for n in normal if n.id != g.source_id and n.id not in normal_used
            ]
            partner = _sample(rng, candidate_normals, 1)[0]
            normal_used.add(partner.id)
            # Randomize which side is A vs B.
            flip = bool(rng.integers(0, 2))
            if flip:
                clip_a_id, clip_b_id = g.id, partner.id
                a_glitched, b_glitched = True, False
            else:
                clip_a_id, clip_b_id = partner.id, g.id
                a_glitched, b_glitched = False, True
            pairs.append(
                Pair(
                    id=f"pair_{pair_idx:04d}",
                    tier=tier,
                    glitch_family=g.glitch_family,
                    clip_a_id=clip_a_id,
                    clip_b_id=clip_b_id,
                    clip_a_glitched=a_glitched,
                    clip_b_glitched=b_glitched,
                )
            )
            pair_idx += 1

    # Gotcha pairs — two normals, both clean.
    gotcha_available = [n for n in gotcha_pool if n.id not in normal_used]
    target_gotcha = config.TARGET_PAIRS["gotcha"]
    gotcha_chosen = _sample(rng, gotcha_available, target_gotcha * 2)
    for i in range(target_gotcha):
        a, b = gotcha_chosen[2 * i], gotcha_chosen[2 * i + 1]
        pairs.append(
            Pair(
                id=f"pair_{pair_idx:04d}",
                tier="gotcha",
                glitch_family="none",
                clip_a_id=a.id,
                clip_b_id=b.id,
                clip_a_glitched=False,
                clip_b_glitched=False,
            )
        )
        pair_idx += 1

    return Selection(pairs=pairs)


def save_selection(selection: Selection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pairs": [asdict(p) for p in selection.pairs]}
    path.write_text(json.dumps(payload, indent=2))


def load_selection(path: Path) -> Selection:
    payload = json.loads(path.read_text())
    return Selection(pairs=[Pair(**p) for p in payload["pairs"]])
