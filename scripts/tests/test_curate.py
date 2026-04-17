"""Unit tests for curation logic."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config, curate


def _flat(n: int = 29, level: float = 0.1) -> np.ndarray:
    return np.full(n, level, dtype=np.float32)


def _spike(n: int = 29, peak_at: int = 15, peak: float = 1.0, base: float = 0.1) -> np.ndarray:
    arr = np.full(n, base, dtype=np.float32)
    arr[peak_at] = peak
    return arr


def test_peak_ratio_flat():
    r = curate.peak_ratio(_flat())
    assert 0.9 <= r <= 1.1


def test_peak_ratio_spike():
    r = curate.peak_ratio(_spike(peak=1.0, base=0.1))
    assert r == pytest.approx(10.0, rel=1e-3)


def test_assign_tier_easy():
    assert curate.assign_tier(_spike(peak=2.0, base=0.1)) == "easy"  # ratio 20


def test_assign_tier_medium():
    # Ratio = 0.5 / 0.1 = 5.0 → medium (between 3.5 and 8.0)
    assert curate.assign_tier(_spike(peak=0.5, base=0.1)) == "medium"


def test_assign_tier_hard():
    # Ratio = 0.3 / 0.1 = 3.0 → hard (< 3.5)
    assert curate.assign_tier(_spike(peak=0.3, base=0.1)) == "hard"


def test_gotcha_filter_passes_flat():
    assert curate.is_gotcha_eligible(_flat(level=0.1))
    assert curate.is_gotcha_eligible(_flat(level=0.2))


def test_gotcha_filter_rejects_spike():
    # Ratio 10 > GOTCHA_MAX_RATIO (1.5) → not eligible
    assert not curate.is_gotcha_eligible(_spike(peak=1.0, base=0.1))


def test_select_pairs_respects_targets_and_tiers():
    # Build a pool of 40 glitched + 40 normal fake entries, with tier variety.
    glitched = []
    for i in range(40):
        if i < 10:
            curve = _spike(peak=2.0 + i * 0.05, base=0.1)  # easy
            tier = "easy"
            family = "teleport"
        elif i < 25:
            curve = _spike(peak=0.5, base=0.1)  # medium
            tier = "medium"
            family = "teleport" if i % 2 == 0 else "time-reversal"
        else:
            curve = _spike(peak=0.3, base=0.1)  # hard
            tier = "hard"
            family = "time-reversal"
        glitched.append(
            curate.GlitchedEntry(
                id=f"g_{i:03d}",
                curve=curve,
                tier=tier,
                glitch_family=family,
                source_id=f"n_{i:03d}",
            )
        )
    normal = [curate.NormalEntry(id=f"n_{i:03d}", curve=_flat(level=0.1)) for i in range(40)]

    selection = curate.select_pairs(glitched=glitched, normal=normal, seed=0)

    counts = {t: 0 for t in ("easy", "medium", "hard", "gotcha")}
    for pair in selection.pairs:
        counts[pair.tier] += 1
    assert counts == config.TARGET_PAIRS

    # Gotcha pairs must have both sides unglitched.
    for pair in selection.pairs:
        if pair.tier == "gotcha":
            assert pair.glitch_family == "none"
            assert pair.clip_a_glitched is False
            assert pair.clip_b_glitched is False


def test_select_pairs_deterministic():
    normal = [curate.NormalEntry(id=f"n_{i:03d}", curve=_flat()) for i in range(40)]
    # Peaks span all three tiers: 2.0→easy(20), 0.5→medium(5), 0.3→hard(3),
    # 1.5→easy(15), 0.4→medium(4). Over 40 entries: 16 easy, 16 medium, 8 hard.
    tier_peaks = [2.0, 0.5, 0.3, 1.5, 0.4]
    glitched = [
        curate.GlitchedEntry(
            id=f"g_{i:03d}",
            curve=_spike(peak=tier_peaks[i % 5], base=0.1),
            tier=curate.assign_tier(_spike(peak=tier_peaks[i % 5], base=0.1)),
            glitch_family="teleport",
            source_id=f"n_{i:03d}",
        )
        for i in range(40)
    ]
    a = curate.select_pairs(glitched=glitched, normal=normal, seed=42)
    b = curate.select_pairs(glitched=glitched, normal=normal, seed=42)
    assert [p.id for p in a.pairs] == [p.id for p in b.pairs]
