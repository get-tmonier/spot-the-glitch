"""Export tests: JSON round-trips through the declared JSON Schema."""

from __future__ import annotations

import json

import jsonschema
import numpy as np
import pytest

from pipeline import config, curate, export


def _fake_selection(n: int = 3) -> curate.Selection:
    pairs = [
        curate.Pair(
            id=f"pair_{i:04d}",
            tier="medium" if i < 2 else "gotcha",
            glitch_family="teleport" if i < 2 else "none",
            clip_a_id=f"g_{i:03d}" if i < 2 else f"n_{i:03d}",
            clip_b_id=f"n_{i:03d}",
            clip_a_glitched=i < 2,
            clip_b_glitched=False,
        )
        for i in range(n)
    ]
    return curate.Selection(pairs=pairs)


def _fake_curves_for(selection: curate.Selection) -> dict[str, np.ndarray]:
    curves = {}
    for p in selection.pairs:
        for cid, glitched in ((p.clip_a_id, p.clip_a_glitched), (p.clip_b_id, p.clip_b_glitched)):
            arr = np.full(config.CLIP_STEPS - 1, 0.1, dtype=np.float32)
            if glitched:
                arr[15] = 0.9
            curves[cid] = arr
    return curves


def test_build_json_matches_schema():
    selection = _fake_selection()
    curves = _fake_curves_for(selection)
    doc = export.build_json(selection, curves)
    schema = json.loads(config.SCHEMA_FILE.read_text())
    jsonschema.validate(doc, schema)


def test_build_json_has_correct_top_level_fields():
    doc = export.build_json(_fake_selection(), _fake_curves_for(_fake_selection()))
    assert doc["schemaVersion"] == 1
    assert doc["clipFps"] == config.ENV_FPS
    assert doc["clipDurationSec"] == config.CLIP_DURATION_SEC
    assert set(doc["distribution"].keys()) == {"easy", "medium", "hard", "gotcha"}


def test_peak_pass_rate_all_spiky():
    curves = [np.array([0.1, 0.1, 0.9, 0.1]), np.array([0.1, 0.1, 1.0, 0.1])]
    rate = export.peak_pass_rate(curves)
    assert rate == 1.0


def test_peak_pass_rate_mixed():
    curves = [
        np.array([0.1, 0.1, 0.9, 0.1]),  # ratio 9 > 3 → pass
        np.array([0.1, 0.1, 0.2, 0.1]),  # ratio 2 < 3 → fail
    ]
    assert export.peak_pass_rate(curves) == pytest.approx(0.5)


def test_gotcha_check_fails_on_spike():
    curves = [np.array([0.1, 0.1, 0.9, 0.1])]
    assert not export.gotcha_curves_flat(curves)


def test_gotcha_check_passes_on_flat():
    curves = [np.array([0.1, 0.11, 0.12, 0.1])]
    assert export.gotcha_curves_flat(curves)
