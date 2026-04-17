"""Unit tests for glitch injection logic."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config, glitch


def _make_traj(steps: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    frames = (rng.integers(0, 255, size=(steps, 16, 16, 3))).astype(np.uint8)
    states = rng.normal(size=(steps, 4)).astype(np.float32)
    states[:, :2] = np.cumsum(np.ones((steps, 2)), axis=0)  # monotonic xy for detection
    actions = rng.normal(size=(steps - 1, 2)).astype(np.float32)
    return glitch.BaseTrajectory(frames=frames, states=states, actions=actions, seed=seed)


def test_teleport_offset_within_bounds():
    traj = _make_traj()
    glitched = glitch.teleport(traj, seed=1)
    offset = np.linalg.norm(
        glitched.states[glitched.injection_index, :2] - traj.states[glitched.injection_index, :2]
    )
    assert config.TELEPORT_MIN_OFFSET_PX <= offset <= config.TELEPORT_MAX_OFFSET_PX


def test_teleport_divergence_exactly_at_injection():
    traj = _make_traj()
    glitched = glitch.teleport(traj, seed=1)
    k = glitched.injection_index
    # Frames/states before k are identical to source.
    np.testing.assert_array_equal(glitched.states[:k], traj.states[:k])
    # States at k are different (teleport applied).
    assert not np.array_equal(glitched.states[k, :2], traj.states[k, :2])


def test_teleport_injection_frame_range():
    traj = _make_traj()
    for s in range(20):
        glitched = glitch.teleport(traj, seed=s)
        assert (
            config.GLITCH_INJECTION_FRAME_MIN
            <= glitched.injection_index
            <= config.GLITCH_INJECTION_FRAME_MAX
        )


def test_time_reversal_segment_length_in_bounds():
    traj = _make_traj()
    for s in range(20):
        glitched = glitch.time_reversal(traj, seed=s)
        length = glitched.metadata["segment_length"]
        assert config.TIMEREV_SEGMENT_MIN <= length <= config.TIMEREV_SEGMENT_MAX


def test_time_reversal_segment_is_reversed_in_frames():
    traj = _make_traj()
    glitched = glitch.time_reversal(traj, seed=7)
    k = glitched.injection_index
    length = glitched.metadata["segment_length"]
    # The reversed segment in glitched equals reverse of the same segment in source.
    np.testing.assert_array_equal(
        glitched.frames[k : k + length],
        traj.frames[k : k + length][::-1],
    )


def test_time_reversal_outside_segment_unchanged():
    traj = _make_traj()
    glitched = glitch.time_reversal(traj, seed=7)
    k = glitched.injection_index
    length = glitched.metadata["segment_length"]
    np.testing.assert_array_equal(glitched.frames[:k], traj.frames[:k])
    np.testing.assert_array_equal(glitched.frames[k + length :], traj.frames[k + length :])


def test_apply_family_dispatches_correctly():
    traj = _make_traj()
    g1 = glitch.apply_family(traj, family="teleport", seed=3)
    assert g1.glitch_family == "teleport"
    g2 = glitch.apply_family(traj, family="time-reversal", seed=3)
    assert g2.glitch_family == "time-reversal"
    with pytest.raises(ValueError):
        glitch.apply_family(traj, family="nonsense", seed=3)


def test_family_mix_respects_shares():
    mix = glitch.plan_family_mix(total=100, seed=0)
    assert len(mix) == 100
    n_tele = sum(1 for f in mix if f == "teleport")
    n_trev = sum(1 for f in mix if f == "time-reversal")
    assert n_tele == int(100 * config.GLITCH_TELEPORT_SHARE)
    assert n_trev == 100 - n_tele
