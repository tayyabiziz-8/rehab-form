"""Sanity tests on synthetic skeletons. Run: python -m pytest tests -q  (or python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehabform.reps import count_reps  # noqa: E402
from rehabform.skeleton import J, MEDIAPIPE_33, features, joint_angles, normalise, to_shared  # noqa: E402


def standing_pose_33(scale: float = 1.0, shift=(0.0, 0.0)) -> np.ndarray:
    """A crude upright skeleton in MediaPipe layout (image coords, y down)."""
    pts = np.zeros((33, 2), dtype=np.float32)
    named = {
        "nose": (0.0, -0.9),
        "l_shoulder": (0.2, -0.6), "r_shoulder": (-0.2, -0.6),
        "l_elbow": (0.25, -0.3), "r_elbow": (-0.25, -0.3),
        "l_wrist": (0.28, 0.0), "r_wrist": (-0.28, 0.0),
        "l_hip": (0.15, 0.0), "r_hip": (-0.15, 0.0),
        "l_knee": (0.15, 0.45), "r_knee": (-0.15, 0.45),
        "l_ankle": (0.15, 0.9), "r_ankle": (-0.15, 0.9),
    }
    for name, (x, y) in named.items():
        pts[MEDIAPIPE_33[name]] = (x * scale + shift[0], y * scale + shift[1])
    return pts


def test_normalise_removes_position_and_scale():
    a = to_shared(standing_pose_33()[None], "mediapipe33")
    b = to_shared(standing_pose_33(scale=3.0, shift=(5.0, -2.0))[None], "mediapipe33")
    assert np.allclose(normalise(a), normalise(b), atol=1e-5)


def test_straight_knee_is_180_degrees():
    seq = to_shared(standing_pose_33()[None], "mediapipe33")
    ang = joint_angles(normalise(seq))
    assert abs(ang["l_knee"][0] - 180.0) < 1.0
    assert ang["trunk_lean"][0] < 1.0


def test_feature_width():
    seq = np.repeat(standing_pose_33()[None], 10, axis=0)
    f = features(seq, "mediapipe33")
    assert f.shape == (10, 13 * 2 + 9)


def test_count_three_squats():
    fps = 30
    t = np.arange(0, 9, 1 / fps)                    # 9 s, 3 reps of 3 s each
    knee = 170 - 40 * (1 - np.cos(2 * np.pi * t / 3))  # 170 -> 90 -> 170
    knee += np.random.default_rng(0).normal(0, 2, size=t.shape)
    reps = count_reps(knee, mode="min", fps=fps)
    assert len(reps) == 3
    assert all(r.depth > 60 for r in reps)


def test_ignores_small_wobble():
    t = np.arange(0, 6, 1 / 30)
    knee = 170 + 5 * np.sin(2 * np.pi * t)          # 5-degree wobble, no real rep
    assert count_reps(knee, mode="min", fps=30) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
