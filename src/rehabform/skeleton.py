"""Shared skeleton format for every dataset and for the phone app.

Every source skeleton (MediaPipe 33, Kinect v2 25, ...) is mapped onto the same
13 joints, then normalised so that camera distance and position do not matter.

Array convention used everywhere in this repo:
    seq: float array of shape (T, J, D)
        T = frames, J = joints, D = 2 or 3 coordinates (x, y[, z])
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. The shared 13-joint layout
# ---------------------------------------------------------------------------
SHARED_JOINTS = [
    "nose",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
]
J = {name: i for i, name in enumerate(SHARED_JOINTS)}

# ---------------------------------------------------------------------------
# 2. Source formats -> shared layout (index in the source skeleton)
# ---------------------------------------------------------------------------
# MediaPipe Pose Landmarker / BlazePose, 33 landmarks.
MEDIAPIPE_33 = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}

# Microsoft Kinect v2, 25 joints (used by IntelliRehabDS and Keraal).
# Kinect has no nose joint, so the head joint stands in for it.
KINECT_V2_25 = {
    "nose": 3,  # Head
    "l_shoulder": 4, "r_shoulder": 8,
    "l_elbow": 5, "r_elbow": 9,
    "l_wrist": 6, "r_wrist": 10,
    "l_hip": 12, "r_hip": 16,
    "l_knee": 13, "r_knee": 17,
    "l_ankle": 14, "r_ankle": 18,
}

# TODO(data lead): fill these in after opening the files. Rehab-Pile reports
# 22 joints for UI-PRMD and 18 for KIMORE, but the joint order must be read
# from each dataset's own documentation, not guessed.
UI_PRMD_22: dict[str, int] = {}
KIMORE_18: dict[str, int] = {}

FORMATS = {
    "mediapipe33": MEDIAPIPE_33,
    "kinect_v2_25": KINECT_V2_25,
    "ui_prmd_22": UI_PRMD_22,
    "kimore_18": KIMORE_18,
}


def to_shared(seq: np.ndarray, fmt: str) -> np.ndarray:
    """Select the 13 shared joints from a source skeleton.

    seq: (T, J_src, D) -> returns (T, 13, D)
    """
    mapping = FORMATS[fmt]
    if not mapping:
        raise NotImplementedError(
            f"Joint mapping for '{fmt}' is empty; fill it in skeleton.py first."
        )
    idx = [mapping[name] for name in SHARED_JOINTS]
    if seq.ndim != 3 or seq.shape[1] <= max(idx):
        raise ValueError(f"Expected (T, J, D) with J > {max(idx)}, got {seq.shape}")
    return seq[:, idx, :].astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Normalisation: centre on the mid-hip, scale by torso length
# ---------------------------------------------------------------------------
def normalise(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Make a (T, 13, D) shared-layout sequence position- and scale-invariant.

    - Every frame is centred on the mid-hip point.
    - The whole sequence is divided by the median torso length
      (mid-shoulder to mid-hip), so body size and camera distance cancel out.
    """
    seq = np.asarray(seq, dtype=np.float32)
    mid_hip = (seq[:, J["l_hip"]] + seq[:, J["r_hip"]]) / 2.0
    mid_sh = (seq[:, J["l_shoulder"]] + seq[:, J["r_shoulder"]]) / 2.0
    centred = seq - mid_hip[:, None, :]
    torso = np.linalg.norm(mid_sh - mid_hip, axis=-1)
    scale = np.nanmedian(torso)
    if not np.isfinite(scale) or scale < eps:
        raise ValueError("Torso length is zero or missing; check the joint mapping.")
    return centred / scale


# ---------------------------------------------------------------------------
# 4. Joint angles (degrees): camera-independent features clinicians understand
# ---------------------------------------------------------------------------
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at point b formed by a-b-c, per frame, in degrees."""
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


ANGLE_DEFS = {
    # name: (a, vertex, c)
    "l_knee": ("l_hip", "l_knee", "l_ankle"),
    "r_knee": ("r_hip", "r_knee", "r_ankle"),
    "l_hip": ("l_shoulder", "l_hip", "l_knee"),
    "r_hip": ("r_shoulder", "r_hip", "r_knee"),
    "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_shoulder_abd": ("l_hip", "l_shoulder", "l_elbow"),
    "r_shoulder_abd": ("r_hip", "r_shoulder", "r_elbow"),
}


def joint_angles(seq: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame joint angles from a (T, 13, D) sequence. Returns name -> (T,)."""
    out = {
        name: _angle(seq[:, J[a]], seq[:, J[v]], seq[:, J[c]])
        for name, (a, v, c) in ANGLE_DEFS.items()
    }
    # Trunk lean: angle between the hip->shoulder line and the vertical axis.
    # Image y points down, so "up" is (0, -1[, 0]).
    mid_hip = (seq[:, J["l_hip"]] + seq[:, J["r_hip"]]) / 2.0
    mid_sh = (seq[:, J["l_shoulder"]] + seq[:, J["r_shoulder"]]) / 2.0
    up = np.zeros(seq.shape[-1], dtype=np.float32)
    up[1] = -1.0
    out["trunk_lean"] = _angle(mid_sh, mid_hip, mid_hip + up)
    return out


def features(seq: np.ndarray, fmt: str | None = None) -> np.ndarray:
    """Full feature pipeline: map -> normalise -> [coords, angles].

    Returns (T, F) with F = 13 * D + 9 angle channels.
    Pass fmt=None if seq is already in the shared 13-joint layout.
    """
    if fmt is not None:
        seq = to_shared(seq, fmt)
    seq = normalise(seq)
    angles = joint_angles(seq)
    ang = np.stack([angles[k] for k in sorted(angles)], axis=-1) / 180.0
    return np.concatenate([seq.reshape(len(seq), -1), ang], axis=-1).astype(np.float32)
