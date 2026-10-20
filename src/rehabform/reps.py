"""Rep segmentation from one joint angle, e.g. the knee angle for a squat.

No training data needed: a rep is one dip (or rise) of the main joint angle.
The same logic runs in the phone app (port to JavaScript later).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

# Main joint angle per exercise and whether a rep is a dip ("min") or rise ("max").
# Extend this table when the v1 exercise list is agreed with a physiotherapist.
EXERCISE_ANGLE = {
    "squat": ("l_knee", "min"),
    "sit_to_stand": ("l_hip", "max"),
    "lunge": ("l_knee", "min"),
    "shoulder_abduction": ("l_shoulder_abd", "max"),
    "elbow_flexion": ("l_elbow", "min"),
}


@dataclass
class Rep:
    start: int  # frame index where the rep starts
    peak: int   # frame index of the deepest / highest point
    end: int    # frame index where the rep ends
    depth: float  # angle change from start to peak, degrees


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average; window in frames (odd number recommended)."""
    if window <= 1:
        return x.astype(float)
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x.astype(float), pad, mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def count_reps(
    angle: np.ndarray,
    mode: str = "min",
    fps: float = 30.0,
    min_depth_deg: float = 20.0,
    min_rep_seconds: float = 0.8,
    smooth_window: int = 5,
) -> list[Rep]:
    """Find reps in a per-frame angle signal.

    mode="min": a rep is a dip (squat knee angle goes 170 -> 90 -> 170).
    mode="max": a rep is a rise (shoulder abduction goes 20 -> 160 -> 20).
    """
    a = smooth(np.asarray(angle, dtype=float), smooth_window)
    sig = -a if mode == "min" else a
    distance = max(1, int(min_rep_seconds * fps))
    peaks, props = find_peaks(sig, prominence=min_depth_deg, distance=distance)

    reps: list[Rep] = []
    for i, p in enumerate(peaks):
        left = int(props["left_bases"][i])
        right = int(props["right_bases"][i])
        # Split shared bases between neighbouring reps at the midpoint.
        if i > 0:
            left = max(left, (peaks[i - 1] + p) // 2)
        if i < len(peaks) - 1:
            right = min(right, (p + peaks[i + 1]) // 2)
        reps.append(Rep(start=left, peak=int(p), end=right,
                        depth=float(abs(a[left] - a[p]))))
    return reps
