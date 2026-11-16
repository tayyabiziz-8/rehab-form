"""Angle spec: the phone measures the app reports, computed exactly as validated.

Every measure here was checked against Vicon on Keraal Group 3 (539 recordings,
3 participants; Nov 2026). What the app may say about each measure, per
exercise, lives in ``config/trust_table.json`` and is read with ``TrustTable``.

Input: MediaPipe / BlazePose landmarks, shape (T, 33, 3), in PIXEL units
(x * width, y * height, z * width), image y pointing down, NOT mirrored
(the person's left side appears on the image right when facing the camera).
For Keraal files, ``rehabform.keraal.load_blazepose`` already returns this.

Measures (degrees)
  arm_elevation_l/r  signed arm angle in the image plane against the
                     shoulder-line perpendicular (chest-like reference):
                     0 = arm down, 90 = out to the side or forwards,
                     180 = overhead. Unwrapped over time.
  elbow_flexion_l/r  elbow bend in the image plane, 0 = straight.
  trunk_lean         sideways tilt of the hip-to-shoulder line from the image
                     vertical; positive = leaning to the person's LEFT.
  trunk_rotation     twist of the shoulder line about the vertical axis, from
                     BlazePose's relative depth; positive = left shoulder back.
                     Depth is weak: valid for timing only.

Each function returns (values, valid): ``valid`` is False on frames where the
measure must not be reported (missing landmarks, or a limb pointing towards or
away from the camera so its image-plane angle is meaningless).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MediaPipe Pose / BlazePose landmark indices (same order in both).
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

MIN_IN_PLANE = 0.5          # limb shorter than this fraction of its reference length -> not valid
DEFAULT_TRUST_TABLE = Path(__file__).resolve().parents[2] / "config" / "trust_table.json"

SIDES = {"l": (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, R_SHOULDER),
         "r": (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, L_SHOULDER)}


def _unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    ok = np.isfinite(a)
    if ok.sum() > 1:
        out[ok] = np.degrees(np.unwrap(np.radians(a[ok])))
    return out


def _length_ok(vec2d: np.ndarray, reference: float | None) -> np.ndarray:
    length = np.linalg.norm(vec2d, axis=-1)
    ref = reference if reference is not None else np.nanpercentile(length, 90)
    return np.isfinite(length) & (length >= MIN_IN_PLANE * ref)


def arm_elevation(lm: np.ndarray, side: str, arm_length_px: float | None = None):
    """Signed arm angle vs the shoulder-line perpendicular. Returns (deg, valid).

    ``arm_length_px``: the upper arm's full image length (e.g. from a calibration
    pose with the arm out to the side). If None, the 90th percentile over the
    sequence is used, which is fine for whole recordings, not for live streams.
    """
    sh, el, _, _, other = SIDES[side]
    P = lm[..., :2]
    arm = P[:, el] - P[:, sh]
    line = P[:, sh] - P[:, other]                                  # outward along the shoulders
    mid_sh = (P[:, L_SHOULDER] + P[:, R_SHOULDER]) / 2
    mid_hip = (P[:, L_HIP] + P[:, R_HIP]) / 2
    perp = np.stack([-line[:, 1], line[:, 0]], axis=1)
    perp[np.sum(perp * (mid_hip - mid_sh), axis=1) < 0] *= -1     # make it point towards the hips
    d = perp / (np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9)
    o = line / (np.linalg.norm(line, axis=1, keepdims=True) + 1e-9)
    angle = np.degrees(np.arctan2(np.sum(arm * o, axis=1), np.sum(arm * d, axis=1)))
    valid = _length_ok(arm, arm_length_px) & np.isfinite(angle)
    angle[~valid] = np.nan
    return _unwrap_deg(angle), valid


def elbow_flexion(lm: np.ndarray, side: str, upper_px: float | None = None, fore_px: float | None = None):
    """Elbow bend in the image plane (0 = straight). Returns (deg, valid)."""
    sh, el, wr, _, _ = SIDES[side]
    P = lm[..., :2]
    upper, fore = P[:, el] - P[:, sh], P[:, wr] - P[:, el]
    cos = np.sum(upper * fore, axis=1) / (np.linalg.norm(upper, axis=1) * np.linalg.norm(fore, axis=1) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    valid = _length_ok(upper, upper_px) & _length_ok(fore, fore_px) & np.isfinite(angle)
    angle[~valid] = np.nan
    return angle, valid


def trunk_lean(lm: np.ndarray):
    """Sideways tilt from the image vertical; + = towards the person's left."""
    P = lm[..., :2]
    trunk = (P[:, L_SHOULDER] + P[:, R_SHOULDER]) / 2 - (P[:, L_HIP] + P[:, R_HIP]) / 2
    angle = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))      # person's left = image right (+x)
    return angle, np.isfinite(angle)


def trunk_rotation(lm: np.ndarray):
    """Shoulder-line twist about the vertical, from BlazePose depth (timing only)."""
    line = lm[:, L_SHOULDER] - lm[:, R_SHOULDER]
    angle = _unwrap_deg(np.degrees(np.arctan2(line[:, 2], line[:, 0])))
    return angle, np.isfinite(angle)


def compute_all(lm: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All measures for a (T, 33, 3) pixel-unit sequence: name -> (values, valid)."""
    out = {}
    for side in ("l", "r"):
        out[f"arm_elevation_{side}"] = arm_elevation(lm, side)
        out[f"elbow_flexion_{side}"] = elbow_flexion(lm, side)
    out["trunk_lean"] = trunk_lean(lm)
    out["trunk_rotation"] = trunk_rotation(lm)
    return out


# ---------------------------------------------------------------------------
# Trust table
# ---------------------------------------------------------------------------
MODES = ("range", "relative", "timing_only", "not_reportable")


@dataclass
class Decision:
    mode: str                 # one of MODES
    note: str                 # short reason, shown to the clinician
    evidence: dict            # validation numbers behind the decision


class TrustTable:
    """Reads config/trust_table.json and answers: may this measure be reported
    for this exercise, and how?"""

    def __init__(self, data: dict):
        self.data = data
        self.measures = data["measures"]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TRUST_TABLE) -> "TrustTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def decide(self, measure: str, exercise: str, value_deg: float | None = None) -> Decision:
        """``measure`` may carry a side suffix (arm_elevation_l). ``value_deg`` lets
        value-dependent rules apply (overhead arms report relative change only)."""
        base = measure[:-2] if measure.endswith(("_l", "_r")) else measure
        spec = self.measures.get(base)
        if spec is None:
            return Decision("not_reportable", f"unknown measure {measure}", {})
        rule = spec["exercises"].get(exercise, spec.get("default", {"mode": "not_reportable",
                                                                     "note": "not validated for this exercise"}))
        mode, note = rule["mode"], rule.get("note", "")
        over = spec.get("overhead_rule")
        if over and value_deg is not None and np.isfinite(value_deg) and value_deg > over["above_deg"] \
                and MODES.index(over["mode"]) > MODES.index(mode):
            mode, note = over["mode"], over["note"]
        return Decision(mode, note, rule.get("evidence", {}))
