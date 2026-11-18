"""Recording-level features (what the detectors threshold) and the per-frame
series behind each one (where in time a finding happens).

Input: BlazePose (T, 33, 3) in pixel units, un-mirrored, sides fixed
(``rehabform.keraal_groups.load_blazepose_g12`` or the live app pipeline).

Every feature is either a validated measure from ``rehabform.angles`` or a
simple geometric ratio in shoulder widths. The definitions here are the ones
the thresholds in config/findings_thresholds.json were fitted on, so the
findings engine and the evaluation scripts must both use this module.
"""
from __future__ import annotations

import numpy as np

from .angles import compute_all

FPS = 10.0  # Keraal 1A/2A video and BlazePose rate
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24


def _p(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def _shoulder_width(P: np.ndarray) -> float:
    return float(np.nanmedian(np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=1)))


def _reaching_side(m: dict) -> str:
    p95 = {s: _p(m[f"arm_elevation_{s}"], 95) for s in "lr"}
    return max(p95, key=lambda s: p95[s] if np.isfinite(p95[s]) else -1e9)


def series(bp: np.ndarray, measures: dict | None = None) -> dict[str, np.ndarray]:
    """Per-frame series, one per feature family (keys = feature names they feed)."""
    m = measures if measures is not None else {k: v for k, (v, _ok) in compute_all(bp).items()}
    P = bp[..., :2]
    sw = _shoulder_width(P)
    hi = _reaching_side(m)
    lo = "r" if hi == "l" else "l"
    T = len(bp)
    lean = m["trunk_lean"]
    lean0 = np.nanmedian(lean[: max(3, T // 10)])
    sh_y = (P[:, L_SH, 1] + P[:, R_SH, 1]) / 2
    arm_mean = np.nanmean(np.stack([m["arm_elevation_l"], m["arm_elevation_r"]]), axis=0)
    elbow_mean = np.nanmean(np.stack([m["elbow_flexion_l"], m["elbow_flexion_r"]]), axis=0)
    rot = m["trunk_rotation"]
    return {
        "arm_peak_high": m[f"arm_elevation_{hi}"],
        "arm_peak_low": m[f"arm_elevation_{lo}"],
        "arm_peak_mean": arm_mean,
        "arm_range_mean": arm_mean,
        "elbow_reach_p50": m[f"elbow_flexion_{hi}"],
        "elbow_reach_p95": m[f"elbow_flexion_{hi}"],
        "elbow_mean_p50": elbow_mean,
        "lean_abs_p95": np.abs(lean - lean0),
        "lean_range": np.abs(lean - lean0),
        "rotation_range": np.abs(rot - np.nanmedian(rot)),
        "wrist_spread_p95": np.linalg.norm(P[:, L_WR] - P[:, R_WR], axis=1) / sw,
        "elbow_spread_p95": np.linalg.norm(P[:, L_EL] - P[:, R_EL], axis=1) / sw,
        "wrist_height_p95": (sh_y - np.minimum(P[:, L_WR, 1], P[:, R_WR, 1])) / sw,
        "elbow_height_p95": (sh_y - (P[:, L_EL, 1] + P[:, R_EL, 1]) / 2) / sw,
        "_reaching_side": hi,
    }


def features(bp: np.ndarray) -> dict:
    """Recording-level features (identical to the 18 Nov pillar2_error_signals run)."""
    m = {k: v for k, (v, _ok) in compute_all(bp).items()}
    T = len(bp)
    P = bp[..., :2]
    sw = _shoulder_width(P)
    f = {"duration_s": T / FPS}
    f["movement"] = float(np.nanmean([np.nanstd(P[:, j], axis=0).sum() for j in (L_WR, R_WR)]) / sw)

    p95 = {s: _p(m[f"arm_elevation_{s}"], 95) for s in "lr"}
    hi = _reaching_side(m)
    lo = "r" if hi == "l" else "l"
    f["arm_peak_high"] = p95[hi]
    f["arm_peak_low"] = p95[lo]
    f["arm_peak_mean"] = float(np.nanmean([p95["l"], p95["r"]]))
    f["arm_range_mean"] = float(np.nanmean([_p(m[f"arm_elevation_{s}"], 95) - _p(m[f"arm_elevation_{s}"], 5)
                                            for s in "lr"]))
    f["elbow_reach_p50"] = _p(m[f"elbow_flexion_{hi}"], 50)
    f["elbow_reach_p95"] = _p(m[f"elbow_flexion_{hi}"], 95)
    f["elbow_mean_p50"] = float(np.nanmean([_p(m[f"elbow_flexion_{s}"], 50) for s in "lr"]))

    lean = m["trunk_lean"]
    lean0 = np.nanmedian(lean[: max(3, T // 10)])
    f["lean_abs_p95"] = _p(np.abs(lean - lean0), 95)
    f["lean_range"] = _p(lean, 95) - _p(lean, 5)
    rot = m["trunk_rotation"]
    f["rotation_range"] = _p(rot, 95) - _p(rot, 5)

    f["wrist_spread_p95"] = _p(np.linalg.norm(P[:, L_WR] - P[:, R_WR], axis=1) / sw, 95)
    f["elbow_spread_p95"] = _p(np.linalg.norm(P[:, L_EL] - P[:, R_EL], axis=1) / sw, 95)
    sh_y = (P[:, L_SH, 1] + P[:, R_SH, 1]) / 2
    f["wrist_height_p95"] = _p((sh_y - np.minimum(P[:, L_WR, 1], P[:, R_WR, 1])) / sw, 95)
    f["elbow_height_p95"] = _p((sh_y - (P[:, L_EL, 1] + P[:, R_EL, 1]) / 2) / sw, 95)
    return f


def peak_window(x: np.ndarray, top_pct: float = 90, max_gap: int = 3) -> tuple[int, int] | None:
    """Frames [start, end] of the longest run where x is at or above its
    ``top_pct`` percentile (short gaps bridged). This is the moment a finding
    refers to: the top of the movement for 'not enough' errors, the worst
    frames for 'too much' errors."""
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return None
    hot = ok & (x >= np.percentile(x[ok], top_pct))
    idx = np.flatnonzero(hot)
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return max(runs, key=lambda r: (r[1] - r[0], -r[0]))
