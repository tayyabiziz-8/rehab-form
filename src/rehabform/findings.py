"""Findings engine: measurements + fitted thresholds + trust table -> structured findings.

A finding is the ONLY thing feedback may talk about. Each one says which
error was checked, whether it was found, the measured value and the typical
value of correct performance (only if the trust table allows quoting that
measure), and the time window it refers to. The feedback writer rewords
findings; the claim checker (rehabform.claims) rejects any sentence that is
not backed by one.

Thresholds come from config/findings_thresholds.json, written by
scripts/pillar2_detectors.py (fitted on Keraal 1A + 2A clinician labels).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .angles import TrustTable, compute_all
from .session_features import FPS, features, peak_window, segment_reps, series

STRICTNESS = {"strict": 0.0, "standard": 1.0, "lenient": 2.0}   # x the fitted borderline band
MIN_VALID_FRACTION = 0.5   # a check needs its measure in at least half of the frames
DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[2] / "config" / "findings_thresholds.json"

# feature -> (trust-table measure or None, plain label, unit, decimals, is a change/range measure)
FEATURE_INFO = {
    "lean_range": ("trunk_lean", "side-to-side range of your upper-body bend", "°", 0, True),
    "lean_abs_p95": ("trunk_lean", "sideways bend of your upper body", "°", 0, True),
    "rotation_range": ("trunk_rotation", "twist of your upper body", "°", 0, True),
    "arm_peak_mean": ("arm_elevation", "height of your arms", "°", 0, False),
    "arm_peak_high": ("arm_elevation", "height of your raised arm", "°", 0, False),
    "arm_range_mean": ("arm_elevation", "range of your arm movement", "°", 0, True),
    "elbow_reach_p50": ("elbow_flexion", "bend in your elbow", "°", 0, False),
    "elbow_reach_p95": ("elbow_flexion", "bend in your elbow", "°", 0, False),
    "elbow_mean_p50": ("elbow_flexion", "bend in your elbows", "°", 0, False),
    "wrist_height_p95": (None, "height of your hands above your shoulders", "shoulder widths", 1, False),
    "elbow_height_p95": (None, "height of your elbows above your shoulders", "shoulder widths", 1, False),
    "wrist_spread_p95": (None, "distance between your hands", "shoulder widths", 1, False),
    "elbow_spread_p95": (None, "distance between your elbows", "shoulder widths", 1, False),
}

# Plain-language wording per detector (patient-facing). Unknown detectors fall back to 'meaning'.
DETECTOR_TEXT = {
    "CTK_arms_not_raised": {"error": "raise your arms higher", "ok": "your arms reached a good height"},
    "CTK_error3": {"error": "keep your arms more open", "ok": "your arm position looked good"},
    "ELK_body_not_tilted": {"error": "bend your upper body further to the side",
                            "ok": "your upper body bent well to the side"},
    "RTK_arms_not_raised": {"error": "keep your arms raised", "ok": "your arms stayed at a good height"},
    "RTK_torso_not_rotated": {"error": "turn your upper body further", "ok": "your upper body turned well"},
}

BODY_PART_TEXT = {"Spine": "upper body", "BothArms": "both arms", "LeftArm": "left arm", "RightArm": "right arm"}


@dataclass
class Finding:
    id: str                     # F1, F2, ... (feedback sentences cite these)
    detector: str
    exercise: str
    status: str                 # "error" | "borderline" | "ok"  (borderline: shown to the clinician only)
    issue: str                  # what was checked (Keraal error meaning)
    advice: str                 # plain-language action (error) or praise (ok)
    body_part: str              # Spine | BothArms | LeftArm | RightArm
    side: str | None            # "left" / "right" when the finding is about one side
    feature: str
    measure_label: str
    unit: str
    quotable: bool              # may the numbers be shown to the user?
    value: float | None         # None when not quotable
    typical: float | None       # typical value of correct performance (None when not quotable)
    threshold: float | None
    severity: str | None        # "small" | "large" for errors when the measure orders severity
    start_s: float | None
    end_s: float | None
    trust_mode: str | None
    trust_note: str = ""
    evidence: dict = field(default_factory=dict)
    window_source: str = ""     # "worst_rep" | "peak" : how start_s / end_s were chosen
    reps: list = field(default_factory=list)   # [{"start_s", "end_s", "peak"}] for the clinician replay

    def allowed_numbers(self) -> list[float]:
        nums = [x for x in (self.start_s, self.end_s) if x is not None]
        if self.quotable:
            nums += [x for x in (self.value, self.typical) if x is not None]
        return nums

    def to_dict(self) -> dict:
        return asdict(self)


def _nanmax(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else float("nan")


def _round(x: float, decimals: int) -> float:
    return float(round(x, decimals)) if decimals else float(int(round(x)))


class FindingsEngine:
    def __init__(self, thresholds_path: str | Path = DEFAULT_THRESHOLDS, trust: TrustTable | None = None,
                 include_unvalidated: bool = False):
        cfg = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
        self.detectors = {k: v for k, v in cfg["detectors"].items() if include_unvalidated or v.get("usable", True)}
        self.trust = trust or TrustTable.load()
        self.skipped: list[str] = []

    def _quotable(self, det: dict, feature: str, exercise: str, value: float) -> tuple[bool, str | None, str]:
        measure, _label, _unit, _dec, is_change = FEATURE_INFO.get(feature, (None, feature, "", 1, False))
        if not det.get("reportable", False):
            return False, None, "proxy measure: used to detect, never quoted"
        if measure is None:
            return True, None, "geometric ratio in shoulder widths"
        d = self.trust.decide(measure, exercise, value_deg=value)
        if d.mode == "range" or (d.mode == "relative" and is_change):
            return True, d.mode, d.note
        return False, d.mode, d.note

    def run(self, bp: np.ndarray, exercise: str, strictness: str = "standard") -> list[Finding]:
        """Findings for one recording / session. ``bp``: (T, 33, 3) pixel units, un-mirrored, sides fixed.
        A check is skipped when its measure is missing in more than half of the frames.

        ``strictness`` (the clinician's setting) scales the borderline band just past each threshold:
        strict = no band (correct everything past the threshold), standard = the fitted band,
        lenient = twice the band. Thresholds themselves never move."""
        if strictness not in STRICTNESS:
            raise ValueError(f"strictness must be one of {sorted(STRICTNESS)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # empty slices when the person is lost
            return self._run(bp, exercise, STRICTNESS[strictness])

    def _run(self, bp: np.ndarray, exercise: str, band_scale: float = 1.0) -> list[Finding]:
        m = {k: v for k, (v, _ok) in compute_all(bp).items()}
        feats = features(bp)
        ser = series(bp, m)
        out: list[Finding] = []
        self.skipped = []                    # checks for this exercise that could not be judged (too little data)
        for det_id, det in self.detectors.items():
            if det["exercise"] != exercise:
                continue
            feat = det["feature"]
            value = feats.get(feat, float("nan"))
            if not np.isfinite(value):
                self.skipped.append(det_id)
                continue
            below = det["direction"] == "below"
            is_error = value < det["threshold"] if below else value > det["threshold"]
            # borderline: just past the threshold, where clinicians themselves disagree -> clinician view only
            margin = (det.get("borderline_margin") or 0.0) * band_scale
            hard = det["threshold"] - margin if below else det["threshold"] + margin
            borderline = is_error and not (value < hard if below else value > hard)
            _measure, label, unit, dec, _chg = FEATURE_INFO.get(feat, (None, feat, "", 1, False))
            quotable, mode, note = self._quotable(det, feat, exercise, value)
            severity = None
            if is_error and not borderline and det.get("severity_cut") is not None:
                beyond = value < det["severity_cut"] if below else value > det["severity_cut"]
                severity = "large" if beyond else "small"
            x = ser.get(feat, np.full(len(bp), np.nan))
            if np.isfinite(x).mean() < MIN_VALID_FRACTION:
                self.skipped.append(det_id)  # person not seen well enough to judge this check
                continue
            reps = segment_reps(x)
            win, win_src = None, "peak"
            # each repetition's top value, from the raw measure (frames where the person was lost are NaN)
            rep_top = [_nanmax(x[a:b + 1]) for a, _p, b in reps]
            if sum(np.isfinite(v) for v in rep_top) >= 2:
                # the repetition that went worst: lowest top for "not enough", highest for "too much"
                k = int(np.nanargmin(rep_top) if below else np.nanargmax(rep_top))
                win, win_src = (reps[k][0], reps[k][2]), "worst_rep"
            if win is None:
                win = peak_window(x)
            side = None
            if feat in ("arm_peak_high", "elbow_reach_p50", "elbow_reach_p95"):
                side = {"l": "left", "r": "right"}[ser["_reaching_side"]]
            text = DETECTOR_TEXT.get(det_id, {"error": det["meaning"], "ok": f"no sign of: {det['meaning']}"})
            typical = det.get("typical_correct")
            out.append(Finding(
                id=f"F{len(out) + 1}", detector=det_id, exercise=exercise,
                status="borderline" if borderline else ("error" if is_error else "ok"), issue=det["meaning"],
                advice=text["error" if is_error else "ok"], body_part=det["body_part"], side=side,
                feature=feat, measure_label=label, unit=unit, quotable=quotable,
                value=_round(value, dec) if quotable else None,
                typical=_round(typical, dec) if quotable and typical is not None else None,
                threshold=_round(det["threshold"], dec) if quotable else None,
                severity=severity,
                start_s=round(win[0] / FPS, 1) if win else None,
                end_s=round(win[1] / FPS, 1) if win else None,
                trust_mode=mode, trust_note=note, evidence=det.get("evidence", {}),
                window_source=win_src,
                reps=[{"start_s": round(a / FPS, 1), "end_s": round(b / FPS, 1),
                       "peak": round(float(v), 3) if np.isfinite(v) else None}
                      for (a, _pk, b), v in zip(reps, rep_top)],
            ))
        # errors first (largest first), then the checks that passed
        out.sort(key=lambda f: ({"error": 0, "borderline": 1, "ok": 2}[f.status], f.severity != "large"))
        for i, f in enumerate(out, 1):
            f.id = f"F{i}"
        return out
