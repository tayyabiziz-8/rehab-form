"""Findings engine, feedback template and claim checker (synthetic, no dataset needed).

    python tests/test_findings_claims.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.claims import check, error_categories  # noqa: E402
from rehabform.feedback import strip_tags, template_feedback  # noqa: E402
from rehabform.findings import Finding, FindingsEngine  # noqa: E402

W, H = 480, 360


def elk_sequence(lean_deg: float, T: int = 100, reps: list[float] | None = None) -> np.ndarray:
    """Person facing the camera, left arm goes overhead while the trunk leans by lean_deg.
    With ``reps`` (one scale per repetition) the bend is repeated, each rep scaled."""
    bp = np.full((T, 33, 3), np.nan, np.float32)
    if reps:
        rest = 10                                              # 1 s still at the start
        per = (T - rest) // len(reps)
        s = np.concatenate([np.zeros(rest)] + [sc * np.sin(np.linspace(0, np.pi, per)) for sc in reps])
        s = np.concatenate([s, np.zeros(T - len(s))])
    else:
        s = np.sin(np.linspace(0, np.pi, T))
    for t in range(T):
        lean = np.radians(lean_deg * s[t])
        hip = np.array([240.0, 280.0])
        up = np.array([np.sin(lean), -np.cos(lean)]) * 150
        mid = hip + up
        bp[t, 23, :2], bp[t, 24, :2] = hip + [20, 0], hip - [20, 0]
        bp[t, 11, :2], bp[t, 12, :2] = mid + [30, 0], mid - [30, 0]
        a = np.radians(170 * s[t])
        ls = bp[t, 11, :2]
        bp[t, 13, :2] = ls + 50 * np.array([np.sin(a), np.cos(a)])
        bp[t, 15, :2] = ls + 100 * np.array([np.sin(a), np.cos(a)])
        rs = bp[t, 12, :2]
        bp[t, 14, :2], bp[t, 16, :2] = rs + [0, 50], rs + [0, 100]
        bp[t, :, 2] = 0
    return bp


def engine(tmp: Path, margin: float = 0.0) -> FindingsEngine:
    cfg = {"detectors": {"ELK_body_not_tilted": {"borderline_margin": margin,
        "exercise": "ELK", "keraal_error": "Error2", "meaning": "body not tilted", "body_part": "Spine",
        "feature": "lean_range", "direction": "below", "threshold": 15.0, "unit": "deg trunk lean range",
        "reportable": True, "typical_correct": 25.0, "severity_cut": 8.0, "evidence": {}}}}
    p = tmp / "thr.json"
    p.write_text(json.dumps(cfg))
    return FindingsEngine(p)


def test_engine_flags_small_lean_and_passes_large_lean():
    with tempfile.TemporaryDirectory() as d:
        eng = engine(Path(d))
        bad = eng.run(elk_sequence(5), "ELK")
        good = eng.run(elk_sequence(30), "ELK")
    assert bad[0].status == "error" and bad[0].quotable and bad[0].value < 15
    assert bad[0].severity == "large"                       # 5 deg is beyond the severity cut (8)
    assert bad[0].start_s is not None and 2 < bad[0].start_s < 8   # window around the peak (middle of 10 s)
    assert good[0].status == "ok"


def test_window_is_the_worst_repetition_and_borderline_is_not_corrected():
    with tempfile.TemporaryDirectory() as d:
        eng = engine(Path(d))
        f = eng.run(elk_sequence(12, T=300, reps=[1.0, 1.0, 0.4, 1.0, 1.0]), "ELK")[0]
        assert f.status == "error" and f.window_source == "worst_rep" and len(f.reps) == 5
        assert 12 <= f.start_s and f.end_s <= 19.5                 # the third repetition (~12.6-18.4 s)
        eng2 = engine(Path(d), margin=5.0)                          # 12 deg is within 5 of the 15 threshold
        b = eng2.run(elk_sequence(12, T=300, reps=[1.0, 1.0, 0.4, 1.0, 1.0]), "ELK")
    assert b[0].status == "borderline"
    text = template_feedback(b)
    assert "[F" not in text and "bend" not in text.lower()          # nothing corrective for the patient


def test_template_feedback_always_passes_the_checker():
    with tempfile.TemporaryDirectory() as d:
        eng = engine(Path(d))
        for lean in (3, 10, 30):
            f = eng.run(elk_sequence(lean), "ELK")
            text = template_feedback(f)
            rep = check(text, f)
            assert rep.ok, (text, rep.reasons())
            assert "[" not in strip_tags(text)


def _finding(**kw) -> Finding:
    base = dict(id="F1", detector="ELK_body_not_tilted", exercise="ELK", status="error", issue="body not tilted",
                advice="bend your upper body further to the side", body_part="Spine", side=None,
                feature="lean_range", measure_label="sideways bend of your upper body", unit="°", quotable=True,
                value=9.0, typical=24.0, threshold=15.0, severity="small", start_s=4.2, end_s=6.0,
                trust_mode="range")
    base.update(kw)
    return Finding(**base)


def test_checker_rejects_invented_claims():
    f = [_finding(), _finding(id="F2", detector="X", status="ok", body_part="BothArms", feature="arm_peak_high",
                              advice="your arm went high", quotable=False, value=None, typical=None,
                              side="left", start_s=5.0, end_s=6.0)]
    ok = "Try to bend your upper body further to the side; you reached 9° and most people reach 24°. [F1]"
    assert check(ok, f).ok
    assert check("Nice work today!", f).ok                                   # generic, allowed
    bad = {
        "You reached 12° of side bend. [F1]": "unsupported_number",
        "Your knees were bent. [F1]": "unsupported_body_part",
        "Lean further to the right. [F1]": "unsupported_side",
        "Your trunk only bent 9°.": "untagged_claim",
        "Raise your arm higher. [F2]": "contradicts_finding",
        "This may cause back pain. [F1]": "out_of_scope",
        "Bend further. [F9]": "unknown_finding",
        "Your left arm reached 170°. [F2]": "unsupported_number",           # F2 is not quotable
    }
    for text, reason in bad.items():
        rep = check(text, f)
        assert not rep.ok and any(r.startswith(reason) for r in rep.reasons()), (text, rep.reasons())
    assert check("Your left arm went up well. [F2]", f).ok                  # side backed by F2


def test_checker_catches_what_gpt_oss_20b_wrote_on_20_nov():
    """Real drafts from the first Groq run that the first checker let through."""
    arms = dict(detector="CTK_arms_not_raised", exercise="CTK", issue="arms not raised enough",
                body_part="BothArms", feature="wrist_height_p95",
                measure_label="height of your hands above your shoulders", unit="shoulder widths",
                threshold=0.8, severity=None, start_s=14.4, end_s=16.4, trust_mode=None)
    ok = _finding(id="F1", status="ok", advice="your arms reached a good height", quotable=True,
                  value=0.9, typical=1.0, **arms)
    err = _finding(id="F1", status="error", advice="raise your arms higher", quotable=True,
                   value=0.8, typical=1.0, **{**arms, "severity": "small"})
    assert not check("Your arms were not raised enough. [F1] Your arms reached a good height. [F1]", [ok]).ok
    assert not check("You reached a good height with your arms. [F1]", [err]).ok        # praise for an error
    assert not check("Try raising your arms higher to reach about 1.0. [F1]", [err]).ok  # no unit
    assert check("Try raising your arms higher, aiming for about 1 shoulder width instead of the "
                 "0.8 shoulder widths you reached. [F1]", [err]).ok
    assert check("Raise your arms higher. [F1]", [err]).ok
    assert check("Your arms reached a good height. [F1]", [ok]).ok


def test_error_categories_for_the_baseline():
    assert error_categories("Try to raise your arms higher.") == {"arms_height"}
    assert error_categories("Lean your body further to the side.") == {"trunk_tilt"}
    assert error_categories("Great job keeping your arms up!") == set()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
