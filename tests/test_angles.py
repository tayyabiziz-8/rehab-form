"""Tests for the angle spec and trust table. Run: python tests/test_angles.py"""
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from rehabform import angles as A  # noqa: E402
from rehabform.angles import TrustTable  # noqa: E402


def pose(arm_l_deg=0.0, elbow_l_deg=0.0, tilt_deg=0.0, T=3):
    """Person facing camera (left side on image right), pixel units, y down."""
    lm = np.full((T, 33, 3), np.nan)
    base = {A.L_SHOULDER: (60, 0), A.R_SHOULDER: (-60, 0), A.L_HIP: (40, 220), A.R_HIP: (-40, 220),
            A.R_ELBOW: (-60, 110), A.R_WRIST: (-60, 220)}
    a, e = np.radians(arm_l_deg), np.radians(arm_l_deg + elbow_l_deg)
    base[A.L_ELBOW] = (60 + 110 * np.sin(a), 110 * np.cos(a))
    base[A.L_WRIST] = (base[A.L_ELBOW][0] + 110 * np.sin(e), base[A.L_ELBOW][1] + 110 * np.cos(e))
    t = np.radians(tilt_deg)          # lean towards the person's left = image right, about the mid-hip
    for k, (x, y) in base.items():
        dy = y - 220
        xr, yr = x * np.cos(t) - dy * np.sin(t), 220 + x * np.sin(t) + dy * np.cos(t)
        lm[:, k] = (480 + xr, 100 + yr, 0.0)
    return lm


def test_known_poses():
    for deg in (0, 45, 90, 135):
        v, ok = A.arm_elevation(pose(arm_l_deg=deg), "l", arm_length_px=110)
        assert ok.all() and abs(v[0] - deg) < 0.5, (deg, v[0])
    v, _ = A.elbow_flexion(pose(arm_l_deg=90, elbow_l_deg=90), "l", 110, 110)
    assert abs(v[0] - 90) < 0.5
    v, _ = A.trunk_lean(pose(tilt_deg=30))
    assert abs(v[0] - 30) < 0.5                       # + = towards the person's left
    v, _ = A.arm_elevation(pose(arm_l_deg=60, tilt_deg=30), "l", arm_length_px=110)
    assert abs(v[0] - 60) < 0.5                       # trunk lean does not leak into the arm angle


def test_arm_pointing_at_camera_is_invalid():
    lm = pose(arm_l_deg=90)
    lm[:, A.L_ELBOW, :2] = lm[:, A.L_SHOULDER, :2] + [15, 5]   # foreshortened: arm points at the camera
    _, ok = A.arm_elevation(lm, "l", arm_length_px=110)
    assert not ok.any()


def test_matches_the_validated_metric():
    from make_synthetic_keraal import write
    from rehabform.keraal import iter_recordings, load_blazepose
    diag = importlib.import_module("diagnose_arm_reference")
    root = Path(tempfile.mkdtemp())
    write(root, "ELK-P1T1-Unknown-C-0", max_deg=150)
    bp = load_blazepose(iter_recordings(root)[0].blazepose)
    mine, _ = A.arm_elevation(bp, "l")
    validated = diag.phone_arm_angles(bp)["shoulder_perp_l"]
    ok = np.isfinite(mine) & np.isfinite(validated)
    assert ok.sum() > 50 and np.nanmax(np.abs(mine[ok] - validated[ok])) < 1e-6


def test_trust_table_decisions():
    tt = TrustTable.load()
    for name, spec in tt.measures.items():
        assert hasattr(A, spec["function"].split(".")[-1]), name
        for ex, rule in spec["exercises"].items():
            assert rule["mode"] in A.MODES, (name, ex)
    assert tt.decide("arm_elevation_l", "CTK", 85).mode == "range"
    assert tt.decide("arm_elevation_l", "CTK", 140).mode == "relative"      # overhead rule
    assert tt.decide("arm_elevation_r", "ELK").mode == "relative"
    assert tt.decide("elbow_flexion_l", "ELK").mode == "range"
    assert tt.decide("elbow_flexion_l", "CTK").mode == "not_reportable"
    assert tt.decide("trunk_rotation", "RTK").mode == "timing_only"
    assert tt.decide("trunk_rotation", "ELK").mode == "not_reportable"
    assert tt.decide("trunk_lean", "SQUAT").mode == "not_reportable"        # unvalidated exercise


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
