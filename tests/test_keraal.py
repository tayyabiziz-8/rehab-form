"""Tests for the Keraal loader fixes. Run: python tests/test_keraal.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic_keraal import write  # noqa: E402
from rehabform.keraal import BLAZEPOSE_ORDER, iter_recordings, load_blazepose  # noqa: E402
from diagnose_shoulder_angles import candidate_angles  # noqa: E402
from rehabform.skeleton import to_shared, normalise  # noqa: E402

ROOT = Path(tempfile.mkdtemp())
write(ROOT, "CTK-P1T1-Unknown-C-0")
REC = iter_recordings(ROOT)[0]


def test_landmarks_ordered_by_name_not_json_order():
    bp = load_blazepose(REC.blazepose)  # JSON keys are written in reversed order
    i = BLAZEPOSE_ORDER.index("Left_shoulder")
    assert np.allclose(bp[0, i, :2], [540.0, 200.0], atol=1e-3)


def test_pixel_units_preserve_angles():
    # Frame where the synthetic arm is near 45 degrees from vertical.
    bp_px = load_blazepose(REC.blazepose)
    bp_raw = load_blazepose(REC.blazepose, image_size=None)
    i_sh, i_el = BLAZEPOSE_ORDER.index("Left_shoulder"), BLAZEPOSE_ORDER.index("Left_elbow")

    def arm_deg(bp, t):
        v = bp[t, i_el, :2] - bp[t, i_sh, :2]
        return np.degrees(np.arctan2(v[0], v[1]))

    angles = [arm_deg(bp_px, t) for t in range(len(bp_px))]
    t45 = int(np.argmin(np.abs(np.array(angles) - 45)))
    assert abs(arm_deg(bp_px, t45) - 45) < 2          # pixels: correct
    assert arm_deg(bp_raw, t45) < 32                   # raw normalised: squashed (~30 deg)


def test_shoulder_vertical_reference_matches_between_systems():
    from rehabform.keraal import load_kinect_positions, resample_sequence
    phone = normalise(to_shared(load_blazepose(REC.blazepose), "mediapipe33"))
    kinect = normalise(to_shared(load_kinect_positions(REC.kinect), "kinect_v2_25"))
    phone = resample_sequence(phone, len(kinect))
    a_phone = candidate_angles(phone, vertical_positive_up=False)["l_vertical_3d"]
    a_kinect = candidate_angles(kinect, vertical_positive_up=True)["l_vertical_3d"]
    assert np.nanmean(np.abs(a_phone - a_kinect)) < 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
