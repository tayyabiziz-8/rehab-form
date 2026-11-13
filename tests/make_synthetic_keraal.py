"""Write a tiny synthetic Keraal Group 3 folder for testing the evaluation scripts.

The 'person' faces the camera and raises the left arm sideways from 0 to 80 degrees
and back, with a straight elbow. BlazePose is written in Keraal's normalised format
(x / 960, y / 544) with the JSON keys in REVERSED order, so tests catch both the
aspect-ratio bug and any reliance on key order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rehabform.keraal import BLAZEPOSE_ORDER, VICON_TARGETS  # noqa: E402

W, H = 960, 544


def arm_angle(t: np.ndarray) -> np.ndarray:
    return np.radians(80) * np.sin(np.pi * t / t[-1]) ** 2


def write(root: Path, rec: str, n_video: int = 100, vicon_hz: int = 100) -> None:
    for d in ("blazepose", "vicon", "kinect", "openpose", "videos"):
        (root / d).mkdir(parents=True, exist_ok=True)
    tv = np.linspace(0, n_video / 10, n_video)
    # Person in pixels: shoulders at y=200, hips at y=420; left side appears on image right.
    pix = {}
    for i, a in enumerate(arm_angle(tv)):
        p = {n: [480.0, 300.0, 0.0] for n in BLAZEPOSE_ORDER}
        p["Left_shoulder"] = [540.0, 200.0, 0.0]; p["Right_shoulder"] = [420.0, 200.0, 0.0]
        p["Left_hip"] = [520.0, 420.0, 0.0];      p["Right_hip"] = [440.0, 420.0, 0.0]
        d = np.array([np.sin(a), np.cos(a)])
        p["Left_elbow"] = [540 + 110 * d[0], 200 + 110 * d[1], 0.0]
        p["Left_wrist"] = [540 + 220 * d[0], 200 + 220 * d[1], 0.0]
        p["Right_elbow"] = [420.0, 310.0, 0.0];   p["Right_wrist"] = [420.0, 420.0, 0.0]
        pix[i] = p
    positions = {}
    for i, p in pix.items():
        # Normalised like Keraal, keys in reversed order on purpose.
        positions[f"{i + 1}.0"] = {n: [p[n][0] / W, p[n][1] / H, p[n][2] / W] for n in reversed(BLAZEPOSE_ORDER)}
    (root / "blazepose" / f"G3-BP-{rec}.json").write_text(json.dumps({"positions": positions}))

    # Kinect (25 joints x 7), metres, y up; same pose as the pixels.
    K = {"ShoulderLeft": 4, "ElbowLeft": 5, "WristLeft": 6, "ShoulderRight": 8, "ElbowRight": 9,
         "WristRight": 10, "HipLeft": 12, "HipRight": 16, "Head": 3}
    src = {"ShoulderLeft": "Left_shoulder", "ElbowLeft": "Left_elbow", "WristLeft": "Left_wrist",
           "ShoulderRight": "Right_shoulder", "ElbowRight": "Right_elbow", "WristRight": "Right_wrist",
           "HipLeft": "Left_hip", "HipRight": "Right_hip", "Head": "Nose"}
    lines = []
    for i in range(n_video):
        row = np.zeros((25, 7)); row[:, 6] = 1.0; row[:, 2] = 2.5
        for kn, idx in K.items():
            x, y, _ = pix[i][src[kn]]
            row[idx, :2] = [x / 500.0, -y / 500.0]
        lines.append(" ".join(f"{v:.6f}" for v in row.ravel()))
    (root / "kinect" / f"G3-Kinect-{rec}.txt").write_text("\n".join(lines) + "\n")

    # Vicon (17 targets x 7), mm, z up; person's left = +y. Targets on segments.
    n_vicon = int(n_video / 10 * vicon_hz)
    tc = np.linspace(0, n_video / 10, n_vicon)
    lines = []
    for a in arm_angle(tc):
        P = np.zeros((17, 7)); P[:, 6] = 1.0
        def s(name, xyz): P[VICON_TARGETS.index(name), :3] = xyz
        ls, rs = np.array([0, 200.0, 1400]), np.array([0, -200.0, 1400])
        s("l_shoulder", ls); s("r_shoulder", rs); s("chest", [0, 0, 1300]); s("hips", [0, 0, 1000])
        s("head", [0, 0, 1600])
        d = np.array([0, np.sin(a), -np.cos(a)])
        s("l_arm", ls + 150 * d); s("l_forearm", ls + 400 * d); s("l_hand", ls + 550 * d)
        s("r_arm", rs + [0, 0, -150]); s("r_forearm", rs + [0, 0, -400]); s("r_hand", rs + [0, 0, -550])
        for name in ("r_thigh", "l_thigh", "r_tibia", "l_tibia", "r_foot", "l_foot"):
            s(name, [0, 100 if name.startswith("l") else -100, 600])
        lines.append(" ".join(f"{v:.6f}" for v in P.ravel()))
    (root / "vicon" / f"G3-Vicon-{rec}.txt").write_text("\n".join(lines) + "\n")
    (root / "openpose" / f"G3-OP-{rec}.json").write_text("{}")
    (root / "videos" / f"G3-Anon-{rec}.avi").write_bytes(b"")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/keraal_synth")
    write(out, "CTK-P1T1-Unknown-C-0")
    write(out, "ELK-P2T1-Unknown-E1-1")
    print(f"wrote synthetic Keraal folder to {out}")
