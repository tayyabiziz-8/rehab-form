"""Compare BlazePose and Vicon angles for one Keraal Group 3 recording."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehabform.keraal import (  # noqa: E402
    iter_recordings,
    load_blazepose,
    load_vicon_positions,
    resample_sequence,
    vicon_to_shared,
)
from rehabform.skeleton import joint_angles, normalise, to_shared  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/keraal")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    recordings = iter_recordings(args.root)
    if not recordings:
        raise SystemExit("No complete Keraal recordings found")
    recording = recordings[args.index]

    phone = normalise(to_shared(load_blazepose(recording.blazepose), "mediapipe33"))
    vicon = vicon_to_shared(load_vicon_positions(recording.vicon))
    vicon = normalise(resample_sequence(vicon, len(phone)))
    phone_angles = joint_angles(phone)
    vicon_angles = joint_angles(vicon, vertical_axis=2, vertical_positive_up=True)

    print(f"recording: {recording.recording_id}")
    print(f"exercise: {recording.exercise}, label: {recording.label}, error: {recording.error}")
    print(f"frames: BlazePose={len(phone)}, Vicon={len(vicon)} after resampling")
    for name in ("l_elbow", "r_elbow", "l_shoulder_abd", "r_shoulder_abd", "trunk_lean"):
        error = np.abs(phone_angles[name] - vicon_angles[name])
        print(f"{name}: MAE={np.nanmean(error):.2f} deg, median={np.nanmedian(error):.2f} deg")


if __name__ == "__main__":
    main()
