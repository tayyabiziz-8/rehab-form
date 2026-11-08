"""Compare Keraal Kinect and Vicon angle measurements across Group 3."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehabform.keraal import (  # noqa: E402
    iter_recordings,
    load_kinect_positions,
    load_vicon_positions,
    resample_sequence,
    vicon_to_shared,
)
from rehabform.skeleton import joint_angles, normalise, to_shared  # noqa: E402

ANGLE_NAMES = ("l_elbow", "r_elbow", "l_shoulder_abd", "r_shoulder_abd", "trunk_lean")


def compare(recording) -> dict[str, float]:
    kinect = normalise(to_shared(load_kinect_positions(recording.kinect), "kinect_v2_25"))
    vicon = vicon_to_shared(load_vicon_positions(recording.vicon))
    vicon = normalise(resample_sequence(vicon, len(kinect)))
    kinect_angles = joint_angles(kinect, vertical_positive_up=True)
    vicon_angles = joint_angles(vicon, vertical_axis=2, vertical_positive_up=True)
    metrics = {}
    for name in ANGLE_NAMES:
        error = np.abs(kinect_angles[name] - vicon_angles[name])
        metrics[f"{name}_mae"] = float(np.nanmean(error))
        metrics[f"{name}_median"] = float(np.nanmedian(error))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/keraal")
    parser.add_argument("--output", default="outputs/keraal_kinect_vicon_per_recording.csv")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    recordings = iter_recordings(args.root)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        recordings = recordings[: args.limit]
    rows = []
    for number, recording in enumerate(recordings, start=1):
        rows.append({
            "recording_id": recording.recording_id,
            "exercise": recording.exercise,
            **compare(recording),
        })
        if number % 50 == 0 or number == len(recordings):
            print(f"processed {number}/{len(recordings)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
