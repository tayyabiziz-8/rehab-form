"""Compare candidate shoulder-angle definitions against Kinect."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehabform.keraal import (  # noqa: E402
    iter_recordings,
    load_blazepose,
    load_kinect_positions,
    resample_sequence,
)
from rehabform.skeleton import J, normalise, to_shared  # noqa: E402


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    cosine = np.sum(ba * bc, axis=-1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def candidate_angles(
    sequence: np.ndarray, vertical_positive_up: bool = False
) -> dict[str, np.ndarray]:
    """Candidate shoulder angles.

    ``vertical_positive_up`` must match the coordinate system: False for
    BlazePose image coordinates (y points down), True for Kinect (y points up).
    Using the image convention for Kinect turns its "up" reference upside down.
    """
    mid_hip = (sequence[:, J["l_hip"]] + sequence[:, J["r_hip"]]) / 2.0
    mid_shoulder = (
        sequence[:, J["l_shoulder"]] + sequence[:, J["r_shoulder"]]
    ) / 2.0
    up = np.zeros(sequence.shape[-1], dtype=np.float32)
    up[1] = 1.0 if vertical_positive_up else -1.0
    result = {}
    for side in ("l", "r"):
        shoulder = sequence[:, J[f"{side}_shoulder"]]
        elbow = sequence[:, J[f"{side}_elbow"]]
        result[f"{side}_torso_3d"] = angle(
            sequence[:, J[f"{side}_hip"]], shoulder, elbow
        )
        result[f"{side}_vertical_3d"] = angle(
            elbow, shoulder, shoulder + up
        )
        result[f"{side}_torso_2d"] = angle(
            mid_hip[:, :2], shoulder[:, :2], elbow[:, :2]
        )
    return result


def evaluate(recording) -> dict[str, float]:
    phone = normalise(
        to_shared(load_blazepose(recording.blazepose), "mediapipe33")
    )
    kinect = normalise(
        to_shared(load_kinect_positions(recording.kinect), "kinect_v2_25")
    )
    phone = resample_sequence(phone, len(kinect))
    phone_angles = candidate_angles(phone, vertical_positive_up=False)
    kinect_angles = candidate_angles(kinect, vertical_positive_up=True)
    result = {}
    for side in ("l", "r"):
        for definition in ("torso_3d", "vertical_3d", "torso_2d"):
            name = f"{side}_{definition}"
            result[f"{name}_mae"] = float(
                np.nanmean(np.abs(phone_angles[name] - kinect_angles[name]))
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/keraal")
    parser.add_argument("--output", default="outputs/keraal_shoulder_angle_diagnostic.csv")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    recordings = iter_recordings(args.root)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        recordings = recordings[: args.limit]

    rows = []
    for number, recording in enumerate(recordings, start=1):
        rows.append(
            {
                "recording_id": recording.recording_id,
                "exercise": recording.exercise,
                **evaluate(recording),
            }
        )
        if number % 50 == 0 or number == len(recordings):
            print(f"processed {number}/{len(recordings)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["exercise"]].append(row)
    for exercise, exercise_rows in sorted(grouped.items()):
        print(exercise)
        for side in ("l", "r"):
            for definition in ("torso_3d", "vertical_3d", "torso_2d"):
                name = f"{side}_{definition}_mae"
                value = np.mean([float(row[name]) for row in exercise_rows])
                print(f"  {name}: {value:.2f} deg")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
