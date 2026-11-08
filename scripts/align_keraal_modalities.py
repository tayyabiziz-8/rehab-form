"""Estimate Kinect-Vicon temporal lag and compare aligned angle errors."""
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
    shift_sequence,
    vicon_to_shared,
)
from rehabform.skeleton import joint_angles, normalise, to_shared  # noqa: E402

ANGLE_NAMES = ("l_elbow", "r_elbow", "l_shoulder_abd", "r_shoulder_abd", "trunk_lean")


def angle_sequences(recording) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    kinect = normalise(to_shared(load_kinect_positions(recording.kinect), "kinect_v2_25"))
    vicon = vicon_to_shared(load_vicon_positions(recording.vicon))
    vicon = normalise(resample_sequence(vicon, len(kinect)))
    return (
        joint_angles(kinect, vertical_positive_up=True),
        joint_angles(vicon, vertical_axis=2, vertical_positive_up=True),
    )


def mean_error(observed: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    values = [
        np.nanmean(np.abs(observed[name] - reference[name]))
        for name in ANGLE_NAMES
    ]
    return float(np.mean(values))


def best_shift(
    observed: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    max_shift: int,
) -> tuple[int, float]:
    candidates = []
    for shift in range(-max_shift, max_shift + 1):
        shifted = {
            name: shift_sequence(reference[name], shift)
            for name in ANGLE_NAMES
        }
        candidates.append((mean_error(observed, shifted), shift))
    error, shift = min(candidates)
    return shift, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/keraal")
    parser.add_argument("--output", default="outputs/keraal_kinect_vicon_alignment.csv")
    parser.add_argument("--max-shift", type=int, default=30)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_shift < 0:
        raise SystemExit("--max-shift must be non-negative")

    recordings = iter_recordings(args.root)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        recordings = recordings[: args.limit]

    rows = []
    for number, recording in enumerate(recordings, start=1):
        observed, reference = angle_sequences(recording)
        before = mean_error(observed, reference)
        shift, after = best_shift(observed, reference, args.max_shift)
        rows.append({
            "recording_id": recording.recording_id,
            "exercise": recording.exercise,
            "best_shift_frames": shift,
            "error_before_deg": before,
            "error_after_deg": after,
            "improvement_deg": before - after,
        })
        if number % 50 == 0 or number == len(recordings):
            print(f"processed {number}/{len(recordings)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shifts = np.asarray([row["best_shift_frames"] for row in rows])
    improvements = np.asarray([row["improvement_deg"] for row in rows])
    print(f"output: {output}")
    print(f"median best shift: {np.median(shifts):.1f} frames")
    print(f"median improvement: {np.median(improvements):.2f} degrees")
    print(f"improved recordings: {np.count_nonzero(improvements > 0)}/{len(rows)}")


if __name__ == "__main__":
    main()
