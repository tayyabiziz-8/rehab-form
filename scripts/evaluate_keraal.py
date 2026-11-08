"""Evaluate BlazePose angles against Vicon across Keraal Group 3."""
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
    load_vicon_positions,
    resample_sequence,
    vicon_to_shared,
)
from rehabform.skeleton import joint_angles, normalise, to_shared  # noqa: E402

ANGLE_NAMES = (
    "l_elbow",
    "r_elbow",
    "l_shoulder_abd",
    "r_shoulder_abd",
    "trunk_lean",
)


def angle_errors(recording) -> dict[str, float]:
    """Return per-angle MAE and median error for one recording."""
    phone = normalise(to_shared(load_blazepose(recording.blazepose), "mediapipe33"))
    vicon = vicon_to_shared(load_vicon_positions(recording.vicon))
    vicon = normalise(resample_sequence(vicon, len(phone)))
    phone_angles = joint_angles(phone)
    vicon_angles = joint_angles(vicon, vertical_axis=2, vertical_positive_up=True)

    metrics = {}
    for name in ANGLE_NAMES:
        error = np.abs(phone_angles[name] - vicon_angles[name])
        metrics[f"{name}_mae"] = float(np.nanmean(error))
        metrics[f"{name}_median"] = float(np.nanmedian(error))
    return metrics


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["exercise"])].append(row)

    summary = []
    for exercise, exercise_rows in sorted(grouped.items()):
        row: dict[str, object] = {
            "exercise": exercise,
            "recordings": len(exercise_rows),
        }
        for name in ANGLE_NAMES:
            for statistic in ("mae", "median"):
                values = np.asarray(
                    [float(item[f"{name}_{statistic}"]) for item in exercise_rows]
                )
                row[f"{name}_{statistic}_mean"] = float(np.mean(values))
                row[f"{name}_{statistic}_std"] = float(np.std(values, ddof=1))
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/keraal")
    parser.add_argument("--output", default="outputs/keraal_angle_per_recording.csv")
    parser.add_argument("--summary-output", default="outputs/keraal_angle_summary.csv")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N recordings")
    args = parser.parse_args()

    recordings = iter_recordings(args.root)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        recordings = recordings[: args.limit]
    if not recordings:
        raise SystemExit("No complete Keraal recordings found")

    rows = []
    for number, recording in enumerate(recordings, start=1):
        metrics = angle_errors(recording)
        rows.append(
            {
                "recording_id": recording.recording_id,
                "exercise": recording.exercise,
                "participant": recording.participant,
                "label": recording.label,
                "error": recording.error,
                **metrics,
            }
        )
        if number % 50 == 0 or number == len(recordings):
            print(f"processed {number}/{len(recordings)}")

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    write_rows(output, rows)
    write_rows(summary_output, summary_rows(rows))
    print(f"per-recording: {output}")
    print(f"summary: {summary_output}")


if __name__ == "__main__":
    main()
