"""Compare standard and diagnostic-swapped Vicon mappings against Kinect."""
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


def mapping_errors(recording, swap_sides: bool) -> dict[str, float]:
    kinect = normalise(to_shared(load_kinect_positions(recording.kinect), "kinect_v2_25"))
    vicon = vicon_to_shared(
        load_vicon_positions(recording.vicon), swap_sides=swap_sides
    )
    vicon = normalise(resample_sequence(vicon, len(kinect)))
    observed = joint_angles(kinect, vertical_positive_up=True)
    reference = joint_angles(vicon, vertical_axis=2, vertical_positive_up=True)
    return {
        name: float(np.nanmean(np.abs(observed[name] - reference[name])))
        for name in ANGLE_NAMES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/keraal")
    parser.add_argument("--output", default="outputs/keraal_mapping_diagnostic.csv")
    args = parser.parse_args()

    rows = []
    for number, recording in enumerate(iter_recordings(args.root), start=1):
        standard = mapping_errors(recording, swap_sides=False)
        swapped = mapping_errors(recording, swap_sides=True)
        row = {
            "recording_id": recording.recording_id,
            "exercise": recording.exercise,
        }
        for name in ANGLE_NAMES:
            row[f"{name}_standard_mae"] = standard[name]
            row[f"{name}_swapped_mae"] = swapped[name]
            row[f"{name}_swapped_minus_standard"] = swapped[name] - standard[name]
        rows.append(row)
        if number % 50 == 0 or number == 540:
            print(f"processed {number}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
