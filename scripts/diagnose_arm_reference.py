"""Which trunk reference makes the phone's arm angle match Vicon?

Background: against Vicon angles from segment orientations (chest -> upper arm),
the phone's arm range matches within ~5 deg in CTK and RTK, but reads 45-56 deg
too large for overhead sideways reaching (ELK), while timing is near-perfect.
The phone measures the arm against the hip-to-shoulder MIDLINE; Vicon measures
it against the CHEST. During a flank stretch the person also bends sideways,
so the two references can diverge exactly when the arm is overhead.

This script computes the phone's signed arm angle (0 = down, 90 = sideways,
180 = overhead) against four references and compares each with Vicon:

  midline        down = mid-shoulder -> mid-hip line, in the image (current metric)
  shoulder_perp  down = perpendicular to the shoulder line, towards the hips
                 (a chest-like reference: follows the upper trunk)
  vertical       down = image vertical (ignores the trunk entirely)
  midline_3d     as midline, but in 3D using BlazePose's relative depth
                 (tests whether projecting onto the image inflates the angle)

Output, per exercise and reference: median r vs Vicon (orientation angle),
median phone range of motion, Vicon range, and the median difference.

Example
  python scripts/diagnose_arm_reference.py --root data/keraal
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rehabform.keraal import BLAZEPOSE_ORDER, iter_recordings, load_blazepose, load_vicon  # noqa: E402
from evaluate_keraal_vicon_offsetfree import compare, from_start, resample, rom, signed_plane_angle  # noqa: E402
from evaluate_vicon_quaternions import quaternion_angles  # noqa: E402

BP = {n: i for i, n in enumerate(BLAZEPOSE_ORDER)}
REFERENCES = ("midline", "shoulder_perp", "vertical", "midline_3d")


def phone_arm_angles(bp: np.ndarray) -> dict[str, np.ndarray]:
    """bp: (T, 33, 3) pixel units, un-mirrored. Returns '<ref>_<side>' -> (T,)."""
    out = {}
    for ref in REFERENCES:
        P = bp if ref == "midline_3d" else bp[..., :2]
        g = lambda name: P[:, BP[name]]  # noqa: E731
        mid_sh = (g("Left_shoulder") + g("Right_shoulder")) / 2
        mid_hip = (g("Left_hip") + g("Right_hip")) / 2
        for side, S, O in (("l", "Left", "Right"), ("r", "Right", "Left")):
            arm = g(f"{S}_elbow") - g(f"{S}_shoulder")
            outward = g(f"{S}_shoulder") - g(f"{O}_shoulder")
            if ref in ("midline", "midline_3d"):
                down = mid_hip - mid_sh
            elif ref == "vertical":
                down = np.tile([0.0, 1.0], (len(P), 1))            # image y points down
            else:  # shoulder_perp: rotate the shoulder line 90 deg, pick the side facing the hips
                line = g("Left_shoulder") - g("Right_shoulder")
                perp = np.stack([-line[:, 1], line[:, 0]], axis=1)
                flip = np.sum(perp * (mid_hip - mid_sh), axis=1) < 0
                perp[flip] *= -1
                down = perp
            out[f"{ref}_{side}"] = signed_plane_angle(arm, down, outward)
    return out


def evaluate(recording, max_lag: int) -> list[dict]:
    bp = load_blazepose(recording.blazepose)
    n = len(bp)
    quat, _ = quaternion_angles(load_vicon(recording.vicon))
    phone = phone_arm_angles(bp)
    rows = []
    for side in ("l", "r"):
        v = from_start(resample(quat[f"quat_arm_{side}"], n))
        for ref in REFERENCES:
            p = from_start(phone[f"{ref}_{side}"])
            res = compare(p, v, signed=True, max_lag=max_lag)     # Vicon axis sign is arbitrary
            rows.append({"recording_id": recording.recording_id, "exercise": recording.exercise,
                         "participant": recording.participant.split("T")[0], "side": side,
                         "reference": ref, "r": res["r"], "change_mae": res["change_mae"],
                         "rom_phone": rom(p), "rom_vicon": rom(v), "rom_diff": rom(p) - rom(v)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/keraal")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--output", default="outputs/arm_reference_per_recording.csv")
    args = ap.parse_args()

    recs = iter_recordings(args.root)
    if args.limit:
        recs = recs[: args.limit]
    rows: list[dict] = []
    for i, rec in enumerate(recs, 1):
        n_video = len(load_blazepose(rec.blazepose, image_size=None))
        rate = len(load_vicon(rec.vicon)) / (n_video / 10.0)
        if not 15.0 <= rate <= 30.0:
            print(f"[skip] {rec.recording_id}: implied Vicon rate {rate:.1f} Hz")
            continue
        rows += evaluate(rec, args.max_lag)
        if i % 50 == 0 or i == len(recs):
            print(f"processed {i}/{len(recs)}")

    warnings.filterwarnings("ignore", message="All-NaN slice")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["exercise"], r["reference"])].append(r)
    print("\nArm angle vs Vicon (orientation), both sides pooled; medians")
    print(f"{'ex':<5}{'reference':<15}{'n':>5}{'r':>7}{'MAE':>7}{'ROM phone':>11}{'ROM Vicon':>11}{'diff':>7}")
    for ex in sorted({k[0] for k in groups}):
        for ref in REFERENCES:
            items = groups.get((ex, ref), [])
            if not items:
                continue
            med = lambda c: float(np.nanmedian([float(i[c]) for i in items]))  # noqa: E731
            print(f"{ex:<5}{ref:<15}{len(items):>5}{med('r'):>7.2f}{med('change_mae'):>7.1f}"
                  f"{med('rom_phone'):>11.0f}{med('rom_vicon'):>11.0f}{med('rom_diff'):>+7.0f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-recording: {out}")


if __name__ == "__main__":
    main()
