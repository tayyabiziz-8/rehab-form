"""Vicon joint angles from segment ORIENTATIONS, compared with the phone and
with the earlier target-POSITION Vicon angles.

Why: after un-mirroring, sideways arm movement tracks Vicon almost perfectly in
time (r ~0.99), yet the phone's range of motion is 30-40 deg larger. Either the
phone overstates, or Vicon's position-based angle understates (for example
because the "shoulder" target moves with the arm). Each Vicon ART target also
records its 3D orientation, which is what joint angles are normally built from.

How (no knowledge of how each target was mounted is needed):
  1. relative orientation of child vs parent segment per frame
       q_rel(t) = inverse(q_parent(t)) * q_child(t)
  2. change from the start pose, expressed in the start frame
       q_delta(t) = inverse(q_rel(0)) * q_rel(t)
  3. rotation vector of q_delta (axis * angle). Its length is the total joint
     rotation since the start ("magnitude").
  4. the joint's main rotation axis = principal direction of those rotation
     vectors over the recording; the projection onto it is a SIGNED single-axis
     angle ("signed"). The axis direction is arbitrary, so the sign may be
     flipped to match the phone (reported). "axis_share" says how much of the
     rotation is about that one axis (1.0 = a pure hinge movement).

Joints
  quat_arm_l/r    chest -> upper arm   (bypasses the shoulder target)
  quat_elbow_l/r  upper arm -> forearm
  quat_trunk      hips -> chest

Comparisons written per recording (all as change from start; ROM = p95 - p5)
  phone vs quaternion   e.g. phone arm_abd_signed_l  vs quat_arm_l
  target vs quaternion  e.g. Vicon target arm_abd_signed_l vs quat_arm_l
The second one answers directly whether the target-based angle understates.

Examples
  python scripts/evaluate_vicon_quaternions.py --limit 5
  python scripts/evaluate_vicon_quaternions.py
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

from rehabform.keraal import VICON_TARGETS, iter_recordings, load_blazepose, load_vicon  # noqa: E402
from evaluate_keraal_vicon_offsetfree import compare, from_start, phone_angles, resample, vicon_angles  # noqa: E402

VI = {name: i for i, name in enumerate(VICON_TARGETS)}

# (comparison name, phone angle, phone variant, quaternion joint, target-based Vicon angle)
PAIRS = [
    ("arm_l", "arm_abd_signed_l", "2d", "quat_arm_l", "arm_abd_signed_l"),
    ("arm_r", "arm_abd_signed_r", "2d", "quat_arm_r", "arm_abd_signed_r"),
    ("elbow_l", "elbow_l", "2d", "quat_elbow_l", "elbow_l"),
    ("elbow_r", "elbow_r", "2d", "quat_elbow_r", "elbow_r"),
    ("trunk_lat", "trunk_lat", "2d", "quat_trunk", "trunk_lat"),
    ("trunk_rot", "trunk_rot", "3d", "quat_trunk", "trunk_rot"),
]


# ---------------------------------------------------------------------------
# Quaternions, (..., 4) as x, y, z, w
# ---------------------------------------------------------------------------
def q_normalise(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    out = q / np.where(n > 1e-9, n, np.nan)
    out[(n[..., 0] < 0.5)] = np.nan          # lost tracking / empty rows
    return out


def q_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def rotvec_deg(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, degrees) of unit quaternions (..., 4)."""
    q = np.where(q[..., 3:4] < 0, -q, q)            # shortest rotation
    v, w = q[..., :3], q[..., 3]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, w)
    axis = v / np.where(s > 1e-12, s, 1.0)[..., None]
    return np.degrees(axis * angle[..., None])


def joint_rotation(q_parent: np.ndarray, q_child: np.ndarray) -> dict[str, np.ndarray | float]:
    """Change-from-start joint rotation between two segments, axis-free."""
    rel = q_mul(q_conj(q_normalise(q_parent)), q_normalise(q_child))
    good = np.isfinite(rel).all(axis=-1)
    if good.sum() < 5:
        nan = np.full(len(rel), np.nan)
        return {"magnitude": nan, "signed": nan, "axis_share": np.nan}
    start = rel[np.argmax(good)]
    delta = q_mul(q_conj(np.broadcast_to(start, rel.shape)), rel)
    rv = rotvec_deg(delta)                          # (T, 3) in the start frame
    mag = np.linalg.norm(rv, axis=-1)
    ok = np.isfinite(rv).all(axis=-1)
    _, sv, vt = np.linalg.svd(rv[ok], full_matrices=False)
    axis = vt[0]
    signed = rv @ axis
    share = float(sv[0] ** 2 / np.sum(sv ** 2)) if np.sum(sv ** 2) > 0 else np.nan
    return {"magnitude": mag, "signed": signed, "axis_share": share}


def quaternion_angles(vicon: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    q = vicon[..., 3:7]
    g = lambda name: q[:, VI[name]]  # noqa: E731
    out, shares = {}, {}
    for side in ("l", "r"):
        for joint, parent, child in (("arm", "chest", f"{side}_arm"), ("elbow", f"{side}_arm", f"{side}_forearm")):
            res = joint_rotation(g(parent), g(child))
            out[f"quat_{joint}_{side}"] = res["signed"]
            out[f"quat_{joint}_{side}_mag"] = res["magnitude"]
            shares[f"quat_{joint}_{side}"] = res["axis_share"]
    res = joint_rotation(g("hips"), g("chest"))
    out["quat_trunk"], out["quat_trunk_mag"] = res["signed"], res["magnitude"]
    shares["quat_trunk"] = res["axis_share"]
    return out, shares


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def rom(x: np.ndarray) -> float:
    return float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5)) if np.isfinite(x).any() else float("nan")


def evaluate(recording, max_lag: int) -> list[dict]:
    bp = load_blazepose(recording.blazepose)
    vicon = load_vicon(recording.vicon)
    n = len(bp)
    phone = phone_angles(bp)
    target = vicon_angles(vicon[..., :3])
    quat, shares = quaternion_angles(vicon)
    rows = []
    for name, p_angle, variant, q_joint, t_angle in PAIRS:
        q = from_start(resample(quat[q_joint], n))
        q_mag = resample(quat[f"{q_joint}_mag"], n)
        p = from_start(phone[variant][p_angle])
        t = from_start(resample(target[t_angle], n))
        vs_phone = compare(p, q, signed=True, max_lag=max_lag)      # quaternion axis sign is arbitrary
        vs_target = compare(t, q, signed=True, max_lag=0)           # same system, no lag
        rows.append({
            "recording_id": recording.recording_id, "exercise": recording.exercise,
            "participant": recording.participant.split("T")[0], "error": recording.error,
            "joint": name, "axis_share": shares[q_joint],
            "rom_quat": rom(q), "rom_quat_magnitude": rom(q_mag),
            "rom_target": rom(t), "rom_phone": rom(p),
            "r_phone_vs_quat": vs_phone["r"], "mae_phone_vs_quat": vs_phone["change_mae"],
            "r_target_vs_quat": vs_target["r"], "mae_target_vs_quat": vs_target["change_mae"],
        })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    warnings.filterwarnings("ignore", message="All-NaN slice")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["exercise"], r["joint"])].append(r)
    out = []
    for (ex, joint), items in sorted(groups.items()):
        med = lambda c: float(np.nanmedian([float(i[c]) for i in items]))  # noqa: E731
        out.append({"exercise": ex, "joint": joint, "n": len(items),
                    "axis_share": med("axis_share"),
                    "rom_phone": med("rom_phone"), "rom_target": med("rom_target"),
                    "rom_quat": med("rom_quat"), "rom_quat_mag": med("rom_quat_magnitude"),
                    "r_phone_quat": med("r_phone_vs_quat"), "mae_phone_quat": med("mae_phone_vs_quat"),
                    "r_target_quat": med("r_target_vs_quat")})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/keraal")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--min-rate", type=float, default=15.0)
    ap.add_argument("--max-rate", type=float, default=30.0)
    ap.add_argument("--output", default="outputs/vicon_quaternion_per_recording.csv")
    ap.add_argument("--summary-output", default="outputs/vicon_quaternion_summary.csv")
    args = ap.parse_args()

    recs = iter_recordings(args.root)
    if args.limit:
        recs = recs[: args.limit]
    rows: list[dict] = []
    for i, rec in enumerate(recs, 1):
        n_video = len(load_blazepose(rec.blazepose, image_size=None))
        rate = len(load_vicon(rec.vicon)) / (n_video / 10.0)
        if not args.min_rate <= rate <= args.max_rate:
            print(f"[skip] {rec.recording_id}: implied Vicon rate {rate:.1f} Hz")
            continue
        rows += evaluate(rec, args.max_lag)
        if i % 50 == 0 or i == len(recs):
            print(f"processed {i}/{len(recs)}")

    summary = summarise(rows)
    print("\nMedians per exercise and joint (ROM in degrees; r: 1 = moves together)")
    print(f"{'ex':<5}{'joint':<11}{'n':>4}{'axis':>6}{'ROM phone':>10}{'ROM target':>11}{'ROM quat':>9}"
          f"{'r phone~quat':>13}{'MAE':>6}{'r target~quat':>14}")
    for s in summary:
        print(f"{s['exercise']:<5}{s['joint']:<11}{s['n']:>4}{s['axis_share']:>6.2f}{s['rom_phone']:>10.0f}"
              f"{s['rom_target']:>11.0f}{s['rom_quat']:>9.0f}{s['r_phone_quat']:>13.2f}"
              f"{s['mae_phone_quat']:>6.1f}{s['r_target_quat']:>14.2f}")
    write_csv(Path(args.output), rows)
    write_csv(Path(args.summary_output), summary)
    print(f"\nper-recording: {args.output}\nsummary: {args.summary_output}")


if __name__ == "__main__":
    main()
