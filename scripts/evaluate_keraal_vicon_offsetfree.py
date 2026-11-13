"""Offset-free BlazePose-vs-Vicon evaluation on Keraal Group 3.

Why "offset-free": Vicon tracks ART targets on body SEGMENTS (mid upper arm,
mid forearm, one pelvis target), not joint centres. Any angle built from those
targets differs from the anatomical angle by a roughly fixed offset, so
absolute-angle errors against Vicon mostly measure that offset. Here we compare
what a clinician actually uses and what the offset cancels out of:

  * change from the starting pose (median of the first frames), and
  * range of motion (ROM) per recording (95th minus 5th percentile).

Vicon angles use segment DIRECTIONS rather than treating targets as joints:
  upper arm = arm target - shoulder target
  forearm   = hand target - forearm target
  trunk     = chest target - hips target

Angles (degrees)
  arm_elev_l/r  upper arm vs trunk (0 = arm by the side)
  elbow_l/r     elbow bend (0 = straight)
  trunk_lat     sideways lean (signed; BlazePose 2D only)
  trunk_rot     twist about the vertical axis (signed; BlazePose 3D only)
  arm_abd_signed_l/r  arm angle in the body's frontal plane, SIGNED and unwrapped:
                0 = arm down, +90 = out to the side, +180 = overhead, >180 =
                past overhead, <0 = across the body. Same trunk-midline and
                shoulder-line references for both systems, so no sign flip is
                applied and a negative r means truly opposite motion.
                (BlazePose 2D only; tests whether overhead failures come from
                the unsigned angle folding back near 180 degrees.)

BlazePose is loaded in pixel units (aspect-ratio corrected) and evaluated in
two variants: "2d" (image x, y) and "3d" (adds BlazePose's relative depth).

Examples
  python scripts/evaluate_keraal_vicon_offsetfree.py --limit 3
  python scripts/evaluate_keraal_vicon_offsetfree.py --key G3-CTK-P1T1-Unknown-C-0
  python scripts/evaluate_keraal_vicon_offsetfree.py          # all recordings
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rehabform.keraal import (  # noqa: E402
    BLAZEPOSE_ORDER,
    VICON_TARGETS,
    iter_recordings,
    load_blazepose,
    load_vicon_positions,
)

BP = {name: i for i, name in enumerate(BLAZEPOSE_ORDER)}
VI = {name: i for i, name in enumerate(VICON_TARGETS)}
SIGNED = {"trunk_lat", "trunk_rot"}
START_FRAMES = 5


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------
def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    ok = np.isfinite(a)
    if ok.sum() > 1:
        out[ok] = np.degrees(np.unwrap(np.radians(a[ok])))
    return out


def signed_plane_angle(arm: np.ndarray, down: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Signed angle (deg) of `arm` from `down`, positive towards `outward`, in the
    plane spanned by down and outward. All inputs (T, D); D = 2 or 3. Unwrapped."""
    d = down / (np.linalg.norm(down, axis=-1, keepdims=True) + 1e-9)
    o = outward - np.sum(outward * d, axis=-1, keepdims=True) * d   # make outward perpendicular to down
    o = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)
    ang = np.degrees(np.arctan2(np.sum(arm * o, axis=-1), np.sum(arm * d, axis=-1)))
    return unwrap_deg(ang)


def phone_angles(bp_pixels: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """bp_pixels: (T, 33, 3) in pixel units. Returns {'2d': {...}, '3d': {...}}."""
    out = {}
    for variant, P in (("2d", bp_pixels[..., :2]), ("3d", bp_pixels)):
        g = lambda name: P[:, BP[name]]  # noqa: E731
        a = {}
        for side, S in (("l", "Left"), ("r", "Right")):
            sh, el, wr, hip = g(f"{S}_shoulder"), g(f"{S}_elbow"), g(f"{S}_wrist"), g(f"{S}_hip")
            a[f"arm_elev_{side}"] = angle_between(el - sh, hip - sh)
            a[f"elbow_{side}"] = angle_between(el - sh, wr - el)
        trunk = (g("Left_shoulder") + g("Right_shoulder")) / 2 - (g("Left_hip") + g("Right_hip")) / 2
        if variant == "2d":
            a["trunk_lat"] = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))  # image y points down
            for side, S, O in (("l", "Left", "Right"), ("r", "Right", "Left")):
                a[f"arm_abd_signed_{side}"] = signed_plane_angle(
                    g(f"{S}_elbow") - g(f"{S}_shoulder"), -trunk, g(f"{S}_shoulder") - g(f"{O}_shoulder"))
        else:
            line = g("Left_shoulder") - g("Right_shoulder")
            a["trunk_rot"] = unwrap_deg(np.degrees(np.arctan2(line[:, 2], line[:, 0])))
        out[variant] = a
    return out


def vicon_angles(vicon: np.ndarray) -> dict[str, np.ndarray]:
    """vicon: (T, 17, 3) target positions, mm, z up."""
    g = lambda name: vicon[:, VI[name]]  # noqa: E731
    a = {}
    trunk_down = g("hips") - g("chest")
    for side in ("l", "r"):
        upper = g(f"{side}_arm") - g(f"{side}_shoulder")
        fore = g(f"{side}_hand") - g(f"{side}_forearm")
        a[f"arm_elev_{side}"] = angle_between(upper, trunk_down)
        a[f"elbow_{side}"] = angle_between(upper, fore)
    for side, other in (("l", "r"), ("r", "l")):
        a[f"arm_abd_signed_{side}"] = signed_plane_angle(
            g(f"{side}_arm") - g(f"{side}_shoulder"), trunk_down,
            g(f"{side}_shoulder") - g(f"{other}_shoulder"))
    lateral = g("l_shoulder")[0] - g("r_shoulder")[0]
    lateral[2] = 0.0
    lateral /= np.linalg.norm(lateral) + 1e-9
    trunk_up = g("chest") - g("hips")
    a["trunk_lat"] = np.degrees(np.arctan2(trunk_up @ lateral, trunk_up[:, 2]))
    line = g("l_shoulder") - g("r_shoulder")
    a["trunk_rot"] = unwrap_deg(np.degrees(np.arctan2(line[:, 1], line[:, 0])))
    return a


# ---------------------------------------------------------------------------
# Time base and metrics
# ---------------------------------------------------------------------------
def resample(x: np.ndarray, n: int) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full(n, np.nan)
    src = np.linspace(0.0, 1.0, len(x))
    return np.interp(np.linspace(0.0, 1.0, n), src[ok], x[ok])


def from_start(x: np.ndarray) -> np.ndarray:
    head = x[:START_FRAMES]
    base = np.nanmedian(head) if np.isfinite(head).any() else np.nanmedian(x)
    return x - base


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if lag > 0:
        out[lag:] = x[:-lag]
    elif lag < 0:
        out[:lag] = x[-lag:]
    else:
        out[:] = x
    return out


def rom(x: np.ndarray) -> float:
    return float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5)) if np.isfinite(x).any() else float("nan")


def compare(phone: np.ndarray, vicon: np.ndarray, signed: bool, max_lag: int) -> dict[str, float]:
    best_lag, best_r = 0, corr(phone, vicon)
    for lag in range(-max_lag, max_lag + 1):
        r = corr(shift(phone, lag), vicon)
        if np.isfinite(r) and (not np.isfinite(best_r) or abs(r) > abs(best_r)):
            best_lag, best_r = lag, r
    p = shift(phone, best_lag)
    flipped = bool(signed and np.isfinite(best_r) and best_r < 0)
    if flipped:
        p = -p
    ok = np.isfinite(p) & np.isfinite(vicon)
    mae = float(np.mean(np.abs(p[ok] - vicon[ok]))) if ok.any() else float("nan")
    rom_p, rom_v = rom(p), rom(vicon)
    return {"lag": best_lag, "flipped": int(flipped),
            "r": abs(best_r) if (signed and np.isfinite(best_r)) else best_r,
            "change_mae": mae, "rom_phone": rom_p, "rom_vicon": rom_v, "rom_err": rom_p - rom_v}


def evaluate(recording, max_lag: int) -> list[dict]:
    bp = load_blazepose(recording.blazepose)          # pixel units
    vicon = load_vicon_positions(recording.vicon)
    n = len(bp)
    va = vicon_angles(vicon)
    rows = []
    for variant, angles in phone_angles(bp).items():
        for name, series in angles.items():
            v = from_start(resample(va[name], n))
            res = compare(from_start(series), v, name in SIGNED, max_lag)
            rows.append({
                "recording_id": recording.recording_id, "exercise": recording.exercise,
                "participant": recording.participant.split("T")[0], "label": recording.label,
                "error": recording.error, "angle": name, "variant": variant,
                "n_video": n, "n_vicon": len(vicon), **res,
            })
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarise(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    warnings.filterwarnings("ignore", message="All-NaN slice")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, items in sorted(groups.items()):
        col = lambda c: np.asarray([float(i[c]) for i in items], dtype=float)  # noqa: E731
        out.append({**dict(zip(keys, key)), "recordings": len(items),
                    "median_r": float(np.nanmedian(col("r"))),
                    "median_change_mae": float(np.nanmedian(col("change_mae"))),
                    "mean_rom_err": float(np.nanmean(col("rom_err"))),
                    "mean_abs_rom_err": float(np.nanmean(np.abs(col("rom_err"))))})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict], keys: tuple[str, ...]) -> None:
    head = "".join(f"{k:<18}" for k in keys) + f"{'n':>5}{'med r':>8}{'med MAE':>9}{'ROM err':>9}{'|ROM err|':>10}"
    print(head)
    for r in rows:
        print("".join(f"{str(r[k]):<18}" for k in keys)
              + f"{r['recordings']:>5}{r['median_r']:>8.2f}{r['median_change_mae']:>9.1f}"
              + f"{r['mean_rom_err']:>9.1f}{r['mean_abs_rom_err']:>10.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/keraal")
    p.add_argument("--key", help="one recording_id, e.g. G3-CTK-P1T1-Unknown-C-0")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-lag", type=int, default=10, help="lag search range, video frames (10 fps)")
    p.add_argument("--min-rate", type=float, default=15.0, help="skip recordings whose implied Vicon rate is below this (Hz)")
    p.add_argument("--max-rate", type=float, default=30.0, help="skip recordings whose implied Vicon rate is above this (Hz)")
    p.add_argument("--output", default="outputs/keraal_vicon_offsetfree_per_recording.csv")
    p.add_argument("--summary-output", default="outputs/keraal_vicon_offsetfree_summary.csv")
    args = p.parse_args()

    recordings = iter_recordings(args.root)
    if args.key:
        recordings = [r for r in recordings if r.recording_id == args.key]
        if not recordings:
            raise SystemExit(f"No recording with id {args.key}")
    if args.limit:
        recordings = recordings[: args.limit]

    rows: list[dict] = []
    for i, rec in enumerate(recordings, 1):
        n_video = len(load_blazepose(rec.blazepose, image_size=None))
        n_vicon = len(load_vicon_positions(rec.vicon))
        rate = n_vicon / (n_video / 10.0)
        if not args.min_rate <= rate <= args.max_rate:
            print(f"[skip] {rec.recording_id}: Vicon/video length gives {rate:.1f} Hz "
                  f"(expected {args.min_rate:g}-{args.max_rate:g}); lengths probably mismatched")
            continue
        rows += evaluate(rec, args.max_lag)
        if i % 50 == 0 or i == len(recordings):
            print(f"processed {i}/{len(recordings)}")
    rate = np.median([r["n_vicon"] / (r["n_video"] / 10.0) for r in rows])
    print(f"Median Vicon rate: {rate:.1f} Hz (video 10 fps)\n")

    by_ex = summarise(rows, ("exercise", "angle", "variant"))
    print("By exercise:")
    print_table(by_ex, ("exercise", "angle", "variant"))
    print("\nBy participant (all exercises):")
    print_table(summarise(rows, ("participant", "angle", "variant")), ("participant", "angle", "variant"))

    write_csv(Path(args.output), rows)
    write_csv(Path(args.summary_output), by_ex)
    print(f"\nper-recording: {args.output}\nsummary: {args.summary_output}")


if __name__ == "__main__":
    main()
