"""Draw BlazePose on Keraal Group 3 video frames to see where the phone pose fails.

For each chosen recording it saves one image with 4 frames side by side:
  1. the start frame,
  2. the frame where Vicon's LEFT arm is highest,
  3. the frame where Vicon's RIGHT arm is highest,
  4. the frame where phone and Vicon disagree most about the left arm.

Left-side limbs are drawn in ORANGE, right-side limbs in BLUE, torso in grey.
"L" and "R" mark BlazePose's wrists. Each frame shows the signed frontal-plane
arm angle (0 = down, 90 = sideways, 180 = overhead) as change from the start
pose, for the phone and for Vicon.

What to look for
  * Arms overhead but the skeleton's wrists/elbows are elsewhere -> BlazePose loses the arm.
  * Orange drawn on the person's right arm -> left/right labels swapped.
  * Skeleton fine, numbers disagree -> the angle definition is the problem.

Examples
  python scripts/overlay_blazepose.py                       # ELK, one correct recording per participant
  python scripts/overlay_blazepose.py --exercise CTK
  python scripts/overlay_blazepose.py --key G3-ELK-P1T1-Unknown-C-0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rehabform.keraal import BLAZEPOSE_ORDER, iter_recordings, load_blazepose, load_vicon_positions  # noqa: E402
from evaluate_keraal_vicon_offsetfree import from_start, phone_angles, resample, vicon_angles  # noqa: E402

BP = {n: i for i, n in enumerate(BLAZEPOSE_ORDER)}
ORANGE, BLUE, GREY = (0, 140, 255), (255, 120, 0), (170, 170, 170)
EDGES = [
    ("Left_shoulder", "Left_elbow", ORANGE), ("Left_elbow", "Left_wrist", ORANGE),
    ("Left_shoulder", "Left_hip", ORANGE),
    ("Right_shoulder", "Right_elbow", BLUE), ("Right_elbow", "Right_wrist", BLUE),
    ("Right_shoulder", "Right_hip", BLUE),
    ("Left_shoulder", "Right_shoulder", GREY), ("Left_hip", "Right_hip", GREY),
]


def put_text(cv2, img, text, org, scale=0.9):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw(cv2, img, pts):
    def p(name):
        x, y = pts[BP[name], :2]
        return None if not np.isfinite([x, y]).all() else (int(round(x)), int(round(y)))
    for a, b, colour in EDGES:
        pa, pb = p(a), p(b)
        if pa and pb:
            cv2.line(img, pa, pb, colour, 5, cv2.LINE_AA)
    for name in ("Left_shoulder", "Left_elbow", "Left_wrist", "Right_shoulder", "Right_elbow", "Right_wrist"):
        q = p(name)
        if q:
            cv2.circle(img, q, 8, ORANGE if name.startswith("Left") else BLUE, -1, cv2.LINE_AA)
    for name, label in (("Left_wrist", "L"), ("Right_wrist", "R")):
        q = p(name)
        if q:
            put_text(cv2, img, label, (q[0] + 10, q[1] - 10), 1.2)


def read_frame(cv2, path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def overlay(recording, out_dir: Path) -> Path | None:
    import cv2

    bp = load_blazepose(recording.blazepose)            # pixel units
    n = len(bp)
    phone = {k: from_start(v) for k, v in phone_angles(bp)["2d"].items()}
    vic = {k: from_start(resample(v, n)) for k, v in vicon_angles(load_vicon_positions(recording.vicon)).items()}
    pl, pr = phone["arm_abd_signed_l"], phone["arm_abd_signed_r"]
    vl, vr = vic["arm_abd_signed_l"], vic["arm_abd_signed_r"]

    picks = [("start", 0),
             ("Vicon L arm highest", int(np.nanargmax(vl))),
             ("Vicon R arm highest", int(np.nanargmax(vr))),
             ("largest L disagreement", int(np.nanargmax(np.abs(pl - vl))))]

    tiles = []
    for title, t in picks:
        frame = read_frame(cv2, recording.video, t)
        if frame is None:
            print(f"[warn] could not read frame {t} of {recording.video.name}")
            continue
        draw(cv2, frame, bp[t])
        put_text(cv2, frame, f"{title} (frame {t})", (12, 38), 1.05)
        put_text(cv2, frame, f"L arm  phone {pl[t]:+6.0f}  Vicon {vl[t]:+6.0f}", (12, 78))
        put_text(cv2, frame, f"R arm  phone {pr[t]:+6.0f}  Vicon {vr[t]:+6.0f}", (12, 114))
        tiles.append(cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
    if not tiles:
        return None
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])

    # Quick automatic check: which image side does BlazePose's "Left" shoulder sit on?
    sh_l, sh_r = bp[0, BP["Left_shoulder"], 0], bp[0, BP["Right_shoulder"], 0]
    side = "image RIGHT (normal for a person facing an unmirrored camera)" if sh_l > sh_r else \
           "image LEFT (person facing away, or video mirrored, or labels swapped)"
    r = np.corrcoef(np.nan_to_num(pl), np.nan_to_num(vl))[0, 1]
    print(f"{recording.recording_id}: BlazePose 'Left' shoulder on {side}; "
          f"signed left-arm r vs Vicon = {r:+.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"overlay_{recording.recording_id}.png"
    cv2.imwrite(str(out), sheet)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/keraal")
    ap.add_argument("--exercise", default="ELK")
    ap.add_argument("--key", help="one recording_id; overrides --exercise")
    ap.add_argument("--label", default="C", help="error label to pick (C = correctly performed)")
    ap.add_argument("--out-dir", default="outputs/overlays")
    args = ap.parse_args()

    recs = iter_recordings(args.root)
    if args.key:
        chosen = [r for r in recs if r.recording_id == args.key]
    else:
        chosen, seen = [], set()
        for r in recs:
            person = r.participant.split("T")[0]
            if r.exercise == args.exercise and r.error == args.label and person not in seen:
                chosen.append(r)
                seen.add(person)
    if not chosen:
        raise SystemExit("No matching recordings")
    for r in chosen:
        out = overlay(r, Path(args.out_dir))
        if out:
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
