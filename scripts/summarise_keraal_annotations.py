"""Summarise a Keraal Group 1A / 2A folder: files, clinician annotations, video
format and whether the shipped BlazePose is mirrored.

What it prints
  1. recordings found, per exercise / subject, and which modalities are missing
  2. every annotation track and attribute name with its value counts, per annotator
  3. annotator A vs B agreement on the global evaluation (when both labelled)
  4. one full annotation that has more than the global-evaluation track
  5. video size / fps / frame count vs BlazePose frame count (first few videos)
  6. mirroring check: BlazePose arm angles vs Kinect arm angles in ELK
     (one arm is raised at a time, so the correct pairing stands out)

What it writes
  outputs/keraal_<group>_annotations_long.csv   one row per annotated element
  outputs/overlays/mirror_<key>.png             with --overlay N: raw vs un-mirrored
                                                skeleton on N ELK videos

Examples (cmd, repo root)
  python scripts\\summarise_keraal_annotations.py data\\keraal2
  python scripts\\summarise_keraal_annotations.py data\\keraal2 --overlay 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.keraal import BLAZEPOSE_ORDER, load_blazepose, load_kinect_positions, resample_sequence  # noqa: E402
from rehabform.keraal_groups import iter_annotated_recordings, read_anvil  # noqa: E402

IMAGE_SIZE_G12 = (480, 360)  # dataset page; checked against the videos below
BP = {n: i for i, n in enumerate(BLAZEPOSE_ORDER)}
# Kinect v2 joint order (readme_files_format.txt)
K_SPINE_BASE, K_SHOULDER_L, K_ELBOW_L, K_SHOULDER_R, K_ELBOW_R, K_SPINE_SHOULDER = 0, 4, 5, 8, 9, 20


def section(title: str) -> None:
    print(f"\n=== {title}")


# ---------------------------------------------------------------------------
# 1-4 annotations
# ---------------------------------------------------------------------------
def global_label(elements: list[dict]) -> str | None:
    for e in elements:
        if (e["track"] or "").lower().startswith("global"):
            vals = list(e["attributes"].values())
            return vals[0] if vals else None
    return None


def summarise_annotations(recs, out_csv: Path) -> None:
    rows, by_track = [], defaultdict(Counter)
    per_annotator_global: dict[str, dict[str, str | None]] = defaultdict(dict)
    example = None
    bad = []
    for r in recs:
        for annotator, path in sorted(r.annotations.items()):
            try:
                parsed = read_anvil(path)
            except Exception as exc:  # report and carry on
                bad.append((path, exc))
                continue
            els = parsed["elements"]
            per_annotator_global[annotator][r.key] = global_label(els)
            tracks = {e["track"] for e in els}
            if example is None and len(tracks) > 1:
                example = (path, parsed)
            for e in els:
                rows.append({
                    "key": r.key, "group": r.group, "exercise": r.exercise, "subject": r.subject,
                    "site": r.site, "annotator": annotator, "track": e["track"],
                    "index": e["index"], "start": e["start"], "end": e["end"],
                    "attributes": json.dumps(e["attributes"], ensure_ascii=False),
                    "anvil_video": parsed["video"],
                })
                if not e["attributes"]:
                    by_track[(annotator, e["track"], "(no attribute)")][""] += 1
                for name, value in e["attributes"].items():
                    by_track[(annotator, e["track"], name)][value] += 1

    section("2. annotation tracks and attribute values")
    for (annotator, track, name), values in sorted(by_track.items()):
        shown = ", ".join(f"{v!r}({n})" for v, n in values.most_common(12))
        more = f" ... +{len(values) - 12} more" if len(values) > 12 else ""
        print(f"[{annotator}] {track} / {name}: {shown}{more}")
    for path, exc in bad:
        print(f"[parse error] {path}: {exc}")

    section("3. annotator agreement on the global evaluation")
    names = sorted(per_annotator_global)
    if len(names) >= 2:
        a, b = names[:2]
        common = sorted(set(per_annotator_global[a]) & set(per_annotator_global[b]))
        pairs = Counter((per_annotator_global[a][k], per_annotator_global[b][k]) for k in common)
        agree = sum(n for (x, y), n in pairs.items() if x == y)
        print(f"{len(common)} recordings labelled by both {a} and {b}; agree on {agree} "
              f"({100 * agree / max(len(common), 1):.0f}%)")
        for (x, y), n in pairs.most_common():
            print(f"  {a}={x!r:<14} {b}={y!r:<14} {n}")
        by_ex = defaultdict(Counter)
        for r in recs:
            if r.key in common:
                by_ex[r.exercise][per_annotator_global[a][r.key]] += 1
        print(f"global label by exercise ({a}):", {k: dict(v) for k, v in sorted(by_ex.items())})
    else:
        print("only one annotator:", names)

    section("4. one full annotation with more than one track")
    if example:
        path, parsed = example
        print(f"{path}  (video src in file: {parsed['video']})")
        for e in parsed["elements"]:
            print(f"  {e['track']!r:<28} [{e['start']}, {e['end']}] {e['attributes']}")
    else:
        print("every annotation has a single track (global evaluation only)")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["key"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nlong table: {out_csv} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 5 video format
# ---------------------------------------------------------------------------
def check_videos(recs, n: int = 5) -> tuple[int, int] | None:
    section("5. video format vs BlazePose frames")
    try:
        import cv2
    except ImportError:
        print("opencv not installed in this Python; skipped (activate .venv)")
        return None
    sizes = Counter()
    for r in [r for r in recs if r.video and r.blazepose][:n]:
        cap = cv2.VideoCapture(str(r.video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with open(r.blazepose, encoding="utf-8") as f:
            bp_frames = len(json.load(f)["positions"])
        sizes[(w, h)] += 1
        print(f"{r.key}: {w}x{h}, {fps:.1f} fps, {count} video frames, {bp_frames} BlazePose frames")
    return sizes.most_common(1)[0][0] if sizes else None


# ---------------------------------------------------------------------------
# 6 mirroring
# ---------------------------------------------------------------------------
def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def bp_arm_angles(bp: np.ndarray) -> dict[str, np.ndarray]:
    """Upper arm vs trunk (shoulder->hip), 2D image plane. Same idea per side."""
    out = {}
    for side in ("Left", "Right"):
        sh, el, hip = bp[:, BP[f"{side}_shoulder"], :2], bp[:, BP[f"{side}_elbow"], :2], bp[:, BP[f"{side}_hip"], :2]
        out[side] = _angle(el - sh, hip - sh)
    return out


def kinect_arm_angles(k: np.ndarray) -> dict[str, np.ndarray]:
    down = k[:, K_SPINE_BASE] - k[:, K_SPINE_SHOULDER]
    return {"Left": _angle(k[:, K_ELBOW_L] - k[:, K_SHOULDER_L], down),
            "Right": _angle(k[:, K_ELBOW_R] - k[:, K_SHOULDER_R], down)}


def _r(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-6 or np.std(b[ok]) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def check_mirroring(recs, image_size, limit: int = 20) -> None:
    section("6. mirroring check (ELK: BlazePose vs Kinect arm angle)")
    votes = Counter()
    for r in [r for r in recs if r.exercise == "ELK" and r.blazepose and r.kinect][:limit]:
        try:
            raw = load_blazepose(r.blazepose, image_size=image_size, unmirror=False)
            kin = load_kinect_positions(r.kinect)
        except Exception as exc:
            print(f"[skip] {r.key}: {exc}")
            continue
        kin = resample_sequence(kin, len(raw))
        b, k = bp_arm_angles(raw), kinect_arm_angles(kin)
        # A still arm has no correlation (NaN), so take the better-matching arm of each pairing
        def best(*rs: float) -> float:
            rs = [x for x in rs if np.isfinite(x)]
            return max(rs) if rs else float("nan")
        same = best(_r(b["Left"], k["Left"]), _r(b["Right"], k["Right"]))
        swapped = best(_r(b["Left"], k["Right"]), _r(b["Right"], k["Left"]))
        if not (np.isfinite(same) or np.isfinite(swapped)):
            verdict = "undecided (no arm movement)"
        elif not np.isfinite(same) or (np.isfinite(swapped) and swapped > same):
            verdict = "MIRRORED (names swapped)"
        else:
            verdict = "not mirrored"
        votes[verdict] += 1
        print(f"{r.key}: r same-side {same:+.2f}  r swapped {swapped:+.2f}  -> {verdict}")
    print("votes:", dict(votes))
    print("Confirm by eye with --overlay N (the skeleton that sits on the body is the right one).")


def overlay(recs, image_size, n: int, out_dir: Path) -> None:
    import cv2

    orange, blue, grey = (0, 140, 255), (255, 120, 0), (170, 170, 170)
    edges = [("Left_shoulder", "Left_elbow", orange), ("Left_elbow", "Left_wrist", orange),
             ("Left_shoulder", "Left_hip", orange), ("Right_shoulder", "Right_elbow", blue),
             ("Right_elbow", "Right_wrist", blue), ("Right_shoulder", "Right_hip", blue),
             ("Left_shoulder", "Right_shoulder", grey), ("Left_hip", "Right_hip", grey)]

    def draw(img, pts, title):
        for a, b, c in edges:
            pa, pb = pts[BP[a], :2], pts[BP[b], :2]
            if np.isfinite([*pa, *pb]).all():
                cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), c, 3, cv2.LINE_AA)
        for name, lab in (("Left_wrist", "L"), ("Right_wrist", "R")):
            q = pts[BP[name], :2]
            if np.isfinite(q).all():
                cv2.putText(img, lab, (int(q[0]) + 6, int(q[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    for r in [r for r in recs if r.exercise == "ELK" and r.video and r.blazepose][:n]:
        raw = load_blazepose(r.blazepose, image_size=image_size, unmirror=False)
        fixed = load_blazepose(r.blazepose, image_size=image_size, unmirror=True)
        # frame where either wrist is highest (smallest y) = arm up
        t = int(np.nanargmin(np.nanmin(fixed[:, [BP["Left_wrist"], BP["Right_wrist"]], 1], axis=1)))
        cap = cv2.VideoCapture(str(r.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"[warn] could not read frame {t} of {r.video.name}")
            continue
        a, b = frame.copy(), frame.copy()
        draw(a, raw[t], f"as shipped (frame {t})")
        draw(b, fixed[t], "un-mirrored")
        out = out_dir / f"mirror_{r.key}.png"
        cv2.imwrite(str(out), np.hstack([a, b]))
        print(f"saved {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--overlay", type=int, default=0, help="save raw vs un-mirrored images for N ELK videos")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    recs = iter_annotated_recordings(args.root)
    if not recs:
        raise SystemExit(f"No Group 1/2 recordings under {args.root} (is the .tar.xz extracted?)")
    group = recs[0].group

    section(f"1. recordings in {args.root}")
    print(f"{len(recs)} recordings, group(s) {sorted({r.group for r in recs})}")
    print("by exercise:", dict(Counter(r.exercise for r in recs)))
    print("by subject :", dict(sorted(Counter(r.subject for r in recs).items())))
    print("by site    :", dict(Counter(r.site for r in recs)))
    print("annotators :", dict(Counter(a for r in recs for a in r.annotations)))
    for attr in ("video", "blazepose", "openpose", "kinect"):
        missing = [r.key for r in recs if getattr(r, attr) is None]
        if missing:
            print(f"missing {attr}: {len(missing)} e.g. {missing[:3]}")
    no_ann = [r.key for r in recs if not r.annotations]
    if no_ann:
        print(f"no annotation: {len(no_ann)} e.g. {no_ann[:3]}")

    summarise_annotations(recs, Path(args.out_dir) / f"keraal_{group}_annotations_long.csv")
    size = check_videos(recs) or IMAGE_SIZE_G12
    check_mirroring(recs, size)
    if args.overlay:
        overlay(recs, size, args.overlay, Path(args.out_dir) / "overlays")


if __name__ == "__main__":
    main()
