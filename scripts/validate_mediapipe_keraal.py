"""Does our own MediaPipe run give the same features and decisions as Keraal's shipped BlazePose?

The thresholds were fitted on Keraal's BlazePose files. The app will run
MediaPipe Pose Landmarker on the phone instead. This runs the landmarker on
Keraal videos and compares, recording by recording:
  * each feature the detectors use (correlation, median difference),
  * each validated check's decision (error / borderline / ok),
  * how often the person is found, and the time per video.

Example (cmd, repo root, .venv active; about 5-20 s per video on a laptop CPU)
  python scripts\\validate_mediapipe_keraal.py data\\keraal1 data\\keraal2 --limit 40
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.findings import DEFAULT_THRESHOLDS, FindingsEngine  # noqa: E402
from rehabform.keraal_groups import iter_annotated_recordings, load_blazepose_g12  # noqa: E402
from rehabform.session_features import features  # noqa: E402
from rehabform.video_pose import cached_pose  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "models" / "pose_landmarker_full.task"))
    ap.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    ap.add_argument("--limit", type=int, default=40, help="random sample of CTK/ELK recordings (0 = all)")
    ap.add_argument("--out", default="outputs/mediapipe_vs_blazepose.csv")
    ap.add_argument("--cache", default="outputs/mediapipe_cache", help="landmarks saved per recording")
    args = ap.parse_args()

    engine = FindingsEngine(args.thresholds)
    exercises = {d["exercise"] for d in engine.detectors.values()}
    feats = sorted({d["feature"] for d in engine.detectors.values()})
    recs = [r for root in args.roots for r in iter_annotated_recordings(root)
            if r.video and r.blazepose and r.exercise in exercises]
    if args.limit:
        recs = random.Random(0).sample(recs, min(args.limit, len(recs)))
    print(f"{len(recs)} recordings ({dict(Counter(r.exercise for r in recs))}); features compared: {feats}")

    rows, decisions = [], Counter()
    for i, r in enumerate(sorted(recs, key=lambda r: r.key), 1):
        try:
            bp_ship, _ = load_blazepose_g12(r.blazepose)
            t0 = time.time()
            bp_mp, info = cached_pose(r.video, args.model, args.cache, r.key)
            secs = time.time() - t0
            f_ship, f_mp = features(bp_ship), features(bp_mp)
            d_ship = {f.detector: f.status for f in engine.run(bp_ship, r.exercise)}
            d_mp = {f.detector: f.status for f in engine.run(bp_mp, r.exercise)}
            d_mp = {det: d_mp.get(det, "skipped") for det in d_ship}   # skipped = person seen in < half the frames
        except Exception as exc:
            import traceback
            print(f"[skip] {r.key}: {type(exc).__name__}: {exc} "
                  f"({traceback.extract_tb(exc.__traceback__)[-1].name})")
            continue
        row = {"key": r.key, "exercise": r.exercise, "frames_shipped": len(bp_ship), "frames_mediapipe": len(bp_mp),
               "person_found": round(1 - info["missing_fraction"], 3), "sides_swapped": info["swapped"],
               "seconds": round(secs, 1)}
        for f in feats:
            row[f"{f}_shipped"] = round(f_ship.get(f, float("nan")), 3)
            row[f"{f}_mediapipe"] = round(f_mp.get(f, float("nan")), 3)
        for det in d_ship:
            row[f"{det}_shipped"], row[f"{det}_mediapipe"] = d_ship[det], d_mp[det]
            decisions[(det, d_ship[det], d_mp[det])] += 1
        rows.append(row)
        print(f"[{i}/{len(recs)}] {r.key}: {secs:.0f} s, person found {row['person_found']:.0%}, "
              + ", ".join(f"{det} {d_ship[det]}->{d_mp.get(det)}" for det in d_ship))

    if not rows:
        raise SystemExit("nothing compared")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                                                 if k in rows[0] else 999))
        w.writeheader()
        w.writerows(rows)
    print("\n=== features: own MediaPipe vs shipped BlazePose")
    for f in feats:
        a = np.array([r[f"{f}_shipped"] for r in rows], float)
        b = np.array([r[f"{f}_mediapipe"] for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3:
            rr = np.corrcoef(a[ok], b[ok])[0, 1]
            print(f"  {f:<20} n={ok.sum():>3}  r {rr:.2f}  median difference (MediaPipe - shipped) "
                  f"{np.median(b[ok] - a[ok]):+.3g}  median |difference| {np.median(np.abs(b[ok] - a[ok])):.3g}")
    print("\n=== decisions: shipped -> MediaPipe")
    for det in sorted({k[0] for k in decisions}):
        same = sum(n for (d, s, m), n in decisions.items() if d == det and s == m)
        tot = sum(n for (d, _s, _m), n in decisions.items() if d == det)
        print(f"  {det:<24} same decision {same}/{tot}  "
              + str({f"{s}->{m}": n for (d, s, m), n in sorted(decisions.items(), key=str) if d == det}))
    print(f"\nperson found: median {np.median([r['person_found'] for r in rows]):.0%}; "
          f"time per video: median {np.median([r['seconds'] for r in rows]):.0f} s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
