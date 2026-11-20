"""Pillar 2, step 1: which phone measures carry each clinician-labelled error?

For every Group 1A / 2A recording this script
  1. reads both annotators' labels and builds one label table
     (A, B, agreement, error type / severity / body part, A's time windows),
  2. reports how much the two clinicians agree (raw % and Cohen's kappa)
     = the ceiling for any automatic method,
  3. loads BlazePose (pixel units, un-mirrored, left/right fixed by geometry)
     and checks the side fix against Kinect on ELK,
  4. computes recording-level features from rehabform.angles
     (peaks, ranges, spreads, amount of movement),
  5. for each exercise and error type: how well each feature separates
     "both clinicians said Correct" from "annotator A marked this error"
     (AUC; 0.5 = no signal, 1.0 = perfect; direction shown as higher/lower).

Error numbers (Error1-3) mean different things per exercise. The AUC table
plus the body-part counts tell us which error is which, and which measure
should carry it in the findings engine. Thresholds come in the next step,
fitted on some patients and tested on the others.

Outputs (outputs/)
  keraal_labels.csv          one row per recording
  keraal_features.csv        one row per recording, features + labels
  keraal_error_auc.csv       exercise x error type x feature

Example (cmd, repo root, .venv active)
  python scripts\\pillar2_error_signals.py data\\keraal1 data\\keraal2
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.session_features import features  # noqa: E402
from rehabform.keraal import load_kinect_positions, resample_sequence  # noqa: E402
from rehabform.keraal_groups import (  # noqa: E402
    annotation_labels, iter_annotated_recordings, load_blazepose_g12,
)
from rehabform.video_pose import cached_pose  # noqa: E402

L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP = 11, 12, 13, 14, 23, 24
K_SPINE_BASE, K_SH_L, K_EL_L, K_SH_R, K_EL_R, K_SPINE_SH = 0, 4, 5, 8, 9, 20
MIN_N = 4  # minimum recordings on each side before an AUC is reported


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    cats = sorted({c for p in pairs for c in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def build_labels(recs) -> list[dict]:
    rows = []
    for r in recs:
        lab = {k: annotation_labels(p) for k, p in r.annotations.items()}
        a, b = lab.get("A", {}), lab.get("B", {})
        ea, eb = a.get("evaluation"), b.get("evaluation")
        a_err = (a.get("errors") or [{}])[0]
        b_err = (b.get("errors") or [{}])[0]
        wins = a.get("windows") or []
        rows.append({
            "key": r.key, "group": r.group, "exercise": r.exercise, "subject": r.subject,
            "site": r.site, "eval_A": ea, "eval_B": eb,
            "agree": ea == eb if ea and eb else None,
            "consensus": ea if ea == eb else "disagree",
            "type_A": a_err.get("type"), "severity_A": a_err.get("severity"), "body_A": a_err.get("body_part"),
            "type_B": b_err.get("type"), "severity_B": b_err.get("severity"), "body_B": b_err.get("body_part"),
            "n_errors_A": len(a.get("errors") or []), "n_errors_B": len(b.get("errors") or []),
            "window_A_start": wins[0]["start"] if wins else None,
            "window_A_end": wins[0]["end"] if wins else None,
            "n_windows_A": len(wins), "n_windows_B": len(b.get("windows") or []),
        })
    return rows


def report_agreement(labels: list[dict]) -> None:
    print("\n=== clinician agreement (the ceiling)")
    for group in sorted({r["group"] for r in labels}) + ["all"]:
        rows = [r for r in labels if group in ("all", r["group"]) and r["eval_A"] and r["eval_B"]]
        four = [(r["eval_A"], r["eval_B"]) for r in rows]
        two = [(r["eval_A"], r["eval_B"]) for r in rows
               if r["eval_A"] in ("Correct", "Incorrect") and r["eval_B"] in ("Correct", "Incorrect")]
        print(f"{group:<4} 4-class: n={len(four)} agree {np.mean([a == b for a, b in four]):.0%} "
              f"kappa {kappa(four):.2f} | Correct vs Incorrect: n={len(two)} "
              f"agree {np.mean([a == b for a, b in two]):.0%} kappa {kappa(two):.2f}")
        for ex in ("CTK", "ELK", "RTK"):
            t = [(a, b) for (a, b), r in zip(four, rows) if r["exercise"] == ex]
            if not t:
                continue
            print(f"     {ex}: n={len(t)} agree {np.mean([a == b for a, b in t]):.0%} kappa {kappa(t):.2f}")
    both_inc = [r for r in labels if r["eval_A"] == "Incorrect" and r["eval_B"] == "Incorrect"
                and r["type_A"] and r["type_B"]]
    if both_inc:
        same = sum(r["type_A"] == r["type_B"] for r in both_inc)
        print(f"error TYPE agreement when both say Incorrect and both give a type: {same}/{len(both_inc)}")

    print("\n=== error types by exercise (annotator A, Incorrect only): type -> body part counts")
    tab = defaultdict(Counter)
    for r in labels:
        if r["eval_A"] == "Incorrect":
            tab[(r["exercise"], r["type_A"])][f"{r['body_A']}/{r['severity_A']}"] += 1
    for (ex, t), c in sorted(tab.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"  {ex} {t}: n={sum(c.values())}  {dict(c.most_common())}")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# side check against Kinect (ELK)
# ---------------------------------------------------------------------------
def _angle(u, v):
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-6 or np.std(b[ok]) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def kinect_side_vote(bp, kinect_path) -> str:
    k = resample_sequence(load_kinect_positions(kinect_path), len(bp))
    down = k[:, K_SPINE_BASE] - k[:, K_SPINE_SH]
    kl, kr = _angle(k[:, K_EL_L] - k[:, K_SH_L], down), _angle(k[:, K_EL_R] - k[:, K_SH_R], down)
    P = bp[..., :2]
    bl = _angle(P[:, L_EL] - P[:, L_SH], P[:, L_HIP] - P[:, L_SH])
    br = _angle(P[:, R_EL] - P[:, R_SH], P[:, R_HIP] - P[:, R_SH])
    same = np.nanmax([_r(bl, kl), _r(br, kr), -2])
    swap = np.nanmax([_r(bl, kr), _r(br, kl), -2])
    if max(same, swap) < 0.5:
        return "unclear"
    return "agrees" if same >= swap else "disagrees"


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(feature of a positive > feature of a negative), ties count half."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < MIN_N or len(neg) < MIN_N:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                      # average ranks for ties
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty slices on short / missing data
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--pose", choices=["shipped", "mediapipe"], default="shipped",
                    help="shipped = Keraal's BlazePose files; mediapipe = our own MediaPipe run on the videos "
                         "(what the app uses; cached in --mp-cache)")
    ap.add_argument("--model", default=str(ROOT / "models" / "pose_landmarker_full.task"))
    ap.add_argument("--mp-cache", default="outputs/mediapipe_cache")
    ap.add_argument("--min-person", type=float, default=0.5,
                    help="mediapipe: skip recordings where the person is found in fewer frames than this")
    args = ap.parse_args()
    suffix = "" if args.pose == "shipped" else "_mediapipe"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs = [r for root in args.roots for r in iter_annotated_recordings(root)]
    labels = build_labels(recs)
    report_agreement(labels)
    by_key = {r["key"]: r for r in labels}

    rows, side_stats, votes = [], Counter(), Counter()
    skipped_person = []
    for i, r in enumerate(recs, 1):
        if not (r.blazepose if args.pose == "shipped" else r.video):
            continue
        try:
            if args.pose == "shipped":
                bp, info = load_blazepose_g12(r.blazepose)
            else:
                bp, info = cached_pose(r.video, args.model, args.mp_cache, r.key)
                if 1 - info["missing_fraction"] < args.min_person:
                    skipped_person.append(r.key)
                    continue
        except Exception as exc:
            print(f"[skip] {r.key}: {exc}")
            continue
        side_stats["swapped" if info["swapped"] else "kept"] += 1
        if r.exercise == "ELK" and r.kinect:
            try:
                votes[kinect_side_vote(bp, r.kinect)] += 1
            except Exception:
                votes["kinect error"] += 1
        row = dict(by_key[r.key])
        row.update(sides_swapped=info["swapped"], frames_reversed=round(info["frames_reversed"], 3))
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in features(bp).items()})
        rows.append(row)
        if i % 50 == 0:
            print(f"processed {i}/{len(recs)}")

    if skipped_person:
        print(f"\n[mediapipe] {len(skipped_person)} recordings skipped, person found in < {args.min_person:.0%} "
              f"of frames: {skipped_person[:8]}{' ...' if len(skipped_person) > 8 else ''}")
    print(f"\n=== pose source: {args.pose} ({len(rows)} recordings with features)")
    print("\n=== left/right fix (after un-mirroring)")
    print(f"recordings whose BlazePose sides were swapped by geometry: {side_stats['swapped']} of "
          f"{sum(side_stats.values())}")
    print(f"ELK, fixed BlazePose vs Kinect arm movement: {dict(votes)}  (want: mostly 'agrees')")
    fr = [r["frames_reversed"] for r in rows if np.isfinite(r["frames_reversed"])]
    if fr:
        print(f"frames still reversed within a recording: median {np.median(fr):.1%}, "
              f"recordings with >10%: {sum(x > 0.10 for x in fr)}")

    with open(out / f"keraal_labels{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0]))
        w.writeheader()
        w.writerows(labels)
    with open(out / f"keraal_features{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    feat_names = [k for k in rows[0] if k not in labels[0] and k not in ("sides_swapped", "frames_reversed")]
    print("\n=== which feature separates each error from 'both said Correct' (AUC, best 4)")
    print("    higher = the feature is larger in the error recordings; lower = smaller")
    auc_rows = []
    for ex in ("CTK", "ELK", "RTK"):
        neg = [r for r in rows if r["exercise"] == ex and r["consensus"] == "Correct"]
        types = Counter(r["type_A"] for r in rows if r["exercise"] == ex and r["eval_A"] == "Incorrect")
        targets = [(t, [r for r in rows if r["exercise"] == ex and r["eval_A"] == "Incorrect" and r["type_A"] == t])
                   for t in sorted(types, key=str)]
        targets.append(("any (A Incorrect)", [r for r in rows if r["exercise"] == ex and r["eval_A"] == "Incorrect"]))
        targets.append(("Incomplete (A)", [r for r in rows if r["exercise"] == ex and r["eval_A"] == "Incomplete"]))
        for t, pos in targets:
            res = []
            for fn in feat_names:
                a = auc(np.array([r[fn] for r in pos], float), np.array([r[fn] for r in neg], float))
                if np.isfinite(a):
                    res.append((max(a, 1 - a), "higher" if a >= 0.5 else "lower", fn, a))
                    auc_rows.append({"exercise": ex, "error": t, "feature": fn, "auc": round(a, 3),
                                     "n_error": len(pos), "n_correct": len(neg),
                                     "subjects_error": len({r["subject"] for r in pos})})
            res.sort(reverse=True)
            subj = len({r["subject"] for r in pos})
            best = ", ".join(f"{fn} {d} {s:.2f}" for s, d, fn, _ in res[:4]) or f"too few (<{MIN_N})"
            print(f"{ex} {t:<18} n={len(pos):>3} vs {len(neg):>3} correct, {subj} subjects: {best}")

    with open(out / f"keraal_error_auc{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exercise", "error", "feature", "auc", "n_error", "n_correct",
                                          "subjects_error"])
        w.writeheader()
        w.writerows(auc_rows)
    print(f"\nwrote {out / f'keraal_labels{suffix}.csv'}, {out / f'keraal_features{suffix}.csv'}, "
          f"{out / f'keraal_error_auc{suffix}.csv'}")


if __name__ == "__main__":
    main()
