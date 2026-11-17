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

from rehabform.angles import compute_all  # noqa: E402
from rehabform.keraal import load_kinect_positions, resample_sequence  # noqa: E402
from rehabform.keraal_groups import (  # noqa: E402
    annotation_labels, iter_annotated_recordings, load_blazepose_g12,
)

FPS = 10.0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
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
def _p(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def features(bp: np.ndarray) -> dict:
    m = {k: v for k, (v, _ok) in compute_all(bp).items()}
    T = len(bp)
    P = bp[..., :2]
    sw = np.nanmedian(np.linalg.norm(P[:, L_SH] - P[:, R_SH], axis=1))  # shoulder width, px
    f = {"duration_s": T / FPS}

    # how much the person moves (wrists, in shoulder widths)
    f["movement"] = float(np.nanmean([np.nanstd(P[:, j], axis=0).sum() for j in (L_WR, R_WR)]) / sw)

    p95 = {s: _p(m[f"arm_elevation_{s}"], 95) for s in "lr"}
    hi = max(p95, key=lambda s: p95[s] if np.isfinite(p95[s]) else -1e9)
    lo = "r" if hi == "l" else "l"
    f["arm_peak_high"] = p95[hi]                 # the higher arm (ELK: the reaching arm)
    f["arm_peak_low"] = p95[lo]                  # the other arm (ELK: should stay down)
    f["arm_peak_mean"] = float(np.nanmean([p95["l"], p95["r"]]))   # CTK / RTK: both arms
    f["arm_range_mean"] = float(np.nanmean([_p(m[f"arm_elevation_{s}"], 95) - _p(m[f"arm_elevation_{s}"], 5)
                                            for s in "lr"]))
    f["elbow_reach_p50"] = _p(m[f"elbow_flexion_{hi}"], 50)
    f["elbow_reach_p95"] = _p(m[f"elbow_flexion_{hi}"], 95)
    f["elbow_mean_p50"] = float(np.nanmean([_p(m[f"elbow_flexion_{s}"], 50) for s in "lr"]))

    lean = m["trunk_lean"]
    lean0 = np.nanmedian(lean[: max(3, T // 10)])                  # offset-free: change from start
    f["lean_abs_p95"] = _p(np.abs(lean - lean0), 95)
    f["lean_range"] = _p(lean, 95) - _p(lean, 5)
    rot = m["trunk_rotation"]
    f["rotation_range"] = _p(rot, 95) - _p(rot, 5)                 # timing-only measure: range is indicative

    # spreads and heights in shoulder widths (CTK "arms not outspread" has no validated measure yet)
    f["wrist_spread_p95"] = _p(np.linalg.norm(P[:, L_WR] - P[:, R_WR], axis=1) / sw, 95)
    f["elbow_spread_p95"] = _p(np.linalg.norm(P[:, L_EL] - P[:, R_EL], axis=1) / sw, 95)
    sh_y = (P[:, L_SH, 1] + P[:, R_SH, 1]) / 2
    f["wrist_height_p95"] = _p((sh_y - np.minimum(P[:, L_WR, 1], P[:, R_WR, 1])) / sw, 95)   # + = above shoulders
    f["elbow_height_p95"] = _p((sh_y - (P[:, L_EL, 1] + P[:, R_EL, 1]) / 2) / sw, 95)
    return f


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
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs = [r for root in args.roots for r in iter_annotated_recordings(root)]
    labels = build_labels(recs)
    report_agreement(labels)
    by_key = {r["key"]: r for r in labels}

    rows, side_stats, votes = [], Counter(), Counter()
    for i, r in enumerate(recs, 1):
        if not r.blazepose:
            continue
        try:
            bp, info = load_blazepose_g12(r.blazepose)
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

    print("\n=== left/right fix (after un-mirroring)")
    print(f"recordings whose BlazePose sides were swapped by geometry: {side_stats['swapped']} of "
          f"{sum(side_stats.values())}")
    print(f"ELK, fixed BlazePose vs Kinect arm movement: {dict(votes)}  (want: mostly 'agrees')")
    fr = [r["frames_reversed"] for r in rows if np.isfinite(r["frames_reversed"])]
    if fr:
        print(f"frames still reversed within a recording: median {np.median(fr):.1%}, "
              f"recordings with >10%: {sum(x > 0.10 for x in fr)}")

    with open(out / "keraal_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0]))
        w.writeheader()
        w.writerows(labels)
    with open(out / "keraal_features.csv", "w", newline="", encoding="utf-8") as f:
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

    with open(out / "keraal_error_auc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exercise", "error", "feature", "auc", "n_error", "n_correct",
                                          "subjects_error"])
        w.writeheader()
        w.writerows(auc_rows)
    print(f"\nwrote {out/'keraal_labels.csv'}, {out/'keraal_features.csv'}, {out/'keraal_error_auc.csv'}")


if __name__ == "__main__":
    main()
