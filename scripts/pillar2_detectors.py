"""Pillar 2, step 2: one simple detector per clinician-labelled error, tested
the honest way (leave one subject out) and compared with clinician-vs-clinician
agreement.

Reads outputs/keraal_features.csv (written by pillar2_error_signals.py), so it
runs in seconds and needs no dataset files.

Detector = one feature + one threshold + a direction ("below" = the error is
flagged when the feature is under the threshold). The threshold is fitted on
all subjects except one (best balanced accuracy between "annotator A marked
this error" and "both clinicians said Correct"), then applied to every
recording of the held-out subject. Pooled over all subjects this gives
sensitivity / specificity that were never fitted on the person being judged.

Then, on the same recordings, three agreements (Cohen's kappa, error flagged
or not):
    A vs B          how much the two clinicians agree on this error  (ceiling)
    system vs A     system vs B
If system-vs-clinician is close to A-vs-B, the detector is as consistent as a
second clinician would be.

Also printed: whether the feature orders severity (Correct -> SmallError ->
LargeError), which is what lets feedback be graded rather than yes/no.

Writes
  outputs/pillar2_detector_results.csv
  config/findings_thresholds.json   thresholds fitted on ALL subjects, for the findings engine

Example
  python scripts\\pillar2_detectors.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# A detector feeds the findings engine only if it held up on people it was not fitted on
MIN_BALANCED = 0.75   # leave-one-subject-out balanced accuracy
MIN_PEOPLE = 4        # different people with the error
MAX_SENS_LOSS = 0.10     # borderline band may cost at most 10 points of sensitivity (vs annotator A, in-sample)
MIN_BAND = 5             # ... and must hold at least 5 rated recordings to say anything
# Exercise envelope: what a session must look like to be judged as this exercise at all. One-sided on purpose:
# healthy people move MORE than Keraal's patients, so only movements that leave the exercise are excluded
# (CTK = hands in front of the face; arms far overhead is a different exercise).
ENVELOPE = {"CTK": [("arm_peak_mean", "max"), ("wrist_height_p95", "max")]}

# Error meanings: Keraal's descriptions, matched to the 18 Nov signal ranking
# (body part + which feature separates the error). 'reportable' = the feature
# is a validated measure that may be shown to the user as a number
# (trust table); proxies may help detection but are never quoted.
DETECTORS = [
    {"id": "CTK_arms_not_raised", "exercise": "CTK", "error": "Error1",
     "meaning": "arms not raised enough", "body_part": "BothArms",
     "candidates": [("wrist_height_p95", "below", True, "shoulder widths above the shoulders"),
                    ("arm_peak_mean", "below", True, "deg arm elevation"),
                    ("elbow_mean_p50", "above", False, "deg image-plane elbow bend (proxy)")]},
    {"id": "CTK_error3", "exercise": "CTK", "error": "Error3",
     "meaning": "to confirm (both arms; elbow-bend signal)", "body_part": "BothArms",
     "candidates": [("elbow_reach_p95", "above", False, "deg image-plane elbow bend (proxy)"),
                    ("elbow_spread_p95", "below", True, "shoulder widths between elbows"),
                    ("wrist_spread_p95", "below", True, "shoulder widths between wrists")]},
    {"id": "ELK_body_not_tilted", "exercise": "ELK", "error": "Error2",
     "meaning": "body not tilted", "body_part": "Spine",
     "candidates": [("lean_range", "below", True, "deg trunk lean range"),
                    ("lean_abs_p95", "below", True, "deg trunk lean from start"),
                    ("lean_abs_rep_min", "below", True, "deg trunk lean, smallest repetition")]},
    {"id": "RTK_arms_not_raised", "exercise": "RTK", "error": "Error1",
     "meaning": "arms not raised enough", "body_part": "BothArms",
     "candidates": [("arm_peak_mean", "below", True, "deg arm elevation"),
                    ("wrist_height_p95", "below", True, "shoulder widths above the shoulders")]},
    {"id": "RTK_torso_not_rotated", "exercise": "RTK", "error": "Error2",
     "meaning": "torso rotation not sufficient (Spine)", "body_part": "Spine",
     "candidates": [("rotation_range", "below", False, "BlazePose depth rotation (timing only)"),
                    ("lean_range", "below", True, "deg trunk lean range")]},
]


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    if len(a) == 0:
        return float("nan")
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def flag(x: np.ndarray, thr: float, direction: str) -> np.ndarray:
    return (x < thr) if direction == "below" else (x > thr)


def fit_threshold(pos: np.ndarray, neg: np.ndarray, direction: str) -> float:
    """Threshold with the best balanced accuracy (midpoints between sorted values)."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.unique(np.concatenate([pos, neg]))
    cands = np.concatenate([[v[0] - 1e-6], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-6]])
    best, best_t = -1.0, float("nan")
    for t in cands:
        ba = (flag(pos, t, direction).mean() + (~flag(neg, t, direction)).mean()) / 2
        if ba > best:
            best, best_t = ba, float(t)
    return best_t


def as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def borderline_band(rows: list[dict], cfg: dict) -> float:
    """Choose a 'borderline' margin just past the threshold and print the trade-off.

    Inside the band a recording is not corrected (clinician view only). The band is
    the zone just past the threshold where the two clinicians disagree MOST often:
    among margins that cost at most MAX_SENS_LOSS of sensitivity and hold at least
    MIN_BAND rated recordings, the one with the highest in-band disagreement (ties ->
    the smaller margin). No such margin -> no band. Capped at half the distance to
    the severity cut, so 'large' errors are never hidden. In-sample: a strictness
    setting, not a validated accuracy figure. (20 Nov: the earlier rule, "smallest
    margin with <= 10% false alarms", ran to the cap on CTK and hid half the errors.)"""
    ex, feat, d, thr = cfg["exercise"], cfg["feature"], cfg["direction"], cfg["threshold"]
    rr = [r for r in rows if r["exercise"] == ex]
    x = np.array([as_float(r[feat]) for r in rr])
    pos = np.array([r["eval_A"] == "Incorrect" and r["type_A"] == cfg["keraal_error"] for r in rr])
    both_ok = np.array([r["consensus"] == "Correct" for r in rr])
    rated = np.array([r["eval_A"] in ("Correct", "Incorrect") and r["eval_B"] in ("Correct", "Incorrect") for r in rr])
    disagree = np.array([r["eval_A"] != r["eval_B"] for r in rr])
    sign = -1 if d == "below" else 1
    cut = cfg.get("severity_cut")
    typical = cfg.get("typical_correct")
    ref = cut if cut is not None else typical
    cap = abs(thr - ref) / 2 if ref is not None else 0.0
    steps = np.linspace(0, cap, 11) if cap > 0 else np.array([0.0])
    print(f"  borderline sweep for {cfg['exercise']} {feat} (threshold {thr:g}, flag {d}):")
    print(f"    {'margin':>7} {'sens vs A':>9} {'false alarms':>12} {'in band':>7} {'clinicians disagree in band':>27}")
    rows_ = []
    for m in steps:
        hard = thr + sign * m
        flagged = flag(x, hard, d)
        band = flag(x, thr, d) & ~flagged
        sens = flagged[pos].mean() if pos.any() else float("nan")
        fa = flagged[both_ok].mean() if both_ok.any() else float("nan")
        n_rated = int((band & rated).sum())
        dis = disagree[band & rated].mean() if n_rated else float("nan")
        rows_.append((float(m), sens, fa, int(band.sum()), n_rated, dis))
        print(f"    {m:7.3f} {sens:9.0%} {fa:12.0%} {band.sum():7d} {dis:27.0%}")
    sens0 = rows_[0][1]
    ok = [r for r in rows_[1:] if np.isfinite(r[5]) and r[4] >= MIN_BAND and sens0 - r[1] <= MAX_SENS_LOSS + 1e-9]
    chosen = 0.0
    if ok:
        best = max(r[5] for r in ok)
        chosen = min(r[0] for r in ok if r[5] >= best - 1e-9)
    out_band = flag(x, thr + sign * chosen, d) | ~flag(x, thr, d)
    base = disagree[rated & out_band].mean() if (rated & out_band).any() else float("nan")
    print(f"    -> margin {chosen:.3f}; clinicians disagree on {base:.0%} of recordings outside the band")
    return round(chosen, 3)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="outputs/keraal_features.csv")
    ap.add_argument("--out", default="outputs/pillar2_detector_results.csv")
    ap.add_argument("--thresholds", default=str(ROOT / "config" / "findings_thresholds.json"))
    args = ap.parse_args()

    with open(args.features, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["person"] = f"{r['group']}-{r['subject']}"      # G1A-R1 and G2A-S1 are different people

    # left/right flicker by exercise (from the side-fix diagnostics)
    print("=== frames whose shoulder order is reversed inside a recording, by exercise")
    for ex in ("CTK", "ELK", "RTK"):
        fr = np.array([as_float(r["frames_reversed"]) for r in rows if r["exercise"] == ex])
        fr = fr[np.isfinite(fr)]
        if fr.size:
            print(f"  {ex}: median {np.median(fr):.1%}, >10% in {np.sum(fr > 0.10)}/{fr.size} recordings, "
                  f"max {fr.max():.0%}")

    results, thresholds = [], {}
    print("\n=== detectors (leave one subject out; pooled over subjects)")
    for det in DETECTORS:
        ex, err = det["exercise"], det["error"]
        rr = [r for r in rows if r["exercise"] == ex]
        if not rr:
            print(f"\n{det['id']}: no {ex} recordings")
            continue
        is_pos = np.array([r["eval_A"] == "Incorrect" and r["type_A"] == err for r in rr])
        is_neg = np.array([r["consensus"] == "Correct" for r in rr])
        both_rated = np.array([r["eval_A"] in ("Correct", "Incorrect") and r["eval_B"] in ("Correct", "Incorrect")
                               for r in rr])
        flag_A = is_pos
        flag_B = np.array([r["eval_B"] == "Incorrect" and r["type_B"] == err for r in rr])
        people = np.array([r["person"] for r in rr])
        sev = np.array([r["severity_A"] or "" for r in rr])
        print(f"\n{det['id']}  ({ex} {err}: {det['meaning']})  n error={is_pos.sum()} "
              f"({len(set(people[is_pos]))} people), n correct={is_neg.sum()}")
        if is_pos.sum() < 4 or is_neg.sum() < 4:
            print("  too few recordings to test")
            continue

        best_for_config = None
        cand = {}
        for feat, direction, reportable, unit in det["candidates"]:
            if feat not in rr[0]:          # features file older than this candidate
                print(f"  {feat:17s} not in the features file (rerun pillar2_error_signals.py) -> skipped")
                continue
            x = np.array([as_float(r[feat]) for r in rr])
            pred = np.zeros(len(rr), bool)
            for p in np.unique(people):
                train, test = people != p, people == p
                t = fit_threshold(x[train & is_pos], x[train & is_neg], direction)
                pred[test] = flag(x[test], t, direction) if np.isfinite(t) else False
            sens = pred[is_pos].mean()
            spec = (~pred[is_neg]).mean()
            k_ab = kappa(flag_A[both_rated], flag_B[both_rated])
            k_sa = kappa(pred[both_rated], flag_A[both_rated])
            k_sb = kappa(pred[both_rated], flag_B[both_rated])
            # severity ordering on the full-data feature values
            med = {s: np.nanmedian(x[m]) if m.any() else float("nan")
                   for s, m in (("Correct", is_neg), ("SmallError", is_pos & (sev == "SmallError")),
                                ("LargeError", is_pos & (sev == "LargeError")))}
            vals = [med["Correct"], med["SmallError"], med["LargeError"]]
            if all(np.isfinite(vals)):
                mono = (vals[0] >= vals[1] >= vals[2]) if direction == "below" else (vals[0] <= vals[1] <= vals[2])
            else:
                mono = None
            t_all = fit_threshold(x[is_pos], x[is_neg], direction)
            tag = "" if reportable else "  [proxy: detect only, never quote]"
            print(f"  {feat:<17} {direction:<5} thr(all) {t_all:8.2f}  LOSO sens {sens:.0%} spec {spec:.0%} "
                  f"bal {(sens + spec) / 2:.0%} | kappa A-B {k_ab:.2f}  sys-A {k_sa:.2f}  sys-B {k_sb:.2f}{tag}")
            print(f"  {'':<17} medians: correct {vals[0]:.2f}, small {vals[1]:.2f}, large {vals[2]:.2f}"
                  f" -> severity ordered: {mono}")
            results.append({"detector": det["id"], "exercise": ex, "error": err, "feature": feat,
                            "direction": direction, "reportable": reportable, "threshold_all": round(t_all, 3),
                            "loso_sensitivity": round(sens, 3), "loso_specificity": round(spec, 3),
                            "loso_balanced": round((sens + spec) / 2, 3), "kappa_A_B": round(k_ab, 3),
                            "kappa_sys_A": round(k_sa, 3), "kappa_sys_B": round(k_sb, 3),
                            "n_error": int(is_pos.sum()), "n_correct": int(is_neg.sum()),
                            "people_error": len(set(people[is_pos])), "severity_ordered": mono,
                            "median_correct": vals[0], "median_small": vals[1], "median_large": vals[2]})
            cand[feat] = {"pred": pred.copy(), "x": x, "direction": direction, "reportable": reportable,
                          "bal": (sens + spec) / 2, "median_correct": vals[0], "unit": unit}
            score = (sens + spec) / 2 + (0.02 if reportable else 0.0)   # prefer a quotable measure on near-ties
            if best_for_config is None or score > best_for_config[0]:
                best_for_config = (score, feat, direction, reportable, unit, t_all, sens, spec, k_ab, k_sa, k_sb, vals, mono)
        if best_for_config:
            _, feat, direction, reportable, unit, t_all, sens, spec, k_ab, k_sa, k_sb, vals, mono = best_for_config
            # severity: between the Small and Large medians, only if the feature orders severity
            cut = round((vals[1] + vals[2]) / 2, 3) if mono else None
            # A proxy (e.g. image elbow bend for "arms not raised") detects a correlate, not the error itself,
            # and can misfire on other movements (21 Nov: arms overhead with bent elbows -> "raise your arms").
            # Guard it with the best direct measure: no error when the direct measure is at or past what
            # correct repetitions typically reach. The guarded detector is what the engine runs.
            guard = None
            if not reportable:
                direct = [(c["bal"], f) for f, c in cand.items() if c["reportable"] and np.isfinite(c["median_correct"])]
                if direct:
                    gf = max(direct)[1]
                    g = cand[gf]
                    clearly_ok = (g["x"] >= g["median_correct"]) if g["direction"] == "below" \
                        else (g["x"] <= g["median_correct"])
                    clearly_ok &= np.isfinite(g["x"])
                    pg = cand[feat]["pred"] & ~clearly_ok
                    sens, spec = pg[is_pos].mean(), (~pg[is_neg]).mean()
                    k_sa, k_sb = kappa(pg[both_rated], flag_A[both_rated]), kappa(pg[both_rated], flag_B[both_rated])
                    guard = {"feature": gf, "direction": g["direction"], "ok_limit": round(float(g["median_correct"]), 3),
                             "unit": g["unit"]}
                    print(f"  guard: {feat} flags only while {gf} has not reached the typical correct value "
                          f"{g['median_correct']:.2f} -> LOSO sens {sens:.0%} spec {spec:.0%} bal {(sens + spec) / 2:.0%} "
                          f"| kappa sys-A {k_sa:.2f} sys-B {k_sb:.2f}")
            people_err = len(set(people[is_pos]))
            usable = bool((sens + spec) / 2 >= MIN_BALANCED and people_err >= MIN_PEOPLE)
            why = "ok" if usable else (f"only {people_err} people with this error" if people_err < MIN_PEOPLE
                                       else f"leave-one-subject-out balanced accuracy {(sens + spec) / 2:.0%} < "
                                            f"{MIN_BALANCED:.0%}")
            print(f"  -> config: {feat} ({'USED' if usable else 'NOT USED: ' + why})")
            thresholds[det["id"]] = {
                "usable": usable, "usable_reason": why,
                "exercise": ex, "keraal_error": err, "meaning": det["meaning"], "body_part": det["body_part"],
                "feature": feat, "direction": direction, "threshold": round(t_all, 3), "unit": unit,
                "reportable": reportable, "guard": guard,
                "typical_correct": round(float(vals[0]), 3) if np.isfinite(vals[0]) else None,
                "severity_cut": cut,
                "evidence": {"loso_sensitivity": round(sens, 3), "loso_specificity": round(spec, 3),
                             "kappa_clinicians": round(k_ab, 3), "kappa_system_A": round(k_sa, 3),
                             "kappa_system_B": round(k_sb, 3), "n_error": int(is_pos.sum()),
                             "n_correct": int(is_neg.sum()), "people_error": people_err},
            }

    print("\n=== exercise envelopes (movements far outside every Keraal recording of the exercise are not judged)")
    envelopes = {}
    for ex, specs in ENVELOPE.items():
        xs_ex = [r for r in rows if r["exercise"] == ex]
        for feat, side in specs:
            x = np.array([as_float(r[feat]) for r in xs_ex])
            x = x[np.isfinite(x)]
            if len(x) < 20:
                continue
            # p5/p95 + a quarter of their spread: with ~90 recordings p1/p99 are single recordings, and on
            # MediaPipe those can be pose failures (21 Nov: CTK limit landed at 301 deg of 'arm elevation')
            lo, hi = np.percentile(x, 5), np.percentile(x, 95)
            pad = 0.25 * (hi - lo)
            limit = hi + pad if side == "max" else lo - pad
            outside = int(np.sum(x > limit) if side == "max" else np.sum(x < limit))
            envelopes.setdefault(ex, []).append({"feature": feat, "side": side, "limit": round(float(limit), 3)})
            print(f"  {ex} {feat}: {side} {limit:.2f}  (Keraal p5 {lo:.2f}, p95 {hi:.2f}, n={len(x)}; "
                  f"{outside} Keraal recordings would be called 'not this exercise')")

    print("\n=== borderline band (no patient correction just past the threshold)")
    for det_id, cfg in thresholds.items():
        cfg["borderline_margin"] = borderline_band(rows, cfg) if cfg["usable"] else 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    Path(args.thresholds).parent.mkdir(parents=True, exist_ok=True)
    with open(args.thresholds, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "date": date.today().isoformat(), "envelopes": envelopes,
                   "source": "Keraal 1A + 2A, annotator A error labels vs both-Correct; thresholds fitted on all "
                             "subjects, evidence from leave-one-subject-out",
                   "detectors": thresholds}, f, indent=2)
    print(f"\nwrote {args.out} and {args.thresholds}")


if __name__ == "__main__":
    main()
