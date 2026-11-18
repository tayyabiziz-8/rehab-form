"""Pillar 2, step 3: findings -> feedback -> claim check, evaluated on Keraal 1A + 2A.

Per recording it runs the findings engine (thresholds from
config/findings_thresholds.json) and the feedback writers, then reports

  Findings vs clinicians (in-sample: thresholds were fitted on these
  recordings; the leave-one-subject-out numbers are in pillar2_detectors.py)
    * flagged vs annotator A / B, per detector
    * time window: does our window overlap A's 'Temporal error' window for that
      error (and the overlap ratio, IoU)?
    * severity: our small/large vs A's SmallError/LargeError
    * side: for A's LeftArm/RightArm labels, do we name the same side?

  Feedback
    * template      always on, must pass the checker 100% (sanity check)
    * grounded LLM  (--llm) rewords findings; checker pass rate BEFORE fallback
                    = how often an LLM invents something; after fallback = 0
    * baseline LLM  (--llm --baseline) sees only the pose sequence; it cannot be
                    claim-checked (no findings), so it is compared on WHAT it
                    asks the person to correct vs the clinicians' labels
                    (approximate keyword categories; spot-check by hand)

Writes outputs/pillar2_feedback_per_recording.csv and
outputs/pillar2_feedback_examples.md (examples for the report / clinician rating).

Examples (cmd, repo root, .venv active)
  python scripts\\pillar2_feedback_eval.py data\\keraal1 data\\keraal2
  python scripts\\pillar2_feedback_eval.py data\\keraal1 data\\keraal2 --llm ^
      --base-url http://localhost:11434/v1 --model qwen2.5:7b --baseline --limit 60
  (hosted: --base-url https://api.groq.com/openai/v1 --model <model> --api-key-env GROQ_API_KEY)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.claims import DETECTOR_CATEGORY, check, error_categories  # noqa: E402
from rehabform.feedback import LLMClient, baseline_prompt, strip_tags, template_feedback, write_feedback  # noqa: E402
from rehabform.findings import DEFAULT_THRESHOLDS, FindingsEngine  # noqa: E402
from rehabform.keraal_groups import annotation_labels, iter_annotated_recordings, load_blazepose_g12  # noqa: E402


def overlap(a: tuple[float, float], b: tuple[float, float], slack: float = 0.5) -> tuple[bool, float]:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    hit = (min(a[1], b[1]) + slack) >= max(a[0], b[0])
    return hit, (inter / union if union > 0 else 0.0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    ap.add_argument("--llm", action="store_true", help="also run the grounded language-model writer")
    ap.add_argument("--baseline", action="store_true", help="also run the pose-only baseline (needs --llm)")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env")
    ap.add_argument("--limit", type=int, default=0, help="LLM runs on a random sample of this many recordings")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not Path(args.thresholds).exists():
        raise SystemExit(f"{args.thresholds} not found: run scripts/pillar2_detectors.py first")
    engine = FindingsEngine(args.thresholds)
    llm = None
    if args.llm:
        if not (args.base_url and args.model):
            raise SystemExit("--llm needs --base-url and --model")
        llm = LLMClient(args.base_url, args.model, args.api_key_env, cache_path=out / "llm_cache.jsonl")

    recs = [r for root in args.roots for r in iter_annotated_recordings(root) if r.blazepose]
    llm_keys = {r.key for r in recs}
    if llm and args.limit:
        llm_keys = set(random.Random(0).sample(sorted(llm_keys), min(args.limit, len(llm_keys))))

    rows, examples = [], []
    det_stats = defaultdict(Counter)
    win_hits, win_iou, sev_pairs, side_pairs = defaultdict(list), defaultdict(list), Counter(), Counter()
    fb = Counter()
    reasons = Counter()
    cat_stats = {m: Counter() for m in ("template", "grounded", "baseline")}

    for i, r in enumerate(recs, 1):
        try:
            bp, _info = load_blazepose_g12(r.blazepose)
        except Exception as exc:
            print(f"[skip] {r.key}: {exc}")
            continue
        lab = {k: annotation_labels(p) for k, p in r.annotations.items()}
        A, B = lab.get("A", {}), lab.get("B", {})
        findings = engine.run(bp, r.exercise)
        a_types = {e["type"] for e in A.get("errors", [])} if A.get("evaluation") == "Incorrect" else set()
        b_types = {e["type"] for e in B.get("errors", [])} if B.get("evaluation") == "Incorrect" else set()
        a_correct, b_correct = A.get("evaluation") == "Correct", B.get("evaluation") == "Correct"

        for f in findings:
            err = engine.detectors[f.detector]["keraal_error"]
            flagged = f.status == "error"
            for who, types, correct in (("A", a_types, a_correct), ("B", b_types, b_correct)):
                if err in types:
                    det_stats[(f.detector, who)]["tp" if flagged else "fn"] += 1
                elif correct:
                    det_stats[(f.detector, who)]["fp" if flagged else "tn"] += 1
            if flagged and err in a_types:
                wins = [w for w in A.get("windows", []) if w["type"] == err and w["start"] is not None]
                if wins and f.start_s is not None:
                    best = max((overlap((f.start_s, f.end_s), (w["start"], w["end"])) for w in wins),
                               key=lambda x: (x[0], x[1]))
                    win_hits[f.detector].append(best[0])
                    win_iou[f.detector].append(best[1])
                a_err = next(e for e in A["errors"] if e["type"] == err)
                if f.severity and a_err.get("severity"):
                    sev_pairs[(f.severity, a_err["severity"])] += 1
                if a_err.get("body_part") in ("LeftArm", "RightArm"):
                    side_pairs[(f.side or "none", a_err["body_part"])] += 1

        # feedback
        tmpl = template_feedback(findings)
        t_rep = check(tmpl, findings)
        fb["template_total"] += 1
        fb["template_pass"] += t_rep.ok
        labelled_cats = {DETECTOR_CATEGORY.get(d) for d, det in engine.detectors.items()
                         if det["exercise"] == r.exercise and det["keraal_error"] in (a_types | b_types)}
        any_correct_label = a_correct or b_correct
        row = {"key": r.key, "exercise": r.exercise, "eval_A": A.get("evaluation"), "eval_B": B.get("evaluation"),
               "types_A": "|".join(sorted(a_types)), "types_B": "|".join(sorted(b_types)),
               "findings": json.dumps([{k: f.to_dict()[k] for k in ("id", "detector", "status", "value", "typical",
                                                                    "severity", "start_s", "end_s")}
                                       for f in findings]),
               "template": strip_tags(tmpl), "template_ok": t_rep.ok}

        def score(mode: str, text: str) -> None:
            cats = error_categories(text)
            st = cat_stats[mode]
            st["n"] += 1
            st["claims"] += len(cats)
            st["claims_on_correct"] += len(cats) if (a_correct and b_correct) else 0
            st["both_correct"] += a_correct and b_correct
            st["labelled_errors"] += len(labelled_cats - {None})
            st["hits"] += len(cats & labelled_cats)

        score("template", tmpl)
        if llm and r.key in llm_keys:
            try:
                res = write_feedback(findings, r.exercise, llm)
                fb["llm_total"] += 1
                fb["llm_draft_pass"] += res.source == "llm"
                fb["llm_draft_sentences"] += len(res.draft_report.sentences)
                fb["llm_draft_unsupported"] += res.draft_report.n_unsupported
                reasons.update(x.split(":")[0] for x in res.draft_report.reasons())
                fb["final_pass"] += res.report.ok
                row.update(grounded_draft=res.draft, grounded_draft_ok=res.draft_report.ok,
                           grounded_draft_reasons="|".join(res.draft_report.reasons()),
                           grounded_final=strip_tags(res.text), grounded_source=res.source)
                score("grounded", res.draft)
                if args.baseline:
                    base = llm(baseline_prompt(bp, r.exercise))
                    row["baseline"] = base
                    score("baseline", base)
            except Exception as exc:
                print(f"[llm error] {r.key}: {exc}")
                fb["llm_errors"] += 1
        rows.append(row)
        if len(examples) < args.examples and ("grounded_draft" in row or not llm):
            examples.append((r, A, B, findings, row))
        if i % 50 == 0:
            print(f"processed {i}/{len(recs)}")

    # ----------------------------------------------------------------- report
    print("\n=== detection vs clinicians (in-sample; LOSO numbers: pillar2_detectors.py)")
    for (det, who), c in sorted(det_stats.items()):
        tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        print(f"  {det:<24} vs {who}: sens {sens:.0%} spec {spec:.0%} precision {prec:.0%}  (tp {tp} fn {fn} fp {fp} tn {tn})")

    print("\n=== time window vs annotator A's 'Temporal error' (true positives with a window)")
    for det in sorted(win_hits):
        h, iou = win_hits[det], win_iou[det]
        print(f"  {det:<24} overlap {np.mean(h):.0%} of {len(h)}, median IoU {np.median(iou):.2f}")
    if sev_pairs:
        print("\n=== severity: ours vs A")
        for (ours, theirs), n in sorted(sev_pairs.items()):
            print(f"  ours {ours:<6} A {theirs:<11} {n}")
        agree = sum(n for (o, t), n in sev_pairs.items() if (o, t) in (("small", "SmallError"), ("large", "LargeError")))
        print(f"  agreement {agree}/{sum(sev_pairs.values())}")
    if side_pairs:
        print("\n=== side: ours vs A's one-arm labels")
        for (ours, theirs), n in sorted(side_pairs.items()):
            print(f"  ours {ours:<6} A {theirs:<9} {n}")

    print("\n=== feedback")
    print(f"  template: passes the claim checker {fb['template_pass']}/{fb['template_total']} (must be all)")
    if fb["llm_total"]:
        print(f"  grounded LLM drafts: {fb['llm_draft_pass']}/{fb['llm_total']} pass the checker; "
              f"unsupported sentences {fb['llm_draft_unsupported']}/{fb['llm_draft_sentences']} "
              f"({fb['llm_draft_unsupported'] / max(fb['llm_draft_sentences'], 1):.1%})")
        print(f"  rejection reasons: {dict(reasons.most_common())}")
        print(f"  after checker + template fallback: {fb['final_pass']}/{fb['llm_total']} pass "
              f"(unsupported claims shown to the patient: 0)")
    if fb["llm_errors"]:
        print(f"  LLM call errors: {fb['llm_errors']}")
    print("\n=== what each writer asks the patient to correct vs clinician labels (approximate keyword categories)")
    for mode, st in cat_stats.items():
        if not st["n"]:
            continue
        print(f"  {mode:<9} n={st['n']:>3}: corrections {st['claims']}, of which on recordings BOTH called Correct "
              f"{st['claims_on_correct']} ({st['both_correct']} such recordings); "
              f"labelled errors addressed {st['hits']}/{st['labelled_errors']}")

    with open(out / "pillar2_feedback_per_recording.csv", "w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = ["# Pillar 2 feedback examples", ""]
    for r, A, B, findings, row in examples:
        md += [f"## {r.key} ({r.exercise})", "",
               f"Clinician A: {A.get('evaluation')} {A.get('errors') or ''}  ",
               f"Clinician B: {B.get('evaluation')} {B.get('errors') or ''}", "",
               "Findings:", ""]
        for fd in findings:
            val = f"{fd.value:g}{fd.unit if fd.unit == '°' else ' ' + fd.unit} (typical {fd.typical:g})" \
                if fd.quotable and fd.value is not None and fd.typical is not None else "not quoted"
            md.append(f"- {fd.id} {fd.detector}: **{fd.status}** {fd.severity or ''}, {val}, "
                      f"window {fd.start_s}–{fd.end_s} s")
        md += ["", f"**Template:** {row['template']}", ""]
        if "grounded_draft" in row:
            md += [f"**Grounded LLM draft** (checker: {'pass' if row['grounded_draft_ok'] else 'REJECTED ' + row['grounded_draft_reasons']}): "
                   f"{row['grounded_draft']}", "", f"**Shown to patient:** {row['grounded_final']}", ""]
        if "baseline" in row:
            md += [f"**Baseline (pose only):** {row['baseline']}", ""]
    (out / "pillar2_feedback_examples.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {out/'pillar2_feedback_per_recording.csv'} and {out/'pillar2_feedback_examples.md'}")


if __name__ == "__main__":
    main()
