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
from rehabform.feedback import (  # noqa: E402
    LLMClient, RateLimited, baseline_prompt, load_env_file, strip_tags, template_feedback, write_feedback,
)
from rehabform.session_features import FPS  # noqa: E402
from rehabform.findings import DEFAULT_THRESHOLDS, FindingsEngine  # noqa: E402
from rehabform.keraal_groups import annotation_labels, iter_annotated_recordings, load_blazepose_g12  # noqa: E402


def overlap(a: tuple[float, float], b: tuple[float, float], slack: float = 0.5) -> tuple[bool, float]:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    hit = (min(a[1], b[1]) + slack) >= max(a[0], b[0])
    return hit, (inter / union if union > 0 else 0.0)


ORIG_FPS_RANGE = (12.0, 30.0)   # plausible original video rates


def estimate_clocks(recs) -> tuple[dict, float, int]:
    """Original frame rate per recording, for mapping clinician times (Anvil seconds, original clock)
    onto the shipped mp4 (same frames played at 10 fps).

    A recording's own estimate = frames / its global-evaluation span, used only when that span is a
    plausible whole-video span: a rate in ORIG_FPS_RANGE and long enough to contain A's windows.
    Others get the median of the plausible ones, capped so A's last window still fits in the video.
    (21 Nov: many spans are 1-9 s for 30-50 s videos, i.e. not the whole video.)"""
    import json as _json
    info, plaus = {}, []
    for r in recs:
        try:
            with open(r.blazepose, encoding="utf-8") as fh:
                n = len(_json.load(fh)["positions"])
        except Exception:
            continue
        a = annotation_labels(r.annotations["A"]) if "A" in r.annotations else {}
        span = (a.get("span") or (None, None))[1]
        ends = [w["end"] for w in a.get("windows", []) if w.get("end") is not None]
        last = max(ends) if ends else 0.0
        fps = n / span if span else float("nan")
        ok = bool(np.isfinite(fps) and ORIG_FPS_RANGE[0] <= fps <= ORIG_FPS_RANGE[1] and span >= last)
        info[r.key] = {"frames": n, "fps": fps if ok else None, "cap": n / last if last > 0 else float("inf")}
        if ok:
            plaus.append(fps)
    typical = float(np.median(plaus)) if plaus else 15.0
    for v in info.values():
        if v["fps"] is None:
            v["fps"] = min(typical, v["cap"])
            v["fallback"] = True
    return info, typical, len(plaus)


def chance_overlap(length: float, total: float, windows: list[tuple[float, float]], slack: float = 0.5) -> float:
    """Probability that a window of this length, placed anywhere, overlaps one of the windows."""
    if total <= length:
        return 1.0
    starts = np.linspace(0, total - length, 200)
    return float(np.mean([any(overlap((s, s + length), w, slack)[0] for w in windows) for s in starts]))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # progress visible in a redirected file
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    ap.add_argument("--llm", action="store_true", help="also run the grounded language-model writer")
    ap.add_argument("--baseline", action="store_true", help="also run the pose-only baseline (needs --llm)")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key-env", help="name of the environment variable (or .env entry) holding the key")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--prompt-version", type=int, choices=[1, 2], default=2,
                    help="1 = the 20 Nov prompts (correction first); 2 = no correction when every check passes (both writers)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="answer budget per call (reasoning included)")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="for reasoning models such as gpt-oss (low = fewer tokens, fewer empty replies)")
    ap.add_argument("--list-models", action="store_true", help="print the endpoint's models and stop")
    ap.add_argument("--limit", type=int, default=0, help="LLM runs on a random sample of this many recordings")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    load_env_file(args.env_file)
    if args.list_models:
        if not args.base_url:
            raise SystemExit("--list-models needs --base-url")
        for m in LLMClient(args.base_url, "none", args.api_key_env, cache_path=out / "llm_cache.jsonl").list_models():
            print(m)
        return
    if not Path(args.thresholds).exists():
        raise SystemExit(f"{args.thresholds} not found: run scripts/pillar2_detectors.py first")
    engine = FindingsEngine(args.thresholds)
    all_detectors = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["detectors"]  # incl. unused
    print("detectors used:", sorted(engine.detectors), "| not used:", sorted(set(all_detectors) - set(engine.detectors)))
    llm = None
    if args.llm:
        if not (args.base_url and args.model):
            raise SystemExit("--llm needs --base-url and --model")
        llm = LLMClient(args.base_url, args.model, args.api_key_env, cache_path=out / "llm_cache.jsonl",
                        reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)

    recs = [r for root in args.roots for r in iter_annotated_recordings(root) if r.blazepose]
    clocks, clock_typical, n_plausible = estimate_clocks(recs)
    print(f"clinician clock: {n_plausible} recordings have a plausible whole-video span; "
          f"their median original rate {clock_typical:.1f} fps is used for the others")
    llm_keys = {r.key for r in recs}
    if llm and args.limit:
        if args.prompt_version == 1:              # exactly the 20 Nov sample
            llm_keys = set(random.Random(0).sample(sorted(llm_keys), min(args.limit, len(llm_keys))))
        else:
            # only exercises with validated checks (both writers have something to say), and a fixed shuffled
            # order so --limit 30 is a subset of --limit 60 and finished calls are reused
            checked = {d["exercise"] for d in engine.detectors.values()}
            pool = sorted(r.key for r in recs if r.exercise in checked)
            llm_keys = set(random.Random(0).sample(pool, len(pool))[:args.limit])

    rows, examples = [], []
    det_stats = defaultdict(Counter)
    win_hits, win_iou, sev_pairs, side_pairs = defaultdict(list), defaultdict(list), Counter(), Counter()
    win_chance, implied_fps = defaultdict(list), []
    win_source, win_anyrep, borderline = defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    win_debug = []
    t_reasons = Counter()
    fb = Counter()
    reasons = Counter()
    per_mode: dict[str, dict] = {m: {} for m in ("template", "grounded", "baseline")}

    for i, r in enumerate(recs, 1):
        try:
            bp, _info = load_blazepose_g12(r.blazepose)
        except Exception as exc:
            print(f"[skip] {r.key}: {exc}")
            continue
        lab = {k: annotation_labels(p) for k, p in r.annotations.items()}
        A, B = lab.get("A", {}), lab.get("B", {})
        findings = engine.run(bp, r.exercise)
        # Anvil times are on the ORIGINAL video's clock; the shipped mp4 has the same frames played at 10 fps.
        # Map Anvil seconds -> our seconds with the global-evaluation span (= whole recording).
        # Some global-evaluation spans do not cover the recording (implied rates of 0.2 or 900 fps were seen);
        # then use the typical original rate. 19.3 fps median matches the 19.4 Hz found for Group 3.
        span_end = (A.get("span") or (None, None))[1]
        ck = clocks.get(r.key, {"fps": clock_typical})
        fps_orig = ck["fps"]
        implied_fps.append(fps_orig)
        fb["clock_fallback"] += bool(ck.get("fallback"))
        scale = fps_orig / FPS          # Anvil seconds x original fps = frame index; / 10 fps = mp4 seconds
        a_types = {e["type"] for e in A.get("errors", [])} if A.get("evaluation") == "Incorrect" else set()
        b_types = {e["type"] for e in B.get("errors", [])} if B.get("evaluation") == "Incorrect" else set()
        a_correct, b_correct = A.get("evaluation") == "Correct", B.get("evaluation") == "Correct"

        for f in findings:
            err = engine.detectors[f.detector]["keraal_error"]
            flagged = f.status == "error"
            if f.status == "borderline":
                label = ("both Correct" if a_correct and b_correct else
                         "both marked this error" if err in a_types and err in b_types else
                         "clinicians disagree" if (err in a_types) != (err in b_types) or a_correct != b_correct
                         else "other")
                borderline[f.detector][label] += 1
            for who, types, correct in (("A", a_types, a_correct), ("B", b_types, b_correct)):
                if err in types:
                    det_stats[(f.detector, who)]["tp" if flagged else "fn"] += 1
                elif correct:
                    det_stats[(f.detector, who)]["fp" if flagged else "tn"] += 1
            if flagged and err in a_types:
                raw = [(w["start"], w["end"]) for w in A.get("windows", []) if w["type"] == err and w["start"] is not None]
                wins = [(a * scale, b * scale) for a, b in raw]
                win_debug.append({"key": r.key, "detector": f.detector, "frames": len(bp), "our_duration_s": len(bp) / FPS,
                                  "anvil_span_end_s": span_end, "orig_fps_used": round(fps_orig, 2),
                                  "A_windows_anvil_s": json.dumps(raw), "A_windows_ours_s": json.dumps(
                                      [(round(a, 1), round(b, 1)) for a, b in wins]),
                                  "our_window_s": json.dumps([f.start_s, f.end_s]), "window_source": f.window_source,
                                  "reps_s": json.dumps([(x["start_s"], x["end_s"]) for x in f.reps])})
                if wins and f.start_s is not None:
                    best = max((overlap((f.start_s, f.end_s), w) for w in wins), key=lambda x: (x[0], x[1]))
                    win_hits[f.detector].append(best[0])
                    win_iou[f.detector].append(best[1])
                    win_chance[f.detector].append(chance_overlap(f.end_s - f.start_s, len(bp) / FPS, wins))
                    win_source[f.detector][f.window_source] += 1
                    if f.reps:   # upper bound: does ANY repetition we found overlap A's window?
                        win_anyrep[f.detector].append(any(overlap((r_["start_s"], r_["end_s"]), w)[0]
                                                          for r_ in f.reps for w in wins))
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
        t_reasons.update(t_rep.reasons())
        labelled_cats = {DETECTOR_CATEGORY.get(d) for d, det in all_detectors.items()
                         if det["exercise"] == r.exercise and det["keraal_error"] in (a_types | b_types)}
        any_correct_label = a_correct or b_correct
        row = {"key": r.key, "exercise": r.exercise, "eval_A": A.get("evaluation"), "eval_B": B.get("evaluation"),
               "types_A": "|".join(sorted(a_types)), "types_B": "|".join(sorted(b_types)),
               "findings": json.dumps([{k: f.to_dict()[k] for k in ("id", "detector", "status", "value", "typical",
                                                                    "severity", "start_s", "end_s")}
                                       for f in findings]),
               "template": strip_tags(tmpl), "template_ok": t_rep.ok}

        both_correct = a_correct and b_correct
        labelled = labelled_cats - {None}

        def score(mode: str, text: str) -> None:
            per_mode[mode][r.key] = (error_categories(text), both_correct, labelled)

        score("template", tmpl)
        if llm and r.key in llm_keys:
            try:
                res = write_feedback(findings, r.exercise, llm, prompt_version=args.prompt_version)
                if res.draft is not None:                  # an LLM call was made (exercise has findings)
                    fb["llm_total"] += 1
                    fb["llm_draft_pass"] += res.source == "llm"
                    fb["llm_draft_sentences"] += len(res.draft_report.sentences)
                    fb["llm_draft_unsupported"] += res.draft_report.n_unsupported
                    reasons.update(x.split(":")[0] for x in res.draft_report.reasons())
                    fb["final_pass"] += res.report.ok
                    row.update(grounded_draft=res.draft, grounded_draft_ok=res.draft_report.ok,
                               grounded_draft_reasons="|".join(res.draft_report.reasons()))
                else:
                    fb["llm_no_findings"] += 1
                row.update(grounded_final=strip_tags(res.text), grounded_source=res.source)
                score("grounded", res.text)                # what the patient would see
                if args.baseline:
                    base = llm(baseline_prompt(bp, r.exercise, version=args.prompt_version))
                    row["baseline"] = base
                    score("baseline", base)
            except RateLimited as exc:
                raise SystemExit(f"\n[stopped] {exc}")
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

    if implied_fps:
        print(f"\n=== clinician clock: original rate used = median {np.median(implied_fps):.1f} fps "
              f"(range {np.min(implied_fps):.1f}-{np.max(implied_fps):.1f}); mp4 plays at {FPS:g}. Windows rescaled; "
              f"{fb['clock_fallback']} recordings used the median {clock_typical:.1f} fps (capped so A's windows fit).")
    print("\n=== time window vs annotator A's 'Temporal error' (true positives with a window)")
    for det in sorted(win_hits):
        h, iou = win_hits[det], win_iou[det]
        print(f"  {det:<24} overlap {np.mean(h):.0%} of {len(h)} (a randomly placed window: "
              f"{np.mean(win_chance[det]):.0%}), median IoU {np.median(iou):.2f}; windows from "
              f"{dict(win_source[det])}")
        if win_anyrep[det]:
            print(f"  {'':<24} any detected repetition overlaps A's window: {np.mean(win_anyrep[det]):.0%} "
                  f"(upper bound if we picked the right repetition)")
    if borderline:
        print("\n=== borderline (just past the threshold; clinician view only, not corrected)")
        for det, c in sorted(borderline.items()):
            print(f"  {det:<24} {sum(c.values())} recordings: {dict(c.most_common())}")
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
    if t_reasons:
        print(f"  template failures (bug, please send): {dict(t_reasons.most_common(8))}")
    if fb["llm_total"]:
        print(f"  grounded LLM drafts: {fb['llm_draft_pass']}/{fb['llm_total']} pass the checker; "
              f"unsupported sentences {fb['llm_draft_unsupported']}/{fb['llm_draft_sentences']} "
              f"({fb['llm_draft_unsupported'] / max(fb['llm_draft_sentences'], 1):.1%})")
        print(f"  rejection reasons: {dict(reasons.most_common())}")
        print(f"  ({fb['llm_no_findings']} sampled recordings had no validated checks: no LLM call, "
              f"the patient sees the 'no checks yet' message)")
        print(f"  after checker + template fallback: {fb['final_pass']}/{fb['llm_total']} pass "
              f"(unsupported claims shown to the patient: 0)")
    if fb["llm_errors"]:
        print(f"  LLM call errors: {fb['llm_errors']}")
    print("\n=== what each writer tells the patient to correct vs the clinicians (approximate keyword categories)")
    print("    'false alarms' = recordings BOTH clinicians called Correct that still got a correction")

    def summary(mode: str, keys) -> str:
        recs_ = [per_mode[mode][k] for k in keys]
        bc = [c for c, both, _l in recs_ if both]
        lab = sum(len(l) for _c, _b, l in recs_)
        hit = sum(len(c & l) for c, _b, l in recs_)
        fa = sum(1 for c in bc if c)
        return (f"n={len(recs_):>3}: false alarms {fa}/{len(bc)} ({fa / max(len(bc), 1):.0%}), "
                f"labelled errors addressed {hit}/{lab} ({hit / max(lab, 1):.0%})")

    for mode in ("template", "grounded", "baseline"):
        if per_mode[mode]:
            print(f"  {mode:<9} all: {summary(mode, list(per_mode[mode]))}")
    paired = set(per_mode["template"])
    for mode in ("grounded", "baseline"):
        if per_mode[mode]:
            paired &= set(per_mode[mode])
    if len(paired) < len(per_mode["template"]):
        print(f"  same recordings for every writer ({len(paired)}):")
        for mode in ("template", "grounded", "baseline"):
            if per_mode[mode]:
                print(f"    {mode:<9} {summary(mode, sorted(paired))}")

    with open(out / "pillar2_feedback_per_recording.csv", "w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if win_debug:
        with open(out / "pillar2_windows_debug.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(win_debug[0]))
            w.writeheader()
            w.writerows(win_debug)
        print(f"time-window details: {out / 'pillar2_windows_debug.csv'}")
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
