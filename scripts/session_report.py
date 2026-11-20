"""One exercise video -> checked feedback, as the app will do it.

    video --MediaPipe--> landmarks (10 fps, pixels, sides fixed)
          --findings engine--> findings (validated checks only, trust table)
          --writer--> template text, or a language model draft that must pass the claim checker
          --> one self-contained HTML report (patient text, clinician view, per-check timeline
              with repetitions, key frame with the skeleton)

Examples (cmd, repo root, .venv active)
  python scripts\\session_report.py data\\keraal2\\videos\\G2A-Anon-ELK-S1-Roscoff-060.mp4 --exercise ELK
  python scripts\\session_report.py my_elk.mp4 --exercise ELK --mirrored          (front-camera video)
  python scripts\\session_report.py my_ctk.mp4 --exercise CTK --llm ^
      --base-url https://api.groq.com/openai/v1 --api-key-env API_KEY --model openai/gpt-oss-120b

Recording tips for your own videos: phone on a stand at chest height, 2-3 m away,
whole upper body and hips in frame, facing the camera, plain background, 3-5 repetitions.
Needs models/pose_landmarker_full.task (MediaPipe Pose Landmarker guide).
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.feedback import EXERCISES, LLMClient, load_env_file, strip_tags, write_feedback  # noqa: E402
from rehabform.findings import APP_THRESHOLDS, FindingsEngine  # noqa: E402
from rehabform.session_features import FPS, series  # noqa: E402
from rehabform.video_pose import pose_from_video  # noqa: E402

EDGES = [(11, 13), (13, 15), (11, 23), (12, 14), (14, 16), (12, 24), (11, 12), (23, 24)]
LEFT = {11, 13, 15, 23}


def svg_timeline(x: np.ndarray, reps: list[dict], window: tuple[float, float] | None, title: str,
                 width: int = 720, height: int = 170) -> str:
    """Inline SVG: the per-frame measure, repetitions shaded, the finding's window outlined."""
    t = np.arange(len(x)) / FPS
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return "<p>(no data)</p>"
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    hi = hi if hi > lo else lo + 1
    pad_l, pad_r, pad_t, pad_b = 44, 10, 22, 24
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    X = lambda tt: pad_l + W * tt / max(t[-1], 1e-6)  # noqa: E731
    Y = lambda v: pad_t + H * (1 - (v - lo) / (hi - lo))  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{pad_l}" y="14" class="t">{html.escape(title)}</text>']
    for k, r in enumerate(reps):
        parts.append(f'<rect x="{X(r["start_s"]):.1f}" y="{pad_t}" width="{X(r["end_s"]) - X(r["start_s"]):.1f}" '
                     f'height="{H}" class="{"rep" if k % 2 == 0 else "rep2"}"/>')
    if window:
        parts.append(f'<rect x="{X(window[0]):.1f}" y="{pad_t}" width="{max(X(window[1]) - X(window[0]), 2):.1f}" '
                     f'height="{H}" class="win"/>')
    pts = " ".join(f"{X(tt):.1f},{Y(v):.1f}" for tt, v in zip(t[ok], x[ok]))
    parts.append(f'<polyline points="{pts}" class="line"/>')
    for v in (lo, hi):
        parts.append(f'<text x="{pad_l - 4}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v:.2g}</text>')
    for s in range(0, int(t[-1]) + 1, max(1, int(t[-1] // 8) or 1)):
        parts.append(f'<text x="{X(s):.1f}" y="{height - 6}" class="ax" text-anchor="middle">{s}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def key_frame(video: Path, bp: np.ndarray, t_s: float, info: dict) -> str | None:
    """The video frame at t_s with the skeleton drawn (orange = left, blue = right), as a data URI."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if info.get("mirrored_input"):
        frame = cv2.flip(frame, 1)
    k = int(np.clip(round(t_s * FPS), 0, len(bp) - 1))
    p = bp[k]
    for a, b in EDGES:
        if np.isfinite(p[[a, b], :2]).all():
            col = (0, 140, 255) if a in LEFT and b in LEFT else (255, 120, 0) if a not in LEFT and b not in LEFT \
                else (200, 200, 200)
            cv2.line(frame, tuple(int(v) for v in p[a, :2]), tuple(int(v) for v in p[b, :2]), col, 3, cv2.LINE_AA)
    scale = 640 / max(frame.shape[1], 1)
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def contact_sheet(video: Path, bp: np.ndarray, info: dict, n: int = 12) -> bytes | None:
    """n frames spread over the video, 4 per row, skeleton drawn or 'NOT FOUND' written."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for k in np.linspace(0, len(bp) - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_MSEC, k / FPS * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        if info.get("mirrored_input"):
            frame = cv2.flip(frame, 1)
        p = bp[k]
        found = np.isfinite(p[11, 0])
        for a, b in EDGES:
            if np.isfinite(p[[a, b], :2]).all():
                col = (0, 140, 255) if a in LEFT and b in LEFT else (255, 120, 0) if a not in LEFT and b not in LEFT \
                    else (200, 200, 200)
                cv2.line(frame, tuple(int(v) for v in p[a, :2]), tuple(int(v) for v in p[b, :2]), col, 3, cv2.LINE_AA)
        cv2.putText(frame, f"{k / FPS:.1f}s" + ("" if found else "  NOT FOUND"), (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        tiles.append(cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1]))))
    cap.release()
    if not tiles:
        return None
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#2563eb;--rep:#2563eb14;--rep2:#2563eb08;--win:#dc2626;--card:#f6f6f7}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;--line:#60a5fa;--rep:#60a5fa22;--rep2:#60a5fa10;--win:#f87171;--card:#1f1f23}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:860px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px} .muted{color:var(--muted)}
.card{background:var(--card);border-radius:10px;padding:14px 16px;margin:10px 0}
.patient{font-size:17px} table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:6px 8px;border-bottom:1px solid #8883;text-align:left;vertical-align:top}
.error{color:var(--win);font-weight:600} .ok{color:#16a34a;font-weight:600} .borderline{color:#d97706;font-weight:600}
svg .t{font-size:12px;fill:var(--fg)} svg .ax{font-size:10px;fill:var(--muted)}
svg .line{fill:none;stroke:var(--line);stroke-width:1.6} svg .rep{fill:var(--rep)} svg .rep2{fill:var(--rep2)}
svg .win{fill:none;stroke:var(--win);stroke-width:1.5;stroke-dasharray:4 3}
img{max-width:100%;border-radius:8px}
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--exercise", required=True, choices=sorted(EXERCISES))
    ap.add_argument("--model", default=str(ROOT / "models" / "pose_landmarker_full.task"))
    ap.add_argument("--mirrored", action="store_true", help="the video is mirrored (front-camera selfie)")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="MediaPipe detection/tracking confidence (default 0.5; try 0.3 if the person is often lost)")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also save 12 frames with the skeleton (or 'NOT FOUND') to see why frames are missed")
    ap.add_argument("--thresholds", default=str(APP_THRESHOLDS),
                    help="default: config/findings_thresholds_mediapipe.json if present (fitted on our own MediaPipe)")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--base-url")
    ap.add_argument("--model-name", "--llm-model", dest="llm_model")
    ap.add_argument("--api-key-env")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--out-dir", default="outputs/sessions")
    args = ap.parse_args()

    video = Path(args.video)
    bp, info = pose_from_video(video, args.model, mirrored=args.mirrored, min_confidence=args.min_confidence)
    engine = FindingsEngine(args.thresholds)
    findings = engine.run(bp, args.exercise)
    if info["missing_fraction"] > 0.5:
        print(f"warning: the person was found in only {1 - info['missing_fraction']:.0%} of frames; "
              f"checks need at least half. Re-record with the whole upper body and hips in view.")
    llm = None
    if args.llm:
        load_env_file(args.env_file)
        if not (args.base_url and args.llm_model):
            raise SystemExit("--llm needs --base-url and --llm-model")
        llm = LLMClient(args.base_url, args.llm_model, args.api_key_env,
                        cache_path=Path(args.out_dir) / "llm_cache.jsonl")
    fb = write_feedback(findings, args.exercise, llm)
    if not findings and engine.skipped:            # checks exist but the person was not seen well enough
        from rehabform.feedback import NOT_SEEN_TEXT
        fb.text, fb.source = NOT_SEEN_TEXT, "not_seen"

    # console summary
    dur = len(bp) / FPS
    print(f"{video.name}: {dur:.1f} s, person found in {1 - info['missing_fraction']:.0%} of frames, "
          f"sides swapped by geometry: {info['swapped']}")
    for f in findings:
        val = f"{f.value:g} (typical {f.typical:g})" if f.quotable and f.value is not None else "not quoted"
        print(f"  {f.id} {f.detector}: {f.status} {f.severity or ''} {val}, window {f.start_s}-{f.end_s} s, "
              f"{len(f.reps)} reps")
    print(f"\nPatient feedback ({fb.source}): {strip_tags(fb.text)}")
    if fb.draft is not None and fb.source != "llm":
        print(f"  (LLM draft rejected: {fb.draft_report.reasons()})")

    # HTML report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ser = series(bp)
    if args.contact_sheet:
        sheet = contact_sheet(video, bp, info)
        if sheet is not None:
            p = Path(args.out_dir) / f"{video.stem}_frames.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(sheet)
            print(f"contact sheet: {p}")
    blocks = []
    for f in findings:
        x = ser.get(f.feature, np.full(len(bp), np.nan))
        win = (f.start_s, f.end_s) if f.start_s is not None else None
        unit = f.unit if f.unit == "°" else f" {f.unit}"
        val = f"{f.value:g}{unit} (correct repetitions typically ~{f.typical:g}{unit})" \
            if f.quotable and f.value is not None else "measured; not quoted (proxy or trust-table rule)"
        ev = f.evidence or {}
        evid = (f"tested on people it was not tuned on: catches {ev['loso_sensitivity']:.0%} of clinician-marked "
                f"errors, leaves {ev['loso_specificity']:.0%} of correct recordings alone "
                f"(agreement with clinicians, kappa {ev.get('kappa_system_A', float('nan')):.2f} / "
                f"{ev.get('kappa_system_B', float('nan')):.2f}; between clinicians {ev.get('kappa_clinicians', float('nan')):.2f})"
                if "loso_sensitivity" in ev else "")
        blocks.append(
            f'<div class="card"><b>{f.id} · {html.escape(f.issue)}</b> — <span class="{f.status}">{f.status}'
            f'{" (" + f.severity + ")" if f.severity else ""}</span><br><span class="muted">{html.escape(f.measure_label)}'
            f': {html.escape(val)}; window {f.start_s}–{f.end_s} s ({f.window_source}); '
            f'{len(f.reps)} repetitions.<br>{html.escape(evid)}</span>'
            f'{svg_timeline(x, f.reps, win, f.measure_label + " over time (dashed = the repetition referred to)")}</div>')
    first = next((f for f in findings if f.start_s is not None), None)
    img = key_frame(video, bp, (first.start_s + first.end_s) / 2, info) if first else None
    draft = ""
    if fb.draft is not None:
        draft = (f'<p class="muted">Language-model draft ({"passed" if fb.source == "llm" else "rejected: " + html.escape(", ".join(fb.draft_report.reasons()))}):'
                 f' {html.escape(fb.draft)}</p>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Session report</title><style>{CSS}</style></head><body>
<h1>{html.escape(EXERCISES[args.exercise].split(':')[0].capitalize())} · session report</h1>
<p class="muted">{html.escape(video.name)} · {dur:.1f} s · person found in {1 - info['missing_fraction']:.0%} of frames · analysed at {FPS:g} fps{' · mirrored input' if args.mirrored else ''}</p>
<h2>What the patient sees</h2><div class="card patient">{html.escape(strip_tags(fb.text))}</div>
<p class="muted">Source: {fb.source}. Every sentence is backed by a finding below; the claim checker rejects anything else.</p>{draft}
<h2>Clinician view</h2>{''.join(blocks) or '<p>No validated checks for this exercise yet.</p>'}
{f'<h2>Key frame</h2><img src="{img}" alt="video frame with skeleton"><p class="muted">Orange = left side, blue = right side. If the colours are swapped, re-run with --mirrored.</p>' if img else ''}
</body></html>"""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{video.stem}_{args.exercise}.html"
    path.write_text(page, encoding="utf-8")
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
