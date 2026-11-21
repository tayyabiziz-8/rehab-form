"""Physio feedback server: landmarks in, checked feedback out. Standard library only.

    python app\\server.py                      -> http://localhost:8000  (reference web app + API)
    python app\\server.py --port 8000 --llm --base-url https://api.groq.com/openai/v1 ^
        --llm-model openai/gpt-oss-120b --api-key-env API_KEY

Endpoints
  GET  /                    the reference web app (app/web/index.html)
  GET  /models/<file>       MediaPipe model files from models/ (so the browser needs no Google download)
  GET  /api/health          {"ok": true, "exercises": {...}, "validated_checks": {...}}
  POST /api/session         landmarks of one recorded set -> findings + feedback (see API.md)

Only landmarks come in; no video ever reaches the server.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.feedback import (  # noqa: E402
    EXERCISES, NOT_EXERCISE_TEXT, NOT_SEEN_TEXT, LLMClient, load_env_file, split_for_app, write_feedback,
)
from rehabform.findings import APP_THRESHOLDS, STRICTNESS, FindingsEngine  # noqa: E402
from rehabform.keraal_groups import canonical_sides  # noqa: E402
from rehabform.session_features import FPS, series  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
MODELS = ROOT / "models"
MAX_BODY = 25 * 1024 * 1024        # a 2-minute set at 30 fps is ~3 MB of JSON
MIN_FRAMES = 20                    # 2 s at 10 fps


class BadRequest(ValueError):
    pass


def landmarks_to_10fps(frames: list, width: float, height: float, mirrored: bool) -> tuple[np.ndarray, dict]:
    """[{"t_ms", "landmarks": 33 x [x, y, z] normalised or null}] -> (T, 33, 3) pixels at 10 fps of real time."""
    t, pts = [], []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict) or "t_ms" not in fr:
            raise BadRequest(f"frames[{i}] needs t_ms and landmarks")
        lm = fr.get("landmarks")
        if lm is None:
            a = np.full((33, 3), np.nan)
        else:
            a = np.asarray(lm, float)
            if a.shape[0] != 33 or a.shape[1] < 3:
                raise BadRequest(f"frames[{i}].landmarks must be 33 x [x, y, z] (got {a.shape})")
            a = a[:, :3]
        t.append(float(fr["t_ms"]))
        pts.append(a)
    t = np.asarray(t) / 1000.0
    if np.any(np.diff(t) <= 0):
        raise BadRequest("t_ms must increase from frame to frame")
    P = np.stack(pts) * np.array([width, height, width])          # normalised -> pixels (z scales with width)
    if mirrored:                                                   # a mirrored image: flip back to the camera's view
        P[..., 0] = width - P[..., 0]
    t = t - t[0]
    grid = np.arange(0, t[-1] + 1e-9, 1.0 / FPS)
    out = np.full((len(grid), 33, 3), np.nan)
    found = np.isfinite(P[:, 11, 0])
    # nearest real frame within half a step; a lost person stays lost (no invented landmarks)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.where(np.abs(t[idx - 1] - grid) <= np.abs(t[idx] - grid), idx - 1, idx)
    close = np.abs(t[near] - grid) <= 0.75 / FPS
    for k, (j, ok) in enumerate(zip(near, close)):
        if ok and found[j]:
            # linear interpolation between neighbours when both have a person
            j2 = j + 1 if t[j] < grid[k] and j + 1 < len(t) else j - 1 if t[j] > grid[k] and j > 0 else j
            if j2 != j and found[j2] and t[j2] != t[j]:
                w = (grid[k] - t[j]) / (t[j2] - t[j])
                out[k] = P[j] + w * (P[j2] - P[j])
            else:
                out[k] = P[j]
    info = {"frames_received": len(frames), "fps_received": round((len(t) - 1) / t[-1], 1) if t[-1] > 0 else None,
            "duration_s": round(float(t[-1]), 2),
            "person_found": round(float(np.isfinite(out[:, 11, 0]).mean()), 3) if len(out) else 0.0}
    return out.astype(np.float32), info


class App:
    def __init__(self, thresholds: Path, llm: LLMClient | None):
        self.engine = FindingsEngine(thresholds)
        self.llm = llm

    def validated(self) -> dict:
        out = {ex: [] for ex in EXERCISES}
        for det_id, d in self.engine.detectors.items():
            out.setdefault(d["exercise"], []).append({"id": det_id, "checks": d["meaning"], "body_part": d["body_part"]})
        return out

    def session(self, req: dict) -> dict:
        ex = req.get("exercise")
        if ex not in EXERCISES:
            raise BadRequest(f"exercise must be one of {sorted(EXERCISES)}")
        strictness = req.get("strictness", "standard")
        if strictness not in STRICTNESS:
            raise BadRequest(f"strictness must be one of {sorted(STRICTNESS)}")
        try:
            width, height = float(req["width"]), float(req["height"])
        except (KeyError, TypeError, ValueError):
            raise BadRequest("width and height (the video's pixel size) are required") from None
        frames = req.get("frames") or []
        bp, info = landmarks_to_10fps(frames, width, height, bool(req.get("mirrored", False)))
        if len(bp) < MIN_FRAMES:
            raise BadRequest(f"recording too short: {info['duration_s']} s (need at least {MIN_FRAMES / FPS:g} s)")
        bp, sides = canonical_sides(bp)
        findings = self.engine.run(bp, ex, strictness=strictness)
        use_llm = bool(req.get("use_llm", False)) and self.llm is not None
        fb = write_feedback(findings, ex, self.llm if use_llm else None)
        status = "ok"
        if not findings and self.engine.skipped:
            status, fb.text, fb.source = "not_seen", NOT_SEEN_TEXT, "not_seen"
        elif not self.engine.conformity["ok"]:
            status, fb.source = "not_this_exercise", "not_this_exercise"
            fb.text = NOT_EXERCISE_TEXT.format(name=EXERCISES[ex].split(":")[0])
        elif not self.validated().get(ex):
            status = "no_checks"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ser = series(bp)
        timelines = {}
        for f in findings:
            x = ser.get(f.feature)
            if x is not None:
                timelines[f.id] = [None if not np.isfinite(v) else round(float(v), 3) for v in x]
        return {
            "ok": True, "status": status, "exercise": ex, "strictness": strictness,
            **info, "analysis_fps": FPS, "sides_swapped": sides["swapped"],
            "feedback": {**split_for_app(fb.text, findings), "source": fb.source,
                         "llm_draft": fb.draft, "llm_rejected_reasons": fb.draft_report.reasons()
                         if fb.draft_report is not None and fb.source != "llm" else []},
            "findings": [f.to_dict() for f in findings],
            "skipped_checks": list(self.engine.skipped),
            "conformity": self.engine.conformity,
            "timelines": timelines,
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rehabform/0.1"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/api/health":
                return self._json(200, {"ok": True, "exercises": EXERCISES, "validated_checks": app.validated(),
                                        "llm": app.llm is not None, "strictness": list(STRICTNESS)})
            if path.startswith("/models/"):
                f = (MODELS / path[len("/models/"):]).resolve()
                if f.parent == MODELS.resolve() and f.is_file():
                    return self._send(200, f.read_bytes(), "application/octet-stream")
                return self._json(404, {"ok": False, "error": f"model not found: {path} (put it in models/)"})
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (WEB / rel).resolve()
            if WEB.resolve() in f.parents and f.is_file():
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype)
            self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/api/session":
                return self._json(404, {"ok": False, "error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._json(413 if n > MAX_BODY else 400, {"ok": False, "error": "empty or too large body"})
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                self._json(200, app.session(req))
            except (BadRequest, json.JSONDecodeError) as e:
                self._json(400, {"ok": False, "error": str(e)})
            except Exception as e:  # keep the server up; report the failure
                traceback.print_exc()
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args) -> None:
            sys.stderr.write("  " + fmt % args + "\n")

    return Handler


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow phones on the same Wi-Fi (see README)")
    ap.add_argument("--thresholds", default=str(APP_THRESHOLDS),
                    help="default: config/findings_thresholds_mediapipe.json if present (fitted on our own MediaPipe)")
    ap.add_argument("--llm", action="store_true", help="let requests with use_llm=true get a checked LLM draft")
    ap.add_argument("--base-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--api-key-env", default="API_KEY")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    args = ap.parse_args()
    llm = None
    if args.llm:
        load_env_file(args.env_file)
        if not (args.base_url and args.llm_model):
            raise SystemExit("--llm needs --base-url and --llm-model")
        llm = LLMClient(args.base_url, args.llm_model, args.api_key_env,
                        cache_path=ROOT / "outputs" / "app_llm_cache.jsonl")
    if not Path(args.thresholds).exists():
        raise SystemExit(f"{args.thresholds} not found: run scripts/pillar2_detectors.py first")
    app = App(Path(args.thresholds), llm)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"thresholds: {args.thresholds}")
    print(f"serving on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}  "
          f"(validated checks: { {k: [c['id'] for c in v] for k, v in app.validated().items()} }, "
          f"LLM: {'on' if llm else 'off'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
