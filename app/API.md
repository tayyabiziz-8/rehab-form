# Physio feedback API (v0.1, 20 Nov 2026)

One call per recorded set. The phone sends **landmarks only** (never video); the server returns findings and feedback that passed the claim checker.

Server: `python app/server.py` → `http://localhost:8000`. Standard library only; it also serves the reference web app (`app/web/index.html`) and the MediaPipe model from `models/`.

## `GET /api/health`

```json
{
  "ok": true,
  "exercises": {"CTK": "hiding face: …", "ELK": "flank stretch: …", "RTK": "torso rotation: …"},
  "validated_checks": {"CTK": [{"id": "CTK_arms_not_raised", "checks": "arms not raised enough", "body_part": "BothArms"}],
                       "ELK": [{"id": "ELK_body_not_tilted", "checks": "body not tilted", "body_part": "Spine"}],
                       "RTK": []},
  "llm": false,
  "strictness": ["strict", "standard", "lenient"]
}
```

Use it to build the exercise picker and to tell the user which exercises have no validated checks yet (RTK today).

## `POST /api/session`

### Request

```json
{
  "exercise": "ELK",
  "strictness": "standard",
  "use_llm": false,
  "width": 1280,
  "height": 720,
  "mirrored": false,
  "frames": [
    {"t_ms": 0.0,   "landmarks": [[0.512, 0.301, -0.21], [0.515, 0.288, -0.20], "… 33 rows …"]},
    {"t_ms": 33.4,  "landmarks": null},
    {"t_ms": 66.7,  "landmarks": [["…"]]}
  ]
}
```

| Field | Meaning |
|---|---|
| `exercise` | `CTK`, `ELK` or `RTK` (Keraal's three exercises). |
| `strictness` | Clinician setting, optional (default `standard`). Scales the borderline band past each threshold: `strict` = none, `standard` = fitted band, `lenient` = 2×. Thresholds never move. |
| `use_llm` | Ask for language-model wording. Only honoured if the server was started with `--llm`; the draft must pass the claim checker or the template text is returned. |
| `width`, `height` | Pixel size of the frames MediaPipe saw (`video.videoWidth/Height`). Needed to turn normalised landmarks into pixels; angles are wrong without it. |
| `mirrored` | `true` only if the **frames given to MediaPipe** were mirrored. A CSS-mirrored preview does not count: keep this `false` when you mirror only the display. |
| `frames[].t_ms` | Time of the frame in milliseconds, strictly increasing. Any rate; the server resamples to 10 fps of real time. |
| `frames[].landmarks` | MediaPipe Pose Landmarker `result.landmarks[0]`: 33 × `[x, y, z]`, normalised (x, y in 0–1, z relative). `null` when no person was found. Round to 5 decimals to keep the body small. |

Limits: at least 2 s of recording, body ≤ 25 MB (a 2-minute set at 30 fps is about 3 MB).

### Response (200)

```json
{
  "ok": true,
  "status": "ok",
  "exercise": "ELK",
  "strictness": "standard",
  "frames_received": 897, "fps_received": 30.0, "duration_s": 29.87,
  "person_found": 0.99, "analysis_fps": 10.0, "sides_swapped": false,
  "feedback": {
    "text": "Bend your upper body further to the side: the sideways bend of your upper body reached 12° (most clearly in the repetition around 12.6–18.3 s), while correct repetitions typically reach about 25°.",
    "sentences": [{"text": "Bend your upper body further …", "cites": ["F1"]}],
    "source": "template_only",
    "llm_draft": null,
    "llm_rejected_reasons": []
  },
  "findings": [{
    "id": "F1", "detector": "ELK_body_not_tilted", "exercise": "ELK",
    "status": "error", "severity": "small",
    "issue": "body not tilted", "advice": "bend your upper body further to the side",
    "body_part": "Spine", "side": null,
    "feature": "lean_range", "measure_label": "sideways bend of your upper body", "unit": "°",
    "quotable": true, "value": 12.0, "typical": 25.0, "threshold": 15.0,
    "start_s": 12.6, "end_s": 18.3, "window_source": "worst_rep",
    "reps": [{"start_s": 0.0, "end_s": 6.7, "peak": 11.995}, "…"],
    "trust_mode": "range", "trust_note": "…",
    "evidence": {"loso_sensitivity": 0.79, "loso_specificity": 0.93, "kappa_clinicians": 0.43,
                 "kappa_system_A": 0.53, "kappa_system_B": 0.43, "n_error": 43, "n_correct": 15, "people_error": 5}
  }],
  "skipped_checks": [],
  "timelines": {"F1": [0.0, 0.1, "… one value per 0.1 s, null where missing …"]}
}
```

| Field | How the app uses it |
|---|---|
| `status` | `ok`; `not_seen` (person visible in < half the frames → show `feedback.text`, which asks for a re-recording); `no_checks` (exercise has no validated checks yet). |
| `feedback.text` | **The only text the patient sees.** Show it as is: no added numbers, body parts or advice. |
| `feedback.sentences[].cites` | Finding ids each sentence relies on. Clinician view: tap a sentence → highlight its finding and window. |
| `feedback.source` | `template_only`, `template` (LLM draft rejected → template used), `llm` (draft passed the checker), `not_seen`. |
| `findings[].status` | `error` (patient is told), `borderline` (**clinician view only**, never patient), `ok` (may be praised). |
| `findings[].quotable` | `false` → never show `value` / `typical` to anyone as a number (proxy measure or trust-table rule). |
| `findings[].start_s` / `end_s` | Seconds from the first frame you sent. The repetition the finding refers to (`window_source = worst_rep`) or the peak moment (`peak`). Use with your stored landmarks for the skeleton replay. |
| `findings[].reps` | Repetitions the server found (seconds). Show as bands on the timeline; the count is the authoritative rep count. |
| `findings[].evidence` | How the check did on people it was not tuned on. Show in the clinician view ("catches 79% of clinician-marked errors, leaves 93% of correct sets alone"). |
| `timelines[F#]` | The finding's measure at 10 fps, for the clinician chart. |

### Errors

`400 {"ok": false, "error": "…"}` for a bad exercise or strictness, missing `width`/`height`, landmarks not 33 × 3, `t_ms` not increasing, or less than 2 s. `413` for a body over 25 MB. `500` with the exception text for a server bug (please report it with the request).

## Conventions that break things if ignored

1. **Landmarks are MediaPipe's 33-point Pose Landmarker output** (`@mediapipe/tasks-vision`, `pose_landmarker_full.task`). Other models or orders are not supported.
2. **Do not mirror the frames MediaPipe sees.** Mirror the on-screen preview with CSS if you like (`transform: scaleX(-1)` on both `<video>` and the overlay `<canvas>`).
3. **Send real timestamps.** Frame-rate drops are fine; reordered or duplicate times are not.
4. **The person must face the camera.** Left/right are fixed on the server assuming that; side-on recordings are not supported.
5. **Whole upper body and hips in view**, camera at chest height, 2–3 m away. Below half the frames with a person, nothing is judged.
6. **Never show `borderline` findings or non-quotable numbers to the patient.**
