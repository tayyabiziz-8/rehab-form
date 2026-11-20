# rehab-form — measured, grounded feedback for home physiotherapy

A phone watches a patient do a prescribed exercise and gives feedback a clinician can check: every sentence points at a measurement, and nothing is said that the measurements do not support.

- **Pillar 1 — measurement validity:** phone pose (BlazePose / MediaPipe) against Vicon on Keraal Group 3 → the angle spec and the trust table (what may be reported, per measure and exercise).
- **Pillar 2 — grounded feedback:** clinician-labelled Keraal Groups 1A/2A → validated error checks, a findings engine, an LLM writer whose sentences must pass a claim checker, compared with an LLM that only sees the pose.
- **App:** browser app + small server. Only landmarks leave the phone, never video.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # numpy, opencv-python, mediapipe (+ torch/aeon for the baseline)
python tests/test_findings_claims.py
python tests/test_keraal_groups.py
```

Use Python 3.10–3.12 (MediaPipe wheels lag new Python releases; 3.11 is what the team uses). Put `pose_landmarker_full.task` in `models/`:

```bash
curl -L -C - --retry 10 -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Data never goes in git (`data/`, `outputs/`, `models/`, `.env` are ignored); share it through Drive or Kaggle.

## Quick start: the app

```bash
python scripts/pillar2_detectors.py      # once: writes config/findings_thresholds.json (needs data/keraal1, data/keraal2)
python app/server.py                     # http://localhost:8000
```

Open http://localhost:8000, pick an exercise, start the camera (or choose a video file), record a set, read the feedback. API contract: `app/API.md`.

## What is here

| Path | What it does | Status |
|---|---|---|
| `src/rehabform/skeleton.py` | Maps MediaPipe (33) and Kinect v2 (25) skeletons to 13 shared joints; normalises (mid-hip centre, torso scale); computes joint angles and trunk lean | Tested |
| `src/rehabform/angles.py` | **Angle spec**: the phone measures the app reports (arm elevation vs shoulder line, elbow flexion, trunk lean, trunk rotation), computed exactly as validated, with validity masks; `TrustTable` reads `config/trust_table.json` | Tested |
| `config/trust_table.json` | **Trust table v3**: per measure and exercise, whether it may be reported (range / relative / timing only / not reportable), why, and the validation numbers; Keraal errors mapped to measures | Thresholds still to calibrate |
| `src/rehabform/reps.py` | Counts reps from one joint angle with peak detection; no training needed | Tested |
| `src/rehabform/models.py` | FCN baseline (Wang et al. 2017), one of the 9 models benchmarked on Rehab-Pile | Compiles; not yet run |
| `scripts/train_rehab_pile.py` | Trains FCN on every fold of one Rehab-Pile dataset; prints accuracy and balanced accuracy, mean ± std | Compiles; not yet run |
| `scripts/extract_mediapipe.py` | Runs MediaPipe Pose Landmarker on a folder of videos; saves `.npz` skeletons | Compiles; not yet run |
| `scripts/check_datasets.py` | First look at a downloaded dataset; decodes IntelliRehabDS and UI-PRMD file names | Tested on fake folders |
| `scripts/evaluate_keraal.py` | Computes BlazePose-vs-Vicon angle errors for all Group 3 recordings | Tested on Keraal Group 3 |
| `scripts/compare_keraal_modalities.py` | Compares Kinect-vs-Vicon angle errors across Group 3 | Tested on Keraal Group 3 |
| `scripts/align_keraal_modalities.py` | Estimates whole-frame Kinect-Vicon lag without changing baseline metrics | Tested on Keraal Group 3 |
| `scripts/diagnose_keraal_mapping.py` | Tests a diagnostic Vicon left/right swap against Kinect without changing baseline mapping | Tested on Keraal Group 3 |
| `scripts/evaluate_keraal_kinect.py` | Computes BlazePose-vs-Kinect angle errors using Kinect's anatomical joints as the primary reference | Tested on Keraal Group 3 |
| `scripts/diagnose_shoulder_angles.py` | Compares candidate shoulder-angle definitions against Kinect without changing production metrics | Vertical-axis fix 8 Nov; rerun needed |
| `scripts/evaluate_keraal_vicon_offsetfree.py` | BlazePose vs Vicon on change from start pose and range of motion (cancels ART-target offsets); by exercise and participant; signed frontal-plane arm angle; skips recordings with mismatched lengths | Run on Group 3; signed angle added |
| `scripts/evaluate_vicon_quaternions.py` | Vicon joint angles from segment orientations (chest→arm, arm→forearm, hips→chest), compared with the phone and with target-position angles; tells whether target-based angles understate ranges | Tested on synthetic data |
| `scripts/diagnose_arm_reference.py` | Phone arm angle against four trunk references (midline, shoulder-line perpendicular, image vertical, 3D midline) vs orientation-based Vicon; explains overhead-reach overstatement | Tested on synthetic data |
| `scripts/overlay_blazepose.py` | Draws BlazePose (left orange, right blue) on 4 video frames per recording: start, Vicon arm peaks, largest disagreement | Tested on synthetic video |
| `tests/test_keraal.py` | Checks BlazePose name ordering, pixel conversion and the vertical-axis fix on a synthetic Keraal folder | Passing |
| `docs/Keraal Group 3 - Initial Angle Evaluation.md` | Obsidian-compatible record of the initial phone-vs-Vicon evaluation | Pre-fix results; rerun pending (see `CHANGES-2026-11-08.md`) |
| `data/README.md` | Where to download each dataset, licences, what may ship in the product | — |
| `src/rehabform/keraal_groups.py` | Keraal Groups 1A/2A: file names, both annotators' Anvil labels (UTF-16), `canonical_sides` (fixes BlazePose left/right by geometry), `load_blazepose_g12` | Tested |
| `src/rehabform/session_features.py` | Recording features the detectors threshold, their per-frame series, `segment_reps`, `peak_window` | Tested |
| `src/rehabform/findings.py` | **Findings engine**: validated checks + trust table → findings (error / borderline / ok, value, typical, severity, worst repetition, evidence); clinician strictness | Tested |
| `src/rehabform/claims.py` | **Claim checker**: every feedback sentence must cite a finding; rejects invented numbers, units, body parts, sides, contradictions, medical wording | Tested (incl. real gpt-oss drafts) |
| `src/rehabform/feedback.py` | Template writer (always passes), grounded LLM prompt, pose-only baseline prompt, OpenAI-compatible `LLMClient` (cache, rate limits), `write_feedback`, `split_for_app` | Tested; Groq runs pending |
| `src/rehabform/video_pose.py` | MediaPipe Pose Landmarker on any video → 10 fps pixel landmarks, sides fixed | Runs; person found only 44% on an anonymised Keraal video (investigating) |
| `config/findings_thresholds.json` | Fitted thresholds, borderline margins, typical values and leave-one-subject-out evidence per check (written by `pillar2_detectors.py`) | CTK arms-height and ELK side-bend in use; 3 checks off |
| `scripts/summarise_keraal_annotations.py` | Layout, label vocabulary, annotator agreement, video format, mirroring check for Group 1A/2A | Run |
| `scripts/pillar2_error_signals.py` | Labels table, clinician kappa, recording features, which feature separates each labelled error (AUC) | Run on 1A + 2A |
| `scripts/pillar2_detectors.py` | One detector per error, leave-one-subject-out, kappa vs each clinician, severity order, borderline sweep → `config/findings_thresholds.json` | Run |
| `scripts/pillar2_feedback_eval.py` | Detection, time windows, severity; template / grounded LLM / pose-only LLM feedback vs clinicians and the claim checker | Run (template); LLM runs pending |
| `scripts/session_report.py` | One video → findings → checked feedback → single HTML report (contact sheet option) | Run |
| `scripts/validate_mediapipe_keraal.py` | Our MediaPipe vs Keraal's shipped BlazePose on the same videos: features and decisions | Pending first full run |
| `app/server.py` | **App server** (standard library): `POST /api/session` landmarks → findings + feedback; serves the reference web app and `models/` | Tested |
| `app/web/index.html` | **Reference web app**: camera or video file → MediaPipe in the browser → live angles and reps → server → patient text + clinician view with replay | Tested with synthetic data; camera untested |
| `app/API.md` | API contract for the app (request, response, conventions, errors) | — |
| `tests/test_keraal_groups.py`, `tests/test_findings_claims.py` | Group 1/2 loading and labels; findings, borderline, worst repetition, template and claim checker | Passing |

"Not yet run" means the dependency was not available where the code was written; expect small fixes on the first run.

## Run order

### Pillar 1 and baseline run order

```bash
# 1. Benchmark baseline (after downloading Rehab-Pile, see data/README.md)
python scripts/train_rehab_pile.py --list
python scripts/train_rehab_pile.py --dataset <NAME> --extract-path data/rehab_pile --max-folds 1 --epochs 50   # smoke test
python scripts/train_rehab_pile.py --dataset <NAME> --extract-path data/rehab_pile                             # full run

# 2. Inspect raw datasets after each download
python scripts/check_datasets.py data/keraal
python scripts/check_datasets.py data/irds --kind irds
python scripts/check_datasets.py data/ui_prmd --kind ui_prmd

# 3. MediaPipe skeletons from video (Keraal, KIMORE)
python scripts/extract_mediapipe.py --videos data/keraal/videos --out data/keraal/mediapipe \
    --model models/pose_landmarker_full.task --limit 10

# 4. Aggregate phone-vs-Vicon angle errors
python scripts/evaluate_keraal.py --root data/keraal

# 5. Validate Kinect-vs-Vicon angle errors
python scripts/compare_keraal_modalities.py --root data/keraal

# 6. Diagnose whole-frame Kinect-vs-Vicon timing lag
python scripts/align_keraal_modalities.py --root data/keraal

# 7. Evaluate phone angles against the Kinect anatomical reference
python scripts/evaluate_keraal_kinect.py --root data/keraal

# 8. Diagnose alternative shoulder-angle definitions
python scripts/diagnose_shoulder_angles.py --root data/keraal

# 9. Offset-free phone-vs-Vicon (change from start pose, range of motion)
python scripts/evaluate_keraal_vicon_offsetfree.py --root data/keraal

# 10. See where phone pose fails: skeleton overlay on video frames
python scripts/overlay_blazepose.py --root data/keraal --exercise ELK

# 11. Vicon joint angles from segment orientations (quaternions)
python scripts/evaluate_vicon_quaternions.py --root data/keraal

# 12. Which trunk reference makes the phone arm angle match Vicon?
python scripts/diagnose_arm_reference.py --root data/keraal
```

Download the `.task` model file from the MediaPipe Pose Landmarker guide into `models/` first.

### Pillar 2 run order (Groups 1A in `data/keraal1`, 2A in `data/keraal2`)

```bash
python scripts/summarise_keraal_annotations.py data/keraal2 --overlay 3
python scripts/pillar2_error_signals.py data/keraal1 data/keraal2
python scripts/pillar2_detectors.py                      # thresholds, borderline band, evidence
python scripts/pillar2_feedback_eval.py data/keraal1 data/keraal2
python scripts/pillar2_feedback_eval.py data/keraal1 data/keraal2 --llm --baseline --limit 60 \
    --base-url https://api.groq.com/openai/v1 --api-key-env API_KEY --model openai/gpt-oss-20b --out-dir outputs/llm_20b
python scripts/validate_mediapipe_keraal.py data/keraal1 data/keraal2 --limit 40
python scripts/session_report.py my_video.mp4 --exercise ELK --contact-sheet
```

The Groq key goes in `.env` as `API_KEY=...` (never committed). The free tier allows about 200,000 tokens per model per day; runs resume from `outputs/*/llm_cache.jsonl`.

## Results so far (Nov 2026)

| What | Result |
|---|---|
| Phone vs Vicon (Group 3, orientation-based Vicon) | Arms r 0.76–0.99; ELK overhead reach overstated (~+26° with shoulder-line reference); trunk side-bend r 0.99, 2.4°; elbows only valid in ELK |
| Clinician agreement (1A + 2A, Correct vs Incorrect) | kappa 0.59 (CTK 0.73, ELK 0.58, RTK 0.50); same error type 78/88 when both say Incorrect |
| ELK "body not tilted" (side-bend range) | leave-one-subject-out balanced accuracy 86%; kappa with clinicians 0.53 / 0.43 vs 0.43 between them |
| CTK "arms not raised enough" (hand height) | leave-one-subject-out sensitivity 96%, specificity 78%; kappa 0.43 / 0.56 vs 0.64 between clinicians |
| CTK Error3, both RTK checks | do not hold up on unseen people (2–3 people each) → switched off |
| Severity (small / large) vs clinician A | 44 of 61 |
| Claim checker | template 299/299; gpt-oss-20b drafts 36/41 passed before the checker (6.8% unsupported sentences), 0 shown after |

## Known TODOs

- [ ] Finish the Groq runs (gpt-oss-20b and -120b) and report the grounded vs pose-only comparison.
- [ ] Find why MediaPipe misses the person on anonymised Keraal videos (contact sheet, `--min-confidence 0.3`); check own phone videos.
- [ ] Time windows: overlap with clinician A's windows is barely above chance; check the worst-repetition windows.
- [ ] Fill `UI_PRMD_22` and `KIMORE_18` joint orders in `skeleton.py` from each dataset's documentation. Do not guess them.
- [ ] KIMORE FCN baseline as mean ± std over seeds and folds.
- [ ] For LITEMV, the Rehab-Pile winner, use the authors' official code rather than re-implementing it.

## Rules for results

- BlazePose angles must be computed in pixel units (`load_blazepose` does this by default; Group 3 = 960 × 544, other groups = 480 × 360).
- Keraal's shipped BlazePose is mirrored (x flipped, Left/Right swapped); `load_blazepose` un-mirrors by default. Verify with `overlay_blazepose.py` before using other groups.
- Against Vicon, report change from start pose and range of motion, not absolute angles.
- BlazePose vs Kinect is agreement, not accuracy.

- Always report mean ± std over all folds, plus balanced accuracy. Datasets are small (often 70–300 samples per exercise), and single runs mislead.
- Never select models on the test fold. `train_rehab_pile.py` selects on training loss only.
- Models trained on Keraal, KIMORE, Fitness-AQA or FLEX are for research results only until licences are confirmed (see `data/README.md`).
- Group 1A/2A BlazePose is mirrored too (confirmed on 6 overlays), and its left/right labels flip in some recordings: use `load_blazepose_g12` (un-mirror + geometry side fix).
- Anvil annotation times use the original video clock (~19 fps, matching Vicon's 19.4 Hz), not the 10 fps the mp4 plays at; rescale before comparing windows.
- A check feeds patient feedback only if it holds up leave-one-subject-out (balanced accuracy ≥ 75%, ≥ 4 people). Numbers are quoted only for validated measures the trust table allows.
