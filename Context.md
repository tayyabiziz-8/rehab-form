# FYP Context — Measured, Grounded Feedback for Home Physiotherapy

**Last updated:** 22 Nov 2026, 14:00 PKT (living file; replaces `FYP_Context_Status.md` and `FYP_Context_Status_v2.md`)
**Team:** Tayyab Irfan Aziz (BSCS23028), Muhammad Zair (BSCS23034), Ali Gohar (BSCS23037), Zain Gulzar (BSCS23151)
**Institution:** Department of Computer Science, Information Technology University (ITU)
**Working title:** *Measured, Grounded Feedback for Home Physiotherapy from a Single Phone*
**Next milestone:** progress presentation to supervisor (~2 Dec 2026, moved from ~28 Nov)
**Current week:** Week 3 of 4 (16–22 Nov 2026)

---

## 0. Right now

**State at 20 Nov (night) — read this first.**
- **Pillar 2 built end to end:** Keraal 1A/2A loader + labels → detectors (leave-one-subject-out) → findings engine (trust table, borderline band, worst-repetition windows, clinician strictness) → feedback (template / checked LLM) → claim checker → evaluation. Two checks in use: **CTK arms not raised** (hand height; LOSO sens 96%, spec 78%) and **ELK body not tilted** (side-bend range; LOSO bal 86%, kappa 0.53/0.43 vs clinicians' 0.43). CTK Error3 + RTK checks switched off.
- **LLM runs (Groq, gpt-oss-20b / -120b):** both hit the 200k tokens/day free limit; finish after reset with the same commands (cache resumes). First 20b numbers: 36/41 drafts pass the checker, 6.8% unsupported sentences, 0 shown.
- **App starter handed to the App lead (20 Nov):** `app-starter.zip` = `app/server.py` (stdlib server, `POST /api/session`), `app/web/index.html` (reference web app: camera/video → MediaPipe JS 0.10.14 → live angles + reps → server → patient text + clinician view + replay), `app/API.md`, README rewrite. Handoff doc: **App Lead Handoff — Physio Feedback App** https://claude.ai/artifact/Ki6WreTBdcy9h3xdGH4YYT (architecture, 15-minute start, API, conventions, screens, what is validated, tasks to ~28 Nov, limitations). App lead also needs `config/findings_thresholds.json` from Danish's machine.
- **Open risk:** our MediaPipe found the person in only 44% of frames on an anonymised Keraal video (blurred face + drawn skeleton) → `--contact-sheet` / `--min-confidence 0.3` runs and own phone recordings pending; `validate_mediapipe_keraal.py` (40 videos) rerun pending.
- **20 Nov night results:** own phone ELK video → person found 100%, sides right, 4 reps, check 'ok' (67° side-to-side range vs ~32° typical) → the phone pipeline works on real video. Anonymised Keraal videos: MediaPipe often misses the person (Roscoff 4–77%). **Thresholds don't transfer between pose estimators:** own MediaPipe vs shipped BlazePose side-bend range r 0.21, hand height r 0.48; ELK decisions changed in 7/20 → refit checks on our own MediaPipe (`pillar2_error_signals.py --pose mediapipe` → `pillar2_detectors.py --features outputs/keraal_features_mediapipe.csv --thresholds config/findings_thresholds_mediapipe.json`). Borderline rule changed (zone of most clinician disagreement, ≤10 pts sensitivity cost, ≥5 recordings → CTK 0.036, ELK 1.13°). Time windows: even any-repetition overlap only 32–40% → alignment problem; `pillar2_windows_debug.csv` to inspect.
- **21 Nov refit on own MediaPipe (253/301 recs):** ELK body-not-tilted → `lean_abs_p95`, LOSO sens 89% spec 91%, kappa 0.57/0.49 vs clinicians 0.43 (4 people with error, 11 correct). CTK arms-not-raised → elbow-bend proxy (not quoted), sens 72% spec 84%, kappa 0.41/0.50 vs 0.61. App + session_report now default to `config/findings_thresholds_mediapipe.json`. Shipped-skeleton borderline (new rule): CTK 0.036 (50% disagree in band vs 9% out), ELK 1.13° (60% vs 26%). **Time-window bug found (inverted clock scale, mine) → fixed;** overlap on the debug dump 45% vs chance 30%; repetition finding misses many reps (next for windows). Handoff doc 'What is validated' updated to MediaPipe numbers.
- **21 Nov first full LLM comparison (gpt-oss-120b, prompt v1, 59 recs):** grounded drafts pass checker 12/39, 49% of sentences unsupported (22 corrections of PASSED checks, 10 untagged, 6 no unit) → 0 shown after fallback. False alarms on both-Correct recordings: pose-only baseline 92% vs grounded 8% (template 8%); labelled errors addressed 84% vs 45% (48%). Baseline invents asymmetries and praises arms that clinicians said were too low. Windows with fixed clock: CTK 36% vs chance 29%, ELK 48% vs 29%. Cause of rejections: v1 prompts said 'most important correction first' → prompt v2 for both writers (no correction when all checks pass; numbers as exact quotable phrases; fair baseline) + CTK/ELK-only prefix-stable sample. Next: v2 runs, --limit 30 --reasoning-effort low, out-dirs llm_*_v2. 20b hit daily limit again.
- **21 Nov (afternoon):** both Groq models hit daily limits again (v2 runs pending). Built `scripts/kimore_benchmark.py` (majority / logreg / MiniRocket / FCN × all KIMORE datasets × folds × 3 seeds, pooled metrics) — to run (CPU for majority/logreg/minirocket; Kaggle GPU for fcn). Related-work page: https://claude.ai/artifact/JoXR293nMZBfqkqZAfsYEF (OpenCap Monocular 4.8° MAE; BioCoach; Agentic physio (Google); Rehab-Pile LITEMV 75.5%; our gap + numbers). Supervisor report §4.3 Pillar 2 results added.
- **21 Nov KIMORE benchmark done** (5 exercises × 5 folds × 3 seeds, pooled balanced accuracy, chance 50): majority 50.0; logistic regression on summary stats 61.1; MiniRocket 61.4; FCN 57.9 (LA: FCN one class in 12/15 fold runs). Best method varies by exercise; 31 test people per exercise → ~3 points per person, differences mostly noise. Conclusion for slides: session-level classifiers ≈ 58–61% with ~70 people → justify clinician-labelled checks. Table in supervisor report §4.1 and `kimore_benchmark_combined.md`.
- **21 Nov own test videos:** `my_ctk` (arms overhead + bent-elbow flex, not CTK) → CTK elbow-bend proxy wrongly said 'raise your arms higher'; `my_elk2` (deliberately reduced bends, seated cross-legged) → 'ok' at 24° vs ~22° typical (thresholds encode patient ROM; cross-legged hides hips; 1 rep found). Fixes (`pillar2-v6-guard-envelope.zip`): proxy guard (proxy flags only while best direct measure < typical correct; evidence recomputed) + exercise envelope (one-sided Keraal p1/p99 limits; CTK arm elevation/hand height max → 'does not look like this exercise', not judged). Recording advice for demo: sit on a stool like Keraal, hips visible, make errors obvious.
- **21 Nov own videos with v6:** guard worked on my_ctk (error→ok) but CTK envelope did NOT fire (overhead still judged as hiding face; report now prints envelope values). my_elk 36° vs my_elk2 24° (MediaPipe thresholds, typical 22°, thr 14.6°) → both ok. elk2 timeline: 3 reps with tops ~6°/25°/7° → recording p95 set by the biggest rep; rep finder cut reps from the bend → weak reps not counted. v7 (`pillar2-v7-reps-from-arm.zip`): reps cut on the raised arm (CTK/ELK), new ELK candidate `lean_abs_rep_min` (smallest rep's bend; LOSO decides), report shows envelope check + per-rep tops. To run: error_signals --pose mediapipe → detectors (mediapipe) → session_report. Danish to record better videos (stool, hips visible, 3+ reps) later.
- **21 Nov night (v7 results → v8):** `lean_abs_rep_min` LOSO bal 71% vs `lean_abs_p95` 90% → not used (correct Keraal recordings also contain a small bend; clinicians judge the whole recording); per-rep tops shown to clinician only. Found corrupted MediaPipe angles (CTK arm_peak_mean −90…267°, arm_peak_high up to 450°, ELK 'bends' 108–181° in Correct recs) → CTK envelope limit was 301° (why my_ctk overhead never caught). Causes: per-frame L/R flicker (median 13–15% of CTK/ELK frames) + unwrap across gaps + upside-down frames. v8 (`pillar2-v8-side-flicker.zip`): per-frame side fix in canonical_sides (also on cache load + app), gap-aware unwrap re-centred, |lean|>90 invalid, envelope p5/p95+25% with count of Keraal recs rejected. Rerun both error_signals (shipped + mediapipe) and both detectors; run tests/test_angles.py on Danish's machine (needs make_synthetic_keraal).
- **21 Nov night v8 rerun:** test_angles passes (validated arm metric unchanged). MediaPipe CTK now uses real arm elevation `arm_peak_mean`: LOSO sens 88% spec 76% (bal 82%), kappa 0.46/0.40 vs clinicians 0.61 (proxy + guard no longer used). MediaPipe ELK `lean_abs_p95` 89/91, kappa 0.51/0.54 vs 0.43. Shipped: CTK wrist height 96/78 (arm_peak_mean 79→85%), ELK lean_range 81/93. CTK envelope (MediaPipe): arm ≤198.8° (2/89 Keraal rejected), hand height ≤2.01 sw (0 rejected). RTK still off. Handoff 'What is validated' table updated. Next: rerun session_report on my_ctk/my_elk/my_elk2 with the new config.
- **22 Nov own videos with v8:** my_ctk overhead still judged hiding face (arm ok, tops 165/155/153°; hand height 1.57 sw < limit 2.01) → limitation: checks test Keraal's errors, not exercise identity (future: hands-near-face shape cue). my_elk2 now 3 reps (13/25/14°), recording ok by design. my_elk 2 reps from arm → findings.py keeps whichever segmentation finds more. Moving on: Groq prompt-v2 LLM runs next (thresholds changed in v8 → all calls new).
- **22 Nov prompt-v2 LLM runs done (both models, 30 recs, 25 with checks):** grounded drafts pass checker 24/25 and 25/25 (v1: 12/39); 1 rejection was a checker false positive ('the right height' → fixed in claims.py). Grounded false alarms on both-Correct 1/8 (= template), corrections where A said Incorrect 10/11 (= template). Pose-only baseline: gpt-oss-20b false alarms 7/8 (88%), addresses 8/11; gpt-oss-120b false alarms 3/8, addresses 6/11 (praises incorrect sets). Grounded checker pass: 20b 24/25, 120b 25/25. LLM drafts rarely quote numbers (20b 4/25, 120b 1/25). Next: template eval rerun (windows) → supervisor report update.
- **22 Nov template rerun (shipped skeleton, v8):** template passes checker 299/299. CTK arms (all-data thresholds): flags 24/26 of A-Error1 (+2 borderline), 11/55 both-Correct (20%); ELK: 31/42 A-Error2 (+4 borderline), 1/15 both-Correct (7%). 102 recs without findings (99 RTK). Windows (v8, clock median 16.9 fps): CTK overlap 45% of 22 vs chance 34% (any rep 61%; windows peak 14 / worst_rep 8), ELK 48% of 29 vs 30% (any rep 77%). Severity agreement 39/55. In-sample vs A: CTK sens 92% spec 80%, ELK 74%/86%. Template false alarms 12/121 (10%). Supervisor report §4.3 updated (CTK arm angle 88/76, ELK kappa 0.51/0.54, LLM v2 table, pose-only failure modes, own-video limitations).
- **Presentation moved to ~2 Dec 2026 (told 21 Nov); deck on hold until asked.**
- **Still for the presentation:** live demo (App lead), Groq comparison numbers, KIMORE mean ± std, related-work page, supervisor report update with Pillar 2.


**Done (14 Nov):** mirroring fix applied; all 5 tests in `tests/test_keraal.py` pass. ELK overlays now show skeleton on the person with correct sides; all BlazePose evaluations rerun (§6e). Supervisor report (§4.2, summary, limitations, trust table) and plan doc updated.
**Done (14 Nov):** quaternion-based Vicon angles run on all 539 recordings (§6f); 6 tests pass. Supervisor report §4.2 (new table vs orientation reference), summary, limitations, methods, trust table v2 updated; plan doc logged.
**Done (15 Nov):** arm-reference diagnostic run on 539 recordings (§6g). **Decision: app arm metric = signed arm angle vs shoulder-line perpendicular (`shoulder_perp`).** Trust table v3; supervisor report + plan doc updated.
**Done (16 Nov):** angle spec + trust table → `angle-spec-trust-table.zip`: `src/rehabform/angles.py` (`arm_elevation` signed vs shoulder-line perpendicular, `elbow_flexion` 2D, `trunk_lean` tilt from vertical + = person's left, `trunk_rotation` depth/timing only; each returns `(values, valid)`, invalid when limb image length < 50% of reference; `TrustTable.decide(measure, exercise, value)`), `config/trust_table.json` (v3: modes range / relative / timing_only / not_reportable per measure × exercise, notes, evidence, overhead rule > 110° → relative, Keraal errors → measures with thresholds null; CTK "arms not outspread" + 3rd variant have no measure yet), `tests/test_angles.py` (4 tests; arm measure equals validated `shoulder_perp`). Plan doc has a human-readable "Angle spec and trust table" section.
**Task in progress (18 Nov):** Groups 1A (`data\keraal1`, 250 recs, patients R1–R6) and 2A (`data\keraal2`, 51 recs, S1–S6) extracted and summarised. Videos 480×360 @ 10 fps, BlazePose frames = video frames. BlazePose MIRRORED in 1A/2A too (6/6 overlays) AND BlazePose left/right labels flip in some recordings after un-mirroring → `canonical_sides` (person's left shoulder+hip on image right; patients face camera); `load_blazepose_g12`. Anvil tracks: Global evaluation (Correct/Incorrect/Incomplete/Motionless), Global error (Evaluation Small/LargeError, type Error1-3/Other/none, bodyPart Spine/BothArms/LeftArm/RightArm), Temporal error (same + window; A gives windows ~always, B rarely). Clinician agreement: 1A 76% (kappa ≈0.59; B stricter: 45 recs A=Correct/B=Incorrect), 2A 92% (kappa ≈0.83). A's labels by exercise (1A): CTK 45C/39I, ELK 32C/41I/10Inc/1M, RTK 58C/7I/17Inc. Delivered `pillar2-error-signals.zip` (`scripts/pillar2_error_signals.py`: labels table, kappa, error type→body part, side fix vs Kinect, recording features, AUC per exercise×error). Waiting for its output. **Signals result (18 Nov):** kappa C-vs-I 0.59 (CTK 0.73, ELK 0.58, RTK 0.50); same error TYPE 78/88 when both say Incorrect (clinicians agree on what, not on how bad). Error map (A): CTK Error1 n=26 arms (wrist height lower AUC 0.85, elbow-bend proxy 0.88) = arms not raised; CTK Error3 n=17 (3 people) elbow-bend 0.81, meaning TBC; CTK Error2 n=1; ELK only Error2 n=43 spine, lean range lower AUC 0.88 = body not tilted; RTK Error1 n=4, RTK Error2 n=6 (too few). Side fix: 14/299 swapped; ELK vs Kinect 68 agree/2 disagree/21 unclear; 71 recs >10% frames reversed (to break down). 2 RTK recs skipped (empty BlazePose). Delivered `pillar2-detectors.zip` (`scripts/pillar2_detectors.py`: LOSO thresholds over 12 people, sens/spec, kappa sys-A/sys-B vs A-B, severity ordering → `config/findings_thresholds.json`). Waiting for output.
**Built 18 Nov (afternoon), `pillar2-feedback.zip`:** findings engine (`findings.py`: thresholds + trust table → findings F1.. with value/typical/severity/time window; numbers quotable only for validated measures), claim checker (`claims.py`: every sentence cites [Fn]; rejects invented numbers, body parts, sides, corrections of passed checks, medical wording), feedback writer (`feedback.py`: template fallback, grounded LLM prompt, pose-only baseline prompt, OpenAI-compatible client with cache), eval (`pillar2_feedback_eval.py`). Order to run: `pillar2_detectors.py` (creates `config/findings_thresholds.json`) → `pillar2_feedback_eval.py` (template) → with `--llm` once an LLM endpoint is chosen (Ollama local or hosted). Detector output not yet seen.
**20 Nov results + fixes (`pillar2-feedback-v2.zip`):** LOSO: CTK arms-not-raised (wrist height) sens 96%/spec 78%, kappa sys-B 0.56 vs clinicians 0.64; ELK body-not-tilted (lean range) bal 86%, kappa sys-A 0.53 / sys-B 0.43 vs clinicians 0.43 (= headline). CTK Error3 + both RTK detectors fail on unseen people → `usable:false` (rule: LOSO bal ≥75%, ≥4 people), skipped by engine. Severity agreement with A 47/64. Template failed checker 42/299 ("looked right" side word; negative heights) → fixed, 720-case sweep passes. Time-window overlap ~0–21% because Anvil times use the original video clock (~15–18 fps vs mp4 10 fps) → rescaled; chance baseline printed. Groq key in `.env` as `API_KEY` (must be gitignored); Groq production models (Nov 2026 docs): openai/gpt-oss-20b, openai/gpt-oss-120b, llama-3.1-8b-instant, llama-3.3-70b-versatile (reported retired Aug 2026 for non-enterprise). Plan: gpt-oss-20b and gpt-oss-120b, --limit 60, --baseline.
**20 Nov rerun:** template passes checker 299/299; template corrections on both-Correct recordings 13/121 (was 58 before unusable detectors were switched off); severity 44/61. Anvil implied original fps median 19.3 (matches G3 Vicon 19.4 Hz) but range 0.2–922 → eval now falls back to 19.3 outside 12–30 fps. Window overlap with A: CTK 50% vs chance 34%, ELK 48% vs 36%, IoU ~0 → windows need per-rep segmentation (next). Groq models available: openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b (no llama). gpt-oss-20b LLM run in progress.
**20 Nov gpt-oss-20b run (partial, 39 call errors):** grounded drafts passed the checker 36/41 (4/59 sentences unsupported = 6.8%: 3 contradicts_finding, 1 untagged); 0 shown after fallback. Errors: baseline prompts > Groq free-tier 8k-token request cap (413) and an RTK no-findings crash → fixed (compact baseline table, max_tokens 2048, no LLM call when no validated checks, paired writer comparison with 'false alarms' on both-Correct recordings). Rerun needed for 20b and 120b (cached calls reused; baseline prompts changed so baseline calls are new).
**20 Nov drafts reviewed:** checker v1 missed meaning errors ("arms were not raised enough" citing a PASSED check; "reach about 1.0" without unit) → contradicts_finding now covers negations + praise-for-error, new number_without_unit rule. Baseline (pose-only) examples: confident wrong advice ("keep your wrists at shoulder height rather than going above" for an arms-raise exercise; "great job lifting both arms above your head" where both clinicians said arms not raised enough; "helps protect the shoulder joint") → good qualitative evidence. Groq free tier: 8k TPM + 200k tokens/day per model (rolling) → ~one baseline call/min, ~40–50 calls/day/model; added --reasoning-effort low. Borderline values (0.8 vs threshold 0.82) give FPs vs both-Correct → consider a 'borderline' status later.
**20 Nov (morning), `pillar2-v3-reps-borderline-video.zip`:** both Groq models hit the 200k/day limit (20b ~half done, 120b ~150/301 processed) → finish when reset. Built: worst-repetition time windows (`segment_reps`, rest = first 1 s), borderline status (clinician-only; margin = smallest giving ≤10% false alarms on both-Correct, capped at half distance to severity cut; also = strictness setting), `video_pose.py` (MediaPipe Tasks on any video → 10 fps pixel landmarks, sides fixed, --mirrored for selfie), `session_report.py` (video → findings → checked feedback → single HTML report: patient text, clinician view with evidence, SVG timeline with reps, key frame), `validate_mediapipe_keraal.py` (own MediaPipe vs shipped BlazePose features/decisions). Grounded prompts change with v3 (windows, borderline) → LLM grounded calls re-run (baseline cached). Next: user runs detectors (borderline sweep), template eval, MediaPipe validation (40 recs), session report on Keraal + own phone video; LLM runs after reset. Presentation ~28 Nov: needs live demo (App lead status unknown) + KIMORE mean±std + related-work page.
**20 Nov night:** MediaPipe model downloaded (curl resumed). First real session_report on G2A-ELK-S1-Roscoff-060: MediaPipe found person in only 44% of frames (shipped BlazePose: all) → all checks skipped. Suspects: anonymised Keraal videos (blurred face, drawn skeleton overlay), small person at 480×360, default 0.5 confidences. Added --min-confidence, --contact-sheet, 'could not see you' patient text, engine.skipped. validate_mediapipe_keraal crashed at summary (None sort) → fixed + landmark cache; rerun pending.
**Then:** findings engine — per-rep measures (angles.py + reps.py) vs thresholds/reference rep → findings; calibrate thresholds and test against 1A/2A doctor labels (precision/recall, body part, time window).
**Also pending:** CTK overlays shared on 14 Nov were the OLD (pre-fix) images — rerun `overlay_blazepose.py --exercise CTK` to confirm alignment there. Groups 1a/2a download (check mirroring with overlay) + Anvil parser (Pillar 2).

---

## 1. One-paragraph summary

A phone app for physiotherapy clinics. The patient exercises at home in front of a phone; the app measures joint angles and range of motion per rep, compares them with clinician targets, and gives two or three specific corrections. Two research pillars: (1) phone measurements validated against Vicon motion capture, and (2) feedback built only from measured findings — a language model rewords findings, a claim checker rejects anything unsupported. Pitch line: "One phone, clinic-grade numbers, and feedback you can verify."

---

## 2. Decision history

1. **Original plan:** TCM on ObjectRelator (semester 1) + OI-ProfNet object-interaction-aware proficiency on Ego-Exo4D (semester 2).
2. **Left it:** best published Ego-Exo4D proficiency ~53% (AdaMVS, 4 classes); 5 synchronised cameras; research-only data; no buyer. FYP panels favour industry projects; industry benefit is the team goal; supervisor allowed free choice.
3. **Procedure verification (IndustReal, Assembly101):** dropped — customers would need own filming and labelling.
4. **Compared physio / hand-hygiene / cricket:** physio chosen (public data with quality labels, clear buyer).
5. **Physio v1:** MediaPipe + per-exercise correct/incorrect classifier; data plan, starter repo, first baseline.
6. **Supervisor feedback (late Oct):** v1 weaker and less ambitious than original. Evidence: "MediaPipe + feedback" physio apps are hackathon-level by May 2026.
7. **Revised direction (31 Oct):** add (2) grounded, checkable language feedback and (3) clinic-grade measurement validated against Vicon.
8. **1 Nov:** "Assess any exercise from one clinician demonstration" kept as a possible expansion, not to be implemented now (group decision via doc comment). Self-supervised pretraining and two-phone fusion also parked.
9. **1 Nov:** KIMORE FCN baseline closed after diagnostics (FCN ≈ majority class). Error detector will start from angle rules.
10. **2 Nov:** Keraal structure confirmed (see §4); angle set changed to upper body/trunk.

---

## 3. Current scope

### Pillar 1 — Measurement validity
- **RQ1:** How closely do single-phone joint angles and range of motion match Vicon during upper-body rehab exercises; which joints, exercises and camera views are reliable enough to report?
- **Data:** Keraal **Group 3** only (3 healthy adults, 540 recordings, correct + simulated errors; video, Kinect, BlazePose and Vicon recorded simultaneously).
- **Angles:** shoulder flexion, shoulder abduction, elbow flexion, trunk lateral bend, trunk rotation (rotation is hard for one 2D camera — report separately).
- **Method:** MediaPipe (and OpenCap Monocular on a subset) on Group 3 video vs the same angles from Vicon; resample Vicon to video frame rate; check residual lag per recording.
- **Metrics:** MAE (degrees) per joint/exercise, ROM error per rep, Bland–Altman limits, ICC, breakdown by view/visibility.
- **Limitation:** 3 subjects → measurement study, not generalisation across bodies. Secondary evidence: KIMORE MediaPipe vs Kinect across 78 people.
- **Product output:** per-joint trust table.

### Pillar 2 — Grounded, checkable feedback
- **RQ2:** Does feedback built from measured biomechanics detect real errors and avoid invented claims, versus a language model describing the pose sequence alone?
- **Data:** Keraal **Groups 1a + 2a** (300 recordings: 249 patient, 51 healthy; Anvil XML labels by a physiotherapy doctor and a physiotherapist: error type, body part, time window; no Vicon).
- **Method:** findings engine (measured vs target, per rep and joint, + error detector) → LLM rewords findings only → claim checker (template fallback).
- **Key insight:** Keraal's 9 errors map largely to angle rules (e.g. "arms not raised enough" → shoulder-flexion threshold; "torso rotation not sufficient" → trunk-rotation range; "body leaned on the side" → lateral lean).
- **Metrics:** detection precision/recall vs labels; body-part and time-window agreement; unsupported-claim rate before/after checker; grounded vs ungrounded comparison; physio usefulness rating if access allows.

### Architecture
- **Phone:** camera + framing guide → MediaPipe pose → angle engine → rep segmenter. Only landmarks/angles leave the phone.
- **Server:** findings engine → feedback writer (LLM) → claim checker → results store → patient screen + clinician dashboard.
- **Inputs to findings engine:** trust table (Pillar 1), clinician targets, error detector (angle rules first; learned model only if it beats the majority baseline).

### Out of scope
Diagnosis/treatment advice, multi-person scenes, medical certification, commercial use (FYP is non-commercial; cite each dataset).

---

## 4. Data

### Keraal (anchor dataset) — confirmed from the paper (arXiv 2407.00521)
| Group | Who | Recordings | Labels | Video | Vicon |
|---|---|---|---|---|---|
| 1a | 6 patients (annotated subset) | 249 | Anvil XML: error, body part, time window | mp4 480×360 | No |
| 1b | 12 patients (rest) | 1,631 | None | mp4 480×360 | No |
| 2a | 6 healthy (annotated subset) | 51 | Anvil XML: error, body part, time window | mp4 480×360 | No |
| 2b | 6 healthy (rest) | 151 | None | mp4 480×360 | No |
| 3 | 3 healthy, correct + simulated errors | 540 | Error label | avi 960×544 | **Yes** |

- All groups: Kinect V2 positions + orientations (tabular); OpenPose/AlphaPose/BlazePose (dictionary; BlazePose = 33 landmarks).
- Vicon skeleton: 17 segments (forearms, arms, chest, hips, head, thighs, shoulders, hands, feet, tibias), positions + orientations.
- Exercises: torso rotation, flank stretch (lateral bend), hiding face (arms flexed 90° at shoulder and elbow).
- Errors (3 per exercise): torso rotation — arms not raised enough / rotation not sufficient / body leaned to the side; flank stretch — opposite arm not along body / body not tilted / upper arm not bent; hiding face — arms not raised enough / arms not outspread enough / arms not raised enough (3rd variant).
- Published baselines: LSTM error classification 53.9 ± 4.8% (torso rotation, best 64.4%), 31.6 ± 6.5% (flank stretch), 49.1 ± 3.3% (hiding face); correct/incorrect detectors label most incorrect reps as correct.
- Size: Group 3 archive ~8.7 GB (`group3.tar.xz`); extracted ~9.5 GB.
- **Group 3 on disk (3 Nov):** `data\keraal\` → `videos` (540 .avi), `vicon` (540 .txt), `kinect` (540 .txt), `blazepose` (540 .json), `openpose` (540 .json), `readme_files_format.txt` (by Maxime Devanne). Every recording has all modalities.
- **BlazePose JSON format:** `{"positions": {"1.0": {"Nose": [x, y, z], "Left_eye_inner": [...], ...}, ...}}` — keyed by frame number (as a string float), named landmarks, normalised x, y and relative z (MediaPipe-style).
- **File names (confirmed by readme):** `G3-<modality>-<exercise>-P<participant>T<trial>-<field>-<label>-<recording>`, e.g. `G3-Vicon-CTK-P1T1-Unknown-C-0.txt`. Modalities: `Anon` (video), `Vicon`, `Kinect`, `BP` (BlazePose), `OP` (OpenPose). Exercise codes CTK, ELK, RTK (same as Rehab-Pile) — which code is which exercise still to confirm by watching a video of each.
- **Folders:** `data\keraal\{videos,vicon,kinect,blazepose,openpose}` (flat, no group subfolder). Other groups use `groupX/<modality>/<exercise>/...` and 1A/2A have an `annotations` folder. Local copies: 1A in `data\keraal1`, 2A in `data\keraal2`.
- **Video:** 10 fps, 960×544 (first recording: 196 frames ≈ 19.6 s). BlazePose: one entry per video frame, keys "1.0" … "196.0".
- **Vicon:** 17 rigid targets (0 RForearm, 1 LForearm, 2 RArm, 3 LArm, 4 Chest, 5 RThigh, 6 LThigh, 7 RShoulder, 8 LShoulder, 9 RHand, 10 LHand, 11 RFoot, 12 LFoot, 13 Hips, 14 Head, 15 RTibia, 16 LTibia); 119 values per row = 17 × (x, y, z, qx, qy, qz, qw); positions in mm, z up (head ≈ 1,210 mm, feet ≈ 120 mm). Targets sit on segments, not joint centres → compare change from start pose and ROM, not absolute angles. Vicon frame rate: to be read from the first comparison run. First frame of the first recording suggests the participant may be seated (thighs near horizontal) — check the video.
- **Kinect:** 25 joints (readme order = standard Kinect v2), 175 values per row = 25 × 7, metres; end joints have zero quaternions.
- **OpenPose:** COCO, 14 joints, 2D. **BlazePose:** 33 named joints, x, y normalised, z = depth relative to video plane.
- **Data sharing rule:** data never goes to git (`.gitignore` excludes `data/`); team shares via Google Drive or a private Kaggle dataset.
- Licence: CC BY-NC-SA (per the Keraal dataset page).
- **Download (moved, 16 Nov 2026):** page https://keraal.enstb.org/KeraalDataset.html (old nguyensmai.free.fr link is dead). Direct archives (http, use curl): `http://keraal.enstb.org/data/group1A.tar.xz` (1.4 GB), `group2A.tar.xz` (338 MB), `group1B.tar.xz` (13 GB), `group2B.tar.xz` (1.3 GB), `group3.tar.xz` (8.2 GB), `readme_files_format.txt`; sample zip `https://keraal.enstb.org/data/keraal_sample_2022.zip`. Code: https://github.com/nguyensmai/KeraalDataset
- **Confirmed by the dataset page:** CTK = hiding face, ELK = flank stretch, RTK = torso rotation; Group 3 labels = error number + body part (e.g. `E1B1` = Error1 Bodypart1). Anvil annotations (1A/2A): correct/incorrect (codes 4 = incomplete, 5 = motionless), error label + significant/small, body part, time window.

### Other datasets
| Dataset | Role | Access |
|---|---|---|
| KIMORE | Secondary measurement check (MediaPipe vs Kinect, 78 people); clinician scores | vrai.dii.univpm.it/node/204704 |
| UI-PRMD | Vicon + Kinect, correct vs `_inc` incorrect files; angle targets; no RGB | webpages.uidaho.edu/ui-prmd |
| FLEX | Expert feedback text for wording comparison | github.com/HaoYin116/FLEX (non-commercial) |
| Rehab-Pile | Benchmark folds via aeon (auto-download) | `load_rehab_pile_dataset(name, split, fold, extract_path)` |

**Rehab-Pile facts:** KIMORE_CLF_BN label = clinician score ≤ 50 → 0, > 50 → 1; one sample = one person's whole session (several reps); 18 Kinect joints × 3D; lengths Sq 557, LA 725, LT 798, TR 813, PR 847; 5 folds, 64–65 train / 6–7 test.

---

## 5. Environment and code

- **Local:** Windows, `cmd` shell; repo at `T:\n-a\fyp\rehab-form`; Python 3.11 venv `.venv`.
- **GitHub:** https://github.com/tayyabiziz-8/rehab-form
- **Kaggle:** Session options → GPU + **Internet On** (needs phone-verified account; "Could not resolve host" = internet off). Install only `pip install -q aeon` (PyTorch preinstalled; do not install requirements.txt). Kaggle runs Python 3.13. Long runs: Save Version → Save & Run All.
- **Repo contents:** `src/rehabform/skeleton.py` (13 shared joints, normalisation, angles), `reps.py`, `models.py` (FCN); `scripts/train_rehab_pile.py`, `extract_mediapipe.py`, `check_datasets.py`; `tests/test_core.py`.
- **Changes applied:** label `.ravel()` + `to_3d()` guard; LR schedule (`ReduceLROnPlateau` 0.5/50/1e-4); per-fold diagnostics (test label counts, predicted counts, majority-baseline accuracy).
- **New (3 Nov):** `src/rehabform/keraal.py` (file-name parser, `find_recordings`, `load_vicon`, `load_kinect`, `load_blazepose`) and `scripts/compare_vicon.py` (BlazePose 2D/3D vs Vicon: arm_elev_L/R, elbow_L/R, trunk_lat, trunk_rot; resample Vicon to video time, change-from-start, lag search ±1 s, r / MAE / ROM error; `--plot`, `--all` → `outputs/keraal_vicon_compare.csv`). Tested on synthetic data only; needs `pandas` and `matplotlib`.
- **Teammate's scripts (8 Nov):** `evaluate_keraal.py`, `compare_keraal.py`, `compare_keraal_modalities.py`, `align_keraal_modalities.py`, `diagnose_keraal_mapping.py`, `evaluate_keraal_kinect.py`, `diagnose_shoulder_angles.py`; their `keraal.py` (`iter_recordings`, `load_blazepose`, `load_vicon_positions`, `load_kinect_positions`, `vicon_to_shared`, `resample_sequence`, `shift_sequence`); `skeleton.joint_angles` gained `vertical_axis` / `vertical_positive_up`. Outputs in `outputs/keraal_*.csv`.
- **Review fixes (8 Nov):** `load_blazepose(path, image_size=(960, 544))` now returns pixel units (x·W, y·H, z·W) and orders landmarks by name (`BLAZEPOSE_ORDER`); pass `image_size=None` for raw, `(480, 360)` for Groups 1a/1b/2a/2b. Synthetic check: absolute shoulder comparison vs Vicon shows ~21° even when phone and Vicon agree perfectly (fixed offset). Claude's earlier standalone `keraal.py`/`compare_vicon.py` are superseded by the teammate's loader + `evaluate_keraal_vicon_offsetfree.py`.
- **Still to do in code:** pooled metrics across folds; Anvil XML parser for Groups 1a/2a; angle spec implementation for upper body; UI-PRMD/KIMORE joint orders; fix `data/README.md` Rehab-Pile line.
- **cmd loop:** `for %e in (LA LT PR TR) do python scripts\train_rehab_pile.py --dataset KIMORE_clf_bn_%e --extract-path data\rehab_pile` (`%%e` in .bat).

---

## 6. Results

| Run | Setup | Avg accuracy | Avg balanced acc |
|---|---|---|---|
| Run 1 (21 Oct) | FCN, 200 epochs, batch 32, fixed LR, CPU | 63.0% | 65.1% |
| Run 2 (1 Nov) | FCN, 1,500 epochs, batch 16, LR schedule, Kaggle GPU | 64.3% | 62.6% |

Per exercise, run 2: LA 68.6, LT 74.3, PR 58.6, Sq 74.3, TR 45.7 (accuracy %).

**Diagnostics (1 Nov, 200 epochs, seeds 0 and 1):**
| Exercise | Seed 0 | Seed 1 | Majority baseline |
|---|---|---|---|
| LA | 68.6% | 65.2% | 71.9% |
| TR | 69.0% | 59.0% | 62.9% |

- LA: FCN predicts all "correct" in folds 0, 2, 3 → identical results across runs; worse than majority baseline.
- TR: weak signal (balanced acc ~0.58–0.71 on two-class folds), seed swing up to 10 points.
- LA folds 1, 4 and TR fold 4 contain one class only → per-fold balanced accuracy inflated.
- Long training worsened TR (overfitting; model selected by lowest training loss).
- **Conclusion:** with ~65 training people, learned classifiers ≈ class balance + noise. Rules for reporting from now on: always show majority baseline; pool predictions across folds; ≥2 seeds.

### 6b. Keraal Group 3 — first evaluation (teammate, 8 Nov; preliminary, has known issues)

Source: `docs/Keraal Group 3 - Initial Angle Evaluation.md` in the repo. Absolute-angle MAE per recording, averaged per exercise (180 recordings each).

| Comparison | Exercise | L elbow | R elbow | L shoulder abd | R shoulder abd | Trunk lean |
|---|---|---|---|---|---|---|
| BlazePose vs Vicon | CTK | 16.2 | 23.3 | 22.7 | 25.8 | 27.4 |
| | ELK | 55.7 | 25.2 | 76.0 | 81.5 | 21.0 |
| | RTK | 58.5 | 19.5 | 23.6 | 23.0 | 15.5 |
| Kinect vs Vicon | CTK | 19.4 | 17.9 | 22.6 | 25.8 | 20.0 |
| | ELK | 51.2 | 22.2 | 23.2 | 26.9 | 12.3 |
| | RTK | 54.0 | 21.4 | 13.6 | 15.0 | 9.3 |
| BlazePose vs Kinect | CTK | 26.6 | 29.8 | 19.6 | 19.7 | 24.1 |
| | ELK | 23.7 | 24.2 | 79.8 | 82.3 | 20.0 |
| | RTK | 27.8 | 27.3 | 22.4 | 20.7 | 20.3 |

Teammate's diagnostics: whole-frame lag negligible (median −1 frame, 0.03° gain); global Vicon left/right swap rejected; authors' MATLAB code has no Vicon-to-anatomy conversion; three shoulder definitions tested, torso-relative 3D best.

**Review findings (Claude, 8 Nov) — fix before reporting:**
1. **Aspect ratio:** BlazePose x/y are normalised by 960/544 separately and never converted to pixels → oblique angles distorted (e.g. a true 45° reads ~30°). Fix: x, z × 960, y × 544 before any angle.
2. **Vicon offsets:** Vicon ART targets treated as joints (elbow = Arm target, wrist = Forearm target, both hips = single Hips target) and compared on absolute angles → built-in fixed offsets. Need offset-free comparison (change from start, ROM) with segment vectors before demoting Vicon.
3. **Kinect is not ground truth:** BlazePose-vs-Kinect = agreement, not accuracy; cannot replace Vicon for the Pillar 1 claim. Fine as a development/debug reference.
4. **Shoulder diagnostic bug:** `diagnose_shoulder_angles.py` uses image "up" (−y) for Kinect too (Kinect y is up) → vertical-reference candidate measured upside-down for Kinect; rerun after fixing.
Also: their `load_blazepose` relies on JSON key order matching the canonical 33-landmark order (only checks the set) — reorder by name; two Keraal loaders now exist (theirs `iter_recordings`, Claude's `find_recordings`) — `compare_vicon.py` won't run against theirs until merged.

**Teammate's good practice to keep:** triangulating with Kinect to separate reference errors from phone errors; not changing production definitions on diagnostics; documenting everything; checking the authors' code and licence.

### 6c. Corrected rerun (after 8 Nov fixes) — first real Pillar 1 numbers

Offset-free BlazePose vs Vicon (change from start pose; ROM = p95 − p5), 180 recordings per exercise:

| Measure | Exercise | Median r | Median change MAE | ROM Vicon vs phone |
|---|---|---|---|---|
| Sideways trunk lean (2D) | ELK | 0.99 (IQR 0.98–1.00) | 2.4° | 46° vs 41° |
| Trunk rotation (3D) | RTK | 0.98 (IQR 0.98–0.99) | 15.4° | 87° vs 132° |
| Arm elevation L (3D) | RTK | 0.74 | 20.5° | 55° vs 87° |
| Arm elevation L (3D) | CTK | 0.57 | 29.3° | 53° vs 79° |
| Arm elevation L (3D) | ELK | −0.52 (all labels, all participants) | 81.2° | 128° vs 134° |
| Elbow (2D/3D) | all | −0.40 to 0.57 | 21–50° | CTK 2D phone ROM ~2× Vicon |

- Arm elevation by participant (3D, median r): P1 0.88 CTK / 0.92 RTK; P2 0.30 / 0.64; P3 0.31 / 0.52.
- ELK anti-correlation is not a simple L/R label swap (ROM-based swap test: no improvement). Hypotheses: unsigned angle folding near 180° with differing trunk references; BlazePose failing on overhead arms (ELK also 72–85° off vs Kinect in every definition).
- Vicon sampling ≈ 19.5 Hz (rows ≈ 2× video frames). Median lag 0 frames. Outlier: `G3-CTK-P2T3-Unknown-E3B3-0` (91 video frames vs 472 Vicon rows) → exclude.
- Sign conventions: ELK trunk_lat and RTK trunk_rot always flipped (consistent convention difference, handled).
- **Exercise codes inferred from data:** ELK = flank stretch (lateral bend), RTK = torso rotation, CTK = hiding face. Confirm by video.
- BlazePose vs Kinect after fixes (absolute MAE): elbows 28–37° (worse than pre-fix, depth now weighted correctly → depth is the weak link); shoulder torso-3D CTK 24–26, ELK 74–79, RTK 20; vertical-3D shoulder now best for CTK (20–22°) and RTK (15–17°); torso-2D improved (CTK 35, RTK 24–28); trunk lean 30–33°.
- **Emerging trust table:** report sideways trunk lean; trunk rotation for timing/relative only; arm elevation with confidence note; not elbow or overhead arm (single front camera) until follow-ups.

### 6d. Mirrored BlazePose (14 Nov)
- Signed frontal-plane arm angle results before the fix: ELK r −0.57 / −0.62 (all participants negative); CTK degenerate (arms forward → 360° unwrap jumps in 39% of recordings); RTK 0.35–0.38.
- Overlays: skeleton drawn mirrored about image centre (P1 sits left of centre → skeleton drawn right of person); in ELK the raised arm is labelled with the wrong side.
- All earlier BlazePose arm/shoulder/elbow numbers (§6b, §6c) are affected by the L/R swap; trunk lean and trunk rotation are not (side-independent).

### 6e. After un-mirroring (14 Nov) — current best numbers

Offset-free BlazePose vs Vicon (539 recordings; one skipped for length mismatch). Median r / median change MAE / mean ROM err (phone − Vicon):

| Measure | CTK | ELK | RTK |
|---|---|---|---|
| Trunk lean (2D) | 0.27 / 4.5° (small motion) | **0.99 / 2.4° / −5°** | 0.74 / 8.1° |
| Trunk rotation (3D) | 0.30 | 0.68 | **0.98 / 15.4° / +37°** |
| Arm elev L/R (2D) | 0.51 / 0.38 | **0.99 / 0.99; 12.5° / 17.8°; +28 / +39°** | 0.47 / 0.51 |
| Arm elev L/R (3D) | 0.55 / 0.39; 26° / 35° | 0.96 / 0.94 | 0.69 / 0.72 |
| Signed frontal arm L/R (2D) | 0.80 / 0.74 (ROM unreliable: gating) | **1.00 / 0.99; 12° / 21°** | 0.70 / 0.76; 16° / 20° |
| Elbow L/R (2D) | 0.44 / 0.40 | **0.73 / 0.67; ROM ±3°** | −0.18 / −0.19 |
| Elbow L/R (3D) | −0.35 / −0.40 | 0.45 / 0.20 | 0.43 / 0.51 |

- By participant, signed arm r: P1 0.99/0.99; P2 0.86/0.76; P3 0.77/0.67.
- BlazePose vs Kinect after un-mirroring: ELK shoulder 74–79° → **18–26°**; elbows 28–36°; trunk lean ~30–33° (absolute, depth-dependent).
- Shoulder diagnostic vs Kinect: vertical-3D best overall (CTK 18.2–18.5, ELK 19.3–20.5, RTK 8.1–10.8); torso-2D best for ELK left (13.4).
- Phone ROM > Vicon ROM for sideways arms (+28–66°) while r ≈ 0.99 → likely Vicon proxy (shoulder target moving with arm) understates; test with quaternions.
- **First trust table:** report sideways trunk lean; sideways arm elevation/frontal arm angle (ranges may read larger than lab); trunk rotation timing/relative only; elbow only for sideways movements; nothing while arm points at camera (CTK forward raise).

### 6f. Vicon angles from segment orientations (14 Nov) — now the primary Vicon reference

Joint rotation from quaternions (axis-free, mounting-independent). Median per recording; ROM differences relative to quaternion Vicon.

| Joint | Ex | axis share | ROM phone / target / quat | r phone~quat | phone − quat | target − quat |
|---|---|---|---|---|---|---|
| Arm L/R | CTK | 0.76 | 91–96 / 55–65 / 94–99 | 0.91 / 0.88 | −5 / −3 | −39 / −21 |
| Arm L/R | RTK | 0.88–0.89 | 81–90 / 51–57 / 74–78 | 0.77 / 0.76 | +4 / −1 | −23 / −37 |
| Arm L/R | ELK | 0.92–0.94 | 185–192 / 141–148 / 129–142 | 0.99 / 0.97 | +45 / +56 | +4 / +4 |
| Elbow L/R | ELK | 0.81–0.83 | 65–70 / 57–72 / 58–65 | 0.73 / 0.72 | 0 / +9 | −5 / +14 |
| Elbow L/R | CTK | 0.98 | 110–115 / 41–50 / 48–52 | 0.55 / 0.56 | +62 / +58 | −4 / −3 |
| Elbow L/R | RTK | 0.99–1.00 | 49–64 / 52–53 / 55 | 0.26 / 0.29 | +5 / −6 | −10 / −3 |
| Trunk lat | ELK | 0.88 | 41 / 46 / 66 | 0.96 | −25 | −22 |
| Trunk rot | RTK | 0.91 | 132 / 87 / 56 | 0.98 | +74 | +34 |

- **Hypothesis confirmed for CTK/RTK arms:** target-position Vicon understated by 21–39°; phone is within ~5° of quaternion Vicon. Earlier "CTK weak" was mostly a reference artefact.
- **ELK overhead reach:** phone overstates +45–56° (both Vicon methods agree) → open question; candidate cause: phone uses hip–shoulder midline as trunk reference while quaternion uses chest orientation.
- **Elbow:** ok in ELK; CTK foreshortening (2×); RTK timing poor.
- **Trunk:** lateral lean — phone ≈ tilt-from-vertical (target method), 25° below chest-vs-pelvis; rotation — phone > 2× spine rotation (depth).
- By participant (r phone~quat, arms): CTK P1 0.97 / P2 0.84–0.90 / P3 0.70–0.72; RTK P1 0.93–0.97 / P2 0.50–0.74 / P3 0.51–0.77; ELK 0.89–0.99 all.
- **Trust table v2:** arm elevation (forward & sideways, below shoulder height) — report range (±5°); overhead — timing/relative only; elbow — sideways only; trunk lean — as tilt from vertical; trunk rotation — timing only; no elbow when forearm points at camera.

### 6g. Arm angle trunk reference (15 Nov)

Phone signed arm angle vs quaternion Vicon (chest→arm); both sides pooled; medians.

| Ex | Reference | r | change MAE | ROM phone / Vicon | diff | |diff| |
|---|---|---|---|---|---|---|
| CTK | midline | 0.90 | 19.7 | 93 / 97 | −3 | 18.8 |
| CTK | **shoulder_perp** | 0.90 | 19.1 | 94 / 97 | −4 | 17.8 |
| CTK | vertical | 0.92 | 15.8 | 91 / 97 | −5 | 15.3 |
| CTK | midline_3d | 0.56 | 103.5 | 193 / 97 | +101 | 105.7 |
| ELK | midline | 0.98 | 18.6 | 188 / 135 | +51 | 51.1 |
| ELK | **shoulder_perp** | 0.98 | 14.0 | 159 / 135 | **+26** | 29.4 |
| ELK | vertical | 0.97 | 27.2 | 218 / 135 | +86 | 86.3 |
| ELK | midline_3d | 0.96 | 34.9 | 170 / 135 | +37 | 49.2 |
| RTK | midline | 0.76 | 16.9 | 87 / 75 | +1 | 11.5 |
| RTK | **shoulder_perp** | 0.78 | 15.9 | 82 / 75 | −2 | 10.1 |
| RTK | vertical | 0.74 | 17.4 | 88 / 75 | +4 | 13.1 |
| RTK | midline_3d | 0.64 | 49.0 | 71 / 75 | −2 | 30.8 |

- ELK shoulder_perp by side: L +22, R +28; by participant: P1 +8, P2 +36, P3 +29.
- Interpretation: about half the ELK overstatement came from the trunk reference (hip–shoulder midline bends with the trunk); remaining ~25° likely BlazePose shoulder point rising with the arm. Vertical worst (adds trunk lean); 3D unstable (BlazePose depth).
- **Trust table v3:** arm elevation vs shoulder line — report range for forward/sideways raises below shoulder height (median error < 5°, typical per-recording 10–18°); overhead — timing/relative, ranges ~25° high; elbow — sideways only; trunk lean — tilt from vertical; trunk rotation — timing only; no elbow when forearm points at camera.

**FCN notes:** 3 conv blocks (128/256/128; kernels 8/5/3) + global average pooling + linear; receptive field ~14 frames; class activation maps possible for "when".

---

## 7. Four-week plan (2–28 Nov 2026)

| Week | Measurement (Pillar 1) | Feedback (Pillar 2) | App | Data & baseline |
|---|---|---|---|---|
| 1 (2–8 Nov) | Keraal download, Group 3 inventory, angle spec | Findings schema, error-to-rule mapping, read BioCoach | Browser camera + MediaPipe + one live angle | Pooled metrics, Keraal Anvil XML loader |
| 2 (9–15 Nov) | MediaPipe vs Vicon, alignment | Findings engine on Keraal reps | Live angles + rep count | Error detector baseline (rules + simple classifier) |
| 3 (16–22 Nov) | Error per joint, OpenCap subset | LLM writer + claim checker | Feedback screen, targets form | Evaluation scripts |
| 4 (23–28 Nov) | Trust table, figures | Faithfulness + detection results | End-to-end demo + recording | Slides, report, tables |

Week-1 checkpoint: usable Group 3 count, angle spec, error-to-rule mapping, live angle in browser. Internal go/no-go end of week 2.

**Presentation deliverables:** phone-vs-Vicon error table + trust table; detection precision/recall + unsupported-claim rate; live demo; KIMORE finding (FCN ≈ majority); related-work page.

---

## 8. Literature

| Paper | Relevance |
|---|---|
| OpenCap Monocular (arXiv 2603.24733, Mar 2026) | Single-phone 3D kinematics validated vs mocap; open source; Pillar 1 comparison |
| BioCoach (CVPR 2026, arXiv 2603.26938) | Biomechanics-grounded VLM coaching; Pillar 2 inspiration |
| Agentic AI for Personalized Physiotherapy (arXiv 2604.21154) | Rule-guarded LLM feedback; VLMs weak on fine-grained motion |
| Keraal (arXiv 2407.00521, IJCNN 2024) | Anchor dataset; four challenges: assessment, error recognition, spatial and temporal localisation |
| Rehab-Pile (arXiv 2507.21018 v2) | 60-dataset benchmark; LITEMV best (~75.5%); one model per exercise |
| SSL-Rehab (CVIU 2025) | Self-supervised pretraining; parked |
| LiFT (arXiv 2506.06480) | Recognition + rep counting across many exercises |
| ISPRS 2025 instructor–patient distance | Prior art for parked "any exercise" idea |
| Portable Biomechanics Laboratory (arXiv 2507.08268) | Handheld-phone kinematics in clinical populations |
| SkillFormer, AdaMVS | Original plan; AdaMVS relevant to parked two-phone fusion |

---

## 9. Docs

- **Revised FYP Direction** (living plan doc, has progress log): https://claude.ai/artifact/NpSEuDjSYgiwGwRfcvXfj5
- **App Lead Handoff — Physio Feedback App** (20 Nov): https://claude.ai/artifact/Ki6WreTBdcy9h3xdGH4YYT
- **Related Work — one page** (21 Nov): https://claude.ai/artifact/JoXR293nMZBfqkqZAfsYEF
- **FYP Progress Report for the supervisor** (8 Nov; §4.2 numbers are pre-correction, update after rerun; names to be added): https://claude.ai/artifact/6jT2yV1pKsKTGJ8Deo4MoF
- FYP Meeting Brief (22 Oct): https://claude.ai/artifact/DwWRVP3UMPgFpxg4Wrv3oc
- Development Kickoff Pack (19 Oct, partly superseded): https://claude.ai/code/artifact/c7ff0203-9430-43fd-b89f-e87503a0e761
- `rehab-form.zip` starter repo; `run1_200ep.csv`, `run2_1500ep.csv`, `diag.csv` results.

---

## 10. Open items

- [ ] Assign the four role owners.
- [x] Confirm presentation date: ~2 Dec 2026.
- [x] Download Keraal Group 3 (540 usable recordings).
- [x] Download Keraal Groups 1a and 2a (in `data\keraal1`, `data\keraal2`, 16 Nov).
- [ ] Finish Groq LLM runs (20b, 120b) after the daily reset.
- [ ] MediaPipe miss rate on Keraal videos: contact sheet + `--min-confidence 0.3`; own phone recordings.
- [ ] App lead: starter running, phone demo over HTTPS, fallback demo video (by 25 Nov).
- [ ] Update the supervisor report with Pillar 2 results.
- [ ] Ask supervisor for a physiotherapist contact.
- [ ] Check OpenCap Monocular licence and compute before week 3.
- [ ] Update kickoff doc and repo `data/README.md` to the revised direction.

---

*Update this file at the end of every working session. Upload the latest version to the project's files so new chats start from here.*
