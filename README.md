# rehab-form — FYP starter kit

Phone-based home-exercise checker for physiotherapy clinics: count reps, score each rep, and name the error, the body part and when it happened.

This repo is the starting point for the ML side: loading datasets into one skeleton format, reproducing a published baseline, and extracting MediaPipe skeletons from video.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_core.py        # should print 5 "ok" lines
```

Use Python 3.10–3.12. MediaPipe wheels lag behind new Python releases; if `pip install mediapipe` fails, create the venv with Python 3.11.

## What is here

| Path | What it does | Status |
|---|---|---|
| `src/rehabform/skeleton.py` | Maps MediaPipe (33) and Kinect v2 (25) skeletons to 13 shared joints; normalises (mid-hip centre, torso scale); computes joint angles and trunk lean | Tested |
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
| `scripts/evaluate_keraal_vicon_offsetfree.py` | BlazePose vs Vicon on change from start pose and range of motion (cancels ART-target offsets); by exercise and participant | Tested on synthetic data; run on Group 3 |
| `tests/test_keraal.py` | Checks BlazePose name ordering, pixel conversion and the vertical-axis fix on a synthetic Keraal folder | Passing |
| `docs/Keraal Group 3 - Initial Angle Evaluation.md` | Obsidian-compatible record of the initial phone-vs-Vicon evaluation | Pre-fix results; rerun pending (see `CHANGES-2026-11-08.md`) |
| `data/README.md` | Where to download each dataset, licences, what may ship in the product | — |

"Not yet run" means PyTorch, aeon and MediaPipe were not available where this was written. Expect small fixes on the first run.

## Run order

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
```

Download the `.task` model file from the MediaPipe Pose Landmarker guide into `models/` first.

## Known TODOs

- [ ] Fill `UI_PRMD_22` and `KIMORE_18` joint orders in `skeleton.py` from each dataset's documentation. Do not guess them.
- [ ] Confirm that UI-PRMD keeps incorrect reps in folders with "incorrect" in the path (`check_datasets.py` assumes this).
- [ ] For LITEMV, the Rehab-Pile winner, use the authors' official code rather than re-implementing it.
- [ ] Compare MediaPipe output on 10 Keraal videos with Keraal's shipped BlazePose skeletons.

## Rules for results

- BlazePose angles must be computed in pixel units (`load_blazepose` does this by default; Group 3 = 960 × 544, other groups = 480 × 360).
- Against Vicon, report change from start pose and range of motion, not absolute angles.
- BlazePose vs Kinect is agreement, not accuracy.

- Always report mean ± std over all folds, plus balanced accuracy. Datasets are small (often 70–300 samples per exercise), and single runs mislead.
- Never select models on the test fold. `train_rehab_pile.py` selects on training loss only.
- Models trained on Keraal, KIMORE, Fitness-AQA or FLEX are for research results only until licences are confirmed (see `data/README.md`).
