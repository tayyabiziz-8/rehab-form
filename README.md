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
```

Download the `.task` model file from the MediaPipe Pose Landmarker guide into `models/` first.

## Known TODOs

- [ ] Fill `UI_PRMD_22` and `KIMORE_18` joint orders in `skeleton.py` from each dataset's documentation. Do not guess them.
- [ ] Confirm that UI-PRMD keeps incorrect reps in folders with "incorrect" in the path (`check_datasets.py` assumes this).
- [ ] For LITEMV, the Rehab-Pile winner, use the authors' official code rather than re-implementing it.
- [ ] Compare MediaPipe output on 10 Keraal videos with Keraal's shipped BlazePose skeletons.

## Rules for results

- Always report mean ± std over all folds, plus balanced accuracy. Datasets are small (often 70–300 samples per exercise), and single runs mislead.
- Never select models on the test fold. `train_rehab_pile.py` selects on training loss only.
- Models trained on Keraal, KIMORE, Fitness-AQA or FLEX are for research results only until licences are confirmed (see `data/README.md`).
