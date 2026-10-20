# Data: where to get each dataset

Access pages were checked on 19 Oct 2026. Nothing here is committed to git; every dataset goes in its own folder under `data/`.
Run `python scripts/check_datasets.py data/<name>` right after each download and paste the output into the team's dataset notes.

| Folder | Dataset | Get it from | Licence | Ship in product? |
|---|---|---|---|---|
| `data/keraal` | Keraal (RGB video, Kinect, BlazePose skeletons, error type + body part + time window) | http://nguyensmai.free.fr/KeraalDataset.html and https://github.com/nguyensmai/KeraalDataset | Not stated in paper; check repo | Ask authors |
| `data/rehab_pile` | Rehab-Pile (60 per-exercise datasets, fixed folds) | https://msd-irimas.github.io/pages/DeepRehabPile/ , then load with aeon | Each source repo's licence | Depends on source |
| `data/ui_prmd` | UI-PRMD (10 subjects x 10 exercises, correct/incorrect) | https://webpages.uidaho.edu/ui-prmd/ | ODC Public Domain | Yes |
| `data/irds` | IntelliRehabDS (29 subjects, 9 exercises, Kinect CSV) | https://zenodo.org/records/4610859 | CC BY 4.0 | Yes, with attribution |
| `data/kimore` | KIMORE (78 subjects, 5 exercises, RGB + depth + skeleton, clinician scores) | https://vrai.dii.univpm.it/node/204704 | "Free", terms not stated | Ask authors |

## Download order and notes

1. **Rehab-Pile** first: it gives benchmark numbers in a day. Follow the step-by-step instructions on the project page, unpack into `data/rehab_pile`, then:
   `python scripts/train_rehab_pile.py --list` and `--dataset <name> --extract-path data/rehab_pile`.
2. **Keraal**: the main training set. Its BlazePose skeletons match MediaPipe, our app's pose model. Record which subset (Group 1a, 1b, 2a, 2b, 3) each file belongs to; only 1a and 2a have body-part and time-window labels.
3. **UI-PRMD**: small, public domain, safe to ship. The segmented Kinect positions are the files we need.
4. **IntelliRehabDS**: the Zenodo record holds about 41 GB of large archives plus one ~199 MB file. Check the file list first; the skeleton CSVs are probably the small file and the large archives depth data. Download only what you need.
5. **KIMORE**: if the full archive fails, download each sub-folder separately (CG/Expert, CG/NotExpert, GPP/BackPain, GPP/Parkinson, GPP/Stroke), as the authors advise.

## Emails to send in week 1

Ask the Keraal and KIMORE authors (contacts on their pages) one question: may a student team train a model on the dataset and use it in a product sold to physiotherapy clinics, with citation? Keep the replies in this folder.

## Comparison-only datasets (do not ship models trained on these)

- Fitness-AQA: https://zenodo.org/records/7310289 (non-commercial)
- FLEX: https://github.com/HaoYin116/FLEX (CC BY-NC-SA 4.0)
- EgoExo-Fitness: https://huggingface.co/datasets/Lymann/EgoExo-Fitness (frames and CLIP features; raw video on request)
