"""KIMORE (Rehab-Pile) baseline, reported the honest way: every exercise, every fold, several seeds,
predictions pooled across folds, next to the majority-class baseline.

Why: on 1 Nov the FCN looked like 63-64% accuracy, but per-fold diagnostics showed it mostly predicts the
majority class (LA: worse than majority). This script makes that visible in one table for the presentation.

Methods (choose with --methods)
  majority   predicts the training majority class (the bar every model must clear)
  logreg     logistic regression on per-channel summary statistics (mean, std, min, max, range); fast, CPU
  minirocket aeon MiniRocket + ridge classifier; a strong, fast time-series baseline (CPU is fine)
  fcn        the repo's FCN (src/rehabform/models.py), same recipe as train_rehab_pile.py (GPU recommended)

Reported per exercise and method
  pooled accuracy / balanced accuracy = all test predictions of all folds together (headline:
  single-class folds make per-fold balanced accuracy meaningless), mean +/- std over seeds;
  per-fold accuracy mean +/- std; how often the model predicted only one class.

Examples
  python scripts/kimore_benchmark.py --extract-path data/rehab_pile --methods majority,logreg,minirocket
  python scripts/kimore_benchmark.py --extract-path /kaggle/working/rehab_pile --methods fcn --seeds 0 1 2 --epochs 200
Outputs: outputs/kimore_benchmark_predictions.csv, outputs/kimore_benchmark_summary.csv, outputs/kimore_benchmark.md
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def kimore_names(extract_path: str | None) -> list[str]:
    import aeon.datasets as ad
    names = sorted(n for n in ad.load_rehab_pile_classification_datasets() if n.upper().startswith("KIMORE"))
    if not names:
        raise SystemExit("No KIMORE classification datasets listed by aeon (check aeon version / --list).")
    return names


def load_fold(name: str, fold: int, extract_path: str | None):
    from train_rehab_pile import load_fold as _load
    return _load(name, fold, extract_path)


# ------------------------------------------------------------------ methods: each returns test predictions
def m_majority(X_tr, y_tr, X_te, seed):
    vals, counts = np.unique(y_tr, return_counts=True)
    return np.full(len(X_te), vals[np.argmax(counts)])


def _summary(X):
    return np.concatenate([X.mean(2), X.std(2), X.min(2), X.max(2), X.max(2) - X.min(2)], axis=1)


def m_logreg(X_tr, y_tr, X_te, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                                                             random_state=seed))
    return clf.fit(_summary(X_tr), y_tr).predict(_summary(X_te))


def m_minirocket(X_tr, y_tr, X_te, seed):
    try:
        from aeon.classification.convolution_based import MiniRocketClassifier
        clf = MiniRocketClassifier(random_state=seed)
    except ImportError:                                    # older aeon
        from aeon.classification.convolution_based import RocketClassifier
        clf = RocketClassifier(rocket_transform="minirocket", random_state=seed)
    return clf.fit(X_tr.astype(np.float32), y_tr).predict(X_te.astype(np.float32))


def m_fcn(X_tr, y_tr, X_te, seed, epochs=200, batch=32, lr=1e-3):
    import torch
    from sklearn.preprocessing import LabelEncoder
    from torch import nn

    from rehabform.models import FCN
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = LabelEncoder().fit(y_tr)
    ytr = enc.transform(y_tr)
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = torch.tensor((X_tr - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((X_te - mu) / sd, dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FCN(Xtr.shape[1], len(enc.classes_)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50, min_lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best, state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(dev)), Ytr[idx].to(dev))
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step(tot / len(Xtr))
        if tot / len(Xtr) < best:                         # model selection on TRAIN loss (test untouched)
            best, state = tot / len(Xtr), {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(dev)).argmax(1).cpu().numpy()
    return enc.inverse_transform(pred)


METHODS = {"majority": m_majority, "logreg": m_logreg, "minirocket": m_minirocket, "fcn": m_fcn}


def acc(y, p):
    return float(np.mean(y == p)) if len(y) else float("nan")


def bacc(y, p):
    cls = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in cls])) if len(cls) else float("nan")


def ms(xs) -> str:
    xs = np.asarray([x for x in xs if np.isfinite(x)])
    return f"{100 * xs.mean():.1f} ± {100 * xs.std():.1f}" if len(xs) else "–"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-path", default=None)
    ap.add_argument("--datasets", nargs="*", help="default: every KIMORE classification dataset aeon lists")
    ap.add_argument("--methods", default="majority,logreg,minirocket")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-folds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=200, help="fcn only")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}; choose from {list(METHODS)}")
    names = args.datasets or kimore_names(args.extract_path)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    preds = []                                            # one row per test case x method x seed
    for name in names:
        folds = []
        for fold in range(args.max_folds):
            try:
                folds.append(load_fold(name, fold, args.extract_path))
            except Exception as e:
                if fold == 0:
                    print(f"[skip] {name}: {e}")
                break
        print(f"\n{name}: {len(folds)} folds")
        for m in methods:
            seeds = [0] if m == "majority" else args.seeds
            for seed in seeds:
                t0 = time.time()
                for k, (X_tr, y_tr, X_te, y_te) in enumerate(folds):
                    fn = METHODS[m]
                    p = fn(X_tr, y_tr, X_te, seed, epochs=args.epochs) if m == "fcn" else fn(X_tr, y_tr, X_te, seed)
                    for i, (yt, yp) in enumerate(zip(y_te, p)):
                        preds.append({"dataset": name, "method": m, "seed": seed, "fold": k, "case": i,
                                      "y_true": yt, "y_pred": yp})
                print(f"  {m:<10} seed {seed}: {time.time() - t0:.0f} s")

    with open(out / "kimore_benchmark_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]))
        w.writeheader()
        w.writerows(preds)

    # ---------------------------------------------------------------- summary
    by = defaultdict(list)
    for r in preds:
        by[(r["dataset"], r["method"], r["seed"])].append(r)
    summ = []
    for name in names:
        for m in methods:
            seeds = sorted({s for (d, mm, s) in by if d == name and mm == m})
            pa, pb, fa, one = [], [], [], 0
            for s in seeds:
                rs = by[(name, m, s)]
                y = np.array([r["y_true"] for r in rs])
                p = np.array([r["y_pred"] for r in rs])
                pa.append(acc(y, p))
                pb.append(bacc(y, p))
                for k in sorted({r["fold"] for r in rs}):
                    yk = np.array([r["y_true"] for r in rs if r["fold"] == k])
                    pk = np.array([r["y_pred"] for r in rs if r["fold"] == k])
                    fa.append(acc(yk, pk))
                    one += len(np.unique(pk)) == 1
            n_fold_runs = len(fa)
            summ.append({"dataset": name, "method": m, "seeds": len(seeds),
                         "pooled_acc": ms(pa), "pooled_bal_acc": ms(pb), "per_fold_acc": ms(fa),
                         "single_class_predictions": f"{one}/{n_fold_runs}"})
    with open(out / "kimore_benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    lines = ["# KIMORE (Rehab-Pile) benchmark", "",
             "Pooled = all test predictions of all folds together; mean ± std over seeds (%).", "",
             "| Dataset | Method | Seeds | Pooled accuracy | Pooled balanced accuracy | Per-fold accuracy | Folds predicting one class only |",
             "|---|---|---|---|---|---|---|"]
    for s in summ:
        lines.append(f"| {s['dataset']} | {s['method']} | {s['seeds']} | {s['pooled_acc']} | {s['pooled_bal_acc']} | "
                     f"{s['per_fold_acc']} | {s['single_class_predictions']} |")
    (out / "kimore_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {out / 'kimore_benchmark_summary.csv'}, {out / 'kimore_benchmark.md'}")


if __name__ == "__main__":
    main()
