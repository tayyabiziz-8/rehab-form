"""Reproduce a Rehab-Pile classification baseline with FCN.

Loads one Rehab-Pile dataset through aeon, trains FCN on every cross-subject
fold, and reports accuracy and balanced accuracy as mean +/- std.

Examples
    python scripts/train_rehab_pile.py --list
    python scripts/train_rehab_pile.py --dataset KIMORE_clf_bn_Sq --extract-path data/rehab_pile
    python scripts/train_rehab_pile.py --dataset KIMORE_clf_bn_Sq --max-folds 1 --epochs 50   # quick smoke test

Dataset names come from --list; "KIMORE_clf_bn_TR" is the example name in the
aeon docs. Check the exact spelling there before a long run.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_fold(name: str, fold: int, extract_path: str | None):
    from aeon.datasets import load_rehab_pile_dataset

    kw = {"extract_path": extract_path} if extract_path else {}
    X_tr, y_tr = load_rehab_pile_dataset(name, split="train", fold=fold, **kw)
    X_te, y_te = load_rehab_pile_dataset(name, split="test", fold=fold, **kw)
    return X_tr, y_tr, X_te, y_te


def train_one(X_tr, y_tr, X_te, y_te, epochs: int, batch: int, lr: float, seed: int):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from torch import nn

    from rehabform.models import FCN

    torch.manual_seed(seed)
    np.random.seed(seed)

    enc = LabelEncoder().fit(y_tr)
    ytr = enc.transform(y_tr)
    # Test classes unseen in training cannot be predicted; keep them as -1.
    known = set(enc.classes_)
    yte = np.array([enc.transform([v])[0] if v in known else -1 for v in y_te])

    # aeon convention: (n_cases, n_channels, n_timepoints). Standardise per
    # channel with training statistics only.
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = torch.tensor((X_tr - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((X_te - mu) / sd, dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FCN(Xtr.shape[1], len(enc.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        epoch_loss = total / len(Xtr)
        # Model selection on TRAIN loss only, so the test fold stays untouched.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(device)).argmax(dim=1).cpu().numpy()
    return accuracy_score(yte, pred), balanced_accuracy_score(yte, pred)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list Rehab-Pile classification datasets and exit")
    p.add_argument("--dataset", help="Rehab-Pile dataset name")
    p.add_argument("--extract-path", default=None, help="folder holding the downloaded Rehab-Pile data")
    p.add_argument("--max-folds", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/rehab_pile_results.csv")
    args = p.parse_args()

    try:
        import aeon.datasets as ad
    except ImportError:
        sys.exit("aeon is not installed: pip install -r requirements.txt")

    if args.list:
        names = ad.load_rehab_pile_classification_datasets()
        print("\n".join(sorted(names)))
        return
    if not args.dataset:
        sys.exit("Pass --dataset NAME (see --list).")

    rows = []
    for fold in range(args.max_folds):
        try:
            X_tr, y_tr, X_te, y_te = load_fold(args.dataset, fold, args.extract_path)
        except Exception as e:  # no more folds, or data not downloaded yet
            if fold == 0:
                sys.exit(
                    f"Could not load {args.dataset} fold 0: {e}\n"
                    "Download Rehab-Pile first (see data/README.md) and pass --extract-path."
                )
            break
        t0 = time.time()
        acc, bacc = train_one(X_tr, y_tr, X_te, y_te, args.epochs, args.batch, args.lr, args.seed)
        rows.append({"dataset": args.dataset, "fold": fold, "acc": acc, "bacc": bacc,
                     "n_train": len(y_tr), "n_test": len(y_te), "seconds": round(time.time() - t0, 1)})
        print(f"fold {fold}: acc={acc:.3f} bacc={bacc:.3f} (train {len(y_tr)}, test {len(y_te)})")

    accs = np.array([r["acc"] for r in rows])
    baccs = np.array([r["bacc"] for r in rows])
    print(f"\n{args.dataset}: {len(rows)} folds  "
          f"accuracy {accs.mean():.3f} +/- {accs.std():.3f}  "
          f"balanced accuracy {baccs.mean():.3f} +/- {baccs.std():.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Per-fold results appended to {out}")


if __name__ == "__main__":
    main()
