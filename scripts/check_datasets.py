"""First look at a downloaded dataset folder.

Prints file counts and sizes by type, peeks into a few files, and decodes the
file-name conventions of datasets that encode labels in the name:

    IntelliRehabDS  SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.txt
                    (CorrectLabel: 1 = correct, 2 = incorrect)
    UI-PRMD         m<movement>_s<subject>_e<episode>_<positions|angles>.txt

Run it on every dataset right after download and paste the output into the
team's dataset notes.

Example
    python scripts/check_datasets.py data/irds --kind irds
    python scripts/check_datasets.py data/ui_prmd --kind ui_prmd
    python scripts/check_datasets.py data/keraal
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

IRDS_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<date>[^_]+)_(?P<gesture>\d+)_(?P<rep>\d+)_(?P<correct>\d+)_(?P<position>[^_.]+)")
UIPRMD_RE = re.compile(r"m(?P<movement>\d+)_s(?P<subject>\d+)_e(?P<episode>\d+)_(?P<kind>positions|angles)", re.I)


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def overview(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = Counter(p.suffix.lower() or "(none)" for p in files)
    size_by_ext = defaultdict(int)
    for p in files:
        size_by_ext[p.suffix.lower() or "(none)"] += p.stat().st_size
    total = sum(size_by_ext.values())
    print(f"\n== {root}  ({len(files)} files, {human(total)})")
    print(f"{'type':<10}{'files':>8}{'size':>12}")
    for ext, n in by_ext.most_common():
        print(f"{ext:<10}{n:>8}{human(size_by_ext[ext]):>12}")
    top = sorted({p.relative_to(root).parts[0] for p in files})[:15]
    print("top-level entries:", ", ".join(top))
    return files


def peek(files: list[Path], k: int = 3) -> None:
    text = [p for p in files if p.suffix.lower() in {".txt", ".csv", ".json", ".xml"}][:k]
    for p in text:
        print(f"\n-- {p.name}")
        try:
            with p.open("r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    print("   ", line.rstrip()[:160])
        except OSError as e:
            print("    cannot read:", e)
    for p in [p for p in files if p.suffix.lower() == ".npz"][:k]:
        import numpy as np
        with np.load(p) as d:
            print(f"\n-- {p.name}: " + ", ".join(f"{key}{d[key].shape}" for key in d.files))


def decode(files: list[Path], kind: str) -> None:
    rx = {"irds": IRDS_RE, "ui_prmd": UIPRMD_RE}[kind]
    parsed = []
    for f in files:
        m = rx.search(f.name)
        if m:
            row = m.groupdict()
            # UI-PRMD keeps incorrect reps in separate folders; we assume the
            # word "incorrect" appears in their path. Confirm on first download.
            row["label"] = "incorrect" if "incorrect" in str(f).lower() else "correct"
            parsed.append(row)
    print(f"\n== decoded {len(parsed)} of {len(files)} file names as {kind}")
    if not parsed:
        print("No names matched; check the convention against the dataset docs.")
        return
    if kind == "irds":
        print("subjects:", len({r['subject'] for r in parsed}))
        table = Counter((int(r["gesture"]), r["correct"]) for r in parsed)
        print(f"{'gesture':>8}{'correct(1)':>12}{'incorrect(2)':>14}{'other':>8}")
        for g in sorted({g for g, _ in table}):
            other = sum(v for (gg, c), v in table.items() if gg == g and c not in {"1", "2"})
            print(f"{g:>8}{table[(g, '1')]:>12}{table[(g, '2')]:>14}{other:>8}")
        print("positions:", dict(Counter(r["position"] for r in parsed)))
    else:
        print("subjects:", len({r['subject'] for r in parsed}))
        table = Counter((int(r["movement"]), r["label"]) for r in parsed if r["kind"].lower() == "positions")
        print(f"{'movement':>9}{'correct':>9}{'incorrect':>11}   (positions files only)")
        for m in sorted({m for m, _ in table}):
            print(f"{m:>9}{table[(m, 'correct')]:>9}{table[(m, 'incorrect')]:>11}")
        print("Labels come from the folder path; if every file shows as correct, fix the rule in decode().")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--kind", choices=["irds", "ui_prmd"], help="decode label-bearing file names")
    args = p.parse_args()
    root = Path(args.folder)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    files = overview(root)
    peek(files)
    if args.kind:
        decode(files, args.kind)


if __name__ == "__main__":
    main()
