"""Print the layout of a downloaded Keraal group folder (1A, 2A, ...) so a loader can be written.

Usage (from the repo root):
    python scripts\\inspect_keraal_groups.py data\\keraal1
    python scripts\\inspect_keraal_groups.py data\\keraal2
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(root: Path) -> None:
    if not root.exists():
        print(f"not found: {root}")
        return
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"== {root}  ({len(files)} files)")

    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for p in files:
        by_dir[p.parent.relative_to(root)].append(p)

    print("\n-- folders (files, extensions, 3 example names)")
    for d in sorted(by_dir):
        items = sorted(by_dir[d])
        exts = Counter(p.suffix.lower() for p in items)
        print(f"{str(d) or '.':<45} {len(items):>5}  {dict(exts)}")
        for p in items[:3]:
            print(f"      {p.name}")

    # Name pattern: count the dash-separated part counts and show the distinct values per position
    stems = [p.stem for p in files if p.suffix.lower() in {".mp4", ".avi", ".json", ".txt", ".anvil", ".xml"}]
    parts_count = Counter(len(s.split("-")) for s in stems)
    print(f"\n-- name parts (split on '-'): {dict(parts_count)}")
    if stems:
        n = parts_count.most_common(1)[0][0]
        cols = defaultdict(Counter)
        for s in stems:
            bits = s.split("-")
            if len(bits) == n:
                for i, b in enumerate(bits):
                    cols[i][b] += 1
        for i in range(n):
            vals = cols[i]
            shown = ", ".join(f"{k}({v})" for k, v in vals.most_common(8))
            more = f" ... +{len(vals) - 8} more" if len(vals) > 8 else ""
            print(f"  part {i}: {len(vals)} distinct: {shown}{more}")

    # One annotation file, first 60 lines
    ann = sorted(p for p in files if p.suffix.lower() in {".anvil", ".xml"}
                 or "annot" in str(p.parent).lower())
    print(f"\n-- annotation-like files: {len(ann)}")
    if ann:
        p = ann[0]
        print(f"first one: {p.relative_to(root)}\n")
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 60:
                    print("... (truncated)")
                    break
                print(line.rstrip())

    # One BlazePose json: first frame keys
    bp = sorted(p for p in files if p.suffix.lower() == ".json" and "blaze" in str(p).lower())
    if bp:
        import json
        with open(bp[0], encoding="utf-8") as f:
            d = json.load(f)
        print(f"\n-- blazepose example: {bp[0].relative_to(root)}")
        print("top-level keys:", list(d)[:5])
        pos = d.get("positions", d)
        if isinstance(pos, dict) and pos:
            k0 = sorted(pos, key=lambda k: float(k) if k.replace('.', '', 1).isdigit() else 0)[0]
            fr = pos[k0]
            print(f"frames: {len(pos)}; first frame key {k0!r}; joints: {list(fr)[:6]} ...")
            if fr:
                j = next(iter(fr))
                print(f"  {j}: {fr[j]}")


if __name__ == "__main__":
    # Windows cmd writes redirected output as cp1252, which cannot hold some annotation characters.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for arg in sys.argv[1:] or ["data/keraal1", "data/keraal2"]:
        main(Path(arg))
        print()