"""Tests for the Group 1A/2A loader and the Anvil parser (synthetic files, no dataset needed).

    python tests/test_keraal_groups.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rehabform.keraal import BLAZEPOSE_ORDER  # noqa: E402
import numpy as np  # noqa: E402
from rehabform.keraal_groups import (  # noqa: E402
    annotation_labels, canonical_sides, iter_annotated_recordings, parse_name, read_anvil,
)

ANVIL = """<?xml version="1.0" encoding="UTF-16"?>
<annotation>
  <head>
    <video src="Vid001.mp4" master="true" />
  </head>
  <body>
    <track name="Global evaluation" type="primary">
      <el index="0" start="0" end="1.48">
        <attribute name="evaluation">Incorrect</attribute>
      </el>
    </track>
    <track name="Error" type="primary">
      <el index="0" start="0.4" end="1.1">
        <attribute name="label">Bend trunk</attribute>
        <attribute name="significance">Significant</attribute>
      </el>
    </track>
  </body>
</annotation>
"""


def test_names_share_a_key_across_modalities():
    a = parse_name("G2A-BP-CTK-S1-Brest-029")
    b = parse_name("G2A-CTK-S1-Brest-029")  # annotation file: no modality
    assert a["modality"] == "BP" and b["modality"] is None
    assert a["key"] == b["key"] == "G2A-CTK-S1-Brest-029"
    assert parse_name("") is None and parse_name("G2A-XX-CTK-S1-Brest-029") is None


def test_utf16_anvil_parses():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.anvil"
        p.write_bytes(ANVIL.encode("utf-16"))  # includes the byte-order mark, like the real files
        parsed = read_anvil(p)
    assert parsed["video"] == "Vid001.mp4"
    tracks = [e["track"] for e in parsed["elements"]]
    assert tracks == ["Global evaluation", "Error"]
    err = parsed["elements"][1]
    assert (err["start"], err["end"]) == (0.4, 1.1)
    assert err["attributes"] == {"label": "Bend trunk", "significance": "Significant"}


def test_folder_scan_pairs_modalities_and_annotators():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for folder, name in [("blazepose", "G2A-BP-ELK-S2-Roscoff-005.json"),
                             ("openpose", "G2A-OP-ELK-S2-Roscoff-005.json"),
                             ("kinect", "G2A-Kinect-ELK-S2-Roscoff-005.txt"),
                             ("videos", "G2A-Anon-ELK-S2-Roscoff-005.mp4"),
                             ("annotatorA", "G2A-ELK-S2-Roscoff-005.anvil"),
                             ("annotatorB", "G2A-ELK-S2-Roscoff-005.anvil"),
                             ("annotatorA", ".anvil")]:  # stray nameless file must be ignored
            (root / folder).mkdir(exist_ok=True)
            (root / folder / name).write_text(
                json.dumps({"positions": {"1.0": {n: [0, 0, 0] for n in BLAZEPOSE_ORDER}}}))
        recs = iter_annotated_recordings(root)
    assert len(recs) == 1
    r = recs[0]
    assert (r.exercise, r.subject, r.site, r.index) == ("ELK", "S2", "Roscoff", "005")
    assert r.blazepose and r.openpose and r.kinect and r.video
    assert sorted(r.annotations) == ["A", "B"]


REAL_ANVIL = ANVIL.replace(
    '<track name="Error" type="primary">', '<track name="Global error" type="primary">').replace(
    '<attribute name="label">Bend trunk</attribute>\n        <attribute name="significance">Significant</attribute>',
    '<attribute name="Evaluation">LargeError</attribute><attribute name="type">Error1</attribute>'
    '<attribute name="bodyPart">BothArms</attribute>')


def test_annotation_labels_use_real_track_names():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.anvil"
        p.write_bytes(REAL_ANVIL.encode("utf-16"))
        lab = annotation_labels(p)
    assert lab["evaluation"] == "Incorrect"
    assert lab["errors"] == [{"type": "Error1", "severity": "LargeError", "body_part": "BothArms"}]


def test_canonical_sides_puts_person_left_on_image_right():
    bp = np.zeros((5, 33, 3), np.float32)
    bp[:, 11, 0], bp[:, 12, 0] = 200, 280      # 'Left' shoulder on image LEFT: labels are wrong
    bp[:, 23, 0], bp[:, 24, 0] = 210, 270
    bp[:, 15, 1] = 10                          # 'Left' wrist raised
    fixed, info = canonical_sides(bp)
    assert info["swapped"] and fixed[0, 11, 0] == 280 and fixed[0, 16, 1] == 10
    again, info2 = canonical_sides(fixed)      # already right: unchanged
    assert not info2["swapped"] and np.array_equal(again, fixed)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
