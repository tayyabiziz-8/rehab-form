"""Keraal Groups 1A / 2A (and 1B / 2B): recordings with clinician annotations.

Layout seen in group2A (16 Nov 2026):
    <root>/annotatorA/G2A-CTK-S1-Brest-029.anvil      (UTF-16 Anvil XML)
    <root>/annotatorB/G2A-CTK-S1-Brest-029.anvil
    <root>/blazepose/G2A-BP-CTK-S1-Brest-029.json
    <root>/openpose/G2A-OP-CTK-S1-Brest-029.json
    <root>/kinect/G2A-Kinect-CTK-S1-Brest-029.txt
    <root>/videos/G2A-Anon-CTK-S1-Brest-029.mp4

Name = <group>-[<modality>-]<exercise>-<subject>-<site>-<index>; annotation
files have no modality part. The recording key drops the modality, so every
file of one recording shares it: G2A-CTK-S1-Brest-029.

There are no Vicon files in these groups. The annotations are the ground truth
for Pillar 2 (do the phone findings agree with what the clinicians marked?).
Two annotators (A and B) labelled the same recordings, so their agreement is the
ceiling any automatic method can reasonably be expected to reach.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODALITY_FOLDERS = {
    # folder: (name token, extension)
    "blazepose": ("BP", ".json"),
    "openpose": ("OP", ".json"),
    "kinect": ("Kinect", ".txt"),
    "videos": ("Anon", ".mp4"),
}
MODALITY_TOKENS = {token for token, _ in MODALITY_FOLDERS.values()}
EXERCISES = ("CTK", "ELK", "RTK")

_NAME = re.compile(
    r"^(?P<group>G\d[A-Z]?)-"
    r"(?:(?P<modality>[A-Za-z]+)-)?"
    r"(?P<exercise>CTK|ELK|RTK)-"
    r"(?P<subject>[^-]+)-"
    r"(?P<site>[^-]+)-"
    r"(?P<index>[^-]+)$"
)


def parse_name(stem: str) -> dict | None:
    """Split a Group 1/2 file stem into its parts; None if it does not match.

    G2A-BP-CTK-S1-Brest-029 -> group G2A, modality BP, exercise CTK,
    subject S1, site Brest, index 029, key G2A-CTK-S1-Brest-029.
    """
    m = _NAME.match(stem)
    if not m:
        return None
    parts = m.groupdict()
    if parts["modality"] is not None and parts["modality"] not in MODALITY_TOKENS:
        return None
    parts["key"] = "-".join(parts[k] for k in ("group", "exercise", "subject", "site", "index"))
    return parts


@dataclass
class AnnotatedRecording:
    key: str
    group: str
    exercise: str
    subject: str
    site: str
    index: str
    blazepose: Path | None = None
    openpose: Path | None = None
    kinect: Path | None = None
    video: Path | None = None
    annotations: dict[str, Path] = field(default_factory=dict)  # annotator -> .anvil


def iter_annotated_recordings(root: str | Path) -> list[AnnotatedRecording]:
    """All recordings under one extracted group folder, sorted by key.

    Annotator folders are any folder whose name starts with 'annotator'
    (annotatorA -> 'A'). Files whose names do not parse are ignored
    (group2A ships a stray, nameless '.anvil' sample in each annotator folder).
    """
    root = Path(root)
    recs: dict[str, AnnotatedRecording] = {}

    def get(info: dict) -> AnnotatedRecording:
        k = info["key"]
        if k not in recs:
            recs[k] = AnnotatedRecording(
                key=k, group=info["group"], exercise=info["exercise"],
                subject=info["subject"], site=info["site"], index=info["index"],
            )
        return recs[k]

    for folder, (_token, ext) in MODALITY_FOLDERS.items():
        for f in sorted(root.rglob(f"{folder}/*{ext}")):
            info = parse_name(f.stem)
            if info:
                setattr(get(info), "video" if folder == "videos" else folder, f)

    for d in sorted(p for p in root.rglob("annotator*") if p.is_dir()):
        annotator = d.name[len("annotator"):] or d.name
        for f in sorted(d.glob("*.anvil")):
            info = parse_name(f.stem)
            if info:
                get(info).annotations[annotator] = f

    return [recs[k] for k in sorted(recs)]


# ---------------------------------------------------------------------------
# Anvil annotations
# ---------------------------------------------------------------------------
def _read_xml_text(path: str | Path) -> str:
    """Anvil files here are UTF-16 with a byte-order mark; fall back to UTF-8."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-16")
    # ElementTree refuses a str that still declares an encoding
    return re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1)


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def read_anvil(path: str | Path) -> dict:
    """Parse one Anvil file into {'video': src, 'elements': [...]}.

    Each element: track, track_type, index, start, end (seconds) and
    attributes {name: text}. Kept generic on purpose: the track and attribute
    names are reported by scripts/summarise_keraal_annotations.py before any
    code relies on them.
    """
    root = ET.fromstring(_read_xml_text(path))
    video = root.find("./head/video")
    out = {"video": video.get("src") if video is not None else None, "elements": []}
    for track in root.iter("track"):
        for el in track.iter("el"):
            attrs = {a.get("name"): (a.text or "").strip() for a in el.iter("attribute")}
            out["elements"].append({
                "track": track.get("name"),
                "track_type": track.get("type"),
                "index": el.get("index"),
                "start": _num(el.get("start")),
                "end": _num(el.get("end")),
                "attributes": attrs,
            })
    return out


# ---------------------------------------------------------------------------
# BlazePose for Groups 1/2: un-mirror, then fix left/right by geometry
# ---------------------------------------------------------------------------
IMAGE_SIZE_G12 = (480, 360)  # checked 17 Nov: every 1A/2A video is 480x360 at 10 fps

# BlazePose landmark indices (MediaPipe order) for the shoulders and hips
_L_SH, _R_SH, _L_HIP, _R_HIP = 11, 12, 23, 24


def _swap_index() -> list[int]:
    from .keraal import BLAZEPOSE_ORDER

    pos = {n: i for i, n in enumerate(BLAZEPOSE_ORDER)}
    out = []
    for n in BLAZEPOSE_ORDER:
        if n.startswith("Left_"):
            out.append(pos["Right_" + n[5:]])
        elif n.startswith("Right_"):
            out.append(pos["Left_" + n[6:]])
        else:
            out.append(pos[n])
    return out


def canonical_sides(bp: np.ndarray) -> tuple[np.ndarray, dict]:
    """Make 'Left_*' the person's real left, assuming they face the camera.

    Why: after un-mirroring (x -> 1 - x, which the 1A/2A overlays confirm),
    BlazePose still labels left and right the wrong way round in some
    recordings (17 Nov: G1A-ELK-R1-Brest-003 and -014 vs -029, same patient,
    same room). Keraal patients face the camera, so the person's left shoulder
    and hip must be on the IMAGE RIGHT (larger x). If the recording's median
    says otherwise, every Left_/Right_ pair is swapped.

    Returns (fixed array, info) with info = {'swapped': bool,
    'frames_reversed': fraction of frames whose own order disagrees with the
    recording after the fix (BlazePose flicker; reported, not changed)}.
    """
    d = (bp[:, _L_SH, 0] - bp[:, _R_SH, 0]) + (bp[:, _L_HIP, 0] - bp[:, _R_HIP, 0])
    ok = np.isfinite(d)
    swapped = bool(ok.any() and np.median(d[ok]) < 0)
    if swapped:
        bp = bp[:, _swap_index()]
        d = -d
    frac = float(np.mean(d[ok] < 0)) if ok.any() else float("nan")
    return bp, {"swapped": swapped, "frames_reversed": frac}


def load_blazepose_g12(path: str | Path, image_size=IMAGE_SIZE_G12) -> tuple[np.ndarray, dict]:
    """Group 1/2 BlazePose in pixel units, un-mirrored, sides fixed by geometry."""
    from .keraal import load_blazepose

    bp = load_blazepose(path, image_size=image_size, unmirror=True)
    return canonical_sides(bp)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def annotation_labels(path: str | Path) -> dict:
    """One annotator's labels for one recording.

    Track names seen in 1A/2A (17 Nov):
      'Global evaluation' / evaluation: Correct | Incorrect | Incomplete | Motionless
      'Global error'      / Evaluation: SmallError | LargeError, type: Error1-3 | Other | none,
                            bodyPart: Spine | BothArms | LeftArm | RightArm
      'Temporal error'    / same attributes, with the time window where it happens
    """
    els = read_anvil(path)["elements"]
    out = {"evaluation": None, "errors": [], "windows": [], "span": None}
    for e in els:
        track = (e["track"] or "").lower()
        a = e["attributes"]
        if track.startswith("global evaluation"):
            out["evaluation"] = a.get("evaluation") or next(iter(a.values()), None)
            out["span"] = (e["start"], e["end"])   # whole recording, in Anvil seconds
        elif track.startswith("global error"):
            if a.get("type") and a.get("type") != "none":
                out["errors"].append({"type": a.get("type"), "severity": a.get("Evaluation"),
                                      "body_part": a.get("bodyPart")})
        elif track.startswith("temporal error") and a:
            out["windows"].append({"type": a.get("type"), "severity": a.get("Evaluation"),
                                   "body_part": a.get("bodyPart"),
                                   "start": e["start"], "end": e["end"]})
    return out
