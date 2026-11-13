"""Load and align Keraal Group 3 recordings."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import SHARED_JOINTS

_NAME_RE = re.compile(
    r"^G3-(?P<modality>[^-]+)-(?P<exercise>[^-]+)-"
    r"(?P<participant>P\d+T\d+)-(?P<label>[^-]+)-(?P<error>[^-]+)-(?P<index>\d+)$"
)

VICON_TARGETS = [
    "r_forearm", "l_forearm", "r_arm", "l_arm", "chest", "r_thigh",
    "l_thigh", "r_shoulder", "l_shoulder", "r_hand", "l_hand",
    "r_foot", "l_foot", "hips", "head", "r_tibia", "l_tibia",
]
_VICON_INDEX = {name: index for index, name in enumerate(VICON_TARGETS)}
# Canonical BlazePose order (readme_files_format.txt, section 4.1). This is the
# same order as MediaPipe Pose, so index i here matches MEDIAPIPE_33 in skeleton.py.
BLAZEPOSE_ORDER = [
    "Nose", "Left_eye_inner", "Left_eye", "Left_eye_outer",
    "Right_eye_inner", "Right_eye", "Right_eye_outer", "Left_ear",
    "Right_ear", "Mouth_left", "Mouth_right", "Left_shoulder",
    "Right_shoulder", "Left_elbow", "Right_elbow", "Left_wrist",
    "Right_wrist", "Left_pinky", "Right_pinky", "Left_index",
    "Right_index", "Left_thumb", "Right_thumb", "Left_hip", "Right_hip",
    "Left_knee", "Right_knee", "Left_ankle", "Right_ankle", "Left_heel",
    "Right_heel", "Left_foot_index", "Right_foot_index",
]
_BLAZEPOSE_NAMES = set(BLAZEPOSE_ORDER)

# Keraal Group 3 videos are 960 x 544. Groups 1a, 1b, 2a and 2b are 480 x 360.
KERAAL_G3_IMAGE_SIZE = (960, 544)


@dataclass(frozen=True)
class KeraalRecording:
    """Paths and metadata for one Group 3 recording."""

    recording_id: str
    exercise: str
    participant: str
    label: str
    error: str
    blazepose: Path
    vicon: Path
    kinect: Path
    openpose: Path
    video: Path


def _modality_path(root: Path, modality: str, recording_id: str, extension: str) -> Path:
    return next(root.glob(f"{modality}/{recording_id}{extension}"), Path())


def iter_recordings(root: str | Path) -> list[KeraalRecording]:
    """Return Group 3 recordings whose five modality files all exist."""
    root = Path(root)
    recordings = []
    for path in sorted((root / "blazepose").glob("G3-BP-*.json")):
        match = _NAME_RE.match(path.stem)
        if match is None:
            raise ValueError(f"Unrecognised Keraal filename: {path.name}")
        common = "-".join(
            [
                "G3",
                match["exercise"],
                match["participant"],
                match["label"],
                match["error"],
                match["index"],
            ]
        )
        files = {
            "blazepose": path,
            "vicon": _modality_path(root, "vicon", f"G3-Vicon-{common[3:]}", ".txt"),
            "kinect": _modality_path(root, "kinect", f"G3-Kinect-{common[3:]}", ".txt"),
            "openpose": _modality_path(root, "openpose", f"G3-OP-{common[3:]}", ".json"),
            "video": _modality_path(root, "videos", f"G3-Anon-{common[3:]}", ".avi"),
        }
        missing = [name for name, file in files.items() if not file.is_file()]
        if missing:
            raise FileNotFoundError(f"{path.name} is missing modalities: {', '.join(missing)}")
        recordings.append(
            KeraalRecording(
                recording_id=common,
                exercise=match["exercise"],
                participant=match["participant"],
                label=match["label"],
                error=match["error"],
                **files,
            )
        )
    return recordings


def load_blazepose(
    path: str | Path,
    image_size: tuple[int, int] | None = KERAAL_G3_IMAGE_SIZE,
) -> np.ndarray:
    """Load a Keraal BlazePose JSON file as ``(frames, 33, 3)``.

    Landmarks are placed in the canonical BlazePose order by NAME, so the
    result does not depend on the key order inside the JSON file. Missing
    landmarks become NaN.

    BlazePose stores x as a fraction of image width and y as a fraction of
    image height, so raw values have different scales on the two axes and
    distort every angle that is not exactly horizontal or vertical. With
    ``image_size=(width, height)`` (the default, Group 3 = 960 x 544) the
    result is converted to pixel units: x * width, y * height, and z * width
    (MediaPipe's z uses roughly the same scale as x). Pass ``image_size=None``
    to get the raw normalised values.
    """
    payload = json.loads(Path(path).read_text())
    positions = payload["positions"]
    frames = sorted(positions, key=float)
    names = set(positions[frames[0]])
    if names != _BLAZEPOSE_NAMES:
        raise ValueError(f"Unexpected BlazePose landmarks in {path}")
    output = np.full((len(frames), len(BLAZEPOSE_ORDER), 3), np.nan, dtype=np.float32)
    for t, frame in enumerate(frames):
        landmarks = positions[frame] or {}
        for j, name in enumerate(BLAZEPOSE_ORDER):
            xyz = landmarks.get(name)
            if xyz is not None and len(xyz) >= 3:
                output[t, j] = xyz[:3]
    if image_size is not None:
        width, height = image_size
        output = output * np.array([width, height, width], dtype=np.float32)
    return output


def load_vicon_positions(path: str | Path) -> np.ndarray:
    """Load Vicon global target positions as ``(frames, 17, 3)``."""
    values = np.loadtxt(path, ndmin=2)
    expected_width = len(VICON_TARGETS) * 7
    if values.shape[1] != expected_width:
        raise ValueError(f"Expected {expected_width} Vicon columns, got {values.shape[1]}")
    return values.reshape(len(values), len(VICON_TARGETS), 7)[:, :, :3].astype(np.float32)


def load_kinect_positions(path: str | Path) -> np.ndarray:
    """Load Kinect joint positions as ``(frames, 25, 3)``."""
    values = np.loadtxt(path, ndmin=2)
    expected_width = 25 * 7
    if values.shape[1] != expected_width:
        raise ValueError(f"Expected {expected_width} Kinect columns, got {values.shape[1]}")
    return values.reshape(len(values), 25, 7)[:, :, :3].astype(np.float32)


def vicon_to_shared(positions: np.ndarray, swap_sides: bool = False) -> np.ndarray:
    """Approximate-map ART Vicon targets to the shared layout.

    ``swap_sides`` is diagnostic only. It tests a possible source-side
    convention mismatch and should not be used as a general correction.

    The Keraal format documentation identifies these as ART targets
    (shoulder, arm, forearm, hand, etc.), not Kinect-style anatomical joint
    centres. The returned points are therefore suitable for diagnostics, but
    not yet a validated anatomical reference.
    """
    if positions.ndim != 3 or positions.shape[1] != len(VICON_TARGETS):
        raise ValueError(f"Expected Vicon positions (T, 17, 3), got {positions.shape}")
    index = _VICON_INDEX
    left, right = ("l", "r") if not swap_sides else ("r", "l")
    mapped = {
        "nose": positions[:, index["head"]],
        "l_shoulder": positions[:, index[f"{left}_shoulder"]],
        "r_shoulder": positions[:, index[f"{right}_shoulder"]],
        "l_elbow": positions[:, index[f"{left}_arm"]],
        "r_elbow": positions[:, index[f"{right}_arm"]],
        "l_wrist": positions[:, index[f"{left}_forearm"]],
        "r_wrist": positions[:, index[f"{right}_forearm"]],
        "l_hip": positions[:, index["hips"]],
        "r_hip": positions[:, index["hips"]],
        "l_knee": positions[:, index[f"{left}_thigh"]],
        "r_knee": positions[:, index[f"{right}_thigh"]],
        "l_ankle": positions[:, index[f"{left}_tibia"]],
        "r_ankle": positions[:, index[f"{right}_tibia"]],
    }
    return np.stack([mapped[name] for name in SHARED_JOINTS], axis=1)


def resample_sequence(sequence: np.ndarray, frame_count: int) -> np.ndarray:
    """Linearly resample a sequence along its frame axis."""
    if sequence.ndim < 1 or len(sequence) < 2:
        raise ValueError("A sequence needs at least two frames to be resampled")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    old = np.linspace(0.0, 1.0, len(sequence))
    new = np.linspace(0.0, 1.0, frame_count)
    flat = sequence.reshape(len(sequence), -1)
    result = np.stack([np.interp(new, old, flat[:, column]) for column in range(flat.shape[1])], axis=1)
    return result.reshape((frame_count,) + sequence.shape[1:]).astype(np.float32)


def shift_sequence(sequence: np.ndarray, shift: int) -> np.ndarray:
    """Shift a sequence by whole frames, padding edge values.

    A positive shift moves later frames earlier, which is useful when the
    reference sequence lags the observed sequence.
    """
    if sequence.ndim < 1:
        raise ValueError(f"Expected a sequence, got shape {sequence.shape}")
    if shift == 0:
        return sequence.copy()
    result = np.empty_like(sequence)
    if shift > 0:
        result[:-shift] = sequence[shift:]
        result[-shift:] = sequence[-1]
    else:
        amount = -shift
        result[amount:] = sequence[:-amount]
        result[:amount] = sequence[0]
    return result
