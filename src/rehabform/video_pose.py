"""Video -> pose sequence in the format the findings engine expects.

  pose_from_video(path, model)  runs MediaPipe Pose Landmarker on a video and
                                returns (bp, info): bp is (T, 33, 3) in PIXEL
                                units at TARGET_FPS of real time, left/right
                                fixed by geometry (person facing the camera).

Why resample to 10 fps: the detectors were fitted on Keraal BlazePose at 10
frames per recording-second, and repetition timing assumes that rate. A phone
records at 30 fps, so every 3rd frame is used (the landmarker still sees each
used frame in order, so its tracking works).

Front-camera ("selfie") videos may be saved mirrored. Then the person's left
arm is on the image left; pass mirrored=True (and check the overlay image the
report saves: orange = left side).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .keraal_groups import canonical_sides

TARGET_FPS = 10.0


def pose_from_video(path: str | Path, model: str | Path, mirrored: bool = False,
                    target_fps: float = TARGET_FPS, max_seconds: float | None = None,
                    min_confidence: float = 0.5) -> tuple[np.ndarray, dict]:
    """min_confidence sets MediaPipe's detection, presence and tracking thresholds (its default is 0.5);
    lower it (e.g. 0.3) when the person is small, far away or partly hidden."""
    import os
    os.environ.setdefault("GLOG_minloglevel", "2")          # hide MediaPipe's harmless start-up warnings
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not Path(model).exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model} (download pose_landmarker_full.task "
                                f"from the MediaPipe Pose Landmarker guide into models/)")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / target_fps)))
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=min_confidence, min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence)
    frames, vis, used = [], [], []
    i = 0
    with vision.PoseLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok or (max_seconds and i / fps > max_seconds):
                break
            if i % step == 0:
                if mirrored:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int(i * 1000.0 / fps))
                if res.pose_landmarks:
                    p = res.pose_landmarks[0]
                    frames.append([[q.x * W, q.y * H, q.z * W] for q in p])
                    vis.append([q.visibility for q in p])
                else:
                    frames.append(np.full((33, 3), np.nan).tolist())
                    vis.append([0.0] * 33)
                used.append(i)
            i += 1
    cap.release()
    bp = np.asarray(frames, np.float32)
    if len(bp) == 0:
        raise ValueError(f"no frames read from {path}")
    missing = float(np.isnan(bp[:, 11, 0]).mean())
    got_fps = fps / step
    if abs(got_fps - target_fps) > 0.05:                  # e.g. 25 fps video -> 12.5 -> resample to 10
        from .keraal import resample_sequence
        bp = resample_sequence(bp, max(2, int(round(len(bp) * target_fps / got_fps)))).astype(np.float32)
    bp, side_info = canonical_sides(bp)
    info = {"video_fps": fps, "frame_step": step, "fps": target_fps, "width": W, "height": H,
            "frames_used": used, "missing_fraction": missing, "min_confidence": min_confidence, "mirrored_input": mirrored, **side_info,
            "visibility": np.asarray(vis, np.float32)}
    return bp, info


def load_landmarks_npz(path: str | Path, width: int, height: int) -> np.ndarray:
    """Output of scripts/extract_mediapipe.py (normalised) -> pixel units."""
    d = np.load(path)
    img = d["image"][..., :3].astype(np.float32)
    return img * np.array([width, height, width], np.float32)
