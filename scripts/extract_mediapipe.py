"""Run MediaPipe Pose Landmarker on videos and save the skeletons.

Use this on datasets that ship RGB video (Keraal, KIMORE) so training data has
the same skeleton format and noise as the phone app.

Output: one .npz per video with
    image  (T, 33, 4)  normalised x, y, z, visibility   (NaN where no person)
    world  (T, 33, 3)  metric world coordinates, metres (NaN where no person)
    fps    float

Setup
    1. pip install -r requirements.txt
    2. Download a model file (lite, full or heavy) from the MediaPipe Pose
       Landmarker guide, e.g. pose_landmarker_full.task, into models/.

Example
    python scripts/extract_mediapipe.py --videos data/keraal/videos --out data/keraal/mediapipe \
        --model models/pose_landmarker_full.task
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def extract(video: Path, landmarker_factory) -> dict:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    image, world = [], []
    frame_i = 0
    with landmarker_factory() as landmarker:  # fresh instance per video: timestamps restart
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_i * 1000.0 / fps)
            res = landmarker.detect_for_video(mp_img, ts_ms)
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                wl = res.pose_world_landmarks[0]
                image.append([[p.x, p.y, p.z, p.visibility] for p in lm])
                world.append([[p.x, p.y, p.z] for p in wl])
            else:
                image.append(np.full((33, 4), np.nan))
                world.append(np.full((33, 3), np.nan))
            frame_i += 1
    cap.release()
    return {"image": np.asarray(image, np.float32), "world": np.asarray(world, np.float32),
            "fps": np.float32(fps)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="folder searched recursively for videos")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True, help="path to a pose_landmarker_*.task file")
    p.add_argument("--limit", type=int, default=0, help="process only the first N videos (0 = all)")
    args = p.parse_args()

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        sys.exit("mediapipe is not installed: pip install -r requirements.txt")

    if not Path(args.model).exists():
        sys.exit(f"Model file not found: {args.model} (see Setup in this file's docstring)")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    def factory():
        return vision.PoseLandmarker.create_from_options(options)

    videos = sorted(v for v in Path(args.videos).rglob("*") if v.suffix.lower() in VIDEO_EXT)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        sys.exit(f"No videos found under {args.videos}")

    out_root = Path(args.out)
    for i, v in enumerate(videos, 1):
        rel = v.relative_to(args.videos).with_suffix(".npz")
        dst = out_root / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = extract(v, factory)
        missing = float(np.isnan(data["image"][:, 0, 0]).mean()) if len(data["image"]) else 1.0
        np.savez_compressed(dst, **data)
        print(f"[{i}/{len(videos)}] {rel}: {len(data['image'])} frames, {missing:.0%} without a person")


if __name__ == "__main__":
    main()
