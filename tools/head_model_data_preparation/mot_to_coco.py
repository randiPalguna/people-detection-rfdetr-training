#!/usr/bin/env python3
"""Convert a single MOT sequence to COCO format with optional FPS subsampling."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Tuple

TARGET_CLASS_IDS = {1, 2, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one MOT sequence to COCO format (single class: head)."
    )
    parser.add_argument(
        "seq_dir",
        type=Path,
        help="Path to MOT sequence (contains gt/gt.txt and images)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for COCO folder",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target FPS for subsampling (e.g., 5)",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Frame offset after subsampling (e.g., 1 for 2, 27, 52...)",
    )
    return parser.parse_args()


def read_seqinfo(seq_dir: Path) -> Dict[str, str]:
    seqinfo = {}
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        return seqinfo

    with seqinfo_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            seqinfo[key.strip()] = value.strip()
    return seqinfo


def get_original_fps(seqinfo: Dict[str, str]) -> float:
    try:
        return float(seqinfo.get("frameRate", 25.0))
    except ValueError:
        return 25.0


def resolve_image_dir(seq_dir: Path, seqinfo: Dict[str, str]) -> Path:
    candidates = []
    im_dir = seqinfo.get("imDir")
    if im_dir:
        candidates.append(seq_dir / im_dir)
    candidates.extend([seq_dir / "img1", seq_dir / "img"])

    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    raise FileNotFoundError(
        "Could not find image directory. Checked imDir, img1, and img."
    )


def normalize_ext(ext: str | None) -> str | None:
    if not ext:
        return None
    ext = ext.strip()
    if not ext:
        return None
    return ext.lower() if ext.startswith(".") else f".{ext.lower()}"


def build_frame_to_file(img_dir: Path, ext: str | None) -> Dict[int, str]:
    frame_to_file: Dict[int, str] = {}
    for img_path in img_dir.iterdir():
        if not img_path.is_file():
            continue
        if ext and img_path.suffix.lower() != ext:
            continue
        stem = img_path.stem
        if not stem.isdigit():
            continue
        frame_id = int(stem)
        frame_to_file[frame_id] = img_path.name
    return frame_to_file


def build_valid_frames(
    frames: Iterable[int],
    original_fps: float,
    target_fps: float | None,
    frame_offset: int,
) -> set[int]:
    frames = sorted(frames)
    if not frames:
        return set()
    if target_fps is None or target_fps >= original_fps:
        return set(frames)

    ratio = original_fps / target_fps
    max_frame = max(frames)
    valid_frames = set()
    idx = 0
    while True:
        frame_id = round(idx * ratio) + 1 + frame_offset
        if frame_id > max_frame:
            break
        valid_frames.add(frame_id)
        idx += 1
    return valid_frames


def get_image_size(path: Path) -> Tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def load_annotations(gt_path: Path) -> Dict[int, list[Tuple[float, float, float, float]]]:
    frame_to_anns: Dict[int, list[Tuple[float, float, float, float]]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue

            try:
                frame_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[2:6])
            except ValueError:
                continue

            if w <= 0 or h <= 0:
                continue

            cls_id = None
            if len(parts) > 7:
                try:
                    cls_id = int(float(parts[7]))
                except ValueError:
                    cls_id = None

            if cls_id not in TARGET_CLASS_IDS:
                continue

            vis = 1.0
            if len(parts) > 8:
                try:
                    vis = float(parts[8])
                except ValueError:
                    vis = 1.0

            if vis <= 0.0:
                continue

            frame_to_anns[frame_id].append((x, y, w, h))

    return frame_to_anns


def main() -> int:
    args = parse_args()
    seq_dir = args.seq_dir
    output_dir = args.output_dir

    if args.target_fps is not None and args.target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    if args.frame_offset < 0:
        raise ValueError("--frame-offset must be >= 0")

    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        print(f"Error: gt.txt not found at {gt_path}")
        return 1

    seqinfo = read_seqinfo(seq_dir)
    original_fps = get_original_fps(seqinfo)
    img_dir = resolve_image_dir(seq_dir, seqinfo)
    ext = normalize_ext(seqinfo.get("imExt"))

    frame_to_file = build_frame_to_file(img_dir, ext)
    if not frame_to_file:
        print(f"Error: No images found in {img_dir}")
        return 1

    frames = sorted(frame_to_file.keys())
    valid_frames = build_valid_frames(
        frames, original_fps, args.target_fps, args.frame_offset
    )
    selected_frames = [f for f in frames if f in valid_frames]

    frame_to_anns = load_annotations(gt_path)

    width = None
    height = None
    if "imWidth" in seqinfo and "imHeight" in seqinfo:
        try:
            width = int(seqinfo["imWidth"])
            height = int(seqinfo["imHeight"])
        except ValueError:
            width = None
            height = None

    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in selected_frames:
        file_name = frame_to_file.get(frame_id)
        if not file_name:
            continue

        src_path = img_dir / file_name
        if not src_path.exists():
            continue

        if width is None or height is None:
            width, height = get_image_size(src_path)

        if width is None or height is None:
            print(f"Warning: Unable to read image size for {src_path}")
            return 1

        dst_path = images_out_dir / file_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

        images.append(
            {
                "id": img_id,
                "file_name": str(PurePosixPath("images") / file_name),
                "width": width,
                "height": height,
            }
        )

        for x, y, w, h in frame_to_anns.get(frame_id, []):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    coco = {
        "info": {
            "description": f"MOT sequence {seq_dir.name}",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "head", "supercategory": "head"},
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / "annotations.json"
    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"Images written: {len(images)}")
    print(f"Annotations written: {len(annotations)}")
    print(f"COCO annotations: {annotations_path}")
    print(f"Images folder: {images_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
