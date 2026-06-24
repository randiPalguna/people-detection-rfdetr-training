#!/usr/bin/env python3
"""Extract one CCHead sequence from a COCO file into a Roboflow-ready folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a single CCHead sequence into COCO format (single class: head)."
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="Path to CCHead COCO annotations (train.json or test.json)",
    )
    parser.add_argument(
        "image_root",
        type=Path,
        help="Root directory used by file_name entries in the COCO file",
    )
    parser.add_argument(
        "sequence",
        type=str,
        help="Sequence name (e.g., 90T_25fps)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the Roboflow-ready folder",
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


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frame_id_from_name(file_name: str) -> int | None:
    stem = PurePosixPath(file_name).stem
    if stem.isdigit():
        return int(stem)
    return None


def sequence_from_name(file_name: str) -> str | None:
    parts = PurePosixPath(file_name).parts
    if not parts:
        return None
    for idx, part in enumerate(parts):
        if part in {"img", "img1"} and idx > 0:
            return parts[idx - 1]
    return parts[0]


def list_sequences(images: Iterable[dict]) -> list[str]:
    sequences = set()
    for img in images:
        file_name = img.get("file_name", "")
        seq = sequence_from_name(file_name)
        if seq:
            sequences.add(seq)
    return sorted(sequences)


def resolve_sequence_dir(image_root: Path, file_name: str, sequence: str) -> Path | None:
    parts = PurePosixPath(file_name).parts
    if sequence not in parts:
        return None
    idx = parts.index(sequence)
    seq_rel = PurePosixPath(*parts[: idx + 1])
    return image_root / seq_rel


def read_seqinfo_fps(seq_dir: Path) -> float | None:
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        return None
    with seqinfo_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("frameRate"):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def build_valid_frames(
    frame_ids: Iterable[int],
    original_fps: float,
    target_fps: float | None,
    frame_offset: int,
) -> set[int]:
    frame_ids = sorted(frame_ids)
    if not frame_ids:
        return set()
    if target_fps is None or target_fps >= original_fps:
        return set(frame_ids)

    ratio = original_fps / target_fps
    max_frame = max(frame_ids)
    valid_frames = set()
    idx = 0
    while True:
        frame_id = round(idx * ratio) + 1 + frame_offset
        if frame_id > max_frame:
            break
        valid_frames.add(frame_id)
        idx += 1
    return valid_frames


def main() -> int:
    args = parse_args()

    if args.target_fps is not None and args.target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    if args.frame_offset < 0:
        raise ValueError("--frame-offset must be >= 0")

    coco = load_coco(args.annotations)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    selected_images = [
        img for img in images
        if sequence_from_name(img.get("file_name", "")) == args.sequence
    ]

    if not selected_images:
        print(f"No images found for sequence '{args.sequence}'.")
        sequences = list_sequences(images)
        if sequences:
            print("Available sequences:")
            for seq in sequences:
                print(f"- {seq}")
        return 1

    def sort_key(img: dict):
        file_name = img.get("file_name", "")
        frame_id = frame_id_from_name(file_name)
        return (frame_id is None, frame_id if frame_id is not None else file_name)

    selected_images.sort(key=sort_key)

    frame_ids = [
        fid
        for fid in (frame_id_from_name(img.get("file_name", "")) for img in selected_images)
        if fid is not None
    ]

    original_fps = 25.0
    if selected_images:
        seq_dir = resolve_sequence_dir(
            args.image_root, selected_images[0].get("file_name", ""), args.sequence
        )
        if seq_dir is not None:
            seq_fps = read_seqinfo_fps(seq_dir)
            if seq_fps is not None:
                original_fps = seq_fps

    valid_frames = set(frame_ids)
    if args.target_fps is not None and frame_ids:
        valid_frames = build_valid_frames(
            frame_ids, original_fps, args.target_fps, args.frame_offset
        )
        selected_images = [
            img for img in selected_images
            if frame_id_from_name(img.get("file_name", "")) in valid_frames
        ]

    selected_image_ids = {img["id"] for img in selected_images if "id" in img}
    if not selected_image_ids:
        print("No valid images after filtering.")
        return 1

    annotations_out = []
    images_out = []
    image_id_map: Dict[int, int] = {}

    images_out_dir = args.output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    new_image_id = 1
    for img in selected_images:
        old_id = img.get("id")
        file_name = img.get("file_name", "")
        if old_id is None or not file_name:
            continue

        src_path = args.image_root / PurePosixPath(file_name)
        if not src_path.exists():
            print(f"Missing image: {src_path}")
            continue

        dst_name = PurePosixPath(file_name).name
        dst_path = images_out_dir / dst_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

        width = img.get("width")
        height = img.get("height")
        if width is None or height is None:
            try:
                from PIL import Image

                with Image.open(src_path) as image:
                    width, height = image.width, image.height
            except Exception:
                print(f"Warning: could not read size for {src_path}")
                return 1

        images_out.append(
            {
                "id": new_image_id,
                "file_name": str(PurePosixPath("images") / dst_name),
                "width": int(width),
                "height": int(height),
            }
        )
        image_id_map[old_id] = new_image_id
        new_image_id += 1

    new_ann_id = 1
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id not in selected_image_ids:
            continue
        new_image_id = image_id_map.get(image_id)
        if new_image_id is None:
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        area = ann.get("area")
        if area is None:
            area = float(bbox[2] * bbox[3])

        ann_out = {
            "id": new_ann_id,
            "image_id": new_image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "iscrowd": ann.get("iscrowd", 0),
        }

        if "segmentation" in ann:
            ann_out["segmentation"] = ann["segmentation"]

        annotations_out.append(ann_out)
        new_ann_id += 1

    coco_out = {
        "info": {
            "description": f"CCHead sequence {args.sequence}",
            "version": "1.0",
        },
        "images": images_out,
        "annotations": annotations_out,
        "categories": [
            {"id": 1, "name": "head", "supercategory": "head"},
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "annotations.json"
    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(coco_out, f, indent=2)

    print(f"Images written: {len(images_out)}")
    print(f"Annotations written: {len(annotations_out)}")
    print(f"COCO annotations: {annotations_path}")
    print(f"Images folder: {images_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
