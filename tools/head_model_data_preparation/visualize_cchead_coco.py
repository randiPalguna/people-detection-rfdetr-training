import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2


def load_coco(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def build_indices(coco: dict):
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # Build lookup tables for fast access.
    id_to_image = {img["id"]: img for img in images}
    id_to_category = {cat["id"]: cat.get("name", str(cat["id"])) for cat in categories}
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    return images, id_to_image, id_to_category, anns_by_image


def list_sequences(images: list[dict]) -> list[str]:
    sequences = set()
    for img in images:
        file_name = img.get("file_name")
        if not file_name:
            continue
        parts = PurePosixPath(file_name).parts
        if parts:
            sequences.add(parts[0])
    return sorted(sequences)


def parse_color(color_str: str) -> tuple[int, int, int]:
    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError("Color must be in 'B,G,R' format, e.g. 0,255,0")
    return tuple(int(p.strip()) for p in parts)


def get_original_fps(sequence_dir: Path, override_fps: float | None) -> float:
    if override_fps is not None:
        return override_fps

    seqinfo_file = sequence_dir / "seqinfo.ini"
    if seqinfo_file.exists():
        with seqinfo_file.open("r") as f:
            for line in f:
                if line.startswith("frameRate"):
                    return float(line.split("=")[1].strip())

    print("Warning: seqinfo.ini not found. Defaulting to 25.0 FPS.")
    return 25.0


def image_sort_key(img: dict):
    file_name = img.get("file_name", "")
    stem = PurePosixPath(file_name).stem
    try:
        return int(stem)
    except ValueError:
        return file_name


def frame_id_from_img(img: dict) -> int | None:
    stem = PurePosixPath(img.get("file_name", "")).stem
    try:
        return int(stem)
    except ValueError:
        return None


def draw_label(img, text: str, x: int, y: int, color: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x0 = max(0, x)
    y0 = max(0, y - text_h - 6)
    cv2.rectangle(img, (x0, y0), (x0 + text_w + 4, y0 + text_h + 4), color, -1)
    cv2.putText(img, text, (x0 + 2, y0 + text_h + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def visualize_sequence(
    coco_path: Path,
    image_root: Path,
    sequence: str,
    output_dir: Path,
    max_images: int,
    target_fps: float | None,
    original_fps: float | None,
    draw_labels: bool,
    color: tuple[int, int, int],
    thickness: int,
    show: bool,
):
    coco = load_coco(coco_path)
    images, _, id_to_category, anns_by_image = build_indices(coco)

    sequences = list_sequences(images)
    if sequence not in sequences:
        print(f"Sequence '{sequence}' not found in annotations.")
        if sequences:
            print("Available sequences:")
            for name in sequences:
                print(f"- {name}")
        return

    selected_images = []
    for img in images:
        file_name = img.get("file_name", "")
        parts = PurePosixPath(file_name).parts
        if parts and parts[0] == sequence:
            selected_images.append(img)

    selected_images.sort(key=image_sort_key)

    if target_fps is not None:
        seq_dir = image_root / sequence
        source_fps = get_original_fps(seq_dir, original_fps)
        if target_fps < source_fps:
            ratio = source_fps / target_fps
            frame_ids = [frame_id_from_img(img) for img in selected_images]
            frame_ids = [fid for fid in frame_ids if fid is not None]
            max_frame = max(frame_ids) if frame_ids else 0

            valid_frames = set()
            idx = 0
            while True:
                frame_id = round(idx * ratio)
                if frame_id > max_frame:
                    break
                valid_frames.add(frame_id)
                idx += 1

            selected_images = [
                img for img in selected_images
                if frame_id_from_img(img) in valid_frames
            ]

    if max_images > 0:
        selected_images = selected_images[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(selected_images)} images for sequence '{sequence}'.")
    for img in selected_images:
        img_id = img["id"]
        rel_path = PurePosixPath(img["file_name"])
        img_path = image_root / rel_path
        if not img_path.exists():
            print(f"Missing image: {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue

        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            if draw_labels:
                cat_name = id_to_category.get(ann.get("category_id"), "unknown")
                conf = ann.get("conf")
                label = f"{cat_name}" if conf is None else f"{cat_name} {conf:.2f}"
                draw_label(frame, label, x, y, color)

        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)

        if show:
            cv2.imshow("CChead COCO", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    print(f"Done. Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CChead COCO annotations for one sequence")
    parser.add_argument("annotations", help="Path to COCO json (train.json or test.json)")
    parser.add_argument("image_root", help="Root folder that contains sequence directories")
    parser.add_argument("sequence", nargs="?", help="Sequence name (first path segment in file_name)")
    parser.add_argument("output_dir", nargs="?", help="Directory to save rendered images")
    parser.add_argument("--list-sequences", action="store_true", help="List sequences in the COCO file and exit")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--target-fps", type=float, default=None, help="Target FPS to subsample visualization (e.g., 15)")
    parser.add_argument("--original-fps", type=float, default=None, help="Override original FPS (otherwise read from seqinfo.ini)")
    parser.add_argument("--draw-labels", action="store_true", help="Draw category labels on boxes")
    parser.add_argument("--color", default="0,255,0", help="Box color as B,G,R (default: 0,255,0)")
    parser.add_argument("--thickness", type=int, default=2, help="Box line thickness")
    parser.add_argument("--show", action="store_true", help="Display images while saving (press q to quit)")

    args = parser.parse_args()

    coco_path = Path(args.annotations)
    image_root = Path(args.image_root)

    coco = load_coco(coco_path)
    images, _, _, _ = build_indices(coco)
    sequences = list_sequences(images)

    if args.list_sequences:
        print("Sequences found:")
        for name in sequences:
            print(f"- {name}")
        return

    if not args.sequence or not args.output_dir:
        parser.error("sequence and output_dir are required unless --list-sequences is set")

    color = parse_color(args.color)

    visualize_sequence(
        coco_path=coco_path,
        image_root=image_root,
        sequence=args.sequence,
        output_dir=Path(args.output_dir),
        max_images=args.max_images,
        target_fps=args.target_fps,
        original_fps=args.original_fps,
        draw_labels=args.draw_labels,
        color=color,
        thickness=args.thickness,
        show=args.show,
    )


if __name__ == "__main__":
    main()
