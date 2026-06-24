"""
MOT Ground Truth Verbose Visualizer (with FPS Subsampling)
Features: Bottom-aligned text, raw stat outputs (Class, Vis, Conf), Auto FPS Subsampling

Usage:
    python visualize_mot_gt.py /data/MOT20/train/MOT20-05 ./debug_output --target-fps 15 --max-results 20
"""

import cv2
import os
import argparse
from pathlib import Path
from collections import defaultdict

COLORS = {
    "ignore_conf_0": (0, 0, 255),   # Red: Ignore region (Usually crowds in MOT20-05)
    "target_search": (0, 165, 255), # Orange: The specific class you are searching for
    "normal": (0, 255, 0),          # Green: Normal classes
}

def get_original_fps(seq_path: Path, override_fps: float) -> float:
    """Reads the original FPS from seqinfo.ini, or uses the override."""
    if override_fps is not None:
        return override_fps
        
    seqinfo_file = seq_path / "seqinfo.ini"
    if seqinfo_file.exists():
        with open(seqinfo_file, "r") as f:
            for line in f:
                if line.startswith("frameRate"):
                    return float(line.split("=")[1].strip())
    
    print("Warning: seqinfo.ini not found. Defaulting to 30.0 FPS.")
    return 30.0

def visualize_mot_gt(seq_dir: str, output_dir: str, find_class: int, max_results: int, target_fps: float, override_fps: float):
    seq_path = Path(seq_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gt_file = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "img1"

    if not gt_file.exists():
        print(f"Error: Could not find gt.txt in {gt_file}")
        return

    print(f"Scanning {seq_path.name}...")
    original_fps = get_original_fps(seq_path, override_fps)
    
    frame_anns = defaultdict(list)
    frames_with_target = set()

    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
                
            frame_id = int(parts[0])
            bb_left, bb_top, bb_width, bb_height = map(float, parts[2:6])
            conf = float(parts[6])
            cls_id = int(float(parts[7]))
            vis = float(parts[8]) if len(parts) > 8 else 1.0

            if find_class is not None and cls_id == find_class:
                frames_with_target.add(frame_id)

            frame_anns[frame_id].append({
                "bbox": [int(bb_left), int(bb_top), int(bb_width), int(bb_height)],
                "conf": conf, "cls": cls_id, "vis": vis
            })

    if find_class is not None:
        frames_to_process = sorted(list(frames_with_target))
        if not frames_to_process:
            print(f"❌ Search complete: Class {find_class} was NOT FOUND in this sequence.")
            return
        print(f"✅ Found Class {find_class} in {len(frames_to_process)} frames!")
    else:
        frames_to_process = sorted(frame_anns.keys())

    # --- SUBSAMPLING LOGIC ---
    if target_fps is not None and target_fps < original_fps:
        print(f"Subsampling from {original_fps} FPS to {target_fps} FPS...")
        ratio = original_fps / target_fps
        max_frame = max(frames_to_process) if frames_to_process else 0
        
        valid_frames = set()
        idx = 0
        while True:
            # Kalkulasi frame berbasis rasio agar bisa menangani pembagian tidak bulat (misal 25 ke 15)
            # MOT frames index dimulai dari 1
            frame_id = round(idx * ratio) + 1
            if frame_id > max_frame:
                break
            valid_frames.add(frame_id)
            idx += 1
            
        frames_to_process = [f for f in frames_to_process if f in valid_frames]
        print(f"Frames reduced to {len(frames_to_process)} total frames.")

    # Terapkan max_results setelah subsampling agar Anda bisa melihat deretan frame yang sudah di-subsample
    if max_results > 0:
        frames_to_process = frames_to_process[:max_results]

    print(f"Drawing bounding boxes for {len(frames_to_process)} frames...")

    for frame_id in frames_to_process:
        img_name = f"{frame_id:06d}.jpg"
        img_file = img_dir / img_name
        
        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        
        for ann in frame_anns[frame_id]:
            x, y, w, h = ann["bbox"]
            conf, cls_id, vis = ann["conf"], ann["cls"], ann["vis"]

            # --- Label & Color Logic ---
            if find_class is not None and cls_id == find_class:
                color = COLORS["target_search"]
                label = f"TARGET: Cls {cls_id} {vis:.2f} {conf}"
            elif conf == 0:
                color = COLORS["ignore_conf_0"]
                label = "" # Label dimatikan seperti kode asli Anda
            else:
                color = COLORS["normal"]
                label = "" # Label dimatikan seperti kode asli Anda

            # 1. Draw Bounding Box Rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            
            if label:
                # 2. Calculate text size
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # 3. Draw Text Background at the BOTTOM (y + h)
                cv2.rectangle(img, (x, y + h), (x + text_w, y + h + text_h + 4), color, -1)
                
                # 4. Draw Text over the background
                cv2.putText(img, label, (x, y + h + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(str(out_path / img_name), img)

    print(f"Done! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MOT Ground Truth")
    parser.add_argument("seq_dir", help="Path to MOT sequence")
    parser.add_argument("output_dir", help="Directory to save images")
    parser.add_argument("--find-class", type=int, default=None, help="Only extract frames containing this specific Class ID")
    parser.add_argument("--max-results", type=int, default=10, help="Max images to generate (default: 10)")
    
    # Argumen Subsampling Baru
    parser.add_argument("--target-fps", type=float, default=None, help="Target FPS to subsample the dataset to (e.g., 15)")
    parser.add_argument("--original-fps", type=float, default=None, help="Force original FPS (will auto-detect from seqinfo.ini if not provided)")
    
    args = parser.parse_args()
    visualize_mot_gt(args.seq_dir, args.output_dir, args.find_class, args.max_results, args.target_fps, args.original_fps)