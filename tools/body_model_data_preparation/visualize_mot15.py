"""
MOT15 Ground Truth Visualizer
Features: Bottom-aligned text, raw stat outputs (ID, Conf)

Usage (To browse the raw stats on the first 10 frames):
    python visualize_mot15_gt.py /data/MOT15/train/PETS09-S2L2 ./debug_output --max-results 10
"""

import cv2
import os
import argparse
from pathlib import Path
from collections import defaultdict

COLORS = {
    "ignore_conf_0": (0, 0, 255),   # Merah: Ignore region / Evaluasi diabaikan
    "normal": (0, 255, 0),          # Hijau: Target Pejalan Kaki (Pedestrian) Normal
}

def visualize_mot15_gt(seq_dir: str, output_dir: str, max_results: int):
    seq_path = Path(seq_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gt_file = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "img1"

    if not gt_file.exists():
        print(f"Error: Could not find gt.txt in {gt_file}")
        return

    print(f"Scanning {seq_path.name}...")
    
    frame_anns = defaultdict(list)

    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            
            # Memastikan minimal ada 7 kolom (Frame sampai Conf)
            if len(parts) < 7:
                continue
                
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            bb_left, bb_top, bb_width, bb_height = map(float, parts[2:6])
            conf = float(parts[6])
            
            # Kolom 7, 8, 9 pada MOT15 adalah koordinat 3D (x, y, z), 
            # bukan Class dan Visibility, jadi kita abaikan saja.

            frame_anns[frame_id].append({
                "id": obj_id,
                "bbox": [int(bb_left), int(bb_top), int(bb_width), int(bb_height)],
                "conf": conf
            })

    frames_to_process = sorted(frame_anns.keys())

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
            conf, obj_id = ann["conf"], ann["id"]

            # --- Label & Color Logic ---
            if conf == 0:
                color = COLORS["ignore_conf_0"]
                label = f"ID: {obj_id} | Conf: {conf:.0f}"
            else:
                color = COLORS["normal"]
                label = f"ID: {obj_id} | Conf: {conf:.0f}"

            # 1. Draw Bounding Box Rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # 2. Calculate text size
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 3. Draw Text Background at the BOTTOM (y + h)
            cv2.rectangle(img, (x, y + h), (x + text_w, y + h + text_h + 4), color, -1)
            
            # 4. Draw Text over the background
            cv2.putText(img, label, (x, y + h + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(str(out_path / img_name), img)

    print(f"Done! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MOT15 Ground Truth")
    parser.add_argument("seq_dir", help="Path to MOT sequence")
    parser.add_argument("output_dir", help="Directory to save images")
    parser.add_argument("--max-results", type=int, default=10, help="Max images to generate (default: 10)")
    
    args = parser.parse_args()
    visualize_mot15_gt(args.seq_dir, args.output_dir, args.max_results)