import os
import json
import cv2
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

def get_seq_info(seq_path: Path):
    """Membaca informasi seqinfo.ini untuk mendapatkan FPS dan dimensi gambar."""
    seqinfo_file = seq_path / "seqinfo.ini"
    fps = 30.0
    width = 1920
    height = 1080
    seq_length = 0

    if seqinfo_file.exists():
        with open(seqinfo_file, "r") as f:
            for line in f:
                if line.startswith("frameRate"):
                    fps = float(line.split("=")[1].strip())
                elif line.startswith("imWidth"):
                    width = int(line.split("=")[1].strip())
                elif line.startswith("imHeight"):
                    height = int(line.split("=")[1].strip())
                elif line.startswith("seqLength"):
                    seq_length = int(line.split("=")[1].strip())
    else:
        print(f"Warning: seqinfo.ini tidak ditemukan di {seq_path}. Menggunakan default (30FPS, 1080p).")
    
    return fps, width, height, seq_length

def convert_mot_to_coco(seq_dir: str, output_dataset_dir: str, target_fps: float = 5.0):
    """Mengonversi MOT17/MOT20 ke format COCO JSON dan menyalin gambar ke folder Roboflow-ready."""
    seq_path = Path(seq_dir)
    gt_file = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "img1"
    
    out_dataset_path = Path(output_dataset_dir)
    out_dataset_path.mkdir(parents=True, exist_ok=True)
    
    if not gt_file.exists():
        print(f"Error: {gt_file} tidak ditemukan!")
        return None

    original_fps, width, height, seq_length = get_seq_info(seq_path)
    print(f"[{seq_path.name}] Original FPS: {original_fps} | Target FPS: {target_fps}")

    # 1. Menghitung Frame Valid untuk Subsampling
    valid_frames = set()
    if target_fps < original_fps:
        ratio = original_fps / target_fps
        idx = 0
        while True:
            frame_id = round(idx * ratio) + 1
            if frame_id > seq_length:
                break
            valid_frames.add(frame_id)
            idx += 1
    else:
        valid_frames = set(range(1, seq_length + 1))

    # 2. Setup Struktur COCO
    coco_data = {
        "info": {"description": f"MOT17/20 {seq_path.name} converted to COCO (Subsampled {target_fps} FPS)"},
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "images": [],
        "annotations": []
    }

    # 3. Menambahkan gambar ke list COCO 'images' DAN menyalin (copy) gambar ke folder output
    image_id_map = {} 
    copied_images_count = 0
    
    for frame_id in sorted(list(valid_frames)):
        img_name = f"{frame_id:06d}.jpg"
        image_id = frame_id 
        image_id_map[frame_id] = image_id
        
        # Proses salin gambar dari MOT seq_dir ke direktori Roboflow dataset
        src_img = img_dir / img_name
        dst_img = out_dataset_path / img_name
        
        if src_img.exists():
            shutil.copy(str(src_img), str(dst_img))
            copied_images_count += 1
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": width,
                "height": height,
                "frame_id": frame_id
            })

    # 4. Parsing gt.txt dan memasukkan ke COCO 'annotations'
    ann_id = 1
    # Kelas yang relevan berdasarkan paper MOT16/17/20: 
    # 1: Pedestrian, 2: Person on vehicle, 7: Static person
    TARGET_CLASSES = {1, 2, 7} 
    
    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
                
            frame_id = int(parts[0])
            
            # Lewati jika frame tidak masuk dalam daftar subsampling
            if frame_id not in valid_frames:
                continue

            obj_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            
            # Format MOT17/20: kolom ke-7=conf, ke-8=class, ke-9=visibility
            conf = float(parts[6])
            cls_id = int(float(parts[7]))
            vis = float(parts[8])

            # --- ATURAN FILTERING SESUAI REQUIREMENT ---
            # 1. Class: Hanya 1, 2, atau 7
            if cls_id not in TARGET_CLASSES:
                continue
                
            # 3. Visibility: >= 0.02
            if vis < 0.02:
                continue

            # Tambahkan ke anotasi COCO
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": image_id_map[frame_id],
                "category_id": 1,               # Semua target dijadikan "person" (ID 1)
                "bbox": [bb_left, bb_top, bb_width, bb_height],
                "area": bb_width * bb_height,
                "track_id": obj_id,             # Menyimpan ID tracking (bonus info)
                "mot_class": cls_id,            # Menyimpan kelas asli MOT (bonus info)
                "visibility": vis,               # Menyimpan tingkat visibilitas (bonus info)
                "iscrowd": 0,                   # Iscrowd dipaksa 0 untuk semua bbox yang disimpan
            })
            ann_id += 1

    # 5. Simpan File JSON dengan nama standar Roboflow
    out_json_path = out_dataset_path / "_annotations.coco.json"
    with open(out_json_path, "w") as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"✅ Konversi & Penyalinan Selesai! Tersimpan di: {out_dataset_path}")
    print(f"   Total Gambar Disalin: {copied_images_count}")
    print(f"   Total Anotasi Valid: {len(coco_data['annotations'])}")
    
    return out_json_path, out_dataset_path

def visualize_coco_json(dataset_dir: Path, json_path: Path, output_dir: str, max_results: int = 10):
    """Membaca COCO JSON dan menggambar bounding box dari direktori dataset baru."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Kelompokkan anotasi berdasarkan image_id
    anns_by_img = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    print(f"\nMembuat Visualisasi untuk {min(max_results, len(coco_data['images']))} gambar pertama...")

    count = 0
    for img_info in coco_data["images"]:
        if count >= max_results:
            break

        img_name = img_info["file_name"]
        img_id = img_info["id"]
        img_file = dataset_dir / img_name  # Baca gambar dari dataset Roboflow yang baru
        
        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        
        # Gambar semua anotasi untuk gambar ini
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = map(int, ann["bbox"])
            track_id = ann.get("track_id", "?")
            mot_class = ann.get("mot_class", "?")
            vis = ann.get("visibility", 0)
 
            # Warna bisa dibedakan berdasarkan kelas asli MOT untuk kemudahan debugging
            if mot_class == 1:
                color = (0, 255, 0)      # Hijau: Pedestrian
            elif mot_class == 2:
                color = (255, 165, 0)    # Biru/Oranye: Person on Vehicle
            elif mot_class == 7:
                color = (0, 255, 255)    # Kuning: Static Person
            else:
                color = (255, 255, 255)  # Putih: Unknown

            label = f"ID:{track_id} C:{mot_class} V:{vis:.2f}"

            # Gambar Box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            
            # Gambar Label
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x, y + h), (x + text_w, y + h + text_h + 4), color, -1)
            cv2.putText(img, label, (x, y + h + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        out_file = out_path / img_name
        cv2.imwrite(str(out_file), img)
        count += 1

    print(f"✅ Visualisasi Selesai! Cek folder '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MOT17/MOT20 to Roboflow-ready COCO and Visualize")
    parser.add_argument("seq_dir", help="Path ke folder sequence MOT17/20 asli (ex: /data/MOT20/train/MOT20-05)")
    parser.add_argument("output_dataset_dir", help="Folder tujuan untuk menyimpan Roboflow dataset (JSON + Images)")
    parser.add_argument("output_visual_dir", help="Folder untuk menyimpan hasil gambar visualisasi")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Target FPS untuk subsampling (default: 5.0)")
    parser.add_argument("--max-visual", type=int, default=10, help="Maksimal gambar yang divisualisasi (default: 10)")
    
    args = parser.parse_args()
    
    # 1. Jalankan Konversi & Pembuatan Direktori Dataset
    json_result_path, new_dataset_path = convert_mot_to_coco(
        seq_dir=args.seq_dir, 
        output_dataset_dir=args.output_dataset_dir,
        target_fps=args.target_fps
    )
    
    # 2. Jalankan Visualisasi dengan membaca dari direktori yang baru saja dibuat
    if json_result_path:
        visualize_coco_json(
            dataset_dir=new_dataset_path, 
            json_path=json_result_path, 
            output_dir=args.output_visual_dir, 
            max_results=args.max_visual
        )