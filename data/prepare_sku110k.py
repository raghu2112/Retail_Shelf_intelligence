"""
SKU-110K Dataset Preparation Script
=====================================
What this does:
  1. Reads SKU-110K CSV annotations
  2. Converts them to YOLOv8 format (normalized xywh)
  3. Copies a subset of images to data/processed/
  4. Creates the dataset YAML config for YOLOv8

SKU-110K annotation format (CSV columns):
  image_name, x1, y1, x2, y2, class, image_width, image_height

YOLOv8 annotation format (per line in .txt):
  class_id cx cy w h   (all normalized 0-1)

HOW TO USE:
  1. Download SKU-110K from: https://github.com/eg4000/SKU110K_CVPR19
  2. Place images in:        data/raw/SKU-110K/images/
  3. Place CSVs in:          data/raw/SKU-110K/annotations/
  4. Run: python data/prepare_sku110k.py
"""

import os
import csv
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

RAW_DIR         = Path("data/raw/SKU-110K")
PROCESSED_DIR   = Path("data/processed")
ANNOTATIONS_DIR = RAW_DIR / "annotations"
IMAGES_DIR      = RAW_DIR / "images"

MAX_TRAIN_IMAGES = 200
MAX_VAL_IMAGES   = 50
CLASS_NAMES      = ["product"]


def convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def read_sku110k_csv(csv_path):
    annotations = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            img_name, x1, y1, x2, y2, cls, img_w, img_h = row[:8]
            try:
                annotations[img_name].append((
                    float(x1), float(y1), float(x2), float(y2),
                    int(img_w), int(img_h)
                ))
            except ValueError:
                continue
    return annotations


def write_yolo_labels(label_path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for (x1, y1, x2, y2, img_w, img_h) in boxes:
            cx, cy, w, h = convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h)
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_split(split, csv_path, max_images, out_dir):
    if not csv_path.exists():
        print(f"  [SKIP] CSV not found: {csv_path}")
        return 0

    print(f"\nProcessing {split} from {csv_path.name} ...")
    annotations = read_sku110k_csv(csv_path)
    image_names = list(annotations.keys())
    random.shuffle(image_names)
    image_names = image_names[:max_images]

    img_out   = out_dir / split / "images"
    label_out = out_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    processed, skipped = 0, 0
    for img_name in image_names:
        img_src = IMAGES_DIR / img_name
        if not img_src.exists():
            candidates = list(IMAGES_DIR.rglob(img_name))
            if not candidates:
                skipped += 1
                continue
            img_src = candidates[0]
        shutil.copy2(img_src, img_out / img_name)
        write_yolo_labels(label_out / (Path(img_name).stem + ".txt"), annotations[img_name])
        processed += 1
        if processed % 50 == 0:
            print(f"  {processed}/{len(image_names)} done ...")

    print(f"  Done: {processed} processed, {skipped} skipped")
    return processed


def create_dataset_yaml(out_dir):
    yaml_path = out_dir / "dataset.yaml"
    config = {
        "path":  str(out_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nDataset YAML -> {yaml_path}")
    return yaml_path


def main():
    print("=" * 50)
    print("SKU-110K -> YOLOv8 Preparation")
    print("=" * 50)
    random.seed(42)

    if not RAW_DIR.exists():
        print(f"\n[ERROR] Not found: {RAW_DIR}")
        print("Download SKU-110K and place at that path.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (ANNOTATIONS_DIR / "annotations_train.csv", MAX_TRAIN_IMAGES),
        "val":   (ANNOTATIONS_DIR / "annotations_val.csv",   MAX_VAL_IMAGES),
    }

    total = 0
    for split, (csv_path, max_imgs) in splits.items():
        total += process_split(split, csv_path, max_imgs, PROCESSED_DIR)

    create_dataset_yaml(PROCESSED_DIR)
    print(f"\nTotal images prepared: {total}")


if __name__ == "__main__":
    main()
