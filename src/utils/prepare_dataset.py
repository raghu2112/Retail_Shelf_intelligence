# src/utils/prepare_dataset.py
#
# PURPOSE:
#   Convert SKU-110K raw annotations (CSV format) into YOLOv8-compatible
#   folder structure and label files, then create the dataset YAML.
#
# SKU-110K download:
#   https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd
#   After extraction you should have:
#       data/raw/
#           images/         (train/, val/, test/ subfolders with .jpg files)
#           annotations/    (annotations_train.csv, annotations_val.csv, annotations_test.csv)
#
# USAGE:
#   python -m src.utils.prepare_dataset
#
# OUTPUT:
#   data/processed/
#       images/train/   ← copied/symlinked images
#       images/val/
#       labels/train/   ← one .txt per image in YOLO format
#       labels/val/
#   models/configs/sku110k.yaml

import os
import shutil
import random
import csv
from pathlib import Path
from tqdm import tqdm
import yaml

# Add project root to path so config imports work
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_sku110k_csv(csv_path: str) -> dict:
    """
    Parse SKU-110K annotation CSV.

    CSV columns (no header in file):
        image_name, x1, y1, x2, y2, class, image_width, image_height

    Returns:
        dict mapping image_name → list of (x1, y1, x2, y2, class, w, h)
    """
    annotations = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            img_name, x1, y1, x2, y2, cls, img_w, img_h = (
                row[0], float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), row[5],
                float(row[6]), float(row[7])
            )
            if img_name not in annotations:
                annotations[img_name] = []
            annotations[img_name].append((x1, y1, x2, y2, cls, img_w, img_h))
    return annotations


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h) -> tuple:
    """
    Convert absolute [x1 y1 x2 y2] → YOLO normalised [cx cy w h].
    All values are in range [0, 1].
    """
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp to valid range (some annotations have tiny float errors)
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
    return cx, cy, w, h


def write_yolo_label(label_path: str, boxes: list, class_id: int = 0):
    """
    Write a YOLO-format label file.
    SKU-110K has only one class: 'object' (retail product).
    We map it to class_id 0.
    """
    with open(label_path, "w") as f:
        for (x1, y1, x2, y2, cls, img_w, img_h) in boxes:
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            if w > 0 and h > 0:  # skip degenerate boxes
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_split(
    split_name: str,
    raw_images_dir: str,
    annotations: dict,
    out_images_dir: str,
    out_labels_dir: str,
    subset_size: int = None
):
    """
    Copy images and write label files for one dataset split (train/val).

    Args:
        split_name:      'train' or 'val'
        raw_images_dir:  path to raw images for this split
        annotations:     dict from read_sku110k_csv()
        out_images_dir:  where to copy images
        out_labels_dir:  where to write .txt labels
        subset_size:     if set, only process this many images
    """
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # Find all images that have annotations
    available = [
        img for img in os.listdir(raw_images_dir)
        if img in annotations and img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not available:
        print(f"  [WARNING] No annotated images found in {raw_images_dir}")
        print(f"  Make sure your CSV filenames match the image filenames exactly.")
        return 0

    # Optionally limit to a subset for faster experimentation
    if subset_size and len(available) > subset_size:
        random.seed(42)
        available = random.sample(available, subset_size)

    print(f"  Processing {len(available)} images for split='{split_name}'")

    processed = 0
    for img_name in tqdm(available, desc=f"  {split_name}"):
        src_img = os.path.join(raw_images_dir, img_name)
        dst_img = os.path.join(out_images_dir, img_name)
        dst_lbl = os.path.join(out_labels_dir, img_name.rsplit(".", 1)[0] + ".txt")

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Write label
        boxes = annotations[img_name]
        write_yolo_label(dst_lbl, boxes)
        processed += 1

    return processed


def write_dataset_yaml(out_path: str, processed_dir: str, class_names: list):
    """
    Write the YOLOv8 dataset YAML file.
    YOLOv8 reads this to find images and class names.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {
        "path": str(Path(processed_dir).resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"\n  Dataset YAML written → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SKU-110K → YOLOv8 Dataset Preparation")
    print("=" * 60)

    # ── Validate raw data exists ──────────────────────────────────────────────
    raw_images_root  = os.path.join(cfg.RAW_DIR, "images")
    annotations_root = os.path.join(cfg.RAW_DIR, "annotations")

    if not os.path.exists(raw_images_root):
        print(f"\n[ERROR] Raw images folder not found: {raw_images_root}")
        print("Please download SKU-110K and extract it so the structure is:")
        print("  data/raw/images/train/")
        print("  data/raw/images/val/")
        print("  data/raw/annotations/annotations_train.csv")
        print("  data/raw/annotations/annotations_val.csv")
        return

    # ── Load annotations ──────────────────────────────────────────────────────
    splits = {
        "train": {
            "csv":    os.path.join(annotations_root, "annotations_train.csv"),
            "images": raw_images_root,
        },
        "val": {
            "csv":    os.path.join(annotations_root, "annotations_val.csv"),
            "images": raw_images_root,
        },
    }

    # Compute subset sizes per split
    # 85% of SUBSET_SIZE for train, 15% for val
    if cfg.SUBSET_SIZE:
        train_subset = int(cfg.SUBSET_SIZE * cfg.TRAIN_RATIO)
        val_subset   = cfg.SUBSET_SIZE - train_subset
    else:
        train_subset = val_subset = None

    total = 0
    for split_name, paths in splits.items():
        print(f"\n[{split_name.upper()}]")

        if not os.path.exists(paths["csv"]):
            print(f"  [SKIP] CSV not found: {paths['csv']}")
            continue

        print(f"  Reading annotations from: {paths['csv']}")
        annotations = read_sku110k_csv(paths["csv"])
        print(f"  Found {len(annotations)} annotated images in CSV")

        out_images = os.path.join(cfg.PROCESSED_DIR, "images", split_name)
        out_labels = os.path.join(cfg.PROCESSED_DIR, "labels", split_name)

        subset = train_subset if split_name == "train" else val_subset
        n = process_split(
            split_name     = split_name,
            raw_images_dir = paths["images"],
            annotations    = annotations,
            out_images_dir = out_images,
            out_labels_dir = out_labels,
            subset_size    = subset,
        )
        total += n

    # ── Write dataset YAML ────────────────────────────────────────────────────
    # SKU-110K has one class: generic retail product
    class_names = ["product"]
    write_dataset_yaml(cfg.DATASET_YAML, cfg.PROCESSED_DIR, class_names)

    print(f"\n{'=' * 60}")
    print(f"Done. Total images processed: {total}")
    print(f"Processed data → {cfg.PROCESSED_DIR}")
    print(f"Dataset YAML   → {cfg.DATASET_YAML}")
    print("=" * 60)
    print("\nNext step: run training with:")
    print("  python -m src.detection.train")


if __name__ == "__main__":
    main()
