# src/detection/train.py
#
# PURPOSE:
#   Train YOLOv8n on the prepared SKU-110K dataset.
#   On CPU this will be slow — 30 epochs on 500 images takes ~2-4 hours.
#   Use EPOCHS=5 in config.py for a quick smoke test first.
#
# USAGE:
#   python -m src.detection.train
#
# OUTPUT:
#   models/checkpoints/best.pt   ← use this for inference
#   models/checkpoints/last.pt   ← last epoch weights
#   runs/detect/train*/          ← training logs, plots (created by ultralytics)

import os
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config as cfg
import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load
from ultralytics import YOLO


def train():
    print("=" * 60)
    print("YOLOv8 Training — Retail Shelf Detection")
    print("=" * 60)
    print(f"  Model:      {cfg.MODEL_NAME}")
    print(f"  Dataset:    {cfg.DATASET_YAML}")
    print(f"  Epochs:     {cfg.EPOCHS}")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Image size: {cfg.IMG_SIZE}")
    print(f"  Config device: {cfg.DEVICE}")

    cuda_available = torch.cuda.is_available()
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    device = cfg.DEVICE
    if device == "cuda" and not cuda_available:
        print("  [WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    if device == "mps" and not mps_available:
        print("  [WARNING] MPS requested but not available. Falling back to CPU.")
        device = "cpu"
    if device == "cuda":
        print(f"  CUDA available: {torch.cuda.device_count()} device(s)")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("  MPS backend available. Using Apple GPU.")
    else:
        if cuda_available or mps_available:
            print("  A GPU backend is available, but using CPU because cfg.DEVICE is set to 'cpu'.")
        else:
            print("  No GPU backend is available. Using CPU.")
    print(f"  Device:     {device}")
    print()

    # Validate dataset YAML exists
    if not os.path.exists(cfg.DATASET_YAML):
        print(f"[ERROR] Dataset YAML not found: {cfg.DATASET_YAML}")
        print("Run dataset preparation first:")
        print("  python -m src.utils.prepare_dataset")
        return

    # Load the model
    # YOLOv8n pretrained on COCO — we fine-tune it on retail products.
    # This is transfer learning: the backbone already knows how to detect
    # objects; we just teach it what retail shelf objects look like.
    model = YOLO(cfg.MODEL_NAME)

    # Train
    results = model.train(
        data      = cfg.DATASET_YAML,
        epochs    = cfg.EPOCHS,
        batch     = cfg.BATCH_SIZE,
        imgsz     = cfg.IMG_SIZE,
        device    = device,
        workers   = cfg.WORKERS,
        project   = cfg.CHECKPOINTS_DIR,
        name      = "train",
        exist_ok  = True,       # overwrite previous run folder
        pretrained= True,       # use COCO weights as starting point
        patience  = 10,         # stop early if val loss doesn't improve
        save      = True,
        plots     = True,       # saves confusion matrix, loss curves etc.
        verbose   = True,

        # CPU-specific: disable mixed precision (only helps on GPU)
        amp       = False,

        # Data augmentation — helps a lot with small datasets
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 0.5,        # mosaic augmentation (4 images combined)
        degrees   = 5.0,        # small rotation
        translate = 0.1,
        scale     = 0.3,
    )

    # Copy best weights to our standard location
    best_src = os.path.join(cfg.CHECKPOINTS_DIR, "train", "weights", "best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, cfg.BEST_WEIGHTS)
        print(f"\nBest weights saved → {cfg.BEST_WEIGHTS}")
    else:
        print(f"\n[WARNING] best.pt not found at expected path: {best_src}")
        print("Check the 'runs/' directory for your weights.")

    # Print summary
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"Weights → {cfg.BEST_WEIGHTS}")
    print("=" * 60)
    print("\nNext step: test inference with:")
    print("  python -m src.detection.detector --image path/to/shelf.jpg")


if __name__ == "__main__":
    train()
