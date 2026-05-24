"""Quick test: run the improved ProductIdentifier on a sample shelf image."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from src.detection.detector import ShelfDetector
from src.analytics.product_identifier import ProductIdentifier
import config as cfg

# Pick a sample image from the validation set
VAL_DIR = os.path.join(cfg.DATA_DIR, "processed", "images", "val")
images = [f for f in os.listdir(VAL_DIR) if f.endswith(".jpg")][:3]

if not images:
    print("No validation images found.")
    sys.exit(1)

# Load detector
print("Loading detector...")
detector = ShelfDetector(model_path=cfg.BEST_WEIGHTS, device=cfg.DEVICE)

# Load identifier
print("Loading product identifier...")
identifier = ProductIdentifier(max_products=5)

for img_name in images:
    img_path = os.path.join(VAL_DIR, img_name)
    print(f"\n{'='*60}")
    print(f"Image: {img_name}")
    print(f"{'='*60}")

    pil_img = Image.open(img_path).convert("RGB")
    img_array = np.array(pil_img)

    # Step 1: Detect
    result = detector.detect(img_array)
    print(f"  Detected: {result.count} products")

    # Build detection dicts (as the API does)
    det_list = [
        {
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            "confidence": round(d.confidence, 4),
            "class_id": d.class_id,
            "class_name": d.class_name,
        } for d in result.detections
    ]

    # Step 2: Identify
    t0 = time.time()
    inventory = identifier.identify(img_array, det_list)
    elapsed = time.time() - t0

    print(f"  Identification time: {elapsed:.1f}s")
    print(f"  Identified: {inventory.total_identified} / {inventory.total_identified + inventory.total_unidentified}")
    print(f"  Unique products: {inventory.unique_products}")
    print(f"  Counts:")
    for name, count in sorted(inventory.counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {count}")
