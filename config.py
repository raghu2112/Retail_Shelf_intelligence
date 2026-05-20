# config.py — central configuration for the entire project
# Change paths here if your folder layout differs

import importlib.util
import os

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR          = os.path.join(ROOT_DIR, "data")
RAW_DIR           = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR     = os.path.join(DATA_DIR, "processed")
REPLAY_BUFFER_DIR = os.path.join(DATA_DIR, "replay_buffer")

MODELS_DIR      = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
CONFIGS_DIR     = os.path.join(MODELS_DIR, "configs")

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_YAML = os.path.join(CONFIGS_DIR, "sku110k.yaml")

# How many images to use for quick experiments (set None to use all)
SUBSET_SIZE = 100

# Train / val split ratio
TRAIN_RATIO = 0.80


def _preferred_device() -> str:
    """Detect the preferred device for PyTorch training/inference."""
    try:
        spec = importlib.util.find_spec("torch")
        if spec is None:
            return "cpu"
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


# ── Model ────────────────────────────────────────────────────────────────────
# Model variants: "yolov8n.pt" (nano), "yolov8s.pt" (small), "yolov8m.pt" (medium)
# Segmentation:   "yolov8s-seg.pt", "yolov8m-seg.pt"
MODEL_NAME    = "yolov8s.pt"     # upgraded from nano to small for better accuracy
BEST_WEIGHTS  = os.path.join(CHECKPOINTS_DIR, "best.pt")

# Detection mode: "detect" or "segment"
DETECTION_MODE = "detect"

# Multi-class support — list all classes the model should detect
# For SKU-110K: single "product" class. Add more for multi-class training.
CLASS_NAMES = ["product"]

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 4            # safe starting batch size for lower-memory GPUs
IMG_SIZE    = 640
# Auto-select the available device backend.
# Supports CUDA for NVIDIA, MPS for Apple, or CPU when no GPU backend is available.
# Adjust batch size down if your GPU has limited VRAM.
DEVICE      = _preferred_device()
WORKERS     = 4            # more workers speeds up data loading on multi-core systems

# ── Inference ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD        = 0.45

# ── Anomaly detection thresholds ─────────────────────────────────────────────
EMPTY_SHELF_MAX_PRODUCTS    = 2
LOW_STOCK_MAX_PRODUCTS      = 5
MISPLACED_IOU_THRESHOLD     = 0.1

# ML anomaly detection
ANOMALY_CONTAMINATION = 0.1     # expected proportion of anomalous samples
ANOMALY_FEATURES = ["total_products", "avg_confidence", "detection_density",
                    "zone_variance", "max_zone_gap"]

# ── OCR ──────────────────────────────────────────────────────────────────────
OCR_ENABLED    = True
OCR_LANGUAGES  = ["en"]
OCR_CONFIDENCE = 0.4            # minimum OCR confidence to keep a text detection

# ── Heatmap ──────────────────────────────────────────────────────────────────
HEATMAP_RADIUS    = 40          # Gaussian blur radius for heatmap
HEATMAP_INTENSITY = 0.6         # overlay opacity (0=transparent, 1=opaque)

# ── Live camera ──────────────────────────────────────────────────────────────
CAMERA_INTERVAL_SEC = 5         # seconds between auto-captures from live feed

# ── Continual learning ────────────────────────────────────────────────────────
REPLAY_BUFFER_MAX_SIZE = 200
REPLAY_SAMPLE_SIZE     = 50
CL_EPOCHS              = 10

# EWC (Elastic Weight Consolidation)
EWC_LAMBDA    = 5000            # importance weight for EWC penalty
EWC_N_SAMPLES = 100             # samples used to estimate Fisher information

# Active learning
AL_UNCERTAINTY_METHOD = "entropy"  # "entropy", "margin", "least_confident"
AL_POOL_SIZE          = 200        # unlabeled pool size to evaluate
AL_QUERY_SIZE         = 20         # how many images to select per round

# ── Async inference ──────────────────────────────────────────────────────────
MAX_QUEUE_SIZE    = 50          # max pending inference jobs
ASYNC_WORKERS     = 2           # number of background inference workers

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "Retail Shelf Intelligence"
