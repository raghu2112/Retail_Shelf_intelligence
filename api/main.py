# api/main.py
#
# PURPOSE:
#   REST API that wraps the detector, counter, and anomaly detector.
#   The Streamlit dashboard calls these endpoints.
#
# ENDPOINTS:
#   POST /detect              — run detection on an uploaded image
#   POST /detect/batch        — batch detection on multiple images
#   GET  /health              — sanity check
#   GET  /buffer/stats        — replay buffer statistics
#   POST /buffer/seed         — seed buffer from processed dataset
#   POST /continual/train     — trigger incremental fine-tuning
#   GET  /analytics/heatmap   — heatmap data for last detection
#   GET  /analytics/shelf-share — shelf share for last detection
#   POST /analytics/ocr       — extract text from shelf image
#   POST /anomaly/ml/train    — train ML anomaly model
#   POST /active-learning/query — query uncertain samples
#
# USAGE:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

import io
import os
import sys
import time
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from src.detection.detector import ShelfDetector
from src.detection.counter import ProductCounter
from src.anomaly.rules import AnomalyDetector
from src.continual_learning.replay_buffer import ReplayBuffer
from src.continual_learning.trainer import IncrementalTrainer


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title        = cfg.DASHBOARD_TITLE,
    description  = "Retail shelf monitoring API with continual learning",
    version      = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Lazy-initialise heavy components once ─────────────────────────────────────

_state = {}
_detection_history: deque = deque(maxlen=500)  # store recent results for ML training
_job_queue: Dict[str, dict] = {}               # async job tracking
_executor = ThreadPoolExecutor(max_workers=cfg.ASYNC_WORKERS)

def get_detector() -> ShelfDetector:
    if "detector" not in _state:
        weights = cfg.BEST_WEIGHTS if os.path.exists(cfg.BEST_WEIGHTS) else cfg.MODEL_NAME
        _state["detector"] = ShelfDetector(model_path=weights, device=cfg.DEVICE)
    return _state["detector"]

def get_counter() -> ProductCounter:
    if "counter" not in _state:
        _state["counter"] = ProductCounter()
    return _state["counter"]

def get_anomaly_detector() -> AnomalyDetector:
    if "anomaly" not in _state:
        _state["anomaly"] = AnomalyDetector()
    return _state["anomaly"]

def get_buffer() -> ReplayBuffer:
    if "buffer" not in _state:
        _state["buffer"] = ReplayBuffer()
    return _state["buffer"]

def get_ml_anomaly():
    if "ml_anomaly" not in _state:
        from src.anomaly.ml_detector import MLAnomalyDetector
        _state["ml_anomaly"] = MLAnomalyDetector()
    return _state["ml_anomaly"]

def get_product_identifier():
    if "identifier" not in _state:
        from src.analytics.product_identifier import ProductIdentifier
        _state["identifier"] = ProductIdentifier()
    return _state["identifier"]


# ── Response models ───────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    product_name: Optional[str] = None

class ZoneStats(BaseModel):
    zone_id: int
    count: int

class AnomalyOut(BaseModel):
    type: str
    severity: str
    description: str
    zone_id: int

class DetectionResponse(BaseModel):
    image_width:    int
    image_height:   int
    total_products: int
    counts_by_class: dict
    avg_confidence:  float
    zones:          List[ZoneStats]
    anomalies:      List[AnomalyOut]
    detections:     List[BoundingBox]
    processing_time_ms: float
    product_inventory: Optional[dict] = None

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ── Core detection helper ─────────────────────────────────────────────────────

def _run_detection(img_array: np.ndarray, filename: str = "image.jpg") -> dict:
    """Run full detection pipeline and return response dict."""
    t0 = time.time()

    detector = get_detector()
    result = detector.detect(img_array)
    result.image_path = filename

    counter = get_counter()
    stats = counter.count(result)

    anomaly_detector = get_anomaly_detector()
    anomalies = anomaly_detector.detect(stats)

    # ML anomaly detection (if trained)
    ml_anomalies = []
    ml_detector = get_ml_anomaly()
    if ml_detector.is_trained:
        ml_result = ml_detector.predict({
            "total_products": stats.total_products,
            "avg_confidence": stats.avg_confidence,
            "zones": [{"zone_id": z.zone_id, "count": z.count} for z in stats.zones],
            "image_width": stats.image_width,
            "image_height": stats.image_height,
        })
        if ml_result:
            ml_anomalies.append(ml_result.to_dict())

    elapsed_ms = (time.time() - t0) * 1000

    det_list = [
        {
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            "confidence": round(d.confidence, 4),
            "class_id": d.class_id,
            "class_name": d.class_name,
        } for d in result.detections
    ]

    # ── Product identification via OCR ────────────────────────────────────
    identifier = get_product_identifier()
    inventory = identifier.identify(img_array, det_list)
    inv_dict = inventory.to_dict()

    # Merge OCR-identified names back onto detections
    # Build a map from crop_index -> product name
    name_map = {p.crop_index: p.name for p in inventory.products}
    for i, det in enumerate(det_list):
        det["product_name"] = name_map.get(i, "Unknown")

    response_data = {
        "image_width": stats.image_width,
        "image_height": stats.image_height,
        "total_products": stats.total_products,
        "counts_by_class": stats.counts_by_class,
        "avg_confidence": round(stats.avg_confidence, 4),
        "zones": [{"zone_id": z.zone_id, "count": z.count} for z in stats.zones],
        "anomalies": [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity,
                "description": a.description,
                "zone_id": a.zone_id,
            } for a in anomalies
        ] + ml_anomalies,
        "detections": det_list,
        "processing_time_ms": round(elapsed_ms, 2),
        # Product inventory from OCR
        "product_inventory": {
            "counts_by_name": inv_dict["counts"],
            "unique_products": inv_dict["unique_products"],
            "total_identified": inv_dict["total_identified"],
            "total_unidentified": inv_dict["total_unidentified"],
        },
    }

    # Store for ML anomaly training
    _detection_history.append(response_data)

    return response_data


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick sanity check."""
    return {
        "status":  "ok",
        "model":   cfg.MODEL_NAME,
        "weights": cfg.BEST_WEIGHTS if os.path.exists(cfg.BEST_WEIGHTS) else "pretrained",
        "device":  cfg.DEVICE,
        "detection_mode": cfg.DETECTION_MODE,
        "history_size": len(_detection_history),
        "ml_anomaly_trained": get_ml_anomaly().is_trained,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """Upload a shelf image, get back detections + anomaly alerts."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    return _run_detection(img_array, file.filename)


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Batch detection on multiple images.
    Returns a list of detection results, one per image.
    """
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({"error": f"{file.filename}: not an image"})
            continue

        raw = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(pil_img)

        data = _run_detection(img_array, file.filename)
        data["filename"] = file.filename
        results.append(data)

    return {"results": results, "total_images": len(results)}


@app.post("/detect/async", response_model=AsyncJobResponse)
async def detect_async(file: UploadFile = File(...)):
    """
    Submit an image for async processing.
    Returns a job_id; poll /detect/async/{job_id} for results.
    """
    if len(_job_queue) >= cfg.MAX_QUEUE_SIZE:
        raise HTTPException(status_code=429, detail="Queue full. Try again later.")

    raw = await file.read()
    job_id = str(uuid.uuid4())[:8]

    _job_queue[job_id] = {"status": "processing", "result": None, "filename": file.filename}

    def _process():
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(pil_img)
        result = _run_detection(img_array, file.filename)
        _job_queue[job_id] = {"status": "done", "result": result, "filename": file.filename}

    _executor.submit(_process)

    return AsyncJobResponse(
        job_id=job_id,
        status="processing",
        message="Job submitted. Poll /detect/async/{job_id} for results.",
    )


@app.get("/detect/async/{job_id}")
def get_async_result(job_id: str):
    """Get the result of an async detection job."""
    if job_id not in _job_queue:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_queue[job_id]


# ── Analytics endpoints ───────────────────────────────────────────────────────

@app.post("/analytics/heatmap")
async def heatmap(file: UploadFile = File(...)):
    """Generate a product density heatmap for an uploaded image."""
    from src.analytics.heatmap import generate_heatmap

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename)

    heat = generate_heatmap(
        data["detections"],
        data["image_width"],
        data["image_height"],
    )

    return {
        "heatmap": heat.tolist(),
        "image_width": data["image_width"],
        "image_height": data["image_height"],
        "total_products": data["total_products"],
    }


@app.post("/analytics/shelf-share")
async def shelf_share(file: UploadFile = File(...)):
    """Calculate shelf share analysis for an uploaded image."""
    from src.analytics.shelf_share import calculate_shelf_share

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename)

    share = calculate_shelf_share(
        data["detections"],
        data["image_width"],
        data["image_height"],
    )

    return {
        "occupancy_rate": round(share.occupancy_rate, 4),
        "occupied_area": share.occupied_area,
        "empty_area": share.empty_area,
        "share_by_class": share.share_by_class,
        "count_by_class": share.count_by_class,
    }


@app.post("/analytics/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text (prices, labels) from a shelf image using OCR."""
    from src.analytics.ocr import ShelfOCR

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename)

    ocr = ShelfOCR()
    ocr_results = ocr.read_with_products(img_array, data["detections"])
    prices = ocr.extract_prices(ocr_results)

    return {
        "texts": [r.to_dict() for r in ocr_results],
        "prices": prices,
        "total_texts": len(ocr_results),
        "total_prices": len(prices),
    }


@app.post("/analytics/identify-products")
async def identify_products(file: UploadFile = File(...)):
    """
    Identify products by name using OCR on each detected product region.
    Returns a grouped inventory with counts per unique product.
    """
    from src.analytics.product_identifier import ProductIdentifier

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename)

    identifier = ProductIdentifier()
    inventory = identifier.identify(img_array, data["detections"])

    return {
        "inventory": inventory.to_dict(),
        "total_detections": data["total_products"],
    }


# ── ML Anomaly endpoints ─────────────────────────────────────────────────────

@app.post("/anomaly/ml/train")
def train_ml_anomaly():
    """Train ML anomaly model from detection history."""
    if len(_detection_history) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 observations, have {len(_detection_history)}. "
                   f"Run more /detect calls first.",
        )

    ml_detector = get_ml_anomaly()
    ml_detector.fit(list(_detection_history))
    return {
        "status": "trained",
        "observations": len(_detection_history),
        "message": "ML anomaly model trained. Future /detect calls will include ML anomaly scores.",
    }


# ── Active learning endpoint ─────────────────────────────────────────────────

class ActiveLearningRequest(BaseModel):
    images_dir: str
    method: Optional[str] = None
    query_size: Optional[int] = None

@app.post("/active-learning/query")
def active_learning_query(req: ActiveLearningRequest):
    """Find the most uncertain images for labeling."""
    from src.continual_learning.active_learning import ActiveLearner

    if not os.path.exists(req.images_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.images_dir}")

    learner = ActiveLearner(
        method=req.method or cfg.AL_UNCERTAINTY_METHOD,
        query_size=req.query_size or cfg.AL_QUERY_SIZE,
    )
    detector = get_detector()
    samples = learner.query_from_directory(detector, req.images_dir)

    return {
        "samples": [s.to_dict() for s in samples],
        "total_queried": len(samples),
        "method": learner.method,
    }


# ── Buffer & training endpoints ───────────────────────────────────────────────

@app.get("/buffer/stats")
def buffer_stats():
    """Return replay buffer statistics."""
    return get_buffer().stats()


@app.post("/buffer/seed")
def seed_buffer(phase: int = 1, max_samples: int = 100):
    """Seed the replay buffer from the processed training dataset."""
    trainer = IncrementalTrainer(buffer=get_buffer())
    trainer.seed_buffer_from_phase(phase=phase, max_samples=max_samples)
    return {"status": "ok", "buffer_size": get_buffer().size}


class ContinualTrainRequest(BaseModel):
    new_images_dir: str
    new_labels_dir: str
    class_names:    List[str]
    phase:          int = 2
    epochs:         Optional[int] = None
    use_ewc:        bool = False


@app.post("/continual/train")
def continual_train(req: ContinualTrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger incremental fine-tuning in the background.
    Optionally use EWC to prevent catastrophic forgetting.
    """
    if not os.path.exists(req.new_images_dir):
        raise HTTPException(status_code=400, detail=f"Images dir not found: {req.new_images_dir}")
    if not os.path.exists(req.new_labels_dir):
        raise HTTPException(status_code=400, detail=f"Labels dir not found: {req.new_labels_dir}")

    trainer = IncrementalTrainer(buffer=get_buffer())

    def run_training():
        trainer.train_new_phase(
            new_images_dir = req.new_images_dir,
            new_labels_dir = req.new_labels_dir,
            class_names    = req.class_names,
            phase          = req.phase,
            epochs         = req.epochs,
        )
        # Reload detector with updated weights
        _state["detector"] = ShelfDetector(
            model_path=cfg.BEST_WEIGHTS, device=cfg.DEVICE
        )
        print("[API] Detector reloaded with updated weights.")

    background_tasks.add_task(run_training)

    return {
        "status":  "training_started",
        "phase":   req.phase,
        "use_ewc": req.use_ewc,
        "message": "Fine-tuning running in background.",
    }
