# Retail Shelf Intelligence

AI-powered retail shelf monitoring with continual learning, FastAPI inference, and a Streamlit dashboard.

---

## What it does

| Feature | How |
|---------|-----|
| Product detection | YOLOv8 fine-tuned on SKU-110K data |
| Product counting | Per-class and per-shelf-zone counts |
| Anomaly detection | Rule-based alerts plus optional ML anomaly model |
| Continual learning | Replay buffer + incremental fine-tuning |
| API | FastAPI REST endpoints |
| Dashboard | Streamlit + Plotly UI |
| Product identification | OCR-based brand/product name extraction |

---

## Project structure

```text
Retail_Shelf_intelligence/
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── future_upgrades.md
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── processed/
│   ├── raw/
│   └── replay_buffer/
├── models/
│   ├── checkpoints/
│   └── configs/
│       └── data_kaggle.yaml
├── runs/
├── src/
│   ├── analytics/
│   │   ├── heatmap.py
│   │   ├── ocr.py
│   │   ├── product_identifier.py
│   │   └── shelf_share.py
│   ├── anomaly/
│   │   ├── ml_detector.py
│   │   └── rules.py
│   ├── continual_learning/
│   │   ├── active_learning.py
│   │   ├── ewc.py
│   │   ├── replay_buffer.py
│   │   └── trainer.py
│   ├── detection/
│   │   ├── counter.py
│   │   ├── detector.py
│   │   └── train.py
│   └── utils/
│       ├── image_utils.py
│       └── prepare_dataset.py
├── tests/
│   ├── test_api.py
│   ├── test_detector.py
│   ├── test_identifier.py
│   ├── test_model_accuracy.py
│   └── evaluate.py
└── yolo26n.pt, yolov8n.pt, yolov8s.pt
```

> Note: `models/checkpoints/best.pt` is the checked-in production checkpoint. Training artifacts under `models/checkpoints/train/` are generated and are ignored by Git.

---

## Quickstart (without Docker)

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt)
# .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

The project can use either a raw SKU-110K CSV export or a Kaggle-style YOLO export.

#### Option A — Raw SKU-110K CSV

Download the SKU-110K dataset, then extract it so the folder structure is:

```text
data/raw/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── annotations_train.csv
    └── annotations_val.csv
```

Then run:

```bash
python -m src.utils.prepare_dataset
```

#### Option B — Kaggle-style YOLO export

If you already have a folder such as `SKU110K_fixed`, the preparation script will detect it automatically and write `models/configs/data_kaggle.yaml` for you.

### 3. Train the model

```bash
python -m src.detection.train
```

This uses the settings in `config.py`:
- `MODEL_NAME` (default: `yolov8s.pt`)
- `EPOCHS` (default: `30`)
- `BATCH_SIZE` (default: `8`)
- `IMG_SIZE` (default: `640`)
- `DEVICE` (auto-selects `cuda`, `mps`, or `cpu`)

### 4. Test inference on a single image

```bash
python -m src.detection.detector --image path/to/shelf.jpg --output output.jpg
```

The detector uses `models/checkpoints/best.pt` by default unless you pass `--weights`.

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the Swagger UI.

### 6. Start the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard expects the API to be running on http://localhost:8000.

### 7. Run the test suite

```bash
pytest tests/ -v
```

If you only want the detector checks:

```bash
pytest tests/test_detector.py -v
```

---

## Quickstart (with Docker)

```bash
docker compose up --build
```

Then open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

The compose file mounts `data/` and `models/` so trained artifacts persist across restarts.

---

## Configuration

`config.py` is the source of truth for all tunable settings.

| Setting | Current default | What it does |
|---------|-----------------|--------------|
| `SUBSET_SIZE` | `None` | Uses all available images unless you set a limit |
| `EPOCHS` | `30` | Training epochs |
| `BATCH_SIZE` | `8` | Batch size for training |
| `IMG_SIZE` | `640` | Input resolution for YOLO training and inference |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum confidence to keep a detection |
| `EMPTY_SHELF_MAX_PRODUCTS` | `2` | Triggers the empty-shelf anomaly |
| `LOW_STOCK_MAX_PRODUCTS` | `5` | Triggers the low-stock anomaly |
| `REPLAY_BUFFER_MAX_SIZE` | `200` | Maximum replay samples stored |
| `REPLAY_SAMPLE_SIZE` | `50` | Replay samples used in each continual-learning round |
| `CL_EPOCHS` | `10` | Fine-tuning epochs for incremental learning |
| `DEVICE` | auto-selected | Uses `cuda`, `mps`, or `cpu` based on the local runtime |

The dataset YAML path is `models/configs/data_kaggle.yaml`.

---

## GPU and accelerator support

This project supports these backends:

- `cuda` for NVIDIA GPUs
- `mps` for Apple Silicon
- `cpu` as the fallback

`config.py` auto-selects the available backend at runtime. To force a backend, set `DEVICE` explicitly in `config.py`.

---

## Continual learning workflow

```bash
# 1. Train the base model
python -m src.detection.train

# 2. Seed the replay buffer with phase-1 samples
python -c "
from src.continual_learning.trainer import IncrementalTrainer
trainer = IncrementalTrainer()
trainer.seed_buffer_from_phase(phase=1)
"

# 3. Add new phase data in YOLO format
# Put the images and labels in:
#   data/phase2/images/
#   data/phase2/labels/

# 4. Fine-tune on the new phase while replaying old samples
python -c "
from src.continual_learning.trainer import IncrementalTrainer
trainer = IncrementalTrainer()
trainer.train_new_phase(
    new_images_dir='data/phase2/images',
    new_labels_dir='data/phase2/labels',
    class_names=['product'],
    phase=2,
)
"

# 5. Compare before / after forgetting metrics
python tests/evaluate.py \
    --before models/checkpoints/phase1_best.pt \
    --after models/checkpoints/best.pt
```

---

## Common errors

**`ModuleNotFoundError: No module named 'ultralytics'`**
Run `pip install -r requirements.txt`.

**`Dataset YAML not found`**
Run `python -m src.utils.prepare_dataset` first. The script also detects a Kaggle-style export automatically.

**`No images found in train set`**
Check that the raw files live under `data/raw/images/` and that the annotation filenames match the image filenames.

**`OOM / killed during training`**
Lower `BATCH_SIZE` or `IMG_SIZE` in `config.py`.

**`API connection refused` in the dashboard**
Start the API first with `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`.
The dashboard does not run standalone in this repo; it expects the API to be available.

**`Training is too slow on CPU`**
Lower `EPOCHS` and `BATCH_SIZE` for a fast smoke test. The current defaults are tuned for stronger accuracy.

---

## Evaluation metrics

- **mAP50** — main metric for detection quality.
- **mAP50-95** — stricter IoU-averaged metric.
- **Forgetting score** — degradation on older classes after incremental fine-tuning.

A good target for the replay-buffer workflow is to keep forgetting small while maintaining good accuracy on new data.

---

## Tech stack

| Layer | Library |
|-------|---------|
| Detection | YOLOv8 (Ultralytics) |
| Training | PyTorch |
| Image processing | OpenCV, Pillow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| OCR / identification | EasyOCR |
| Containerisation | Docker + Compose |
| Tests | pytest |
