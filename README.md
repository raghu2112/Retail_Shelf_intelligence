# Retail Shelf Intelligence

AI-powered retail shelf monitoring with continual learning.

---

## What it does

| Feature | How |
|---------|-----|
| Product detection | YOLOv8n fine-tuned on SKU-110K |
| Product counting | Per-class and per-shelf-zone counts |
| Anomaly detection | Rule-based (empty shelf, low stock, misplaced) |
| Continual learning | Experience replay buffer + incremental fine-tuning |
| API | FastAPI REST endpoints |
| Dashboard | Streamlit with Plotly charts |

---

## Quickstart (without Docker)

### 1. Install dependencies
```bash
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows:
# .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download SKU-110K
Download from: https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd

Extract so your structure looks like:
```
data/raw/
    images/
        train/   ← .jpg files
        val/     ← .jpg files
    annotations/
        annotations_train.csv
        annotations_val.csv
```

### 3. Prepare dataset
```bash
python -m src.utils.prepare_dataset
```
This converts SKU-110K CSV annotations into YOLOv8 format.
The number of images is controlled by `config.py` → `SUBSET_SIZE`.

### 4. Train
```bash
python -m src.detection.train
```
The project auto-selects the best available device backend in `config.py`: `cuda`, `mps`, or `cpu`.
On CPU, training can be slow; lower `EPOCHS`, `SUBSET_SIZE`, or `BATCH_SIZE` for a quick test.

### 5. Test detection
```bash
python -m src.detection.detector --image path/to/shelf.jpg --output output.jpg
```

### 6. Run tests
```bash
pytest tests/test_detector.py -v
```

### 7. Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/docs for Swagger UI.

### 8. Start dashboard
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501

---

## Quickstart (with Docker)

```bash
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs:  http://localhost:8000/docs

---

## GPU and accelerator support

This project supports multiple device backends:
- `cuda` for NVIDIA GPUs
- `mps` for Apple silicon
- `cpu` as a fallback when no GPU backend is available

The backend is auto-selected in `config.py` using the available PyTorch runtime.
If you want to force a specific device, update `DEVICE` in `config.py`.

---

## Continual learning workflow

```bash
# Step 1: Train initial model (Phase 1)
python -m src.detection.train

# Step 2: Seed replay buffer with Phase 1 samples
python -c "
from src.continual_learning.trainer import IncrementalTrainer
t = IncrementalTrainer()
t.seed_buffer_from_phase(phase=1)
"

# Step 3: When new product images arrive (Phase 2)
# Put them in:  data/phase2/images/  and  data/phase2/labels/
# Then run:
python -c "
from src.continual_learning.trainer import IncrementalTrainer
t = IncrementalTrainer()
t.train_new_phase(
    new_images_dir='data/phase2/images',
    new_labels_dir='data/phase2/labels',
    class_names=['product'],
    phase=2,
)
"

# Step 4: Measure forgetting
python tests/evaluate.py \
    --before models/checkpoints/phase1_best.pt \
    --after  models/checkpoints/best.pt
```

---

## Project structure

```
retail-shelf-intelligence/
├── config.py                          ← all settings in one place
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/                           ← SKU-110K download goes here
│   ├── processed/                     ← YOLOv8-format dataset
│   └── replay_buffer/                 ← old training samples
│
├── models/
│   ├── checkpoints/best.pt            ← trained weights
│   └── configs/sku110k.yaml           ← dataset YAML
│
├── src/
│   ├── detection/
│   │   ├── detector.py                ← YOLOv8 inference wrapper
│   │   ├── counter.py                 ← product counting + zone stats
│   │   └── train.py                   ← training script
│   ├── anomaly/
│   │   └── rules.py                   ← rule-based anomaly detection
│   ├── continual_learning/
│   │   ├── replay_buffer.py           ← experience replay storage
│   │   └── trainer.py                 ← incremental fine-tuning
│   └── utils/
│       ├── image_utils.py             ← shared image helpers
│       └── prepare_dataset.py         ← SKU-110K → YOLOv8 converter
│
├── api/
│   └── main.py                        ← FastAPI app
├── dashboard/
│   └── app.py                         ← Streamlit dashboard
└── tests/
    ├── test_detector.py               ← pytest test suite
    └── evaluate.py                    ← mAP + forgetting metrics
```

---

## Configuration

All tunable parameters live in `config.py`. The important ones:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| SUBSET_SIZE | 100 | How many images to use from SKU-110K |
| EPOCHS | 10 | Training epochs (lower for faster testing) |
| BATCH_SIZE | 4 | Lower to 1 or 2 if you get out-of-memory errors on small GPUs |
| CONFIDENCE_THRESHOLD | 0.35 | Minimum detection confidence |
| EMPTY_SHELF_MAX_PRODUCTS | 2 | Zone product count to trigger "empty" alert |
| LOW_STOCK_MAX_PRODUCTS | 5 | Zone product count to trigger "low stock" alert |
| REPLAY_BUFFER_MAX_SIZE | 200 | Max samples stored in replay buffer |
| REPLAY_SAMPLE_SIZE | 50 | Old samples mixed in per fine-tuning round |
| CL_EPOCHS | 10 | Fine-tuning epochs per continual learning phase |

---

## Common errors

**`ModuleNotFoundError: No module named 'ultralytics'`**
Run `pip install -r requirements.txt` inside your virtualenv.

**`Dataset YAML not found`**
Run `python -m src.utils.prepare_dataset` first.

**`No images found in train set`**
Check that SKU-110K is extracted to `data/raw/` and that your CSV filenames match the image filenames exactly.

**`OOM / killed during training`**
Lower `BATCH_SIZE` to 4 or `IMG_SIZE` to 416 in config.py.

**`API connection refused` in dashboard**
Start the API first: `uvicorn api.main:app --port 8000`
Or run the dashboard in standalone mode: `streamlit run dashboard/app.py -- --standalone`

**Training is too slow on CPU**
Set `EPOCHS = 5`, `SUBSET_SIZE = 100`, and `BATCH_SIZE = 4` for a quick demo. The model won't be accurate but the pipeline will work end-to-end.

---

## Evaluation metrics explained

- **mAP50**: mean Average Precision at IoU=0.50. Main metric. Target: >0.5 on SKU-110K subset.
- **mAP50-95**: stricter version averaged across IoU thresholds. Lower, normal.
- **Forgetting score**: mAP drop on old classes after fine-tuning new ones. Target: <0.02 with replay buffer active.

---

## Tech stack

| Layer | Library |
|-------|---------|
| Detection | YOLOv8n (Ultralytics) |
| Training | PyTorch |
| Image processing | OpenCV, Pillow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Compose |
| Tests | pytest |
