# Retail Shelf Intelligence — Future Upgrades

## 🔬 Model & Detection

| # | Upgrade | Why | Effort |
|---|---------|-----|--------|
| 1 | **GPU training** — Train on full SKU-110K (11K+ images) with CUDA | Current 500-image subset limits accuracy; full dataset would significantly boost mAP | Low |
| 2 | **Upgrade to YOLOv8s/m** — Use a larger model variant | YOLOv8n is optimized for speed, not accuracy; medium model yields better shelf detections | Low |
| 3 | **Multi-class detection** — Train per-product-category labels (beverages, snacks, toiletries) | Currently all products are a single "product" class; per-category classification enables richer analytics | Medium |
| 4 | **Instance segmentation** — Switch to YOLOv8-seg for pixel-level masks | Enables precise shelf occupancy measurement (% of shelf space used) instead of bounding boxes | Medium |
| 5 | **OCR integration** — Add Tesseract/PaddleOCR for price tag & label reading | Extract product names and prices directly from shelf images | Medium |
| 6 | **Planogram compliance** — Compare detected layout against expected planogram | Flag products placed in wrong positions by comparing to a reference layout | Hard |

## 📊 Analytics & Anomaly Detection

| # | Upgrade | Why | Effort |
|---|---------|-----|--------|
| 7 | **Time-series tracking** — Store detection results with timestamps in a database | Track stock levels over time, identify restocking patterns, predict out-of-stock events | Medium |
| 8 | **ML-based anomaly detection** — Replace rule-based logic with isolation forest or autoencoder | Learns normal shelf patterns and detects unusual states without manual thresholds | Medium |
| 9 | **Heatmap visualization** — Overlay product density heatmaps on shelf images | Visual "hot zones" showing where products cluster and where gaps exist | Low |
| 10 | **Shelf share analysis** — Calculate percentage of shelf space per brand/category | Key retail KPI — show which brands dominate shelf space | Medium |
| 11 | **Restocking alerts** — Predict when stock will run out based on depletion trends | Proactive notifications before shelves go empty | Hard |

## 🖥️ Dashboard & UX

| # | Upgrade | Why | Effort |
|---|---------|-----|--------|
| 12 | **Live camera feed** — Accept RTSP/webcam streams instead of static uploads | Real-time monitoring of shelf conditions with periodic auto-capture | Medium |
| 13 | **Multi-image batch upload** — Process multiple images at once with comparison view | Analyze an entire store in one go | Low |
| 14 | **Before/after comparison** — Side-by-side view of same shelf at different times | Visually track restocking effectiveness | Low |
| 15 | **Export reports** — PDF/CSV export of detection results and anomaly reports | Shareable reports for store managers | Low |
| 16 | **User authentication** — Login system with role-based access (admin, viewer) | Required for production deployment with multiple users | Medium |

## 🔧 Infrastructure & DevOps

| # | Upgrade | Why | Effort |
|---|---------|-----|--------|
| 17 | **Database integration** — PostgreSQL/SQLite for storing detection history | Currently no persistence; all results are ephemeral | Medium |
| 18 | **Redis caching** — Cache model predictions for recently analyzed images | Avoid re-running inference on duplicate uploads | Low |
| 19 | **Async inference queue** — Celery/RQ for background processing | Current API blocks during inference; queue enables handling multiple requests | Medium |
| 20 | **Cloud deployment** — Deploy to AWS/GCP with GPU instances | Enable fast inference and scale to multiple stores | Medium |
| 21 | **CI/CD pipeline** — GitHub Actions for automated testing + Docker builds | Currently manual testing; automate quality gates on every push | Low |
| 22 | **Model versioning** — MLflow or DVC for tracking model experiments | Track hyperparameters, metrics, and weights across training runs | Medium |
| 23 | **API rate limiting & monitoring** — Add rate limits, Prometheus metrics, logging | Production hardening for real-world use | Low |

## 🧠 Continual Learning Enhancements

| # | Upgrade | Why | Effort |
|---|---------|-----|--------|
| 24 | **Active learning** — Auto-select the most uncertain images for human labeling | Reduce annotation effort by focusing on samples where the model struggles | Hard |
| 25 | **Elastic weight consolidation (EWC)** — Advanced anti-forgetting technique | Better than experience replay alone at preventing catastrophic forgetting | Hard |
| 26 | **Federated learning** — Train across multiple store locations without sharing data | Privacy-preserving learning from distributed retail locations | Hard |
| 27 | **Auto-labeling pipeline** — Use high-confidence predictions as pseudo-labels for retraining | Bootstrap new training data from the model's own predictions | Medium |

---

## Recommended Priority Order

> [!TIP]
> Start with **high-impact, low-effort** upgrades first.

### Phase 1 — Quick Wins
- \#9 Heatmap visualization
- \#13 Multi-image batch upload
- \#15 Export reports
- \#21 CI/CD pipeline

### Phase 2 — Core Improvements
- \#1 GPU training on full dataset
- \#7 Time-series tracking + database
- \#12 Live camera feed
- \#17 Database integration

### Phase 3 — Advanced Features
- \#3 Multi-class detection
- \#6 Planogram compliance
- \#24 Active learning
- \#5 OCR integration
