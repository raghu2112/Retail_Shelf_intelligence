# tests/evaluate.py
#
# PURPOSE:
#   Evaluate detection performance and measure catastrophic forgetting.
#   Run after training and after each continual learning phase.
#
#   Metrics computed:
#     - mAP50, mAP50-95 (standard object detection metrics)
#     - Per-class precision and recall
#     - Forgetting score: accuracy drop on old classes after new phase
#
# USAGE:
#   # Evaluate current model on val set
#   python tests/evaluate.py
#
#   # Evaluate a specific weights file
#   python tests/evaluate.py --weights models/checkpoints/best.pt
#
#   # Compare two checkpoints (measures forgetting)
#   python tests/evaluate.py --before models/checkpoints/phase1_best.pt \
#                             --after  models/checkpoints/phase2_best.pt

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


def evaluate_model(weights_path: str, dataset_yaml: str = None) -> dict:
    """
    Run YOLOv8 validation and return metrics dict.

    Args:
        weights_path: .pt file to evaluate
        dataset_yaml: dataset YAML (defaults to cfg.DATASET_YAML)

    Returns:
        dict with mAP50, mAP50-95, precision, recall
    """
    from ultralytics import YOLO

    dataset_yaml = dataset_yaml or cfg.DATASET_YAML

    if not os.path.exists(weights_path):
        print(f"[Eval] Weights not found: {weights_path}")
        return {}

    if not os.path.exists(dataset_yaml):
        print(f"[Eval] Dataset YAML not found: {dataset_yaml}")
        print("Run prepare_dataset.py first.")
        return {}

    print(f"\n[Eval] Evaluating: {weights_path}")
    print(f"[Eval] Dataset:    {dataset_yaml}")

    model = YOLO(weights_path)
    results = model.val(
        data    = dataset_yaml,
        imgsz   = cfg.IMG_SIZE,
        batch   = cfg.BATCH_SIZE,
        device  = cfg.DEVICE,
        verbose = False,
    )

    metrics = {
        "weights":     weights_path,
        "mAP50":       float(results.box.map50),
        "mAP50_95":    float(results.box.map),
        "precision":   float(results.box.mp),
        "recall":      float(results.box.mr),
    }

    # Per-class metrics if available
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        class_metrics = {}
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = model.names.get(int(cls_idx), str(cls_idx))
            class_metrics[cls_name] = {
                "AP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            }
        metrics["per_class"] = class_metrics

    return metrics


def compute_forgetting_score(before_metrics: dict, after_metrics: dict) -> dict:
    """
    Measure how much the model forgot after continual learning.

    Forgetting = (mAP before fine-tuning) - (mAP after fine-tuning)
    on the SAME validation set (original classes).

    A positive forgetting score means the model got worse on old classes.
    Ideally this should be close to 0 with replay buffer.

    Args:
        before_metrics: metrics dict from evaluate_model() before CL
        after_metrics:  metrics dict from evaluate_model() after CL

    Returns:
        dict with forgetting scores per metric
    """
    if not before_metrics or not after_metrics:
        return {}

    forgetting = {
        "mAP50_forgetting":    before_metrics["mAP50"]    - after_metrics["mAP50"],
        "mAP50_95_forgetting": before_metrics["mAP50_95"] - after_metrics["mAP50_95"],
        "precision_change":    after_metrics["precision"]  - before_metrics["precision"],
        "recall_change":       after_metrics["recall"]     - before_metrics["recall"],
    }

    # Per-class forgetting
    before_cls = before_metrics.get("per_class", {})
    after_cls  = after_metrics.get("per_class", {})
    per_class_forgetting = {}
    for cls in before_cls:
        if cls in after_cls:
            per_class_forgetting[cls] = (
                before_cls[cls]["AP50"] - after_cls[cls]["AP50"]
            )
    if per_class_forgetting:
        forgetting["per_class"] = per_class_forgetting

    return forgetting


def print_metrics(metrics: dict, label: str = ""):
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  mAP50:       {metrics.get('mAP50', 0):.4f}")
    print(f"  mAP50-95:    {metrics.get('mAP50_95', 0):.4f}")
    print(f"  Precision:   {metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {metrics.get('recall', 0):.4f}")
    if "per_class" in metrics:
        print("\n  Per-class AP50:")
        for cls, m in metrics["per_class"].items():
            print(f"    {cls}: {m['AP50']:.4f}")
    print(f"{'─'*50}")


def print_forgetting(forgetting: dict):
    print(f"\n{'─'*50}")
    print("  Forgetting Analysis")
    print(f"{'─'*50}")
    mAP_forget = forgetting.get("mAP50_forgetting", 0)
    arrow = "↓" if mAP_forget > 0 else "↑" if mAP_forget < 0 else "→"
    print(f"  mAP50 change:      {arrow} {abs(mAP_forget):.4f}")
    print(f"  mAP50-95 change:   {forgetting.get('mAP50_95_forgetting', 0):+.4f}")
    print(f"  Precision change:  {forgetting.get('precision_change', 0):+.4f}")
    print(f"  Recall change:     {forgetting.get('recall_change', 0):+.4f}")

    if "per_class" in forgetting:
        print("\n  Per-class forgetting (positive = forgot, negative = improved):")
        for cls, delta in forgetting["per_class"].items():
            flag = "⚠️ " if delta > 0.05 else "✅ "
            print(f"    {flag}{cls}: {delta:+.4f}")
    print(f"{'─'*50}")

    # Verdict
    if mAP_forget > 0.05:
        print("\n  ⚠️  Significant forgetting detected (>5% mAP drop).")
        print("     Consider increasing REPLAY_SAMPLE_SIZE in config.py.")
    elif mAP_forget > 0.01:
        print("\n  🟡 Minor forgetting detected. Acceptable for a prototype.")
    else:
        print("\n  ✅ Minimal forgetting. Replay buffer is working well.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model and measure forgetting")
    parser.add_argument("--weights", default=cfg.BEST_WEIGHTS, help="Weights to evaluate")
    parser.add_argument("--before",  default=None, help="Pre-CL weights (for forgetting comparison)")
    parser.add_argument("--after",   default=None, help="Post-CL weights (for forgetting comparison)")
    parser.add_argument("--dataset", default=None, help="Dataset YAML override")
    parser.add_argument("--save",    default=None, help="Save metrics JSON to this path")
    args = parser.parse_args()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # ── Forgetting comparison ─────────────────────────────────────────────────
    if args.before and args.after:
        before_metrics = evaluate_model(args.before, args.dataset)
        after_metrics  = evaluate_model(args.after,  args.dataset)

        print_metrics(before_metrics, label="BEFORE continual learning")
        print_metrics(after_metrics,  label="AFTER continual learning")

        forgetting = compute_forgetting_score(before_metrics, after_metrics)
        print_forgetting(forgetting)

        if args.save:
            with open(args.save, "w") as f:
                json.dump({
                    "before":     before_metrics,
                    "after":      after_metrics,
                    "forgetting": forgetting,
                }, f, indent=2)
            print(f"\nMetrics saved → {args.save}")

    # ── Single model evaluation ────────────────────────────────────────────────
    else:
        metrics = evaluate_model(args.weights, args.dataset)
        if metrics:
            print_metrics(metrics, label=os.path.basename(args.weights))
        else:
            print("\nNo metrics returned. Check your weights and dataset paths.")

        if args.save and metrics:
            with open(args.save, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved → {args.save}")


if __name__ == "__main__":
    main()
