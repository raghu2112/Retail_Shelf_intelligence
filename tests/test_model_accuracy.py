# tests/test_model_accuracy.py
#
# PURPOSE:
#   Compare accuracy of different trained model weights.
#   Auto-discovers .pt files and runs YOLOv8 validation.
#
# USAGE:
#   python tests/test_model_accuracy.py              # interactive menu
#   python tests/test_model_accuracy.py --model yolov8s.pt  # specific model
#   python tests/test_model_accuracy.py --all         # compare all models

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

# Torch safe load patch to avoid weights_only issues on PyTorch 2.6+ / 2.12+ / Python 3.14+
import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load


def discover_models() -> list:
    """Find all .pt weights in the project root and checkpoints folder."""
    models = []
    
    # 1. Project root
    root_dir = Path(cfg.ROOT_DIR)
    for p in root_dir.glob("*.pt"):
        models.append(p.resolve())
        
    # 2. Checkpoints dir
    checkpoints_dir = Path(cfg.CHECKPOINTS_DIR)
    if checkpoints_dir.exists():
        for p in checkpoints_dir.rglob("*.pt"):
            models.append(p.resolve())
            
    # Deduplicate keeping order
    seen = set()
    deduped = []
    for m in models:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
            
    return deduped


def evaluate_single_model(model_path: Path) -> dict:
    """Run YOLOv8 validation on the model and return metrics dict."""
    from ultralytics import YOLO
    
    model_str = str(model_path)
    print(f"\nEvaluating: {model_path.name}")
    print(f"Path:       {model_str}")
    print(f"Device:     {cfg.DEVICE}")
    print("-" * 50)
    
    t0 = time.time()
    try:
        model = YOLO(model_str)
        results = model.val(
            data=cfg.DATASET_YAML,
            imgsz=cfg.IMG_SIZE,
            batch=cfg.BATCH_SIZE,
            device=cfg.DEVICE,
            verbose=False,
        )
        elapsed = time.time() - t0
        
        # Get metrics
        map50 = float(results.box.map50)
        map50_95 = float(results.box.map)
        precision = float(results.box.mp)
        recall = float(results.box.mr)
        # speed is dict of {preprocess, inference, postprocess, loss} in ms
        speed_inf = float(results.speed.get("inference", 0.0))
        speed_total = speed_inf + float(results.speed.get("preprocess", 0.0)) + float(results.speed.get("postprocess", 0.0))
        
        return {
            "name": model_path.name,
            "path": model_str,
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "speed_inf": speed_inf,
            "speed_total": speed_total,
            "success": True
        }
    except Exception as e:
        print(f"Error evaluating {model_path.name}: {e}")
        return {
            "name": model_path.name,
            "path": model_str,
            "success": False,
            "error": str(e)
        }


def format_table(headers, rows) -> str:
    """Render a clean, formatted ASCII table using box-drawing characters."""
    # Find column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(str(val)))
            
    # Build borders
    top_border = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
    mid_border = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
    bot_border = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"
    
    # Format header
    header_str = "│ " + " │ ".join(f"{h:<{widths[idx]}}" for idx, h in enumerate(headers)) + " │"
    
    # Format rows
    row_strs = []
    for row in rows:
        row_str = "│ " + " │ ".join(f"{str(val):<{widths[idx]}}" for idx, val in enumerate(row)) + " │"
        row_strs.append(row_str)
        
    return "\n".join([top_border, header_str, mid_border] + row_strs + [bot_border])


def display_results(results_list: list):
    """Sort and display evaluation results in a premium table format."""
    successful = [r for r in results_list if r.get("success")]
    failed = [r for r in results_list if not r.get("success")]
    
    if not successful:
        print("\nNo models were successfully evaluated.")
        return
        
    # Sort successful by mAP50 descending
    successful.sort(key=lambda x: x["mAP50"], reverse=True)
    
    # Find the best for each metric
    best_map50 = max(successful, key=lambda x: x["mAP50"])["mAP50"]
    best_map95 = max(successful, key=lambda x: x["mAP50_95"])["mAP50_95"]
    best_prec = max(successful, key=lambda x: x["precision"])["precision"]
    best_rec = max(successful, key=lambda x: x["recall"])["recall"]
    best_speed = min(successful, key=lambda x: x["speed_inf"])["speed_inf"]
    
    headers = ["Model Name", "mAP50", "mAP50-95", "Precision", "Recall", "Inference Speed", "Status"]
    
    rows = []
    for r in successful:
        # Mark best values
        map50_str = f"{r['mAP50']:.4f}" + (" *" if r["mAP50"] == best_map50 and len(successful) > 1 else "")
        map95_str = f"{r['mAP50_95']:.4f}" + (" *" if r["mAP50_95"] == best_map95 and len(successful) > 1 else "")
        prec_str = f"{r['precision']:.4f}" + (" *" if r["precision"] == best_prec and len(successful) > 1 else "")
        rec_str = f"{r['recall']:.4f}" + (" *" if r["recall"] == best_rec and len(successful) > 1 else "")
        speed_str = f"{r['speed_inf']:.1f} ms" + (" *" if r["speed_inf"] == best_speed and len(successful) > 1 else "")
        
        status = "Best Model" if r["mAP50"] == best_map50 and len(successful) > 1 else "Ready"
        
        rows.append([
            r["name"],
            map50_str,
            map95_str,
            prec_str,
            rec_str,
            speed_str,
            status
        ])
        
    print("\n" + "="*80)
    print("MODEL ACCURACY EVALUATION RESULTS")
    print("Sorted by mAP50 (higher is better)")
    if len(successful) > 1:
        print("Note: '*' indicates the best performer in that column.")
    print("="*80)
    print(format_table(headers, rows))
    
    if failed:
        print("\nFailed to evaluate:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of retail shelf YOLOv8 models")
    parser.add_argument("--model", type=str, help="Path/name of a specific model file (.pt)")
    parser.add_argument("--all", action="store_true", help="Evaluate all discovered models and compare them")
    args = parser.parse_args()
    
    # 1. Discover models
    discovered = discover_models()
    if not discovered:
        print(f"[ERROR] No .pt model files found in root or checkpoints directory ({cfg.CHECKPOINTS_DIR}).")
        return
        
    # Verify dataset yaml exists
    if not os.path.exists(cfg.DATASET_YAML):
        print(f"[ERROR] Dataset configuration YAML not found at: {cfg.DATASET_YAML}")
        print("Please run dataset preparation first: python -m src.utils.prepare_dataset")
        return
        
    results = []
    
    # Case A: Specific model via argument
    if args.model:
        model_path = Path(args.model)
        # Search in discovered or exact path
        match = None
        for d in discovered:
            if d.name == model_path.name or str(d) == str(model_path.resolve()):
                match = d
                break
        
        if not match:
            if model_path.exists():
                match = model_path
            else:
                print(f"[ERROR] Model file '{args.model}' not found in project or exact path.")
                return
                
        r = evaluate_single_model(match)
        results.append(r)
        display_results(results)
        
    # Case B: Evaluate all models
    elif args.all:
        print(f"Found {len(discovered)} models to evaluate.")
        for idx, m in enumerate(discovered):
            print(f"\n[{idx+1}/{len(discovered)}] ", end="")
            r = evaluate_single_model(m)
            results.append(r)
        display_results(results)
        
    # Case C: Interactive menu
    else:
        print("="*60)
        print("Retail Shelf Intelligence — Model Evaluation Tool")
        print("="*60)
        print("Discovered models:")
        for idx, m in enumerate(discovered):
            # Try to show a relative path for cleaner display
            try:
                rel_path = m.relative_to(cfg.ROOT_DIR)
            except ValueError:
                rel_path = m.name
            print(f"  {idx + 1:2d}) {rel_path}")
            
        print(f"  {len(discovered) + 1:2d}) [Evaluate All Models]")
        print(f"  {len(discovered) + 2:2d}) Exit")
        print("-" * 60)
        
        try:
            choice = input(f"Select a model to evaluate (1-{len(discovered)+2}): ").strip()
            choice_val = int(choice)
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            return
            
        if choice_val == len(discovered) + 2:
            print("Exiting.")
            return
        elif choice_val == len(discovered) + 1:
            print("\nEvaluating all discovered models...")
            for idx, m in enumerate(discovered):
                print(f"\n[{idx+1}/{len(discovered)}] ", end="")
                r = evaluate_single_model(m)
                results.append(r)
            display_results(results)
        elif 1 <= choice_val <= len(discovered):
            selected = discovered[choice_val - 1]
            r = evaluate_single_model(selected)
            results.append(r)
            display_results(results)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
