# src/continual_learning/trainer.py
#
# PURPOSE:
#   Incrementally fine-tune the YOLOv8 model when new product classes arrive.
#   Mixes new samples with old replay buffer samples to prevent forgetting.
#
# WORKFLOW:
#   Phase 1 → Train normally (train.py)
#             Seed replay buffer with phase 1 samples
#
#   Phase 2 → Call IncrementalTrainer.train_new_phase()
#             It automatically:
#               1. Pulls old samples from replay buffer
#               2. Combines them with new data
#               3. Fine-tunes the existing model (not from scratch)
#               4. Updates the replay buffer with new samples
#
# USAGE:
#   from src.continual_learning.trainer import IncrementalTrainer
#   trainer = IncrementalTrainer()
#   trainer.seed_buffer_from_phase(phase=1)
#   trainer.train_new_phase(
#       new_images_dir="data/phase2/images",
#       new_labels_dir="data/phase2/labels",
#       class_names=["product"],
#       phase=2
#   )

import os
import sys
import shutil
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg
from src.continual_learning.replay_buffer import ReplayBuffer


class IncrementalTrainer:
    """
    Manages incremental fine-tuning with experience replay.

    Args:
        buffer:        ReplayBuffer instance (created if not provided)
        weights_path:  starting weights (defaults to cfg.BEST_WEIGHTS)
    """

    def __init__(
        self,
        buffer: ReplayBuffer = None,
        weights_path: str = None,
    ):
        self.buffer       = buffer or ReplayBuffer()
        self.weights_path = weights_path or cfg.BEST_WEIGHTS

    def seed_buffer_from_phase(
        self,
        phase: int = 1,
        images_dir: str = None,
        labels_dir: str = None,
        max_samples: int = None,
    ):
        """
        Populate the replay buffer from the processed dataset.
        Call this once after initial training to preserve phase 1 knowledge.

        Args:
            phase:      phase number label (used for tracking)
            images_dir: defaults to processed train images
            labels_dir: defaults to processed train labels
            max_samples: max images to add (defaults to REPLAY_BUFFER_MAX_SIZE)
        """
        images_dir = images_dir or os.path.join(cfg.PROCESSED_DIR, "images", "train")
        labels_dir = labels_dir or os.path.join(cfg.PROCESSED_DIR, "labels", "train")

        if not os.path.exists(images_dir):
            print(f"[Trainer] Images dir not found: {images_dir}")
            print("Run prepare_dataset.py first.")
            return

        max_add = max_samples or cfg.REPLAY_BUFFER_MAX_SIZE
        n = self.buffer.add_from_directory(images_dir, labels_dir, phase=phase, max_add=max_add)
        print(f"[Trainer] Seeded replay buffer with {n} phase-{phase} samples.")
        print(f"[Trainer] Buffer stats: {self.buffer.stats()}")

    def train_new_phase(
        self,
        new_images_dir: str,
        new_labels_dir: str,
        class_names: list,
        phase: int = 2,
        epochs: int = None,
        output_weights: str = None,
    ):
        """
        Fine-tune the model on new data while replaying old samples.

        Args:
            new_images_dir: directory of new product images (YOLO format)
            new_labels_dir: directory of matching label .txt files
            class_names:    list of ALL class names (old + new)
            phase:          phase number (for buffer bookkeeping)
            epochs:         training epochs (defaults to cfg.CL_EPOCHS)
            output_weights: where to save updated weights
        """
        epochs         = epochs or cfg.CL_EPOCHS
        output_weights = output_weights or cfg.BEST_WEIGHTS

        if not os.path.exists(self.weights_path):
            print(f"[Trainer] Weights not found: {self.weights_path}")
            print("Train the base model first with: python -m src.detection.train")
            return

        print(f"\n{'='*60}")
        print(f"Incremental Training — Phase {phase}")
        print(f"{'='*60}")
        print(f"  Base weights:  {self.weights_path}")
        print(f"  New data:      {new_images_dir}")
        print(f"  Replay buffer: {self.buffer.size} samples")
        print(f"  Epochs:        {epochs}")

        # ── Step 1: Build combined dataset in a temp directory ────────────────
        with tempfile.TemporaryDirectory(prefix="cl_train_") as tmpdir:
            combined_images = os.path.join(tmpdir, "images", "train")
            combined_labels = os.path.join(tmpdir, "labels", "train")
            val_images      = os.path.join(tmpdir, "images", "val")
            val_labels      = os.path.join(tmpdir, "labels", "val")

            os.makedirs(combined_images, exist_ok=True)
            os.makedirs(combined_labels, exist_ok=True)
            os.makedirs(val_images,      exist_ok=True)
            os.makedirs(val_labels,      exist_ok=True)

            # Copy new images and labels
            n_new = self._copy_split(new_images_dir, new_labels_dir, combined_images, combined_labels, prefix="new_")
            print(f"\n  New samples added: {n_new}")

            # Export replay buffer samples
            replay_tmp = os.path.join(tmpdir, "replay")
            self.buffer.export_for_training(replay_tmp, n_samples=cfg.REPLAY_SAMPLE_SIZE)
            replay_imgs = os.path.join(replay_tmp, "images")
            replay_lbls = os.path.join(replay_tmp, "labels")
            n_replay = self._copy_split(replay_imgs, replay_lbls, combined_images, combined_labels, prefix="replay_")
            print(f"  Replay samples added: {n_replay}")

            # Use a small portion of new data as val
            n_val = self._copy_split(
                new_images_dir, new_labels_dir,
                val_images, val_labels,
                prefix="val_", max_count=max(1, n_new // 5)
            )
            print(f"  Val samples: {n_val}")
            print(f"  Total training samples: {n_new + n_replay}")

            # ── Step 2: Write dataset YAML ────────────────────────────────────
            yaml_path = os.path.join(tmpdir, "cl_dataset.yaml")
            dataset_config = {
                "path":  tmpdir,
                "train": "images/train",
                "val":   "images/val",
                "nc":    len(class_names),
                "names": class_names,
            }
            with open(yaml_path, "w") as f:
                yaml.dump(dataset_config, f)

            # ── Step 3: Fine-tune ──────────────────────────────────────────────
            from ultralytics import YOLO
            model = YOLO(self.weights_path)

            print(f"\n  Starting fine-tuning...")
            results = model.train(
                data      = yaml_path,
                epochs    = epochs,
                batch     = cfg.BATCH_SIZE,
                imgsz     = cfg.IMG_SIZE,
                device    = cfg.DEVICE,
                workers   = cfg.WORKERS,
                project   = os.path.join(cfg.CHECKPOINTS_DIR, f"phase{phase}"),
                name      = "finetune",
                exist_ok  = True,
                pretrained= False,  # we're starting FROM our weights, not COCO
                amp       = False,
                lr0       = 0.001,  # lower LR for fine-tuning (don't destroy old weights)
                lrf       = 0.01,
                freeze    = 10,     # freeze first 10 backbone layers to preserve features
                verbose   = True,
            )

        # ── Step 4: Save best weights ──────────────────────────────────────────
        best_src = os.path.join(
            cfg.CHECKPOINTS_DIR, f"phase{phase}", "finetune", "weights", "best.pt"
        )
        if os.path.exists(best_src):
            shutil.copy2(best_src, output_weights)
            print(f"\n  Updated weights saved → {output_weights}")
        else:
            print(f"\n  [WARNING] Could not find best.pt at: {best_src}")

        # ── Step 5: Update replay buffer with new data ─────────────────────────
        self.buffer.add_from_directory(
            new_images_dir, new_labels_dir,
            phase=phase,
            max_add=cfg.REPLAY_SAMPLE_SIZE,
        )

        mAP = results.results_dict.get("metrics/mAP50(B)", "N/A")
        print(f"\n{'='*60}")
        print(f"Phase {phase} complete. mAP50: {mAP}")
        print(f"Replay buffer size: {self.buffer.size}")
        print(f"{'='*60}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _copy_split(
        self,
        src_images: str,
        src_labels: str,
        dst_images: str,
        dst_labels: str,
        prefix: str = "",
        max_count: int = None,
    ) -> int:
        """Copy images + labels from src to dst directories."""
        if not os.path.exists(src_images):
            return 0

        files = [
            f for f in os.listdir(src_images)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if max_count:
            import random
            files = random.sample(files, min(max_count, len(files)))

        copied = 0
        for fname in files:
            src_img = os.path.join(src_images, fname)
            dst_img = os.path.join(dst_images, prefix + fname)
            shutil.copy2(src_img, dst_img)

            stem    = Path(fname).stem
            src_lbl = os.path.join(src_labels, stem + ".txt")
            dst_lbl = os.path.join(dst_labels, prefix + stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()
            copied += 1
        return copied
