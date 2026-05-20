# src/continual_learning/replay_buffer.py
#
# PURPOSE:
#   Store a fixed-size sample of old training images.
#   When the model learns new products, we mix old samples in so it
#   doesn't forget what it already knows. This is called "Experience Replay".
#
# HOW IT WORKS:
#   1. After training on phase 1 products, we store N random samples.
#   2. When training on phase 2 products, we add those stored samples
#      to the new training batch.
#   3. The model sees old + new examples together → catastrophic forgetting reduced.
#
# USAGE:
#   from src.continual_learning.replay_buffer import ReplayBuffer
#   buf = ReplayBuffer()
#   buf.add(image_path, label_path)
#   samples = buf.sample(50)

import os
import sys
import json
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class BufferEntry:
    """One entry in the replay buffer: an image and its YOLO label file."""
    image_path: str
    label_path: str
    class_names: List[str]      # classes present in this image
    phase: int                  # which training phase added this entry
    timestamp: float = 0.0


class ReplayBuffer:
    """
    Fixed-size reservoir of (image, label) pairs from previous training phases.

    Uses reservoir sampling to keep a representative set when buffer is full.
    The buffer state is persisted to disk as a JSON index file.

    Args:
        max_size:    maximum number of entries to store
        buffer_dir:  directory where image/label copies are stored
    """

    INDEX_FILE = "buffer_index.json"

    def __init__(
        self,
        max_size:   int = None,
        buffer_dir: str = None,
    ):
        self.max_size   = max_size or cfg.REPLAY_BUFFER_MAX_SIZE
        self.buffer_dir = buffer_dir or cfg.REPLAY_BUFFER_DIR
        self.images_dir = os.path.join(self.buffer_dir, "images")
        self.labels_dir = os.path.join(self.buffer_dir, "labels")
        self.index_path = os.path.join(self.buffer_dir, self.INDEX_FILE)

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.entries: List[BufferEntry] = []
        self._total_seen = 0   # for reservoir sampling
        self._load_index()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        image_path: str,
        label_path: str,
        class_names: List[str] = None,
        phase: int = 0,
    ):
        """
        Add one (image, label) pair to the buffer.
        Copies files to the buffer directory.
        Uses reservoir sampling once the buffer is full.
        """
        import time

        self._total_seen += 1

        # Copy files to buffer storage with a unique name
        stem = f"phase{phase}_{self._total_seen:06d}"
        dst_img = os.path.join(self.images_dir, stem + Path(image_path).suffix)
        dst_lbl = os.path.join(self.labels_dir, stem + ".txt")

        shutil.copy2(image_path, dst_img)
        if os.path.exists(label_path):
            shutil.copy2(label_path, dst_lbl)
        else:
            # Create empty label file if no annotations
            open(dst_lbl, "w").close()

        entry = BufferEntry(
            image_path   = dst_img,
            label_path   = dst_lbl,
            class_names  = class_names or [],
            phase        = phase,
            timestamp    = time.time(),
        )

        if len(self.entries) < self.max_size:
            # Buffer not full yet — just append
            self.entries.append(entry)
        else:
            # Reservoir sampling: replace a random existing entry
            # This ensures all seen samples have equal probability of being kept
            replace_idx = random.randint(0, self._total_seen - 1)
            if replace_idx < self.max_size:
                # Remove old file copies
                old = self.entries[replace_idx]
                for p in (old.image_path, old.label_path):
                    if os.path.exists(p):
                        os.remove(p)
                self.entries[replace_idx] = entry
            else:
                # New entry not selected — clean up its copies
                for p in (dst_img, dst_lbl):
                    if os.path.exists(p):
                        os.remove(p)

        self._save_index()

    def add_from_directory(
        self,
        images_dir: str,
        labels_dir: str,
        phase: int = 0,
        max_add: int = None,
    ):
        """
        Bulk-add from a directory of images and labels.
        Useful for seeding the buffer after phase 1 training.

        Args:
            images_dir: folder of .jpg/.png images
            labels_dir: folder of matching .txt label files
            phase:      training phase number
            max_add:    if set, randomly sample at most this many images
        """
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if max_add and len(image_files) > max_add:
            image_files = random.sample(image_files, max_add)

        print(f"[ReplayBuffer] Adding {len(image_files)} images from phase {phase}")
        added = 0
        for img_name in image_files:
            img_path = os.path.join(images_dir, img_name)
            lbl_name = Path(img_name).stem + ".txt"
            lbl_path = os.path.join(labels_dir, lbl_name)
            self.add(img_path, lbl_path, phase=phase)
            added += 1

        print(f"[ReplayBuffer] Buffer size: {len(self.entries)} / {self.max_size}")
        return added

    def sample(self, n: int = None) -> List[BufferEntry]:
        """
        Sample n random entries from the buffer.
        Returns all entries if n is None or n > buffer size.
        """
        n = n or cfg.REPLAY_SAMPLE_SIZE
        if len(self.entries) <= n:
            return list(self.entries)
        return random.sample(self.entries, n)

    def export_for_training(
        self,
        output_dir: str,
        n_samples: int = None,
    ) -> tuple:
        """
        Export a sample of buffer entries to a directory for use in training.
        Returns (images_dir, labels_dir) paths.

        This is called by the trainer to get old samples before fine-tuning.
        """
        samples = self.sample(n_samples)
        out_images = os.path.join(output_dir, "images")
        out_labels = os.path.join(output_dir, "labels")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        for i, entry in enumerate(samples):
            suffix = Path(entry.image_path).suffix
            shutil.copy2(entry.image_path, os.path.join(out_images, f"replay_{i:04d}{suffix}"))
            if os.path.exists(entry.label_path):
                shutil.copy2(entry.label_path, os.path.join(out_labels, f"replay_{i:04d}.txt"))

        print(f"[ReplayBuffer] Exported {len(samples)} replay samples → {output_dir}")
        return out_images, out_labels

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict:
        """Return buffer statistics."""
        phase_counts = {}
        class_counts = {}
        for e in self.entries:
            phase_counts[e.phase] = phase_counts.get(e.phase, 0) + 1
            for cls in e.class_names:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        return {
            "total":        self.size,
            "max_size":     self.max_size,
            "total_seen":   self._total_seen,
            "by_phase":     phase_counts,
            "by_class":     class_counts,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_index(self):
        data = {
            "total_seen": self._total_seen,
            "entries":    [asdict(e) for e in self.entries],
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_index(self):
        if not os.path.exists(self.index_path):
            return
        with open(self.index_path) as f:
            data = json.load(f)
        self._total_seen = data.get("total_seen", 0)
        self.entries = []
        for e in data.get("entries", []):
            entry = BufferEntry(**e)
            # Only load entries whose files still exist
            if os.path.exists(entry.image_path):
                self.entries.append(entry)
        print(f"[ReplayBuffer] Loaded {len(self.entries)} entries from index.")
