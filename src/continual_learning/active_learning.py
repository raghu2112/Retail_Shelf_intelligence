# src/continual_learning/active_learning.py
#
# PURPOSE:
#   Select the most informative unlabeled images for human annotation.
#   Reduces labeling effort by focusing on samples where the model is
#   most uncertain.

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class UncertaintySample:
    """An image scored by model uncertainty."""
    image_path: str
    uncertainty_score: float
    num_detections: int
    avg_confidence: float
    method: str

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "uncertainty_score": round(self.uncertainty_score, 4),
            "num_detections": self.num_detections,
            "avg_confidence": round(self.avg_confidence, 4),
            "method": self.method,
        }


class ActiveLearner:
    """
    Active learning query strategy for retail shelf detection.

    Strategies:
      - entropy: high entropy in confidence distribution → uncertain
      - margin: small gap between top-2 confidences → confused
      - least_confident: lowest max confidence → unsure
      - detection_count: unusual number of detections → edge case
    """

    def __init__(
        self,
        method: str = None,
        pool_size: int = None,
        query_size: int = None,
    ):
        self.method = method or cfg.AL_UNCERTAINTY_METHOD
        self.pool_size = pool_size or cfg.AL_POOL_SIZE
        self.query_size = query_size or cfg.AL_QUERY_SIZE

    def score_uncertainty(
        self,
        confidences: List[float],
        method: str = None,
    ) -> float:
        """
        Compute uncertainty score for a single image's detections.

        Args:
            confidences: list of detection confidence scores
            method: scoring method override

        Returns:
            Uncertainty score (higher = more uncertain)
        """
        method = method or self.method

        if not confidences:
            return 1.0  # no detections = highly uncertain

        confs = np.array(confidences)

        if method == "entropy":
            # Shannon entropy of confidence distribution
            # Normalise confidences to sum to 1
            probs = confs / confs.sum() if confs.sum() > 0 else confs
            probs = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs * np.log(probs))
            return float(entropy)

        elif method == "margin":
            # Margin between highest and second-highest confidence
            if len(confs) < 2:
                return 1.0 - confs[0] if len(confs) == 1 else 1.0
            sorted_confs = np.sort(confs)[::-1]
            margin = sorted_confs[0] - sorted_confs[1]
            return float(1.0 - margin)  # smaller margin = more uncertain

        elif method == "least_confident":
            # 1 - max confidence
            return float(1.0 - confs.max())

        elif method == "detection_count":
            # Penalise unusual detection counts (too few or too many)
            # Assumes normal range is 20–200 for retail shelves
            count = len(confs)
            if count < 5:
                return 1.0
            elif count > 300:
                return 0.8
            return float(1.0 - np.mean(confs))

        else:
            return float(1.0 - np.mean(confs))

    def query(
        self,
        detector,
        image_paths: List[str],
    ) -> List[UncertaintySample]:
        """
        Score a pool of unlabeled images and return the most uncertain ones.

        Args:
            detector: ShelfDetector instance
            image_paths: list of image file paths to evaluate

        Returns:
            Top-k most uncertain samples, sorted by uncertainty (descending)
        """
        # Limit pool size
        pool = image_paths[:self.pool_size]
        scored = []

        for img_path in pool:
            try:
                result = detector.detect(img_path)
                confs = [d.confidence for d in result.detections]
                score = self.score_uncertainty(confs)
                scored.append(UncertaintySample(
                    image_path=img_path,
                    uncertainty_score=score,
                    num_detections=len(result.detections),
                    avg_confidence=float(np.mean(confs)) if confs else 0,
                    method=self.method,
                ))
            except Exception as e:
                print(f"[ActiveLearning] Skipping {img_path}: {e}")
                continue

        # Sort by uncertainty (highest first)
        scored.sort(key=lambda s: s.uncertainty_score, reverse=True)
        return scored[:self.query_size]

    def query_from_directory(
        self,
        detector,
        images_dir: str,
    ) -> List[UncertaintySample]:
        """
        Convenience: scan a directory of images and return top uncertain samples.
        """
        image_paths = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        return self.query(detector, image_paths)
