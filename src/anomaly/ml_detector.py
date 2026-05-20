# src/anomaly/ml_detector.py
#
# PURPOSE:
#   ML-based anomaly detection using Isolation Forest.
#   Learns "normal" shelf patterns from historical data and flags
#   unusual states without manual thresholds.

import sys
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class MLAnomaly:
    """ML-detected anomaly with score."""
    anomaly_score: float       # -1 = anomalous, 1 = normal (sklearn convention)
    confidence: float          # normalised score (0 = normal, 1 = anomalous)
    features: Dict[str, float]
    is_anomaly: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "type": "ml_anomaly",
            "severity": "high" if self.confidence > 0.8 else "medium" if self.confidence > 0.5 else "low",
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "zone_id": -1,
        }


class MLAnomalyDetector:
    """
    Isolation Forest anomaly detector for shelf statistics.

    Workflow:
      1. Collect normal shelf stats over time (fit)
      2. Score new observations (predict)
      3. Flag statistical outliers

    Features extracted from each observation:
      - total_products
      - avg_confidence
      - detection_density
      - zone_variance (variance in product counts across zones)
      - max_zone_gap (biggest difference between adjacent zones)
    """

    def __init__(
        self,
        contamination: float = None,
        model_path: str = None,
    ):
        self.contamination = contamination or cfg.ANOMALY_CONTAMINATION
        self.model_path = model_path or os.path.join(
            cfg.MODELS_DIR, "anomaly_model.pkl"
        )
        self._model = None
        self._fitted = False
        self._load_model()

    def _load_model(self):
        """Load a previously trained model if available."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            self._fitted = True

    def _save_model(self):
        """Persist model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)

    @staticmethod
    def extract_features(stats_dict: dict) -> Dict[str, float]:
        """
        Extract feature vector from a detection stats dictionary.

        Args:
            stats_dict: dict with total_products, avg_confidence, zones, etc.
        """
        zones = stats_dict.get("zones", [])
        zone_counts = [z.get("count", 0) for z in zones]

        total = stats_dict.get("total_products", 0)
        img_w = stats_dict.get("image_width", 640)
        img_h = stats_dict.get("image_height", 480)
        area = img_w * img_h

        features = {
            "total_products": float(total),
            "avg_confidence": float(stats_dict.get("avg_confidence", 0)),
            "detection_density": float(total / area) if area > 0 else 0,
            "zone_variance": float(np.var(zone_counts)) if zone_counts else 0,
            "max_zone_gap": float(
                max(abs(zone_counts[i] - zone_counts[i + 1])
                    for i in range(len(zone_counts) - 1))
                if len(zone_counts) > 1 else 0
            ),
        }
        return features

    def fit(self, history: List[dict], save: bool = True):
        """
        Train the anomaly model on historical shelf observations.

        Args:
            history: list of stats dicts (from API /detect responses)
            save: whether to persist the model
        """
        from sklearn.ensemble import IsolationForest

        if len(history) < 10:
            print(f"[ML Anomaly] Need at least 10 observations, got {len(history)}. Skipping.")
            return

        # Extract feature matrix
        feature_names = list(self.extract_features(history[0]).keys())
        X = np.array([
            [self.extract_features(obs)[f] for f in feature_names]
            for obs in history
        ])

        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self._model.fit(X)
        self._fitted = True

        if save:
            self._save_model()
        print(f"[ML Anomaly] Model trained on {len(history)} observations.")

    def predict(self, stats_dict: dict) -> Optional[MLAnomaly]:
        """
        Score a single observation.

        Returns MLAnomaly if anomalous, None if normal.
        """
        if not self._fitted or self._model is None:
            return None

        features = self.extract_features(stats_dict)
        feature_names = sorted(features.keys())
        X = np.array([[features[f] for f in feature_names]])

        score = self._model.decision_function(X)[0]
        prediction = self._model.predict(X)[0]

        # Convert score to 0–1 confidence (lower decision_function = more anomalous)
        confidence = max(0, min(1, -score))

        is_anomaly = prediction == -1

        if is_anomaly:
            # Find the most unusual feature
            top_feature = max(features.items(), key=lambda x: abs(x[1]))
            return MLAnomaly(
                anomaly_score=score,
                confidence=confidence,
                features=features,
                is_anomaly=True,
                description=(
                    f"ML model detected unusual shelf state "
                    f"(confidence: {confidence:.0%}). "
                    f"Most notable: {top_feature[0]}={top_feature[1]:.2f}"
                ),
            )
        return None

    @property
    def is_trained(self) -> bool:
        return self._fitted
