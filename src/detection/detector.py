"""
YOLOv8 Detector Wrapper
========================
This is the core detection module.

What it does:
  - Loads a YOLOv8 model (pretrained or fine-tuned)
  - Runs inference on a single image
  - Returns structured detection results

Why a wrapper?
  The rest of the system (API, dashboard, continual learning)
  all call this one class. If you swap YOLOv8 for another model,
  you only change this file.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


@dataclass
class Detection:
    """Single detected object."""
    box: List[float]         # [x1, y1, x2, y2] in pixels
    class_id: int
    class_name: str
    confidence: float
    mask: object = None      # optional segmentation mask (numpy array)

    # Convenience properties for accessing box coordinates by name
    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None


@dataclass
class DetectionResult:
    """Full result from one image inference pass."""
    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: float = 0.0
    image_path: str = ""

    @property
    def count(self) -> int:
        return len(self.detections)

    def count_by_class(self) -> dict:
        """Return {class_name: count} for all detections."""
        counts = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Return a new DetectionResult with only detections above the threshold."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
            image_path=self.image_path,
        )

    def boxes(self) -> List[List[float]]:
        return [d.box for d in self.detections]

    def class_ids(self) -> List[int]:
        return [d.class_id for d in self.detections]

    def confidences(self) -> List[float]:
        return [d.confidence for d in self.detections]


class ShelfDetector:
    """
    YOLOv8-based retail shelf product detector.

    Usage:
        detector = ShelfDetector("yolov8n.pt")
        result = detector.detect("shelf_image.jpg")
        print(result.count_by_class())
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path to .pt weights file, or "yolov8n.pt" to auto-download
            confidence_threshold: Minimum confidence to keep a detection
            iou_threshold: NMS IoU threshold (higher = fewer merged boxes)
            device: "cpu" or "cuda" or "mps"
        """
        self.model_path = model_path
        self.conf = confidence_threshold
        self.iou  = iou_threshold
        self.device = device

        print(f"Loading model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.class_names: List[str] = list(self.model.names.values())
        print(f"Model loaded. Classes: {self.class_names}")

    def detect(self, image_input) -> DetectionResult:
        """
        Run detection on an image.

        Args:
            image_input: file path (str/Path), numpy array (RGB), or PIL image

        Returns:
            DetectionResult with all detected objects
        """
        import time
        start = time.time()

        results = self.model.predict(
            source=image_input,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.time() - start) * 1000

        # ultralytics returns a list; we always process one image
        result = results[0]
        h, w   = result.orig_shape

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id  = int(box.cls[0].item())
                conf    = float(box.conf[0].item())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)

                detections.append(Detection(
                    box=[x1, y1, x2, y2],
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                ))

        return DetectionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            inference_time_ms=elapsed_ms,
        )

    def segment(self, image_input) -> DetectionResult:
        """
        Run instance segmentation (requires a YOLOv8-seg model).
        Returns detections with pixel-level masks.

        Args:
            image_input: file path, numpy array (RGB), or PIL image

        Returns:
            DetectionResult with mask field populated on each Detection
        """
        import time
        start = time.time()

        results = self.model.predict(
            source=image_input,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.time() - start) * 1000
        result = results[0]
        h, w = result.orig_shape

        detections = []
        if result.boxes is not None:
            masks = result.masks  # may be None if model is not -seg
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)

                mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i].data.cpu().numpy().squeeze()

                detections.append(Detection(
                    box=[x1, y1, x2, y2],
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    mask=mask,
                ))

        return DetectionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            inference_time_ms=elapsed_ms,
        )

    def detect_and_visualize(self, image_input) -> tuple:
        """
        Run detection and return (DetectionResult, annotated_image_numpy).
        Convenience method used by the dashboard.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.image_utils import load_image
        from utils.visualizer import draw_detections

        # Load image to numpy if it's a path
        if isinstance(image_input, (str, Path)):
            image_np = load_image(str(image_input))
        else:
            image_np = np.array(image_input)

        result = self.detect(image_np)

        annotated = draw_detections(
            image_np,
            result.boxes(),
            result.class_ids(),
            result.confidences(),
            self.class_names,
        )

        return result, annotated

    def train(self, data_yaml: str, epochs: int = 20, batch: int = 4, imgsz: int = 640):
        """
        Fine-tune the model on new data.
        Used by the continual learning module.

        Args:
            data_yaml: path to dataset.yaml
            epochs: training epochs (keep low on CPU)
            batch: batch size (4 is safe on CPU)
            imgsz: image size
        """
        print(f"Starting training: {epochs} epochs, batch={batch}, imgsz={imgsz}")
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project="models/checkpoints",
            name="finetune_run",
            exist_ok=True,
        )
        print("Training complete.")

    def save(self, output_path: str):
        """Save current model weights."""
        self.model.save(output_path)
        print(f"Model saved to: {output_path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "ShelfDetector":
        """Load a previously saved fine-tuned checkpoint."""
        return cls(model_path=checkpoint_path, **kwargs)


if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Ensure the project root is in sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.image_utils import save_image

    parser = argparse.ArgumentParser(description="Test Retail Shelf Detector")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save annotated image")
    parser.add_argument("--weights", type=str, default="models/checkpoints/best.pt", help="Path to trained weights")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    print(f"Initializing detector with {args.weights}...")
    detector = ShelfDetector(model_path=args.weights)
    print(f"Running detection on {args.image}...")
    
    result, annotated = detector.detect_and_visualize(args.image)
    
    print(f"Found {result.count} products.")
    save_image(annotated, args.output)
    print(f"Annotated image saved to {args.output}")
