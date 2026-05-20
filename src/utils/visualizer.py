"""
Bounding box and annotation visualizer.
Draws detection results onto shelf images.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

# Fixed color palette — one color per class index (cycles if > 20 classes)
COLORS = [
    (255, 56,  56),   # red
    (56,  255, 56),   # green
    (56,  56,  255),  # blue
    (255, 165, 0),    # orange
    (255, 0,   255),  # magenta
    (0,   255, 255),  # cyan
    (255, 255, 0),    # yellow
    (128, 0,   128),  # purple
    (0,   128, 128),  # teal
    (128, 128, 0),    # olive
]


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    class_ids: List[int],
    confidences: List[float],
    class_names: List[str],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.

    Args:
        image: RGB numpy array
        boxes: list of [x1, y1, x2, y2] in pixel coordinates
        class_ids: list of class index for each box
        confidences: list of confidence scores
        class_names: list mapping class index to name
        thickness: box border thickness

    Returns:
        Annotated image (copy, original unchanged)
    """
    annotated = image.copy()

    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[cls_id % len(COLORS)]
        label = f"{class_names[cls_id]} {conf:.2f}"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    return annotated


def draw_anomaly_zones(
    image: np.ndarray,
    anomalies: List[Dict],
) -> np.ndarray:
    """
    Draw anomaly overlays on the image.

    Each anomaly dict has:
        - type: str ("empty_shelf", "low_stock", "misplaced")
        - zone: [x1, y1, x2, y2] or None for full image
        - message: str
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    anomaly_colors = {
        "empty_shelf": (255, 0, 0),      # red
        "low_stock":   (255, 165, 0),    # orange
        "misplaced":   (255, 255, 0),    # yellow
    }

    for anomaly in anomalies:
        atype = anomaly.get("type", "unknown")
        zone  = anomaly.get("zone", [0, 0, w, h])
        msg   = anomaly.get("message", atype)
        color = anomaly_colors.get(atype, (200, 200, 200))

        x1, y1, x2, y2 = zone

        # Semi-transparent fill
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        # Border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Message
        cv2.putText(
            annotated, msg,
            (x1 + 4, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 2, cv2.LINE_AA
        )

    return annotated
