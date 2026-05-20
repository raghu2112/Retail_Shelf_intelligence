# src/analytics/heatmap.py
#
# PURPOSE:
#   Generate product density heatmaps from detection results.
#   Visualises where products cluster and where gaps exist on the shelf.

import cv2
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


def generate_heatmap(
    detections: List[dict],
    image_width: int,
    image_height: int,
    radius: int = None,
) -> np.ndarray:
    """
    Create a heatmap array from detection bounding boxes.

    Each detection contributes a Gaussian blob centred at its box midpoint.
    Returns a float32 array (0–1) of shape (image_height, image_width).

    Args:
        detections: list of dicts with x1/y1/x2/y2 keys
        image_width:  width in pixels
        image_height: height in pixels
        radius: Gaussian blur kernel radius (default: from config)
    """
    radius = radius or cfg.HEATMAP_RADIUS
    heat = np.zeros((image_height, image_width), dtype=np.float32)

    for det in detections:
        cx = int((det["x1"] + det["x2"]) / 2)
        cy = int((det["y1"] + det["y2"]) / 2)
        w = int(det["x2"] - det["x1"])
        h = int(det["y2"] - det["y1"])

        # Draw a filled ellipse at the centre of each detection
        cv2.ellipse(heat, (cx, cy), (max(w // 2, 1), max(h // 2, 1)),
                    0, 0, 360, 1.0, -1)

    # Apply Gaussian blur for smooth heatmap
    ksize = radius * 2 + 1
    heat = cv2.GaussianBlur(heat, (ksize, ksize), 0)

    # Normalise to 0–1
    if heat.max() > 0:
        heat = heat / heat.max()

    return heat


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    intensity: float = None,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap onto an image.

    Args:
        image:     RGB numpy array (H, W, 3)
        heatmap:   float32 array (H, W) in range [0, 1]
        intensity: overlay opacity (default: from config)
        colormap:  OpenCV colormap

    Returns:
        Blended image as uint8 RGB array
    """
    intensity = intensity or cfg.HEATMAP_INTENSITY

    # Convert heatmap to coloured overlay
    heat_uint8 = (heatmap * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(image, 1 - intensity, heat_color, intensity, 0)
    return blended


def generate_gap_map(
    detections: List[dict],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Create an inverted heatmap highlighting shelf GAPS (empty areas).
    Useful for identifying restocking opportunities.

    Returns: float32 array (0–1) where 1 = empty area, 0 = product present
    """
    heatmap = generate_heatmap(detections, image_width, image_height)
    gap_map = 1.0 - heatmap
    return gap_map
