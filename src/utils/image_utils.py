"""
Image preprocessing utilities.
Handles loading, resizing, and normalizing shelf images.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and convert to RGB.
    OpenCV loads in BGR by default — we convert to RGB for consistency.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resize image to target_size x target_size while preserving aspect ratio.
    Pads with gray (114, 114, 114) — standard YOLOv8 letterbox padding.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save a numpy RGB image to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array (RGB)."""
    return np.array(pil_image.convert("RGB"))


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy RGB array to PIL image."""
    return Image.fromarray(image.astype(np.uint8))
