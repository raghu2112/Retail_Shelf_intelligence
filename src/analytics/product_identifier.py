# src/analytics/product_identifier.py
#
# PURPOSE:
#   Identify individual products by name using OCR on cropped detection regions.
#   Groups and counts each unique product found on the shelf.

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class IdentifiedProduct:
    """A single product with its extracted name and detection info."""
    name: str
    confidence: float         # OCR confidence for the name
    detection_confidence: float  # YOLO detection confidence
    bbox: List[float]         # [x1, y1, x2, y2]
    all_texts: List[str]      # all OCR text found in this crop
    crop_index: int           # index in original detections list

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ocr_confidence": round(self.confidence, 3),
            "detection_confidence": round(self.detection_confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox],
            "all_texts": self.all_texts,
        }


@dataclass
class ProductInventory:
    """Full product inventory extracted from a shelf image."""
    products: List[IdentifiedProduct]
    counts: Dict[str, int]            # {product_name: count}
    total_identified: int             # products with a readable name
    total_unidentified: int           # products where OCR found nothing
    unique_products: int              # number of distinct product names

    def to_dict(self) -> dict:
        # Sort counts by frequency (most common first)
        sorted_counts = dict(
            sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        )
        return {
            "counts": sorted_counts,
            "unique_products": self.unique_products,
            "total_identified": self.total_identified,
            "total_unidentified": self.total_unidentified,
            "products": [p.to_dict() for p in self.products],
        }

    def summary(self) -> str:
        lines = [
            f"Product Inventory: {self.unique_products} unique products, "
            f"{self.total_identified} identified, "
            f"{self.total_unidentified} unidentified",
            "",
        ]
        for name, count in sorted(
            self.counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)


class ProductIdentifier:
    """
    Identifies products by name using OCR on cropped detection regions.

    Workflow:
      1. Take YOLO detection bounding boxes
      2. Crop each detection region from the image (with padding)
      3. Run OCR on the cropped region
      4. Extract the most prominent text as the product name
      5. Clean and normalise names
      6. Group and count by product name
    """

    def __init__(
        self,
        ocr_languages: List[str] = None,
        min_ocr_confidence: float = None,
        crop_padding: float = 0.1,
        min_name_length: int = 2,
        max_products: int = 100,
        gpu: bool = False,
    ):
        """
        Args:
            ocr_languages: languages for EasyOCR
            min_ocr_confidence: minimum OCR confidence to accept text
            crop_padding: fraction of box size to pad when cropping (0.1 = 10%)
            min_name_length: minimum character length for a valid product name
            max_products: max number of products to OCR (largest first)
            gpu: use GPU for OCR
        """
        self.ocr_languages = ocr_languages or cfg.OCR_LANGUAGES
        self.min_ocr_confidence = min_ocr_confidence or cfg.OCR_CONFIDENCE
        self.crop_padding = crop_padding
        self.min_name_length = min_name_length
        self.max_products = max_products
        self._reader = None
        self._gpu = gpu

    @property
    def reader(self):
        """Lazy-load EasyOCR reader."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                self.ocr_languages,
                gpu=self._gpu,
                verbose=False,
            )
        return self._reader

    def _preprocess_crop_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Apply heavy image processing to improve OCR text readability on tiny/blurred labels."""
        import cv2
        import numpy as np
        
        # 1. Resize to optimal height for OCR (around 48px to 64px)
        # EasyOCR runs a CNN, so massive images slow it down and tiny images cause failures.
        height, width = crop.shape[:2]
        target_h = 48
        scale = target_h / max(height, 1) # Avoid division by zero
        
        # Don't scale down if it's already big, only scale up small texts
        if scale > 1.0:
            target_w = int(width * scale)
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = crop.copy()

        # 2. Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        
        # 3. Sharpen the image using a kernel
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        return enhanced

    def _crop_detection(
        self,
        image: np.ndarray,
        bbox: List[float],
    ) -> np.ndarray:
        """Crop a detection region from the image with padding."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)

        cx1 = max(0, int(x1) - pad_x)
        cy1 = max(0, int(y1) - pad_y)
        cx2 = min(w, int(x2) + pad_x)
        cy2 = min(h, int(y2) + pad_y)

        return image[cy1:cy2, cx1:cx2]

    def _clean_name(self, text: str) -> str:
        """Clean and normalise extracted product name."""
        import re

        # Remove excessive whitespace
        name = " ".join(text.split())

        # Remove purely numeric strings, prices, weights
        # but keep names that contain numbers (e.g. "7up", "V8")
        if re.match(r'^[\d\s.,£$€₹%]+$', name):
            return ""

        # Remove common noise patterns
        noise = [
            r'^\d+[gml]+$',       # weights like "500g", "250ml"
            r'^\d+\s*[xX×]\s*\d+', # sizes like "6x250"
            r'^[£$€₹]\s*\d',      # prices
            r'^\d+p$',             # pence prices
        ]
        for pattern in noise:
            if re.match(pattern, name, re.IGNORECASE):
                return ""

        # Capitalise properly
        name = name.strip()
        if name.isupper() and len(name) > 3:
            name = name.title()

        return name

    def _pick_best_name(self, ocr_results: list) -> Tuple[str, float]:
        """
        From OCR results for a single crop, pick the best product name.

        Prioritises:
          1. Longest text that passes cleaning
          2. Highest confidence among valid texts
        """
        candidates = []

        for bbox_points, text, conf in ocr_results:
            if conf < self.min_ocr_confidence:
                continue
            cleaned = self._clean_name(text)
            if len(cleaned) >= self.min_name_length:
                candidates.append((cleaned, conf, len(cleaned)))

        if not candidates:
            return ("", 0.0)

        # Sort by length (longer names are more informative), then confidence
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return (candidates[0][0], candidates[0][1])

    def identify(
        self,
        image: np.ndarray,
        detections: List[dict],
    ) -> ProductInventory:
        """
        Identify products in an image by name.

        Processes at most `max_products` detections (largest bounding boxes
        first, since bigger products have more readable labels).

        Args:
            image: RGB numpy array of the shelf image
            detections: list of detection dicts with x1/y1/x2/y2 keys

        Returns:
            ProductInventory with names, counts, and per-product details
        """
        products = []
        name_counter: Counter = Counter()
        unidentified = 0

        # Sort detections by bounding box area (largest first) and take top N
        indexed_dets = list(enumerate(detections))
        indexed_dets.sort(
            key=lambda x: (x[1]["x2"] - x[1]["x1"]) * (x[1]["y2"] - x[1]["y1"]),
            reverse=True,
        )
        selected = indexed_dets[:self.max_products]
        skipped = len(detections) - len(selected)

        for i, det in selected:
            bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]

            # Crop the detection region
            crop = self._crop_detection(image, bbox)

            # Skip tiny crops that won't have readable text
            if crop.shape[0] < 15 or crop.shape[1] < 15:
                unidentified += 1
                continue
            
            # Preprocess to boost text contrast
            enhanced_crop = self._preprocess_crop_for_ocr(crop)

            # Run OCR on the crop
            try:
                ocr_results = self.reader.readtext(enhanced_crop)
                # print(f"DEBUG OCR on crop shape {crop.shape}: found {len(ocr_results)} texts")
            except Exception as e:
                # print(f"DEBUG OCR Error: {e}")
                unidentified += 1
                continue

            # Extract all readable text
            all_texts = [
                text.strip()
                for _, text, conf in ocr_results
                if conf >= self.min_ocr_confidence and text.strip()
            ]

            # Pick the best name
            name, ocr_conf = self._pick_best_name(ocr_results)

            if name:
                name_counter[name] += 1
                products.append(IdentifiedProduct(
                    name=name,
                    confidence=ocr_conf,
                    detection_confidence=det.get("confidence", 0),
                    bbox=bbox,
                    all_texts=all_texts,
                    crop_index=i,
                ))
            else:
                unidentified += 1

        # Count skipped products as unidentified
        unidentified += skipped

        return ProductInventory(
            products=products,
            counts=dict(name_counter),
            total_identified=len(products),
            total_unidentified=unidentified,
            unique_products=len(name_counter),
        )
