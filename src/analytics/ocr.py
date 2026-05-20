# src/analytics/ocr.py
#
# PURPOSE:
#   Extract text from shelf images (price tags, product labels, brand names)
#   using EasyOCR. Results are paired with nearest detection bounding box.

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class OCRResult:
    """Single OCR text detection."""
    text: str
    confidence: float
    bbox: List[int]         # [x1, y1, x2, y2]
    nearest_product: Optional[dict] = None  # nearest detection box

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "nearest_product": self.nearest_product,
        }


class ShelfOCR:
    """
    OCR reader for retail shelf images.

    Uses EasyOCR for text extraction, then links detected text
    to the nearest product bounding box.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or cfg.OCR_LANGUAGES
        self._reader = None
        self._gpu = gpu

    @property
    def reader(self):
        """Lazy-load EasyOCR reader (heavy import)."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self._gpu,
                verbose=False,
            )
        return self._reader

    def read(
        self,
        image: np.ndarray,
        min_confidence: float = None,
    ) -> List[OCRResult]:
        """
        Extract text from an image.

        Args:
            image: RGB numpy array
            min_confidence: minimum confidence threshold

        Returns:
            List of OCRResult objects
        """
        min_confidence = min_confidence or cfg.OCR_CONFIDENCE

        raw_results = self.reader.readtext(image)
        results = []

        for bbox_points, text, conf in raw_results:
            if conf < min_confidence:
                continue

            # Convert polygon to [x1, y1, x2, y2]
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

            results.append(OCRResult(
                text=text.strip(),
                confidence=conf,
                bbox=bbox,
            ))

        return results

    def read_with_products(
        self,
        image: np.ndarray,
        detections: List[dict],
        min_confidence: float = None,
    ) -> List[OCRResult]:
        """
        Extract text and link each to the nearest product detection.

        Args:
            image:      RGB numpy array
            detections: list of dicts with x1/y1/x2/y2/class_name keys
            min_confidence: OCR confidence threshold

        Returns:
            List of OCRResult with nearest_product filled in
        """
        ocr_results = self.read(image, min_confidence)

        for ocr in ocr_results:
            ocr_cx = (ocr.bbox[0] + ocr.bbox[2]) / 2
            ocr_cy = (ocr.bbox[1] + ocr.bbox[3]) / 2

            best_dist = float("inf")
            best_det = None

            for det in detections:
                det_cx = (det["x1"] + det["x2"]) / 2
                det_cy = (det["y1"] + det["y2"]) / 2
                dist = ((ocr_cx - det_cx) ** 2 + (ocr_cy - det_cy) ** 2) ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            ocr.nearest_product = best_det

        return ocr_results

    def extract_prices(self, ocr_results: List[OCRResult]) -> List[dict]:
        """
        Filter OCR results to find price-like patterns.

        Returns list of {text, value, bbox, product}
        """
        import re
        prices = []
        price_pattern = re.compile(r'[£$€₹]?\s*\d+[.,]\d{2}')

        for ocr in ocr_results:
            matches = price_pattern.findall(ocr.text)
            for match in matches:
                # Extract numeric value
                numeric = re.sub(r'[^\d.,]', '', match).replace(',', '.')
                try:
                    value = float(numeric)
                    prices.append({
                        "text": match.strip(),
                        "value": value,
                        "bbox": ocr.bbox,
                        "product": ocr.nearest_product,
                    })
                except ValueError:
                    continue

        return prices
