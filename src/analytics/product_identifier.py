# src/analytics/product_identifier.py
#
# PURPOSE:
#   Identify individual products by name using OCR on cropped detection regions.
#   Groups and counts each unique product found on the shelf.
#
# IMPROVEMENTS (v2):
#   - Multi-strategy preprocessing: tries 3 enhancement pipelines per crop
#   - Multi-scale OCR: tries original + 2× upscale for tiny products
#   - Fuzzy name grouping: merges near-identical names (e.g. "Coca Cola" / "Coca-cola")
#   - Smarter name extraction: concatenates nearby text fragments
#   - Larger crop padding: captures more of the label
#   - Lower minimum crop size: fewer products skipped

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
from difflib import SequenceMatcher

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
      3. Run multi-strategy preprocessing on the crop
      4. Run OCR at multiple scales and pick the richest result
      5. Extract and concatenate the most informative text fragments
      6. Clean and normalise names
      7. Fuzzy-group similar names and count by product
    """

    # ── Noise patterns compiled once ─────────────────────────────────────
    _NOISE_PATTERNS = [
        re.compile(r'^[\d\s.,£$€₹%/:×xX\-]+$'),  # pure numbers / symbols
        re.compile(r'^\d+\s*[gG][rR]?[mM]?[sS]?$'),  # weights: 500g, 250gms
        re.compile(r'^\d+\s*[mM][lL]$'),               # volumes: 250ml
        re.compile(r'^\d+\s*[lL]$'),                    # litres: 2l
        re.compile(r'^\d+\s*[kK][gG]$'),               # kilos: 1kg
        re.compile(r'^\d+\s*[oO][zZ]$'),               # ounces: 12oz
        re.compile(r'^\d+\s*[xX×]\s*\d+'),             # multipacks: 6x250
        re.compile(r'^[£$€₹]\s*\d'),                   # prices: $3.99
        re.compile(r'^\d+[pP]$'),                       # pence: 99p
        re.compile(r'^\d+\.\d{2}$'),                    # bare prices: 3.99
        re.compile(r'^(net|wt|vol|qty|exp|mfg|mrp|best before|use by)', re.I),
        re.compile(r'^www\.|\.com|\.in|\.co', re.I),   # URLs
        re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'),  # dates
        re.compile(r'^[A-Z]{1,2}\d{4,}$'),              # barcodes / serial numbers
    ]

    def __init__(
        self,
        ocr_languages: List[str] = None,
        min_ocr_confidence: float = None,
        crop_padding: float = 0.15,
        min_name_length: int = 2,
        max_products: int = 100,
        gpu: bool = None,
        fuzzy_threshold: float = 0.78,
    ):
        """
        Args:
            ocr_languages: languages for EasyOCR
            min_ocr_confidence: minimum OCR confidence to accept text
            crop_padding: fraction of box size to pad when cropping (0.15 = 15%).
                          ACCURACY TIP: Adjust this to 0.20 or 0.25 to prevent cutting off text on label edges.
            min_name_length: minimum character length for a valid product name
            max_products: max number of products to OCR (largest first)
            gpu: use GPU for OCR (defaults to True if cfg.DEVICE is cuda)
            fuzzy_threshold: similarity ratio above which two names are merged
            
        ACCURACY TIPS:
        1. PREDEFINED DICTIONARY (Catalog Matching): Create a hardcoded list of expected brand names
           (e.g., ["Coca-Cola", "Pepsi", "Colgate"]) and use SequenceMatcher to snap the raw OCR output 
           to the closest valid spelling. This is the single most effective way to eliminate minor OCR typos.
        2. ALTERNATIVE OCR: Swap EasyOCR with PaddleOCR inside `src/analytics/ocr.py` for superior text 
           extraction on curved, stylized, or vertical packaging fonts.
        """
        self.ocr_languages = ocr_languages or cfg.OCR_LANGUAGES
        self.min_ocr_confidence = min_ocr_confidence or cfg.OCR_CONFIDENCE
        self.crop_padding = crop_padding
        self.min_name_length = min_name_length
        self.max_products = max_products
        self._reader = None
        self._gpu = gpu if gpu is not None else (cfg.DEVICE == "cuda")
        self.fuzzy_threshold = fuzzy_threshold

        # Brand Catalog for spelling correction and fuzzy mapping
        self.brand_catalog = {
            "Coca-Cola": ["coca", "cola", "coke", "ccca", "c0la", "ccba", "ca-co"],
            "Fanta": ["fanta", "fata", "fant", "fnt"],
            "Sprite": ["sprite", "sprt", "spri"],
            "Dr Pepper": ["pepper", "dr", "pep", "dr.pep"],
            "Minute Maid": ["minute", "maid", "minut", "maid"],
            "Pepsi": ["pepsi", "pep"],
            "Tim Hortons": ["tim", "cafe", "tmcafe", "horton"],
            "Profissimo": ["profissimo", "profis"],
            "Pure": ["pure", "pur"],
            "Calggy": ["calggy", "calg"],
        }

    def _match_catalog(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        
        # 1. Keyword matching
        for brand, keywords in self.brand_catalog.items():
            for kw in keywords:
                if kw in text_lower:
                    return brand
                    
        # 2. Fuzzy similarity matching using SequenceMatcher
        for brand in self.brand_catalog.keys():
            ratio = SequenceMatcher(None, brand.lower(), text_lower).ratio()
            if ratio >= 0.65:
                return brand
                
        return None

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

    # ── Multi-strategy preprocessing ─────────────────────────────────────

    def _preprocess_strategies(self, crop: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessed versions of a crop.
        OCR will be run on each and the richest result is kept.

        Strategies:
          1. Original RGB (upscaled) - preserves natural color gradients and text detection structures
          2. Grayscale + CLAHE  (good for colored labels and shadows)
          3. Grayscale + Sharpen  (good for blurry characters)
        """
        import cv2

        results = []
        height, width = crop.shape[:2]

        # ── Upscale crops aggressively to ensure text is large enough to read ──────────
        min_h = 256
        if height < min_h:
            scale = min_h / max(height, 1)
            new_w = max(int(width * scale), 1)
            crop = cv2.resize(crop, (new_w, min_h), interpolation=cv2.INTER_CUBIC)

        # Strategy 1: Upscaled RGB crop (highly recommended for deep learning OCR)
        results.append(crop)

        # Strategy 2: Grayscale + CLAHE
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        s2 = clahe.apply(gray)
        results.append(s2)

        # Strategy 3: Grayscale + Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        s3 = cv2.filter2D(gray, -1, kernel)
        results.append(s3)

        return results

    # ── Multi-scale OCR ──────────────────────────────────────────────────

    def _run_multiscale_ocr(self, crop: np.ndarray) -> list:
        """
        Run OCR using multiple preprocessing strategies and return
        the result set with the highest total confidence yield.
        """
        strategies = self._preprocess_strategies(crop)

        best_results = []
        best_score = -1.0

        for enhanced in strategies:
            try:
                ocr_results = self.reader.readtext(enhanced)
            except Exception:
                continue

            # Score = total confidence of valid-length texts
            score = sum(
                conf for _, text, conf in ocr_results
                if conf >= self.min_ocr_confidence and len(text.strip()) >= self.min_name_length
            )
            if score > best_score:
                best_score = score
                best_results = ocr_results

        return best_results

    # ── Crop helper ──────────────────────────────────────────────────────

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

    # ── Name cleaning ────────────────────────────────────────────────────

    def _clean_name(self, text: str) -> str:
        """Clean and normalise extracted product name."""
        # Remove excessive whitespace
        name = " ".join(text.split())

        # Reject if it matches any noise pattern
        for pattern in self._NOISE_PATTERNS:
            if pattern.match(name):
                return ""

        # Remove leading/trailing punctuation (but keep internal ones like hyphens)
        name = name.strip(".,;:!?|/\\()[]{}<>\"'`~@#^&*_=+ ")

        # Skip single characters
        if len(name) < self.min_name_length:
            return ""

        # Capitalise properly: ALL-CAPS words > 3 chars → Title Case
        if name.isupper() and len(name) > 3:
            name = name.title()

        return name.strip()

    # ── Name extraction ──────────────────────────────────────────────────

    def _pick_best_name(self, ocr_results: list) -> Tuple[str, float]:
        """
        From OCR results for a single crop, build the best product name.
        Uses a predefined brand catalog to correct typos and map fuzzy detections.
        """
        # 1. First, try catalog matching on ALL OCR fragments (even lower confidence ones)
        for bbox_points, text, conf in ocr_results:
            if conf >= 0.15:  # lower threshold allowed for catalog matching
                cleaned = self._clean_name(text)
                if cleaned:
                    matched_brand = self._match_catalog(cleaned)
                    if matched_brand:
                        # Success! Found a catalog match
                        return (matched_brand, max(conf, 0.85))  # boost confidence on catalog match

        # 2. Fallback to standard cleaning/concatenation if no catalog match is found
        candidates = []

        for bbox_points, text, conf in ocr_results:
            if conf < self.min_ocr_confidence:
                continue
            cleaned = self._clean_name(text)
            if len(cleaned) >= self.min_name_length:
                candidates.append((cleaned, conf, len(cleaned)))

        if not candidates:
            return ("", 0.0)

        # Sort by length desc, then confidence desc
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

        best_name = candidates[0][0]
        best_conf = candidates[0][1]

        # If the best name is short (≤ 5 chars) and there's a second
        # candidate, try concatenating them for a fuller product name.
        if len(best_name) <= 5 and len(candidates) >= 2:
            second = candidates[1][0]
            combined = f"{best_name} {second}"
            # Only use combined if the second fragment isn't a duplicate
            if second.lower() != best_name.lower():
                best_name = combined
                best_conf = (best_conf + candidates[1][1]) / 2

        return (best_name, best_conf)

    # ── Fuzzy name grouping ──────────────────────────────────────────────

    def _fuzzy_group_names(self, name_counter: Counter) -> Counter:
        """
        Merge product names that are near-identical due to OCR typos.

        e.g. "Coca Cola", "Coca-Cola", "Coca cola" → keep the most
        frequent spelling and sum all counts.
        """
        if len(name_counter) <= 1:
            return name_counter

        names = list(name_counter.keys())
        merged = Counter()
        used = set()

        # Sort by frequency so the most common spelling is the canonical one
        sorted_names = sorted(names, key=lambda n: name_counter[n], reverse=True)

        for name in sorted_names:
            if name in used:
                continue

            canonical = name
            total_count = name_counter[name]

            for other in sorted_names:
                if other == name or other in used:
                    continue

                ratio = SequenceMatcher(
                    None,
                    canonical.lower(),
                    other.lower(),
                ).ratio()

                if ratio >= self.fuzzy_threshold:
                    total_count += name_counter[other]
                    used.add(other)

            merged[canonical] = total_count
            used.add(name)

        return merged

    # ── Main identification pipeline ─────────────────────────────────────

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

            # Skip truly tiny crops (reduced from 15px to 10px)
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                unidentified += 1
                continue

            # Run multi-strategy, multi-scale OCR
            try:
                ocr_results = self._run_multiscale_ocr(crop)
            except Exception:
                unidentified += 1
                continue

            # Extract all readable text
            all_texts = [
                text.strip()
                for _, text, conf in ocr_results
                if conf >= self.min_ocr_confidence and text.strip()
            ]

            # Pick the best name (with fragment concatenation)
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

        # Fuzzy-group similar product names
        grouped_counts = self._fuzzy_group_names(name_counter)

        # Update product names to their canonical (grouped) form
        # Build a mapping: old_name → canonical_name
        canonical_map = {}
        raw_names = list(name_counter.keys())
        for canonical, _ in grouped_counts.items():
            for raw in raw_names:
                ratio = SequenceMatcher(None, canonical.lower(), raw.lower()).ratio()
                if ratio >= self.fuzzy_threshold or raw == canonical:
                    canonical_map[raw] = canonical

        for product in products:
            product.name = canonical_map.get(product.name, product.name)

        return ProductInventory(
            products=products,
            counts=dict(grouped_counts),
            total_identified=len(products),
            total_unidentified=unidentified,
            unique_products=len(grouped_counts),
        )
