"""
Product Counter
===============
Counts detected products per class and per shelf zone.

Why a separate file?
  Counting logic can get complex (zone-based counting, deduplication,
  confidence weighting). Keeping it separate makes it easy to improve
  without touching the detector.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from src.detection.detector import Detection, DetectionResult


# ── Zone-based counting (functional API) ──────────────────────────────────────

@dataclass
class ZoneCount:
    """Product counts for a rectangular zone of the shelf image."""
    zone_id: str
    zone_box: List[float]      # [x1, y1, x2, y2] in pixels
    counts: Dict[str, int]     # {class_name: count}

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def count_all_products(result: DetectionResult) -> Dict[str, int]:
    """
    Count detections by class across the entire image.

    Returns: {class_name: count}
    """
    return result.count_by_class()


def count_by_zone(
    result: DetectionResult,
    zones: List[Dict],
) -> List[ZoneCount]:
    """
    Count products inside each defined shelf zone.

    A zone is a dict: {"id": str, "box": [x1, y1, x2, y2]}
    A detection is assigned to a zone if its box CENTER falls inside it.

    Args:
        result:  DetectionResult from the detector
        zones:   List of zone dicts

    Returns:
        List of ZoneCount — one per zone
    """
    zone_counts = []

    for zone in zones:
        zone_id  = zone["id"]
        zx1, zy1, zx2, zy2 = zone["box"]
        class_counts: Dict[str, int] = {}

        for det in result.detections:
            # Use center point of detection box
            cx = (det.box[0] + det.box[2]) / 2
            cy = (det.box[1] + det.box[3]) / 2

            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        zone_counts.append(ZoneCount(
            zone_id=zone_id,
            zone_box=[zx1, zy1, zx2, zy2],
            counts=class_counts,
        ))

    return zone_counts


def build_default_zones(image_width: int, image_height: int, num_zones: int = 3) -> List[Dict]:
    """
    Divide the image into equal vertical strips (left/center/right shelf sections).
    Use this when you don't have manually defined zones.
    """
    zone_width = image_width / num_zones
    zones = []
    for i in range(num_zones):
        x1 = int(i * zone_width)
        x2 = int((i + 1) * zone_width)
        zones.append({
            "id":  f"zone_{i+1}",
            "box": [x1, 0, x2, image_height],
        })
    return zones


def summarize_counts(counts: Dict[str, int]) -> str:
    """Human-readable summary of product counts."""
    if not counts:
        return "No products detected."
    lines = [f"  {name}: {count}" for name, count in sorted(counts.items())]
    total = sum(counts.values())
    lines.append(f"  TOTAL: {total}")
    return "\n".join(lines)


# ── Class-based API (used by API server and anomaly detection) ────────────────

@dataclass
class ShelfZone:
    """A shelf zone with its detected products."""
    zone_id: int
    zone_box: List[float]
    count: int = 0
    detections: List[Detection] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        class_counts: Dict[str, int] = {}
        for d in self.detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        return class_counts


@dataclass
class ShelfStats:
    """Aggregated statistics for a shelf image."""
    total_products: int = 0
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    detection_density: float = 0.0
    zones: List[ShelfZone] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0


class ProductCounter:
    """
    Counts detected products per class and per shelf zone.
    Used by the API server and anomaly detection module.
    """

    def __init__(self, n_zones: int = 4):
        self.n_zones = n_zones

    def count(self, result: DetectionResult) -> ShelfStats:
        """
        Count products in the detection result, split into zones.

        Args:
            result: DetectionResult from ShelfDetector

        Returns:
            ShelfStats with per-zone breakdowns
        """
        img_w = result.image_width
        img_h = result.image_height

        # Build zones (horizontal strips)
        zone_height = img_h / self.n_zones if self.n_zones > 0 else img_h
        zones: List[ShelfZone] = []
        for i in range(self.n_zones):
            y1 = int(i * zone_height)
            y2 = int((i + 1) * zone_height)
            zones.append(ShelfZone(
                zone_id=i,
                zone_box=[0, y1, img_w, y2],
            ))

        # Assign detections to zones
        for det in result.detections:
            cy = (det.box[1] + det.box[3]) / 2
            zone_idx = min(int(cy / zone_height), self.n_zones - 1) if zone_height > 0 else 0
            zones[zone_idx].detections.append(det)
            zones[zone_idx].count += 1

        # Compute aggregate stats
        total = len(result.detections)
        counts_by_class = result.count_by_class()
        avg_conf = (
            sum(d.confidence for d in result.detections) / total
            if total > 0 else 0.0
        )
        density = total / (img_w * img_h) if (img_w * img_h) > 0 else 0.0

        return ShelfStats(
            total_products=total,
            counts_by_class=counts_by_class,
            avg_confidence=avg_conf,
            detection_density=density,
            zones=zones,
            image_width=img_w,
            image_height=img_h,
        )
