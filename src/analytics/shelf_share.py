# src/analytics/shelf_share.py
#
# PURPOSE:
#   Calculate shelf share — the percentage of shelf space occupied
#   by each product class. Key retail KPI.

import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class ShelfShareResult:
    """Shelf share analysis results."""
    total_shelf_area: float           # total image area (pixels²)
    occupied_area: float              # total area covered by detections
    empty_area: float                 # unoccupied area
    occupancy_rate: float             # occupied / total (0–1)
    share_by_class: Dict[str, float]  # {class_name: percentage}
    area_by_class: Dict[str, float]   # {class_name: pixels²}
    count_by_class: Dict[str, int]    # {class_name: count}

    def summary(self) -> str:
        lines = [
            f"Shelf Occupancy: {self.occupancy_rate * 100:.1f}%",
            f"Occupied: {self.occupied_area:.0f} px²  |  Empty: {self.empty_area:.0f} px²",
            "",
            "Share by class:",
        ]
        for cls, share in sorted(self.share_by_class.items(),
                                  key=lambda x: x[1], reverse=True):
            count = self.count_by_class.get(cls, 0)
            lines.append(f"  {cls}: {share:.1f}% ({count} items)")
        return "\n".join(lines)


def calculate_shelf_share(
    detections: List[dict],
    image_width: int,
    image_height: int,
) -> ShelfShareResult:
    """
    Calculate shelf share from detection results.

    Uses a pixel mask to calculate exact occupancy (handling overlaps)
    and groups by OCR-extracted product_name if available.

    Args:
        detections: list of dicts with x1/y1/x2/y2 and optional product_name
        image_width:  image width in pixels
        image_height: image height in pixels

    Returns:
        ShelfShareResult with occupancy and per-product share
    """
    import numpy as np

    if not detections:
        return ShelfShareResult(0, 0, 0, 0.0, {}, {}, {})

    # Calculate the active shelf area (bounding box of all products combined)
    # This ignores floors/ceilings to give a realistic occupancy metric
    min_x = int(min(det["x1"] for det in detections))
    min_y = int(min(det["y1"] for det in detections))
    max_x = int(max(det["x2"] for det in detections))
    max_y = int(max(det["y2"] for det in detections))
    
    # Clamp to image boundaries
    min_x, max_x = max(0, min_x), min(image_width, max_x)
    min_y, max_y = max(0, min_y), min(image_height, max_y)

    total_area = max(0, max_x - min_x) * max(0, max_y - min_y)
    if total_area <= 0:
        return ShelfShareResult(0, 0, 0, 0.0, {}, {}, {})

    # Create a boolean mask for the active shelf area
    occupancy_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)

    area_by_name: Dict[str, float] = {}
    count_by_name: Dict[str, int] = {}

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        # Clamp coordinates
        x1, x2 = max(0, x1), min(image_width, x2)
        y1, y2 = max(0, y1), min(image_height, y2)

        # Mark occupied pixels in the cropped mask
        mask_x1, mask_x2 = max(0, x1 - min_x), max(0, x2 - min_x)
        mask_y1, mask_y2 = max(0, y1 - min_y), max(0, y2 - min_y)
        if mask_x1 < mask_x2 and mask_y1 < mask_y2:
            occupancy_mask[mask_y1:mask_y2, mask_x1:mask_x2] = True

        area = (x2 - x1) * (y2 - y1)
        
        # Prefer product_name from OCR, fallback to 'Unidentified' instead of raw class
        name = det.get("product_name")
        if not name or name == "Unknown":
            name = "Unidentified"

        area_by_name[name] = area_by_name.get(name, 0) + area
        count_by_name[name] = count_by_name.get(name, 0) + 1

    # Exact occupied pixels (no double counting overlaps)
    occupied = float(occupancy_mask.sum())
    empty = max(0, total_area - occupied)
    occupancy_rate = occupied / total_area

    # Share as percentage of OCCUPIED area
    share_by_name = {}
    total_bbox_area = sum(area_by_name.values())
    for name, area in area_by_name.items():
        share_by_name[name] = (area / total_bbox_area * 100) if total_bbox_area > 0 else 0

    return ShelfShareResult(
        total_shelf_area=total_area,
        occupied_area=occupied,
        empty_area=empty,
        occupancy_rate=min(occupancy_rate, 1.0),
        share_by_class=share_by_name,
        area_by_class=area_by_name,
        count_by_class=count_by_name,
    )
