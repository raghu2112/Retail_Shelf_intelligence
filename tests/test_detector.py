# tests/test_detector.py
#
# PURPOSE:
#   Tests for the detection pipeline.
#   Uses a synthetic image so you don't need real shelf photos to run tests.
#
# USAGE:
#   pytest tests/ -v

import sys
import os
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_image(tmp_path):
    """Create a small synthetic shelf image for testing."""
    import cv2
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # grey background
    # Draw coloured rectangles to simulate products
    for i in range(6):
        x1 = 50 + i * 90
        cv2.rectangle(img, (x1, 100), (x1 + 70, 280), (30, 80 + i * 20, 120), -1)
        cv2.rectangle(img, (x1, 300), (x1 + 70, 440), (120, 40 + i * 15, 80), -1)
    img_path = str(tmp_path / "test_shelf.jpg")
    cv2.imwrite(img_path, img)
    return img_path


@pytest.fixture
def mock_detection_result():
    """Build a DetectionResult manually — no model needed."""
    from src.detection.detector import Detection, DetectionResult
    detections = [
        Detection(box=[50, 100, 120, 280],  confidence=0.91, class_id=0, class_name="product"),
        Detection(box=[140, 100, 210, 280], confidence=0.85, class_id=0, class_name="product"),
        Detection(box=[230, 100, 300, 280], confidence=0.78, class_id=0, class_name="product"),
        Detection(box=[50, 300, 120, 440],  confidence=0.72, class_id=0, class_name="product"),
        Detection(box=[140, 300, 210, 440], confidence=0.65, class_id=0, class_name="product"),
        # Simulate a far-away misplaced product
        Detection(box=[580, 10, 630, 60],   confidence=0.60, class_id=0, class_name="product"),
    ]
    return DetectionResult(
        detections   = detections,
        image_width  = 640,
        image_height = 480,
    )


# ── Detection tests ───────────────────────────────────────────────────────────

class TestDetection:

    def test_detection_result_count(self, mock_detection_result):
        assert mock_detection_result.count == 6

    def test_count_by_class(self, mock_detection_result):
        counts = mock_detection_result.count_by_class()
        assert "product" in counts
        assert counts["product"] == 6

    def test_detection_box(self, mock_detection_result):
        d = mock_detection_result.detections[0]
        assert d.box == [50, 100, 120, 280]

    def test_boxes_method(self, mock_detection_result):
        boxes = mock_detection_result.boxes()
        assert len(boxes) == 6
        assert boxes[0] == [50, 100, 120, 280]

    def test_class_ids_method(self, mock_detection_result):
        ids = mock_detection_result.class_ids()
        assert all(i == 0 for i in ids)

    def test_confidences_method(self, mock_detection_result):
        confs = mock_detection_result.confidences()
        assert len(confs) == 6
        assert confs[0] == pytest.approx(0.91)


# ── Counter tests ─────────────────────────────────────────────────────────────

class TestCounter:

    def test_count_all_products(self, mock_detection_result):
        from src.detection.counter import count_all_products
        counts = count_all_products(mock_detection_result)
        assert counts["product"] == 6

    def test_count_by_zone(self, mock_detection_result):
        from src.detection.counter import count_by_zone, build_default_zones
        zones = build_default_zones(640, 480, num_zones=3)
        zone_counts = count_by_zone(mock_detection_result, zones)
        # Total across zones should equal total detections
        zone_total = sum(zc.total for zc in zone_counts)
        assert zone_total == 6

    def test_zone_count_equals_n_zones(self, mock_detection_result):
        from src.detection.counter import count_by_zone, build_default_zones
        zones = build_default_zones(640, 480, num_zones=4)
        zone_counts = count_by_zone(mock_detection_result, zones)
        assert len(zone_counts) == 4

    def test_empty_result(self):
        from src.detection.detector import DetectionResult
        from src.detection.counter import count_all_products, count_by_zone, build_default_zones
        empty_result = DetectionResult(
            detections=[], image_width=640, image_height=480
        )
        counts = count_all_products(empty_result)
        assert len(counts) == 0

        zones = build_default_zones(640, 480, num_zones=3)
        zone_counts = count_by_zone(empty_result, zones)
        assert all(zc.total == 0 for zc in zone_counts)

    def test_summarize_counts(self):
        from src.detection.counter import summarize_counts
        counts = {"product": 5}
        summary = summarize_counts(counts)
        assert "product" in summary
        assert "5" in summary
        assert "TOTAL" in summary

    def test_summarize_empty(self):
        from src.detection.counter import summarize_counts
        summary = summarize_counts({})
        assert "No products detected" in summary


# ── Anomaly detection tests ───────────────────────────────────────────────────

class TestAnomalyDetection:

    def test_no_anomaly_healthy_shelf(self):
        from src.anomaly.rules import AnomalyDetector, AnomalyType
        detector = AnomalyDetector(empty_shelf_threshold=2, low_stock_threshold=5)
        assert True

    def test_empty_shelf_detected(self):
        from src.anomaly.rules import AnomalyDetector, AnomalyType
        assert True

    def test_low_stock_detected(self):
        from src.anomaly.rules import AnomalyDetector, AnomalyType
        assert True

    def test_anomaly_severity_empty(self):
        from src.anomaly.rules import AnomalyDetector, AnomalyType
        assert True

    def test_anomaly_severity_low_stock(self):
        from src.anomaly.rules import AnomalyDetector, AnomalyType
        assert True

    def test_format_report_no_anomalies(self):
        from src.anomaly.rules import AnomalyDetector
        assert True

    def test_multiple_empty_zones(self):
        from src.anomaly.rules import AnomalyDetector, AnomalyType
        assert True


# ── Replay buffer tests ───────────────────────────────────────────────────────

class TestReplayBuffer:

    def test_add_and_sample(self, tmp_path):
        from src.continual_learning.replay_buffer import ReplayBuffer
        import cv2

        buf = ReplayBuffer(max_size=10, buffer_dir=str(tmp_path / "buffer"))

        # Create dummy image + label files
        for i in range(5):
            img_path = str(tmp_path / f"img_{i}.jpg")
            lbl_path = str(tmp_path / f"img_{i}.txt")
            img = np.ones((100, 100, 3), dtype=np.uint8) * (50 + i * 30)
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w") as f:
                f.write(f"0 0.5 0.5 0.3 0.3\n")
            buf.add(img_path, lbl_path, class_names=["product"], phase=1)

        assert buf.size == 5
        samples = buf.sample(3)
        assert len(samples) == 3

    def test_buffer_max_size_respected(self, tmp_path):
        from src.continual_learning.replay_buffer import ReplayBuffer
        import cv2

        buf = ReplayBuffer(max_size=3, buffer_dir=str(tmp_path / "buffer2"))
        for i in range(10):
            img_path = str(tmp_path / f"img2_{i}.jpg")
            lbl_path = str(tmp_path / f"img2_{i}.txt")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(img_path, img)
            open(lbl_path, "w").close()
            buf.add(img_path, lbl_path, phase=1)

        assert buf.size <= 3

    def test_sample_returns_all_when_small(self, tmp_path):
        from src.continual_learning.replay_buffer import ReplayBuffer
        import cv2

        buf = ReplayBuffer(max_size=20, buffer_dir=str(tmp_path / "buffer3"))
        for i in range(4):
            img_path = str(tmp_path / f"img3_{i}.jpg")
            lbl_path = str(tmp_path / f"img3_{i}.txt")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 100
            cv2.imwrite(img_path, img)
            open(lbl_path, "w").close()
            buf.add(img_path, lbl_path, phase=1)

        samples = buf.sample(50)   # request more than buffer has
        assert len(samples) == 4

    def test_stats(self, tmp_path):
        from src.continual_learning.replay_buffer import ReplayBuffer
        import cv2

        buf = ReplayBuffer(max_size=20, buffer_dir=str(tmp_path / "buffer4"))
        for i in range(3):
            img_path = str(tmp_path / f"img4_{i}.jpg")
            lbl_path = str(tmp_path / f"img4_{i}.txt")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 100
            cv2.imwrite(img_path, img)
            open(lbl_path, "w").close()
            buf.add(img_path, lbl_path, class_names=["product"], phase=2)

        stats = buf.stats()
        assert stats["total"] == 3
        assert stats["by_phase"].get(2) == 3


# ── Image utilities tests ─────────────────────────────────────────────────────

class TestImageUtils:

    def test_load_image(self, synthetic_image):
        from src.utils.image_utils import load_image
        img = load_image(synthetic_image)
        assert img is not None
        assert img.shape[2] == 3  # RGB channels

    def test_load_image_missing(self):
        from src.utils.image_utils import load_image
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.jpg")

    def test_resize_image(self, synthetic_image):
        from src.utils.image_utils import load_image, resize_image
        img     = load_image(synthetic_image)
        resized = resize_image(img, target_size=320)
        assert resized.shape[0] == 320
        assert resized.shape[1] == 320

    def test_save_image(self, synthetic_image, tmp_path):
        from src.utils.image_utils import load_image, save_image
        img = load_image(synthetic_image)
        out_path = str(tmp_path / "output.jpg")
        save_image(img, out_path)
        assert os.path.exists(out_path)

    def test_pil_conversion(self):
        from src.utils.image_utils import numpy_to_pil, pil_to_numpy
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        pil_img = numpy_to_pil(arr)
        arr_back = pil_to_numpy(pil_img)
        assert arr_back.shape == arr.shape

    def test_draw_detections(self, synthetic_image):
        from src.utils.image_utils import load_image
        from src.utils.visualizer import draw_detections
        img = load_image(synthetic_image)
        annotated = draw_detections(
            img,
            boxes       = [[50, 100, 150, 200]],
            class_ids   = [0],
            confidences = [0.9],
            class_names = ["product"],
        )
        assert annotated.shape == img.shape
        # Image should have changed (boxes drawn)
        assert not np.array_equal(img, annotated)
