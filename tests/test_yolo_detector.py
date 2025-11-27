"""
Comprehensive tests for models/yolo_detector.py.

Tests cover:
- Detection dataclass validation
- YOLODetector initialization
- Single image detection
- Batch detection
- Threshold updates
- Error handling
"""

import logging
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from models.model_manager import ModelManager
from models.yolo_detector import Detection, YOLODetector

logger = logging.getLogger(__name__)


class TestDetectionDataclass:
    """Test Detection dataclass functionality."""

    def test_detection_valid(self) -> None:
        """Test creating valid detection."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=16,
            class_name="dog",
        )

        assert det.bbox == (100, 100, 200, 200)
        assert det.confidence == 0.85
        assert det.class_id == 16
        assert det.class_name == "dog"

    def test_detection_invalid_confidence_too_high(self) -> None:
        """Test detection with confidence > 1.0 raises error."""
        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=1.5,
                class_id=16,
                class_name="dog",
            )

    def test_detection_invalid_confidence_negative(self) -> None:
        """Test detection with negative confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=-0.1,
                class_id=16,
                class_name="dog",
            )

    def test_detection_invalid_bbox_x(self) -> None:
        """Test detection with invalid bbox (x2 <= x1) raises error."""
        with pytest.raises(ValueError, match="Invalid bbox"):
            Detection(
                bbox=(200, 100, 100, 200),  # x2 < x1
                confidence=0.85,
                class_id=16,
                class_name="dog",
            )

    def test_detection_invalid_bbox_y(self) -> None:
        """Test detection with invalid bbox (y2 <= y1) raises error."""
        with pytest.raises(ValueError, match="Invalid bbox"):
            Detection(
                bbox=(100, 200, 200, 100),  # y2 < y1
                confidence=0.85,
                class_id=16,
                class_name="dog",
            )

    def test_detection_to_dict(self) -> None:
        """Test converting detection to dictionary."""
        det = Detection(
            bbox=(100, 150, 300, 350),
            confidence=0.92,
            class_id=17,
            class_name="cat",
        )

        data = det.to_dict()

        assert data["bbox"] == (100, 150, 300, 350)
        assert data["confidence"] == 0.92
        assert data["class_id"] == 17
        assert data["class_name"] == "cat"
        assert data["width"] == 200
        assert data["height"] == 200

    def test_detection_get_area(self) -> None:
        """Test calculating detection area."""
        det = Detection(
            bbox=(0, 0, 100, 50),
            confidence=0.75,
            class_id=16,
            class_name="dog",
        )

        assert det.get_area() == 5000

    def test_detection_get_center(self) -> None:
        """Test calculating detection center."""
        det = Detection(
            bbox=(100, 100, 300, 200),
            confidence=0.75,
            class_id=16,
            class_name="dog",
        )

        center = det.get_center()
        assert center == (200, 150)

    def test_detection_zero_area_valid(self) -> None:
        """Test detection with minimum valid bbox."""
        det = Detection(
            bbox=(100, 100, 101, 101),
            confidence=0.5,
            class_id=16,
            class_name="dog",
        )

        assert det.get_area() == 1


class TestYOLODetectorInitialization:
    """Test YOLODetector initialization."""

    @patch("models.yolo_detector.ModelManager")
    def test_init_default_params(self, mock_manager_cls: Mock) -> None:
        """Test initialization with default parameters."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)

        assert detector.model_manager == mock_manager
        assert detector.model_name == "yolov8m.pt"
        assert detector.confidence_threshold == 0.25
        assert detector.iou_threshold == 0.45
        assert detector.max_detections == 20
        mock_manager.load_yolo_model.assert_called_once_with("yolov8m.pt")

    @patch("models.yolo_detector.ModelManager")
    def test_init_custom_params(self, mock_manager_cls: Mock) -> None:
        """Test initialization with custom parameters."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(
            mock_manager,
            model_name="yolov8n.pt",
            confidence_threshold=0.5,
            iou_threshold=0.6,
            max_detections=10,
        )

        assert detector.model_name == "yolov8n.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.6
        assert detector.max_detections == 10
        mock_manager.load_yolo_model.assert_called_once_with("yolov8n.pt")

    @patch("models.yolo_detector.ModelManager")
    def test_init_invalid_confidence(self, mock_manager_cls: Mock) -> None:
        """Test initialization with invalid confidence threshold."""
        mock_manager = MagicMock()

        with pytest.raises(ValueError, match="confidence_threshold must be 0-1"):
            YOLODetector(mock_manager, confidence_threshold=1.5)

    @patch("models.yolo_detector.ModelManager")
    def test_init_invalid_iou(self, mock_manager_cls: Mock) -> None:
        """Test initialization with invalid IoU threshold."""
        mock_manager = MagicMock()

        with pytest.raises(ValueError, match="iou_threshold must be 0-1"):
            YOLODetector(mock_manager, iou_threshold=-0.1)

    @patch("models.yolo_detector.ModelManager")
    def test_init_invalid_max_detections(self, mock_manager_cls: Mock) -> None:
        """Test initialization with invalid max detections."""
        mock_manager = MagicMock()

        with pytest.raises(ValueError, match="max_detections must be >= 1"):
            YOLODetector(mock_manager, max_detections=0)


class TestYOLODetectorDetection:
    """Test single image detection."""

    @patch("models.yolo_detector.ModelManager")
    def test_detect_with_results(self, mock_manager_cls: Mock) -> None:
        """Test detection with valid results."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        # Mock detection results - xyxy needs to be tensor-like with cpu() method
        mock_tensor = MagicMock()
        mock_tensor.cpu().numpy().astype.return_value = np.array([100, 100, 200, 200], dtype=int)
        
        mock_box1 = MagicMock()
        mock_box1.xyxy = [mock_tensor]
        mock_box1.conf = [0.85]
        mock_box1.cls = [16]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1]
        mock_model.return_value = [mock_result]
        mock_model.names = {16: "dog"}

        detector = YOLODetector(mock_manager)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections, annotated = detector.detect(image, return_annotated=False)

        assert len(detections) == 1
        assert detections[0].class_name == "dog"
        assert detections[0].confidence == 0.85
        assert detections[0].bbox == (100, 100, 200, 200)
        assert annotated is None

    @patch("models.yolo_detector.ModelManager")
    def test_detect_with_annotated(self, mock_manager_cls: Mock) -> None:
        """Test detection with annotated image."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = []
        annotated_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_result.plot.return_value = annotated_img
        mock_model.return_value = [mock_result]

        detector = YOLODetector(mock_manager)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections, annotated = detector.detect(image, return_annotated=True)

        assert len(detections) == 0
        assert annotated is not None
        mock_result.plot.assert_called_once()

    @patch("models.yolo_detector.ModelManager")
    def test_detect_no_results(self, mock_manager_cls: Mock) -> None:
        """Test detection with no results."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]

        detector = YOLODetector(mock_manager)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections, _ = detector.detect(image)

        assert len(detections) == 0

    @patch("models.yolo_detector.ModelManager")
    def test_detect_invalid_image_none(self, mock_manager_cls: Mock) -> None:
        """Test detection with None image."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)

        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(None)  # type: ignore[arg-type]

    @patch("models.yolo_detector.ModelManager")
    def test_detect_invalid_image_empty(self, mock_manager_cls: Mock) -> None:
        """Test detection with empty image."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(empty_image)

    @patch("models.yolo_detector.ModelManager")
    def test_detect_custom_threshold(self, mock_manager_cls: Mock) -> None:
        """Test detection with custom confidence threshold."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        detector = YOLODetector(mock_manager, confidence_threshold=0.25)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detector.detect(image, conf_threshold=0.75)

        # Verify model called with custom threshold
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["conf"] == 0.75

    @patch("models.yolo_detector.ModelManager")
    def test_detect_invalid_detection_skipped(self, mock_manager_cls: Mock) -> None:
        """Test that invalid detections are skipped with warning."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        # Mock invalid detection (x2 < x1)
        mock_tensor1 = MagicMock()
        mock_tensor1.cpu().numpy().astype.return_value = np.array([200, 100, 100, 200], dtype=int)
        mock_box1 = MagicMock()
        mock_box1.xyxy = [mock_tensor1]
        mock_box1.conf = [0.85]
        mock_box1.cls = [16]

        # Mock valid detection
        mock_tensor2 = MagicMock()
        mock_tensor2.cpu().numpy().astype.return_value = np.array([100, 100, 200, 200], dtype=int)
        mock_box2 = MagicMock()
        mock_box2.xyxy = [mock_tensor2]
        mock_box2.conf = [0.90]
        mock_box2.cls = [17]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_model.return_value = [mock_result]
        mock_model.names = {16: "dog", 17: "cat"}

        detector = YOLODetector(mock_manager)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections, _ = detector.detect(image)

        # Only valid detection should be returned
        assert len(detections) == 1
        assert detections[0].class_name == "cat"


class TestYOLODetectorBatchDetection:
    """Test batch detection."""

    @patch("models.yolo_detector.ModelManager")
    def test_detect_batch_multiple_images(self, mock_manager_cls: Mock) -> None:
        """Test batch detection on multiple images."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        # Mock results for 3 images
        mock_results = []
        for i in range(3):
            mock_tensor = MagicMock()
            mock_tensor.cpu().numpy().astype.return_value = np.array([100, 100, 200, 200], dtype=int)
            mock_box = MagicMock()
            mock_box.xyxy = [mock_tensor]
            mock_box.conf = [0.85 + i * 0.05]
            mock_box.cls = [16]

            mock_result = MagicMock()
            mock_result.boxes = [mock_box]
            mock_results.append(mock_result)

        mock_model.return_value = mock_results
        mock_model.names = {16: "dog"}

        detector = YOLODetector(mock_manager)
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        all_detections = detector.detect_batch(images)

        assert len(all_detections) == 3
        for detections in all_detections:
            assert len(detections) == 1
            assert detections[0].class_name == "dog"

    @patch("models.yolo_detector.ModelManager")
    def test_detect_batch_empty_list(self, mock_manager_cls: Mock) -> None:
        """Test batch detection with empty image list."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)

        with pytest.raises(ValueError, match="Images list cannot be empty"):
            detector.detect_batch([])

    @patch("models.yolo_detector.ModelManager")
    def test_detect_batch_custom_threshold(self, mock_manager_cls: Mock) -> None:
        """Test batch detection with custom threshold."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        detector = YOLODetector(mock_manager, confidence_threshold=0.25)
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]

        detector.detect_batch(images, conf_threshold=0.8)

        # Verify model called with custom threshold
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["conf"] == 0.8


class TestYOLODetectorThresholdUpdates:
    """Test threshold update functionality."""

    @patch("models.yolo_detector.ModelManager")
    def test_update_confidence_threshold(self, mock_manager_cls: Mock) -> None:
        """Test updating confidence threshold."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager, confidence_threshold=0.25)
        detector.update_thresholds(confidence=0.75)

        assert detector.confidence_threshold == 0.75

    @patch("models.yolo_detector.ModelManager")
    def test_update_iou_threshold(self, mock_manager_cls: Mock) -> None:
        """Test updating IoU threshold."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager, iou_threshold=0.45)
        detector.update_thresholds(iou=0.6)

        assert detector.iou_threshold == 0.6

    @patch("models.yolo_detector.ModelManager")
    def test_update_max_detections(self, mock_manager_cls: Mock) -> None:
        """Test updating max detections."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager, max_detections=20)
        detector.update_thresholds(max_det=50)

        assert detector.max_detections == 50

    @patch("models.yolo_detector.ModelManager")
    def test_update_all_thresholds(self, mock_manager_cls: Mock) -> None:
        """Test updating all thresholds at once."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)
        detector.update_thresholds(confidence=0.5, iou=0.55, max_det=15)

        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.55
        assert detector.max_detections == 15

    @patch("models.yolo_detector.ModelManager")
    def test_update_invalid_confidence(self, mock_manager_cls: Mock) -> None:
        """Test updating with invalid confidence threshold."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)

        with pytest.raises(ValueError, match="confidence must be 0-1"):
            detector.update_thresholds(confidence=1.5)

    @patch("models.yolo_detector.ModelManager")
    def test_update_invalid_iou(self, mock_manager_cls: Mock) -> None:
        """Test updating with invalid IoU threshold."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)

        with pytest.raises(ValueError, match="iou must be 0-1"):
            detector.update_thresholds(iou=-0.1)

    @patch("models.yolo_detector.ModelManager")
    def test_update_invalid_max_det(self, mock_manager_cls: Mock) -> None:
        """Test updating with invalid max detections."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(mock_manager)

        with pytest.raises(ValueError, match="max_det must be >= 1"):
            detector.update_thresholds(max_det=0)

    @patch("models.yolo_detector.ModelManager")
    def test_update_none_values_no_change(self, mock_manager_cls: Mock) -> None:
        """Test that None values don't change thresholds."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model

        detector = YOLODetector(
            mock_manager, confidence_threshold=0.3, iou_threshold=0.5, max_detections=25
        )

        detector.update_thresholds(confidence=None, iou=None, max_det=None)

        assert detector.confidence_threshold == 0.3
        assert detector.iou_threshold == 0.5
        assert detector.max_detections == 25

    @patch("models.yolo_detector.ModelManager")
    def test_detect_exception_handling(self, mock_manager_cls: Mock) -> None:
        """Test detect method exception handling."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model
        
        detector = YOLODetector(mock_manager)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock model to raise exception
        mock_model.side_effect = RuntimeError("Model error")
        
        with pytest.raises(RuntimeError, match="Model error"):
            detector.detect(img)

    @patch("models.yolo_detector.ModelManager")
    def test_detect_batch_exception_handling(self, mock_manager_cls: Mock) -> None:
        """Test detect_batch method exception handling."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model
        
        detector = YOLODetector(mock_manager)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock model to raise exception
        mock_model.side_effect = RuntimeError("Batch error")
        
        with pytest.raises(RuntimeError, match="Batch error"):
            detector.detect_batch([img, img])

    @patch("models.yolo_detector.ModelManager")
    def test_get_model_info(self, mock_manager_cls: Mock) -> None:
        """Test get_model_info returns correct information."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_model.names = {0: "cat", 1: "dog", 2: "bird"}
        mock_model.model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.model.parameters.return_value = iter([mock_param])
        mock_manager.load_yolo_model.return_value = mock_model
        
        detector = YOLODetector(mock_manager, model_name="yolov8n.pt")
        
        info = detector.get_model_info()
        
        assert info["model_name"] == "yolov8n.pt"
        assert "confidence_threshold" in info
        assert "iou_threshold" in info
        assert "max_detections" in info
        assert "device" in info
        assert "num_classes" in info
        assert "class_names" in info
        assert isinstance(info["class_names"], list)

    @patch("models.yolo_detector.ModelManager")
    def test_repr(self, mock_manager_cls: Mock) -> None:
        """Test YOLODetector __repr__ method."""
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_yolo_model.return_value = mock_model
        
        detector = YOLODetector(
            mock_manager,
            model_name="yolov8n.pt",
            confidence_threshold=0.35,
            iou_threshold=0.55
        )
        
        repr_str = repr(detector)
        
        assert "YOLODetector" in repr_str
        assert "yolov8n.pt" in repr_str
        assert "0.35" in repr_str
        assert "0.55" in repr_str
