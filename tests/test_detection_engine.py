"""
Unit tests for the detection engine module.

Tests cover:
- Detection engine initialization
- Single image detection
- Batch detection
- Error handling
- Configuration parameters
"""

import logging
from pathlib import Path
from typing import List

import pytest

from core.detection_engine import DetectionEngine, DetectionResult
from models.model_manager import ModelManager
from utils.validators import ValidationError

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestDetectionEngine:
    """Test suite for DetectionEngine class."""

    def test_init_default_params(self) -> None:
        """Test detection engine initialization with default parameters."""
        engine = DetectionEngine()

        assert engine is not None
        assert engine.model_manager is not None
        assert engine.confidence_threshold == 0.25
        assert engine.iou_threshold == 0.45

        logger.info("Detection engine initialized with defaults")

    def test_init_custom_params(self) -> None:
        """Test detection engine initialization with custom parameters."""
        engine = DetectionEngine(
            confidence_threshold=0.5,
            iou_threshold=0.6,
            model_name="yolov8n.pt",
        )

        assert engine.confidence_threshold == 0.5
        assert engine.iou_threshold == 0.6

        logger.info("Detection engine initialized with custom params")

    def test_invalid_confidence_threshold(self) -> None:
        """Test that invalid confidence threshold raises error."""
        with pytest.raises(ValidationError, match="Confidence threshold must be between"):
            DetectionEngine(confidence_threshold=1.5)

        with pytest.raises(ValidationError, match="Confidence threshold must be between"):
            DetectionEngine(confidence_threshold=-0.1)

    def test_invalid_iou_threshold(self) -> None:
        """Test that invalid IoU threshold raises error."""
        with pytest.raises(ValidationError, match="IoU threshold must be between"):
            DetectionEngine(iou_threshold=1.5)

        with pytest.raises(ValidationError, match="IoU threshold must be between"):
            DetectionEngine(iou_threshold=-0.1)

    @pytest.mark.slow
    def test_detect_single_image(self, sample_image: Path) -> None:
        """
        Test detection on single image.

        Args:
            sample_image: Path to test image
        """
        engine = DetectionEngine(confidence_threshold=0.1)  # Low threshold for test
        result = engine.detect(sample_image)

        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.image_path == sample_image
        assert result.image_size is not None
        assert len(result.image_size) == 2
        assert result.num_detections >= 0
        assert len(result.detections) == result.num_detections

        logger.info(
            f"Detected {result.num_detections} objects in {sample_image.name}"
        )

    @pytest.mark.slow
    def test_detect_empty_image(self, empty_image: Path) -> None:
        """
        Test detection on image with no objects.

        Args:
            empty_image: Path to blank image
        """
        engine = DetectionEngine()
        result = engine.detect(empty_image)

        assert result is not None
        assert result.num_detections >= 0  # May or may not detect anything
        assert len(result.detections) == result.num_detections

        logger.info(f"Detected {result.num_detections} objects in blank image")

    @pytest.mark.slow
    def test_batch_detect(self, sample_images: List[Path]) -> None:
        """
        Test batch detection on multiple images.

        Args:
            sample_images: List of test image paths
        """
        engine = DetectionEngine(confidence_threshold=0.1)
        results = engine.detect_batch(sample_images)

        assert results is not None
        assert len(results) == len(sample_images)

        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.num_detections >= 0
            assert len(result.detections) == result.num_detections

        total_detections = sum(r.num_detections for r in results)
        logger.info(
            f"Batch detected {total_detections} objects across {len(sample_images)} images"
        )

    def test_detect_invalid_path(self, invalid_image_path: Path) -> None:
        """
        Test detection with non-existent image path.

        Args:
            invalid_image_path: Path that doesn't exist
        """
        engine = DetectionEngine()

        with pytest.raises(FileNotFoundError):
            engine.detect(invalid_image_path)

    def test_detect_corrupted_image(self, corrupted_image: Path) -> None:
        """
        Test detection with corrupted image file.

        Args:
            corrupted_image: Path to corrupted file
        """
        engine = DetectionEngine()

        with pytest.raises(Exception):  # Should raise some kind of error
            engine.detect(corrupted_image)

    @pytest.mark.slow
    def test_detection_result_properties(self, sample_image: Path) -> None:
        """
        Test DetectionResult properties and methods.

        Args:
            sample_image: Path to test image
        """
        engine = DetectionEngine(confidence_threshold=0.1)
        result = engine.detect(sample_image)

        # Test dictionary conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "image_path" in result_dict
        assert "num_detections" in result_dict
        assert "detections" in result_dict

        # Test filtering
        if result.num_detections > 0:
            high_conf = result.filter_by_confidence(0.8)
            assert high_conf.num_detections <= result.num_detections

            # Test class filtering
            if result.detections:
                first_class = result.detections[0].class_name
                filtered = result.filter_by_class(first_class)
                assert all(d.class_name == first_class for d in filtered.detections)

    def test_confidence_threshold_filtering(self, sample_image: Path) -> None:
        """
        Test that confidence threshold properly filters detections.

        Args:
            sample_image: Path to test image
        """
        # Detect with low threshold
        engine_low = DetectionEngine(confidence_threshold=0.1)
        result_low = engine_low.detect(sample_image)

        # Detect with high threshold
        engine_high = DetectionEngine(confidence_threshold=0.8)
        result_high = engine_high.detect(sample_image)

        # High threshold should have fewer or equal detections
        assert result_high.num_detections <= result_low.num_detections

        logger.info(
            f"Low threshold: {result_low.num_detections}, "
            f"High threshold: {result_high.num_detections}"
        )

    @pytest.mark.slow
    def test_detection_consistency(self, sample_image: Path) -> None:
        """
        Test that detection is consistent across multiple runs.

        Args:
            sample_image: Path to test image
        """
        engine = DetectionEngine(confidence_threshold=0.25)

        # Run detection twice
        result1 = engine.detect(sample_image)
        result2 = engine.detect(sample_image)

        # Should get same number of detections
        assert result1.num_detections == result2.num_detections

        # Check that detection counts are identical
        if result1.num_detections > 0:
            # Boxes may have slight numerical differences, but counts should match
            assert len(result1.detections) == len(result2.detections)

        logger.info(f"Consistent detections: {result1.num_detections}")


@pytest.mark.unit
class TestModelManager:
    """Test suite for ModelManager class."""

    def test_model_manager_init(self) -> None:
        """Test model manager initialization."""
        manager = ModelManager()

        assert manager is not None
        assert manager.device in ["cpu", "cuda", "mps"]

        logger.info(f"ModelManager initialized with device: {manager.device}")

    def test_device_selection(self) -> None:
        """Test automatic device selection."""
        manager = ModelManager()

        # Should select a valid device
        assert manager.device in ["cpu", "cuda", "mps"]

        # CPU should always be available as fallback
        if manager.device == "cpu":
            logger.info("Using CPU (expected on non-GPU systems)")
        else:
            logger.info(f"Using accelerated device: {manager.device}")

    @pytest.mark.slow
    def test_load_yolo_model(self, yolo_model_path: Path) -> None:
        """
        Test loading YOLO model.

        Args:
            yolo_model_path: Path to YOLO model file
        """
        manager = ModelManager()
        model = manager.load_yolo_model(str(yolo_model_path))

        assert model is not None
        logger.info(f"Successfully loaded model: {yolo_model_path}")

    def test_load_invalid_model(self) -> None:
        """Test loading non-existent model file."""
        manager = ModelManager()

        with pytest.raises(Exception):  # Should raise error for invalid path
            manager.load_yolo_model("nonexistent_model.pt")
