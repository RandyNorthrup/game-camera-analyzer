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

        with pytest.raises(ValidationError):
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
"""
Comprehensive tests for core/detection_engine.py module.

Tests detection pipeline, result packaging, batch processing, and metadata tracking.
"""

import logging
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from core.detection_engine import DetectionEngine, DetectionResult
from models.yolo_detector import Detection


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    def test_detection_result_initialization(self, tmp_path: Path) -> None:
        """Test DetectionResult basic initialization."""
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [Detection((10, 10, 50, 50), 0.9, 0, "dog")]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        assert result.image_path == image_path
        assert result.image.shape == (480, 640, 3)
        assert len(result.detections) == 1

    def test_detection_result_with_metadata(self, tmp_path: Path) -> None:
        """Test DetectionResult with custom metadata."""
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {"processing_time": 1.23, "model": "yolov8n"}
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=[],
            metadata=metadata,
        )
        
        assert result.metadata["processing_time"] == 1.23
        assert result.metadata["model"] == "yolov8n"

    def test_detection_result_invalid_path_type(self, tmp_path: Path) -> None:
        """Test DetectionResult validation fails for invalid path type."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(TypeError, match="image_path must be Path"):
            DetectionResult(
                image_path="not_a_path",  # type: ignore
                image=image,
                detections=[],
            )

    def test_detection_result_invalid_image_type(self, tmp_path: Path) -> None:
        """Test DetectionResult validation fails for invalid image type."""
        image_path = tmp_path / "test.jpg"
        
        with pytest.raises(TypeError, match="image must be ndarray"):
            DetectionResult(
                image_path=image_path,
                image="not_an_array",  # type: ignore
                detections=[],
            )

    def test_detection_result_invalid_detections_type(self, tmp_path: Path) -> None:
        """Test DetectionResult validation fails for invalid detections type."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(TypeError, match="detections must be list"):
            DetectionResult(
                image_path=image_path,
                image=image,
                detections="not_a_list",  # type: ignore
            )

    def test_detection_result_invalid_detection_item(self, tmp_path: Path) -> None:
        """Test DetectionResult validation fails for invalid detection items."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(TypeError, match="detections\\[0\\] must be Detection"):
            DetectionResult(
                image_path=image_path,
                image=image,
                detections=["not_a_detection"],  # type: ignore
            )

    def test_image_size_property(self, tmp_path: Path) -> None:
        """Test image_size property returns correct dimensions."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=[],
        )
        
        height, width = result.image_size
        assert height == 480
        assert width == 640

    def test_num_detections_property(self, tmp_path: Path) -> None:
        """Test num_detections property returns correct count."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection((10, 10, 50, 50), 0.9, 0, "dog"),
            Detection((100, 100, 150, 150), 0.85, 1, "cat"),
        ]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        assert result.num_detections == 2

    def test_get_detection_count(self, tmp_path: Path) -> None:
        """Test get_detection_count method (legacy)."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [Detection((10, 10, 50, 50), 0.9, 0, "dog")]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        assert result.get_detection_count() == 1

    def test_has_detections_true(self, tmp_path: Path) -> None:
        """Test has_detections returns True when detections exist."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [Detection((10, 10, 50, 50), 0.9, 0, "dog")]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        assert result.has_detections() is True

    def test_has_detections_false(self, tmp_path: Path) -> None:
        """Test has_detections returns False when no detections."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=[],
        )
        
        assert result.has_detections() is False

    def test_get_class_counts(self, tmp_path: Path) -> None:
        """Test get_class_counts returns correct counts per class."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection((10, 10, 50, 50), 0.9, 0, "dog"),
            Detection((100, 100, 150, 150), 0.85, 0, "dog"),
            Detection((200, 200, 250, 250), 0.75, 1, "cat"),
        ]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        counts = result.get_class_counts()
        
        assert counts["dog"] == 2
        assert counts["cat"] == 1

    def test_filter_by_class(self, tmp_path: Path) -> None:
        """Test filtering detections by class name."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection((10, 10, 50, 50), 0.9, 0, "dog"),
            Detection((100, 100, 150, 150), 0.85, 1, "cat"),
            Detection((200, 200, 250, 250), 0.75, 0, "dog"),
        ]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        dogs = result.filter_by_class("dog")
        
        assert len(dogs) == 2
        assert all(d.class_name == "dog" for d in dogs)

    def test_filter_by_confidence(self, tmp_path: Path) -> None:
        """Test filtering detections by confidence threshold."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection((10, 10, 50, 50), 0.9, 0, "dog"),
            Detection((100, 100, 150, 150), 0.85, 1, "cat"),
            Detection((200, 200, 250, 250), 0.6, 0, "dog"),
        ]
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
        )
        
        high_conf = result.filter_by_confidence(0.8)
        
        assert len(high_conf) == 2
        assert all(d.confidence >= 0.8 for d in high_conf)

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test converting DetectionResult to dictionary."""
        image_path = tmp_path / "test.jpg"
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [Detection((10, 10, 50, 50), 0.9, 0, "dog")]
        metadata = {"processing_time": 1.23}
        
        result = DetectionResult(
            image_path=image_path,
            image=image,
            detections=detections,
            metadata=metadata,
        )
        
        data = result.to_dict()
        
        assert data["image_path"] == str(image_path)
        assert data["image_size"] == (480, 640)
        assert data["num_detections"] == 1
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class_name"] == "dog"
        assert data["metadata"]["processing_time"] == 1.23


class TestDetectionEngine:
    """Test suite for DetectionEngine class."""

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_init_default_parameters(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test DetectionEngine initialization with default parameters."""
        engine = DetectionEngine()
        
        assert engine.model_name == "yolov8n.pt"
        assert engine.confidence_threshold == 0.25
        assert engine.iou_threshold == 0.45
        assert engine.max_detections == 20
        assert engine.preprocess_size is None

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_init_custom_parameters(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test DetectionEngine initialization with custom parameters."""
        engine = DetectionEngine(
            model_name="yolov8m.pt",
            confidence_threshold=0.5,
            iou_threshold=0.6,
            max_detections=10,
            preprocess_size=(800, 600),
        )
        
        assert engine.model_name == "yolov8m.pt"
        assert engine.confidence_threshold == 0.5
        assert engine.iou_threshold == 0.6
        assert engine.max_detections == 10
        assert engine.preprocess_size == (800, 600)

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_init_invalid_confidence(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test DetectionEngine fails with invalid confidence."""
        with pytest.raises(Exception):  # Will raise from validate_confidence_threshold
            DetectionEngine(confidence_threshold=1.5)

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_init_invalid_max_detections(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test DetectionEngine fails with invalid max_detections."""
        with pytest.raises(ValueError, match="max_detections must be >= 1"):
            DetectionEngine(max_detections=0)

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_init_invalid_preprocess_size_length(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test DetectionEngine fails with invalid preprocess_size length."""
        with pytest.raises(ValueError, match="preprocess_size must be"):
            DetectionEngine(preprocess_size=(640,))  # type: ignore

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_init_invalid_preprocess_dimensions(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test DetectionEngine fails with invalid preprocess dimensions."""
        with pytest.raises(ValueError, match="preprocess_size dimensions must be >= 1"):
            DetectionEngine(preprocess_size=(0, 640))

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.resize_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_image_basic(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_resize: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test basic image processing."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        mock_validate.return_value = (image_path, (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        detection = Detection((10, 10, 50, 50), 0.9, 0, "dog")
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([detection], None)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine()
        result = engine.process_image(image_path)
        
        # Verify
        assert result.image_path == image_path
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "dog"
        assert "processing_time_seconds" in result.metadata

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.resize_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_image_with_preprocessing(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_resize: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test image processing with preprocessing."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        mock_validate.return_value = (image_path, (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        resized = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        mock_resize.return_value = resized
        
        detection = Detection((10, 10, 50, 50), 0.9, 0, "dog")
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([detection], None)
        mock_detector_cls.return_value = mock_detector
        
        # Execute with preprocessing
        engine = DetectionEngine(preprocess_size=(640, 640))
        result = engine.process_image(image_path)
        
        # Verify preprocessing was called
        mock_resize.assert_called_once()
        assert result.metadata["preprocessing_applied"] is True

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.resize_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_image_with_annotated(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_resize: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test image processing with annotated output."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        mock_validate.return_value = (image_path, (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        annotated = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], annotated)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine()
        result = engine.process_image(image_path, return_annotated=True)
        
        # Verify annotated image returned
        assert result.annotated_image is not None

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.resize_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_image_save_annotated(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_resize: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test image processing with saving annotated image."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        save_path = tmp_path / "annotated.jpg"
        mock_validate.return_value = (image_path, (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        annotated = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], annotated)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine()
        result = engine.process_image(image_path, save_annotated=save_path)
        
        # Verify save was called
        mock_save.assert_called_once()

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_image_handles_errors(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test image processing handles errors gracefully."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        mock_validate.return_value = (image_path, (640, 480))
        
        mock_load.side_effect = RuntimeError("Load failed")
        
        # Execute and verify error
        engine = DetectionEngine()
        with pytest.raises(RuntimeError, match="Image processing failed"):
            engine.process_image(image_path)

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_batch_success(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test batch processing multiple images."""
        # Setup
        image_paths = [tmp_path / f"test{i}.jpg" for i in range(3)]
        for path in image_paths:
            path.touch()
        mock_validate.return_value = (image_paths[0], (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], None)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine()
        results = engine.process_batch(image_paths)
        
        # Verify
        assert len(results) == 3

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_batch_empty_list(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test batch processing fails with empty list."""
        engine = DetectionEngine()
        
        with pytest.raises(ValueError, match="image_paths cannot be empty"):
            engine.process_batch([])

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_process_batch_handles_errors(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test batch processing continues after individual errors."""
        # Setup
        image_paths = [tmp_path / f"test{i}.jpg" for i in range(3)]
        for path in image_paths:
            path.touch()
        mock_validate.return_value = (image_paths[0], (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # First call fails, second succeeds, third fails
        mock_load.side_effect = [
            RuntimeError("Load failed"),
            (image, {"format": "JPEG"}),
            RuntimeError("Load failed"),
        ]
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], None)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine()
        results = engine.process_batch(image_paths)
        
        # Verify only successful result returned
        assert len(results) == 1

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_detect_alias(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test detect() method is alias for process_image()."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        mock_validate.return_value = (image_path, (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], None)
        mock_detector_cls.return_value = mock_detector
        
        # Execute both methods
        engine = DetectionEngine()
        result1 = engine.detect(image_path)
        result2 = engine.process_image(image_path)
        
        # Verify same behavior
        assert result1.image_path == result2.image_path

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_detect_batch_alias(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test detect_batch() method is alias for process_batch()."""
        # Setup
        image_paths = [tmp_path / "test.jpg"]
        image_paths[0].touch()
        mock_validate.return_value = (image_paths[0], (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], None)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine()
        results = engine.detect_batch(image_paths)
        
        # Verify
        assert len(results) == 1

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_update_thresholds(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test updating detection thresholds."""
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        
        engine = DetectionEngine()
        engine.update_thresholds(confidence=0.5, iou=0.6, max_det=15)
        
        assert engine.confidence_threshold == 0.5
        assert engine.iou_threshold == 0.6
        assert engine.max_detections == 15
        mock_detector.update_thresholds.assert_called_once()

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_update_thresholds_partial(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test updating only some thresholds."""
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        
        engine = DetectionEngine(confidence_threshold=0.3, iou_threshold=0.5)
        engine.update_thresholds(confidence=0.7)
        
        assert engine.confidence_threshold == 0.7
        assert engine.iou_threshold == 0.5  # Unchanged

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_update_thresholds_invalid_max_det(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test updating thresholds with invalid max_det."""
        engine = DetectionEngine()
        
        with pytest.raises(ValueError, match="max_det must be >= 1"):
            engine.update_thresholds(max_det=0)

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_get_model_info(
        self, mock_manager_cls: Mock, mock_detector_cls: Mock
    ) -> None:
        """Test getting model information."""
        mock_detector = MagicMock()
        mock_detector.get_model_info.return_value = {"model": "yolov8n", "device": "cpu"}
        mock_detector_cls.return_value = mock_detector
        
        engine = DetectionEngine()
        info = engine.get_model_info()
        
        assert info["model"] == "yolov8n"
        assert info["device"] == "cpu"

    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_repr(self, mock_manager_cls: Mock, mock_detector_cls: Mock) -> None:
        """Test string representation."""
        engine = DetectionEngine(
            model_name="yolov8m.pt",
            confidence_threshold=0.5,
            iou_threshold=0.6,
        )
        
        repr_str = repr(engine)
        
        assert "DetectionEngine" in repr_str
        assert "yolov8m.pt" in repr_str
        assert "0.5" in repr_str


class TestIntegration:
    """Integration tests for detection engine."""

    @patch("core.detection_engine.validate_image_readable")
    @patch("core.detection_engine.save_image")
    @patch("core.detection_engine.load_image")
    @patch("core.detection_engine.YOLODetector")
    @patch("core.detection_engine.ModelManager")
    def test_full_detection_workflow(
        self,
        mock_manager_cls: Mock,
        mock_detector_cls: Mock,
        mock_load: Mock,
        mock_save: Mock,
        mock_validate: Mock,
        tmp_path: Path,
    ) -> None:
        """Test complete detection workflow with multiple detections."""
        # Setup
        image_path = tmp_path / "test.jpg"
        image_path.touch()
        mock_validate.return_value = (image_path, (640, 480))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load.return_value = (image, {"format": "JPEG"})
        
        detections = [
            Detection((10, 10, 50, 50), 0.9, 0, "dog"),
            Detection((100, 100, 150, 150), 0.85, 1, "cat"),
            Detection((200, 200, 250, 250), 0.75, 0, "dog"),
        ]
        annotated = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = (detections, annotated)
        mock_detector_cls.return_value = mock_detector
        
        # Execute
        engine = DetectionEngine(confidence_threshold=0.3)
        result = engine.process_image(image_path, return_annotated=True)
        
        # Verify complete result
        assert result.num_detections == 3
        assert result.has_detections()
        
        # Test filtering
        dogs = result.filter_by_class("dog")
        assert len(dogs) == 2
        
        high_conf = result.filter_by_confidence(0.85)
        assert len(high_conf) == 2
        
        # Test class counts
        counts = result.get_class_counts()
        assert counts["dog"] == 2
        assert counts["cat"] == 1
        
        # Test dict conversion
        data = result.to_dict()
        assert data["num_detections"] == 3
        assert len(data["detections"]) == 3


class TestBatchDetectWithSaveAnnotated:
    """Test batch detection with annotated image saving."""

    def test_batch_detect_with_save_annotated_dir(
        self, sample_images: List[Path], tmp_path: Path
    ) -> None:
        """
        Test batch detection with saving annotated images to directory.

        Args:
            sample_images: List of test image paths
            tmp_path: Pytest temporary directory fixture
        """
        engine = DetectionEngine(confidence_threshold=0.1)
        output_dir = tmp_path / "annotated_batch"

        # Execute batch detection with save directory
        results = engine.detect_batch(
            sample_images, save_annotated_dir=str(output_dir), return_annotated=True
        )

        # Verify results
        assert results is not None
        assert len(results) == len(sample_images)
        assert output_dir.exists()
        assert output_dir.is_dir()

        # Verify annotated images were saved
        saved_files = list(output_dir.glob("annotated_*.jpg"))
        assert len(saved_files) == len(sample_images)

        # Verify each result has annotated image
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.annotated_image is not None
            assert isinstance(result.annotated_image, np.ndarray)

        logger.info(
            f"Batch detection saved {len(saved_files)} annotated images to {output_dir}"
        )

    def test_batch_detect_save_annotated_creates_directory(
        self, sample_images: List[Path], tmp_path: Path
    ) -> None:
        """
        Test that batch detection creates save directory if it doesn't exist.

        Args:
            sample_images: List of test image paths
            tmp_path: Pytest temporary directory fixture
        """
        engine = DetectionEngine(confidence_threshold=0.1)
        output_dir = tmp_path / "nested" / "annotated" / "output"

        # Directory shouldn't exist yet
        assert not output_dir.exists()

        # Execute - should create directory
        results = engine.detect_batch(
            sample_images, save_annotated_dir=str(output_dir), return_annotated=False
        )

        # Verify directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()
        assert len(results) == len(sample_images)

        # Verify files were saved
        saved_files = list(output_dir.glob("annotated_*.jpg"))
        assert len(saved_files) == len(sample_images)

    def test_batch_detect_without_save_annotated(
        self, sample_images: List[Path]
    ) -> None:
        """
        Test batch detection without saving annotated images.

        Args:
            sample_images: List of test image paths
        """
        engine = DetectionEngine(confidence_threshold=0.1)

        # Execute without save_annotated_dir
        results = engine.detect_batch(sample_images, return_annotated=False)

        # Verify results
        assert results is not None
        assert len(results) == len(sample_images)

        # Verify no annotated images in results
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.annotated_image is None
