"""
Comprehensive tests for core.batch_processor module.

Tests BatchConfig, BatchProgress, and BatchProcessor with all pipeline stages.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from core.batch_processor import (
    BatchConfig,
    BatchProcessor,
    BatchProcessingError,
    BatchProgress,
)
from core.classification_engine import ClassificationResult
from core.cropping_engine import CropResult
from core.detection_engine import Detection, DetectionResult


# ================================================================================
# Test BatchConfig
# ================================================================================


class TestBatchConfig:
    """Test suite for BatchConfig dataclass."""

    def test_batch_config_defaults(self) -> None:
        """Test default configuration values."""
        config = BatchConfig()

        assert config.detect is True
        assert config.classify is True
        assert config.crop is True
        assert config.export_csv is True
        assert config.save_annotated is False
        assert config.continue_on_error is True
        assert config.max_workers == 1

    def test_batch_config_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BatchConfig(
            detect=False,
            classify=False,
            crop=False,
            export_csv=False,
            save_annotated=True,
            continue_on_error=False,
            max_workers=4,
        )

        assert config.detect is False
        assert config.classify is False
        assert config.crop is False
        assert config.export_csv is False
        assert config.save_annotated is True
        assert config.continue_on_error is False
        assert config.max_workers == 4

    def test_batch_config_invalid_max_workers_zero(self) -> None:
        """Test that max_workers=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            BatchConfig(max_workers=0)

    def test_batch_config_invalid_max_workers_negative(self) -> None:
        """Test that negative max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            BatchConfig(max_workers=-1)


# ================================================================================
# Test BatchProgress
# ================================================================================


class TestBatchProgress:
    """Test suite for BatchProgress dataclass."""

    def test_batch_progress_defaults(self) -> None:
        """Test default progress values."""
        progress = BatchProgress()

        assert progress.total_images == 0
        assert progress.processed_images == 0
        assert progress.successful_images == 0
        assert progress.failed_images == 0
        assert progress.total_detections == 0
        assert progress.total_classifications == 0
        assert progress.total_crops == 0
        assert progress.current_image == ""
        assert progress.errors == []
        assert isinstance(progress.start_time, float)

    def test_batch_progress_custom_values(self) -> None:
        """Test custom progress values."""
        progress = BatchProgress(
            total_images=10,
            processed_images=5,
            successful_images=4,
            failed_images=1,
            total_detections=15,
            total_classifications=12,
            total_crops=12,
            current_image="test.jpg",
        )

        assert progress.total_images == 10
        assert progress.processed_images == 5
        assert progress.successful_images == 4
        assert progress.failed_images == 1
        assert progress.total_detections == 15
        assert progress.total_classifications == 12
        assert progress.total_crops == 12
        assert progress.current_image == "test.jpg"

    def test_get_progress_percent_zero_total(self) -> None:
        """Test progress percentage with zero total images."""
        progress = BatchProgress(total_images=0, processed_images=0)

        assert progress.get_progress_percent() == 0.0

    def test_get_progress_percent_half_complete(self) -> None:
        """Test progress percentage halfway through."""
        progress = BatchProgress(total_images=10, processed_images=5)

        assert progress.get_progress_percent() == 50.0

    def test_get_progress_percent_complete(self) -> None:
        """Test progress percentage when complete."""
        progress = BatchProgress(total_images=10, processed_images=10)

        assert progress.get_progress_percent() == 100.0

    def test_get_elapsed_time(self) -> None:
        """Test elapsed time calculation."""
        start = time.time()
        progress = BatchProgress()
        progress.start_time = start

        time.sleep(0.1)  # Sleep 100ms

        elapsed = progress.get_elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 0.2  # Should be close to 0.1s

    def test_get_estimated_remaining_none_when_no_progress(self) -> None:
        """Test estimated remaining time returns None with no progress."""
        progress = BatchProgress(total_images=10, processed_images=0)

        assert progress.get_estimated_remaining() is None

    def test_get_estimated_remaining_with_progress(self) -> None:
        """Test estimated remaining time calculation."""
        progress = BatchProgress(total_images=10, processed_images=5)
        progress.start_time = time.time() - 5.0  # 5 seconds elapsed

        estimated = progress.get_estimated_remaining()

        assert estimated is not None
        assert abs(estimated - 5.0) < 0.1  # 5 images at 1s/image = ~5s

    def test_to_dict(self) -> None:
        """Test converting progress to dictionary."""
        progress = BatchProgress(
            total_images=10,
            processed_images=5,
            successful_images=4,
            failed_images=1,
            total_detections=15,
            total_classifications=12,
            total_crops=12,
            current_image="test.jpg",
        )
        progress.errors = ["error1", "error2"]

        result = progress.to_dict()

        assert result["total_images"] == 10
        assert result["processed_images"] == 5
        assert result["successful_images"] == 4
        assert result["failed_images"] == 1
        assert result["total_detections"] == 15
        assert result["total_classifications"] == 12
        assert result["total_crops"] == 12
        assert result["progress_percent"] == 50.0
        assert result["current_image"] == "test.jpg"
        assert result["error_count"] == 2
        assert isinstance(result["elapsed_time_sec"], float)
        assert isinstance(result["estimated_remaining_sec"], float)


# ================================================================================
# Test BatchProcessor Initialization
# ================================================================================


class TestBatchProcessorInitialization:
    """Test suite for BatchProcessor initialization."""

    def test_init_creates_output_directories(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test that initialization creates required output directories."""
        output_dir = tmp_path / "output"

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=sample_species_db,
        )

        # Output dir should exist
        assert processor.output_dir.exists()
        # Subdirs are created by engines (crops_dir, csv_dir) or on demand (annotated_dir)
        assert processor.crops_dir.exists()  # Created by CroppingEngine
        assert processor.csv_dir.exists()  # Created by CSVExporter
        # annotated_dir is NOT created until first annotated save

    def test_init_with_custom_configs(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test initialization with custom configurations."""
        from core.cropping_engine import CropConfig
        from core.csv_exporter import ExportConfig

        output_dir = tmp_path / "output"
        batch_config = BatchConfig(classify=False, crop=False)
        crop_config = CropConfig(padding=0.2, square=True)
        export_config = ExportConfig(include_metadata=False)

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=sample_species_db,
            batch_config=batch_config,
            crop_config=crop_config,
            export_config=export_config,
            detection_confidence=0.5,
            classification_confidence=0.7,
            use_feature_classifier=True,
        )

        assert processor.batch_config.classify is False
        assert processor.batch_config.crop is False
        assert processor.classification_engine is None
        assert processor.cropping_engine is None

    def test_init_validates_species_db_path(self, tmp_path: Path) -> None:
        """Test that missing species database raises error."""
        from utils.validators import ValidationError

        with pytest.raises((FileNotFoundError, ValidationError)):
            BatchProcessor(
                output_dir=tmp_path / "output",
                species_db_path=tmp_path / "nonexistent_db.json",
            )

    def test_init_with_progress_callback(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test initialization with progress callback."""
        callback = MagicMock()

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            progress_callback=callback,
        )

        assert processor.progress_callback == callback

    def test_repr(self, tmp_path: Path, sample_species_db: Path) -> None:
        """Test string representation."""
        output_dir = tmp_path / "output"

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=sample_species_db,
        )

        repr_str = repr(processor)

        assert "BatchProcessor" in repr_str
        assert str(output_dir) in repr_str
        assert "detect=True" in repr_str
        assert "classify=True" in repr_str
        assert "crop=True" in repr_str


# ================================================================================
# Test BatchProcessor - Directory Processing
# ================================================================================


class TestBatchProcessorDirectoryProcessing:
    """Test suite for directory processing methods."""

    def test_process_directory_finds_images(
        self, tmp_path: Path, sample_species_db: Path, sample_test_images: List[Path]
    ) -> None:
        """Test processing directory finds and processes images."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Copy test images to input directory
        for test_img in sample_test_images:
            target = input_dir / test_img.name
            import shutil

            shutil.copy(test_img, target)

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=sample_species_db,
            batch_config=BatchConfig(classify=False, crop=False, export_csv=False),
        )

        progress = processor.process_directory(input_dir)

        assert progress.total_images == len(sample_test_images)
        assert progress.processed_images == len(sample_test_images)

    def test_process_directory_nonexistent_raises_error(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test that nonexistent directory raises FileNotFoundError."""
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
        )

        with pytest.raises(FileNotFoundError):
            processor.process_directory(tmp_path / "nonexistent")

    def test_process_directory_empty_returns_zero_progress(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test that empty directory returns progress with zero images."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
        )

        progress = processor.process_directory(input_dir)

        assert progress.total_images == 0
        assert progress.processed_images == 0

    def test_process_directory_recursive(
        self, tmp_path: Path, sample_species_db: Path, sample_test_images: List[Path]
    ) -> None:
        """Test recursive directory search."""
        input_dir = tmp_path / "input"
        subdir = input_dir / "subdir"
        subdir.mkdir(parents=True)

        # Place images in subdirectory
        for test_img in sample_test_images:
            target = subdir / test_img.name
            import shutil

            shutil.copy(test_img, target)

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(classify=False, crop=False, export_csv=False),
        )

        # Non-recursive should find nothing
        progress_non_recursive = processor.process_directory(input_dir, recursive=False)
        assert progress_non_recursive.total_images == 0

        # Recursive should find images
        progress_recursive = processor.process_directory(input_dir, recursive=True)
        assert progress_recursive.total_images == len(sample_test_images)

    def test_process_directory_custom_patterns(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test custom file patterns."""
        from utils.image_utils import save_image

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create test images with different extensions
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        save_image(img, input_dir / "test.jpg")
        save_image(img, input_dir / "test.png")
        save_image(img, input_dir / "test.bmp")

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(classify=False, crop=False, export_csv=False),
        )

        # Only JPG
        progress = processor.process_directory(input_dir, file_patterns=["*.jpg"])
        assert progress.total_images == 1

        # JPG and PNG
        progress = processor.process_directory(
            input_dir, file_patterns=["*.jpg", "*.png"]
        )
        assert progress.total_images == 2


# ================================================================================
# Test BatchProcessor - Image Processing
# ================================================================================


class TestBatchProcessorImageProcessing:
    """Test suite for image processing methods."""

    @patch("core.batch_processor.load_image")
    @patch("core.batch_processor.DetectionEngine")
    def test_process_images_basic(
        self,
        mock_detection_cls: Mock,
        mock_load_image: Mock,
        tmp_path: Path,
        sample_species_db: Path,
        sample_test_images: List[Path],
    ) -> None:
        """Test basic image processing."""
        # Mock load_image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = (test_img, {})

        # Mock detection engine
        mock_engine = MagicMock()
        mock_detection_cls.return_value = mock_engine

        # Mock detection results (no detections)
        mock_det_result = DetectionResult(
            image_path=sample_test_images[0],
            image=test_img,
            detections=[],
        )
        mock_engine.process_image.return_value = mock_det_result

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(classify=False, crop=False, export_csv=False),
        )

        progress = processor.process_images(sample_test_images[:1])

        assert progress.total_images == 1
        assert progress.processed_images == 1
        assert progress.successful_images == 1
        assert progress.failed_images == 0

    @patch("core.batch_processor.load_image")
    @patch("core.batch_processor.DetectionEngine")
    def test_process_images_with_detections(
        self,
        mock_detection_cls: Mock,
        mock_load_image: Mock,
        tmp_path: Path,
        sample_species_db: Path,
        sample_test_images: List[Path],
    ) -> None:
        """Test processing images with detections."""
        # Mock load_image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = (test_img, {})

        mock_engine = MagicMock()
        mock_detection_cls.return_value = mock_engine

        # Mock detection with 2 animals
        detection1 = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="deer",
        )
        detection2 = Detection(
            bbox=(300, 300, 400, 400),
            confidence=0.85,
            class_id=0,
            class_name="deer",
        )

        mock_det_result = DetectionResult(
            image_path=sample_test_images[0],
            image=test_img,
            detections=[detection1, detection2],
        )
        mock_engine.process_image.return_value = mock_det_result

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(classify=False, crop=False, export_csv=False),
        )

        progress = processor.process_images(sample_test_images[:1])

        assert progress.total_detections == 2
        assert progress.successful_images == 1

    @patch("core.batch_processor.load_image")
    @patch("core.batch_processor.DetectionEngine")
    def test_process_images_error_continues(
        self,
        mock_detection_cls: Mock,
        mock_load_image: Mock,
        tmp_path: Path,
        sample_species_db: Path,
        sample_test_images: List[Path],
    ) -> None:
        """Test that processing continues on error when configured."""
        # Mock load_image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = (test_img, {})

        mock_engine = MagicMock()
        mock_detection_cls.return_value = mock_engine

        # First image raises error, second succeeds
        mock_engine.process_image.side_effect = [
            Exception("Test error"),
            DetectionResult(
                image_path=sample_test_images[1],
                image=test_img,
                detections=[],
            ),
        ]

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(
                continue_on_error=True, classify=False, crop=False, export_csv=False
            ),
        )

        progress = processor.process_images(sample_test_images[:2])

        assert progress.processed_images == 2
        assert progress.failed_images == 1
        assert progress.successful_images == 1
        assert len(progress.errors) == 1
        assert "Test error" in progress.errors[0]

    @patch("core.batch_processor.DetectionEngine")
    def test_process_images_error_stops(
        self,
        mock_detection_cls: Mock,
        tmp_path: Path,
        sample_species_db: Path,
        sample_test_images: List[Path],
    ) -> None:
        """Test that processing stops on error when configured."""
        mock_engine = MagicMock()
        mock_detection_cls.return_value = mock_engine
        mock_engine.process_image.side_effect = Exception("Test error")

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(
                continue_on_error=False, classify=False, crop=False, export_csv=False
            ),
        )

        with pytest.raises(BatchProcessingError, match="Test error"):
            processor.process_images(sample_test_images[:1])

    def test_process_images_progress_callback(
        self, tmp_path: Path, sample_species_db: Path, sample_test_images: List[Path]
    ) -> None:
        """Test that progress callback is called."""
        callback = MagicMock()

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(classify=False, crop=False, export_csv=False),
            progress_callback=callback,
        )

        processor.process_images(sample_test_images[:1])

        # Callback should be called for each image
        assert callback.call_count >= 1
        # First call should have BatchProgress as argument
        call_args = callback.call_args_list[0][0][0]
        assert isinstance(call_args, BatchProgress)


# ================================================================================
# Test BatchProcessor - Pipeline Integration
# ================================================================================


class TestBatchProcessorPipeline:
    """Test suite for full pipeline integration."""

    @patch("core.batch_processor.load_image")
    @patch("core.batch_processor.ClassificationEngine")
    @patch("core.batch_processor.DetectionEngine")
    def test_full_pipeline_with_classification(
        self,
        mock_detection_cls: Mock,
        mock_classification_cls: Mock,
        mock_load_image: Mock,
        tmp_path: Path,
        sample_species_db: Path,
        sample_test_images: List[Path],
    ) -> None:
        """Test full pipeline with detection and classification."""
        # Mock load_image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_load_image.return_value = (test_img, {})

        # Mock detection
        mock_det_engine = MagicMock()
        mock_detection_cls.return_value = mock_det_engine

        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="animal",
        )
        mock_det_result = DetectionResult(
            image_path=sample_test_images[0],
            image=test_img,
            detections=[detection],
        )
        mock_det_engine.process_image.return_value = mock_det_result

        # Mock classification
        mock_class_engine = MagicMock()
        mock_classification_cls.return_value = mock_class_engine

        class_result = ClassificationResult(
            species_id="deer_whitetailed",
            common_name="white-tailed deer",
            scientific_name="Odocoileus virginianus",
            confidence=0.85,
            yolo_class="animal",
            yolo_confidence=0.9,
        )
        mock_class_engine.classify_detection.return_value = class_result

        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=BatchConfig(
                classify=True, crop=False, export_csv=False, save_annotated=False
            ),
        )

        progress = processor.process_images(sample_test_images[:1])

        assert progress.total_detections == 1
        assert progress.total_classifications == 1
        mock_class_engine.classify_detection.assert_called_once()

    def test_get_statistics(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test get_statistics method."""
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
        )

        stats = processor.get_statistics()

        assert isinstance(stats, dict)
        assert "cropping" in stats
        # CSV exporter doesn't have get_statistics, so "export" won't be present


# ================================================================================
# Fixtures
# ================================================================================


@pytest.fixture
def sample_species_db(tmp_path: Path) -> Path:
    """Create a sample species database file."""
    import json

    db_path = tmp_path / "species_db.json"
    db_data = {
        "white-tailed deer": {
            "scientific_name": "Odocoileus virginianus",
            "common_names": ["white-tailed deer", "whitetail"],
        }
    }

    with open(db_path, "w") as f:
        json.dump(db_data, f)

    return db_path


@pytest.fixture
def sample_test_images(tmp_path: Path) -> List[Path]:
    """Create sample test images."""
    from utils.image_utils import save_image

    images_dir = tmp_path / "test_images"
    images_dir.mkdir()

    image_paths: List[Path] = []

    for i in range(2):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        path = images_dir / f"test_{i+1}.jpg"
        save_image(img, path)
        image_paths.append(path)

    return image_paths


class TestBatchProcessorCallbacks:
    """Test progress callback error handling."""

    @patch("core.batch_processor.DetectionEngine")
    def test_progress_callback_exception(
        self, mock_detection_engine_cls: Mock, tmp_path: Path, 
        sample_images: List[Path], sample_species_db: Path
    ) -> None:
        """Test that exceptions in progress callback are handled gracefully."""
        mock_detection = MagicMock()
        mock_detection.detect_batch.return_value = [[], [], []]
        mock_engine = MagicMock()
        mock_engine.detection_engine = mock_detection
        mock_detection_engine_cls.return_value = mock_detection
        
        # Create callback that raises exception
        def bad_callback(progress: Any) -> None:
            raise RuntimeError("Callback error")
        
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            progress_callback=bad_callback
        )
        
        # Should complete processing despite callback errors
        results = processor.process_images(sample_images)
        
        assert results.total_images == 3
        # Processing should continue even with callback errors

    @patch("core.batch_processor.DetectionEngine")
    @patch("core.batch_processor.CSVExporter")
    def test_csv_export_exception(
        self, mock_csv_cls: Mock, mock_detection_engine_cls: Mock,
        tmp_path: Path, sample_images: List[Path], sample_species_db: Path
    ) -> None:
        """Test that CSV export exceptions are handled gracefully."""
        mock_detection = MagicMock()
        mock_detection.detect_batch.return_value = [[], [], []]
        mock_engine = MagicMock()
        mock_engine.detection_engine = mock_detection
        mock_detection_engine_cls.return_value = mock_detection
        
        # Make CSV exporter raise exception on export_combined method
        mock_exporter = MagicMock()
        mock_exporter.export_combined.side_effect = RuntimeError("CSV error")
        mock_csv_cls.return_value = mock_exporter
        
        # Use batch_config with export_csv=True
        from core.batch_processor import BatchConfig
        batch_config = BatchConfig(export_csv=True)
        
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=batch_config
        )
        
        # Should complete even if CSV export fails
        results = processor.process_images(sample_images)
        
        assert results.total_images == 3
        assert any("CSV export failed" in error for error in results.errors)


class TestBatchProcessorErrorCoverage:
    """Additional tests for error handling coverage."""

    def test_process_single_image_detect_disabled(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test processing image with detection disabled returns early (line 416)."""
        from core.batch_processor import BatchConfig
        
        # Disable detection
        batch_cfg = BatchConfig(detect=False)
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=batch_cfg,
        )
        
        # Create test image
        test_img = tmp_path / "empty.jpg"
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img), img_array)
        
        # Should return early with all None (line 416)
        det, cls, crop = processor._process_single_image(test_img)
        
        assert det is None
        assert cls is None
        assert crop is None

    def test_process_single_image_no_detections(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test processing image with no detections returns early (line 428-430)."""
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
        )
        
        # Create test image
        test_img = tmp_path / "empty.jpg"
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img), img_array)
        
        # Mock detection to return no detections
        with patch.object(processor.detection_engine, 'process_image') as mock_process:
            mock_result = MagicMock()
            mock_result.has_detections.return_value = False
            mock_result.detections = []
            mock_process.return_value = mock_result
            
            det, cls, crop = processor._process_single_image(test_img)
            
            # Should return early with None for classification and crop
            assert det is not None
            assert cls is None
            assert crop is None

    def test_classification_exception_handling(
        self, tmp_path: Path, sample_species_db: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that classification exceptions are caught and logged."""
        from core.batch_processor import BatchConfig
        
        batch_cfg = BatchConfig(classify=True, crop=False)
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=batch_cfg,
        )
        
        # Create test image
        test_img = tmp_path / "test.jpg"
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img), img_array)
        
        # Mock detection with one detection
        mock_detection = Detection(
            bbox=(10, 20, 50, 60),
            confidence=0.9,
            class_id=0,
            class_name="animal"
        )
        
        with patch.object(processor.detection_engine, 'process_image') as mock_process:
            mock_result = MagicMock()
            mock_result.has_detections.return_value = True
            mock_result.detections = [mock_detection]
            mock_process.return_value = mock_result
            
            # Make classification raise exception (lines 442-444)
            with patch.object(
                processor.classification_engine, 
                'classify_detection',
                side_effect=RuntimeError("Classification error")
            ):
                with caplog.at_level(logging.ERROR):
                    det, cls, crop = processor._process_single_image(test_img)
                
                # Should catch exception and append None
                assert cls is not None
                assert None in cls
                assert any("Classification failed" in record.message for record in caplog.records)

    def test_cropping_exception_handling(
        self, tmp_path: Path, sample_species_db: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that cropping exceptions are caught and logged."""
        from core.batch_processor import BatchConfig
        
        batch_cfg = BatchConfig(classify=False, crop=True)
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=batch_cfg,
        )
        
        # Create test image
        test_img = tmp_path / "test.jpg"
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img), img_array)
        
        # Mock detection with one detection
        mock_detection = Detection(
            bbox=(10, 20, 50, 60),
            confidence=0.9,
            class_id=0,
            class_name="animal"
        )
        
        with patch.object(processor.detection_engine, 'process_image') as mock_process:
            mock_result = MagicMock()
            mock_result.has_detections.return_value = True
            mock_result.detections = [mock_detection]
            mock_process.return_value = mock_result
            
            # Make cropping raise exception (lines 465-467)
            with patch.object(
                processor.cropping_engine,
                'crop_detection',
                side_effect=RuntimeError("Cropping error")
            ):
                with caplog.at_level(logging.ERROR):
                    det, cls, crop = processor._process_single_image(test_img)
                
                # Should catch exception and append None
                assert crop is not None
                assert None in crop
                assert any("Cropping failed" in record.message for record in caplog.records)

    def test_csv_export_exception_handling(
        self, tmp_path: Path, sample_species_db: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that CSV export exceptions are caught and logged (lines 373-375)."""
        from core.batch_processor import BatchConfig
        
        batch_cfg = BatchConfig(detect=True, classify=False, crop=False, export_csv=True)
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
            batch_config=batch_cfg,
        )
        
        # Create test images
        test_dir = tmp_path / "images"
        test_dir.mkdir()
        test_img = test_dir / "test.jpg"
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img), img_array)
        
        # Make CSV exporter raise exception
        with patch.object(processor.csv_exporter, 'export_combined', side_effect=RuntimeError("Export error")):
            with caplog.at_level(logging.ERROR):
                progress = processor.process_images([test_img])
                
                # Should catch exception and log error (lines 373-375)
                assert any("Failed to export CSV" in record.message for record in caplog.records)
                assert any("CSV export failed" in error for error in progress.errors)

    def test_get_statistics_method(
        self, tmp_path: Path, sample_species_db: Path
    ) -> None:
        """Test get_statistics method returns stats from all engines (line 484)."""
        processor = BatchProcessor(
            output_dir=tmp_path / "output",
            species_db_path=sample_species_db,
        )
        
        # Get statistics
        stats = processor.get_statistics()
        
        # Should return dictionary with engine stats (line 484)
        assert isinstance(stats, dict)
        # Detection and classification engines don't have get_statistics method
        # Only cropping and export have stats
        assert "cropping" in stats
        assert isinstance(stats["cropping"], dict)
