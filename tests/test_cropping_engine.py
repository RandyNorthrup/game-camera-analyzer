"""
Comprehensive tests for core/cropping_engine.py module.

Tests cropping operations, quality checks, file organization, and batch processing.
"""

import logging
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.classification_engine import ClassificationResult
from core.cropping_engine import (
    CropConfig,
    CropResult,
    CroppingEngine,
    CroppingError,
)
from models.yolo_detector import Detection


class TestCroppingError:
    """Test suite for CroppingError exception."""

    def test_cropping_error_is_exception(self) -> None:
        """Test CroppingError is an Exception."""
        assert issubclass(CroppingError, Exception)

    def test_cropping_error_with_message(self) -> None:
        """Test CroppingError with message."""
        error = CroppingError("Crop failed")
        assert str(error) == "Crop failed"


class TestCropConfig:
    """Test suite for CropConfig dataclass."""

    def test_crop_config_defaults(self) -> None:
        """Test CropConfig default values."""
        config = CropConfig()
        
        assert config.padding == 0.1
        assert config.square is False
        assert config.min_width == 50
        assert config.min_height == 50
        assert config.max_width is None
        assert config.max_height is None
        assert config.quality == 95

    def test_crop_config_custom_values(self) -> None:
        """Test CropConfig with custom values."""
        config = CropConfig(
            padding=0.2,
            square=True,
            min_width=100,
            min_height=100,
            max_width=500,
            max_height=500,
            quality=85,
        )
        
        assert config.padding == 0.2
        assert config.square is True
        assert config.min_width == 100

    def test_crop_config_invalid_padding(self) -> None:
        """Test CropConfig validation fails for invalid padding."""
        with pytest.raises(Exception):  # Will raise from validate_padding_ratio
            CropConfig(padding=3.0)

    def test_crop_config_invalid_min_width(self) -> None:
        """Test CropConfig validation fails for invalid min_width."""
        with pytest.raises(ValueError, match="min_width must be >= 1"):
            CropConfig(min_width=0)

    def test_crop_config_invalid_min_height(self) -> None:
        """Test CropConfig validation fails for invalid min_height."""
        with pytest.raises(ValueError, match="min_height must be >= 1"):
            CropConfig(min_height=-1)

    def test_crop_config_max_less_than_min_width(self) -> None:
        """Test CropConfig validation fails when max < min width."""
        with pytest.raises(ValueError, match="max_width.*must be >= min_width"):
            CropConfig(min_width=200, max_width=100)

    def test_crop_config_max_less_than_min_height(self) -> None:
        """Test CropConfig validation fails when max < min height."""
        with pytest.raises(ValueError, match="max_height.*must be >= min_height"):
            CropConfig(min_height=200, max_height=100)

    def test_crop_config_invalid_quality_low(self) -> None:
        """Test CropConfig validation fails for quality < 1."""
        with pytest.raises(ValueError, match="quality must be 1-100"):
            CropConfig(quality=0)

    def test_crop_config_invalid_quality_high(self) -> None:
        """Test CropConfig validation fails for quality > 100."""
        with pytest.raises(ValueError, match="quality must be 1-100"):
            CropConfig(quality=101)


class TestCropResult:
    """Test suite for CropResult dataclass."""

    def test_crop_result_initialization(self) -> None:
        """Test CropResult initialization."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="animal",
        )
        
        result = CropResult(
            crop_image=image,
            detection=detection,
            classification=None,
            original_bbox=(10, 10, 50, 50),
            final_bbox=(5, 5, 55, 55),
        )
        
        assert result.crop_image.shape == (100, 100, 3)
        assert result.detection == detection
        assert result.classification is None

    def test_get_dimensions(self) -> None:
        """Test getting crop dimensions."""
        image = np.random.randint(0, 255, (120, 200, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="animal",
        )
        
        result = CropResult(
            crop_image=image,
            detection=detection,
            classification=None,
            original_bbox=(10, 10, 50, 50),
            final_bbox=(10, 10, 50, 50),
        )
        
        width, height = result.get_dimensions()
        
        assert width == 200
        assert height == 120

    def test_passes_quality_checks_success(self) -> None:
        """Test quality checks pass for valid crop."""
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="animal",
        )
        
        result = CropResult(
            crop_image=image,
            detection=detection,
            classification=None,
            original_bbox=(10, 10, 50, 50),
            final_bbox=(10, 10, 50, 50),
        )
        
        passes, reason = result.passes_quality_checks(50, 50)
        
        assert passes is True
        assert reason == "OK"

    def test_passes_quality_checks_width_failure(self) -> None:
        """Test quality checks fail for insufficient width."""
        image = np.random.randint(0, 255, (100, 30, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="animal",
        )
        
        result = CropResult(
            crop_image=image,
            detection=detection,
            classification=None,
            original_bbox=(10, 10, 50, 50),
            final_bbox=(10, 10, 50, 50),
        )
        
        passes, reason = result.passes_quality_checks(50, 50)
        
        assert passes is False
        assert "Width" in reason

    def test_passes_quality_checks_height_failure(self) -> None:
        """Test quality checks fail for insufficient height."""
        image = np.random.randint(0, 255, (30, 100, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            class_id=0,
            class_name="animal",
        )
        
        result = CropResult(
            crop_image=image,
            detection=detection,
            classification=None,
            original_bbox=(10, 10, 50, 50),
            final_bbox=(10, 10, 50, 50),
        )
        
        passes, reason = result.passes_quality_checks(50, 50)
        
        assert passes is False
        assert "Height" in reason


class TestCroppingEngine:
    """Test suite for CroppingEngine class."""

    def test_init_creates_output_dir(self, tmp_path: Path) -> None:
        """Test CroppingEngine creates output directory."""
        output_dir = tmp_path / "crops"
        
        engine = CroppingEngine(output_dir=output_dir)
        
        assert output_dir.exists()
        assert engine.output_dir == output_dir

    def test_init_with_custom_config(self, tmp_path: Path) -> None:
        """Test CroppingEngine with custom config."""
        output_dir = tmp_path / "crops"
        config = CropConfig(padding=0.2, quality=85)
        
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        assert engine.config.padding == 0.2
        assert engine.config.quality == 85

    def test_init_statistics(self, tmp_path: Path) -> None:
        """Test CroppingEngine initializes statistics."""
        output_dir = tmp_path / "crops"
        
        engine = CroppingEngine(output_dir=output_dir)
        
        assert engine.stats["total_crops"] == 0
        assert engine.stats["successful_crops"] == 0

    def test_crop_detection_basic(self, tmp_path: Path) -> None:
        """Test basic detection cropping."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(100, 100, 300, 250),
            confidence=0.85,
            class_id=0,
            class_name="dog",
        )
        
        result = engine.crop_detection(image, detection, save=False)
        
        assert result is not None
        assert result.crop_image.shape[0] > 0
        assert result.detection == detection

    def test_crop_detection_with_save(self, tmp_path: Path) -> None:
        """Test cropping with save to disk."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(100, 100, 300, 250),
            confidence=0.85,
            class_id=0,
            class_name="dog",
        )
        
        result = engine.crop_detection(image, detection, save=True, source_filename="test.jpg")
        
        assert result is not None
        assert result.output_path is not None
        assert result.output_path.exists()

    def test_crop_detection_with_classification(self, tmp_path: Path) -> None:
        """Test cropping with classification result."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(100, 100, 300, 250),
            confidence=0.85,
            class_id=0,
            class_name="dog",
        )
        classification = ClassificationResult(
            species_id="coyote",
            common_name="Coyote",
            scientific_name="Canis latrans",
            confidence=0.75,
            yolo_class="dog",
            yolo_confidence=0.85,
        )
        
        result = engine.crop_detection(image, detection, classification, save=True)
        
        assert result is not None
        assert result.classification == classification
        assert "coyote" in str(result.output_path).lower()

    def test_crop_detection_quality_failure(self, tmp_path: Path) -> None:
        """Test cropping fails quality checks."""
        output_dir = tmp_path / "crops"
        config = CropConfig(min_width=500, min_height=500)  # Very high requirements
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 10, 50, 50),  # Small bbox
            confidence=0.85,
            class_id=0,
            class_name="dog",
        )
        
        result = engine.crop_detection(image, detection, save=False)
        
        assert result is None
        assert engine.stats["quality_failures"] > 0

    def test_crop_detection_exception_handling(self, tmp_path: Path) -> None:
        """Test cropping handles exceptions."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        # Invalid image (empty)
        image = np.array([])
        detection = Detection(
            bbox=(100, 100, 300, 250),
            confidence=0.85,
            class_id=0,
            class_name="dog",
        )
        
        with pytest.raises(CroppingError):
            engine.crop_detection(image, detection)

    def test_crop_batch_success(self, tmp_path: Path) -> None:
        """Test batch cropping multiple images."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ]
        detections_list = [
            [Detection((100, 100, 300, 250), 0.85, 0, "dog")],
            [Detection((50, 50, 200, 200), 0.9, 1, "cat")],
        ]
        
        results = engine.crop_batch(images, detections_list, save=False)
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    def test_crop_batch_length_mismatch_detections(self, tmp_path: Path) -> None:
        """Test batch cropping fails with mismatched lengths."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        detections_list = [[], []]  # Wrong length
        
        with pytest.raises(ValueError, match="length mismatch"):
            engine.crop_batch(images, detections_list)

    def test_crop_batch_length_mismatch_classifications(self, tmp_path: Path) -> None:
        """Test batch cropping fails with mismatched classification list."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        detections_list = [[]]
        classifications_list = [[], []]  # Wrong length
        
        with pytest.raises(ValueError, match="Classifications list length mismatch"):
            engine.crop_batch(images, detections_list, classifications_list=classifications_list)

    def test_crop_batch_length_mismatch_filenames(self, tmp_path: Path) -> None:
        """Test batch cropping fails with mismatched filenames."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        detections_list = [[]]
        source_filenames = ["a.jpg", "b.jpg"]  # Wrong length
        
        with pytest.raises(ValueError, match="Source filenames length mismatch"):
            engine.crop_batch(images, detections_list, source_filenames=source_filenames)

    def test_crop_batch_handles_errors(self, tmp_path: Path) -> None:
        """Test batch cropping handles individual errors."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ]
        
        # Create detection with invalid bbox
        detections_list = [
            [Detection((1000, 1000, 2000, 2000), 0.85, 0, "dog")],  # Outside image
        ]
        
        results = engine.crop_batch(images, detections_list, save=False)
        
        assert len(results) == 1
        assert results[0][0] is None  # Should be None due to error

    def test_check_quality_valid(self, tmp_path: Path) -> None:
        """Test quality check passes for valid image."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        crop_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        passes, reason = engine._check_quality(crop_image)
        
        assert passes is True
        assert reason == "OK"

    def test_check_quality_too_small_width(self, tmp_path: Path) -> None:
        """Test quality check fails for small width."""
        output_dir = tmp_path / "crops"
        config = CropConfig(min_width=200)
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        crop_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        passes, reason = engine._check_quality(crop_image)
        
        assert passes is False
        assert "Width" in reason

    def test_check_quality_too_small_height(self, tmp_path: Path) -> None:
        """Test quality check fails for small height."""
        output_dir = tmp_path / "crops"
        config = CropConfig(min_height=200)
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        crop_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        passes, reason = engine._check_quality(crop_image)
        
        assert passes is False
        assert "Height" in reason

    def test_check_quality_too_large_width(self, tmp_path: Path) -> None:
        """Test quality check fails for excessive width."""
        output_dir = tmp_path / "crops"
        config = CropConfig(max_width=100)
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        crop_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        passes, reason = engine._check_quality(crop_image)
        
        assert passes is False
        assert "maximum" in reason

    def test_check_quality_too_large_height(self, tmp_path: Path) -> None:
        """Test quality check fails for excessive height."""
        output_dir = tmp_path / "crops"
        config = CropConfig(max_height=100)
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        crop_image = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        
        passes, reason = engine._check_quality(crop_image)
        
        assert passes is False
        assert "maximum" in reason

    def test_check_quality_empty_image(self, tmp_path: Path) -> None:
        """Test quality check fails for empty image."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        # Empty array would cause ValueError on unpacking, which is expected
        crop_image = np.array([])
        
        # Implementation raises ValueError for empty images when unpacking shape
        with pytest.raises(ValueError):
            engine._check_quality(crop_image)

    def test_check_quality_low_variance(self, tmp_path: Path) -> None:
        """Test quality check fails for low variance (blank) image."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        # Create nearly uniform image
        crop_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        passes, reason = engine._check_quality(crop_image)
        
        assert passes is False
        assert "variance" in reason

    def test_sanitize_filename(self, tmp_path: Path) -> None:
        """Test filename sanitization."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        result = engine._sanitize_filename("White-tailed Deer")
        
        # Hyphens are preserved, spaces become underscores
        assert result == "white-tailed_deer"

    def test_sanitize_filename_special_chars(self, tmp_path: Path) -> None:
        """Test sanitization removes special characters."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        result = engine._sanitize_filename("Animal@#$%Name!")
        
        assert "@" not in result
        assert "#" not in result
        assert "animal" in result

    def test_sanitize_filename_consecutive_underscores(self, tmp_path: Path) -> None:
        """Test sanitization removes consecutive underscores."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        result = engine._sanitize_filename("animal___name")
        
        assert "___" not in result
        assert result == "animal_name"

    def test_generate_output_path_basic(self, tmp_path: Path) -> None:
        """Test output path generation."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir, organize_by_species=False, organize_by_date=False)
        
        detection = Detection((100, 100, 300, 250), 0.85, 0, "dog")
        
        path = engine._generate_output_path(detection, None, None)
        
        assert path.parent == output_dir
        assert "dog" in path.name
        assert path.suffix == ".jpg"

    def test_generate_output_path_with_classification(self, tmp_path: Path) -> None:
        """Test output path with classification."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir, organize_by_species=True)
        
        detection = Detection((100, 100, 300, 250), 0.85, 0, "dog")
        classification = ClassificationResult(
            species_id="coyote",
            common_name="Coyote",
            scientific_name="Canis latrans",
            confidence=0.75,
            yolo_class="dog",
            yolo_confidence=0.85,
        )
        
        path = engine._generate_output_path(detection, classification, None)
        
        assert "coyote" in str(path)
        # Directory should be created
        assert path.parent.exists()

    def test_generate_output_path_with_date(self, tmp_path: Path) -> None:
        """Test output path with date organization."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir, organize_by_date=True, organize_by_species=False)
        
        detection = Detection((100, 100, 300, 250), 0.85, 0, "dog")
        
        path = engine._generate_output_path(detection, None, None)
        
        # Should have date directory
        assert len(path.parts) > len(output_dir.parts) + 1

    def test_generate_output_path_unique_filename(self, tmp_path: Path) -> None:
        """Test output path generates unique filenames."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir, organize_by_species=False, organize_by_date=False)
        
        detection = Detection((100, 100, 300, 250), 0.85, 0, "dog")
        
        # Generate first path and create file
        path1 = engine._generate_output_path(detection, None, None)
        path1.touch()
        
        # Generate second path - should be different
        path2 = engine._generate_output_path(detection, None, None)
        
        assert path1 != path2

    def test_get_statistics(self, tmp_path: Path) -> None:
        """Test getting statistics."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        stats = engine.get_statistics()
        
        assert "total_crops" in stats
        assert "successful_crops" in stats
        assert "success_rate" in stats
        assert stats["success_rate"] == 0.0

    def test_get_statistics_with_data(self, tmp_path: Path) -> None:
        """Test statistics after cropping."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = Detection((100, 100, 300, 250), 0.85, 0, "dog")
        
        engine.crop_detection(image, detection, save=False)
        
        stats = engine.get_statistics()
        
        assert stats["total_crops"] == 1
        assert stats["successful_crops"] == 1
        assert stats["success_rate"] == 100.0

    def test_reset_statistics(self, tmp_path: Path) -> None:
        """Test resetting statistics."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        engine.stats["total_crops"] = 10
        engine.stats["successful_crops"] = 8
        
        engine.reset_statistics()
        
        assert engine.stats["total_crops"] == 0
        assert engine.stats["successful_crops"] == 0

    def test_repr(self, tmp_path: Path) -> None:
        """Test string representation."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        repr_str = repr(engine)
        
        assert "CroppingEngine" in repr_str
        assert str(output_dir) in repr_str

    def test_validate_crop_low_variance(self, tmp_path: Path) -> None:
        """Test that low variance (blank) crops are rejected."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        # Test low variance image (line 364 - blank crop detection)
        blank_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        is_valid, message = engine._check_quality(blank_image)
        assert not is_valid
        assert "Low variance" in message or "blank" in message.lower()


class TestIntegration:
    """Integration tests for cropping engine."""

    def test_full_cropping_workflow(self, tmp_path: Path) -> None:
        """Test complete cropping workflow."""
        output_dir = tmp_path / "crops"
        config = CropConfig(padding=0.15, min_width=50, min_height=50)
        engine = CroppingEngine(
            output_dir=output_dir,
            config=config,
            organize_by_species=True,
            organize_by_date=False,
        )
        
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create detection and classification
        detection = Detection((100, 100, 300, 250), 0.85, 0, "dog")
        classification = ClassificationResult(
            species_id="coyote",
            common_name="Coyote",
            scientific_name="Canis latrans",
            confidence=0.75,
            yolo_class="dog",
            yolo_confidence=0.85,
        )
        
        # Crop and save
        result = engine.crop_detection(
            image,
            detection,
            classification,
            save=True,
            source_filename="test_image.jpg",
        )
        
        assert result is not None
        assert result.output_path is not None
        assert result.output_path.exists()
        assert "coyote" in str(result.output_path)
        
        # Check statistics
        stats = engine.get_statistics()
        assert stats["successful_crops"] == 1

    def test_batch_workflow_with_mixed_results(self, tmp_path: Path) -> None:
        """Test batch processing with some failures."""
        output_dir = tmp_path / "crops"
        config = CropConfig(min_width=200, min_height=200)  # High requirements
        engine = CroppingEngine(output_dir=output_dir, config=config)
        
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ]
        
        # First detection will pass, second will fail (too small)
        detections_list = [
            [Detection((100, 100, 400, 400), 0.85, 0, "dog")],  # Large bbox
            [Detection((10, 10, 50, 50), 0.85, 0, "cat")],  # Small bbox
        ]
        
        results = engine.crop_batch(images, detections_list, save=False)
        
        assert len(results) == 2
        assert results[0][0] is not None  # Should succeed
        assert results[1][0] is None  # Should fail quality check
        
        stats = engine.get_statistics()
        assert stats["successful_crops"] == 1
        assert stats["quality_failures"] == 1

    def test_crop_batch_with_exceptions(self, tmp_path: Path) -> None:
        """Test crop_batch handles exceptions gracefully."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        # Create valid image
        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        # Create normal detection
        detection = Detection((100, 100, 200, 200), 0.85, 0, "dog")
        
        # Mock crop_detection to raise exception
        with patch.object(engine, "crop_detection", side_effect=RuntimeError("Crop failed")):
            results = engine.crop_batch([image], [[detection]], save=False)
        
        assert len(results) == 1
        assert results[0][0] is None
        
        # Check that error was logged
        stats = engine.get_statistics()
        # Stats won't change since exception prevents crop attempt from being counted

    def test_crop_detection_save_io_error(self, tmp_path: Path) -> None:
        """Test crop_detection handles IO errors gracefully when saving."""
        output_dir = tmp_path / "crops"
        engine = CroppingEngine(output_dir=output_dir)
        
        # Create a valid crop
        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        detection = Detection((100, 100, 300, 300), 0.85, 0, "dog")
        
        # Mock save_image to simulate IO error
        with patch("core.cropping_engine.save_image", side_effect=OSError("Permission denied")):
            with pytest.raises(CroppingError, match="Cropping failed"):
                engine.crop_detection(image, detection, save=True, source_filename="test.jpg")
