"""
Comprehensive tests for utils/validators.py module.

Tests validation functions for paths, images, parameters, and ranges.
"""

import logging
import os
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from utils.validators import (
    SUPPORTED_IMAGE_FORMATS,
    ValidationError,
    validate_batch_size,
    validate_confidence_threshold,
    validate_directory_writable,
    validate_file_exists,
    validate_file_readable,
    validate_image_dimensions,
    validate_image_format,
    validate_image_readable,
    validate_in_choices,
    validate_in_range,
    validate_iou_threshold,
    validate_model_path,
    validate_output_path,
    validate_padding_ratio,
    validate_positive_integer,
)


class TestValidationError:
    """Test suite for ValidationError exception."""

    def test_validation_error_is_exception(self) -> None:
        """Test ValidationError is an Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_with_message(self) -> None:
        """Test ValidationError with message."""
        error = ValidationError("Test error")
        assert str(error) == "Test error"


class TestValidateFileExists:
    """Test suite for validate_file_exists function."""

    def test_validate_existing_file(self, tmp_path: Path) -> None:
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        result = validate_file_exists(test_file)
        
        assert result == test_file.resolve()

    def test_validate_existing_directory(self, tmp_path: Path) -> None:
        """Test validation of existing directory."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        result = validate_file_exists(test_dir, "directory")
        
        assert result == test_dir.resolve()

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test validation fails for nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(ValidationError, match="not found"):
            validate_file_exists(test_file)

    def test_validate_with_custom_file_type(self, tmp_path: Path) -> None:
        """Test validation with custom file type description."""
        test_file = tmp_path / "nonexistent.jpg"

        with pytest.raises(ValidationError, match="Image not found"):
            validate_file_exists(test_file, "image")

    def test_validate_with_home_expansion(self, tmp_path: Path) -> None:
        """Test validation with tilde expansion."""
        with patch("pathlib.Path.expanduser") as mock_expand:
            test_file = tmp_path / "test.txt"
            test_file.touch()
            mock_expand.return_value = test_file

            result = validate_file_exists("~/test.txt")
            
            assert result == test_file.resolve()

    def test_validate_invalid_path_string(self) -> None:
        """Test validation with invalid path string."""
        # Test with null bytes which are invalid in paths
        with pytest.raises(ValidationError, match="Invalid.*path"):
            validate_file_exists("\x00invalid")


class TestValidateFileReadable:
    """Test suite for validate_file_readable function."""

    def test_validate_readable_file(self, tmp_path: Path) -> None:
        """Test validation of readable file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = validate_file_readable(test_file)
        
        assert result == test_file.resolve()

    def test_validate_nonreadable_file(self, tmp_path: Path) -> None:
        """Test validation fails for non-readable file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        test_file.chmod(0o000)

        try:
            with pytest.raises(ValidationError, match="not readable"):
                validate_file_readable(test_file)
        finally:
            test_file.chmod(0o644)

    def test_validate_nonexistent_file_readable(self, tmp_path: Path) -> None:
        """Test validation fails for nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(ValidationError, match="not found"):
            validate_file_readable(test_file)


class TestValidateDirectoryWritable:
    """Test suite for validate_directory_writable function."""

    def test_validate_writable_directory(self, tmp_path: Path) -> None:
        """Test validation of writable directory."""
        result = validate_directory_writable(tmp_path)
        
        assert result == tmp_path.resolve()

    def test_validate_nonexistent_directory_no_create(self, tmp_path: Path) -> None:
        """Test validation fails for nonexistent directory without create flag."""
        test_dir = tmp_path / "nonexistent"

        with pytest.raises(ValidationError, match="Directory not found"):
            validate_directory_writable(test_dir, create=False)

    def test_validate_nonexistent_directory_with_create(self, tmp_path: Path) -> None:
        """Test validation creates nonexistent directory with create flag."""
        test_dir = tmp_path / "newdir"

        result = validate_directory_writable(test_dir, create=True)
        
        assert result == test_dir.resolve()
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_validate_nested_directory_creation(self, tmp_path: Path) -> None:
        """Test validation creates nested directories."""
        test_dir = tmp_path / "level1" / "level2" / "level3"

        result = validate_directory_writable(test_dir, create=True)
        
        assert result == test_dir.resolve()
        assert test_dir.exists()

    def test_validate_file_as_directory(self, tmp_path: Path) -> None:
        """Test validation fails when path is a file not directory."""
        test_file = tmp_path / "file.txt"
        test_file.touch()

        with pytest.raises(ValidationError, match="not a directory"):
            validate_directory_writable(test_file)

    def test_validate_nonwritable_directory(self, tmp_path: Path) -> None:
        """Test validation fails for non-writable directory."""
        test_dir = tmp_path / "readonly"
        test_dir.mkdir()
        test_dir.chmod(0o444)

        try:
            with pytest.raises(ValidationError, match="not writable"):
                validate_directory_writable(test_dir)
        finally:
            test_dir.chmod(0o755)

    def test_validate_creation_permission_error(self, tmp_path: Path) -> None:
        """Test validation handles permission error during creation."""
        test_dir = tmp_path / "protected" / "newdir"

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            with pytest.raises(ValidationError, match="Cannot create directory"):
                validate_directory_writable(test_dir, create=True)

    def test_validate_invalid_path_exception(self) -> None:
        """Test validation handles invalid path exceptions."""
        # Create a path that will cause ValueError when resolved
        with patch("pathlib.Path.expanduser", side_effect=ValueError("Invalid path")):
            with pytest.raises(ValidationError, match="Invalid directory path"):
                validate_directory_writable("/some/path")

    def test_validate_invalid_path_runtime_error(self) -> None:
        """Test validation handles runtime errors during path resolution."""
        with patch("pathlib.Path.resolve", side_effect=RuntimeError("Cannot resolve")):
            with pytest.raises(ValidationError, match="Invalid directory path"):
                validate_directory_writable("/some/path")


class TestValidateOutputPath:
    """Test suite for validate_output_path function."""

    def test_validate_new_output_path(self, tmp_path: Path) -> None:
        """Test validation of new output path."""
        output_file = tmp_path / "output.txt"

        result = validate_output_path(output_file)
        
        assert result == output_file.resolve()

    def test_validate_existing_path_no_overwrite(self, tmp_path: Path) -> None:
        """Test validation fails for existing file without overwrite."""
        output_file = tmp_path / "existing.txt"
        output_file.touch()

        with pytest.raises(ValidationError, match="already exists"):
            validate_output_path(output_file, overwrite=False)

    def test_validate_existing_path_with_overwrite(self, tmp_path: Path) -> None:
        """Test validation succeeds for existing file with overwrite."""
        output_file = tmp_path / "existing.txt"
        output_file.touch()

        result = validate_output_path(output_file, overwrite=True)
        
        assert result == output_file.resolve()

    def test_validate_output_parent_not_exists(self, tmp_path: Path) -> None:
        """Test validation fails when parent directory doesn't exist."""
        output_file = tmp_path / "nonexistent" / "output.txt"

        with pytest.raises(ValidationError, match="Parent directory does not exist"):
            validate_output_path(output_file)

    def test_validate_output_parent_not_writable(self, tmp_path: Path) -> None:
        """Test validation fails when parent directory is not writable."""
        parent_dir = tmp_path / "readonly"
        parent_dir.mkdir()
        output_file = parent_dir / "output.txt"

        # Mock os.access to simulate non-writable directory
        with patch("os.access", return_value=False):
            with pytest.raises(ValidationError, match="not writable"):
                validate_output_path(output_file)

    def test_validate_output_path_invalid_path_exception(self, tmp_path: Path) -> None:
        """Test validation handles invalid path exceptions."""
        with patch("pathlib.Path.expanduser", side_effect=ValueError("Invalid path")):
            with pytest.raises(ValidationError, match="Invalid output"):
                validate_output_path("/some/invalid/path.txt")

    def test_validate_output_path_runtime_error(self, tmp_path: Path) -> None:
        """Test validation handles runtime errors during path resolution."""
        with patch("pathlib.Path.resolve", side_effect=RuntimeError("Cannot resolve")):
            with pytest.raises(ValidationError, match="Invalid output"):
                validate_output_path("/some/path.txt")


class TestValidateImageFormat:
    """Test suite for validate_image_format function."""

    def test_validate_jpg_format(self, tmp_path: Path) -> None:
        """Test validation of JPG format."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()

        result = validate_image_format(test_file)
        
        assert result == test_file.resolve()

    def test_validate_jpeg_format(self, tmp_path: Path) -> None:
        """Test validation of JPEG format."""
        test_file = tmp_path / "test.jpeg"
        test_file.touch()

        result = validate_image_format(test_file)
        
        assert result == test_file.resolve()

    def test_validate_png_format(self, tmp_path: Path) -> None:
        """Test validation of PNG format."""
        test_file = tmp_path / "test.png"
        test_file.touch()

        result = validate_image_format(test_file)
        
        assert result == test_file.resolve()

    def test_validate_case_insensitive(self, tmp_path: Path) -> None:
        """Test validation is case-insensitive."""
        test_file = tmp_path / "test.JPG"
        test_file.touch()

        result = validate_image_format(test_file)
        
        assert result == test_file.resolve()

    def test_validate_unsupported_format(self, tmp_path: Path) -> None:
        """Test validation fails for unsupported format."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with pytest.raises(ValidationError, match="Unsupported image format"):
            validate_image_format(test_file)

    def test_validate_all_supported_formats(self, tmp_path: Path) -> None:
        """Test all supported formats are validated."""
        for fmt in SUPPORTED_IMAGE_FORMATS:
            test_file = tmp_path / f"test{fmt}"
            test_file.touch()
            
            result = validate_image_format(test_file)
            assert result == test_file.resolve()


class TestValidateImageReadable:
    """Test suite for validate_image_readable function."""

    def test_validate_readable_image_quick(self, tmp_path: Path) -> None:
        """Test quick validation of readable image."""
        test_file = tmp_path / "test.jpg"
        
        # Create a simple image
        img = Image.new("RGB", (100, 50), color="red")
        img.save(test_file)

        result_path, dimensions = validate_image_readable(test_file, check_corruption=False)
        
        assert result_path == test_file.resolve()
        assert dimensions == (100, 50)

    def test_validate_readable_image_full(self, tmp_path: Path) -> None:
        """Test full validation of readable image."""
        test_file = tmp_path / "test.png"
        
        # Create a simple image
        img = Image.new("RGB", (200, 100), color="blue")
        img.save(test_file)

        result_path, dimensions = validate_image_readable(test_file, check_corruption=True)
        
        assert result_path == test_file.resolve()
        assert dimensions == (200, 100)

    def test_validate_grayscale_image(self, tmp_path: Path) -> None:
        """Test validation of grayscale image."""
        test_file = tmp_path / "gray.png"
        
        # Create grayscale image
        img = Image.new("L", (150, 150), color=128)
        img.save(test_file)

        result_path, dimensions = validate_image_readable(test_file, check_corruption=True)
        
        assert result_path == test_file.resolve()
        assert dimensions == (150, 150)

    def test_validate_corrupted_image(self, tmp_path: Path) -> None:
        """Test validation fails for corrupted image."""
        test_file = tmp_path / "corrupted.jpg"
        test_file.write_bytes(b"not an image")

        with pytest.raises(ValidationError, match="Cannot"):
            validate_image_readable(test_file, check_corruption=True)

    def test_validate_empty_file(self, tmp_path: Path) -> None:
        """Test validation fails for empty file."""
        test_file = tmp_path / "empty.jpg"
        test_file.touch()

        with pytest.raises(ValidationError):
            validate_image_readable(test_file, check_corruption=True)

    def test_validate_quick_check_exception(self, tmp_path: Path) -> None:
        """Test quick validation handles PIL exceptions."""
        test_file = tmp_path / "bad.jpg"
        test_file.write_bytes(b"invalid image data")
        
        with pytest.raises(ValidationError, match="Cannot read image"):
            validate_image_readable(test_file, check_corruption=False)

    def test_validate_opencv_fails_pil_fallback(self, tmp_path: Path) -> None:
        """Test PIL fallback when OpenCV fails to read image."""
        test_file = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(test_file)
        
        # Mock cv2.imread to return None, forcing PIL fallback
        with patch("utils.validators.cv2.imread", return_value=None):
            result_path, dimensions = validate_image_readable(test_file, check_corruption=True)
            
            assert result_path == test_file.resolve()
            assert dimensions == (100, 100)

    def test_validate_empty_array_after_read(self, tmp_path: Path) -> None:
        """Test validation detects empty array after reading."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        # Mock both cv2 and PIL to return None/empty
        with patch("utils.validators.cv2.imread", return_value=None):
            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.__enter__ = MagicMock(return_value=mock_img)
                mock_img.__exit__ = MagicMock(return_value=False)
                mock_img.size = (0, 0)
                mock_open.return_value = mock_img
                
                with patch("numpy.array", return_value=None):
                    with pytest.raises(ValidationError, match="empty or corrupted"):
                        validate_image_readable(test_file, check_corruption=True)

    def test_validate_grayscale_2d_array(self, tmp_path: Path) -> None:
        """Test validation handles 2D grayscale arrays correctly."""
        test_file = tmp_path / "gray.png"
        img = Image.new("L", (80, 60), color=100)
        img.save(test_file)
        
        # Ensure we get 2D array by using grayscale
        result_path, dimensions = validate_image_readable(test_file, check_corruption=True)
        
        assert result_path == test_file.resolve()
        assert dimensions == (80, 60)

    def test_validate_reraises_validation_error(self, tmp_path: Path) -> None:
        """Test that ValidationError is re-raised correctly."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"corrupt")
        
        # This should raise ValidationError which gets re-raised
        with pytest.raises(ValidationError):
            validate_image_readable(test_file, check_corruption=True)


class TestValidateImageDimensions:
    """Test suite for validate_image_dimensions function."""

    def test_validate_dimensions_no_constraints(self) -> None:
        """Test validation with no constraints."""
        validate_image_dimensions((640, 480))  # Should not raise

    def test_validate_dimensions_min_width(self) -> None:
        """Test validation with minimum width."""
        validate_image_dimensions((640, 480), min_width=320)  # Should not raise

        with pytest.raises(ValidationError, match="width.*less than minimum"):
            validate_image_dimensions((100, 480), min_width=320)

    def test_validate_dimensions_min_height(self) -> None:
        """Test validation with minimum height."""
        validate_image_dimensions((640, 480), min_height=240)  # Should not raise

        with pytest.raises(ValidationError, match="height.*less than minimum"):
            validate_image_dimensions((640, 100), min_height=240)

    def test_validate_dimensions_max_width(self) -> None:
        """Test validation with maximum width."""
        validate_image_dimensions((640, 480), max_width=1920)  # Should not raise

        with pytest.raises(ValidationError, match="width.*exceeds maximum"):
            validate_image_dimensions((2000, 480), max_width=1920)

    def test_validate_dimensions_max_height(self) -> None:
        """Test validation with maximum height."""
        validate_image_dimensions((640, 480), max_height=1080)  # Should not raise

        with pytest.raises(ValidationError, match="height.*exceeds maximum"):
            validate_image_dimensions((640, 2000), max_height=1080)

    def test_validate_dimensions_all_constraints(self) -> None:
        """Test validation with all constraints."""
        validate_image_dimensions(
            (640, 480),
            min_width=320,
            min_height=240,
            max_width=1920,
            max_height=1080,
        )  # Should not raise

    def test_validate_dimensions_boundary_values(self) -> None:
        """Test validation at boundary values."""
        # Exact minimum
        validate_image_dimensions((320, 240), min_width=320, min_height=240)
        
        # Exact maximum
        validate_image_dimensions((1920, 1080), max_width=1920, max_height=1080)


class TestValidateConfidenceThreshold:
    """Test suite for validate_confidence_threshold function."""

    def test_validate_valid_threshold(self) -> None:
        """Test validation of valid confidence threshold."""
        result = validate_confidence_threshold(0.5)
        assert result == 0.5

    def test_validate_zero_threshold(self) -> None:
        """Test validation of zero threshold."""
        result = validate_confidence_threshold(0.0)
        assert result == 0.0

    def test_validate_one_threshold(self) -> None:
        """Test validation of 1.0 threshold."""
        result = validate_confidence_threshold(1.0)
        assert result == 1.0

    def test_validate_threshold_above_one(self) -> None:
        """Test validation fails for threshold > 1.0."""
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate_confidence_threshold(1.5)

    def test_validate_negative_threshold(self) -> None:
        """Test validation fails for negative threshold."""
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate_confidence_threshold(-0.1)

    def test_validate_non_numeric_threshold(self) -> None:
        """Test validation fails for non-numeric threshold."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_confidence_threshold("0.5")  # type: ignore

    def test_validate_integer_threshold(self) -> None:
        """Test validation accepts integer threshold."""
        result = validate_confidence_threshold(1)
        assert result == 1.0
        assert isinstance(result, float)


class TestValidateIouThreshold:
    """Test suite for validate_iou_threshold function."""

    def test_validate_valid_iou(self) -> None:
        """Test validation of valid IoU threshold."""
        result = validate_iou_threshold(0.45)
        assert result == 0.45

    def test_validate_iou_boundaries(self) -> None:
        """Test validation at boundary values."""
        assert validate_iou_threshold(0.0) == 0.0
        assert validate_iou_threshold(1.0) == 1.0

    def test_validate_iou_out_of_range(self) -> None:
        """Test validation fails for out of range IoU."""
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate_iou_threshold(1.1)

    def test_validate_iou_non_numeric(self) -> None:
        """Test validation fails for non-numeric IoU."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_iou_threshold("0.45")  # type: ignore


class TestValidatePositiveInteger:
    """Test suite for validate_positive_integer function."""

    def test_validate_positive_integer(self) -> None:
        """Test validation of positive integer."""
        result = validate_positive_integer(10, "test_param")
        assert result == 10

    def test_validate_one(self) -> None:
        """Test validation of 1."""
        result = validate_positive_integer(1, "test_param")
        assert result == 1

    def test_validate_zero_fails(self) -> None:
        """Test validation fails for zero."""
        with pytest.raises(ValidationError, match="must be >= 1"):
            validate_positive_integer(0, "test_param")

    def test_validate_negative_fails(self) -> None:
        """Test validation fails for negative."""
        with pytest.raises(ValidationError, match="must be >= 1"):
            validate_positive_integer(-5, "test_param")

    def test_validate_custom_min_value(self) -> None:
        """Test validation with custom minimum value."""
        result = validate_positive_integer(10, "test_param", min_value=5)
        assert result == 10

        with pytest.raises(ValidationError, match="must be >= 5"):
            validate_positive_integer(3, "test_param", min_value=5)

    def test_validate_non_integer(self) -> None:
        """Test validation fails for non-integer."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_integer(5.5, "test_param")  # type: ignore


class TestValidateInRange:
    """Test suite for validate_in_range function."""

    def test_validate_in_range_integer(self) -> None:
        """Test validation of integer in range."""
        result = validate_in_range(50, "test_param", 0, 100)
        assert result == 50

    def test_validate_in_range_float(self) -> None:
        """Test validation of float in range."""
        result = validate_in_range(0.75, "test_param", 0.0, 1.0)
        assert result == 0.75

    def test_validate_at_min_boundary(self) -> None:
        """Test validation at minimum boundary."""
        result = validate_in_range(0, "test_param", 0, 100)
        assert result == 0

    def test_validate_at_max_boundary(self) -> None:
        """Test validation at maximum boundary."""
        result = validate_in_range(100, "test_param", 0, 100)
        assert result == 100

    def test_validate_below_range(self) -> None:
        """Test validation fails below range."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_in_range(-1, "test_param", 0, 100)

    def test_validate_above_range(self) -> None:
        """Test validation fails above range."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_in_range(101, "test_param", 0, 100)

    def test_validate_non_numeric(self) -> None:
        """Test validation fails for non-numeric."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_in_range("50", "test_param", 0, 100)  # type: ignore


class TestValidateInChoices:
    """Test suite for validate_in_choices function."""

    def test_validate_valid_choice(self) -> None:
        """Test validation of valid choice."""
        result = validate_in_choices("option1", "test_param", ["option1", "option2"])
        assert result == "option1"

    def test_validate_invalid_choice(self) -> None:
        """Test validation fails for invalid choice."""
        with pytest.raises(ValidationError, match="must be one of"):
            validate_in_choices("option3", "test_param", ["option1", "option2"])

    def test_validate_case_sensitive(self) -> None:
        """Test case-sensitive validation."""
        with pytest.raises(ValidationError, match="must be one of"):
            validate_in_choices("OPTION1", "test_param", ["option1", "option2"])

    def test_validate_case_insensitive(self) -> None:
        """Test case-insensitive validation."""
        result = validate_in_choices(
            "OPTION1", "test_param", ["option1", "option2"], case_sensitive=False
        )
        assert result == "OPTION1"  # Original case preserved

    def test_validate_non_string(self) -> None:
        """Test validation fails for non-string."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_in_choices(123, "test_param", ["option1", "option2"])  # type: ignore


class TestValidateBatchSize:
    """Test suite for validate_batch_size function."""

    def test_validate_valid_batch_size(self) -> None:
        """Test validation of valid batch size."""
        result = validate_batch_size(8)
        assert result == 8

    def test_validate_default_max(self) -> None:
        """Test validation with default maximum."""
        result = validate_batch_size(64)
        assert result == 64

    def test_validate_custom_max(self) -> None:
        """Test validation with custom maximum."""
        result = validate_batch_size(100, max_batch_size=128)
        assert result == 100

    def test_validate_exceeds_max(self) -> None:
        """Test validation fails when exceeding maximum."""
        with pytest.raises(ValidationError):
            validate_batch_size(100, max_batch_size=64)

    def test_validate_zero_batch_size(self) -> None:
        """Test validation fails for zero batch size."""
        with pytest.raises(ValidationError):
            validate_batch_size(0)


class TestValidatePaddingRatio:
    """Test suite for validate_padding_ratio function."""

    def test_validate_valid_padding(self) -> None:
        """Test validation of valid padding ratio."""
        result = validate_padding_ratio(0.15)
        assert result == 0.15

    def test_validate_zero_padding(self) -> None:
        """Test validation of zero padding."""
        result = validate_padding_ratio(0.0)
        assert result == 0.0

    def test_validate_max_padding(self) -> None:
        """Test validation at maximum padding."""
        result = validate_padding_ratio(2.0)
        assert result == 2.0

    def test_validate_negative_padding(self) -> None:
        """Test validation fails for negative padding."""
        with pytest.raises(ValidationError):
            validate_padding_ratio(-0.1)

    def test_validate_excessive_padding(self) -> None:
        """Test validation fails for excessive padding."""
        with pytest.raises(ValidationError):
            validate_padding_ratio(2.5)


class TestValidateModelPath:
    """Test suite for validate_model_path function."""

    def test_validate_pt_model(self, tmp_path: Path) -> None:
        """Test validation of .pt model file."""
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake model data")

        result = validate_model_path(model_file)
        
        assert result == model_file.resolve()

    def test_validate_pth_model(self, tmp_path: Path) -> None:
        """Test validation of .pth model file."""
        model_file = tmp_path / "model.pth"
        model_file.write_bytes(b"fake model data")

        result = validate_model_path(model_file)
        
        assert result == model_file.resolve()

    def test_validate_onnx_model(self, tmp_path: Path) -> None:
        """Test validation of .onnx model file."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake model data")

        result = validate_model_path(model_file)
        
        assert result == model_file.resolve()

    def test_validate_unusual_extension(self, tmp_path: Path) -> None:
        """Test validation logs warning for unusual extension."""
        model_file = tmp_path / "model.txt"
        model_file.write_bytes(b"fake model data")

        with patch("utils.validators.logger") as mock_logger:
            result = validate_model_path(model_file)
            
            assert result == model_file.resolve()
            mock_logger.warning.assert_called()

    def test_validate_nonexistent_model(self, tmp_path: Path) -> None:
        """Test validation fails for nonexistent model."""
        model_file = tmp_path / "nonexistent.pt"

        with pytest.raises(ValidationError, match="not found"):
            validate_model_path(model_file)


class TestConstants:
    """Test suite for module constants."""

    def test_supported_image_formats_is_set(self) -> None:
        """Test SUPPORTED_IMAGE_FORMATS is a set."""
        assert isinstance(SUPPORTED_IMAGE_FORMATS, set)

    def test_supported_formats_has_common_formats(self) -> None:
        """Test SUPPORTED_IMAGE_FORMATS includes common formats."""
        assert ".jpg" in SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in SUPPORTED_IMAGE_FORMATS
        assert ".png" in SUPPORTED_IMAGE_FORMATS

    def test_supported_formats_lowercase(self) -> None:
        """Test all formats are lowercase."""
        for fmt in SUPPORTED_IMAGE_FORMATS:
            assert fmt == fmt.lower()
            assert fmt.startswith(".")


class TestIntegration:
    """Integration tests for validators module."""

    def test_full_image_validation_workflow(self, tmp_path: Path) -> None:
        """Test complete image validation workflow."""
        # Create test image
        test_file = tmp_path / "test.jpg"
        img = Image.new("RGB", (640, 480), color="green")
        img.save(test_file)

        # Validate format
        path1 = validate_image_format(test_file)
        assert path1.exists()

        # Validate readable
        path2, dimensions = validate_image_readable(test_file)
        assert dimensions == (640, 480)

        # Validate dimensions
        validate_image_dimensions(dimensions, min_width=320, max_width=1920)

    def test_full_output_validation_workflow(self, tmp_path: Path) -> None:
        """Test complete output path validation workflow."""
        output_dir = tmp_path / "output"
        
        # Validate output directory
        validate_directory_writable(output_dir, create=True)
        assert output_dir.exists()

        # Validate output file path
        output_file = output_dir / "result.csv"
        validated_path = validate_output_path(output_file)
        assert validated_path.parent == output_dir

    def test_parameter_validation_chain(self) -> None:
        """Test chaining multiple parameter validations."""
        # Confidence threshold
        confidence = validate_confidence_threshold(0.75)
        assert 0.0 <= confidence <= 1.0

        # IoU threshold
        iou = validate_iou_threshold(0.45)
        assert 0.0 <= iou <= 1.0

        # Batch size
        batch = validate_batch_size(16)
        assert 1 <= batch <= 64

        # Padding
        padding = validate_padding_ratio(0.15)
        assert 0.0 <= padding <= 2.0
