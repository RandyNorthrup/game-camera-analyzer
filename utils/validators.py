"""
Input validation utilities for the Game Camera Analyzer application.

This module provides comprehensive validation functions for:
- File paths (existence, readability, writability)
- Image files (format, dimensions, corruption)
- Parameter values (ranges, types, allowed values)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ValidationError(Exception):
    """Exception raised when validation fails."""

    pass


def validate_file_exists(path: Union[str, Path], file_type: str = "file") -> Path:
    """
    Validate that a file or directory exists.

    Args:
        path: Path to validate
        file_type: Type description for error messages (e.g., "image", "config")

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path does not exist or is not accessible
    """
    try:
        path_obj = Path(path).expanduser().resolve()
    except (ValueError, RuntimeError) as e:
        raise ValidationError(f"Invalid {file_type} path '{path}': {e}") from e

    if not path_obj.exists():
        raise ValidationError(f"{file_type.capitalize()} not found: {path_obj}")

    logger.debug(f"Validated {file_type} exists: {path_obj}")
    return path_obj


def validate_file_readable(path: Union[str, Path], file_type: str = "file") -> Path:
    """
    Validate that a file is readable.

    Args:
        path: Path to validate
        file_type: Type description for error messages

    Returns:
        Validated Path object

    Raises:
        ValidationError: If file is not readable
    """
    path_obj = validate_file_exists(path, file_type)

    if not os.access(path_obj, os.R_OK):
        raise ValidationError(f"{file_type.capitalize()} is not readable: {path_obj}")

    logger.debug(f"Validated {file_type} is readable: {path_obj}")
    return path_obj


def validate_directory_writable(path: Union[str, Path], create: bool = False) -> Path:
    """
    Validate that a directory is writable.

    Args:
        path: Directory path to validate
        create: If True, create directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If directory is not writable or cannot be created
    """
    try:
        path_obj = Path(path).expanduser().resolve()
    except (ValueError, RuntimeError) as e:
        raise ValidationError(f"Invalid directory path '{path}': {e}") from e

    if not path_obj.exists():
        if create:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path_obj}")
            except (OSError, PermissionError) as e:
                raise ValidationError(f"Cannot create directory '{path_obj}': {e}") from e
        else:
            raise ValidationError(f"Directory not found: {path_obj}")

    if not path_obj.is_dir():
        raise ValidationError(f"Path is not a directory: {path_obj}")

    if not os.access(path_obj, os.W_OK):
        raise ValidationError(f"Directory is not writable: {path_obj}")

    logger.debug(f"Validated directory is writable: {path_obj}")
    return path_obj


def validate_output_path(
    path: Union[str, Path], overwrite: bool = False, file_type: str = "file"
) -> Path:
    """
    Validate an output file path.

    Args:
        path: Output path to validate
        overwrite: If True, allow overwriting existing files
        file_type: Type description for error messages

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path exists and overwrite is False, or parent directory not writable
    """
    try:
        path_obj = Path(path).expanduser().resolve()
    except (ValueError, RuntimeError) as e:
        raise ValidationError(f"Invalid output {file_type} path '{path}': {e}") from e

    if path_obj.exists() and not overwrite:
        raise ValidationError(f"Output {file_type} already exists (overwrite=False): {path_obj}")

    # Validate parent directory is writable
    parent_dir = path_obj.parent
    if not parent_dir.exists():
        raise ValidationError(f"Parent directory does not exist: {parent_dir}")

    if not os.access(parent_dir, os.W_OK):
        raise ValidationError(f"Parent directory is not writable: {parent_dir}")

    logger.debug(f"Validated output {file_type} path: {path_obj}")
    return path_obj


def validate_image_format(path: Union[str, Path]) -> Path:
    """
    Validate that a file has a supported image format.

    Args:
        path: Path to image file

    Returns:
        Validated Path object

    Raises:
        ValidationError: If file format is not supported
    """
    path_obj = validate_file_exists(path, "image")

    suffix = path_obj.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise ValidationError(
            f"Unsupported image format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
        )

    logger.debug(f"Validated image format: {path_obj}")
    return path_obj


def validate_image_readable(
    path: Union[str, Path], check_corruption: bool = True
) -> Tuple[Path, Tuple[int, int]]:
    """
    Validate that an image file is readable and optionally check for corruption.

    Args:
        path: Path to image file
        check_corruption: If True, attempt to load image to check for corruption

    Returns:
        Tuple of (validated Path object, image dimensions (width, height))

    Raises:
        ValidationError: If image cannot be read or is corrupted
    """
    path_obj = validate_image_format(path)
    path_obj = validate_file_readable(path_obj, "image")

    if not check_corruption:
        # Quick validation using PIL without loading full image
        try:
            with Image.open(path_obj) as img:
                dimensions = img.size  # (width, height)
            logger.debug(f"Validated image (quick): {path_obj}, size={dimensions}")
            return path_obj, dimensions
        except Exception as e:
            raise ValidationError(f"Cannot read image '{path_obj}': {e}") from e

    # Full validation by loading image data
    try:
        # Try with OpenCV first (faster)
        img_array = cv2.imread(str(path_obj))
        if img_array is None:
            # Fall back to PIL
            with Image.open(path_obj) as img:
                img_array = np.array(img)

        if img_array is None or img_array.size == 0:
            raise ValidationError(f"Image is empty or corrupted: {path_obj}")

        # Get dimensions (height, width, channels) from numpy array
        if len(img_array.shape) == 2:
            height, width = img_array.shape
        else:
            height, width = img_array.shape[:2]

        dimensions = (width, height)
        logger.debug(f"Validated image (full): {path_obj}, size={dimensions}")
        return path_obj, dimensions

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Cannot load image '{path_obj}': {e}") from e


def validate_image_dimensions(
    dimensions: Tuple[int, int],
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> None:
    """
    Validate image dimensions against constraints.

    Args:
        dimensions: Image dimensions as (width, height)
        min_width: Minimum width in pixels
        min_height: Minimum height in pixels
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels

    Raises:
        ValidationError: If dimensions don't meet constraints
    """
    width, height = dimensions

    if min_width is not None and width < min_width:
        raise ValidationError(f"Image width {width} is less than minimum {min_width}")

    if min_height is not None and height < min_height:
        raise ValidationError(f"Image height {height} is less than minimum {min_height}")

    if max_width is not None and width > max_width:
        raise ValidationError(f"Image width {width} exceeds maximum {max_width}")

    if max_height is not None and height > max_height:
        raise ValidationError(f"Image height {height} exceeds maximum {max_height}")

    logger.debug(f"Validated image dimensions: {width}x{height}")


def validate_confidence_threshold(threshold: float) -> float:
    """
    Validate a confidence threshold value.

    Args:
        threshold: Confidence threshold (should be between 0 and 1)

    Returns:
        Validated threshold

    Raises:
        ValidationError: If threshold is not in valid range
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError(
            f"Confidence threshold must be numeric, got {type(threshold).__name__}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValidationError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")

    logger.debug(f"Validated confidence threshold: {threshold}")
    return float(threshold)


def validate_positive_integer(value: int, name: str, min_value: int = 1) -> int:
    """
    Validate a positive integer parameter.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value (default: 1)

    Returns:
        Validated integer

    Raises:
        ValidationError: If value is not a valid positive integer
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")

    if value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")

    logger.debug(f"Validated {name}: {value}")
    return value


def validate_in_range(
    value: Union[int, float],
    name: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> Union[int, float]:
    """
    Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not in range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")

    if not min_value <= value <= max_value:
        raise ValidationError(f"{name} must be between {min_value} and {max_value}, got {value}")

    logger.debug(f"Validated {name}: {value} (range: {min_value}-{max_value})")
    return value


def validate_in_choices(
    value: str, name: str, choices: List[str], case_sensitive: bool = True
) -> str:
    """
    Validate that a string value is one of the allowed choices.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        choices: List of allowed values
        case_sensitive: If False, perform case-insensitive comparison

    Returns:
        Validated value (original case preserved)

    Raises:
        ValidationError: If value is not in choices
    """
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")

    compare_value = value if case_sensitive else value.lower()
    compare_choices = choices if case_sensitive else [c.lower() for c in choices]

    if compare_value not in compare_choices:
        raise ValidationError(f"{name} must be one of {choices}, got '{value}'")

    logger.debug(f"Validated {name}: {value}")
    return value


def validate_batch_size(batch_size: int, max_batch_size: int = 64) -> int:
    """
    Validate batch size for processing.

    Args:
        batch_size: Batch size to validate
        max_batch_size: Maximum allowed batch size

    Returns:
        Validated batch size

    Raises:
        ValidationError: If batch size is invalid
    """
    result = validate_in_range(batch_size, "batch_size", 1, max_batch_size)
    return int(result)


def validate_padding_ratio(padding: float) -> float:
    """
    Validate padding ratio for cropping.

    Args:
        padding: Padding ratio (typically 0.0 to 1.0)

    Returns:
        Validated padding ratio

    Raises:
        ValidationError: If padding is invalid
    """
    return validate_in_range(padding, "padding_ratio", 0.0, 2.0)


def validate_model_path(path: Union[str, Path]) -> Path:
    """
    Validate a model file path.

    Args:
        path: Path to model file

    Returns:
        Validated Path object

    Raises:
        ValidationError: If model file is invalid
    """
    path_obj = validate_file_readable(path, "model")

    # Check file extension
    suffix = path_obj.suffix.lower()
    valid_extensions = {".pt", ".pth", ".onnx", ".pb", ".h5"}

    if suffix not in valid_extensions:
        logger.warning(
            f"Model file has unusual extension '{suffix}', " f"expected one of {valid_extensions}"
        )

    logger.debug(f"Validated model path: {path_obj}")
    return path_obj


if __name__ == "__main__":
    """Test validation functions."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing validators...")

    # Test file existence
    try:
        validate_file_exists("/nonexistent/path")
        print("❌ Should have raised ValidationError for nonexistent file")
    except ValidationError as e:
        print(f"✅ File existence validation: {e}")

    # Test confidence threshold
    try:
        validate_confidence_threshold(0.5)
        print("✅ Valid confidence threshold: 0.5")
    except ValidationError as e:
        print(f"❌ Unexpected error: {e}")

    try:
        validate_confidence_threshold(1.5)
        print("❌ Should have raised ValidationError for threshold > 1.0")
    except ValidationError as e:
        print(f"✅ Confidence threshold validation: {e}")

    # Test positive integer
    try:
        validate_positive_integer(10, "test_param")
        print("✅ Valid positive integer: 10")
    except ValidationError as e:
        print(f"❌ Unexpected error: {e}")

    try:
        validate_positive_integer(-5, "test_param")
        print("❌ Should have raised ValidationError for negative value")
    except ValidationError as e:
        print(f"✅ Positive integer validation: {e}")

    # Test in_choices
    try:
        validate_in_choices("yolov8", "model", ["yolov8", "faster-rcnn"])
        print("✅ Valid choice: 'yolov8'")
    except ValidationError as e:
        print(f"❌ Unexpected error: {e}")

    try:
        validate_in_choices("invalid", "model", ["yolov8", "faster-rcnn"])
        print("❌ Should have raised ValidationError for invalid choice")
    except ValidationError as e:
        print(f"✅ Choice validation: {e}")

    # Test directory creation
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "new_dir"
        try:
            validate_directory_writable(test_dir, create=True)
            print(f"✅ Directory created and validated: {test_dir}")
        except ValidationError as e:
            print(f"❌ Unexpected error: {e}")

    print("\nAll validator tests completed!")
