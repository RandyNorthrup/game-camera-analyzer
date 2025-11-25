"""
Image processing utilities for the Game Camera Analyzer application.

This module provides:
- Image loading with format conversion
- EXIF metadata extraction
- Image preprocessing (resize, normalize)
- Thumbnail generation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    """Exception raised when image cannot be loaded."""

    pass


def load_image(
    path: Union[str, Path], color_mode: str = "BGR"
) -> Tuple[npt.NDArray[Any], Dict[str, Any]]:
    """
    Load an image and extract metadata.

    Args:
        path: Path to image file
        color_mode: Color mode for output image ("BGR", "RGB", "GRAY")

    Returns:
        Tuple of (image array, metadata dict)

    Raises:
        ImageLoadError: If image cannot be loaded
        ValueError: If color_mode is invalid
    """
    valid_modes = {"BGR", "RGB", "GRAY"}
    if color_mode not in valid_modes:
        raise ValueError(f"color_mode must be one of {valid_modes}, got '{color_mode}'")

    path_obj = Path(path)
    if not path_obj.exists():
        raise ImageLoadError(f"Image file not found: {path_obj}")

    try:
        # Load image with OpenCV (fastest)
        img = cv2.imread(str(path_obj))
        if img is None:
            # Fall back to PIL
            with Image.open(path_obj) as pil_img:
                img = np.array(pil_img)
                # PIL loads as RGB, convert if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    if color_mode == "BGR":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif color_mode == "GRAY":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif len(img.shape) == 2:
                    # Already grayscale
                    if color_mode in ("BGR", "RGB"):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        if color_mode == "RGB":
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # OpenCV loaded as BGR, convert if needed
            if color_mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif color_mode == "GRAY":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img is None or img.size == 0:
            raise ImageLoadError(f"Loaded image is empty: {path_obj}")

        # Extract metadata
        metadata = extract_metadata(path_obj)

        logger.debug(f"Loaded image: {path_obj}, shape={img.shape}, mode={color_mode}")
        return img, metadata

    except ImageLoadError:
        raise
    except Exception as e:
        raise ImageLoadError(f"Failed to load image '{path_obj}': {e}") from e


def extract_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract EXIF metadata from an image.

    Args:
        path: Path to image file

    Returns:
        Dictionary with metadata fields:
        - timestamp: datetime object or None
        - camera_make: str or None
        - camera_model: str or None
        - gps_latitude: float or None
        - gps_longitude: float or None
        - gps_altitude: float or None
        - width: int
        - height: int
        - orientation: int or None
    """
    path_obj = Path(path)
    metadata: Dict[str, Any] = {
        "timestamp": None,
        "camera_make": None,
        "camera_model": None,
        "gps_latitude": None,
        "gps_longitude": None,
        "gps_altitude": None,
        "width": None,
        "height": None,
        "orientation": None,
    }

    try:
        with Image.open(path_obj) as img:
            metadata["width"] = img.width
            metadata["height"] = img.height

            # Extract EXIF data
            exif_data = img._getexif()
            if exif_data is None:
                logger.debug(f"No EXIF data in image: {path_obj}")
                return metadata

            # Parse EXIF tags
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)

                if tag_name == "DateTime" or tag_name == "DateTimeOriginal":
                    try:
                        # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                        metadata["timestamp"] = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not parse datetime '{value}': {e}")

                elif tag_name == "Make":
                    metadata["camera_make"] = str(value).strip()

                elif tag_name == "Model":
                    metadata["camera_model"] = str(value).strip()

                elif tag_name == "Orientation":
                    metadata["orientation"] = int(value)

                elif tag_name == "GPSInfo" and isinstance(value, dict):
                    # Parse GPS data
                    gps_data = {}
                    for gps_tag_id, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag_name] = gps_value

                    # Extract GPS coordinates
                    if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
                        lat = _convert_gps_to_degrees(
                            gps_data["GPSLatitude"], gps_data["GPSLatitudeRef"]
                        )
                        metadata["gps_latitude"] = lat

                    if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
                        lon = _convert_gps_to_degrees(
                            gps_data["GPSLongitude"], gps_data["GPSLongitudeRef"]
                        )
                        metadata["gps_longitude"] = lon

                    if "GPSAltitude" in gps_data:
                        alt = gps_data["GPSAltitude"]
                        if isinstance(alt, tuple):
                            metadata["gps_altitude"] = float(alt[0]) / float(alt[1])
                        else:
                            metadata["gps_altitude"] = float(alt)

            logger.debug(f"Extracted metadata from {path_obj}: {metadata}")

    except Exception as e:
        logger.warning(f"Could not extract metadata from '{path_obj}': {e}")

    return metadata


def _convert_gps_to_degrees(gps_coords: Tuple[Any, Any, Any], ref: str) -> Optional[float]:
    """
    Convert GPS coordinates from EXIF format to decimal degrees.

    Args:
        gps_coords: Tuple of (degrees, minutes, seconds)
        ref: Reference direction ('N', 'S', 'E', 'W')

    Returns:
        Decimal degrees or None if conversion fails
    """
    try:
        # Handle tuple format (numerator, denominator)
        def to_float(val: Any) -> float:
            if isinstance(val, tuple) and len(val) == 2:
                return float(val[0]) / float(val[1])
            return float(val)

        degrees = to_float(gps_coords[0])
        minutes = to_float(gps_coords[1])
        seconds = to_float(gps_coords[2])

        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

        # Apply reference direction
        if ref in ("S", "W"):
            decimal = -decimal

        return decimal

    except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        logger.debug(f"Could not convert GPS coordinates: {e}")
        return None


def resize_image(
    img: npt.NDArray[Any],
    target_size: Union[int, Tuple[int, int]],
    maintain_aspect: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> npt.NDArray[Any]:
    """
    Resize an image.

    Args:
        img: Input image array
        target_size: Target size as int (max dimension) or (width, height)
        maintain_aspect: If True, maintain aspect ratio
        interpolation: OpenCV interpolation method

    Returns:
        Resized image array

    Raises:
        ValueError: If target_size is invalid
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty")

    h, w = img.shape[:2]

    if isinstance(target_size, int):
        # Resize based on maximum dimension
        if maintain_aspect:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w = new_h = target_size
    elif isinstance(target_size, tuple) and len(target_size) == 2:
        new_w, new_h = target_size
        if not maintain_aspect:
            # Use exact dimensions
            pass
        else:
            # Fit within target dimensions while maintaining aspect
            scale = min(new_w / w, new_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
    else:
        raise ValueError(f"target_size must be int or (width, height) tuple, got {target_size}")

    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Invalid target dimensions: {new_w}x{new_h}")

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    return resized


def normalize_image(
    img: npt.NDArray[Any], mean: Tuple[float, float, float], std: Tuple[float, float, float]
) -> npt.NDArray[Any]:
    """
    Normalize image using mean and standard deviation.

    Args:
        img: Input image array (BGR or RGB)
        mean: Mean values for each channel
        std: Standard deviation values for each channel

    Returns:
        Normalized image array (float32)
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must have 3 channels, got shape {img.shape}")

    # Convert to float32
    img_float = img.astype(np.float32) / 255.0

    # Normalize
    normalized: npt.NDArray[Any] = (img_float - np.array(mean)) / np.array(std)

    logger.debug("Normalized image")
    return normalized


def create_thumbnail(
    img: npt.NDArray[Any], size: int = 256, output_path: Optional[Path] = None
) -> npt.NDArray[Any]:
    """
    Create a thumbnail from an image.

    Args:
        img: Input image array
        size: Maximum dimension for thumbnail
        output_path: Optional path to save thumbnail

    Returns:
        Thumbnail image array

    Raises:
        ValueError: If input is invalid
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty")

    # Resize while maintaining aspect ratio
    thumbnail = resize_image(img, size, maintain_aspect=True, interpolation=cv2.INTER_AREA)

    # Save if output path provided
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), thumbnail)
            if not success:
                logger.warning(f"Could not save thumbnail to {output_path}")
            else:
                logger.debug(f"Saved thumbnail to {output_path}")
        except Exception as e:
            logger.warning(f"Error saving thumbnail: {e}")

    return thumbnail


def crop_bbox(
    img: npt.NDArray[Any],
    bbox: Tuple[int, int, int, int],
    padding: float = 0.0,
    square: bool = False,
) -> npt.NDArray[Any]:
    """
    Crop a region from an image with optional padding.

    Args:
        img: Input image array
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Padding ratio (0.0 = no padding, 0.5 = 50% padding)
        square: If True, make crop square by expanding to largest dimension

    Returns:
        Cropped image array

    Raises:
        ValueError: If bbox is invalid
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Validate bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {bbox}")

    # Calculate crop dimensions
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Apply padding
    if padding > 0:
        pad_w = int(bbox_w * padding)
        pad_h = int(bbox_h * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

    # Make square if requested
    if square:
        crop_w = x2 - x1
        crop_h = y2 - y1
        max_dim = max(crop_w, crop_h)

        # Expand to square centered on original crop
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        x1 = max(0, center_x - max_dim // 2)
        y1 = max(0, center_y - max_dim // 2)
        x2 = min(w, x1 + max_dim)
        y2 = min(h, y1 + max_dim)

        # Adjust if we hit image boundary
        if x2 - x1 < max_dim:
            x1 = max(0, x2 - max_dim)
        if y2 - y1 < max_dim:
            y1 = max(0, y2 - max_dim)

    # Crop
    cropped = img[y1:y2, x1:x2]

    logger.debug(f"Cropped bbox {bbox} -> {(x1, y1, x2, y2)}, shape={cropped.shape}")
    return cropped


def save_image(img: npt.NDArray[Any], output_path: Union[str, Path], quality: int = 95) -> None:
    """
    Save an image to disk.

    Args:
        img: Image array to save
        output_path: Output file path
        quality: JPEG quality (1-100, higher is better)

    Raises:
        ValueError: If image is invalid
        OSError: If file cannot be written
    """
    if img is None or img.size == 0:
        raise ValueError("Cannot save empty image")

    path_obj = Path(output_path)

    # Create parent directory if needed
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise OSError(f"Cannot create output directory: {e}") from e

    # Determine save parameters based on format
    params = []
    if path_obj.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path_obj.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    # Save image
    try:
        success = cv2.imwrite(str(path_obj), img, params)
        if not success:
            raise OSError(f"cv2.imwrite returned False for {path_obj}")
        logger.debug(f"Saved image to {path_obj}")
    except Exception as e:
        raise OSError(f"Failed to save image to '{path_obj}': {e}") from e


if __name__ == "__main__":
    """Test image utilities."""
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing image utilities...")

    # Create a test image
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_img_path = tmpdir_path / "test.jpg"

        # Create a simple test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img_path), test_img)

        # Test load_image
        try:
            img, metadata = load_image(test_img_path, color_mode="BGR")
            print(f"✅ Loaded image: shape={img.shape}, metadata keys={list(metadata.keys())}")
        except Exception as e:
            print(f"❌ Failed to load image: {e}")

        # Test resize_image
        try:
            resized = resize_image(img, 320, maintain_aspect=True)
            print(f"✅ Resized image: {img.shape} -> {resized.shape}")
        except Exception as e:
            print(f"❌ Failed to resize: {e}")

        # Test create_thumbnail
        try:
            thumbnail_path = tmpdir_path / "thumb.jpg"
            thumbnail = create_thumbnail(img, size=128, output_path=thumbnail_path)
            print(f"✅ Created thumbnail: {thumbnail.shape}, saved={thumbnail_path.exists()}")
        except Exception as e:
            print(f"❌ Failed to create thumbnail: {e}")

        # Test crop_bbox
        try:
            bbox = (100, 100, 300, 300)
            cropped = crop_bbox(img, bbox, padding=0.1, square=False)
            print(f"✅ Cropped bbox {bbox}: shape={cropped.shape}")
        except Exception as e:
            print(f"❌ Failed to crop: {e}")

        # Test save_image
        try:
            output_path = tmpdir_path / "output.jpg"
            save_image(cropped, output_path, quality=90)
            print(f"✅ Saved image: {output_path.exists()}")
        except Exception as e:
            print(f"❌ Failed to save: {e}")

    print("\nAll image utility tests completed!")
