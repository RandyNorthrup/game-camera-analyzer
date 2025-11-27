"""
Comprehensive tests for utils/image_utils.py module.

Tests image loading, processing, metadata extraction, and saving.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from utils.image_utils import (
    ImageLoadError,
    _convert_gps_to_degrees,
    create_thumbnail,
    crop_bbox,
    extract_metadata,
    load_image,
    normalize_image,
    resize_image,
    save_image,
)


class TestImageLoadError:
    """Test suite for ImageLoadError exception."""

    def test_image_load_error_is_exception(self) -> None:
        """Test ImageLoadError is an Exception."""
        assert issubclass(ImageLoadError, Exception)

    def test_image_load_error_with_message(self) -> None:
        """Test ImageLoadError with message."""
        error = ImageLoadError("Failed to load")
        assert str(error) == "Failed to load"


class TestLoadImage:
    """Test suite for load_image function."""

    def test_load_valid_image_bgr(self, tmp_path: Path) -> None:
        """Test loading valid image in BGR mode."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        img, metadata = load_image(img_path, color_mode="BGR")

        assert img.shape == (100, 150, 3)
        assert metadata["width"] == 150
        assert metadata["height"] == 100

    def test_load_valid_image_rgb(self, tmp_path: Path) -> None:
        """Test loading valid image in RGB mode."""
        img_path = tmp_path / "test.png"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        img, metadata = load_image(img_path, color_mode="RGB")

        assert img.shape == (100, 150, 3)
        assert isinstance(metadata, dict)

    def test_load_valid_image_gray(self, tmp_path: Path) -> None:
        """Test loading image in grayscale mode."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        img, metadata = load_image(img_path, color_mode="GRAY")

        assert len(img.shape) == 2
        assert img.shape == (100, 150)

    def test_load_image_invalid_color_mode(self, tmp_path: Path) -> None:
        """Test loading fails with invalid color mode."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        with pytest.raises(ValueError, match="color_mode must be one of"):
            load_image(img_path, color_mode="INVALID")

    def test_load_nonexistent_image(self, tmp_path: Path) -> None:
        """Test loading fails for nonexistent file."""
        img_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(ImageLoadError, match="not found"):
            load_image(img_path)

    def test_load_image_fallback_to_pil(self, tmp_path: Path) -> None:
        """Test fallback to PIL when OpenCV fails."""
        img_path = tmp_path / "test.png"
        pil_img = Image.new("RGB", (100, 150), color=(255, 0, 0))
        pil_img.save(img_path)

        with patch("cv2.imread", return_value=None):
            img, metadata = load_image(img_path, color_mode="RGB")

            assert img.shape == (150, 100, 3)

    def test_load_grayscale_to_bgr(self, tmp_path: Path) -> None:
        """Test loading grayscale image and converting to BGR."""
        img_path = tmp_path / "gray.png"
        gray_img = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        cv2.imwrite(str(img_path), gray_img)

        with patch("cv2.imread", return_value=None):
            pil_img = Image.fromarray(gray_img, mode="L")
            pil_img.save(img_path)
            
            img, metadata = load_image(img_path, color_mode="BGR")
            
            assert img.shape == (100, 150, 3)

    def test_load_empty_image(self, tmp_path: Path) -> None:
        """Test loading fails for empty image."""
        img_path = tmp_path / "empty.jpg"
        img_path.write_bytes(b"")

        with pytest.raises(ImageLoadError):
            load_image(img_path)

    def test_load_corrupted_image(self, tmp_path: Path) -> None:
        """Test loading fails for corrupted image."""
        img_path = tmp_path / "corrupted.jpg"
        img_path.write_bytes(b"not a valid image file")

        with pytest.raises(ImageLoadError, match="Failed to load"):
            load_image(img_path)

    def test_load_image_string_path(self, tmp_path: Path) -> None:
        """Test loading with string path."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        img, metadata = load_image(str(img_path), color_mode="BGR")

        assert img.shape == (100, 150, 3)

    def test_load_pil_fallback_bgr(self, tmp_path: Path) -> None:
        """Test PIL fallback loading RGB image with BGR conversion."""
        img_path = tmp_path / "test.png"
        pil_img = Image.new("RGB", (100, 150), color=(255, 0, 0))
        pil_img.save(img_path)

        with patch("cv2.imread", return_value=None):
            img, metadata = load_image(img_path, color_mode="BGR")
            
            assert img.shape == (150, 100, 3)
            # Red in RGB should become blue channel in BGR
            assert img[0, 0, 2] > 200  # Blue channel high

    def test_load_pil_fallback_gray(self, tmp_path: Path) -> None:
        """Test PIL fallback loading RGB image with grayscale conversion."""
        img_path = tmp_path / "test.png"
        pil_img = Image.new("RGB", (100, 150), color=(128, 128, 128))
        pil_img.save(img_path)

        with patch("cv2.imread", return_value=None):
            img, metadata = load_image(img_path, color_mode="GRAY")
            
            assert len(img.shape) == 2
            assert img.shape == (150, 100)

    def test_load_pil_grayscale_to_bgr(self, tmp_path: Path) -> None:
        """Test PIL fallback loading grayscale image and converting to BGR."""
        img_path = tmp_path / "gray.png"
        pil_img = Image.new("L", (100, 150), color=128)
        pil_img.save(img_path)

        with patch("cv2.imread", return_value=None):
            img, metadata = load_image(img_path, color_mode="BGR")
            
            assert img.shape == (150, 100, 3)
            # All channels should be equal for grayscale
            assert np.allclose(img[:, :, 0], img[:, :, 1])
            assert np.allclose(img[:, :, 1], img[:, :, 2])

    def test_load_pil_grayscale_to_rgb(self, tmp_path: Path) -> None:
        """Test PIL fallback loading grayscale image and converting to RGB."""
        img_path = tmp_path / "gray.png"
        pil_img = Image.new("L", (100, 150), color=128)
        pil_img.save(img_path)

        with patch("cv2.imread", return_value=None):
            img, metadata = load_image(img_path, color_mode="RGB")
            
            assert img.shape == (150, 100, 3)
            # All channels should be equal for grayscale
            assert np.allclose(img[:, :, 0], img[:, :, 1])

    def test_load_opencv_rgb_conversion(self, tmp_path: Path) -> None:
        """Test OpenCV loading with RGB conversion."""
        img_path = tmp_path / "test.jpg"
        # Create BGR image (OpenCV default)
        test_img = np.zeros((100, 150, 3), dtype=np.uint8)
        test_img[:, :, 2] = 255  # Red channel in BGR
        cv2.imwrite(str(img_path), test_img)

        img, metadata = load_image(img_path, color_mode="RGB")
        
        # In RGB, red should be in channel 0
        assert img[0, 0, 0] > 200

    def test_load_opencv_gray_conversion(self, tmp_path: Path) -> None:
        """Test OpenCV loading with grayscale conversion."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        img, metadata = load_image(img_path, color_mode="GRAY")
        
        assert len(img.shape) == 2


class TestExtractMetadata:
    """Test suite for extract_metadata function."""

    def test_extract_metadata_no_exif(self, tmp_path: Path) -> None:
        """Test metadata extraction from image without EXIF."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        metadata = extract_metadata(img_path)

        assert metadata["width"] == 150
        assert metadata["height"] == 100
        assert metadata["timestamp"] is None
        assert metadata["camera_make"] is None

    def test_extract_metadata_with_datetime(self, tmp_path: Path) -> None:
        """Test metadata extraction with datetime."""
        img_path = tmp_path / "test.jpg"
        pil_img = Image.new("RGB", (100, 150))
        
        # Create EXIF with datetime
        from PIL.ExifTags import TAGS
        exif = pil_img.getexif()
        datetime_tag = [k for k, v in TAGS.items() if v == "DateTime"][0]
        exif[datetime_tag] = "2024:01:15 14:30:45"
        
        pil_img.save(img_path, exif=exif)

        metadata = extract_metadata(img_path)

        assert metadata["width"] == 100
        assert metadata["height"] == 150
        assert isinstance(metadata["timestamp"], datetime)
        assert metadata["timestamp"].year == 2024

    def test_extract_metadata_with_camera_info(self, tmp_path: Path) -> None:
        """Test metadata extraction with camera information."""
        img_path = tmp_path / "test.jpg"
        pil_img = Image.new("RGB", (100, 150))
        
        from PIL.ExifTags import TAGS
        exif = pil_img.getexif()
        make_tag = [k for k, v in TAGS.items() if v == "Make"][0]
        model_tag = [k for k, v in TAGS.items() if v == "Model"][0]
        exif[make_tag] = "Canon"
        exif[model_tag] = "EOS 5D"
        
        pil_img.save(img_path, exif=exif)

        metadata = extract_metadata(img_path)

        assert metadata["camera_make"] == "Canon"
        assert metadata["camera_model"] == "EOS 5D"

    def test_extract_metadata_with_orientation(self, tmp_path: Path) -> None:
        """Test metadata extraction with orientation."""
        img_path = tmp_path / "test.jpg"
        pil_img = Image.new("RGB", (100, 150))
        
        from PIL.ExifTags import TAGS
        exif = pil_img.getexif()
        orient_tag = [k for k, v in TAGS.items() if v == "Orientation"][0]
        exif[orient_tag] = 6
        
        pil_img.save(img_path, exif=exif)

        metadata = extract_metadata(img_path)

        assert metadata["orientation"] == 6

    def test_extract_metadata_invalid_datetime(self, tmp_path: Path) -> None:
        """Test metadata extraction with invalid datetime."""
        img_path = tmp_path / "test.jpg"
        pil_img = Image.new("RGB", (100, 150))
        
        from PIL.ExifTags import TAGS
        exif = pil_img.getexif()
        datetime_tag = [k for k, v in TAGS.items() if v == "DateTime"][0]
        exif[datetime_tag] = "invalid datetime"
        
        pil_img.save(img_path, exif=exif)

        metadata = extract_metadata(img_path)

        assert metadata["timestamp"] is None

    def test_extract_metadata_file_error(self, tmp_path: Path) -> None:
        """Test metadata extraction handles file errors."""
        img_path = tmp_path / "nonexistent.jpg"

        metadata = extract_metadata(img_path)

        # Should return empty metadata without raising
        assert metadata["width"] is None
        assert metadata["height"] is None


class TestExtractMetadataGPS:
    """Test suite for GPS metadata extraction."""

    def test_extract_metadata_with_gps_latitude(self, tmp_path: Path) -> None:
        """Test metadata extraction with GPS latitude."""
        img_path = tmp_path / "test.jpg"
        
        # Mock Image.open to return GPS data
        with patch("utils.image_utils.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 150
            
            gps_info = {
                1: "N",  # GPSLatitudeRef
                2: ((40, 1), (42, 1), (51, 1)),  # GPSLatitude
            }
            
            mock_exif = {
                34853: gps_info  # GPSInfo tag
            }
            mock_img._getexif.return_value = mock_exif
            mock_open.return_value.__enter__.return_value = mock_img
            
            metadata = extract_metadata(img_path)
            
            assert "gps_latitude" in metadata
            assert metadata["gps_latitude"] is not None
            assert abs(metadata["gps_latitude"] - 40.714167) < 0.001

    def test_extract_metadata_with_gps_longitude(self, tmp_path: Path) -> None:
        """Test metadata extraction with GPS longitude."""
        img_path = tmp_path / "test.jpg"
        
        with patch("utils.image_utils.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 150
            
            gps_info = {
                3: "W",  # GPSLongitudeRef
                4: ((74, 1), (0, 1), (21, 1)),  # GPSLongitude
            }
            
            mock_exif = {
                34853: gps_info  # GPSInfo tag
            }
            mock_img._getexif.return_value = mock_exif
            mock_open.return_value.__enter__.return_value = mock_img
            
            metadata = extract_metadata(img_path)
            
            assert "gps_longitude" in metadata
            assert metadata["gps_longitude"] is not None
            assert metadata["gps_longitude"] < 0  # West is negative

    def test_extract_metadata_with_gps_altitude_tuple(self, tmp_path: Path) -> None:
        """Test metadata extraction with GPS altitude as tuple."""
        img_path = tmp_path / "test.jpg"
        
        with patch("utils.image_utils.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 150
            
            gps_info = {
                6: (12345, 100),  # GPSAltitude as tuple (numerator, denominator)
            }
            
            mock_exif = {
                34853: gps_info  # GPSInfo tag
            }
            mock_img._getexif.return_value = mock_exif
            mock_open.return_value.__enter__.return_value = mock_img
            
            metadata = extract_metadata(img_path)
            
            assert "gps_altitude" in metadata
            assert metadata["gps_altitude"] == 123.45

    def test_extract_metadata_with_gps_altitude_float(self, tmp_path: Path) -> None:
        """Test metadata extraction with GPS altitude as float."""
        img_path = tmp_path / "test.jpg"
        
        with patch("utils.image_utils.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 150
            
            gps_info = {
                6: 123.45,  # GPSAltitude as float
            }
            
            mock_exif = {
                34853: gps_info  # GPSInfo tag
            }
            mock_img._getexif.return_value = mock_exif
            mock_open.return_value.__enter__.return_value = mock_img
            
            metadata = extract_metadata(img_path)
            
            assert "gps_altitude" in metadata
            assert metadata["gps_altitude"] == 123.45

    def test_extract_metadata_logs_debug(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test metadata extraction logs debug information."""
        img_path = tmp_path / "test.jpg"
        
        with patch("utils.image_utils.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 150
            
            # Add some EXIF to trigger the debug log at line 185
            mock_exif = {
                271: "Canon",  # Make tag
            }
            mock_img._getexif.return_value = mock_exif
            mock_open.return_value.__enter__.return_value = mock_img
            
            with caplog.at_level(logging.DEBUG):
                metadata = extract_metadata(img_path)
            
            # Check that debug log was generated (line 185)
            assert any("Extracted metadata from" in record.message for record in caplog.records)


class TestLoadImageDebugLogging:
    """Test suite for load_image debug logging."""

    def test_load_image_logs_debug(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test load_image logs debug information."""
        img_path = tmp_path / "test.jpg"
        test_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)
        
        with caplog.at_level(logging.DEBUG):
            img, metadata = load_image(img_path)
            
        # Check that debug log was generated (line 83)
        assert any("Loaded image:" in record.message for record in caplog.records)

    def test_extract_metadata_logs_warning_on_exception(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test extract_metadata logs warning when exception occurs."""
        img_path = tmp_path / "test.jpg"
        
        with patch("utils.image_utils.Image.open", side_effect=Exception("Test error")):
            with caplog.at_level(logging.WARNING):
                metadata = extract_metadata(img_path)
                
            # Check that warning log was generated (line 188)
            assert any("Could not extract metadata" in record.message for record in caplog.records)


class TestConvertGpsToDegrees:
    """Test suite for _convert_gps_to_degrees function."""

    def test_convert_gps_north(self) -> None:
        """Test GPS conversion for northern latitude."""
        coords = ((40, 1), (42, 1), (51, 1))
        result = _convert_gps_to_degrees(coords, "N")

        assert result is not None
        assert abs(result - 40.714167) < 0.001

    def test_convert_gps_south(self) -> None:
        """Test GPS conversion for southern latitude."""
        coords = ((33, 1), (52, 1), (0, 1))
        result = _convert_gps_to_degrees(coords, "S")

        assert result is not None
        assert result < 0

    def test_convert_gps_east(self) -> None:
        """Test GPS conversion for eastern longitude."""
        coords = ((151, 1), (12, 1), (30, 1))
        result = _convert_gps_to_degrees(coords, "E")

        assert result is not None
        assert result > 0

    def test_convert_gps_west(self) -> None:
        """Test GPS conversion for western longitude."""
        coords = ((74, 1), (0, 1), (21, 1))
        result = _convert_gps_to_degrees(coords, "W")

        assert result is not None
        assert result < 0

    def test_convert_gps_tuple_format(self) -> None:
        """Test GPS conversion with tuple values."""
        coords = ((40, 1), (42, 1), (51, 1))
        result = _convert_gps_to_degrees(coords, "N")

        assert result is not None

    def test_convert_gps_invalid_coords(self) -> None:
        """Test GPS conversion handles invalid coordinates."""
        coords = (40, 42, "invalid")  # type: ignore
        result = _convert_gps_to_degrees(coords, "N")

        assert result is None

    def test_convert_gps_zero_division(self) -> None:
        """Test GPS conversion handles zero division."""
        coords = ((40, 0), (42, 1), (51, 1))
        result = _convert_gps_to_degrees(coords, "N")

        assert result is None


class TestResizeImage:
    """Test suite for resize_image function."""

    def test_resize_by_max_dimension(self) -> None:
        """Test resizing by maximum dimension."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized = resize_image(img, 320, maintain_aspect=True)

        assert max(resized.shape[:2]) == 320
        assert resized.shape[2] == 3

    def test_resize_to_exact_dimensions(self) -> None:
        """Test resizing to exact dimensions."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized = resize_image(img, (400, 300), maintain_aspect=False)

        assert resized.shape == (300, 400, 3)

    def test_resize_maintain_aspect_with_tuple(self) -> None:
        """Test resizing with aspect ratio maintained using tuple."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized = resize_image(img, (400, 300), maintain_aspect=True)

        # Should fit within 400x300
        assert resized.shape[1] <= 400
        assert resized.shape[0] <= 300

    def test_resize_empty_image(self) -> None:
        """Test resizing fails for empty image."""
        img = np.array([])

        with pytest.raises(ValueError, match="empty"):
            resize_image(img, 320)

    def test_resize_invalid_target_size(self) -> None:
        """Test resizing fails with invalid target size."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="target_size must be"):
            resize_image(img, "invalid")  # type: ignore

    def test_resize_negative_dimensions(self) -> None:
        """Test resizing fails with negative dimensions."""
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid target dimensions"):
            resize_image(img, 0)

    def test_resize_grayscale_image(self) -> None:
        """Test resizing grayscale image."""
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        resized = resize_image(img, 320, maintain_aspect=True)

        assert len(resized.shape) == 2
        assert max(resized.shape) == 320

    def test_resize_interpolation_method(self) -> None:
        """Test resizing with different interpolation."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized = resize_image(img, 320, interpolation=cv2.INTER_NEAREST)

        assert max(resized.shape[:2]) == 320


class TestNormalizeImage:
    """Test suite for normalize_image function."""

    def test_normalize_image(self) -> None:
        """Test image normalization."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalized = normalize_image(img, mean, std)

        # Should be float32 or float64
        assert normalized.dtype in (np.float32, np.float64)
        assert normalized.shape == img.shape

    def test_normalize_grayscale_fails(self) -> None:
        """Test normalization fails for grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        with pytest.raises(ValueError, match="must have 3 channels"):
            normalize_image(img, mean, std)

    def test_normalize_wrong_channels(self) -> None:
        """Test normalization fails for wrong number of channels."""
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        with pytest.raises(ValueError, match="must have 3 channels"):
            normalize_image(img, mean, std)

    def test_normalize_values_range(self) -> None:
        """Test normalized values are in expected range."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        normalized = normalize_image(img, mean, std)

        # 128/255 = 0.502, (0.502 - 0.5) / 0.5 ≈ 0.004
        assert normalized.min() >= -10  # Reasonable range
        assert normalized.max() <= 10


class TestCreateThumbnail:
    """Test suite for create_thumbnail function."""

    def test_create_thumbnail_default_size(self) -> None:
        """Test creating thumbnail with default size."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        thumbnail = create_thumbnail(img)

        assert max(thumbnail.shape[:2]) == 256

    def test_create_thumbnail_custom_size(self) -> None:
        """Test creating thumbnail with custom size."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        thumbnail = create_thumbnail(img, size=128)

        assert max(thumbnail.shape[:2]) == 128

    def test_create_thumbnail_with_output(self, tmp_path: Path) -> None:
        """Test creating and saving thumbnail."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output_path = tmp_path / "thumb.jpg"

        thumbnail = create_thumbnail(img, size=128, output_path=output_path)

        assert max(thumbnail.shape[:2]) == 128
        assert output_path.exists()

    def test_create_thumbnail_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test thumbnail creation creates parent directory."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output_path = tmp_path / "subdir" / "thumb.jpg"

        thumbnail = create_thumbnail(img, size=128, output_path=output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_create_thumbnail_empty_image(self) -> None:
        """Test thumbnail creation fails for empty image."""
        img = np.array([])

        with pytest.raises(ValueError, match="empty"):
            create_thumbnail(img)

    def test_create_thumbnail_save_failure(self, tmp_path: Path) -> None:
        """Test thumbnail handles save failure gracefully."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output_path = tmp_path / "thumb.jpg"

        with patch("cv2.imwrite", return_value=False):
            thumbnail = create_thumbnail(img, output_path=output_path)
            
            # Should return thumbnail even if save fails
            assert thumbnail.shape[2] == 3

    def test_create_thumbnail_save_exception(self, tmp_path: Path) -> None:
        """Test thumbnail handles save exception."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output_path = tmp_path / "thumb.jpg"

        with patch("cv2.imwrite", side_effect=Exception("Write error")):
            thumbnail = create_thumbnail(img, output_path=output_path)
            
            assert thumbnail.shape[2] == 3


class TestCropBbox:
    """Test suite for crop_bbox function."""

    def test_crop_bbox_no_padding(self) -> None:
        """Test cropping without padding."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 300)

        cropped = crop_bbox(img, bbox, padding=0.0)

        assert cropped.shape == (200, 200, 3)

    def test_crop_bbox_with_padding(self) -> None:
        """Test cropping with padding."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (200, 200, 300, 300)

        cropped = crop_bbox(img, bbox, padding=0.5)

        # Should be larger than original bbox
        assert cropped.shape[0] > 100
        assert cropped.shape[1] > 100

    def test_crop_bbox_square(self) -> None:
        """Test cropping to square."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 250)  # Non-square

        cropped = crop_bbox(img, bbox, square=True)

        assert cropped.shape[0] == cropped.shape[1]

    def test_crop_bbox_empty_image(self) -> None:
        """Test cropping fails for empty image."""
        img = np.array([])
        bbox = (100, 100, 200, 200)

        with pytest.raises(ValueError, match="empty"):
            crop_bbox(img, bbox)

    def test_crop_bbox_invalid_bbox(self) -> None:
        """Test cropping fails for invalid bbox."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (300, 300, 100, 100)  # x2 < x1, y2 < y1

        with pytest.raises(ValueError, match="Invalid bounding box"):
            crop_bbox(img, bbox)

    def test_crop_bbox_at_boundary(self) -> None:
        """Test cropping at image boundary."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (0, 0, 100, 100)

        cropped = crop_bbox(img, bbox, padding=0.5)

        # Should not exceed image boundaries
        assert cropped.shape[0] <= 480
        assert cropped.shape[1] <= 640

    def test_crop_bbox_grayscale(self) -> None:
        """Test cropping grayscale image."""
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        bbox = (100, 100, 300, 300)

        cropped = crop_bbox(img, bbox)

        assert len(cropped.shape) == 2
        assert cropped.shape == (200, 200)

    def test_crop_bbox_square_at_right_edge(self) -> None:
        """Test square cropping when bbox is at right edge of image."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Bbox near right edge that will need adjustment
        bbox = (500, 100, 630, 300)

        cropped = crop_bbox(img, bbox, square=True)

        # Should produce square crop without exceeding boundaries
        assert cropped.shape[0] == cropped.shape[1]
        assert cropped.shape[0] <= 480
        assert cropped.shape[1] <= 640

    def test_crop_bbox_square_at_bottom_edge(self) -> None:
        """Test square cropping when bbox is at bottom edge of image."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Bbox near bottom edge that will need adjustment
        bbox = (100, 350, 300, 470)

        cropped = crop_bbox(img, bbox, square=True)

        # Should produce square crop without exceeding boundaries
        assert cropped.shape[0] == cropped.shape[1]
        assert cropped.shape[0] <= 480
        assert cropped.shape[1] <= 640


class TestSaveImage:
    """Test suite for save_image function."""

    def test_save_image_jpg(self, tmp_path: Path) -> None:
        """Test saving JPG image."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "output.jpg"

        save_image(img, output_path, quality=90)

        assert output_path.exists()

    def test_save_image_png(self, tmp_path: Path) -> None:
        """Test saving PNG image."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "output.png"

        save_image(img, output_path)

        assert output_path.exists()

    def test_save_image_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test save creates parent directory."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "subdir1" / "subdir2" / "output.jpg"

        save_image(img, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_empty_image(self, tmp_path: Path) -> None:
        """Test saving fails for empty image."""
        img = np.array([])
        output_path = tmp_path / "output.jpg"

        with pytest.raises(ValueError, match="empty"):
            save_image(img, output_path)

    def test_save_image_string_path(self, tmp_path: Path) -> None:
        """Test saving with string path."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "output.jpg"

        save_image(img, str(output_path))

        assert output_path.exists()

    def test_save_image_write_failure(self, tmp_path: Path) -> None:
        """Test save handles write failure."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "output.jpg"

        with patch("cv2.imwrite", return_value=False):
            with pytest.raises(OSError, match="returned False"):
                save_image(img, output_path)

    def test_save_image_mkdir_failure(self, tmp_path: Path) -> None:
        """Test save handles directory creation failure."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "subdir" / "output.jpg"

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            with pytest.raises(OSError, match="Cannot create output directory"):
                save_image(img, output_path)

    def test_save_image_write_exception(self, tmp_path: Path) -> None:
        """Test save handles write exception."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_path = tmp_path / "output.jpg"

        with patch("cv2.imwrite", side_effect=Exception("Write error")):
            with pytest.raises(OSError, match="Failed to save"):
                save_image(img, output_path)


class TestIntegration:
    """Integration tests for image_utils module."""

    def test_full_image_processing_workflow(self, tmp_path: Path) -> None:
        """Test complete image processing workflow."""
        # Create test image
        img_path = tmp_path / "input.jpg"
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        # Load image
        img, metadata = load_image(img_path, color_mode="BGR")
        assert img.shape == (480, 640, 3)

        # Resize
        resized = resize_image(img, 320, maintain_aspect=True)
        assert max(resized.shape[:2]) == 320

        # Crop
        bbox = (50, 50, 200, 200)
        cropped = crop_bbox(resized, bbox, padding=0.1)
        assert cropped.shape[0] > 0

        # Save
        output_path = tmp_path / "output.jpg"
        save_image(cropped, output_path)
        assert output_path.exists()

    def test_thumbnail_workflow(self, tmp_path: Path) -> None:
        """Test thumbnail generation workflow."""
        # Create test image
        img_path = tmp_path / "photo.jpg"
        test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        # Load and create thumbnail
        img, _ = load_image(img_path)
        thumb_path = tmp_path / "thumb.jpg"
        thumbnail = create_thumbnail(img, size=256, output_path=thumb_path)

        assert max(thumbnail.shape[:2]) == 256
        assert thumb_path.exists()

    def test_normalization_workflow(self, tmp_path: Path) -> None:
        """Test image normalization workflow."""
        # Create and load image
        img_path = tmp_path / "input.jpg"
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), test_img)

        img, _ = load_image(img_path, color_mode="RGB")

        # Normalize (ImageNet stats)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalized = normalize_image(img, mean, std)

        # Should be float32 or float64
        assert normalized.dtype in (np.float32, np.float64)
        assert normalized.shape == img.shape
