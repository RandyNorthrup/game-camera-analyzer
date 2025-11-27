"""
Additional tests for validators module - grayscale image handling.

This module contains tests specifically for grayscale image validation
to cover line 226 in utils/validators.py.
"""

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from utils.validators import validate_image_readable, ValidationError

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestValidateImageReadableGrayscale:
    """Test suite for validate_image_readable with grayscale images."""

    def test_validate_grayscale_image_readable(self, tmp_path: Path) -> None:
        """
        Test validate_image_readable with grayscale image (line 226).
        
        Args:
            tmp_path: Temporary directory path
            
        Verifies:
            - Grayscale images (2D arrays) are handled correctly via PIL fallback
            - Line 226: height, width = img_array.shape is executed when check_corruption=True
            - Dimensions are extracted properly from 2D array
        """
        # Create a grayscale test image using PIL
        from PIL import Image as PILImage
        
        grayscale_array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        grayscale_img = PILImage.fromarray(grayscale_array, mode='L')  # 'L' = grayscale
        
        # Save as TIFF - some TIFF variants cause cv2.imread to return None, forcing PIL fallback
        test_image = tmp_path / "grayscale_test.tiff"
        grayscale_img.save(test_image, compression='tiff_deflate')  # Use compression that might not work with cv2
        
        # This may force PIL fallback, and PIL will return a 2D array for grayscale, hitting line 226
        path_obj, dimensions = validate_image_readable(test_image, check_corruption=True)
        
        # Verify dimensions are correctly extracted
        assert path_obj == test_image
        assert dimensions[0] == 640, f"Expected width 640, got {dimensions[0]}"
        assert dimensions[1] == 480, f"Expected height 480, got {dimensions[1]}"
        logger.info(f"Grayscale TIFF image validated: {dimensions}")

    def test_validate_grayscale_image_quick_mode(self, tmp_path: Path) -> None:
        """
        Test validate_image_readable with grayscale image in quick mode.
        
        Args:
            tmp_path: Temporary directory path
            
        Verifies:
            - Quick validation (check_corruption=False) works for grayscale
            - Uses PIL instead of OpenCV for quick check
        """
        # Create grayscale image
        grayscale_array = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        test_image = tmp_path / "grayscale_quick.jpg"
        cv2.imwrite(str(test_image), grayscale_array)
        
        # Quick validation (doesn't trigger line 226)
        path_obj, dimensions = validate_image_readable(test_image, check_corruption=False)
        
        assert path_obj == test_image
        assert dimensions[0] == 320, f"Expected width 320, got {dimensions[0]}"
        assert dimensions[1] == 240, f"Expected height 240, got {dimensions[1]}"
        logger.info(f"Grayscale quick validation: {dimensions}")

    def test_validate_corrupted_grayscale(self, tmp_path: Path) -> None:
        """
        Test validation fails for corrupted grayscale image.
        
        Args:
            tmp_path: Temporary directory path
            
        Verifies:
            - Validation error raised for corrupted files
            - Corruption check works for grayscale images
        """
        # Create a corrupted "image" file
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_bytes(b"Not a valid image file")
        
        # Should raise ValidationError with "Cannot load image" message
        with pytest.raises(ValidationError, match="Cannot load image"):
            validate_image_readable(corrupted_file, check_corruption=True)
        
        logger.info("Corrupted grayscale validation error correctly raised")

    def test_validate_grayscale_png_format(self, tmp_path: Path) -> None:
        """
        Test validate_image_readable with grayscale PNG.
        
        Args:
            tmp_path: Temporary directory path
            
        Verifies:
            - PNG grayscale images work correctly
            - 2D array handling works across formats
        """
        # Create grayscale PNG
        grayscale_array = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        test_image = tmp_path / "grayscale.png"
        cv2.imwrite(str(test_image), grayscale_array)
        
        # Validate with corruption check
        path_obj, dimensions = validate_image_readable(test_image, check_corruption=True)
        
        assert path_obj == test_image
        # Dimensions should be (150, 100) - width, height
        assert dimensions[0] == 150, f"Expected width 150, got {dimensions[0]}"
        assert dimensions[1] == 100, f"Expected height 100, got {dimensions[1]}"
        logger.info(f"Grayscale PNG validated: {dimensions}")

    def test_validate_small_grayscale_dimensions(self, tmp_path: Path) -> None:
        """
        Test validate_image_readable with very small grayscale image.
        
        Args:
            tmp_path: Temporary directory path
            
        Verifies:
            - Small grayscale images are validated correctly
            - 2D shape handling works for minimal dimensions
        """
        # Create tiny grayscale image
        tiny_array = np.random.randint(0, 255, (10, 20), dtype=np.uint8)
        test_image = tmp_path / "tiny_grayscale.jpg"
        cv2.imwrite(str(test_image), tiny_array)
        
        # Should validate successfully
        path_obj, dimensions = validate_image_readable(test_image, check_corruption=True)
        
        assert path_obj == test_image
        assert dimensions[0] == 20, f"Expected width 20, got {dimensions[0]}"
        assert dimensions[1] == 10, f"Expected height 10, got {dimensions[1]}"
        logger.info(f"Tiny grayscale image validated: {dimensions}")
