"""
Cropping engine for extracting and saving detected animals.

This module provides functionality to crop detected animals from images,
apply quality checks, generate semantic filenames, and organize output.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from core.classification_engine import ClassificationResult
from models.yolo_detector import Detection
from utils.image_utils import crop_bbox, save_image
from utils.validators import (
    validate_directory_writable,
    validate_padding_ratio,
)

logger = logging.getLogger(__name__)


class CroppingError(Exception):
    """Exception raised for cropping errors."""

    pass


@dataclass
class CropConfig:
    """
    Configuration for cropping operations.

    Attributes:
        padding: Padding ratio around bounding box (0.0-1.0)
        square: Whether to make crops square
        min_width: Minimum crop width in pixels
        min_height: Minimum crop height in pixels
        max_width: Maximum crop width in pixels (None = no limit)
        max_height: Maximum crop height in pixels (None = no limit)
        quality: JPEG quality for saved crops (1-100)
    """

    padding: float = 0.1
    square: bool = False
    min_width: int = 50
    min_height: int = 50
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    quality: int = 95

    def __post_init__(self) -> None:
        """Validate crop configuration."""
        validate_padding_ratio(self.padding)

        if self.min_width < 1:
            raise ValueError(f"min_width must be >= 1, got {self.min_width}")
        if self.min_height < 1:
            raise ValueError(f"min_height must be >= 1, got {self.min_height}")

        if self.max_width is not None and self.max_width < self.min_width:
            raise ValueError(
                f"max_width ({self.max_width}) must be >= min_width ({self.min_width})"
            )
        if self.max_height is not None and self.max_height < self.min_height:
            raise ValueError(
                f"max_height ({self.max_height}) must be >= " f"min_height ({self.min_height})"
            )

        if not 1 <= self.quality <= 100:
            raise ValueError(f"quality must be 1-100, got {self.quality}")


@dataclass
class CropResult:
    """
    Result of a cropping operation.

    Attributes:
        crop_image: The cropped image array
        detection: Original detection that was cropped
        classification: Optional classification result
        original_bbox: Original bounding box (x1, y1, x2, y2)
        final_bbox: Final bounding box after padding/adjustments
        output_path: Path where crop was saved (if saved)
        metadata: Additional metadata about the crop
    """

    crop_image: NDArray[np.uint8]
    detection: Detection
    classification: Optional[ClassificationResult]
    original_bbox: Tuple[int, int, int, int]
    final_bbox: Tuple[int, int, int, int]
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_dimensions(self) -> Tuple[int, int]:
        """Get crop dimensions (width, height)."""
        return self.crop_image.shape[1], self.crop_image.shape[0]

    def passes_quality_checks(self, min_width: int, min_height: int) -> Tuple[bool, str]:
        """
        Check if crop meets quality requirements.

        Args:
            min_width: Minimum required width
            min_height: Minimum required height

        Returns:
            Tuple of (passes, reason)
        """
        width, height = self.get_dimensions()

        if width < min_width:
            return False, f"Width {width} < minimum {min_width}"
        if height < min_height:
            return False, f"Height {height} < minimum {min_height}"

        return True, "OK"


class CroppingEngine:
    """
    Engine for cropping and saving detected animals.

    This engine handles:
    - Cropping detections with configurable padding
    - Quality checks (minimum/maximum dimensions)
    - Semantic filename generation
    - Organized output directory structure
    - Batch processing of multiple detections
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[CropConfig] = None,
        organize_by_species: bool = True,
        organize_by_date: bool = True,
    ) -> None:
        """
        Initialize cropping engine.

        Args:
            output_dir: Base directory for saving crops
            config: Crop configuration (uses defaults if None)
            organize_by_species: Create subdirectories for each species
            organize_by_date: Create subdirectories by date

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If output_dir is invalid
        """
        logger.info("Initializing CroppingEngine")

        self.output_dir = Path(output_dir)
        validate_directory_writable(self.output_dir, create=True)

        self.config = config or CropConfig()
        self.organize_by_species = organize_by_species
        self.organize_by_date = organize_by_date

        # Track statistics
        self.stats = {
            "total_crops": 0,
            "successful_crops": 0,
            "failed_crops": 0,
            "quality_failures": 0,
        }

        logger.info(
            f"CroppingEngine initialized: output={self.output_dir}, "
            f"padding={self.config.padding}, "
            f"organize_by_species={organize_by_species}"
        )

    def crop_detection(
        self,
        image: NDArray[np.uint8],
        detection: Detection,
        classification: Optional[ClassificationResult] = None,
        save: bool = True,
        source_filename: Optional[str] = None,
    ) -> Optional[CropResult]:
        """
        Crop a single detection from an image.

        Args:
            image: Full source image
            detection: Detection to crop
            classification: Optional classification result
            save: Whether to save the crop to disk
            source_filename: Original image filename (for naming crops)

        Returns:
            CropResult or None if crop fails quality checks

        Raises:
            CroppingError: If cropping fails
        """
        try:
            logger.debug(
                f"Cropping detection: class={detection.class_name}, " f"bbox={detection.bbox}"
            )

            # Extract crop with padding
            crop_image = crop_bbox(
                image,
                detection.bbox,
                padding=self.config.padding,
                square=self.config.square,
            )

            # Calculate final bbox (after padding)
            x1, y1, x2, y2 = detection.bbox

            # Quality checks
            passes, reason = self._check_quality(crop_image)
            if not passes:
                logger.debug(f"Crop failed quality check: {reason}")
                self.stats["quality_failures"] += 1
                self.stats["failed_crops"] += 1
                return None

            # Create result
            result = CropResult(
                crop_image=crop_image,
                detection=detection,
                classification=classification,
                original_bbox=detection.bbox,
                final_bbox=(x1, y1, x2, y2),  # Simplified for now
                metadata={
                    "padding": self.config.padding,
                    "square": self.config.square,
                    "source_filename": source_filename,
                },
            )

            # Save if requested
            if save:
                output_path = self._generate_output_path(detection, classification, source_filename)
                save_image(crop_image, output_path, quality=self.config.quality)
                result.output_path = output_path
                logger.info(f"Saved crop: {output_path}")

            self.stats["successful_crops"] += 1
            self.stats["total_crops"] += 1

            return result

        except Exception as e:
            logger.error(f"Failed to crop detection: {e}", exc_info=True)
            self.stats["failed_crops"] += 1
            self.stats["total_crops"] += 1
            raise CroppingError(f"Cropping failed: {e}") from e

    def crop_batch(
        self,
        images: List[NDArray[np.uint8]],
        detections_list: List[List[Detection]],
        classifications_list: Optional[List[List[Optional[ClassificationResult]]]] = None,
        source_filenames: Optional[List[str]] = None,
        save: bool = True,
    ) -> List[List[Optional[CropResult]]]:
        """
        Crop detections from multiple images.

        Args:
            images: List of source images
            detections_list: List of detection lists (one per image)
            classifications_list: Optional list of classification lists
            source_filenames: Optional list of source filenames
            save: Whether to save crops to disk

        Returns:
            List of lists of CropResult (one list per image)

        Raises:
            ValueError: If list lengths don't match
        """
        if len(images) != len(detections_list):
            raise ValueError(
                f"Images and detections length mismatch: "
                f"{len(images)} vs {len(detections_list)}"
            )

        if classifications_list is not None and len(classifications_list) != len(images):
            raise ValueError(
                f"Classifications list length mismatch: "
                f"{len(classifications_list)} vs {len(images)}"
            )

        if source_filenames is not None and len(source_filenames) != len(images):
            raise ValueError(
                f"Source filenames length mismatch: " f"{len(source_filenames)} vs {len(images)}"
            )

        logger.info(f"Processing batch of {len(images)} images")

        results: List[List[Optional[CropResult]]] = []

        for i, (image, detections) in enumerate(zip(images, detections_list)):
            image_results: List[Optional[CropResult]] = []

            classifications = (
                classifications_list[i] if classifications_list else [None] * len(detections)
            )
            filename = source_filenames[i] if source_filenames else None

            logger.debug(f"Processing image {i+1}/{len(images)}: {len(detections)} detections")

            for detection, classification in zip(detections, classifications):
                try:
                    result = self.crop_detection(image, detection, classification, save, filename)
                    image_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to crop detection in image {i+1}: {e}")
                    image_results.append(None)

            results.append(image_results)

        successful = sum(1 for img_results in results for r in img_results if r is not None)
        total = sum(len(img_results) for img_results in results)

        logger.info(f"Batch cropping complete: {successful}/{total} successful")

        return results

    def _check_quality(self, crop_image: NDArray[np.uint8]) -> Tuple[bool, str]:
        """
        Check if crop meets quality requirements.

        Args:
            crop_image: Cropped image array

        Returns:
            Tuple of (passes, reason)
        """
        height, width = crop_image.shape[:2]

        # Check minimum dimensions
        if width < self.config.min_width:
            return False, f"Width {width} < minimum {self.config.min_width}"
        if height < self.config.min_height:
            return False, f"Height {height} < minimum {self.config.min_height}"

        # Check maximum dimensions
        if self.config.max_width is not None and width > self.config.max_width:
            return False, f"Width {width} > maximum {self.config.max_width}"
        if self.config.max_height is not None and height > self.config.max_height:
            return False, f"Height {height} > maximum {self.config.max_height}"

        # Check if image is empty or corrupted
        if crop_image.size == 0:
            return False, "Empty image"

        # Check for extremely low variance (likely a blank crop)
        if crop_image.std() < 1.0:
            return False, "Low variance (possibly blank)"

        return True, "OK"

    def _generate_output_path(
        self,
        detection: Detection,
        classification: Optional[ClassificationResult],
        source_filename: Optional[str],
    ) -> Path:
        """
        Generate semantic output path for a crop.

        Path structure:
        - If organize_by_species: output_dir/species_name/filename
        - If organize_by_date: output_dir/YYYY-MM-DD/filename
        - Otherwise: output_dir/filename

        Filename format:
        - With classification: species_name_timestamp_confidence.jpg
        - Without classification: yolo_class_timestamp_confidence.jpg

        Args:
            detection: Detection object
            classification: Optional classification result
            source_filename: Original image filename

        Returns:
            Path object for output file
        """
        # Build directory structure
        output_path = self.output_dir

        # Add date subdirectory if requested
        if self.organize_by_date:
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_path = output_path / date_str

        # Add species subdirectory if requested
        if self.organize_by_species and classification:
            species_dir = self._sanitize_filename(classification.common_name)
            output_path = output_path / species_dir
        elif self.organize_by_species:
            # Fallback to YOLO class
            yolo_dir = self._sanitize_filename(detection.class_name)
            output_path = output_path / yolo_dir

        # Create directory if needed
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        if classification:
            species_name = self._sanitize_filename(classification.common_name)
            confidence = int(classification.confidence * 100)
            filename = f"{species_name}_{timestamp}_c{confidence}.jpg"
        else:
            class_name = self._sanitize_filename(detection.class_name)
            confidence = int(detection.confidence * 100)
            filename = f"{class_name}_{timestamp}_c{confidence}.jpg"

        # Add source filename prefix if provided
        if source_filename:
            source_base = Path(source_filename).stem
            source_sanitized = self._sanitize_filename(source_base)[:30]  # Limit length
            filename = f"{source_sanitized}_{filename}"

        # Ensure unique filename
        full_path = output_path / filename
        counter = 1
        while full_path.exists():
            stem = full_path.stem
            full_path = output_path / f"{stem}_{counter}.jpg"
            counter += 1

        return full_path

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string for use in filenames.

        Args:
            name: String to sanitize

        Returns:
            Sanitized string
        """
        # Convert to lowercase
        sanitized = name.lower()

        # Replace spaces and special chars with underscore
        sanitized = re.sub(r"[^\w\-]", "_", sanitized)

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        return sanitized

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cropping statistics.

        Returns:
            Dictionary with statistics
        """
        success_rate = (
            self.stats["successful_crops"] / self.stats["total_crops"] * 100
            if self.stats["total_crops"] > 0
            else 0.0
        )

        return {
            **self.stats,
            "success_rate": round(success_rate, 2),
        }

    def reset_statistics(self) -> None:
        """Reset cropping statistics."""
        self.stats = {
            "total_crops": 0,
            "successful_crops": 0,
            "failed_crops": 0,
            "quality_failures": 0,
        }
        logger.info("Statistics reset")

    def __repr__(self) -> str:
        """String representation of engine."""
        return (
            f"CroppingEngine(output={self.output_dir}, "
            f"padding={self.config.padding}, "
            f"crops={self.stats['successful_crops']})"
        )


if __name__ == "__main__":
    """Test cropping engine."""
    import sys
    import tempfile
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing Cropping Engine...\n")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "crops"

        # Test configuration
        config = CropConfig(padding=0.15, square=False, min_width=50, min_height=50, quality=90)

        print(
            f"Crop config: padding={config.padding}, "
            f"min_size={config.min_width}x{config.min_height}"
        )

        # Create engine
        engine = CroppingEngine(
            output_dir=output_dir,
            config=config,
            organize_by_species=True,
            organize_by_date=False,
        )

        print(f"{engine}\n")

        # Create test image
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

        # Create test detection
        from models.yolo_detector import Detection

        test_detection = Detection(
            bbox=(100, 100, 300, 250), confidence=0.85, class_id=16, class_name="dog"
        )

        print("Testing single crop:")
        print(f"  Detection: {test_detection.class_name} at {test_detection.bbox}")

        # Crop without classification
        result = engine.crop_detection(
            test_image, test_detection, save=True, source_filename="test_image.jpg"
        )

        if result:
            width, height = result.get_dimensions()
            print("\n✅ Crop successful:")
            print(f"  Size: {width}x{height}")
            print(f"  Saved: {result.output_path}")
            print(f"  Quality: {result.passes_quality_checks(50, 50)}")

        # Test with classification
        from core.classification_engine import ClassificationResult

        test_classification = ClassificationResult(
            species_id="coyote",
            common_name="Coyote",
            scientific_name="Canis latrans",
            confidence=0.75,
            yolo_class="dog",
            yolo_confidence=0.85,
        )

        print("\nTesting crop with classification:")
        result2 = engine.crop_detection(
            test_image,
            test_detection,
            test_classification,
            save=True,
            source_filename="test_image.jpg",
        )

        if result2:
            print(f"✅ Classified crop saved: {result2.output_path}")

        # Test batch
        print("\nTesting batch cropping:")
        images = [test_image, test_image]
        detections_list = [[test_detection], [test_detection, test_detection]]

        batch_results = engine.crop_batch(images, detections_list, save=True)

        total_crops = sum(len(img_results) for img_results in batch_results)
        successful_crops = sum(
            1 for img_results in batch_results for r in img_results if r is not None
        )

        print(f"✅ Batch complete: {successful_crops}/{total_crops} successful")

        # Show statistics
        print(f"\nStatistics: {engine.get_statistics()}")

        # List created files
        print(f"\nCreated files in {output_dir}:")
        if output_dir.exists():
            for file in sorted(output_dir.rglob("*.jpg")):
                rel_path = file.relative_to(output_dir)
                print(f"  {rel_path}")

    print("\nCropping engine tests completed!")
