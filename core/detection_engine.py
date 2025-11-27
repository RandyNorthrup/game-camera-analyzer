"""
Detection engine for coordinating image loading, preprocessing, and animal detection.

This module provides the DetectionEngine class that orchestrates the complete
detection pipeline: loading images, preprocessing, running YOLOv8 detection,
and packaging results with metadata.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from models.model_manager import ModelManager
from models.yolo_detector import Detection, YOLODetector
from utils.image_utils import load_image, resize_image, save_image, enhance_low_light, denoise_image
from utils.validators import (
    validate_confidence_threshold,
    validate_iou_threshold,
    validate_image_readable,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Complete detection result for an image.

    Contains the original image path, loaded image, detections, metadata,
    and optionally the annotated image with bounding boxes drawn.

    Attributes:
        image_path: Path to the source image
        image: The loaded image array (RGB format)
        detections: List of Detection objects found in the image
        metadata: Additional metadata (file size, dimensions, processing time, etc.)
        annotated_image: Optional annotated image with bounding boxes drawn
    """

    image_path: Path
    image: NDArray[np.uint8]
    detections: List[Detection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotated_image: Optional[NDArray[np.uint8]] = None

    def __post_init__(self) -> None:
        """Validate detection result after initialization."""
        if not isinstance(self.image_path, Path):
            raise TypeError(f"image_path must be Path, got {type(self.image_path)}")

        if not isinstance(self.image, np.ndarray):
            raise TypeError(f"image must be ndarray, got {type(self.image)}")

        if not isinstance(self.detections, list):
            raise TypeError(f"detections must be list, got {type(self.detections)}")

        # Validate all detections
        for i, det in enumerate(self.detections):
            if not isinstance(det, Detection):
                raise TypeError(f"detections[{i}] must be Detection, got {type(det)}")

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Get image size as (height, width) tuple.

        Returns:
            Tuple of (height, width) in pixels
        """
        return (self.image.shape[0], self.image.shape[1])

    @property
    def num_detections(self) -> int:
        """
        Get the number of detections.

        Returns:
            Count of detections in the image
        """
        return len(self.detections)

    def get_detection_count(self) -> int:
        """Get the number of detections (legacy method, prefer num_detections property)."""
        return len(self.detections)

    def has_detections(self) -> bool:
        """Check if any detections were found."""
        return len(self.detections) > 0

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get count of detections per class.

        Returns:
            Dictionary mapping class names to detection counts
        """
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def filter_by_class(self, class_name: str) -> List[Detection]:
        """
        Filter detections by class name.

        Args:
            class_name: The class name to filter by

        Returns:
            List of detections matching the class name
        """
        return [det for det in self.detections if det.class_name == class_name]

    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """
        Filter detections by minimum confidence.

        Args:
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            List of detections with confidence >= min_confidence
        """
        validate_confidence_threshold(min_confidence)
        return [det for det in self.detections if det.confidence >= min_confidence]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection result to dictionary format.

        Returns:
            Dictionary containing detection result data including:
            - image_path: str path to the image
            - image_size: tuple of (height, width)
            - num_detections: count of detections
            - detections: list of detection dicts
            - metadata: result metadata
        """
        return {
            "image_path": str(self.image_path),
            "image_size": self.image_size,
            "num_detections": self.num_detections,
            "detections": [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                }
                for det in self.detections
            ],
            "metadata": self.metadata,
        }


class DetectionEngine:
    """
    High-level detection engine coordinating the complete detection pipeline.

    This class manages:
    - Image loading and validation
    - Image preprocessing (resizing, normalization)
    - YOLOv8 detection via YOLODetector
    - Result packaging with metadata
    - Batch processing with progress tracking

    Example:
        >>> engine = DetectionEngine(
        ...     model_name="yolov8n.pt",
        ...     confidence_threshold=0.3
        ... )
        >>> result = engine.process_image("path/to/image.jpg")
        >>> print(f"Found {result.get_detection_count()} animals")
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        preprocess_size: Optional[Tuple[int, int]] = None,
        model_manager: Optional[ModelManager] = None,
        enhance_low_light: bool = True,
        denoise_images: bool = False,
        low_light_threshold: int = 80,
        denoise_strength: int = 3,
    ) -> None:
        """
        Initialize detection engine.

        Args:
            model_name: YOLOv8 model name (e.g., "yolov8n.pt", "yolov8m.pt")
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            max_detections: Maximum number of detections per image
            preprocess_size: Optional resize dimensions (width, height) for preprocessing
            model_manager: Optional ModelManager instance (created if not provided)
            enhance_low_light: Automatically enhance dark images for better detection
            denoise_images: Apply denoising to reduce false positives
            low_light_threshold: Mean brightness threshold for enhancement (0-255)
            denoise_strength: Denoising strength (1-10)

        Raises:
            ValueError: If confidence or IoU thresholds are invalid
            ModelLoadError: If model cannot be loaded
        """
        logger.info("Initializing DetectionEngine")

        # Validate parameters
        validate_confidence_threshold(confidence_threshold)
        validate_iou_threshold(iou_threshold)

        if max_detections < 1:
            raise ValueError(f"max_detections must be >= 1, got {max_detections}")

        if preprocess_size is not None:
            if len(preprocess_size) != 2:
                raise ValueError(f"preprocess_size must be (width, height), got {preprocess_size}")
            if preprocess_size[0] < 1 or preprocess_size[1] < 1:
                raise ValueError(f"preprocess_size dimensions must be >= 1, got {preprocess_size}")

        # Store configuration
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.preprocess_size = preprocess_size
        self.enhance_low_light = enhance_low_light
        self.denoise_images = denoise_images
        self.low_light_threshold = low_light_threshold
        self.denoise_strength = denoise_strength

        # Initialize model manager and detector
        self.model_manager = model_manager or ModelManager()
        self.detector = YOLODetector(
            model_manager=self.model_manager,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        logger.info(
            f"DetectionEngine initialized: model={model_name}, "
            f"conf={confidence_threshold}, iou={iou_threshold}, "
            f"enhance_low_light={enhance_low_light}, denoise={denoise_images}"
        )

    def process_image(
        self,
        image_path: str | Path,
        return_annotated: bool = False,
        save_annotated: Optional[str | Path] = None,
    ) -> DetectionResult:
        """
        Process a single image through the complete detection pipeline.

        Steps:
        1. Load and validate image
        2. Preprocess (resize if configured)
        3. Run detection
        4. Package results with metadata
        5. Optionally save annotated image

        Args:
            image_path: Path to the image file
            return_annotated: Whether to include annotated image in result
            save_annotated: Optional path to save annotated image

        Returns:
            DetectionResult with detections and metadata

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image is invalid
            RuntimeError: If detection fails
        """
        import time

        start_time = time.time()

        # Validate and convert path
        image_path = Path(image_path)
        validate_image_readable(image_path)

        logger.info(f"Processing image: {image_path}")

        try:
            # Load image (returns tuple of image and metadata)
            image, img_metadata = load_image(image_path, color_mode="RGB")
            original_shape = image.shape
            logger.debug(f"Loaded image: shape={original_shape}")

            # Apply image enhancements if configured
            enhanced_image = image
            if self.enhance_low_light:
                # Convert to BGR for OpenCV enhancement
                bgr_image = image[:, :, ::-1].copy()
                bgr_enhanced = enhance_low_light(
                    bgr_image,
                    auto_adjust=True,
                    gamma=2.2,
                    clahe_clip_limit=2.0,
                )
                # Convert back to RGB
                enhanced_image = bgr_enhanced[:, :, ::-1]
                logger.debug("Applied low-light enhancement")

            if self.denoise_images:
                # Convert to BGR for OpenCV denoising
                bgr_image = enhanced_image[:, :, ::-1].copy()
                bgr_denoised = denoise_image(
                    bgr_image,
                    method="bilateral",
                    strength=self.denoise_strength,
                )
                # Convert back to RGB
                enhanced_image = bgr_denoised[:, :, ::-1]
                logger.debug(f"Applied denoising with strength={self.denoise_strength}")

            # Preprocess if needed
            processed_image = enhanced_image
            if self.preprocess_size is not None:
                processed_image = resize_image(
                    enhanced_image, target_size=self.preprocess_size, maintain_aspect=False
                )
                logger.debug(f"Preprocessed image: {original_shape} -> {processed_image.shape}")

            # Run detection
            detections, annotated = self.detector.detect(
                processed_image, return_annotated=return_annotated or save_annotated is not None
            )

            # If we preprocessed, we need to scale bounding boxes back to original size
            if self.preprocess_size is not None and detections:
                scale_x = original_shape[1] / self.preprocess_size[0]
                scale_y = original_shape[0] / self.preprocess_size[1]

                scaled_detections: List[Detection] = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    scaled_bbox = (
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y),
                    )
                    scaled_detections.append(
                        Detection(
                            bbox=scaled_bbox,
                            confidence=det.confidence,
                            class_id=det.class_id,
                            class_name=det.class_name,
                        )
                    )
                detections = scaled_detections
                logger.debug(f"Scaled {len(detections)} bounding boxes to original size")

            # Save annotated image if requested
            if save_annotated is not None and annotated is not None:
                save_path = Path(save_annotated)
                save_image(annotated, save_path)
                logger.info(f"Saved annotated image: {save_path}")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build metadata
            metadata = {
                "filename": image_path.name,
                "file_size_bytes": image_path.stat().st_size,
                "original_shape": original_shape,
                "processed_shape": processed_image.shape,
                "preprocessing_applied": self.preprocess_size is not None,
                "num_detections": len(detections),
                "processing_time_seconds": round(processing_time, 3),
                "model_name": self.model_name,
                "confidence_threshold": self.confidence_threshold,
            }

            logger.info(
                f"Detection complete: {len(detections)} detections " f"in {processing_time:.3f}s"
            )

            return DetectionResult(
                image_path=image_path,
                image=image,
                detections=detections,
                metadata=metadata,
                annotated_image=annotated if return_annotated else None,
            )

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Image processing failed: {e}") from e

    def process_batch(
        self,
        image_paths: List[str | Path],
        return_annotated: bool = False,
        save_annotated_dir: Optional[str | Path] = None,
    ) -> List[DetectionResult]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image file paths
            return_annotated: Whether to include annotated images in results
            save_annotated_dir: Optional directory to save annotated images

        Returns:
            List of DetectionResult objects

        Raises:
            ValueError: If image_paths is empty
            RuntimeError: If batch processing fails
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        logger.info(f"Processing batch of {len(image_paths)} images")

        results: List[DetectionResult] = []

        # Create save directory if needed
        if save_annotated_dir is not None:
            save_dir = Path(save_annotated_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created annotated output directory: {save_dir}")

        # Process each image
        for i, image_path in enumerate(image_paths, 1):
            try:
                logger.debug(f"Processing image {i}/{len(image_paths)}: {image_path}")

                # Determine save path for annotated image
                save_path = None
                if save_annotated_dir is not None:
                    save_path = Path(save_annotated_dir) / f"annotated_{Path(image_path).name}"

                # Process image
                result = self.process_image(
                    image_path,
                    return_annotated=return_annotated,
                    save_annotated=save_path,
                )
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Failed to process image {i}/{len(image_paths)} " f"({image_path}): {e}"
                )
                # Continue with remaining images
                continue

        logger.info(f"Batch processing complete: {len(results)}/{len(image_paths)} successful")

        return results

    # Convenience method aliases for backward compatibility and test compatibility
    def detect(
        self,
        image_path: str | Path,
        return_annotated: bool = False,
        save_annotated: Optional[str | Path] = None,
    ) -> DetectionResult:
        """
        Convenience alias for process_image().

        This method provides backward compatibility with older API versions
        and matches test expectations. All functionality is delegated to
        process_image().

        Args:
            image_path: Path to the image file
            return_annotated: Whether to include annotated image in result
            save_annotated: Optional path to save annotated image

        Returns:
            DetectionResult with detections and metadata

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image is invalid
            RuntimeError: If detection fails
        """
        return self.process_image(
            image_path=image_path,
            return_annotated=return_annotated,
            save_annotated=save_annotated,
        )

    def detect_batch(
        self,
        image_paths: List[str | Path],
        return_annotated: bool = False,
        save_annotated_dir: Optional[str | Path] = None,
    ) -> List[DetectionResult]:
        """
        Convenience alias for process_batch().

        This method provides backward compatibility with older API versions
        and matches test expectations. All functionality is delegated to
        process_batch().

        Args:
            image_paths: List of image file paths
            return_annotated: Whether to include annotated images in results
            save_annotated_dir: Optional directory to save annotated images

        Returns:
            List of DetectionResult objects

        Raises:
            ValueError: If image_paths is empty
            RuntimeError: If batch processing fails
        """
        return self.process_batch(
            image_paths=image_paths,
            return_annotated=return_annotated,
            save_annotated_dir=save_annotated_dir,
        )

    def update_thresholds(
        self,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> None:
        """
        Update detection thresholds at runtime.

        Args:
            confidence: New confidence threshold (0-1)
            iou: New IoU threshold for NMS (0-1)
            max_det: New maximum detections per image

        Raises:
            ValueError: If any threshold is invalid
        """
        if confidence is not None:
            validate_confidence_threshold(confidence)
            self.confidence_threshold = confidence
            logger.info(f"Updated confidence threshold: {confidence}")

        if iou is not None:
            validate_confidence_threshold(iou)
            self.iou_threshold = iou
            logger.info(f"Updated IoU threshold: {iou}")

        if max_det is not None:
            if max_det < 1:
                raise ValueError(f"max_det must be >= 1, got {max_det}")
            self.max_detections = max_det
            logger.info(f"Updated max detections: {max_det}")

        # Update detector
        self.detector.update_thresholds(confidence, iou, max_det)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return self.detector.get_model_info()

    def __repr__(self) -> str:
        """String representation of DetectionEngine."""
        return (
            f"DetectionEngine(model={self.model_name}, "
            f"conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold})"
        )


if __name__ == "__main__":
    """Test detection engine."""
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing Detection Engine...\n")

    # Create engine
    engine = DetectionEngine(
        model_name="yolov8n.pt",
        confidence_threshold=0.25,
        preprocess_size=(640, 640),
    )

    print(f"{engine}\n")

    # Get model info
    model_info = engine.get_model_info()
    print("Model info:")
    for key, value in model_info.items():
        if key == "class_names":
            print(f"  {key}: {len(value)} classes")
        else:
            print(f"  {key}: {value}")

    # Test with random image (save to temp file)
    print("\nTesting detection on random image...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Save test image temporarily
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        save_image(test_image, tmp_path)

    try:
        # Process image
        result = engine.process_image(tmp_path, return_annotated=True)

        print("\n✅ Processing complete")
        print(f"  Detections: {result.get_detection_count()}")
        print(f"  Processing time: {result.metadata['processing_time_seconds']}s")
        print(f"  File size: {result.metadata['file_size_bytes']} bytes")

        if result.has_detections():
            print("\nDetection details:")
            for i, det in enumerate(result.detections, 1):
                print(f"  {i}. {det.class_name} (conf={det.confidence:.2f})")

            # Test filtering
            class_counts = result.get_class_counts()
            print(f"\nClass counts: {class_counts}")

        # Test threshold update
        print("\nTesting threshold update...")
        engine.update_thresholds(confidence=0.5, iou=0.5, max_det=10)
        print("✅ Thresholds updated")

    finally:
        # Cleanup temp file
        tmp_path.unlink()

    print("\nDetection engine tests completed!")
