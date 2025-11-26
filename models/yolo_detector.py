"""
YOLOv8 detector wrapper for the Game Camera Analyzer application.

This module provides a high-level interface to YOLOv8 detection with:
- Confidence filtering
- Non-maximum suppression (NMS)
- Batch processing
- Structured result format
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from models.model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single animal detection."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float  # Detection confidence (0.0-1.0)
    class_id: int  # YOLO class ID
    class_name: str  # Human-readable class name

    def __post_init__(self) -> None:
        """Validate detection parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {self.bbox}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert detection to dictionary format.

        Returns:
            Dictionary with detection information
        """
        return {
            "bbox": self.bbox,
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "width": self.bbox[2] - self.bbox[0],
            "height": self.bbox[3] - self.bbox[1],
        }

    def get_area(self) -> int:
        """
        Calculate bounding box area.

        Returns:
            Area in pixels
        """
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def get_center(self) -> Tuple[int, int]:
        """
        Calculate bounding box center.

        Returns:
            Center coordinates (x, y)
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class YOLODetector:
    """
    Wrapper for YOLOv8 detection model.

    This class provides a high-level interface for animal detection
    using YOLOv8 with configurable parameters and batch processing.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        model_name: str = "yolov8m.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_manager: ModelManager instance for loading models
            model_name: YOLOv8 model variant to use
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image

        Raises:
            ValueError: If thresholds are invalid
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be 0-1, got {confidence_threshold}")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be 0-1, got {iou_threshold}")

        if max_detections < 1:
            raise ValueError(f"max_detections must be >= 1, got {max_detections}")

        self.model_manager = model_manager
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        # Load model
        logger.info(f"Initializing YOLO detector with {model_name}")
        self.model = self.model_manager.load_yolo_model(model_name)
        logger.info("YOLO detector ready")

    def detect(
        self,
        image: npt.NDArray[Any],
        conf_threshold: Optional[float] = None,
        return_annotated: bool = False,
    ) -> Tuple[List[Detection], Optional[npt.NDArray[Any]]]:
        """
        Detect animals in an image.

        Args:
            image: Input image as numpy array (RGB or BGR)
            conf_threshold: Override default confidence threshold
            return_annotated: If True, return annotated image

        Returns:
            Tuple of (detections list, annotated image or None)

        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        threshold = conf_threshold or self.confidence_threshold

        logger.debug(f"Running detection with confidence={threshold}")

        try:
            # Run inference
            results = self.model(
                image,
                conf=threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )[0]

            # Parse results
            detections = self._parse_results(results)

            logger.info(f"Detected {len(detections)} animals")

            # Generate annotated image if requested
            annotated_image = None
            if return_annotated:
                annotated_image = results.plot()
                logger.debug("Generated annotated image")

            return detections, annotated_image

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise

    def _parse_results(self, results: Any) -> List[Detection]:
        """
        Parse YOLO results into Detection objects.

        Args:
            results: YOLO results object

        Returns:
            List of Detection objects
        """
        detections: List[Detection] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Extract confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # Create detection object
            try:
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
                detections.append(detection)
            except ValueError as e:
                logger.warning(f"Invalid detection skipped: {e}")
                continue

        return detections

    def detect_batch(
        self,
        images: List[npt.NDArray[Any]],
        conf_threshold: Optional[float] = None,
    ) -> List[List[Detection]]:
        """
        Detect animals in multiple images (batch processing).

        Args:
            images: List of input images
            conf_threshold: Override default confidence threshold

        Returns:
            List of detection lists (one per image)

        Raises:
            ValueError: If images list is empty
        """
        if not images:
            raise ValueError("Images list cannot be empty")

        threshold = conf_threshold or self.confidence_threshold

        logger.info(f"Running batch detection on {len(images)} images")

        try:
            # Run batch inference
            results = self.model(
                images,
                conf=threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )

            # Parse results for each image
            all_detections = []
            for result in results:
                detections = self._parse_results(result)
                all_detections.append(detections)

            total_detections = sum(len(d) for d in all_detections)
            logger.info(f"Batch detection complete: {total_detections} total detections")

            return all_detections

        except Exception as e:
            logger.error(f"Batch detection failed: {e}", exc_info=True)
            raise

    def update_thresholds(
        self,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> None:
        """
        Update detection thresholds.

        Args:
            confidence: New confidence threshold
            iou: New IoU threshold
            max_det: New maximum detections

        Raises:
            ValueError: If thresholds are invalid
        """
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be 0-1, got {confidence}")
            self.confidence_threshold = confidence
            logger.info(f"Updated confidence threshold: {confidence}")

        if iou is not None:
            if not 0.0 <= iou <= 1.0:
                raise ValueError(f"iou must be 0-1, got {iou}")
            self.iou_threshold = iou
            logger.info(f"Updated IoU threshold: {iou}")

        if max_det is not None:
            if max_det < 1:
                raise ValueError(f"max_det must be >= 1, got {max_det}")
            self.max_detections = max_det
            logger.info(f"Updated max detections: {max_det}")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "device": str(next(self.model.model.parameters()).device),
            "num_classes": len(self.model.names),
            "class_names": list(self.model.names.values()),
        }

    def __repr__(self) -> str:
        """String representation of YOLODetector."""
        return (
            f"YOLODetector(model={self.model_name}, "
            f"conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold})"
        )


if __name__ == "__main__":
    """Test YOLO detector."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing YOLO Detector...")

    # Create model manager and detector
    model_mgr = ModelManager()
    detector = YOLODetector(
        model_manager=model_mgr,
        model_name="yolov8n.pt",
        confidence_threshold=0.25,
    )

    print(f"\n{detector}")

    # Get model info
    model_info = detector.get_model_info()
    print("\nModel info:")
    for key, value in model_info.items():
        if key == "class_names":
            print(f"  {key}: {len(value)} classes")
        else:
            print(f"  {key}: {value}")

    # Test with a random image
    print("\nTesting detection on random image...")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    detections, annotated = detector.detect(test_image, return_annotated=True)
    print(f"✅ Detection complete: {len(detections)} detections")

    if detections:
        print("\nDetection details:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det.class_name} (conf={det.confidence:.2f})")
            print(f"     bbox={det.bbox}, area={det.get_area()} px")

    # Test batch detection
    print("\nTesting batch detection...")
    batch_images = [test_image, test_image, test_image]
    batch_results = detector.detect_batch(batch_images)
    print(f"✅ Batch detection complete: {len(batch_results)} results")

    # Test threshold update
    print("\nTesting threshold update...")
    detector.update_thresholds(confidence=0.5, iou=0.5, max_det=10)
    print("✅ Thresholds updated")

    print("\nYOLO detector tests completed!")
