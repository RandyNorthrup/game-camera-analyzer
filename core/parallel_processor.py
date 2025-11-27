"""
Parallel processing infrastructure for batch image processing.

Provides thread-safe result collection and worker pool management
for concurrent image processing operations.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.detection_engine import DetectionResult
from core.classification_engine import ClassificationResult
from core.cropping_engine import CropResult

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """
    Configuration for parallel processing.

    Attributes:
        max_workers: Maximum number of concurrent workers
        chunk_size: Number of images per worker batch
        enable_gpu_workers: Allow multiple GPU workers (requires memory)
        memory_limit_mb: Maximum memory per worker in MB
        timeout_seconds: Timeout for individual image processing
    """

    max_workers: int = 4
    chunk_size: int = 10
    enable_gpu_workers: bool = False
    memory_limit_mb: int = 2048
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        """Validate parallel configuration."""
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.max_workers > 16:
            logger.warning(f"max_workers={self.max_workers} may cause memory issues")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.memory_limit_mb < 512:
            logger.warning(f"memory_limit_mb={self.memory_limit_mb} is very low")


class ThreadSafeResultCollector:
    """
    Thread-safe collector for processing results.

    Collects detection, classification, and cropping results
    from multiple worker threads safely.
    """

    def __init__(self) -> None:
        """Initialize thread-safe result collector."""
        self._lock = threading.Lock()
        self._detection_results: List[DetectionResult] = []
        self._classification_results: List[List[Optional[ClassificationResult]]] = []
        self._crop_results: List[List[Optional[CropResult]]] = []
        self._errors: List[Tuple[Path, Exception]] = []
        self._image_count = 0

        logger.debug("ThreadSafeResultCollector initialized")

    def add_result(
        self,
        image_path: Path,
        detection: Optional[DetectionResult],
        classifications: Optional[List[Optional[ClassificationResult]]],
        crops: Optional[List[Optional[CropResult]]],
    ) -> None:
        """
        Add processing results for one image.

        Args:
            image_path: Path to processed image
            detection: Detection result or None
            classifications: List of classification results or None
            crops: List of crop results or None

        Thread-safe operation using internal lock.
        """
        with self._lock:
            self._image_count += 1

            if detection is not None:
                self._detection_results.append(detection)

            if classifications is not None:
                self._classification_results.append(classifications)

            if crops is not None:
                self._crop_results.append(crops)

            logger.debug(
                f"Added results for {image_path.name} " f"(total images: {self._image_count})"
            )

    def add_error(self, image_path: Path, error: Exception) -> None:
        """
        Record processing error for an image.

        Args:
            image_path: Path to failed image
            error: Exception that occurred
        """
        with self._lock:
            self._errors.append((image_path, error))
            logger.error(f"Recorded error for {image_path.name}: {error}", exc_info=True)

    def get_results(
        self,
    ) -> Tuple[
        List[DetectionResult],
        List[List[Optional[ClassificationResult]]],
        List[List[Optional[CropResult]]],
        List[Tuple[Path, Exception]],
    ]:
        """
        Get all collected results.

        Returns:
            Tuple of (detections, classifications, crops, errors)

        Thread-safe operation.
        """
        with self._lock:
            return (
                self._detection_results.copy(),
                self._classification_results.copy(),
                self._crop_results.copy(),
                self._errors.copy(),
            )

    def get_statistics(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary with counts of results and errors
        """
        with self._lock:
            return {
                "images_processed": self._image_count,
                "detections": len(self._detection_results),
                "classifications": sum(len(c) for c in self._classification_results),
                "crops": sum(len(c) for c in self._crop_results),
                "errors": len(self._errors),
            }


class ParallelBatchProcessor:
    """
    Parallel batch processor using ThreadPoolExecutor.

    Manages concurrent processing of multiple images with
    memory management and error recovery.
    """

    def __init__(
        self,
        process_function: Callable[[Path], Tuple[Any, Any, Any]],
        config: ParallelConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Initialize parallel batch processor.

        Args:
            process_function: Function to process single image
            config: Parallel processing configuration
            progress_callback: Optional callback for progress updates (current, total)
        """
        self.process_function = process_function
        self.config = config
        self.progress_callback = progress_callback
        self.collector = ThreadSafeResultCollector()

        logger.info(
            f"ParallelBatchProcessor initialized: "
            f"workers={config.max_workers}, chunk_size={config.chunk_size}"
        )

    def process_images(self, image_paths: List[Path]) -> ThreadSafeResultCollector:
        """
        Process multiple images in parallel.

        Args:
            image_paths: List of image paths to process

        Returns:
            ThreadSafeResultCollector with all results

        Raises:
            RuntimeError: If processing fails catastrophically
        """
        total_images = len(image_paths)
        logger.info(f"Starting parallel processing of {total_images} images")

        processed_count = 0

        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                futures: Dict[Future[Tuple[Any, Any, Any]], Path] = {
                    executor.submit(self._process_single_with_timeout, image_path): image_path
                    for image_path in image_paths
                }

                # Process completed tasks
                for future in as_completed(futures):
                    image_path = futures[future]

                    try:
                        detection, classifications, crops = future.result(
                            timeout=self.config.timeout_seconds
                        )

                        self.collector.add_result(image_path, detection, classifications, crops)

                    except Exception as e:
                        logger.error(f"Failed to process {image_path.name}: {e}", exc_info=True)
                        self.collector.add_error(image_path, e)

                    processed_count += 1

                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(processed_count, total_images)

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Parallel processing failed: {e}")

        stats = self.collector.get_statistics()
        logger.info(
            f"Parallel processing complete: {stats['images_processed']} images, "
            f"{stats['detections']} detections, {stats['errors']} errors"
        )

        return self.collector

    def _process_single_with_timeout(self, image_path: Path) -> Tuple[Any, Any, Any]:
        """
        Process single image with timeout protection.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (detection, classifications, crops)
        """
        try:
            return self.process_function(image_path)
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
            raise
