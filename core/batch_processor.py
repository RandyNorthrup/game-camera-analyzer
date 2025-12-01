"""
Batch processor for orchestrating the complete wildlife analysis pipeline.

This module coordinates detection, classification, cropping, and export operations
across multiple images with progress tracking and error recovery.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from core.classification_engine import ClassificationEngine, ClassificationResult
from core.cropping_engine import CropConfig, CropResult, CroppingEngine
from core.csv_exporter import CSVExporter, ExportConfig
from core.detection_engine import DetectionEngine, DetectionResult
from utils.image_utils import load_image
from utils.validators import validate_directory_writable, validate_file_exists

logger = logging.getLogger(__name__)


class BatchProcessingError(Exception):
    """Exception raised for batch processing errors."""

    pass


@dataclass
class BatchConfig:
    """
    Configuration for batch processing operations.

    Attributes:
        detect: Whether to run detection
        classify: Whether to run classification
        crop: Whether to crop detections
        export_csv: Whether to export results to CSV
        save_annotated: Whether to save annotated images
        continue_on_error: Continue processing if an image fails
        max_workers: Maximum number of parallel workers (not implemented yet)
    """

    detect: bool = True
    classify: bool = True
    crop: bool = True
    export_csv: bool = True
    save_annotated: bool = False
    continue_on_error: bool = True
    max_workers: int = 1  # Future: parallel processing

    def __post_init__(self) -> None:
        """Validate batch configuration."""
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")


@dataclass
class BatchProgress:
    """
    Progress tracking for batch processing.

    Attributes:
        total_images: Total number of images to process
        processed_images: Number of images processed
        successful_images: Number of successfully processed images
        failed_images: Number of failed images
        total_detections: Total number of detections across all images
        total_classifications: Total number of successful classifications
        total_crops: Total number of successful crops
        start_time: Processing start time
        current_image: Name of currently processing image
        errors: List of error messages
    """

    total_images: int = 0
    processed_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_detections: int = 0
    total_classifications: int = 0
    total_crops: int = 0
    start_time: float = field(default_factory=time.time)
    current_image: str = ""
    errors: List[str] = field(default_factory=list)

    def get_progress_percent(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images / self.total_images) * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def get_estimated_remaining(self) -> Optional[float]:
        """Get estimated remaining time in seconds."""
        if self.processed_images == 0:
            return None
        elapsed = self.get_elapsed_time()
        rate = elapsed / self.processed_images
        remaining = (self.total_images - self.processed_images) * rate
        return remaining

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary."""
        return {
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "successful_images": self.successful_images,
            "failed_images": self.failed_images,
            "total_detections": self.total_detections,
            "total_classifications": self.total_classifications,
            "total_crops": self.total_crops,
            "progress_percent": round(self.get_progress_percent(), 2),
            "elapsed_time_sec": round(self.get_elapsed_time(), 2),
            "estimated_remaining_sec": round(self.get_estimated_remaining() or 0, 2),
            "current_image": self.current_image,
            "error_count": len(self.errors),
        }


class BatchProcessor:
    """
    Orchestrates the complete wildlife detection and analysis pipeline.

    This processor coordinates:
    1. Image loading and validation
    2. Animal detection (YOLOv8)
    3. Species classification
    4. Crop extraction
    5. CSV export
    6. Progress tracking

    Example:
        >>> processor = BatchProcessor(
        ...     output_dir="results",
        ...     species_db_path="data/species_db.json"
        ... )
        >>> results = processor.process_directory("images/")
        >>> print(f"Processed {results.successful_images} images")
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        species_db_path: Union[str, Path],
        batch_config: Optional[BatchConfig] = None,
        crop_config: Optional[CropConfig] = None,
        export_config: Optional[ExportConfig] = None,
        detection_confidence: float = 0.25,
        classification_confidence: float = 0.5,
        use_feature_classifier: bool = False,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        model_name: str = "yolov8m.pt",
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        device: str = "auto",
        enhance_low_light: bool = True,
        denoise_images: bool = False,
    ) -> None:
        """
        Initialize batch processor.

        Args:
            output_dir: Base output directory for all results
            species_db_path: Path to species database JSON
            batch_config: Batch processing configuration
            crop_config: Crop configuration
            export_config: Export configuration
            detection_confidence: Minimum confidence for detections
            classification_confidence: Minimum confidence for classifications
            use_feature_classifier: Whether to use CNN feature classifier
            progress_callback: Optional callback for progress updates
            model_name: YOLO model to use (e.g., "yolov8n.pt", "yolov8m.pt")
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections per image
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            enhance_low_light: Automatically enhance dark images
            denoise_images: Apply denoising to reduce false positives

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If required files not found
        """
        logger.info("Initializing BatchProcessor")

        self.output_dir = Path(output_dir)
        validate_directory_writable(self.output_dir, create=True)

        self.species_db_path = Path(species_db_path)
        validate_file_exists(self.species_db_path, "species database")

        self.batch_config = batch_config or BatchConfig()
        self.crop_config = crop_config or CropConfig()
        self.export_config = export_config or ExportConfig()
        self.progress_callback = progress_callback

        # Store detection settings
        self.model_name = model_name
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.enhance_low_light = enhance_low_light
        self.denoise_images = denoise_images

        # Create output subdirectories
        self.crops_dir = self.output_dir / "crops"
        self.csv_dir = self.output_dir / "csv"
        self.annotated_dir = self.output_dir / "annotated"

        # Initialize engines
        logger.info("Initializing processing engines...")

        # Create model manager with device setting
        from models.model_manager import ModelManager
        model_manager = ModelManager(device=device if device != "auto" else None)

        self.detection_engine = DetectionEngine(
            model_name=model_name,
            confidence_threshold=detection_confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            model_manager=model_manager,
            enhance_low_light=enhance_low_light,
            denoise_images=denoise_images,
        )

        if self.batch_config.classify:
            self.classification_engine = ClassificationEngine(
                species_db_path=self.species_db_path,
                use_feature_classifier=use_feature_classifier,
                confidence_threshold=classification_confidence,
            )
        else:
            self.classification_engine = None

        if self.batch_config.crop:
            self.cropping_engine = CroppingEngine(
                output_dir=self.crops_dir,
                config=self.crop_config,
                organize_by_species=True,
                organize_by_date=True,
            )
        else:
            self.cropping_engine = None

        if self.batch_config.export_csv:
            self.csv_exporter = CSVExporter(
                output_dir=self.csv_dir, config=self.export_config
            )
        else:
            self.csv_exporter = None

        logger.info(
            f"BatchProcessor initialized: "
            f"detect={self.batch_config.detect}, "
            f"classify={self.batch_config.classify}, "
            f"crop={self.batch_config.crop}"
        )

    def process_directory(
        self,
        input_dir: Union[str, Path],
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
    ) -> BatchProgress:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing images
            recursive: Whether to search subdirectories
            file_patterns: List of glob patterns (e.g., ["*.jpg", "*.png"])

        Returns:
            BatchProgress with processing statistics

        Raises:
            FileNotFoundError: If input directory doesn't exist
            BatchProcessingError: If processing fails critically
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        logger.info(f"Processing directory: {input_path}")

        # Find all images
        if file_patterns is None:
            file_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

        image_files: List[Path] = []
        for pattern in file_patterns:
            if recursive:
                image_files.extend(input_path.rglob(pattern))
            else:
                image_files.extend(input_path.glob(pattern))

        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return BatchProgress(total_images=0)

        logger.info(f"Found {len(image_files)} images to process")

        return self.process_images(image_files)

    def process_images(
        self,
        image_paths: List[Union[str, Path]],
        use_parallel: Optional[bool] = None
    ) -> BatchProgress:
        """
        Process a list of images with optional parallel execution.

        Args:
            image_paths: List of image file paths
            use_parallel: Use parallel processing (None = auto-detect from config)

        Returns:
            BatchProgress with processing statistics

        Raises:
            BatchProcessingError: If processing fails critically
        """
        # Determine if parallel processing should be used
        should_use_parallel = use_parallel if use_parallel is not None else (
            self.batch_config.max_workers > 1
        )
        
        if should_use_parallel and self.batch_config.max_workers > 1:
            logger.info(
                f"Using parallel processing with {self.batch_config.max_workers} workers"
            )
            return self._process_images_parallel(image_paths)
        else:
            logger.info("Using sequential processing")
            return self._process_images_sequential(image_paths)
    
    def _process_images_sequential(self, image_paths: List[Union[str, Path]]) -> BatchProgress:
        """
        Process images sequentially (original implementation).

        Args:
            image_paths: List of image file paths

        Returns:
            BatchProgress with processing statistics

        Raises:
            BatchProcessingError: If processing fails critically
        """
        progress = BatchProgress(total_images=len(image_paths))

        # Storage for results
        all_detection_results: List[DetectionResult] = []
        all_classification_results: List[List[Optional[ClassificationResult]]] = []
        all_crop_results: List[List[Optional[CropResult]]] = []

        logger.info(f"Starting batch processing of {len(image_paths)} images")

        for i, image_path in enumerate(image_paths, 1):
            image_path = Path(image_path)
            progress.current_image = image_path.name
            progress.processed_images = i

            try:
                # Process single image
                det_result, class_results, crop_results = self._process_single_image(
                    image_path
                )

                # Store results
                if det_result is not None:
                    all_detection_results.append(det_result)
                    progress.successful_images += 1
                    progress.total_detections += len(det_result.detections)

                if class_results is not None:
                    all_classification_results.append(class_results)
                    progress.total_classifications += sum(
                        1 for c in class_results if c is not None
                    )

                if crop_results is not None:
                    all_crop_results.append(crop_results)
                    progress.total_crops += sum(
                        1 for c in crop_results if c is not None
                    )

            except Exception as e:
                progress.failed_images += 1
                error_msg = f"{image_path.name}: {str(e)}"
                progress.errors.append(error_msg)
                logger.error(f"Failed to process {image_path}: {e}", exc_info=True)

                if not self.batch_config.continue_on_error:
                    raise BatchProcessingError(
                        f"Processing failed on {image_path}: {e}"
                    ) from e

            # Call progress callback
            if self.progress_callback:
                try:
                    self.progress_callback(progress)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}", exc_info=True)

            # Log progress periodically
            if i % 10 == 0 or i == len(image_paths):
                logger.info(
                    f"Progress: {i}/{len(image_paths)} "
                    f"({progress.get_progress_percent():.1f}%), "
                    f"{progress.total_detections} detections, "
                    f"elapsed: {progress.get_elapsed_time():.1f}s"
                )

        # Export results to CSV
        if self.batch_config.export_csv and self.csv_exporter and all_detection_results:
            try:
                logger.info("Exporting results to CSV...")
                self.csv_exporter.export_combined(
                    all_detection_results,
                    all_classification_results if all_classification_results else None,
                    all_crop_results if all_crop_results else None,
                    output_filename="batch_results.csv",
                )
                logger.info("CSV export complete")
            except Exception as e:
                logger.error(f"Failed to export CSV: {e}", exc_info=True)
                progress.errors.append(f"CSV export failed: {e}")

        # Log final summary
        logger.info(
            f"Batch processing complete: "
            f"{progress.successful_images}/{progress.total_images} successful, "
            f"{progress.failed_images} failed, "
            f"{progress.total_detections} detections, "
            f"{progress.total_classifications} classifications, "
            f"{progress.total_crops} crops, "
            f"time: {progress.get_elapsed_time():.1f}s"
        )

        return progress
    
    def _process_images_parallel(self, image_paths: List[Union[str, Path]]) -> BatchProgress:
        """
        Process images using parallel workers.

        Args:
            image_paths: List of image paths to process

        Returns:
            BatchProgress with statistics

        Raises:
            BatchProcessingError: If processing fails critically
        """
        from core.parallel_processor import ParallelConfig, ParallelBatchProcessor
        
        progress = BatchProgress(total_images=len(image_paths))
        
        # Create parallel config from batch config
        parallel_config = ParallelConfig(
            max_workers=self.batch_config.max_workers,
            chunk_size=10,
            enable_gpu_workers=False,  # Conservative default
            memory_limit_mb=2048,
            timeout_seconds=300
        )
        
        # Create progress callback
        def on_progress(current: int, total: int) -> None:
            progress.processed_images = current
            if self.progress_callback:
                self.progress_callback(progress)
            
            # Log progress periodically
            if current % 10 == 0 or current == total:
                logger.info(
                    f"Progress: {current}/{total} "
                    f"({progress.get_progress_percent():.1f}%)"
                )
        
        # Convert paths to Path objects
        paths = [Path(p) for p in image_paths]
        
        # Create parallel processor
        parallel_processor = ParallelBatchProcessor(
            process_function=self._process_single_image,
            config=parallel_config,
            progress_callback=on_progress
        )
        
        # Process images
        try:
            collector = parallel_processor.process_images(paths)
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}", exc_info=True)
            if not self.batch_config.continue_on_error:
                raise BatchProcessingError(f"Parallel processing failed: {e}") from e
            return progress
        
        # Aggregate results
        detections, classifications, crops, errors = collector.get_results()
        
        progress.successful_images = len(detections)
        progress.failed_images = len(errors)
        progress.total_detections = sum(
            len(d.detections) if d else 0 for d in detections
        )
        progress.total_classifications = sum(
            len([c for c in cls if c]) if cls else 0 for cls in classifications
        )
        progress.total_crops = sum(
            len([c for c in crp if c]) if crp else 0 for crp in crops
        )
        
        # Collect error messages
        for image_path, error in errors:
            progress.errors.append(f"{image_path.name}: {error}")
        
        # Export results to CSV
        if self.batch_config.export_csv and self.csv_exporter and detections:
            try:
                logger.info("Exporting results to CSV...")
                self.csv_exporter.export_combined(
                    detections,
                    classifications if classifications else None,
                    crops if crops else None,
                    output_filename="batch_results.csv",
                )
                logger.info("CSV export complete")
            except Exception as e:
                logger.error(f"Failed to export CSV: {e}", exc_info=True)
                progress.errors.append(f"CSV export failed: {e}")
        
        # Log final summary
        logger.info(
            f"Parallel batch processing complete: "
            f"{progress.successful_images}/{progress.total_images} successful, "
            f"{progress.failed_images} failed, "
            f"{progress.total_detections} detections, "
            f"{progress.total_classifications} classifications, "
            f"{progress.total_crops} crops, "
            f"time: {progress.get_elapsed_time():.1f}s"
        )
        
        return progress

    def _process_single_image(
        self, image_path: Path
    ) -> Tuple[
        Optional[DetectionResult],
        Optional[List[Optional[ClassificationResult]]],
        Optional[List[Optional[CropResult]]],
    ]:
        """
        Process a single image through the complete pipeline.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (DetectionResult, classifications, crops)

        Raises:
            Exception: If processing fails
        """
        logger.debug(f"Processing image: {image_path}")

        # Load image
        image, metadata = load_image(image_path, color_mode="RGB")

        # Step 1: Detection
        if not self.batch_config.detect:
            return None, None, None

        det_result = self.detection_engine.process_image(
            image_path,
            return_annotated=self.batch_config.save_annotated,
            save_annotated=(
                self.annotated_dir / image_path.name
                if self.batch_config.save_annotated
                else None
            ),
        )

        if not det_result.has_detections():
            logger.debug(f"No detections in {image_path.name}")
            return det_result, None, None

        # Step 2: Classification
        classification_results: Optional[List[Optional[ClassificationResult]]] = None
        if self.batch_config.classify and self.classification_engine:
            classification_results = []
            for detection in det_result.detections:
                try:
                    result = self.classification_engine.classify_detection(
                        detection, image
                    )
                    classification_results.append(result)
                except Exception as e:
                    logger.error(f"Classification failed: {e}")
                    classification_results.append(None)

        # Step 3: Cropping
        crop_results: Optional[List[Optional[CropResult]]] = None
        if self.batch_config.crop and self.cropping_engine:
            crop_results = []
            for j, detection in enumerate(det_result.detections):
                classification = (
                    classification_results[j]
                    if classification_results and j < len(classification_results)
                    else None
                )
                try:
                    crop = self.cropping_engine.crop_detection(
                        image,
                        detection,
                        classification,
                        save=True,
                        source_filename=image_path.name,
                    )
                    crop_results.append(crop)
                except Exception as e:
                    logger.error(f"Cropping failed: {e}")
                    crop_results.append(None)

        return det_result, classification_results, crop_results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics from all engines.

        Returns:
            Dictionary with statistics from all components
        """
        stats: Dict[str, Any] = {}

        if self.cropping_engine:
            stats["cropping"] = self.cropping_engine.get_statistics()

        if self.csv_exporter and hasattr(self.csv_exporter, "get_statistics"):
            stats["export"] = self.csv_exporter.get_statistics()

        return stats

    def __repr__(self) -> str:
        """String representation of processor."""
        return (
            f"BatchProcessor(output={self.output_dir}, "
            f"detect={self.batch_config.detect}, "
            f"classify={self.batch_config.classify}, "
            f"crop={self.batch_config.crop})"
        )


if __name__ == "__main__":
    """Test batch processor."""
    import sys
    import tempfile
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing Batch Processor...\n")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        input_dir = tmppath / "images"
        output_dir = tmppath / "results"
        input_dir.mkdir()

        # Create test images
        print(f"Creating test images in {input_dir}...")
        from utils.image_utils import save_image

        for i in range(3):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            save_image(test_image, input_dir / f"test_image_{i+1}.jpg")

        print("Created 3 test images\n")

        # Create batch config
        batch_config = BatchConfig(
            detect=True,
            classify=True,
            crop=True,
            export_csv=True,
            save_annotated=False,
            continue_on_error=True,
        )

        # Progress callback
        def progress_callback(progress: BatchProgress) -> None:
            if progress.processed_images % 1 == 0:
                print(
                    f"  Progress: {progress.processed_images}/{progress.total_images} "
                    f"({progress.get_progress_percent():.0f}%) - "
                    f"{progress.current_image}"
                )

        # Create processor
        db_path = Path(__file__).parent.parent / "data" / "species_db.json"

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=db_path,
            batch_config=batch_config,
            detection_confidence=0.25,
            classification_confidence=0.3,
            use_feature_classifier=False,
            progress_callback=progress_callback,
        )

        print(f"{processor}\n")

        # Process directory
        print("Processing images...")
        progress = processor.process_directory(input_dir)

        # Show results
        print("\n✅ Batch processing complete!")
        print("\nResults:")
        print(f"  Total images: {progress.total_images}")
        print(f"  Successful: {progress.successful_images}")
        print(f"  Failed: {progress.failed_images}")
        print(f"  Detections: {progress.total_detections}")
        print(f"  Classifications: {progress.total_classifications}")
        print(f"  Crops: {progress.total_crops}")
        print(f"  Elapsed time: {progress.get_elapsed_time():.2f}s")

        if progress.errors:
            print(f"\nErrors ({len(progress.errors)}):")
            for error in progress.errors[:5]:
                print(f"  - {error}")

        # Show statistics
        stats = processor.get_statistics()
        print("\nEngine statistics:")
        for engine, engine_stats in stats.items():
            print(f"  {engine}: {engine_stats}")

        # List created files
        print("\nCreated files:")
        for subdir in ["crops", "csv"]:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                print(f"  {subdir}/: {file_count} files")

    print("\nBatch processor tests completed!")
