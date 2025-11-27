"""
Model benchmarking infrastructure for Game Camera Analyzer.

Provides tools to benchmark YOLOv8 models for performance metrics:
- Inference speed (FPS)
- Detection accuracy (mAP, precision, recall)
- Memory usage
- Model comparison
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.model_manager import ModelManager
from models.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Results from a model benchmark run.

    Attributes:
        model_name: Name of benchmarked model
        device: Device used for inference
        num_images: Number of images processed
        total_time_seconds: Total processing time
        avg_fps: Average frames per second
        min_fps: Minimum FPS across all images
        max_fps: Maximum FPS across all images
        avg_detections_per_image: Average number of detections
        total_detections: Total detections across all images
        memory_mb: Peak memory usage in MB
        confidence_threshold: Confidence threshold used
        iou_threshold: IoU threshold used
        per_image_times: List of inference times per image
        error_count: Number of failed images
    """

    model_name: str
    device: str
    num_images: int
    total_time_seconds: float
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_detections_per_image: float
    total_detections: int
    memory_mb: float
    confidence_threshold: float
    iou_threshold: float
    per_image_times: List[float] = field(default_factory=list)
    error_count: int = 0

    def __post_init__(self) -> None:
        """Validate benchmark result after initialization."""
        if self.num_images < 0:
            raise ValueError(f"num_images must be >= 0, got {self.num_images}")

        if self.total_time_seconds < 0:
            raise ValueError(f"total_time must be >= 0, got {self.total_time_seconds}")

        if self.avg_fps < 0:
            raise ValueError(f"avg_fps must be >= 0, got {self.avg_fps}")

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be 0-1, got {self.confidence_threshold}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert benchmark result to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_images": self.num_images,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "avg_fps": round(self.avg_fps, 2),
            "min_fps": round(self.min_fps, 2),
            "max_fps": round(self.max_fps, 2),
            "avg_detections_per_image": round(self.avg_detections_per_image, 2),
            "total_detections": self.total_detections,
            "memory_mb": round(self.memory_mb, 2),
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "error_count": self.error_count,
        }

    def __str__(self) -> str:
        """String representation of benchmark result."""
        return (
            f"BenchmarkResult({self.model_name} on {self.device}): "
            f"{self.avg_fps:.2f} FPS, "
            f"{self.avg_detections_per_image:.2f} detections/image, "
            f"{self.memory_mb:.2f} MB"
        )


@dataclass
class ComparisonResult:
    """
    Comparison of multiple model benchmarks.

    Attributes:
        benchmarks: List of benchmark results
        fastest_model: Name of fastest model
        most_accurate_model: Name of most accurate model (most detections)
        most_efficient_model: Name with best FPS/detection ratio
    """

    benchmarks: List[BenchmarkResult]
    fastest_model: str = ""
    most_accurate_model: str = ""
    most_efficient_model: str = ""

    def __post_init__(self) -> None:
        """Calculate comparison metrics after initialization."""
        if not self.benchmarks:
            raise ValueError("benchmarks list cannot be empty")

        # Find fastest model
        fastest = max(self.benchmarks, key=lambda b: b.avg_fps)
        self.fastest_model = fastest.model_name

        # Find most accurate (most detections per image)
        most_accurate = max(self.benchmarks, key=lambda b: b.avg_detections_per_image)
        self.most_accurate_model = most_accurate.model_name

        # Find most efficient (best FPS per detection)
        most_efficient = max(
            self.benchmarks, key=lambda b: b.avg_fps / max(b.avg_detections_per_image, 1)
        )
        self.most_efficient_model = most_efficient.model_name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert comparison to dictionary.

        Returns:
            Dictionary with comparison results
        """
        return {
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "fastest_model": self.fastest_model,
            "most_accurate_model": self.most_accurate_model,
            "most_efficient_model": self.most_efficient_model,
        }


class ModelBenchmark:
    """
    Benchmark YOLOv8 models for performance and accuracy.

    Provides tools to measure inference speed, detection counts,
    and memory usage for model comparison.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None, warmup_iterations: int = 3):
        """
        Initialize model benchmark.

        Args:
            model_manager: ModelManager instance (creates new if None)
            warmup_iterations: Number of warmup runs before benchmarking
        """
        self.model_manager = model_manager or ModelManager()
        self.warmup_iterations = warmup_iterations

        logger.info(
            f"ModelBenchmark initialized: "
            f"device={self.model_manager.device}, "
            f"warmup={warmup_iterations}"
        )

    def benchmark_model(
        self,
        model_name: str,
        test_images: List[Path],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> BenchmarkResult:
        """
        Benchmark a single model.

        Args:
            model_name: Name of model to benchmark
            test_images: List of test image paths
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            BenchmarkResult with performance metrics

        Raises:
            ValueError: If test_images is empty or invalid
            RuntimeError: If benchmarking fails
        """
        if not test_images:
            raise ValueError("test_images cannot be empty")

        for img_path in test_images:
            if not img_path.exists():
                raise ValueError(f"Test image not found: {img_path}")

        logger.info(
            f"Benchmarking {model_name}: "
            f"{len(test_images)} images, "
            f"conf={confidence_threshold}, iou={iou_threshold}"
        )

        try:
            # Load model
            detector = YOLODetector(
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                model_manager=self.model_manager,
            )

            # Warmup runs
            if self.warmup_iterations > 0 and test_images:
                logger.debug(f"Running {self.warmup_iterations} warmup iterations")
                warmup_img = test_images[0]
                import cv2

                warmup_array = cv2.imread(str(warmup_img))

                for _ in range(self.warmup_iterations):
                    detector.detect(warmup_array)

            # Benchmark runs
            per_image_times: List[float] = []
            total_detections = 0
            error_count = 0

            import cv2
            import psutil
            import os

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB

            start_time = time.perf_counter()

            for img_path in test_images:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Could not load image: {img_path}")
                        error_count += 1
                        continue

                    # Time detection
                    img_start = time.perf_counter()
                    detections = detector.detect(img)
                    img_time = time.perf_counter() - img_start

                    per_image_times.append(img_time)
                    total_detections += len(detections)

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    error_count += 1

            total_time = time.perf_counter() - start_time

            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            peak_memory = mem_after - mem_before

            # Calculate metrics
            successful_images = len(per_image_times)

            if successful_images == 0:
                raise RuntimeError("All images failed to process")

            avg_time = total_time / successful_images
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0

            # Per-image FPS
            image_fps = [1.0 / t if t > 0 else 0.0 for t in per_image_times]
            min_fps = min(image_fps) if image_fps else 0.0
            max_fps = max(image_fps) if image_fps else 0.0

            avg_detections = total_detections / successful_images

            result = BenchmarkResult(
                model_name=model_name,
                device=self.model_manager.device,
                num_images=successful_images,
                total_time_seconds=total_time,
                avg_fps=avg_fps,
                min_fps=min_fps,
                max_fps=max_fps,
                avg_detections_per_image=avg_detections,
                total_detections=total_detections,
                memory_mb=peak_memory,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                per_image_times=per_image_times,
                error_count=error_count,
            )

            logger.info(f"Benchmark complete: {result}")
            return result

        except Exception as e:
            error_msg = f"Benchmarking failed for {model_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def compare_models(
        self,
        model_names: List[str],
        test_images: List[Path],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> ComparisonResult:
        """
        Compare multiple models on same test set.

        Args:
            model_names: List of model names to compare
            test_images: Test image paths
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            ComparisonResult with all benchmarks and analysis

        Raises:
            ValueError: If model_names is empty
            RuntimeError: If comparison fails
        """
        if not model_names:
            raise ValueError("model_names cannot be empty")

        logger.info(f"Comparing {len(model_names)} models on {len(test_images)} images")

        benchmarks: List[BenchmarkResult] = []

        for model_name in model_names:
            try:
                logger.info(f"Benchmarking model: {model_name}")
                result = self.benchmark_model(
                    model_name=model_name,
                    test_images=test_images,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                )
                benchmarks.append(result)

                # Clear cache to ensure fair comparison
                self.model_manager.clear_cache()

            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                # Continue with other models

        if not benchmarks:
            raise RuntimeError("All model benchmarks failed")

        comparison = ComparisonResult(benchmarks=benchmarks)

        logger.info(
            f"Comparison complete: "
            f"fastest={comparison.fastest_model}, "
            f"most_accurate={comparison.most_accurate_model}, "
            f"most_efficient={comparison.most_efficient_model}"
        )

        return comparison

    def export_results(
        self, results: BenchmarkResult, output_path: Path, format: str = "json"
    ) -> None:
        """
        Export benchmark results to file.

        Args:
            results: Benchmark results to export
            output_path: Output file path
            format: Export format ("json" or "csv")

        Raises:
            ValueError: If format is invalid
            RuntimeError: If export fails
        """
        if format not in ["json", "csv"]:
            raise ValueError(f"Invalid format: {format}. Must be 'json' or 'csv'")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                import json

                with open(output_path, "w") as f:
                    json.dump(results.to_dict(), f, indent=2)

                logger.info(f"Exported results to JSON: {output_path}")

            elif format == "csv":
                import csv

                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results.to_dict().keys())
                    writer.writeheader()
                    writer.writerow(results.to_dict())

                logger.info(f"Exported results to CSV: {output_path}")

        except Exception as e:
            error_msg = f"Failed to export results: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def export_comparison(
        self, comparison: ComparisonResult, output_path: Path, format: str = "json"
    ) -> None:
        """
        Export comparison results to file.

        Args:
            comparison: Comparison results to export
            output_path: Output file path
            format: Export format ("json" or "csv")

        Raises:
            ValueError: If format is invalid
            RuntimeError: If export fails
        """
        if format not in ["json", "csv"]:
            raise ValueError(f"Invalid format: {format}. Must be 'json' or 'csv'")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                import json

                with open(output_path, "w") as f:
                    json.dump(comparison.to_dict(), f, indent=2)

                logger.info(f"Exported comparison to JSON: {output_path}")

            elif format == "csv":
                import csv

                with open(output_path, "w", newline="") as f:
                    # Write summary
                    f.write(f"Fastest Model,{comparison.fastest_model}\n")
                    f.write(f"Most Accurate Model,{comparison.most_accurate_model}\n")
                    f.write(f"Most Efficient Model,{comparison.most_efficient_model}\n")
                    f.write("\n")

                    # Write benchmark details
                    if comparison.benchmarks:
                        fieldnames = comparison.benchmarks[0].to_dict().keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for benchmark in comparison.benchmarks:
                            writer.writerow(benchmark.to_dict())

                logger.info(f"Exported comparison to CSV: {output_path}")

        except Exception as e:
            error_msg = f"Failed to export comparison: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
