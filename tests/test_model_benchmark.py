"""
Tests for model benchmark module.

Tests benchmarking infrastructure, performance metrics calculation,
model comparison, and result export functionality.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from models.model_benchmark import (
    BenchmarkResult,
    ComparisonResult,
    ModelBenchmark,
)
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    Create temporary output directory for test files.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_images(tmp_path: Path) -> list[Path]:
    """
    Create test images for benchmarking.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        List of test image paths
    """
    images_dir = tmp_path / "test_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for i in range(5):
        # Create simple test images
        img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        img_path = images_dir / f"test_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        image_paths.append(img_path)

    return image_paths


@pytest.fixture
def mock_model_manager() -> ModelManager:
    """
    Create mock ModelManager for testing.

    Returns:
        Mock ModelManager instance
    """
    manager = Mock(spec=ModelManager)
    manager.device = "cpu"
    manager.clear_cache = Mock()
    return manager


@pytest.fixture
def benchmark(mock_model_manager: ModelManager) -> ModelBenchmark:
    """
    Create ModelBenchmark instance for testing.

    Args:
        mock_model_manager: Mock ModelManager

    Returns:
        ModelBenchmark instance
    """
    return ModelBenchmark(model_manager=mock_model_manager, warmup_iterations=0)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_valid_benchmark_result(self) -> None:
        """Test valid BenchmarkResult creation."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.5,
            total_detections=35,
            memory_mb=512.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        assert result.model_name == "yolov8n.pt"
        assert result.device == "cpu"
        assert result.num_images == 10
        assert result.avg_fps == 2.0

    def test_invalid_num_images(self) -> None:
        """Test BenchmarkResult rejects negative num_images."""
        with pytest.raises(ValueError, match="num_images must be >= 0"):
            BenchmarkResult(
                model_name="yolov8n.pt",
                device="cpu",
                num_images=-1,
                total_time_seconds=5.0,
                avg_fps=2.0,
                min_fps=1.5,
                max_fps=2.5,
                avg_detections_per_image=3.5,
                total_detections=35,
                memory_mb=512.0,
                confidence_threshold=0.25,
                iou_threshold=0.45,
            )

    def test_invalid_total_time(self) -> None:
        """Test BenchmarkResult rejects negative total_time."""
        with pytest.raises(ValueError, match="total_time must be >= 0"):
            BenchmarkResult(
                model_name="yolov8n.pt",
                device="cpu",
                num_images=10,
                total_time_seconds=-1.0,
                avg_fps=2.0,
                min_fps=1.5,
                max_fps=2.5,
                avg_detections_per_image=3.5,
                total_detections=35,
                memory_mb=512.0,
                confidence_threshold=0.25,
                iou_threshold=0.45,
            )

    def test_invalid_avg_fps(self) -> None:
        """Test BenchmarkResult rejects negative avg_fps."""
        with pytest.raises(ValueError, match="avg_fps must be >= 0"):
            BenchmarkResult(
                model_name="yolov8n.pt",
                device="cpu",
                num_images=10,
                total_time_seconds=5.0,
                avg_fps=-1.0,
                min_fps=1.5,
                max_fps=2.5,
                avg_detections_per_image=3.5,
                total_detections=35,
                memory_mb=512.0,
                confidence_threshold=0.25,
                iou_threshold=0.45,
            )

    def test_invalid_confidence_threshold(self) -> None:
        """Test BenchmarkResult rejects invalid confidence_threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be 0-1"):
            BenchmarkResult(
                model_name="yolov8n.pt",
                device="cpu",
                num_images=10,
                total_time_seconds=5.0,
                avg_fps=2.0,
                min_fps=1.5,
                max_fps=2.5,
                avg_detections_per_image=3.5,
                total_detections=35,
                memory_mb=512.0,
                confidence_threshold=1.5,
                iou_threshold=0.45,
            )

    def test_to_dict(self) -> None:
        """Test BenchmarkResult to_dict conversion."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.123,
            avg_fps=2.456,
            min_fps=1.567,
            max_fps=2.789,
            avg_detections_per_image=3.5,
            total_detections=35,
            memory_mb=512.123,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        result_dict = result.to_dict()

        assert result_dict["model_name"] == "yolov8n.pt"
        assert result_dict["device"] == "cpu"
        assert result_dict["num_images"] == 10
        assert result_dict["total_time_seconds"] == 5.123
        assert result_dict["avg_fps"] == 2.46  # Rounded to 2 decimals
        assert result_dict["memory_mb"] == 512.12  # Rounded to 2 decimals

    def test_str_representation(self) -> None:
        """Test BenchmarkResult string representation."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.5,
            total_detections=35,
            memory_mb=512.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        str_repr = str(result)

        assert "yolov8n.pt" in str_repr
        assert "cpu" in str_repr
        assert "2.00 FPS" in str_repr


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_valid_comparison_result(self) -> None:
        """Test valid ComparisonResult creation."""
        benchmark1 = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        benchmark2 = BenchmarkResult(
            model_name="yolov8s.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=10.0,
            avg_fps=1.0,
            min_fps=0.8,
            max_fps=1.2,
            avg_detections_per_image=5.0,
            total_detections=50,
            memory_mb=512.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        comparison = ComparisonResult(benchmarks=[benchmark1, benchmark2])

        assert len(comparison.benchmarks) == 2
        assert comparison.fastest_model == "yolov8n.pt"
        assert comparison.most_accurate_model == "yolov8s.pt"

    def test_empty_benchmarks_list(self) -> None:
        """Test ComparisonResult rejects empty benchmarks list."""
        with pytest.raises(ValueError, match="benchmarks list cannot be empty"):
            ComparisonResult(benchmarks=[])

    def test_most_efficient_calculation(self) -> None:
        """Test ComparisonResult calculates most efficient model correctly."""
        # Model 1: 4 FPS, 2 detections/image = 2.0 FPS per detection
        benchmark1 = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=2.5,
            avg_fps=4.0,
            min_fps=3.5,
            max_fps=4.5,
            avg_detections_per_image=2.0,
            total_detections=20,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        # Model 2: 2 FPS, 5 detections/image = 0.4 FPS per detection
        benchmark2 = BenchmarkResult(
            model_name="yolov8s.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.8,
            max_fps=2.2,
            avg_detections_per_image=5.0,
            total_detections=50,
            memory_mb=512.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        comparison = ComparisonResult(benchmarks=[benchmark1, benchmark2])

        # yolov8n has better FPS/detection ratio
        assert comparison.most_efficient_model == "yolov8n.pt"

    def test_to_dict(self) -> None:
        """Test ComparisonResult to_dict conversion."""
        benchmark1 = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        comparison = ComparisonResult(benchmarks=[benchmark1])
        result_dict = comparison.to_dict()

        assert "benchmarks" in result_dict
        assert "fastest_model" in result_dict
        assert "most_accurate_model" in result_dict
        assert "most_efficient_model" in result_dict
        assert len(result_dict["benchmarks"]) == 1


class TestModelBenchmark:
    """Tests for ModelBenchmark class."""

    def test_initialization(self, mock_model_manager: ModelManager) -> None:
        """Test ModelBenchmark initialization."""
        benchmark = ModelBenchmark(model_manager=mock_model_manager, warmup_iterations=3)

        assert benchmark.model_manager == mock_model_manager
        assert benchmark.warmup_iterations == 3

    def test_initialization_creates_default_manager(self) -> None:
        """Test ModelBenchmark creates default ModelManager if none provided."""
        with patch("models.model_benchmark.ModelManager") as mock_manager_class:
            mock_instance = Mock()
            mock_manager_class.return_value = mock_instance

            benchmark = ModelBenchmark()

            assert benchmark.model_manager == mock_instance

    @patch("models.model_benchmark.YOLODetector")
    @patch("cv2.imread")
    @patch("psutil.Process")
    def test_benchmark_model_success(
        self,
        mock_process: Mock,
        mock_imread: Mock,
        mock_detector_class: Mock,
        benchmark: ModelBenchmark,
        test_images: list[Path],
    ) -> None:
        """Test successful model benchmarking."""
        # Setup mocks
        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        mock_detector = Mock()
        mock_detector.detect.return_value = [Mock(), Mock()]  # 2 detections per image
        mock_detector_class.return_value = mock_detector

        # Mock memory info
        mock_mem_info = Mock()
        mock_mem_info.rss = 1024 * 1024 * 512  # 512 MB
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = mock_mem_info
        mock_process.return_value = mock_process_instance

        # Run benchmark
        result = benchmark.benchmark_model(
            model_name="yolov8n.pt",
            test_images=test_images[:3],  # Use 3 images
            confidence_threshold=0.3,
            iou_threshold=0.5,
        )

        # Verify results
        assert result.model_name == "yolov8n.pt"
        assert result.device == "cpu"
        assert result.num_images == 3
        assert result.total_detections == 6  # 2 detections * 3 images
        assert result.avg_detections_per_image == 2.0
        assert result.confidence_threshold == 0.3
        assert result.iou_threshold == 0.5
        assert result.avg_fps > 0
        assert result.error_count == 0

    def test_benchmark_model_empty_images(self, benchmark: ModelBenchmark) -> None:
        """Test benchmark_model rejects empty image list."""
        with pytest.raises(ValueError, match="test_images cannot be empty"):
            benchmark.benchmark_model(
                model_name="yolov8n.pt",
                test_images=[],
            )

    def test_benchmark_model_nonexistent_image(self, benchmark: ModelBenchmark) -> None:
        """Test benchmark_model rejects non-existent image paths."""
        with pytest.raises(ValueError, match="Test image not found"):
            benchmark.benchmark_model(
                model_name="yolov8n.pt",
                test_images=[Path("/nonexistent/image.jpg")],
            )

    @patch("models.model_benchmark.YOLODetector")
    @patch("cv2.imread")
    @patch("psutil.Process")
    def test_benchmark_model_with_errors(
        self,
        mock_process: Mock,
        mock_imread: Mock,
        mock_detector_class: Mock,
        benchmark: ModelBenchmark,
        test_images: list[Path],
    ) -> None:
        """Test benchmark handles image processing errors."""
        # Setup mocks - imread fails for some images
        mock_imread.side_effect = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # Success
            None,  # Fail to load
            np.zeros((480, 640, 3), dtype=np.uint8),  # Success
        ]

        mock_detector = Mock()
        mock_detector.detect.return_value = [Mock()]
        mock_detector_class.return_value = mock_detector

        mock_mem_info = Mock()
        mock_mem_info.rss = 1024 * 1024 * 256
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = mock_mem_info
        mock_process.return_value = mock_process_instance

        result = benchmark.benchmark_model(
            model_name="yolov8n.pt",
            test_images=test_images[:3],
        )

        assert result.num_images == 2  # Only 2 successful
        assert result.error_count == 1

    @patch("models.model_benchmark.YOLODetector")
    def test_benchmark_model_all_failures(
        self,
        mock_detector_class: Mock,
        benchmark: ModelBenchmark,
        test_images: list[Path],
    ) -> None:
        """Test benchmark fails when all images fail to process."""
        mock_detector_class.side_effect = Exception("Detector failed")

        with pytest.raises(RuntimeError, match="Benchmarking failed"):
            benchmark.benchmark_model(
                model_name="yolov8n.pt",
                test_images=test_images[:1],
            )

    @patch("models.model_benchmark.YOLODetector")
    @patch("cv2.imread")
    @patch("psutil.Process")
    def test_compare_models(
        self,
        mock_process: Mock,
        mock_imread: Mock,
        mock_detector_class: Mock,
        benchmark: ModelBenchmark,
        test_images: list[Path],
    ) -> None:
        """Test comparing multiple models."""
        # Setup mocks
        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img

        mock_detector = Mock()
        mock_detector.detect.return_value = [Mock(), Mock()]
        mock_detector_class.return_value = mock_detector

        mock_mem_info = Mock()
        mock_mem_info.rss = 1024 * 1024 * 256
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = mock_mem_info
        mock_process.return_value = mock_process_instance

        comparison = benchmark.compare_models(
            model_names=["yolov8n.pt", "yolov8s.pt"],
            test_images=test_images[:2],
        )

        assert len(comparison.benchmarks) == 2
        assert comparison.fastest_model in ["yolov8n.pt", "yolov8s.pt"]
        assert comparison.most_accurate_model in ["yolov8n.pt", "yolov8s.pt"]

        # Verify cache was cleared between models
        assert benchmark.model_manager.clear_cache.call_count == 2

    def test_compare_models_empty_list(self, benchmark: ModelBenchmark) -> None:
        """Test compare_models rejects empty model list."""
        with pytest.raises(ValueError, match="model_names cannot be empty"):
            benchmark.compare_models(
                model_names=[],
                test_images=[Path("test.jpg")],
            )

    @patch("models.model_benchmark.YOLODetector")
    def test_compare_models_all_fail(
        self,
        mock_detector_class: Mock,
        benchmark: ModelBenchmark,
        test_images: list[Path],
    ) -> None:
        """Test compare_models fails when all benchmarks fail."""
        mock_detector_class.side_effect = Exception("Failed")

        with pytest.raises(RuntimeError, match="All model benchmarks failed"):
            benchmark.compare_models(
                model_names=["yolov8n.pt", "yolov8s.pt"],
                test_images=test_images[:1],
            )

    def test_export_results_json(self, benchmark: ModelBenchmark, temp_output_dir: Path) -> None:
        """Test exporting benchmark results to JSON."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        output_path = temp_output_dir / "benchmark_results.json"

        benchmark.export_results(result, output_path, format="json")

        assert output_path.exists()

        # Verify content
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["model_name"] == "yolov8n.pt"
        assert data["device"] == "cpu"
        assert data["num_images"] == 10

    def test_export_results_csv(self, benchmark: ModelBenchmark, temp_output_dir: Path) -> None:
        """Test exporting benchmark results to CSV."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        output_path = temp_output_dir / "benchmark_results.csv"

        benchmark.export_results(result, output_path, format="csv")

        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "model_name" in content
        assert "yolov8n.pt" in content
        assert "cpu" in content

    def test_export_results_invalid_format(
        self, benchmark: ModelBenchmark, temp_output_dir: Path
    ) -> None:
        """Test export_results rejects invalid format."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        with pytest.raises(ValueError, match="Invalid format"):
            benchmark.export_results(result, temp_output_dir / "test.txt", format="xml")

    def test_export_comparison_json(self, benchmark: ModelBenchmark, temp_output_dir: Path) -> None:
        """Test exporting comparison results to JSON."""
        benchmark1 = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        comparison = ComparisonResult(benchmarks=[benchmark1])
        output_path = temp_output_dir / "comparison.json"

        benchmark.export_comparison(comparison, output_path, format="json")

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "benchmarks" in data
        assert "fastest_model" in data

    def test_export_comparison_csv(self, benchmark: ModelBenchmark, temp_output_dir: Path) -> None:
        """Test exporting comparison results to CSV."""
        benchmark1 = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        benchmark2 = BenchmarkResult(
            model_name="yolov8s.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=10.0,
            avg_fps=1.0,
            min_fps=0.8,
            max_fps=1.2,
            avg_detections_per_image=5.0,
            total_detections=50,
            memory_mb=512.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )

        comparison = ComparisonResult(benchmarks=[benchmark1, benchmark2])
        output_path = temp_output_dir / "comparison.csv"

        benchmark.export_comparison(comparison, output_path, format="csv")

        assert output_path.exists()

        content = output_path.read_text()
        assert "Fastest Model" in content
        assert "yolov8n.pt" in content or "yolov8s.pt" in content
