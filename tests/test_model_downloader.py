"""
Tests for model downloader module.

Tests downloading, caching, validation, and metadata management
for YOLOv8 models.
"""

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models.model_downloader import (
    ModelDownloader,
    ModelInfo,
    DownloadProgress,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_download_dir(tmp_path: Path) -> Path:
    """
    Create temporary download directory.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to download directory
    """
    download_dir = tmp_path / "models"
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


@pytest.fixture
def downloader(temp_download_dir: Path) -> ModelDownloader:
    """
    Create ModelDownloader with temporary directory.

    Args:
        temp_download_dir: Temporary download directory

    Returns:
        ModelDownloader instance
    """
    return ModelDownloader(download_dir=temp_download_dir)


@pytest.fixture
def sample_model_file(temp_download_dir: Path) -> Path:
    """
    Create a sample .pt file for testing.

    Args:
        temp_download_dir: Temporary download directory

    Returns:
        Path to sample model file
    """
    model_path = temp_download_dir / "test_model.pt"
    model_path.write_bytes(b"fake model data")
    return model_path


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_valid_model_info(self) -> None:
        """Test valid ModelInfo creation."""
        info = ModelInfo(
            name="yolov8n.pt",
            size="n",
            description="Test model",
            parameters_millions=3.2,
            macs_billions=8.7,
            size_mb=6.2,
        )

        assert info.name == "yolov8n.pt"
        assert info.size == "n"
        assert info.is_downloaded is False

    def test_invalid_name_without_pt(self) -> None:
        """Test ModelInfo rejects names without .pt extension."""
        with pytest.raises(ValueError, match="must end with .pt"):
            ModelInfo(
                name="yolov8n",
                size="n",
                description="Test",
                parameters_millions=3.2,
                macs_billions=8.7,
                size_mb=6.2,
            )

    def test_invalid_size(self) -> None:
        """Test ModelInfo rejects invalid size values."""
        with pytest.raises(ValueError, match="Model size must be one of"):
            ModelInfo(
                name="yolov8z.pt",
                size="z",
                description="Test",
                parameters_millions=3.2,
                macs_billions=8.7,
                size_mb=6.2,
            )

    def test_negative_parameters(self) -> None:
        """Test ModelInfo rejects negative parameters."""
        with pytest.raises(ValueError, match="Parameters must be >= 0"):
            ModelInfo(
                name="yolov8n.pt",
                size="n",
                description="Test",
                parameters_millions=-1.0,
                macs_billions=8.7,
                size_mb=6.2,
            )

    def test_negative_size(self) -> None:
        """Test ModelInfo rejects negative file size."""
        with pytest.raises(ValueError, match="Size must be >= 0 MB"):
            ModelInfo(
                name="yolov8n.pt",
                size="n",
                description="Test",
                parameters_millions=3.2,
                macs_billions=8.7,
                size_mb=-5.0,
            )


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_initial_progress(self) -> None:
        """Test DownloadProgress initial state."""
        progress = DownloadProgress(model_name="yolov8n.pt")

        assert progress.model_name == "yolov8n.pt"
        assert progress.total_bytes == 0
        assert progress.downloaded_bytes == 0
        assert progress.percent_complete == 0.0
        assert progress.is_complete is False

    def test_update_progress(self) -> None:
        """Test DownloadProgress update method."""
        progress = DownloadProgress(model_name="yolov8n.pt")

        progress.update(downloaded=50, total=100, speed_mbps=10.0)

        assert progress.downloaded_bytes == 50
        assert progress.total_bytes == 100
        assert progress.percent_complete == 50.0
        assert progress.speed_mbps == 10.0

    def test_update_with_zero_total(self) -> None:
        """Test DownloadProgress handles zero total bytes."""
        progress = DownloadProgress(model_name="yolov8n.pt")

        progress.update(downloaded=50, total=0)

        assert progress.percent_complete == 0.0

    def test_eta_calculation(self) -> None:
        """Test DownloadProgress ETA calculation."""
        progress = DownloadProgress(model_name="yolov8n.pt")

        # 100 MB total, 50 MB downloaded, 10 MB/s speed
        # Remaining: 50 MB, ETA: 5 seconds
        progress.update(downloaded=50 * 1024 * 1024, total=100 * 1024 * 1024, speed_mbps=10.0)

        assert progress.eta_seconds == pytest.approx(5.0, rel=0.1)


class TestModelDownloader:
    """Tests for ModelDownloader class."""

    def test_initialization(self, temp_download_dir: Path) -> None:
        """Test ModelDownloader initialization."""
        downloader = ModelDownloader(download_dir=temp_download_dir)

        assert downloader.download_dir == temp_download_dir
        assert temp_download_dir.exists()

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Test ModelDownloader creates download directory."""
        new_dir = tmp_path / "new_models"

        downloader = ModelDownloader(download_dir=new_dir)

        assert new_dir.exists()
        assert downloader.download_dir == new_dir

    def test_list_available_models(self, downloader: ModelDownloader) -> None:
        """Test listing available models."""
        models = downloader.list_available_models()

        assert len(models) == 5
        assert all(isinstance(m, ModelInfo) for m in models)

        model_names = [m.name for m in models]
        assert "yolov8n.pt" in model_names
        assert "yolov8s.pt" in model_names
        assert "yolov8m.pt" in model_names
        assert "yolov8l.pt" in model_names
        assert "yolov8x.pt" in model_names

    def test_get_model_info_existing(self, downloader: ModelDownloader) -> None:
        """Test getting info for existing model."""
        info = downloader.get_model_info("yolov8n.pt")

        assert info is not None
        assert info.name == "yolov8n.pt"
        assert info.size == "n"
        assert info.parameters_millions == 3.2

    def test_get_model_info_nonexistent(self, downloader: ModelDownloader) -> None:
        """Test getting info for non-existent model."""
        info = downloader.get_model_info("yolov8z.pt")

        assert info is None

    def test_get_storage_info_empty(self, downloader: ModelDownloader) -> None:
        """Test storage info with no downloaded models."""
        storage_info = downloader.get_storage_info()

        assert storage_info["downloaded_count"] == 0
        assert storage_info["total_size_mb"] == 0.0
        assert storage_info["download_dir"] == str(downloader.download_dir)

    def test_get_storage_info_with_models(
        self, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test storage info with downloaded models."""
        # Create fake model files
        (temp_download_dir / "yolov8n.pt").write_bytes(b"x" * 1024 * 1024)  # 1 MB
        (temp_download_dir / "yolov8s.pt").write_bytes(b"x" * 2 * 1024 * 1024)  # 2 MB

        storage_info = downloader.get_storage_info()

        assert storage_info["downloaded_count"] == 2
        assert storage_info["total_size_mb"] == pytest.approx(3.0, rel=0.1)
        assert len(storage_info["models"]) == 2

    @patch("models.model_downloader.YOLO")
    def test_download_model_success(
        self, mock_yolo: Mock, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test successful model download."""
        # Setup mocks - create ultralytics cache
        with patch("models.model_downloader.Path.home") as mock_home:
            mock_home.return_value = temp_download_dir
            cache_dir = temp_download_dir / ".cache" / "ultralytics"
            cache_dir.mkdir(parents=True)
            cache_path = cache_dir / "yolov8n.pt"
            cache_path.write_bytes(b"model data")

            result = downloader.download_model("yolov8n.pt")

            assert result.name == "yolov8n.pt"
            assert result.is_downloaded

    def test_download_model_invalid_name(self, downloader: ModelDownloader) -> None:
        """Test download with invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            downloader.download_model("invalid_model.pt")

    @patch("models.model_downloader.YOLO")
    def test_download_model_already_exists(
        self, mock_yolo: Mock, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test download when model already exists."""
        # Create existing model file
        model_path = temp_download_dir / "yolov8n.pt"
        model_path.write_bytes(b"existing model")

        result = downloader.download_model("yolov8n.pt", force_redownload=False)

        assert result.name == "yolov8n.pt"
        assert result.is_downloaded
        # YOLO should not be called since file exists
        mock_yolo.assert_not_called()

    def test_calculate_sha256(self, downloader: ModelDownloader, sample_model_file: Path) -> None:
        """Test SHA-256 checksum calculation."""
        checksum = downloader._calculate_sha256(sample_model_file)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 is 64 hex characters

    def test_delete_model_success(
        self, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test successful model deletion."""
        # Create model file
        model_path = temp_download_dir / "yolov8n.pt"
        model_path.write_bytes(b"model data")

        result = downloader.delete_model("yolov8n.pt")

        assert result is True
        assert not model_path.exists()

    def test_delete_model_not_found(self, downloader: ModelDownloader) -> None:
        """Test deleting non-existent model."""
        result = downloader.delete_model("yolov8n.pt")

        assert result is False

    @patch("models.model_downloader.YOLO")
    def test_validate_model_valid(
        self, mock_yolo: Mock, downloader: ModelDownloader, sample_model_file: Path
    ) -> None:
        """Test validating a valid model file."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_yolo.return_value = mock_model

        result = downloader.validate_model(sample_model_file)

        assert result is True

    def test_validate_model_not_found(self, downloader: ModelDownloader) -> None:
        """Test validating non-existent model file."""
        result = downloader.validate_model(Path("/nonexistent/model.pt"))

        assert result is False

    def test_validate_model_wrong_extension(
        self, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test validating file with wrong extension."""
        wrong_file = temp_download_dir / "model.txt"
        wrong_file.write_bytes(b"data")

        result = downloader.validate_model(wrong_file)

        assert result is False

    @patch("models.model_downloader.YOLO")
    def test_validate_model_load_failure(
        self, mock_yolo: Mock, downloader: ModelDownloader, sample_model_file: Path
    ) -> None:
        """Test validating model that fails to load."""
        mock_yolo.side_effect = Exception("Load failed")

        result = downloader.validate_model(sample_model_file)

        assert result is False

    @patch("models.model_downloader.YOLO")
    def test_import_custom_model_success(
        self, mock_yolo: Mock, downloader: ModelDownloader, sample_model_file: Path
    ) -> None:
        """Test successful custom model import."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_yolo.return_value = mock_model

        result = downloader.import_custom_model(sample_model_file, "custom.pt")

        assert result.name == "custom.pt"
        assert result.size == "custom"
        assert result.is_downloaded

    def test_import_custom_model_not_found(self, downloader: ModelDownloader) -> None:
        """Test importing non-existent model file."""
        with pytest.raises(ValueError, match="Source file not found"):
            downloader.import_custom_model(Path("/nonexistent/model.pt"))

    def test_import_custom_model_wrong_extension(
        self, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test importing file with wrong extension."""
        wrong_file = temp_download_dir / "model.txt"
        wrong_file.write_bytes(b"data")

        with pytest.raises(ValueError, match="Model must be .pt file"):
            downloader.import_custom_model(wrong_file)

    @patch("models.model_downloader.YOLO")
    def test_import_custom_model_invalid(
        self, mock_yolo: Mock, downloader: ModelDownloader, sample_model_file: Path
    ) -> None:
        """Test importing invalid model file."""
        mock_yolo.side_effect = Exception("Invalid model")

        with pytest.raises(ValueError, match="Invalid YOLOv8 model"):
            downloader.import_custom_model(sample_model_file)

    def test_metadata_persistence(self, temp_download_dir: Path) -> None:
        """Test metadata is saved and loaded correctly."""
        # Create downloader and add metadata
        downloader1 = ModelDownloader(download_dir=temp_download_dir)
        downloader1._metadata["yolov8n.pt"] = {
            "download_date": datetime.now().isoformat(),
            "sha256": "abc123",
            "source": "test",
        }
        downloader1._save_metadata()

        # Create new downloader instance
        downloader2 = ModelDownloader(download_dir=temp_download_dir)

        # Metadata should be loaded
        assert "yolov8n.pt" in downloader2._metadata
        assert downloader2._metadata["yolov8n.pt"]["sha256"] == "abc123"

    def test_metadata_corrupted_file(self, temp_download_dir: Path) -> None:
        """Test handling of corrupted metadata file."""
        # Create corrupted metadata file
        metadata_file = temp_download_dir / "models_metadata.json"
        metadata_file.write_text("corrupted json {")

        # Should initialize with empty metadata
        downloader = ModelDownloader(download_dir=temp_download_dir)

        assert downloader._metadata == {}


class TestModelDownloaderIntegration:
    """Integration tests for ModelDownloader."""

    def test_download_and_delete_workflow(
        self, downloader: ModelDownloader, temp_download_dir: Path
    ) -> None:
        """Test complete download and delete workflow."""
        # Create fake downloaded model
        model_path = temp_download_dir / "yolov8n.pt"
        model_path.write_bytes(b"model data")

        # Check model is listed as downloaded
        models = downloader.list_available_models()
        yolo_n = next(m for m in models if m.name == "yolov8n.pt")
        assert yolo_n.is_downloaded

        # Delete model
        result = downloader.delete_model("yolov8n.pt")
        assert result is True

        # Check model is no longer downloaded
        models = downloader.list_available_models()
        yolo_n = next(m for m in models if m.name == "yolov8n.pt")
        assert not yolo_n.is_downloaded

    @patch("models.model_downloader.YOLO")
    def test_import_and_validate_workflow(
        self, mock_yolo: Mock, downloader: ModelDownloader, sample_model_file: Path
    ) -> None:
        """Test import and validation workflow."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict = Mock()
        mock_yolo.return_value = mock_model

        # Import model
        result = downloader.import_custom_model(sample_model_file, "custom.pt")
        assert result.name == "custom.pt"

        # Validate imported model
        imported_path = downloader.download_dir / "custom.pt"
        is_valid = downloader.validate_model(imported_path)
        assert is_valid
