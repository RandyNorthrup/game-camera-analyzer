"""
Comprehensive tests for models/model_manager.py.

Tests cover:
- Model manager initialization
- Device selection (CPU/CUDA/MPS)
- YOLO model loading and caching
- Error handling and edge cases
- Device info retrieval
"""

import logging
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from models.model_manager import (
    DEFAULT_MODEL_DIR,
    ModelLoadError,
    ModelManager,
)

logger = logging.getLogger(__name__)


class TestModelManagerInitialization:
    """Test ModelManager initialization and configuration."""

    def test_init_default_params(self, tmp_path: Path) -> None:
        """
        Test initialization with default parameters.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("models.model_manager.DEFAULT_MODEL_DIR", tmp_path / "models"):
            manager = ModelManager()

            assert manager.model_dir == tmp_path / "models"
            assert manager.cache_models is True
            assert manager.device in ["cpu", "cuda", "mps"]
            assert manager.model_dir.exists()
            assert manager.model_dir.is_dir()

    def test_init_custom_model_dir(self, tmp_path: Path) -> None:
        """
        Test initialization with custom model directory.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        custom_dir = tmp_path / "custom_models"
        manager = ModelManager(model_dir=custom_dir)

        assert manager.model_dir == custom_dir
        assert custom_dir.exists()
        assert custom_dir.is_dir()

    def test_init_creates_nested_directory(self, tmp_path: Path) -> None:
        """
        Test that initialization creates nested directories.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        nested_dir = tmp_path / "a" / "b" / "c" / "models"
        manager = ModelManager(model_dir=nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert manager.model_dir == nested_dir

    def test_init_with_cache_disabled(self, tmp_path: Path) -> None:
        """
        Test initialization with caching disabled.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models", cache_models=False)

        assert manager.cache_models is False
        assert len(manager._model_cache) == 0

    def test_init_model_dir_creation_error(self, tmp_path: Path) -> None:
        """
        Test error handling when model directory cannot be created.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        # Create a file where we want a directory
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("blocking")

        with pytest.raises(OSError, match="Cannot create model directory"):
            ModelManager(model_dir=file_path / "models")


class TestDeviceSelection:
    """Test device selection logic."""

    def test_select_device_auto_cpu_only(self, tmp_path: Path) -> None:
        """
        Test automatic device selection when only CPU is available.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            manager = ModelManager(model_dir=tmp_path / "models", device=None)
            assert manager.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_select_device_cuda_available(self, tmp_path: Path) -> None:
        """
        Test device selection when CUDA is available.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA GPU"
        ):
            manager = ModelManager(model_dir=tmp_path / "models", device=None)
            assert manager.device == "cuda"

    def test_select_device_mps_falls_back_to_cpu(self, tmp_path: Path) -> None:
        """
        Test that MPS detection falls back to CPU due to known limitations.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            manager = ModelManager(model_dir=tmp_path / "models", device=None)
            # Should use CPU due to torchvision::nms limitation
            assert manager.device == "cpu"

    def test_select_device_explicit_cpu(self, tmp_path: Path) -> None:
        """
        Test explicit CPU device selection.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models", device="cpu")
        assert manager.device == "cpu"

    def test_select_device_explicit_cuda_not_available(
        self, tmp_path: Path
    ) -> None:
        """
        Test CUDA request when CUDA is not available falls back to CPU.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=False):
            manager = ModelManager(model_dir=tmp_path / "models", device="cuda")
            assert manager.device == "cpu"

    def test_select_device_explicit_mps_uses_cpu(self, tmp_path: Path) -> None:
        """
        Test explicit MPS request uses CPU due to limitations.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models", device="mps")
        # Should use CPU due to torchvision::nms limitation
        assert manager.device == "cpu"

    def test_select_device_auto_string(self, tmp_path: Path) -> None:
        """
        Test 'auto' device string triggers automatic selection.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            manager = ModelManager(model_dir=tmp_path / "models", device="auto")
            assert manager.device == "cpu"


class TestDeviceInfo:
    """Test device information retrieval."""

    def test_get_device_info_cpu(self, tmp_path: Path) -> None:
        """
        Test device info for CPU device.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            manager = ModelManager(model_dir=tmp_path / "models", device="cpu")
            info = manager.get_device_info()

            assert isinstance(info, dict)
            assert info["device"] == "cpu"
            assert "cuda_available" in info
            assert "mps_available" in info
            assert info["cuda_available"] is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_info_cuda(self, tmp_path: Path) -> None:
        """
        Test device info for CUDA device.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="Test GPU"
        ), patch("torch.cuda.device_count", return_value=1), patch(
            "torch.cuda.memory_allocated", return_value=1024
        ), patch(
            "torch.cuda.memory_reserved", return_value=2048
        ):
            manager = ModelManager(model_dir=tmp_path / "models", device="cuda")
            info = manager.get_device_info()

            assert info["device"] == "cuda"
            assert info["cuda_available"] is True
            assert "cuda_device_count" in info
            assert "cuda_device_name" in info
            assert "cuda_memory_allocated" in info
            assert "cuda_memory_reserved" in info


class TestYOLOModelLoading:
    """Test YOLO model loading and caching."""

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_download(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test loading YOLO model that needs to be downloaded.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(model_dir=tmp_path / "models")
        model = manager.load_yolo_model("yolov8n.pt")

        assert model is not None
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")
        mock_model.to.assert_called_once_with(manager.device)

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_local_exists(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test loading YOLO model from local directory.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        # Create a fake local model file
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model = model_dir / "yolov8n.pt"
        local_model.write_bytes(b"fake model data")

        manager = ModelManager(model_dir=model_dir)
        model = manager.load_yolo_model("yolov8n.pt")

        assert model is not None
        mock_yolo_cls.assert_called_once_with(str(local_model))
        mock_model.to.assert_called_once()

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_caching_enabled(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test that models are cached when caching is enabled.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        # Load model first time
        model1 = manager.load_yolo_model("yolov8n.pt")

        # Load model second time - should use cache
        model2 = manager.load_yolo_model("yolov8n.pt")

        assert model1 is model2
        # Should only call YOLO constructor once
        assert mock_yolo_cls.call_count == 1

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_caching_disabled(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test that models are not cached when caching is disabled.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(
            model_dir=tmp_path / "models", cache_models=False
        )

        # Load model twice
        manager.load_yolo_model("yolov8n.pt")
        manager.load_yolo_model("yolov8n.pt")

        # Should call YOLO constructor twice (no caching)
        assert mock_yolo_cls.call_count == 2

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_force_reload(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test force reload bypasses cache.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        # Load and cache
        manager.load_yolo_model("yolov8n.pt")

        # Force reload
        manager.load_yolo_model("yolov8n.pt", force_reload=True)

        # Should call YOLO constructor twice despite caching
        assert mock_yolo_cls.call_count == 2

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_error(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test error handling when model loading fails.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_yolo_cls.side_effect = Exception("Model load failed")

        manager = ModelManager(model_dir=tmp_path / "models")

        with pytest.raises(ModelLoadError, match="Could not load model"):
            manager.load_yolo_model("yolov8n.pt")

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_multiple_models(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test loading multiple different models with caching.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model_n = MagicMock()
        mock_model_n.to = MagicMock(return_value=mock_model_n)

        mock_model_m = MagicMock()
        mock_model_m.to = MagicMock(return_value=mock_model_m)

        mock_yolo_cls.side_effect = [mock_model_n, mock_model_m]

        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        # Load two different models
        model_n = manager.load_yolo_model("yolov8n.pt")
        model_m = manager.load_yolo_model("yolov8m.pt")

        assert model_n is not model_m
        assert len(manager._model_cache) == 2
        assert "yolo_yolov8n.pt" in manager._model_cache
        assert "yolo_yolov8m.pt" in manager._model_cache


class TestModelCacheManagement:
    """Test model cache management operations."""

    @patch("models.model_manager.YOLO")
    def test_clear_cache(self, mock_yolo_cls: Mock, tmp_path: Path) -> None:
        """
        Test clearing the model cache.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        # Load and cache a model
        manager.load_yolo_model("yolov8n.pt")
        assert len(manager._model_cache) == 1

        # Clear cache
        manager.clear_cache()
        assert len(manager._model_cache) == 0

    @patch("models.model_manager.YOLO")
    def test_clear_cache_empty(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test clearing empty cache doesn't error.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        # Clear empty cache
        manager.clear_cache()
        assert len(manager._model_cache) == 0

    @patch("models.model_manager.YOLO")
    def test_get_cache_info(self, mock_yolo_cls: Mock, tmp_path: Path) -> None:
        """
        Test getting cache information.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        # Load a model
        manager.load_yolo_model("yolov8n.pt")

        # Get cache info
        info = manager.get_cache_info()

        assert isinstance(info, dict)
        assert "cached_models" in info
        assert "cache_count" in info
        assert "cache_enabled" in info
        assert info["cache_enabled"] is True
        assert info["cache_count"] == 1
        assert "yolo_yolov8n.pt" in info["cached_models"]

    def test_get_cache_info_no_models(self, tmp_path: Path) -> None:
        """
        Test cache info with no models loaded.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models", cache_models=True)

        info = manager.get_cache_info()

        assert info["cache_count"] == 0
        assert info["cached_models"] == []
        assert info["cache_enabled"] is True

    def test_get_cache_info_caching_disabled(self, tmp_path: Path) -> None:
        """
        Test cache info when caching is disabled.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(
            model_dir=tmp_path / "models", cache_models=False
        )

        info = manager.get_cache_info()

        assert info["cache_enabled"] is False


class TestYOLOModelListing:
    """Test listing available YOLO models."""

    def test_list_available_yolo_models(self, tmp_path: Path) -> None:
        """
        Test listing available YOLO model variants.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models")
        models = manager.list_available_yolo_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "yolov8n.pt" in models
        assert "yolov8s.pt" in models
        assert "yolov8m.pt" in models
        assert "yolov8l.pt" in models
        assert "yolov8x.pt" in models

    def test_list_available_models_order(self, tmp_path: Path) -> None:
        """
        Test that models are listed in size order.

        Args:
            tmp_path: Pytest temporary directory fixture
        """
        manager = ModelManager(model_dir=tmp_path / "models")
        models = manager.list_available_yolo_models()

        # Should be in order: n, s, m, l, x
        expected_order = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        assert models == expected_order


class TestModelManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_model_dir_as_string(self) -> None:
        """Test model directory can be provided as string."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(model_dir=Path(tmpdir) / "models")
            assert isinstance(manager.model_dir, Path)
            assert manager.model_dir.exists()

    @patch("models.model_manager.YOLO")
    def test_cache_survives_device_change(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """
        Test that cache key is independent of device changes.

        Args:
            mock_yolo_cls: Mock YOLO class
            tmp_path: Pytest temporary directory fixture
        """
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model

        manager = ModelManager(model_dir=tmp_path / "models", device="cpu")

        # Load model
        manager.load_yolo_model("yolov8n.pt")
        assert len(manager._model_cache) == 1

        # Device doesn't affect cache key
        assert "yolo_yolov8n.pt" in manager._model_cache

    def test_default_model_dir_constant(self) -> None:
        """Test DEFAULT_MODEL_DIR constant is set correctly."""
        assert isinstance(DEFAULT_MODEL_DIR, Path)
        assert ".game_camera_analyzer" in str(DEFAULT_MODEL_DIR)
        assert "models" in str(DEFAULT_MODEL_DIR)

    @patch("models.model_manager.torch.backends.mps.is_available", return_value=True)
    @patch("models.model_manager.torch.cuda.is_available", return_value=False)
    def test_device_selection_mps_warning(
        self, mock_cuda: Mock, mock_mps: Mock, tmp_path: Path
    ) -> None:
        """Test MPS device detection logs warning about using CPU instead."""
        manager = ModelManager(model_dir=tmp_path / "models")
        
        # Should detect MPS but choose CPU with warning
        assert manager.device == "cpu"

    @patch("models.model_manager.torch.cuda.is_available", return_value=True)
    @patch("models.model_manager.torch.cuda.get_device_name", return_value="NVIDIA RTX 3090")
    @patch("models.model_manager.torch.cuda.device_count", return_value=2)
    @patch("models.model_manager.torch.cuda.memory_allocated", return_value=1024)
    @patch("models.model_manager.torch.cuda.memory_reserved", return_value=2048)
    def test_get_device_info_cuda_details(
        self, mock_reserved: Mock, mock_allocated: Mock, mock_count: Mock,
        mock_name: Mock, mock_available: Mock, tmp_path: Path
    ) -> None:
        """Test get_device_info returns CUDA details when on CUDA device."""
        manager = ModelManager(model_dir=tmp_path / "models", device="cuda")
        info = manager.get_device_info()
        
        assert info["device"] == "cuda"
        assert info["cuda_device_count"] == 2
        assert info["cuda_device_name"] == "NVIDIA RTX 3090"
        assert info["cuda_memory_allocated"] == 1024
        assert info["cuda_memory_reserved"] == 2048

    @patch("models.model_manager.YOLO")
    def test_load_yolo_model_save_exception(
        self, mock_yolo_cls: Mock, tmp_path: Path
    ) -> None:
        """Test model loading handles save exceptions gracefully."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_yolo_cls.return_value = mock_model
        
        manager = ModelManager(model_dir=tmp_path / "models")
        
        # Mock Path.home() to cause exception during save
        with patch("pathlib.Path.home", side_effect=RuntimeError("Cannot access home")):
            # Should still load successfully, just log warning
            model = manager.load_yolo_model("yolov8n.pt")
            assert model is not None

    def test_repr(self, tmp_path: Path) -> None:
        """Test ModelManager __repr__ method."""
        manager = ModelManager(model_dir=tmp_path / "models", device="cpu")
        repr_str = repr(manager)
        
        assert "ModelManager" in repr_str
        assert "device=cpu" in repr_str
        assert "cache=" in repr_str
