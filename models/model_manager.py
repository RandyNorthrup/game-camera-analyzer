"""
Model management for the Game Camera Analyzer application.

This module handles:
- Model loading and caching
- Device selection (CPU/GPU/MPS)
- Model downloading and validation
- Version tracking
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


# Default model directory
DEFAULT_MODEL_DIR: Path = Path.home() / ".game_camera_analyzer" / "models"


class ModelLoadError(Exception):
    """Exception raised when model cannot be loaded."""

    pass


class ModelManager:
    """
    Manages loading and caching of detection and classification models.

    This class handles model lifecycle including downloading, loading,
    device placement, and caching for efficient reuse.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        device: Optional[str] = None,
        cache_models: bool = True,
    ):
        """
        Initialize model manager.

        Args:
            model_dir: Directory for storing models. Defaults to ~/.game_camera_analyzer/models
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            cache_models: If True, cache loaded models in memory

        Raises:
            OSError: If model directory cannot be created
        """
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.cache_models = cache_models
        self._model_cache: Dict[str, Any] = {}

        # Create model directory if it doesn't exist
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model directory: {self.model_dir}")
        except (OSError, PermissionError) as e:
            raise OSError(f"Cannot create model directory '{self.model_dir}': {e}") from e

        # Determine device
        self.device = self._select_device(device)
        logger.info(f"Using device: {self.device}")

    def _select_device(self, device: Optional[str]) -> str:
        """
        Select the best available device for inference.

        Args:
            device: Requested device or None for auto-selection

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if device is not None and device != "auto":
            # Validate requested device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            elif device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
            return device

        # Auto-select best available device
        if torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {cuda_device}")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
        else:
            logger.info("Using CPU (no GPU acceleration available)")
            return "cpu"

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device.

        Returns:
            Dictionary with device information
        """
        info: Dict[str, Any] = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }

        if self.device == "cuda":
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
            info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)

        return info

    def load_yolo_model(
        self, model_name: str = "yolov8m.pt", force_reload: bool = False
    ) -> Any:  # Returns YOLO model but type is not exposed
        """
        Load a YOLOv8 detection model.

        Args:
            model_name: Name or path of YOLOv8 model (e.g., 'yolov8n.pt', 'yolov8m.pt')
            force_reload: If True, reload even if cached

        Returns:
            Loaded YOLO model

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        cache_key = f"yolo_{model_name}"

        # Check cache
        if not force_reload and self.cache_models and cache_key in self._model_cache:
            logger.debug(f"Using cached YOLO model: {model_name}")
            return self._model_cache[cache_key]

        logger.info(f"Loading YOLO model: {model_name}")

        try:
            # Check if model exists locally
            model_path = self.model_dir / model_name
            if model_path.exists():
                logger.info(f"Loading local model: {model_path}")
                model = YOLO(str(model_path))
            else:
                # Download from Ultralytics
                logger.info(f"Downloading model: {model_name}")
                model = YOLO(model_name)

                # Save to local directory for future use
                try:
                    # Copy downloaded model to our model directory
                    import shutil

                    # Find where Ultralytics cached the model
                    ultralytics_cache = Path.home() / ".cache" / "ultralytics"
                    if ultralytics_cache.exists():
                        for cached_file in ultralytics_cache.rglob(model_name):
                            shutil.copy2(cached_file, model_path)
                            logger.info(f"Saved model to: {model_path}")
                            break
                except Exception as e:
                    logger.warning(f"Could not save model locally: {e}")

            # Move model to device
            model.to(self.device)
            logger.info(f"Model loaded successfully: {model_name} on {self.device}")

            # Cache model
            if self.cache_models:
                self._model_cache[cache_key] = model
                logger.debug(f"Cached model: {cache_key}")

            return model

        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_name}': {e}")
            raise ModelLoadError(f"Could not load model '{model_name}': {e}") from e

    def list_available_yolo_models(self) -> List[str]:
        """
        List available YOLOv8 model variants.

        Returns:
            List of model names
        """
        return [
            "yolov8n.pt",  # Nano - fastest, least accurate
            "yolov8s.pt",  # Small
            "yolov8m.pt",  # Medium - recommended balance
            "yolov8l.pt",  # Large
            "yolov8x.pt",  # Extra large - most accurate, slowest
        ]

    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        count = len(self._model_cache)
        self._model_cache.clear()
        logger.info(f"Cleared model cache ({count} models)")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_count": len(self._model_cache),
            "cache_enabled": self.cache_models,
        }

    def __repr__(self) -> str:
        """String representation of ModelManager."""
        return f"ModelManager(device={self.device}, cache={len(self._model_cache)} models)"


if __name__ == "__main__":
    """Test model manager."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing Model Manager...")

    # Create model manager
    manager = ModelManager()
    print(f"\n{manager}")

    # Get device info
    device_info = manager.get_device_info()
    print("\nDevice Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    # List available models
    print("\nAvailable YOLO models:")
    for model in manager.list_available_yolo_models():
        print(f"  - {model}")

    # Test loading a model (nano for speed)
    print("\nLoading YOLOv8 nano model...")
    try:
        yolo_model: Any = manager.load_yolo_model("yolov8n.pt")
        print(f"✅ Model loaded successfully: {type(yolo_model).__name__}")

        # Get model info
        print("\nModel details:")
        print(f"  Device: {next(yolo_model.model.parameters()).device}")
        print(f"  Number of parameters: {sum(p.numel() for p in yolo_model.model.parameters()):,}")

        # Check cache
        cache_info = manager.get_cache_info()
        print("\nCache info:")
        for key, value in cache_info.items():
            print(f"  {key}: {value}")

    except ModelLoadError as e:
        print(f"❌ Failed to load model: {e}")

    print("\nModel manager tests completed!")
