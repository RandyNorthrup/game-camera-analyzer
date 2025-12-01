"""
Model downloading and management for Game Camera Analyzer.

Handles downloading YOLOv8 models from GitHub releases, local caching,
metadata extraction, and version tracking.
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# Default model storage directory - use consistent location across all modules
# This matches DEFAULT_MODEL_DIR in model_manager.py
DEFAULT_DOWNLOAD_DIR: Path = Path.home() / ".game_camera_analyzer" / "models"


@dataclass
class ModelInfo:
    """
    Information about a YOLOv8 model.

    Attributes:
        name: Model name (e.g., "yolov8n.pt")
        size: Model variant size (n, s, m, l, x)
        description: Human-readable description
        parameters_millions: Estimated parameter count in millions
        macs_billions: Estimated GFLOPS/MACs in billions
        size_mb: Approximate file size in MB
        local_path: Path to local file if downloaded
        download_url: Official download URL
        is_downloaded: Whether model exists locally
        download_date: When model was downloaded
        sha256: SHA-256 checksum if known
    """

    name: str
    size: str
    description: str
    parameters_millions: float
    macs_billions: float
    size_mb: float
    local_path: Optional[Path] = None
    download_url: str = ""
    is_downloaded: bool = False
    download_date: Optional[datetime] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate model info after initialization."""
        if not self.name.endswith(".pt"):
            raise ValueError(f"Model name must end with .pt, got {self.name}")

        valid_sizes = {"n", "s", "m", "l", "x", "custom"}
        if self.size not in valid_sizes:
            raise ValueError(f"Model size must be one of {valid_sizes}, got {self.size}")

        if self.parameters_millions < 0:
            raise ValueError(f"Parameters must be >= 0, got {self.parameters_millions}")

        if self.size_mb < 0:
            raise ValueError(f"Size must be >= 0 MB, got {self.size_mb}")


@dataclass
class DownloadProgress:
    """
    Progress information for model download.

    Attributes:
        model_name: Name of model being downloaded
        total_bytes: Total file size in bytes
        downloaded_bytes: Bytes downloaded so far
        percent_complete: Download progress percentage (0-100)
        speed_mbps: Current download speed in MB/s
        eta_seconds: Estimated time remaining in seconds
        is_complete: Whether download is finished
        error_message: Error message if download failed
    """

    model_name: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    percent_complete: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    is_complete: bool = False
    error_message: Optional[str] = None

    def update(self, downloaded: int, total: int, speed_mbps: float = 0.0) -> None:
        """
        Update download progress.

        Args:
            downloaded: Bytes downloaded
            total: Total bytes
            speed_mbps: Download speed in MB/s
        """
        self.downloaded_bytes = downloaded
        self.total_bytes = total
        self.speed_mbps = speed_mbps

        if total > 0:
            self.percent_complete = (downloaded / total) * 100

            if speed_mbps > 0:
                remaining_mb = (total - downloaded) / (1024 * 1024)
                self.eta_seconds = remaining_mb / speed_mbps


class ModelDownloader:
    """
    Download and manage YOLOv8 models.

    Handles downloading models from Ultralytics, local caching,
    metadata tracking, and version management.
    """

    # Official YOLOv8 model specifications
    OFFICIAL_MODELS: Dict[str, ModelInfo] = {
        "yolov8n.pt": ModelInfo(
            name="yolov8n.pt",
            size="n",
            description="YOLOv8 Nano - Fastest inference, good for testing",
            parameters_millions=3.2,
            macs_billions=8.7,
            size_mb=6.2,
            download_url=(
                "https://github.com/ultralytics/assets/releases/" "download/v8.2.0/yolov8n.pt"
            ),
        ),
        "yolov8s.pt": ModelInfo(
            name="yolov8s.pt",
            size="s",
            description="YOLOv8 Small - Fast with better accuracy",
            parameters_millions=11.2,
            macs_billions=28.6,
            size_mb=21.5,
            download_url=(
                "https://github.com/ultralytics/assets/releases/" "download/v8.2.0/yolov8s.pt"
            ),
        ),
        "yolov8m.pt": ModelInfo(
            name="yolov8m.pt",
            size="m",
            description="YOLOv8 Medium - Balanced speed and accuracy (recommended)",
            parameters_millions=25.9,
            macs_billions=78.9,
            size_mb=49.7,
            download_url=(
                "https://github.com/ultralytics/assets/releases/" "download/v8.2.0/yolov8m.pt"
            ),
        ),
        "yolov8l.pt": ModelInfo(
            name="yolov8l.pt",
            size="l",
            description="YOLOv8 Large - High accuracy, slower inference",
            parameters_millions=43.7,
            macs_billions=165.2,
            size_mb=83.7,
            download_url=(
                "https://github.com/ultralytics/assets/releases/" "download/v8.2.0/yolov8l.pt"
            ),
        ),
        "yolov8x.pt": ModelInfo(
            name="yolov8x.pt",
            size="x",
            description="YOLOv8 Extra Large - Maximum accuracy, slowest inference",
            parameters_millions=68.2,
            macs_billions=257.8,
            size_mb=130.5,
            download_url=(
                "https://github.com/ultralytics/assets/releases/" "download/v8.2.0/yolov8x.pt"
            ),
        ),
    }

    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize model downloader.

        Args:
            download_dir: Directory for storing downloaded models

        Raises:
            OSError: If download directory cannot be created
        """
        self.download_dir = download_dir or DEFAULT_DOWNLOAD_DIR

        try:
            self.download_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model download directory: {self.download_dir}")
        except (OSError, PermissionError) as e:
            raise OSError(f"Cannot create download directory '{self.download_dir}': {e}") from e

        self._metadata_file = self.download_dir / "models_metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata about downloaded models from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    self._metadata = json.load(f)
                logger.debug(f"Loaded metadata for {len(self._metadata)} models")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load metadata: {e}, starting fresh")
                self._metadata: Dict[str, Dict] = {}
        else:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Save metadata about downloaded models to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
            logger.debug("Saved model metadata")
        except OSError as e:
            logger.error(f"Failed to save metadata: {e}")

    def list_available_models(self) -> List[ModelInfo]:
        """
        Get list of all available YOLOv8 models.

        Returns:
            List of ModelInfo objects with download status
        """
        models = []

        for model_name, model_info in self.OFFICIAL_MODELS.items():
            # Create copy with updated local info
            info = ModelInfo(
                name=model_info.name,
                size=model_info.size,
                description=model_info.description,
                parameters_millions=model_info.parameters_millions,
                macs_billions=model_info.macs_billions,
                size_mb=model_info.size_mb,
                download_url=model_info.download_url,
            )

            # Check if downloaded locally
            local_path = self.download_dir / model_name
            if local_path.exists():
                info.local_path = local_path
                info.is_downloaded = True

                # Load metadata if available
                if model_name in self._metadata:
                    meta = self._metadata[model_name]
                    if "download_date" in meta:
                        info.download_date = datetime.fromisoformat(meta["download_date"])
                    if "sha256" in meta:
                        info.sha256 = meta["sha256"]

            models.append(info)

        return models

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of model (e.g., "yolov8n.pt")

        Returns:
            ModelInfo or None if model not found
        """
        models = self.list_available_models()
        for model in models:
            if model.name == model_name:
                return model
        return None

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        force_redownload: bool = False,
    ) -> ModelInfo:
        """
        Download a YOLOv8 model.

        Args:
            model_name: Name of model to download
            progress_callback: Optional callback(DownloadProgress) for progress updates
            force_redownload: If True, re-download even if exists

        Returns:
            ModelInfo with download results

        Raises:
            ValueError: If model name is invalid
            RuntimeError: If download fails
        """
        if model_name not in self.OFFICIAL_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. " f"Available: {list(self.OFFICIAL_MODELS.keys())}"
            )

        model_info = self.OFFICIAL_MODELS[model_name]
        local_path = self.download_dir / model_name

        # Check if already exists
        if local_path.exists() and not force_redownload:
            logger.info(f"Model {model_name} already downloaded")
            return self.get_model_info(model_name) or model_info

        logger.info(f"Downloading {model_name} from Ultralytics...")

        progress = DownloadProgress(model_name=model_name)

        # Initial progress callback - download starting
        if progress_callback:
            progress.percent_complete = 0.0
            progress_callback(progress)

        try:
            # Download directly from GitHub using requests
            download_url = model_info.download_url
            logger.info(f"Downloading from: {download_url}")

            # Download with progress tracking
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            start_time = time.time()

            # Write to file in chunks
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Update progress callback
                        if progress_callback and total_size > 0:
                            elapsed = time.time() - start_time
                            speed = (downloaded_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            progress.update(downloaded_size, total_size, speed)
                            progress_callback(progress)

            logger.info(f"Model downloaded to: {local_path}")

            # Calculate SHA-256 checksum
            sha256 = self._calculate_sha256(local_path)

            # Save metadata
            self._metadata[model_name] = {
                "download_date": datetime.now().isoformat(),
                "sha256": sha256,
                "source": "github",
                "url": download_url,
            }
            self._save_metadata()

            progress.is_complete = True
            progress.percent_complete = 100.0

            if progress_callback:
                progress_callback(progress)

            logger.info(f"Successfully downloaded {model_name}")

            # Return updated model info
            result = self.get_model_info(model_name)
            if result:
                return result
            return model_info

        except requests.RequestException as e:
            error_msg = f"Failed to download {model_name}: {e}"
            logger.error(error_msg, exc_info=True)
            progress.error_message = str(e)

            if progress_callback:
                progress_callback(progress)

            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to download {model_name}: {e}"
            logger.error(error_msg, exc_info=True)
            progress.error_message = str(e)

            if progress_callback:
                progress_callback(progress)

            raise RuntimeError(error_msg) from e

    def _calculate_sha256(self, file_path: Path) -> str:
        """
        Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal SHA-256 checksum
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read in 64KB chunks
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a downloaded model.

        Args:
            model_name: Name of model to delete

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If deletion fails
        """
        local_path = self.download_dir / model_name

        if not local_path.exists():
            logger.warning(f"Model {model_name} not found locally")
            return False

        try:
            local_path.unlink()
            logger.info(f"Deleted model: {model_name}")

            # Remove from metadata
            if model_name in self._metadata:
                del self._metadata[model_name]
                self._save_metadata()

            return True

        except OSError as e:
            error_msg = f"Failed to delete {model_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def validate_model(self, model_path: Path) -> bool:
        """
        Validate that a model file is a valid YOLOv8 model.

        Args:
            model_path: Path to model file

        Returns:
            True if valid, False otherwise
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        if not model_path.suffix == ".pt":
            logger.error(f"Model must be .pt file, got {model_path.suffix}")
            return False

        try:
            # Try to load the model
            logger.debug(f"Validating model: {model_path}")
            model = YOLO(str(model_path))

            # Check that it has expected methods
            if not hasattr(model, "predict"):
                logger.error("Model missing 'predict' method")
                return False

            logger.info(f"Model validation successful: {model_path.name}")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}", exc_info=True)
            return False

    def import_custom_model(self, source_path: Path, model_name: Optional[str] = None) -> ModelInfo:
        """
        Import a custom model file.

        Args:
            source_path: Path to custom model file
            model_name: Optional name for model (defaults to source filename)

        Returns:
            ModelInfo for imported model

        Raises:
            ValueError: If source file invalid
            RuntimeError: If import fails
        """
        if not source_path.exists():
            raise ValueError(f"Source file not found: {source_path}")

        if not source_path.suffix == ".pt":
            raise ValueError(f"Model must be .pt file, got {source_path.suffix}")

        # Validate model before importing
        if not self.validate_model(source_path):
            raise ValueError(f"Invalid YOLOv8 model: {source_path}")

        # Determine destination name
        dest_name = model_name or source_path.name
        if not dest_name.endswith(".pt"):
            dest_name += ".pt"

        dest_path = self.download_dir / dest_name

        try:
            # Copy model to download directory
            logger.info(f"Importing custom model: {source_path} -> {dest_path}")
            shutil.copy2(source_path, dest_path)

            # Calculate checksum
            sha256 = self._calculate_sha256(dest_path)

            # Get file size
            size_mb = dest_path.stat().st_size / (1024 * 1024)

            # Save metadata
            self._metadata[dest_name] = {
                "download_date": datetime.now().isoformat(),
                "sha256": sha256,
                "source": "custom_import",
                "original_path": str(source_path),
            }
            self._save_metadata()

            # Create ModelInfo
            model_info = ModelInfo(
                name=dest_name,
                size="custom",
                description=f"Custom model imported from {source_path.name}",
                parameters_millions=0.0,  # Unknown for custom models
                macs_billions=0.0,
                size_mb=size_mb,
                local_path=dest_path,
                is_downloaded=True,
                download_date=datetime.now(),
                sha256=sha256,
            )

            logger.info(f"Successfully imported custom model: {dest_name}")
            return model_info

        except Exception as e:
            error_msg = f"Failed to import custom model: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about model storage.

        Returns:
            Dictionary with storage statistics
        """
        downloaded_models = [m for m in self.list_available_models() if m.is_downloaded]
        total_size_mb = sum(m.size_mb for m in downloaded_models)

        # Calculate directory size from actual files
        actual_size_bytes = sum(
            f.stat().st_size for f in self.download_dir.glob("*.pt") if f.is_file()
        )
        actual_size_mb = actual_size_bytes / (1024 * 1024)

        return {
            "download_dir": str(self.download_dir),
            "downloaded_count": len(downloaded_models),
            "total_size_mb": round(actual_size_mb, 2),
            "estimated_size_mb": round(total_size_mb, 2),
            "models": [m.name for m in downloaded_models],
        }
