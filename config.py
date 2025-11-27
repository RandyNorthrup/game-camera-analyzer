"""
Configuration management for the Game Camera Analyzer application.

This module provides centralized configuration management with support for:
- Default configuration from JSON file
- Environment variable overrides
- User configuration file
- Runtime configuration updates
- Configuration validation
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default paths
DEFAULT_CONFIG_FILE: Path = Path(__file__).parent / "resources" / "default_config.json"
USER_CONFIG_DIR: Path = Path.home() / ".game_camera_analyzer"
USER_CONFIG_FILE: Path = USER_CONFIG_DIR / "config.json"


@dataclass
class DetectionConfig:
    """Configuration for animal detection."""

    model: str = "yolov8m.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 20
    input_size: int = 640
    device: str = "auto"


@dataclass
class ClassificationConfig:
    """Configuration for species classification."""

    model: str = "efficientnet_b0"
    confidence_threshold: float = 0.7
    top_k: int = 5
    use_feature_classifier: bool = False
    max_alternatives: int = 3
    species_db: str = "data/species_db.json"


@dataclass
class ProcessingConfig:
    """Configuration for image processing."""

    batch_size: int = 8
    num_workers: int = 4
    use_gpu: bool = True
    cache_models: bool = True
    enhance_low_light: bool = True  # Automatically enhance dark images
    denoise_images: bool = False  # Apply denoising to reduce false positives
    low_light_threshold: int = 80  # Mean brightness threshold for enhancement
    denoise_strength: int = 3  # Denoising strength (1-10)


@dataclass
class CroppingConfig:
    """Configuration for image cropping."""

    strategy: str = "context"
    padding_ratio: float = 0.15
    min_size: int = 224
    max_size: int = 1024
    square_crop: bool = False
    # GUI fields
    padding: float = 0.1
    square_crops: bool = False
    organize_by_species: bool = True
    min_width: int = 0
    min_height: int = 0
    max_width: int = 0
    max_height: int = 0
    jpeg_quality: int = 95


@dataclass
class OutputConfig:
    """Configuration for output organization."""

    base_dir: str = "output"
    organize_by_species: bool = True
    organize_by_date: bool = True
    organize_by_camera: bool = True
    create_thumbnails: bool = True
    thumbnail_size: int = 256
    save_metadata_json: bool = False
    export_csv: bool = True
    # GUI fields
    csv_delimiter: str = ","
    include_confidence: bool = True
    include_alternatives: bool = True
    include_timestamps: bool = True
    save_annotated: bool = False


@dataclass
class CSVExportConfig:
    """Configuration for CSV export."""

    master_filename: str = "detections_master.csv"
    summary_filename: str = "species_summary.csv"
    include_gps: bool = True
    include_exif: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class ModelConfig:
    """Configuration for model management."""

    download_dir: str = str(
        Path.home() / "Documents" / "GameCameraAnalyzer" / "models"
    )


@dataclass
class GUIConfig:
    """Configuration for GUI."""

    theme: str = "default"
    window_width: int = 1280
    window_height: int = 800
    remember_window_size: bool = True
    show_confidence_scores: bool = True
    auto_process: bool = False
    last_directory: str = ""


@dataclass
class AppConfig:
    """Main application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cropping: CroppingConfig = field(default_factory=CroppingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    csv_export: CSVExportConfig = field(default_factory=CSVExportConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)


class ConfigManager:
    """
    Manages application configuration from multiple sources.

    Configuration is loaded in the following priority (highest to lowest):
    1. Runtime updates via set_value()
    2. Environment variables (prefixed with GCA_)
    3. User configuration file (~/.game_camera_analyzer/config.json)
    4. Default configuration file (resources/default_config.json)
    """

    def __init__(
        self,
        default_config_path: Optional[Path] = None,
        user_config_path: Optional[Path] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            default_config_path: Path to default configuration file
            user_config_path: Path to user configuration file

        Raises:
            FileNotFoundError: If default configuration file not found
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        self._default_config_path = default_config_path or DEFAULT_CONFIG_FILE
        self._user_config_path = user_config_path or USER_CONFIG_FILE
        self._runtime_config: Dict[str, Any] = {}
        self._config: AppConfig

        # Load configuration
        self._load_configuration()

    def _load_configuration(self) -> None:
        """
        Load configuration from all sources.

        Raises:
            FileNotFoundError: If default configuration file not found
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        # Load default configuration
        if not self._default_config_path.exists():
            raise FileNotFoundError(
                f"Default configuration file not found: {self._default_config_path}"
            )

        try:
            with open(self._default_config_path, "r", encoding="utf-8") as f:
                default_config = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in default config: {e}")
            raise json.JSONDecodeError(f"Invalid default configuration: {e}", e.doc, e.pos) from e

        # Load user configuration if it exists
        user_config: Dict[str, Any] = {}
        if self._user_config_path.exists():
            try:
                with open(self._user_config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                logger.info(f"Loaded user configuration from {self._user_config_path}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid user configuration, using defaults: {e}")
            except OSError as e:
                logger.warning(f"Could not read user configuration: {e}")

        # Merge configurations (user overrides default)
        merged_config = self._merge_dicts(default_config, user_config)

        # Apply environment variable overrides
        merged_config = self._apply_env_overrides(merged_config)

        # Apply runtime overrides
        merged_config = self._merge_dicts(merged_config, self._runtime_config)

        # Convert to dataclass structure
        self._config = self._dict_to_config(merged_config)

    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with override values

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides.

        Environment variables should be prefixed with GCA_ (Game Camera Analyzer).
        For nested keys, use double underscore: GCA_DETECTION__CONFIDENCE_THRESHOLD

        Args:
            config: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        prefix = "GCA_"
        for env_var, env_value in os.environ.items():
            if not env_var.startswith(prefix):
                continue

            # Remove prefix and split on double underscore
            key_path = env_var[len(prefix) :].lower().split("__")

            # Navigate to the correct position in config dict
            current = config
            for key in key_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value (attempt type conversion)
            final_key = key_path[-1]
            try:
                # Try to parse as JSON for complex types
                current[final_key] = json.loads(env_value)
            except json.JSONDecodeError:
                # Fall back to string value
                current[final_key] = env_value

            logger.debug(f"Applied environment override: {env_var}={env_value}")

        return config

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """
        Convert configuration dictionary to AppConfig dataclass.

        Args:
            config_dict: Configuration dictionary

        Returns:
            AppConfig instance
        """
        return AppConfig(
            detection=DetectionConfig(**config_dict.get("detection", {})),
            classification=ClassificationConfig(**config_dict.get("classification", {})),
            processing=ProcessingConfig(**config_dict.get("processing", {})),
            cropping=CroppingConfig(**config_dict.get("cropping", {})),
            output=OutputConfig(**config_dict.get("output", {})),
            csv_export=CSVExportConfig(**config_dict.get("csv_export", {})),
            logging_config=LoggingConfig(**config_dict.get("logging", {})),
            gui=GUIConfig(**config_dict.get("gui", {})),
        )

    @property
    def config(self) -> AppConfig:
        """Get the current application configuration."""
        return self._config

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dotted key path.

        Args:
            key_path: Dotted path to configuration value (e.g., "detection.confidence_threshold")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config_manager.get_value("detection.confidence_threshold")
            0.25
        """
        keys = key_path.split(".")
        current: Any = self._config

        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return default

        return current

    def set_value(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value at runtime.

        Note: This only affects runtime configuration and does not persist to file.

        Args:
            key_path: Dotted path to configuration value
            value: New value to set

        Raises:
            ValueError: If key_path is invalid
        """
        keys = key_path.split(".")
        current = self._runtime_config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        logger.debug(f"Set runtime config: {key_path}={value}")

        # Reload configuration to apply changes
        self._load_configuration()

    def save_user_config(self) -> None:
        """
        Save current runtime configuration to user config file.

        Raises:
            OSError: If unable to write configuration file
            PermissionError: If insufficient permissions
        """
        # Create user config directory if it doesn't exist
        try:
            self._user_config_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot create config directory: {e}")
            raise

        # Save runtime configuration
        try:
            with open(self._user_config_path, "w", encoding="utf-8") as f:
                json.dump(self._runtime_config, f, indent=2)
            logger.info(f"Saved user configuration to {self._user_config_path}")
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot write user configuration: {e}")
            raise

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults by clearing runtime overrides."""
        self._runtime_config = {}
        self._load_configuration()
        logger.info("Configuration reset to defaults")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.

    Returns:
        ConfigManager instance

    Example:
        >>> config_mgr = get_config_manager()
        >>> threshold = config_mgr.config.detection.confidence_threshold
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """
    Get the current application configuration.

    Returns:
        AppConfig instance

    Example:
        >>> config = get_config()
        >>> print(config.detection.confidence_threshold)
        0.25
    """
    return get_config_manager().config


if __name__ == "__main__":
    # Test configuration management
    from utils.logger import setup_logging

    setup_logging()

    config_mgr = get_config_manager()
    print("Configuration loaded successfully!")
    print(f"\nDetection model: {config_mgr.config.detection.model}")
    print(f"Confidence threshold: {config_mgr.config.detection.confidence_threshold}")
    print(f"Output directory: {config_mgr.config.output.base_dir}")

    # Test get_value
    value = config_mgr.get_value("detection.confidence_threshold")
    print(f"\nGot value via get_value: {value}")

    # Test set_value
    config_mgr.set_value("detection.confidence_threshold", 0.5)
    print(f"Updated threshold: {config_mgr.config.detection.confidence_threshold}")
