"""
Comprehensive tests for config.py module.

Tests configuration management, loading, merging, and persistence.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from config import (
    AppConfig,
    CSVExportConfig,
    ClassificationConfig,
    ConfigManager,
    CroppingConfig,
    DEFAULT_CONFIG_FILE,
    DetectionConfig,
    GUIConfig,
    LoggingConfig,
    OutputConfig,
    ProcessingConfig,
    USER_CONFIG_DIR,
    USER_CONFIG_FILE,
    get_config,
    get_config_manager,
)


class TestDataclasses:
    """Test suite for configuration dataclasses."""

    def test_detection_config_defaults(self) -> None:
        """Test DetectionConfig default values."""
        config = DetectionConfig()
        
        assert config.model == "yolov8m.pt"
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.max_detections == 20
        assert config.input_size == 640
        assert config.device == "auto"

    def test_classification_config_defaults(self) -> None:
        """Test ClassificationConfig default values."""
        config = ClassificationConfig()
        
        assert config.model == "efficientnet_b0"
        assert config.confidence_threshold == 0.7
        assert config.top_k == 5
        assert config.use_feature_classifier is False
        assert config.max_alternatives == 3
        assert config.species_db == "data/species_db.json"

    def test_processing_config_defaults(self) -> None:
        """Test ProcessingConfig default values."""
        config = ProcessingConfig()
        
        assert config.batch_size == 8
        assert config.num_workers == 4
        assert config.use_gpu is True
        assert config.cache_models is True

    def test_cropping_config_defaults(self) -> None:
        """Test CroppingConfig default values."""
        config = CroppingConfig()
        
        assert config.strategy == "context"
        assert config.padding_ratio == 0.15
        assert config.min_size == 224
        assert config.max_size == 1024
        assert config.square_crop is False

    def test_output_config_defaults(self) -> None:
        """Test OutputConfig default values."""
        config = OutputConfig()
        
        assert config.base_dir == "output"
        assert config.organize_by_species is True
        assert config.create_thumbnails is True
        assert config.thumbnail_size == 256
        assert config.export_csv is True

    def test_csv_export_config_defaults(self) -> None:
        """Test CSVExportConfig default values."""
        config = CSVExportConfig()
        
        assert config.master_filename == "detections_master.csv"
        assert config.summary_filename == "species_summary.csv"
        assert config.include_gps is True
        assert config.include_exif is True

    def test_logging_config_defaults(self) -> None:
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.console_level == "INFO"
        assert config.file_level == "DEBUG"
        assert config.max_log_size_mb == 10
        assert config.backup_count == 5

    def test_gui_config_defaults(self) -> None:
        """Test GUIConfig default values."""
        config = GUIConfig()
        
        assert config.theme == "default"
        assert config.window_width == 1280
        assert config.window_height == 800
        assert config.remember_window_size is True
        assert config.show_confidence_scores is True
        assert config.auto_process is False

    def test_app_config_initialization(self) -> None:
        """Test AppConfig creates all sub-configs."""
        config = AppConfig()
        
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.classification, ClassificationConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.cropping, CroppingConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.csv_export, CSVExportConfig)
        assert isinstance(config.logging_config, LoggingConfig)
        assert isinstance(config.gui, GUIConfig)


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_init_with_default_config(self, tmp_path: Path) -> None:
        """Test ConfigManager initialization with default config."""
        # Create a minimal config file
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.3},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)

        # Dataclass defaults fill in missing fields
        assert isinstance(manager.config, AppConfig)
        assert isinstance(manager.config.detection, DetectionConfig)

    def test_init_missing_default_config(self, tmp_path: Path) -> None:
        """Test initialization fails when default config missing."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Default configuration file not found"):
            ConfigManager(default_config_path=nonexistent)

    def test_init_invalid_json(self, tmp_path: Path) -> None:
        """Test initialization fails with invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError, match="Invalid default configuration"):
            ConfigManager(default_config_path=config_file)

    def test_user_config_overrides_default(self, tmp_path: Path) -> None:
        """Test user config overrides default config."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        default_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        user_data = {
            "detection": {"confidence_threshold": 0.5},
        }

        default_config.write_text(json.dumps(default_data))
        user_config.write_text(json.dumps(user_data))

        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )

        assert manager.config.detection.confidence_threshold == 0.5

    def test_user_config_invalid_json_ignored(self, tmp_path: Path) -> None:
        """Test invalid user config is ignored gracefully."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        default_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }

        default_config.write_text(json.dumps(default_data))
        user_config.write_text("{ invalid json }")

        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )

        # Should use default value
        assert manager.config.detection.confidence_threshold == 0.25

    def test_user_config_read_error_ignored(self, tmp_path: Path) -> None:
        """Test user config read error is handled gracefully."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        default_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }

        default_config.write_text(json.dumps(default_data))
        # Create user config but it will be unreadable
        user_config.write_text(json.dumps({}))
        user_config.chmod(0o000)

        try:
            manager = ConfigManager(
                default_config_path=default_config,
                user_config_path=user_config,
            )

            # Should use default value even with read error
            assert isinstance(manager.config, AppConfig)
        finally:
            user_config.chmod(0o644)

    def test_merge_dicts_nested(self, tmp_path: Path) -> None:
        """Test nested dictionary merging."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.3},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)

        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 4}

        result = manager._merge_dicts(base, override)

        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2
        assert result["d"] == 3
        assert result["e"] == 4

    def test_env_overrides_applied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable overrides are applied."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("GCA_DETECTION__CONFIDENCE_THRESHOLD", "0.8")

        manager = ConfigManager(default_config_path=config_file)

        assert manager.config.detection.confidence_threshold == 0.8

    def test_env_override_json_parsing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable with JSON value."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("GCA_DETECTION__MAX_DETECTIONS", "50")

        manager = ConfigManager(default_config_path=config_file)

        assert manager.config.detection.max_detections == 50

    def test_env_override_string_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable falls back to string."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"model": "yolov8m.pt"},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("GCA_DETECTION__MODEL", "custom_model.pt")

        manager = ConfigManager(default_config_path=config_file)

        assert manager.config.detection.model == "custom_model.pt"

    def test_get_value_simple(self, tmp_path: Path) -> None:
        """Test get_value with simple path."""
        config_file = tmp_path / "config.json"
        user_config = tmp_path / "user.json"  # Isolated user config
        
        config_data = {
            "detection": {
                "confidence_threshold": 0.3,
                "model": "yolov8m.pt",
                "iou_threshold": 0.45,
                "max_detections": 20,
                "input_size": 640,
                "device": "auto",
            },
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(
            default_config_path=config_file,
            user_config_path=user_config,
        )

        value = manager.get_value("detection.confidence_threshold")

        assert value == 0.3

    def test_get_value_default(self, tmp_path: Path) -> None:
        """Test get_value returns default for missing key."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)

        value = manager.get_value("nonexistent.key", default="fallback")

        assert value == "fallback"

    def test_set_value_updates_runtime(self, tmp_path: Path) -> None:
        """Test set_value updates runtime configuration."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)
        manager.set_value("detection.confidence_threshold", 0.7)

        assert manager.config.detection.confidence_threshold == 0.7

    def test_set_value_nested(self, tmp_path: Path) -> None:
        """Test set_value with nested keys."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)
        manager.set_value("detection.iou_threshold", 0.6)

        assert manager.config.detection.iou_threshold == 0.6

    def test_save_user_config(self, tmp_path: Path) -> None:
        """Test saving user configuration."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        default_config.write_text(json.dumps(config_data))

        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )
        manager.set_value("detection.confidence_threshold", 0.8)
        manager.save_user_config()

        assert user_config.exists()
        saved_data = json.loads(user_config.read_text())
        assert saved_data["detection"]["confidence_threshold"] == 0.8

    def test_save_user_config_creates_directory(self, tmp_path: Path) -> None:
        """Test save creates parent directory."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "subdir" / "user.json"

        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        default_config.write_text(json.dumps(config_data))

        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )
        manager.save_user_config()

        assert user_config.exists()
        assert user_config.parent.exists()

    def test_save_user_config_mkdir_error(self, tmp_path: Path) -> None:
        """Test save handles directory creation error."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                manager.save_user_config()

    def test_save_user_config_write_error(self, tmp_path: Path) -> None:
        """Test save handles write error."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)

        with patch("builtins.open", side_effect=PermissionError("Write denied")):
            with pytest.raises(PermissionError):
                manager.save_user_config()

    def test_reset_to_defaults(self, tmp_path: Path) -> None:
        """Test resetting configuration to defaults."""
        # Create isolated user config to prevent pollution
        test_user_config = tmp_path / "test_user_config.json"
        
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {"confidence_threshold": 0.25},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(
            default_config_path=config_file, user_config_path=test_user_config
        )
        manager.set_value("detection.confidence_threshold", 0.9)
        assert manager.config.detection.confidence_threshold == 0.9

        manager.reset_to_defaults()

        assert manager.config.detection.confidence_threshold == 0.25

    def test_config_property(self, tmp_path: Path) -> None:
        """Test config property returns AppConfig."""
        config_file = tmp_path / "config.json"
        config_data = {
            "detection": {},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(default_config_path=config_file)

        assert isinstance(manager.config, AppConfig)


class TestGlobalFunctions:
    """Test suite for global configuration functions."""

    def test_get_config_manager_singleton(self) -> None:
        """Test get_config_manager returns same instance."""
        with patch("config.ConfigManager") as mock_manager_class:
            mock_instance = MagicMock()
            mock_manager_class.return_value = mock_instance

            # Reset global
            import config
            config._config_manager = None

            manager1 = get_config_manager()
            manager2 = get_config_manager()

            assert manager1 is manager2
            mock_manager_class.assert_called_once()

    def test_get_config(self) -> None:
        """Test get_config returns AppConfig."""
        with patch("config.get_config_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_config = AppConfig()
            mock_manager.config = mock_config
            mock_get_manager.return_value = mock_manager

            config = get_config()

            assert config is mock_config


class TestConstants:
    """Test suite for module constants."""

    def test_default_config_file_exists(self) -> None:
        """Test default config file path is defined."""
        assert DEFAULT_CONFIG_FILE is not None
        assert isinstance(DEFAULT_CONFIG_FILE, Path)

    def test_user_config_paths_defined(self) -> None:
        """Test user config paths are defined."""
        assert USER_CONFIG_DIR is not None
        assert USER_CONFIG_FILE is not None
        assert isinstance(USER_CONFIG_DIR, Path)
        assert isinstance(USER_CONFIG_FILE, Path)


class TestIntegration:
    """Integration tests for config module."""

    def test_full_configuration_workflow(self, tmp_path: Path) -> None:
        """Test complete configuration workflow."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        # Create default config
        default_data = {
            "detection": {"confidence_threshold": 0.25, "model": "yolov8m.pt"},
            "classification": {"confidence_threshold": 0.7},
            "processing": {"batch_size": 8},
            "cropping": {"padding_ratio": 0.15},
            "output": {"base_dir": "output"},
            "csv_export": {"include_gps": True},
            "logging": {"level": "INFO"},
            "gui": {"theme": "default"},
        }
        default_config.write_text(json.dumps(default_data))

        # Initialize manager
        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )

        # Check defaults loaded
        assert manager.config.detection.confidence_threshold == 0.25

        # Update runtime config
        manager.set_value("detection.confidence_threshold", 0.5)
        assert manager.config.detection.confidence_threshold == 0.5

        # Save to user config
        manager.save_user_config()
        assert user_config.exists()

        # Reset to defaults clears runtime, but user config persists on disk
        manager.reset_to_defaults()
        # After reset, user config is loaded again from disk
        # So value will be 0.5 (from saved user config), not 0.25
        assert manager.config.detection.confidence_threshold == 0.5

    def test_configuration_priority(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration priority: runtime > env > user > default."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        default_data = {
            "detection": {"confidence_threshold": 0.2},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }
        user_data = {
            "detection": {"confidence_threshold": 0.3},
        }

        default_config.write_text(json.dumps(default_data))
        user_config.write_text(json.dumps(user_data))

        # User config overrides default
        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )
        assert manager.config.detection.confidence_threshold == 0.3

        # Env var overrides user config
        monkeypatch.setenv("GCA_DETECTION__CONFIDENCE_THRESHOLD", "0.4")
        manager._load_configuration()
        assert manager.config.detection.confidence_threshold == 0.4

        # Runtime overrides env var
        manager.set_value("detection.confidence_threshold", 0.5)
        assert manager.config.detection.confidence_threshold == 0.5

    def test_env_override_string_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable override with string value (non-JSON)."""
        default_config = tmp_path / "default.json"
        user_config = tmp_path / "user.json"

        default_data = {
            "detection": {"model": "yolov8m.pt"},
            "classification": {},
            "processing": {},
            "cropping": {},
            "output": {"base_dir": "output"},
            "csv_export": {},
            "logging": {},
            "gui": {},
        }

        default_config.write_text(json.dumps(default_data))
        user_config.write_text("{}") 

        # String value that's not valid JSON
        monkeypatch.setenv("GCA_OUTPUT__BASE_DIR", "/custom/output/path")
        
        manager = ConfigManager(
            default_config_path=default_config,
            user_config_path=user_config,
        )
        
        # Should use string value directly (fallback from JSON parse failure)
        assert manager.config.output.base_dir == "/custom/output/path"
