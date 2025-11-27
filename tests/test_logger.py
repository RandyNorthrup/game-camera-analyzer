"""
Comprehensive tests for utils/logger.py module.

Tests logging initialization, handlers, formatters, levels, and error handling.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from utils.logger import (
    DEFAULT_LOG_DIR,
    DETAILED_FORMAT,
    SIMPLE_FORMAT,
    get_logger,
    log_exception,
    set_log_level,
    setup_logging,
)


class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_setup_logging_default_params(self, tmp_path: Path) -> None:
        """Test logging setup with default parameters."""
        log_dir = tmp_path / "logs"

        setup_logging(log_dir=log_dir)

        # Verify log directory created
        assert log_dir.exists()
        assert log_dir.is_dir()

        # Verify log files created
        assert (log_dir / "game_camera_analyzer.log").exists()
        assert (log_dir / "errors.log").exists()

        # Verify root logger configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) >= 2  # At least console and file handlers

    def test_setup_logging_custom_levels(self, tmp_path: Path) -> None:
        """Test logging setup with custom log levels."""
        log_dir = tmp_path / "logs"

        setup_logging(
            log_dir=log_dir,
            log_level=logging.DEBUG,
            console_level=logging.WARNING,
            file_level=logging.INFO,
        )

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Check handler levels
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        assert len(console_handlers) >= 1
        assert len(file_handlers) >= 1

        # Console handler should have WARNING level
        console_handler = console_handlers[0]
        assert console_handler.level == logging.WARNING

    def test_setup_logging_custom_rotation(self, tmp_path: Path) -> None:
        """Test logging setup with custom rotation parameters."""
        log_dir = tmp_path / "logs"
        max_bytes = 5 * 1024 * 1024  # 5MB
        backup_count = 10

        setup_logging(
            log_dir=log_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        assert len(rotating_handlers) >= 1

        # Check rotation settings
        handler = rotating_handlers[0]
        assert handler.maxBytes == max_bytes
        assert handler.backupCount == backup_count

    def test_setup_logging_creates_directory(self, tmp_path: Path) -> None:
        """Test that setup_logging creates missing directories."""
        log_dir = tmp_path / "deeply" / "nested" / "logs"
        assert not log_dir.exists()

        setup_logging(log_dir=log_dir)

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_setup_logging_permission_error(self, tmp_path: Path) -> None:
        """Test setup_logging handles permission errors on directory creation."""
        log_dir = tmp_path / "logs"

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Cannot create log directory"):
                setup_logging(log_dir=log_dir)

    def test_setup_logging_os_error(self, tmp_path: Path) -> None:
        """Test setup_logging handles OS errors on directory creation."""
        log_dir = tmp_path / "logs"

        with patch("pathlib.Path.mkdir", side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="Cannot create log directory"):
                setup_logging(log_dir=log_dir)

    def test_setup_logging_removes_existing_handlers(self, tmp_path: Path) -> None:
        """Test that setup_logging removes existing handlers."""
        log_dir = tmp_path / "logs"

        # Set up logging once
        setup_logging(log_dir=log_dir)
        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)

        # Set up again - should remove old handlers
        setup_logging(log_dir=log_dir)
        final_handler_count = len(root_logger.handlers)

        # Should have same number of handlers, not double
        assert final_handler_count == initial_handler_count

    def test_setup_logging_formatters(self, tmp_path: Path) -> None:
        """Test that handlers have correct formatters."""
        log_dir = tmp_path / "logs"

        setup_logging(log_dir=log_dir)

        root_logger = logging.getLogger()

        # Console handler should have simple format
        console_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(console_handlers) >= 1
        console_formatter = console_handlers[0].formatter
        assert console_formatter is not None
        assert console_formatter._fmt == SIMPLE_FORMAT

        # File handlers should have detailed format
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1
        file_formatter = file_handlers[0].formatter
        assert file_formatter is not None
        assert file_formatter._fmt == DETAILED_FORMAT

    def test_setup_logging_file_handler_error(self, tmp_path: Path) -> None:
        """Test setup_logging continues when file handler creation fails."""
        log_dir = tmp_path / "logs"

        with patch(
            "logging.handlers.RotatingFileHandler",
            side_effect=PermissionError("Cannot write to file"),
        ):
            # Should not raise, just log warning
            setup_logging(log_dir=log_dir)

            # Should still have console handler
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) >= 1

    def test_setup_logging_error_handler(self, tmp_path: Path) -> None:
        """Test that error handler is created with ERROR level."""
        log_dir = tmp_path / "logs"

        setup_logging(log_dir=log_dir)

        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        # Should have at least 2 rotating handlers (main and error)
        assert len(rotating_handlers) >= 2

        # Check for error handler
        error_handlers = [h for h in rotating_handlers if h.level == logging.ERROR]
        assert len(error_handlers) >= 1

    def test_setup_logging_default_directory(self, tmp_path: Path) -> None:
        """Test setup_logging uses DEFAULT_LOG_DIR when not specified."""
        # Use tmp_path to avoid permissions issues
        with patch("utils.logger.DEFAULT_LOG_DIR", tmp_path / "default_logs"):
            setup_logging(log_dir=None)

            # Should have created default directory
            default_dir = tmp_path / "default_logs"
            assert default_dir.exists()


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_different_names(self) -> None:
        """Test get_logger returns different loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"

    def test_get_logger_same_name_returns_same_instance(self) -> None:
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        assert logger1 is logger2


class TestSetLogLevel:
    """Test suite for set_log_level function."""

    def test_set_log_level_debug(self, tmp_path: Path) -> None:
        """Test setting log level to DEBUG."""
        setup_logging(log_dir=tmp_path / "logs")

        set_log_level(logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_set_log_level_info(self, tmp_path: Path) -> None:
        """Test setting log level to INFO."""
        setup_logging(log_dir=tmp_path / "logs")

        set_log_level(logging.INFO)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_set_log_level_warning(self, tmp_path: Path) -> None:
        """Test setting log level to WARNING."""
        setup_logging(log_dir=tmp_path / "logs")

        set_log_level(logging.WARNING)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_set_log_level_error(self, tmp_path: Path) -> None:
        """Test setting log level to ERROR."""
        setup_logging(log_dir=tmp_path / "logs")

        set_log_level(logging.ERROR)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    def test_set_log_level_critical(self, tmp_path: Path) -> None:
        """Test setting log level to CRITICAL."""
        setup_logging(log_dir=tmp_path / "logs")

        set_log_level(logging.CRITICAL)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.CRITICAL

    def test_set_log_level_invalid_int(self, tmp_path: Path) -> None:
        """Test set_log_level raises ValueError for invalid integer."""
        setup_logging(log_dir=tmp_path / "logs")

        with pytest.raises(ValueError, match="Invalid log level"):
            set_log_level(999)

    def test_set_log_level_invalid_type(self, tmp_path: Path) -> None:
        """Test set_log_level raises ValueError for non-integer."""
        setup_logging(log_dir=tmp_path / "logs")

        with pytest.raises(ValueError, match="Invalid log level"):
            set_log_level("DEBUG")  # type: ignore


class TestLogException:
    """Test suite for log_exception function."""

    def test_log_exception_with_exc_info(self, tmp_path: Path) -> None:
        """Test log_exception includes exception info."""
        setup_logging(log_dir=tmp_path / "logs")
        logger = get_logger("test")

        try:
            raise ValueError("Test error")
        except ValueError:
            with patch.object(logger, "error") as mock_error:
                log_exception(logger, "An error occurred")

                mock_error.assert_called_once()
                args, kwargs = mock_error.call_args
                assert args[0] == "An error occurred"
                assert kwargs["exc_info"] is True

    def test_log_exception_without_exc_info(self, tmp_path: Path) -> None:
        """Test log_exception without exception info."""
        setup_logging(log_dir=tmp_path / "logs")
        logger = get_logger("test")

        with patch.object(logger, "error") as mock_error:
            log_exception(logger, "An error occurred", exc_info=False)

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert args[0] == "An error occurred"
            assert kwargs["exc_info"] is False

    def test_log_exception_with_context(self, tmp_path: Path) -> None:
        """Test log_exception includes additional context."""
        setup_logging(log_dir=tmp_path / "logs")
        logger = get_logger("test")

        context = {"user_id": 123, "operation": "process_image"}

        with patch.object(logger, "error") as mock_error:
            log_exception(logger, "Processing failed", **context)

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert args[0] == "Processing failed"
            assert kwargs["extra"]["user_id"] == 123
            assert kwargs["extra"]["operation"] == "process_image"

    def test_log_exception_writes_to_log(self, tmp_path: Path) -> None:
        """Test log_exception actually writes to log file."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir)
        logger = get_logger("test")

        try:
            raise RuntimeError("Test exception")
        except RuntimeError:
            log_exception(logger, "Exception caught", test_key="test_value")

        # Check error log file
        error_log = log_dir / "errors.log"
        assert error_log.exists()

        log_content = error_log.read_text()
        assert "Exception caught" in log_content
        assert "RuntimeError" in log_content
        assert "Test exception" in log_content


class TestConstants:
    """Test suite for module constants."""

    def test_default_log_dir_is_path(self) -> None:
        """Test DEFAULT_LOG_DIR is a Path instance."""
        assert isinstance(DEFAULT_LOG_DIR, Path)

    def test_default_log_dir_includes_app_name(self) -> None:
        """Test DEFAULT_LOG_DIR includes application name."""
        assert ".game_camera_analyzer" in str(DEFAULT_LOG_DIR)
        assert "logs" in str(DEFAULT_LOG_DIR)

    def test_detailed_format_has_required_fields(self) -> None:
        """Test DETAILED_FORMAT includes required fields."""
        assert "%(asctime)s" in DETAILED_FORMAT
        assert "%(name)s" in DETAILED_FORMAT
        assert "%(levelname)s" in DETAILED_FORMAT
        assert "%(filename)s" in DETAILED_FORMAT
        assert "%(lineno)d" in DETAILED_FORMAT
        assert "%(funcName)s" in DETAILED_FORMAT
        assert "%(message)s" in DETAILED_FORMAT

    def test_simple_format_has_required_fields(self) -> None:
        """Test SIMPLE_FORMAT includes required fields."""
        assert "%(asctime)s" in SIMPLE_FORMAT
        assert "%(levelname)s" in SIMPLE_FORMAT
        assert "%(message)s" in SIMPLE_FORMAT


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_full_logging_workflow(self, tmp_path: Path) -> None:
        """Test complete logging workflow."""
        log_dir = tmp_path / "logs"

        # Setup logging
        setup_logging(log_dir=log_dir, log_level=logging.DEBUG)

        # Get logger and log messages
        logger = get_logger("integration_test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify log files
        main_log = log_dir / "game_camera_analyzer.log"
        error_log = log_dir / "errors.log"

        assert main_log.exists()
        assert error_log.exists()

        main_content = main_log.read_text()
        assert "Debug message" in main_content
        assert "Info message" in main_content
        assert "Warning message" in main_content
        assert "Error message" in main_content

        error_content = error_log.read_text()
        assert "Error message" in error_content
        assert "Debug message" not in error_content

    def test_log_level_filtering(self, tmp_path: Path) -> None:
        """Test that log levels filter messages correctly."""
        log_dir = tmp_path / "logs"

        # Setup with WARNING level
        setup_logging(log_dir=log_dir, log_level=logging.WARNING)

        logger = get_logger("filter_test")
        logger.debug("Should not appear")
        logger.info("Should not appear")
        logger.warning("Should appear")
        logger.error("Should appear")

        main_log = log_dir / "game_camera_analyzer.log"
        content = main_log.read_text()

        assert "Should not appear" not in content
        assert "Should appear" in content

    def test_log_rotation(self, tmp_path: Path) -> None:
        """Test log file rotation."""
        log_dir = tmp_path / "logs"

        # Setup with small max_bytes
        max_bytes = 100  # Very small to trigger rotation
        setup_logging(log_dir=log_dir, max_bytes=max_bytes, backup_count=3)

        logger = get_logger("rotation_test")

        # Write enough to trigger rotation
        for i in range(50):
            logger.info(f"Log message number {i} with some extra text to increase size")

        # Check for backup files
        log_files = list(log_dir.glob("game_camera_analyzer.log*"))
        assert len(log_files) > 1  # Should have main file and at least one backup
