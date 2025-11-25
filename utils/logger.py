"""
Logging configuration for the Game Camera Analyzer application.

This module provides centralized logging configuration with support for:
- File and console handlers
- Log rotation
- Structured logging with context
- Multiple log levels
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# Default log directory
DEFAULT_LOG_DIR: Path = Path.home() / ".game_camera_analyzer" / "logs"

# Log format strings
DETAILED_FORMAT: str = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
SIMPLE_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure application-wide logging with file and console handlers.

    Args:
        log_dir: Directory for log files. Defaults to ~/.game_camera_analyzer/logs
        log_level: Root logger level. Defaults to INFO
        console_level: Console handler level. Defaults to INFO
        file_level: File handler level. Defaults to DEBUG
        max_bytes: Maximum size of log file before rotation. Defaults to 10MB
        backup_count: Number of backup log files to keep. Defaults to 5

    Raises:
        OSError: If log directory cannot be created
        PermissionError: If insufficient permissions for log directory
    """
    # Use default log directory if not specified
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    # Create log directory if it doesn't exist
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        sys.stderr.write(f"ERROR: Cannot create log directory {log_dir}: Permission denied\n")
        raise PermissionError(f"Cannot create log directory {log_dir}") from e
    except OSError as e:
        sys.stderr.write(f"ERROR: Cannot create log directory {log_dir}: {e}\n")
        raise OSError(f"Cannot create log directory {log_dir}") from e

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(SIMPLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create rotating file handler for all logs with detailed format
    all_log_file = log_dir / "game_camera_analyzer.log"
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            all_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(DETAILED_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        sys.stderr.write(f"WARNING: Cannot create log file {all_log_file}: {e}\n")
        # Continue without file handler

    # Create separate error log file
    error_log_file = log_dir / "errors.log"
    try:
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(DETAILED_FORMAT)
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
    except (PermissionError, OSError) as e:
        sys.stderr.write(f"WARNING: Cannot create error log file {error_log_file}: {e}\n")
        # Continue without error handler

    # Log initialization message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialized",
        extra={
            "log_dir": str(log_dir),
            "log_level": logging.getLevelName(log_level),
            "console_level": logging.getLevelName(console_level),
            "file_level": logging.getLevelName(file_level),
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    """
    Change the log level for the root logger.

    Args:
        level: New log level (e.g., logging.DEBUG, logging.INFO)

    Raises:
        ValueError: If level is not a valid log level
    """
    if not isinstance(level, int) or level not in (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ):
        raise ValueError(f"Invalid log level: {level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    logger = get_logger(__name__)
    logger.info(f"Log level changed to {logging.getLevelName(level)}")


def log_exception(
    logger: logging.Logger,
    message: str,
    exc_info: bool = True,
    **kwargs: object,
) -> None:
    """
    Log an exception with additional context.

    Args:
        logger: Logger instance to use
        message: Error message
        exc_info: Whether to include exception info. Defaults to True
        **kwargs: Additional context to include in extra dict

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, "Operation failed", param=value)
    """
    logger.error(message, exc_info=exc_info, extra=kwargs)


if __name__ == "__main__":
    # Test logging configuration
    setup_logging(log_level=logging.DEBUG)
    logger = get_logger(__name__)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test exception")
    except ValueError:
        log_exception(logger, "Test exception caught", test_param="test_value")

    print("\nLogging test completed. Check logs directory for output.")
