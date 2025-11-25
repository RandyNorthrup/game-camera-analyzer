"""
Game Camera Animal Recognition Application

Main entry point for the application. Handles initialization, configuration,
logging setup, and GUI launch.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import get_config_manager
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="game-camera-analyzer",
        description="Automated wildlife detection and classification for game camera images",
        epilog="For more information, see README.md",
    )

    # Input/output options
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input image file or directory (launches GUI if not specified)",
        metavar="PATH",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for processed images and metadata",
        metavar="DIR",
    )

    # Processing options
    parser.add_argument(
        "--confidence",
        type=float,
        help="Detection confidence threshold (0.0-1.0)",
        metavar="FLOAT",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing multiple images",
        metavar="N",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference (auto=automatic detection)",
    )

    # Configuration options
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to custom configuration file",
        metavar="FILE",
    )

    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults",
    )

    # Export options
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV (enabled by default in GUI mode)",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV export",
    )

    # Logging options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output except errors",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (default: logs/app.log)",
        metavar="FILE",
    )

    # Mode options
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Force command-line mode (no GUI)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> None:
    """
    Set up application environment based on command-line arguments.

    Args:
        args: Parsed command-line arguments

    Raises:
        SystemExit: If setup fails
    """
    # Determine log level
    if args.verbose:
        log_level_int = logging.DEBUG
        log_level_name = "DEBUG"
    elif args.quiet:
        log_level_int = logging.ERROR
        log_level_name = "ERROR"
    else:
        log_level_int = logging.INFO
        log_level_name = "INFO"

    # Set up logging
    try:
        log_dir = Path("logs")
        if args.log_file:
            log_file_path = Path(args.log_file)
            log_dir = log_file_path.parent

        setup_logging(
            log_dir=log_dir,
            log_level=log_level_int,
            console_level=log_level_int,
            file_level=logging.DEBUG,
        )
        logger.info("=" * 60)
        logger.info("Game Camera Animal Recognition Application")
        logger.info("=" * 60)
        logger.info(f"Log level: {log_level_name}")

    except Exception as e:
        print(f"ERROR: Failed to set up logging: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize configuration
    try:
        config_mgr = get_config_manager()

        # Reset config if requested
        if args.reset_config:
            config_mgr.reset_to_defaults()
            logger.info("Configuration reset to defaults")

        # Apply command-line overrides
        if args.confidence is not None:
            config_mgr.set_value("detection.confidence_threshold", args.confidence)
            logger.info(f"Set confidence threshold: {args.confidence}")

        if args.batch_size is not None:
            config_mgr.set_value("processing.batch_size", args.batch_size)
            logger.info(f"Set batch size: {args.batch_size}")

        if args.device is not None:
            config_mgr.set_value("detection.device", args.device)
            logger.info(f"Set device: {args.device}")

        if args.output is not None:
            config_mgr.set_value("output.base_dir", args.output)
            logger.info(f"Set output directory: {args.output}")

        if args.csv:
            config_mgr.set_value("output.export_csv", True)
        elif args.no_csv:
            config_mgr.set_value("output.export_csv", False)

        logger.info("Configuration initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        sys.exit(1)


def run_cli_mode(input_path: Path) -> int:
    """
    Run application in command-line mode (batch processing).

    Args:
        input_path: Path to input file or directory

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Starting CLI mode")
    logger.info(f"Input: {input_path}")

    try:
        from utils.validators import validate_file_exists

        # Validate input
        input_path = validate_file_exists(input_path, "input")

        if input_path.is_file():
            logger.info("Processing single file")
            # TODO: Implement single file processing
            logger.warning("CLI mode not yet implemented - coming in Phase 2")
            return 0

        elif input_path.is_dir():
            logger.info("Processing directory")
            # TODO: Implement batch processing
            logger.warning("CLI mode not yet implemented - coming in Phase 2")
            return 0

        else:
            logger.error(f"Invalid input path: {input_path}")
            return 1

    except Exception as e:
        logger.error(f"CLI processing failed: {e}", exc_info=True)
        return 1


def run_gui_mode() -> int:
    """
    Run application in GUI mode.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Starting GUI mode")

    try:
        # Import PySide6 here to avoid loading if not needed
        from PySide6.QtWidgets import QApplication

        # Create application instance
        app = QApplication(sys.argv)
        app.setApplicationName("Game Camera Analyzer")
        app.setOrganizationName("Wildlife Research")
        app.setOrganizationDomain("github.com/RandyNorthrup")

        # Import and create main window
        # TODO: Implement main window in Phase 6
        logger.warning("GUI not yet implemented - coming in Phase 6")
        logger.info("For now, please use --cli mode with --input option")

        # Placeholder message
        from PySide6.QtWidgets import QMessageBox

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Game Camera Analyzer")
        msg.setText("GUI Coming Soon!")
        msg.setInformativeText(
            "The graphical interface is under development.\n\n"
            "Current status:\n"
            "✅ Phase 1: Foundation (Complete)\n"
            "🔄 Phase 2: Detection Engine (Next)\n"
            "⏳ Phase 6: GUI (Planned)\n\n"
            "For now, please check back soon or use CLI mode."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

        return 0

    except ImportError as e:
        logger.error(f"Failed to import GUI dependencies: {e}")
        logger.error("Please ensure PySide6 is installed: pip install PySide6")
        return 1

    except Exception as e:
        logger.error(f"GUI failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """
    Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    args = parse_arguments()

    # Set up environment
    setup_environment(args)

    try:
        # Determine mode
        if args.cli or args.input:
            # CLI mode
            if not args.input:
                logger.error("CLI mode requires --input argument")
                return 1

            input_path = Path(args.input)
            return run_cli_mode(input_path)

        else:
            # GUI mode (default)
            return run_gui_mode()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        return 1

    finally:
        logger.info("Application shutdown")


if __name__ == "__main__":
    sys.exit(main())
