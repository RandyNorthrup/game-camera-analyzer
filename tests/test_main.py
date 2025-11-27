"""
Comprehensive tests for main.py entry point module.

Tests command-line argument parsing, environment setup, and application modes.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from main import (
    main,
    parse_arguments,
    run_cli_mode,
    run_gui_mode,
    setup_environment,
)


class TestParseArguments:
    """Test suite for parse_arguments function."""

    def test_parse_no_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing with no arguments (GUI mode)."""
        monkeypatch.setattr(sys, "argv", ["main.py"])
        
        args = parse_arguments()
        
        assert args.input is None
        assert args.output is None
        assert args.cli is False
        assert args.verbose is False
        assert args.quiet is False

    def test_parse_input_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing with input path."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--input", "test.jpg"])
        
        args = parse_arguments()
        
        assert args.input == "test.jpg"

    def test_parse_output_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing with output directory."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--output", "results"])
        
        args = parse_arguments()
        
        assert args.output == "results"

    def test_parse_confidence_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing confidence threshold."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--confidence", "0.8"])
        
        args = parse_arguments()
        
        assert args.confidence == 0.8

    def test_parse_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing batch size."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--batch-size", "16"])
        
        args = parse_arguments()
        
        assert args.batch_size == 16

    def test_parse_device_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing device selection (auto)."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--device", "auto"])
        
        args = parse_arguments()
        
        assert args.device == "auto"

    def test_parse_device_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing device selection (cpu)."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--device", "cpu"])
        
        args = parse_arguments()
        
        assert args.device == "cpu"

    def test_parse_device_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing device selection (cuda)."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--device", "cuda"])
        
        args = parse_arguments()
        
        assert args.device == "cuda"

    def test_parse_device_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing device selection (mps)."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--device", "mps"])
        
        args = parse_arguments()
        
        assert args.device == "mps"

    def test_parse_config_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing custom config file path."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", "custom.json"])
        
        args = parse_arguments()
        
        assert args.config == "custom.json"

    def test_parse_reset_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing reset config flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--reset-config"])
        
        args = parse_arguments()
        
        assert args.reset_config is True

    def test_parse_csv_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing CSV export flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--csv"])
        
        args = parse_arguments()
        
        assert args.csv is True

    def test_parse_no_csv_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing no CSV flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--no-csv"])
        
        args = parse_arguments()
        
        assert args.no_csv is True

    def test_parse_verbose_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing verbose flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--verbose"])
        
        args = parse_arguments()
        
        assert args.verbose is True

    def test_parse_quiet_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing quiet flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--quiet"])
        
        args = parse_arguments()
        
        assert args.quiet is True

    def test_parse_log_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing log file path."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--log-file", "custom.log"])
        
        args = parse_arguments()
        
        assert args.log_file == "custom.log"

    def test_parse_cli_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing CLI mode flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--cli"])
        
        args = parse_arguments()
        
        assert args.cli is True

    def test_parse_short_input_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing short input flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "-i", "test.jpg"])
        
        args = parse_arguments()
        
        assert args.input == "test.jpg"

    def test_parse_short_output_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing short output flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "-o", "results"])
        
        args = parse_arguments()
        
        assert args.output == "results"

    def test_parse_short_config_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing short config flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "-c", "config.json"])
        
        args = parse_arguments()
        
        assert args.config == "config.json"

    def test_parse_short_verbose_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing short verbose flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "-v"])
        
        args = parse_arguments()
        
        assert args.verbose is True

    def test_parse_short_quiet_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing short quiet flag."""
        monkeypatch.setattr(sys, "argv", ["main.py", "-q"])
        
        args = parse_arguments()
        
        assert args.quiet is True

    def test_parse_multiple_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing multiple arguments together."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "main.py",
                "-i", "input.jpg",
                "-o", "output",
                "--confidence", "0.75",
                "--device", "cuda",
                "--verbose",
            ],
        )
        
        args = parse_arguments()
        
        assert args.input == "input.jpg"
        assert args.output == "output"
        assert args.confidence == 0.75
        assert args.device == "cuda"
        assert args.verbose is True


class TestSetupEnvironment:
    """Test suite for setup_environment function."""

    def test_setup_environment_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test environment setup with default settings."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                setup_environment(args)
                
                # Verify config manager was called
                mock_config.assert_called_once()

    def test_setup_environment_verbose(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test environment setup with verbose logging."""
        args = argparse.Namespace(
            verbose=True,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging") as mock_setup:
            with patch("main.get_config_manager"):
                setup_environment(args)
                
                # Verify DEBUG level was set
                call_args = mock_setup.call_args
                assert call_args[1]["log_level"] == logging.DEBUG

    def test_setup_environment_quiet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test environment setup with quiet mode."""
        args = argparse.Namespace(
            verbose=False,
            quiet=True,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging") as mock_setup:
            with patch("main.get_config_manager"):
                setup_environment(args)
                
                # Verify ERROR level was set
                call_args = mock_setup.call_args
                assert call_args[1]["log_level"] == logging.ERROR

    def test_setup_environment_custom_log_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test environment setup with custom log file."""
        log_file = tmp_path / "custom" / "app.log"
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=str(log_file),
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging") as mock_setup:
            with patch("main.get_config_manager"):
                setup_environment(args)
                
                # Verify log dir was set correctly
                call_args = mock_setup.call_args
                assert call_args[1]["log_dir"] == log_file.parent

    def test_setup_environment_reset_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with config reset."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=True,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify reset was called
                mock_mgr.reset_to_defaults.assert_called_once()

    def test_setup_environment_confidence_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with confidence threshold override."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=0.85,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify confidence was set
                mock_mgr.set_value.assert_any_call("detection.confidence_threshold", 0.85)

    def test_setup_environment_batch_size_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with batch size override."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=32,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify batch size was set
                mock_mgr.set_value.assert_any_call("processing.batch_size", 32)

    def test_setup_environment_device_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with device override."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device="cuda",
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify device was set
                mock_mgr.set_value.assert_any_call("detection.device", "cuda")

    def test_setup_environment_output_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with output directory override."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output="custom_output",
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify output dir was set
                mock_mgr.set_value.assert_any_call("output.base_dir", "custom_output")

    def test_setup_environment_csv_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with CSV export enabled."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=True,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify CSV was enabled
                mock_mgr.set_value.assert_any_call("output.export_csv", True)

    def test_setup_environment_csv_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup with CSV export disabled."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=True,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify CSV was disabled
                mock_mgr.set_value.assert_any_call("output.export_csv", False)

    def test_setup_environment_logging_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup handles logging failure."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging", side_effect=Exception("Logging failed")):
            with pytest.raises(SystemExit) as exc_info:
                setup_environment(args)
            
            assert exc_info.value.code == 1

    def test_setup_environment_config_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment setup handles config initialization failure."""
        args = argparse.Namespace(
            verbose=False,
            quiet=False,
            log_file=None,
            reset_config=False,
            confidence=None,
            batch_size=None,
            device=None,
            output=None,
            csv=False,
            no_csv=False,
        )

        with patch("main.setup_logging"):
            with patch("main.get_config_manager", side_effect=Exception("Config failed")):
                with pytest.raises(SystemExit) as exc_info:
                    setup_environment(args)
                
                assert exc_info.value.code == 1


class TestRunCliMode:
    """Test suite for run_cli_mode function."""

    def test_run_cli_mode_returns_error(self, tmp_path: Path) -> None:
        """Test CLI mode returns error code (not implemented)."""
        input_path = tmp_path / "test.jpg"
        input_path.touch()

        with patch("main.logger"):
            result = run_cli_mode(input_path)
        
        assert result == 1

    def test_run_cli_mode_logs_warning(self, tmp_path: Path) -> None:
        """Test CLI mode logs appropriate warning messages."""
        input_path = tmp_path / "test.jpg"
        input_path.touch()

        with patch("main.logger") as mock_logger:
            run_cli_mode(input_path)
            
            # Verify warning was logged
            assert mock_logger.warning.called
            warning_call = mock_logger.warning.call_args[0][0]
            assert "not yet implemented" in warning_call.lower()


class TestRunGuiMode:
    """Test suite for run_gui_mode function."""

    def test_run_gui_mode_success(self) -> None:
        """Test successful GUI mode launch."""
        mock_app = Mock()
        mock_app.exec.return_value = 0
        mock_window = Mock()

        with patch("PySide6.QtWidgets.QApplication", return_value=mock_app):
            with patch("gui.main_window.MainWindow", return_value=mock_window):
                with patch("main.logger"):
                    result = run_gui_mode()
        
        assert result == 0
        mock_window.show.assert_called_once()
        mock_app.exec.assert_called_once()

    def test_run_gui_mode_sets_app_properties(self) -> None:
        """Test GUI mode sets application properties."""
        mock_app = Mock()
        mock_app.exec.return_value = 0

        with patch("PySide6.QtWidgets.QApplication", return_value=mock_app):
            with patch("gui.main_window.MainWindow"):
                with patch("main.logger"):
                    run_gui_mode()
        
        mock_app.setApplicationName.assert_called_once_with("Game Camera Analyzer")
        mock_app.setOrganizationName.assert_called_once_with("Wildlife Research")
        mock_app.setOrganizationDomain.assert_called_once()

    def test_run_gui_mode_import_error(self) -> None:
        """Test GUI mode handles import errors."""
        with patch("builtins.__import__", side_effect=ImportError("PySide6 not found")):
            with patch("main.logger") as mock_logger:
                result = run_gui_mode()
        
        assert result == 1
        mock_logger.error.assert_called()

    def test_run_gui_mode_exception(self) -> None:
        """Test GUI mode handles general exceptions."""
        mock_app = Mock()
        mock_app.exec.side_effect = RuntimeError("GUI crashed")

        with patch("PySide6.QtWidgets.QApplication", return_value=mock_app):
            with patch("gui.main_window.MainWindow"):
                with patch("main.logger") as mock_logger:
                    result = run_gui_mode()
        
        assert result == 1
        mock_logger.error.assert_called()


class TestMain:
    """Test suite for main function."""

    def test_main_gui_mode_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main function defaults to GUI mode."""
        monkeypatch.setattr(sys, "argv", ["main.py"])

        with patch("main.setup_environment"):
            with patch("main.run_gui_mode", return_value=0) as mock_gui:
                result = main()
        
        assert result == 0
        mock_gui.assert_called_once()

    def test_main_cli_mode_with_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main function with CLI mode and input."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--input", "test.jpg"])

        with patch("main.setup_environment"):
            with patch("main.run_cli_mode", return_value=1) as mock_cli:
                result = main()
        
        assert result == 1
        mock_cli.assert_called_once()

    def test_main_cli_mode_without_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main function with CLI flag but no input."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--cli"])

        with patch("main.setup_environment"):
            with patch("main.logger"):
                result = main()
        
        assert result == 1

    def test_main_keyboard_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main function handles keyboard interrupt."""
        monkeypatch.setattr(sys, "argv", ["main.py"])

        with patch("main.setup_environment"):
            with patch("main.run_gui_mode", side_effect=KeyboardInterrupt):
                with patch("main.logger"):
                    result = main()
        
        assert result == 130

    def test_main_unhandled_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main function handles unhandled exceptions."""
        monkeypatch.setattr(sys, "argv", ["main.py"])

        with patch("main.setup_environment"):
            with patch("main.run_gui_mode", side_effect=RuntimeError("Crash")):
                with patch("main.logger") as mock_logger:
                    result = main()
        
        assert result == 1
        mock_logger.critical.assert_called()

    def test_main_logs_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main function logs shutdown message."""
        monkeypatch.setattr(sys, "argv", ["main.py"])

        with patch("main.setup_environment"):
            with patch("main.run_gui_mode", return_value=0):
                with patch("main.logger") as mock_logger:
                    main()
        
        # Verify shutdown was logged
        calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("shutdown" in str(call).lower() for call in calls)


class TestIntegration:
    """Integration tests for main module."""

    def test_full_argument_parsing_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete flow from arguments to environment setup."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "main.py",
                "--verbose",
                "--confidence", "0.9",
                "--device", "cpu",
            ],
        )

        args = parse_arguments()
        
        assert args.verbose is True
        assert args.confidence == 0.9
        assert args.device == "cpu"

        with patch("main.setup_logging"):
            with patch("main.get_config_manager") as mock_config:
                mock_mgr = Mock()
                mock_config.return_value = mock_mgr
                
                setup_environment(args)
                
                # Verify all settings were applied
                mock_mgr.set_value.assert_any_call("detection.confidence_threshold", 0.9)
                mock_mgr.set_value.assert_any_call("detection.device", "cpu")

    def test_version_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test version flag displays version and exits."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--version"])

        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        
        assert exc_info.value.code == 0
