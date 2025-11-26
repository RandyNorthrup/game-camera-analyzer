"""
Main window for the Game Camera Analyzer application.

This module provides the primary GUI interface with:
- File/directory selection
- Batch processing controls
- Real-time progress tracking
- Results display
- Settings access
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import ConfigManager, get_config_manager
from core.batch_processor import BatchConfig, BatchProcessor, BatchProgress
from gui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """
    Background thread for batch processing operations.

    Signals:
        progress_updated: Emitted when processing progress changes
        processing_complete: Emitted when processing finishes
        processing_error: Emitted when an error occurs
    """

    progress_updated = Signal(object)  # BatchProgress object
    processing_complete = Signal(object)  # BatchProgress object
    processing_error = Signal(str)  # Error message

    def __init__(
        self,
        processor: BatchProcessor,
        input_path: Path,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize processing thread.

        Args:
            processor: BatchProcessor instance
            input_path: Path to image or directory to process
            parent: Parent widget
        """
        super().__init__(parent)
        self.processor = processor
        self.input_path = input_path
        self._is_running = True

    def run(self) -> None:
        """Execute batch processing in background thread."""
        try:
            logger.info(f"Starting background processing: {self.input_path}")

            # Process based on input type
            if self.input_path.is_file():
                # Process single file
                progress = self.processor.process_images([self.input_path])
            else:
                # Process directory
                progress = self.processor.process_directory(self.input_path)

            if self._is_running:
                self.processing_complete.emit(progress)
                logger.info("Background processing complete")

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            if self._is_running:
                self.processing_error.emit(str(e))

    def stop(self) -> None:
        """Stop the processing thread gracefully."""
        logger.info("Stopping processing thread")
        self._is_running = False


class MainWindow(QMainWindow):
    """
    Main application window for Game Camera Analyzer.

    Provides interface for:
    - Selecting images/directories for processing
    - Configuring processing options
    - Monitoring progress
    - Viewing results
    """

    def __init__(self) -> None:
        """Initialize main window."""
        super().__init__()

        self.config_manager = get_config_manager()
        self.processor: Optional[BatchProcessor] = None
        self.processing_thread: Optional[ProcessingThread] = None
        self.input_path: Optional[Path] = None

        self._setup_ui()
        self._connect_signals()
        self._load_settings()

        logger.info("Main window initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Game Camera Analyzer")
        self.setMinimumSize(800, 600)
        self.setObjectName("testid_main_window")

        # Create central widget and layout
        central_widget = QWidget()
        central_widget.setObjectName("testid_central_widget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Add UI sections
        main_layout.addWidget(self._create_input_section())
        main_layout.addWidget(self._create_progress_section())
        main_layout.addWidget(self._create_results_section())
        main_layout.addWidget(self._create_actions_section())

    def _create_input_section(self) -> QGroupBox:
        """
        Create input selection section.

        Returns:
            QGroupBox containing input controls
        """
        group = QGroupBox("Input Selection")
        group.setObjectName("testid_input_group")
        layout = QVBoxLayout()

        # Current input label
        self.input_label = QLabel("No input selected")
        self.input_label.setObjectName("testid_input_label")
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)

        # Button row
        button_layout = QHBoxLayout()

        self.select_file_btn = QPushButton("Select Image")
        self.select_file_btn.setObjectName("testid_select_file_button")
        self.select_file_btn.setToolTip("Select a single image file to process")
        button_layout.addWidget(self.select_file_btn)

        self.select_dir_btn = QPushButton("Select Directory")
        self.select_dir_btn.setObjectName("testid_select_directory_button")
        self.select_dir_btn.setToolTip("Select a directory to process all images")
        button_layout.addWidget(self.select_dir_btn)

        layout.addLayout(button_layout)
        group.setLayout(layout)

        return group

    def _create_progress_section(self) -> QGroupBox:
        """
        Create progress tracking section.

        Returns:
            QGroupBox containing progress controls
        """
        group = QGroupBox("Processing Progress")
        group.setObjectName("testid_progress_group")
        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("testid_progress_bar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("testid_status_label")
        layout.addWidget(self.status_label)

        # Details label (ETA, speed, etc.)
        self.details_label = QLabel("")
        self.details_label.setObjectName("testid_details_label")
        layout.addWidget(self.details_label)

        group.setLayout(layout)
        return group

    def _create_results_section(self) -> QGroupBox:
        """
        Create results display section.

        Returns:
            QGroupBox containing results display
        """
        group = QGroupBox("Results")
        group.setObjectName("testid_results_group")
        layout = QVBoxLayout()

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setObjectName("testid_results_text")
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        layout.addWidget(self.results_text)

        group.setLayout(layout)
        return group

    def _create_actions_section(self) -> QWidget:
        """
        Create action buttons section.

        Returns:
            QWidget containing action buttons
        """
        widget = QWidget()
        widget.setObjectName("testid_actions_widget")
        layout = QHBoxLayout()

        # Settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("testid_settings_button")
        self.settings_btn.setToolTip("Configure processing settings")
        layout.addWidget(self.settings_btn)

        # Spacer
        layout.addStretch()

        # Process button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setObjectName("testid_process_button")
        self.process_btn.setEnabled(False)
        self.process_btn.setToolTip("Start batch processing")
        layout.addWidget(self.process_btn)

        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("testid_stop_button")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop current processing")
        layout.addWidget(self.stop_btn)

        widget.setLayout(layout)
        return widget

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        self.select_file_btn.clicked.connect(self._on_select_file)
        self.select_dir_btn.clicked.connect(self._on_select_directory)
        self.process_btn.clicked.connect(self._on_start_processing)
        self.stop_btn.clicked.connect(self._on_stop_processing)
        self.settings_btn.clicked.connect(self._on_open_settings)

    def _load_settings(self) -> None:
        """Load application settings from configuration."""
        try:
            # Load last used directory
            last_dir = self.config_manager.get_value("gui.last_directory")
            if last_dir:
                self.input_path = Path(last_dir)
                self.input_label.setText(f"Last used: {last_dir}")

            logger.debug("Settings loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

    def _save_settings(self) -> None:
        """Save application settings to configuration."""
        try:
            if self.input_path:
                if self.input_path.is_file():
                    save_path = str(self.input_path.parent)
                else:
                    save_path = str(self.input_path)

                self.config_manager.set_value("gui.last_directory", save_path)
                self.config_manager.save_user_config()

            logger.debug("Settings saved successfully")

        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")

    def _on_select_file(self) -> None:
        """Handle file selection."""
        try:
            start_dir = ""
            if self.input_path and self.input_path.is_dir():
                start_dir = str(self.input_path)

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image",
                start_dir,
                "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*.*)",
            )

            if file_path:
                self.input_path = Path(file_path)
                self.input_label.setText(f"Selected file: {file_path}")
                self.process_btn.setEnabled(True)
                self._save_settings()
                logger.info(f"Selected file: {file_path}")

        except Exception as e:
            logger.error(f"File selection error: {e}", exc_info=True)
            self._show_error("File Selection Error", str(e))

    def _on_select_directory(self) -> None:
        """Handle directory selection."""
        try:
            start_dir = ""
            if self.input_path and self.input_path.is_dir():
                start_dir = str(self.input_path)

            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Directory",
                start_dir,
                QFileDialog.Option.ShowDirsOnly,
            )

            if dir_path:
                self.input_path = Path(dir_path)
                # Count images in directory
                image_count = self._count_images(self.input_path)
                self.input_label.setText(
                    f"Selected directory: {dir_path}\n({image_count} images found)"
                )
                self.process_btn.setEnabled(True)
                self._save_settings()
                logger.info(f"Selected directory: {dir_path} ({image_count} images)")

        except Exception as e:
            logger.error(f"Directory selection error: {e}", exc_info=True)
            self._show_error("Directory Selection Error", str(e))

    def _count_images(self, directory: Path) -> int:
        """
        Count image files in directory.

        Args:
            directory: Directory to search

        Returns:
            Number of image files found
        """
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        count = 0

        try:
            for item in directory.rglob("*"):
                if item.is_file() and item.suffix.lower() in extensions:
                    count += 1
        except Exception as e:
            logger.warning(f"Error counting images: {e}")

        return count

    def _on_start_processing(self) -> None:
        """Handle start processing button."""
        if not self.input_path or not self.input_path.exists():
            self._show_error("Invalid Input", "Please select a valid file or directory")
            return

        try:
            # Create batch config from settings
            batch_config = BatchConfig(
                detect=True,
                classify=True,
                crop=True,
                export_csv=self.config_manager.get_value("output.export_csv", True),
                save_annotated=self.config_manager.get_value(
                    "output.save_annotated", False
                ),
                continue_on_error=True,
            )

            # Get configuration values
            output_dir = Path(
                self.config_manager.get_value("output.base_dir", "./output")
            )
            species_db_path = self.config_manager.get_value(
                "classification.species_db", "data/species_db.json"
            )
            detection_conf = self.config_manager.get_value(
                "detection.confidence_threshold", 0.25
            )
            classification_conf = self.config_manager.get_value(
                "classification.threshold", 0.5
            )
            use_features = self.config_manager.get_value(
                "classification.use_feature_classifier", False
            )

            # Create processor
            self.processor = BatchProcessor(
                output_dir=output_dir,
                species_db_path=species_db_path,
                batch_config=batch_config,
                detection_confidence=detection_conf,
                classification_confidence=classification_conf,
                use_feature_classifier=use_features,
            )

            # Create and start processing thread
            self.processing_thread = ProcessingThread(
                self.processor,
                self.input_path,
                self,
            )
            self.processing_thread.progress_updated.connect(self._on_progress_update)
            self.processing_thread.processing_complete.connect(
                self._on_processing_complete
            )
            self.processing_thread.processing_error.connect(self._on_processing_error)

            # Update UI state
            self.process_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.select_file_btn.setEnabled(False)
            self.select_dir_btn.setEnabled(False)
            self.settings_btn.setEnabled(False)

            # Clear previous results
            self.results_text.clear()
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting processing...")

            # Start processing
            self.processing_thread.start()
            logger.info("Processing started")

        except Exception as e:
            logger.error(f"Failed to start processing: {e}", exc_info=True)
            self._show_error("Processing Error", f"Failed to start processing: {e}")
            self._reset_ui_state()

    def _on_stop_processing(self) -> None:
        """Handle stop processing button."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.status_label.setText("Stopping processing...")
            self.stop_btn.setEnabled(False)
            self.processing_thread.stop()
            self.processing_thread.wait(5000)  # Wait up to 5 seconds
            self._reset_ui_state()
            logger.info("Processing stopped by user")

    def _on_progress_update(self, progress: BatchProgress) -> None:
        """
        Handle progress update from processing thread.

        Args:
            progress: Updated progress information
        """
        try:
            # Update progress bar
            if progress.total_images > 0:
                percentage = int((progress.processed_images / progress.total_images) * 100)
                self.progress_bar.setValue(percentage)

            # Update status
            status_text = (
                f"Processing: {progress.processed_images}/{progress.total_images} images"
            )
            if progress.current_image:
                status_text += f" - {progress.current_image}"
            self.status_label.setText(status_text)

            # Update details
            elapsed = progress.get_elapsed_time()
            eta = progress.get_estimated_remaining()
            speed = progress.processed_images / elapsed if elapsed > 0 else 0

            details = f"Elapsed: {elapsed:.1f}s"
            if eta and eta > 0:
                details += f" | ETA: {eta:.1f}s"
            if speed > 0:
                details += f" | Speed: {speed:.2f} img/s"
            details += f" | Detections: {progress.total_detections}"

            self.details_label.setText(details)

        except Exception as e:
            logger.warning(f"Error updating progress: {e}")

    def _on_processing_complete(self, progress: BatchProgress) -> None:
        """
        Handle processing completion.

        Args:
            progress: Final progress information
        """
        try:
            self.progress_bar.setValue(100)
            self.status_label.setText("Processing complete!")

            # Build results summary
            results = []
            results.append("=" * 50)
            results.append("PROCESSING COMPLETE")
            results.append("=" * 50)
            results.append(f"Total images: {progress.total_images}")
            results.append(f"Successful: {progress.successful_images}")
            results.append(f"Failed: {progress.failed_images}")
            results.append(f"Total detections: {progress.total_detections}")
            results.append(f"Classifications: {progress.total_classifications}")
            results.append(f"Crops saved: {progress.total_crops}")
            results.append(f"Processing time: {progress.get_elapsed_time():.2f}s")

            if progress.errors:
                results.append(f"\nErrors ({len(progress.errors)}):")
                for error in progress.errors[:5]:  # Show first 5 errors
                    results.append(f"  • {error}")
                if len(progress.errors) > 5:
                    results.append(f"  ... and {len(progress.errors) - 5} more")

            # Get statistics from processor
            if self.processor:
                stats = self.processor.get_statistics()
                if stats:
                    results.append("\nEngine Statistics:")
                    for engine, engine_stats in stats.items():
                        results.append(f"  {engine}: {engine_stats}")

            self.results_text.setPlainText("\n".join(results))

            self._reset_ui_state()
            logger.info("Processing completed successfully")

            # Show completion message
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Successfully processed {progress.successful_images} of "
                f"{progress.total_images} images.\n\n"
                f"Found {progress.total_detections} animals.",
            )

        except Exception as e:
            logger.error(f"Error handling completion: {e}", exc_info=True)
            self._reset_ui_state()

    def _on_processing_error(self, error_message: str) -> None:
        """
        Handle processing error.

        Args:
            error_message: Error message
        """
        self.status_label.setText("Processing failed")
        self.results_text.setPlainText(f"ERROR: {error_message}")
        self._reset_ui_state()

        logger.error(f"Processing error: {error_message}")
        self._show_error("Processing Error", error_message)

    def _on_open_settings(self) -> None:
        """Handle settings button."""
        try:
            dialog = SettingsDialog(self.config_manager, self)
            result = dialog.exec()

            if result == SettingsDialog.DialogCode.Accepted:
                if dialog.has_changes():
                    logger.info("Settings updated")
                    QMessageBox.information(
                        self,
                        "Settings Saved",
                        "Settings have been saved successfully.\n\n"
                        "New settings will be applied to the next processing run.",
                    )

        except Exception as e:
            logger.error(f"Failed to open settings dialog: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Settings Error",
                f"Failed to open settings: {e}",
            )

    def _reset_ui_state(self) -> None:
        """Reset UI to ready state."""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_file_btn.setEnabled(True)
        self.select_dir_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)

    def _show_error(self, title: str, message: str) -> None:
        """
        Show error message dialog.

        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Handle window close event.

        Args:
            event: Close event
        """
        # Stop processing if running
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.processing_thread.stop()
            self.processing_thread.wait(5000)

        # Save settings
        self._save_settings()

        logger.info("Main window closing")
        event.accept()
