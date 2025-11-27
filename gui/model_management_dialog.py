"""
Model Management Dialog for Game Camera Analyzer.

Provides GUI for:
- Downloading YOLOv8 models
- Model version comparison
- Custom model upload
- Performance benchmarking
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QProgressBar,
    QLabel,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QGroupBox,
    QTextEdit,
    QComboBox,
    QSpinBox,
)

from config import get_config_manager
from models.model_downloader import ModelDownloader, ModelInfo, DownloadProgress
from models.model_benchmark import ModelBenchmark, BenchmarkResult, ComparisonResult
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class DownloadThread(QThread):
    """Background thread for model downloads."""

    progress_updated = Signal(object)  # DownloadProgress
    download_complete = Signal(object)  # ModelInfo
    download_failed = Signal(str)  # error message

    def __init__(
        self, downloader: ModelDownloader, model_name: str, force_redownload: bool = False
    ):
        """
        Initialize download thread.

        Args:
            downloader: ModelDownloader instance
            model_name: Model to download
            force_redownload: Force re-download if exists
        """
        super().__init__()
        self.downloader = downloader
        self.model_name = model_name
        self.force_redownload = force_redownload

    def run(self) -> None:
        """Execute download in background thread."""
        try:
            logger.info(f"Starting download thread for {self.model_name}")

            def progress_callback(progress: DownloadProgress) -> None:
                self.progress_updated.emit(progress)

            model_info = self.downloader.download_model(
                model_name=self.model_name,
                progress_callback=progress_callback,
                force_redownload=self.force_redownload,
            )

            self.download_complete.emit(model_info)
            logger.info(f"Download thread completed for {self.model_name}")

        except Exception as e:
            error_msg = f"Download failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.download_failed.emit(error_msg)


class BenchmarkThread(QThread):
    """Background thread for model benchmarking."""

    benchmark_complete = Signal(object)  # BenchmarkResult
    benchmark_failed = Signal(str)  # error message
    progress_updated = Signal(int, int)  # current, total

    def __init__(
        self,
        benchmark: ModelBenchmark,
        model_name: str,
        test_images: List[Path],
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        """
        Initialize benchmark thread.

        Args:
            benchmark: ModelBenchmark instance
            model_name: Model to benchmark
            test_images: Test image paths
            confidence: Confidence threshold
            iou: IoU threshold
        """
        super().__init__()
        self.benchmark = benchmark
        self.model_name = model_name
        self.test_images = test_images
        self.confidence = confidence
        self.iou = iou

    def run(self) -> None:
        """Execute benchmark in background thread."""
        try:
            logger.info(f"Starting benchmark thread for {self.model_name}")

            result = self.benchmark.benchmark_model(
                model_name=self.model_name,
                test_images=self.test_images,
                confidence_threshold=self.confidence,
                iou_threshold=self.iou,
            )

            self.benchmark_complete.emit(result)
            logger.info(f"Benchmark thread completed for {self.model_name}")

        except Exception as e:
            error_msg = f"Benchmark failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.benchmark_failed.emit(error_msg)


class ModelManagementDialog(QDialog):
    """
    Model management dialog.

    Provides interface for downloading, benchmarking, and
    managing YOLOv8 models.
    """

    def __init__(
        self, model_manager: Optional[ModelManager] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize model management dialog.

        Args:
            model_manager: ModelManager instance (creates new if None)
            parent: Parent widget
        """
        super().__init__(parent)

        self.config_manager = get_config_manager()
        self.model_manager = model_manager or ModelManager()

        # Get download directory from config
        download_dir_str = self.config_manager.get_value("model.download_dir")
        download_dir = Path(download_dir_str) if download_dir_str else None
        self.downloader = ModelDownloader(download_dir=download_dir)
        self.benchmark = ModelBenchmark(model_manager=self.model_manager)

        self.download_thread: Optional[DownloadThread] = None
        self.benchmark_thread: Optional[BenchmarkThread] = None

        self._setup_ui()
        self._load_model_list()
        self._connect_signals()

        logger.debug("Model management dialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Model Management")
        self.setMinimumSize(800, 600)
        self.setObjectName("testid_model_management_dialog")

        layout = QVBoxLayout(self)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setObjectName("testid_model_tabs")

        self.tabs.addTab(self._create_download_tab(), "Download Models")
        self.tabs.addTab(self._create_benchmark_tab(), "Benchmark")
        self.tabs.addTab(self._create_comparison_tab(), "Comparison")

        layout.addWidget(self.tabs)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setObjectName("testid_close_btn")
        close_btn.clicked.connect(self.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _create_download_tab(self) -> QWidget:
        """
        Create download tab.

        Returns:
            Widget with download interface
        """
        widget = QWidget()
        widget.setObjectName("testid_download_tab")
        layout = QVBoxLayout(widget)

        # Download directory configuration
        dir_group = QGroupBox("Download Directory")
        dir_group.setObjectName("testid_download_dir_group")
        dir_layout = QHBoxLayout()

        dir_layout.addWidget(QLabel("Models Location:"))

        self.download_dir_edit = QLabel()
        self.download_dir_edit.setObjectName("testid_download_dir_label")
        self.download_dir_edit.setText(str(self.downloader.download_dir))
        self.download_dir_edit.setWordWrap(True)
        dir_layout.addWidget(self.download_dir_edit, stretch=1)

        self.change_dir_btn = QPushButton("Change...")
        self.change_dir_btn.setObjectName("testid_change_dir_btn")
        dir_layout.addWidget(self.change_dir_btn)

        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Model list table
        models_group = QGroupBox("Available Models")
        models_group.setObjectName("testid_models_group")
        models_layout = QVBoxLayout()

        self.models_table = QTableWidget()
        self.models_table.setObjectName("testid_models_table")
        self.models_table.setColumnCount(6)
        self.models_table.setHorizontalHeaderLabels(
            ["Model", "Size", "Parameters (M)", "Size (MB)", "Status", "Description"]
        )
        self.models_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        models_layout.addWidget(self.models_table)
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        # Download controls
        controls_layout = QHBoxLayout()

        self.download_btn = QPushButton("Download Selected")
        self.download_btn.setObjectName("testid_download_btn")

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setObjectName("testid_delete_btn")

        self.import_btn = QPushButton("Import Custom Model")
        self.import_btn.setObjectName("testid_import_btn")

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("testid_refresh_btn")

        controls_layout.addWidget(self.download_btn)
        controls_layout.addWidget(self.delete_btn)
        controls_layout.addWidget(self.import_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.refresh_btn)

        layout.addLayout(controls_layout)

        # Progress bar
        self.download_progress = QProgressBar()
        self.download_progress.setObjectName("testid_download_progress")
        self.download_progress.setVisible(False)
        layout.addWidget(self.download_progress)

        self.download_status = QLabel("")
        self.download_status.setObjectName("testid_download_status")
        self.download_status.setVisible(False)
        layout.addWidget(self.download_status)

        # Storage info
        self.storage_label = QLabel("")
        self.storage_label.setObjectName("testid_storage_label")
        layout.addWidget(self.storage_label)

        return widget

    def _create_benchmark_tab(self) -> QWidget:
        """
        Create benchmark tab.

        Returns:
            Widget with benchmark interface
        """
        widget = QWidget()
        widget.setObjectName("testid_benchmark_tab")
        layout = QVBoxLayout(widget)

        # Model selection
        model_group = QGroupBox("Benchmark Configuration")
        model_group.setObjectName("testid_benchmark_config_group")
        model_layout = QVBoxLayout()

        # Model selector
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))

        self.benchmark_model_combo = QComboBox()
        self.benchmark_model_combo.setObjectName("testid_benchmark_model_combo")
        model_select_layout.addWidget(self.benchmark_model_combo, stretch=1)
        model_layout.addLayout(model_select_layout)

        # Test images selector
        test_images_layout = QHBoxLayout()
        test_images_layout.addWidget(QLabel("Test Images:"))

        self.test_images_path_label = QLabel("No directory selected")
        self.test_images_path_label.setObjectName("testid_test_images_path")
        test_images_layout.addWidget(self.test_images_path_label, stretch=1)

        self.select_test_images_btn = QPushButton("Select Directory")
        self.select_test_images_btn.setObjectName("testid_select_test_images_btn")
        test_images_layout.addWidget(self.select_test_images_btn)

        model_layout.addLayout(test_images_layout)

        # Number of images
        num_images_layout = QHBoxLayout()
        num_images_layout.addWidget(QLabel("Max Images:"))

        self.num_images_spin = QSpinBox()
        self.num_images_spin.setObjectName("testid_num_images_spin")
        self.num_images_spin.setRange(1, 1000)
        self.num_images_spin.setValue(50)
        num_images_layout.addWidget(self.num_images_spin)
        num_images_layout.addStretch()

        model_layout.addLayout(num_images_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Run benchmark button
        self.run_benchmark_btn = QPushButton("Run Benchmark")
        self.run_benchmark_btn.setObjectName("testid_run_benchmark_btn")
        layout.addWidget(self.run_benchmark_btn)

        # Progress
        self.benchmark_progress = QProgressBar()
        self.benchmark_progress.setObjectName("testid_benchmark_progress")
        self.benchmark_progress.setVisible(False)
        layout.addWidget(self.benchmark_progress)

        # Results
        results_group = QGroupBox("Benchmark Results")
        results_group.setObjectName("testid_benchmark_results_group")
        results_layout = QVBoxLayout()

        self.benchmark_results_text = QTextEdit()
        self.benchmark_results_text.setObjectName("testid_benchmark_results_text")
        self.benchmark_results_text.setReadOnly(True)
        results_layout.addWidget(self.benchmark_results_text)

        # Export button
        export_layout = QHBoxLayout()

        self.export_benchmark_btn = QPushButton("Export Results")
        self.export_benchmark_btn.setObjectName("testid_export_benchmark_btn")
        self.export_benchmark_btn.setEnabled(False)
        export_layout.addWidget(self.export_benchmark_btn)
        export_layout.addStretch()

        results_layout.addLayout(export_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.test_images_dir: Optional[Path] = None
        self.current_benchmark_result: Optional[BenchmarkResult] = None

        return widget

    def _create_comparison_tab(self) -> QWidget:
        """
        Create comparison tab.

        Returns:
            Widget with comparison interface
        """
        widget = QWidget()
        widget.setObjectName("testid_comparison_tab")
        layout = QVBoxLayout(widget)

        # Instructions
        info_label = QLabel(
            "Compare multiple models on the same test set. "
            "Configure test images in the Benchmark tab first."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Comparison table
        comparison_group = QGroupBox("Model Comparison")
        comparison_group.setObjectName("testid_comparison_group")
        comparison_layout = QVBoxLayout()

        self.comparison_table = QTableWidget()
        self.comparison_table.setObjectName("testid_comparison_table")
        self.comparison_table.setColumnCount(7)
        self.comparison_table.setHorizontalHeaderLabels(
            [
                "Model",
                "Avg FPS",
                "Min FPS",
                "Max FPS",
                "Detections/Image",
                "Total Detections",
                "Memory (MB)",
            ]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        comparison_layout.addWidget(self.comparison_table)
        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)

        # Run comparison button
        self.run_comparison_btn = QPushButton("Run Comparison (All Downloaded Models)")
        self.run_comparison_btn.setObjectName("testid_run_comparison_btn")
        layout.addWidget(self.run_comparison_btn)

        # Summary
        summary_group = QGroupBox("Summary")
        summary_group.setObjectName("testid_summary_group")
        summary_layout = QVBoxLayout()

        self.summary_text = QTextEdit()
        self.summary_text.setObjectName("testid_summary_text")
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        summary_layout.addWidget(self.summary_text)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Export button
        export_layout = QHBoxLayout()

        self.export_comparison_btn = QPushButton("Export Comparison")
        self.export_comparison_btn.setObjectName("testid_export_comparison_btn")
        self.export_comparison_btn.setEnabled(False)
        export_layout.addWidget(self.export_comparison_btn)
        export_layout.addStretch()

        layout.addLayout(export_layout)

        self.current_comparison_result: Optional[ComparisonResult] = None

        return widget

    def _connect_signals(self) -> None:
        """Connect signals to slots."""
        # Download tab
        self.change_dir_btn.clicked.connect(self._on_change_download_dir)
        self.download_btn.clicked.connect(self._on_download_clicked)
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        self.import_btn.clicked.connect(self._on_import_clicked)
        self.refresh_btn.clicked.connect(self._load_model_list)

        # Benchmark tab
        self.select_test_images_btn.clicked.connect(self._on_select_test_images)
        self.run_benchmark_btn.clicked.connect(self._on_run_benchmark)
        self.export_benchmark_btn.clicked.connect(self._on_export_benchmark)

        # Comparison tab
        self.run_comparison_btn.clicked.connect(self._on_run_comparison)
        self.export_comparison_btn.clicked.connect(self._on_export_comparison)

    def _load_model_list(self) -> None:
        """Load and display available models."""
        try:
            models = self.downloader.list_available_models()

            self.models_table.setRowCount(len(models))

            for i, model in enumerate(models):
                self.models_table.setItem(i, 0, QTableWidgetItem(model.name))
                self.models_table.setItem(i, 1, QTableWidgetItem(model.size.upper()))
                self.models_table.setItem(
                    i, 2, QTableWidgetItem(f"{model.parameters_millions:.1f}")
                )
                self.models_table.setItem(i, 3, QTableWidgetItem(f"{model.size_mb:.1f}"))

                status = "Downloaded" if model.is_downloaded else "Not Downloaded"
                status_item = QTableWidgetItem(status)
                if model.is_downloaded:
                    status_item.setForeground(Qt.GlobalColor.darkGreen)
                self.models_table.setItem(i, 4, status_item)

                self.models_table.setItem(i, 5, QTableWidgetItem(model.description))

            # Update storage info
            storage_info = self.downloader.get_storage_info()
            self.storage_label.setText(
                f"Storage: {storage_info['downloaded_count']} models, "
                f"{storage_info['total_size_mb']:.1f} MB"
            )

            # Update benchmark combo
            downloaded_models = [m for m in models if m.is_downloaded]
            self.benchmark_model_combo.clear()
            for model in downloaded_models:
                self.benchmark_model_combo.addItem(model.name)

            logger.debug(f"Loaded {len(models)} models")

        except Exception as e:
            logger.error(f"Failed to load model list: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load model list: {e}")

    def _on_change_download_dir(self) -> None:
        """Handle change download directory button click."""
        current_dir = str(self.downloader.download_dir)

        new_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Model Download Directory",
            current_dir,
            QFileDialog.Option.ShowDirsOnly,
        )

        if not new_dir:
            return

        new_path = Path(new_dir)

        # Check if directory is writable
        if not os.access(new_path, os.W_OK):
            QMessageBox.critical(self, "Permission Error", f"Cannot write to directory: {new_path}")
            return

        # Confirm if changing to a different directory with existing models
        existing_models = list(new_path.glob("*.pt"))
        if existing_models and new_path != self.downloader.download_dir:
            reply = QMessageBox.question(
                self,
                "Existing Models Found",
                f"Found {len(existing_models)} model(s) in {new_path}.\n\n" f"Use this directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            # Update config
            self.config_manager.set_value("model.download_dir", str(new_path))
            self.config_manager.save_user_config()

            # Create new downloader with new path
            self.downloader = ModelDownloader(download_dir=new_path)

            # Update UI
            self.download_dir_edit.setText(str(new_path))

            # Reload model list
            self._load_model_list()

            QMessageBox.information(
                self, "Directory Changed", f"Model download directory changed to:\n{new_path}"
            )

            logger.info(f"Changed model download directory to: {new_path}")

        except Exception as e:
            logger.error(f"Failed to change download directory: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to change download directory: {e}")

    def _on_download_clicked(self) -> None:
        """Handle download button click."""
        selected_rows = self.models_table.selectionModel().selectedRows()

        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a model to download")
            return

        row = selected_rows[0].row()
        model_name = self.models_table.item(row, 0).text()

        # Check if already downloaded
        model_info = self.downloader.get_model_info(model_name)
        if model_info and model_info.is_downloaded:
            reply = QMessageBox.question(
                self,
                "Already Downloaded",
                f"{model_name} is already downloaded. Re-download?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        self._start_download(model_name, force_redownload=True)

    def _start_download(self, model_name: str, force_redownload: bool = False) -> None:
        """
        Start model download in background thread.

        Args:
            model_name: Model to download
            force_redownload: Force re-download if exists
        """
        self.download_progress.setVisible(True)
        self.download_status.setVisible(True)
        self.download_status.setText(f"Downloading {model_name}...")
        self.download_progress.setValue(0)

        self.download_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.import_btn.setEnabled(False)

        self.download_thread = DownloadThread(self.downloader, model_name, force_redownload)

        self.download_thread.progress_updated.connect(self._on_download_progress)
        self.download_thread.download_complete.connect(self._on_download_complete)
        self.download_thread.download_failed.connect(self._on_download_failed)

        self.download_thread.start()

    def _on_download_progress(self, progress: DownloadProgress) -> None:
        """
        Handle download progress update.

        Args:
            progress: Download progress info
        """
        self.download_progress.setValue(int(progress.percent_complete))

        if progress.speed_mbps > 0:
            self.download_status.setText(
                f"Downloading {progress.model_name}: "
                f"{progress.percent_complete:.1f}% "
                f"({progress.speed_mbps:.1f} MB/s)"
            )

    def _on_download_complete(self, model_info: ModelInfo) -> None:
        """
        Handle download completion.

        Args:
            model_info: Downloaded model info
        """
        self.download_progress.setValue(100)
        self.download_status.setText(f"Download complete: {model_info.name}")

        self.download_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.import_btn.setEnabled(True)

        self._load_model_list()

        QMessageBox.information(
            self, "Download Complete", f"Successfully downloaded {model_info.name}"
        )

    def _on_download_failed(self, error_message: str) -> None:
        """
        Handle download failure.

        Args:
            error_message: Error description
        """
        self.download_status.setText(f"Download failed: {error_message}")

        self.download_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.import_btn.setEnabled(True)

        QMessageBox.critical(self, "Download Failed", error_message)

    def _on_delete_clicked(self) -> None:
        """Handle delete button click."""
        selected_rows = self.models_table.selectionModel().selectedRows()

        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete")
            return

        row = selected_rows[0].row()
        model_name = self.models_table.item(row, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {model_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.downloader.delete_model(model_name)
                self._load_model_list()
                QMessageBox.information(self, "Deleted", f"Deleted {model_name}")
            except Exception as e:
                logger.error(f"Failed to delete model: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete model: {e}")

    def _on_import_clicked(self) -> None:
        """Handle import button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt)"
        )

        if not file_path:
            return

        try:
            model_info = self.downloader.import_custom_model(Path(file_path))
            self._load_model_list()

            QMessageBox.information(
                self, "Import Complete", f"Successfully imported {model_info.name}"
            )

        except Exception as e:
            logger.error(f"Failed to import model: {e}", exc_info=True)
            QMessageBox.critical(self, "Import Failed", str(e))

    def _on_select_test_images(self) -> None:
        """Handle select test images button click."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Test Images Directory")

        if not dir_path:
            return

        self.test_images_dir = Path(dir_path)

        # Count images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f
            for f in self.test_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        self.test_images_path_label.setText(
            f"{self.test_images_dir.name} ({len(image_files)} images)"
        )

        logger.info(f"Selected test images directory: {self.test_images_dir}")

    def _on_run_benchmark(self) -> None:
        """Handle run benchmark button click."""
        if not self.test_images_dir:
            QMessageBox.warning(
                self, "No Test Images", "Please select a test images directory first"
            )
            return

        model_name = self.benchmark_model_combo.currentText()
        if not model_name:
            QMessageBox.warning(
                self, "No Model", "No downloaded models available. Please download a model first."
            )
            return

        # Get test images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        test_images = [
            f
            for f in self.test_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        max_images = self.num_images_spin.value()
        test_images = test_images[:max_images]

        if not test_images:
            QMessageBox.warning(self, "No Images", f"No images found in {self.test_images_dir}")
            return

        self.benchmark_progress.setVisible(True)
        self.benchmark_progress.setRange(0, 0)  # Indeterminate
        self.run_benchmark_btn.setEnabled(False)
        self.benchmark_results_text.setText(f"Running benchmark on {len(test_images)} images...")

        self.benchmark_thread = BenchmarkThread(self.benchmark, model_name, test_images)

        self.benchmark_thread.benchmark_complete.connect(self._on_benchmark_complete)
        self.benchmark_thread.benchmark_failed.connect(self._on_benchmark_failed)

        self.benchmark_thread.start()

    def _on_benchmark_complete(self, result: BenchmarkResult) -> None:
        """
        Handle benchmark completion.

        Args:
            result: Benchmark results
        """
        self.benchmark_progress.setVisible(False)
        self.run_benchmark_btn.setEnabled(True)
        self.current_benchmark_result = result
        self.export_benchmark_btn.setEnabled(True)

        # Display results
        results_text = f"""
Benchmark Results: {result.model_name}
{'='*60}

Device: {result.device}
Images Processed: {result.num_images}
Errors: {result.error_count}

Performance:
  Average FPS: {result.avg_fps:.2f}
  Min FPS: {result.min_fps:.2f}
  Max FPS: {result.max_fps:.2f}
  Total Time: {result.total_time_seconds:.2f}s

Detections:
  Total Detections: {result.total_detections}
  Avg Detections/Image: {result.avg_detections_per_image:.2f}

Configuration:
  Confidence Threshold: {result.confidence_threshold}
  IoU Threshold: {result.iou_threshold}

Memory:
  Peak Memory: {result.memory_mb:.2f} MB
"""

        self.benchmark_results_text.setText(results_text)

        QMessageBox.information(
            self, "Benchmark Complete", f"Benchmark completed: {result.avg_fps:.2f} FPS"
        )

    def _on_benchmark_failed(self, error_message: str) -> None:
        """
        Handle benchmark failure.

        Args:
            error_message: Error description
        """
        self.benchmark_progress.setVisible(False)
        self.run_benchmark_btn.setEnabled(True)

        self.benchmark_results_text.setText(f"Benchmark failed: {error_message}")

        QMessageBox.critical(self, "Benchmark Failed", error_message)

    def _on_export_benchmark(self) -> None:
        """Handle export benchmark button click."""
        if not self.current_benchmark_result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Benchmark Results",
            f"benchmark_{self.current_benchmark_result.model_name}.json",
            "JSON Files (*.json);;CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            output_path = Path(file_path)
            format_type = "json" if output_path.suffix == ".json" else "csv"

            self.benchmark.export_results(
                self.current_benchmark_result, output_path, format=format_type
            )

            QMessageBox.information(
                self, "Export Complete", f"Results exported to {output_path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to export results: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", str(e))

    def _on_run_comparison(self) -> None:
        """Handle run comparison button click."""
        if not self.test_images_dir:
            QMessageBox.warning(
                self, "No Test Images", "Please configure test images in the Benchmark tab first"
            )
            return

        # Get downloaded models
        models = self.downloader.list_available_models()
        downloaded_models = [m.name for m in models if m.is_downloaded]

        if len(downloaded_models) < 2:
            QMessageBox.warning(
                self, "Not Enough Models", "Please download at least 2 models for comparison"
            )
            return

        # Get test images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        test_images = [
            f
            for f in self.test_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        max_images = self.num_images_spin.value()
        test_images = test_images[:max_images]

        if not test_images:
            QMessageBox.warning(self, "No Images", f"No images found in {self.test_images_dir}")
            return

        self.run_comparison_btn.setEnabled(False)
        self.comparison_table.setRowCount(0)
        self.summary_text.setText(f"Running comparison on {len(downloaded_models)} models...")

        try:
            comparison = self.benchmark.compare_models(
                model_names=downloaded_models, test_images=test_images
            )

            self._display_comparison(comparison)
            self.current_comparison_result = comparison
            self.export_comparison_btn.setEnabled(True)

            QMessageBox.information(
                self, "Comparison Complete", f"Compared {len(comparison.benchmarks)} models"
            )

        except Exception as e:
            logger.error(f"Comparison failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Comparison Failed", str(e))

        finally:
            self.run_comparison_btn.setEnabled(True)

    def _display_comparison(self, comparison: ComparisonResult) -> None:
        """
        Display comparison results.

        Args:
            comparison: Comparison results to display
        """
        self.comparison_table.setRowCount(len(comparison.benchmarks))

        for i, result in enumerate(comparison.benchmarks):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(result.model_name))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{result.avg_fps:.2f}"))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{result.min_fps:.2f}"))
            self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{result.max_fps:.2f}"))
            self.comparison_table.setItem(
                i, 4, QTableWidgetItem(f"{result.avg_detections_per_image:.2f}")
            )
            self.comparison_table.setItem(i, 5, QTableWidgetItem(str(result.total_detections)))
            self.comparison_table.setItem(i, 6, QTableWidgetItem(f"{result.memory_mb:.2f}"))

        # Display summary
        summary_text = f"""
Comparison Summary
{'='*60}

Fastest Model: {comparison.fastest_model}
Most Accurate Model: {comparison.most_accurate_model}
Most Efficient Model: {comparison.most_efficient_model}

Total Models Compared: {len(comparison.benchmarks)}
"""

        self.summary_text.setText(summary_text)

    def _on_export_comparison(self) -> None:
        """Handle export comparison button click."""
        if not self.current_comparison_result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Results",
            "model_comparison.json",
            "JSON Files (*.json);;CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            output_path = Path(file_path)
            format_type = "json" if output_path.suffix == ".json" else "csv"

            self.benchmark.export_comparison(
                self.current_comparison_result, output_path, format=format_type
            )

            QMessageBox.information(
                self, "Export Complete", f"Comparison exported to {output_path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to export comparison: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", str(e))

    def closeEvent(self, event) -> None:
        """Handle dialog close event."""
        # Clean up threads
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.quit()
            self.download_thread.wait()

        if self.benchmark_thread and self.benchmark_thread.isRunning():
            self.benchmark_thread.quit()
            self.benchmark_thread.wait()

        event.accept()
