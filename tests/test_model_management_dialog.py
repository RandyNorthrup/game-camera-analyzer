"""
Tests for model management dialog GUI.

Tests model download UI, benchmarking UI, comparison UI,
and background thread operations.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QMessageBox

from gui.model_management_dialog import (
    DownloadThread,
    BenchmarkThread,
    ModelManagementDialog,
)
from models.model_benchmark import BenchmarkResult, ComparisonResult
from models.model_downloader import DownloadProgress, ModelInfo
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_model_manager() -> Mock:
    """
    Create mock ModelManager for testing.
    
    Returns:
        Mock ModelManager instance
    """
    manager = Mock(spec=ModelManager)
    manager.device = "cpu"
    manager.models_dir = Path("/tmp/models")
    return manager


@pytest.fixture
def mock_downloader() -> Mock:
    """
    Create mock ModelDownloader for testing.
    
    Returns:
        Mock ModelDownloader instance
    """
    downloader = Mock()
    
    # Mock list_available_models
    model_info_n = ModelInfo(
        name="yolov8n.pt",
        size="n",
        description="Nano",
        parameters_millions=3.2,
        macs_billions=8.7,
        size_mb=6.2,
        is_downloaded=True,
        local_path=Path("/tmp/models/yolov8n.pt"),
    )
    
    model_info_s = ModelInfo(
        name="yolov8s.pt",
        size="s",
        description="Small",
        parameters_millions=11.2,
        macs_billions=28.6,
        size_mb=22.5,
        is_downloaded=False,
    )
    
    downloader.list_available_models.return_value = [model_info_n, model_info_s]
    downloader.get_storage_info.return_value = {
        "total_size_mb": 28.7,
        "num_models": 2,
        "models_dir": "/tmp/models",
    }
    
    return downloader


@pytest.fixture
def mock_benchmark() -> Mock:
    """
    Create mock ModelBenchmark for testing.
    
    Returns:
        Mock ModelBenchmark instance
    """
    benchmark = Mock()
    
    # Mock benchmark result
    result = BenchmarkResult(
        model_name="yolov8n.pt",
        device="cpu",
        num_images=10,
        total_time_seconds=5.0,
        avg_fps=2.0,
        min_fps=1.5,
        max_fps=2.5,
        avg_detections_per_image=3.5,
        total_detections=35,
        memory_mb=256.0,
        confidence_threshold=0.25,
        iou_threshold=0.45,
    )
    
    benchmark.benchmark_model.return_value = result
    
    return benchmark


@pytest.fixture
def dialog(
    qtbot, mock_model_manager: Mock, mock_downloader: Mock, mock_benchmark: Mock
) -> ModelManagementDialog:
    """
    Create ModelManagementDialog for testing.
    
    Args:
        qtbot: pytest-qt bot fixture
        mock_model_manager: Mock ModelManager
        mock_downloader: Mock ModelDownloader
        mock_benchmark: Mock ModelBenchmark
    
    Returns:
        ModelManagementDialog instance
    """
    with patch(
        "gui.model_management_dialog.ModelDownloader", return_value=mock_downloader
    ):
        with patch(
            "gui.model_management_dialog.ModelBenchmark", return_value=mock_benchmark
        ):
            dialog = ModelManagementDialog(model_manager=mock_model_manager)
            qtbot.addWidget(dialog)
            return dialog


class TestDownloadThread:
    """Tests for DownloadThread class."""
    
    def test_download_thread_success(self, qtbot, mock_downloader: Mock) -> None:
        """Test successful download in background thread."""
        model_info = ModelInfo(
            name="yolov8n.pt",
            size="n",
            description="Nano",
            parameters_millions=3.2,
            macs_billions=8.7,
            size_mb=6.2,
            is_downloaded=True,
            local_path=Path("/tmp/models/yolov8n.pt"),
        )
        
        mock_downloader.download_model.return_value = model_info
        
        thread = DownloadThread(
            downloader=mock_downloader,
            model_name="yolov8n.pt",
            force_redownload=False,
        )
        
        # Capture signals
        completed_models = []
        
        def on_complete(model: ModelInfo) -> None:
            completed_models.append(model)
        
        thread.download_complete.connect(on_complete)
        
        # Run thread
        with qtbot.waitSignal(thread.download_complete, timeout=5000):
            thread.start()
        
        thread.wait()
        
        assert len(completed_models) == 1
        assert completed_models[0].name == "yolov8n.pt"
        mock_downloader.download_model.assert_called_once()
    
    def test_download_thread_failure(self, qtbot, mock_downloader: Mock) -> None:
        """Test download failure in background thread."""
        mock_downloader.download_model.side_effect = RuntimeError("Download failed")
        
        thread = DownloadThread(
            downloader=mock_downloader, model_name="yolov8n.pt", force_redownload=False
        )
        
        # Capture signals
        error_messages = []
        
        def on_error(msg: str) -> None:
            error_messages.append(msg)
        
        thread.download_failed.connect(on_error)
        
        # Run thread
        with qtbot.waitSignal(thread.download_failed, timeout=5000):
            thread.start()
        
        thread.wait()
        
        assert len(error_messages) == 1
        assert "Download failed" in error_messages[0]
    
    def test_download_thread_progress_updates(
        self, qtbot, mock_downloader: Mock
    ) -> None:
        """Test progress updates during download."""
        progress_updates = []
        
        def progress_callback(progress: DownloadProgress) -> None:
            progress_updates.append(progress)
        
        def mock_download(model_name: str, progress_callback=None, **kwargs):
            if progress_callback:
                # Simulate progress updates
                for i in range(3):
                    progress = DownloadProgress(
                        model_name=model_name,
                        total_bytes=1000,
                        downloaded_bytes=(i + 1) * 300,
                        percent_complete=(i + 1) * 30.0,
                    )
                    progress_callback(progress)
            
            return ModelInfo(
                name=model_name,
                size="n",
                description="Nano",
                parameters_millions=3.2,
                macs_billions=8.7,
                size_mb=6.2,
                is_downloaded=True,
            )
        
        mock_downloader.download_model.side_effect = mock_download
        
        thread = DownloadThread(
            downloader=mock_downloader, model_name="yolov8n.pt", force_redownload=False
        )
        
        thread.progress_updated.connect(lambda p: progress_updates.append(p))
        
        with qtbot.waitSignal(thread.download_complete, timeout=5000):
            thread.start()
        
        thread.wait()
        
        assert len(progress_updates) >= 3


class TestBenchmarkThread:
    """Tests for BenchmarkThread class."""
    
    def test_benchmark_thread_success(self, qtbot, mock_benchmark: Mock) -> None:
        """Test successful benchmark in background thread."""
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.5,
            total_detections=35,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        
        mock_benchmark.benchmark_model.return_value = result
        
        thread = BenchmarkThread(
            benchmark=mock_benchmark,
            model_name="yolov8n.pt",
            test_images=[Path("test1.jpg"), Path("test2.jpg")],
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        
        # Capture signals
        results = []
        
        def on_complete(res: BenchmarkResult) -> None:
            results.append(res)
        
        thread.benchmark_complete.connect(on_complete)
        
        # Run thread
        with qtbot.waitSignal(thread.benchmark_complete, timeout=5000):
            thread.start()
        
        thread.wait()
        
        assert len(results) == 1
        assert results[0].model_name == "yolov8n.pt"
        assert results[0].avg_fps == 2.0
    
    def test_benchmark_thread_failure(self, qtbot, mock_benchmark: Mock) -> None:
        """Test benchmark failure in background thread."""
        mock_benchmark.benchmark_model.side_effect = RuntimeError("Benchmark failed")
        
        thread = BenchmarkThread(
            benchmark=mock_benchmark,
            model_name="yolov8n.pt",
            test_images=[Path("test1.jpg")],
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        
        # Capture signals
        error_messages = []
        
        def on_error(msg: str) -> None:
            error_messages.append(msg)
        
        thread.benchmark_failed.connect(on_error)
        
        # Run thread
        with qtbot.waitSignal(thread.benchmark_failed, timeout=5000):
            thread.start()
        
        thread.wait()
        
        assert len(error_messages) == 1
        assert "Benchmark failed" in error_messages[0]


class TestModelManagementDialog:
    """Tests for ModelManagementDialog class."""
    
    def test_dialog_initialization(self, dialog: ModelManagementDialog) -> None:
        """Test dialog initializes correctly."""
        assert dialog.windowTitle() == "Model Management"
        assert dialog.model_manager is not None
        assert dialog.downloader is not None
        assert dialog.benchmark is not None
        
        # Check tabs exist
        assert dialog.tab_widget.count() == 3
        assert dialog.tab_widget.tabText(0) == "Download Models"
        assert dialog.tab_widget.tabText(1) == "Benchmark"
        assert dialog.tab_widget.tabText(2) == "Compare Models"
    
    def test_download_tab_widgets_exist(self, dialog: ModelManagementDialog) -> None:
        """Test download tab has required widgets."""
        # Check for test IDs
        assert dialog.findChild(object, "testid_models_table") is not None
        assert dialog.findChild(object, "testid_download_button") is not None
        assert dialog.findChild(object, "testid_delete_button") is not None
        assert dialog.findChild(object, "testid_import_button") is not None
        assert dialog.findChild(object, "testid_refresh_button") is not None
        assert dialog.findChild(object, "testid_download_progress") is not None
        assert dialog.findChild(object, "testid_storage_label") is not None
    
    def test_benchmark_tab_widgets_exist(self, dialog: ModelManagementDialog) -> None:
        """Test benchmark tab has required widgets."""
        assert dialog.findChild(object, "testid_benchmark_model_combo") is not None
        assert dialog.findChild(object, "testid_test_images_button") is not None
        assert dialog.findChild(object, "testid_num_images_spin") is not None
        assert dialog.findChild(object, "testid_confidence_spin") is not None
        assert dialog.findChild(object, "testid_iou_spin") is not None
        assert dialog.findChild(object, "testid_run_benchmark_button") is not None
        assert dialog.findChild(object, "testid_benchmark_results_text") is not None
    
    def test_comparison_tab_widgets_exist(self, dialog: ModelManagementDialog) -> None:
        """Test comparison tab has required widgets."""
        assert dialog.findChild(object, "testid_comparison_table") is not None
        assert dialog.findChild(object, "testid_run_comparison_button") is not None
        assert dialog.findChild(object, "testid_comparison_summary_text") is not None
    
    def test_load_models_table(
        self, dialog: ModelManagementDialog, mock_downloader: Mock
    ) -> None:
        """Test loading models into table."""
        models_table = dialog.findChild(object, "testid_models_table")
        
        # Should have 2 rows (from mock)
        assert models_table.rowCount() == 2
        
        # Check first row (yolov8n - downloaded)
        assert models_table.item(0, 0).text() == "yolov8n.pt"
        assert models_table.item(0, 1).text() == "Nano"
        assert "✓" in models_table.item(0, 4).text()  # Downloaded status
        
        # Check second row (yolov8s - not downloaded)
        assert models_table.item(1, 0).text() == "yolov8s.pt"
        assert models_table.item(1, 1).text() == "Small"
        assert "✗" in models_table.item(1, 4).text()  # Not downloaded status
    
    def test_refresh_models(
        self, qtbot, dialog: ModelManagementDialog, mock_downloader: Mock
    ) -> None:
        """Test refreshing models list."""
        refresh_button = dialog.findChild(object, "testid_refresh_button")
        
        # Click refresh
        qtbot.mouseClick(refresh_button, Qt.MouseButton.LeftButton)
        
        # Should call list_available_models
        assert mock_downloader.list_available_models.call_count >= 2  # Initial + refresh
    
    def test_download_button_starts_download(
        self, qtbot, dialog: ModelManagementDialog, mock_downloader: Mock
    ) -> None:
        """Test download button initiates download."""
        models_table = dialog.findChild(object, "testid_models_table")
        download_button = dialog.findChild(object, "testid_download_button")
        
        # Select second row (not downloaded)
        models_table.selectRow(1)
        
        # Mock download to return immediately
        model_info = ModelInfo(
            name="yolov8s.pt",
            size="s",
            description="Small",
            parameters_millions=11.2,
            macs_billions=28.6,
            size_mb=22.5,
            is_downloaded=True,
            local_path=Path("/tmp/models/yolov8s.pt"),
        )
        mock_downloader.download_model.return_value = model_info
        
        # Click download
        with patch.object(dialog, "_on_download_clicked") as mock_click:
            qtbot.mouseClick(download_button, Qt.MouseButton.LeftButton)
            mock_click.assert_called_once()
    
    def test_delete_button_deletes_model(
        self, qtbot, dialog: ModelManagementDialog, mock_downloader: Mock
    ) -> None:
        """Test delete button removes downloaded model."""
        models_table = dialog.findChild(object, "testid_models_table")
        delete_button = dialog.findChild(object, "testid_delete_button")
        
        # Select first row (downloaded)
        models_table.selectRow(0)
        
        # Mock QMessageBox to auto-confirm
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ):
            # Click delete
            with patch.object(dialog, "_on_delete_clicked") as mock_delete:
                qtbot.mouseClick(delete_button, Qt.MouseButton.LeftButton)
                mock_delete.assert_called_once()
    
    def test_import_custom_model(
        self, qtbot, dialog: ModelManagementDialog, mock_downloader: Mock, tmp_path: Path
    ) -> None:
        """Test importing custom model."""
        import_button = dialog.findChild(object, "testid_import_button")
        
        # Create fake model file
        custom_model = tmp_path / "custom_model.pt"
        custom_model.write_text("fake model data")
        
        model_info = ModelInfo(
            name="custom_model.pt",
            size="custom",
            description="Custom imported model",
            parameters_millions=0.0,
            macs_billions=0.0,
            size_mb=1.0,
            is_downloaded=True,
            local_path=custom_model,
        )
        mock_downloader.import_custom_model.return_value = model_info
        
        # Mock file dialog to return custom model path
        with patch(
            "gui.model_management_dialog.QFileDialog.getOpenFileName",
            return_value=(str(custom_model), "*.pt"),
        ):
            with patch.object(dialog, "_on_import_clicked") as mock_import:
                qtbot.mouseClick(import_button, Qt.MouseButton.LeftButton)
                mock_import.assert_called_once()
    
    def test_benchmark_model_selection(
        self, dialog: ModelManagementDialog, mock_downloader: Mock
    ) -> None:
        """Test benchmark model combo box populated."""
        combo = dialog.findChild(object, "testid_benchmark_model_combo")
        
        # Should have downloaded models (1 in mock)
        assert combo.count() >= 1
        assert "yolov8n.pt" in [combo.itemText(i) for i in range(combo.count())]
    
    def test_select_test_images_directory(
        self, qtbot, dialog: ModelManagementDialog, tmp_path: Path
    ) -> None:
        """Test selecting test images directory."""
        test_images_button = dialog.findChild(object, "testid_test_images_button")
        
        # Create test images directory
        images_dir = tmp_path / "test_images"
        images_dir.mkdir()
        (images_dir / "img1.jpg").write_text("fake")
        (images_dir / "img2.jpg").write_text("fake")
        
        # Mock directory dialog
        with patch(
            "gui.model_management_dialog.QFileDialog.getExistingDirectory",
            return_value=str(images_dir),
        ):
            qtbot.mouseClick(test_images_button, Qt.MouseButton.LeftButton)
            
            # Check path label updated
            assert dialog.test_images_path == images_dir
    
    def test_run_benchmark(
        self, qtbot, dialog: ModelManagementDialog, mock_benchmark: Mock, tmp_path: Path
    ) -> None:
        """Test running benchmark."""
        # Setup: select model and images
        combo = dialog.findChild(object, "testid_benchmark_model_combo")
        combo.setCurrentIndex(0)  # Select first model
        
        images_dir = tmp_path / "test_images"
        images_dir.mkdir()
        (images_dir / "img1.jpg").write_text("fake")
        dialog.test_images_path = images_dir
        
        run_button = dialog.findChild(object, "testid_run_benchmark_button")
        
        # Mock benchmark result
        result = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=1,
            total_time_seconds=1.0,
            avg_fps=1.0,
            min_fps=1.0,
            max_fps=1.0,
            avg_detections_per_image=2.0,
            total_detections=2,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        mock_benchmark.benchmark_model.return_value = result
        
        with patch.object(dialog, "_on_run_benchmark") as mock_run:
            qtbot.mouseClick(run_button, Qt.MouseButton.LeftButton)
            mock_run.assert_called_once()
    
    def test_run_comparison(
        self, qtbot, dialog: ModelManagementDialog, mock_benchmark: Mock
    ) -> None:
        """Test running model comparison."""
        run_button = dialog.findChild(object, "testid_run_comparison_button")
        
        # Setup test images
        dialog.test_images_path = Path("/tmp/test_images")
        
        # Mock comparison result
        result1 = BenchmarkResult(
            model_name="yolov8n.pt",
            device="cpu",
            num_images=10,
            total_time_seconds=5.0,
            avg_fps=2.0,
            min_fps=1.5,
            max_fps=2.5,
            avg_detections_per_image=3.0,
            total_detections=30,
            memory_mb=256.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        
        comparison = ComparisonResult(benchmarks=[result1])
        mock_benchmark.compare_models.return_value = comparison
        
        with patch.object(dialog, "_on_run_comparison") as mock_run:
            qtbot.mouseClick(run_button, Qt.MouseButton.LeftButton)
            mock_run.assert_called_once()
    
    def test_storage_info_display(
        self, dialog: ModelManagementDialog, mock_downloader: Mock
    ) -> None:
        """Test storage info label shows correct information."""
        storage_label = dialog.findChild(object, "testid_storage_label")
        
        # Should display storage info
        label_text = storage_label.text()
        assert "28.7" in label_text or "29" in label_text  # Size in MB
        assert "2" in label_text  # Number of models
    
    def test_download_progress_visibility(
        self, dialog: ModelManagementDialog
    ) -> None:
        """Test download progress bar visibility."""
        progress_bar = dialog.findChild(object, "testid_download_progress")
        
        # Initially hidden or at 0
        assert not progress_bar.isVisible() or progress_bar.value() == 0
    
    def test_benchmark_results_display(
        self, dialog: ModelManagementDialog
    ) -> None:
        """Test benchmark results text area exists and is writable."""
        results_text = dialog.findChild(object, "testid_benchmark_results_text")
        
        # Should be read-only
        assert results_text.isReadOnly()
        
        # Should be able to set text
        test_text = "Test results"
        results_text.setPlainText(test_text)
        assert test_text in results_text.toPlainText()
    
    def test_comparison_summary_display(
        self, dialog: ModelManagementDialog
    ) -> None:
        """Test comparison summary text area exists."""
        summary_text = dialog.findChild(object, "testid_comparison_summary_text")
        
        # Should be read-only
        assert summary_text.isReadOnly()
    
    def test_confidence_threshold_range(
        self, dialog: ModelManagementDialog
    ) -> None:
        """Test confidence threshold spin box has correct range."""
        confidence_spin = dialog.findChild(object, "testid_confidence_spin")
        
        assert confidence_spin.minimum() == 0.0
        assert confidence_spin.maximum() == 1.0
        assert confidence_spin.singleStep() == 0.05
    
    def test_iou_threshold_range(self, dialog: ModelManagementDialog) -> None:
        """Test IOU threshold spin box has correct range."""
        iou_spin = dialog.findChild(object, "testid_iou_spin")
        
        assert iou_spin.minimum() == 0.0
        assert iou_spin.maximum() == 1.0
        assert iou_spin.singleStep() == 0.05
    
    def test_num_images_range(self, dialog: ModelManagementDialog) -> None:
        """Test number of images spin box has correct range."""
        num_images_spin = dialog.findChild(object, "testid_num_images_spin")
        
        assert num_images_spin.minimum() >= 1
        assert num_images_spin.maximum() >= 100
    
    def test_dialog_accepts_on_close(self, qtbot, dialog: ModelManagementDialog) -> None:
        """Test dialog can be closed."""
        # Dialog should be closable
        assert dialog.result() == QDialog.DialogCode.Rejected
        
        # Simulate accept
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
    
    def test_dialog_rejects_on_cancel(
        self, qtbot, dialog: ModelManagementDialog
    ) -> None:
        """Test dialog can be rejected."""
        dialog.reject()
        assert dialog.result() == QDialog.DialogCode.Rejected


class TestModelManagementDialogIntegration:
    """Integration tests for ModelManagementDialog."""
    
    @patch("gui.model_management_dialog.ModelDownloader")
    @patch("gui.model_management_dialog.ModelBenchmark")
    def test_full_download_workflow(
        self,
        mock_benchmark_class: Mock,
        mock_downloader_class: Mock,
        qtbot,
        mock_model_manager: Mock,
    ) -> None:
        """Test complete download workflow from UI to completion."""
        # Setup mocks
        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader
        
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        
        model_info_not_downloaded = ModelInfo(
            name="yolov8n.pt",
            size="n",
            description="Nano",
            parameters_millions=3.2,
            macs_billions=8.7,
            size_mb=6.2,
            is_downloaded=False,
        )
        
        model_info_downloaded = ModelInfo(
            name="yolov8n.pt",
            size="n",
            description="Nano",
            parameters_millions=3.2,
            macs_billions=8.7,
            size_mb=6.2,
            is_downloaded=True,
            local_path=Path("/tmp/models/yolov8n.pt"),
        )
        
        mock_downloader.list_available_models.return_value = [model_info_not_downloaded]
        mock_downloader.download_model.return_value = model_info_downloaded
        mock_downloader.get_storage_info.return_value = {
            "total_size_mb": 6.2,
            "num_models": 1,
            "models_dir": "/tmp/models",
        }
        
        # Create dialog
        dialog = ModelManagementDialog(model_manager=mock_model_manager)
        qtbot.addWidget(dialog)
        
        # Verify initial state
        models_table = dialog.findChild(object, "testid_models_table")
        assert models_table.rowCount() == 1
        assert "✗" in models_table.item(0, 4).text()
        
        # Select model and click download
        models_table.selectRow(0)
        download_button = dialog.findChild(object, "testid_download_button")
        
        # Simulate download completion via signal
        QTimer.singleShot(100, lambda: dialog._on_download_complete(model_info_downloaded))
        
        qtbot.mouseClick(download_button, Qt.MouseButton.LeftButton)
        
        # Allow event processing
        qtbot.wait(200)
    
    @patch("gui.model_management_dialog.ModelDownloader")
    @patch("gui.model_management_dialog.ModelBenchmark")
    def test_benchmark_workflow_without_images(
        self,
        mock_benchmark_class: Mock,
        mock_downloader_class: Mock,
        qtbot,
        mock_model_manager: Mock,
    ) -> None:
        """Test benchmark workflow shows error when no images selected."""
        # Setup mocks
        mock_downloader = Mock()
        mock_downloader_class.return_value = mock_downloader
        
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        
        model_info = ModelInfo(
            name="yolov8n.pt",
            size="n",
            description="Nano",
            parameters_millions=3.2,
            macs_billions=8.7,
            size_mb=6.2,
            is_downloaded=True,
            local_path=Path("/tmp/models/yolov8n.pt"),
        )
        
        mock_downloader.list_available_models.return_value = [model_info]
        mock_downloader.get_storage_info.return_value = {
            "total_size_mb": 6.2,
            "num_models": 1,
            "models_dir": "/tmp/models",
        }
        
        # Create dialog
        dialog = ModelManagementDialog(model_manager=mock_model_manager)
        qtbot.addWidget(dialog)
        
        # Try to run benchmark without selecting images
        run_button = dialog.findChild(object, "testid_run_benchmark_button")
        
        with patch.object(QMessageBox, "warning") as mock_warning:
            qtbot.mouseClick(run_button, Qt.MouseButton.LeftButton)
            
            # Should show warning
            mock_warning.assert_called_once()
