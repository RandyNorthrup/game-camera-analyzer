"""
Settings dialog for the Game Camera Analyzer application.

This module provides configuration interface for:
- Detection settings (confidence, IoU, model)
- Classification settings (threshold, feature classifier)
- Crop settings (padding, dimensions, quality)
- Export settings (CSV format, output paths)
"""

import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import ConfigManager

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """
    Settings configuration dialog.

    Provides tabs for different setting categories:
    - Detection: Model selection, confidence, IoU thresholds
    - Classification: Species threshold, feature classifier
    - Cropping: Padding, dimensions, quality
    - Export: Output paths, CSV format
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize settings dialog.

        Args:
            config_manager: Configuration manager instance
            parent: Parent widget
        """
        super().__init__(parent)

        self.config_manager = config_manager
        self.original_values: dict[str, Any] = {}

        self._setup_ui()
        self._load_settings()
        self._connect_signals()

        logger.debug("Settings dialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)
        self.setObjectName("testid_settings_dialog")

        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setObjectName("testid_settings_tabs")

        # Add tabs
        self.tabs.addTab(self._create_detection_tab(), "Detection")
        self.tabs.addTab(self._create_classification_tab(), "Classification")
        self.tabs.addTab(self._create_cropping_tab(), "Cropping")
        self.tabs.addTab(self._create_export_tab(), "Export")

        layout.addWidget(self.tabs)

        # Button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.button_box.setObjectName("testid_button_box")
        layout.addWidget(self.button_box)

    def _create_detection_tab(self) -> QWidget:
        """
        Create detection settings tab.

        Returns:
            Widget containing detection settings
        """
        widget = QWidget()
        widget.setObjectName("testid_detection_tab")
        layout = QVBoxLayout(widget)

        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_group.setObjectName("testid_model_group")
        model_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("testid_model_combo")
        self.model_combo.addItems([
            "yolov8n.pt (Nano - Fast)",
            "yolov8s.pt (Small)",
            "yolov8m.pt (Medium - Balanced)",
            "yolov8l.pt (Large)",
            "yolov8x.pt (Extra Large - Accurate)",
        ])
        self.model_combo.setToolTip("Select YOLOv8 model variant (larger = more accurate but slower)")
        model_layout.addRow("Model:", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.setObjectName("testid_device_combo")
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setToolTip("Device for inference (auto = automatic detection)")
        model_layout.addRow("Device:", self.device_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Detection parameters group
        params_group = QGroupBox("Detection Parameters")
        params_group.setObjectName("testid_params_group")
        params_layout = QFormLayout()

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setObjectName("testid_confidence_spin")
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSuffix(" (0.01-0.99)")
        self.confidence_spin.setToolTip("Minimum confidence for detections (higher = fewer false positives)")
        params_layout.addRow("Confidence Threshold:", self.confidence_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setObjectName("testid_iou_spin")
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSuffix(" (0.01-0.99)")
        self.iou_spin.setToolTip("IoU threshold for NMS (higher = less overlap allowed)")
        params_layout.addRow("IoU Threshold:", self.iou_spin)

        self.max_detections_spin = QSpinBox()
        self.max_detections_spin.setObjectName("testid_max_detections_spin")
        self.max_detections_spin.setRange(1, 100)
        self.max_detections_spin.setSingleStep(1)
        self.max_detections_spin.setToolTip("Maximum detections per image")
        params_layout.addRow("Max Detections:", self.max_detections_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        layout.addStretch()
        return widget

    def _create_classification_tab(self) -> QWidget:
        """
        Create classification settings tab.

        Returns:
            Widget containing classification settings
        """
        widget = QWidget()
        widget.setObjectName("testid_classification_tab")
        layout = QVBoxLayout(widget)

        # Classification settings group
        class_group = QGroupBox("Classification Settings")
        class_group.setObjectName("testid_class_group")
        class_layout = QFormLayout()

        self.class_threshold_spin = QDoubleSpinBox()
        self.class_threshold_spin.setObjectName("testid_class_threshold_spin")
        self.class_threshold_spin.setRange(0.01, 0.99)
        self.class_threshold_spin.setSingleStep(0.05)
        self.class_threshold_spin.setDecimals(2)
        self.class_threshold_spin.setSuffix(" (0.01-0.99)")
        self.class_threshold_spin.setToolTip("Minimum confidence for species classification")
        class_layout.addRow("Classification Threshold:", self.class_threshold_spin)

        self.use_feature_check = QCheckBox("Enable feature-based classifier")
        self.use_feature_check.setObjectName("testid_use_feature_check")
        self.use_feature_check.setToolTip("Use deep learning features for classification (slower but more accurate)")
        class_layout.addRow("", self.use_feature_check)

        self.max_alternatives_spin = QSpinBox()
        self.max_alternatives_spin.setObjectName("testid_max_alternatives_spin")
        self.max_alternatives_spin.setRange(0, 10)
        self.max_alternatives_spin.setSingleStep(1)
        self.max_alternatives_spin.setToolTip("Number of alternative species to include")
        class_layout.addRow("Max Alternatives:", self.max_alternatives_spin)

        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # Species database group
        db_group = QGroupBox("Species Database")
        db_group.setObjectName("testid_db_group")
        db_layout = QFormLayout()

        db_path_layout = QHBoxLayout()
        self.species_db_edit = QLineEdit()
        self.species_db_edit.setObjectName("testid_species_db_edit")
        self.species_db_edit.setReadOnly(True)
        self.species_db_edit.setToolTip("Path to species database JSON file")
        db_path_layout.addWidget(self.species_db_edit)

        self.species_db_btn = QPushButton("Browse...")
        self.species_db_btn.setObjectName("testid_species_db_button")
        db_path_layout.addWidget(self.species_db_btn)

        db_layout.addRow("Database File:", db_path_layout)

        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        layout.addStretch()
        return widget

    def _create_cropping_tab(self) -> QWidget:
        """
        Create cropping settings tab.

        Returns:
            Widget containing cropping settings
        """
        widget = QWidget()
        widget.setObjectName("testid_cropping_tab")
        layout = QVBoxLayout(widget)

        # Crop parameters group
        params_group = QGroupBox("Crop Parameters")
        params_group.setObjectName("testid_crop_params_group")
        params_layout = QFormLayout()

        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setObjectName("testid_padding_spin")
        self.padding_spin.setRange(0.0, 2.0)
        self.padding_spin.setSingleStep(0.1)
        self.padding_spin.setDecimals(1)
        self.padding_spin.setSuffix(" (0.0-2.0)")
        self.padding_spin.setToolTip("Padding around detection (0.1 = 10% of bbox size)")
        params_layout.addRow("Padding Factor:", self.padding_spin)

        self.square_crops_check = QCheckBox("Force square crops")
        self.square_crops_check.setObjectName("testid_square_crops_check")
        self.square_crops_check.setToolTip("Expand crops to square aspect ratio")
        params_layout.addRow("", self.square_crops_check)

        self.organize_species_check = QCheckBox("Organize by species")
        self.organize_species_check.setObjectName("testid_organize_species_check")
        self.organize_species_check.setToolTip("Create subdirectories for each species")
        params_layout.addRow("", self.organize_species_check)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Dimension constraints group
        dims_group = QGroupBox("Dimension Constraints")
        dims_group.setObjectName("testid_dims_group")
        dims_layout = QFormLayout()

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setObjectName("testid_min_width_spin")
        self.min_width_spin.setRange(0, 10000)
        self.min_width_spin.setSingleStep(10)
        self.min_width_spin.setSuffix(" px (0 = no limit)")
        self.min_width_spin.setToolTip("Minimum crop width in pixels")
        dims_layout.addRow("Min Width:", self.min_width_spin)

        self.min_height_spin = QSpinBox()
        self.min_height_spin.setObjectName("testid_min_height_spin")
        self.min_height_spin.setRange(0, 10000)
        self.min_height_spin.setSingleStep(10)
        self.min_height_spin.setSuffix(" px (0 = no limit)")
        self.min_height_spin.setToolTip("Minimum crop height in pixels")
        dims_layout.addRow("Min Height:", self.min_height_spin)

        self.max_width_spin = QSpinBox()
        self.max_width_spin.setObjectName("testid_max_width_spin")
        self.max_width_spin.setRange(0, 10000)
        self.max_width_spin.setSingleStep(100)
        self.max_width_spin.setSuffix(" px (0 = no limit)")
        self.max_width_spin.setToolTip("Maximum crop width in pixels")
        dims_layout.addRow("Max Width:", self.max_width_spin)

        self.max_height_spin = QSpinBox()
        self.max_height_spin.setObjectName("testid_max_height_spin")
        self.max_height_spin.setRange(0, 10000)
        self.max_height_spin.setSingleStep(100)
        self.max_height_spin.setSuffix(" px (0 = no limit)")
        self.max_height_spin.setToolTip("Maximum crop height in pixels")
        dims_layout.addRow("Max Height:", self.max_height_spin)

        dims_group.setLayout(dims_layout)
        layout.addWidget(dims_group)

        # Quality settings group
        quality_group = QGroupBox("Quality Settings")
        quality_group.setObjectName("testid_quality_group")
        quality_layout = QFormLayout()

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setObjectName("testid_jpeg_quality_spin")
        self.jpeg_quality_spin.setRange(1, 100)
        self.jpeg_quality_spin.setSingleStep(5)
        self.jpeg_quality_spin.setSuffix(" (1-100)")
        self.jpeg_quality_spin.setToolTip("JPEG quality for saved crops (higher = better quality, larger files)")
        quality_layout.addRow("JPEG Quality:", self.jpeg_quality_spin)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        layout.addStretch()
        return widget

    def _create_export_tab(self) -> QWidget:
        """
        Create export settings tab.

        Returns:
            Widget containing export settings
        """
        widget = QWidget()
        widget.setObjectName("testid_export_tab")
        layout = QVBoxLayout(widget)

        # Output paths group
        paths_group = QGroupBox("Output Paths")
        paths_group.setObjectName("testid_paths_group")
        paths_layout = QFormLayout()

        # Base output directory
        base_path_layout = QHBoxLayout()
        self.base_dir_edit = QLineEdit()
        self.base_dir_edit.setObjectName("testid_base_dir_edit")
        self.base_dir_edit.setToolTip("Base directory for all output files")
        base_path_layout.addWidget(self.base_dir_edit)

        self.base_dir_btn = QPushButton("Browse...")
        self.base_dir_btn.setObjectName("testid_base_dir_button")
        base_path_layout.addWidget(self.base_dir_btn)

        paths_layout.addRow("Output Directory:", base_path_layout)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        # CSV settings group
        csv_group = QGroupBox("CSV Export Settings")
        csv_group.setObjectName("testid_csv_group")
        csv_layout = QFormLayout()

        self.export_csv_check = QCheckBox("Enable CSV export")
        self.export_csv_check.setObjectName("testid_export_csv_check")
        self.export_csv_check.setToolTip("Export results to CSV files")
        csv_layout.addRow("", self.export_csv_check)

        self.csv_delimiter_combo = QComboBox()
        self.csv_delimiter_combo.setObjectName("testid_csv_delimiter_combo")
        self.csv_delimiter_combo.addItems(["Comma (,)", "Semicolon (;)", "Tab"])
        self.csv_delimiter_combo.setToolTip("CSV field delimiter")
        csv_layout.addRow("Delimiter:", self.csv_delimiter_combo)

        self.include_confidence_check = QCheckBox("Include confidence values")
        self.include_confidence_check.setObjectName("testid_include_confidence_check")
        csv_layout.addRow("", self.include_confidence_check)

        self.include_alternatives_check = QCheckBox("Include alternative species")
        self.include_alternatives_check.setObjectName("testid_include_alternatives_check")
        csv_layout.addRow("", self.include_alternatives_check)

        self.include_timestamps_check = QCheckBox("Include timestamps")
        self.include_timestamps_check.setObjectName("testid_include_timestamps_check")
        csv_layout.addRow("", self.include_timestamps_check)

        csv_group.setLayout(csv_layout)
        layout.addWidget(csv_group)

        # Other export options group
        other_group = QGroupBox("Other Options")
        other_group.setObjectName("testid_other_export_group")
        other_layout = QFormLayout()

        self.save_annotated_check = QCheckBox("Save annotated images")
        self.save_annotated_check.setObjectName("testid_save_annotated_check")
        self.save_annotated_check.setToolTip("Save images with detection boxes drawn")
        other_layout.addRow("", self.save_annotated_check)

        other_group.setLayout(other_layout)
        layout.addWidget(other_group)

        layout.addStretch()
        return widget

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._on_restore_defaults
        )

        self.species_db_btn.clicked.connect(self._on_browse_species_db)
        self.base_dir_btn.clicked.connect(self._on_browse_base_dir)

    def _load_settings(self) -> None:
        """Load settings from configuration manager."""
        try:
            # Detection settings
            model_name = self.config_manager.get_value("detection.model", "yolov8m.pt")
            model_map = {
                "yolov8n.pt": 0,
                "yolov8s.pt": 1,
                "yolov8m.pt": 2,
                "yolov8l.pt": 3,
                "yolov8x.pt": 4,
            }
            self.model_combo.setCurrentIndex(model_map.get(model_name, 2))

            device = self.config_manager.get_value("detection.device", "auto")
            device_index = max(0, self.device_combo.findText(device))
            self.device_combo.setCurrentIndex(device_index)

            self.confidence_spin.setValue(
                self.config_manager.get_value("detection.confidence_threshold", 0.25)
            )
            self.iou_spin.setValue(
                self.config_manager.get_value("detection.iou_threshold", 0.45)
            )
            self.max_detections_spin.setValue(
                self.config_manager.get_value("detection.max_detections", 20)
            )

            # Classification settings
            self.class_threshold_spin.setValue(
                self.config_manager.get_value("classification.threshold", 0.5)
            )
            self.use_feature_check.setChecked(
                self.config_manager.get_value("classification.use_feature_classifier", False)
            )
            self.max_alternatives_spin.setValue(
                self.config_manager.get_value("classification.max_alternatives", 3)
            )
            self.species_db_edit.setText(
                self.config_manager.get_value("classification.species_db", "data/species_db.json")
            )

            # Cropping settings
            self.padding_spin.setValue(
                self.config_manager.get_value("cropping.padding", 0.1)
            )
            self.square_crops_check.setChecked(
                self.config_manager.get_value("cropping.square_crops", False)
            )
            self.organize_species_check.setChecked(
                self.config_manager.get_value("cropping.organize_by_species", True)
            )
            self.min_width_spin.setValue(
                self.config_manager.get_value("cropping.min_width", 0)
            )
            self.min_height_spin.setValue(
                self.config_manager.get_value("cropping.min_height", 0)
            )
            self.max_width_spin.setValue(
                self.config_manager.get_value("cropping.max_width", 0)
            )
            self.max_height_spin.setValue(
                self.config_manager.get_value("cropping.max_height", 0)
            )
            self.jpeg_quality_spin.setValue(
                self.config_manager.get_value("cropping.jpeg_quality", 95)
            )

            # Export settings
            self.base_dir_edit.setText(
                self.config_manager.get_value("output.base_dir", "./output")
            )
            self.export_csv_check.setChecked(
                self.config_manager.get_value("output.export_csv", True)
            )

            delimiter = self.config_manager.get_value("output.csv_delimiter", ",")
            delimiter_map = {",": 0, ";": 1, "\t": 2}
            self.csv_delimiter_combo.setCurrentIndex(delimiter_map.get(delimiter, 0))

            self.include_confidence_check.setChecked(
                self.config_manager.get_value("output.include_confidence", True)
            )
            self.include_alternatives_check.setChecked(
                self.config_manager.get_value("output.include_alternatives", True)
            )
            self.include_timestamps_check.setChecked(
                self.config_manager.get_value("output.include_timestamps", True)
            )
            self.save_annotated_check.setChecked(
                self.config_manager.get_value("output.save_annotated", False)
            )

            # Store original values for comparison
            self._store_original_values()

            logger.debug("Settings loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load settings: {e}", exc_info=True)

    def _store_original_values(self) -> None:
        """Store current values for change detection."""
        self.original_values = {
            "detection.model": self._get_model_name(),
            "detection.device": self.device_combo.currentText(),
            "detection.confidence_threshold": self.confidence_spin.value(),
            "detection.iou_threshold": self.iou_spin.value(),
            "detection.max_detections": self.max_detections_spin.value(),
            "classification.threshold": self.class_threshold_spin.value(),
            "classification.use_feature_classifier": self.use_feature_check.isChecked(),
            "classification.max_alternatives": self.max_alternatives_spin.value(),
            "classification.species_db": self.species_db_edit.text(),
            "cropping.padding": self.padding_spin.value(),
            "cropping.square_crops": self.square_crops_check.isChecked(),
            "cropping.organize_by_species": self.organize_species_check.isChecked(),
            "cropping.min_width": self.min_width_spin.value(),
            "cropping.min_height": self.min_height_spin.value(),
            "cropping.max_width": self.max_width_spin.value(),
            "cropping.max_height": self.max_height_spin.value(),
            "cropping.jpeg_quality": self.jpeg_quality_spin.value(),
            "output.base_dir": self.base_dir_edit.text(),
            "output.export_csv": self.export_csv_check.isChecked(),
            "output.csv_delimiter": self._get_csv_delimiter(),
            "output.include_confidence": self.include_confidence_check.isChecked(),
            "output.include_alternatives": self.include_alternatives_check.isChecked(),
            "output.include_timestamps": self.include_timestamps_check.isChecked(),
            "output.save_annotated": self.save_annotated_check.isChecked(),
        }

    def _get_model_name(self) -> str:
        """
        Get model name from combo box selection.

        Returns:
            Model filename
        """
        model_names = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        return model_names[self.model_combo.currentIndex()]

    def _get_csv_delimiter(self) -> str:
        """
        Get CSV delimiter from combo box selection.

        Returns:
            Delimiter character
        """
        delimiters = [",", ";", "\t"]
        return delimiters[self.csv_delimiter_combo.currentIndex()]

    def _save_settings(self) -> None:
        """Save settings to configuration manager."""
        try:
            # Detection settings
            self.config_manager.set_value("detection.model", self._get_model_name())
            self.config_manager.set_value("detection.device", self.device_combo.currentText())
            self.config_manager.set_value("detection.confidence_threshold", self.confidence_spin.value())
            self.config_manager.set_value("detection.iou_threshold", self.iou_spin.value())
            self.config_manager.set_value("detection.max_detections", self.max_detections_spin.value())

            # Classification settings
            self.config_manager.set_value("classification.threshold", self.class_threshold_spin.value())
            self.config_manager.set_value("classification.use_feature_classifier", self.use_feature_check.isChecked())
            self.config_manager.set_value("classification.max_alternatives", self.max_alternatives_spin.value())
            self.config_manager.set_value("classification.species_db", self.species_db_edit.text())

            # Cropping settings
            self.config_manager.set_value("cropping.padding", self.padding_spin.value())
            self.config_manager.set_value("cropping.square_crops", self.square_crops_check.isChecked())
            self.config_manager.set_value("cropping.organize_by_species", self.organize_species_check.isChecked())
            self.config_manager.set_value("cropping.min_width", self.min_width_spin.value())
            self.config_manager.set_value("cropping.min_height", self.min_height_spin.value())
            self.config_manager.set_value("cropping.max_width", self.max_width_spin.value())
            self.config_manager.set_value("cropping.max_height", self.max_height_spin.value())
            self.config_manager.set_value("cropping.jpeg_quality", self.jpeg_quality_spin.value())

            # Export settings
            self.config_manager.set_value("output.base_dir", self.base_dir_edit.text())
            self.config_manager.set_value("output.export_csv", self.export_csv_check.isChecked())
            self.config_manager.set_value("output.csv_delimiter", self._get_csv_delimiter())
            self.config_manager.set_value("output.include_confidence", self.include_confidence_check.isChecked())
            self.config_manager.set_value("output.include_alternatives", self.include_alternatives_check.isChecked())
            self.config_manager.set_value("output.include_timestamps", self.include_timestamps_check.isChecked())
            self.config_manager.set_value("output.save_annotated", self.save_annotated_check.isChecked())

            # Persist to disk
            self.config_manager.save_user_config()

            logger.info("Settings saved successfully")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}", exc_info=True)
            raise

    def _on_accept(self) -> None:
        """Handle OK button click."""
        try:
            # Validate settings
            if not self._validate_settings():
                return

            # Save settings
            self._save_settings()

            self.accept()

        except Exception as e:
            logger.error(f"Failed to accept settings: {e}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Settings Error",
                f"Failed to save settings: {e}",
            )

    def _validate_settings(self) -> bool:
        """
        Validate all settings.

        Returns:
            True if settings are valid, False otherwise
        """
        from PySide6.QtWidgets import QMessageBox

        # Validate output directory
        base_dir = self.base_dir_edit.text().strip()
        if not base_dir:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "Output directory cannot be empty.",
            )
            self.tabs.setCurrentIndex(3)  # Export tab
            self.base_dir_edit.setFocus()
            return False

        # Validate species database path
        species_db = self.species_db_edit.text().strip()
        if not species_db:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "Species database path cannot be empty.",
            )
            self.tabs.setCurrentIndex(1)  # Classification tab
            self.species_db_edit.setFocus()
            return False

        # Validate dimension constraints
        min_width = self.min_width_spin.value()
        max_width = self.max_width_spin.value()
        if max_width > 0 and min_width > max_width:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "Minimum width cannot be greater than maximum width.",
            )
            self.tabs.setCurrentIndex(2)  # Cropping tab
            self.min_width_spin.setFocus()
            return False

        min_height = self.min_height_spin.value()
        max_height = self.max_height_spin.value()
        if max_height > 0 and min_height > max_height:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "Minimum height cannot be greater than maximum height.",
            )
            self.tabs.setCurrentIndex(2)  # Cropping tab
            self.min_height_spin.setFocus()
            return False

        return True

    def _on_restore_defaults(self) -> None:
        """Handle Restore Defaults button click."""
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all settings to defaults?\n\n"
            "This will reset all configuration values.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.config_manager.reset_to_defaults()
                self._load_settings()
                logger.info("Settings restored to defaults")

                QMessageBox.information(
                    self,
                    "Defaults Restored",
                    "All settings have been restored to defaults.",
                )

            except Exception as e:
                logger.error(f"Failed to restore defaults: {e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to restore defaults: {e}",
                )

    def _on_browse_species_db(self) -> None:
        """Handle Browse button for species database."""
        try:
            current_path = self.species_db_edit.text()
            start_dir = str(Path(current_path).parent) if current_path else "."

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Species Database",
                start_dir,
                "JSON Files (*.json);;All Files (*.*)",
            )

            if file_path:
                self.species_db_edit.setText(file_path)
                logger.debug(f"Selected species database: {file_path}")

        except Exception as e:
            logger.error(f"Error browsing for species database: {e}", exc_info=True)

    def _on_browse_base_dir(self) -> None:
        """Handle Browse button for base output directory."""
        try:
            current_path = self.base_dir_edit.text()
            start_dir = current_path if current_path else "."

            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory",
                start_dir,
                QFileDialog.Option.ShowDirsOnly,
            )

            if dir_path:
                self.base_dir_edit.setText(dir_path)
                logger.debug(f"Selected output directory: {dir_path}")

        except Exception as e:
            logger.error(f"Error browsing for output directory: {e}", exc_info=True)

    def has_changes(self) -> bool:
        """
        Check if settings have been modified.

        Returns:
            True if settings have changed, False otherwise
        """
        current_values = {
            "detection.model": self._get_model_name(),
            "detection.device": self.device_combo.currentText(),
            "detection.confidence_threshold": self.confidence_spin.value(),
            "detection.iou_threshold": self.iou_spin.value(),
            "detection.max_detections": self.max_detections_spin.value(),
            "classification.threshold": self.class_threshold_spin.value(),
            "classification.use_feature_classifier": self.use_feature_check.isChecked(),
            "classification.max_alternatives": self.max_alternatives_spin.value(),
            "classification.species_db": self.species_db_edit.text(),
            "cropping.padding": self.padding_spin.value(),
            "cropping.square_crops": self.square_crops_check.isChecked(),
            "cropping.organize_by_species": self.organize_species_check.isChecked(),
            "cropping.min_width": self.min_width_spin.value(),
            "cropping.min_height": self.min_height_spin.value(),
            "cropping.max_width": self.max_width_spin.value(),
            "cropping.max_height": self.max_height_spin.value(),
            "cropping.jpeg_quality": self.jpeg_quality_spin.value(),
            "output.base_dir": self.base_dir_edit.text(),
            "output.export_csv": self.export_csv_check.isChecked(),
            "output.csv_delimiter": self._get_csv_delimiter(),
            "output.include_confidence": self.include_confidence_check.isChecked(),
            "output.include_alternatives": self.include_alternatives_check.isChecked(),
            "output.include_timestamps": self.include_timestamps_check.isChecked(),
            "output.save_annotated": self.save_annotated_check.isChecked(),
        }

        return current_values != self.original_values
