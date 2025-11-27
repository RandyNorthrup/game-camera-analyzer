"""
Comprehensive tests for core.csv_exporter module.

Tests ExportConfig, CSVExporter, and all export methods.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.classification_engine import ClassificationResult
from core.cropping_engine import CropResult
from core.csv_exporter import CSVExportError, CSVExporter, ExportConfig
from core.detection_engine import DetectionResult
from models.yolo_detector import Detection


# ================================================================================
# Test ExportConfig
# ================================================================================


class TestExportConfig:
    """Test suite for ExportConfig dataclass."""

    def test_export_config_defaults(self) -> None:
        """Test default configuration values."""
        config = ExportConfig()

        assert config.include_metadata is True
        assert config.include_bbox_details is True
        assert config.include_alternatives is False
        assert config.timestamp_format == "%Y-%m-%d %H:%M:%S"
        assert config.delimiter == ","
        assert config.decimal_separator == "."

    def test_export_config_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ExportConfig(
            include_metadata=False,
            include_bbox_details=False,
            include_alternatives=True,
            timestamp_format="%Y%m%d_%H%M%S",
            delimiter=";",
            decimal_separator=",",
        )

        assert config.include_metadata is False
        assert config.include_bbox_details is False
        assert config.include_alternatives is True
        assert config.timestamp_format == "%Y%m%d_%H%M%S"
        assert config.delimiter == ";"
        assert config.decimal_separator == ","

    def test_export_config_invalid_delimiter(self) -> None:
        """Test that invalid delimiter raises ValueError."""
        with pytest.raises(ValueError, match="delimiter must be one of"):
            ExportConfig(delimiter=":")

    def test_export_config_invalid_decimal_separator(self) -> None:
        """Test that invalid decimal separator raises ValueError."""
        with pytest.raises(ValueError, match="decimal_separator must be"):
            ExportConfig(decimal_separator="_")

    def test_export_config_valid_delimiters(self) -> None:
        """Test all valid delimiters."""
        for delimiter in [",", ";", "\t", "|"]:
            config = ExportConfig(delimiter=delimiter)
            assert config.delimiter == delimiter

    def test_export_config_valid_decimal_separators(self) -> None:
        """Test all valid decimal separators."""
        for separator in [".", ","]:
            config = ExportConfig(decimal_separator=separator)
            assert config.decimal_separator == separator


# ================================================================================
# Test CSVExporter Initialization
# ================================================================================


class TestCSVExporterInitialization:
    """Test suite for CSVExporter initialization."""

    def test_init_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "output"

        exporter = CSVExporter(output_dir=output_dir)

        assert exporter.output_dir.exists()
        assert exporter.output_dir.is_dir()

    def test_init_with_custom_config(self, tmp_path: Path) -> None:
        """Test initialization with custom configuration."""
        config = ExportConfig(
            include_metadata=False,
            delimiter=";",
        )

        exporter = CSVExporter(
            output_dir=tmp_path / "output",
            config=config,
        )

        assert exporter.config.include_metadata is False
        assert exporter.config.delimiter == ";"

    def test_init_default_config(self, tmp_path: Path) -> None:
        """Test initialization with default configuration."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        assert isinstance(exporter.config, ExportConfig)
        assert exporter.config.include_metadata is True

    def test_init_statistics(self, tmp_path: Path) -> None:
        """Test that statistics are initialized."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        stats = exporter.get_stats()
        assert stats["files_exported"] == 0
        assert stats["rows_exported"] == 0
        assert stats["errors"] == 0

    def test_repr(self, tmp_path: Path) -> None:
        """Test string representation."""
        output_dir = tmp_path / "output"
        exporter = CSVExporter(output_dir=output_dir)

        repr_str = repr(exporter)

        assert "CSVExporter" in repr_str
        assert str(output_dir) in repr_str


# ================================================================================
# Test Detection Export
# ================================================================================


class TestDetectionExport:
    """Test suite for detection export methods."""

    def test_export_detections_basic(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test basic detection export."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_detections(
            sample_detection_results,
            output_filename="detections.csv",
        )

        assert output_path.exists()
        assert output_path.name == "detections.csv"

        # Verify CSV content
        df = pd.read_csv(output_path)
        assert len(df) == 2  # 2 detections total
        assert "yolo_class" in df.columns
        assert "confidence" in df.columns

    def test_export_detections_empty_list(self, tmp_path: Path) -> None:
        """Test exporting empty detection list."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_detections(
            [],
            output_filename="empty.csv",
        )

        # Path is returned but file not created for empty data
        assert output_path == tmp_path / "output" / "empty.csv"

    def test_export_detections_with_bbox_details(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test detection export with bbox details."""
        config = ExportConfig(include_bbox_details=True)
        exporter = CSVExporter(output_dir=tmp_path / "output", config=config)

        output_path = exporter.export_detections(sample_detection_results)

        df = pd.read_csv(output_path)
        assert "bbox_x1" in df.columns
        assert "bbox_y1" in df.columns
        assert "bbox_x2" in df.columns
        assert "bbox_y2" in df.columns
        assert "bbox_width" in df.columns
        assert "bbox_height" in df.columns
        assert "bbox_area" in df.columns

    def test_export_detections_without_bbox_details(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test detection export without bbox details."""
        config = ExportConfig(include_bbox_details=False)
        exporter = CSVExporter(output_dir=tmp_path / "output", config=config)

        output_path = exporter.export_detections(sample_detection_results)

        df = pd.read_csv(output_path)
        assert "bbox" in df.columns
        assert "bbox_x1" not in df.columns

    def test_export_detections_with_metadata(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test detection export with metadata."""
        config = ExportConfig(include_metadata=True)
        exporter = CSVExporter(output_dir=tmp_path / "output", config=config)

        output_path = exporter.export_detections(sample_detection_results)

        df = pd.read_csv(output_path)
        assert "image_width" in df.columns
        assert "image_height" in df.columns

    def test_export_detections_append_mode(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test appending to existing detection file."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        # First export
        output_path = exporter.export_detections(
            sample_detection_results[:1],
            append=False,
        )

        df1 = pd.read_csv(output_path)
        rows_first = len(df1)

        # Append more
        exporter.export_detections(
            sample_detection_results[1:],
            append=True,
        )

        df2 = pd.read_csv(output_path)
        assert len(df2) > rows_first

    def test_export_detections_custom_delimiter(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test detection export with custom delimiter."""
        config = ExportConfig(delimiter=";")
        exporter = CSVExporter(output_dir=tmp_path / "output", config=config)

        output_path = exporter.export_detections(sample_detection_results)

        # Read with custom delimiter
        df = pd.read_csv(output_path, sep=";")
        assert len(df) == 2

    def test_export_detections_statistics(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test that statistics are updated after export."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        exporter.export_detections(sample_detection_results)

        stats = exporter.get_stats()
        assert stats["files_exported"] == 1
        assert stats["rows_exported"] == 2
        assert stats["errors"] == 0


# ================================================================================
# Test Classification Export
# ================================================================================


class TestClassificationExport:
    """Test suite for classification export methods."""

    def test_export_classifications_basic(
        self,
        tmp_path: Path,
        sample_detection_results: List[DetectionResult],
        sample_classification_results: List[List[Optional[ClassificationResult]]],
    ) -> None:
        """Test basic classification export."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_classifications(
            sample_detection_results,
            sample_classification_results,
            output_filename="classifications.csv",
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert "species_common_name" in df.columns
        assert "classification_confidence" in df.columns

    def test_export_classifications_length_mismatch(
        self,
        tmp_path: Path,
        sample_detection_results: List[DetectionResult],
    ) -> None:
        """Test that length mismatch raises ValueError."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        with pytest.raises(ValueError, match="length mismatch"):
            exporter.export_classifications(
                sample_detection_results,
                [],  # Empty classification results
            )

    def test_export_classifications_with_alternatives(
        self,
        tmp_path: Path,
        sample_detection_results: List[DetectionResult],
    ) -> None:
        """Test classification export with alternative matches."""
        config = ExportConfig(include_alternatives=True)
        exporter = CSVExporter(output_dir=tmp_path / "output", config=config)

        # Create classification with alternatives
        classification_with_alt = ClassificationResult(
            species_id="deer_whitetailed",
            common_name="white-tailed deer",
            scientific_name="Odocoileus virginianus",
            confidence=0.85,
            yolo_class="deer",
            yolo_confidence=0.9,
            alternative_matches=[("deer_mule", 0.12), ("elk", 0.03)],
        )

        classifications = [[classification_with_alt]]

        output_path = exporter.export_classifications(
            sample_detection_results[:1],
            classifications,
        )

        df = pd.read_csv(output_path)
        assert "alternative_species" in df.columns
        assert "alternative_confidences" in df.columns


# ================================================================================
# Test Crop Export
# ================================================================================


class TestCropExport:
    """Test suite for crop export methods."""

    def test_export_crops_basic(
        self, tmp_path: Path, sample_crop_results: List[List[Optional[CropResult]]]
    ) -> None:
        """Test basic crop export."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_crops(
            sample_crop_results,
            output_filename="crops.csv",
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert "crop_path" in df.columns
        assert "crop_width" in df.columns
        assert "crop_height" in df.columns

    def test_export_crops_empty_list(self, tmp_path: Path) -> None:
        """Test exporting empty crop list."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_crops(
            [],
            output_filename="empty_crops.csv",
        )

        # Path is returned but file not created for empty data
        assert output_path == tmp_path / "output" / "empty_crops.csv"


# ================================================================================
# Test Combined Export
# ================================================================================


class TestCombinedExport:
    """Test suite for combined export methods."""

    def test_export_combined_all_data(
        self,
        tmp_path: Path,
        sample_detection_results: List[DetectionResult],
        sample_classification_results: List[List[Optional[ClassificationResult]]],
        sample_crop_results: List[List[Optional[CropResult]]],
    ) -> None:
        """Test combined export with all data types."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_combined(
            sample_detection_results,
            sample_classification_results,
            sample_crop_results,
            output_filename="combined.csv",
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert "yolo_class" in df.columns
        assert "species_common_name" in df.columns
        assert "crop_path" in df.columns

    def test_export_combined_detections_only(
        self,
        tmp_path: Path,
        sample_detection_results: List[DetectionResult],
    ) -> None:
        """Test combined export with only detection data."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_combined(
            sample_detection_results,
            classification_results=None,
            crop_results=None,
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 2
        # Empty classification/crop columns should exist but be empty (pandas uses NaN for missing values)
        assert "species_id" in df.columns
        assert pd.isna(df["species_id"].iloc[0])

    def test_export_combined_empty_detections(self, tmp_path: Path) -> None:
        """Test combined export with no detections creates empty CSV with headers."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        output_path = exporter.export_combined(
            [],  # No detection results
            classification_results=None,
            crop_results=None,
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 0
        # Headers should still exist
        assert "yolo_class" in df.columns


# ================================================================================
# Test Error Handling
# ================================================================================


class TestErrorHandling:
    """Test suite for error handling."""

    def test_export_with_io_error(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test that IO errors are caught and wrapped."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        # Try to write to a read-only location (will fail)
        import os

        readonly_file = tmp_path / "output" / "readonly.csv"
        readonly_file.touch()
        os.chmod(readonly_file, 0o444)  # Read-only

        try:
            with pytest.raises(CSVExportError):
                exporter.export_detections(
                    sample_detection_results,
                    output_filename="readonly.csv",
                )

            stats = exporter.get_stats()
            assert stats["errors"] > 0
        finally:
            os.chmod(readonly_file, 0o644)  # Restore permissions


# ================================================================================
# Test Statistics and Utilities
# ================================================================================


class TestStatisticsAndUtilities:
    """Test suite for statistics and utility methods."""

    def test_get_stats(self, tmp_path: Path) -> None:
        """Test get_stats returns copy of statistics."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        stats = exporter.get_stats()

        # Modify returned stats
        stats["files_exported"] = 999

        # Original stats should be unchanged
        assert exporter.get_stats()["files_exported"] == 0

    def test_reset_statistics(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult]
    ) -> None:
        """Test reset_statistics clears counters."""
        exporter = CSVExporter(output_dir=tmp_path / "output")

        # Export some data
        exporter.export_detections(sample_detection_results)

        assert exporter.get_stats()["files_exported"] > 0

        # Reset
        exporter.reset_statistics()

        stats = exporter.get_stats()
        assert stats["files_exported"] == 0
        assert stats["rows_exported"] == 0
        assert stats["errors"] == 0


# ================================================================================
# Fixtures
# ================================================================================


@pytest.fixture
def sample_detection_results(tmp_path: Path) -> List[DetectionResult]:
    """Create sample detection results."""
    from utils.image_utils import save_image

    image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    path1 = tmp_path / "image1.jpg"
    path2 = tmp_path / "image2.jpg"

    save_image(image1, path1)
    save_image(image2, path2)

    detection1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="deer",
    )

    detection2 = Detection(
        bbox=(300, 300, 400, 400),
        confidence=0.85,
        class_id=1,
        class_name="bear",
    )

    result1 = DetectionResult(
        image_path=path1,
        image=image1,
        detections=[detection1],
        metadata={"processing_time_seconds": 0.5},
    )

    result2 = DetectionResult(
        image_path=path2,
        image=image2,
        detections=[detection2],
        metadata={"processing_time_seconds": 0.6},
    )

    return [result1, result2]


@pytest.fixture
def sample_classification_results() -> List[List[Optional[ClassificationResult]]]:
    """Create sample classification results."""
    classification1 = ClassificationResult(
        species_id="deer_whitetailed",
        common_name="white-tailed deer",
        scientific_name="Odocoileus virginianus",
        confidence=0.85,
        yolo_class="deer",
        yolo_confidence=0.9,
        metadata={"method": "lookup"},
    )

    classification2 = ClassificationResult(
        species_id="bear_black",
        common_name="black bear",
        scientific_name="Ursus americanus",
        confidence=0.8,
        yolo_class="bear",
        yolo_confidence=0.85,
        metadata={"method": "lookup"},
    )

    return [[classification1], [classification2]]


@pytest.fixture
def sample_crop_results(tmp_path: Path) -> List[List[Optional[CropResult]]]:
    """Create sample crop results."""
    crop_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    detection1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="deer",
    )

    crop1 = CropResult(
        crop_image=crop_img,
        detection=detection1,
        classification=None,
        original_bbox=(100, 100, 200, 200),
        final_bbox=(100, 100, 200, 200),
        output_path=tmp_path / "crop1.jpg",
    )

    return [[crop1]]


class TestCSVExporterErrorHandling:
    """Test error handling in CSV export."""

    def test_export_classifications_no_data_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test export with no classifications logs warning."""
        exporter = CSVExporter(output_dir=tmp_path / "csv")
        
        # Create DetectionResult with no detections
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        empty_result = DetectionResult(
            image_path=tmp_path / "test.jpg",
            image=empty_image,
            detections=[],
            metadata={},
        )
        
        with caplog.at_level(logging.WARNING):
            output = exporter.export_classifications(
                detection_results=[empty_result],
                classification_results=[[]],
                output_filename="empty.csv"
            )
        
        # Should log warning about no classifications (line 222-223)
        assert any("No classifications to export" in record.message for record in caplog.records)
        # Function returns path but doesn't create file when there are no rows
        assert not output.exists()

    def test_export_classifications_exception(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult],
        sample_classification_results: List[List[Optional[ClassificationResult]]]
    ) -> None:
        """Test export handles pandas exceptions."""
        exporter = CSVExporter(output_dir=tmp_path / "csv")
        
        # Mock pandas to raise exception (lines 247-250)
        with patch("pandas.DataFrame.to_csv", side_effect=RuntimeError("Pandas error")):
            with pytest.raises(CSVExportError, match="Classification export failed"):
                exporter.export_classifications(
                    detection_results=sample_detection_results,
                    classification_results=sample_classification_results,
                    output_filename="test.csv"
                )
        
        # Verify error stats incremented
        assert exporter.stats["errors"] == 1

    def test_export_crops_exception(
        self, tmp_path: Path, sample_crop_results: List[List[Optional[CropResult]]]
    ) -> None:
        """Test crop export handles exceptions."""
        exporter = CSVExporter(output_dir=tmp_path / "csv")
        
        # Mock pandas to raise exception (lines 311-314)
        with patch("pandas.DataFrame.to_csv", side_effect=RuntimeError("Pandas error")):
            with pytest.raises(CSVExportError, match="Crop export failed"):
                exporter.export_crops(
                    crop_results=sample_crop_results,
                    output_filename="crops.csv"
                )
        
        # Verify error stats incremented
        assert exporter.stats["errors"] == 1

    def test_export_combined_exception(
        self, tmp_path: Path, sample_detection_results: List[DetectionResult],
        sample_classification_results: List[List[Optional[ClassificationResult]]],
        sample_crop_results: List[List[Optional[CropResult]]]
    ) -> None:
        """Test combined export handles exceptions."""
        exporter = CSVExporter(output_dir=tmp_path / "csv")
        
        # Mock DataFrame to raise exception (lines 397-400)
        with patch("pandas.DataFrame.to_csv", side_effect=RuntimeError("Write error")):
            with pytest.raises(CSVExportError, match="Combined export failed"):
                exporter.export_combined(
                    detection_results=sample_detection_results,
                    classification_results=sample_classification_results,
                    crop_results=sample_crop_results,
                    output_filename="combined.csv"
                )
        
        # Verify error stats incremented
        assert exporter.stats["errors"] == 1

    def test_crop_row_with_classification(
        self, tmp_path: Path
    ) -> None:
        """Test crop row includes classification data when present."""
        exporter = CSVExporter(output_dir=tmp_path / "csv")
        
        # Create crop result with classification
        classification = ClassificationResult(
            species_id="DEER",
            common_name="White-tailed Deer",
            scientific_name="Odocoileus virginianus",
            confidence=0.95,
            yolo_class="animal",
            yolo_confidence=0.98,
        )
        
        crop_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = Detection(
            bbox=(10, 20, 110, 120),
            confidence=0.98,
            class_id=0,
            class_name="animal",
        )
        
        crop_result = CropResult(
            crop_image=crop_img,
            detection=detection,
            classification=classification,
            original_bbox=(10, 20, 110, 120),
            final_bbox=(10, 20, 110, 120),
            output_path=tmp_path / "crop.jpg",
        )
        
        # Export crop results (line 503 adds classification data)
        output = exporter.export_crops(
            crop_results=[[crop_result]],
            output_filename="test_crops.csv"
        )
        
        # Verify classification fields in CSV
        df = pd.read_csv(output)
        assert "species_id" in df.columns
        assert "species_name" in df.columns
        assert df.iloc[0]["species_id"] == "DEER"
        assert df.iloc[0]["species_name"] == "White-tailed Deer"

