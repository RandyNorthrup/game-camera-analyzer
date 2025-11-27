"""
CSV export functionality for detection and classification results.

This module provides functionality to export detection, classification,
and cropping results to CSV format for analysis and record-keeping.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from core.classification_engine import ClassificationResult
from core.cropping_engine import CropResult
from core.detection_engine import DetectionResult
from models.yolo_detector import Detection
from utils.validators import validate_directory_writable

logger = logging.getLogger(__name__)


class CSVExportError(Exception):
    """Exception raised for CSV export errors."""

    pass


@dataclass
class ExportConfig:
    """
    Configuration for CSV export operations.

    Attributes:
        include_metadata: Include image metadata columns
        include_bbox_details: Include detailed bbox coordinates
        include_alternatives: Include alternative classification matches
        timestamp_format: Format string for timestamps
        delimiter: CSV delimiter character
        decimal_separator: Decimal separator for floats
    """

    include_metadata: bool = True
    include_bbox_details: bool = True
    include_alternatives: bool = False
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    delimiter: str = ","
    decimal_separator: str = "."

    def __post_init__(self) -> None:
        """Validate export configuration."""
        if self.delimiter not in [",", ";", "\t", "|"]:
            raise ValueError(
                f"delimiter must be one of [',', ';', '\\t', '|'], got '{self.delimiter}'"
            )
        if self.decimal_separator not in [".", ","]:
            raise ValueError(
                f"decimal_separator must be '.' or ',', got '{self.decimal_separator}'"
            )


class CSVExporter:
    """
    Exporter for detection and classification results to CSV format.

    This exporter handles:
    - Detection results export (bbox, class, confidence)
    - Classification results export (species, confidence)
    - Crop results export (file paths, dimensions)
    - Metadata export (timestamps, image info)
    - Batch processing with append support
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[ExportConfig] = None,
    ) -> None:
        """
        Initialize CSV exporter.

        Args:
            output_dir: Directory for saving CSV files
            config: Export configuration (uses defaults if None)

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If output_dir is invalid
        """
        logger.info("Initializing CSVExporter")

        self.output_dir = Path(output_dir)
        validate_directory_writable(self.output_dir, create=True)

        self.config = config or ExportConfig()

        # Track statistics
        self.stats = {
            "files_exported": 0,
            "rows_exported": 0,
            "errors": 0,
        }

        logger.info(f"CSVExporter initialized: output={self.output_dir}")

    def export_detections(
        self,
        detection_results: List[DetectionResult],
        output_filename: str = "detections.csv",
        append: bool = False,
    ) -> Path:
        """
        Export detection results to CSV.

        Args:
            detection_results: List of DetectionResult objects
            output_filename: Output CSV filename
            append: Whether to append to existing file

        Returns:
            Path to the exported CSV file

        Raises:
            CSVExportError: If export fails
        """
        try:
            output_path = self.output_dir / output_filename
            logger.info(f"Exporting {len(detection_results)} detection results to {output_path}")

            # Build rows
            rows: List[Dict[str, Any]] = []

            for det_result in detection_results:
                for detection in det_result.detections:
                    row = self._detection_to_row(det_result, detection)
                    rows.append(row)

            if not rows:
                logger.warning("No detections to export")
                return output_path

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Export to CSV
            mode = "a" if append and output_path.exists() else "w"
            header = not (append and output_path.exists())

            df.to_csv(
                output_path,
                index=False,
                sep=self.config.delimiter,
                decimal=self.config.decimal_separator,
                mode=mode,
                header=header,
            )

            self.stats["files_exported"] += 1
            self.stats["rows_exported"] += len(rows)

            logger.info(f"Exported {len(rows)} rows to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export detections: {e}", exc_info=True)
            self.stats["errors"] += 1
            raise CSVExportError(f"Detection export failed: {e}") from e

    def export_classifications(
        self,
        detection_results: List[DetectionResult],
        classification_results: List[List[Optional[ClassificationResult]]],
        output_filename: str = "classifications.csv",
        append: bool = False,
    ) -> Path:
        """
        Export classification results to CSV.

        Args:
            detection_results: List of DetectionResult objects
            classification_results: List of classification lists (one per detection result)
            output_filename: Output CSV filename
            append: Whether to append to existing file

        Returns:
            Path to the exported CSV file

        Raises:
            CSVExportError: If export fails
            ValueError: If list lengths don't match
        """
        if len(detection_results) != len(classification_results):
            raise ValueError(
                f"Detection and classification lists length mismatch: "
                f"{len(detection_results)} vs {len(classification_results)}"
            )

        try:
            output_path = self.output_dir / output_filename
            logger.info(
                f"Exporting {len(detection_results)} classification results to {output_path}"
            )

            # Build rows
            rows: List[Dict[str, Any]] = []

            for det_result, classifications in zip(
                detection_results, classification_results
            ):
                for detection, classification in zip(
                    det_result.detections, classifications
                ):
                    if classification is not None:
                        row = self._classification_to_row(
                            det_result, detection, classification
                        )
                        rows.append(row)

            if not rows:
                logger.warning("No classifications to export")
                return output_path

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Export to CSV
            mode = "a" if append and output_path.exists() else "w"
            header = not (append and output_path.exists())

            df.to_csv(
                output_path,
                index=False,
                sep=self.config.delimiter,
                decimal=self.config.decimal_separator,
                mode=mode,
                header=header,
            )

            self.stats["files_exported"] += 1
            self.stats["rows_exported"] += len(rows)

            logger.info(f"Exported {len(rows)} rows to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export classifications: {e}", exc_info=True)
            self.stats["errors"] += 1
            raise CSVExportError(f"Classification export failed: {e}") from e

    def export_crops(
        self,
        crop_results: List[List[Optional[CropResult]]],
        output_filename: str = "crops.csv",
        append: bool = False,
    ) -> Path:
        """
        Export crop results to CSV.

        Args:
            crop_results: List of lists of CropResult objects
            output_filename: Output CSV filename
            append: Whether to append to existing file

        Returns:
            Path to the exported CSV file

        Raises:
            CSVExportError: If export fails
        """
        try:
            output_path = self.output_dir / output_filename
            logger.info(f"Exporting crop results to {output_path}")

            # Build rows
            rows: List[Dict[str, Any]] = []

            for image_crops in crop_results:
                for crop_result in image_crops:
                    if crop_result is not None:
                        row = self._crop_to_row(crop_result)
                        rows.append(row)

            if not rows:
                logger.warning("No crops to export")
                return output_path

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Export to CSV
            mode = "a" if append and output_path.exists() else "w"
            header = not (append and output_path.exists())

            df.to_csv(
                output_path,
                index=False,
                sep=self.config.delimiter,
                decimal=self.config.decimal_separator,
                mode=mode,
                header=header,
            )

            self.stats["files_exported"] += 1
            self.stats["rows_exported"] += len(rows)

            logger.info(f"Exported {len(rows)} rows to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export crops: {e}", exc_info=True)
            self.stats["errors"] += 1
            raise CSVExportError(f"Crop export failed: {e}") from e

    def export_combined(
        self,
        detection_results: List[DetectionResult],
        classification_results: Optional[
            List[List[Optional[ClassificationResult]]]
        ] = None,
        crop_results: Optional[List[List[Optional[CropResult]]]] = None,
        output_filename: str = "combined_results.csv",
        append: bool = False,
    ) -> Path:
        """
        Export combined detection, classification, and crop results to CSV.

        Args:
            detection_results: List of DetectionResult objects
            classification_results: Optional list of classification lists
            crop_results: Optional list of crop lists
            output_filename: Output CSV filename
            append: Whether to append to existing file

        Returns:
            Path to the exported CSV file

        Raises:
            CSVExportError: If export fails
        """
        try:
            output_path = self.output_dir / output_filename
            logger.info(f"Exporting combined results to {output_path}")

            # Build rows
            rows: List[Dict[str, Any]] = []

            for i, det_result in enumerate(detection_results):
                # Handle sparse classification/crop results - only images with detections have entries
                # classification_results and crop_results are parallel to detection_results
                # but may be None or shorter if some images had no detections
                classifications = (
                    classification_results[i] if classification_results and i < len(classification_results) else None
                )
                crops = crop_results[i] if crop_results and i < len(crop_results) else None

                for j, detection in enumerate(det_result.detections):
                    classification = (
                        classifications[j] if classifications and j < len(classifications)
                        else None
                    )
                    crop = crops[j] if crops and j < len(crops) else None

                    row = self._combined_to_row(
                        det_result, detection, classification, crop
                    )
                    rows.append(row)

            # Create DataFrame - even if empty, to write headers
            if rows:
                df = pd.DataFrame(rows)
            else:
                # No detections, but create empty DataFrame with expected columns
                logger.warning("No combined results to export, creating empty CSV with headers")
                df = pd.DataFrame(columns=self._get_combined_columns())

            # Export to CSV
            mode = "a" if append and output_path.exists() else "w"
            header = not (append and output_path.exists())

            df.to_csv(
                output_path,
                index=False,
                sep=self.config.delimiter,
                decimal=self.config.decimal_separator,
                mode=mode,
                header=header,
            )

            self.stats["files_exported"] += 1
            self.stats["rows_exported"] += len(rows)

            logger.info(f"Exported {len(rows)} rows to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export combined results: {e}", exc_info=True)
            self.stats["errors"] += 1
            raise CSVExportError(f"Combined export failed: {e}") from e

    def _detection_to_row(
        self, det_result: DetectionResult, detection: Detection
    ) -> Dict[str, Any]:
        """Convert detection to CSV row dictionary."""
        row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime(self.config.timestamp_format),
            "image_path": str(det_result.image_path),
            "image_filename": det_result.image_path.name,
            "yolo_class": detection.class_name,
            "yolo_class_id": detection.class_id,
            "confidence": round(detection.confidence, 4),
        }

        if self.config.include_bbox_details:
            x1, y1, x2, y2 = detection.bbox
            row.update(
                {
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_width": x2 - x1,
                    "bbox_height": y2 - y1,
                    "bbox_area": detection.get_area(),
                }
            )
        else:
            row["bbox"] = str(detection.bbox)

        if self.config.include_metadata:
            row.update(
                {
                    "image_width": det_result.image.shape[1],
                    "image_height": det_result.image.shape[0],
                    "processing_time_sec": det_result.metadata.get(
                        "processing_time_seconds", 0
                    ),
                }
            )

        return row

    def _classification_to_row(
        self,
        det_result: DetectionResult,
        detection: Detection,
        classification: ClassificationResult,
    ) -> Dict[str, Any]:
        """Convert classification to CSV row dictionary."""
        row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime(self.config.timestamp_format),
            "image_path": str(det_result.image_path),
            "image_filename": det_result.image_path.name,
            "yolo_class": detection.class_name,
            "yolo_confidence": round(detection.confidence, 4),
            "species_id": classification.species_id,
            "species_common_name": classification.common_name,
            "species_scientific_name": classification.scientific_name,
            "classification_confidence": round(classification.confidence, 4),
            "classification_method": classification.metadata.get("method", "unknown"),
        }

        if self.config.include_bbox_details:
            x1, y1, x2, y2 = detection.bbox
            row.update(
                {
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                }
            )

        if self.config.include_alternatives and classification.alternative_matches:
            alt_species = [
                alt[0] for alt in classification.alternative_matches[:3]
            ]  # Top 3
            alt_confs = [
                round(alt[1], 4) for alt in classification.alternative_matches[:3]
            ]
            row["alternative_species"] = "|".join(alt_species)
            row["alternative_confidences"] = "|".join(map(str, alt_confs))

        return row

    def _crop_to_row(self, crop_result: CropResult) -> Dict[str, Any]:
        """Convert crop result to CSV row dictionary."""
        width, height = crop_result.get_dimensions()

        row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime(self.config.timestamp_format),
            "crop_path": str(crop_result.output_path)
            if crop_result.output_path
            else "",
            "yolo_class": crop_result.detection.class_name,
            "confidence": round(crop_result.detection.confidence, 4),
            "crop_width": width,
            "crop_height": height,
        }

        if crop_result.classification:
            row.update(
                {
                    "species_id": crop_result.classification.species_id,
                    "species_name": crop_result.classification.common_name,
                }
            )

        if self.config.include_bbox_details:
            x1, y1, x2, y2 = crop_result.original_bbox
            row.update({"bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2})

        return row

    def _combined_to_row(
        self,
        det_result: DetectionResult,
        detection: Detection,
        classification: Optional[ClassificationResult],
        crop: Optional[CropResult],
    ) -> Dict[str, Any]:
        """Convert combined results to CSV row dictionary."""
        row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime(self.config.timestamp_format),
            "image_path": str(det_result.image_path),
            "image_filename": det_result.image_path.name,
            "yolo_class": detection.class_name,
            "yolo_confidence": round(detection.confidence, 4),
            "detection_confidence": round(detection.confidence, 4),  # Alias for backward compatibility
        }

        # Add classification data
        if classification:
            row.update(
                {
                    "species_id": classification.species_id,
                    "species_common_name": classification.common_name,
                    "species_scientific_name": classification.scientific_name,
                    "species_confidence": round(classification.confidence, 4),
                    "classification_method": classification.metadata.get(
                        "method", "unknown"
                    ),
                }
            )
        else:
            row.update(
                {
                    "species_id": "",
                    "species_common_name": "",
                    "species_scientific_name": "",
                    "species_confidence": 0.0,
                    "classification_method": "",
                }
            )

        # Add crop data
        if crop:
            width, height = crop.get_dimensions()
            row.update(
                {
                    "crop_path": str(crop.output_path) if crop.output_path else "",
                    "crop_width": width,
                    "crop_height": height,
                }
            )
        else:
            row.update({"crop_path": "", "crop_width": 0, "crop_height": 0})

        # Add bbox details
        if self.config.include_bbox_details:
            x1, y1, x2, y2 = detection.bbox
            row.update(
                {
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_width": x2 - x1,
                    "bbox_height": y2 - y1,
                }
            )

        # Add image metadata
        if self.config.include_metadata:
            row.update(
                {
                    "image_width": det_result.image.shape[1],
                    "image_height": det_result.image.shape[0],
                }
            )

        return row

    def _get_combined_columns(self) -> List[str]:
        """
        Get the list of expected column names for combined CSV export.

        Returns:
            List of column names in the expected order
        """
        columns = [
            "timestamp",
            "image_path",
            "image_filename",
            "yolo_class",
            "yolo_confidence",
            "detection_confidence",  # Alias for backward compatibility
            "species_id",
            "species_common_name",
            "species_scientific_name",
            "species_confidence",
            "classification_method",
            "crop_path",
            "crop_width",
            "crop_height",
        ]

        if self.config.include_bbox_details:
            columns.extend([
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bbox_width",
                "bbox_height",
            ])

        return columns

    def get_stats(self) -> Dict[str, int]:
        """Get export statistics."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset export statistics."""
        self.stats = {"files_exported": 0, "rows_exported": 0, "errors": 0}
        logger.info("Statistics reset")

    def __repr__(self) -> str:
        """String representation of exporter."""
        return (
            f"CSVExporter(output={self.output_dir}, "
            f"files={self.stats['files_exported']}, "
            f"rows={self.stats['rows_exported']})"
        )


if __name__ == "__main__":
    """Test CSV exporter."""
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing CSV Exporter...\n")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "exports"

        # Create exporter
        config = ExportConfig(
            include_metadata=True, include_bbox_details=True, include_alternatives=True
        )

        exporter = CSVExporter(output_dir=output_dir, config=config)
        print(f"{exporter}\n")

        # Create test data
        from models.yolo_detector import Detection

        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        test_detection = Detection(
            bbox=(100, 100, 300, 250), confidence=0.85, class_id=16, class_name="dog"
        )

        det_result = DetectionResult(
            image_path=Path("test_image.jpg"),
            image=test_image,
            detections=[test_detection],
            metadata={"processing_time_seconds": 1.23},
        )

        # Test detection export
        print("Testing detection export:")
        csv_path = exporter.export_detections([det_result])
        print(f"✅ Exported detections to: {csv_path}")

        # Read and display
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print("  Columns: " + ", ".join(df.columns.tolist()[:5]) + "...")

        # Test classification export
        from core.classification_engine import ClassificationResult

        test_classification = ClassificationResult(
            species_id="coyote",
            common_name="Coyote",
            scientific_name="Canis latrans",
            confidence=0.75,
            yolo_class="dog",
            yolo_confidence=0.85,
            alternative_matches=[("gray_fox", 0.65), ("raccoon", 0.55)],
            metadata={"method": "rule_based"},
        )

        print("\nTesting classification export:")
        csv_path = exporter.export_classifications(
            [det_result], [[test_classification]]
        )
        print(f"✅ Exported classifications to: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

        # Test combined export
        print("\nTesting combined export:")
        csv_path = exporter.export_combined(
            [det_result], [[test_classification]], None
        )
        print(f"✅ Exported combined results to: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print("\nSample row:")
        for col, val in df.iloc[0].items():
            print(f"  {col}: {val}")

        # Show statistics
        print(f"\nStatistics: {exporter.get_statistics()}")

        # List created files
        print(f"\nCreated CSV files in {output_dir}:")
        for file in sorted(output_dir.glob("*.csv")):
            size = file.stat().st_size
            print(f"  {file.name} ({size} bytes)")

    print("\nCSV exporter tests completed!")
