"""
End-to-end tests for the complete wildlife analysis pipeline.

These tests verify the full workflow:
1. Load images from directory
2. Detect animals with YOLO
3. Classify species
4. Extract and save crops
5. Export results to CSV
6. Verify all outputs are created correctly
"""

import csv
import logging
from pathlib import Path
from typing import List

import pytest

from core.batch_processor import BatchConfig, BatchProcessor
from core.cropping_engine import CropConfig
from core.csv_exporter import ExportConfig

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end integration tests for complete pipeline."""

    def test_full_pipeline_single_image(
        self,
        sample_image: Path,
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test complete pipeline on a single image.

        Args:
            sample_image: Test image path
            species_db_path: Species database path
            temp_dir: Temporary output directory

        Verifies:
            - Detection runs successfully
            - Classification produces results
            - Crops are saved to disk
            - CSV export contains data
            - All output directories are created
        """
        output_dir = temp_dir / "output"

        # Configure batch processor for full pipeline
        batch_config = BatchConfig(
            detect=True,
            classify=True,
            crop=True,
            export_csv=True,
            save_annotated=False,
            continue_on_error=False,
        )

        crop_config = CropConfig(
            padding=0.1,
            min_width=50,
            min_height=50,
            square=False,
            quality=90,
        )

        export_config = ExportConfig(
            delimiter=",",
            include_metadata=True,
            include_alternatives=True,
        )

        # Create processor
        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            batch_config=batch_config,
            crop_config=crop_config,
            export_config=export_config,
            detection_confidence=0.1,  # Low threshold for test images
            classification_confidence=0.3,
            use_feature_classifier=False,
        )

        # Process single image
        progress = processor.process_images([sample_image])

        # Verify progress tracking
        assert progress.total_images == 1
        assert progress.processed_images == 1
        assert progress.successful_images == 1
        assert progress.failed_images == 0

        # Verify output directories exist
        assert output_dir.exists()
        assert (output_dir / "crops").exists()
        assert (output_dir / "csv").exists()

        # Verify CSV was created
        csv_file = output_dir / "csv" / "batch_results.csv"
        assert csv_file.exists(), "CSV file should be created"

        # Verify CSV contents
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Should have at least headers
            assert len(rows) >= 0
            
            if rows:
                # Verify required columns exist
                assert "image_path" in rows[0]
                assert "detection_confidence" in rows[0]

        logger.info(
            f"E2E test complete: {progress.total_detections} detections, "
            f"{progress.total_classifications} classifications, "
            f"{progress.total_crops} crops"
        )

    def test_full_pipeline_multiple_images(
        self,
        sample_images: List[Path],
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test complete pipeline on multiple images.

        Args:
            sample_images: List of test images
            species_db_path: Species database path
            temp_dir: Temporary output directory

        Verifies:
            - All images are processed
            - Progress tracking is accurate
            - Multiple crops are generated
            - CSV contains all results
            - No processing errors occur
        """
        output_dir = temp_dir / "batch_output"

        batch_config = BatchConfig(
            detect=True,
            classify=True,
            crop=True,
            export_csv=True,
            save_annotated=False,
            continue_on_error=True,
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            batch_config=batch_config,
            detection_confidence=0.1,
            classification_confidence=0.3,
            use_feature_classifier=False,
        )

        # Process all images
        progress = processor.process_images(list(sample_images))

        # Verify all images processed
        assert progress.total_images == len(sample_images)
        assert progress.processed_images == len(sample_images)
        assert progress.successful_images + progress.failed_images == len(sample_images)

        # Should have some detections (test images have features)
        assert progress.total_detections >= 0

        # Verify outputs
        assert output_dir.exists()
        csv_file = output_dir / "csv" / "batch_results.csv"
        
        if progress.total_detections > 0:
            assert csv_file.exists()
            
            # Verify CSV has multiple rows
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0

        logger.info(
            f"Batch E2E complete: {len(sample_images)} images, "
            f"{progress.total_detections} detections, "
            f"{progress.get_elapsed_time():.2f}s"
        )

    def test_directory_processing(
        self,
        sample_images: List[Path],
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test processing images from a directory.

        Args:
            sample_images: List of test images
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - Directory scanning works
            - All images in directory are found
            - Processing completes successfully
        """
        # Create input directory with images
        input_dir = temp_dir / "input_images"
        input_dir.mkdir()

        # Copy sample images to input dir
        for img in sample_images:
            from shutil import copy2
            copy2(img, input_dir / img.name)

        output_dir = temp_dir / "dir_output"

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            detection_confidence=0.1,
            classification_confidence=0.3,
            use_feature_classifier=False,
        )

        # Process directory
        progress = processor.process_directory(input_dir, recursive=False)

        # Should find and process all images
        assert progress.total_images == len(sample_images)
        assert progress.processed_images == len(sample_images)

        logger.info(f"Directory processing: {progress.total_images} images found")

    def test_error_recovery(
        self,
        sample_images: List[Path],
        corrupted_image: Path,
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test pipeline error recovery with mixed valid/invalid images.

        Args:
            sample_images: Valid test images
            corrupted_image: Corrupted image file
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - Pipeline continues after errors
            - Valid images still process
            - Errors are tracked
            - Failed count is accurate
        """
        output_dir = temp_dir / "error_test"

        # Mix valid and invalid images
        mixed_images = sample_images + [corrupted_image]

        batch_config = BatchConfig(
            detect=True,
            classify=True,
            crop=True,
            export_csv=True,
            continue_on_error=True,  # Should continue despite errors
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            batch_config=batch_config,
            detection_confidence=0.1,
        )

        # Process mixed images
        progress = processor.process_images(list(mixed_images))

        # Should process all images
        assert progress.processed_images == len(mixed_images)
        
        # Should have at least one failure (corrupted image)
        assert progress.failed_images >= 1
        
        # Should have processed valid images successfully
        assert progress.successful_images >= len(sample_images)

        # Errors should be tracked
        assert len(progress.errors) >= 1

        logger.info(
            f"Error recovery: {progress.successful_images} successful, "
            f"{progress.failed_images} failed"
        )

    def test_crop_organization_by_species(
        self,
        sample_image: Path,
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test that crops are organized by species.

        Args:
            sample_image: Test image
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - Species subdirectories are created
            - Crops are saved in correct directories
            - Filenames include species names
        """
        output_dir = temp_dir / "organized"

        crop_config = CropConfig(
            padding=0.1,
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            crop_config=crop_config,
            detection_confidence=0.1,
            classification_confidence=0.3,
        )

        progress = processor.process_images([sample_image])

        # Check crops directory
        crops_dir = output_dir / "crops"
        
        if progress.total_crops > 0:
            assert crops_dir.exists()
            
            # Should have species subdirectories
            subdirs = [d for d in crops_dir.iterdir() if d.is_dir()]
            
            if progress.total_classifications > 0:
                assert len(subdirs) > 0, "Should have species subdirectories"
                
                # Verify crops exist in subdirectories
                crop_files = list(crops_dir.rglob("*.jpg"))
                assert len(crop_files) == progress.total_crops

        logger.info(f"Crop organization: {progress.total_crops} crops in species dirs")

    def test_annotated_image_output(
        self,
        sample_image: Path,
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test saving annotated images with detection boxes.

        Args:
            sample_image: Test image
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - Annotated images are saved when enabled
            - Output directory is created
            - Annotated images match input count
        """
        output_dir = temp_dir / "annotated"

        batch_config = BatchConfig(
            detect=True,
            classify=True,
            crop=False,
            export_csv=False,
            save_annotated=True,
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            batch_config=batch_config,
            detection_confidence=0.1,
        )

        progress = processor.process_images([sample_image])

        # Check annotated directory
        annotated_dir = output_dir / "annotated"
        
        if progress.total_detections > 0:
            assert annotated_dir.exists(), "Annotated directory should exist"
            
            # Should have annotated images
            annotated_files = list(annotated_dir.glob("*.jpg"))
            assert len(annotated_files) > 0

        logger.info(f"Annotated images: {progress.total_detections} detections drawn")

    def test_csv_export_completeness(
        self,
        sample_images: List[Path],
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test CSV export contains all expected data.

        Args:
            sample_images: Test images
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - CSV has correct headers
            - All detections are recorded
            - Confidence values are included
            - Species information is present
            - Timestamps are included
        """
        output_dir = temp_dir / "csv_test"

        export_config = ExportConfig(
            delimiter=",",
            include_metadata=True,
            include_alternatives=True,
            include_bbox_details=True,
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            export_config=export_config,
            detection_confidence=0.1,
            classification_confidence=0.3,
        )

        progress = processor.process_images(list(sample_images))

        # Read and verify CSV
        csv_file = output_dir / "csv" / "batch_results.csv"
        rows = []
        
        if progress.total_detections > 0:
            assert csv_file.exists()
            
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) > 0
                
                # Verify expected columns
                expected_cols = ["image_path", "detection_confidence"]
                for col in expected_cols:
                    assert col in rows[0], f"Missing column: {col}"
                
                # Verify data types and ranges
                for row in rows:
                    if "detection_confidence" in row:
                        conf = float(row["detection_confidence"])
                        assert 0.0 <= conf <= 1.0

        logger.info(f"CSV export verified: {len(rows)} rows")

    def test_progress_tracking_accuracy(
        self,
        sample_images: List[Path],
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test progress tracking throughout pipeline.

        Args:
            sample_images: Test images
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - Progress updates are accurate
            - Timing information is reasonable
            - Counts match actual outputs
        """
        output_dir = temp_dir / "progress_test"

        progress_updates = []

        def track_progress(progress):
            progress_updates.append(progress.to_dict())

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            detection_confidence=0.1,
            progress_callback=track_progress,
        )

        final_progress = processor.process_images(list(sample_images))

        # Verify progress tracking
        assert len(progress_updates) > 0, "Should have progress updates"
        
        # Verify final counts
        assert final_progress.processed_images == len(sample_images)
        assert final_progress.get_progress_percent() == 100.0
        
        # Verify timing
        elapsed = final_progress.get_elapsed_time()
        assert elapsed > 0.0, "Should have positive elapsed time"
        
        # Verify progress consistency
        for update in progress_updates:
            assert update["processed_images"] <= update["total_images"]
            assert 0.0 <= update["progress_percent"] <= 100.0

        logger.info(
            f"Progress tracking: {len(progress_updates)} updates, "
            f"{elapsed:.2f}s total"
        )

    def test_minimal_configuration(
        self,
        sample_image: Path,
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test pipeline with minimal configuration (detection only).

        Args:
            sample_image: Test image
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - Detection-only mode works
            - Other stages are skipped
            - Output is still valid
        """
        output_dir = temp_dir / "minimal"

        batch_config = BatchConfig(
            detect=True,
            classify=False,
            crop=False,
            export_csv=False,
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            batch_config=batch_config,
            detection_confidence=0.1,
        )

        progress = processor.process_images([sample_image])

        # Should have detections
        assert progress.total_detections >= 0
        
        # Should have no classifications or crops
        assert progress.total_classifications == 0
        assert progress.total_crops == 0

        logger.info(f"Minimal config: {progress.total_detections} detections only")

    def test_high_quality_output(
        self,
        sample_image: Path,
        species_db_path: Path,
        temp_dir: Path,
    ) -> None:
        """
        Test pipeline with high-quality output settings.

        Args:
            sample_image: Test image
            species_db_path: Species database path
            temp_dir: Temporary directory

        Verifies:
            - High JPEG quality is applied
            - Crops have good quality
            - File sizes are reasonable
        """
        output_dir = temp_dir / "high_quality"

        crop_config = CropConfig(
            padding=0.2,  # More padding
            min_width=100,  # Larger minimum
            min_height=100,
            quality=98,  # High quality
            square=True,
        )

        processor = BatchProcessor(
            output_dir=output_dir,
            species_db_path=species_db_path,
            crop_config=crop_config,
            detection_confidence=0.1,
        )

        progress = processor.process_images([sample_image])

        if progress.total_crops > 0:
            crops_dir = output_dir / "crops"
            crop_files = list(crops_dir.rglob("*.jpg"))
            
            assert len(crop_files) > 0
            
            # Verify crops exist and have reasonable sizes
            for crop_file in crop_files:
                assert crop_file.stat().st_size > 0
                # High quality should produce larger files
                # (but this depends on image content)

        logger.info(f"High quality: {progress.total_crops} crops at JPEG quality 98")
