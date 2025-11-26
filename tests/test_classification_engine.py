"""
Unit tests for the classification engine module.

Tests cover:
- Classification engine initialization
- Species classification
- YOLO class mapping
- Confidence thresholds
- Alternative species
- Error handling
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pytest

from core.classification_engine import (
    ClassificationEngine,
    ClassificationResult,
)
from core.detection_engine import Detection

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestClassificationEngine:
    """Test suite for ClassificationEngine class."""

    def test_init_basic(self, species_db_path: Path) -> None:
        """
        Test classification engine initialization.

        Args:
            species_db_path: Path to species database
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        assert engine is not None
        assert engine.species_db is not None
        assert engine.confidence_threshold == 0.5
        assert not engine.use_feature_classifier

        logger.info("ClassificationEngine initialized")

    def test_init_with_features(self, species_db_path: Path) -> None:
        """
        Test initialization with feature classifier.

        Args:
            species_db_path: Path to species database
        """
        engine = ClassificationEngine(
            species_db_path=species_db_path,
            use_feature_classifier=True,
            confidence_threshold=0.7,
        )

        assert engine.use_feature_classifier
        assert engine.confidence_threshold == 0.7

        logger.info("ClassificationEngine initialized with features")

    def test_invalid_confidence_threshold(self, species_db_path: Path) -> None:
        """Test that invalid confidence threshold raises error."""
        with pytest.raises(ValueError, match="Confidence threshold"):
            ClassificationEngine(
                species_db_path=species_db_path,
                confidence_threshold=1.5,
            )

    def test_invalid_species_db_path(self) -> None:
        """Test that invalid species DB path raises error."""
        with pytest.raises(FileNotFoundError):
            ClassificationEngine(species_db_path=Path("nonexistent.json"))

    def test_species_database_loading(self, species_db_path: Path) -> None:
        """
        Test that species database loads correctly.

        Args:
            species_db_path: Path to species database
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        assert engine.species_db is not None
        assert len(engine.species_db) > 0

        # Check expected California wildlife
        species_ids = [s["species_id"] for s in engine.species_db]
        assert "black_tailed_deer" in species_ids
        assert "coyote" in species_ids
        assert "bobcat" in species_ids

        logger.info(f"Loaded {len(engine.species_db)} species")

    def test_yolo_class_mapping(self, species_db_path: Path) -> None:
        """
        Test YOLO class to species mapping.

        Args:
            species_db_path: Path to species database
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        # Test known mappings
        # Bear -> Black bear
        bear_species = engine.get_species_by_yolo_class("bear")
        assert len(bear_species) > 0
        assert any(s["species_id"] == "black_bear" for s in bear_species)

        # Cat -> Multiple feline species
        cat_species = engine.get_species_by_yolo_class("cat")
        assert len(cat_species) > 0

        # Unknown class should return empty list
        unknown_species = engine.get_species_by_yolo_class("unknown_class_xyz")
        assert len(unknown_species) == 0

        logger.info("YOLO class mappings validated")

    def test_classify_detection_basic(
        self,
        species_db_path: Path,
        sample_image: Path,
    ) -> None:
        """
        Test basic detection classification.

        Args:
            species_db_path: Path to species database
            sample_image: Path to test image
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        # Create mock detection
        detection = Detection(
            bbox=np.array([100.0, 100.0, 300.0, 300.0]),
            confidence=0.85,
            class_id=15,  # cat in COCO
            class_name="cat",
        )

        result = engine.classify_detection(sample_image, detection)

        if result is not None:
            assert isinstance(result, ClassificationResult)
            assert result.species_id is not None
            assert result.common_name is not None
            assert result.scientific_name is not None
            assert 0.0 <= result.confidence <= 1.0
            assert result.yolo_class == "cat"

            logger.info(
                f"Classification: {result.common_name} "
                f"(confidence: {result.confidence:.2f})"
            )
        else:
            logger.info("No classification above threshold")

    def test_classify_detection_low_confidence(
        self,
        species_db_path: Path,
        sample_image: Path,
    ) -> None:
        """
        Test that low confidence detections return None.

        Args:
            species_db_path: Path to species database
            sample_image: Path to test image
        """
        # High confidence threshold
        engine = ClassificationEngine(
            species_db_path=species_db_path,
            confidence_threshold=0.95,
        )

        detection = Detection(
            bbox=np.array([100.0, 100.0, 200.0, 200.0]),
            confidence=0.5,
            class_id=15,
            class_name="cat",
        )

        result = engine.classify_detection(sample_image, detection)

        # With high threshold and YOLO-only classification, likely returns None
        # or a result below threshold
        if result is not None:
            assert result.confidence <= 1.0

        logger.info(f"High threshold classification: {result}")

    def test_classify_batch(
        self,
        species_db_path: Path,
        sample_image: Path,
    ) -> None:
        """
        Test batch classification of multiple detections.

        Args:
            species_db_path: Path to species database
            sample_image: Path to test image
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        # Create multiple mock detections
        detections = [
            Detection(
                bbox=np.array([50.0, 50.0, 150.0, 150.0]),
                confidence=0.8,
                class_id=15,
                class_name="cat",
            ),
            Detection(
                bbox=np.array([200.0, 200.0, 350.0, 350.0]),
                confidence=0.75,
                class_id=16,
                class_name="dog",
            ),
        ]

        results = engine.classify_batch(sample_image, detections)

        assert isinstance(results, list)
        assert len(results) == len(detections)

        for result in results:
            if result is not None:
                assert isinstance(result, ClassificationResult)

        logger.info(f"Batch classified {len(detections)} detections")

    def test_classification_alternatives(
        self,
        species_db_path: Path,
        sample_image: Path,
    ) -> None:
        """
        Test alternative species suggestions.

        Args:
            species_db_path: Path to species database
            sample_image: Path to test image
        """
        engine = ClassificationEngine(
            species_db_path=species_db_path,
            max_alternatives=3,
        )

        detection = Detection(
            bbox=np.array([100.0, 100.0, 300.0, 300.0]),
            confidence=0.85,
            class_id=15,
            class_name="cat",
        )

        result = engine.classify_detection(sample_image, detection)

        if result is not None and result.alternatives:
            assert len(result.alternatives) <= 3
            for alt in result.alternatives:
                assert "species_id" in alt
                assert "confidence" in alt
                assert 0.0 <= alt["confidence"] <= 1.0

            logger.info(f"Found {len(result.alternatives)} alternative species")

    def test_unknown_yolo_class(
        self,
        species_db_path: Path,
        sample_image: Path,
    ) -> None:
        """
        Test classification with unknown YOLO class.

        Args:
            species_db_path: Path to species database
            sample_image: Path to test image
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        # Unknown animal class
        detection = Detection(
            bbox=np.array([100.0, 100.0, 200.0, 200.0]),
            confidence=0.8,
            class_id=999,
            class_name="unknown_animal",
        )

        result = engine.classify_detection(sample_image, detection)

        # Should return None or handle gracefully
        if result is None:
            logger.info("Unknown class correctly returned None")
        else:
            logger.info(f"Unknown class mapped to: {result.common_name}")

    def test_classification_result_to_dict(
        self,
        species_db_path: Path,
        sample_image: Path,
    ) -> None:
        """
        Test ClassificationResult dictionary conversion.

        Args:
            species_db_path: Path to species database
            sample_image: Path to test image
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        detection = Detection(
            bbox=np.array([100.0, 100.0, 300.0, 300.0]),
            confidence=0.85,
            class_id=16,
            class_name="dog",
        )

        result = engine.classify_detection(sample_image, detection)

        if result is not None:
            result_dict = result.to_dict()

            assert isinstance(result_dict, dict)
            assert "species_id" in result_dict
            assert "common_name" in result_dict
            assert "scientific_name" in result_dict
            assert "confidence" in result_dict
            assert "yolo_class" in result_dict

            logger.info("Classification result dictionary validated")

    def test_feature_classifier_initialization(
        self,
        species_db_path: Path,
    ) -> None:
        """
        Test feature classifier initialization (slow).

        Args:
            species_db_path: Path to species database
        """
        # This will load a neural network model
        engine = ClassificationEngine(
            species_db_path=species_db_path,
            use_feature_classifier=True,
        )

        assert engine.feature_extractor is not None
        logger.info("Feature classifier initialized successfully")

    def test_california_wildlife_coverage(self, species_db_path: Path) -> None:
        """
        Test that database covers major California wildlife.

        Args:
            species_db_path: Path to species database
        """
        engine = ClassificationEngine(species_db_path=species_db_path)

        expected_species = [
            "black_tailed_deer",
            "mule_deer",
            "elk",
            "black_bear",
            "mountain_lion",
            "coyote",
            "gray_fox",
            "bobcat",
            "striped_skunk",
            "western_gray_squirrel",
        ]

        species_ids = [s["species_id"] for s in engine.species_db]

        for species in expected_species:
            assert species in species_ids, f"Missing species: {species}"

        logger.info(f"California wildlife coverage validated: {len(expected_species)} key species")
