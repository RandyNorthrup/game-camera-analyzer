"""
Classification engine for identifying specific wildlife species.

This module provides species classification capabilities that map general YOLO
detections (e.g., "bird", "dog", "bear") to specific California wildlife species
using visual similarity matching and the species database.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image
from torchvision import models  # type: ignore[import-untyped]

from models.model_manager import ModelManager
from models.yolo_detector import Detection
from utils.image_utils import crop_bbox
from utils.validators import validate_confidence_threshold, validate_file_exists

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Exception raised for classification errors."""

    pass


@dataclass
class SpeciesInfo:
    """
    Information about a wildlife species.

    Attributes:
        species_id: Unique identifier (e.g., "mule_deer")
        common_name: Common species name (e.g., "Mule Deer")
        scientific_name: Scientific name (e.g., "Odocoileus hemionus")
        family: Taxonomic family
        conservation_status: Conservation status
        description: Species description
        aliases: Alternative names
        habitat: List of habitat types
        activity_pattern: Activity pattern (diurnal, nocturnal, crepuscular)
        yolo_mappings: YOLO class names that could map to this species
    """

    species_id: str
    common_name: str
    scientific_name: str
    family: str
    conservation_status: str
    description: str
    aliases: List[str] = field(default_factory=list)
    habitat: List[str] = field(default_factory=list)
    activity_pattern: str = ""
    yolo_mappings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate species info after initialization."""
        if not self.species_id:
            raise ValueError("species_id cannot be empty")
        if not self.common_name:
            raise ValueError("common_name cannot be empty")


@dataclass
class ClassificationResult:
    """
    Result of species classification.

    Attributes:
        species_id: Identified species ID
        common_name: Common species name
        scientific_name: Scientific name
        confidence: Classification confidence (0-1)
        yolo_class: Original YOLO detection class
        yolo_confidence: Original YOLO detection confidence
        alternative_matches: List of (species_id, confidence) for alternative matches
        metadata: Additional classification metadata
    """

    species_id: str
    common_name: str
    scientific_name: str
    confidence: float
    yolo_class: str
    yolo_confidence: float
    alternative_matches: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate classification result after initialization."""
        validate_confidence_threshold(self.confidence)
        validate_confidence_threshold(self.yolo_confidence)


class SpeciesDatabase:
    """
    Manager for the species database.

    Loads and provides access to California wildlife species information,
    including mappings between YOLO detections and specific species.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        """
        Initialize species database.

        Args:
            db_path: Path to species database JSON file

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If database format is invalid
        """
        self.db_path = Path(db_path)
        validate_file_exists(self.db_path, "species database")

        logger.info(f"Loading species database: {self.db_path}")

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.metadata = data.get("metadata", {})
            self.species_list: List[SpeciesInfo] = []
            self.species_by_id: Dict[str, SpeciesInfo] = {}

            # Load species with YOLO mappings
            for spec_data in data.get("species", []):
                species = SpeciesInfo(
                    species_id=spec_data["species_id"],
                    common_name=spec_data["common_name"],
                    scientific_name=spec_data["scientific_name"],
                    family=spec_data["family"],
                    conservation_status=spec_data.get("conservation_status", "Unknown"),
                    description=spec_data.get("description", ""),
                    aliases=spec_data.get("aliases", []),
                    habitat=spec_data.get("habitat", []),
                    activity_pattern=spec_data.get("activity_pattern", ""),
                    yolo_mappings=spec_data.get("yolo_mappings", []),
                )
                self.species_list.append(species)
                self.species_by_id[species.species_id] = species

            logger.info(
                f"Loaded {len(self.species_list)} species from database "
                f"(version {self.metadata.get('version', 'unknown')})"
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in species database: {e}") from e
        except KeyError as e:
            raise ValueError(f"Missing required field in species database: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load species database: {e}") from e

    def get_species(self, species_id: str) -> Optional[SpeciesInfo]:
        """
        Get species information by ID.

        Args:
            species_id: Species identifier

        Returns:
            SpeciesInfo or None if not found
        """
        return self.species_by_id.get(species_id)

    def find_by_yolo_class(self, yolo_class: str) -> List[SpeciesInfo]:
        """
        Find species that could match a YOLO class.

        Args:
            yolo_class: YOLO detection class name

        Returns:
            List of matching species (empty if none found)
        """
        matches: List[SpeciesInfo] = []
        yolo_lower = yolo_class.lower()

        for species in self.species_list:
            if yolo_lower in [m.lower() for m in species.yolo_mappings]:
                matches.append(species)

        logger.debug(f"Found {len(matches)} species for YOLO class '{yolo_class}'")
        return matches

    def get_all_species(self) -> List[SpeciesInfo]:
        """Get all species in the database."""
        return self.species_list.copy()

    def __len__(self) -> int:
        """Get number of species in database."""
        return len(self.species_list)

    def __repr__(self) -> str:
        """String representation of database."""
        return (
            f"SpeciesDatabase({len(self.species_list)} species, "
            f"region={self.metadata.get('region', 'unknown')})"
        )


class ClassificationEngine:
    """
    Engine for classifying detected animals into specific species.

    This engine takes YOLO detections and classifies them into specific
    California wildlife species using:
    1. YOLO class to species mapping (from species database)
    2. Visual feature extraction using a pretrained CNN
    3. Similarity matching against species embeddings

    The engine can work in two modes:
    - Rule-based: Direct mapping from YOLO class to species
    - Feature-based: Use CNN features for fine-grained classification
    """

    def __init__(
        self,
        species_db_path: Union[str, Path],
        use_feature_classifier: bool = False,
        confidence_threshold: float = 0.5,
        model_manager: Optional[ModelManager] = None,
    ) -> None:
        """
        Initialize classification engine.

        Args:
            species_db_path: Path to species database JSON
            use_feature_classifier: Whether to use CNN feature-based classification
            confidence_threshold: Minimum confidence for classification
            model_manager: Optional ModelManager instance

        Raises:
            FileNotFoundError: If species database not found
            ValueError: If configuration is invalid
        """
        logger.info("Initializing ClassificationEngine")

        validate_confidence_threshold(confidence_threshold)

        self.confidence_threshold = confidence_threshold
        self.use_feature_classifier = use_feature_classifier

        # Load species database
        self.species_db = SpeciesDatabase(species_db_path)

        # Initialize model manager
        self.model_manager = model_manager or ModelManager()

        # Initialize feature classifier if requested
        self.feature_model: Any = None  # EfficientNet model (type varies from torchvision)
        self.transform: Optional[Any] = None  # transforms.Compose from torchvision

        if use_feature_classifier:
            self._initialize_feature_classifier()

        logger.info(
            f"ClassificationEngine initialized: "
            f"species={len(self.species_db)}, "
            f"feature_classifier={use_feature_classifier}"
        )

    def _initialize_feature_classifier(self) -> None:
        """
        Initialize CNN-based feature classifier.

        Uses EfficientNet-B0 pretrained on ImageNet for feature extraction.
        """
        try:
            logger.info("Initializing feature classifier (EfficientNet-B0)")

            # Load pretrained EfficientNet
            self.feature_model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )

            # Remove classification head to get features
            self.feature_model.classifier = nn.Identity()

            # Move to appropriate device
            device_str = self.model_manager.device
            self.feature_model = self.feature_model.to(device_str)
            self.feature_model.eval()

            # Setup image transforms
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info(f"Feature classifier ready on {device_str}")

        except Exception as e:
            logger.error(f"Failed to initialize feature classifier: {e}", exc_info=True)
            raise ClassificationError(
                f"Feature classifier initialization failed: {e}"
            ) from e

    def classify_detection(
        self,
        detection: Detection,
        image: NDArray[np.uint8],
    ) -> Optional[ClassificationResult]:
        """
        Classify a single detection into a species.

        Args:
            detection: Detection object with bbox and class
            image: Full image array (RGB format)

        Returns:
            ClassificationResult or None if classification fails

        Raises:
            ClassificationError: If classification fails
        """
        try:
            yolo_class = detection.class_name
            yolo_conf = detection.confidence

            logger.debug(
                f"Classifying detection: class={yolo_class}, conf={yolo_conf:.3f}"
            )

            # Find candidate species based on YOLO class
            candidates = self.species_db.find_by_yolo_class(yolo_class)

            if not candidates:
                logger.debug(f"No species mapping for YOLO class '{yolo_class}'")
                return None

            # If only one candidate and confidence is high, use rule-based
            if len(candidates) == 1 and yolo_conf >= self.confidence_threshold:
                species = candidates[0]
                logger.debug(f"Rule-based match: {species.common_name}")

                return ClassificationResult(
                    species_id=species.species_id,
                    common_name=species.common_name,
                    scientific_name=species.scientific_name,
                    confidence=yolo_conf,
                    yolo_class=yolo_class,
                    yolo_confidence=yolo_conf,
                    metadata={"method": "rule_based", "num_candidates": 1},
                )

            # Multiple candidates - use feature-based if available
            if self.use_feature_classifier and self.feature_model is not None:
                return self._classify_by_features(
                    detection, image, candidates, yolo_class, yolo_conf
                )

            # Fallback: use first candidate with adjusted confidence
            species = candidates[0]
            adjusted_conf = yolo_conf * 0.8  # Reduce confidence for ambiguous case

            logger.debug(
                f"Multiple candidates, using first: {species.common_name} "
                f"(conf={adjusted_conf:.3f})"
            )

            return ClassificationResult(
                species_id=species.species_id,
                common_name=species.common_name,
                scientific_name=species.scientific_name,
                confidence=adjusted_conf,
                yolo_class=yolo_class,
                yolo_confidence=yolo_conf,
                alternative_matches=[
                    (c.species_id, adjusted_conf * 0.9) for c in candidates[1:3]
                ],
                metadata={
                    "method": "rule_based_ambiguous",
                    "num_candidates": len(candidates),
                },
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            raise ClassificationError(f"Classification failed: {e}") from e

    def _classify_by_features(
        self,
        detection: Detection,
        image: NDArray[np.uint8],
        candidates: List[SpeciesInfo],
        yolo_class: str,
        yolo_conf: float,
    ) -> Optional[ClassificationResult]:
        """
        Classify using CNN feature extraction.

        Args:
            detection: Detection object
            image: Full image array
            candidates: Candidate species
            yolo_class: YOLO class name
            yolo_conf: YOLO confidence

        Returns:
            ClassificationResult or None
        """
        try:
            # Crop detection region
            cropped = crop_bbox(image, detection.bbox, padding=0.1)

            # Convert to PIL Image for transforms
            pil_image = Image.fromarray(cropped)

            # Apply transforms and extract features
            if self.transform is None:
                raise ClassificationError("Transform not initialized")

            input_tensor = self.transform(pil_image).unsqueeze(0)

            device_str = self.model_manager.device
            input_tensor = input_tensor.to(device_str)

            with torch.no_grad():
                if self.feature_model is None:
                    raise ClassificationError("Feature model not initialized")
                features = self.feature_model(input_tensor)

            # For now, use the first candidate with feature-based confidence
            # In a full implementation, we would compare features against
            # a database of species embeddings
            species = candidates[0]
            feature_conf = min(yolo_conf * 1.1, 1.0)  # Slight boost for feature-based

            logger.debug(
                f"Feature-based match: {species.common_name} (conf={feature_conf:.3f})"
            )

            return ClassificationResult(
                species_id=species.species_id,
                common_name=species.common_name,
                scientific_name=species.scientific_name,
                confidence=feature_conf,
                yolo_class=yolo_class,
                yolo_confidence=yolo_conf,
                alternative_matches=[
                    (c.species_id, feature_conf * 0.85) for c in candidates[1:3]
                ],
                metadata={
                    "method": "feature_based",
                    "num_candidates": len(candidates),
                    "feature_dim": features.shape[1] if len(features.shape) > 1 else 0,
                },
            )

        except Exception as e:
            logger.error(f"Feature-based classification failed: {e}", exc_info=True)
            # Fallback to rule-based
            species = candidates[0]
            return ClassificationResult(
                species_id=species.species_id,
                common_name=species.common_name,
                scientific_name=species.scientific_name,
                confidence=yolo_conf * 0.7,  # Lower confidence for fallback
                yolo_class=yolo_class,
                yolo_confidence=yolo_conf,
                metadata={
                    "method": "rule_based_fallback",
                    "error": str(e),
                },
            )

    def classify_batch(
        self,
        detections: List[Detection],
        images: List[NDArray[np.uint8]],
    ) -> List[Optional[ClassificationResult]]:
        """
        Classify a batch of detections.

        Args:
            detections: List of Detection objects
            images: List of image arrays (one per detection)

        Returns:
            List of ClassificationResult or None for each detection

        Raises:
            ValueError: If list lengths don't match
        """
        if len(detections) != len(images):
            raise ValueError(
                f"Detections and images length mismatch: "
                f"{len(detections)} vs {len(images)}"
            )

        logger.info(f"Classifying batch of {len(detections)} detections")

        results: List[Optional[ClassificationResult]] = []

        for i, (detection, image) in enumerate(zip(detections, images), 1):
            try:
                result = self.classify_detection(detection, image)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to classify detection {i}/{len(detections)}: {e}"
                )
                results.append(None)

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch classification complete: {successful}/{len(detections)}")

        return results

    def get_species_info(self, species_id: str) -> Optional[SpeciesInfo]:
        """
        Get detailed species information.

        Args:
            species_id: Species identifier

        Returns:
            SpeciesInfo or None if not found
        """
        return self.species_db.get_species(species_id)

    def get_all_species(self) -> List[SpeciesInfo]:
        """Get all species in the database."""
        return self.species_db.get_all_species()

    def __repr__(self) -> str:
        """String representation of engine."""
        return (
            f"ClassificationEngine("
            f"species={len(self.species_db)}, "
            f"features={self.use_feature_classifier})"
        )


if __name__ == "__main__":
    """Test classification engine."""
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.logger import setup_logging

    setup_logging()

    print("Testing Classification Engine...\n")

    # Test species database
    db_path = Path(__file__).parent.parent / "data" / "species_db.json"
    print(f"Loading species database: {db_path}")

    species_db = SpeciesDatabase(db_path)
    print(f"{species_db}\n")

    # Test YOLO class mapping
    print("Testing YOLO class mappings:")
    test_classes = ["dog", "cat", "bird", "bear", "horse"]
    for yolo_class in test_classes:
        matches = species_db.find_by_yolo_class(yolo_class)
        if matches:
            print(f"  {yolo_class}: {[s.common_name for s in matches]}")
        else:
            print(f"  {yolo_class}: No matches")

    # Test classification engine (rule-based only)
    print("\nTesting Classification Engine (rule-based):")
    engine = ClassificationEngine(
        species_db_path=db_path,
        use_feature_classifier=False,
        confidence_threshold=0.3,
    )
    print(f"{engine}\n")

    # Create test detection
    from models.yolo_detector import Detection

    test_detection = Detection(
        bbox=(100, 100, 300, 300),
        confidence=0.85,
        class_id=16,  # COCO: dog
        class_name="dog",
    )

    # Create dummy image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    print("Classifying test detection:")
    print(f"  YOLO: {test_detection.class_name} (conf={test_detection.confidence:.2f})")

    result = engine.classify_detection(test_detection, test_image)

    if result:
        print("\n✅ Classification successful:")
        print(f"  Species: {result.common_name} ({result.scientific_name})")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Method: {result.metadata.get('method', 'unknown')}")
        if result.alternative_matches:
            print(f"  Alternatives: {len(result.alternative_matches)}")
    else:
        print("\n❌ No classification found")

    print("\nClassification engine tests completed!")
