"""Core detection and processing engines."""

from core.classification_engine import (
    ClassificationEngine,
    ClassificationResult,
    SpeciesDatabase,
    SpeciesInfo,
)
from core.detection_engine import DetectionEngine, DetectionResult

__all__ = [
    "DetectionEngine",
    "DetectionResult",
    "ClassificationEngine",
    "ClassificationResult",
    "SpeciesDatabase",
    "SpeciesInfo",
]
