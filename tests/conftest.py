"""
Pytest configuration and shared fixtures for test suite.

This module provides:
- Test fixtures for images, models, and configurations
- Temporary directory management
- Mock data generation
- Common test utilities
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest
from PIL import Image

from config import ConfigManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Get test data directory path.

    Returns:
        Path to test_data directory
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Create temporary directory for test outputs.

    Yields:
        Path to temporary directory

    Cleanup:
        Removes directory after test
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def test_config(temp_dir: Path) -> ConfigManager:
    """
    Create test configuration manager.

    Args:
        temp_dir: Temporary directory for outputs

    Returns:
        ConfigManager instance with test settings
    """
    config = ConfigManager()

    # Override with test-specific values
    config.set_value("output.base_dir", str(temp_dir / "output"))
    config.set_value("detection.confidence_threshold", 0.25)
    config.set_value("detection.iou_threshold", 0.45)
    config.set_value("classification.threshold", 0.5)
    config.set_value("classification.use_feature_classifier", False)
    config.set_value("cropping.padding", 0.1)
    config.set_value("cropping.jpeg_quality", 95)
    config.set_value("output.export_csv", True)

    return config


@pytest.fixture(scope="function")
def sample_image(temp_dir: Path) -> Path:
    """
    Create sample test image with animal-like features.

    Args:
        temp_dir: Temporary directory for saving

    Returns:
        Path to saved test image
    """
    # Create 640x480 RGB image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some "object-like" features (bright rectangles simulating animals)
    # Deer-sized object
    img_array[100:200, 200:350] = [139, 69, 19]  # Brown color

    # Smaller animal
    img_array[300:350, 400:480] = [128, 128, 128]  # Gray color

    # Convert to PIL Image
    img = Image.fromarray(img_array, mode="RGB")

    # Save to temp directory
    img_path = temp_dir / "sample_image.jpg"
    img.save(img_path, quality=95)

    logger.debug(f"Created sample image: {img_path}")
    return img_path


@pytest.fixture(scope="function")
def sample_images(temp_dir: Path) -> List[Path]:
    """
    Create multiple sample test images.

    Args:
        temp_dir: Temporary directory for saving

    Returns:
        List of paths to test images
    """
    images = []

    for i in range(3):
        # Create varied images
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Add different "animal" features to each
        if i == 0:
            # Large animal
            img_array[50:250, 100:400] = [139, 69, 19]
        elif i == 1:
            # Two smaller animals
            img_array[100:180, 150:250] = [128, 128, 128]
            img_array[250:330, 350:450] = [160, 82, 45]
        else:
            # One small animal
            img_array[200:280, 300:400] = [105, 105, 105]

        img = Image.fromarray(img_array, mode="RGB")
        img_path = temp_dir / f"test_image_{i:02d}.jpg"
        img.save(img_path, quality=95)
        images.append(img_path)

    logger.debug(f"Created {len(images)} sample images")
    return images


@pytest.fixture(scope="function")
def empty_image(temp_dir: Path) -> Path:
    """
    Create image with no animal-like features (blank).

    Args:
        temp_dir: Temporary directory for saving

    Returns:
        Path to blank image
    """
    # Create solid color image (no detections expected)
    img_array = np.full((480, 640, 3), 100, dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    img_path = temp_dir / "empty_image.jpg"
    img.save(img_path, quality=95)

    logger.debug(f"Created empty image: {img_path}")
    return img_path


@pytest.fixture(scope="function")
def species_db_path() -> Path:
    """
    Get path to species database.

    Returns:
        Path to species_db.json
    """
    return Path("data/species_db.json")


@pytest.fixture(scope="session")
def yolo_model_path() -> Path:
    """
    Get path to YOLO model file.

    Returns:
        Path to yolov8n.pt (or other available model)
    """
    # Check for available models
    models = [
        Path("yolov8n.pt"),
        Path("models/yolov8n.pt"),
    ]

    for model_path in models:
        if model_path.exists():
            return model_path

    # If no model found, download will happen automatically by Ultralytics
    return Path("yolov8n.pt")


@pytest.fixture(scope="function")
def mock_detection_result() -> Dict[str, Any]:
    """
    Create mock detection result for testing.

    Returns:
        Dictionary with detection data
    """
    return {
        "image_path": "test_image.jpg",
        "image_size": (640, 480),
        "num_detections": 2,
        "boxes": [
            [100.0, 50.0, 300.0, 250.0],  # x1, y1, x2, y2
            [350.0, 200.0, 500.0, 350.0],
        ],
        "confidences": [0.85, 0.72],
        "classes": [15, 15],  # YOLO 'cat' class
        "class_names": ["cat", "cat"],
    }


@pytest.fixture(scope="function")
def mock_classification_results() -> List[Dict[str, Any]]:
    """
    Create mock classification results for testing.

    Returns:
        List of classification result dictionaries
    """
    return [
        {
            "species_id": "bobcat",
            "common_name": "Bobcat",
            "scientific_name": "Lynx rufus",
            "confidence": 0.78,
            "yolo_class": "cat",
            "alternatives": [
                {"species_id": "mountain_lion", "confidence": 0.15},
            ],
        },
        {
            "species_id": "domestic_cat",
            "common_name": "Domestic Cat",
            "scientific_name": "Felis catus",
            "confidence": 0.62,
            "yolo_class": "cat",
            "alternatives": [],
        },
    ]


@pytest.fixture(scope="function")
def mock_crop_results() -> List[Dict[str, Any]]:
    """
    Create mock crop results for testing.

    Returns:
        List of crop result dictionaries
    """
    return [
        {
            "crop_path": "output/crops/bobcat/test_image_00_bobcat_001.jpg",
            "bbox": [100, 50, 300, 250],
            "crop_size": (220, 220),
            "quality_score": 0.92,
            "species": "bobcat",
        },
        {
            "crop_path": "output/crops/domestic_cat/test_image_00_domestic_cat_001.jpg",
            "bbox": [350, 200, 500, 350],
            "crop_size": (165, 165),
            "quality_score": 0.85,
            "species": "domestic_cat",
        },
    ]


@pytest.fixture(scope="function")
def invalid_image_path(temp_dir: Path) -> Path:
    """
    Get path to non-existent image file.

    Args:
        temp_dir: Temporary directory

    Returns:
        Path that doesn't exist
    """
    return temp_dir / "nonexistent_image.jpg"


@pytest.fixture(scope="function")
def corrupted_image(temp_dir: Path) -> Path:
    """
    Create corrupted image file for error testing.

    Args:
        temp_dir: Temporary directory

    Returns:
        Path to corrupted file
    """
    corrupt_path = temp_dir / "corrupted.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"This is not a valid image file")

    logger.debug(f"Created corrupted image: {corrupt_path}")
    return corrupt_path


# Pytest configuration
def pytest_configure(config: Any) -> None:
    """
    Configure pytest with custom markers and settings.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gui: marks tests as GUI tests")


# Enable verbose logging in tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
