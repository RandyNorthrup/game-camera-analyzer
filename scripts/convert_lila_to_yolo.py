#!/usr/bin/env python3
"""
Convert LILA Camera Traps dataset to YOLO format.

This script:
1. Reads LILA dataset from HuggingFace format
2. Converts bounding box annotations to YOLO format (normalized x_center, y_center, width, height)
3. Creates train/val/test splits
4. Generates dataset.yaml for YOLOv8 training
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# California species (matching download script)
CALIFORNIA_SPECIES = [
    "mule_deer", "roosevelt_elk", "tule_elk", "wild_pig", "desert_bighorn_sheep", "pronghorn",
    "black_bear", "mountain_lion", "bobcat", "coyote", "gray_fox", "kit_fox",
    "san_joaquin_kit_fox", "badger", "raccoon", "striped_skunk", "spotted_skunk",
    "ringtail", "virginia_opossum", "river_otter", "marten", "fisher", "long_tailed_weasel",
    "western_gray_squirrel", "california_ground_squirrel", "california_kangaroo_rat",
    "woodrat", "black_tailed_jackrabbit", "brush_rabbit", "desert_cottontail",
    "wild_turkey", "california_quail", "common_raven", "roadrunner", "unknown"
]


def convert_bbox_to_yolo(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x_min, y_min, width, height] to YOLO format.
    
    YOLO format: [x_center, y_center, width, height] (all normalized 0-1)
    
    Args:
        bbox: Bounding box as [x_min, y_min, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to 0-1
    """
    x_min, y_min, box_width, box_height = bbox
    
    # Calculate center
    x_center = x_min + (box_width / 2)
    y_center = y_min + (box_height / 2)
    
    # Normalize to 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_width / img_width
    height_norm = box_height / img_height
    
    # Clamp to valid range
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def create_yolo_annotation(
    image_path: Path,
    annotations: List[Dict],
    species_to_id: Dict[str, int],
    output_path: Path
) -> bool:
    """
    Create YOLO format annotation file for an image.
    
    Args:
        image_path: Path to image file
        annotations: List of annotation dicts with 'bbox' and 'species' keys
        species_to_id: Mapping from species name to class ID
        output_path: Path to save YOLO annotation file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Convert each annotation
        yolo_lines = []
        for ann in annotations:
            species = ann.get('species', 'unknown')
            bbox = ann.get('bbox')
            
            if bbox is None or len(bbox) != 4:
                logger.warning(f"Invalid bbox for {image_path}: {bbox}")
                continue
            
            # Get class ID
            class_id = species_to_id.get(species, species_to_id.get('unknown', 34))
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox, img_width, img_height
            )
            
            # Create YOLO line: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        if not yolo_lines:
            logger.warning(f"No valid annotations for {image_path}")
            return False
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return False


def split_dataset(
    image_paths: List[Path],
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        image_paths: List of image file paths
        split_ratios: Tuple of (train, val, test) ratios (must sum to 1.0)
        
    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    assert abs(sum(split_ratios) - 1.0) < 0.001, "Split ratios must sum to 1.0"
    
    # Shuffle
    np.random.seed(42)  # For reproducibility
    shuffled_paths = np.random.permutation(image_paths)
    
    # Calculate split indices
    n_total = len(shuffled_paths)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    
    # Split
    train_paths = shuffled_paths[:n_train].tolist()
    val_paths = shuffled_paths[n_train:n_train + n_val].tolist()
    test_paths = shuffled_paths[n_train + n_val:].tolist()
    
    logger.info(f"Split dataset: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    return train_paths, val_paths, test_paths


def create_dataset_yaml(
    output_dir: Path,
    species_list: List[str]
) -> None:
    """
    Create dataset.yaml configuration file for YOLOv8.
    
    Args:
        output_dir: Root directory of YOLO format dataset
        species_list: List of species names in class order
    """
    yaml_content = f"""# California Wildlife Dataset Configuration
# Auto-generated by convert_lila_to_yolo.py

# Dataset root (absolute path)
path: {output_dir.absolute()}

# Splits
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(species_list)}

# Class names (in order)
names:
"""
    
    for i, species in enumerate(species_list):
        yaml_content += f"  {i}: {species}\n"
    
    yaml_path = output_dir / "california_wildlife.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created dataset configuration: {yaml_path}")


def convert_lila_to_yolo(
    input_dir: Path,
    output_dir: Path,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> None:
    """
    Convert LILA dataset to YOLO format.
    
    Args:
        input_dir: Directory containing LILA dataset
        output_dir: Directory to save YOLO format dataset
        split_ratios: Train/val/test split ratios
    """
    logger.info(f"Converting LILA dataset from {input_dir} to YOLO format at {output_dir}")
    
    # Create species to ID mapping
    species_to_id = {species: i for i, species in enumerate(CALIFORNIA_SPECIES)}
    
    # Find all images and annotations
    # Assuming structure: input_dir/images/*.jpg and input_dir/annotations.json
    images_dir = input_dir / "images"
    annotations_file = input_dir / "annotations.json"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Load annotations
    import json
    with open(annotations_file) as f:
        annotations_data = json.load(f)
    
    # Group annotations by image
    image_annotations = {}
    for ann in annotations_data.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Get all image paths
    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images")
    
    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(image_paths, split_ratios)
    
    # Process each split
    for split_name, split_paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths)
    ]:
        logger.info(f"Processing {split_name} split ({len(split_paths)} images)...")
        
        images_output_dir = output_dir / "images" / split_name
        labels_output_dir = output_dir / "labels" / split_name
        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        for img_path in tqdm(split_paths, desc=f"Converting {split_name}"):
            # Get annotations for this image
            image_id = img_path.stem
            annotations = image_annotations.get(image_id, [])
            
            if not annotations:
                logger.warning(f"No annotations for {img_path}")
                continue
            
            # Copy image
            output_img_path = images_output_dir / img_path.name
            shutil.copy2(img_path, output_img_path)
            
            # Create YOLO annotation
            output_label_path = labels_output_dir / f"{img_path.stem}.txt"
            if create_yolo_annotation(img_path, annotations, species_to_id, output_label_path):
                success_count += 1
        
        logger.info(f"Successfully converted {success_count}/{len(split_paths)} images in {split_name}")
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir, CALIFORNIA_SPECIES)
    
    logger.info("Conversion complete!")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert LILA Camera Traps dataset to YOLO format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing LILA dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for YOLO format dataset"
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test split ratios (comma-separated, must sum to 1.0)"
    )
    
    args = parser.parse_args()
    
    # Parse split ratios
    split_ratios = tuple(map(float, args.split_ratio.split(',')))
    if len(split_ratios) != 3:
        raise ValueError("Split ratio must have exactly 3 values (train,val,test)")
    if abs(sum(split_ratios) - 1.0) >= 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    
    # Run conversion
    convert_lila_to_yolo(args.input, args.output, split_ratios)


if __name__ == "__main__":
    main()
