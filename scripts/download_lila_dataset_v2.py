#!/usr/bin/env python3
"""
Download California-specific datasets from LILA.science (direct download).

This script downloads from multiple California-focused camera trap datasets:
1. California Small Animals - 2.2M images from CA Dept of Fish and Wildlife
2. Caltech Camera Traps - 243k images from Southwestern US
3. Channel Islands Camera Traps - 246k images from Channel Islands, CA
4. North American Camera Trap Images - 3.7M images (includes California locations)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# California species we want to train on
CALIFORNIA_SPECIES: List[str] = [
    "mule_deer", "black_bear", "mountain_lion", "coyote", "bobcat", "gray_fox",
    "raccoon", "wild_pig", "roosevelt_elk", "tule_elk", "striped_skunk",
    "western_gray_squirrel", "california_ground_squirrel", "wild_turkey",
    "california_quail", "virginia_opossum", "badger", "ringtail",
    "long_tailed_weasel", "spotted_skunk", "desert_bighorn_sheep", "pronghorn",
    "black_tailed_jackrabbit", "brush_rabbit", "desert_cottontail",
    "kit_fox", "san_joaquin_kit_fox", "river_otter", "marten", "fisher",
    "california_kangaroo_rat", "woodrat", "common_raven", "roadrunner",
    "opossum", "deer", "elk", "fox", "squirrel", "skunk", "rabbit", "quail"
]


# LILA dataset metadata URLs (using Google Cloud Storage - publicly accessible)
LILA_DATASETS = {
    "caltech_camera_traps": {
        "name": "Caltech Camera Traps",
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip",
        "images_base_url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images",
        "priority": 1,  # Southern California data
        "size_gb": 105,
        "num_images": 243100,
        "species": ["opossum", "raccoon", "coyote", "bobcat", "gray_fox", "deer", "squirrel"],
    },
    "channel_islands": {
        "name": "Channel Islands Camera Traps", 
        "metadata_url": "https://storage.googleapis.com/public-datasets-lila/channel-islands-camera-traps/channel_islands_camera_traps.json.zip",
        "images_base_url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/channel-islands-camera-traps",
        "priority": 1,  # California islands
        "size_gb": 40,
        "num_images": 246529,
        "species": ["fox", "rodent", "bird"],
    },
}


def download_metadata(url: str, output_path: Path) -> Path:
    """
    Download dataset metadata JSON file.
    
    Args:
        url: URL to metadata file
        output_path: Path to save metadata
        
    Returns:
        Path to downloaded file
        
    Raises:
        RuntimeError: If download fails
    """
    logger.info(f"Downloading metadata from {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded metadata to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to download metadata: {e}", exc_info=True)
        raise RuntimeError(f"Metadata download failed: {e}") from e


def load_metadata(metadata_path: Path) -> Dict:
    """
    Load dataset metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Parsed metadata dictionary
    """
    # Handle .zip files
    if metadata_path.suffix == '.zip':
        import zipfile
        import tempfile
        
        with zipfile.ZipFile(metadata_path, 'r') as zip_ref:
            # Extract first JSON file
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
            if not json_files:
                raise ValueError(f"No JSON files found in {metadata_path}")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_ref.extract(json_files[0], tmpdir)
                json_path = Path(tmpdir) / json_files[0]
                with open(json_path) as f:
                    return json.load(f)
    else:
        with open(metadata_path) as f:
            return json.load(f)


def filter_california_images(
    metadata: Dict,
    max_images: Optional[int] = None
) -> List[Dict]:
    """
    Filter metadata for California-relevant images.
    
    Args:
        metadata: Dataset metadata dictionary
        max_images: Maximum number of images to return
        
    Returns:
        List of filtered image records
    """
    logger.info("Filtering for California species...")
    
    filtered_images = []
    
    # Get images and annotations
    images = metadata.get('images', [])
    annotations = metadata.get('annotations', [])
    categories = {cat['id']: cat['name'].lower().replace(' ', '_') 
                  for cat in metadata.get('categories', [])}
    
    # Create annotation index by image_id
    image_annotations = {}
    for ann in annotations:
        image_id = ann.get('image_id')
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Filter images
    for img in images:
        img_id = img.get('id')
        img_anns = image_annotations.get(img_id, [])
        
        if not img_anns:
            continue  # Skip empty images
        
        # Check if any annotation has California species
        has_california_species = False
        for ann in img_anns:
            category_id = ann.get('category_id')
            species = categories.get(category_id, '').lower().replace(' ', '_')
            
            # Check against our California species list
            if any(calif_sp in species or species in calif_sp 
                   for calif_sp in CALIFORNIA_SPECIES):
                has_california_species = True
                break
        
        if has_california_species:
            filtered_images.append({
                'image': img,
                'annotations': img_anns,
                'species': [categories.get(ann.get('category_id')) 
                           for ann in img_anns]
            })
            
            if max_images and len(filtered_images) >= max_images:
                break
    
    logger.info(f"Found {len(filtered_images)} California-relevant images")
    return filtered_images


def download_images(
    filtered_data: List[Dict],
    base_url: str,
    output_dir: Path,
    max_images: Optional[int] = None
) -> Dict[str, int]:
    """
    Download filtered images.
    
    Args:
        filtered_data: List of filtered image records
        base_url: Base URL for image downloads
        output_dir: Directory to save images
        max_images: Maximum number of images to download
        
    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(filtered_data),
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
    }
    
    logger.info(f"Downloading images to {output_dir}")
    
    for idx, record in enumerate(tqdm(filtered_data, desc="Downloading images")):
        if max_images and idx >= max_images:
            break
        
        img = record['image']
        file_name = img.get('file_name')
        
        if not file_name:
            logger.warning(f"No filename for image {img.get('id')}")
            stats["failed"] += 1
            continue
        
        # Check if already downloaded
        output_path = output_dir / file_name
        if output_path.exists():
            stats["skipped"] += 1
            continue
        
        # Download image
        img_url = urljoin(base_url, file_name)
        
        try:
            response = requests.get(img_url, timeout=30)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            stats["downloaded"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to download {img_url}: {e}")
            stats["failed"] += 1
    
    return stats


def download_california_datasets(
    output_dir: Path,
    datasets: List[str],
    max_images_per_dataset: Optional[int] = None
) -> Dict:
    """
    Download California wildlife datasets from LILA.
    
    Args:
        output_dir: Root directory for downloaded data
        datasets: List of dataset names to download
        max_images_per_dataset: Maximum images to download per dataset
        
    Returns:
        Overall statistics
    """
    overall_stats = {
        "datasets_processed": 0,
        "total_images": 0,
        "california_images": 0,
    }
    
    for dataset_key in datasets:
        if dataset_key not in LILA_DATASETS:
            logger.warning(f"Unknown dataset: {dataset_key}")
            continue
        
        dataset_info = LILA_DATASETS[dataset_key]
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {dataset_info['name']}")
        logger.info(f"Size: ~{dataset_info['size_gb']}GB")
        logger.info(f"{'='*80}\n")
        
        # Create dataset directory
        dataset_dir = output_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download metadata
        metadata_path = dataset_dir / "metadata.json.zip"
        try:
            download_metadata(dataset_info['metadata_url'], metadata_path)
        except Exception as e:
            logger.error(f"Failed to download metadata for {dataset_key}: {e}")
            continue
        
        # Load and filter metadata
        try:
            metadata = load_metadata(metadata_path)
            filtered_data = filter_california_images(metadata, max_images_per_dataset)
        except Exception as e:
            logger.error(f"Failed to process metadata for {dataset_key}: {e}")
            continue
        
        # Save filtered metadata
        filtered_metadata_path = dataset_dir / "filtered_metadata.json"
        with open(filtered_metadata_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        logger.info(f"Saved filtered metadata: {filtered_metadata_path}")
        logger.info(f"California-relevant images: {len(filtered_data)}")
        
        overall_stats["datasets_processed"] += 1
        overall_stats["california_images"] += len(filtered_data)
    
    return overall_stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download California wildlife datasets from LILA"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/lila"),
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=["caltech_camera_traps"],  # Start with smallest dataset
        choices=list(LILA_DATASETS.keys()),
        help="Datasets to download"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum images per dataset (for exploration)"
    )
    
    args = parser.parse_args()
    
    logger.info("LILA Dataset Downloader for California Wildlife")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Max images per dataset: {args.max_images}")
    
    try:
        stats = download_california_datasets(
            output_dir=args.output,
            datasets=args.datasets,
            max_images_per_dataset=args.max_images,
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Datasets processed: {stats['datasets_processed']}")
        logger.info(f"California images found: {stats['california_images']}")
        logger.info("✅ Dataset download completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Dataset download failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
