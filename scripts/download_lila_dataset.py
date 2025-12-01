#!/usr/bin/env python3
"""
Download and prepare LILA Camera Traps dataset for California wildlife training.

This script downloads the LILA dataset from Hugging Face and filters it for
California-specific species to prepare training data for a custom YOLOv8 model.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# California species we want to train on
CALIFORNIA_SPECIES: List[str] = [
    "mule_deer",
    "black_bear", 
    "mountain_lion",
    "coyote",
    "bobcat",
    "gray_fox",
    "raccoon",
    "wild_pig",
    "roosevelt_elk",
    "tule_elk",
    "striped_skunk",
    "western_gray_squirrel",
    "california_ground_squirrel",
    "wild_turkey",
    "california_quail",
    "virginia_opossum",
    "badger",
    "ringtail",
    "long_tailed_weasel",
    "spotted_skunk",
    "desert_bighorn_sheep",
    "pronghorn",
    "black_tailed_jackrabbit",
    "brush_rabbit",
    "desert_cottontail",
    "kit_fox",
    "san_joaquin_kit_fox",
    "river_otter",
    "marten",
    "fisher",
    "california_kangaroo_rat",
    "woodrat",
    "common_raven",
    "roadrunner",
]


def download_lila_dataset(
    output_dir: Path,
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Download LILA Camera Traps dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save processed dataset
        max_samples: Maximum samples to download (None for all)
        cache_dir: Cache directory for Hugging Face datasets
        
    Returns:
        Dictionary with statistics about downloaded data
        
    Raises:
        RuntimeError: If download fails
    """
    logger.info("Starting LILA dataset download...")
    logger.info(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset from Hugging Face
        logger.info("Loading dataset from Hugging Face...")
        dataset = load_dataset(
            "society-ethics/lila_camera_traps",
            cache_dir=str(cache_dir) if cache_dir else None,
            streaming=True if max_samples else False,  # Stream for partial downloads
        )
        
        logger.info(f"Dataset loaded successfully")
        
        # Statistics
        stats: Dict[str, int] = {
            "total_images": 0,
            "california_images": 0,
            "species_counts": {},
        }
        
        # Process dataset
        split = "train" if "train" in dataset else list(dataset.keys())[0]
        logger.info(f"Processing split: {split}")
        
        for idx, sample in enumerate(tqdm(dataset[split], desc="Processing images")):
            if max_samples and idx >= max_samples:
                break
                
            stats["total_images"] += 1
            
            # Extract metadata
            species = sample.get("species", "").lower().replace(" ", "_")
            location = sample.get("location", "").lower()
            
            # Check if California species
            if species in CALIFORNIA_SPECIES or "california" in location:
                stats["california_images"] += 1
                
                # Track species
                if species not in stats["species_counts"]:
                    stats["species_counts"][species] = 0
                stats["species_counts"][species] += 1
        
        # Log statistics
        logger.info(f"\n{'='*60}")
        logger.info("DOWNLOAD STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total images processed: {stats['total_images']}")
        logger.info(f"California-relevant images: {stats['california_images']}")
        logger.info(f"Unique species found: {len(stats['species_counts'])}")
        logger.info(f"\nTop 10 species by count:")
        
        sorted_species = sorted(
            stats['species_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for species, count in sorted_species:
            logger.info(f"  {species}: {count} images")
        
        logger.info(f"{'='*60}\n")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}", exc_info=True)
        raise RuntimeError(f"Dataset download failed: {e}") from e


def main() -> int:
    """Main entry point."""
    # Configuration
    output_dir = Path("training_data/lila")
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    max_samples = 1000  # Start with 1000 samples to explore
    
    logger.info("LILA Dataset Downloader for California Wildlife")
    logger.info(f"Target species: {len(CALIFORNIA_SPECIES)}")
    
    try:
        stats = download_lila_dataset(
            output_dir=output_dir,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )
        
        logger.info("✅ Dataset download completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Dataset download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
