#!/usr/bin/env python3
"""
Train custom California wildlife YOLOv8 model.

This script trains a YOLOv8 model on the California wildlife dataset with
optimized hyperparameters for camera trap imagery.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_california_wildlife_model(
    base_model: str = "yolov8m.pt",
    data_yaml: Path = Path("training_data/yolo_format/california_wildlife.yaml"),
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 16,
    device: str = "0",
    project: Path = Path("runs/train"),
    name: str = "california_wildlife_v1",
    resume: bool = False,
    resume_from: Optional[Path] = None
) -> None:
    """
    Train YOLOv8 model on California wildlife dataset.
    
    Args:
        base_model: Base YOLOv8 model (yolov8n/s/m/l/x.pt)
        data_yaml: Path to dataset configuration file
        epochs: Number of training epochs
        imgsz: Training image size (pixels)
        batch: Batch size (adjust based on GPU memory)
        device: Device to use ('0' for GPU 0, 'cpu' for CPU, '0,1' for multi-GPU)
        project: Project directory for saving results
        name: Experiment name
        resume: Resume from last checkpoint
        resume_from: Resume from specific checkpoint path
    """
    logger.info("=" * 80)
    logger.info("California Wildlife YOLOv8 Training")
    logger.info("=" * 80)
    
    # Validate data file
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    
    # Load model
    if resume_from:
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        model = YOLO(str(resume_from))
    else:
        logger.info(f"Loading base model: {base_model}")
        model = YOLO(base_model)
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Data: {data_yaml}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Batch size: {batch}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output: {project}/{name}")
    
    # Train
    logger.info("Starting training...")
    try:
        results = model.train(
            # Dataset
            data=str(data_yaml),
            
            # Training duration
            epochs=epochs,
            
            # Image settings
            imgsz=imgsz,
            
            # Batch and workers
            batch=batch,
            workers=8,
            
            # Device
            device=device,
            
            # Output
            project=str(project),
            name=name,
            exist_ok=False,
            
            # Transfer learning
            pretrained=True,
            
            # Optimization
            optimizer='AdamW',
            lr0=0.001,  # Initial learning rate
            lrf=0.01,   # Final learning rate (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            
            # Augmentation (wildlife-specific)
            hsv_h=0.015,    # Hue
            hsv_s=0.7,      # Saturation
            hsv_v=0.4,      # Value/brightness
            degrees=15.0,   # Rotation
            translate=0.1,  # Translation
            scale=0.5,      # Scaling
            shear=0.0,      # No shear
            perspective=0.0, # No perspective
            flipud=0.0,     # No vertical flip (animals don't walk upside down)
            fliplr=0.5,     # 50% horizontal flip
            mosaic=1.0,     # Mosaic augmentation
            mixup=0.0,      # No mixup
            
            # Loss weights
            box=7.5,        # Box loss
            cls=0.5,        # Classification loss
            dfl=1.5,        # DFL loss
            
            # Validation
            val=True,
            plots=True,
            
            # Checkpointing
            save=True,
            save_period=10,  # Save every 10 epochs
            
            # Early stopping
            patience=20,     # Stop if no improvement for 20 epochs
            
            # Multi-scale training
            multi_scale=True,
            
            # Other
            verbose=True,
            resume=resume,
        )
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Best model: {project}/{name}/weights/best.pt")
        logger.info(f"Last model: {project}/{name}/weights/last.pt")
        logger.info(f"Results: {project}/{name}/results.csv")
        logger.info(f"Plots: {project}/{name}/*.png")
        
        # Log final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logger.info("\nFinal Metrics:")
            logger.info(f"  mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")
            logger.info(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
            logger.info(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
            logger.info(f"  Recall: {metrics.get('metrics/recall(B)', 0):.3f}")
        
        logger.info("\nNext steps:")
        logger.info("1. Evaluate model: python scripts/evaluate_model.py")
        logger.info("2. Test inference: python scripts/test_inference.py")
        logger.info("3. Integrate into app: Copy best.pt to ~/.game_camera_analyzer/models/")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train custom California wildlife YOLOv8 model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Base YOLOv8 model size"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training_data/yolo_format/california_wildlife.yaml"),
        help="Path to dataset YAML configuration"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Training image size (pixels)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (adjust based on GPU memory)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (0 for GPU 0, cpu for CPU, 0,1 for multi-GPU)"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Project directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="california_wildlife_v1",
        help="Experiment name"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from specific checkpoint"
    )
    
    args = parser.parse_args()
    
    train_california_wildlife_model(
        base_model=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
