#!/usr/bin/env python3
"""
Evaluate trained California wildlife YOLOv8 model.

This script evaluates model performance on the test set and generates
detailed per-class metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    split: str = "test",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    save_json: bool = True,
    output_dir: Path = Path("evaluation_results")
) -> Dict:
    """
    Evaluate YOLOv8 model on dataset.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset configuration
        split: Dataset split to evaluate ('val' or 'test')
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        save_json: Save results to JSON file
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("=" * 80)
    logger.info("Model Evaluation")
    logger.info("=" * 80)
    
    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Run validation
    logger.info(f"Evaluating on {split} split...")
    results = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,
        save_json=save_json,
    )
    
    # Extract metrics
    metrics = {
        "model": str(model_path.name),
        "dataset": str(data_yaml.name),
        "split": split,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
    }
    
    # Overall metrics
    if hasattr(results, 'box'):
        box_metrics = results.box
        metrics.update({
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
        })
        
        logger.info("\n" + "=" * 80)
        logger.info("Overall Metrics")
        logger.info("=" * 80)
        logger.info(f"mAP50:    {metrics['map50']:.3f}")
        logger.info(f"mAP50-95: {metrics['map50_95']:.3f}")
        logger.info(f"Precision: {metrics['precision']:.3f}")
        logger.info(f"Recall:    {metrics['recall']:.3f}")
    
    # Per-class metrics
    if hasattr(results, 'box') and hasattr(box_metrics, 'ap_class_index'):
        class_names = model.names
        per_class_metrics = []
        
        logger.info("\n" + "=" * 80)
        logger.info("Per-Class Metrics")
        logger.info("=" * 80)
        logger.info(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'mAP50':<12}")
        logger.info("-" * 80)
        
        for i, class_idx in enumerate(box_metrics.ap_class_index):
            class_name = class_names[int(class_idx)]
            class_metrics = {
                "class_id": int(class_idx),
                "class_name": class_name,
                "precision": float(box_metrics.p[i]) if i < len(box_metrics.p) else 0.0,
                "recall": float(box_metrics.r[i]) if i < len(box_metrics.r) else 0.0,
                "map50": float(box_metrics.ap50[i]) if i < len(box_metrics.ap50) else 0.0,
                "map50_95": float(box_metrics.ap[i]) if i < len(box_metrics.ap) else 0.0,
            }
            per_class_metrics.append(class_metrics)
            
            logger.info(
                f"{class_name:<30} "
                f"{class_metrics['precision']:>11.3f} "
                f"{class_metrics['recall']:>11.3f} "
                f"{class_metrics['map50']:>11.3f}"
            )
        
        metrics["per_class"] = per_class_metrics
        
        # Identify best and worst performing classes
        sorted_by_map = sorted(per_class_metrics, key=lambda x: x['map50'], reverse=True)
        
        logger.info("\n" + "=" * 80)
        logger.info("Top 5 Best Performing Classes")
        logger.info("=" * 80)
        for i, cls in enumerate(sorted_by_map[:5], 1):
            logger.info(f"{i}. {cls['class_name']}: mAP50={cls['map50']:.3f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Top 5 Worst Performing Classes")
        logger.info("=" * 80)
        for i, cls in enumerate(sorted_by_map[-5:], 1):
            logger.info(f"{i}. {cls['class_name']}: mAP50={cls['map50']:.3f}")
    
    # Save results
    if save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"evaluation_{model_path.stem}_{split}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\nSaved JSON results: {json_path}")
        
        # Save CSV (per-class metrics)
        if "per_class" in metrics:
            csv_path = output_dir / f"evaluation_{model_path.stem}_{split}_per_class.csv"
            df = pd.DataFrame(metrics["per_class"])
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV results: {csv_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    
    return metrics


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained California wildlife YOLOv8 model"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights (best.pt)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training_data/yolo_format/california_wildlife.yaml"),
        help="Path to dataset YAML configuration"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_json=True,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
