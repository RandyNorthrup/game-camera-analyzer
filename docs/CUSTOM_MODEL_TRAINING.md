# Custom California Wildlife Model Training Guide

Complete guide for training a custom YOLOv8 model specifically for California wildlife detection using the LILA Camera Traps dataset.

## Overview

This guide walks through creating a **species-specific detection model** that can distinguish between individual California wildlife species (e.g., "mule deer" vs "elk" vs "mountain lion") rather than relying on generic COCO classes (cat, dog, horse).

### Why Custom Training?

**Current Limitations (COCO-based YOLOv8):**
- ❌ Only 7 relevant classes (bird, cat, dog, horse, sheep, cow, bear)
- ❌ Cannot distinguish similar species (all deer → "horse")
- ❌ Not trained on wildlife-specific features
- ❌ Poor performance on camera trap conditions

**Benefits of Custom Model:**
- ✅ **35 species-level classes** - Direct classification to species
- ✅ **Wildlife-optimized** - Trained on actual game camera images
- ✅ **California-specific** - Tuned for local species morphology
- ✅ **Better accuracy** - Understands wildlife poses, lighting, partial occlusion
- ✅ **No mapping needed** - Direct species predictions

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5 (8 cores)
- RAM: 16GB
- Storage: 100GB available
- GPU: NVIDIA RTX 3060 (8GB VRAM)

**Recommended:**
- CPU: Intel i7/i9 / AMD Ryzen 7/9
- RAM: 32GB
- Storage: 250GB SSD
- GPU: NVIDIA RTX 3080/4080 (12-16GB VRAM)

**Professional:**
- CPU: AMD Threadripper / Intel Xeon
- RAM: 64GB+
- Storage: 500GB NVMe SSD
- GPU: NVIDIA RTX 4090 / A100 (24GB+ VRAM)

### Software Requirements

```bash
# Python 3.10-3.12
python --version  # Should be 3.10.x, 3.11.x, or 3.12.x

# CUDA Toolkit (if using NVIDIA GPU)
nvidia-smi  # Should show your GPU

# Already installed in project:
- ultralytics
- torch
- opencv-python
- pandas
- numpy

# New dependencies (already installed):
- datasets (Hugging Face)
- roboflow
- supervision
```

## Dataset: LILA Camera Traps

**Source**: [Hugging Face - LILA Camera Traps](https://huggingface.co/datasets/society-ethics/lila_camera_traps)

### Dataset Overview

- **Size**: 5+ million camera trap images
- **Sources**: 20+ wildlife monitoring projects
- **Geographic Coverage**: North America, Europe, Africa, Australia
- **Species**: 500+ animal species
- **Annotations**: Bounding boxes + species labels
- **Quality**: Professional wildlife survey data

### California-Only Filtering Strategy

The download script filters the LILA dataset using two criteria:
1. **Species-based filtering**: Only images labeled with our 35 California target species
2. **Location-based filtering**: Images taken in California or with "california" in metadata

This ensures we train **exclusively on California wildlife** in their natural California habitats, avoiding:
- ❌ African species morphology (different deer, big cats, etc.)
- ❌ Non-California terrain and lighting conditions
- ❌ Species that don't exist in California
- ✅ Pure California data for maximum accuracy

### Target Species (35 Classes)

```python
CALIFORNIA_SPECIES = [
    # Large Mammals
    "mule_deer",
    "roosevelt_elk",
    "tule_elk",
    "wild_pig",
    "desert_bighorn_sheep",
    "pronghorn",
    
    # Predators
    "black_bear",
    "mountain_lion",
    "bobcat",
    "coyote",
    "gray_fox",
    "kit_fox",
    "san_joaquin_kit_fox",
    "badger",
    
    # Medium Mammals
    "raccoon",
    "striped_skunk",
    "spotted_skunk",
    "ringtail",
    "virginia_opossum",
    "river_otter",
    "marten",
    "fisher",
    "long_tailed_weasel",
    
    # Small Mammals
    "western_gray_squirrel",
    "california_ground_squirrel",
    "california_kangaroo_rat",
    "woodrat",
    "black_tailed_jackrabbit",
    "brush_rabbit",
    "desert_cottontail",
    
    # Birds
    "wild_turkey",
    "california_quail",
    "common_raven",
    "roadrunner",
]
```

## Step 1: Dataset Preparation

### 1.1 Download LILA Dataset

```bash
# Activate virtual environment
source venv/bin/activate

# Run download script (starts with 1000 samples to explore)
python scripts/download_lila_dataset.py

# Expected output:
# - Downloads to training_data/lila/
# - Shows statistics for each species
# - Creates manifest file
```

### 1.2 Filter for California Species

The download script automatically filters for:
- Images with California species labels
- Images from California locations
- Images from similar ecosystems

### 1.3 Data Statistics

After download, check coverage:

```bash
python scripts/analyze_dataset.py

# Target minimum per species:
# - Common species (deer, coyote): 2000+ images
# - Medium species (bobcat, fox): 1000+ images  
# - Rare species (fisher, kit fox): 500+ images
```

### 1.4 Address Data Gaps

If species are under-represented:

**Option A: Add More Data Sources**
```bash
# Download from additional sources
python scripts/download_supplementary.py \
    --species "fisher,san_joaquin_kit_fox" \
    --min-images 500
```

**Option B: Data Augmentation**
```python
# Heavy augmentation for rare species
augmentations = [
    "horizontal_flip",
    "rotate_15",
    "brightness_adjust",
    "gaussian_blur",
    "motion_blur",
    "crop_and_pad",
]
```

**Option C: Manual Collection**
```bash
# Request data from wildlife researchers
# California Department of Fish and Wildlife
# University research programs
# Citizen science projects (iNaturalist)
```

## Step 2: Data Annotation & Formatting

### 2.1 Annotation Format

YOLOv8 requires annotations in this format:

```
# Format: class_id x_center y_center width height
# All coordinates normalized to 0-1
0 0.5 0.5 0.3 0.4
```

### 2.2 Convert LILA Format to YOLO

```bash
python scripts/convert_lila_to_yolo.py \
    --input training_data/lila \
    --output training_data/yolo_format \
    --split-ratio 0.8,0.1,0.1  # train/val/test
```

### 2.3 Directory Structure

```
training_data/yolo_format/
├── images/
│   ├── train/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image_001.txt  # YOLO format annotations
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── california_wildlife.yaml  # Dataset configuration
```

### 2.4 Dataset YAML Configuration

```yaml
# california_wildlife.yaml
path: /absolute/path/to/training_data/yolo_format
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 35

# Class names (in order)
names:
  0: mule_deer
  1: black_bear
  2: mountain_lion
  3: coyote
  4: bobcat
  5: gray_fox
  6: raccoon
  7: wild_pig
  8: roosevelt_elk
  9: tule_elk
  10: striped_skunk
  11: western_gray_squirrel
  12: california_ground_squirrel
  13: wild_turkey
  14: california_quail
  15: virginia_opossum
  16: badger
  17: ringtail
  18: long_tailed_weasel
  19: spotted_skunk
  20: desert_bighorn_sheep
  21: pronghorn
  22: black_tailed_jackrabbit
  23: brush_rabbit
  24: desert_cottontail
  25: kit_fox
  26: san_joaquin_kit_fox
  27: river_otter
  28: marten
  29: fisher
  30: california_kangaroo_rat
  31: woodrat
  32: common_raven
  33: roadrunner
  34: unknown
```

## Step 3: Training Configuration

### 3.1 Choose Base Model

Options for transfer learning:

| Model | Parameters | Size | Speed | Accuracy | Use Case |
|-------|-----------|------|-------|----------|----------|
| yolov8n | 3.2M | 6MB | Fastest | Good | Quick experiments |
| yolov8s | 11.2M | 22MB | Fast | Better | Balanced testing |
| yolov8m | 25.9M | 50MB | Medium | Great | **Recommended** |
| yolov8l | 43.7M | 84MB | Slow | Excellent | High accuracy |
| yolov8x | 68.2M | 131MB | Slowest | Best | Maximum accuracy |

**Recommendation**: Start with **yolov8m** for best balance of speed and accuracy.

### 3.2 Training Hyperparameters

Create `training/config.yaml`:

```yaml
# Model
model: yolov8m.pt  # Pretrained COCO weights

# Dataset
data: training_data/yolo_format/california_wildlife.yaml

# Training
epochs: 100
batch: 16  # Adjust based on GPU memory
imgsz: 1280  # Larger images for distant animals
device: 0  # GPU 0, or 'cpu' for CPU training

# Optimization
optimizer: AdamW
lr0: 0.001  # Initial learning rate
lrf: 0.01  # Final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005

# Augmentation (wildlife-specific)
hsv_h: 0.015  # Hue augmentation
hsv_s: 0.7    # Saturation
hsv_v: 0.4    # Value/brightness
degrees: 15.0  # Rotation
translate: 0.1  # Translation
scale: 0.5  # Scaling
shear: 0.0  # Shear
perspective: 0.0  # Perspective
flipud: 0.0  # Flip up-down (no - animals don't walk upside down)
fliplr: 0.5  # Flip left-right (yes - 50% chance)
mosaic: 1.0  # Mosaic augmentation
mixup: 0.0  # Mixup augmentation

# Loss weights
box: 7.5  # Box loss weight
cls: 0.5  # Classification loss weight
dfl: 1.5  # DFL loss weight

# Validation
val: True
plots: True
save: True
save_period: 10  # Save checkpoint every 10 epochs

# Early stopping
patience: 20  # Stop if no improvement for 20 epochs

# Multi-scale training
multi_scale: True

# Other
workers: 8  # Data loading workers
project: runs/train
name: california_wildlife_v1
exist_ok: False
pretrained: True
verbose: True
```

## Step 4: Training Process

### 4.1 Basic Training Script

Create `scripts/train_model.py`:

```python
#!/usr/bin/env python3
"""Train custom California wildlife YOLOv8 model."""

import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_california_wildlife_model():
    """Train YOLOv8 model on California wildlife dataset."""
    
    # Load pretrained model
    logger.info("Loading base model: yolov8m.pt")
    model = YOLO('yolov8m.pt')
    
    # Train
    logger.info("Starting training...")
    results = model.train(
        data='training_data/yolo_format/california_wildlife.yaml',
        epochs=100,
        imgsz=1280,
        batch=16,
        device=0,  # GPU 0
        workers=8,
        project='runs/train',
        name='california_wildlife_v1',
        exist_ok=False,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )
    
    logger.info(f"Training complete! Results: {results}")
    logger.info(f"Best model saved to: runs/train/california_wildlife_v1/weights/best.pt")


if __name__ == "__main__":
    train_california_wildlife_model()
```

### 4.2 Run Training

```bash
# Activate environment
source venv/bin/activate

# Start training (will take 24-72 hours on RTX 3080)
python scripts/train_model.py

# Monitor progress with TensorBoard
tensorboard --logdir runs/train
# Open browser to http://localhost:6006
```

### 4.3 Training Metrics to Watch

Monitor these metrics during training:

**Loss Metrics:**
- `box_loss` - Bounding box regression (target: < 0.05)
- `cls_loss` - Classification loss (target: < 0.5)
- `dfl_loss` - Distribution focal loss (target: < 1.5)

**Performance Metrics:**
- `mAP50` - Mean Average Precision @ IoU 0.5 (target: > 0.7)
- `mAP50-95` - mAP averaged over IoU 0.5-0.95 (target: > 0.5)
- `Precision` - True positives / (True positives + False positives) (target: > 0.75)
- `Recall` - True positives / (True positives + False negatives) (target: > 0.7)

**Training Progress:**
- Losses should steadily decrease
- mAP should steadily increase
- If plateauing, early stopping will trigger after `patience` epochs

### 4.4 Expected Training Time

| GPU | Batch Size | Time per Epoch | 100 Epochs |
|-----|------------|----------------|------------|
| RTX 3060 (8GB) | 8 | 30 min | 50 hours |
| RTX 3080 (10GB) | 16 | 20 min | 33 hours |
| RTX 4090 (24GB) | 32 | 10 min | 17 hours |
| A100 (40GB) | 64 | 5 min | 8 hours |

## Step 5: Model Evaluation

### 5.1 Evaluate on Test Set

```bash
python scripts/evaluate_model.py \
    --model runs/train/california_wildlife_v1/weights/best.pt \
    --data training_data/yolo_format/california_wildlife.yaml
```

### 5.2 Per-Species Performance

Check which species perform well/poorly:

```python
# scripts/analyze_results.py
results = model.val()

# Per-class metrics
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    print(f"  Precision: {results.box.p[i]:.3f}")
    print(f"  Recall: {results.box.r[i]:.3f}")
    print(f"  mAP50: {results.box.ap50[i]:.3f}")
```

### 5.3 Confusion Matrix

Identify commonly confused species:

```python
# Generate confusion matrix
model.val(plots=True)
# See: runs/train/california_wildlife_v1/confusion_matrix.png
```

### 5.4 Visual Inspection

```bash
# Test on sample images
python scripts/test_inference.py \
    --model runs/train/california_wildlife_v1/weights/best.pt \
    --source test_images/ \
    --conf 0.25 \
    --save
```

## Step 6: Model Optimization (Optional)

### 6.1 Hyperparameter Tuning

```python
# Auto-tune hyperparameters
model.tune(
    data='california_wildlife.yaml',
    epochs=30,
    iterations=300,
    optimizer='AdamW',
    plots=True,
    save=True,
)
```

### 6.2 Model Pruning

Reduce model size while maintaining accuracy:

```python
# Prune to 75% of original size
model.prune(amount=0.25)
```

### 6.3 Quantization (INT8)

For faster inference on edge devices:

```bash
# Export to INT8 TensorRT
python scripts/export_model.py \
    --model best.pt \
    --format engine \
    --int8
```

## Step 7: Integration into Application

### 7.1 Copy Trained Model

```bash
# Copy best model to application models directory
cp runs/train/california_wildlife_v1/weights/best.pt \
   ~/.game_camera_analyzer/models/california_wildlife_v1.pt
```

### 7.2 Update Model Manager

Modify `models/model_manager.py`:

```python
def list_available_yolo_models(self) -> List[str]:
    """List available YOLOv8 model variants."""
    return [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "california_wildlife_v1.pt",  # Custom model
    ]
```

### 7.3 Update Species Database

Custom model doesn't need YOLO class mappings:

```python
# No more yolo_mappings needed!
# Model directly outputs species
{
    "species_id": "mule_deer",
    "common_name": "Mule Deer",
    # yolo_mappings field no longer used
}
```

### 7.4 Update Detection Engine

```python
# core/detection_engine.py
# Model automatically uses correct class names
# No mapping layer needed anymore!
```

## Step 8: Continuous Improvement

### 8.1 Collect Real-World Data

Add "Contribute Data" feature to application:

```python
# User can submit:
# - Correctly identified images (reinforcement)
# - Incorrectly identified images (corrections)
# - New species not in training set
```

### 8.2 Active Learning

Prioritize labeling images where model is uncertain:

```python
# Images with confidence 0.4-0.6 need review
if 0.4 < confidence < 0.6:
    flag_for_review(image, species, confidence)
```

### 8.3 Periodic Retraining

Schedule retraining with new data:

```bash
# Every 3-6 months or after 5000 new images
python scripts/retrain_model.py \
    --base-model california_wildlife_v1.pt \
    --new-data training_data/v2/ \
    --output california_wildlife_v2.pt
```

## Expected Results

### Performance Goals

**Detection (mAP50):**
- Large mammals (deer, elk, bear): 85-92%
- Medium mammals (coyote, bobcat, fox): 75-85%
- Small mammals (squirrels, rabbits): 65-75%
- Birds: 70-80%

**Species Classification Accuracy:**
- Easy pairs (bear vs deer): 95%+
- Medium pairs (bobcat vs mountain lion): 80-90%
- Hard pairs (different deer species): 70-80%

### Comparison to COCO Model

| Metric | COCO YOLOv8m | Custom Model | Improvement |
|--------|--------------|--------------|-------------|
| Detection mAP | 72% | 85% | +18% |
| Species ID | N/A | 78% | New capability |
| False positives | 15% | 8% | -47% |
| Inference speed | 150ms | 165ms | -10% (acceptable) |

## Troubleshooting

### Low mAP (<0.5)

**Causes:**
- Insufficient training data
- Poor data quality (blurry, dark images)
- Too aggressive augmentation
- Wrong image size

**Solutions:**
- Add more training images (2000+ per class)
- Filter low-quality images
- Reduce augmentation strength
- Increase image size to 1280 or 1536

### Overfitting

**Symptoms:**
- Training loss low, validation loss high
- mAP on train set > validation set by 15%+

**Solutions:**
- Add more training data
- Increase augmentation
- Add dropout layers
- Reduce model size (use yolov8s instead of yolov8m)
- Early stopping (reduce patience)

### Specific Species Perform Poorly

**Solutions:**
- Collect more data for that species
- Check for annotation errors
- Increase class weight for rare species
- Use focal loss

### Training Takes Too Long

**Solutions:**
- Use smaller model (yolov8n or yolov8s)
- Reduce image size (640 instead of 1280)
- Use fewer workers
- Enable mixed precision training (automatic)

## Cost Estimates

**GPU Cloud Training:**
- AWS g4dn.xlarge (T4): $0.50/hr × 50 hrs = $25
- AWS p3.2xlarge (V100): $3.06/hr × 20 hrs = $61
- Google Cloud A100: $2.95/hr × 10 hrs = $30

**Annotation Services:**
- Professional (Scale AI): $0.10-0.30 per image
- Freelance (Upwork): $0.05-0.15 per image
- Self-annotation: Time cost only

**Storage:**
- Dataset: 50-100GB (~$2-5/month S3)
- Model checkpoints: 20GB (~$0.50/month)

**Total Budget:**
- DIY with own GPU: $0 (electricity only)
- Cloud training only: $30-60
- Full service (data + training): $500-2000

## Next Steps

1. **Run dataset download**: `python scripts/download_lila_dataset.py`
2. **Analyze coverage**: Check species distribution
3. **Convert to YOLO format**: Format annotations
4. **Start training**: Begin with yolov8m for 100 epochs
5. **Evaluate results**: Check mAP and per-species metrics
6. **Integrate into app**: Replace COCO model
7. **Deploy to users**: Update application version
8. **Collect feedback**: Implement data collection feature

## Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [LILA Camera Traps Dataset](https://huggingface.co/datasets/society-ethics/lila_camera_traps)
- [Roboflow Training Guide](https://docs.roboflow.com/)
- [YOLOv8 Training Tips](https://github.com/ultralytics/ultralytics/wiki)

---

**Ready to train? Start with:** `python scripts/download_lila_dataset.py`
