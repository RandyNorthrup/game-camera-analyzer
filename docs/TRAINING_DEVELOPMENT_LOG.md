# California Wildlife Model Training - Development Log

**Project**: Custom YOLOv8 model for California wildlife species detection  
**Started**: December 1, 2025  
**Status**: 🔄 In Progress - Infrastructure Setup

---

## 📋 Current Phase: Infrastructure & Data Preparation

### ✅ Completed Tasks

#### 1. Training Infrastructure Setup
- [x] Created directory structure (`training_data/`, `docs/`)
- [x] Installed required dependencies:
  - `datasets==4.4.1` (Hugging Face dataset loading)
  - `huggingface_hub==1.1.5` (dataset access)
  - `roboflow==1.2.11` (annotation management)
  - `supervision==0.27.0` (CV utilities)
  - `opencv-python-headless==4.10.0.84`
  - `pyarrow==22.0.0` (efficient data handling)

#### 2. Documentation Created
- [x] **CUSTOM_MODEL_TRAINING.md** - Complete 600+ line training guide covering:
  - Hardware requirements and cost estimates
  - Dataset preparation workflow
  - Training configuration and hyperparameters
  - Evaluation metrics and benchmarks
  - Integration into application
  - Troubleshooting guide
  - California-only data filtering strategy

#### 3. Scripts Created
- [x] **download_lila_dataset.py** (122 lines)
  - Downloads LILA Camera Traps dataset from Hugging Face
  - Filters for 34 California species
  - Location-based filtering (California only)
  - Progress tracking and statistics
  - Configurable sampling (default: 1000 images)

- [x] **convert_lila_to_yolo.py** (233 lines)
  - Converts LILA annotations to YOLO format
  - Bbox conversion: [x_min, y_min, w, h] → [x_center, y_center, w, h] (normalized)
  - Creates train/val/test splits (80/10/10)
  - Generates `california_wildlife.yaml` for YOLOv8
  - Validates image dimensions and annotations

- [x] **train_model.py** (175 lines)
  - YOLOv8 training wrapper with California-optimized settings
  - Configurable base model (n/s/m/l/x)
  - Wildlife-specific augmentations
  - Checkpoint saving every 10 epochs
  - Early stopping (patience=20)
  - Multi-scale training support

- [x] **evaluate_model.py** (195 lines)
  - Comprehensive model evaluation on test set
  - Overall metrics (mAP50, mAP50-95, precision, recall)
  - Per-class performance breakdown
  - Identifies best/worst performing species
  - Exports JSON and CSV results
  - Generates confusion matrix

#### 4. Design Decisions
- [x] **California-only data**: Rejected non-California datasets (Snapshot Serengeti, etc.)
  - Reason: Different species morphology, terrain, lighting conditions
  - Strategy: Species + location filtering on LILA dataset
  
- [x] **YOLOv8m as recommended base model**
  - Reason: Best balance of accuracy (mAP ~0.85 target) vs speed (165ms)
  - Alternative: yolov8s for faster experiments, yolov8x for maximum accuracy

- [x] **Image size 1280px**
  - Reason: Camera traps often have distant animals, need detail for species distinction
  - Trade-off: Slower training but better small animal detection

- [x] **Direct LILA.science downloads**
  - Reason: Hugging Face dataset uses deprecated loading scripts
  - Solution: Use Google Cloud Storage URLs for metadata + images
  - Benefit: Publicly accessible, no authentication required

#### 5. Dataset Sources Documentation
- [x] **Created DATASET_SOURCES.md** - Comprehensive shareable reference
  - 3 California-specific datasets with download links
  - Multiple download methods (gsutil, azcopy, wget, our script)
  - Training strategy (phased approach)
  - All 35 target species listed
  - License information (CDLA Permissive)

---

## 🎯 Current Objectives

### Immediate Next Steps (Today)
1. ✅ **Test dataset download script**
   - Fixed Hugging Face dataset loading issue (deprecated script format)
   - Found direct LILA.science download URLs (Google Cloud Storage)
   - Created `download_lila_dataset_v2.py` with correct URLs
   - Successfully tested with Caltech dataset metadata download

2. **Download and analyze Caltech metadata**
   ```bash
   python scripts/download_lila_dataset_v2.py --datasets caltech_camera_traps --max-images 100
   ```
   - Parse metadata JSON to understand structure
   - Identify California species coverage
   - Check annotation quality

3. **Create data conversion pipeline**
   - Test YOLO format conversion on sample data
   - Validate bounding box transformations
   - Generate dataset.yaml configuration

### Short-term Goals (This Week)
- [ ] Download full California subset from LILA (~10k-50k images estimated)
- [ ] Convert all data to YOLO format
- [ ] Run initial training experiment (yolov8n, 10 epochs) to validate pipeline
- [ ] Document any issues or needed adjustments

### Medium-term Goals (Next 2 Weeks)
- [ ] Full training run (yolov8m, 100 epochs)
- [ ] Model evaluation and per-species analysis
- [ ] Identify commonly confused species pairs
- [ ] Hyperparameter tuning if needed
- [ ] Integration testing with main application

---

## 📊 Dataset Status

### Caltech Camera Traps - Initial Analysis ✅

**Status**: Metadata downloaded and filtered (Dec 1, 2025)

**Sample Analysis (100 images)**:
- ✅ Successfully downloaded 9MB metadata
- ✅ Filtered for California species
- ✅ Found species distribution:
  - Opossum: 20 images
  - Rabbit: 17 images  
  - Deer: 16 images
  - Coyote: 13 images
  - Raccoon: 12 images
  - Cat: 11 images
  - Bobcat: 6 images
  - Squirrel: 2 images
  - Skunk: 1 image
  - Fox: 1 image

**Image Specifications**:
- Resolution: 2048×1494px (high quality)
- Format: JPEG
- Bounding boxes: COCO format annotations included

**Next Steps**: Scale up to full dataset (243k images)

### Data Sources
- **Primary**: LILA Camera Traps (Hugging Face: `society-ethics/lila_camera_traps`)
- **Backup options** (if gaps exist):
  - California Department of Fish and Wildlife archives
  - iNaturalist California observations
  - University research datasets (UC Davis, SDSU)

---

## 🔧 Technical Configuration

### Training Environment
- **Hardware**: TBD (need to verify GPU availability)
- **Python**: 3.12.12
- **Virtual Environment**: `/Users/user/game-camera-analyzer/venv`
- **CUDA**: Status unknown (check with `nvidia-smi`)

### Model Configuration
```yaml
Base Model: yolov8m.pt (pretrained COCO weights)
Image Size: 1280px
Batch Size: 16 (adjust based on GPU memory)
Epochs: 100
Optimizer: AdamW
Learning Rate: 0.001 → 0.00001 (cosine decay)
Early Stopping: 20 epochs patience
```

### Augmentation Pipeline
```yaml
Horizontal Flip: 50%
Rotation: ±15°
Translation: ±10%
Scale: 0.5-1.5x
HSV: h=0.015, s=0.7, v=0.4
Mosaic: 100%
Mixup: 0% (disabled for wildlife)
Vertical Flip: 0% (animals don't walk upside down)
```

---

## ⚠️ Issues & Blockers

### Current Issues

#### 🔴 BLOCKER: Hugging Face LILA Dataset Access Failed
**Date**: December 1, 2025 12:50 PM  
**Error**: `RuntimeError: Dataset scripts are no longer supported, but found lila_camera_traps.py`

**Root Cause**: The `society-ethics/lila_camera_traps` dataset on Hugging Face uses deprecated dataset loading scripts that are no longer supported by the `datasets` library (v4.4.1+).

**Impact**: Cannot download training data via Hugging Face

**Investigation**:
- Hugging Face deprecated custom dataset scripts
- Need to find alternative download method
- Options: Direct LILA.science download, pre-processed datasets, or different source

**Solution Found**:
- LILA.science hosts direct downloads with metadata JSON files
- Identified 3 California-specific datasets:
  1. **California Small Animals** (2.2M images) - CA Dept of Fish and Wildlife
  2. **Caltech Camera Traps** (243k images) - Southern California
  3. **Channel Islands Camera Traps** (246k images) - CA islands
- Created new download script: `download_lila_dataset_v2.py`

**Next Action**: Test new direct download approach with Caltech dataset (smallest at 18GB)

**Status Update**: ✅ Script updated with correct Google Cloud Storage URLs  
**Documentation Created**: `docs/DATASET_SOURCES.md` - Shareable reference with all download links

**Metadata Analysis Results**:
- Successfully downloaded Caltech metadata (9MB)
- Filtered 100 sample images showing good California species coverage
- Found 10 species in sample: opossum, rabbit, deer, coyote, raccoon, cat, bobcat, squirrel, skunk, fox
- Image quality confirmed: 2048×1494px with COCO format bounding boxes

### Potential Risks
1. **Insufficient data for rare species**
   - Impact: Poor performance on fisher, kit fox, etc.
   - Mitigation: Data augmentation + supplementary data collection

2. **LILA dataset access/download failures**
   - Impact: Cannot proceed with training
   - Mitigation: Alternative data sources prepared

3. **GPU memory constraints**
   - Impact: Reduced batch size → longer training
   - Mitigation: Use smaller model or reduce image size

4. **Annotation quality issues**
   - Impact: Model learns incorrect labels
   - Mitigation: Manual validation of sample annotations

---

## 📈 Success Metrics

### Model Performance Targets

**Detection (mAP50)**:
- Large mammals (deer, elk, bear): **85-92%**
- Medium mammals (coyote, bobcat, fox): **75-85%**
- Small mammals (squirrels, rabbits): **65-75%**
- Birds: **70-80%**

**Species Classification Accuracy**:
- Easy pairs (bear vs deer): **95%+**
- Medium pairs (bobcat vs mountain lion): **80-90%**
- Hard pairs (mule deer vs elk): **70-80%**

**Overall Goals**:
- mAP50: **>75%** (minimum acceptable)
- mAP50-95: **>55%**
- Precision: **>75%**
- Recall: **>70%**

**Comparison to Current COCO Model**:
- Detection improvement: **+15-20%**
- Species ID capability: **New (N/A → 78%)**
- False positive reduction: **-40-50%**

---

## 💡 Lessons Learned

### 1. California-Only Data Philosophy
- **Learning**: Using non-California data (African datasets) would introduce distribution shift
- **Decision**: Strict filtering for California species + California locations only
- **Impact**: Better generalization to target deployment environment

### 2. Data Requirements
- **Learning**: 35 classes require substantial data (ideally 1000+ per class)
- **Challenge**: Some California species are rare (fisher, kit fox)
- **Strategy**: Combine augmentation + active learning + user data collection post-deployment

---

## 🔜 Next Session Plan

**Priority 1**: Run dataset download and analyze results
```bash
source venv/bin/activate
python scripts/download_lila_dataset.py
```

**Priority 2**: Make decision on data sufficiency
- If sufficient (>500 per species): Proceed with full download
- If insufficient: Plan supplementary data collection

**Priority 3**: Begin data conversion
```bash
python scripts/convert_lila_to_yolo.py \
    --input training_data/lila \
    --output training_data/yolo_format \
    --split-ratio 0.8,0.1,0.1
```

**Priority 4**: Quick validation training run
```bash
python scripts/train_model.py \
    --model yolov8n.pt \
    --epochs 10 \
    --batch 32 \
    --name validation_run
```

---

## 📝 Notes & Observations

- Current COCO model limitations well documented (7 classes, no species distinction)
- User correctly identified issue with non-California data suggestion
- Infrastructure ready for immediate data download
- All scripts have comprehensive error handling and logging
- Training guide covers troubleshooting for common issues

---

**Last Updated**: December 1, 2025  
**Next Update**: After dataset download completion
