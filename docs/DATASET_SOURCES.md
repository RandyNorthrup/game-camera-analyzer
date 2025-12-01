# California Wildlife Dataset Sources

Quick reference guide for accessing California-specific camera trap datasets for custom model training.

---

## 🎯 Primary Datasets (California-Specific)

### 1. Caltech Camera Traps ⭐ RECOMMENDED
**Location**: Southwestern United States (Southern California)  
**Size**: 243,100 images (105GB)  
**Species**: 21 categories including opossum, raccoon, coyote, bobcat, gray fox, deer, squirrel  
**Bounding Boxes**: ~66,000 annotations  
**Empty Images**: ~70% (typical for camera traps)

**Download Links**:
- **Metadata (Image-level)**: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip (9MB)
- **Metadata (Bounding boxes)**: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_bboxes_20200316.json (35MB)
- **Train/Val Splits**: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/CaltechCameraTrapsSplits_v0.json (4KB)
- **Images (Full dataset)**: 
  - GCP: `gs://public-datasets-lila/caltech-unzipped/cct_images`
  - Azure: https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images
  - AWS: `s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped/cct_images`
  - Tar.gz: https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/cct_images.tar.gz

**Citation**:
```
Sara Beery, Grant Van Horn, Pietro Perona. 
Recognition in Terra Incognita. 
Proceedings of the 15th European Conference on Computer Vision (ECCV 2018).
```

**More Info**: https://lila.science/datasets/caltech-camera-traps  
**Project Page**: https://beerys.github.io/CaltechCameraTraps/

---

### 2. Channel Islands Camera Traps
**Location**: Channel Islands, California  
**Size**: 246,529 images (40GB estimated)  
**Species**: Fox, rodents, birds (73 camera locations)  
**Bounding Boxes**: Yes (all animals annotated)

**Download Links**:
- **Metadata**: https://storage.googleapis.com/public-datasets-lila/channel-islands-camera-traps/channel_islands_camera_traps.json.zip
- **Images**: https://lilawildlife.blob.core.windows.net/lila-wildlife/channel-islands-camera-traps

**Provider**: The Nature Conservancy

**More Info**: https://lila.science/datasets/channel-islands-camera-traps/

---

### 3. California Small Animals ⚠️ VERY LARGE
**Location**: California Department of Fish and Wildlife  
**Size**: 2,278,071 images (~350GB+)  
**Species**: Small mammals following standardized protocol  
**Quality**: High-quality professional wildlife monitoring data

**Note**: This is an extremely large dataset. Start with Caltech or Channel Islands first.

**More Info**: https://lila.science/datasets/california-small-animals/  
**Protocol**: https://storymaps.arcgis.com/collections/a6a06b22b05f43cd863568b2690b08c1?item=1

---

## 🌎 Supplementary Datasets (Western US)

### Idaho Camera Traps
**Size**: 1.5M images  
**Species**: 62 categories (deer, elk, cattle are most common)  
**Location**: Idaho (similar ecosystems to Northern California)

**More Info**: https://lila.science/datasets/idaho-camera-traps/

---

### North American Camera Trap Images (NACTI)
**Size**: 3.7M images  
**Species**: 28 categories (cattle, boar, red deer most common)  
**Locations**: 5 locations across United States (may include California)

**More Info**: https://lila.science/datasets/nacti

---

## 📚 Additional Resources

### LILA Main Portal
**Website**: https://lila.science/datasets  
**Description**: Complete catalog of all wildlife camera trap datasets

### MegaDetector Results
**URL**: https://lila.science/megadetector-results-for-camera-trap-datasets/  
**Description**: Pre-computed detection results for all LILA datasets using MegaDetector

### Taxonomy Mapping
**URL**: https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/  
**Description**: Guide for mapping species labels to common taxonomy

### Download Guidelines
**URL**: https://lila.science/image-access  
**Description**: Best practices for downloading specific subsets without giant zipfiles

---

## 🛠️ Download Tools

### Using Our Script
```bash
# Download Caltech dataset (metadata + filter for California species)
python scripts/download_lila_dataset_v2.py \
    --datasets caltech_camera_traps \
    --max-images 1000 \
    --output training_data/lila
```

### Using Google Cloud SDK (for large downloads)
```bash
# Install gsutil
pip install gsutil

# Download Caltech images
gsutil -m cp -r gs://public-datasets-lila/caltech-unzipped/cct_images training_data/lila/caltech/
```

### Using Azure CLI
```bash
# Install azcopy
brew install azcopy

# Download Caltech images
azcopy copy \
    'https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/*' \
    'training_data/lila/caltech/' \
    --recursive
```

### Using wget (for metadata)
```bash
# Download Caltech metadata
wget https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip
wget https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_bboxes_20200316.json
```

---

## 📊 Recommended Training Strategy

### Phase 1: Initial Training (Small Scale)
1. **Download**: Caltech Camera Traps (start with 10,000 images)
2. **Train**: Quick validation run (yolov8n, 10 epochs)
3. **Evaluate**: Check if approach is working

### Phase 2: Full Training (Medium Scale)
1. **Download**: Full Caltech dataset (243k images)
2. **Optionally add**: Channel Islands (246k images)
3. **Train**: Production model (yolov8m, 100 epochs)
4. **Evaluate**: Full metrics and per-species analysis

### Phase 3: Enhancement (Large Scale - if needed)
1. **Download**: California Small Animals dataset (2.2M images)
2. **Combine**: All datasets with data augmentation
3. **Train**: Large-scale model (yolov8l or yolov8x)
4. **Fine-tune**: Hyperparameter optimization

---

## 🎯 Target California Species

Our model will focus on these 35 species found in California:

**Large Mammals**: mule_deer, roosevelt_elk, tule_elk, wild_pig, desert_bighorn_sheep, pronghorn

**Predators**: black_bear, mountain_lion, bobcat, coyote, gray_fox, kit_fox, san_joaquin_kit_fox, badger

**Medium Mammals**: raccoon, striped_skunk, spotted_skunk, ringtail, virginia_opossum, river_otter, marten, fisher, long_tailed_weasel

**Small Mammals**: western_gray_squirrel, california_ground_squirrel, california_kangaroo_rat, woodrat, black_tailed_jackrabbit, brush_rabbit, desert_cottontail

**Birds**: wild_turkey, california_quail, common_raven, roadrunner

---

## 📝 License Information

All LILA datasets are released under the **Community Data License Agreement (CDLA) - Permissive variant**.

**License URL**: https://cdla.io/permissive-1-0/

**Summary**: Free to use for research and commercial purposes with attribution.

---

## 💡 Tips for Efficient Downloads

1. **Start small**: Download 1,000-10,000 images first to validate pipeline
2. **Use cloud tools**: gsutil or azcopy are faster than wget for large datasets
3. **Filter early**: Use metadata JSON to identify relevant images before downloading
4. **Parallel downloads**: Use `-m` flag with gsutil for parallel transfers
5. **Resume capability**: Cloud tools support resuming interrupted downloads

---

**Last Updated**: December 1, 2025  
**Maintained by**: Game Camera Analyzer Project  
**For issues**: See main project README
