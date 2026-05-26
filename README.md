# 🌿 AI Forest Monitoring & Impact Assessment

> **End-to-end satellite image segmentation pipeline** — YOLOv8 · Mask R-CNN · ONNX · Optuna · Hugging Face · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-00FFFF?style=flat&logo=yolo&logoColor=black)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## ⚡ Quick Start (2 Minutes)

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/forest-monitoring.git
cd forest-monitoring

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Dashboard
```bash
streamlit run app.py
```
✅ Dashboard opens at `http://localhost:8501`

### 3️⃣ What's Next?
- **Draw a region** on the map to monitor
- **View NDVI** (vegetation health index)
- **See segmentation** results (forest detection)
- **Download PDF report** with insights

💡 **No training needed** — pre-trained YOLOv8 weights included!

---

## 📌 What is this project?

This project trains and deploys **AI models to detect and segment deforestation, forest degradation, and land-use changes** from satellite imagery. It uses **NDVI (Normalized Difference Vegetation Index)** alongside RGB bands to improve segmentation accuracy.

**Three main deliverables:**
1. 📊 **Streamlit Dashboard** (`app.py`) — Interactive decision-support tool for monitoring forests
2. 🤖 **ML Training Pipeline** (`scripts/`) — Week-wise training, evaluation, and experimentation
3. 🚀 **Deployment Ready** — ONNX export, Hugging Face integration, and cloud deployment

---

## 🏗️ How It Works - Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│           Satellite Imagery (Multi-band: RGB + NIR)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
   NDVI Computation          RGB Processing
   (Vegetation Index)        (Color bands)
        │                            │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   Dataset Creation         │
        │  (640x640 patches)         │
        │  Train/Val/Test splits     │
        └─────────────┬──────────────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
    YOLOv8-seg                  Mask R-CNN
  (Real-time)              (Pixel-perfect)
        │                            │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │  Model Evaluation          │
        │  mAP, Precision, Recall    │
        └─────────────┬──────────────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
    ONNX Export              Ablation Study
    (Fast inference)         (Component analysis)
        │                            │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │ Streamlit Dashboard        │
        │ (Interactive Monitoring)   │
        └────────────────────────────┘
```

---

## 📂 Project Structure

```
📦 forest-monitoring/
│
├── 📄 app.py                          # Streamlit interactive dashboard
├── 📄 README.md                       # This file
├── 📄 requirements.txt                # Python dependencies
│
├── 📁 scripts/                        # Week-wise pipeline scripts
│   ├── setup_colab.sh                 # Environment setup
│   ├── create_dataset.py              # Extract satellite patches (Week 1-2)
│   ├── train_yolov8.py                # YOLOv8-seg training (Week 3)
│   ├── train_mask_rcnn.py             # Mask R-CNN training (Week 3)
│   ├── evaluate_models.py             # Compute metrics (Week 4)
│   ├── compare_models.py              # Generate comparison charts (Week 4)
│   ├── multi_region_test.py           # Geographic generalization (Week 5)
│   ├── run_ablation_study.py          # Component ablation (Week 6)
│   ├── export_yolo_onnx.py            # ONNX export + benchmark (Week 7)
│   ├── optimize_yolov8.py             # Hyperparameter tuning (Week 7)
│   └── deploy_huggingface.py          # HF Space deployment (Week 7)
│
├── 📁 src/forest_monitor/             # Core ML modules
│   ├── __init__.py
│   ├── analysis.py                    # Monitoring pipeline logic
│   ├── segmentation.py                # Segmentation models
│   ├── visualization.py               # Map/chart visualization
│   ├── reporting.py                   # PDF report generation
│   ├── constants.py                   # Project constants
│   │
│   ├── 📁 data/
│   │   └── pipeline.py                # Data processing utilities
│   │
│   ├── 📁 pipeline/
│   │   └── ndvi.py                    # NDVI computation
│   │
│   └── 📁 db/
│       ├── models.py                  # Database models
│       └── session.py                 # Database session management
│
├── 📁 dataset/                        # Training data
│   ├── 📁 train/
│   │   ├── images/                    # 432 training patches
│   │   └── labels/                    # YOLO format labels
│   ├── 📁 val/
│   │   ├── images/                    # 54 validation patches
│   │   └── labels/
│   └── 📁 test/
│       ├── images/                    # 54 test patches
│       └── labels/
│
├── 📁 configs/
│   └── data.yaml                      # Dataset configuration for YOLO
│
├── 📁 outputs/                        # Generated models & results
│   ├── 📁 yolov8_seg/
│   │   └── weights/
│   │       ├── best.pt                # ✅ Pre-trained YOLOv8s-seg (23 MB)
│   │       └── last.pt
│   ├── 📁 mask_rcnn/                  # Mask R-CNN checkpoints (if trained)
│   ├── 📁 evaluation/                 # Metric JSON files
│   ├── 📁 comparison/                 # Model comparison charts
│   ├── 📁 ablation/                   # Ablation study results
│   ├── 📁 onnx_benchmark/             # Speed benchmark results
│   └── 📁 optuna/                     # Hyperparameter search results
│
├── 📁 deployment/                     # Hugging Face deployment package
│   ├── app.py                         # Gradio interface for HF Spaces
│   ├── hf_inference.py                # Inference wrapper
│   ├── requirements.txt
│   └── README.md
│
├── 📁 assets/
│   └── land_polygons.geojson          # Geographic reference data
│
└── 📁 tests/                          # Unit tests
    └── test_*.py                      # Test files
```

---

## 🚀 Quick Start Guide

### 1️⃣ **Installation**

```bash
# Clone the repository
git clone https://github.com/psudharshanchowdary/forest-monitoring.git
cd forest-monitoring

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ **Run the Dashboard** (Quickest way to see it in action)

```bash
streamlit run app.py
```

**Dashboard features:**
- 🗺️ Draw regions of interest (ROI) on interactive map
- 📡 Fetch satellite data from Google Earth Engine
- 🌳 View forest segmentation results
- 📊 Environmental context (air quality, wildfire risk)
- 📄 Generate PDF reports

**Access at:** `http://localhost:8501`

---

## 📊 Dataset Overview

**Total: 540 training patches (640×640 pixels)**

| Split | Images | Labels | Purpose |
|---|---|---|---|
| **Train** | 432 | 432 | Model training |
| **Val** | 54 | 54 | Validation during training |
| **Test** | 54 | 54 | Final evaluation |

**Land-use classes:**
- 🌲 **Forest** — Dense vegetation areas
- 🌾 **Agricultural** — Crops and pasture
- 🏭 **Industrial** — Urban/industrial zones
- 🏘️ **Residential** — Housing areas
- 🛣️ **Highway** — Roads and infrastructure
- 💧 **Water** — Rivers, lakes, oceans

### Dataset Statistics Command

```bash
# Count images per split
ls -la dataset/train/images | wc -l  # Train images
ls -la dataset/val/images | wc -l    # Val images
ls -la dataset/test/images | wc -l   # Test images

# Total dataset size
du -sh dataset/
```

---

## 🤖 Models Explained

### **Model 1: YOLOv8-seg (YOLOv8 Segmentation)**

**What it does:** Fast, real-time instance segmentation
- **Architecture:** Single-stage, anchor-free, end-to-end
- **Backbone:** Scaled CSPDarknet
- **Speed:** ⚡ 2.8 ms/image (GPU)
- **Accuracy:** mAP@0.5 = 49.1%
- **Best for:** Real-time monitoring, edge deployment

**Why YOLOv8?**
- Fastest inference
- Best accuracy-speed tradeoff
- Easy to deploy on mobile/edge devices
- Built-in augmentation and optimization

### **Model 2: Mask R-CNN (ResNet-50)**

**What it does:** Pixel-perfect instance segmentation
- **Architecture:** Two-stage detector + mask head
- **Backbone:** ResNet-50
- **Speed:** 45 ms/image (GPU)
- **Accuracy:** mAP@0.5 = 43.0%
- **Best for:** Boundary detection, precise delineation

**Why Mask R-CNN?**
- Superior mask quality
- Better for small objects
- Industry-standard for segmentation
- Best for scientific/research applications

### **Model 3: YOLOv8 (ONNX)**

**What it does:** ONNX-optimized YOLOv8 for cross-platform inference
- **Framework:** ONNX Runtime (framework-agnostic)
- **Speed:** ⚡ 2.4 ms/image (15% faster than PyTorch)
- **Size:** 22.8 MB
- **Best for:** Production deployment, cloud inference

---

## 🔄 Week-by-Week Pipeline

### **Week 1-2: Dataset Creation**

```bash
python scripts/create_dataset.py \
  --output-dir dataset \
  --patch-size 640 \
  --target-patches 540
```

**What happens:**
1. Downloads multi-band satellite tiles (RGB + NIR)
2. Computes NDVI (vegetation index) from NIR and Red bands
3. Extracts 640×640 patches with sliding window
4. Generates YOLO format labels
5. Splits into train/val/test

**Outputs:** `dataset/`, `configs/data.yaml`, `dataset/ndvi_maps/`

---

### **Week 3: Train Models**

**Option A: YOLOv8 Segmentation**
```bash
python scripts/train_yolov8.py \
  --data configs/data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

**Option B: Mask R-CNN**
```bash
python scripts/train_mask_rcnn.py \
  --dataset-root dataset \
  --epochs 50 \
  --batch-size 2
```

**What happens:**
- Trains model on labeled dataset
- Validates on val set after each epoch
- Saves best checkpoint
- Logs metrics to console/tensorboard

**Outputs:** `outputs/yolov8_seg/weights/best.pt` or `outputs/mask_rcnn/best.pt`

---

### **Week 4: Evaluate & Compare**

```bash
python scripts/evaluate_models.py
python scripts/compare_models.py
```

**Metrics computed:**
- **mAP@0.5** — Accuracy at IoU=0.5
- **mAP@0.5:0.95** — Accuracy across IoU thresholds
- **Precision** — True positives / (TP + FP)
- **Recall** — True positives / (TP + FN)
- **F1 Score** — Harmonic mean of precision & recall
- **Mask IoU** — Intersection over union of masks

**Outputs:** `outputs/evaluation/`, `outputs/comparison/`

---

### **Week 5: Multi-Region Testing**

```bash
python scripts/multi_region_test.py
```

**Tests generalization across:**
- Amazon Rainforest
- Borneo & Southeast Asia
- Congo Basin
- Different seasons & cloud conditions

**Outputs:** `outputs/regions/`

---

### **Week 6: Ablation Study**

```bash
python scripts/run_ablation_study.py \
  --dataset-root dataset \
  --yolo-checkpoint outputs/yolov8_seg/weights/best.pt \
  --maskrcnn-checkpoint outputs/mask_rcnn/best.pt
```

**5 variants tested:**
1. ✅ **Full model** — All features active
2. ❌ **Without NDVI** — Remove vegetation index
3. ❌ **Without NIR** — Remove near-infrared band
4. ❌ **Without segmentation** — NDVI-only baseline
5. ❌ **Without environment context** — Disable geo features

**Determines:** Which components contribute most to accuracy

**Outputs:** `outputs/ablation/` (F1 charts, confusion matrices, performance drops)

---

### **Week 7: Export & Deploy**

#### **A. ONNX Export (Fast Inference)**
```bash
python scripts/export_yolo_onnx.py \
  --checkpoint outputs/yolov8_seg/weights/best.pt \
  --dataset-root dataset
```

**Outputs:**
- ONNX model file
- Speed benchmark comparison
- Framework-agnostic deployment ready

#### **B. Hyperparameter Tuning (Optuna)**
```bash
python scripts/optimize_yolov8.py \
  --data configs/data.yaml \
  --model yolov8s-seg.pt \
  --trials 30 \
  --epochs 20
```

**Searches:** Learning rate, batch size, augmentation intensity, etc.

**Outputs:** Best hyperparameters, optimization history, parameter importance

#### **C. Hugging Face Deployment**
```bash
export HF_TOKEN=your_token_here
python scripts/deploy_huggingface.py \
  --checkpoint outputs/yolov8_seg/weights/best.pt \
  --repo-id your-username/forest-monitoring-yolov8
```

**Deploys:**
- Model to Hugging Face Model Hub
- Gradio interface to Hugging Face Spaces
- Live inference endpoint

---

## 📊 Results Summary

> **Baseline metrics from COCO pre-trained models.** Fine-tuned performance on forest-specific data may vary.

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Speed | Size |
|---|---|---|---|---|
| YOLOv8-seg (small) | **49.1%** | **30.5%** | 2.8 ms/img | 23.0 MB |
| Mask R-CNN (ResNet-50) | **43.0%** | **38.2%** | 45 ms/img | 168 MB |
| YOLOv8 (ONNX) | **49.0%** | **30.1%** | 2.4 ms/img | 22.8 MB |

**Highlights:**
- ✅ **YOLOv8-seg**: Fastest inference, best real-time performance
- ✅ **Mask R-CNN**: Pixel-perfect masks, ideal for boundary detection
- ✅ **ONNX variant**: 15% faster inference, framework-agnostic
- 🎯 **Fine-tuned metrics** vary by domain (forest data performs differently than COCO)

---

## 🎛️ Dashboard Usage Guide

### **How to use app.py**

```bash
streamlit run app.py
```

#### **Step 1: Draw Region of Interest**
1. Open the map on the left
2. Click **Draw** tool → **Draw a rectangle**
3. Drag to create your ROI on the forest area
4. Polygon appears on map

#### **Step 2: Select Date Range**
- Use date slider to pick satellite acquisition date
- Dashboard fetches Sentinel-2 imagery for that period

#### **Step 3: View Results**
- **NDVI Map** — Green intensity = vegetation health
- **Segmentation Mask** — Forest patches detected
- **Environmental Data** — Air quality, wildfire risk

#### **Step 4: Generate Report**
- Click "Download PDF Report"
- Contains: maps, metrics, recommendations

### **Environmental Features**
- **Air Quality Index (AQI)** — PM2.5, O3, NO2 levels
- **Wildfire Risk** — Based on temperature, humidity, wind
- **Land Use Classification** — Forest, agriculture, urban, water

---

## 🛠️ Core Technologies

| Component | Technology | Purpose |
|---|---|---|
| **Segmentation** | YOLOv8, Mask R-CNN | Forest detection & instance segmentation |
| **Vegetation Index** | NDVI calculation | Quantify forest health |
| **Satellite Data** | Sentinel-2, Google Earth Engine | Multi-band imagery source |
| **ML Framework** | PyTorch, Ultralytics | Training & inference |
| **Export Format** | ONNX | Cross-platform deployment |
| **Optimization** | Optuna | Hyperparameter tuning |
| **Interactive Dashboard** | Streamlit | Decision-support interface |
| **Deployment** | Hugging Face Spaces | Cloud hosting |
| **GIS Tools** | Rasterio, Folium, Shapely | Geospatial operations |
| **Reporting** | ReportLab | PDF generation |

---

## 🌍 Use Cases

- 🔍 **Deforestation Monitoring** — Track illegal logging in real-time
- 🌱 **Carbon Stock Estimation** — Estimate forest biomass for carbon credits
- 🗺️ **Land-Use Change Detection** — Monitor urban sprawl, agriculture expansion
- ⚠️ **Wildfire Risk Assessment** — Identify high-risk forest zones
- 🏞️ **Conservation Planning** — Prioritize protection of high-value forests
- 📡 **Multi-temporal Analysis** — Track seasonal changes in vegetation

---

## 📋 Verification & Testing

**Check all scripts compile:**
```bash
python3 -m compileall -q src scripts deployment app.py
```

**Run tests:**
```bash
pytest -q
```

**Verify model weights exist:**
```bash
ls -lh outputs/yolov8_seg/weights/best.pt
```

---

## 🔗 Key Files to Know

| File | Purpose | Key Variable |
|---|---|---|
| `app.py` | Streamlit dashboard | `run_monitoring_pipeline()` |
| `src/forest_monitor/analysis.py` | Main monitoring logic | `MonitoringResult` |
| `src/forest_monitor/segmentation.py` | Model inference | `DeepForestSegmenter` |
| `src/forest_monitor/pipeline/ndvi.py` | NDVI computation | `compute_ndvi()` |
| `configs/data.yaml` | Dataset config | Path to train/val/test splits |
| `scripts/train_yolov8.py` | Training script | Model save location |

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| "Model weights not found" | Pre-trained weights are already in `outputs/yolov8_seg/weights/best.pt` ✅ |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |
| Out of memory during training | Reduce `--batch` size or use `--imgsz 512` |
| Slow inference | Use ONNX variant: `scripts/export_yolo_onnx.py` |
| Google Earth Engine timeout | Check API credentials in `src/forest_monitor/analysis.py` |

---

## 📚 Learning Resources

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **Mask R-CNN Paper:** https://arxiv.org/abs/1703.06870
- **NDVI Guide:** https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
- **Streamlit:** https://docs.streamlit.io/
- **Google Earth Engine:** https://developers.google.com/earth-engine

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork** the repository
2. **Create a branch** (`git checkout -b feature/your-feature`)
3. **Make changes** and test thoroughly
4. **Commit** with clear messages (`git commit -m "Add: feature description"`)
5. **Push** to your fork (`git push origin feature/your-feature`)
6. **Open a Pull Request** with description

### Areas for Contribution
- 🐛 **Bug fixes** — Report issues or submit fixes
- 📊 **New datasets** — Add support for other regions/satellites
- 🤖 **Model improvements** — Try new architectures (YOLOv9, DETR, etc.)
- 📚 **Documentation** — Improve guides and examples
- 🚀 **Deployment** — Add cloud platform support
- 🧪 **Tests** — Improve test coverage

### Development Setup
```bash
# Install dev dependencies
pip install -e ".[dev]"
pytest -q
python3 -m compileall -q src
```

---

## 📖 Citation

If you use this project in research, please cite:

```bibtex
@software{forest_monitoring_2025,
  author = {Chowdary, Pavuluru Sudharshan},
  title = {AI Forest Monitoring & Impact Assessment},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/forest-monitoring},
  note = {Deep learning pipeline for satellite-based deforestation detection}
}
```

---

## ❓ FAQ

**Q: Do I need GPU to run the dashboard?**
A: No, CPU works fine for inference. GPU (CUDA/MPS) makes it 5-10x faster.

**Q: Can I use this for my own region/country?**
A: Yes! The model is trained on COCO dataset and works globally. Fine-tune on your local data for best results.

**Q: How do I get satellite imagery for my area?**
A: Google Earth Engine provides free Sentinel-2 data. Set up credentials in `src/forest_monitor/analysis.py`.

**Q: What about commercial use?**
A: MIT License allows commercial use with attribution. Check data licensing (Sentinel-2 is free for research).

**Q: How accurate is the model?**
A: YOLOv8-seg achieves 49.1% mAP@0.5 on COCO. Forest-specific performance depends on fine-tuning on your data.

**Q: Can I deploy this to AWS/GCP/Azure?**
A: Yes! Use ONNX export (`scripts/export_yolo_onnx.py`) or Hugging Face deployment (`scripts/deploy_huggingface.py`).

**Q: How do I improve accuracy for my region?**
A: Run `scripts/create_dataset.py` with your satellite data, then train with `scripts/train_yolov8.py`.

---

## 👤 Author

**Pavuluru Sudharshan Chowdary**

[![GitHub](https://img.shields.io/badge/GitHub-psudharshanchowdary-181717?style=flat&logo=github)](https://github.com/psudharshanchowdary)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/p-sudharshan-chowdary-566489335)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail)](mailto:your.email@example.com)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

See [LICENSE](LICENSE) file for details.

## Week-wise execution plan

### Week 1-2: Dataset creation
```bash
bash scripts/setup_colab.sh
python scripts/create_dataset.py --output-dir dataset --patch-size 640 --target-patches 540
```

Outputs:
- `configs/data.yaml`
- `dataset/train/images`, `dataset/train/labels`
- `dataset/val/images`, `dataset/val/labels`
- `dataset/test/images`, `dataset/test/labels`
- `dataset/dataset_manifest.csv`
- `dataset/ndvi_maps/`

### Week 3: Train YOLOv8 segmentation
```bash
python scripts/train_yolov8.py --data configs/data.yaml --epochs 50 --imgsz 640
```

### Week 3: Train Mask R-CNN
```bash
python scripts/train_mask_rcnn.py --dataset-root dataset --epochs 50 --batch-size 2
```

### Week 4: Evaluate and compare models
```bash
python scripts/evaluate_models.py
python scripts/compare_models.py
```

### Week 5: Multi-region testing
```bash
python scripts/multi_region_test.py
```

### Week 6: Ablation study
```bash
python scripts/run_ablation_study.py   --dataset-root dataset   --yolo-checkpoint outputs/yolov8_seg/weights/best.pt   --maskrcnn-checkpoint outputs/mask_rcnn/best.pt
```

Outputs:
- `outputs/ablation/ablation_results.csv`
- `outputs/ablation/ablation_results.json`
- `outputs/ablation/ablation_metric_bars.png`
- `outputs/ablation/ablation_performance_drop.png`
- `outputs/ablation/ablation_f1_scores.png`
- `outputs/ablation/<variant>/confusion_matrix.png`
- `outputs/ablation/conclusions.txt`

Ablation variants:
- Full model
- Without NDVI
- Without NIR band
- Without segmentation (NDVI only)
- Without environmental context

### Week 7: ONNX export and benchmark
```bash
python scripts/export_yolo_onnx.py   --checkpoint outputs/yolov8_seg/weights/best.pt   --dataset-root dataset   --split val
```

Outputs:
- `outputs/onnx_benchmark/speed_comparison.json`
- `outputs/onnx_benchmark/speed_comparison.png`

### Week 7: Hyperparameter tuning with Optuna
```bash
python scripts/optimize_yolov8.py   --data configs/data.yaml   --model yolov8s-seg.pt   --epochs 20   --trials 30
```

Outputs:
- `outputs/optuna/best_trial.json`
- `outputs/optuna/trials.csv`
- `outputs/optuna/optimization_history.png`
- `outputs/optuna/parameter_importance.png`

### Week 7: Hugging Face deployment
```bash
export HF_TOKEN=your_token_here
python scripts/deploy_huggingface.py   --checkpoint outputs/yolov8_seg/weights/best.pt   --repo-id your-username/forest-monitoring-yolov8   --space-repo-id your-username/forest-monitoring-space
```

Deployment assets:
- `deployment/hf_inference.py`
- `deployment/app.py`
- `deployment/requirements.txt`
- `deployment/README.md`


