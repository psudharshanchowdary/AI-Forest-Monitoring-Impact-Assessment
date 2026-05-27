# 🌿 AI Forest Monitoring & Impact Assessment

> I built this to make it easier to understand what's happening to our forests — using satellite imagery, AI segmentation, and a dashboard that anyone can actually use.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-00FFFF?style=flat&logo=yolo&logoColor=black)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## ⚡ Get Up and Running in 2 Minutes

You don't need a PhD to use this. Here's all you need to do: 
### 1️⃣ Clone & Set Up

```bash
git clone https://github.com/YOUR_USERNAME/forest-monitoring.git
cd forest-monitoring

# Create a virtual environment (keeps things tidy)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install everything you need
pip install -r requirements.txt
```

### 2️⃣ Launch the Dashboard
```bash
streamlit run app.py
```

That's it — the dashboard opens at `http://localhost:8501`. No setup wizards, no YAML configuration hell.

### 3️⃣ Start Exploring
Once it's running, you can:
- **Draw a region** on the map that you want to monitor
- **View NDVI** — a vegetation health score derived from satellite bands
- **See where the forest is** — and where it's disappearing
- **Download a PDF report** with everything laid out clearly

💡 **You don't need to train anything.** Pre-trained YOLOv8 weights are already included.
---

## 📌 What Is This, and Why Does It Exist?

Forests are disappearing faster than most people realize. Illegal logging, agricultural expansion, and industrial development are changing the landscape — often in remote areas that no one is watching.

This project is my attempt to change that. It uses **AI-powered satellite image analysis** to detect deforestation, track forest degradation, and flag land-use changes. The backbone is **NDVI (Normalized Difference Vegetation Index)** combined with RGB imagery — together, they give a much clearer picture of forest health than visible light alone.

There are three main things this project delivers:

1. 📊 **An interactive Streamlit dashboard** (`app.py`) — draw your region, get your results
2. 🤖 **A full ML training pipeline** (`scripts/`) — week-by-week, from raw satellite data to trained models
3. 🚀 **Production-ready deployment** — ONNX export, Hugging Face integration, cloud-ready

---

## 🏗️ How It All Fits Together

Here's the big picture of how data flows through the system:

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
    (Fast inference)         (What actually matters?)
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

I've tried to keep things organized by purpose so you can jump in wherever makes sense for you:

```
📦 forest-monitoring/
│
├── 📄 app.py                          # The dashboard — start here
├── 📄 README.md                       # You're reading it
├── 📄 requirements.txt                # All dependencies
│
├── 📁 scripts/                        # Week-wise pipeline scripts
│   ├── setup_colab.sh                 # Environment setup
│   ├── create_dataset.py              # Satellite patch extraction (Week 1-2)
│   ├── train_yolov8.py                # YOLOv8-seg training (Week 3)
│   ├── train_mask_rcnn.py             # Mask R-CNN training (Week 3)
│   ├── evaluate_models.py             # Metric computation (Week 4)
│   ├── compare_models.py              # Comparison charts (Week 4)
│   ├── multi_region_test.py           # Geographic generalization (Week 5)
│   ├── run_ablation_study.py          # Ablation analysis (Week 6)
│   ├── export_yolo_onnx.py            # ONNX export + benchmarks (Week 7)
│   ├── optimize_yolov8.py             # Hyperparameter search (Week 7)
│   └── deploy_huggingface.py          # Deploy to HF Spaces (Week 7)
│
├── 📁 src/forest_monitor/             # Core ML modules
│   ├── __init__.py
│   ├── analysis.py                    # Monitoring pipeline logic
│   ├── segmentation.py                # Model inference wrappers
│   ├── visualization.py               # Maps and charts
│   ├── reporting.py                   # PDF report generation
│   ├── constants.py                   # Project-wide constants
│   │
│   ├── 📁 data/
│   │   └── pipeline.py                # Data processing utilities
│   │
│   ├── 📁 pipeline/
│   │   └── ndvi.py                    # NDVI computation
│   │
│   └── 📁 db/
│       ├── models.py                  # Database models
│       └── session.py                 # Session management
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
│   └── data.yaml                      # Dataset config for YOLO
│
├── 📁 outputs/                        # Everything the models produce
│   ├── 📁 yolov8_seg/weights/
│   │   ├── best.pt                    # ✅ Pre-trained YOLOv8s-seg (23 MB)
│   │   └── last.pt
│   ├── 📁 mask_rcnn/
│   ├── 📁 evaluation/
│   ├── 📁 comparison/
│   ├── 📁 ablation/
│   ├── 📁 onnx_benchmark/
│   └── 📁 optuna/
│
├── 📁 deployment/                     # Hugging Face package
│   ├── app.py                         # Gradio interface
│   ├── hf_inference.py                # Inference wrapper
│   ├── requirements.txt
│   └── README.md
│
├── 📁 assets/
│   └── land_polygons.geojson          # Geographic reference data
│
└── 📁 tests/
    └── test_*.py
```

---

## 📊 The Dataset

I put together **540 training patches, each 640×640 pixels**, covering a range of environments and land-use types.

| Split | Images | Labels | What it's used for |
|---|---|---|---|
| **Train** | 432 | 432 | Teaching the model |
| **Val** | 54 | 54 | Checking it during training |
| **Test** | 54 | 54 | Final honest evaluation |

**The six land-use classes the model learns to recognize:**
- 🌲 **Forest** — Dense vegetation
- 🌾 **Agricultural** — Crops and pasture
- 🏭 **Industrial** — Urban and industrial zones
- 🏘️ **Residential** — Housing areas
- 🛣️ **Highway** — Roads and infrastructure
- 💧 **Water** — Rivers, lakes, oceans

Want to check your dataset quickly?

```bash
ls -la dataset/train/images | wc -l  # How many training images?
ls -la dataset/val/images | wc -l    # Validation set size?
du -sh dataset/                      # Total disk space?
```

---

## 🤖 The Models

I trained and compared two models — each with its own strengths. Here's an honest breakdown:

### YOLOv8-seg — Speed is its superpower

YOLOv8 is a single-stage detector that runs incredibly fast without sacrificing too much accuracy. It's what I'd recommend for real-time monitoring or anything where latency matters.

- **Architecture:** Anchor-free, end-to-end single-stage
- **Backbone:** Scaled CSPDarknet
- **Speed:** ⚡ 2.8 ms/image on GPU
- **Accuracy:** mAP@0.5 = 49.1%
- **Best for:** Real-time monitoring, edge deployment, quick iteration

### Mask R-CNN — When precision matters more than speed

Mask R-CNN is a two-stage detector that takes longer but produces much cleaner mask boundaries. If you need pixel-level accuracy — for scientific analysis or boundary delineation — this is the one.

- **Architecture:** Two-stage detector with a dedicated mask head
- **Backbone:** ResNet-50
- **Speed:** 45 ms/image on GPU
- **Accuracy:** mAP@0.5 = 43.0%
- **Best for:** Precise boundary detection, research applications

### YOLOv8 (ONNX) — The production-ready version

The ONNX export of YOLOv8 is actually 15% faster than the PyTorch version, and it runs anywhere — no PyTorch dependency required.

- **Speed:** ⚡ 2.4 ms/image
- **Size:** 22.8 MB
- **Best for:** Cloud deployment, production systems

---

## 🔄 The Full Pipeline, Week by Week

This project was built over seven weeks. Here's what happened each week and how to reproduce it:

### Weeks 1–2: Building the Dataset

```bash
python scripts/create_dataset.py \
  --output-dir dataset \
  --patch-size 640 \
  --target-patches 540
```

This script downloads multi-band satellite tiles, computes NDVI from the NIR and Red bands, slices everything into 640×640 patches, generates YOLO-format labels, and splits the data into train/val/test. You end up with a clean, annotated dataset ready for training.

**Outputs:** `dataset/`, `configs/data.yaml`, `dataset/ndvi_maps/`

---

### Week 3: Training the Models

**Train YOLOv8-seg:**
```bash
python scripts/train_yolov8.py \
  --data configs/data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

**Train Mask R-CNN:**
```bash
python scripts/train_mask_rcnn.py \
  --dataset-root dataset \
  --epochs 50 \
  --batch-size 2
```

Both scripts validate on the val set after every epoch and save the best checkpoint automatically.

**Outputs:** `outputs/yolov8_seg/weights/best.pt` or `outputs/mask_rcnn/best.pt`

---

### Week 4: Evaluating and Comparing

```bash
python scripts/evaluate_models.py
python scripts/compare_models.py
```

I computed the full suite of metrics here — not just accuracy, but precision, recall, F1, and mask IoU. The comparison script generates side-by-side charts so you can actually see where each model wins.

**Metrics:**
- **mAP@0.5** — Accuracy at IoU threshold of 0.5
- **mAP@0.5:0.95** — Accuracy averaged across multiple IoU thresholds
- **Precision** — How often the model is right when it fires
- **Recall** — How much of the actual forest it catches
- **F1 Score** — The balance between precision and recall
- **Mask IoU** — How well the masks overlap with ground truth

**Outputs:** `outputs/evaluation/`, `outputs/comparison/`

---

### Week 5: Testing Across Different Regions

```bash
python scripts/multi_region_test.py
```

A model that only works on one region isn't very useful. This script tested generalization across the Amazon Rainforest, Borneo, and the Congo Basin — including different seasons and cloud conditions.

**Outputs:** `outputs/regions/`

---

### Week 6: Ablation Study — What Actually Matters?

```bash
python scripts/run_ablation_study.py \
  --dataset-root dataset \
  --yolo-checkpoint outputs/yolov8_seg/weights/best.pt \
  --maskrcnn-checkpoint outputs/mask_rcnn/best.pt
```

I ran five variants of the model to figure out which components are actually carrying their weight:

1. ✅ **Full model** — Everything enabled
2. ❌ **Without NDVI** — Does removing the vegetation index hurt?
3. ❌ **Without NIR** — How important is the near-infrared band?
4. ❌ **Without segmentation** — What if we just used NDVI as a baseline?
5. ❌ **Without environmental context** — Does geographic context help?

**Outputs:** F1 charts, confusion matrices, performance drop analysis — all in `outputs/ablation/`

---

### Week 7: Making It Production-Ready

#### ONNX Export

```bash
python scripts/export_yolo_onnx.py \
  --checkpoint outputs/yolov8_seg/weights/best.pt \
  --dataset-root dataset
```

Converts the model to ONNX format and benchmarks it against the PyTorch version. The speed improvement is real — about 15% faster.

#### Hyperparameter Tuning with Optuna

```bash
python scripts/optimize_yolov8.py \
  --data configs/data.yaml \
  --model yolov8s-seg.pt \
  --trials 30 \
  --epochs 20
```

Optuna searches over learning rate, batch size, augmentation intensity, and more. After 30 trials, it tells you what settings actually matter.

**Outputs:** `outputs/optuna/best_trial.json`, optimization history, parameter importance charts

#### Deploy to Hugging Face

```bash
export HF_TOKEN=your_token_here
python scripts/deploy_huggingface.py \
  --checkpoint outputs/yolov8_seg/weights/best.pt \
  --repo-id your-username/forest-monitoring-yolov8
```

This pushes the model to Hugging Face Hub and spins up a Gradio interface on HF Spaces. Anyone can then run inference directly from their browser.

---

## 📊 Results Summary

> These are baseline metrics using COCO pre-trained weights. Fine-tuned performance on forest-specific data will look different depending on your region and dataset quality.

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Speed | Size |
|---|---|---|---|---|
| YOLOv8-seg (small) | **49.1%** | **30.5%** | 2.8 ms/img | 23.0 MB |
| Mask R-CNN (ResNet-50) | **43.0%** | **38.2%** | 45 ms/img | 168 MB |
| YOLOv8 (ONNX) | **49.0%** | **30.1%** | 2.4 ms/img | 22.8 MB |

**My take:**
- Use **YOLOv8-seg** if you need real-time results or are deploying on limited hardware
- Use **Mask R-CNN** if boundary accuracy matters more than speed
- Use the **ONNX version** in production — same accuracy, faster and more portable

---

## 🎛️ Using the Dashboard

```bash
streamlit run app.py
```

#### Step 1: Draw Your Region
Open the map, click the Draw tool, and drag a rectangle over the area you want to monitor. It can be as small as a village or as large as a national park.

#### Step 2: Pick a Date Range
Use the date slider to select when you want satellite data from. The dashboard fetches Sentinel-2 imagery for that period automatically.

#### Step 3: See What the Model Found
- **NDVI Map** — Greener = healthier vegetation
- **Segmentation Mask** — Where the model thinks forest exists (and where it's gone)
- **Environmental Context** — Air quality, wildfire risk score, land classification

#### Step 4: Download the Report
Hit "Download PDF Report" and you get a clean document with maps, metrics, and recommendations you can actually share with someone.

---

## 🛠️ What's Under the Hood

| Component | Technology | Why I chose it |
|---|---|---|
| **Segmentation** | YOLOv8, Mask R-CNN | Best accuracy-speed tradeoffs available |
| **Vegetation Index** | NDVI | Reliable, satellite-agnostic health metric |
| **Satellite Data** | Sentinel-2, Google Earth Engine | Free, high-quality, global coverage |
| **ML Framework** | PyTorch, Ultralytics | Mature ecosystems, great documentation |
| **Export Format** | ONNX | Deploy anywhere without PyTorch |
| **Optimization** | Optuna | Bayesian search beats grid search |
| **Dashboard** | Streamlit | Fast to build, easy to use |
| **Deployment** | Hugging Face Spaces | Free, accessible, no DevOps required |
| **GIS Tools** | Rasterio, Folium, Shapely | The standard toolkit for geospatial Python |
| **Reporting** | ReportLab | Reliable PDF generation |

---

## 🌍 What Can You Use This For?

- 🔍 **Deforestation Monitoring** — Flag areas where forest cover is shrinking
- 🌱 **Carbon Stock Estimation** — Estimate biomass for carbon credit calculations
- 🗺️ **Land-Use Change Detection** — Track urban sprawl or agricultural expansion over time
- ⚠️ **Wildfire Risk Assessment** — Identify high-risk zones before fire season
- 🏞️ **Conservation Planning** — Prioritize which areas need protection most urgently
- 📡 **Multi-temporal Analysis** — Compare the same location across seasons or years

---

## ✅ Verification & Testing

If you want to make sure everything is working before diving in:

```bash
# Check all scripts compile without errors
python3 -m compileall -q src scripts deployment app.py

# Run the test suite
pytest -q

# Confirm the pre-trained weights are present
ls -lh outputs/yolov8_seg/weights/best.pt
```

---

## 🔗 Files Worth Knowing About

| File | What it does |
|---|---|
| [app.py](app.py) | The dashboard — this is where most users will spend their time |
| [src/forest_monitor/analysis.py](src/forest_monitor/analysis.py) | Main monitoring logic, `MonitoringResult` class |
| [src/forest_monitor/segmentation.py](src/forest_monitor/segmentation.py) | Model inference, `DeepForestSegmenter` |
| [src/forest_monitor/pipeline/ndvi.py](src/forest_monitor/pipeline/ndvi.py) | NDVI computation, `compute_ndvi()` |
| [configs/data.yaml](configs/data.yaml) | Dataset config for YOLO training |
| [scripts/train_yolov8.py](scripts/train_yolov8.py) | YOLOv8 training script |

---

## 🐛 When Things Go Wrong

Most issues have simple fixes. Here's what I've run into:

| Problem | Fix |
|---|---|
| "Model weights not found" | They're already in `outputs/yolov8_seg/weights/best.pt` — double-check the path |
| Streamlit port in use | Run `streamlit run app.py --server.port 8502` |
| Out of memory during training | Drop the batch size with `--batch 4`, or use `--imgsz 512` |
| Slow inference | Export to ONNX — it's ~15% faster |
| Google Earth Engine timeout | Your API credentials might need refreshing in `src/forest_monitor/analysis.py` |

---

## 📚 Further Reading

If you want to go deeper on any of the technologies involved:

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **Mask R-CNN Paper:** https://arxiv.org/abs/1703.06870
- **NDVI Explained:** https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
- **Streamlit Docs:** https://docs.streamlit.io/
- **Google Earth Engine:** https://developers.google.com/earth-engine

---



## ❓ Questions I Get Asked a Lot

**Do I need a GPU to run the dashboard?**
No — CPU is totally fine for inference. A GPU (CUDA or Apple MPS) makes it 5–10× faster, but it's not required.

**Can I use this for my own country or region?**
Yes. The base model was trained on COCO data and works globally. For best results on a specific region, fine-tune it on local satellite patches using `scripts/train_yolov8.py`.

**How do I get satellite imagery for my area?**
Google Earth Engine provides free Sentinel-2 data. Set up your API credentials in `src/forest_monitor/analysis.py`.

**Is this okay for commercial use?**
The satellite data (Sentinel-2) is free for research. Check the data provider's terms for commercial use.

**How accurate is it?**
YOLOv8-seg hits 49.1% mAP@0.5 on COCO. That number will change once you fine-tune on forest-specific data — usually for the better.

**Can I deploy this on AWS, GCP, or Azure?**
Yes. Either use the ONNX export (`scripts/export_yolo_onnx.py`) or push to Hugging Face (`scripts/deploy_huggingface.py`) for a quick cloud endpoint.

---

## Week-by-Week Execution Summary

### Weeks 1–2: Dataset Creation
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

### Week 3: Train YOLOv8 Segmentation
```bash
python scripts/train_yolov8.py --data configs/data.yaml --epochs 50 --imgsz 640
```

### Week 3: Train Mask R-CNN
```bash
python scripts/train_mask_rcnn.py --dataset-root dataset --epochs 50 --batch-size 2
```

### Week 4: Evaluate and Compare
```bash
python scripts/evaluate_models.py
python scripts/compare_models.py
```

### Week 5: Multi-Region Testing
```bash
python scripts/multi_region_test.py
```

### Week 6: Ablation Study
```bash
python scripts/run_ablation_study.py \
  --dataset-root dataset \
  --yolo-checkpoint outputs/yolov8_seg/weights/best.pt \
  --maskrcnn-checkpoint outputs/mask_rcnn/best.pt
```

Outputs:
- `outputs/ablation/ablation_results.csv`
- `outputs/ablation/ablation_results.json`
- `outputs/ablation/ablation_metric_bars.png`
- `outputs/ablation/ablation_performance_drop.png`
- `outputs/ablation/ablation_f1_scores.png`
- `outputs/ablation/<variant>/confusion_matrix.png`
- `outputs/ablation/conclusions.txt`

Ablation variants tested: full model, without NDVI, without NIR band, without segmentation (NDVI only), without environmental context.

### Week 7: ONNX Export and Benchmark
```bash
python scripts/export_yolo_onnx.py \
  --checkpoint outputs/yolov8_seg/weights/best.pt \
  --dataset-root dataset \
  --split val
```

Outputs:
- `outputs/onnx_benchmark/speed_comparison.json`
- `outputs/onnx_benchmark/speed_comparison.png`

### Week 7: Hyperparameter Tuning with Optuna
```bash
python scripts/optimize_yolov8.py \
  --data configs/data.yaml \
  --model yolov8s-seg.pt \
  --epochs 20 \
  --trials 30
```

Outputs:
- `outputs/optuna/best_trial.json`
- `outputs/optuna/trials.csv`
- `outputs/optuna/optimization_history.png`
- `outputs/optuna/parameter_importance.png`

### Week 7: Hugging Face Deployment
```bash
export HF_TOKEN=your_token_here
python scripts/deploy_huggingface.py \
  --checkpoint outputs/yolov8_seg/weights/best.pt \
  --repo-id your-username/forest-monitoring-yolov8 \
  --space-repo-id your-username/forest-monitoring-space
```

Deployment assets:
- `deployment/hf_inference.py`
- `deployment/app.py`
- `deployment/requirements.txt`
- `deployment/README.md`



## 🛠️ Technologies Used

Here's a full breakdown of every tool and library in the stack, and the honest reason each one is here:

| Category | Technology | Why |
|---|---|---|
| **Instance Segmentation** | YOLOv8-seg, Mask R-CNN | Two different philosophies — speed vs. precision |
| **Deep Learning Framework** | PyTorch 2.4+, Ultralytics | Rock-solid ecosystem, great community support |
| **Vegetation Analysis** | NDVI (custom implementation) | Physics-based, satellite-agnostic health metric |
| **Satellite Imagery** | Sentinel-2 via Google Earth Engine | Free, globally available, multi-band |
| **Model Export** | ONNX Runtime | Cross-platform deployment without PyTorch dependency |
| **Hyperparameter Tuning** | Optuna | Bayesian search — smarter than grid or random search |
| **Dashboard** | Streamlit | Fastest way to build a usable ML interface |
| **Cloud Deployment** | Hugging Face Spaces + Gradio | Free hosting, no DevOps, accessible to anyone |
| **Geospatial** | Rasterio, Folium, Shapely, GeoPandas | The standard Python GIS toolkit |
| **PDF Reporting** | ReportLab | Reliable, programmatic PDF generation |
| **Database** | SQLAlchemy | Session and result persistence |
| **Testing** | pytest | Clean, simple test runner |
| **Language** | Python 3.11+ | Best ML ecosystem, latest performance improvements |

---

## ✨ Features

Here's what this project can actually do:

### 🗺️ Interactive Map Dashboard
- Draw any region of interest directly on a live map
- Supports rectangles and polygons — from a single field to an entire region
- Instant feedback once your ROI is defined

### 📡 Satellite Data Integration
- Fetches real Sentinel-2 multi-band imagery via Google Earth Engine
- Date range selection — compare the same region across different time periods
- Multi-band processing: RGB + Near-Infrared (NIR)

### 🌿 NDVI Vegetation Health Mapping
- Computes NDVI from NIR and Red bands
- Color-coded output: brighter green = healthier, denser forest
- Quick visual indicator of where vegetation is stressed or missing

### 🤖 AI Segmentation (Two Models)
- **YOLOv8-seg** — Real-time forest detection, 2.8 ms/image
- **Mask R-CNN** — Pixel-precise boundary detection, better for scientific use
- Choose the one that fits your use case

### 🌍 Environmental Context
- **Air Quality Index (AQI)** — PM2.5, O3, NO2 levels in the monitored region
- **Wildfire Risk Score** — Derived from temperature, humidity, and wind data
- **Land Classification** — Forest, agriculture, urban, water breakdown

### 📄 PDF Report Generation
- One-click download of a full report
- Includes: maps, NDVI stats, segmentation results, environmental data, recommendations
- Share-ready — clean layout that makes sense to non-technical stakeholders

### 🚀 Production Deployment Options
- ONNX export for 15% faster inference without PyTorch
- Hugging Face Spaces deployment with a Gradio interface
- Cloud-agnostic — works on AWS, GCP, Azure

---

## 🔁 The Process

This wasn't built in a weekend. It was a 7-week structured pipeline, and here's honestly how it went:

**Weeks 1–2 — Getting the data right**
I started by figuring out how to get usable satellite data. Google Earth Engine has everything, but the API takes some getting used to. The main challenge was extracting clean 640×640 patches that were spatially diverse and actually representative of different forest conditions. NDVI computation was straightforward once the band ordering was sorted.

**Week 3 — Training two very different models**
Running YOLOv8 was smooth — the Ultralytics library handles a lot of the boilerplate. Mask R-CNN was more work to configure, especially getting the custom dataset loader to produce the right mask format. The performance gap between the two was smaller than I expected at first.

**Week 4 — Honest evaluation**
This week was humbling. The numbers looked decent on the surface, but drilling into per-class metrics showed the model struggled more with agricultural and residential classes — things that can look similar to degraded forest in satellite imagery. I kept those findings in the results rather than hiding them.

**Week 5 — Does it generalize?**
Testing on the Amazon, Borneo, and Congo Basin revealed real limitations. The model trained mostly on one biome doesn't automatically transfer perfectly to another. This was expected, but it was important to document.

**Week 6 — Ablation: what actually matters?**
Removing NDVI hurt the most. Removing the NIR band hurt almost as much. This confirmed that pure RGB segmentation is meaningfully worse for forest monitoring — the extra satellite bands genuinely carry information that RGB can't replicate.

**Week 7 — Wrapping it up for real use**
ONNX export, Optuna tuning, and the Hugging Face deployment. Getting the Streamlit dashboard to a state I'd actually show someone took more polish than the ML work.

---

## 🧠 What I Learned

Some things I genuinely didn't expect going in:

- **Satellite data is messier than you think.** Cloud cover, atmospheric distortion, and inconsistent band naming across providers are real problems. Data cleaning took longer than model training.

- **NDVI is surprisingly powerful.** I expected it to be a minor addition — it turned out to be one of the most impactful components in the ablation study.

- **YOLOv8 is fast, but Mask R-CNN masks are genuinely cleaner.** The difference is most obvious at forest edges and in regions with fragmented land cover.

- **Optuna is worth the setup time.** Manual hyperparameter tuning is frustrating and inefficient. Letting Optuna run 30 trials and then reading the parameter importance chart is a much better use of time.

- **Building a usable interface is its own skill.** Getting the Streamlit dashboard to a state where someone who knows nothing about the ML pipeline can use it required as much thought as the model work.

- **Geographic generalization is hard.** A model that works on one continent doesn't automatically work on another. Domain adaptation and fine-tuning on local data matters.

---

## 📈 Overall Growth

Looking back at where this project started vs. where it ended up:

- Went from zero experience with geospatial Python to being comfortable with Rasterio, Folium, and Google Earth Engine's API
- Got hands-on with two fundamentally different segmentation architectures and understood their real tradeoffs beyond just benchmark numbers
- Learned how to think about model evaluation honestly — not just reporting the best number, but understanding *why* it is what it is
- Built something end-to-end: from raw satellite imagery to a deployed interface that anyone can open in a browser
- Gained a much deeper appreciation for how much work goes into real-world ML systems vs. notebook experiments

This project pushed me further into production ML thinking — not just "does the model work" but "can someone actually use this, and what breaks when they try."

---

## 🔮 How It Can Be Improved

Honestly, there's a lot still to do. Here's what I'd tackle next:

**Model improvements:**
- Fine-tune on forest-specific labeled data rather than relying on COCO pre-training
- Try newer architectures — SAM (Segment Anything), YOLOv9, or a transformer-based segmentation model
- Train a dedicated change-detection model rather than comparing individual segmentation outputs

**Data improvements:**
- Add more geographic diversity to the training set
- Include time-series patches so the model learns temporal change patterns
- Add synthetic data augmentation for cloud cover and seasonal variation

**System improvements:**
- Real-time alerts when deforestation is detected in a monitored region
- A proper database-backed history so you can track a region over months or years
- Better handling of large ROIs — currently the pipeline can struggle with very large polygons
- Mobile-friendly dashboard layout

**Deployment improvements:**
- Automated retraining pipeline when new labeled data becomes available
- API endpoint so other tools can query the model programmatically
- Proper authentication if this were ever deployed for multiple users

---

## 🚀 Running the Project

Everything you need to get this running from scratch:

### Prerequisites
- Python 3.11 or higher
- `pip` (comes with Python)
- A terminal
- Optional but recommended: a GPU (CUDA or Apple MPS) for faster inference

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/forest-monitoring.git
cd forest-monitoring
```

### Step 2 — Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Launch the dashboard
```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501`. The pre-trained YOLOv8 weights are already included — no training needed.

### Step 5 — (Optional) Run the full ML pipeline
If you want to train from scratch or run the evaluation pipeline, follow the [week-by-week execution steps](#week-by-week-execution-summary) below.

### Verify everything is set up correctly
```bash
# Check all code compiles
python3 -m compileall -q src scripts deployment app.py

# Run tests
pytest -q

# Confirm model weights exist
ls -lh outputs/yolov8_seg/weights/best.pt
```


## 🎬 Demo Video

> A walkthrough of the dashboard — drawing a region, fetching satellite data, viewing the segmentation results, and downloading the PDF report.

📹 **[Watch the Demo on Google Drive](https://drive.google.com/file/d/19ta8GgF1wSXdQCdAajh4qC0m3_XcifbG/view?usp=sharing)**

---

## 📄 Results Report

The full PDF report — covering model evaluation metrics, NDVI analysis, segmentation results, and environmental data — is available here:

📊 **[View the Results PDF](https://drive.google.com/file/d/1DjNRD-mG7P-bwxzJyu_D2QXvBgOBkoTY/view?usp=sharing)**

---
