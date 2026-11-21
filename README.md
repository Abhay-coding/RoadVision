# RoadVision: Autonomous Road-Scene Semantic Understanding & Robustness System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-SegFormer-yellow.svg)](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
[![Clean mIoU](https://img.shields.io/badge/Clean%20mIoU-38.98%25-success.svg)]()
[![Pixel Accuracy](https://img.shields.io/badge/Pixel%20Accuracy-94.47%25-success.svg)]()

RoadVision is an evaluation-ready autonomous-driving road-scene understanding system. It provides real-time semantic segmentation on dashcam images and video streams, quantitative confusion-matrix evaluation across 19 Cityscapes classes, and a deterministic 25-condition adverse weather robustness benchmark (Low Light, Atmospheric Fog, Rain, Gaussian Blur, and Gaussian Sensor Noise).

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Single-Image Inference](#1-single-image-inference)
  - [2. Dashcam Video Inference](#2-dashcam-video-inference)
  - [3. Adverse Weather Robustness Testing](#3-adverse-weather-robustness-testing)
  - [4. Quantitative Dataset Evaluation](#4-quantitative-dataset-evaluation)
  - [5. Streamlit Interactive Dashboard](#5-streamlit-interactive-dashboard)
- [Automated Testing](#-automated-testing)
- [Experimental Results](#-experimental-results)
- [Viva Preparation](#-viva-preparation)
- [Known Limitations](#-known-limitations)

---

## 🌟 Key Features

1. **Autonomous Driving Semantic Understanding:**
   - Classifies 19 Cityscapes road categories: Road, Sidewalk, Building, Wall, Fence, Pole, Traffic Light, Traffic Sign, Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle.
2. **Dual Model Support:**
   - Primary: **SegFormer-B0** (`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, 3.7M parameters).
   - Secondary: **DeepLabV3-MobileNetV3** from torchvision for PASCAL VOC object classes. The Cityscapes ground-truth benchmarks use SegFormer.
3. **OpenCV Video Stream Processing:**
   - Dashcam video inference with configurable frame skipping ($k$-factor), latency tracking, and real-time HUD telemetry.
4. **Adverse Environmental Robustness Benchmark:**
   - Deterministic 5-condition corruption pipeline with 5 severity levels: Low Illumination, Koschmieder Fog, Rain Streaks, Optical Blur, and Sensor Shot Noise.
5. **Rigorous Quantitative Metrics:**
   - Confusion-matrix evaluator computing True Positives, False Positives, False Negatives, per-class IoU, Mean IoU (mIoU), Overall Pixel Accuracy, and absolute/relative degradation drops.
6. **Interactive Streamlit Web Dashboard:**
   - Multi-tab application covering Single-Image Segmentation, Video Analysis, Robustness Lab with difference heatmaps, and Model Evaluation plots.

---

## 🏗 System Architecture

```
                             RoadVision System Pipeline
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Input Image / Video    │────▶│ Image Preprocessing &  │────▶│  SegFormer-B0 /        │
│  (Camera / Dashcam)     │     │ Resolution Scaling     │     │  DeepLabV3-MobileNetV3 │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                            │
      ┌─────────────────────────────────────────────────────────────────────┘
      ▼
┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Logits Tensor           │────▶│ Per-Pixel Class Argmax │────▶│ Colorization & Alpha   │
│ (1, 19, H_out, W_out)   │     │ (Bilinear Upsampling)  │     │ Blended Overlay        │
└─────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                            │
      ┌─────────────────────────────────────────────────────────────────────┘
      ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Output Deliverables: Segmented Video HUD, Class Distribution, Metrics, Robustness Lab  │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
RoadVision/
├── app.py                     # Streamlit multi-page interactive dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and guide
├── .gitignore                 # Cache and artifact ignore rules
├── configs/
│   └── default_config.yaml    # System settings, Cityscapes classes, palette & parameters
├── src/
│   ├── __init__.py            # Package initializer
│   ├── inference.py           # RoadVisionSegmentor inference engine
│   ├── video_processor.py     # VideoProcessor OpenCV stream pipeline
│   ├── corruptions.py         # Environmental corruptions (Low Light, Fog, Rain, Blur, Noise)
│   ├── visualization.py       # Mask colorizer, alpha overlay, diff map & legend generator
│   ├── metrics.py             # SegmentationEvaluator & degradation calculator
│   └── dataset.py             # RoadSceneDataset loader & sample video generator
├── scripts/
│   ├── run_inference.py       # Command-line image and video inference runner
│   ├── evaluate.py            # Quantitative evaluation script
│   ├── robustness_test.py     # 25-condition robustness benchmark runner
│   └── create_sample_video.py # Dashcam video generator
├── data/
│   ├── samples/               # Authentic Cityscapes road-scene sample images and videos
│   └── labels/                # Segmentation ground truth masks
├── outputs/
│   ├── predictions/           # Generated masks, overlays, segmented videos & telemetry
│   ├── plots/                 # Per-class IoU, degradation curves & comparison grids
│   └── metrics/               # Machine-readable JSON evaluation logs
├── tests/
│   ├── __init__.py            # Test package initializer
│   ├── test_metrics.py        # Unit tests for confusion matrix, IoU, ignore index
│   ├── test_corruptions.py    # Unit tests for corruption determinism & monotonicity
│   └── test_inference.py      # Integration tests for configuration and model forward pass
└── docs/
    ├── methodology.md         # Detailed mathematical formulation & architecture details
    ├── results.md             # Measured experimental findings & benchmark tables
    └── viva_preparation.md    # 15 complete viva oral questions and answers
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11 or 3.12 (verified on Windows with Python 3.11)
- High-capacity storage recommended for model weight caching

### Clone & Install Dependencies
```powershell
git clone --branch fix/video-playback-ui https://github.com/Zylus08/RoadVision.git
cd RoadVision

# Create an isolated environment and install dependencies
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activate the virtual environment before running the commands below. In PowerShell, run `.\.venv\Scripts\Activate.ps1`. The first inference run downloads the selected model weights. Once the weights are cached, set `$env:HF_HUB_OFFLINE = '1'` to run SegFormer offline. The video pipeline uses FFmpeg from `imageio-ffmpeg` to produce H.264 MP4.

---

## 🚀 Quick Start Guide

### 1. Single-Image Inference
Run segmentation on any road scene image:
```powershell
python scripts/run_inference.py --input data/samples/road_scene_001.png
```
*Outputs generated in `outputs/predictions/`:*
- `road_scene_001_mask.png` (Color-coded mask)
- `road_scene_001_overlay.png` (Alpha blended overlay)
- `road_scene_001_telemetry.json` (Latency, FPS, class breakdown)

### 2. Dashcam Video Inference
Process a dashcam video stream with telemetry HUD:
```powershell
python scripts/run_inference.py --input data/samples/sample_dashcam.mp4 --frame-skip 2
```
*Outputs generated:*
- `outputs/predictions/sample_dashcam_segmented.mp4` (Annotated dashcam video)
- `outputs/predictions/sample_dashcam_video_stats.json` (Frame counts, model latency vs e2e latency)

### 3. Adverse Weather Robustness Testing
Run the complete 25-condition robustness grid (Low light, Fog, Rain, Blur, Noise $\times$ 5 severities):
```powershell
python scripts/robustness_test.py --data-dir data/samples/authentic --label-dir data/labels/authentic --samples 4 --seed 42
```
*Outputs generated:*
- `outputs/metrics/robustness_benchmark.json` (Complete quantitative drop metrics)
- `outputs/plots/robustness_degradation_curves.png` (Degradation curves)
- `outputs/plots/robustness_max_severity_drop.png` (Severity 5 drop comparison)
- `outputs/plots/robustness_visual_comparison_grid.png` (Visual matrix)

### 4. Quantitative Dataset Evaluation
Evaluate model predictions against ground truth labels:
```powershell
python scripts/evaluate.py --data-dir data/samples/authentic --label-dir data/labels/authentic --max-samples 5
```
*Outputs generated:*
- `outputs/metrics/clean_eval.json`
- `outputs/plots/per_class_iou.png`

### 5. Streamlit Interactive Dashboard
Launch the web interface:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Execute the test suite to verify metric correctness, corruption determinism, and inference dimensions:
```powershell
python -m unittest discover -s tests -v
```

Run the deliverable smoke test to validate the published artifacts and execute image inference on an authentic sample.

```powershell
python scripts/smoke_test_deliverables.py
```

Prediction files are generated locally. The evaluation dashboard reads the published benchmark reports and plots checked into Git.

----------------------------------------------------------------------
Ran 19 tests in 10.10s

OK (19/19 Passed)
```

---

## 📊 Experimental Results

> **Evaluation Subset Note:** The quantitative clean baseline and adverse weather robustness benchmark were evaluated on an authentic pilot validation subset consisting of 5 paired driving sequences from the Cityscapes Frankfurt validation set (`frankfurt_000000_000294` through `001751`). Due to CPU execution mode, this 5-scene subset serves as a rigorous pilot validation sanity check rather than a full 500-image Cityscapes benchmark.

The published benchmark uses 512×512 SegFormer input. Interactive inference and the general evaluation command use the current configuration, which defaults to 1024×1024. To reproduce the published settings and save new results locally, run:

```powershell
python scripts/reproduce_clean_eval.py --output-dir outputs/test-runs/metrics
python scripts/run_verified_robustness_benchmark.py --output-dir outputs/test-runs/metrics --plot-dir outputs/test-runs/plots
```

### Benchmark Conventions:
- **Dataset-Level Aggregate mIoU:** **38.98%** (computed from accumulated $19 \times 19$ confusion matrix over $9,117,536$ valid pixels across all 18 evaluated classes).
- **Image-Level Mean mIoU:** **46.79%** (arithmetic mean of individual scene mIoUs: $57.58\%$, $57.45\%$, $47.30\%$, $34.77\%$, $36.85\%$).
- **Overall Pixel Accuracy:** **94.47%** (excluding void/boundary index 255).

### Verified Robustness Performance Matrix:

| Condition | Verified Clean mIoU | Severity 5 mIoU | Relative Degradation (%) | Worst Affected Categories |
|---|---|---|---|---|
| **Atmospheric Fog** | 38.98% | 17.16% | **56.0% Drop** | Horizon, Distant Vehicles, Signs |
| **Low Light** | 38.98% | 29.23% | **25.0% Drop** | Sidewalk, Road Boundaries, Poles |
| **Gaussian Blur** | 38.98% | 29.23% | **25.0% Drop** | Traffic Lights, Signs, Poles |
| **Gaussian Noise** | 38.98% | 31.50% | **19.2% Drop** | Texture Boundaries, Distant Objects |
| **Rain Streaks** | 38.98% | 36.72% | **5.8% Drop** | Small Pedestrians, Lane Markings |

Detailed analysis and curves are provided in [docs/results.md](docs/results.md) and [docs/audit_report.md](docs/audit_report.md).

---

## 🎓 Viva Preparation

A comprehensive oral examination study guide covering all 15 semester laboratory evaluation questions is provided in [docs/viva_preparation.md](docs/viva_preparation.md).

---

## ⚠️ Known Limitations

1. **CPU Execution Throughput:** In the reported CPU benchmark, forward-pass inference runs at ~1.7 to 2.0 FPS (~550 ms) for high-resolution road scenes. GPU acceleration (CUDA) is recommended for real-time (30+ FPS) operations.
2. **Evaluation Sample Scale:** Quantitative metrics are validated on a 5-scene pilot subset. Running the full 500-image Cityscapes validation split on CPU requires several hours.
3. **Temporal Consistency:** Frames in video processing are processed per-frame without optical-flow temporal smoothing, which can produce subtle mask flicker across consecutive frames.
4. **Synthetic Domain Gap:** Synthetic corruptions approximate adverse weather mathematically (Koschmieder atmospheric scattering, directional motion blur streaks), but cannot replicate all complex real-world physical dynamics (e.g., lens water droplets, dynamic windshield wiper motion, road puddle reflections).

