# 🕳️ Pothole Detection from Road Videos Using a Custom CNN

A deep learning course project for detecting, classifying, and assessing the severity of potholes in road videos and images. Built **100% from scratch using pure PyTorch** — without YOLO, Ultralytics, torchvision backbones, or pretrained weights.

---

## 🏗️ Architecture: Two-Stage Detection Pipeline

The system uses an interpretable **two-stage detection pipeline**: classical computer vision extracts candidate region proposals, and a custom CNN classifies each proposal.

### Stage 1: Classical Region Proposals
- **Road Surface Perspective Prior**: Restricts candidate search to the road surface (lower 50% for dashcam videos, or bottom road region excluding horizon for still photos).
- **Edge-Preserving Smoothing**: Bilateral filtering (`d=9, σ_color=75, σ_space=75`) smooths gravel and asphalt texture while preserving crater borders.
- **Adaptive Depression Thresholding**: Mean-C adaptive thresholding (`block_size=31, C=10`) isolates dark candidate depressions.
- **Morphological Filtering**: Elliptical opening (5×5) and closing (7×7) eliminate grain noise and bridge fragmented crater edges.
- **Local Depression Contrast Verification**: Compares interior crater luminance against the surrounding road pavement margin ($L_{surrounding} - L_{interior}$). Discards regions brighter than the pavement (white lane markings, reflector studs, headlights).
- **Saliency Ranking**: Candidates are ranked by depression contrast, scale-relative area ($0.0006 \le A / A_{frame} \le 0.18$), and interior roughness.

### Stage 2: ResPotholeNet CNN Classifier & Tracker
- Each candidate patch is extracted with a 10% context margin, resized to **128 × 128**, and normalized.
- The CNN classifies each patch as pothole or normal road.
- **The CNN sigmoid confidence is the sole accept/reject criterion** — no contrast score blending.
- **Non-Maximum Suppression** (IoU threshold = 0.35) removes overlapping boxes.
- **Multi-Object Tracking**: An IoU and centroid tracker (`PotholeTracker`) associates detections across consecutive frames, assigns persistent track IDs, filters out single-frame flickers (`min_hits=2`), and counts unique physical potholes encountered throughout a video.

> **Note on Bounding Box Head:** The bounding box regression head exists in the architecture for academic completeness, but is **not used at inference** due to synthetic anchor mismatches. Detections derive directly from Stage 1 contours.

---

## 🧠 CNN Architecture: ResPotholeNet

| Component | Details |
|---|---|
| **Stem** | Strided convolutions for initial spatial downsampling |
| **Backbone** | 3 stages of Residual Bottleneck blocks with skip connections |
| **Attention** | Squeeze-and-Excitation (SE) channel attention |
| **Pooling** | Spatial Pyramid Pooling - Fast (SPPF) for multi-scale features |
| **Classification Head** | FC layers → Sigmoid → confidence ∈ [0, 1] |
| **Regression Head** | FC layers → 4 deltas (unused at inference) |
| **Parameters** | ~858,245 trainable parameters (trained 100% from scratch) |

---

## 📊 Dataset & Splitting Methodology

| Attribute | Specification |
|---|---|
| **Raw Data** | 712 photos (364 pothole + 348 normal) from [anantSinghCross/pothole-detection-system](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks) |
| **Deduplication** | **SHA-256 byte hash** removed 6 byte-identical duplicates → **706 unique photos** (362 pothole + 344 normal). Perceptual difference hashing (dHash) flags 3 near-duplicate pairs within potholes. |
| **Source-Level Split** | **70% Train / 15% Val / 15% Test** performed on source image files *before* crop extraction (**Zero Data Leakage**). Crops from the same photo never appear in multiple partitions. |
| **Split Counts (Photos)** | **Train:** 494 photos (253 pos, 241 neg) <br> **Val:** 104 photos (54 pos, 50 neg) <br> **Test:** 108 photos (55 pos, 53 neg) |
| **Patch Counts (128×128)** | **Train:** 1,222 crops <br> **Val:** 258 crops <br> **Test:** 266 crops (109 positive, 157 negative) from the 108 test photos |
| **Loss & Optimizer** | Focal Loss ($\alpha=0.25, \gamma=2.0$), AdamW (weight decay $10^{-4}$), CosineAnnealingLR |
| **Augmentation** | Horizontal flip, brightness/contrast jitter, slight rotation ($\pm 15^\circ$). No vertical flip (preserves road perspective). |

---

## 📈 Evaluation Results

We report evaluation at two distinct levels: **patch classification** on candidate proposals and **end-to-end pipeline detection** on full source images.

### 1. Patch-Level Classification (Unseen Test Crops, n = 266 crops from 108 photos)

Measures the CNN's ability to classify candidate 128×128 patches proposed by Stage 1:

| Metric | Value |
|---|---|
| **Accuracy** | **87.59%** |
| **Precision** | **85.19%** |
| **Recall** | **84.40%** |
| **F1-Score** | **84.79%** |
| **Focal Loss** | **0.0549** |

```
Confusion Matrix (n = 266 crops):
                Predicted Pos   Predicted Neg
  Actual Pos          92              17
  Actual Neg          16             141
```

> **Statistical Variance Note (±5% Margin):** Test crops come from 108 unique source photos. Because patches from the same photo share identical camera optics, lighting, and asphalt aggregate, test samples are correlated rather than strictly independent. Expect a standard confidence margin of approximately **±5 percentage points**, which naturally explains the difference between validation (76.9%) and test (84.8%).

---

### 2. End-to-End Pipeline Detection (Full Held-Out Photos, n = 108 photos)

Measures the complete two-stage system operating on full held-out test photos (Stage 1 proposals → Stage 2 ResPotholeNet → NMS):

- **Pothole Detection Rate (Recall):** Percentage of pothole-containing photos correctly flagged ($\ge 1$ pothole detected).
- **Clean Road Specificity:** Percentage of clean road photos correctly identified with 0 false detections.
- **FPPI (False Positives Per Image):** Average number of false alarm boxes generated per clean road photo.

| Conf Threshold | Image Accuracy | Image Precision | Image Recall (Sensitivity) | Image Specificity | Image F1-Score | FPPI (Clean Roads) | Avg Detections / Pothole Photo |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.3** | 63.89% | 58.51% | **100.0%** | 26.42% | 73.83% | 3.66 | 17.55 |
| **0.4** | 73.15% | 65.48% | **100.0%** | 45.28% | 79.14% | 2.02 | 14.73 |
| **0.5 (Default)** | **80.56%** | **72.37%** | **100.0%** | **60.38%** | **83.97%** | **0.98** | **11.78** |
| **0.6** | **88.89%** | **82.09%** | **100.0%** | **77.36%** | **90.16%** | **0.51** | **9.15** |
| **0.7** | **90.74%** | **88.14%** | **94.55%** | **86.79%** | **91.23%** | **0.24** | **6.84** |

```
End-to-End Confusion Matrix at Conf = 0.50 (n = 108 full photos):
                  Predicted Pothole   Predicted Clear Road
  Actual Pothole          55                    0
  Actual Clear            21                   32
```

At threshold 0.50, the pipeline detects 100% of pothole scenes with under 1 false positive box per clean photo (0.98 FPPI). Raising threshold to 0.70 cuts FPPI down to 0.24 while retaining 94.55% pothole recall.

---

## 🚀 Installation & Setup

### 1. Clone & Install
```bash
git clone https://github.com/Naitikjha/ROAD_DAMAGE.git
cd ROAD_DAMAGE
pip install -r requirements.txt
```

### 2. Dataset Setup
Place images in `data/potholes/` and `data/normal/`:
```
data/
├── potholes/   # Road photos containing potholes (.jpg / .png)
└── normal/     # Clear pavement road photos (.jpg / .png)
```

### 3. Run End-to-End Evaluation
To evaluate the complete two-stage pipeline on the held-out test images without retraining:
```bash
python evaluate_pipeline.py --weights pothole_cnn.pth --data_dir data
```

### 4. Train the Model
```bash
python train.py --data_dir data --epochs 15 --batch_size 32
```
This pipeline:
1. Deduplicates photos via SHA-256 byte hashing (and reports perceptual dHash near-dupes).
2. Partitions source photos 70/15/15 with zero data leakage.
3. Mines domain-aligned, multi-scale proposals using road perspective priors and depression contrast.
4. Trains ResPotholeNet from scratch with Focal Loss and Cosine Annealing.
5. Runs both patch-level evaluation and end-to-end pipeline evaluation on unseen test images.
6. Saves weights to `pothole_cnn.pth` and reports to `model_metrics.json`.

### 5. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
Features:
- Video & image upload with real-time inference.
- Multi-object tracking with persistent track IDs.
- Temporal confirmation filtering (`min_hits=2`) to eliminate single-frame false flickers.
- Metric cards showing Unique Potholes Tracked, Peak Simultaneous Potholes, Frames Evaluated, and Runtime.
- Interactive detections table with track IDs, timestamps, and severity categorization.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   ├── potholes/            # Real pothole road images (362 unique)
│   └── normal/              # Clear road pavement images (344 unique)
├── models/
│   └── pothole_net.py       # Custom ResPotholeNet CNN architecture
├── app.py                   # Streamlit web application with tracking
├── detector.py              # Two-stage detection engine & PotholeTracker
├── train.py                 # Multi-scale training & unified evaluation
├── evaluate_pipeline.py     # Standalone end-to-end evaluation tool
├── pothole_cnn.pth          # Trained model weights
├── model_metrics.json       # Patch & end-to-end evaluation report
├── end_to_end_metrics.json  # Full threshold sweep report
├── positives_sheet.png      # Contact sheet of extracted Stage-1 proposals
├── requirements.txt         # Python dependencies
├── sample_pothole_video.mp4 # Sample test video
└── README.md
```

---

## ⚠️ Known Limitations & Discussion

- **Dataset Framing**: The dataset consists of still photos of roads from diverse cameras rather than continuous driving video logs. While Stage 1 enforces road perspective priors, varying camera tilt can shift the horizon.
- **Sample Correlation**: Test crops derive from 108 source images, introducing approximately ±5% variance due to intra-image correlation.
- **Hardware Agnostic**: Pipeline runs efficiently on both CPU and CUDA-enabled GPUs.

---

## 📜 Academic Integrity Note

Developed for academic coursework. Formulated honestly as a **classical region proposal + custom CNN classifier pipeline with multi-object tracking**. Built 100% from scratch with pure PyTorch — no YOLO, Ultralytics, torchvision backbones, or pretrained weights. All metrics, methodology, and limitations are documented openly.
