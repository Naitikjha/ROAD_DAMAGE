# 🕳️ Pothole Detection from Road Videos Using a Custom CNN

A deep learning course project for detecting, classifying, and assessing the severity of potholes in road videos and images. Built **100% from scratch using pure PyTorch** — without YOLO, Ultralytics, torchvision backbones, or pretrained weights.

---

## 🏗️ Architecture: Two-Stage Detection Pipeline

The system uses an interpretable **two-stage detection pipeline**: classical computer vision extracts candidate region proposals, and a custom CNN classifies each proposal.

### Stage 1: Classical Region Proposals
- **Road Surface Perspective Prior**: Restricts candidate search to the road surface (lower 50% for dashcam videos, or bottom road region excluding horizon and bottom watermark borders for still photos).
- **Edge-Preserving Smoothing**: Bilateral filtering (`d=9, σ_color=75, σ_space=75`) smooths gravel and asphalt texture while preserving crater borders.
- **Adaptive Depression Thresholding**: Mean-C adaptive thresholding (`block_size=31, C=10`) isolates dark candidate depressions.
- **Morphological Filtering**: Elliptical opening (5×5) and closing (7×7) eliminate grain noise and bridge fragmented crater edges.
- **Local Depression Contrast & Border Filtering**: Compares interior crater luminance against the surrounding road pavement margin ($L_{surrounding} - L_{interior}$). Discards regions clipped against frame boundaries (cars, watermarks, shoes) or brighter than pavement.
- **Saliency Ranking**: Candidates are ranked by depression contrast, scale-relative area ($0.0008 \le A / A_{frame} \le 0.18$), and interior roughness.

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
| **Deduplication** | **SHA-256 byte hash** removed 6 byte-identical duplicates. **Perceptual dHash** (Hamming distance $\le 2$) removed 1 near-duplicate pair $\rightarrow$ **705 clean unique source photos** (361 pothole + 344 normal). |
| **Deterministic Split** | File paths are `sorted()` before splitting to guarantee 100% cross-platform reproducibility. Split is **70% Train / 15% Val / 15% Test** on source photos (**Zero Data Leakage**). |
| **Split Counts (Photos)** | **Train:** 492 photos (252 pos, 240 neg) <br> **Val:** 105 photos (54 pos, 51 neg) <br> **Test:** 108 photos (55 pos, 53 neg) |
| **Patch Counts (128×128)** | **Train:** 915 crops <br> **Val:** 196 crops <br> **Test:** 199 crops (106 positive, 93 negative) from the 108 test photos |
| **Symmetric Sampling** | Both positive and negative classes are sampled with the **exact same rule** (top 2 Stage-1 candidate proposals per photo) to eliminate size and rank shortcuts. |
| **Loss & Optimizer** | Focal Loss ($\alpha=0.25, \gamma=2.0$), AdamW (weight decay $10^{-4}$), CosineAnnealingLR |
| **Augmentation** | Horizontal flip, brightness/contrast jitter, slight rotation ($\pm 15^\circ$). No vertical flip (preserves ground perspective). |

---

## 📈 Evaluation Results

We report evaluation at two distinct levels: **patch classification** on candidate proposals and **end-to-end pipeline detection** on full source images.

### 1. Patch-Level Classification (Unseen Test Crops, n = 199 crops from 108 photos)

Measures the CNN's ability to classify candidate 128×128 patches proposed by Stage 1:

| Metric | Value |
|---|---|
| **Accuracy** | **86.93%** |
| **Precision** | **92.55%** |
| **Recall** | **82.08%** |
| **F1-Score** | **87.00%** |
| **Focal Loss** | **0.0364** |

```
Patch Confusion Matrix (n = 199 crops):
                Predicted Pos   Predicted Neg
  Actual Pos          87              19
  Actual Neg           7              86
```

> **Statistical Variance Note (±5% Margin):** Test crops originate from 108 unique source photos. Because patches from the same photo share identical camera optics, lighting, and asphalt aggregate, test samples are correlated rather than strictly independent. Expect an estimated standard confidence margin of approximately **±5 percentage points**, accounting for variations between validation (85.4%) and test (87.0%).

---

### 2. End-to-End Pipeline Evaluation (Full Held-Out Photos, n = 108 photos)

Measures the complete two-stage system operating on full held-out test photos (Stage 1 proposals → Stage 2 ResPotholeNet → NMS).

#### Threshold Selection Methodology
To avoid test-set tuning bias, the optimal operating threshold is **selected strictly on the VALIDATION partition** by sweeping candidate thresholds $\in [0.3, 0.8]$ to maximize Validation F1 score:
- **Optimal Validation Threshold:** **0.50** (Validation F1 = 82.81%, Recall = 98.15%, Specificity = 58.82%).
- The held-out **test set is evaluated ONCE** at this validation-selected threshold (0.50).

#### Test Set Performance at Validation-Selected Threshold (0.50)

| Metric | Test Set Value | Description |
|---|---|---|
| **Image-Level Accuracy** | **87.04%** | Percentage of all test photos correctly categorized |
| **Image-Level Precision** | **79.71%** | Precision among photos flagged as containing potholes |
| **Image-Level Recall** | **100.0%** | 55 of 55 pothole photos have $\ge 1$ detection |
| **Image Specificity** | **73.58%** | 39 of 53 clean road photos correctly have 0 detections |
| **Image-Level F1-Score** | **88.71%** | Harmonic mean of image precision and recall |
| **Clean Road FPPI** | **0.642** | Average false positive boxes per clean road photo |
| **Avg Pothole Detections**| **7.35** | Average bounding boxes per pothole-containing photo |

```
Image-Level Confusion Matrix at Threshold 0.50 (n = 108 full photos):
                  Predicted Pothole   Predicted Clear Road
  Actual Pothole          55                    0
  Actual Clear            14                   39
```

#### ⚠️ Critical Discussion & Limitations of Image-Level Metrics
- **Surrogate Nature of Image-Level Recall:** A photo counts as "detected" if *any* candidate box passes the threshold. Because pothole photos generate an average of 7.35 passing boxes, 100% image-level recall is an expected artifact of multi-box proposals and **does not prove that the bounding box tightly bounds the pothole**.
- **False Alarms on Clean Roads:** At threshold 0.50, 14 of 53 clean road photos (26.4%) trigger at least one false alarm box (0.642 FPPI), caused by severe cracks, dark asphalt patches, or shadow seams.
- **True Object Detection vs. Image Screening:** Real object detection accuracy (e.g., mAP@0.5) requires hand-annotated ground-truth bounding box coordinates with IoU $\ge 0.5$. Because the underlying dataset provides only image-level labels without coordinate annotations, this end-to-end evaluation measures **coarse frame screening performance**, not tight bounding box localization.

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
To evaluate the complete two-stage pipeline on the held-out test images:
```bash
python evaluate_pipeline.py --weights pothole_cnn.pth --data_dir data
```

### 4. Train the Model
```bash
python train.py --data_dir data --epochs 15 --batch_size 32
```
This pipeline:
1. Deduplicates photos via SHA-256 byte hashing and dHash perceptual deduplication.
2. Sorts file paths and partitions source photos 70/15/15 with zero data leakage.
3. Samples symmetric Stage-1 proposals across both classes (top 2 proposals per photo).
4. Trains ResPotholeNet from scratch with Focal Loss and Cosine Annealing.
5. Selects the optimal threshold on the validation set, then evaluates on unseen test photos.
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
│   ├── potholes/            # Real pothole road images (361 clean)
│   └── normal/              # Clear road pavement images (344 clean)
├── models/
│   └── pothole_net.py       # Custom ResPotholeNet CNN architecture
├── app.py                   # Streamlit web application with tracking
├── detector.py              # Two-stage detection engine & PotholeTracker
├── train.py                 # Symmetric training & validation tuning pipeline
├── evaluate_pipeline.py     # Standalone end-to-end evaluation tool
├── pothole_cnn.pth          # Retrained model weights
├── model_metrics.json       # Complete patch & validation-tuned metrics
├── end_to_end_metrics.json  # Full validation sweep & test metrics
├── positives_sheet.png      # Contact sheet of extracted Stage-1 proposals
├── requirements.txt         # Python dependencies
├── sample_pothole_video.mp4 # Sample test video
└── README.md
```

---

## 📜 Academic Integrity Note

Developed for academic coursework. Formulated honestly as a **classical region proposal + custom CNN classifier pipeline with multi-object tracking**. Built 100% from scratch with pure PyTorch — no YOLO, Ultralytics, torchvision backbones, or pretrained weights. All metrics, methodology, and limitations are documented openly.
