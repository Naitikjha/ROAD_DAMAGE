# 🕳️ Pothole Detection from Road Videos Using a Custom CNN

A deep learning course project for detecting, classifying, and tracking potholes in road videos and images. Built **100% from scratch using pure PyTorch** (`torch.nn`) — without YOLO, Ultralytics, torchvision backbones, or pretrained weights in the primary architecture.

---

## 🏗️ Architecture: Two-Stage Detection Pipeline

The system employs a two-stage detection pipeline: classical computer vision nominates candidate region proposals, and a custom CNN classifies each proposal.

```
Input Video/Image
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Classical Region Proposals (OpenCV)          │
│ • Adaptive Road Surface ROI ('auto' dashcam/photo)     │
│ • Bilateral Edge-Preserving Filter (d=9, σ=75, σ=75)   │
│ • Adaptive Mean-C Thresholding (block=31, C=10)        │
│ • Elliptical Morphological Opening & Closing           │
│ • Local Depression Contrast & Edge Anisotropy Filter   │
└────────────────────────────────────────────────────────┘
       │ [Candidate Bounding Boxes]
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: ResPotholeNet CNN Classifier (PyTorch)        │
│ • 128×128 Normalized Patches (10% context margin)      │
│ • 3-Stage Residual Bottlenecks + Skip Connections      │
│ • Squeeze-and-Excitation (SE) Channel Attention        │
│ • Spatial Pyramid Pooling Fast (SPPF)                  │
│ • Sigmoid Confidence ∈ [0, 1] (Sole Accept Criterion)  │
│ • Non-Maximum Suppression (IoU = 0.35)                 │
└────────────────────────────────────────────────────────┘
       │ [Confirmed Bounding Boxes]
       ▼
┌────────────────────────────────────────────────────────┐
│ Temporal Tracking & Explainability                     │
│ • Kalman Filter Multi-Object Tracking (SORT ICIP 2016) │
│ • 7D State Vector [u, v, s, r, u_dot, v_dot, s_dot]^T   │
│ • Bipartite Greedy IoU Association (min_hits=2)        │
│ • Grad-CAM XAI Saliency Visualizations                 │
└────────────────────────────────────────────────────────┘
```

### Stage 1: Classical Region Proposals
- **Adaptive Road Surface ROI**: Dynamically adjusts search area based on frame aspect ratio (`roi_mode="auto"` — lower 45–100% for wide dashcam footage; 15–95% for still photos).
- **Edge-Preserving Smoothing**: Bilateral filtering (`d=9, σ_color=75, σ_space=75`) smooths coarse gravel aggregate while preserving crater edges.
- **Adaptive Depression Thresholding**: Mean-C adaptive thresholding (`block_size=31, C=10`) isolates dark depression contours.
- **Morphological Filtering**: Elliptical opening ($5\times 5$) and closing ($7\times 7$) eliminate salt-and-pepper grain and bridge fragmented crater boundaries.
- **Local Depression Contrast & Anisotropy**: Compares candidate interior luminance against the surrounding road margin. Discards clipped boxes, highly saturated colorful objects (vegetation/vehicles), and dominant vertical slats (fences/facades).
- **Saliency Ranking**: Candidates are ranked by depression contrast, relative area ($0.0008 \le A/A_{frame} \le 0.18$), and texture variance.

### Stage 2: ResPotholeNet CNN Classifier & Tracker
- Each candidate proposal is cropped with a 10% context margin, resized to **128 × 128**, and normalized.
- **The CNN sigmoid confidence is the sole accept/reject criterion** — no heuristic contrast blending.
- **Non-Maximum Suppression (NMS)** (IoU threshold = 0.35) prunes duplicate overlapping proposals.
- **Kalman Multi-Object Tracking (SORT)**: `PotholeTracker` implements the SORT algorithm (Bewley et al., ICIP 2016) with a 7D state vector $[u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$, greedy bipartite IoU matching, `min_hits=2` temporal flicker suppression, and `max_age=8` track retirement. Aligns with **Unit III Kalman Filter** course topic.

---

## 🧠 CNN Architecture: ResPotholeNet

| Component | Architecture Details |
|---|---|
| **Input** | $3 \times 128 \times 128$ RGB road patch |
| **Stem** | Conv 3→32 (stride 2) + Conv 32→64 (stride 1) + BatchNorm + SiLU ($128 \rightarrow 64$) |
| **Stage 1** | Down Conv 64→64 (stride 2) + 2× ResBlocks with SE Attention ($64 \rightarrow 32$) |
| **Stage 2** | Down Conv 64→128 (stride 2) + 2× ResBlocks with SE Attention ($32 \rightarrow 16$) |
| **Stage 3** | Down Conv 128→256 (stride 2) + SPPF Module (multi-scale receptive field) ($16 \rightarrow 8$) |
| **Head** | AdaptiveAvgPool2d(1, 1) → Linear(256, 128) → Dropout(0.3) → Linear(128, 32) → Linear(32, 1) → Sigmoid |
| **Trainable Params** | **853,985 parameters** (100% custom from scratch, zero pretrained weights) |

> **Note on Regression Head:** The model is strictly a patch classifier. The bounding box regression head was removed because the underlying dataset only contains image-level labels without coordinate annotations. All 853,985 parameters serve classification exclusively.

---

## 🔬 Explainable AI (XAI): Grad-CAM

[`models/gradcam.py`](models/gradcam.py) implements **Gradient-weighted Class Activation Mapping** (Selvaraju et al., ICCV 2017) to satisfy the course **Unit IV / XAI** requirement:
- Hooks into the deepest SPPF convolution layer (`model.sppf.cv2.conv`).
- Computes gradient-weighted activation heatmaps to reveal which visual features (crater borders, shadow gradients, asphalt texture contrast) drive pothole predictions.
- Integrated directly into the Streamlit UI with interactive heatmap and visual attention overlay displays.

---

## 📊 Pretrained Baseline Comparison (Unit II: Transfer Learning)

[`baseline.py`](baseline.py) fine-tunes a torchvision **ResNet-18 (ImageNet-1k pretrained)** on the identical dataset partition and patch crops, satisfying the **Unit II Model Adaptation** requirement:

| Metric | ResPotholeNet (From Scratch) | Pretrained ResNet-18 (Fine-Tuned) |
|---|---|---|
| **Training Paradigm** | Random initialization (Kaiming normal) | ImageNet-1k pretrained + fine-tuned |
| **Trainable Parameters** | **853,985** (1x) | **11,180,609** (~13.1x heavier) |
| **Test Accuracy** | 86.93% (95% CI: [81.5%, 90.9%]) | 81.41% (95% CI: [75.4%, 86.2%]) |
| **Test Precision** | 92.55% | 85.11% |
| **Test Recall** | 82.08% | 78.43% |
| **Test F1-Score** | 87.00% | 81.77% |
| **Inference Latency** | ~2.5 ms / patch | ~4.1 ms / patch |

> ⚠️ **Statistical Significance Note:** With $n=199$ test crops from 108 source photos, the 95% Wilson confidence intervals overlap (81.5% vs 86.2%). **We do not claim that ResPotholeNet outperforms ResNet-18.** Rather, the custom model demonstrates that a lightweight, domain-tailored architecture (~854k parameters) achieves comparable classification capability with 13x fewer parameters and zero reliance on external ImageNet weights.

---

## 📊 Dataset & Splitting Methodology

| Attribute | Specification |
|---|---|
| **Raw Source Photos** | 712 photos (364 pothole, 348 normal) |
| **Deduplication** | SHA-256 byte hashing (6 exact duplicates removed) + perceptual dHash (Hamming $\le 2$, 1 near-duplicate removed) + **cross-class deduplication** $\rightarrow$ **705 clean unique source photos** |
| **Deterministic Split** | Filenames sorted before splitting: **70% Train (492) / 15% Val (105) / 15% Test (108 photos)** |
| **Data Leakage Prevention** | Source photo level partitioning guarantees patches from the same photo never cross train/val/test splits. |
| **Symmetric Sampling** | Top 2 Stage-1 candidate proposals per photo extracted symmetrically for both positive and negative classes using `roi_mode="auto"`. |
| **Patch Counts (128×128)** | Train: 915 crops \| Val: 196 crops \| Test: 199 crops (106 positive, 93 negative) |
| **Loss & Optimizer** | Focal Loss ($\alpha=0.25, \gamma=2.0$), AdamW (weight decay $10^{-4}$), Cosine Annealing learning rate schedule. |

---

## 📈 Multi-Level Empirical Evaluation

To maintain scientific integrity, performance is evaluated at three distinct levels:

### 1. Patch-Level Classification ($n = 199$ unseen crops from 108 photos)

| Metric | Measured Value | Wilson 95% Confidence Interval |
|---|---|---|
| **Accuracy** | **86.93%** (173 / 199) | [81.54%, 90.93%] |
| **Precision** | **92.55%** (87 / 94) | [85.42%, 96.35%] |
| **Recall** | **82.08%** (87 / 106) | [73.69%, 88.21%] |
| **F1-Score** | **87.00%** | — |
| **Focal Loss** | **0.0364** | — |

```
Patch Confusion Matrix:
                Predicted Pothole   Predicted Normal
  Actual Pothole       87                  19
  Actual Normal         7                  86
```

---

### 2. End-to-End Image-Level Screening ($n = 108$ full held-out photos)

Operating threshold selected strictly on the **Validation set** (0.50), then evaluated **once** on the unseen test set:

| Metric | Test Set Value | Description |
|---|---|---|
| **Image-Level Accuracy** | **87.04%** (94 / 108) | Photos correctly categorized as pothole vs normal |
| **Image-Level Precision** | **79.71%** (55 / 69) | Precision when at least one box is detected |
| **Image-Level Recall** | **100.0%** (55 / 55) | 95% CI: [93.47%, 100.0%] |
| **Image Specificity** | **73.58%** (39 / 53) | Clean roads with zero false alarm boxes (95% CI: [60.42%, 83.56%]) |
| **Clean Road FPPI** | **0.642** | Average false positive boxes per clean road photo |
| **Avg Boxes / Pothole Photo** | **7.35** | Average candidate boxes triggered per pothole photo |

```
Image-Level Screening Confusion Matrix:
                    Predicted Pothole   Predicted Clear Road
  Actual Pothole          55                    0
  Actual Clear            14                   39
```

---

### 3. Box-Level Ground-Truth Detection Benchmark (IoU $\ge 0.50$, $n = 30$ photos)

Evaluated using [`evaluate_box_level.py`](evaluate_box_level.py) on the 30 manually annotated ground-truth photos in [`data/test_ground_truth_boxes.json`](data/test_ground_truth_boxes.json):

| Metric | Measured Value | Meaning & Context |
|---|---|---|
| **Annotated Test Images** | **30** | Hand-annotated subset of test pothole photos |
| **Ground-Truth Pothole Boxes** | **30** | Exactly 1 reference box per test photo |
| **True Positive Boxes (IoU $\ge 0.50$)** | **2** | Only 2 predicted boxes achieved $\ge 50\%$ IoU |
| **False Positive Boxes (Pothole Photos)** | **241** | High candidate count on complex road textures |
| **False Negative Boxes (Missed)** | **28** | Boxes that failed to meet 0.50 IoU overlap |
| **Clean Road False Alarm Boxes** | **17** | False positive boxes across 30 clean road photos |
| **Box Precision** | **0.82%** | Due to heavy multi-candidate proposal generation |
| **Box Recall** | **6.67%** | 95% CI: [1.85%, 21.32%] |
| **Box F1-Score** | **1.47%** | Harmonic mean of box precision and recall |
| **Mean IoU (Matched True Positives)** | **56.10%** | Quality of overlap on the 2 successful matches |
| **mAP @ IoU 0.50 (AP_50)** | **0.20%** | Average precision under standard Pascal VOC metric |

---

## 🔍 Failure Diagnosis & Root Cause Analysis

### Why is Box-Level Localization (6.7%) Weak Despite High Image Recall (100%)?

We conducted an empirical ablation to determine whether the box-level failure stems from Stage 1 classical proposals or the Stage 2 CNN classifier:

1. **Stage 1 Recall Ceiling (Root Cause)**:
   - When running Stage 1 region proposals **alone** against the ground-truth boxes with no CNN filtering, only **2 of the 30 ground-truth boxes achieved an IoU $\ge 0.50$** (Stage 1 candidate recall ceiling = **6.67%**, mean maximum IoU = **11.1%**).
   - Because the CNN can only classify candidates nominated by Stage 1, **Stage 1 is the primary bottleneck for box localization**. Classical adaptive thresholding fragments crater boundaries and isolates small dark sub-regions rather than the complete depression.
2. **Image-Level Recall vs Localization**:
   - The 100% image-level recall is an expected artifact of multi-candidate nomination: since the pipeline produces an average of 7.35 passing boxes per pothole photo, at least one box triggers on every photo. This confirms frame screening capability, but **not** precise object boundary localization.
3. **False Alarms on Clean Roads**:
   - 26.4% of clean road photos trigger at least one false positive box at threshold 0.50 (0.642 FPPI). Asphalt patches, deep pavement cracks, and harsh shadow seams resemble depressions to classical filters.
4. **Proxy Supervision Label Noise**:
   - Because the training set lacks bounding box coordinates, positive training patches were extracted using the top-2 Stage-1 classical proposals. Visual examination of [`positives_sheet.png`](positives_sheet.png) reveals that **only ~10 of 24 crops (~42%) depict true potholes**, while **~14 crops (~58%) capture non-pothole road noise** (shadows, cracks, dark asphalt, road boundaries). This label noise naturally limits classifier precision.
5. **Severity Metric Disclaimer**:
   - Severity categorization (Low / Medium / High) in the application is an unvalidated heuristic based strictly on relative bounding box pixel area ($A < 2\%$, $2\% \le A < 6\%$, $A \ge 6\%$). No ground-truth depth or physical dimension measurements exist in the dataset.

---

## 🚀 Installation & Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/Naitikjha/ROAD_DAMAGE.git
cd ROAD_DAMAGE
pip install -r requirements.txt
```

### 2. Dataset Structure
```
data/
├── potholes/                   # Pothole road images (.jpg / .png)
├── normal/                     # Clear road pavement images (.jpg / .png)
├── test_ground_truth_boxes.json # 30 hand-annotated test bounding boxes
└── test_images_meta.json       # Metadata for test partitions
```

### 3. Run the Streamlit Application (Localhost)
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
- **Upload video or image**: Real-time detection with Kalman multi-object tracking.
- **Grad-CAM XAI**: Expand the explainability panel to inspect class activation heatmaps and visual attention overlays.
- **Tracking Analytics**: Persistent track IDs, flicker filtering (`min_hits=2`), and unique pothole counts.

### 4. Run Evaluations & Benchmarks

```bash
# 1. Box-Level Ground Truth Evaluation (IoU >= 0.50 on annotated subset)
python evaluate_box_level.py

# 2. End-to-End Pipeline Evaluation (Full test images screening)
python evaluate_pipeline.py --weights pothole_cnn.pth --data_dir data

# 3. Pretrained ResNet-18 Transfer Learning Baseline (Unit II)
python baseline.py --epochs 8 --batch_size 32

# 4. Retrain Model from Scratch
python train.py --data_dir data --epochs 15 --batch_size 32
```

---

## 📁 Repository Structure

```
├── .gitignore               # Ignores raw images but tracks data/*.json annotations
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── app.py                   # Streamlit web app (Kalman SORT tracking + Grad-CAM)
├── detector.py              # Two-stage detection engine + KalmanBoxTracker + PotholeTracker
├── train.py                 # Training pipeline (cross-class dedup + auto ROI + Focal Loss)
├── baseline.py              # ResNet-18 fine-tuning baseline (Unit II)
├── evaluate_pipeline.py     # End-to-end full image evaluation
├── evaluate_box_level.py    # Box IoU evaluation on ground-truth annotations
├── pothole_cnn.pth          # Sanitized trained ResPotholeNet weights (~854k params)
├── model_metrics.json       # Patch-level evaluation metrics
├── end_to_end_metrics.json  # Full validation sweep & test metrics
├── box_level_metrics.json   # Detailed box-level benchmark metrics
├── baseline_comparison.json # ResNet-18 vs ResPotholeNet comparison metrics
├── positives_sheet.png      # Contact sheet of extracted Stage-1 proposals
├── sample_gradcam.png       # Sample Grad-CAM attention overlay
├── sample_pothole_video.mp4 # Sample dashcam video for testing
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation and evaluation report
```

---

## 📜 Academic Integrity Note

Developed for academic coursework. Formulated honestly as a **classical region proposal + custom CNN classifier pipeline with Kalman multi-object tracking and Grad-CAM explainability**. Built 100% from scratch with pure PyTorch — no YOLO, Ultralytics, torchvision backbones, or pretrained weights in the primary architecture. A fine-tuned ResNet-18 baseline is provided for Unit II comparison. All limitations, including box-level candidate recall ceilings, clean road false alarms, and label noise from proxy supervision, are documented openly without inflated claims.
