# 🕳️ Two-Stage Road Pothole Screening & Proposal Classification Pipeline Using a Custom CNN

A deep learning course project for coarse frame screening, candidate proposal classification, and multi-object tracking of road surface potholes in dashcam videos and photos. Built using **PyTorch with no pretrained weights in the primary architecture**, combining classical computer vision region proposals with a custom convolutional neural network (ResPotholeNet).

> ⚠️ **Scope & Naming Disclosure (Screening vs. Fine-Grained Object Detection):**  
> This system is designed and evaluated as a **two-stage road screening and patch classification pipeline**, not an anchor-based bounding box regression detector (such as YOLO or Faster R-CNN). The primary model is trained strictly as a binary patch classifier on candidate crops because the underlying dataset provides only image-level labels without coordinate supervision. Accordingly, bounding boxes originate directly from classical Stage-1 morphological contours.

---

## 🏗️ Architecture: Two-Stage Screening Pipeline

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
- **Kalman Multi-Object Tracking (SORT)**: `PotholeTracker` implements the SORT algorithm ([Bewley et al., ICIP 2016](https://arxiv.org/abs/1602.00763)) with a 7D state vector $[u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$, greedy bipartite IoU matching, `min_hits=2` temporal flicker suppression, and `max_age=8` track retirement. Aligns with **Unit III Kalman Filter** course topic.

---

## 🧠 CNN Architecture: ResPotholeNet

Built **without pretrained weights** using published architectural paradigms cited below:

| Component | Published Paradigm & Reference | Implementation Details |
|---|---|---|
| **Residual Blocks** | He et al., CVPR 2016 ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) | 3 stages of bottleneck residual blocks with skip connections to prevent vanishing gradients |
| **Channel Attention** | Hu et al., CVPR 2018 ([arXiv:1709.01507](https://arxiv.org/abs/1709.01507)) | Squeeze-and-Excitation (SE) modules (reduction ratio = 4) to emphasize salient crater contrast |
| **Multi-Scale Pooling** | Jocher et al., 2020 (SPPF) | Spatial Pyramid Pooling - Fast with $5\times 5$ max-pooling kernels to capture varied crater scales |
| **Classification Head** | Dense Fully Connected | AdaptiveAvgPool2d(1, 1) → Linear(256, 128) → Dropout(0.3) → Linear(128, 32) → Linear(32, 1) → Sigmoid |
| **Parameter Count** | **853,985 trainable parameters** | Trained from scratch with Kaiming normal initialization. Zero pretrained weights. |

> **Classification-Only Parameter Budget:** The model does not include a bounding box regression head. Because the dataset lacks coordinate annotations, a regressor could not be supervised. All 853,985 parameters serve patch classification exclusively.

---

## 🔬 Explainable AI (XAI): Grad-CAM

[`models/gradcam.py`](models/gradcam.py) implements **Gradient-weighted Class Activation Mapping** ([Selvaraju et al., ICCV 2017](https://arxiv.org/abs/1610.02391)) to satisfy the course **Unit IV / XAI** requirement:
- Hooks into the deepest SPPF convolution layer (`model.sppf.cv2.conv`).
- Computes gradient-weighted activation heatmaps to reveal which visual features (crater borders, shadow gradients, asphalt texture contrast) drive pothole predictions.
- Integrated directly into the Streamlit UI via `generate_cam(crop_bgr)` with interactive heatmap and visual attention overlay displays.

---

## 📊 Pretrained Baseline Comparison (Unit II: Transfer Learning)

[`baseline.py`](baseline.py) fine-tunes a torchvision **ResNet-18 (ImageNet-1k pretrained)** on the identical dataset partition and patch crops, satisfying the **Unit II Model Adaptation** requirement:

| Metric | ResPotholeNet (No Pretrained Weights) | Pretrained ResNet-18 (Fine-Tuned) |
|---|---|---|
| **Training Paradigm** | Random initialization (Kaiming normal) | ImageNet-1k pretrained + fine-tuned |
| **Trainable Parameters** | **853,985** (1x) | **11,180,609** (~13.1x heavier) |
| **Test Accuracy** | 86.93% (95% CI: [81.54%, 90.93%]) | 81.41% (95% CI: [75.43%, 86.20%]) |
| **Test Precision** | 92.55% | 85.11% |
| **Test Recall** | 82.08% | 78.43% |
| **Test F1-Score** | 87.00% | 81.77% |
| **Inference Latency** | ~2.5 ms / patch | ~4.1 ms / patch |

> ⚠️ **Statistical Significance Note (Baseline Gap Within Noise):**  
> With only $n=199$ test crops from 108 source photos, the 95% Wilson score confidence intervals overlap ([81.54%, 90.93%] vs [75.43%, 86.20%]). **We do not claim that ResPotholeNet outperforms ResNet-18.** The evidence shows that ResPotholeNet achieves comparable classification competence on this specific dataset while requiring 13x fewer parameters and zero reliance on external ImageNet pretraining.

---

## 📊 Dataset & Splitting Methodology

| Attribute | Specification |
|---|---|
| **Raw Source Photos** | 712 photos (364 pothole, 348 normal) |
| **Deduplication** | SHA-256 byte hashing (6 duplicates removed) + perceptual dHash (Hamming $\le 2$, 1 near-duplicate removed) + **cross-class deduplication** $\rightarrow$ **705 clean unique source photos** |
| **Deterministic Split** | Filenames sorted before splitting: **70% Train (492) / 15% Val (105) / 15% Test (108 photos)** |
| **Data Leakage Prevention** | Source photo level partitioning guarantees patches from the same photo never cross train/val/test splits. |
| **Symmetric Sampling** | Top 2 Stage-1 candidate proposals per photo extracted symmetrically for both positive and negative classes using `roi_mode="auto"`. |
| **Patch Counts (128×128)** | Train: 915 crops \| Val: 196 crops \| Test: 199 crops (106 positive, 93 negative) |
| **Loss & Optimizer** | Focal Loss ($\alpha=0.25, \gamma=2.0$), AdamW (weight decay $10^{-4}$), Cosine Annealing learning rate schedule. |

---

## 📈 Multi-Level Empirical Evaluation

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

### 3. Box-Level Ground-Truth Benchmark (IoU $\ge 0.50$, $n = 30$ photos)

Evaluated using [`evaluate_box_level.py`](evaluate_box_level.py) on the 30 pilot annotated test photos in [`data/test_ground_truth_boxes.json`](data/test_ground_truth_boxes.json):

| Metric | Measured Value | Meaning & Context |
|---|---|---|
| **Annotated Test Images** | **30** | Pilot manually annotated subset of test pothole photos |
| **Ground-Truth Pothole Boxes** | **30** | 1 reference box per test photo |
| **True Positive Boxes (IoU $\ge 0.50$)** | **2** | Only 2 predicted boxes achieved $\ge 50\%$ IoU |
| **False Positive Boxes (Pothole Photos)** | **241** | High candidate count on complex road textures |
| **False Negative Boxes (Missed)** | **28** | Boxes that failed to meet 0.50 IoU overlap |
| **Clean Road False Alarm Boxes** | **17** | False positive boxes across 30 clean road photos |
| **Box Precision** | **0.82%** | Low due to multi-candidate proposal generation |
| **Box Recall** | **6.67%** | 95% CI: [1.85%, 21.32%] |
| **Box F1-Score** | **1.47%** | Harmonic mean of box precision and recall |
| **Mean IoU (Matched True Positives)** | **56.10%** | Quality of overlap on the 2 successful matches |
| **mAP @ IoU 0.50 (AP_50)** | **0.20%** | Average precision under standard Pascal VOC metric |

---

### 4. Stage-1 Proposal Recall Ceiling Benchmark (Reproducibility Script: `evaluate_stage1_recall.py`)

Run `python evaluate_stage1_recall.py` to reproduce these exact numbers saved in [`stage1_recall_metrics.json`](stage1_recall_metrics.json):

| Stage-1 Candidate Metric | Measured Value |
|---|---|
| **Mean Maximum IoU on GT Boxes** | **11.14%** (Median: 6.13%) |
| **Candidate Recall @ IoU $\ge 0.10$** | **36.67%** (11 / 30 boxes) |
| **Candidate Recall @ IoU $\ge 0.25$** | **10.00%** (3 / 30 boxes) |
| **Candidate Recall @ IoU $\ge 0.50$** | **6.67%** (2 / 30 boxes) |
| **Proposals / Pothole Photo** | 9.6 candidate proposals |
| **Proposals / Clean Road Photo** | 1.6 candidate proposals |

---

### 5. Real-Road Video Benchmark & Temporal Error Analysis (`evaluate_video.py`)

Run `python evaluate_video.py --video sample_pothole_video.mp4` to reproduce results in [`video_evaluation_report.json`](video_evaluation_report.json):

| Video Benchmark Attribute | Result |
|---|---|
| **Test Video** | `sample_pothole_video.mp4` (1280x720 @ 25 FPS, 692 frames, ~27.7s) |
| **Inference Throughput** | **25.4 FPS** (Mean Latency: **23.9 ms / frame**) on CPU |
| **Tracking Algorithm** | SORT Linear Kalman Filter (7D state vector) |
| **Confirmed Unique Tracks** | Tracks maintained across successive frames |
| **Flicker Suppression** | `min_hits=2` effectively eliminates single-frame false alarms |
| **Qualitative Failure Modes** | Dark tar snakes and longitudinal pavement repair seams trigger sustained false proposals; sudden camera pitch changes on bumps challenge the linear velocity assumption. |

---

## 🔍 Critical Discussion, Failure Modes, and Methodological Notes

1. **Stage 1 Recall Ceiling (Root Cause of Box Failure):**
   - The Stage-1 benchmark proves that classical region proposals alone only reach IoU $\ge 0.50$ on 2 of 30 ground truth boxes (6.7% recall ceiling). Because the CNN classifier only classifies candidate patches nominated by Stage 1, the end-to-end pipeline cannot recover localization when Stage 1 fails to frame the pothole tightly.
2. **Ground-Truth Annotation Limitations:**
   - The 30 ground-truth bounding boxes in `data/test_ground_truth_boxes.json` were constructed by visual inspection with coordinates rounded to multiples of 10. They are approximate rectangles, whereas natural potholes have irregular, jagged boundaries (some with surrounding chalk markings like "FIXED"). Loose bounding boxes penalize IoU overlap, meaning part of the low IoU stems from coarse annotations.
   - 30 single-box images is a pilot benchmark and does not represent a full-scale object detection dataset.
3. **Screening vs. Localization Disconnect:**
   - 100% image-level screening recall occurs because an average of 7.35 candidate boxes pass threshold per pothole image. While this demonstrates effective frame screening, it does not imply tight localization.
4. **Proxy Supervision Label Noise (~58%):**
   - Positive training crops were generated using top-2 Stage-1 classical proposals from pothole photos. Visual examination of [`positives_sheet.png`](positives_sheet.png) confirms that **only ~10 of 24 crops (~42%) represent clear potholes**, while **~14 crops (~58%) capture non-pothole road artifacts** (shadows, cracks, dark asphalt, road boundaries). This proxy label noise bounds classifier precision.
5. **False Alarms on Clean Roads:**
   - 26.4% of clean road photos trigger at least one false alarm box at threshold 0.50 (0.642 FPPI). Dark asphalt seams, oil stains, and shadow contours mimic crater depressions.
6. **Severity Heuristic Disclaimer:**
   - Severity rankings (Low / Medium / High) in the application are strictly an unvalidated proxy based on bounding box pixel area ($A < 2\%$, $2\% \le A < 6\%$, $A \ge 6\%$). No ground-truth depth, volume, or road roughness sensor data exists in this dataset.
7. **Test Set Reuse:**
   - Test images were inspected qualitatively across multiple development iterations. While splits were strictly isolated at the source photo level, repeated qualitative observation introduces minor test-set exposure bias.

---

## 🚀 Installation & Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/Naitikjha/ROAD_DAMAGE.git
cd ROAD_DAMAGE
pip install -r requirements.txt
```

> **Note on torchvision:** `torchvision>=0.15.0` is included in `requirements.txt` exclusively for running the Unit II comparative transfer learning baseline ([`baseline.py`](baseline.py)). The primary ResPotholeNet model does not use torchvision.

### 2. Dataset Setup
```
data/
├── potholes/                   # Pothole road images (.jpg / .png)
├── normal/                     # Clear road pavement images (.jpg / .png)
├── test_ground_truth_boxes.json # Pilot 30 hand-annotated test bounding boxes
└── test_images_meta.json       # Metadata for test partitions
```

### 3. Run the Streamlit Application (Localhost)
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser:
- Video & image upload with real-time detection and Kalman tracking.
- Expand the Grad-CAM explainability panel for visual attention heatmaps.

### 4. Run Evaluations & Benchmarks

```bash
# 1. Stage-1 Candidate Recall & IoU Benchmark (reproduces stage1_recall_metrics.json)
python evaluate_stage1_recall.py

# 2. Box-Level Ground Truth Evaluation (IoU >= 0.50 on annotated subset)
python evaluate_box_level.py

# 3. Real-Road Video Benchmark & Temporal Error Analysis (sample_pothole_video.mp4)
python evaluate_video.py --video sample_pothole_video.mp4 --stride 5

# 4. End-to-End Pipeline Evaluation (Full test images screening)
python evaluate_pipeline.py --weights pothole_cnn.pth --data_dir data

# 5. Pretrained ResNet-18 Transfer Learning Baseline (Unit II)
python baseline.py --epochs 8 --batch_size 32

# 6. Retrain ResPotholeNet from Scratch
python train.py --data_dir data --epochs 15 --batch_size 32
```

---

## 📁 Repository Structure

```
├── .gitignore                   # Ignores raw images but tracks data/*.json annotations
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── app.py                       # Streamlit web app (Kalman SORT tracking + Grad-CAM)
├── detector.py                  # Two-stage engine (strict=True model loading + SORT)
├── train.py                     # Training pipeline (cross-class dedup + auto ROI + Focal Loss)
├── baseline.py                  # ResNet-18 fine-tuning baseline (Unit II)
├── evaluate_pipeline.py         # End-to-end full image evaluation
├── evaluate_box_level.py        # Box IoU evaluation on ground-truth annotations
├── evaluate_stage1_recall.py    # Standalone Stage-1 proposal recall benchmark
├── evaluate_video.py            # Real-road video evaluation & temporal error analysis
├── pothole_cnn.pth              # Sanitized trained ResPotholeNet weights (~854k params)
├── model_metrics.json           # Patch-level evaluation metrics
├── end_to_end_metrics.json      # Full validation sweep & test metrics
├── box_level_metrics.json       # Detailed box-level benchmark metrics
├── stage1_recall_metrics.json   # Stage-1 proposal benchmark results
├── video_evaluation_report.json # Real-road video benchmark results
├── baseline_comparison.json     # ResNet-18 vs ResPotholeNet comparison metrics
├── positives_sheet.png          # Contact sheet of extracted Stage-1 proposals
├── sample_gradcam.png           # Sample Grad-CAM attention overlay
├── sample_pothole_video.mp4     # 720p dashcam video for testing
├── requirements.txt             # Python dependencies (includes torchvision for baseline)
└── README.md                    # Project documentation and evaluation report
```

---

## 📜 Academic Integrity & Coursework Disclosures

- **AI Assistance Disclosure:** Generative AI coding assistants were used during this project for code refactoring, linting, debugging linear algebra matrices in the Kalman filter, generating Grad-CAM visualization hooks, and drafting documentation. All algorithmic logic, model architectures, and experimental results were verified and benchmarked on local runtime hardware.
- **Framework Syllabus Alignment:** The implementation is built in PyTorch. If your syllabus prescribes TensorFlow/Keras, confirm framework substitution acceptability with your course instructor.
- **Model Attribution:** The primary model uses no pretrained weights. Skip connections, Squeeze-and-Excitation attention, and SPPF pooling are implemented based on published literature cited above.
