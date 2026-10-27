# 🕳️ Pothole Detection from Road Videos Using a Custom CNN

A deep learning course project for detecting, classifying, and assessing the severity of potholes in road videos and images. Built **100% from scratch using pure PyTorch** — without YOLO, Ultralytics, torchvision backbones, or pretrained weights.

---

## 🏗️ Architecture: Two-Stage Detection Pipeline

The system uses a **two-stage pipeline**: classical computer vision generates region proposals, and a custom CNN classifies each proposal.

### Stage 1: Classical Region Proposals
- Extracts the road surface ROI (bottom 48% of the frame).
- Bilateral filtering (`d=9, σ_color=75, σ_space=75`) smooths asphalt texture while preserving edges.
- Adaptive thresholding + morphological opening/closing isolates dark candidate regions.
- Contour extraction with aspect-ratio filtering (`0.4 < aspect < 4.5`) produces candidate bounding boxes.

### Stage 2: ResPotholeNet CNN Classifier
- Each candidate patch is resized to **128 × 128** and normalized.
- The CNN classifies each patch as pothole or normal road.
- **The CNN sigmoid confidence is the sole accept/reject criterion** — no contrast score blending.
- Non-Maximum Suppression (IoU threshold = 0.35) removes overlapping boxes.

> **Note:** The bounding box regression head exists in the architecture but is **not used at inference** due to a training/inference target mismatch. The boxes come directly from Stage 1 contours. See the project report for ablation discussion.

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
| **Parameters** | ~858,245 trainable parameters |

---

## 📊 Training Details & Evaluation Scope

| Attribute | Specification |
|---|---|
| **Training Input** | Domain-aligned 128 × 128 crops from real road photos + Stage-1 hard negatives (shadows, cracks, oil stains) |
| **Data Source** | Real road images (~712 source photos: 364 pothole + 348 normal) from [anantSinghCross/pothole-detection-system](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks) |
| **Deduplication** | SHA-256 hash verification to eliminate exact/near duplicate photos across splits |
| **Split** | 70% Train / 15% Validation / 15% Test (held-out test set n = 108) |
| **Loss Function** | Focal Loss (α=0.25, γ=2.0) for binary classification |
| **Optimizer** | AdamW (weight_decay=1e-4) with CosineAnnealingLR scheduler |
| **Augmentation** | Horizontal flip (no vertical flip to preserve ground perspective), brightness/contrast jitter, rotation (±15°) |
| **Pretrained Weights** | **None** (trained 100% from scratch) |

### ⚠️ Evaluation Scope: Patch Classification vs. End-to-End Detection

- **What the metrics measure:** The test metrics in [`model_metrics.json`](file:///d:/NAITIK/DL%20PROJECT/ROAD_POTHHOLE/model_metrics.json) (Accuracy: 92.59%, Precision: 90.74%, Recall: 94.23%, F1: 92.45%, n=108) measure the CNN's ability to classify **128 × 128 patches** as pothole vs. normal road.
- **What they do not measure:** They do **not** represent end-to-end bounding box detection accuracy (e.g., mAP@0.5). Bounding boxes are proposed by Stage 1 OpenCV contours; the dataset provides image-level labels without ground-truth bounding box coordinates.
- **Sample size sensitivity:** On n=108 test samples (52 positive), a single error shifts metrics by ~1%. All numbers should be interpreted in this context.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ROAD_POTHHOLE
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
```bash
# The dataset is auto-organized into data/potholes/ and data/normal/
# Or clone manually:
git clone --depth 1 https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks.git temp_repo
# Then run: python -c "see train.py instructions"
```

### 4. Train the Model
```bash
python train.py --data_dir data --epochs 15 --batch_size 16
```

This produces:
- `pothole_cnn.pth` — best model weights (saved at highest validation F1)
- `model_metrics.json` — test set evaluation report

### 5. Run the Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

> ⚠️ The app **requires** `pothole_cnn.pth` to exist. If missing, it will display an error instructing you to train first.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml          # Upload limit configuration
├── data/
│   ├── potholes/            # Real pothole road images
│   └── normal/              # Clear road pavement images
├── models/
│   └── pothole_net.py       # Custom CNN architecture (ResPotholeNet)
├── app.py                   # Streamlit web application
├── detector.py              # Two-stage detection engine
├── train.py                 # Training, validation & evaluation pipeline
├── pothole_cnn.pth          # Trained model weights
├── model_metrics.json       # Test set evaluation metrics
├── requirements.txt         # Python dependencies
├── sample_pothole_video.mp4 # Sample test video
├── .gitignore
└── README.md
```

---

## ⚠️ Known Limitations

- **Patch Classification vs. Detection Accuracy**: The 92.45% F1 score evaluates whether the CNN correctly identifies a 128×128 crop as a pothole or clear road patch. It does not measure whole-frame object detection accuracy (mAP), because the bounding boxes are derived from Stage 1 OpenCV morphological contours rather than ground-truth annotated boxes.
- **Sample Size Sensitivity**: Held-out test set size is n = 108 crops (52 pothole, 56 normal); a single misclassification moves the accuracy/F1 score by ~1%.
- **ROI Hardcoded**: Stage 1 inspects the lower 48% of the frame, which assumes standard forward-facing dashcam perspective. It may miss hazards on tilted or cropped angles.
- **No Tracking Across Frames**: Counts reflect detections per evaluated frame rather than unique persistent potholes.
- **Regression Head Unused at Inference**: The model retains its bounding box regression architecture for academic parity, but inference uses Stage 1 contours directly to avoid synthetic anchor mismatches.
- **Frame Skipping**: With `every_n > 1`, processing skips intermediate frames for real-time speed, maintaining persistent overlay from the previous evaluation.

---

## 📜 Academic Integrity Note

Developed for academic coursework. Formulated honestly as a **classical region proposal + custom CNN classifier pipeline**. No YOLO, Ultralytics, torchvision backbones, or pretrained weights used. All limitations are documented above.
