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

## 📊 Training Details

| Attribute | Specification |
|---|---|
| **Training Data** | Real road images (~712 images: 364 pothole + 348 normal) |
| **Data Source** | [anantSinghCross/pothole-detection-system](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks) |
| **Split** | 70% Train / 15% Validation / 15% Test (by image) |
| **Loss** | Focal Loss (α=0.25, γ=2.0) — standard formulation |
| **Optimizer** | AdamW (weight_decay=1e-4) with CosineAnnealing LR |
| **Augmentation** | Random H/V flip, brightness/contrast jitter, rotation (±15°) |
| **Pretrained Weights** | **None** (trained from scratch) |

Evaluation metrics (Precision, Recall, F1, Confusion Matrix) are computed on the held-out test set and saved to `model_metrics.json`.

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

- **ROI hardcoded**: Stage 1 uses the bottom 48% of the frame, which may fail on cropped or tilted footage.
- **No tracking**: Pothole count is per-frame, not unique potholes across a video. A tracker (SORT/DeepSORT) is needed for unique counting.
- **Regression head unused**: The bbox refinement head was trained with synthetic targets that don't match inference crops. It is intentionally disabled.
- **Dataset size**: ~712 images is small. Performance would improve with larger datasets (e.g., RDD2022 with thousands of images).
- **Frame skipping**: With `every_n > 1`, the "Frames Processed" metric shows only frames where the CNN ran.

---

## 📜 Academic Integrity Note

Developed for academic coursework. Formulated honestly as a **classical region proposal + custom CNN classifier pipeline**. No YOLO, Ultralytics, torchvision backbones, or pretrained weights used. All limitations are documented above.
