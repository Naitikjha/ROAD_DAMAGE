# 🕳️ Pothole Detection from Road Videos Using a Custom CNN

A deep learning course project for detecting, classifying, and assessing the severity of potholes in road videos and images. Built **100% from scratch using pure PyTorch** — without YOLO, Ultralytics, torchvision backbones, or pretrained weights.

---

## 🏗️ Architecture: Two-Stage Detection Pipeline (R-CNN Paradigm)

Rather than using an off-the-shelf blackbox detector, the system employs a two-stage computer vision & deep learning architecture:

1. **Stage 1: Road-Surface Region Proposals (RPN / Morphological)**
   - Extracts road surface Region of Interest (perspective ROI).
   - Bilateral filtering (`d=9, sigmaColor=75, sigmaSpace=75`) to preserve crater rim edges while smoothing asphalt aggregate grain.
   - Multi-scale adaptive thresholding and morphological closing/opening to isolate candidate road depressions.
   - Contour extraction and aspect-ratio geometry filtering (`0.4 < aspect < 4.5`) generate candidate bounding boxes.

2. **Stage 2: ResPotholeNet Deep Patch Classifier (Custom CNN)**
   - Candidate patches are resized to **128 × 128** and normalized against road pavement distribution.
   - **Stem:** Strided convolutions for initial spatial downsampling.
   - **Residual Bottlenecks:** 3 stages of residual blocks with skip connections to prevent gradient vanishing.
   - **Squeeze-and-Excitation (SE) Attention:** Re-calibrates channel weights to focus on dark cavity depth features.
   - **Spatial Pyramid Pooling - Fast (SPPF):** Multi-scale receptive field aggregation (captures small cracks to large craters).
   - **Dual Head:**
     - Classification Head (Sigmoid probability) — **primary decision maker with full veto power**.
     - Bounding Box Refinement Head (Smooth L1 regression delta).
   - **Non-Maximum Suppression (NMS):** Eliminates overlapping redundant proposals (`IoU threshold = 0.35`).

---

## 📊 Model Specifications

| Attribute | Specification |
|---|---|
| **Architecture** | ResPotholeNet (SE-ResNet + SPPF) |
| **Parameters** | 858,245 trainable parameters |
| **Input Resolution** | 128 × 128 patches |
| **Loss Formulation** | Focal Loss ($\alpha=0.25, \gamma=2.0$) + Smooth L1 Loss |
| **Pretrained Weights** | **None** (Trained completely from scratch) |
| **Training Split** | 70% Train / 15% Validation / 15% Test |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ROAD_POTHHOLE
```

### 2. Install Required Packages
```bash
pip install streamlit opencv-python numpy imageio-ffmpeg torch pandas
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🏋️ Training & Evaluation

The training script supports both real image datasets (e.g., RDD2022 / Kaggle) and structured synthetic benchmark generation:

```bash
# Train on synthetic benchmark dataset with train/val/test evaluation:
python train.py --epochs 8 --batch_size 16

# Train on real road dataset (folder structured with /potholes and /normal):
python train.py --data_dir /path/to/road_dataset --epochs 15 --batch_size 32
```

Outputs:
- Best weights: `pothole_cnn.pth`
- Evaluation report: `model_metrics.json` (includes Precision, Recall, F1-score, Confusion Matrix)

---

## 🌟 Frontend Features & Fixes

- **Decoupled Tabs:** Video and Image tabs operate on separate session states; uploading an image will not wipe video results.
- **Reliable File Tracking:** Identifies uploads by file metadata (`name + size`) rather than volatile memory IDs.
- **H.264 Browser Encoding:** Automatic conversion to H.264 `yuv420p` with `faststart` and dimension padding (`ceil(iw/2)*2`) for seamless browser playback.
- **Throttled Progress Updates:** Progress bar updates are throttled to prevent UI stutter on long videos.
- **Accurate Metric Reporting:** Displays total frames read and frames evaluated by the CNN.
- **Chronological Detections Log:** Detections are logged in time sequence with severity (Low, Medium, High) derived from physical perspective area.
- **Cache Management:** One-click "Clear Temp Files & Cache" button in the sidebar to prevent disk buildup.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml          # Upload limit configuration
├── models/
│   └── pothole_net.py       # Custom CNN architecture (ResPotholeNet)
├── app.py                   # Streamlit web application
├── detector.py              # Two-stage detection engine
├── train.py                 # Training, validation & evaluation pipeline
├── pothole_cnn.pth          # Trained model weights
├── model_metrics.json       # Unseen test set evaluation metrics
├── sample_pothole_video.mp4 # Sample test video
├── .gitignore
└── README.md
```

---

## 📜 Academic Integrity Note
Developed for academic coursework. Formulated honestly as a **Two-Stage Region-Proposal + Custom CNN Verification Architecture** compliant with project requirements (zero pretrained models).
