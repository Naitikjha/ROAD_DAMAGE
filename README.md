# 🕳️ Pothole Detection from Road Videos Using a Custom CNN

An end-to-end deep learning project for detecting and categorizing potholes in road videos using a **Custom Convolutional Neural Network (ResPotholeNet)** trained from scratch without any pretrained models or backbones.

---

## 🌟 Highlights
- **100% From-Scratch CNN**: No YOLO, Ultralytics, torchvision backbones, or pretrained weights used.
- **ResPotholeNet Architecture**: Incorporates Residual Bottlenecks, Squeeze-and-Excitation (SE) channel attention, and Multi-Scale Spatial Pyramid Pooling (SPPF).
- **Dual Head Output**: Simultaneous cavity confidence scoring and bounding box boundary regression.
- **Severity Classification**: Automatically categorizes pothole severity (Low, Medium, High) based on physical area relative to frame perspective.
- **Interactive Streamlit Frontend**: Video and Image detection tabs, live bounding box overlays, metric cards, H.264 browser-compatible re-encoding, and download capabilities.

---

## 🏗️ Architecture: ResPotholeNet

| Layer / Component | Details |
|---|---|
| **Input Resolution** | 128 × 128 candidate patches |
| **Stem** | Conv2D + BatchNorm + SiLU downsampling |
| **Backbone** | 3-stage Residual Bottlenecks with SE Attention |
| **Attention** | Squeeze-and-Excitation (channel re-calibration) |
| **Receptive Field** | Spatial Pyramid Pooling - Fast (SPPF) |
| **Parameters** | 858,245 parameters |
| **Loss Formulation** | Focal Loss + Smooth L1 Regression Loss |

---

## 🚀 Installation & Running

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ROAD_POTHHOLE
```

### 2. Install Dependencies
```bash
pip install streamlit opencv-python numpy imageio-ffmpeg torch
```

### 3. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📁 Repository Structure
```
├── .streamlit/
│   └── config.toml          # Streamlit upload size configuration
├── models/
│   └── pothole_net.py       # Custom CNN architecture (ResPotholeNet)
├── app.py                   # Main Streamlit web application
├── detector.py              # Detection engine & candidate extraction
├── train.py                 # Custom model training pipeline
├── pothole_cnn.pth          # Trained model weights
├── sample_pothole_video.mp4 # Sample road test footage
├── .gitignore
└── README.md
```

---

## 📜 License
Developed for Academic & Course Project Submission.
