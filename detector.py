"""
Two-Stage Road Pothole Detection Pipeline:
- Stage 1: Classical Region Proposals (Bilateral filtering, adaptive thresholding,
           morphological filtering, and perspective ROI contour grouping)
- Stage 2: ResPotholeNet CNN Patch Classification (sole decision maker)

Architecture: Classical region proposals + custom CNN binary classifier.
The CNN confidence score is the ONLY factor in accept/reject decisions.
The bounding box regression head is NOT used at inference (training/inference
target mismatch — see project report for ablation discussion).

No pretrained backbones or transfer learning weights used.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from models.pothole_net import ResPotholeNet


class PotholeDetector:
    """
    Two-Stage Pothole Detection Engine.
    Stage 1: Classical morphological region proposals.
    Stage 2: ResPotholeNet CNN binary classifier (sole decision maker).
    """
    def __init__(self, weights_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ResPotholeNet(in_channels=3).to(self.device).float()

        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location=self.device,
                                    weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded trained ResPotholeNet weights from {weights_path}")
        else:
            raise FileNotFoundError(
                f"Weights file not found: '{weights_path}'. "
                "Please train the model first with: python train.py --data_dir data"
            )

        self.model.float()
        self.model.eval()

    def extract_candidates(self, frame: np.ndarray) -> list:
        """
        Stage 1: Extract candidate pothole bounding boxes using classical CV.
        Uses bilateral filtering, adaptive thresholding, and morphological ops.
        """
        h, w = frame.shape[:2]
        # Road region of interest: bottom 48% of the frame (avoids sky, horizon)
        roi_top = int(h * 0.52)
        roi = frame[roi_top:, :]

        # Convert to LAB for luminance extraction
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

        # Smooth road aggregate texture while preserving crater borders
        blurred = cv2.bilateralFilter(l_chan, 9, 75, 75)

        # Adaptive thresholding for depression detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )

        # Morphological filtering to group cavity fragments and suppress grain
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Area filter: ignore tiny noise and oversized blobs
            if 350 < area < (h * w * 0.15):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / float(bh) if bh > 0 else 0
                # Potholes generally have aspect ratio 0.4 to 4.5
                if 0.4 < aspect < 4.5:
                    # Map coordinates back to original frame
                    orig_y = by + roi_top
                    candidates.append((bx, orig_y, bx + bw, orig_y + bh))

        return candidates

    def predict(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list:
        """
        Detect potholes in a single frame.

        Stage 1: Classical region proposals generate candidate boxes.
        Stage 2: Each candidate patch is classified by ResPotholeNet.
                 The CNN sigmoid confidence is the SOLE accept/reject criterion.

        Returns:
            list of {"box": (x1, y1, x2, y2), "score": float}
        """
        h, w = frame.shape[:2]
        candidate_boxes = self.extract_candidates(frame)

        if not candidate_boxes:
            return []

        # Prepare patches for CNN evaluation
        patch_tensors = []
        valid_boxes = []

        for (x1, y1, x2, y2) in candidate_boxes:
            # Context margin around box
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            patch = frame[cy1:cy2, cx1:cx2]
            if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
                continue

            # Preprocess patch for ResPotholeNet: resize to 128x128 and normalise
            resized = cv2.resize(patch, (128, 128))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
            std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
            rgb = (rgb - mean) / std
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
            patch_tensors.append(tensor)
            valid_boxes.append((x1, y1, x2, y2))

        if not patch_tensors:
            return []

        batch = torch.cat(patch_tensors, dim=0).to(self.device)
        with torch.no_grad():
            cnn_conf, _ = self.model(batch)
            # Regression head output is intentionally ignored at inference.
            # See project report for discussion of training/inference mismatch.
            cnn_scores = cnn_conf.squeeze(-1).cpu().numpy()

        detections = []
        for i, (bx1, by1, bx2, by2) in enumerate(valid_boxes):
            # CNN confidence is the sole decision criterion
            score = float(np.clip(cnn_scores[i], 0.0, 1.0))

            if score >= conf_threshold:
                detections.append({
                    "box": (bx1, by1, bx2, by2),
                    "score": round(score, 3),
                })

        # Apply Non-Maximum Suppression to eliminate overlapping boxes
        return self.non_max_suppression(detections, iou_thresh=0.35)

    @staticmethod
    def non_max_suppression(detections: list, iou_thresh: float = 0.35) -> list:
        """Standard Non-Maximum Suppression."""
        if not detections:
            return []

        boxes = np.array([d["box"] for d in detections], dtype=float)
        scores = np.array([d["score"] for d in detections], dtype=float)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [detections[idx] for idx in keep]
