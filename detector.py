"""
Two-Stage Road Pothole Detection Pipeline:
- Stage 1: Classical Region Proposals (Bilateral filtering, adaptive thresholding,
           morphological filtering, road perspective ROI, and local depression contrast)
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


def extract_stage1_candidates(
    frame: np.ndarray,
    use_roi: bool = True,
    roi_mode: str = None,
    max_candidates: int = 40,
) -> list:
    """
    Stage 1: Extract candidate pothole bounding boxes using classical CV.
    Uses bilateral filtering, adaptive thresholding, morphological filtering,
    local depression contrast scoring, color saturation rejection, and
    vertical edge anisotropy filtering (to eliminate fences and building facades).

    Parameters
    ----------
    frame : np.ndarray
        BGR image or video frame.
    use_roi : bool
        Legacy toggle. If True, uses dashcam road surface ROI; if False, uses photo ROI.
    roi_mode : str, optional
        'dashcam' (bottom 55% road region), 'photo' (15% to 95% surface), or 'auto'.
    max_candidates : int
        Maximum number of candidate proposals to return, sorted by saliency score.

    Returns
    -------
    list of (x1, y1, x2, y2) tuples
    """
    h, w = frame.shape[:2]

    # Resolve ROI strategy
    if roi_mode is None:
        mode = "dashcam" if use_roi else "photo"
    else:
        mode = roi_mode

    if mode == "auto":
        mode = "dashcam" if (w / float(max(h, 1)) >= 1.4 and h >= 300) else "photo"

    if mode == "dashcam":
        roi_top = int(h * 0.45)
        roi_bot = h
    elif mode == "photo":
        roi_top = int(h * 0.15) if h > 200 else 0
        roi_bot = int(h * 0.95) if h > 200 else h
    else:
        roi_top = 0
        roi_bot = h

    roi = frame[roi_top:roi_bot, :]
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 30 or roi_w < 30:
        return []

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

    # Dynamic area filters relative to frame resolution
    min_area = max(300, int(0.0008 * h * w))
    max_area = int(0.18 * h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.35 < aspect < 3.5:
                # Exclude boxes clipped against frame borders
                if bx <= 4 or by <= 4 or (bx + bw) >= roi_w - 4 or (by + bh) >= roi_h - 4:
                    continue

                orig_y = by + roi_top
                # Reject candidates sitting too high in image (sky/buildings horizon)
                if (orig_y / float(h)) < 0.18:
                    continue

                inner = l_chan[by:by + bh, bx:bx + bw]
                inner_bgr = roi[by:by + bh, bx:bx + bw]

                # 1. Color Saturation Filter:
                # Road asphalt and potholes are neutral grey/black/brown (low saturation).
                # Discard colorful non-road elements (vegetation, blue sky, cars, vibrant signs).
                hsv_inner = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
                mean_sat = float(np.mean(hsv_inner[:, :, 1]))
                if mean_sat > 75.0:
                    continue

                # 2. Vertical Anisotropy / Slat Filter:
                # Fences and building facades have strong dominant vertical gradients.
                # Potholes have 2D closed or isotropic edge distributions.
                gx = cv2.Sobel(inner, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(inner, cv2.CV_32F, 0, 1, ksize=3)
                gx_energy = float(np.mean(np.abs(gx)))
                gy_energy = float(np.mean(np.abs(gy)))
                if gy_energy > 0 and (gx_energy / (gy_energy + 1e-4) > 3.0):
                    continue

                # 3. Local Depression Contrast: compare candidate interior with surrounding road margin
                pad_x = max(6, int(bw * 0.30))
                pad_y = max(6, int(bh * 0.30))
                ox1 = max(0, bx - pad_x)
                oy1 = max(0, by - pad_y)
                ox2 = min(roi_w, bx + bw + pad_x)
                oy2 = min(roi_h, by + bh + pad_y)

                outer = l_chan[oy1:oy2, ox1:ox2]
                inner_sum = float(np.sum(inner))
                outer_sum = float(np.sum(outer))
                inner_cnt = inner.size
                outer_cnt = outer.size

                margin_cnt = outer_cnt - inner_cnt
                if margin_cnt < 16:
                    continue

                margin_mean = (outer_sum - inner_sum) / margin_cnt
                inner_mean = inner_sum / max(inner_cnt, 1)
                contrast = margin_mean - inner_mean

                # Pothole must be a physical depression (darker than surrounding road)
                if contrast < 0.0:
                    continue

                inner_std = float(np.std(inner))
                margin_std = float(np.std(outer))
                # Reject patches where margin is completely uniform (graphics, banners, sky)
                if margin_std < 5.0 or inner_std < 6.0:
                    continue

                # Saliency score: depression contrast + scale + interior texture
                score = (contrast + 5.0) * np.sqrt(area) * (1.0 + min(inner_std, 40.0) / 40.0)
                candidates.append((bx, orig_y, bx + bw, orig_y + bh, score))

    candidates.sort(key=lambda b: b[4], reverse=True)
    return [(b[0], b[1], b[2], b[3]) for b in candidates[:max_candidates]]


def extract_crop(
    frame: np.ndarray,
    box: tuple,
    crop_size: int = 128,
    pad_ratio: float = 0.1,
) -> np.ndarray:
    """
    Extract a bounding box with context margin and resize to (crop_size, crop_size).
    Returns None if patch is invalid or out of bounds.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    patch = frame[cy1:cy2, cx1:cx2]
    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return None
    return cv2.resize(patch, (crop_size, crop_size))


def normalize_patch(patch_bgr: np.ndarray) -> np.ndarray:
    """Normalize a 128x128 BGR patch using dataset mean and standard deviation."""
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
    return (rgb - mean) / std


class KalmanBoxTracker:
    """
    Individual object tracker using a Linear Kalman Filter for 2D bounding boxes.
    Follows SORT (Bewley et al., ICIP 2016).
    State vector: [u, v, s, r, u_dot, v_dot, s_dot]^T
      - u, v: center horizontal and vertical pixel coordinates
      - s: scale (bounding box area w * h)
      - r: aspect ratio (w / h), assumed approximately constant
      - u_dot, v_dot, s_dot: velocity components
    """

    count = 0

    def __init__(self, bbox: tuple):
        self.id = KalmanBoxTracker.count + 1
        KalmanBoxTracker.count += 1

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        u = float(bbox[0]) + w / 2.0
        v = float(bbox[1]) + h / 2.0
        s = w * h
        r = w / h

        # State vector
        self.x = np.array([u, v, s, r, 0.0, 0.0, 0.0]).reshape((7, 1))

        # State transition matrix F (constant velocity model)
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0  # u += u_dot
        self.F[1, 5] = 1.0  # v += v_dot
        self.F[2, 6] = 1.0  # s += s_dot

        # Measurement matrix H
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 100.0, 1.0, 1000.0, 1000.0, 10000.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 10.0, 0.01, 1.0, 1.0, 10.0]).astype(np.float32)
        self.R = np.diag([10.0, 10.0, 100.0, 1.0]).astype(np.float32)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = bbox

    def predict(self) -> tuple:
        """Advance the state vector using the constant velocity model."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: tuple):
        """Update the state vector with an observed measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_bbox = bbox

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        u = float(bbox[0]) + w / 2.0
        v = float(bbox[1]) + h / 2.0
        s = w * h
        r = w / h
        z = np.array([u, v, s, r], dtype=np.float32).reshape((4, 1))

        # Innovation residual
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(7, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> tuple:
        """Return the current bounding box estimate as (x1, y1, x2, y2)."""
        u, v, s, r = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        s = max(10.0, s)
        r = max(0.1, min(10.0, r))
        w = np.sqrt(s * r)
        h = s / max(w, 1e-4)
        x1 = max(0, u - w / 2.0)
        y1 = max(0, v - h / 2.0)
        x2 = x1 + w
        y2 = y1 + h
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


class PotholeTracker:
    """
    Linear Kalman Filter Pothole Tracker with Temporal Confirmation.
    - Uses KalmanBoxTracker (SORT algorithm, Bewley et al., ICIP 2016)
      to maintain constant-velocity state estimates for detected potholes.
    - Bipartite matching via IoU association between predicted and detected boxes.
    - Assigns persistent, unique track IDs.
    - Suppresses single-frame flickers (requires min_hits before confirmation).
    """

    def __init__(self, iou_thresh: float = 0.20, max_age: int = 8, min_hits: int = 2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []  # List of KalmanBoxTracker instances
        self.meta = {}      # tid -> dict with metadata (scores, severities, frames)
        self.confirmed_track_ids = set()

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / max(union, 1e-6)

    def update(self, detections: list, frame_idx: int = 0) -> list:
        """
        Update tracker with detections from current frame using Kalman filter prediction & update.

        Parameters
        ----------
        detections : list of dicts with keys "box", "score", (optionally "severity")
        frame_idx  : current frame index

        Returns
        -------
        list of active detections, each enriched with "track_id" and "hits"
        """
        # 1. Predict new locations of existing Kalman trackers
        predicted_boxes = []
        for trk in self.trackers:
            pred_box = trk.predict()
            predicted_boxes.append(pred_box)

        # 2. Compute IoU association matrix
        num_dets = len(detections)
        num_trks = len(self.trackers)
        matched_dets = set()
        matched_trks = set()

        if num_dets > 0 and num_trks > 0:
            iou_pairs = []
            for d_idx, det in enumerate(detections):
                for t_idx, trk in enumerate(self.trackers):
                    iou = self._compute_iou(det["box"], predicted_boxes[t_idx])
                    iou_pairs.append((iou, d_idx, t_idx))

            # Greedy bipartite matching by descending IoU
            iou_pairs.sort(key=lambda x: x[0], reverse=True)
            for iou, d_idx, t_idx in iou_pairs:
                if d_idx in matched_dets or t_idx in matched_trks:
                    continue
                if iou >= self.iou_thresh:
                    # Update Kalman filter with observation
                    det = detections[d_idx]
                    self.trackers[t_idx].update(det["box"])
                    tid = self.trackers[t_idx].id

                    # Update metadata
                    meta = self.meta[tid]
                    meta["last_frame"] = frame_idx
                    meta["score"] = det["score"]
                    meta["max_score"] = max(meta["max_score"], det["score"])
                    meta["hits"] = self.trackers[t_idx].hits
                    if "severity" in det:
                        meta["severity"] = det["severity"]
                        sev_rank = {"low": 1, "medium": 2, "high": 3}
                        if sev_rank.get(det["severity"], 1) > sev_rank.get(meta["peak_severity"], 1):
                            meta["peak_severity"] = det["severity"]

                    if meta["hits"] >= self.min_hits:
                        meta["confirmed"] = True
                        self.confirmed_track_ids.add(tid)

                    matched_dets.add(d_idx)
                    matched_trks.add(t_idx)

        # 3. Initialize new Kalman trackers for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                trk = KalmanBoxTracker(det["box"])
                tid = trk.id
                self.trackers.append(trk)
                confirmed = (self.min_hits <= 1)
                if confirmed:
                    self.confirmed_track_ids.add(tid)
                self.meta[tid] = {
                    "id": tid,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "hits": 1,
                    "score": det["score"],
                    "max_score": det["score"],
                    "severity": det.get("severity", "low"),
                    "peak_severity": det.get("severity", "low"),
                    "confirmed": confirmed,
                }

        # 4. Prune dead trackers that exceeded max_age without observations
        active_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                active_trackers.append(trk)
        self.trackers = active_trackers

        # 5. Return active confirmed detections in current frame
        output_dets = []
        for trk in self.trackers:
            tid = trk.id
            meta = self.meta.get(tid, {})
            # Only output if updated on this frame and confirmed
            if trk.time_since_update == 0 and (meta.get("confirmed", False) or self.min_hits <= 1):
                box = trk.get_state()
                output_dets.append({
                    "box": box,
                    "score": meta.get("score", 0.5),
                    "severity": meta.get("severity", "low"),
                    "track_id": tid,
                    "hits": trk.hits,
                })

        return output_dets

    def get_unique_count(self) -> int:
        """Total number of confirmed unique potholes tracked."""
        return len(self.confirmed_track_ids)

    def get_track_summary(self) -> list:
        """Summary records for all confirmed tracks."""
        summary = []
        for tid in sorted(self.confirmed_track_ids):
            if tid in self.meta:
                m = self.meta[tid]
                summary.append({
                    "track_id": m["id"],
                    "first_frame": m["first_frame"],
                    "last_frame": m["last_frame"],
                    "total_hits": m["hits"],
                    "max_score": round(m["max_score"], 3),
                    "peak_severity": m["peak_severity"],
                })
        return summary


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
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            raw_state = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
            model_keys = set(self.model.state_dict().keys())
            clean_state = {k: v for k, v in raw_state.items() if k in model_keys}
            self.model.load_state_dict(clean_state, strict=False)
            print(f"Loaded trained ResPotholeNet weights from {weights_path}")
        else:
            raise FileNotFoundError(
                f"Weights file not found: '{weights_path}'. "
                "Please train the model first with: python train.py --data_dir data"
            )

        self.model.float()
        self.model.eval()

    def extract_candidates(self, frame: np.ndarray, use_roi: bool = True, roi_mode: str = None) -> list:
        """Stage 1: Extract candidate bounding boxes."""
        return extract_stage1_candidates(frame, use_roi=use_roi, roi_mode=roi_mode)

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        use_roi: bool = True,
        roi_mode: str = None,
    ) -> list:
        """
        Detect potholes in a single frame.

        Stage 1: Classical region proposals generate candidate boxes.
        Stage 2: Each candidate patch is classified by ResPotholeNet.
                 The CNN sigmoid confidence is the SOLE accept/reject criterion.

        Returns:
            list of {"box": (x1, y1, x2, y2), "score": float}
        """
        candidate_boxes = self.extract_candidates(frame, use_roi=use_roi, roi_mode=roi_mode)
        if not candidate_boxes:
            return []

        patch_tensors = []
        valid_boxes = []

        for box in candidate_boxes:
            patch = extract_crop(frame, box, crop_size=128, pad_ratio=0.1)
            if patch is None:
                continue

            norm = normalize_patch(patch)
            tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
            patch_tensors.append(tensor)
            valid_boxes.append(box)

        if not patch_tensors:
            return []

        batch = torch.cat(patch_tensors, dim=0).to(self.device)
        with torch.no_grad():
            cnn_conf, _ = self.model(batch)
            cnn_scores = cnn_conf.squeeze(-1).cpu().numpy()

        # Handle single patch output dimension
        if cnn_scores.ndim == 0:
            cnn_scores = np.array([float(cnn_scores)])

        detections = []
        for i, (bx1, by1, bx2, by2) in enumerate(valid_boxes):
            score = float(np.clip(cnn_scores[i], 0.0, 1.0))
            if score >= conf_threshold:
                detections.append({
                    "box": (bx1, by1, bx2, by2),
                    "score": round(score, 3),
                })

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
