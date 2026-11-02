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

import time
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import cv2
import numpy as np
import torch
from models.pothole_net import ResPotholeNet


def load_validation_threshold(
    metrics_path: str = "model_metrics_windows.json",
    default: float = 0.50,
) -> float:
    """
    Load the confidence threshold chosen during validation tuning (Step 3).
    Ensures zero test leakage: the threshold is read directly from validation tuning metrics.
    """
    p = Path(metrics_path)
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        thresh = data.get("chosen_inference_threshold")
        if thresh is not None:
            return float(thresh)
    except Exception:
        pass
    return default


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


PROPOSALS = "sliding"  # Options: "sliding" | "legacy"


def generate_sliding_windows(
    frame: np.ndarray,
    roi_mode: str = "auto",
    stride_ratio: float = 0.50,
    max_windows: int = 300,
    scale_fractions: Optional[List[float]] = None,
    aspect_ratios: Optional[List[Tuple[int, int]]] = None,
) -> list:
    """
    Generate bounded multi-scale sliding window candidate boxes over the road surface ROI.

    Plain-words viva explanation:
    1. Road ROI: Restricts candidate search to the drivable road surface so the CNN
       never wastes budget on sky, horizon, or dashboard.
    2. Multi-Scale Sampling: Generates boxes at relative scales (default [0.20, 0.35, 0.52])
       times min(H, W).  For a 1280x720 dashcam frame these produce windows of roughly
       144, 252, and 374 px — covering small road cracks up to large craters.  Three
       aspect ratios are used: 1:1 (roughly square potholes), 2:1 (wide transverse
       ruts), and 3:1 (longitudinal cracks and trenches).
    3. Stride Setting: step = stride_ratio * window_dimension (default: 0.50).
       Adjusting stride_ratio allows trading speed vs. spatial resolution:
       a larger stride (e.g. 0.65) generates fewer windows for faster FPS;
       a smaller stride (e.g. 0.45) increases localization overlap.
    4. Bounded Budget — per-bucket subsampling: The total count is held to
       max_windows (default 300).  When the count exceeds the budget we subsample
       *within each (scale, aspect) bucket* uniformly so every scale and aspect ratio
       keeps at least one window.

    Parameters
    ----------
    frame : np.ndarray
        BGR image or video frame.
    roi_mode : str
        'auto', 'dashcam', or 'photo'.
    stride_ratio : float
        Fraction of box dimension used as the spatial step (default: 0.50).
    max_windows : int
        Hard upper bound on candidates returned per frame (default: 300).
    scale_fractions : list of float, optional
        Custom scale multipliers relative to min(H, W). Default: [0.20, 0.35, 0.52].
    aspect_ratios : list of (w, h) tuples, optional
        Aspect ratios to use. Default: [(1, 1), (2, 1), (3, 1)].

    Returns
    -------
    list of (x1, y1, x2, y2) integer tuples
    """
    h, w = frame.shape[:2]
    if roi_mode == "auto":
        mode = "dashcam" if (w / float(max(h, 1)) >= 1.4 and h >= 300) else "photo"
    else:
        mode = roi_mode

    if mode == "dashcam":
        y_min, y_max = int(h * 0.45), h
    elif mode == "photo":
        y_min = int(h * 0.15) if h > 200 else 0
        y_max = int(h * 0.95) if h > 200 else h
    else:
        y_min, y_max = 0, h

    roi_h = y_max - y_min
    roi_w = w
    base = min(h, w)

    if scale_fractions is None:
        scale_fractions = [0.20, 0.35, 0.52]
    if aspect_ratios is None:
        aspect_ratios = [(1, 1), (2, 1), (3, 1)]

    scales = [max(16, int(base * s)) for s in scale_fractions]

    # ── Collect windows per (scale, aspect) bucket ───────────────────────────
    # Keeping buckets separate lets us subsample proportionally later so that
    # large-scale windows (fewer but important for big potholes) are not wiped
    # out by the more numerous small-scale windows.
    buckets: list = []          # list of lists, one per valid (scale, aspect)
    for s in scales:
        for ar_w, ar_h in aspect_ratios:
            bh = int(s / np.sqrt(ar_w / ar_h))
            bw = int(bh * (ar_w / ar_h))
            if bh > roi_h or bw > roi_w or bh < 24 or bw < 24:
                continue

            step_y = max(18, int(bh * stride_ratio))
            step_x = max(18, int(bw * stride_ratio))

            bucket: list = []
            for y1 in range(y_min, y_max - bh + 1, step_y):
                for x1 in range(0, roi_w - bw + 1, step_x):
                    bucket.append((x1, y1, x1 + bw, y1 + bh))
            if bucket:
                buckets.append(bucket)

    total = sum(len(b) for b in buckets)
    if total == 0:
        return []

    # ── Per-bucket uniform subsampling when over budget ──────────────────────
    # Each bucket gets a quota proportional to its size.  Within the bucket,
    # windows are picked at evenly spaced indices so spatial coverage is
    # maintained rather than always biasing toward the top-left corner.
    if total > max_windows:
        windows: list = []
        for bucket in buckets:
            # Proportional quota, at least 1 window per non-empty bucket
            quota = max(1, int(round(len(bucket) / total * max_windows)))
            indices = np.linspace(0, len(bucket) - 1, min(quota, len(bucket)), dtype=int)
            windows.extend(bucket[i] for i in indices)
    else:
        windows = [w for b in buckets for w in b]

    return windows


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


def weighted_box_merge(detections: list, iou_thresh: float = 0.35) -> list:
    """
    Weighted Box Merge (WBM / Weighted Box Fusion):
    Instead of greedily discarding overlapping detections like standard NMS,
    groups boxes with IoU >= iou_thresh into clusters and computes a
    confidence-weighted average of their coordinates:
        coord_merged = sum(conf_i * coord_i) / sum(conf_i)
    The cluster's final confidence score is the maximum confidence among the constituent
    boxes.

    Viva explanation:
    - Standard NMS discards all but the single highest-scoring box in an overlapping group,
      which can cause boundary jitter across consecutive video frames.
    - Weighted Box Merge combines all overlapping positive sliding windows, weighting
      their coordinates by predicted CNN confidence:
          x_merged = sum(conf_i * x_i) / sum(conf_i)
      This pulls the final bounding box toward the consensus center of highest pothole confidence,
      producing tighter, smoother box predictions for irregularly shaped road damage.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    clusters: List[List[Dict[str, Any]]] = []

    for det in sorted_dets:
        box_a = det["box"]
        matched_cluster = None
        for cluster in clusters:
            leader_box = cluster[0]["box"]
            xA = max(box_a[0], leader_box[0])
            yA = max(box_a[1], leader_box[1])
            xB = min(box_a[2], leader_box[2])
            yB = min(box_a[3], leader_box[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            areaA = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            areaB = (leader_box[2] - leader_box[0]) * (leader_box[3] - leader_box[1])
            union = areaA + areaB - inter
            iou = inter / max(union, 1e-6)
            if iou >= iou_thresh:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster.append(det)
        else:
            clusters.append([det])

    merged = []
    for cluster in clusters:
        total_score = sum(d["score"] for d in cluster)
        max_score = max(d["score"] for d in cluster)

        if total_score > 0:
            avg_x1 = sum(d["score"] * d["box"][0] for d in cluster) / total_score
            avg_y1 = sum(d["score"] * d["box"][1] for d in cluster) / total_score
            avg_x2 = sum(d["score"] * d["box"][2] for d in cluster) / total_score
            avg_y2 = sum(d["score"] * d["box"][3] for d in cluster) / total_score
        else:
            avg_x1, avg_y1, avg_x2, avg_y2 = cluster[0]["box"]

        merged_box = (int(round(avg_x1)), int(round(avg_y1)), int(round(avg_x2)), int(round(avg_y2)))
        res_item = {
            "box": merged_box,
            "score": round(max_score, 3),
            "merged_count": len(cluster),
        }
        if "severity" in cluster[0]:
            res_item["severity"] = cluster[0]["severity"]
        merged.append(res_item)

    return merged


class PotholeDetector:
    """
    Two-Stage Pothole Detection Engine.
    Stage 1: Multi-scale sliding window proposals (or legacy classical CV contours).
    Stage 2: ResPotholeNet CNN binary classifier (sole decision maker).
    Stage 3: Box suppression / fusion (standard NMS or Weighted Box Merge).
    """

    def __init__(
        self,
        weights_path: str = None,
        device: str = None,
        proposals: str = PROPOSALS,
        conf_threshold: float = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.proposals = proposals
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        else:
            self.conf_threshold = load_validation_threshold()

        self.model = ResPotholeNet(in_channels=3).to(self.device).float()

        # If weights_path not specified, prefer retrained sliding window weights if available
        if weights_path is None:
            if Path("pothole_cnn_windows.pth").exists():
                weights_path = "pothole_cnn_windows.pth"
            elif Path("pothole_cnn.pth").exists():
                weights_path = "pothole_cnn.pth"

        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Loaded trained ResPotholeNet weights from {weights_path} (proposals={self.proposals}, threshold={self.conf_threshold:.2f})")
        else:
            raise FileNotFoundError(
                f"Weights file not found: '{weights_path}'. "
                "Please train the model first with: python train.py --data_dir data"
            )

        self.model.float()
        self.model.eval()

    def extract_candidates(
        self,
        frame: np.ndarray,
        use_roi: bool = True,
        roi_mode: str = None,
        stride_ratio: float = 0.50,
        max_windows: int = 300,
        scale_fractions: Optional[List[float]] = None,
    ) -> list:
        """Extract candidate bounding boxes via sliding windows or legacy classical CV."""
        if self.proposals == "sliding":
            return generate_sliding_windows(
                frame,
                roi_mode=roi_mode or "auto",
                stride_ratio=stride_ratio,
                max_windows=max_windows,
                scale_fractions=scale_fractions,
            )
        else:
            return extract_stage1_candidates(frame, use_roi=use_roi, roi_mode=roi_mode)

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = None,
        use_roi: bool = True,
        roi_mode: str = None,
        batch_size: int = 64,
        iou_thresh: float = 0.35,
        nms_mode: str = "nms",
        stride_ratio: float = 0.50,
        max_windows: int = 300,
        scale_fractions: Optional[List[float]] = None,
    ) -> list:
        """
        Detect potholes in a single frame.

        Stage 1: Candidate proposals (sliding windows or legacy classical contours).
        Stage 2: Batched ResPotholeNet classification.
                 The CNN sigmoid confidence is the SOLE accept/reject criterion.
        Stage 3: Non-Maximum Suppression (standard NMS or Weighted Box Merge).

        Returns:
            list of {"box": (x1, y1, x2, y2), "score": float}
        """
        dets, _ = self.predict_timed(
            frame=frame,
            conf_threshold=conf_threshold,
            use_roi=use_roi,
            roi_mode=roi_mode,
            batch_size=batch_size,
            iou_thresh=iou_thresh,
            nms_mode=nms_mode,
            stride_ratio=stride_ratio,
            max_windows=max_windows,
            scale_fractions=scale_fractions,
        )
        return dets

    def predict_timed(
        self,
        frame: np.ndarray,
        conf_threshold: float = None,
        use_roi: bool = True,
        roi_mode: str = None,
        batch_size: int = 64,
        iou_thresh: float = 0.35,
        nms_mode: str = "nms",
        stride_ratio: float = 0.50,
        max_windows: int = 300,
        scale_fractions: Optional[List[float]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Detect potholes in a single frame and report execution latency breakdown (in ms).

        Returns:
            detections: list of {"box": (x1, y1, x2, y2), "score": float, ...}
            timing: dict with proposal_ms, crop_prep_ms, cnn_inference_ms, nms_ms, total_ms, fps, num_windows
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        t_start = time.perf_counter()

        # 1. Candidate proposals
        t0 = time.perf_counter()
        candidate_boxes = self.extract_candidates(
            frame,
            use_roi=use_roi,
            roi_mode=roi_mode,
            stride_ratio=stride_ratio,
            max_windows=max_windows,
            scale_fractions=scale_fractions,
        )
        t1 = time.perf_counter()
        proposal_ms = (t1 - t0) * 1000.0

        if not candidate_boxes:
            t_end = time.perf_counter()
            return [], {
                "proposal_ms": round(proposal_ms, 2),
                "crop_prep_ms": 0.0,
                "cnn_inference_ms": 0.0,
                "nms_ms": 0.0,
                "total_ms": round((t_end - t_start) * 1000.0, 2),
                "fps": round(1000.0 / max((t_end - t_start) * 1000.0, 1e-3), 2),
                "num_windows": 0,
            }

        # 2. Patch extraction & normalization
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

        t2 = time.perf_counter()
        crop_prep_ms = (t2 - t1) * 1000.0

        if not patch_tensors:
            t_end = time.perf_counter()
            return [], {
                "proposal_ms": round(proposal_ms, 2),
                "crop_prep_ms": round(crop_prep_ms, 2),
                "cnn_inference_ms": 0.0,
                "nms_ms": 0.0,
                "total_ms": round((t_end - t_start) * 1000.0, 2),
                "fps": round(1000.0 / max((t_end - t_start) * 1000.0, 1e-3), 2),
                "num_windows": len(candidate_boxes),
            }

        # 3. Batched ResPotholeNet CNN classification
        cnn_scores_list = []
        with torch.no_grad():
            for b_idx in range(0, len(patch_tensors), batch_size):
                batch = torch.cat(patch_tensors[b_idx:b_idx + batch_size], dim=0).to(self.device)
                cnn_conf, _ = self.model(batch)
                scores = cnn_conf.squeeze(-1).cpu().numpy()
                if scores.ndim == 0:
                    scores = np.array([float(scores)])
                cnn_scores_list.extend(scores.tolist())

        t3 = time.perf_counter()
        cnn_inference_ms = (t3 - t2) * 1000.0

        # 4. Thresholding & NMS / Weighted Box Merge
        raw_detections = []
        for i, (bx1, by1, bx2, by2) in enumerate(valid_boxes):
            score = float(np.clip(cnn_scores_list[i], 0.0, 1.0))
            if score >= conf_threshold:
                raw_detections.append({
                    "box": (bx1, by1, bx2, by2),
                    "score": round(score, 3),
                })

        if nms_mode in ("merge", "wbm", "weighted"):
            final_detections = weighted_box_merge(raw_detections, iou_thresh=iou_thresh)
        else:
            final_detections = self.non_max_suppression(raw_detections, iou_thresh=iou_thresh)

        t4 = time.perf_counter()
        nms_ms = (t4 - t3) * 1000.0
        total_ms = (t4 - t_start) * 1000.0

        timing_stats = {
            "proposal_ms": round(proposal_ms, 2),
            "crop_prep_ms": round(crop_prep_ms, 2),
            "cnn_inference_ms": round(cnn_inference_ms, 2),
            "nms_ms": round(nms_ms, 2),
            "total_ms": round(total_ms, 2),
            "fps": round(1000.0 / max(total_ms, 1e-3), 2),
            "num_windows": len(candidate_boxes),
        }

        return final_detections, timing_stats

    def predict_and_track(
        self,
        frame: np.ndarray,
        tracker: Any,
        frame_idx: int,
        conf_threshold: float = None,
        use_roi: bool = True,
        roi_mode: str = None,
        batch_size: int = 64,
        iou_thresh: float = 0.35,
        nms_mode: str = "nms",
        stride_ratio: float = 0.50,
        max_windows: int = 300,
        scale_fractions: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end frame processing: detect bounding boxes and pass them to PotholeTracker.
        Maintains unique track IDs, Kalman smoothing, and hit confirmation across frames.
        """
        detections = self.predict(
            frame=frame,
            conf_threshold=conf_threshold,
            use_roi=use_roi,
            roi_mode=roi_mode,
            batch_size=batch_size,
            iou_thresh=iou_thresh,
            nms_mode=nms_mode,
            stride_ratio=stride_ratio,
            max_windows=max_windows,
            scale_fractions=scale_fractions,
        )
        return tracker.update(detections, frame_idx=frame_idx)

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
