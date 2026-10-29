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
    max_candidates: int = 40,
) -> list:
    """
    Stage 1: Extract candidate pothole bounding boxes using classical CV.
    Uses bilateral filtering, adaptive thresholding, morphological filtering,
    and local depression contrast scoring.

    Parameters
    ----------
    frame : np.ndarray
        BGR image or video frame.
    use_roi : bool
        If True, restricts search to bottom 50% road region (for dashcam videos).
        If False, processes road portion of photo (ignoring top horizon).
    max_candidates : int
        Maximum number of candidate proposals to return, sorted by saliency score.

    Returns
    -------
    list of (x1, y1, x2, y2) tuples
    """
    h, w = frame.shape[:2]
    # Restrict to road surface: bottom 50% for dashcam videos,
    # or exclude top 15% (sky/horizon/trees) for still photos if height allows.
    roi_top = int(h * 0.50) if use_roi else (int(h * 0.15) if h > 200 else 0)
    roi = frame[roi_top:, :]
    roi_h, roi_w = roi.shape[:2]

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
    min_area = max(250, int(0.0006 * h * w))
    max_area = int(0.18 * h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.35 < aspect < 3.5:
                # Avoid touching the top border of full photos (sky/cars)
                if not use_roi and by <= 2:
                    continue

                # Local depression contrast: compare candidate interior with surrounding road margin
                pad_x = max(4, int(bw * 0.25))
                pad_y = max(4, int(bh * 0.25))
                ox1 = max(0, bx - pad_x)
                oy1 = max(0, by - pad_y)
                ox2 = min(roi_w, bx + bw + pad_x)
                oy2 = min(roi_h, by + bh + pad_y)

                inner = l_chan[by:by + bh, bx:bx + bw]
                outer = l_chan[oy1:oy2, ox1:ox2]
                inner_sum = float(np.sum(inner))
                outer_sum = float(np.sum(outer))
                inner_cnt = inner.size
                outer_cnt = outer.size

                margin_cnt = outer_cnt - inner_cnt
                if margin_cnt > 0:
                    margin_mean = (outer_sum - inner_sum) / margin_cnt
                    inner_mean = inner_sum / max(inner_cnt, 1)
                    contrast = margin_mean - inner_mean
                else:
                    contrast = 0.0

                # Reject regions that are significantly brighter than surrounding pavement
                # (e.g. white road stripes, reflector studs, bright highlights)
                if contrast < -2.5:
                    continue

                inner_std = float(np.std(inner))
                orig_y = by + roi_top

                # Saliency score: depression contrast + scale + interior texture
                score = (contrast + 3.0) * np.sqrt(area) * (1.0 + min(inner_std, 40.0) / 40.0)
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


class PotholeTracker:
    """
    Lightweight IoU & Centroid Object Tracker with Temporal Confirmation.
    - Associates detections across consecutive frames using IoU and centroid distance.
    - Assigns persistent, unique track IDs.
    - Counts unique potholes over a video stream.
    - Suppresses single-frame flickers (requires min_hits before confirmation).
    """

    def __init__(self, iou_thresh: float = 0.20, max_age: int = 5, min_hits: int = 2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = {}
        self.history = {}
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

    @staticmethod
    def _compute_centroid_dist(boxA, boxB):
        cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
        cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
        return float(np.hypot(cA[0] - cB[0], cA[1] - cB[1]))

    def update(self, detections: list, frame_idx: int = 0) -> list:
        """
        Update tracker with detections from current frame.

        Parameters
        ----------
        detections : list of dicts with keys "box", "score", (optionally "severity")
        frame_idx  : current frame index

        Returns
        -------
        list of active detections, each enriched with "track_id" and "hits"
        """
        det_matched = [False] * len(detections)
        active_ids = list(self.tracks.keys())
        track_matched = {tid: False for tid in active_ids}

        cost_matrix = []
        for d_idx, det in enumerate(detections):
            for tid in active_ids:
                tbox = self.tracks[tid]["box"]
                iou = self._compute_iou(det["box"], tbox)
                dist = self._compute_centroid_dist(det["box"], tbox)
                w = max(det["box"][2] - det["box"][0], 1)
                h = max(det["box"][3] - det["box"][1], 1)
                norm_dist = dist / float(np.hypot(w, h))
                cost_matrix.append((iou, -norm_dist, d_idx, tid))

        cost_matrix.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for iou, neg_norm_dist, d_idx, tid in cost_matrix:
            if det_matched[d_idx] or track_matched[tid]:
                continue
            if iou >= self.iou_thresh or neg_norm_dist >= -0.85:
                det = detections[d_idx]
                trk = self.tracks[tid]
                trk["box"] = det["box"]
                trk["score"] = det["score"]
                if "severity" in det:
                    trk["severity"] = det["severity"]
                    sev_rank = {"low": 1, "medium": 2, "high": 3}
                    if sev_rank.get(det["severity"], 1) > sev_rank.get(trk["peak_severity"], 1):
                        trk["peak_severity"] = det["severity"]
                trk["max_score"] = max(trk["max_score"], det["score"])
                trk["hits"] += 1
                trk["age"] = 0
                trk["last_frame"] = frame_idx
                if trk["hits"] >= self.min_hits:
                    trk["confirmed"] = True
                    self.confirmed_track_ids.add(tid)

                det_matched[d_idx] = True
                track_matched[tid] = True

        for d_idx, det in enumerate(detections):
            if not det_matched[d_idx]:
                tid = self.next_id
                self.next_id += 1
                confirmed = (self.min_hits <= 1)
                if confirmed:
                    self.confirmed_track_ids.add(tid)
                self.tracks[tid] = {
                    "id": tid,
                    "box": det["box"],
                    "score": det["score"],
                    "severity": det.get("severity", "low"),
                    "hits": 1,
                    "age": 0,
                    "confirmed": confirmed,
                    "max_score": det["score"],
                    "peak_severity": det.get("severity", "low"),
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                }
                track_matched[tid] = True

        dead_ids = []
        for tid in active_ids:
            if not track_matched.get(tid, False):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    dead_ids.append(tid)

        # Store latest state in history before deletion
        for tid, trk in self.tracks.items():
            if trk["confirmed"]:
                self.history[tid] = {
                    "track_id": trk["id"],
                    "first_frame": trk["first_frame"],
                    "last_frame": trk["last_frame"],
                    "total_hits": trk["hits"],
                    "max_score": round(trk["max_score"], 3),
                    "peak_severity": trk["peak_severity"],
                }

        for tid in dead_ids:
            del self.tracks[tid]

        output_dets = []
        for tid, trk in self.tracks.items():
            if trk["age"] == 0 and (trk["confirmed"] or self.min_hits <= 1):
                output_dets.append({
                    "box": trk["box"],
                    "score": trk["score"],
                    "severity": trk["severity"],
                    "track_id": trk["id"],
                    "hits": trk["hits"],
                })

        return output_dets

    def get_unique_count(self) -> int:
        """Total number of confirmed unique potholes tracked."""
        return len(self.confirmed_track_ids)

    def get_track_summary(self) -> list:
        """Summary records for all confirmed tracks."""
        return [self.history[tid] for tid in sorted(self.history.keys())]


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

    def extract_candidates(self, frame: np.ndarray, use_roi: bool = True) -> list:
        """Stage 1: Extract candidate bounding boxes."""
        return extract_stage1_candidates(frame, use_roi=use_roi)

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        use_roi: bool = True,
    ) -> list:
        """
        Detect potholes in a single frame.

        Stage 1: Classical region proposals generate candidate boxes.
        Stage 2: Each candidate patch is classified by ResPotholeNet.
                 The CNN sigmoid confidence is the SOLE accept/reject criterion.

        Returns:
            list of {"box": (x1, y1, x2, y2), "score": float}
        """
        candidate_boxes = self.extract_candidates(frame, use_roi=use_roi)
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
