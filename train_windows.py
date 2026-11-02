"""
train_windows.py

Step 3: Retrain ResPotholeNet on multi-scale sliding windows labeled with IoU
boundaries, using chronological (not random) train/val splits.

Plain-Words Explanation for Viva:
1. Why retrain on sliding windows?
   Previously, the CNN was only trained on proxy crops from Stage-1 classical
   filters (which had ~58% label noise and very few real pothole shapes).
   By retraining directly on sliding windows labeled against ground-truth boxes,
   the CNN learns what a true centred pothole looks like at various scales, and
   learns to reject background road patches (hard negatives from the same frames).
2. IoU Labeling Policy:
   - Positive (label 1): window IoU >= 0.50 with any GT pothole box.
   - Negative (label 0): window IoU < 0.20 with ALL GT boxes.
   - Ambiguous (skipped): 0.20 <= IoU < 0.50 excluded so the network is not
     confused by boundary-clipping patches.
3. Class Balancing & Augmentation:
   Negative windows vastly outnumber positives.  We cap negatives at
   negative_ratio * n_positives (default 2x) so the model sees both classes
   equally, then augment positives with horizontal flip, brightness/contrast
   jitter, and colour channel jitter.
4. From-Scratch Training (no pretrained weights):
   ResPotholeNet uses Kaiming-normal initialisation.  Training uses AdamW with
   cosine annealing LR.  We track the best epoch by val F1 (early stopping).
5. Validation-Based Threshold Selection (NEW):
   After training completes, sweep_val_threshold scores every val patch once
   (no gradient, raw probabilities) and tries candidate thresholds
   [0.30, 0.35, ..., 0.90].  The threshold that maximises val F1 is chosen.
   The TEST set is NEVER used in this sweep — it is evaluated exactly once
   when --run_test is passed.
6. Chronological Splits (NEW):
   Image paths come from data/splits/{train,val,test}_manifest.csv produced by
   tools/make_splits.py.  An assertion verifies that no segment_id appears in
   more than one split.

Usage:
    # Standard retraining (no test evaluation)
    python train_windows.py --epochs 15 --batch_size 32

    # Run test evaluation exactly once after training
    python train_windows.py --epochs 15 --batch_size 32 --run_test

    # Use a custom splits directory
    python train_windows.py --splits_dir data/splits --epochs 15

Notes:
    Label your extracted frames as YOLO .txt files alongside the JPEG files
    in data/extracted_frames/ before running this script.
    YOLO format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    (class_id is ignored — every box is treated as a pothole).
"""

import csv
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.pothole_net import ResPotholeNet, count_parameters
from detector import generate_sliding_windows, extract_crop, normalize_patch
from evaluate_box_level import compute_iou


# ─────────────────────────────────────────────────────────────────────────────
#  Loss Function (Focal Loss for Class Imbalance)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1).float()
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)
        pt = torch.where(target == 1, pred, 1.0 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        loss = -focal_weight * torch.log(pt)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Ground-Truth Box Loading (YOLO txt + JSON fallback)
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_boxes_for_image(
    image_path: Path,
    json_annotations: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Load ground-truth bounding boxes for a given image.
    Supports:
      1. YOLO .txt file alongside the image ({stem}.txt) with normalized (cx, cy, w, h).
      2. Lookup in json_annotations dict (e.g. test_ground_truth_boxes.json).
      3. Empty list for clean/normal road images.
    """
    boxes = []
    # 1. Check for YOLO .txt file
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            ih, iw = img.shape[:2]
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, bw, bh = map(float, parts[:5])
                        x1 = int((cx - bw / 2.0) * iw)
                        y1 = int((cy - bh / 2.0) * ih)
                        x2 = int((cx + bw / 2.0) * iw)
                        y2 = int((cy + bh / 2.0) * ih)
                        boxes.append((max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)))
            if boxes:
                return boxes

    # 2. Check JSON annotations
    if json_annotations:
        fname = image_path.name
        if fname in json_annotations:
            for b in json_annotations[fname].get("boxes", []):
                boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
#  Sliding Window Patch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowPotholeDataset(Dataset):
    """
    Dataset of multi-scale sliding window crops labeled with IoU boundaries:
      - Positive: IoU >= 0.50 with a GT pothole box
      - Negative: IoU < 0.20 with all GT boxes
      - Skipped: 0.20 <= IoU < 0.50
    """
    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],
        augment: bool = False,
    ):
        """
        samples: list of (patch_bgr_128x128, label)
        """
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch, label = self.samples[idx]
        patch = patch.copy()

        # Augmentation on training patches
        if self.augment:
            # 1. Random horizontal flip
            if random.random() > 0.5:
                patch = cv2.flip(patch, 1)

            # 2. Random brightness and contrast jitter
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)  # contrast
                beta = random.uniform(-15, 15)     # brightness
                patch = np.clip(alpha * patch + beta, 0, 255).astype(np.uint8)

            # 3. Slight color channel jitter
            if random.random() > 0.4:
                b_shift = random.randint(-8, 8)
                g_shift = random.randint(-8, 8)
                r_shift = random.randint(-8, 8)
                patch = np.clip(patch.astype(np.int16) + [b_shift, g_shift, r_shift], 0, 255).astype(np.uint8)

        norm = normalize_patch(patch)
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
        target = torch.tensor([label], dtype=torch.float32)
        return tensor, target


def extract_window_samples(
    image_paths: List[Path],
    json_annotations: Optional[Dict[str, Any]] = None,
    negative_ratio: float = 1.5,
) -> List[Tuple[np.ndarray, int]]:
    """
    Generate sliding windows for images, label them with IoU policy, extract 128x128 patches in memory,
    and balance positive and negative classes.
    """
    pos_samples: List[Tuple[np.ndarray, int]] = []
    neg_samples: List[Tuple[np.ndarray, int]] = []

    for ipath in image_paths:
        if not ipath.exists():
            continue
        img = cv2.imread(str(ipath))
        if img is None:
            continue

        gt_boxes = load_gt_boxes_for_image(ipath, json_annotations)
        roi_mode = "dashcam" if (img.shape[1] / max(img.shape[0], 1) >= 1.4) else "photo"
        windows = generate_sliding_windows(img, roi_mode=roi_mode, stride_ratio=0.55, max_windows=300)

        for w in windows:
            if not gt_boxes:
                # Clean road frame: background negative
                patch = extract_crop(img, w, crop_size=128, pad_ratio=0.08)
                if patch is not None:
                    neg_samples.append((patch, 0))
            else:
                max_iou = max([compute_iou(w, gb) for gb in gt_boxes], default=0.0)
                if max_iou >= 0.50:
                    patch = extract_crop(img, w, crop_size=128, pad_ratio=0.08)
                    if patch is not None:
                        pos_samples.append((patch, 1))
                elif max_iou < 0.20:
                    patch = extract_crop(img, w, crop_size=128, pad_ratio=0.08)
                    if patch is not None:
                        neg_samples.append((patch, 0))
                # 0.20 <= max_iou < 0.50 is intentionally skipped

    print(f"  Raw Window Counts: {len(pos_samples)} Positive (IoU >= 0.5) | {len(neg_samples)} Negative (IoU < 0.2)", flush=True)

    # Class balancing: balance negatives relative to positives
    max_negs = max(30, int(len(pos_samples) * negative_ratio))
    if len(neg_samples) > max_negs:
        random.seed(42)
        neg_samples = random.sample(neg_samples, max_negs)

    balanced = pos_samples + neg_samples
    random.seed(42)
    random.shuffle(balanced)
    print(f"  -> Balanced Dataset: {len(balanced)} total window samples ({len(pos_samples)} pos, {len(neg_samples)} neg)", flush=True)
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            conf, _ = model(x)
            scores = conf.squeeze(-1).cpu().numpy()
            targets = y.squeeze(-1).numpy()
            all_preds.extend((scores >= 0.5).astype(int).tolist())
            all_targets.extend(targets.astype(int).tolist())

    tp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 0)

    acc = (tp + tn) / max(len(all_targets), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Manifest-Based Split Loader  (NEW — Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def load_paths_from_manifest(manifest_csv: str) -> Tuple[List[Path], List[str]]:
    """
    Load image paths and segment IDs from a split manifest CSV.

    Plain-words viva explanation:
    tools/make_splits.py writes one CSV per split (train / val / test).
    Each row has image_path (relative) and segment_id (e.g.
    sample_pothole_video.mp4::seg_01_train).  We return both so the caller
    can assert that no segment_id appears in more than one split.

    Parameters
    ----------
    manifest_csv : str
        Path to {train,val,test}_manifest.csv.

    Returns
    -------
    (image_paths, segment_ids)
    """
    mp = Path(manifest_csv)
    if not mp.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_csv}\n"
            "Run: python tools/make_splits.py --manifest data/manifest.csv"
        )
    with open(mp, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    image_paths = [Path(r["image_path"]) for r in rows]
    segment_ids = [r["segment_id"] for r in rows]
    return image_paths, segment_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Validation Threshold Sweep  (NEW — Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_val_threshold(
    model: nn.Module,
    val_samples: List[Tuple[np.ndarray, int]],
    device: torch.device,
    thresholds: Optional[List[float]] = None,
    batch_size: int = 64,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find the confidence threshold that maximises F1 on the validation set.

    Plain-words viva explanation:
    We run exactly one forward pass over all val patches (no threshold applied,
    just raw sigmoid probabilities).  We then iterate over candidate thresholds
    [0.30, 0.35, ..., 0.90] in steps of 0.05, compute precision/recall/F1 at
    each, and return the threshold with the highest F1.  This is the
    ONLY place the validation set influences the threshold; the test set is
    never consulted until --run_test is explicitly passed.

    Parameters
    ----------
    model       : Trained ResPotholeNet in eval() mode.
    val_samples : List of (patch_bgr_128x128, label) tuples.
    device      : Torch device.
    thresholds  : Candidate thresholds to sweep (default: 0.30 to 0.90 step 0.05).
    batch_size  : Inference batch size.

    Returns
    -------
    (best_threshold, sweep_report_dict)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]

    val_ds = SlidingWindowPotholeDataset(val_samples, augment=False)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            conf, _ = model(x)
            probs = conf.squeeze(-1).cpu().numpy()
            labels = y.squeeze(-1).numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.astype(int).tolist())

    best_thresh = 0.50
    best_f1 = -1.0
    sweep_table: List[Dict[str, Any]] = []

    for t in thresholds:
        preds = [1 if p >= t else 0 for p in all_probs]
        tp = sum(1 for p, gt in zip(preds, all_labels) if p == 1 and gt == 1)
        fp = sum(1 for p, gt in zip(preds, all_labels) if p == 1 and gt == 0)
        fn = sum(1 for p, gt in zip(preds, all_labels) if p == 0 and gt == 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2.0 * prec * rec / max(prec + rec, 1e-6)
        sweep_table.append({
            "threshold": t,
            "precision": round(prec * 100, 2),
            "recall":    round(rec  * 100, 2),
            "f1":        round(f1   * 100, 2),
        })
        if f1 > best_f1:
            best_f1   = f1
            best_thresh = t

    return best_thresh, {
        "best_threshold": best_thresh,
        "best_val_f1_at_threshold": round(best_f1 * 100, 2),
        "threshold_sweep": sweep_table,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Routine
# ─────────────────────────────────────────────────────────────────────────────

def train_windows(
    splits_dir: str = "data/splits",
    gt_json: str = "data/test_ground_truth_boxes.json",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    negative_ratio: float = 2.0,
    save_path: str = "pothole_cnn_windows.pth",
    metrics_path: str = "model_metrics_windows.json",
    run_test: bool = False,
) -> Dict[str, Any]:
    """
    Retrain ResPotholeNet on sliding-window crops from chronological splits.

    Data flow:
      1. Load train/val image paths from manifest CSVs (tools/make_splits.py output).
      2. Assert no segment_id appears in both train and val (temporal integrity).
      3. Generate sliding windows, label with IoU policy, balance classes.
      4. Train with FocalLoss + AdamW + cosine LR.  Save best epoch by val F1.
      5. Sweep confidence threshold on val — pick best.  Never touch test here.
      6. Optionally (--run_test) evaluate on test split exactly once.
      7. Save weights + full metrics JSON.
    """
    print("=" * 70)
    print("    STEP 3 — RETRAINING ResPotholeNet ON SLIDING WINDOWS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Splits dir      : {splits_dir}")
    print(f"Save path       : {save_path}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : {batch_size}")
    print(f"Neg / pos ratio : {negative_ratio}")

    # ── 1. Load split manifests ───────────────────────────────────────────────
    train_paths, train_segs = load_paths_from_manifest(f"{splits_dir}/train_manifest.csv")
    val_paths,   val_segs   = load_paths_from_manifest(f"{splits_dir}/val_manifest.csv")

    # ── 2. Strict non-overlap assertion ──────────────────────────────────────
    # Plain-words: adjacent video frames look nearly identical.  If a segment
    # appeared in both train and val the model would essentially see val frames
    # during training, making generalisation metrics meaningless.
    train_seg_set = set(train_segs)
    val_seg_set   = set(val_segs)
    overlap = train_seg_set & val_seg_set
    assert len(overlap) == 0, (
        f"Contamination: these segment(s) appear in both train and val: {overlap}\n"
        "Re-run tools/make_splits.py to fix the split."
    )
    print(f"\nSplit non-overlap assertion: PASSED")
    print(f"  Train : {len(train_paths)} frames  segments={train_seg_set}")
    print(f"  Val   : {len(val_paths)} frames  segments={val_seg_set}")

    # ── 3. Optional JSON annotation fallback for data/potholes photos ─────────
    # If the user has already annotated the static pothole photo set
    # (data/test_ground_truth_boxes.json) those boxes are also used.  Extracted
    # frames are labeled exclusively from YOLO .txt files placed alongside the
    # JPEG files in data/extracted_frames/.
    json_annotations: Dict[str, Any] = {}
    gt_path = Path(gt_json)
    if gt_path.exists():
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        json_annotations = gt_data.get("annotations", {})
        print(f"  JSON annotations: {len(json_annotations)} images from {gt_json}")
    else:
        print(f"  JSON annotations: not found at {gt_json} (using YOLO txt labels only)")

    # ── 4. Extract & label sliding windows ────────────────────────────────────
    print("\nExtracting & labeling windows (train split)...", flush=True)
    train_samples = extract_window_samples(
        train_paths, json_annotations, negative_ratio=negative_ratio
    )

    print("\nExtracting & labeling windows (val split)...", flush=True)
    val_samples = extract_window_samples(
        val_paths, json_annotations, negative_ratio=negative_ratio
    )

    n_train_pos = sum(1 for _, lbl in train_samples if lbl == 1)
    n_val_pos   = sum(1 for _, lbl in val_samples   if lbl == 1)
    print(f"\nDataset summary:")
    print(f"  Train : {len(train_samples)} windows  ({n_train_pos} pos, {len(train_samples)-n_train_pos} neg)")
    print(f"  Val   : {len(val_samples)} windows  ({n_val_pos} pos, {len(val_samples)-n_val_pos} neg)")

    if n_train_pos == 0:
        print("\n  WARNING: Zero positive windows found in train split.")
        print("  Label your extracted frames as YOLO .txt files in data/extracted_frames/")
        print("  and re-run.  Training will proceed but the model cannot learn potholes.")

    train_ds = SlidingWindowPotholeDataset(train_samples, augment=True)
    val_ds   = SlidingWindowPotholeDataset(val_samples,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 5. Model from scratch ─────────────────────────────────────────────────
    print("\nInitialising ResPotholeNet from random weights (no pretrained backbone)...")
    model  = ResPotholeNet(in_channels=3).to(device)
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1    = -1.0
    best_state_dict: Optional[Dict] = None
    best_metrics: Dict[str, Any]    = {}

    # ── 6. Training loop (unchanged structure) ────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Acc':^9} | {'Val Prec':^9} | {'Val Rec':^9} | {'Val F1':^9}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        val_metrics = evaluate_model(model, val_loader, device)

        print(
            f"{epoch:^7} | {train_loss:^12.4f} | {val_metrics['accuracy']:^8.1f}% | "
            f"{val_metrics['precision']:^8.1f}% | {val_metrics['recall']:^8.1f}% | {val_metrics['f1_score']:^8.1f}%"
        )

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1    = val_metrics["f1_score"]
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics    = dict(val_metrics)
            best_metrics["epoch"]      = epoch
            best_metrics["train_loss"] = round(train_loss, 4)

    # ── 7. Save best weights (strict=True compatible) ─────────────────────────
    # Save the raw state-dict so that  model.load_state_dict(torch.load(...), strict=True)
    # works without any wrapper keys.
    if best_state_dict is not None:
        torch.save(best_state_dict, save_path)
        print(f"\nSaved best weights -> {save_path}  (Val F1 @ threshold 0.50 = {best_val_f1:.1f}%)")

        # Reload best weights into model for threshold sweep
        model.load_state_dict(best_state_dict)
    else:
        print("\nWARNING: No improvement observed; weights not saved.")

    # ── 8. Threshold sweep on validation (never on test) ─────────────────────
    # Plain-words: we score all val patches at raw probability, then try every
    # threshold in [0.30, 0.90] and pick the one with best val F1.  Doing this
    # on val (not test) respects the rule "choose threshold on validation,
    # run test exactly once".
    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD SWEEP (test set not touched)")
    print("=" * 70)

    best_thresh, thresh_report = sweep_val_threshold(
        model, val_samples, device, batch_size=batch_size
    )

    print(f"{'Threshold':^11} | {'Precision':^10} | {'Recall':^8} | {'F1':^8}")
    print("-" * 46)
    for row in thresh_report["threshold_sweep"]:
        marker = " <-- BEST" if row["threshold"] == best_thresh else ""
        print(f"  {row['threshold']:.2f}      |  {row['precision']:>7.2f}%  |  {row['recall']:>5.2f}%  |  {row['f1']:>5.2f}%{marker}")
    print(f"\nChosen threshold for inference: {best_thresh:.2f}  (Val F1 = {thresh_report['best_val_f1_at_threshold']:.2f}%)")

    # ── 9. Test evaluation — exactly once, only if --run_test ────────────────
    test_metrics: Optional[Dict[str, Any]] = None
    if run_test:
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION  (one-time only — do not re-run to select thresholds)")
        print("=" * 70)
        test_paths, test_segs = load_paths_from_manifest(f"{splits_dir}/test_manifest.csv")

        # Assert test has no overlap with train or val
        test_seg_set = set(test_segs)
        assert len(train_seg_set & test_seg_set) == 0, \
            f"Contamination: train and test share segments {train_seg_set & test_seg_set}"
        assert len(val_seg_set & test_seg_set) == 0, \
            f"Contamination: val and test share segments {val_seg_set & test_seg_set}"

        print(f"Test frames: {len(test_paths)}  segments={test_seg_set}")
        print("Extracting & labeling test windows...", flush=True)
        test_samples = extract_window_samples(test_paths, json_annotations, negative_ratio=2.0)
        n_test_pos = sum(1 for _, lbl in test_samples if lbl == 1)
        print(f"Test windows: {len(test_samples)} ({n_test_pos} pos, {len(test_samples)-n_test_pos} neg)")

        # Use the validation-chosen threshold for test evaluation
        test_ds     = SlidingWindowPotholeDataset(test_samples, augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Collect raw probabilities then apply chosen threshold
        model.eval()
        all_probs_test: List[float] = []
        all_labels_test: List[int]  = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                conf, _ = model(x)
                all_probs_test.extend(conf.squeeze(-1).cpu().numpy().tolist())
                all_labels_test.extend(y.squeeze(-1).numpy().astype(int).tolist())

        preds_test = [1 if p >= best_thresh else 0 for p in all_probs_test]
        tp = sum(1 for p, gt in zip(preds_test, all_labels_test) if p == 1 and gt == 1)
        fp = sum(1 for p, gt in zip(preds_test, all_labels_test) if p == 1 and gt == 0)
        fn = sum(1 for p, gt in zip(preds_test, all_labels_test) if p == 0 and gt == 1)
        tn = sum(1 for p, gt in zip(preds_test, all_labels_test) if p == 0 and gt == 0)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2.0 * prec * rec / max(prec + rec, 1e-6)
        acc  = (tp + tn) / max(len(all_labels_test), 1)

        test_metrics = {
            "threshold_used": best_thresh,
            "accuracy":  round(acc  * 100, 2),
            "precision": round(prec * 100, 2),
            "recall":    round(rec  * 100, 2),
            "f1_score":  round(f1   * 100, 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "test_windows": len(test_samples),
            "test_pos_windows": n_test_pos,
        }
        print(f"\nTest Results (threshold={best_thresh:.2f}):")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1        : {f1*100:.2f}%")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ── 10. Save full metrics JSON ────────────────────────────────────────────
    metrics_report: Dict[str, Any] = {
        "training_approach": "Step 3: Multi-Scale Sliding Window IoU-Supervised Retraining",
        "split_method": "Contiguous chronological segments per source video (no random shuffle)",
        "parameters": params,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "iou_positive_threshold": 0.50,
        "iou_negative_threshold": 0.20,
        "negative_ratio": negative_ratio,
        "best_epoch": best_metrics.get("epoch", epochs),
        "best_val_metrics_at_0.50": best_metrics,
        "threshold_selection": thresh_report,
        "chosen_inference_threshold": best_thresh,
        "weights_file": save_path,
        "dataset_summary": {
            "train_images": len(train_paths),
            "val_images":   len(val_paths),
            "train_windows": len(train_samples),
            "val_windows":   len(val_samples),
            "train_pos_windows": n_train_pos,
            "val_pos_windows":   n_val_pos,
        },
    }
    if test_metrics is not None:
        metrics_report["test_metrics"] = test_metrics

    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\nSaved full metrics report -> {metrics_path}")
    print("=" * 70)
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Retrain ResPotholeNet on sliding-window crops labeled by IoU."
    )
    parser.add_argument(
        "--splits_dir", type=str, default="data/splits",
        help="Directory containing train/val/test_manifest.csv from tools/make_splits.py"
    )
    parser.add_argument(
        "--gt_json", type=str, default="data/test_ground_truth_boxes.json",
        help="Optional JSON annotation file for static pothole photos"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Initial learning rate for AdamW (default: 0.001)"
    )
    parser.add_argument(
        "--negative_ratio", type=float, default=2.0,
        help="Max negatives = negative_ratio * n_positives (default: 2.0)"
    )
    parser.add_argument(
        "--save_path", type=str, default="pothole_cnn_windows.pth",
        help="Where to save best model weights (default: pothole_cnn_windows.pth)"
    )
    parser.add_argument(
        "--metrics_path", type=str, default="model_metrics_windows.json",
        help="Where to save the training metrics JSON"
    )
    parser.add_argument(
        "--run_test", action="store_true",
        help="Evaluate on the test split exactly once after training (do not use for threshold tuning)"
    )
    args = parser.parse_args()

    train_windows(
        splits_dir=args.splits_dir,
        gt_json=args.gt_json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        negative_ratio=args.negative_ratio,
        save_path=args.save_path,
        metrics_path=args.metrics_path,
        run_test=args.run_test,
    )
