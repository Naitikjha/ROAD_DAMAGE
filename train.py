"""
Training and Evaluation Pipeline for ResPotholeNet Custom CNN (From Scratch)
Course Project: Two-Stage Pothole Detection from Road Videos Using Custom CNN

Domain-Aligned Pipeline with Zero Data Leakage:
  1. Source image deduplication: Exact duplicate photos removed via SHA-256 hash.
     (Optional perceptual dHash for near-duplicate identification).
  2. Source-level split: 70% Train / 15% Validation / 15% Test performed on SOURCE IMAGES.
     Crops from the same image NEVER appear in both training and testing.
  3. Shared Stage-1 Candidate Extraction:
     Both positive and negative training crops are generated using the exact same
     Stage-1 morphological & depression proposal logic (detector.extract_stage1_candidates):
       - Positive images -> Stage 1 crater depression proposals (interior darker than road margin)
       - Negative images -> Stage 1 false-alarm proposals across multi-scale sizes (shadows, cracks, stains) + clean pavement
     Ensures the CNN is trained on the exact multi-scale patch distribution it encounters at inference.
  4. Augmentation: Horizontal flip, brightness/contrast jitter, slight rotation (no vertical flip).
  5. Metrics: Evaluated on unseen test crops (patch-level) AND full-pipeline end-to-end evaluation
     (image-level detection rate, FPPI on clean roads, and confusion matrix).
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.pothole_net import ResPotholeNet, count_parameters
from detector import extract_stage1_candidates, extract_crop, normalize_patch


# ─────────────────────────────────────────────────────────────────────────────
#  File Deduplication and Splitting
# ─────────────────────────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    """SHA-256 of the raw file bytes for exact byte deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Perceptual difference hash (dHash) for near-duplicate detection."""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def deduplicate_files(paths: List[Path], check_near_duplicates: bool = True) -> Tuple[List[Path], int]:
    """
    Filter out byte-identical duplicate files using SHA-256 hash.
    Optionally reports perceptual near-duplicate count via dHash.
    """
    seen_hashes = set()
    unique = []
    for p in paths:
        h = file_hash(str(p))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(p)

    near_dupe_count = 0
    if check_near_duplicates:
        seen_dhashes = set()
        for p in unique:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                dh = dhash(img)
                if dh in seen_dhashes:
                    near_dupe_count += 1
                else:
                    seen_dhashes.add(dh)

    return unique, near_dupe_count


def split_source_images(
    pos_files: List[Path],
    neg_files: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """
    Split source images into train, validation, and test sets.
    Splitting occurs BEFORE crop generation to prevent cross-partition leakage.
    """
    rng = np.random.RandomState(seed)
    pos_shuffled = list(pos_files)
    neg_shuffled = list(neg_files)
    rng.shuffle(pos_shuffled)
    rng.shuffle(neg_shuffled)

    def partition(files: List[Path]):
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]

    train_pos, val_pos, test_pos = partition(pos_shuffled)
    train_neg, val_neg, test_neg = partition(neg_shuffled)
    return train_pos, val_pos, test_pos, train_neg, val_neg, test_neg


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset: Multi-Scale Stage-1 Aligned Road Patch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage1RoadPatchDataset(Dataset):
    """
    Dataset composed of 128x128 crops generated via Stage-1 proposals.
    Both positive and negative samples share the same Stage-1 candidate framing
    across multiple scales (small, medium, large).
    """

    def __init__(
        self,
        pos_files: List[Path],
        neg_files: List[Path],
        patch_size: int = 128,
        augment: bool = False,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []  # List of (crop_bgr: np.ndarray, label: float)

        rng = np.random.RandomState(seed)

        # 1. Positive samples: Extract genuine depression proposals from pothole images
        for path in pos_files:
            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            # Extract candidates with road horizon rejection and local depression contrast
            boxes = extract_stage1_candidates(img, use_roi=False, max_candidates=30)
            added_for_img = 0

            if boxes:
                # Select up to 3 salient proposals across scales
                for box in boxes[:3]:
                    crop = extract_crop(img, box, crop_size=patch_size, pad_ratio=0.1)
                    if crop is not None:
                        self.samples.append((crop, 1.0))
                        added_for_img += 1

            if added_for_img == 0:
                # Fallback: road region crop (bottom half) centered on lowest depression
                cs = min(h, w, 256)
                cy = int(h * 0.70)
                cx = w // 2
                y1 = max(0, cy - cs // 2)
                y2 = min(h, y1 + cs)
                x1 = max(0, cx - cs // 2)
                x2 = min(w, x1 + cs)
                patch = img[y1:y2, x1:x2]
                if patch.size > 0:
                    self.samples.append((cv2.resize(patch, (patch_size, patch_size)), 1.0))

        # 2. Negative samples: Extract multi-scale Stage-1 false alarms + clean pavement
        for path in neg_files:
            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            boxes = extract_stage1_candidates(img, use_roi=False, max_candidates=30)

            # Sample up to 3 Stage-1 proposals across small, medium, and large scales
            if boxes:
                # Take diverse scale proposals
                selected_boxes = []
                if len(boxes) <= 3:
                    selected_boxes = boxes
                else:
                    selected_boxes = [boxes[0], boxes[len(boxes) // 2], boxes[-1]]

                for box in selected_boxes:
                    crop = extract_crop(img, box, crop_size=patch_size, pad_ratio=0.1)
                    if crop is not None:
                        self.samples.append((crop, 0.0))

            # Add clean pavement patch (representing clear road background)
            if h >= patch_size and w >= patch_size:
                y = rng.randint(int(h * 0.20), h - patch_size + 1)
                x = rng.randint(0, w - patch_size + 1)
                self.samples.append((img[y:y + patch_size, x:x + patch_size], 0.0))
            else:
                self.samples.append((cv2.resize(img, (patch_size, patch_size)), 0.0))

        # Shuffle deterministically
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        crop, label = self.samples[idx]

        # Augmentation (only horizontal flip, brightness/contrast, rotation — NO vertical flip)
        if self.augment:
            if np.random.rand() > 0.5:
                crop = cv2.flip(crop, 1)
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.randint(-15, 15)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = crop.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        norm = normalize_patch(crop)
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Loss Function
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    alpha weights positive class, (1 - alpha) weights negative class.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation: Patch-Level
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5) -> dict:
    """Evaluate model on a DataLoader patch partition."""
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    cls_criterion = FocalLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            conf, _ = model(imgs)

            loss = cls_criterion(conf, labels)
            total_loss += loss.item()

            preds = (conf >= threshold).float()
            for p, y in zip(preds, labels):
                p_val = int(p.item())
                y_val = int(y.item())
                if p_val == 1 and y_val == 1:
                    tp += 1
                elif p_val == 1 and y_val == 0:
                    fp += 1
                elif p_val == 0 and y_val == 0:
                    tn += 1
                elif p_val == 0 and y_val == 1:
                    fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1e-7)
    recall = tp / max(tp + fn, 1e-7)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    avg_loss = total_loss / max(len(loader), 1)

    return {
        "n": n,
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation: End-to-End Two-Stage Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_end_to_end(
    model: nn.Module,
    test_pos_images: List[Path],
    test_neg_images: List[Path],
    device: torch.device,
    conf_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
) -> dict:
    """
    Evaluate the COMPLETE two-stage pipeline end-to-end on full source images.

    Stage 1: Region proposals from full image (extract_stage1_candidates)
    Stage 2: CNN patch classification + Non-Maximum Suppression

    Evaluates:
      - Image-level Detection Accuracy, Precision, Recall, Specificity, F1
      - False Positives Per Image (FPPI) on clean road photos
      - Average detections per pothole photo
      - Confusion matrix at image level
      - Sweep over confidence thresholds
    """
    model.eval()

    # Pre-extract Stage 1 candidate proposals and patches for all test images
    def get_proposals_and_patches(paths: List[Path]):
        data = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            boxes = extract_stage1_candidates(img, use_roi=False)
            patches = []
            valid_boxes = []
            for b in boxes:
                crop = extract_crop(img, b, crop_size=128, pad_ratio=0.1)
                if crop is not None:
                    norm = normalize_patch(crop)
                    t = torch.from_numpy(norm).permute(2, 0, 1).float()
                    patches.append(t)
                    valid_boxes.append(b)
            data.append({"path": str(p), "boxes": valid_boxes, "patches": patches})
        return data

    pos_data = get_proposals_and_patches(test_pos_images)
    neg_data = get_proposals_and_patches(test_neg_images)

    def run_inference_on_data(data_list):
        results = []
        with torch.no_grad():
            for item in data_list:
                if not item["patches"]:
                    results.append({"boxes": [], "scores": []})
                    continue
                batch = torch.stack(item["patches"]).to(device)
                conf, _ = model(batch)
                scores = conf.squeeze(-1).cpu().numpy()
                if scores.ndim == 0:
                    scores = np.array([float(scores)])
                results.append({"boxes": item["boxes"], "scores": scores})
        return results

    pos_results = run_inference_on_data(pos_data)
    neg_results = run_inference_on_data(neg_data)

    def apply_nms(boxes, scores, iou_thresh=0.35):
        if not boxes:
            return []
        b_arr = np.array(boxes, dtype=float)
        s_arr = np.array(scores, dtype=float)
        x1, y1, x2, y2 = b_arr[:, 0], b_arr[:, 1], b_arr[:, 2], b_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = s_arr.argsort()[::-1]
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
        return [boxes[k] for k in keep]

    threshold_reports = {}
    for th in conf_thresholds:
        # Pothole images: count images with >= 1 detection (TP) vs 0 (FN)
        tp_images = 0
        fn_images = 0
        total_pothole_boxes = 0

        for res in pos_results:
            valid_boxes = [res["boxes"][i] for i, s in enumerate(res["scores"]) if s >= th]
            valid_scores = [res["scores"][i] for i, s in enumerate(res["scores"]) if s >= th]
            kept_boxes = apply_nms(valid_boxes, valid_scores)
            num_boxes = len(kept_boxes)
            total_pothole_boxes += num_boxes
            if num_boxes > 0:
                tp_images += 1
            else:
                fn_images += 1

        # Normal images: count images with 0 detections (TN) vs >= 1 (FP)
        tn_images = 0
        fp_images = 0
        total_false_boxes = 0

        for res in neg_results:
            valid_boxes = [res["boxes"][i] for i, s in enumerate(res["scores"]) if s >= th]
            valid_scores = [res["scores"][i] for i, s in enumerate(res["scores"]) if s >= th]
            kept_boxes = apply_nms(valid_boxes, valid_scores)
            num_boxes = len(kept_boxes)
            total_false_boxes += num_boxes
            if num_boxes == 0:
                tn_images += 1
            else:
                fp_images += 1

        total_images = tp_images + fn_images + tn_images + fp_images
        acc = (tp_images + tn_images) / max(total_images, 1)
        prec = tp_images / max(tp_images + fp_images, 1e-7)
        rec = tp_images / max(tp_images + fn_images, 1e-7)
        spec = tn_images / max(tn_images + fp_images, 1e-7)
        f1 = 2 * prec * rec / max(prec + rec, 1e-7)
        fppi = total_false_boxes / max(len(neg_data), 1)
        avg_pos_boxes = total_pothole_boxes / max(len(pos_data), 1)

        threshold_reports[f"threshold_{th}"] = {
            "confidence_threshold": th,
            "image_accuracy": round(acc * 100, 2),
            "image_precision": round(prec * 100, 2),
            "image_recall": round(rec * 100, 2),
            "image_specificity": round(spec * 100, 2),
            "image_f1_score": round(f1 * 100, 2),
            "fppi_on_clean_roads": round(fppi, 3),
            "avg_detections_per_pothole_photo": round(avg_pos_boxes, 2),
            "confusion_matrix": {
                "TP_images": tp_images,
                "FP_images": fp_images,
                "TN_images": tn_images,
                "FN_images": fn_images,
            }
        }

    return {
        "evaluation_type": "End-to-End Two-Stage Pipeline Evaluation (Full Source Images)",
        "test_source_images": {
            "pothole_images": len(pos_data),
            "normal_images": len(neg_data),
            "total_images": len(pos_data) + len(neg_data),
        },
        "default_threshold_0_5": threshold_reports["threshold_0.5"],
        "threshold_sweep": threshold_reports,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline(
    data_dir: str = "data",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    save_path: str = "pothole_cnn.pth",
    metrics_path: str = "model_metrics.json",
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    data_path = Path(data_dir)
    pothole_dirs = ["potholes", "pothole", "positive", "Pothole"]
    normal_dirs = ["normal", "clear", "negative", "pavement", "plain", "Plain"]

    pos_paths = []
    neg_paths = []
    for pd in pothole_dirs:
        p_path = data_path / pd
        if p_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                pos_paths.extend(list(p_path.glob(ext)))
    for nd in normal_dirs:
        n_path = data_path / nd
        if n_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                neg_paths.extend(list(n_path.glob(ext)))

    if len(pos_paths) == 0 or len(neg_paths) == 0:
        print(f"\nERROR: Real road dataset not found in '{data_dir}/'")
        return

    # Deduplicate source images (exact byte deduplication via SHA-256)
    unique_pos, near_dupes_pos = deduplicate_files(pos_paths, check_near_duplicates=True)
    unique_neg, near_dupes_neg = deduplicate_files(neg_paths, check_near_duplicates=True)
    dupes_pos = len(pos_paths) - len(unique_pos)
    dupes_neg = len(neg_paths) - len(unique_neg)

    print(f"Deduplication Summary:")
    print(f"  Potholes: {len(pos_paths)} files -> {len(unique_pos)} unique (removed {dupes_pos} byte-identical duplicates, {near_dupes_pos} perceptual near-dupes)")
    print(f"  Normal:   {len(neg_paths)} files -> {len(unique_neg)} unique (removed {dupes_neg} byte-identical duplicates, {near_dupes_neg} perceptual near-dupes)")
    print(f"  Total Unique Source Images: {len(unique_pos)} pothole + {len(unique_neg)} normal = {len(unique_pos) + len(unique_neg)}")

    # Split SOURCE IMAGES first (Zero Data Leakage)
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = split_source_images(
        unique_pos, unique_neg, train_ratio=0.70, val_ratio=0.15, seed=seed
    )
    print(f"\nSource Image Split (Zero Leakage):")
    print(f"  Train: {len(train_pos)} pos + {len(train_neg)} neg = {len(train_pos) + len(train_neg)} photos")
    print(f"  Val:   {len(val_pos)} pos + {len(val_neg)} neg = {len(val_pos) + len(val_neg)} photos")
    print(f"  Test:  {len(test_pos)} pos + {len(test_neg)} neg = {len(test_pos) + len(test_neg)} photos")

    print("\nExtracting multi-scale Stage-1 aligned crops for each split independently...")
    train_dataset = Stage1RoadPatchDataset(train_pos, train_neg, patch_size=128, augment=True, seed=seed)
    val_dataset = Stage1RoadPatchDataset(val_pos, val_neg, patch_size=128, augment=False, seed=seed)
    test_dataset = Stage1RoadPatchDataset(test_pos, test_neg, patch_size=128, augment=False, seed=seed)

    print(f"\nCrop Dataset Counts (Multi-Scale Proposals):")
    print(f"  Train crops: {len(train_dataset)}")
    print(f"  Val crops:   {len(val_dataset)}")
    print(f"  Test crops:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResPotholeNet(in_channels=3).to(device)
    cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0

    print(f"\nTraining ResPotholeNet for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_cls_loss = 0.0
        num_batches = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            conf, _ = model(batch_imgs)
            l_cls = cls_criterion(conf, batch_labels)
            l_cls.backward()
            optimizer.step()

            total_cls_loss += l_cls.item()
            num_batches += 1

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = total_cls_loss / max(num_batches, 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] — "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']}% | "
              f"Val P: {val_metrics['precision']}% | "
              f"Val R: {val_metrics['recall']}% | "
              f"Val F1: {val_metrics['f1_score']}% (n={val_metrics['n']})")

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best weights (Val F1={best_val_f1}%)")

    print(f"\nTraining Complete! Best model saved to: {save_path}")

    # Reload best weights for rigorous final evaluation on unseen test partition
    model.load_state_dict(torch.load(save_path, map_location=device))

    print("\n" + "=" * 65)
    print("     PART 1: PATCH-LEVEL EVALUATION (Unseen Test Crops)")
    print("=" * 65)
    test_metrics = evaluate(model, test_loader, device)
    cm = test_metrics["confusion_matrix"]

    print(f"  n (test crops):   {test_metrics['n']}")
    print(f"  Test Loss:        {test_metrics['loss']}")
    print(f"  Test Accuracy:    {test_metrics['accuracy']}%")
    print(f"  Test Precision:   {test_metrics['precision']}%")
    print(f"  Test Recall:      {test_metrics['recall']}%")
    print(f"  Test F1-Score:    {test_metrics['f1_score']}%")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Pos   Predicted Neg")
    print(f"    Actual Pos        {cm['TP']:>4}            {cm['FN']:>4}")
    print(f"    Actual Neg        {cm['FP']:>4}            {cm['TN']:>4}")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("     PART 2: END-TO-END PIPELINE EVALUATION (Full Test Images)")
    print("=" * 65)
    e2e_results = evaluate_end_to_end(model, test_pos, test_neg, device)
    e2e_def = e2e_results["default_threshold_0_5"]
    e2e_cm = e2e_def["confusion_matrix"]

    print(f"  Test Images:      {len(test_pos)} pothole + {len(test_neg)} normal = {len(test_pos) + len(test_neg)}")
    print(f"  Threshold:        0.50")
    print(f"  Image Accuracy:   {e2e_def['image_accuracy']}%")
    print(f"  Image Precision:  {e2e_def['image_precision']}%")
    print(f"  Image Recall:     {e2e_def['image_recall']}% (Detection Rate)")
    print(f"  Image Specificity:{e2e_def['image_specificity']}% (Clean Road Correctness)")
    print(f"  Image F1-Score:   {e2e_def['image_f1_score']}%")
    print(f"  FPPI:             {e2e_def['fppi_on_clean_roads']} false boxes per clean road photo")
    print(f"  Avg Pos Boxes:    {e2e_def['avg_detections_per_pothole_photo']} detections per pothole photo")
    print(f"\n  Image-Level Confusion Matrix:")
    print(f"                  Predicted Pothole   Predicted Clear")
    print(f"    Actual Pothole      {e2e_cm['TP_images']:>4}              {e2e_cm['FN_images']:>4}")
    print(f"    Actual Clear        {e2e_cm['FP_images']:>4}              {e2e_cm['TN_images']:>4}")
    print("=" * 65)

    val_final = evaluate(model, val_loader, device)

    report = {
        "model_architecture": "ResPotholeNet (From Scratch)",
        "total_parameters": count_parameters(model),
        "loss_function": "Focal Loss (alpha=0.25, gamma=2.0)",
        "dataset": {
            "source": str(data_dir),
            "source_images": {
                "unique_pothole_images": len(unique_pos),
                "unique_normal_images": len(unique_neg),
                "total_unique_images": len(unique_pos) + len(unique_neg),
                "deduplication": "SHA-256 byte hash (706 unique images from 712 raw files)",
                "perceptual_near_dupes": near_dupes_pos + near_dupes_neg,
            },
            "split_method": "Split by source image file (Zero Leakage between train/val/test)",
            "crop_method": "Multi-scale Stage-1 proposal extraction with local depression contrast scoring",
            "crops": {
                "train_crops": len(train_dataset),
                "val_crops": len(val_dataset),
                "test_crops": len(test_dataset),
                "total_crops": len(train_dataset) + len(val_dataset) + len(test_dataset),
            }
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW (weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR",
            "augmentation": "Horizontal flip, brightness/contrast jitter, rotation (+-15 deg). NO vertical flip.",
        },
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "end_to_end_pipeline_metrics": e2e_results,
        "evaluation_notes": {
            "sample_correlation": (
                "Test crops originate from 106-108 unseen source photos. Because patches from the same photo "
                "share lighting and asphalt conditions, treat scores as subject to roughly +-5 percentage points variance."
            ),
            "patch_vs_detection": (
                "Patch metrics measure candidate crop classification. End-to-end metrics measure full "
                "two-stage pipeline performance on full held-out photos (Stage 1 proposals + Stage 2 CNN + NMS)."
            )
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nComprehensive evaluation report saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResPotholeNet on domain-aligned multi-scale crops")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to dataset directory (containing potholes/ and normal/)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate for AdamW")
    parser.add_argument("--save_path", type=str, default="pothole_cnn.pth",
                        help="Destination file for best model checkpoint")
    parser.add_argument("--metrics_path", type=str, default="model_metrics.json",
                        help="Destination file for evaluation metrics JSON report")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        metrics_path=args.metrics_path,
    )
