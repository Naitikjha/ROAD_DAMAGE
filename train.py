"""
Training and Evaluation Pipeline for ResPotholeNet Custom CNN (From Scratch)
Course Project: Two-Stage Pothole Detection from Road Videos Using Custom CNN

Domain-Aligned Pipeline with Zero Data Leakage & Reproducibility:
  1. Source image deduplication: Exact byte duplicates removed via SHA-256 hash.
     Perceptual near-duplicates removed via dHash (Hamming distance <= 2).
  2. Deterministic source-level split: Sorted file paths partitioned into
     70% Train / 15% Validation / 15% Test performed on SOURCE IMAGES.
     Crops from the same image NEVER appear across partitions.
  3. Symmetric Stage-1 Candidate Extraction:
     Both positive and negative training crops are generated using the EXACT same
     Stage-1 morphological & depression proposal logic (detector.extract_stage1_candidates):
       - Positive images -> Top 2 Stage 1 depression proposals
       - Negative images -> Top 2 Stage 1 proposals (hard negatives across scales)
     Zero structural rank or size asymmetry between positive and negative classes.
  4. Augmentation: Horizontal flip, brightness/contrast jitter, slight rotation (no vertical flip).
  5. Validation-Tuned Threshold: Optimal operating threshold is selected on the VALIDATION set,
     then evaluated once on the held-out TEST set (preventing test set leakage).
  6. Multi-Level Metrics: Patch-level classification metrics + end-to-end full pipeline evaluation.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

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


def deduplicate_files(paths: List[Path], dhash_threshold: int = 2) -> Tuple[List[Path], int, int]:
    """
    Remove byte-identical duplicates (SHA-256) AND perceptual near-duplicates (dHash).
    Guarantees that near-duplicate photos cannot leak across train/val/test splits.
    """
    # 1. Exact byte deduplication
    seen_sha = set()
    sha_unique = []
    byte_dupes = 0
    for p in paths:
        h = file_hash(str(p))
        if h not in seen_sha:
            seen_sha.add(h)
            sha_unique.append(p)
        else:
            byte_dupes += 1

    # 2. Perceptual near-duplicate removal
    seen_dhashes = {}
    clean_paths = []
    near_dupes = 0

    for p in sha_unique:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        dh = dhash(img)
        is_dupe = False
        for exh in seen_dhashes:
            dist = bin(dh ^ exh).count("1")
            if dist <= dhash_threshold:
                is_dupe = True
                near_dupes += 1
                break
        if not is_dupe:
            seen_dhashes[dh] = p
            clean_paths.append(p)

    return clean_paths, byte_dupes, near_dupes


def split_source_images(
    pos_files: List[Path],
    neg_files: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """
    Split source images into train, validation, and test sets.
    Lists are sorted first to ensure deterministic, cross-platform reproducibility.
    Splitting occurs BEFORE crop generation to prevent cross-partition leakage.
    """
    pos_sorted = sorted(pos_files)
    neg_sorted = sorted(neg_files)

    rng = np.random.RandomState(seed)
    pos_shuffled = list(pos_sorted)
    neg_shuffled = list(neg_sorted)
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
#  Dataset: Symmetric Stage-1 Road Patch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage1RoadPatchDataset(Dataset):
    """
    Dataset composed of 128x128 crops generated via Stage-1 proposals.
    Both positive and negative samples share the EXACT same Stage-1 candidate
    sampling strategy (top 2 proposals per photo) to eliminate size/rank shortcuts.
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

        # 1. Positive samples: Top 2 Stage-1 candidate proposals from pothole images
        for path in sorted(pos_files):
            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            boxes = extract_stage1_candidates(img, use_roi=False, max_candidates=20)
            added = 0

            if boxes:
                for box in boxes[:2]:
                    crop = extract_crop(img, box, crop_size=patch_size, pad_ratio=0.1)
                    if crop is not None:
                        self.samples.append((crop, 1.0))
                        added += 1

            if added == 0:
                # Road center fallback
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

        # 2. Negative samples: Top 2 Stage-1 candidate proposals from normal images
        # Exactly identical sampling rule as positive class
        for path in sorted(neg_files):
            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            boxes = extract_stage1_candidates(img, use_roi=False, max_candidates=20)
            added = 0

            if boxes:
                for box in boxes[:2]:
                    crop = extract_crop(img, box, crop_size=patch_size, pad_ratio=0.1)
                    if crop is not None:
                        self.samples.append((crop, 0.0))
                        added += 1

            if added == 0:
                # Clean road center fallback
                cs = min(h, w, 256)
                cy = int(h * 0.70)
                cx = w // 2
                y1 = max(0, cy - cs // 2)
                y2 = min(h, y1 + cs)
                x1 = max(0, cx - cs // 2)
                x2 = min(w, x1 + cs)
                patch = img[y1:y2, x1:x2]
                if patch.size > 0:
                    self.samples.append((cv2.resize(patch, (patch_size, patch_size)), 0.0))

        # Deterministic shuffle
        rng = np.random.RandomState(seed)
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
    """Focal Loss for binary classification."""
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
#  Evaluation: End-to-End Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_on_images(
    model: nn.Module,
    pos_images: List[Path],
    neg_images: List[Path],
    device: torch.device,
    conf_threshold: float = 0.5,
) -> dict:
    """
    Run full two-stage pipeline on source images at a specific threshold.
    """
    model.eval()

    def process_group(paths):
        counts = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            boxes = extract_stage1_candidates(img, use_roi=False)
            if not boxes:
                counts.append(0)
                continue

            patches = []
            for b in boxes:
                crop = extract_crop(img, b, crop_size=128, pad_ratio=0.1)
                if crop is not None:
                    norm = normalize_patch(crop)
                    patches.append(torch.from_numpy(norm).permute(2, 0, 1).float())

            if not patches:
                counts.append(0)
                continue

            batch = torch.stack(patches).to(device)
            with torch.no_grad():
                conf, _ = model(batch)
                scores = conf.squeeze(-1).cpu().numpy()
                if scores.ndim == 0:
                    scores = np.array([float(scores)])

            # Count detections passing threshold
            passing_boxes = [boxes[i] for i, s in enumerate(scores) if s >= conf_threshold]
            counts.append(len(passing_boxes))
        return counts

    pos_counts = process_group(pos_images)
    neg_counts = process_group(neg_images)

    tp_images = sum(1 for c in pos_counts if c > 0)
    fn_images = sum(1 for c in pos_counts if c == 0)
    tn_images = sum(1 for c in neg_counts if c == 0)
    fp_images = sum(1 for c in neg_counts if c > 0)

    total_images = tp_images + fn_images + tn_images + fp_images
    acc = (tp_images + tn_images) / max(total_images, 1)
    prec = tp_images / max(tp_images + fp_images, 1e-7)
    rec = tp_images / max(tp_images + fn_images, 1e-7)
    spec = tn_images / max(tn_images + fp_images, 1e-7)
    f1 = 2 * prec * rec / max(prec + rec, 1e-7)
    fppi = sum(neg_counts) / max(len(neg_counts), 1)
    avg_pos = sum(pos_counts) / max(len(pos_counts), 1)

    return {
        "confidence_threshold": round(conf_threshold, 2),
        "image_accuracy": round(acc * 100, 2),
        "image_precision": round(prec * 100, 2),
        "image_recall": round(rec * 100, 2),
        "image_specificity": round(spec * 100, 2),
        "image_f1_score": round(f1 * 100, 2),
        "fppi_on_clean_roads": round(fppi, 3),
        "avg_detections_per_pothole_photo": round(avg_pos, 2),
        "confusion_matrix": {
            "TP_images": tp_images,
            "FP_images": fp_images,
            "TN_images": tn_images,
            "FN_images": fn_images,
        }
    }


def select_best_threshold_on_val(
    model: nn.Module,
    val_pos: List[Path],
    val_neg: List[Path],
    device: torch.device,
    candidates: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
) -> Tuple[float, dict, dict]:
    """
    Select optimal operating threshold strictly on the VALIDATION source images.
    Returns: (best_threshold, best_val_report, all_val_reports)
    """
    all_reports = {}
    best_th = 0.5
    best_f1 = -1.0
    best_report = None

    for th in candidates:
        rep = run_pipeline_on_images(model, val_pos, val_neg, device, conf_threshold=th)
        all_reports[f"threshold_{th:.1f}"] = rep
        if rep["image_f1_score"] > best_f1:
            best_f1 = rep["image_f1_score"]
            best_th = th
            best_report = rep

    return best_th, best_report, all_reports


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

    raw_pos_paths = []
    raw_neg_paths = []
    for pd in pothole_dirs:
        p_path = data_path / pd
        if p_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                raw_pos_paths.extend(list(p_path.glob(ext)))
    for nd in normal_dirs:
        n_path = data_path / nd
        if n_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                raw_neg_paths.extend(list(n_path.glob(ext)))

    if len(raw_pos_paths) == 0 or len(raw_neg_paths) == 0:
        print(f"\nERROR: Real road dataset not found in '{data_dir}/'")
        return

    # Sort raw paths for 100% deterministic reproducibility
    raw_pos_paths = sorted(raw_pos_paths)
    raw_neg_paths = sorted(raw_neg_paths)

    # Deduplicate: remove SHA-256 byte duplicates AND perceptual dHash near-duplicates
    clean_pos, byte_pos, near_pos = deduplicate_files(raw_pos_paths, dhash_threshold=2)
    clean_neg, byte_neg, near_neg = deduplicate_files(raw_neg_paths, dhash_threshold=2)

    print(f"Dataset Deduplication:")
    print(f"  Potholes: {len(raw_pos_paths)} raw -> removed {byte_pos} byte-identical + {near_pos} perceptual near-dupes -> {len(clean_pos)} clean")
    print(f"  Normal:   {len(raw_neg_paths)} raw -> removed {byte_neg} byte-identical + {near_neg} perceptual near-dupes -> {len(clean_neg)} clean")
    print(f"  Total Clean Unique Source Photos: {len(clean_pos) + len(clean_neg)}")

    # Split SOURCE IMAGES deterministically (Zero Data Leakage)
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = split_source_images(
        clean_pos, clean_neg, train_ratio=0.70, val_ratio=0.15, seed=seed
    )
    print(f"\nSource Image Split (Zero Leakage):")
    print(f"  Train: {len(train_pos)} pos + {len(train_neg)} neg = {len(train_pos) + len(train_neg)} photos")
    print(f"  Val:   {len(val_pos)} pos + {len(val_neg)} neg = {len(val_pos) + len(val_neg)} photos")
    print(f"  Test:  {len(test_pos)} pos + {len(test_neg)} neg = {len(test_pos) + len(test_neg)} photos")

    print("\nExtracting symmetric Stage-1 proposals for each split independently...")
    train_dataset = Stage1RoadPatchDataset(train_pos, train_neg, patch_size=128, augment=True, seed=seed)
    val_dataset = Stage1RoadPatchDataset(val_pos, val_neg, patch_size=128, augment=False, seed=seed)
    test_dataset = Stage1RoadPatchDataset(test_pos, test_neg, patch_size=128, augment=False, seed=seed)

    print(f"\nCrop Counts (Symmetric Proposals):")
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

    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device))

    print("\n" + "=" * 65)
    print("     THRESHOLD SELECTION ON VALIDATION SOURCE IMAGES")
    print("=" * 65)
    best_th, best_val_e2e, all_val_e2e = select_best_threshold_on_val(model, val_pos, val_neg, device)
    print(f"Optimal threshold chosen on Validation set: {best_th:.1f}")
    print(f"Validation Image F1: {best_val_e2e['image_f1_score']}% | Rec: {best_val_e2e['image_recall']}% | Spec: {best_val_e2e['image_specificity']}%")

    print("\n" + "=" * 65)
    print("     PART 1: PATCH-LEVEL EVALUATION (Unseen Test Crops)")
    print("=" * 65)
    test_metrics = evaluate(model, test_loader, device, threshold=0.5)
    cm = test_metrics["confusion_matrix"]

    print(f"  n (test crops):   {test_metrics['n']}")
    print(f"  Test Loss:        {test_metrics['loss']}")
    print(f"  Test Accuracy:    {test_metrics['accuracy']}%")
    print(f"  Test Precision:   {test_metrics['precision']}%")
    print(f"  Test Recall:      {test_metrics['recall']}%")
    print(f"  Test F1-Score:    {test_metrics['f1_score']}%")
    print(f"\n  Patch Confusion Matrix:")
    print(f"                  Predicted Pos   Predicted Neg")
    print(f"    Actual Pos        {cm['TP']:>4}            {cm['FN']:>4}")
    print(f"    Actual Neg        {cm['FP']:>4}            {cm['TN']:>4}")
    print("=" * 65)

    print("\n" + "=" * 65)
    print(f"     PART 2: END-TO-END PIPELINE EVALUATION (Held-out Test Photos)")
    print(f"     (Evaluated at Validation-Selected Threshold = {best_th:.1f})")
    print("=" * 65)
    test_e2e_best = run_pipeline_on_images(model, test_pos, test_neg, device, conf_threshold=best_th)
    test_e2e_default = run_pipeline_on_images(model, test_pos, test_neg, device, conf_threshold=0.5)
    e2e_cm = test_e2e_best["confusion_matrix"]

    print(f"  Test Photos:      {len(test_pos)} pothole + {len(test_neg)} normal = {len(test_pos) + len(test_neg)}")
    print(f"  Threshold:        {best_th:.1f} (Selected on Val)")
    print(f"  Image Accuracy:   {test_e2e_best['image_accuracy']}%")
    print(f"  Image Precision:  {test_e2e_best['image_precision']}%")
    print(f"  Image Recall:     {test_e2e_best['image_recall']}% (Detection Rate)")
    print(f"  Image Specificity:{test_e2e_best['image_specificity']}% (Clean Road Correctness)")
    print(f"  Image F1-Score:   {test_e2e_best['image_f1_score']}%")
    print(f"  Clean Road FPPI:  {test_e2e_best['fppi_on_clean_roads']} false boxes per clean road photo")
    print(f"  Avg Detections:   {test_e2e_best['avg_detections_per_pothole_photo']} detections per pothole photo")
    print(f"\n  Image-Level Confusion Matrix (at threshold {best_th:.1f}):")
    print(f"                  Predicted Pothole   Predicted Clear")
    print(f"    Actual Pothole      {e2e_cm['TP_images']:>4}              {e2e_cm['FN_images']:>4}")
    print(f"    Actual Clear        {e2e_cm['FP_images']:>4}              {e2e_cm['TN_images']:>4}")
    print("=" * 65)

    val_final = evaluate(model, val_loader, device, threshold=0.5)

    report = {
        "model_architecture": "ResPotholeNet (From Scratch)",
        "total_parameters": count_parameters(model),
        "loss_function": "Focal Loss (alpha=0.25, gamma=2.0)",
        "dataset": {
            "source": str(data_dir),
            "source_images": {
                "raw_pothole_images": len(raw_pos_paths),
                "raw_normal_images": len(raw_neg_paths),
                "clean_pothole_images": len(clean_pos),
                "clean_normal_images": len(clean_neg),
                "total_clean_images": len(clean_pos) + len(clean_neg),
                "byte_identical_duplicates_removed": byte_pos + byte_neg,
                "perceptual_near_duplicates_removed": near_pos + near_neg,
            },
            "split_method": "Deterministic sorted split by source photo (Zero Leakage)",
            "crop_method": "Symmetric Stage-1 proposal sampling (top 2 proposals per image for both classes)",
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
        "validation_threshold_tuning": {
            "selected_threshold": best_th,
            "selection_criterion": "Maximize Validation Image-Level F1 Score",
            "val_results_at_selected_threshold": best_val_e2e,
            "val_threshold_sweep": all_val_e2e,
        },
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "test_end_to_end_at_val_threshold": test_e2e_best,
        "test_end_to_end_at_default_0_5": test_e2e_default,
        "evaluation_notes": {
            "sample_correlation": (
                "Test crops originate from unseen source photos. Because patches from the same photo "
                "share lighting and asphalt conditions, treat patch metrics as subject to roughly +-5 percentage points variance."
            ),
            "end_to_end_screening_vs_detection": (
                "Image-level detection measures whether ANY candidate box in a full frame passes threshold. "
                "It serves as a coarse frame screening metric. It does NOT represent bounding box mAP@0.5, "
                "which strictly requires hand-annotated bounding box coordinates."
            )
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nEvaluation report saved to: {metrics_path}")

    # Also save end_to_end_metrics.json for standalone inspection
    with open("end_to_end_metrics.json", "w") as f:
        json.dump({
            "selected_threshold": best_th,
            "val_tuning": all_val_e2e,
            "test_at_selected_threshold": test_e2e_best,
            "test_at_threshold_0_5": test_e2e_default,
        }, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResPotholeNet on domain-aligned symmetric crops")
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
