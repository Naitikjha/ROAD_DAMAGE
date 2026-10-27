"""
Training and Evaluation Pipeline for ResPotholeNet Custom CNN (From Scratch)
Course Project: Two-Stage Pothole Detection from Road Videos Using Custom CNN

Domain-Aligned Pipeline with Zero Data Leakage:
  1. Source image deduplication: Exact duplicate photos removed via SHA-256 hash.
  2. Source-level split: 70% Train / 15% Validation / 15% Test performed on SOURCE IMAGES.
     Crops from the same image NEVER appear in both training and testing.
  3. Shared Stage-1 Candidate Extraction:
     Both positive and negative training crops are generated using the exact same
     Stage-1 morphological proposal logic (detector.extract_stage1_candidates + extract_crop):
       - Positive images -> Stage 1 crater proposals (potholes)
       - Negative images -> Stage 1 false-alarm proposals (shadows, cracks, stains) + pavement
     Ensures the CNN is trained on the exact patch distribution it encounters at inference.
  4. Augmentation: Horizontal flip, brightness/contrast jitter, slight rotation (no vertical flip).
  5. Metrics: Evaluated on unseen test crops with explicit sample counts (n).
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
    """SHA-256 of the raw file bytes for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate_files(paths: List[Path]) -> List[Path]:
    """Filter out duplicate files using SHA-256 hash."""
    seen_hashes = set()
    unique = []
    for p in paths:
        h = file_hash(str(p))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(p)
    return unique


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
#  Dataset: Stage-1 Crop Road Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Stage1RoadPatchDataset(Dataset):
    """
    Dataset composed of 128x128 crops generated via Stage-1 proposals.
    Both positive and negative samples share the same Stage-1 candidate framing.
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

        # 1. Positive samples: Extract Stage-1 candidate proposals from pothole images
        for path in pos_files:
            img = cv2.imread(str(path))
            if img is None:
                continue

            # Shared Stage-1 candidate proposals (without horizon ROI crop)
            boxes = extract_stage1_candidates(img, use_roi=False)
            if boxes:
                # Top proposals correspond to largest depressions/crater contours
                for box in boxes[:2]:
                    crop = extract_crop(img, box, crop_size=patch_size, pad_ratio=0.1)
                    if crop is not None:
                        self.samples.append((crop, 1.0))
            else:
                # Fallback: center crop if no contour met threshold
                h, w = img.shape[:2]
                cs = min(h, w, 256)
                cy, cx = h // 2, w // 2
                patch = img[max(0, cy - cs // 2):min(h, cy + cs // 2),
                            max(0, cx - cs // 2):min(w, cx + cs // 2)]
                if patch.size > 0:
                    self.samples.append((cv2.resize(patch, (patch_size, patch_size)), 1.0))

        # 2. Negative samples: Extract Stage-1 false alarms (shadows, cracks) + pavement
        for path in neg_files:
            img = cv2.imread(str(path))
            if img is None:
                continue

            h, w = img.shape[:2]
            # Shared Stage-1 candidate proposals on normal roads
            boxes = extract_stage1_candidates(img, use_roi=False)
            for box in boxes[:2]:
                crop = extract_crop(img, box, crop_size=patch_size, pad_ratio=0.1)
                if crop is not None:
                    self.samples.append((crop, 0.0))

            # Include clean pavement patch
            if h >= patch_size and w >= patch_size:
                y = rng.randint(0, h - patch_size + 1)
                x = rng.randint(0, w - patch_size + 1)
                self.samples.append((img[y:y + patch_size, x:x + patch_size], 0.0))
            else:
                self.samples.append((cv2.resize(img, (patch_size, patch_size)), 0.0))

        # Shuffle deterministic
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
                crop = cv2.flip(crop, 1)  # horizontal flip only
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.randint(-15, 15)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = crop.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Normalize with shared function
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
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5) -> dict:
    """Evaluate model on a DataLoader partition."""
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
        print("Expected structure:")
        print("  data/")
        print("    potholes/  <- pothole road images")
        print("    normal/    <- clear road pavement images")
        return

    # Deduplicate source images
    unique_pos = deduplicate_files(pos_paths)
    unique_neg = deduplicate_files(neg_paths)
    dupes_pos = len(pos_paths) - len(unique_pos)
    dupes_neg = len(neg_paths) - len(unique_neg)
    if dupes_pos + dupes_neg > 0:
        print(f"Removed {dupes_pos + dupes_neg} duplicate images via SHA-256 hash.")
    print(f"Unique source images: {len(unique_pos)} pothole + {len(unique_neg)} normal = {len(unique_pos) + len(unique_neg)}")

    # Split SOURCE IMAGES first (Zero Data Leakage)
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = split_source_images(
        unique_pos, unique_neg, train_ratio=0.70, val_ratio=0.15, seed=seed
    )
    print(f"Source Image Split:")
    print(f"  Train: {len(train_pos)} pos + {len(train_neg)} neg = {len(train_pos) + len(train_neg)}")
    print(f"  Val:   {len(val_pos)} pos + {len(val_neg)} neg = {len(val_pos) + len(val_neg)}")
    print(f"  Test:  {len(test_pos)} pos + {len(test_neg)} neg = {len(test_pos) + len(test_neg)}")

    print("\nExtracting Stage-1 aligned crops for each split independently...")
    train_dataset = Stage1RoadPatchDataset(train_pos, train_neg, patch_size=128, augment=True, seed=seed)
    val_dataset = Stage1RoadPatchDataset(val_pos, val_neg, patch_size=128, augment=False, seed=seed)
    test_dataset = Stage1RoadPatchDataset(test_pos, test_neg, patch_size=128, augment=False, seed=seed)

    print(f"Crop Dataset Counts:")
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

    print("\n" + "=" * 60)
    print("     FINAL TEST SET EVALUATION (Unseen Source Images)")
    print("=" * 60)
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
    print("=" * 60)
    print("\n  NOTE: Evaluated on crops from source photos that were NEVER seen in training.")
    print("  Measures patch classification accuracy on Stage-1 candidate proposals.")

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
                "deduplication": "SHA-256 hash"
            },
            "split_method": "Split by source image file (Zero Leakage between train/val/test)",
            "crop_method": "Shared Stage-1 morphological proposal extraction (detector.extract_stage1_candidates)",
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
        "evaluation_scope": (
            "Patch classification metrics on Stage-1 candidate proposals from unseen source images (n="
            + str(test_metrics['n']) + "). Bounding boxes originate from Stage-1 classical OpenCV contours."
        ),
    }

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nEvaluation report saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResPotholeNet on domain-aligned Stage-1 crops")
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
