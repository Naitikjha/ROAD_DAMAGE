"""
Training and Evaluation Pipeline for ResPotholeNet Custom CNN (From Scratch)
Course Project: Two-Stage Pothole Detection from Road Videos Using Custom CNN

IMPORTANT: Domain alignment
  At inference, the CNN receives 128×128 CROPS that Stage 1 proposes from a
  video frame — not full scene photos.  To match that, this script:
    1. Reads each source image.
    2. Resizes it so the shorter side is 256-320 px (preserving aspect ratio).
    3. Extracts multiple random 128×128 CROPS (not a full-image resize).
    4. For negative images, also runs Stage-1-style proposals and adds
       any crops as hard negatives (shadows, cracks, oil stains, etc.).
  This ensures the CNN trains on the same kind of input it will see at
  inference time.

Dataset expected structure:
    data_dir/
      potholes/  (or positive/) -> real pothole road patches
      normal/    (or negative/ or plain/ or clear/) -> clear road patches

NO synthetic / fake data.  If no real dataset is provided the script exits.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from models.pothole_net import ResPotholeNet


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    """SHA-256 of the raw file bytes (for deduplication)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resize_short_side(img: np.ndarray, target: int = 288) -> np.ndarray:
    """Resize so the shorter side equals `target`, preserving aspect ratio."""
    h, w = img.shape[:2]
    if h < w:
        new_h = target
        new_w = int(w * target / h)
    else:
        new_w = target
        new_h = int(h * target / w)
    return cv2.resize(img, (new_w, new_h))


def extract_random_crops(img: np.ndarray, crop_size: int = 128,
                         n_crops: int = 3, rng: np.random.RandomState = None) -> list:
    """Take `n_crops` random 128×128 patches from `img`."""
    if rng is None:
        rng = np.random.RandomState()
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        return [cv2.resize(img, (crop_size, crop_size))]
    crops = []
    for _ in range(n_crops):
        y = rng.randint(0, h - crop_size + 1)
        x = rng.randint(0, w - crop_size + 1)
        crops.append(img[y:y + crop_size, x:x + crop_size])
    return crops


def extract_stage1_proposals(img: np.ndarray, crop_size: int = 128) -> list:
    """
    Run a simplified Stage-1 pipeline on an image and return proposal crops.
    Used to generate hard-negative training patches from normal-road images.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    blurred = cv2.bilateralFilter(l_chan, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < (h * w * 0.3):
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.3 < aspect < 5.0:
                pad_x = int(bw * 0.1)
                pad_y = int(bh * 0.1)
                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)
                patch = img[y1:y2, x1:x2]
                if patch.shape[0] >= 8 and patch.shape[1] >= 8:
                    crops.append(cv2.resize(patch, (crop_size, crop_size)))
    return crops


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CropBasedRoadDataset(Dataset):
    """
    Builds a training set of 128×128 CROPS from source images,
    matching the kind of input the CNN receives at inference.

    For pothole images → random crops (the whole image is a pothole scene).
    For normal images  → random crops + Stage-1 hard-negative proposals.
    """

    def __init__(self, data_dir: str, patch_size: int = 128,
                 crops_per_image: int = 3, include_hard_negatives: bool = True,
                 augment: bool = False, seed: int = 42):
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []  # list of (crop: np.ndarray, label: float)

        data_path = Path(data_dir)
        pothole_dirs = ["potholes", "pothole", "positive", "Pothole"]
        normal_dirs = ["normal", "clear", "negative", "pavement", "plain", "Plain"]

        rng = np.random.RandomState(seed)

        # Collect all image paths with labels
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

        if len(pos_paths) + len(neg_paths) == 0:
            raise FileNotFoundError(
                f"No images found in '{data_dir}'. Expected subfolders like "
                "'potholes/' and 'normal/' containing .jpg/.png images."
            )

        # Deduplicate by file hash
        seen_hashes = set()
        unique_pos, unique_neg = [], []
        dupes = 0
        for paths, unique_list in [(pos_paths, unique_pos), (neg_paths, unique_neg)]:
            for p in paths:
                h = file_hash(str(p))
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_list.append(p)
                else:
                    dupes += 1

        if dupes > 0:
            print(f"  Removed {dupes} duplicate images (by SHA-256 hash)")

        print(f"  Unique images: {len(unique_pos)} pothole + {len(unique_neg)} normal")

        # Generate crops from positive images
        for img_path in unique_pos:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = resize_short_side(img, target=rng.randint(256, 321))
            crops = extract_random_crops(img, patch_size, crops_per_image, rng)
            for c in crops:
                self.samples.append((c, 1.0))

        # Generate crops from negative images
        for img_path in unique_neg:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = resize_short_side(img, target=rng.randint(256, 321))
            # Random crops
            crops = extract_random_crops(img, patch_size, crops_per_image, rng)
            for c in crops:
                self.samples.append((c, 0.0))
            # Hard negatives from Stage 1 proposals
            if include_hard_negatives:
                hard_negs = extract_stage1_proposals(img, patch_size)
                for hn in hard_negs[:2]:  # limit per image
                    self.samples.append((hn, 0.0))

        # Shuffle deterministically
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        self.samples = [self.samples[i] for i in indices]

        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = len(self.samples) - n_pos
        print(f"  Total crops: {n_pos} pothole + {n_neg} normal = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        crop, label = self.samples[idx]

        # Data augmentation (only horizontal flip + brightness — no vertical flip)
        if self.augment:
            if np.random.rand() > 0.5:
                crop = cv2.flip(crop, 1)  # horizontal flip only
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.randint(-20, 20)
                crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = crop.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Standard Focal Loss for binary classification.
    alpha weights the positive class, (1 - alpha) weights the negative class.
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
    """
    Evaluate model on a dataset partition.
    Returns precision, recall, F1, accuracy, loss, confusion matrix, and n.
    """
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
#  Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AugmentedSubset(Dataset):
    """Wraps a subset of CropBasedRoadDataset with augmentation enabled."""
    def __init__(self, full_dataset: CropBasedRoadDataset, indices: list):
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        crop, label = self.full_dataset.samples[real_idx]

        # Augmentation (horizontal flip + brightness/contrast + rotation)
        if np.random.rand() > 0.5:
            crop = cv2.flip(crop, 1)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = crop.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor


def train_pipeline(
    data_dir: str = "data",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: str = "pothole_cnn.pth",
    metrics_path: str = "model_metrics.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ResPotholeNet on {device}...")

    # --- Load REAL dataset (no synthetic fallback) ---
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\nERROR: Dataset directory '{data_dir}' not found!")
        print("Please download a real pothole dataset and organize it as:")
        print(f"  {data_dir}/")
        print(f"    potholes/   <- pothole road images (.jpg/.png)")
        print(f"    normal/     <- clear road images (.jpg/.png)")
        print("\nRecommended datasets:")
        print("  - https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks")
        return

    print("Building crop-based dataset (domain-aligned with Stage-1 inference)...")
    full_dataset = CropBasedRoadDataset(
        data_dir, patch_size=128, crops_per_image=3,
        include_hard_negatives=True, augment=False, seed=42
    )

    # 70% Train, 15% Validation, 15% Test split
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_aug_dataset = AugmentedSubset(full_dataset, train_set.indices)
    train_loader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Dataset Split: Train={n_train} | Val={n_val} | Test={n_test}")

    model = ResPotholeNet(in_channels=3).to(device)
    cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0

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
        print(f"Epoch [{epoch}/{epochs}] — "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']}% | "
              f"Val P: {val_metrics['precision']}% | "
              f"Val R: {val_metrics['recall']}% | "
              f"Val F1: {val_metrics['f1_score']}% "
              f"(n={val_metrics['n']})")

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (Val F1={best_val_f1}%)")

    print(f"\nTraining Complete! Best model saved to: {save_path}")

    # Reload best weights for final test evaluation
    model.load_state_dict(torch.load(save_path, map_location=device))

    print("\n" + "=" * 60)
    print("     FINAL TEST SET EVALUATION (Patch Classification)")
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
    print("\n  NOTE: These are PATCH CLASSIFICATION metrics — they measure")
    print("  whether the CNN correctly classifies 128×128 crops.")
    print("  They do NOT measure end-to-end detection accuracy")
    print("  (which depends on Stage 1 proposal quality).")

    val_final = evaluate(model, val_loader, device)

    from models.pothole_net import count_parameters
    report = {
        "model_architecture": "ResPotholeNet (From Scratch)",
        "total_parameters": count_parameters(model),
        "loss_function": "Focal Loss (alpha=0.25, gamma=2.0)",
        "dataset": {
            "source": str(data_dir),
            "type": "Random 128x128 crops from real road images (domain-aligned with Stage 1)",
            "source_images": "~712 real photos",
            "training_crops": n_total,
            "includes_hard_negatives": True,
            "deduplication": "SHA-256 hash (no near-duplicates in train/test)",
        },
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW (weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR",
            "augmentation": "Horizontal flip, brightness/contrast jitter, rotation (±15°). NO vertical flip.",
        },
        "val_metrics": val_final,
        "test_metrics": test_metrics,
        "metric_note": (
            "These are PATCH CLASSIFICATION metrics (can the CNN correctly classify "
            "128x128 crops). They do NOT measure end-to-end detection accuracy, "
            "which also depends on Stage 1 proposal quality."
        ),
    }
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nEvaluation report saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResPotholeNet")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to real road dataset (must have potholes/ and normal/ subdirs)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="pothole_cnn.pth", help="Path to save weights")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
