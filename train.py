"""
Training and Evaluation Pipeline for ResPotholeNet Custom CNN (From Scratch)
Course Project: Two-Stage Pothole Detection from Road Videos Using Custom CNN

Trains on REAL pothole images (binary classification patches).
Dataset expected structure:
    data_dir/
      potholes/  (or positive/) -> real pothole road patches
      normal/    (or negative/ or plain/ or clear/) -> clear road patches

Key Features:
- Train / Validation / Test split (70% / 15% / 15%) by image.
- Focal Loss (class-imbalanced road cavity classification).
- Data augmentation: random flip, rotation, color jitter.
- Rigorous Evaluation: Precision, Recall, F1-Score, Accuracy, Confusion Matrix.
- Saves best model weights (pothole_cnn.pth) and evaluation report (model_metrics.json).

NO synthetic / fake data. If no real dataset is provided, the script exits with
an error message pointing the user to download a dataset.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from models.pothole_net import ResPotholeNet


class RealRoadPotholeDataset(Dataset):
    """
    Loads real road patches from a directory structured as:
    data_dir/
      potholes/ (or positive/ or pothole/) -> positive road cavity images
      normal/   (or negative/ or plain/ or clear/ or pavement/) -> clear road images
    """
    def __init__(self, data_dir: str, patch_size: int = 128, augment: bool = False):
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []  # list of (image_path, label: 1.0 or 0.0)

        data_path = Path(data_dir)
        pothole_dirs = ["potholes", "pothole", "positive", "Pothole"]
        normal_dirs = ["normal", "clear", "negative", "pavement", "plain", "Plain"]

        # Find positive samples
        for pd in pothole_dirs:
            p_path = data_path / pd
            if p_path.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img in p_path.glob(ext):
                        self.samples.append((str(img), 1.0))

        # Find negative samples
        for nd in normal_dirs:
            n_path = data_path / nd
            if n_path.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img in n_path.glob(ext):
                        self.samples.append((str(img), 0.0))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{data_dir}'. Expected subfolders like "
                f"'potholes/' and 'normal/' containing .jpg/.png images."
            )

        pos = sum(1 for _, l in self.samples if l == 1.0)
        neg = len(self.samples) - pos
        print(f"  Found {pos} pothole images + {neg} normal images = {len(self.samples)} total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback black patch if file is corrupt
            img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.patch_size, self.patch_size))

        # Data augmentation (training only)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)
            # Random brightness/contrast jitter
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.randint(-20, 20)     # brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            # Random rotation (small angles)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor


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
        # Apply alpha to positive class, (1 - alpha) to negative class
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce
        return focal_loss.mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5) -> dict:
    """
    Evaluate model on a dataset partition.
    Returns precision, recall, F1, accuracy, loss, and confusion matrix.
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

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1e-7)
    recall = tp / max(tp + fn, 1e-7)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    avg_loss = total_loss / max(len(loader), 1)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }
    }


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
        print("  - RDD2022: https://github.com/sekilab/RoadDamageDetector")
        return

    full_dataset = RealRoadPotholeDataset(data_dir, augment=False)
    print(f"Loaded real dataset from '{data_dir}' with {len(full_dataset)} samples.")

    # 70% Train, 15% Validation, 15% Test split
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Create augmented wrapper for training data
    train_aug_dataset = AugmentedSubset(full_dataset, train_set.indices)
    train_loader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
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

        # Validation at epoch end
        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = total_cls_loss / max(num_batches, 1)
        print(f"Epoch [{epoch}/{epochs}] — "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']}% | "
              f"Val P: {val_metrics['precision']}% | "
              f"Val R: {val_metrics['recall']}% | "
              f"Val F1: {val_metrics['f1_score']}%")

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (Val F1={best_val_f1}%)")

    print(f"\nTraining Complete! Best model saved to: {save_path}")

    # --- Reload best weights for final test evaluation ---
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Rigorous Evaluation on Unseen Test Set
    print("\n" + "=" * 60)
    print("           FINAL TEST SET EVALUATION METRICS")
    print("=" * 60)
    test_metrics = evaluate(model, test_loader, device)
    cm = test_metrics["confusion_matrix"]

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

    # Also evaluate on validation set for completeness
    val_final = evaluate(model, val_loader, device)

    # Save metrics to disk
    from models.pothole_net import count_parameters
    report = {
        "model_architecture": "ResPotholeNet (From Scratch)",
        "backbone": "3-Stage Residual Bottlenecks + Squeeze-and-Excitation Attention",
        "pooling": "Spatial Pyramid Pooling - Fast (SPPF)",
        "total_parameters": count_parameters(model),
        "loss_function": "Focal Loss (alpha=0.25, gamma=2.0)",
        "dataset": {
            "source": str(data_dir),
            "type": "Real road images (no synthetic data)",
            "total_images": n_total,
        },
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW (weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR",
            "augmentation": "Random flip (H+V), brightness/contrast jitter, rotation (±15°)"
        },
        "val_metrics": val_final,
        "test_metrics": test_metrics
    }
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation report saved to: {metrics_path}")


class AugmentedSubset(Dataset):
    """Wraps a subset of RealRoadPotholeDataset with augmentation enabled."""
    def __init__(self, full_dataset: RealRoadPotholeDataset, indices: list):
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.full_dataset.samples[real_idx]

        img = cv2.imread(img_path)
        ps = self.full_dataset.patch_size
        if img is None:
            img = np.zeros((ps, ps, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (ps, ps))

        # Augmentation
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 0)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return tensor, label_tensor


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
        save_path=args.save_path
    )
