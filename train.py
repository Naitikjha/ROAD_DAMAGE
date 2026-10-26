"""
Training and Evaluation Pipeline for ResPotholeNet Custom CNN (From Scratch)
Course Project: Two-Stage Pothole Detection from Road Videos Using Custom CNN

Key Features:
- Stage 2 Patch Verification Network (ResPotholeNet: SE-ResNet + SPPF)
- Supports Real Image Datasets (e.g., RDD2022 / Kaggle / custom folder structure)
  with fallback to structured road-texture dataset generator.
- Train / Validation / Test split (70% / 15% / 15%).
- Focal Loss (imbalanced road cavity classification) + SmoothL1 bounding box refinement.
- Rigorous Evaluation: Precision, Recall, F1-Score, Accuracy, and Confusion Matrix.
- Saves model weights (pothole_cnn.pth) and evaluation summary (model_metrics.json).
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
    Loads real road patches from directory structured as:
    data_dir/
      potholes/  (or positive/) -> positive road cavity images
      normal/    (or negative/) -> clear road pavement images
    """
    def __init__(self, data_dir: str, patch_size: int = 128):
        self.patch_size = patch_size
        self.samples = []  # list of (image_path, label: 1.0 or 0.0)

        data_path = Path(data_dir)
        pothole_dirs = ["potholes", "pothole", "positive", "cracks"]
        normal_dirs = ["normal", "clear", "negative", "pavement"]

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback black patch if corrupt
            img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.patch_size, self.patch_size))

        # Random horizontal/vertical flip augmentation
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        bbox_delta = torch.zeros(4, dtype=torch.float32)
        return tensor, label_tensor, bbox_delta


class SyntheticRoadPotholeDataset(Dataset):
    """
    Deterministic & realistic road patch synthesizer for benchmarking
    when real dataset directory is not yet downloaded locally.
    Generates structured asphalt grain, illumination gradients, and cavity contours.
    """
    def __init__(self, num_samples: int = 1000, patch_size: int = 128, seed: int = 42):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Reproducible pseudo-random state per sample index
        rng = np.random.RandomState(self.seed + idx)

        is_pothole = 1.0 if (idx % 2 == 0) else 0.0

        # Base asphalt background texture (rough distribution)
        base = rng.normal(loc=0.45, scale=0.08, size=(self.patch_size, self.patch_size, 3))

        # Add road lighting gradient (sunlight angle / road perspective)
        grad = np.linspace(rng.uniform(-0.1, 0.0), rng.uniform(0.0, 0.1), self.patch_size)
        base += grad[:, None, None]

        if is_pothole:
            # Generate irregular cavity depression
            center_x = rng.randint(self.patch_size // 4, 3 * self.patch_size // 4)
            center_y = rng.randint(self.patch_size // 4, 3 * self.patch_size // 4)
            rx = rng.randint(18, self.patch_size // 3)
            ry = rng.randint(14, self.patch_size // 3)

            y, x = np.ogrid[:self.patch_size, :self.patch_size]
            dist_sq = ((x - center_x) / rx) ** 2 + ((y - center_y) / ry) ** 2

            # Dark interior depression
            interior_mask = dist_sq <= 1.0
            base[interior_mask] *= rng.uniform(0.3, 0.6)

            # High-frequency fracture rim
            rim_mask = (dist_sq > 0.82) & (dist_sq <= 1.22)
            base[rim_mask] += rng.normal(0, 0.12, size=base[rim_mask].shape)

            dx = float((center_x - self.patch_size / 2) / float(self.patch_size))
            dy = float((center_y - self.patch_size / 2) / float(self.patch_size))
            dw = float((rx * 2 - self.patch_size / 2) / float(self.patch_size))
            dh = float((ry * 2 - self.patch_size / 2) / float(self.patch_size))
            bbox_delta = np.array([dx, dy, dw, dh], dtype=np.float32)
        else:
            # Normal road: asphalt with minor aggregates, no cavity
            bbox_delta = np.zeros(4, dtype=np.float32)

        image = np.clip(base, 0.0, 1.0).astype(np.float32)
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor([is_pothole], dtype=torch.float32)
        delta = torch.from_numpy(bbox_delta).float()
        return tensor, label, delta


class FocalLoss(nn.Module):
    """Focal Loss to focus on hard road distress cavities vs uniform pavement."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal_loss.mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    """Evaluate model on a dataset partition and compute precision, recall, F1, accuracy, and confusion matrix."""
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    cls_criterion = FocalLoss()

    with torch.no_grad():
        for imgs, labels, _ in loader:
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
    data_dir: str = None,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: str = "pothole_cnn.pth",
    metrics_path: str = "model_metrics.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ResPotholeNet Custom CNN from scratch on {device}...")

    # Load dataset
    if data_dir and Path(data_dir).exists():
        full_dataset = RealRoadPotholeDataset(data_dir)
        print(f"Loaded real dataset from '{data_dir}' with {len(full_dataset)} samples.")
    else:
        full_dataset = SyntheticRoadPotholeDataset(num_samples=1000, seed=42)
        print(f"Using structured synthetic benchmark dataset with {len(full_dataset)} samples.")

    # 70% Train, 15% Validation, 15% Test split
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Dataset Split: Train={n_train} | Val={n_val} | Test={n_test}")

    model = ResPotholeNet(in_channels=3).to(device)
    cls_criterion = FocalLoss()
    reg_criterion = nn.SmoothL1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        for batch_imgs, batch_labels, batch_deltas in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            batch_deltas = batch_deltas.to(device)

            optimizer.zero_grad()
            conf, deltas = model(batch_imgs)

            l_cls = cls_criterion(conf, batch_labels)
            pos_mask = (batch_labels > 0.5).squeeze(-1)
            if pos_mask.sum() > 0:
                l_reg = reg_criterion(deltas[pos_mask], batch_deltas[pos_mask])
            else:
                l_reg = torch.tensor(0.0, device=device)

            loss = l_cls + 0.5 * l_reg
            loss.backward()
            optimizer.step()

            total_cls_loss += l_cls.item()
            total_reg_loss += l_reg.item()

        scheduler.step()

        # Validation at epoch end
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch}/{epochs}] — Train Loss: {total_cls_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_metrics['accuracy']}% | Val F1: {val_metrics['f1_score']}%")

        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining Complete! Best model saved to: {save_path}")

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
    print(f"  Confusion Matrix: TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}")
    print("=" * 60)

    # Save metrics to disk
    report = {
        "model_architecture": "ResPotholeNet (From Scratch)",
        "backbone": "3-Stage Residual Bottlenecks + Squeeze-and-Excitation Attention",
        "pooling": "Spatial Pyramid Pooling - Fast (SPPF)",
        "total_parameters": 858245,
        "loss_function": "Focal Loss (alpha=0.25, gamma=2.0) + Smooth L1 Loss",
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "test_metrics": test_metrics
    }
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation report saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResPotholeNet")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to real road dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
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
