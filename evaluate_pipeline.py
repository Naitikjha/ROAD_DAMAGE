"""
Standalone End-to-End Pipeline Evaluation for Two-Stage Pothole Detection.

Runs the complete detection pipeline on the held-out source test images:
  Stage 1: Region proposals from full image (extract_stage1_candidates)
  Stage 2: ResPotholeNet patch classification + Non-Maximum Suppression

Selection Methodology:
  - Optimal operating threshold is selected on the VALIDATION source images.
  - Test partition is evaluated ONCE at the validation-selected threshold to prevent leakage.

Evaluates:
  - Image-level Detection Accuracy, Precision, Recall, Specificity, and F1
  - False Positives Per Image (FPPI) on clean road photos
  - Average detections per pothole photo
  - Image-level Confusion Matrix

Usage:
  python evaluate_pipeline.py --weights pothole_cnn.pth --data_dir data
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch

from models.pothole_net import ResPotholeNet
from train import (
    deduplicate_files,
    split_source_images,
    select_best_threshold_on_val,
    run_pipeline_on_images,
)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Evaluation on Full Test Images")
    parser.add_argument("--weights", type=str, default="pothole_cnn.pth",
                        help="Path to trained model weights")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to dataset directory")
    parser.add_argument("--metrics_out", type=str, default="end_to_end_metrics.json",
                        help="Output path for evaluation JSON report")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for source image partition (must match train.py)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running End-to-End Evaluation on {device}...")

    data_path = Path(args.data_dir)
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

    raw_pos_paths = sorted(raw_pos_paths)
    raw_neg_paths = sorted(raw_neg_paths)

    clean_pos, byte_pos, near_pos = deduplicate_files(raw_pos_paths, dhash_threshold=2)
    clean_neg, byte_neg, near_neg = deduplicate_files(raw_neg_paths, dhash_threshold=2)

    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = split_source_images(
        clean_pos, clean_neg, train_ratio=0.70, val_ratio=0.15, seed=args.seed
    )

    print(f"Dataset: {len(clean_pos) + len(clean_neg)} clean unique photos ({len(clean_pos)} pothole, {len(clean_neg)} clean)")
    print(f"Held-out Unseen Test Set: {len(test_pos)} pothole photos + {len(test_neg)} normal photos = {len(test_pos) + len(test_neg)} photos")

    model = ResPotholeNet(in_channels=3).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("\nSelecting optimal operating threshold on VALIDATION images...")
    best_th, best_val_rep, all_val_reps = select_best_threshold_on_val(model, val_pos, val_neg, device)
    print(f"  Selected Threshold: {best_th:.1f} (Validation F1 = {best_val_rep['image_f1_score']}%)")

    print(f"\nEvaluating Held-Out Test Set at validation-selected threshold ({best_th:.1f})...")
    test_rep_best = run_pipeline_on_images(model, test_pos, test_neg, device, conf_threshold=best_th)
    test_rep_default = run_pipeline_on_images(model, test_pos, test_neg, device, conf_threshold=0.5)

    cm = test_rep_best["confusion_matrix"]

    print("\n" + "=" * 65)
    print(f"     END-TO-END EVALUATION ON TEST PHOTOS (Threshold = {best_th:.1f})")
    print("=" * 65)
    print(f"  Test Source Images:    {len(test_pos) + len(test_neg)} ({len(test_pos)} pos, {len(test_neg)} neg)")
    print(f"  Image-Level Accuracy:  {test_rep_best['image_accuracy']}%")
    print(f"  Image-Level Precision: {test_rep_best['image_precision']}%")
    print(f"  Image-Level Recall:    {test_rep_best['image_recall']}% (Detection Rate)")
    print(f"  Image Specificity:     {test_rep_best['image_specificity']}% (Clean Road Correctness)")
    print(f"  Image F1-Score:        {test_rep_best['image_f1_score']}%")
    print(f"  Clean Road FPPI:       {test_rep_best['fppi_on_clean_roads']} false positive boxes / clean photo")
    print(f"  Avg Pothole Detections:{test_rep_best['avg_detections_per_pothole_photo']} detections / pothole photo")
    print(f"\n  Image-Level Confusion Matrix:")
    print(f"                    Predicted Pothole   Predicted Clear")
    print(f"    Actual Pothole        {cm['TP_images']:>4}              {cm['FN_images']:>4}")
    print(f"    Actual Clear          {cm['FP_images']:>4}              {cm['TN_images']:>4}")
    print("=" * 65)

    out_data = {
        "selected_threshold": best_th,
        "validation_sweep": all_val_reps,
        "test_results_at_selected_threshold": test_rep_best,
        "test_results_at_threshold_0_5": test_rep_default,
    }

    with open(args.metrics_out, "w") as f:
        json.dump(out_data, f, indent=4)
    print(f"\nDetailed end-to-end report saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()
