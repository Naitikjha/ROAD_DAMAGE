"""
Standalone End-to-End Pipeline Evaluation for Two-Stage Pothole Detection.

Runs the complete detection pipeline on the held-out source test images:
  Stage 1: Region proposals from full image (extract_stage1_candidates)
  Stage 2: ResPotholeNet patch classification + Non-Maximum Suppression

Evaluates:
  - Image-level Detection Accuracy, Precision, Recall (Sensitivity), Specificity, and F1
  - False Positives Per Image (FPPI) on clean road photos
  - Average detections per pothole photo
  - Image-level Confusion Matrix
  - Threshold Sensitivity Analysis (0.3 to 0.7)

Usage:
  python evaluate_pipeline.py --weights pothole_cnn.pth --data_dir data
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch

from models.pothole_net import ResPotholeNet
from train import deduplicate_files, split_source_images, evaluate_end_to_end


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

    unique_pos, _ = deduplicate_files(pos_paths, check_near_duplicates=False)
    unique_neg, _ = deduplicate_files(neg_paths, check_near_duplicates=False)

    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = split_source_images(
        unique_pos, unique_neg, train_ratio=0.70, val_ratio=0.15, seed=args.seed
    )

    print(f"Held-out Unseen Test Set: {len(test_pos)} pothole photos + {len(test_neg)} normal photos = {len(test_pos) + len(test_neg)} photos")

    model = ResPotholeNet(in_channels=3).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("\nRunning two-stage pipeline on all test images across thresholds...")
    results = evaluate_end_to_end(model, test_pos, test_neg, device, conf_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
    def_rep = results["default_threshold_0_5"]
    cm = def_rep["confusion_matrix"]

    print("\n" + "=" * 65)
    print("     END-TO-END PIPELINE EVALUATION RESULTS (Conf Threshold = 0.50)")
    print("=" * 65)
    print(f"  Test Source Images:    {results['test_source_images']['total_images']} ({results['test_source_images']['pothole_images']} pos, {results['test_source_images']['normal_images']} neg)")
    print(f"  Image-Level Accuracy:  {def_rep['image_accuracy']}%")
    print(f"  Image-Level Precision: {def_rep['image_precision']}%")
    print(f"  Image-Level Recall:    {def_rep['image_recall']}% (Detection Rate on Pothole Images)")
    print(f"  Image Specificity:     {def_rep['image_specificity']}% (Correct Clean Road Flagging)")
    print(f"  Image F1-Score:        {def_rep['image_f1_score']}%")
    print(f"  Clean Road FPPI:       {def_rep['fppi_on_clean_roads']} false positive boxes per clean photo")
    print(f"  Avg Pothole Detections:{def_rep['avg_detections_per_pothole_photo']} boxes per pothole photo")
    print(f"\n  Image-Level Confusion Matrix:")
    print(f"                    Predicted Pothole   Predicted Clear")
    print(f"    Actual Pothole        {cm['TP_images']:>4}              {cm['FN_images']:>4}")
    print(f"    Actual Clear          {cm['FP_images']:>4}              {cm['TN_images']:>4}")
    print("=" * 65)

    print("\nThreshold Sweep Analysis:")
    print("  Thresh | Accuracy | Precision | Recall | Specificity | F1-Score | FPPI")
    print("  " + "-" * 62)
    for k, v in results["threshold_sweep"].items():
        th = v["confidence_threshold"]
        print(f"   {th:.1f}   |  {v['image_accuracy']:>5.1f}%  |   {v['image_precision']:>5.1f}%  | {v['image_recall']:>5.1f}% |    {v['image_specificity']:>5.1f}%   |  {v['image_f1_score']:>5.1f}%  | {v['fppi_on_clean_roads']:>4.2f}")
    print("=" * 65)

    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed end-to-end report saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()
