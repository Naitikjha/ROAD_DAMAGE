"""
Box-Level Evaluation for Road Pothole Detection.

⚠️  SCOPE LIMITATION — READ BEFORE INTERPRETING RESULTS:
The main pothole dataset provides only IMAGE-LEVEL labels (pothole / normal).
There are NO ground-truth bounding box coordinates for the full test set.
This evaluation only covers the small MANUALLY ANNOTATED subset defined in
`data/test_ground_truth_boxes.json`. Results (IoU, box precision, box recall,
mAP@0.50) are valid only for that annotated subset and CANNOT be extrapolated
to the full 108-photo test set. Do NOT report these as dataset-wide mAP numbers.

Computes (for the annotated subset only):
- Intersection over Union (IoU)
- Bounding Box Precision, Recall, and F1-Score at IoU >= 0.50
- Mean IoU across matched true positive detections
- Average Precision (AP_50)
- Box-level vs. Scene-level comparative performance

Answers the question (for annotated images only):
"Does the predicted box actually sit on the pothole?"
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch

from detector import PotholeDetector


def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / max(union, 1e-6)


def evaluate_box_level(
    gt_json_path: str = "data/test_ground_truth_boxes.json",
    potholes_dir: str = "data/potholes",
    normal_dir: str = "data/normal",
    weights_path: str = "pothole_cnn.pth",
    conf_threshold: float = 0.50,
    iou_threshold: float = 0.50,
    save_results: str = "box_level_metrics.json",
):
    print("=" * 70)
    print("       BOX-LEVEL OBJECT DETECTION EVALUATION (IoU >= 0.50)")
    print("=" * 70)
    print()
    print("  *** SCOPE LIMITATION: ***")
    print("  The main dataset has only IMAGE-LEVEL labels. This evaluation")
    print("  operates on the MANUALLY ANNOTATED subset in test_ground_truth_boxes.json.")
    print("  Results are valid ONLY for that subset and must NOT be reported")
    print("  as dataset-wide mAP or as a substitute for full-dataset box-level metrics.")
    print()

    # 1. Load ground truth annotations
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    annotations = gt_data["annotations"]
    print(f"Loaded ground truth annotations for {len(annotations)} test pothole images.")

    # 2. Initialize detector
    detector = PotholeDetector(weights_path=weights_path)

    # Evaluation counters
    total_gt_boxes = 0
    tp_boxes = 0
    fp_boxes = 0
    fn_boxes = 0
    matched_ious = []

    # Detection confidence lists for AP calculation
    all_detections = []  # list of (score, is_tp)

    # Per-image detailed log
    per_image_results = {}

    for fname, entry in annotations.items():
        img_path = Path(potholes_dir) / fname
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_boxes = entry["boxes"]
        total_gt_boxes += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)

        # Run two-stage detector
        preds = detector.predict(img, conf_threshold=conf_threshold, roi_mode="photo")

        # Sort predictions by confidence descending
        preds.sort(key=lambda d: d["score"], reverse=True)

        img_tp = 0
        img_fp = 0
        img_ious = []

        for pred in preds:
            pbox = pred["box"]
            score = pred["score"]

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gbox in enumerate(gt_boxes):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                tp_boxes += 1
                img_tp += 1
                gt_matched[best_gt_idx] = True
                matched_ious.append(best_iou)
                img_ious.append(best_iou)
                all_detections.append((score, 1))
            else:
                fp_boxes += 1
                img_fp += 1
                all_detections.append((score, 0))

        img_fn = sum(1 for matched in gt_matched if not matched)
        fn_boxes += img_fn

        per_image_results[fname] = {
            "gt_count": len(gt_boxes),
            "pred_count": len(preds),
            "tp": img_tp,
            "fp": img_fp,
            "fn": img_fn,
            "mean_iou": round(float(np.mean(img_ious)), 3) if img_ious else 0.0,
        }

    # 3. Clean road false alarms (test normal images)
    # Check false positive bounding boxes on clean roads
    clean_photos = sorted(list(Path(normal_dir).glob("*.*")))[:30]
    clean_fp_boxes = 0
    clean_photos_with_fp = 0

    for cp in clean_photos:
        cimg = cv2.imread(str(cp))
        if cimg is None:
            continue
        cpreds = detector.predict(cimg, conf_threshold=conf_threshold, roi_mode="photo")
        if cpreds:
            clean_fp_boxes += len(cpreds)
            clean_photos_with_fp += 1
            for cp_det in cpreds:
                all_detections.append((cp_det["score"], 0))

    # Compute overall box-level metrics
    precision = tp_boxes / max(tp_boxes + fp_boxes, 1)
    recall = tp_boxes / max(tp_boxes + fn_boxes, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    # Compute Average Precision (AP_50) via precision-recall curve
    all_detections.sort(key=lambda x: x[0], reverse=True)
    cum_tp = 0
    cum_fp = 0
    recalls = []
    precisions = []

    for _, is_tp in all_detections:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        rec = cum_tp / max(total_gt_boxes, 1)
        prec = cum_tp / max(cum_tp + cum_fp, 1)
        recalls.append(rec)
        precisions.append(prec)

    # 11-point interpolation or AUC for AP
    ap_50 = 0.0
    if recalls:
        # Standard VOC 11-point interpolated AP
        for t in np.arange(0.0, 1.1, 0.1):
            prec_at_rec = [p for r, p in zip(recalls, precisions) if r >= t]
            if prec_at_rec:
                ap_50 += max(prec_at_rec) / 11.0

    print("\n" + "=" * 70)
    print("                       BOX-LEVEL RESULTS")
    print("=" * 70)
    print(f"  Annotated Pothole Photos:    {len(annotations)}")
    print(f"  Ground-Truth Pothole Boxes:  {total_gt_boxes}")
    print(f"  True Positive Boxes (IoU>=.5): {tp_boxes}")
    print(f"  False Positive Boxes (Pothole):{fp_boxes}")
    print(f"  False Negative Boxes (Missed): {fn_boxes}")
    print(f"  Clean Road False Boxes:      {clean_fp_boxes} (in {clean_photos_with_fp}/{len(clean_photos)} clean photos)")
    print("-" * 70)
    print(f"  Box-Level Precision:         {precision * 100:.2f}%")
    print(f"  Box-Level Recall:            {recall * 100:.2f}%")
    print(f"  Box-Level F1-Score:          {f1 * 100:.2f}%")
    print(f"  Mean IoU on Matched Potholes:{mean_iou * 100:.2f}%")
    print(f"  mAP @ IoU 0.50 (AP_50):      {ap_50 * 100:.2f}%")
    print("=" * 70)

    results = {
        "evaluation_type": "Box-Level Ground Truth Evaluation (IoU >= 0.50)",
        "annotated_images_count": len(annotations),
        "total_ground_truth_boxes": total_gt_boxes,
        "true_positives": tp_boxes,
        "false_positives_pothole_photos": fp_boxes,
        "false_negatives_missed": fn_boxes,
        "clean_road_false_alarm_boxes": clean_fp_boxes,
        "box_precision": round(precision * 100, 2),
        "box_recall": round(recall * 100, 2),
        "box_f1_score": round(f1 * 100, 2),
        "mean_iou_on_true_positives": round(mean_iou * 100, 2),
        "average_precision_ap50": round(ap_50 * 100, 2),
        "per_image_results": per_image_results,
    }

    with open(save_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed box-level metrics to {save_results}")
    return results


if __name__ == "__main__":
    evaluate_box_level()
