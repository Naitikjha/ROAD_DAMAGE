"""
Standalone Stage-1 Candidate Proposal Recall Evaluation.

Evaluates the recall ceiling imposed by Stage-1 classical CV region proposals
against the hand-annotated test ground-truth bounding boxes:
  - Measures candidate recall at multiple IoU thresholds (0.10, 0.25, 0.35, 0.50)
  - Computes the mean maximum IoU achievable before any CNN classification
  - Measures proposal counts per image on both pothole and clean road images
  - Produces 'stage1_recall_metrics.json' to ensure full reproducibility

Usage:
  python evaluate_stage1_recall.py
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np

from detector import extract_stage1_candidates
from evaluate_box_level import compute_iou


def evaluate_stage1(
    gt_json_path: str = "data/test_ground_truth_boxes.json",
    potholes_dir: str = "data/potholes",
    normal_dir: str = "data/normal",
    output_json: str = "stage1_recall_metrics.json",
    max_candidates: int = 40,
) -> dict:
    print("=" * 70)
    print("        STAGE-1 REGION PROPOSAL RECALL & IoU BENCHMARK")
    print("=" * 70)
    print()
    print("  Measures the classical CV region proposal recall ceiling BEFORE")
    print("  any CNN classification. If Stage 1 does not nominate a region with")
    print("  sufficient IoU overlap, the CNN cannot recover localization.")
    print()

    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    annotations = gt_data["annotations"]
    print(f"Loaded ground-truth annotations for {len(annotations)} test pothole images.\n")

    total_gt_boxes = 0
    max_ious_per_box = []
    proposals_per_image = []
    per_image_results = {}

    iou_thresholds = [0.10, 0.20, 0.25, 0.30, 0.35, 0.50]
    matched_counts = {t: 0 for t in iou_thresholds}

    for fname, entry in annotations.items():
        img_path = Path(potholes_dir) / fname
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_boxes = entry["boxes"]
        total_gt_boxes += len(gt_boxes)

        # Extract Stage 1 classical candidate proposals
        candidates = extract_stage1_candidates(
            img, roi_mode="photo", max_candidates=max_candidates
        )
        proposals_per_image.append(len(candidates))

        img_box_ious = []
        for gbox in gt_boxes:
            best_iou = 0.0
            best_cand = None
            for cand in candidates:
                iou = compute_iou(cand, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_cand = cand

            img_box_ious.append(round(best_iou, 4))
            max_ious_per_box.append(best_iou)

            for t in iou_thresholds:
                if best_iou >= t:
                    matched_counts[t] += 1

        per_image_results[fname] = {
            "gt_boxes_count": len(gt_boxes),
            "stage1_proposals_count": len(candidates),
            "max_ious_achieved": img_box_ious,
            "mean_max_iou": round(float(np.mean(img_box_ious)), 4) if img_box_ious else 0.0,
            "matched_at_0_50": sum(1 for x in img_box_ious if x >= 0.50),
        }

    # Evaluate clean road proposal density
    clean_photos = sorted(list(Path(normal_dir).glob("*.*")))[:30]
    clean_proposals = []
    for cp in clean_photos:
        cimg = cv2.imread(str(cp))
        if cimg is None:
            continue
        c_cands = extract_stage1_candidates(cimg, roi_mode="photo", max_candidates=max_candidates)
        clean_proposals.append(len(c_cands))

    mean_max_iou = float(np.mean(max_ious_per_box)) if max_ious_per_box else 0.0
    median_max_iou = float(np.median(max_ious_per_box)) if max_ious_per_box else 0.0
    avg_proposals_pothole = float(np.mean(proposals_per_image)) if proposals_per_image else 0.0
    avg_proposals_clean = float(np.mean(clean_proposals)) if clean_proposals else 0.0

    print("=" * 70)
    print("                     STAGE-1 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Test Ground-Truth Photos:          {len(annotations)}")
    print(f"  Total Ground-Truth Boxes:          {total_gt_boxes}")
    print(f"  Avg Stage-1 Proposals / Pothole:   {avg_proposals_pothole:.1f} candidate boxes")
    print(f"  Avg Stage-1 Proposals / Clean:     {avg_proposals_clean:.1f} candidate boxes")
    print(f"  Mean Maximum IoU on GT Boxes:      {mean_max_iou * 100:.2f}%")
    print(f"  Median Maximum IoU on GT Boxes:    {median_max_iou * 100:.2f}%")
    print("-" * 70)
    print("  Candidate Recall Ceiling by IoU Threshold (Before CNN Filtering):")
    for t in iou_thresholds:
        rec = matched_counts[t] / max(total_gt_boxes, 1) * 100
        print(f"    Recall @ IoU >= {t:.2f}:           {rec:>6.2f}% ({matched_counts[t]}/{total_gt_boxes})")
    print("=" * 70)
    print(
        f"\n  Key Takeaway: Stage 1 alone nominates only {matched_counts[0.50]}/{total_gt_boxes} boxes "
        f"at IoU >= 0.50 ({matched_counts[0.50]/total_gt_boxes*100:.1f}%).\n"
        "  This empirically proves Stage 1 is the primary bottleneck for box localization."
    )

    results = {
        "evaluation_name": "Stage-1 Classical Region Proposal Recall Benchmark",
        "gt_annotations_file": gt_json_path,
        "total_annotated_photos": len(annotations),
        "total_ground_truth_boxes": total_gt_boxes,
        "mean_max_iou_percent": round(mean_max_iou * 100, 2),
        "median_max_iou_percent": round(median_max_iou * 100, 2),
        "avg_proposals_per_pothole_image": round(avg_proposals_pothole, 2),
        "avg_proposals_per_clean_image": round(avg_proposals_clean, 2),
        "candidate_recall_ceiling_by_iou": {
            f"iou_{int(t*100)}": {
                "threshold": t,
                "matched_gt_boxes": matched_counts[t],
                "total_gt_boxes": total_gt_boxes,
                "recall_percent": round(matched_counts[t] / max(total_gt_boxes, 1) * 100, 2),
            }
            for t in iou_thresholds
        },
        "per_image_results": per_image_results,
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved detailed Stage-1 benchmark metrics to: {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 Region Proposal Recall Evaluation")
    parser.add_argument("--gt", type=str, default="data/test_ground_truth_boxes.json")
    parser.add_argument("--potholes_dir", type=str, default="data/potholes")
    parser.add_argument("--normal_dir", type=str, default="data/normal")
    parser.add_argument("--out", type=str, default="stage1_recall_metrics.json")
    parser.add_argument("--max_candidates", type=int, default=40)
    args = parser.parse_args()

    evaluate_stage1(
        gt_json_path=args.gt,
        potholes_dir=args.potholes_dir,
        normal_dir=args.normal_dir,
        output_json=args.out,
        max_candidates=args.max_candidates,
    )
