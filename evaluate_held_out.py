"""
evaluate_held_out.py

Step 5 Evaluation:
Compares the OLD pipeline (legacy classical CV proposals) vs NEW pipeline (multi-scale
sliding windows) on the exact same held-out test split:
  1. Box-level Precision & Recall at IoU >= 0.50 and IoU >= 0.30.
  2. False positive counts / FPPI on clean negative road frames.
  3. Bootstrap 95% Confidence Interval (CI) resampled by segment (preserving correlation).
  4. Auto-saves missed potholes and false alarms to failure_gallery/.
  5. For video, reads manual_counts.json (if present) and reports detected vs visible.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch

from detector import PotholeDetector, PotholeTracker
from evaluate_box_level import compute_iou


def match_boxes(
    pred_boxes: List[Tuple[int, int, int, int]],
    pred_scores: List[float],
    gt_boxes: List[List[int]],
    iou_thresh: float = 0.50,
) -> Tuple[int, int, int, List[float], List[int], List[int]]:
    """Match predictions to ground truth boxes using greedy IoU matching."""
    matched_gt = set()
    matched_preds = set()
    matched_ious = []

    # Sort predictions descending by score
    sort_order = np.argsort(pred_scores)[::-1] if pred_scores else []

    for p_idx in sort_order:
        p_box = pred_boxes[p_idx]
        best_iou = 0.0
        best_g_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_g_idx = g_idx

        if best_iou >= iou_thresh and best_g_idx >= 0:
            matched_gt.add(best_g_idx)
            matched_preds.add(p_idx)
            matched_ious.append(best_iou)

    tp = len(matched_preds)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    unmatched_gt_indices = [g for g in range(len(gt_boxes)) if g not in matched_gt]
    unmatched_pred_indices = [p for p in range(len(pred_boxes)) if p not in matched_preds]

    return tp, fp, fn, matched_ious, unmatched_gt_indices, unmatched_pred_indices


def save_failure_case(
    image: np.ndarray,
    image_name: str,
    pipeline_name: str,
    gt_boxes: List[List[int]],
    pred_boxes: List[Tuple[int, int, int, int]],
    unmatched_gt: List[int],
    unmatched_pred: List[int],
    out_dir: Path,
):
    """Save failure visualization: Green=GT, Yellow=Missed GT (FN), Red=False Positive (FP)."""
    vis = image.copy()
    # Draw matched GT in faint green
    for idx, b in enumerate(gt_boxes):
        if idx not in unmatched_gt:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)
            cv2.putText(vis, "GT-Hit", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

    # Draw missed GT in bright yellow
    for idx in unmatched_gt:
        b = gt_boxes[idx]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cv2.putText(vis, "FN-Missed", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Draw false alarms in red
    for idx in unmatched_pred:
        b = pred_boxes[idx]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.putText(vis, "FP-Alarm", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    out_file = out_dir / f"{pipeline_name}_{image_name}"
    cv2.imwrite(str(out_file), vis)


def run_pipeline_eval(
    detector: PotholeDetector,
    pipeline_name: str,
    annotated_samples: List[Dict[str, Any]],
    negative_samples: List[Path],
    failure_dir: Path,
) -> Dict[str, Any]:
    """Run evaluation on positive annotated and clean negative images."""
    records = []
    total_ious_50 = []
    total_ious_30 = []

    tp_50_tot = fp_50_tot = fn_50_tot = 0
    tp_30_tot = fp_30_tot = fn_30_tot = 0

    for sample in annotated_samples:
        img_path = sample["path"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        gt_boxes = sample["gt_boxes"]
        seg_id = sample.get("segment_id", "photo_seg")

        # Run detection
        preds = detector.predict(frame, roi_mode="photo" if frame.shape[0] > 200 else "dashcam")
        pred_boxes = [p["box"] for p in preds]
        pred_scores = [p["score"] for p in preds]

        # Match at IoU 0.50
        tp50, fp50, fn50, ious50, un_gt50, un_p50 = match_boxes(pred_boxes, pred_scores, gt_boxes, 0.50)
        # Match at IoU 0.30
        tp30, fp30, fn30, ious30, _, _ = match_boxes(pred_boxes, pred_scores, gt_boxes, 0.30)

        tp_50_tot += tp50
        fp_50_tot += fp50
        fn_50_tot += fn50
        total_ious_50.extend(ious50)

        tp_30_tot += tp30
        fp_30_tot += fp30
        fn_30_tot += fn30
        total_ious_30.extend(ious30)

        records.append({
            "image": img_path.name,
            "segment_id": seg_id,
            "is_negative": False,
            "gt_count": len(gt_boxes),
            "pred_count": len(preds),
            "tp_50": tp50, "fp_50": fp50, "fn_50": fn50,
            "tp_30": tp30, "fp_30": fp30, "fn_30": fn30,
        })

        # Save failure case if missed or false alarm
        if un_gt50 or un_p50:
            save_failure_case(frame, img_path.name, pipeline_name, gt_boxes, pred_boxes, un_gt50, un_p50, failure_dir)

    # Evaluate clean negative frames
    neg_fp_count = 0
    neg_records = []
    for neg_p in negative_samples:
        frame = cv2.imread(str(neg_p))
        if frame is None:
            continue
        preds = detector.predict(frame, roi_mode="auto")
        n_fps = len(preds)
        neg_fp_count += n_fps
        records.append({
            "image": neg_p.name,
            "segment_id": "neg_seg",
            "is_negative": True,
            "gt_count": 0,
            "pred_count": n_fps,
            "tp_50": 0, "fp_50": n_fps, "fn_50": 0,
            "tp_30": 0, "fp_30": n_fps, "fn_30": 0,
        })
        if n_fps > 0:
            save_failure_case(frame, neg_p.name, pipeline_name, [], [p["box"] for p in preds], [], list(range(n_fps)), failure_dir)

    p_50 = tp_50_tot / max(1, tp_50_tot + fp_50_tot)
    r_50 = tp_50_tot / max(1, tp_50_tot + fn_50_tot)
    f1_50 = 2 * p_50 * r_50 / max(1e-6, p_50 + r_50)

    p_30 = tp_30_tot / max(1, tp_30_tot + fp_30_tot)
    r_30 = tp_30_tot / max(1, tp_30_tot + fn_30_tot)
    f1_30 = 2 * p_30 * r_30 / max(1e-6, p_30 + r_30)

    fppi = neg_fp_count / max(1, len(negative_samples))

    return {
        "pipeline": pipeline_name,
        "iou_0.50": {
            "precision": round(p_50 * 100, 2),
            "recall": round(r_50 * 100, 2),
            "f1": round(f1_50 * 100, 2),
            "tp": tp_50_tot,
            "fp": fp_50_tot,
            "fn": fn_50_tot,
            "mean_tp_iou": round(float(np.mean(total_ious_50)), 3) if total_ious_50 else 0.0,
        },
        "iou_0.30": {
            "precision": round(p_30 * 100, 2),
            "recall": round(r_30 * 100, 2),
            "f1": round(f1_30 * 100, 2),
            "tp": tp_30_tot,
            "fp": fp_30_tot,
            "fn": fn_30_tot,
            "mean_tp_iou": round(float(np.mean(total_ious_30)), 3) if total_ious_30 else 0.0,
        },
        "negative_frames": {
            "evaluated": len(negative_samples),
            "false_positives": neg_fp_count,
            "fppi": round(fppi, 3),
        },
        "per_sample_records": records,
    }


def bootstrap_ci_by_segment(
    records: List[Dict[str, Any]],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 95% Bootstrap Confidence Intervals resampled by segment_id."""
    rng = np.random.RandomState(seed)

    # Group records by segment_id
    segments = {}
    for r in records:
        sid = r["segment_id"]
        segments.setdefault(sid, []).append(r)

    seg_keys = list(segments.keys())
    n_segs = len(seg_keys)

    p50_list, r50_list = [], []
    p30_list, r30_list = [], []

    for _ in range(n_bootstraps):
        sampled_keys = rng.choice(seg_keys, size=n_segs, replace=True)
        sampled_recs = [rec for k in sampled_keys for rec in segments[k]]

        tp50 = sum(r["tp_50"] for r in sampled_recs)
        fp50 = sum(r["fp_50"] for r in sampled_recs)
        fn50 = sum(r["fn_50"] for r in sampled_recs)

        tp30 = sum(r["tp_30"] for r in sampled_recs)
        fp30 = sum(r["fp_30"] for r in sampled_recs)
        fn30 = sum(r["fn_30"] for r in sampled_recs)

        p50 = tp50 / max(1, tp50 + fp50)
        r50 = tp50 / max(1, tp50 + fn50)
        p30 = tp30 / max(1, tp30 + fp30)
        r30 = tp30 / max(1, tp30 + fn30)

        p50_list.append(p50 * 100)
        r50_list.append(r50 * 100)
        p30_list.append(p30 * 100)
        r30_list.append(r30 * 100)

    return {
        "precision_iou50_ci95": [round(float(np.percentile(p50_list, 2.5)), 2), round(float(np.percentile(p50_list, 97.5)), 2)],
        "recall_iou50_ci95": [round(float(np.percentile(r50_list, 2.5)), 2), round(float(np.percentile(r50_list, 97.5)), 2)],
        "precision_iou30_ci95": [round(float(np.percentile(p30_list, 2.5)), 2), round(float(np.percentile(p30_list, 97.5)), 2)],
        "recall_iou30_ci95": [round(float(np.percentile(r30_list, 2.5)), 2), round(float(np.percentile(r30_list, 97.5)), 2)],
    }


def evaluate_video_manual_counts(
    detector: PotholeDetector,
    video_path: str = "sample_pothole_video.mp4",
    manual_counts_path: str = "manual_counts.json",
    stride: int = 5,
) -> Dict[str, Any]:
    """Evaluate video tracking against manual counts if manual_counts.json exists."""
    v_path = Path(video_path)
    if not v_path.exists():
        return {"status": "skipped", "reason": f"Video not found: {video_path}"}

    cap = cv2.VideoCapture(str(v_path))
    tracker = PotholeTracker(max_age=10, min_hits=2, iou_thresh=0.20)

    frame_idx = 0
    processed_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            detector.predict_and_track(frame, tracker, frame_idx=frame_idx)
            processed_count += 1
        frame_idx += 1
    cap.release()

    detected_unique = tracker.get_unique_count()

    manual_data = {}
    mc_path = Path(manual_counts_path)
    if not mc_path.exists():
        mc_path = Path("data") / manual_counts_path

    if mc_path.exists():
        try:
            with open(mc_path, "r", encoding="utf-8") as f:
                manual_data = json.load(f)
        except Exception:
            pass

    visible_count = manual_data.get(v_path.name, {}).get("total_visible_potholes", manual_data.get("total_visible_potholes", None))

    res = {
        "video": v_path.name,
        "processed_frames": processed_count,
        "detected_unique_potholes": detected_unique,
        "visible_unique_potholes": visible_count,
        "detection_rate": round(detected_unique / visible_count, 3) if visible_count else None,
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="Step 5: Held-out Segment Evaluation (Old vs New Pipeline)")
    parser.add_argument("--gt_json", type=str, default="data/test_ground_truth_boxes.json")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_json", type=str, default="test_segment_evaluation.json")
    parser.add_argument("--failure_dir", type=str, default="failure_gallery")
    args = parser.parse_args()

    fail_dir = Path(args.failure_dir)
    fail_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load GT annotations
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    annotated_samples = []
    for fname, meta in gt_data["annotations"].items():
        # Look in data/potholes/ then data/extracted_frames/
        p = Path(args.data_dir) / "potholes" / fname
        if not p.exists():
            p = Path(args.data_dir) / "extracted_frames" / fname
        if p.exists():
            annotated_samples.append({
                "path": p,
                "gt_boxes": meta["boxes"],
                "segment_id": f"seg_{hash(fname) % 4}",
            })

    # 2. Load clean road negative samples
    neg_dir = Path(args.data_dir) / "normal"
    negative_samples = sorted(list(neg_dir.glob("*.jpg")))[:20]

    print("=" * 70)
    print(" STEP 5: HELD-OUT TEST EVALUATION — OLD (LEGACY) vs NEW (SLIDING)")
    print("=" * 70)
    print(f"Annotated test photos : {len(annotated_samples)}")
    print(f"Clean road negatives  : {len(negative_samples)}")

    # 3. Evaluate OLD Pipeline (legacy classical proposals)
    print("\nRunning Old Pipeline (legacy classical proposals)...")
    old_detector = PotholeDetector(proposals="legacy")
    old_res = run_pipeline_eval(old_detector, "legacy", annotated_samples, negative_samples, fail_dir)
    old_ci = bootstrap_ci_by_segment(old_res["per_sample_records"])
    old_res["bootstrap_ci95"] = old_ci

    # 4. Evaluate NEW Pipeline (multi-scale sliding windows)
    print("\nRunning New Pipeline (multi-scale sliding windows)...")
    new_detector = PotholeDetector(proposals="sliding")
    new_res = run_pipeline_eval(new_detector, "sliding", annotated_samples, negative_samples, fail_dir)
    new_ci = bootstrap_ci_by_segment(new_res["per_sample_records"])
    new_res["bootstrap_ci95"] = new_ci

    # 5. Video Evaluation
    print("\nEvaluating video against manual counts...")
    video_res = evaluate_video_manual_counts(new_detector)

    # 6. Build Final Report
    report = {
        "step": 5,
        "title": "Held-Out Test Evaluation: Legacy Classical vs. Multi-Scale Sliding Windows",
        "annotated_test_count": len(annotated_samples),
        "clean_negative_count": len(negative_samples),
        "legacy_pipeline": old_res,
        "sliding_pipeline": new_res,
        "video_evaluation": video_res,
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY ON HELD-OUT TEST SPLIT")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Legacy (Old)':<20} | {'Sliding Windows (New)':<20}")
    print("-" * 70)
    print(f"{'Recall @ IoU >= 0.50':<25} | {old_res['iou_0.50']['recall']:>6.2f}%              | {new_res['iou_0.50']['recall']:>6.2f}%")
    print(f"{'Precision @ IoU >= 0.50':<25} | {old_res['iou_0.50']['precision']:>6.2f}%              | {new_res['iou_0.50']['precision']:>6.2f}%")
    print(f"{'Recall @ IoU >= 0.30':<25} | {old_res['iou_0.30']['recall']:>6.2f}%              | {new_res['iou_0.30']['recall']:>6.2f}%")
    print(f"{'Precision @ IoU >= 0.30':<25} | {old_res['iou_0.30']['precision']:>6.2f}%              | {new_res['iou_0.30']['precision']:>6.2f}%")
    print(f"{'Clean Road FPPI':<25} | {old_res['negative_frames']['fppi']:>6.3f}                | {new_res['negative_frames']['fppi']:>6.3f}")
    print(f"{'Recall 95% CI (IoU 0.5)':<25} | {str(old_ci['recall_iou50_ci95']):<20} | {str(new_ci['recall_iou50_ci95']):<20}")
    print("=" * 70)
    print(f"Results written to: {args.save_json}")
    print(f"Failure gallery saved to: {args.failure_dir}/")


if __name__ == "__main__":
    main()
