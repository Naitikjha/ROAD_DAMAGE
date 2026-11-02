"""
tools/score_windows.py

Step 2 — Multi-Scale Sliding Window Scoring Benchmark.

Plain-Words Explanation for Viva:
1.  Why a separate scoring script?
    Before retraining the CNN on window-labeled data (Step 3), we need to
    confirm that the window generator (Step 2) produces sensible boxes and that
    the existing CNN can be run over them in manageable time.  This script
    measures both: it runs the full proposal + batch-inference pipeline on real
    frames from our manifest and prints per-frame timing, window counts, and
    which windows the current (not yet retrained) CNN finds most suspicious.
2.  Batched CNN inference:
    Each sliding window is cropped to 128x128 and normalised.  All patches for
    one frame are stacked into a single tensor and passed to ResPotholeNet in
    mini-batches of batch_size (default 64).  GPU transfers happen once per
    batch, not once per window, keeping overhead low.
3.  What the scores mean right now:
    The CNN has NOT yet been retrained on window-labeled data.  Its current
    scores are based on the proxy crop training from the original pipeline.
    Scores above 0.50 are displayed as high-confidence; these numbers will
    shift substantially after Step 3 retraining.  The purpose here is purely
    to measure throughput and verify the pipeline end-to-end.

Usage (run from project root):
    python tools/score_windows.py
    python tools/score_windows.py --n_frames 10 --conf_thresh 0.45 --vis_dir vis_windows
    python tools/score_windows.py --image data/extracted_frames/sample_pothole_video_f000090.jpg
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch

# ── Project-local imports (run from project root) ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import generate_sliding_windows, extract_crop, normalize_patch
from models.pothole_net import ResPotholeNet


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device) -> ResPotholeNet:
    """
    Load ResPotholeNet weights with strict=True as required by project rules.

    Plain-words viva explanation:
    strict=True means the checkpoint must contain exactly the same set of
    parameter names as the model definition.  Any mismatch raises an error
    immediately rather than silently producing wrong results.
    """
    model = ResPotholeNet(in_channels=3).to(device).float()
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Train the model first:  python train.py --data_dir data"
        )
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded weights: {path}  (strict=True, device={device})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Per-Frame Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_frame(
    frame: np.ndarray,
    model: ResPotholeNet,
    device: torch.device,
    roi_mode: str = "auto",
    batch_size: int = 64,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """
    Generate sliding windows for one frame, score all patches through the CNN.

    Plain-words viva explanation:
    - generate_sliding_windows produces up to 300 candidate boxes.
    - extract_crop cuts each box out of the frame (with a small context margin)
      and resizes it to 128x128 px.
    - normalize_patch converts BGR to RGB and applies (pixel - mean) / std
      normalisation consistent with training preprocessing.
    - Patches are batched (batch_size per forward pass) to amortise GPU
      transfer overhead.
    - The model returns (conf, None); conf is a (B,1) sigmoid tensor.

    Returns
    -------
    (results, elapsed_ms, n_windows)
    """
    t_start = time.perf_counter()

    windows = generate_sliding_windows(frame, roi_mode=roi_mode, max_windows=300)
    n_windows = len(windows)

    if n_windows == 0:
        return [], 0.0, 0

    patch_tensors: List[torch.Tensor] = []
    valid_boxes: List[tuple] = []
    for box in windows:
        patch = extract_crop(frame, box, crop_size=128, pad_ratio=0.08)
        if patch is None:
            continue
        norm = normalize_patch(patch)
        tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
        patch_tensors.append(tensor)
        valid_boxes.append(box)

    if not patch_tensors:
        return [], 0.0, n_windows

    all_scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(patch_tensors), batch_size):
            batch = torch.stack(patch_tensors[start:start + batch_size]).to(device)
            conf, _ = model(batch)
            scores = conf.squeeze(-1).cpu().numpy()
            if scores.ndim == 0:
                scores = np.array([float(scores)])
            all_scores.extend(scores.tolist())

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    results = [
        {"box": list(box), "score": round(float(np.clip(s, 0.0, 1.0)), 4)}
        for box, s in zip(valid_boxes, all_scores)
    ]
    return results, elapsed_ms, n_windows


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_windows(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
    conf_thresh: float = 0.50,
    max_bg_windows: int = 50,
) -> np.ndarray:
    """
    Draw windows on the frame image for visual verification.

    Plain-words viva explanation:
    Drawing every window would clutter the image.  We draw a representative
    sample of low-confidence windows in light blue (to confirm the grid covers
    the road surface) and all high-confidence windows in orange/red with their
    score printed alongside.
    """
    vis = frame.copy()

    low_conf = [r for r in results if r["score"] < conf_thresh]
    high_conf = sorted(
        [r for r in results if r["score"] >= conf_thresh],
        key=lambda r: r["score"],
        reverse=True,
    )

    # Representative sample of background windows
    if low_conf:
        step = max(1, len(low_conf) // max_bg_windows)
        for r in low_conf[::step]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 150, 60), 1)

    # High-confidence windows: colour interpolated by score
    for r in high_conf:
        x1, y1, x2, y2 = r["box"]
        s = r["score"]
        g = int(255 * max(0.0, 1.0 - 2.0 * (s - 0.5)))
        red = int(255 * min(1.0, 2.0 * (s - 0.5)))
        colour = (0, g, red)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        label = f"{s:.2f}"
        tx, ty = x1, max(y1 - 4, 14)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)

    cv2.putText(
        vis,
        f"Windows: {len(results)}  High-conf(>={conf_thresh:.2f}): {len(high_conf)}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score sliding windows with the existing CNN and report throughput."
    )
    parser.add_argument("--weights", type=str, default="pothole_cnn.pth")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--image", type=str, default=None,
                        help="Score a single image instead of manifest frames")
    parser.add_argument("--n_frames", type=int, default=5,
                        help="Number of frames to score from manifest (default: 5)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--conf_thresh", type=float, default=0.50)
    parser.add_argument("--roi_mode", type=str, default="auto",
                        choices=["auto", "dashcam", "photo"])
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Directory to save per-frame window visualizations")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Save benchmark results to JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    # ── Select frames ─────────────────────────────────────────────────────────
    if args.image:
        frame_paths = [Path(args.image)]
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            sys.exit(1)
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n = min(args.n_frames, len(rows))
        indices = np.linspace(0, len(rows) - 1, n, dtype=int)
        frame_paths = [Path(rows[i]["image_path"]) for i in indices]

    vis_dir: Optional[Path] = None
    if args.vis_dir:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Score each frame ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("   STEP 2 — SLIDING WINDOW SCORING BENCHMARK")
    print("=" * 72)
    print(f"Device:          {device}")
    print(f"Weights:         {args.weights}")
    print(f"Frames to score: {len(frame_paths)}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Conf threshold:  {args.conf_thresh}")
    print(f"ROI mode:        {args.roi_mode}")
    print("-" * 72)
    print(f"{'Frame':<42} {'Windows':>8} {'ms/frame':>9} {'High-conf':>10}")
    print("-" * 72)

    per_frame_data: List[Dict[str, Any]] = []
    all_times: List[float] = []
    all_window_counts: List[int] = []

    for fp in frame_paths:
        if not fp.exists():
            print(f"  MISSING: {fp}")
            continue
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"  UNREADABLE: {fp}")
            continue

        results, elapsed_ms, n_windows = score_frame(
            frame, model, device,
            roi_mode=args.roi_mode,
            batch_size=args.batch_size,
        )
        high_conf = [r for r in results if r["score"] >= args.conf_thresh]
        all_times.append(elapsed_ms)
        all_window_counts.append(n_windows)

        fname = fp.name
        display = fname if len(fname) <= 40 else "..." + fname[-37:]
        print(f"  {display:<40} {n_windows:>8}  {elapsed_ms:>8.1f}  {len(high_conf):>10}")

        record: Dict[str, Any] = {
            "image_path": str(fp),
            "n_windows_generated": n_windows,
            "n_windows_scored": len(results),
            "n_high_conf": len(high_conf),
            "elapsed_ms": round(elapsed_ms, 2),
            "top5_windows": sorted(results, key=lambda r: r["score"], reverse=True)[:5],
        }
        per_frame_data.append(record)

        if vis_dir is not None:
            vis = draw_windows(frame, results, conf_thresh=args.conf_thresh)
            out_path = vis_dir / f"windows_{fp.stem}.jpg"
            cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if not all_times:
        print("No frames processed.")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    budget_ok = max(all_window_counts) <= 300
    print("=" * 72)
    print("SUMMARY")
    print(f"  Frames scored          : {len(all_times)}")
    print(f"  Mean windows/frame     : {np.mean(all_window_counts):.1f}  "
          f"(max {max(all_window_counts)}, budget<=300: {'PASS' if budget_ok else 'FAIL'})")
    print(f"  Mean ms/frame          : {np.mean(all_times):.1f}")
    print(f"  Effective throughput   : {1000.0 / np.mean(all_times):.1f} fps")

    if vis_dir:
        print(f"  Visualisations        : {vis_dir}/")

    if args.out_json:
        report = {
            "step": 2,
            "description": "Sliding window generation + CNN scoring benchmark",
            "device": str(device),
            "weights": args.weights,
            "n_frames": len(all_times),
            "batch_size": args.batch_size,
            "conf_thresh": args.conf_thresh,
            "stride_ratio": 0.50,
            "scales_fracs": [0.20, 0.35, 0.52],
            "aspect_ratios": ["1:1", "2:1", "3:1"],
            "max_windows_budget": 300,
            "stats": {
                "mean_windows_per_frame": round(float(np.mean(all_window_counts)), 1),
                "max_windows_per_frame": int(max(all_window_counts)),
                "budget_pass": budget_ok,
                "mean_ms_per_frame": round(float(np.mean(all_times)), 2),
                "min_ms_per_frame": round(float(min(all_times)), 2),
                "max_ms_per_frame": round(float(max(all_times)), 2),
                "effective_fps": round(1000.0 / float(np.mean(all_times)), 2),
            },
            "per_frame": per_frame_data,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Results JSON          : {args.out_json}")

    print("=" * 72)


if __name__ == "__main__":
    main()
