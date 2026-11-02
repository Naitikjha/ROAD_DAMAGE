"""
tools/make_splits.py

Deterministic dataset splitting by source video or contiguous time segments.
Part of Step 1 in the multi-scale sliding-window detection overhaul.

Plain-Words Explanation for Viva:
1. Why NOT random frame splitting?
   In continuous video, frame t and frame t+1 (recorded fractions of a second apart)
   show virtually identical asphalt texture, sunlight angle, and pothole contours.
   If random frames went into train and test, the test set would effectively test
   memorization rather than generalisation. This is known as temporal data leakage.
2. Contiguous Segment Splitting:
   We sort the frames by chronological order (or group by source video).
   We partition the timeline into continuous non-overlapping segments:
     - Train: first ~70% of chronological timeline (e.g. 0.0s to 19.4s)
     - Val:   next ~15% of chronological timeline (e.g. 19.4s to 23.6s)
     - Test:  final ~15% of chronological timeline (e.g. 23.6s to 27.6s)
3. Hard Boundary Assertion:
   We explicitly assert that no video or time segment appears in more than one
   split, guaranteeing zero temporal overlap across train, validation, and test.

Usage:
    python tools/make_splits.py --manifest data/manifest.csv --out_dir data/splits
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    Load extracted frame records from manifest.csv.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run tools/extract_frames.py first.")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame_idx"] = int(row["frame_idx"])
            row["timestamp_sec"] = float(row["timestamp_sec"])
            row["width"] = int(row["width"])
            row["height"] = int(row["height"])
            records.append(row)
    return records


def split_video_records_contiguously(
    video_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split a single video's sorted frame records into contiguous chronological time segments.

    Plain-words viva explanation:
    Orders frames by frame_idx. Finds cutoff indices for ~70% train, ~15% val, ~15% test.
    Assigns each partition a unique segment_id (e.g. video_seg_train, video_seg_val, video_seg_test)
    and verifies with hard assertions that no segment crosses partition boundaries.
    """
    # Sort chronologically
    sorted_recs = sorted(video_records, key=lambda r: r["frame_idx"])
    n = len(sorted_recs)

    if n < 3:
        raise ValueError(f"Need at least 3 frames to make 3-way split, got {n}")

    # Calculate cutoff points
    idx_val = max(1, int(round(n * train_ratio)))
    idx_test = max(idx_val + 1, int(round(n * (train_ratio + val_ratio))))
    idx_test = min(idx_test, n - 1)  # Ensure at least 1 test frame

    video_name = sorted_recs[0]["video"]
    seg_train_id = f"{video_name}::seg_01_train"
    seg_val_id   = f"{video_name}::seg_02_val"
    seg_test_id  = f"{video_name}::seg_03_test"

    train_recs = []
    for r in sorted_recs[:idx_val]:
        r_copy = dict(r)
        r_copy["segment_id"] = seg_train_id
        r_copy["split"] = "train"
        train_recs.append(r_copy)

    val_recs = []
    for r in sorted_recs[idx_val:idx_test]:
        r_copy = dict(r)
        r_copy["segment_id"] = seg_val_id
        r_copy["split"] = "val"
        val_recs.append(r_copy)

    test_recs = []
    for r in sorted_recs[idx_test:]:
        r_copy = dict(r)
        r_copy["segment_id"] = seg_test_id
        r_copy["split"] = "test"
        test_recs.append(r_copy)

    # Segment metadata
    meta = {
        "video": video_name,
        "total_frames": n,
        "train_segment": {
            "segment_id": seg_train_id,
            "count": len(train_recs),
            "frame_range": [train_recs[0]["frame_idx"], train_recs[-1]["frame_idx"]],
            "time_range_sec": [train_recs[0]["timestamp_sec"], train_recs[-1]["timestamp_sec"]],
        },
        "val_segment": {
            "segment_id": seg_val_id,
            "count": len(val_recs),
            "frame_range": [val_recs[0]["frame_idx"], val_recs[-1]["frame_idx"]],
            "time_range_sec": [val_recs[0]["timestamp_sec"], val_recs[-1]["timestamp_sec"]],
        },
        "test_segment": {
            "segment_id": seg_test_id,
            "count": len(test_recs),
            "frame_range": [test_recs[0]["frame_idx"], test_recs[-1]["frame_idx"]],
            "time_range_sec": [test_recs[0]["timestamp_sec"], test_recs[-1]["timestamp_sec"]],
        },
    }

    return train_recs, val_recs, test_recs, meta


def make_splits(
    manifest_path: str = "data/manifest.csv",
    out_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Partition manifest records into train, validation, and test splits
    grouped by contiguous video time segments with strict non-overlap assertions.
    """
    records = load_manifest(Path(manifest_path))

    # Group records by video source
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_video.setdefault(r["video"], []).append(r)

    print("=" * 65)
    print("      DATASET SPLITTING BY CONTIGUOUS TIME SEGMENT")
    print("=" * 65)
    print(f"Total Source Videos: {len(by_video)}")
    print(f"Total Frames:        {len(records)}")
    print(f"Target Split Ratio:  {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {(1.0-train_ratio-val_ratio)*100:.0f}% Test\n")

    all_train: List[Dict[str, Any]] = []
    all_val: List[Dict[str, Any]] = []
    all_test: List[Dict[str, Any]] = []
    segments_summary: List[Dict[str, Any]] = []

    for vname, v_recs in by_video.items():
        tr, va, te, meta = split_video_records_contiguously(
            v_recs, train_ratio=train_ratio, val_ratio=val_ratio
        )
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        segments_summary.append(meta)

    # ── Strict Non-Overlap Assertions (Rule Compliance) ──────────────────────
    train_segments = {r["segment_id"] for r in all_train}
    val_segments = {r["segment_id"] for r in all_val}
    test_segments = {r["segment_id"] for r in all_test}

    assert len(train_segments & val_segments) == 0, (
        f"Contamination error: Train and Val share segment(s): {train_segments & val_segments}"
    )
    assert len(train_segments & test_segments) == 0, (
        f"Contamination error: Train and Test share segment(s): {train_segments & test_segments}"
    )
    assert len(val_segments & test_segments) == 0, (
        f"Contamination error: Val and Test share segment(s): {val_segments & test_segments}"
    )

    # Assert no individual frame path is shared
    train_paths = {r["image_path"] for r in all_train}
    val_paths = {r["image_path"] for r in all_val}
    test_paths = {r["image_path"] for r in all_test}

    assert len(train_paths & val_paths) == 0, "Train and Val share image frames!"
    assert len(train_paths & test_paths) == 0, "Train and Test share image frames!"
    assert len(val_paths & test_paths) == 0, "Val and Test share image frames!"

    # ── Write Output Split Files ─────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video", "segment_id", "split", "frame_idx", "timestamp_sec",
        "image_path", "image_filename", "width", "height"
    ]

    for split_name, split_recs in [("train", all_train), ("val", all_val), ("test", all_test)]:
        csv_file = out_path / f"{split_name}_manifest.csv"
        txt_file = out_path / f"{split_name}.txt"

        # Write CSV manifest
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_recs)

        # Write txt list of image paths (standard for YOLO / custom datasets)
        with open(txt_file, "w", encoding="utf-8") as f:
            for r in split_recs:
                f.write(f"{r['image_path']}\n")

    summary_file = out_path / "splits_summary.json"
    summary_data = {
        "split_method": "Contiguous Chronological Time Segments per Source Video",
        "total_source_videos": len(by_video),
        "total_frames": len(records),
        "split_counts": {
            "train": len(all_train),
            "val": len(all_val),
            "test": len(all_test),
        },
        "split_percentages": {
            "train": round(len(all_train) / len(records) * 100, 2),
            "val": round(len(all_val) / len(records) * 100, 2),
            "test": round(len(all_test) / len(records) * 100, 2),
        },
        "segments": segments_summary,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("Contiguous Time Segment Breakdown:")
    for sm in segments_summary:
        print(f"  Video: {sm['video']}")
        print(f"    Train: {sm['train_segment']['count']} frames "
              f"(Frames {sm['train_segment']['frame_range'][0]}-{sm['train_segment']['frame_range'][1]}, "
              f"Time {sm['train_segment']['time_range_sec'][0]}s-{sm['train_segment']['time_range_sec'][1]}s)")
        print(f"    Val:   {sm['val_segment']['count']} frames "
              f"(Frames {sm['val_segment']['frame_range'][0]}-{sm['val_segment']['frame_range'][1]}, "
              f"Time {sm['val_segment']['time_range_sec'][0]}s-{sm['val_segment']['time_range_sec'][1]}s)")
        print(f"    Test:  {sm['test_segment']['count']} frames "
              f"(Frames {sm['test_segment']['frame_range'][0]}-{sm['test_segment']['frame_range'][1]}, "
              f"Time {sm['test_segment']['time_range_sec'][0]}s-{sm['test_segment']['time_range_sec'][1]}s)")

    print("-" * 65)
    print(f"Split Totals:")
    print(f"  Train: {len(all_train)} frames ({len(all_train)/len(records)*100:.1f}%)")
    print(f"  Val:   {len(all_val)} frames ({len(all_val)/len(records)*100:.1f}%)")
    print(f"  Test:  {len(all_test)} frames ({len(all_test)/len(records)*100:.1f}%)")
    print(f"\nSaved split manifests and text path lists to: {out_path}")
    print(f"Summary JSON:                                {summary_file}")
    print("=" * 65)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video frames by contiguous time segments.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv",
                        help="Path to manifest.csv generated by extract_frames.py")
    parser.add_argument("--out_dir", type=str, default="data/splits",
                        help="Output directory for split files")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Train fraction (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation fraction (default: 0.15)")
    args = parser.parse_args()

    make_splits(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
