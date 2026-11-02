"""
tools/extract_frames.py

Frame extraction and perceptual near-duplicate suppression for video dataset creation.
Part of Step 1 in the multi-scale sliding-window detection overhaul.

Plain-Words Explanation for Viva:
1. compute_dhash: Downscales a frame to a small 9x8 grayscale thumbnail and compares
   each pixel to its horizontal neighbour. This produces a 64-bit binary fingerprint
   representing luminance gradients. Small lighting changes or compression artifacts
   barely change this hash.
2. hamming_distance: Counts how many bits differ between two 64-bit dHashes using XOR.
   A distance <= threshold (default 3) means the two frames look almost identical
   (e.g., car is stopped at a traffic light or moving very slowly).
3. extract_frames_from_video: Reads video frame-by-frame, samples every N-th frame
   (stride), skips consecutive frames that are perceptual near-duplicates of the
   last accepted frame, saves the unique frames to disk as JPEG, and appends
   metadata (video name, frame index, timestamp, path, dimensions) to manifest.csv.

Usage:
    python tools/extract_frames.py --video sample_pothole_video.mp4 --step 10 --out_dir data/extracted_frames
    python tools/extract_frames.py --video_dir . --step 15 --out_dir data/extracted_frames
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


def compute_dhash(image_gray: np.ndarray, hash_size: int = 8) -> int:
    """
    Compute a 64-bit difference hash (dHash) from a grayscale image.

    Plain-words viva explanation:
    Resizes the image to (hash_size + 1, hash_size) = 9x8 pixels.
    For each row, compares adjacent columns: if left pixel > right pixel, bit is 1, else 0.
    Produces 8 rows x 8 comparisons = 64 bits.
    """
    resized = cv2.resize(image_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack boolean array into integer
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Count the number of differing bits between two 64-bit integers.

    Plain-words viva explanation:
    XOR sets bits to 1 wherever the two hashes disagree. Counting set bits
    gives the number of gradient mismatches between the two frames.
    """
    return bin(hash1 ^ hash2).count("1")


def find_video_files(input_path: str) -> List[Path]:
    """
    Resolve single video file or scan directory for common video extensions.
    """
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        extensions = {".mp4", ".avi", ".mov", ".mkv"}
        videos = [f for f in sorted(p.iterdir()) if f.suffix.lower() in extensions]
        return videos
    return []


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    step: int = 10,
    dhash_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """
    Extract frames from a single video, dropping near-duplicates based on dHash.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    out_dir : Path
        Directory where extracted JPEGs will be saved.
    step : int
        Sample every step-th frame (temporal downsampling).
    dhash_threshold : int
        Max Hamming distance to consider two consecutive sampled frames duplicates.

    Returns
    -------
    List of metadata dicts for manifest.csv.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = video_path.name

    print(f"Processing: {video_name} ({total_frames} frames, {width}x{height} @ {fps:.1f} fps)")

    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    last_accepted_hash: Optional[int] = None
    frame_idx = 0
    sampled_count = 0
    dropped_dupes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            sampled_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cur_hash = compute_dhash(gray)

            # Check if near-duplicate of last accepted frame
            if last_accepted_hash is not None:
                dist = hamming_distance(cur_hash, last_accepted_hash)
                if dist <= dhash_threshold:
                    dropped_dupes += 1
                    frame_idx += 1
                    continue

            # Save unique frame
            frame_filename = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            save_path = out_dir / frame_filename
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            timestamp_sec = round(frame_idx / max(fps, 1e-4), 3)
            records.append({
                "video": video_name,
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "image_path": str(save_path).replace("\\", "/"),
                "image_filename": frame_filename,
                "width": width,
                "height": height,
            })
            last_accepted_hash = cur_hash

        frame_idx += 1

    cap.release()
    accepted = len(records)
    print(f"  -> Sampled {sampled_count} frames | Dropped {dropped_dupes} near-dupes | Saved {accepted} frames")
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract frames from road video(s) and drop near-duplicates.")
    parser.add_argument("--video", type=str, default="sample_pothole_video.mp4",
                        help="Path to video file or directory containing videos")
    parser.add_argument("--step", type=int, default=10,
                        help="Sample every N-th frame (default: 10)")
    parser.add_argument("--dhash_thresh", type=int, default=3,
                        help="Hamming distance threshold for near-duplicate rejection (default: 3)")
    parser.add_argument("--out_dir", type=str, default="data/extracted_frames",
                        help="Directory to save extracted frame images")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv",
                        help="Output path for manifest.csv")
    args = parser.parse_args()

    videos = find_video_files(args.video)
    if not videos:
        print(f"Error: No video files found at '{args.video}'.")
        return

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    for v in videos:
        recs = extract_frames_from_video(
            video_path=v,
            out_dir=out_dir,
            step=args.step,
            dhash_threshold=args.dhash_thresh,
        )
        all_records.extend(recs)

    if not all_records:
        print("Warning: Zero frames were extracted.")
        return

    # Write manifest.csv
    fieldnames = ["video", "frame_idx", "timestamp_sec", "image_path", "image_filename", "width", "height"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 60)
    print(f"Extraction Complete!")
    print(f"Total Unique Frames Extracted: {len(all_records)}")
    print(f"Output Directory:             {out_dir}")
    print(f"Manifest File:                {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
