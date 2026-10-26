"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Pothole Detection from Road Videos — Streamlit Frontend                  ║
║   Course project — Custom CNN (no pretrained models)                       ║
║                                                                             ║
║   REQUIRED PACKAGES:                                                        ║
║       pip install streamlit opencv-python numpy imageio-ffmpeg             ║
║                                                                             ║
║   RUN:                                                                      ║
║       streamlit run app.py                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (edit these; no magic numbers in the rest of the code)
# ─────────────────────────────────────────────────────────────────────────────

# The custom CNN operates at this resolution internally.
# NOTE: Rescaling is done INSIDE your CNN code, not here.
#       predict() receives the original-size frame and must return boxes
#       in the original frame's pixel coordinates.
MODEL_INPUT_SIZE = (416, 416)

# Severity is derived from relative box area (box_area / frame_area).
# These are the upper-bound ratios for each bucket:
#   ratio <= LOW_THRESH  -> "low"
#   ratio <= MED_THRESH  -> "medium"
#   ratio >  MED_THRESH  -> "high"
SEVERITY_THRESHOLDS = {
    "low":    0.02,   # box covers <= 2% of the frame
    "medium": 0.06,   # box covers 2-6% of the frame
}

# Drawing colours in BGR (OpenCV order)
SEVERITY_COLORS = {
    "low":    (0, 200,   0),   # green
    "medium": (0, 140, 255),   # orange
    "high":   (0,   0, 220),   # red
}

# File-upload constraints
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov"]
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]
MAX_UPLOAD_MB           = 200          # mirrors .streamlit/config.toml
MAX_VIDEO_DURATION_S    = 300          # 5 minutes

# Root folder for per-session temporary files
TEMP_ROOT = Path(tempfile.gettempdir()) / "pothole_app"

# ── DEMO MODE ─────────────────────────────────────────────────────────────────
# DEMO_MODE is now False by default because our custom CNN (ResPotholeNet)
# is trained and plugged in! You can still toggle it in the sidebar if needed.
DEMO_MODE = False
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL INTERFACE — custom CNN (ResPotholeNet)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load and cache our custom CNN detector (ResPotholeNet)."""
    try:
        from detector import PotholeDetector
        return PotholeDetector(weights_path="pothole_cnn.pth")
    except Exception as e:
        print(f"Error loading custom CNN: {e}")
        return None


def predict(frame: np.ndarray, conf: float, demo_mode: bool = DEMO_MODE) -> list:
    """
    Run pothole detection on a single frame.
    Uses our custom ResPotholeNet CNN (no pretrained models).
    """
    if demo_mode:
        h, w = frame.shape[:2]
        candidates = [
            {"box": (int(w * 0.38), int(h * 0.68), int(w * 0.60), int(h * 0.82)), "score": 0.86},
            {"box": (int(w * 0.18), int(h * 0.75), int(w * 0.30), int(h * 0.84)), "score": 0.64},
            {"box": (int(w * 0.65), int(h * 0.70), int(w * 0.92), int(h * 0.93)), "score": 0.91},
        ]
        return [c for c in candidates if c["score"] >= conf]

    detector = load_model()
    if detector is not None:
        return detector.predict(frame, conf_threshold=conf)
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_severity(box: tuple, frame_shape: tuple) -> str:
    """
    Map a bounding box to a severity level using the relative box area.

    Severity is intentionally decoupled from the model's confidence score
    so that it reflects the physical size of the pothole in the frame.

    Parameters
    ----------
    box         : (x1, y1, x2, y2) in original pixel coordinates
    frame_shape : (height, width, channels)

    Returns
    -------
    "low" | "medium" | "high"
    """
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w
    box_area   = max(0, x2 - x1) * max(0, y2 - y1)

    if frame_area == 0:
        return "low"

    ratio = box_area / frame_area

    if ratio <= SEVERITY_THRESHOLDS["low"]:
        return "low"
    elif ratio <= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "high"


def draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the frame.

    Parameters
    ----------
    frame      : BGR numpy array (original size)
    detections : list of dicts with keys "box", "score", "severity"

    Returns
    -------
    Annotated BGR numpy array (new copy — original is not modified)
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        severity        = det["severity"]
        score           = det["score"]
        color           = SEVERITY_COLORS.get(severity, (255, 255, 255))

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

        # Label background + text
        label      = f"{severity.upper()}  {score:.2f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Keep label inside frame
        label_y = max(y1 - 6, th + baseline + 4)
        cv2.rectangle(out,
                      (x1, label_y - th - baseline - 4),
                      (x1 + tw + 4, label_y),
                      color, cv2.FILLED)
        # White text on coloured background for readability
        cv2.putText(out, label,
                    (x1 + 2, label_y - baseline - 1),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def to_h264(input_path: str, output_path: str) -> str:
    """
    Re-encode a video to H.264 / yuv420p using the ffmpeg binary bundled
    with imageio-ffmpeg.

    OpenCV's mp4v output is not playable in most browsers; H.264 is.
    -movflags +faststart moves the moov atom to the beginning of the file
    so the browser can start playing before the full download completes.

    Parameters
    ----------
    input_path  : path to the raw OpenCV-written video
    output_path : destination path for the H.264 file

    Returns
    -------
    output_path on success

    Raises
    ------
    RuntimeError if ffmpeg exits with a non-zero code
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(f"Could not locate ffmpeg binary: {exc}") from exc

    cmd = [
        ffmpeg_exe,
        "-y",                        # overwrite output without asking
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",                # quality (lower = better, 23 is sane default)
        "-movflags", "+faststart",   # browser-friendly streaming
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg re-encoding failed (exit {result.returncode}).\n"
            f"stderr:\n{result.stderr[-2000:]}"   # last 2 KB of log
        )

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  TEMP-DIRECTORY MANAGEMENT (per session)
# ─────────────────────────────────────────────────────────────────────────────

def make_session_temp_dir() -> Path:
    """
    Create (or reuse) a unique temporary directory for the current session.
    Stored in st.session_state["temp_dir"] so it persists across reruns.
    """
    if "temp_dir" not in st.session_state:
        session_id = str(uuid.uuid4())
        temp_dir   = TEMP_ROOT / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["temp_dir"] = temp_dir
    return st.session_state["temp_dir"]


def cleanup_temp_dir():
    """Remove the session temp directory and all its contents."""
    temp_dir = st.session_state.get("temp_dir")
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path: str,
    output_path: str,
    conf: float,
    every_n: int,
    progress_callback=None,
    demo_mode: bool = DEMO_MODE,
) -> dict:
    """
    Process a video file frame-by-frame and write an annotated output.

    Parameters
    ----------
    input_path        : path to the uploaded video
    output_path       : path for the raw (pre-H.264) output video
    conf              : confidence threshold passed to predict()
    every_n           : call predict() every N frames; reuse last result otherwise
    progress_callback : optional callable(float 0-1) for updating a progress bar.
                        Decoupled from Streamlit so this function is testable.

    Returns
    -------
    dict with keys:
        frames_processed  : int
        max_potholes      : int   (max detections in one frame — placeholder metric)
        processing_time_s : float
        detections_log    : list of {frame, timestamp_s, severity, score}

    Raises
    ------
    ValueError   for invalid / empty video
    RuntimeError for exceeded duration limit
    """
    cap    = None
    writer = None

    try:
        # ── Open capture ──────────────────────────────────────────────────────
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file. Is the file corrupted?")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames == 0:
            raise ValueError("Video reports 0 frames. The file may be empty or corrupted.")
        if fps == 0:
            raise ValueError(
                "Could not read FPS from video. The file may not be a valid video."
            )

        # ── Duration guard ────────────────────────────────────────────────────
        duration_s = total_frames / fps
        if duration_s > MAX_VIDEO_DURATION_S:
            raise RuntimeError(
                f"Video is {duration_s:.0f}s long.  "
                f"Maximum allowed is {MAX_VIDEO_DURATION_S}s "
                f"({MAX_VIDEO_DURATION_S // 60} min).  "
                "Please trim the video before uploading."
            )

        # ── VideoWriter ───────────────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ── Loop state ────────────────────────────────────────────────────────
        last_dets        = []   # enriched detections (with severity) reused on skips
        frames_read      = 0
        frames_processed = 0
        max_potholes     = 0
        detections_log   = []
        start_time       = time.perf_counter()

        # ── Frame loop ────────────────────────────────────────────────────────
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1

            # ── Predict on every Nth frame ────────────────────────────────────
            if frame_idx % every_n == 0:
                raw_dets = predict(frame, conf, demo_mode=demo_mode)

                # ── PLUG TRACKING / TEMPORAL FILTERING HERE ──────────────────
                # Replace 'tracked_dets' with the output of your tracker
                # (e.g. SORT, DeepSORT, ByteTrack).
                #
                # Goal: assign a stable ID to each pothole so that it is
                # counted only once across the entire video, not once per frame.
                #
                # For now we pass raw_dets straight through.
                tracked_dets = raw_dets   # <- replace with tracker output
                # ─────────────────────────────────────────────────────────────

                # Attach severity (computed from box area, not score)
                enriched = []
                for det in tracked_dets:
                    sev = get_severity(det["box"], frame.shape)
                    enriched.append({**det, "severity": sev})

                last_dets = enriched
                frames_processed += 1

                # Log one entry per detection on this frame
                timestamp_s = round(frame_idx / fps, 2)
                for det in enriched:
                    detections_log.append({
                        "frame":       frame_idx,
                        "timestamp_s": timestamp_s,
                        "severity":    det["severity"],
                        "score":       round(det["score"], 3),
                    })

            # Track max potholes in a single frame (placeholder metric)
            max_potholes = max(max_potholes, len(last_dets))

            # Draw and write
            annotated = draw_boxes(frame, last_dets)
            writer.write(annotated)

            # Progress callback (decoupled from Streamlit)
            if progress_callback is not None:
                progress_callback((frame_idx + 1) / total_frames)

        # ── Guard: zero frames read ───────────────────────────────────────────
        if frames_read == 0:
            raise ValueError(
                "No frames could be read from the video.  "
                "The file may be empty or the codec is unsupported."
            )

        elapsed = time.perf_counter() - start_time

        return {
            "frames_processed":  frames_processed,
            "max_potholes":      max_potholes,
            "processing_time_s": round(elapsed, 2),
            "detections_log":    detections_log,
        }

    finally:
        # Always release resources, even if an exception was raised
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def process_image(image_bgr: np.ndarray, conf: float, demo_mode: bool = DEMO_MODE) -> dict:
    """
    Run pothole detection on a single image.

    Parameters
    ----------
    image_bgr : BGR numpy array (decoded from uploaded image)
    conf      : confidence threshold
    demo_mode : bool

    Returns
    -------
    dict with keys:
        annotated_image : np.ndarray  (BGR)
        detections      : list of dicts with box, score, severity
    """
    raw_dets = predict(image_bgr, conf, demo_mode=demo_mode)

    enriched = []
    for det in raw_dets:
        sev = get_severity(det["box"], image_bgr.shape)
        enriched.append({**det, "severity": sev})

    annotated = draw_boxes(image_bgr, enriched)
    return {"annotated_image": annotated, "detections": enriched}


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_css():
    """Inject page-wide CSS — hero header, metric cards, misc polish."""
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Hero header ── */
        .hero {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            border-radius: 16px;
            padding: 2.4rem 2.8rem 2rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at 80% 20%,
                        rgba(120,80,255,0.25) 0%, transparent 60%);
        }
        .hero h1 {
            margin: 0 0 0.4rem;
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }
        .hero p {
            margin: 0;
            font-size: 1rem;
            font-weight: 300;
            color: rgba(255,255,255,0.72);
        }
        .hero .badge {
            display: inline-block;
            background: rgba(255,255,255,0.12);
            color: rgba(255,255,255,0.85);
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            margin-bottom: 0.7rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* ── Metric cards ── */
        .metric-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.18);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.28);
        }
        .metric-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: rgba(160,160,200,0.85);
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #a78bfa;
            line-height: 1;
        }
        .metric-note {
            font-size: 0.65rem;
            color: rgba(160,160,200,0.6);
            margin-top: 0.3rem;
        }

        /* ── Subtle section divider ── */
        .section-divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.08);
            margin: 1.5rem 0;
        }

        /* ── Upload area ── */
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px !important;
        }

        /* ── Tab styling ── */
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.9rem;
        }

        /* ── Download button ── */
        [data-testid="stDownloadButton"] button {
            border-radius: 10px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the gradient hero banner at the top of the page."""
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Course Project &middot; Custom CNN &middot; No Pretrained Models</div>
            <h1>Pothole Detection</h1>
            <p>Upload a road video or image &mdash; your custom CNN will locate potholes
               and overlay bounding boxes coloured by severity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple:
    """Render sidebar controls and return (conf_threshold, every_n)."""
    with st.sidebar:
        st.markdown("## Settings")
        st.markdown("---")

        conf = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Detections with score below this value are discarded.",
        )
        st.caption(
            "Lower -> more detections (may include false positives).  "
            "Higher -> only confident detections."
        )

        st.markdown("")

        every_n = st.slider(
            "Process every Nth frame",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Skip frames to speed up processing; last result is reused on skipped frames.",
        )
        st.caption(
            "N=1: slowest, most accurate.  "
            "N=5: fastest, detections reused across 5 frames."
        )

        st.markdown("---")

        with st.expander("About the Model", expanded=False):
            st.markdown(
                """
                **ResPotholeNet Architecture (From Scratch)**

                | Attribute | Specification |
                |---|---|
                | **Backbone** | Custom Residual Bottlenecks + SE Attention |
                | **Attention** | Squeeze-and-Excitation (SE) Channel Attention |
                | **Multi-scale** | Spatial Pyramid Pooling (SPPF) |
                | **Parameters** | 858,245 parameters |
                | **Pretrained** | **None** (100% custom from scratch) |
                | **Loss** | Focal Loss + Smooth L1 Loss |
                | **Dual Head** | Pothole Confidence + Bounding Box Regression |

                ---
                **Course Compliance:** No YOLO, Ultralytics, torchvision, or transfer learning backbones used.
                """
            )

        st.markdown("---")

        demo_mode = st.toggle(
            "🧪 Demo Mode (Test UI)",
            value=DEMO_MODE,
            help="Generates test bounding boxes on the road so you can verify the video pipeline and UI before connecting your CNN.",
        )
        if demo_mode:
            st.info("Demo Mode is active. Turn off when your CNN is connected.", icon="💡")

    return conf, every_n, demo_mode


def render_metric_cards(stats: dict):
    """Render 3 styled metric cards from a stats dict."""
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Frames Processed</div>
                <div class="metric-value">{stats['frames_processed']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Max Potholes / Frame</div>
                <div class="metric-value">{stats['max_potholes']}</div>
                <div class="metric-note">Placeholder — add tracking to count unique potholes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Processing Time</div>
                <div class="metric-value">{stats['processing_time_s']}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detections_table(log: list):
    """Render the detections log as a styled dataframe."""
    st.markdown("#### Detections Log")
    if not log:
        st.info(
            "No detections to display.  "
            "Once your CNN is connected, each detected pothole will appear here.",
            icon="ℹ️",
        )
        return

    import pandas as pd

    df = pd.DataFrame(log)
    df.columns = ["Frame #", "Timestamp (s)", "Severity", "Confidence"]

    severity_order = {"high": 0, "medium": 1, "low": 2}
    df["_order"] = df["Severity"].map(severity_order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)


def clear_results_if_new_file(file_obj):
    """
    Detect when a different file has been uploaded and wipe stale results
    from session_state so they do not show for the new file.
    """
    file_id = id(file_obj)
    if st.session_state.get("uploaded_file_id") != file_id:
        for key in ["result_video_path", "result_image", "stats", "detections_log"]:
            st.session_state.pop(key, None)
        st.session_state["uploaded_file_id"] = file_id


# ─── VIDEO TAB ───────────────────────────────────────────────────────────────

def render_video_tab(conf: float, every_n: int, demo_mode: bool = DEMO_MODE):
    """Render the video upload, detection, and results section."""
    st.markdown("### Upload a Road Video")
    st.caption(
        f"Supported: {', '.join(f'.{f}' for f in SUPPORTED_VIDEO_FORMATS)}  |  "
        f"Max {MAX_UPLOAD_MB} MB  |  Max {MAX_VIDEO_DURATION_S // 60} minutes"
    )

    sample_file = Path("sample_pothole_video.mp4")

    col_upload, col_sample = st.columns([3, 2])
    with col_upload:
        uploaded = st.file_uploader(
            "Drop your video here",
            type=SUPPORTED_VIDEO_FORMATS,
            key="video_uploader",
            label_visibility="collapsed",
        )
    with col_sample:
        if sample_file.exists():
            st.markdown("**Or test with sample video:**")
            if st.button("🎬 Use sample_pothole_video.mp4", use_container_width=True):
                st.session_state["use_sample_video"] = True
                for key in ["result_video_path", "result_image", "stats", "detections_log"]:
                    st.session_state.pop(key, None)

    active_video_path = None
    if uploaded is not None:
        st.session_state["use_sample_video"] = False
        clear_results_if_new_file(uploaded)
        temp_dir    = make_session_temp_dir()
        upload_path = str(temp_dir / f"input_{uploaded.name}")
        with open(upload_path, "wb") as f:
            f.write(uploaded.read())
        active_video_path = upload_path
    elif st.session_state.get("use_sample_video", False) and sample_file.exists():
        temp_dir    = make_session_temp_dir()
        upload_path = str(temp_dir / "input_sample_pothole_video.mp4")
        if not Path(upload_path).exists():
            shutil.copyfile(str(sample_file), upload_path)
        active_video_path = upload_path

    if active_video_path is None:
        st.info(
            "Upload a road video (.mp4, .avi, .mov) or click 'Use sample_pothole_video.mp4' to get started.",
            icon="🛣️",
        )
        return

    temp_dir = make_session_temp_dir()

    # Two-column video display
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.video(active_video_path)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    detect_btn = st.button(
        "Detect Potholes",
        type="primary",
        use_container_width=True,
        key="detect_video_btn",
    )

    if detect_btn:
        raw_output  = str(temp_dir / "output_raw.mp4")
        h264_output = str(temp_dir / "output_h264.mp4")

        progress_bar = st.progress(0, text="Processing frames...")
        status_text  = st.empty()

        try:
            status_text.markdown("Running detection...")

            stats = process_video(
                input_path=active_video_path,
                output_path=raw_output,
                conf=conf,
                every_n=every_n,
                progress_callback=lambda p: progress_bar.progress(
                    min(p, 1.0),
                    text=f"Processing frames... {int(p * 100)}%",
                ),
                demo_mode=demo_mode,
            )

            status_text.markdown("Re-encoding to H.264 for browser playback...")
            to_h264(raw_output, h264_output)

            # Persist in session_state so results survive reruns
            detections_log = stats.pop("detections_log", [])
            st.session_state["result_video_path"] = h264_output
            st.session_state["stats"]             = stats
            st.session_state["detections_log"]    = detections_log

            progress_bar.progress(1.0, text="Done!")
            status_text.success("Detection complete!")

        except (ValueError, RuntimeError) as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error: {exc}")
            return

        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Unexpected error: {exc}")
            return

    # Show results — persisted across reruns via session_state
    if "result_video_path" in st.session_state:
        result_path = st.session_state["result_video_path"]

        with col_det:
            st.markdown("**Detected**")
            st.video(result_path)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_metric_cards(st.session_state["stats"])

        st.markdown("")
        with open(result_path, "rb") as f:
            st.download_button(
                label="Download Result Video",
                data=f,
                file_name="pothole_detected.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_detections_table(st.session_state.get("detections_log", []))


# ─── IMAGE TAB ───────────────────────────────────────────────────────────────

def render_image_tab(conf: float, demo_mode: bool = DEMO_MODE):
    """Render the image upload, detection, and results section."""
    st.markdown("### Upload a Road Image")
    st.caption(
        f"Supported: {', '.join(f'.{f}' for f in SUPPORTED_IMAGE_FORMATS)}  |  "
        f"Max {MAX_UPLOAD_MB} MB"
    )

    uploaded = st.file_uploader(
        "Drop your image here",
        type=SUPPORTED_IMAGE_FORMATS,
        key="image_uploader",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info(
            "Upload a road image (.jpg, .jpeg, .png) to get started.",
            icon="🖼️",
        )
        return

    clear_results_if_new_file(uploaded)

    # Decode image
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode the image. Please upload a valid JPG or PNG file.")
        return

    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    detect_btn = st.button(
        "Detect Potholes",
        type="primary",
        use_container_width=True,
        key="detect_image_btn",
    )

    if detect_btn:
        with st.spinner("Running detection..."):
            try:
                result = process_image(image_bgr, conf, demo_mode=demo_mode)
                st.session_state["result_image"]   = result["annotated_image"]
                st.session_state["detections_log"] = [
                    {
                        "frame":       "N/A",
                        "timestamp_s": "N/A",
                        "severity":    d["severity"],
                        "score":       round(d["score"], 3),
                    }
                    for d in result["detections"]
                ]
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                return

    if "result_image" in st.session_state:
        with col_det:
            st.markdown("**Detected**")
            st.image(
                cv2.cvtColor(st.session_state["result_image"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        n_det = len(st.session_state.get("detections_log", []))
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Potholes Found</div>
                    <div class="metric-value">{n_det}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Save annotated image to temp dir for download
        temp_dir     = make_session_temp_dir()
        img_out_path = str(temp_dir / "detected_image.jpg")
        cv2.imwrite(img_out_path, st.session_state["result_image"])

        with c2:
            with open(img_out_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Image",
                    data=f,
                    file_name="pothole_detected.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_detections_table(st.session_state.get("detections_log", []))


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Pothole Detector | Custom CNN",
        page_icon="🕳️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()
    render_hero()

    # Sidebar returns the settings used by both tabs
    conf, every_n, demo_mode = render_sidebar()

    tab_video, tab_image = st.tabs(["🎬  Video", "🖼️  Image"])

    with tab_video:
        render_video_tab(conf, every_n, demo_mode)

    with tab_image:
        render_image_tab(conf, demo_mode)


if __name__ == "__main__":
    main()
