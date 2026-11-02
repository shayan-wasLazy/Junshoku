
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

from node_graph.model import NodeGraph
from node_graph.graph import apply_graph


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

CONFIG_PATH = ROOT / "node_graph" / "configs" / "default.yaml"
CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "node_graph_finetuned" / "best.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SIDE = 512


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Junshoku Studio",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
    .main {
        background-color: #101114;
    }

    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #f4f4f5;
    }

    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Avoid downloading pretrained VGG weights.
    # The checkpoint already contains encoder weights.
    model_cfg = dict(cfg)
    model_cfg["model"] = dict(cfg["model"])
    model_cfg["model"]["pretrained"] = False

    model = NodeGraph(model_cfg).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, cfg


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def resize_rgb(image, max_side=MAX_SIDE):

    h, w = image.shape[:2]

    scale = min(1.0, max_side / max(h, w))

    if scale < 1:
        image = cv2.resize(
            image,
            (
                max(1, round(w * scale)),
                max(1, round(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


def image_to_tensor(image):

    image = np.asarray(image.convert("RGB"))

    image = resize_rgb(image)

    tensor = torch.from_numpy(
        image.astype(np.float32) / 255.0
    )

    return tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def frame_to_tensor(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = resize_rgb(rgb)

    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    )

    return tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def tensor_to_image(tensor):

    image = (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .cpu()
        .numpy()
    )

    return (image * 255).astype(np.uint8)


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------


def grade_image(model, cfg, source, reference):

    # Resize reference to match source dimensions.
    # Required because the spatial feature maps are concatenated.
    reference = torch.nn.functional.interpolate(
        reference,
        size=source.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    with torch.inference_mode():

        prediction = model(source, reference)

        output = apply_graph(source, prediction, cfg)

    return output



# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def process_video(
    model,
    cfg,
    video_path,
    reference,
    start_time,
    end_time,
    output_path,
    progress_bar,
    status_text,
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_time * fps))
    end_frame = min(
        total_frames,
        int(end_time * fps),
    )

    if end_frame <= start_frame:
        cap.release()
        raise ValueError("Invalid video trim range.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize video writer.")

    frame_count = end_frame - start_frame

    processed = 0

    try:

        with torch.inference_mode():

            while processed < frame_count:

                success, frame = cap.read()

                if not success:
                    break

                source = frame_to_tensor(frame)

                graded = grade_image(
                    model,
                    cfg,
                    source,
                    reference,
                )

                output = tensor_to_image(graded)

                output = cv2.resize(
                    output,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

                output = cv2.cvtColor(
                    output,
                    cv2.COLOR_RGB2BGR,
                )

                writer.write(output)

                processed += 1

                progress_bar.progress(
                    min(processed / frame_count, 1.0)
                )

                status_text.write(
                    f"Processing frame {processed}/{frame_count}"
                )

    finally:
        cap.release()
        writer.release()

    if processed == 0:
        raise RuntimeError("No frames were processed.")

    return processed


# --------------------------------------------------
# AUDIO PRESERVATION
# --------------------------------------------------

def mux_audio(source_path, graded_path, final_path, start_time):

    # Requires FFmpeg.
    # If the original video has no audio, use the graded video directly.

    try:

        duration = None

        cap = cv2.VideoCapture(str(graded_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        duration = frame_count / fps

        command = [
            "ffmpeg",
            "-y",
            "-i", str(graded_path),
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(source_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(final_path),
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        return final_path

    except (FileNotFoundError, subprocess.CalledProcessError):

        # Fallback: deliver graded video without audio.
        return graded_path


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

st.title("JUNSHOKU")
st.caption("AI-powered spatially adaptive cinematic colour grading")

st.divider()

# Load model
try:

    with st.spinner("Loading Junshoku model..."):
        model, cfg = load_model()

    st.success(
        f"Model loaded successfully | Device: {DEVICE}"
    )

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("System")

    st.write(f"**Device:** {DEVICE}")

    if DEVICE.type == "cuda":

        st.write(
            f"**GPU:** {torch.cuda.get_device_name(0)}"
        )

    st.divider()

    st.header("About Junshoku")

    st.write("""
    Junshoku uses a VGG16-based feature encoder,
    global parameter prediction and three independent
    spatial context maps.

    The grading pipeline consists of:

    1. Exposure and white balance
    2. Tonal contrast
    3. Creative colour
    """)


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("Source Video")

    uploaded_video = st.file_uploader(
        "Upload source video",
        type=["mp4", "mov", "avi", "mkv"],
    )

with right:

    st.subheader("Reference Image")

    uploaded_reference = st.file_uploader(
        "Upload reference image",
        type=["jpg", "jpeg", "png", "webp"],
    )


# --------------------------------------------------
# VIDEO PREVIEW AND TRIM
# --------------------------------------------------

if uploaded_video is not None:

    video_bytes = uploaded_video.getvalue()

    st.video(video_bytes)

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ) as temp_video:

        temp_video.write(video_bytes)
        source_path = Path(temp_video.name)

    cap = cv2.VideoCapture(str(source_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    cap.release()

    st.caption(
        f"Duration: {duration:.2f}s | "
        f"FPS: {fps:.2f} | "
        f"Frames: {total_frames}"
    )

    st.subheader("Timeline")

    trim_start = st.number_input(
        "Start time (seconds)",
        min_value=0.0,
        max_value=float(duration),
        value=0.0,
        step=0.5,
    )

    trim_end = st.number_input(
        "End time (seconds)",
        min_value=0.0,
        max_value=float(duration),
        value=float(duration),
        step=0.5,
    )

else:

    source_path = None
    duration = 0.0
    trim_start = 0.0
    trim_end = 0.0


# --------------------------------------------------
# REFERENCE PREVIEW
# --------------------------------------------------

if uploaded_reference is not None:

    reference_image = Image.open(uploaded_reference).convert("RGB")

    st.subheader("Reference Image")

    st.image(
        reference_image,
        use_container_width=True,
    )

    reference_tensor = image_to_tensor(reference_image)

else:

    reference_tensor = None


# --------------------------------------------------
# PROCESSING
# --------------------------------------------------

st.divider()

if uploaded_video is not None and uploaded_reference is not None:

    if trim_end <= trim_start:

        st.warning("End time must be greater than start time.")

    else:

        if st.button(
            "GENERATE GRADED VIDEO",
            type="primary",
        ):

            progress = st.progress(0)

            status = st.empty()

            try:

                with tempfile.TemporaryDirectory() as tmp:

                    tmp = Path(tmp)

                    graded_path = tmp / "graded.mp4"
                    final_path = tmp / "junshoku_final.mp4"

                    with st.spinner("Grading video..."):

                        processed = process_video(
                            model=model,
                            cfg=cfg,
                            video_path=source_path,
                            reference=reference_tensor,
                            start_time=trim_start,
                            end_time=trim_end,
                            output_path=graded_path,
                            progress_bar=progress,
                            status_text=status,
                        )

                    status.success(
                        f"Successfully processed {processed} frames."
                    )

                    result_path = mux_audio(
                        source_path,
                        graded_path,
                        final_path,
                        trim_start,
                    )

                    result_bytes = result_path.read_bytes()

                    st.session_state["graded_video"] = result_bytes

                    st.success("Junshoku grading completed!")

            except Exception as e:

                st.error(f"Processing failed: {e}")

            finally:

                source_path.unlink(missing_ok=True)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

if "graded_video" in st.session_state:

    st.divider()

    st.header("Final Output")

    st.video(st.session_state["graded_video"])

    st.download_button(
        label="Download Graded Video",
        data=st.session_state["graded_video"],
        file_name="junshoku_graded.mp4",
        mime="video/mp4",
        type="primary",
    )

st.divider()

st.caption(
    "JUNSHOKU STUDIO | Spatially Adaptive Cinematic Colour Grading"
)
