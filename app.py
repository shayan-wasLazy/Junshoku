from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from io import BytesIO

from node_graph.model import NodeGraph
from node_graph.graph import apply_graph


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "node_graph" / "configs" / "default.yaml"

# Windows is case-insensitive, but the repository may use either spelling.
CHECKPOINT_CANDIDATES = [
    ROOT / "Checkpoints" / "node_graph_finetuned" / "best.pt",
    ROOT / "checkpoints" / "node_graph_finetuned" / "best.pt",
]
CHECKPOINT_PATH = next(
    (path for path in CHECKPOINT_CANDIDATES if path.is_file()),
    CHECKPOINT_CANDIDATES[0],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SIDE = 512


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Junshoku Studio",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #101114; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #f4f4f5; }
    .stButton button { width: 100%; border-radius: 8px; min-height: 45px; }
</style>
""", unsafe_allow_html=True)

st.title("JUNSHOKU")
st.caption("AI-powered spatially adaptive cinematic colour grading")
st.divider()


# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(checkpoint_path: str):
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        expected = "\n".join(str(p) for p in CHECKPOINT_CANDIDATES)
        raise FileNotFoundError(
            "Trained checkpoint not found. Expected one of:\n" + expected
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    # The checkpoint contains the VGG encoder weights; do not download them.
    model_cfg = dict(cfg)
    model_cfg["model"] = dict(cfg["model"])
    model_cfg["model"]["pretrained"] = False

    model = NodeGraph(model_cfg).to(DEVICE)
    state = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


# --------------------------------------------------
# IMAGE / TENSOR UTILITIES
# --------------------------------------------------

def resize_rgb(array: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    height, width = array.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return array


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"))
    array = resize_rgb(array)
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = resize_rgb(rgb)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .cpu()
        .numpy()
    )
    return np.round(array * 255).astype(np.uint8)


def grade_image(
    model: NodeGraph,
    cfg: dict,
    source: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    # NodeGraph concatenates source/reference feature maps at each VGG scale.
    # Match their spatial dimensions before the forward pass.
    reference = F.interpolate(
        reference,
        size=source.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    with torch.inference_mode():
        prediction = model(source, reference)
        return apply_graph(source, prediction, cfg)


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def process_video(
    model,
    cfg,
    video_path: Path,
    reference: torch.Tensor,
    start_time: float,
    end_time: float,
    output_path: Path,
    progress_bar,
    status_text,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_time * fps))
    end_frame = min(total_frames, int(end_time * fps))
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

    total_to_process = end_frame - start_frame
    processed = 0

    try:
        with torch.inference_mode():
            while processed < total_to_process:
                success, frame = cap.read()
                if not success:
                    break

                source = frame_to_tensor(frame)
                graded = grade_image(model, cfg, source, reference)
                output = tensor_to_rgb(graded)
                output = cv2.resize(
                    output, (width, height), interpolation=cv2.INTER_LINEAR
                )
                writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

                processed += 1
                progress_bar.progress(processed / total_to_process)
                status_text.write(
                    f"Processing frame {processed}/{total_to_process}"
                )
    finally:
        cap.release()
        writer.release()

    if processed == 0:
        raise RuntimeError("No video frames were processed.")
    return processed


def mux_audio(
    source_path: Path,
    graded_path: Path,
    final_path: Path,
    start_time: float,
):
    # Keep the graded video even if the source has no audio or FFmpeg is absent.
    cap = cv2.VideoCapture(str(graded_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps

    command = [
        "ffmpeg", "-y",
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

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return final_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        return graded_path


# --------------------------------------------------
# LOAD MODEL AND SIDEBAR
# --------------------------------------------------

try:
    with st.spinner("Loading Junshoku model..."):
        model, cfg = load_model(str(CHECKPOINT_PATH))
    st.success(f"Model loaded | Device: {DEVICE}")
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

with st.sidebar:
    st.header("System")
    st.write(f"**Device:** {DEVICE}")
    if DEVICE.type == "cuda":
        st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
    st.divider()
    st.header("About Junshoku")
    st.write(
        "VGG16 feature encoder with global parameter prediction and "
        "three independent spatial context maps."
    )
    st.write(
        "Grading nodes: exposure/white balance → tonal contrast → creative colour."
    )


# --------------------------------------------------
# INPUT MODE AND REFERENCE
# --------------------------------------------------

mode = st.radio(
    "Choose input type",
    ["Image", "Video"],
    horizontal=True,
    key="input_mode",
)

left, right = st.columns(2)

with left:
    st.subheader(f"Source {mode}")
    if mode == "Image":
        source_upload = st.file_uploader(
            "Upload source image",
            type=["jpg", "jpeg", "png", "webp"],
            key="source_image",
        )
    else:
        source_upload = st.file_uploader(
            "Upload source video",
            type=["mp4", "mov", "avi", "mkv"],
            key="source_video",
        )

with right:
    st.subheader("Reference Image")
    reference_upload = st.file_uploader(
        "Upload target colour look",
        type=["jpg", "jpeg", "png", "webp"],
        key="reference_image",
    )

reference_image = None
reference_tensor = None

if reference_upload is not None:
    try:
        reference_image = Image.open(reference_upload).convert("RGB")
        reference_tensor = pil_to_tensor(reference_image)
        st.image(
            reference_image,
            caption="Reference look",
            use_container_width=True,
        )
    except Exception as error:
        st.error(f"Could not read reference image: {error}")


# --------------------------------------------------
# IMAGE MODE
# --------------------------------------------------

if mode == "Image":
    if source_upload is not None:
        try:
            source_image = Image.open(source_upload).convert("RGB")
            source_tensor = pil_to_tensor(source_image)

            st.subheader("Source Preview")
            st.image(source_image, use_container_width=True)

            if reference_tensor is not None:
                if st.button("GENERATE GRADED IMAGE", type="primary"):
                    with st.spinner("Applying Junshoku colour grade..."):
                        result = grade_image(
                            model, cfg, source_tensor, reference_tensor
                        )
                        result_rgb = tensor_to_rgb(result)

                    st.session_state["graded_image"] = result_rgb
                    st.success("Image grading completed.")

        except Exception as error:
            st.error(f"Could not read source image: {error}")

    if "graded_image" in st.session_state:
        st.divider()
        st.subheader("Before / After Comparison")
        before, after = st.columns(2)
        with before:
            st.image(source_image, caption="Original", use_container_width=True)
        with after:
            st.image(
                st.session_state["graded_image"],
                caption="Junshoku graded",
                use_container_width=True,
            )

        output_image = Image.fromarray(st.session_state["graded_image"])
        buffer = BytesIO()
        output_image.save(buffer, format="PNG")

        st.download_button(
            "DOWNLOAD GRADED IMAGE",
            data=buffer.getvalue(),
            file_name="junshoku_graded.png",
            mime="image/png",
            type="primary",
        )


# --------------------------------------------------
# VIDEO MODE
# --------------------------------------------------

else:
    if source_upload is not None:
        video_bytes = source_upload.getvalue()
        st.video(video_bytes)

        with tempfile.NamedTemporaryFile(
            suffix=Path(source_upload.name).suffix or ".mp4",
            delete=False,
        ) as temp_video:
            temp_video.write(video_bytes)
            source_path = Path(temp_video.name)

        cap = cv2.VideoCapture(str(source_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        st.caption(
            f"Duration: {duration:.2f}s | FPS: {fps:.2f} | "
            f"Frames: {total_frames}"
        )

        start_time, end_time = 0.0, duration
        if duration > 0:
            st.subheader("Timeline / Trim")
            start_time = st.number_input(
                "Start time (seconds)",
                min_value=0.0,
                max_value=float(duration),
                value=0.0,
                step=0.5,
                key="trim_start",
            )
            end_time = st.number_input(
                "End time (seconds)",
                min_value=0.1,
                max_value=float(duration),
                value=float(duration),
                step=0.5,
                key="trim_end",
            )

        if reference_tensor is not None:
            if st.button("GENERATE GRADED VIDEO", type="primary"):
                progress = st.progress(0)
                status = st.empty()

                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir = Path(temp_dir)
                        graded_path = temp_dir / "graded.mp4"
                        final_path = temp_dir / "junshoku_final.mp4"

                        with st.spinner("Grading video frames..."):
                            processed = process_video(
                                model,
                                cfg,
                                source_path,
                                reference_tensor,
                                start_time,
                                end_time,
                                graded_path,
                                progress,
                                status,
                            )

                        status.success(f"Processed {processed} frames.")

                        result_path = mux_audio(
                            source_path,
                            graded_path,
                            final_path,
                            start_time,
                        )

                        st.session_state["graded_video"] = result_path.read_bytes()

                    st.success("Video grading completed.")
                except Exception as error:
                    st.error(f"Video processing failed: {error}")
                finally:
                    source_path.unlink(missing_ok=True)

    if "graded_video" in st.session_state:
        st.divider()
        st.subheader("Graded Video")
        st.video(st.session_state["graded_video"])
        st.download_button(
            "DOWNLOAD GRADED VIDEO",
            data=st.session_state["graded_video"],
            file_name="junshoku_graded.mp4",
            mime="video/mp4",
            type="primary",
        )


st.divider()
st.caption("JUNSHOKU STUDIO | Spatially Adaptive Cinematic Colour Grading")
