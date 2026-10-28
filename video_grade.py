
from pathlib import Path
import subprocess
import shutil
import time

import cv2
import numpy as np
import torch
from PIL import Image

from Models.VGGBase import PredictModel
from lut_train import apply_lut_bank


# ============================================================
# CONFIGURATION — CHANGE THESE VALUES
# ============================================================

ROOT = Path(r"F:\Junshoku")

VIDEO_PATH = ROOT / "input_video.mp4"
REFERENCE_PATH = ROOT / "reference.jpg"

CHECKPOINT_PATH = ROOT / "checkpoints_tst2k" / "best.pt"
OUTPUT_PATH = ROOT / "Outputs" / "graded_video.mp4"

# Maximum number of seconds to process.
# Example: 10 = process the first 10 seconds.
# Set to None to process the entire video.
DURATION_SECONDS = 3

IMAGE_SIZE = 256
LUT_SIZE = 16
NUM_LUTS = 9

# Set True to preserve the source audio when FFmpeg is installed.
PRESERVE_AUDIO = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def load_rgb_image(path, size):
    """Load an image, resize it and convert it to a [1, 3, H, W] tensor."""
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    return tensor.to(DEVICE)


def load_checkpoint(model, checkpoint_path):
    """Load common checkpoint formats."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        state_dict = None

        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            # Supports checkpoints containing only the model state dict.
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle checkpoints saved from DataParallel.
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {
            k.removeprefix("module."): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model


def grade_frame(frame_bgr, reference_tensor, model):
    """
    Grade one OpenCV BGR frame.
    Returns a BGR frame at the original resolution.
    """
    original_height, original_width = frame_bgr.shape[:2]

    # OpenCV BGR -> RGB.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize for the trained model.
    resized = cv2.resize(
        frame_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    source_array = resized.astype(np.float32) / 255.0
    source_tensor = (
        torch.from_numpy(source_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):
            luts, weights = model(source_tensor, reference_tensor)

        # Keep LUT application in float32 for stability.
        transformed = apply_lut_bank(
            source_tensor.float(),
            luts.float()
        )

        # Blend the LUT outputs using the predicted spatial weights.
        graded = (
            transformed * weights.unsqueeze(2).float()
        ).sum(dim=1).clamp(0.0, 1.0)

    # Tensor [1, 3, H, W] -> RGB uint8 image.
    graded_array = (
        graded[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)

    # Resize back to the video's original resolution.
    graded_array = cv2.resize(
        graded_array,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC
    )

    # RGB -> BGR for OpenCV VideoWriter.
    return cv2.cvtColor(graded_array, cv2.COLOR_RGB2BGR)


# ============================================================
# MAIN VIDEO PIPELINE
# ============================================================

def main():
    start_time = time.time()

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Reference image not found: {REFERENCE_PATH}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print(f"Device: {DEVICE}")
    print("Loading Junshoku model...")

    model = PredictModel(
        lut_size=LUT_SIZE,
        num_luts=NUM_LUTS
    ).to(DEVICE)

    model = load_checkpoint(model, CHECKPOINT_PATH)

    reference_tensor = load_rgb_image(
        REFERENCE_PATH,
        IMAGE_SIZE
    )

    print("Model and reference image loaded.")

    # --------------------------------------------------------
    # Open input video
    # --------------------------------------------------------
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read valid video metadata.")

    video_duration = total_frames / fps

    if DURATION_SECONDS is None:
        frames_to_process = total_frames
    else:
        if DURATION_SECONDS <= 0:
            cap.release()
            raise ValueError("DURATION_SECONDS must be positive or None.")

        frames_to_process = min(
            total_frames,
            int(round(DURATION_SECONDS * fps))
        )

    actual_duration = frames_to_process / fps

    print("\nInput video:")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps:.3f}")
    print(f"  Total duration: {video_duration:.2f} seconds")
    print(f"  Processing: {actual_duration:.2f} seconds")
    print(f"  Frames to process: {frames_to_process}")

    # Use a temporary silent video; audio is muxed afterward.
    silent_path = OUTPUT_PATH.with_name(
        OUTPUT_PATH.stem + "_silent.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(silent_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {silent_path}")

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------
    print("\nGrading video frames...")

    processed = 0

    try:
        while processed < frames_to_process:
            success, frame = cap.read()

            if not success:
                break

            graded_frame = grade_frame(
                frame,
                reference_tensor,
                model
            )

            writer.write(graded_frame)
            processed += 1

            if processed % max(1, int(fps)) == 0 or processed == frames_to_process:
                percentage = 100.0 * processed / frames_to_process
                print(
                    f"\rProgress: {processed}/{frames_to_process} "
                    f"frames ({percentage:.1f}%)",
                    end="",
                    flush=True
                )
    finally:
        cap.release()
        writer.release()

    print()

    if processed == 0:
        raise RuntimeError("No frames were processed.")

    # --------------------------------------------------------
    # Preserve audio using FFmpeg, if available
    # --------------------------------------------------------
    ffmpeg = shutil.which("ffmpeg")

    if PRESERVE_AUDIO and ffmpeg:
        print("Combining graded video with original audio...")

        command = [
            ffmpeg,
            "-y",
            "-i", str(silent_path),
            "-i", str(VIDEO_PATH),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", f"{processed / fps:.6f}",
            "-movflags", "+faststart",
            str(OUTPUT_PATH)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("\nFFmpeg failed:")
            print(result.stderr[-3000:])
            print("Keeping the silent graded video instead.")
            shutil.copy2(silent_path, OUTPUT_PATH)
        else:
            silent_path.unlink(missing_ok=True)

    else:
        shutil.copy2(silent_path, OUTPUT_PATH)
        silent_path.unlink(missing_ok=True)

        if PRESERVE_AUDIO and not ffmpeg:
            print(
                "WARNING: FFmpeg was not found. "
                "The output will not contain audio."
            )

    elapsed = time.time() - start_time

    print("\n" + "=" * 55)
    print("VIDEO GRADING COMPLETE")
    print("=" * 55)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames processed: {processed}")
    print(f"Output duration: {processed / fps:.2f} seconds")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Average processing speed: {processed / elapsed:.2f} FPS")


if __name__ == "__main__":
    main()
