from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from PIL import Image

from node_graph.graph import apply_graph
from node_graph.model import NodeGraph
from node_graph.nodes import creative_colour, exposure_wb, masked_node, tonal_contrast
from node_graph.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
STATIC = WEB / "static"
CONFIG_PATH = Path(os.getenv("JUNSHOKU_CONFIG", ROOT / "node_graph/configs/default.yaml"))
CHECKPOINT = Path(os.getenv("JUNSHOKU_CHECKPOINT", ROOT / "checkpoints/node_graph_finetuned/best.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Junshoku Studio API", version="1.0.0")
cors_origins = [origin.strip() for origin in os.getenv("JUNSHOKU_CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_model = None
_cfg = None


def model_and_config():
    global _model, _cfg
    if _cfg is None:
        _cfg = load_config(str(CONFIG_PATH))
    if _model is None:
        if not CHECKPOINT.is_file():
            raise HTTPException(503, "Model checkpoint is not installed on this server yet.")
        try:
            model_cfg = dict(_cfg)
            model_cfg["model"] = dict(_cfg["model"])
            # The complete encoder weights are in the checkpoint; avoid a runtime VGG download.
            model_cfg["model"]["pretrained"] = False
            _model = NodeGraph(model_cfg).to(DEVICE)
            state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
            _model.load_state_dict(state["model"])
            _model.eval()
        except Exception as exc:
            _model = None
            raise HTTPException(500, f"Could not load model checkpoint: {exc}") from exc
    return _model, _cfg


def read_rgb(path: Path, max_side: int = 512):
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def read_video_frame(path: Path, max_side: int = 512):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "Could not read a frame from the uploaded video.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def infer_tensor(model, cfg, source, reference):
    with torch.inference_mode():
        prediction = model(source, reference)
        return apply_graph(source, prediction, cfg), prediction


async def save_upload(upload: UploadFile, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{upload.filename or 'Upload'} is empty.")
    destination.write_bytes(data)
    return destination


def run_ffmpeg(args: list[str]):
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True, check=True)
        return proc
    except FileNotFoundError as exc:
        raise HTTPException(500, "FFmpeg is not installed on the server.") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(400, exc.stderr[-1800:] or "FFmpeg could not process this media.") from exc


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_checkpoint": CHECKPOINT.is_file(), "device": str(DEVICE), "ffmpeg": shutil.which("ffmpeg") is not None}


@app.post("/api/grade")
async def grade_video(
    video: UploadFile = File(...),
    reference: UploadFile = File(...),
    trim_start: float = Form(0),
    trim_end: float = Form(-1),
    cuts: str = Form("[]"),
):
    model, cfg = model_and_config()
    try:
        cut_ranges = json.loads(cuts)
        if not isinstance(cut_ranges, list):
            raise ValueError("cuts must be a list")
    except Exception as exc:
        raise HTTPException(400, f"Invalid cut list: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="junshoku_") as tmp:
        work = Path(tmp)
        source_path = await save_upload(video, work / "source_video")
        ref_path = await save_upload(reference, work / "reference_image")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise HTTPException(400, "Could not read the uploaded video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        start = max(0.0, float(trim_start))
        end = duration if float(trim_end) < 0 else min(duration, float(trim_end))
        if end <= start:
            cap.release()
            raise HTTPException(400, "Trim end must be later than trim start.")
        excluded = []
        for item in cut_ranges:
            a, b = max(start, float(item[0])), min(end, float(item[1]))
            if b > a:
                excluded.append((a, b))
        excluded.sort()
        keep = []
        cursor = start
        for a, b in excluded:
            if a > cursor:
                keep.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < end:
            keep.append((cursor, end))
        if not keep:
            cap.release()
            raise HTTPException(400, "Your cuts remove the entire selected clip.")

        reference_tensor = read_rgb(ref_path)
        writer_path = work / "graded_video.mp4"
        writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise HTTPException(500, "Could not initialize video encoder.")
        frame_index = 0
        written = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_step = max(1, frame_count // 100)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if not any(a <= timestamp < b for a, b in keep):
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(1.0, 512 / max(h, w))
            if scale < 1:
                rgb_small = cv2.resize(rgb, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
            else:
                rgb_small = rgb
            tensor = torch.from_numpy(rgb_small.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                graded, _ = infer_tensor(model, cfg, tensor, reference_tensor)
            out = (graded.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            if out.shape[:2] != (h, w):
                out = cv2.resize(out, (w,h), interpolation=cv2.INTER_LINEAR)
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            written += 1
        cap.release()
        writer.release()
        if written == 0:
            raise HTTPException(400, "No video frames fell inside the selected edit range.")

        # Keep the original audio and apply the same trim/cut segments.
        final_path = work / "junshoku_graded.mp4"
        filter_parts = []
        concat_inputs = []
        for i, (a, b) in enumerate(keep):
            filter_parts.append(f"[0:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[a{i}]")
        audio_filter = ";".join(filter_parts + [f"{''.join(concat_inputs)}concat=n={len(keep)}:v=0:a=1[aout]"])
        try:
            run_ffmpeg(["-y", "-i", str(source_path), "-i", str(writer_path), "-filter_complex", audio_filter,
                        "-map", "1:v:0", "-map", "[aout]", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(final_path)])
        except HTTPException:
            # Silent videos are valid; mux the graded video without an audio stream.
            run_ffmpeg(["-y", "-i", str(writer_path), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-movflags", "+faststart", str(final_path)])
        delivery_path = Path(tempfile.gettempdir()) / f"junshoku_{uuid.uuid4().hex}.mp4"
        shutil.copy2(final_path, delivery_path)
        return FileResponse(delivery_path, media_type="video/mp4", filename="junshoku_graded.mp4",
                            background=BackgroundTask(delivery_path.unlink, missing_ok=True))


@app.post("/api/lut")
async def export_lut(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    size: int = Form(33),
):
    if size not in (17, 33):
        raise HTTPException(400, "LUT size must be 17 or 33.")
    model, cfg = model_and_config()
    with tempfile.TemporaryDirectory(prefix="junshoku_lut_") as tmp:
        work = Path(tmp)
        src_path = await save_upload(source, work / "source_image")
        ref_path = await save_upload(reference, work / "reference_image")
        src = read_video_frame(src_path, max_side=512)
        ref = read_rgb(ref_path, max_side=512)
        _, prediction = infer_tensor(model, cfg, src, ref)
        params = [p.detach() for p in prediction["params"]]
        gates = [m.detach().mean(dim=(2,3), keepdim=True) for m in prediction["maps"]]
        axis = torch.linspace(0, 1, size, device=DEVICE)
        b, g, r = torch.meshgrid(axis, axis, axis, indexing="ij")
        grid = torch.stack([r, g, b], dim=-1).reshape(-1, 3)
        transformed_chunks = []
        for chunk in grid.split(4096):
            x = chunk.T.reshape(1, 3, 1, -1)
            # Expand each predicted global control across the colour cube.
            p = [v.expand(x.shape[0], -1) for v in params]
            m = [v.expand(1, 1, 1, x.shape[-1]) for v in gates]
            y = masked_node(x, exposure_wb(x, p[0], cfg), m[0])
            y = masked_node(y, tonal_contrast(y, p[1], cfg), m[1])
            y = masked_node(y, creative_colour(y, p[2], cfg), m[2])
            transformed_chunks.append(y.squeeze(0).permute(1,2,0).reshape(-1,3).cpu().numpy())
        values = np.concatenate(transformed_chunks, axis=0)
        lut_path = work / "junshoku_grade.cube"
        with lut_path.open("w", encoding="utf-8") as f:
            f.write('TITLE "Junshoku spatially averaged grade"\n')
            f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
            for rgb in values:
                f.write(f"{rgb[0]:.7f} {rgb[1]:.7f} {rgb[2]:.7f}\n")
        delivery_path = Path(tempfile.gettempdir()) / f"junshoku_{uuid.uuid4().hex}.cube"
        shutil.copy2(lut_path, delivery_path)
        return FileResponse(delivery_path, media_type="application/octet-stream", filename="junshoku_grade.cube",
                            background=BackgroundTask(delivery_path.unlink, missing_ok=True))
