# Junshoku Studio web app

The web editor is served by FastAPI and uses the existing `node_graph` model for inference.

## Features
- Upload a source video and a reference still.
- Preview footage in the browser, set in/out points, and define cut ranges.
- Export a graded MP4 with the selected trims/cuts. The server attempts to preserve and cut the original audio.
- Export a 17³ or 33³ `.cube` LUT.

## Model checkpoint
The server requires the trained checkpoint at:
`checkpoints/node_graph_finetuned/best.pt`

Override with `JUNSHOKU_CHECKPOINT`. The checkpoint is intentionally not committed to Git because trained PyTorch weights are large and may contain experiment-specific artifacts. Without it, the UI loads but inference endpoints return HTTP 503.

## Run locally (PowerShell)
From the repository root, activate the project environment and install the web dependencies:

```powershell
pip install -r webapp/requirements.txt
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

FFmpeg must be installed and available on PATH. Open `http://127.0.0.1:8000`.

## LUT limitation
A standard 3D `.cube` LUT maps RGB values to RGB values; it cannot represent spatially varying context maps. Junshoku therefore averages the three predicted context maps when sampling a portable global LUT. The downloaded LUT is a global approximation of the grade, not an exact representation of the spatially adaptive video result.

## Deployment
The included Dockerfile is intended for a Render Docker web service using the repository root as build context and `webapp/Dockerfile` as the Dockerfile path. The service needs FFmpeg and enough RAM for CPU PyTorch inference. The checkpoint must be supplied through a persistent disk or a secure external checkpoint location before inference can be enabled. Do not deploy the free 512 MB instance for model inference; it is not sized for this PyTorch model.
