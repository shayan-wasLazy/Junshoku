# Junshoku Studio

## Free deployment architecture

- **Frontend:** GitHub Pages (free static hosting).
- **Inference API:** run FastAPI on your own Windows PC.
- **Public API URL:** Cloudflare Quick Tunnel (free temporary URL).
- Your PC must stay powered on, connected to the internet, and running the model server while people use the app. Quick Tunnel URLs change when restarted.

This avoids paid cloud compute. It is not an always-on hosted AI service.

## 1. Deploy the frontend on GitHub Pages

1. Push this branch to GitHub.
2. Open repository **Settings → Pages**.
3. Under **Build and deployment → Source**, select **GitHub Actions**.
4. Open **Actions** and wait for **Deploy Junshoku Studio frontend** to finish.
5. The site URL will be shown in the workflow deployment summary. It normally has the form `https://OWNER.github.io/REPOSITORY/`.

The workflow publishes only the static files in `webapp/static/`. It does not deploy the Python model.

## 2. Run the backend on your Windows PC

Use the repository root in PowerShell. Activate the environment that contains the trained model's PyTorch dependencies.

Install the web dependencies:

```powershell
pip install -r webapp/requirements.txt
```

Install FFmpeg and make sure `ffmpeg` is on PATH. The API uses it to mux audio and encode the final MP4.

Place your trained checkpoint at:

```text
checkpoints/node_graph_finetuned/best.pt
```

Start the API:

```powershell
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Check it locally at `http://127.0.0.1:8000/api/health`. The response should report `"model_checkpoint": true`.

## 3. Expose the backend for free

Install Cloudflare Tunnel on Windows:

```powershell
winget install --id Cloudflare.cloudflared
```

Open a second PowerShell window and run:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Cloudflare prints a public URL similar to `https://random-words.trycloudflare.com`. Keep this terminal open.

## 4. Connect the Pages frontend to your API

Edit `webapp/static/config.js` and set the URL printed by Cloudflare:

```javascript
window.JUNSHOKU_API_BASE = "https://YOUR-QUICK-TUNNEL.trycloudflare.com";
```

Commit and push the change to `junshoku-editor`. The GitHub Pages workflow will redeploy the frontend. Do not add a trailing slash to the URL.

The backend currently allows cross-origin requests by default so the Pages frontend can reach it. For a restricted deployment, set `JUNSHOKU_CORS_ORIGINS` to your exact Pages origin.

## Important limitations

- Quick Tunnels are temporary. Restarting the tunnel can give you a new URL; update `config.js` and push again.
- The computer hosting the model must stay on. If it sleeps or shuts down, grading and LUT export stop working.
- Large videos take time and upload through your home internet connection.
- The free frontend hosting does not provide model inference.
- A 3D `.cube` LUT is a global approximation. The model's spatial context maps are averaged because a standard 3D LUT cannot represent spatially varying per-pixel masks.
- Never expose a production inference API without authentication, upload limits, and rate limiting. The free demo configuration is intended for personal/testing use.

## Local-only alternative

Run the API and open `http://127.0.0.1:8000`; the same app works without GitHub Pages or Cloudflare.
