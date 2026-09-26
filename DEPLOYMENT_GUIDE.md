# DermaAI: Production Deployment & Operations Guide

## Overview

DermaAI is a production-grade academic research demonstration system for 7-class dermoscopic skin lesion classification on the retrospective HAM10000 dataset. It couples a frozen dual-model probability ensemble (85% Experiment 8 + 15% Experiment 13, MobileNetV3-Large + CBAM, 4,326,297 parameters each) with real-time dual Grad-CAM attention explainability.

> [!IMPORTANT]
> **RESEARCH USE ONLY NOTICE**:
> DermaAI is an academic research demonstrator. It is NOT a certified medical diagnostic device, has NOT undergone clinical trials, and must NEVER replace professional clinical judgment, dermatoscopic evaluation, biopsy, or physician-patient decision-making.

---

## Architecture Topology

```
   ┌────────────────────────────────────────────────────────┐
   │                  Next.js 16 Web UI                     │
   │      (React 19, Tailwind CSS 4, Lucide Icons)          │
   └──────────┬─────────────────────────────────┬───────────┘
              │ Direct API Calls                │ Proxy Fallback
              ▼                                 ▼
   ┌────────────────────────────────────────────────────────┐
   │             FastAPI Production Backend                 │
   │               (Uvicorn, Python 3.11)                   │
   │                                                        │
   │  Endpoints:                                            │
   │   - GET  /api/health       (Runtime health & status)   │
   │   - GET  /api/model-info   (Architecture specifications│
   │   - POST /api/predict      (Ensemble + Grad-CAM)       │
   └────────────────────────┬───────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌───────────────────────┐   ┌────────────────────────┐
   │ Experiment 8 (Frozen) │   │ Experiment 13 (Frozen) │
   │  4,326,297 parameters │   │  4,326,297 parameters  │
   │      Weight: 0.85     │   │      Weight: 0.15      │
   └───────────────────────┘   └────────────────────────┘
```

---

## Prerequisites & System Requirements

- **Python**: 3.10+ (tested on Python 3.11)
- **Node.js**: 18.x or 20.x+ (tested on Node v20/v22)
- **RAM**: Minimum 2 GB RAM (models use ~150 MB RAM in eval mode)
- **Disk Space**: ~250 MB for checkpoints and code (excluding raw HAM10000 images)
- **Hardware**: CPU-native (GPU optional, auto-detected via `DERMAAI_DEVICE`)

---

## Environment Variables Reference

| Variable | Target | Default | Description |
|---|---|---|---|
| `PORT` | Backend | `8000` | Port for the Uvicorn HTTP server |
| `DERMAAI_DEVICE` | Backend | `cpu` | PyTorch execution device (`cpu` or `cuda`) |
| `DERMAAI_ALLOWED_ORIGINS` | Backend | `*` | Comma-separated list of allowed CORS origins |
| `DERMAAI_MODELS_DIR` | Backend | `ai/models` | Custom absolute or relative path to models directory |
| `DERMAAI_EXP8_CHECKPOINT` | Backend | `ai/models/experiment8_best_model.pt` | Path override for Experiment 8 checkpoint |
| `DERMAAI_EXP13_CHECKPOINT` | Backend | `ai/models/experiment13_best_model.pt` | Path override for Experiment 13 checkpoint |
| `DERMAAI_MAX_UPLOAD_MB` | Backend | `10` | Maximum allowable image upload payload in megabytes |
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Public FastAPI backend URL for client requests |
| `DERMAAI_BACKEND_URL` | Frontend | `http://localhost:8000` | Internal server-to-server URL for Next.js API proxy |

---

## 1. Local Development Setup

### Step 1: Backend Setup
```bash
# From workspace root
python -m venv ai/.venv
# Windows:
ai\.venv\Scripts\activate
# Linux/macOS:
source ai/.venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt

# Run automated smoke test
python smoke_test_dermaai.py

# Launch FastAPI server
python backend/main.py
# Server starts at http://localhost:8000
```

### Step 2: Frontend Setup
```bash
# In a separate terminal
npm install
npm run dev
# Web application available at http://localhost:3000
```

---

## 2. Docker Container Deployment

The backend includes a minimal, multi-platform Docker container definition (`Dockerfile`).

### Build Container
```bash
docker build -t dermaai-backend:latest .
```

### Run Container
```bash
docker run -d \
  --name dermaai-service \
  -p 8000:8000 \
  -e PORT=8000 \
  -e DERMAAI_DEVICE=cpu \
  -e DERMAAI_ALLOWED_ORIGINS="*" \
  --restart unless-stopped \
  dermaai-backend:latest
```

### Check Container Health
```bash
curl http://localhost:8000/api/health
```

---

## 3. Cloud Deployment: Render (Blueprint)

DermaAI includes a turnkey `render.yaml` configuration that deploys both services:

1. Connect your Git repository to [Render](https://render.com).
2. Choose **New > Blueprint**.
3. Select this repository. Render automatically reads `render.yaml` and spins up:
   - `dermaai-backend`: Docker web service with automatic healthcheck on `/api/health`.
   - `dermaai-frontend`: Node.js web service running Next.js with `NEXT_PUBLIC_API_URL` auto-linked to the backend.

---

## 4. Cloud Deployment: Vercel (Frontend) + Standalone Backend

If hosting the frontend on Vercel:

1. Import repository into [Vercel](https://vercel.com).
2. Set Environment Variables in Vercel project settings:
   - `NEXT_PUBLIC_API_URL`: Your hosted FastAPI backend URL (e.g. `https://dermaai-backend.onrender.com`).
   - `DERMAAI_BACKEND_URL`: Same hosted backend URL.
3. Deploy. The Next.js frontend will communicate with the hosted backend and provide fallback proxying.

---

## 5. Verification & Acceptance Testing

Before going live at a presentation or hackathon, execute the full test battery:

```bash
# 1. Rapid synthetic smoke test (7/7 pass)
python smoke_test_dermaai.py

# 2. Production contract & frozen checkpoint audit (7/7 pass)
python verify_production_dermaai.py

# 3. Acceptance verification (13/13 pass)
python verify_dermaai_demo.py

# 4. Backend unit test suite (9/9 pass)
python backend/test_demo.py
```

---

## 6. Troubleshooting Common Issues

### Issue 1: CORS Error in Browser Console
- **Symptom**: `Access to fetch at '...' from origin '...' has been blocked by CORS policy`
- **Solution**: Set `DERMAAI_ALLOWED_ORIGINS="https://yourfrontend.domain.com"` or `"*"` on the backend.

### Issue 2: Checkpoint SHA-256 Mismatch
- **Symptom**: `Model loading failed: SHA256 checksum mismatch`
- **Solution**: Verify that `ai/models/experiment8_best_model.pt` (SHA: `ccc579cb2087545f...`) and `ai/models/experiment13_best_model.pt` (SHA: `9fab1b1218fddc86...`) were downloaded with Git LFS or intact binary transfer.

### Issue 3: Backend Times Out on Cold Start
- **Symptom**: First request takes 15–30 seconds on free tier cloud providers.
- **Solution**: The Next.js UI features progressive multi-stage feedback (`Loading image...`, `Running ensemble...`, `Generating Grad-CAM...`) to inform the user during model warm-up.
