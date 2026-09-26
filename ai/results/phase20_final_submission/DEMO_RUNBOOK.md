# DermaAI: Live Demo Runbook & Rehearsal Guide

This operational runbook provides step-by-step instructions for conducting a flawless live demonstration of DermaAI during hackathons, conference presentations, and technical defenses.

---

## 1. Pre-Flight Setup & Environment Check (10 Minutes Prior)

### Step 1: Launch Backend Server
In Terminal 1:
```bash
# Activate virtual environment
ai\.venv\Scripts\activate       # Windows
# source ai/.venv/bin/activate  # Linux/macOS

# Start FastAPI Uvicorn server on port 8000
python backend/main.py
```
*Expected log output:*
```
[DermaAI] Initializing Frozen Model Service...
[DermaAI] Loading Experiment 8 from .../ai/models/experiment8_best_model.pt...
[DermaAI] Exp8 loaded successfully (Params: 4,326,297)
[DermaAI] Loading Experiment 13 from .../ai/models/experiment13_best_model.pt...
[DermaAI] Exp13 loaded successfully (Params: 4,326,297)
[DermaAI] Ensemble = 0.85 (Exp8) / 0.15 (Exp13)
[DermaAI] Ready for inference.
Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Verify Health & Metadata Endpoints
In a browser or via cURL:
- Health check: `http://localhost:8000/api/health`  
  *Expect: `{"status":"ok","models_loaded":true,"input_size":"224x224",...}`*
- Model metadata: `http://localhost:8000/api/model-info`  
  *Expect: `{"architecture":"DermaAI_MobileNetV3","parameters_per_model":4326297,...}`*

### Step 3: Launch Frontend Web App
In Terminal 2:
```bash
npm run dev
# Or for production mode:
# npm run build && npm run start
```
*Expected output: Next.js server ready at `http://localhost:3000`.*

### Step 4: Open Browser & Validate UI
1. Navigate to `http://localhost:3000`.
2. Verify that the top research banner is visible:
   *`Academic Research Demonstrator: Strictly for image-level classification on retrospective HAM10000 data. Not for clinical diagnosis.`*
3. Look at the status indicator in the top navbar: it should display **"Online"** or show model specifications modal.

---

## 2. Dedicated Demonstration Image Policy

> [!IMPORTANT]
> **STRICT ZERO TEST-SET LEAKAGE POLICY**:
> You MUST NEVER use any of the 988 held-out project test split images during a live demonstration.

### Recommended Demo Assets:
1. **Public Domain Synthetic Asset**:
   - Location: `ai/dataset/demo_sample_non_test.jpg` (or any public ISIC archive image strictly verified outside the project test split).
   - Provenance: External dermoscopic sample or synthetic test sample.
2. **Synthetic In-Memory Demo Image**:
   - The test script `smoke_test_dermaai.py` can generate a synthetic RGB lesion on demand.

---

## 3. Step-by-Step Live Demonstration Walkthrough

| Step # | Screen Action | Explanatory Commentary |
|---|---|---|
| **1** | Open DermaAI homepage (`http://localhost:3000`) | Point out the clean UI, the top research banner, and the 85/15 ensemble badge. |
| **2** | Highlight Architecture Modal | Click **"Model Architecture & Specs"** to show 4,326,297 parameters per model and verified SHA-256 hashes. |
| **3** | Ingest Image | Drag and drop the non-test demonstration lesion into the central dropzone. |
| **4** | Initiate Inference | Click the cyan **"Analyze Lesion Image"** button. |
| **5** | Observe Multi-Stage Progress | Point out the live pipeline feedback: *Loading image... $\to$ Running ensemble... $\to$ Generating Grad-CAM...* |
| **6** | Reveal Predicted Class & Confidence | Highlight the output card: Predicted label (e.g., *Melanocytic Nevus*), confidence percentage, and ensemble weighting. |
| **7** | Examine Top-3 Ranking | Walk through the top-3 ranked predictions. |
| **8** | Review 7-Class Distribution | Show the complete horizontal probability distribution bars, noting that sum is exactly 1.00. |
| **9** | Inspect Dual Grad-CAM Heatmaps | Scroll to the Grad-CAM panel: compare Experiment 8 (global context) vs Experiment 13 (focal pigment network). |
| **10** | Reiterate Argmax Equivalence | Reference the Experiment 15 note: threshold tuning $\theta^*=0.50$ verified natural argmax optimality. |
| **11** | Review Frozen Test Benchmark | Scroll to the benchmark card: Accuracy 80.57%, Balanced Accuracy 75.69%, Macro F1 70.51%, Melanoma F1 52.43% ($N=988$). |
| **12** | Emphasize Research Disclaimer | Conclude by reading the mandatory **RESEARCH USE ONLY** notice. |

---

## 4. Failure Plan & Presentation Fallbacks

In the event of hardware, network, or server failures during a live demonstration, follow this fallback hierarchy:

### Fallback Level 1: Backend Connection Error
- **Symptom**: Red error toast: *"Unable to connect to DermaAI inference backend"*.
- **Action**: Check Terminal 1. Restart backend via `python backend/main.py`. The Next.js UI automatically retries via internal proxy.

### Fallback Level 2: Complete Port / Runtime Failure
- **Symptom**: System refuses connections or Python environment crashes.
- **Action**: Seamlessly transition to the **Offline Presentation Mode**:
  1. Open folder `ai/results/phase20_final_submission/FINAL_PROJECT_SCREENSHOTS/`.
  2. Walk the judges through the high-fidelity sequential screenshots (`1_landing_page.png` through `8_disclaimer_notice.png`).
  3. Clearly state: *"We are presenting captured UI execution logs to ensure transparent adherence to time constraints."*

### Fallback Level 3: Network / Display Disconnection
- **Symptom**: Projector or conference Wi-Fi cuts out.
- **Action**: Present the compiled PowerPoint deck `HACKATHON_FINAL_PPT.pptx` or high-resolution PDF poster `DERMAAI_POSTER.pdf` directly from your local laptop screen.

> [!CAUTION]
> **INTEGRITY RULE**:
> NEVER fake a live inference call. If running offline, explicitly inform the audience that they are viewing recorded offline verification runs.
