# DermaAI: Final Demo Pre-Flight Checklist (22 Items)

Use this checklist before delivering a live demonstration, hackathon submission, or presentation to verify system health.

---

## Section A: Frozen Model & Checkpoint Integrity

- [x] **Item 1: Experiment 8 Checkpoint Presence & Size**  
  File exists at `ai/models/experiment8_best_model.pt` with exact byte size of `17,512,600` bytes.
- [x] **Item 2: Experiment 8 Cryptographic Hash**  
  SHA-256 matches `ccc579cb2087545fa1118b7551015bec17c0682ec04066268d3a46aa5d23be2c`.
- [x] **Item 3: Experiment 13 Checkpoint Presence & Size**  
  File exists at `ai/models/experiment13_best_model.pt` with exact byte size of `17,512,921` bytes.
- [x] **Item 4: Experiment 13 Cryptographic Hash**  
  SHA-256 matches `9fab1b1218fddc86e060f322506bafa691429ef2620d7ff1c26041dc7b52ef21`.

---

## Section B: Mathematical & Pipeline Rigor

- [x] **Item 5: Zero Trainable Parameters**  
  Both neural networks load in `eval()` mode with `requires_grad = False` on all parameters ($4,326,297$ each; 0 trainable).
- [x] **Item 6: Convex Ensemble Formulation**  
  Ensemble weights are strictly locked to $0.85$ (Experiment 8) and $0.15$ (Experiment 13), summing to $1.0000$.
- [x] **Item 7: Argmax Decision Rule**  
  Prediction is determined by $\arg\max(P_{\text{ens}})$, matching Experiment 15 validation findings ($\theta^*=0.50$ leaves test predictions unchanged).
- [x] **Item 8: Probability Conservation**  
  Output probabilities across all 7 classes sum to $1.0000 \pm 10^{-4}$ on all inputs.

---

## Section C: Explainability & Attention Mechanism

- [x] **Item 9: Grad-CAM Target Layer Alignment**  
  Gradients and activations are extracted from `model.attention` (CBAM module) with feature tensor shape $(1, 960, 7, 7)$.
- [x] **Item 10: Dual Model Explainability**  
  Both Experiment 8 and Experiment 13 heatmaps are computed and rendered side-by-side.
- [x] **Item 11: Base64 Heatmap Integrity**  
  Heatmaps are properly encoded as `data:image/png;base64,...` data URIs, valid for inline browser display.

---

## Section D: Backend API & Service Reliability

- [x] **Item 12: Health Endpoint Contract**  
  `GET /api/health` returns HTTP 200 with `status: ok`, `models_loaded: true`, and correct ensemble weights.
- [x] **Item 13: Model Metadata Contract**  
  `GET /api/model-info` (and `/api/models/info`) returns HTTP 200 with full architecture, class codes, and parameter counts.
- [x] **Item 14: Dynamic Port Binding**  
  Server honors the `$PORT` environment variable, defaulting to `8000`.
- [x] **Item 15: Configurable CORS Origins**  
  CORS middleware parses `DERMAAI_ALLOWED_ORIGINS` to permit secure cross-origin communication.

---

## Section E: Frontend User Experience & Resilience

- [x] **Item 16: Next.js Production Build**  
  Application compiles cleanly with zero TypeScript errors via `npm run build`.
- [x] **Item 17: Multi-Stage Progress Indicator**  
  User sees real-time progress transitions (*Loading image...*, *Running ensemble...*, *Generating Grad-CAM...*).
- [x] **Item 18: Client & Server Proxy Fallback**  
  The web app connects directly to FastAPI via `NEXT_PUBLIC_API_URL` or routes transparently through `/api/predict`.

---

## Section F: Security, Privacy & Ethical Compliance

- [x] **Item 19: In-Memory Image Handling**  
  Uploaded files are processed entirely in memory; zero patient or demo images are persisted to disk.
- [x] **Item 20: Input Validation & Sanitization**  
  Non-image files return HTTP 400; files exceeding 10 MB return HTTP 413; corrupted bytes are safely caught.
- [x] **Item 21: Mandatory Research Use Disclaimer**  
  "RESEARCH USE ONLY" notice is prominently displayed in the top alert banner, results cards, and API metadata.
- [x] **Item 22: Strict Data Population Isolation**  
  Zero images from the 988-image test set are used during demo operation, and validation statistics ($N=1,010$) remain strictly separated.

---

## Acceptance Summary
- **Total Verification Items**: 22 / 22 Verified
- **Status**: **READY FOR LIVE DEMONSTRATION & SUBMISSION**
