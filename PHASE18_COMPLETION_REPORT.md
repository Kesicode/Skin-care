# PHASE 18 — DERMAAI DEMO COMPLETION REPORT

**Production-Quality Demo Engineering for the Frozen HAM10000 7-Class Research System**

---

## 1. Executive Summary

Phase 18 successfully engineered and verified the complete production-grade demonstration application around the frozen DermaAI research system. In accordance with strict experimental protocols:
- **Zero Retraining & Zero Tuning**: Neither model was retrained; no weights, parameters, or hyperparameters were altered. Both underlying checkpoints (`experiment8_best_model.pt` and `experiment13_best_model.pt`) remain bitwise identical to their research originals.
- **Frozen Pipeline**: Reproduces the exact Experiment 14 probability ensemble ($0.85 \cdot P_8 + 0.15 \cdot P_{13}$) with the verified standard argmax decision rule (re-confirmed by Experiment 15 where $\theta^* = 0.50$ made zero prediction changes).
- **Dual Grad-CAM Explainability**: Generates component attention heatmaps from `model.attention` (`CBAMBlock`, feature shape $[1, 960, 7, 7]$) targeting the ensemble predicted class, rendered side-by-side in the UI.
- **FastAPI + Next.js Stack**: Implemented a stateless, CPU-optimized FastAPI backend and a responsive, accessible Next.js 16 (App Router) research-demonstrator frontend with live backend health monitoring.
- **Automated Acceptance**: Passed all 13 rigorous criteria in `verify_dermaai_demo.py` and all 9 unit tests in `backend/test_demo.py`.
- **Final Status**: **`DEMO COMPLETE`**.

---

## 2. System Architecture

```
                   USER
                     │
                     ▼
          NEXT.JS WEB APPLICATION
          (Responsive UI, Tailwind CSS)
                     │
                     │ multipart/form-data upload
                     ▼
              FASTAPI BACKEND
                     │
             deterministic 224×224
           ImageNet RGB normalization
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     EXPERIMENT 8          EXPERIMENT 13
     FROZEN MODEL          FROZEN MODEL
     (alpha=0.75)          (1.20x mel wt)
          │                     │
       P8 probabilities      P13 probabilities
          │                     │
          └──────────┬──────────┘
                     ▼
              85 / 15 ENSEMBLE
                     │
        P = 0.85 · P8 + 0.15 · P13
                     │
                     ▼
             7-CLASS PREDICTION
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      prediction  confidence   top-3
     (argmax P)   (max P)    distribution
                     │
                     ▼
             Dual Grad-CAM Engine
            (model.attention: CBAM)
                     │
                     ▼
             JSON API RESPONSE
          (Base64 PNG Overlays)
                     │
                     ▼
             NEXT.JS DASHBOARD
```

---

## 3. Frozen Model Configuration

The application loads both component checkpoints once into memory at backend startup:

| Model Component | Checkpoint Path | Architecture | Backbone | Attention | Param Count | SHA-256 Checksum | File Size |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **Experiment 8** | `ai/models/experiment8_best_model.pt` | `DermaAI_MobileNetV3` | MobileNetV3-Large | CBAM | 4,326,297 | `ccc579cb2087545fa1118b7551015bec17c0682ec04066268d3a46aa5d23be2c` | 17,512,600 B |
| **Experiment 13** | `ai/models/experiment13_best_model.pt` | `DermaAI_MobileNetV3` | MobileNetV3-Large | CBAM | 4,326,297 | `9fab1b1218fddc86e060f322506bafa691429ef2620d7ff1c26041dc7b52ef21` | 17,512,921 B |

Both models run in `model.eval()` mode with `requires_grad=False` (gradient graphs are enabled only transiently on input tensors for Grad-CAM generation and immediately released).

---

## 4. Ensemble Configuration

The inference engine strictly computes:
$$P_8 = \text{softmax}(\text{logits}_8)$$
$$P_{13} = \text{softmax}(\text{logits}_{13})$$
$$P_{\text{final}} = 0.85 \cdot P_8 + 0.15 \cdot P_{13}$$
$$\hat{y} = \arg\max_{c \in \{0..6\}} P_{\text{final}}(c)$$
$$\text{confidence} = \max_{c \in \{0..6\}} P_{\text{final}}(c)$$

- **Decision Rule**: Standard argmax.
- **Experiment 15 Status**: Documented in the API and UI; threshold tuning on validation data selected $\theta^*=0.50$, resulting in zero changed predictions.

---

## 5. Preprocessing

The preprocessing pipeline implements the deterministic evaluation transforms:
1. Decode image stream in memory via PIL.
2. Ensure RGB format (`img.convert("RGB")`).
3. Bilinear resize to $224 \times 224$ pixels.
4. Convert to PyTorch Tensor.
5. Standard ImageNet channel-wise normalization:
   $$\text{mean} = [0.485, 0.456, 0.406], \quad \text{std} = [0.229, 0.224, 0.225]$$
6. No random cropping, rotation, flipping, color jitter, or test-time augmentation (TTA).

---

## 6. FastAPI Backend

Organized into modular, clean components under `backend/app/`:
- [`backend/app/config.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/config.py): Constants, file paths, checksums, class orders, device configuration.
- [`backend/app/schemas.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/schemas.py): Strict Pydantic models for request/response serialization.
- [`backend/app/utils.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/utils.py): Cryptographic SHA-256 verification, safe image validation, base64 encoding.
- [`backend/app/preprocessing.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/preprocessing.py): Deterministic $224 \times 224$ ImageNet transforms.
- [`backend/app/model_service.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/model_service.py): Singleton model manager, startup integrity verification, forward sanity checks.
- [`backend/app/gradcam.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/gradcam.py): Hook-based Grad-CAM generator on `model.attention`.
- [`backend/app/inference.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/inference.py): Dual-model ensemble coordinator, confidence extraction, top-3 generation.
- [`backend/app/main.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/main.py): FastAPI app with CORS, health, model-info, and predict routes.

---

## 7. Frontend

Implemented using Next.js 16 (App Router), TypeScript, Tailwind CSS, and Lucide icons in [`src/app/page.tsx`](file:///c:/Users/kesi/Downloads/Skin-care/src/app/page.tsx):
- **Branding**: "DermaAI — AI-Assisted Dermoscopic Image Classification Research Demonstration".
- **Live Health Status**: Real-time status badge ("Demo backend connected" / "Backend unavailable").
- **Drag & Drop Upload**: Accessible image zone with format and 10 MB size validation, image preview, "Analyze Image", and "Remove Image" actions.
- **Genuine Loading Experience**: Real multi-stage pipeline state transitions ("Loading image...", "Running ensemble...", "Computing class probabilities...", "Generating Grad-CAM...").
- **Results Dashboard**:
  - Predicted Class Name and Code.
  - Model ensemble confidence percentage.
  - Top-3 predictions ranked by probability.
  - Full 7-class probability horizontal distribution bars.
- **Grad-CAM Visualizer**: Side-by-side comparison of original dermoscopy, Exp 8 CAM, and Exp 13 CAM with clear interpretability disclaimers.
- **Research Transparency**: Collapsible "About this model" card detailing architecture, input size, and weights.
- **Benchmark Panel**: Project benchmark metrics with explicit population separation between test split ($N=988$) and validation split ($N=1,010$).
- **Accessibility**: Keyboard navigable, visible focus states, alt text, high-contrast labels.

---

## 8. Prediction Pipeline

1. Uploaded image is verified in memory (format, header, max 10 MB, dimensions).
2. Converted to RGB and preprocessed to a $[1, 3, 224, 224]$ tensor.
3. Passed through frozen Model A (Exp 8) and Model B (Exp 13) within `torch.inference_mode()`.
4. Softmax probabilities computed and weighted: $P_{\text{final}} = 0.85 P_8 + 0.15 P_{13}$.
5. Predicted class determined via argmax; confidence extracted via max probability.
6. Top-3 predictions sorted and 7-class distribution formatted.
7. Dual Grad-CAM computed for the target class $\hat{y}$ on each model's CBAM attention layer.
8. Output packaged into a JSON response with base64 PNG overlays and research notices.

---

## 9. Confidence Output

- Terminology is strictly labeled **"Model confidence"** or **"Ensemble confidence"**.
- Banned terms ("diagnostic certainty", "probability of disease", "clinical confidence") are completely eliminated.
- Low-confidence advisory (< 60%): Discloses that the prediction has relatively low confidence and should be treated as uncertain.

---

## 10. Grad-CAM Implementation

- Target module: `model.attention` (`CBAMBlock`).
- Verified activation shape: $[1, 960, 7, 7]$.
- Gradient calculation: Channel-wise Global Average Pooling (GAP) of gradients $\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$.
- Class Activation Map: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$, min-max normalized to $[0, 1]$.
- Rendering: Bilinear upsampling to original input dimensions, JET colormap blending ($\alpha=0.5$), exported as base64-encoded PNG.
- Disclosures: Labeled "Grad-CAM — Experiment 8 component" and "Grad-CAM — Experiment 13 component", with the mandatory notice that Grad-CAM is an interpretability aid, not proof of clinical reasoning or causality.

---

## 11. API Endpoints

| Method | Path | Request Body | Response Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | None | Returns `status`, `service`, `models_loaded`, `ensemble`, `device` |
| `GET` | `/api/model-info` | None | Returns architecture metadata, classes, parameters, and decision rule |
| `POST` | `/api/predict` | `multipart/form-data` (`image`) | Returns `prediction`, `confidence`, `probabilities`, `top3`, `gradcam` |

---

## 12. Security & Input Validation

- **Memory-Only Processing**: Images are processed directly in memory via `BytesIO`; no user uploads are permanently stored on disk.
- **Payload Limits**: Files larger than 10 MB are rejected with HTTP 413.
- **Integrity Validation**: PIL verifies header integrity before decompression. Corrupted files trigger HTTP 400.
- **Dimension Guards**: Rejects images smaller than $16 \times 16$ or larger than $8192 \times 8192$.
- **Path Sanitization**: No user-supplied file paths or checkpoint paths are accepted from clients.

---

## 13. Research Disclaimer

A prominent, high-importance banner is displayed in the demo UI and embedded in every API response:
> **RESEARCH USE ONLY:**
> This system is an academic research demonstrator for image-level classification of dermoscopic images on the retrospective HAM10000 dataset. It is NOT a certified medical diagnostic device, has NOT undergone clinical trial evaluation, and must NEVER be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic examination, or physician decision-making in patient care. Model predictions may be incorrect.

---

## 14. Benchmark Presentation

The benchmark panel presents the official held-out test split results ($N=988$):
- **Accuracy**: $80.57\%$
- **Balanced Accuracy**: $75.69\%$
- **Macro F1-Score**: $0.7051$
- **Weighted F1-Score**: $0.8106$
- **Melanoma Recall**: $50.94\%$
- **Melanoma Precision**: $54.00\%$
- **Melanoma F1-Score**: $0.5243$

Prominently disclosed as image-level benchmark metrics on the project's HAM10000 test split, explicitly stating they are **NOT** clinical diagnostic accuracy.

---

## 15. Test/Validation Population Separation

- **Held-Out Test Set ($N=988$)**: Reserved exclusively for benchmark metrics. Never used during demo testing or acceptance scripting.
- **Validation Set ($N=1,010$)**: Used for Phase 16 error analysis and case selection.
- **Synthetic/Unit Test Images**: Used exclusively for backend testing (`backend/test_demo.py`) and acceptance verification (`verify_dermaai_demo.py`), maintaining zero data leakage.

---

## 16. Verification Results

### Automated Acceptance Script (`verify_dermaai_demo.py`)
```
=================================================================
DERMAAI DEMO ACCEPTANCE VERIFICATION (PHASE 18)
=================================================================
    Model Loading: PASS
    Exp8 Checkpoint: PASS
    Exp13 Checkpoint: PASS
    Architecture: PASS
    Parameter Count: PASS
    Class Mapping: PASS
    Preprocessing: PASS
    Ensemble Formula: PASS
    Argmax Decision: PASS
    Confidence: PASS
    Grad-CAM Exp8: PASS
    Grad-CAM Exp13: PASS
    API: PASS
=================================================================
FINAL STATUS: DEMO COMPLETE
=================================================================
```

### Unit Test Suite (`backend/test_demo.py`)
- Ran 9 automated unit tests covering all components.
- Result: **9/9 tests passed in 0.518s (OK)**.

### Next.js Production Build
- Command: `npm run build`
- Result: **Compiled successfully in 2.1s with zero TypeScript or linting errors**.

---

## 17. Files Created / Updated

1. [`backend/app/config.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/config.py)
2. [`backend/app/schemas.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/schemas.py)
3. [`backend/app/utils.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/utils.py)
4. [`backend/app/preprocessing.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/preprocessing.py)
5. [`backend/app/gradcam.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/gradcam.py)
6. [`backend/app/model_service.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/model_service.py)
7. [`backend/app/inference.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/inference.py)
8. [`backend/app/main.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/app/main.py)
9. [`backend/main.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/main.py)
10. [`backend/requirements.txt`](file:///c:/Users/kesi/Downloads/Skin-care/backend/requirements.txt)
11. [`backend/test_demo.py`](file:///c:/Users/kesi/Downloads/Skin-care/backend/test_demo.py)
12. [`src/app/page.tsx`](file:///c:/Users/kesi/Downloads/Skin-care/src/app/page.tsx)
13. [`src/components/layout/Navbar.tsx`](file:///c:/Users/kesi/Downloads/Skin-care/src/components/layout/Navbar.tsx)
14. [`src/app/api/health/route.ts`](file:///c:/Users/kesi/Downloads/Skin-care/src/app/api/health/route.ts)
15. [`src/app/api/predict/route.ts`](file:///c:/Users/kesi/Downloads/Skin-care/src/app/api/predict/route.ts)
16. [`verify_dermaai_demo.py`](file:///c:/Users/kesi/Downloads/Skin-care/verify_dermaai_demo.py)
17. [`DERMAAI_DEMO_MANIFEST.json`](file:///c:/Users/kesi/Downloads/Skin-care/DERMAAI_DEMO_MANIFEST.json)
18. [`README_DERMAAI.md`](file:///c:/Users/kesi/Downloads/Skin-care/README_DERMAAI.md)
19. [`PHASE18_COMPLETION_REPORT.md`](file:///c:/Users/kesi/Downloads/Skin-care/PHASE18_COMPLETION_REPORT.md)

---

## 18. Known Issues

- None. All backend routes, model checks, Grad-CAM overlays, and frontend builds pass cleanly without warnings or errors.

---

## 19. Final Demo Status

```
============================================================
FINAL DEMO STATUS: DEMO COMPLETE
============================================================
```

*Execution is complete. Research experiments remain permanently frozen; no further model retraining, tuning, or checkpoint modification will be performed.*
