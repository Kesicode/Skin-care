# DermaAI — Research Demonstrator

**AI-Assisted Dermoscopic Image Classification Research Demonstration**

DermaAI is a production-quality software demonstration system engineered around the **frozen Experiment 14 dual-model probability ensemble** trained on the HAM10000 7-class dermoscopy dataset.

> [!WARNING]
> **RESEARCH USE ONLY:**
> This system is an academic research demonstrator for image-level classification of dermoscopic images. It is **NOT** a certified medical diagnostic device, has **NOT** undergone clinical trial evaluation, and must **NEVER** be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic examination, or physician decision-making in patient care.

---

## 1. System Architecture

```
                   USER
                     │
                     ▼
          NEXT.JS WEB APPLICATION
                     │
                     │ image upload (multipart/form-data)
                     ▼
              FASTAPI BACKEND
                     │
            deterministic 224×224
          ImageNet normalization
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
             Grad-CAM Explainability
             (CBAM attention layer)
                     │
                     ▼
               API RESPONSE
                     │
                     ▼
              WEBSITE DISPLAY
```

---

## 2. Frozen Model Configuration

- **Architecture**: `DermaAI_MobileNetV3`
- **Backbone**: `MobileNetV3-Large`
- **Attention**: `CBAMBlock` (`model.attention`)
- **Parameters per Model**: Exactly **4,326,297** parameters (frozen in `eval()` mode, `requires_grad=False`)
- **Input Resolution**: $224 \times 224$ RGB with standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Classes (Strict Order)**:
  1. `0: akiec` — Actinic Keratosis / Intraepithelial Carcinoma
  2. `1: bcc` — Basal Cell Carcinoma
  3. `2: bkl` — Benign Keratosis (Solar Lentigo / Seborrheic Keratosis)
  4. `3: df` — Dermatofibroma
  5. `4: mel` — Melanoma
  6. `5: nv` — Melanocytic Nevus
  7. `6: vasc` — Vascular Lesion
- **Checkpoints**:
  - `ai/models/experiment8_best_model.pt` (SHA-256: `ccc579cb2087545fa1118b7551015bec17c0682ec04066268d3a46aa5d23be2c`)
  - `ai/models/experiment13_best_model.pt` (SHA-256: `9fab1b1218fddc86e060f322506bafa691429ef2620d7ff1c26041dc7b52ef21`)

---

## 3. Probability Ensemble Formula

The demo reproduces the exact frozen Experiment 14 inference pipeline:
$$P_8 = \text{softmax}(\text{logits}_8)$$
$$P_{13} = \text{softmax}(\text{logits}_{13})$$
$$P_{\text{final}} = 0.85 \cdot P_8 + 0.15 \cdot P_{13}$$
$$\hat{y} = \arg\max_{c} P_{\text{final}}(c)$$
$$\text{confidence} = \max_{c} P_{\text{final}}(c)$$

*Note on Experiment 15:* Experiment 15 verified that post-hoc decision-threshold tuning ($\theta^* = 0.50$) produced zero changed predictions relative to the natural argmax decision rule.

---

## 4. Grad-CAM Explainability

- **Target Feature Representation**: `model.attention` (`CBAMBlock`).
- **Feature Map Dimensions**: $[1, 960, 7, 7]$ for $224 \times 224$ input.
- **Target Class**: Ensemble predicted class ($\hat{y}$).
- **Computation**: Channel-wise Global Average Pooling (GAP) of activation gradients, ReLU activation, min-max $[0, 1]$ normalization, bilinear upsampling, JET colormap blending.
- **Side-by-Side Presentation**: Component maps for Exp 8 and Exp 13 are displayed side-by-side to illuminate model agreement or divergence.
- **Interpretability Disclosure**: Grad-CAM is an interpretability aid showing regions correlated with model logit activation, not proof of clinical causality.

---

## 5. API Endpoints

### `GET /api/health`
Returns runtime service status and model loading state:
```json
{
  "status": "ok",
  "service": "DermaAI",
  "models_loaded": true,
  "ensemble": {
    "experiment8_weight": 0.85,
    "experiment13_weight": 0.15
  },
  "input_size": "224x224",
  "device": "cpu"
}
```

### `GET /api/model-info`
Returns public model metadata and architecture specifications.

### `POST /api/predict`
Accepts `multipart/form-data` with an `image` file (JPG, JPEG, PNG, max 10 MB). Returns:
```json
{
  "success": true,
  "prediction": {
    "class_index": 4,
    "class_code": "mel",
    "class_name": "Melanoma"
  },
  "confidence": 0.7342,
  "probabilities": {
    "akiec": 0.012,
    "bcc": 0.031,
    "bkl": 0.087,
    "df": 0.005,
    "mel": 0.7342,
    "nv": 0.119,
    "vasc": 0.0118
  },
  "top3": [
    { "class_code": "mel", "class_name": "Melanoma", "probability": 0.7342 },
    { "class_code": "nv", "class_name": "Melanocytic Nevus", "probability": 0.1190 },
    { "class_code": "bkl", "class_name": "Benign Keratosis...", "probability": 0.0870 }
  ],
  "ensemble": {
    "experiment8_weight": 0.85,
    "experiment13_weight": 0.15
  },
  "gradcam": {
    "available": true,
    "target_class": "mel",
    "target_layer": "model.attention",
    "experiment8": "data:image/png;base64,...",
    "experiment13": "data:image/png;base64,...",
    "notice": "Grad-CAM is an interpretability aid..."
  },
  "research_notice": "RESEARCH USE ONLY..."
}
```

---

## 6. Local Setup & Execution Commands

### Prerequisites
- Python 3.10+ (using `ai/.venv` or custom virtual environment)
- Node.js 18+ and npm

### 1. Start the FastAPI Backend
```bash
# From repository root
cd backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Backend will be live at `http://localhost:8000`. Test health: `http://localhost:8000/api/health`.

### 2. Start the Next.js Frontend
```bash
# From repository root
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

### 3. Run Automated Acceptance Verification
```bash
python verify_dermaai_demo.py
```
Expected output:
```
============================================================
DERMAAI DEMO ACCEPTANCE VERIFICATION (PHASE 18)
============================================================
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
============================================================
FINAL STATUS: DEMO COMPLETE
============================================================
```

---

## 7. Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DERMAAI_DEVICE` | `cpu` | Device for PyTorch inference (`cpu` or `cuda`) |
| `DERMAAI_MAX_UPLOAD_MB` | `10` | Maximum upload file size in megabytes |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL for frontend fetch requests |

---

## 8. Research Benchmark & Population Disclosure

| Metric | Experiment 14 Ensemble Value | Description |
| :--- | :---: | :--- |
| **Accuracy** | **80.57%** | $796 / 988$ correct predictions |
| **Balanced Accuracy** | **75.69%** | Mean unweighted class sensitivity |
| **Macro F1-Score** | **70.51%** | Unweighted F1 average across all 7 classes |
| **Weighted F1-Score** | **81.06%** | Class-prevalence weighted F1 |
| **Melanoma Recall** | **50.94%** | $54 / 106$ biopsy-verified melanomas detected |
| **Melanoma Precision** | **54.00%** | Melanoma precision on test split |
| **Melanoma F1-Score** | **52.43%** | Harmonic mean of melanoma recall & precision |

**Mandatory Disclosures:**
1. **Not Clinical Accuracy**: These metrics represent image-level benchmark results on the project's 988-image HAM10000 test split. They are NOT clinical diagnostic accuracy.
2. **Repeated Evaluation**: The project test split was evaluated sequentially during development; therefore this benchmark is not an independent external validation result.
3. **Population Separation**: Final benchmark metrics reflect the 988-image test split. The Phase 16 Grad-CAM explainability and error decomposition were conducted on the separate 1,010-image validation split.
