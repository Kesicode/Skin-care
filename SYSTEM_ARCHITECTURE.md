# DermaAI: System Architecture & Data Flow

This document details the architectural design, component responsibilities, data flow, and deployment topology of the **DermaAI** research demonstration platform.

---

## 1. High-Level System Architecture Diagram

```mermaid
flowchart TD
    subgraph Client ["Client Layer (Browser)"]
        UI["Next.js 16 (React 19 + Tailwind CSS 4)"]
        Dropzone["Drag & Drop File Ingestion"]
        Dashboard["Probabilities & Top-3 Distribution"]
        Heatmaps["Dual Grad-CAM Heatmap Viewer"]
    end

    subgraph Gateway ["Edge & API Layer"]
        Proxy["Next.js Route Handlers (/api/predict, /api/health)"]
        FastAPI["FastAPI 0.110+ Service (Uvicorn ASGI)"]
        CORS["CORS Middleware (DERMAAI_ALLOWED_ORIGINS)"]
        Validator["In-Memory Image Validator (PIL, max 10MB)"]
    end

    subgraph Pipeline ["Frozen ML Inference Engine"]
        Prep["Preprocessing (Resize 224x224, ImageNet Normalization)"]
        
        subgraph ModelService ["Singleton Model Service"]
            direction LR
            M8["Model A: Exp 8 Checkpoint\n(MobileNetV3 + CBAM)\n4,326,297 params\nWeight: 0.85"]
            M13["Model B: Exp 13 Checkpoint\n(MobileNetV3 + CBAM)\n4,326,297 params\nWeight: 0.15"]
        end

        Blend["Probability Blending:\nP_ens = 0.85 * P8 + 0.15 * P13"]
        Argmax["Decision Rule:\ny_hat = argmax(P_ens)"]
        
        subgraph Explainability ["Dual Attention Explainability"]
            GC8["Grad-CAM Exp 8\n(model.attention)"]
            GC13["Grad-CAM Exp 13\n(model.attention)"]
        end
    end

    UI --> Dropzone
    Dropzone -->|POST Multipart| Proxy
    Proxy -->|Internal Fetch| FastAPI
    Dropzone -.->|Direct Fetch Option| FastAPI
    FastAPI --> CORS
    CORS --> Validator
    Validator --> Prep
    Prep --> M8
    Prep --> M13
    M8 -->|P8| Blend
    M13 -->|P13| Blend
    Blend --> Argmax
    Argmax --> GC8
    Argmax --> GC13
    GC8 --> Heatmaps
    GC13 --> Heatmaps
    Argmax --> Dashboard
    Blend --> Dashboard
```

---

## 2. Detailed Component Breakdown

### 2.1 Web Presentation Layer (Next.js 16)
- **Framework**: Next.js 16 with Turbopack, React 19, and Tailwind CSS 4.
- **Key Modules**:
  - `src/app/page.tsx`: Single-page reactive application orchestrating image upload, stage-based progress indicators, metric visualization, and Grad-CAM side-by-side display.
  - `src/components/layout/Navbar.tsx`: Header navigation with status badges and research context links.
  - `src/app/api/predict/route.ts` & `src/app/api/health/route.ts`: Resilient Next.js server route handlers providing automatic failover proxying to the FastAPI backend.

### 2.2 API & Serving Layer (FastAPI)
- **Framework**: FastAPI with Uvicorn ASGI server running on Python 3.11.
- **Port Management**: Dynamically honors `$PORT` environment variable (default: `8000`) for cloud PaaS (Render, Railway, Heroku).
- **CORS Management**: Configurable origins via `DERMAAI_ALLOWED_ORIGINS` (supports wildcard `*` or comma-separated lists).
- **In-Memory Safety**: Images are streamed directly to RAM via `BytesIO`, validated with Pillow, and discarded immediately after inference. Zero disk writes ensure compliance with privacy and data protection principles.

### 2.3 Frozen Model Inference Engine
- **Singleton Pattern**: `ModelService` loads both PyTorch checkpoints once during server startup into global memory.
- **Integrity Validation**: Verifies SHA-256 hashes (`ccc579cb...` for Exp 8, `9fab1b12...` for Exp 13) and parameter counts ($4,326,297$ each) before accepting traffic.
- **Strict Freezing**:
  - `model.eval()` invoked on both models.
  - All parameter tensors have `requires_grad = False`.
  - Zero weight updates, zero optimizers, zero training routines.

### 2.4 Mathematical Probability Blending
- **Forward Pass**:
  Given an input tensor $x \in \mathbb{R}^{1 \times 3 \times 224 \times 224}$:
  $$z_8 = f_{\theta_8}(x), \quad z_{13} = f_{\theta_{13}}(x)$$
  $$P_8 = \text{Softmax}(z_8), \quad P_{13} = \text{Softmax}(z_{13})$$
- **Ensemble Blend**:
  $$P_{\text{ens}} = 0.85 \cdot P_8 + 0.15 \cdot P_{13}$$
- **Argmax Decision**:
  $$\hat{c} = \arg\max_{j \in \{0 \dots 6\}} P_{\text{ens}}^j$$
  $$\text{Confidence} = P_{\text{ens}}^{\hat{c}}$$

### 2.5 Explainability Subsystem (Dual Grad-CAM)
- **Target Layer**: `model.attention` (the Convolutional Block Attention Module connecting the backbone feature extractor to the classification head).
- **Activation Volume**: $A \in \mathbb{R}^{1 \times 960 \times 7 \times 7}$.
- **Gradients**:
  $$\alpha_k = \frac{1}{7 \times 7} \sum_{i=1}^7 \sum_{j=1}^7 \frac{\partial z_{\hat{c}}}{\partial A_{k,i,j}}$$
- **Localization Map**:
  $$L_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k=1}^{960} \alpha_k A_k \right)$$
- The resulting $7 \times 7$ saliency map is bilinearly upsampled to $224 \times 224$, normalized to $[0, 1]$, colorized with the OpenCV `COLORMAP_JET` palette, blended with the original lesion image at $\alpha=0.5$, and converted to base64 PNG for inline browser rendering.

---

## 3. Data Flow Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Clinician
    participant UI as Next.js Frontend
    participant API as FastAPI Backend
    participant MS as ModelService (Singleton)
    participant GC as GradCAM Engine

    User->>UI: Select/Drop Dermoscopic Image
    UI->>UI: Client Validation (format, size <10MB)
    UI->>API: POST /api/predict (Multipart Form)
    API->>API: Validate MIME & decode PIL Image
    API->>MS: run_dermaai_inference(PIL_Image)
    MS->>MS: Preprocess: Resize 224x224, Normalize
    par Parallel Model Execution
        MS->>MS: Forward Exp8 -> Softmax -> P8
        MS->>MS: Forward Exp13 -> Softmax -> P13
    end
    MS->>MS: Calculate P_ens = 0.85*P8 + 0.15*P13
    MS->>MS: Determine predicted class = argmax(P_ens)
    par Dual Grad-CAM Computation
        MS->>GC: Backward pass on Exp8 for predicted class
        GC-->>MS: Base64 Overlay Exp8
        MS->>GC: Backward pass on Exp13 for predicted class
        GC-->>MS: Base64 Overlay Exp13
    end
    MS-->>API: PredictionResponse Object
    API-->>UI: JSON (prediction, probabilities, top3, gradcam)
    UI-->>User: Render Results Card + Side-by-Side Heatmaps
```

---

## 4. Production Deployment Topology

```
                  ┌───────────────────────────────┐
                  │          Internet             │
                  └──────────────┬────────────────┘
                                 │ HTTPS (Port 443)
                                 ▼
                  ┌───────────────────────────────┐
                  │    Cloud Reverse Proxy /      │
                  │       CDN (Cloudflare)        │
                  └──────┬─────────────────┬──────┘
                         │                 │
           Host: app.dermaai.com           Host: api.dermaai.com
                         │                 │
                         ▼                 ▼
          ┌────────────────────────┐  ┌────────────────────────┐
          │   Next.js Front-end    │  │   FastAPI Back-end     │
          │     (Vercel/Render)    │  │ (Docker on Render/GCP) │
          │                        │  │                        │
          │ Node 20 Runtime        │  │ Python 3.11 Runtime    │
          │ Next.js 16 Turbopack   │  │ Uvicorn ASGI Server    │
          │ Next Route Handlers    │  │ In-Memory Model Cache  │
          └────────────────────────┘  └────────────────────────┘
```
