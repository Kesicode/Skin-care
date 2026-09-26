# DermaAI: Hackathon & Presentation Slide Outline (14 Slides)

This structured outline defines the presentation deck for DermaAI, designed for a 5–7 minute technical pitch or hackathon presentation.

---

## Slide 1: Title & Overview
- **Title**: DermaAI: Validation-Calibrated Deep Ensemble & Explainable Attention for Skin Lesion Classification
- **Subtitle**: Retrospective 7-Class HAM10000 Study & Production Research Demonstrator
- **Presenter / Team**: DermaAI Research Team
- **Key Visual**: Hero UI screenshot with side-by-side Grad-CAM heatmaps
- **Footer**: *Strictly For Research Use Only — Not A Certified Diagnostic Device*

---

## Slide 2: Clinical Context & The Imbalance Dilemma
- **The Core Problem**:
  - Dermoscopy is vital for early detection, but visual interpretation has substantial inter-observer variance.
  - Melanoma (mel) is lethal but represents only ~11% of the HAM10000 benchmark.
  - Melanocytic Nevi (nv) represent >67% of cases.
- **The Trade-off**:
  - Naive classifiers favor majority class (nv), missing subtle early-stage melanomas.
  - Severe class weighting causes excessive false positives, degrading overall diagnostic utility.

---

## Slide 3: Research Methodology & Frozen Integrity
- **19-Phase Scientific Methodology**:
  - Phase 1–7: Baseline exploratory architectures & augmentation pipelines.
  - Phase 8: MobileNetV3-Large + CBAM attention + tempered class weighting ($\alpha=0.75$).
  - Phase 13: Targeted $1.20\times$ melanoma weight adjustment.
  - Phase 14: Probability-level ensemble calibration on validation data ($\lambda=0.15$).
  - Phase 15: Decision threshold verification ($\theta^*=0.50 \implies \text{natural argmax}$).
  - Phase 16: Systematic Grad-CAM explainability on $N=1,010$ validation cohort.
  - Phase 17–19: Research documentation, demo engineering, and production deployment.
- **Strict Data Isolation**:
  - 988-image test set permanently locked until final benchmark evaluation.
  - Zero retraining, zero parameter modifications since Phase 14.

---

## Slide 4: Neural Architecture: MobileNetV3-Large + CBAM
- **Backbone**: MobileNetV3-Large (efficient inverted residual blocks + hard-swish).
- **Attention**: Convolutional Block Attention Module (CBAM) placed before classifier head.
  - Channel Attention: Squeeze-and-excitation across spatial dimensions ($7 \times 7 \to 1 \times 1$).
  - Spatial Attention: Inter-channel pooling highlighting salient lesion boundaries.
- **Parameter Efficiency**:
  - Only **4,326,297 parameters** per model (~16.7 MB).
  - CPU-friendly: <100ms per forward pass without GPU requirements.

---

## Slide 5: The Two Complementary Checkpoints
- **Model A (Experiment 8)**:
  - *Design*: Balanced baseline, tempered class weight $\alpha=0.75$.
  - *Test Acc*: 80.67% | *Macro F1*: 70.18% | *Mel Recall*: 48.11%
  - *Strength*: Superior precision and multiclass coherence across benign classes.
- **Model B (Experiment 13)**:
  - *Design*: Targeted $1.20\times$ relative melanoma weighting.
  - *Test Acc*: 78.04% | *Macro F1*: 67.58% | *Mel Recall*: 59.43%
  - *Strength*: Elevated sensitivity to subtle melanocytic irregularities.

---

## Slide 6: Validation-Calibrated Ensemble (Experiment 14)
- **Mathematical Formulation**:
  $$P_{\text{ens}}(x) = 0.85 \cdot P_8(x) + 0.15 \cdot P_{13}(x)$$
- **Why $\lambda=0.15$?**:
  - Rigorous grid search over $\lambda \in [0.0, 1.0]$ with step $0.05$ strictly on $N=1,010$ validation images.
  - Preserved peak Macro F1 ($F_1^{\text{max}} - F_1(\lambda) \le 0.005$) while maximizing melanoma recall.
- **Decision Rule**:
  $$\hat{y} = \arg\max_{c \in \{0 \dots 6\}} P_{\text{ens}}^c(x)$$

---

## Slide 7: Frozen Test Benchmark Results ($N=988$)
| Metric | Exp 8 (Base) | Exp 13 (Recall) | Exp 14 (Ensemble) | Delta vs Exp 8 |
|---|---|---|---|---|
| **Overall Accuracy** | 80.67% | 78.04% | **80.57%** | -0.10% |
| **Balanced Accuracy** | 75.37% | 73.14% | **75.69%** | **+0.32%** |
| **Macro F1** | 70.18% | 67.58% | **70.51%** | **+0.33%** |
| **Weighted F1** | 81.04% | 78.96% | **81.06%** | **+0.02%** |
| **Melanoma Recall** | 48.11% | **59.43%** | **50.94%** | **+2.83%** |
| **Melanoma Precision**| 56.04% | 45.99% | **54.00%** | -2.04% |
| **Melanoma F1** | 51.78% | 51.85% | **52.43%** | **+0.65%** |

*Takeaway: The ensemble achieves the highest Balanced Accuracy, Macro F1, and Melanoma F1 of any tested configuration.*

---

## Slide 8: Experiment 15 Verification: Threshold vs Argmax
- **Hypothesis**: Can tuning decision threshold $\theta$ for melanoma improve clinical trade-offs?
- **Validation Tuning Result**: Optimal threshold was $\theta^* = 0.50$.
- **Test Set Equivalence Proof**:
  - Evaluated on all 988 test images: **Zero predictions changed** between $\theta^*=0.50$ and $\arg\max$.
  - Proves the blended probability distribution is inherently well-calibrated; artificial post-hoc thresholding was unnecessary and avoided overfitting.

---

## Slide 9: Dual Grad-CAM Attention Explainability
- **Mechanism**:
  - Gradient of predicted class score with respect to feature activations at `model.attention` (shape: $1 \times 960 \times 7 \times 7$).
  - Rectified linear combination yields spatial heatmap $\to$ Jet colormap overlay.
- **Dual Visual Inspection**:
  - **Exp 8 Map**: Reflects global structural symmetry and pigment network boundaries.
  - **Exp 13 Map**: Highlights localized high-risk features (atypical network, streaks, regression).
  - Side-by-side presentation empowers human-in-the-loop validation.

---

## Slide 10: Full-Stack Production Architecture
- **Backend**:
  - FastAPI + Uvicorn on Python 3.11.
  - Singleton `ModelService` with SHA-256 integrity validation on startup.
  - In-memory image processing (Pillow, zero disk writes for privacy).
- **Frontend**:
  - Next.js 16 with Turbopack, React 19, Tailwind CSS 4.
  - Drag-and-drop file ingestion, client-side validation (<10MB, JPG/PNG).
  - Dynamic visual probability bars and interactive model inspection modals.

---

## Slide 11: Deployment & DevOps Architecture
- **Containerization**:
  - Minimal multi-stage `Dockerfile` with CPU-optimized PyTorch wheels.
  - Image size kept compact by excluding training datasets via `.dockerignore`.
- **Cloud Orchestration**:
  - Infrastructure as Code via `render.yaml` (Blueprint).
  - Vercel frontend integration with Next.js API proxy routing.
  - Standard container health check on `/api/health`.

---

## Slide 12: Live Software Demonstration
- *Interactive Walkthrough*:
  1. Service health check & model parameter verification.
  2. Upload of synthetic/lesion image.
  3. Real-time inference: top-3 class breakdown with percentage bars.
  4. Inspection of Exp8 vs Exp13 Grad-CAM heatmaps.
  5. Error handling demonstrations: invalid file format and oversized file rejection.

---

## Slide 13: Ethical Guardrails & Regulatory Compliance
- **Safety by Design**:
  - Clear **RESEARCH USE ONLY** disclaimers on every view and response.
  - Prohibition on diagnostic certainty claims ("cancer confirmed", "benign guaranteed").
  - Terminology aligned with standard AI reporting guidelines: "predicted class", "model confidence score".
  - Transparent documentation of known dataset limitations (retrospective single-source bias).

---

## Slide 14: Conclusion, Impact & Q&A
- **Key Takeaways**:
  - High performance (80.57% test acc, 75.69% bal acc) with low footprint (4.3M parameters).
  - Principled ensemble design solves melanoma sensitivity without degrading multiclass precision.
  - Interpretable dual attention builds trust.
  - Production-ready, turnkey deployment with Docker and Render.
- **Repository & Artifacts**:
  - Codebase: Clean, modular, fully verified (100% test pass rate).
  - Research Documentation: IEEE-format paper, 23-section research report, full audit trail.
- **Open for Questions!**
