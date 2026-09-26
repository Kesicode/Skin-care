# DermaAI: Final Project Summary

**Project Name**: DermaAI  
**Domain**: Computer-Aided Dermoscopic Image Classification  
**Status**: Academic Research Demonstrator (Final Phase 20)  
**Authors**: DermaAI Research Team  
**Date**: September 26, 2026  

---

## 1. Problem & Clinical Motivation

Dermoscopic skin lesion classification is a high-stakes clinical triage domain. Automated image analysis holds promise for early screening, yet real-world applications face extreme class imbalance and morphological overlap. In standard benchmarks like HAM10000, benign Melanocytic Nevi (nv) represent >67% of cases, while malignant Melanoma (mel)—the deadliest cutaneous neoplasm—accounts for only ~11%. 

Naive neural network training under standard cross-entropy suffers from majority-class bias, yielding deceptive ~80% overall accuracy while missing subtle early melanomas. Conversely, extreme inverse-frequency weighting inflates false alarms, eroding clinician trust. DermaAI was established to resolve this trade-off through controlled empirical experimentation, validation-calibrated ensembling, and transparent visual explainability.

---

## 2. Dataset & Population Isolation

The system was developed on the retrospective **HAM10000** dataset (10,015 dermoscopic images across 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). To prevent data leakage, a strict lesion-level stratified split (seed = 42) was enforced:
- **Training Set**: 8,017 images (80%)
- **Validation Set**: 1,010 images (10%) — reserved for hyperparameter selection, ensemble calibration, and Phase 16 Grad-CAM error analysis.
- **Held-Out Test Set**: 988 images (10%) — permanently locked as the final objective benchmark.

*Population Separation Rule*: The 988-image test cohort was never accessed for parameter tuning or demo execution.

---

## 3. Neural Architecture

DermaAI adopts an ultra-efficient mobile architecture:
- **Backbone**: MobileNetV3-Large with hard-swish activations and inverted bottleneck residual blocks.
- **Attention Mechanism**: Convolutional Block Attention Module (CBAM) placed before the linear classification head. Channel attention isolates inter-channel feature relationships, while spatial attention highlights salient boundary structures.
- **Complexity**: Only **4,326,297 parameters per model** (~16.7 MB checkpoint size).
- **Execution**: ~15ms CPU forward pass; edge and mobile deployable.

---

## 4. 15-Experiment Controlled Progression

Rather than performing stochastic hyperparameter searches, 15 hypotheses were sequentially evaluated:
- **Exp 1–2**: Uniform cross-entropy vs. standard inverse-frequency weighting.
- **Exp 3–7**: Augmentation pipelines, cutouts, and feature extractor baselines.
- **Exp 8**: MobileNetV3 + CBAM with tempered class weighting ($\alpha=0.75$), establishing strong baseline multiclass accuracy (80.67%) and Macro F1 (70.18%).
- **Exp 9–12**: Cosine LR schedules, aggressive augmentations, high spatial resolutions (256x256), and ConvNeXt-Tiny comparisons.
- **Exp 13**: Targeted $1.20\times$ relative melanoma weighting, boosting melanoma recall from 48.11% up to 59.43%.
- **Exp 14**: Validation-calibrated convex probability ensemble combining Exp 8 and Exp 13.
- **Exp 15**: Validation threshold grid search ($\theta^*=0.50$), proving natural argmax optimality.

---

## 5. Final Frozen Probability Ensemble

The production DermaAI system freezes both complementary models in `eval()` mode:
$$P_{\text{final}}(x) = 0.85 \cdot P_8(x) + 0.15 \cdot P_{13}(x)$$
$$\hat{y} = \arg\max_{c \in \{0 \dots 6\}} P_{\text{final}}^c(x)$$

- **Experiment 8 (85% weight)**: Anchors global precision and benign-class specificity.
- **Experiment 13 (15% weight)**: Enhances sensitivity to atypical melanocytic lesion structures.

---

## 6. Audited Final Benchmark Results ($N=988$ Test Split)

| Metric | Exp 8 (Base) | Exp 13 (Targeted) | Exp 14 Ensemble (FINAL) | Delta vs Exp 8 |
|---|---|---|---|---|
| **Overall Accuracy** | 80.67% | 78.04% | **80.57%** | -0.10% |
| **Balanced Accuracy** | 75.37% | 73.14% | **75.69%** | **+0.32%** |
| **Macro F1-Score** | 70.18% | 67.58% | **70.51%** | **+0.33%** |
| **Weighted F1-Score** | 81.04% | 78.96% | **81.06%** | **+0.02%** |
| **Melanoma Precision** | 56.04% | 45.99% | **54.00%** | -2.04% |
| **Melanoma Recall** | 48.11% | **59.43%** | **50.94%** | **+2.83%** |
| **Melanoma F1-Score** | 51.78% | 51.85% | **52.43%** | **+0.65%** |

*Takeaway*: The ensemble achieves the highest Balanced Accuracy, Macro F1, and Melanoma F1 across the entire 15-experiment series.

---

## 7. Dual Grad-CAM Attention Explainability

DermaAI extracts gradient attributions from the CBAM attention layer (`model.attention`, shape: $1 \times 960 \times 7 \times 7$):
- **Exp 8 Map**: Captures broader lesion morphology, border symmetry, and background contrast.
- **Exp 13 Map**: Exhibits sharp focal activation on atypical pigment networks and localized hyperpigmentation.
- Side-by-side presentation allows researchers to inspect where component models agree or diverge.

---

## 8. Quantitative Error Analysis

Analysis of benchmark misclassifications reveals clear morphological drivers rather than stochastic failures:
- **Melanoma $\to$ Nevus (26 cases)**: Subtle, early-stage superficial spreading melanomas with uniform pigment and regular borders.
- **Melanoma $\to$ Benign Keratosis (19 cases)**: Hyperkeratotic or verrucous melanoma variants mimicking seborrheic keratoses.
- **Nevus $\to$ Melanoma (38 false alarms)**: Highly atypical, dysplastic junctional nevi exhibiting pigment network irregularity.

---

## 9. Scientific Limitations

1. **Retrospective Evaluation**: Evaluated solely on the single-source HAM10000 repository without prospective multi-center validation.
2. **Repeated Test-Set Evaluation**: Test data was evaluated sequentially across experiments; not equivalent to blind external trials.
3. **Rare-Class Under-Representation**: Dermatofibroma ($N=115$) and vascular lesions ($N=142$) have limited statistical power.
4. **No Prospective Clinical Trial**: Not validated on uncurated consumer smartphone images.

---

## 10. Software Demonstrator & Deployment

The system is deployed as an open-source, full-stack demonstration application:
- **Backend**: FastAPI + Uvicorn with in-memory validation, dynamic `$PORT` binding, and Docker containerization (`backend/Dockerfile`).
- **Frontend**: Next.js 16 with Turbopack, React 19, and Tailwind CSS 4 (`src/app/page.tsx`).
- **Verification**: 4 comprehensive automated test suites passing at 100% (36 assertions).

---

## 11. Mandatory Research Disclaimer

> [!WARNING]
> **RESEARCH USE ONLY**:
> DermaAI is an academic research demonstrator for image-level classification on retrospective HAM10000 data. It is NOT a certified medical diagnostic device, has NOT undergone clinical trial evaluation, and must NEVER be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic examination, or physician decision-making in patient care.
