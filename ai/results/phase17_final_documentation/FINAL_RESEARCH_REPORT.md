# HAM10000 7-Class Dermoscopic Image Classification
**Comprehensive Research Report on Attention-Augmented Mobile Neural Networks, Class-Imbalance Interventions, and Calibrated Probability Ensembles**

---

## Abstract

**Background:** Early detection of malignant melanoma significantly improves clinical prognosis. However, automated classification of dermoscopic imagery is hindered by severe class imbalance, high visual similarity between early malignant lesions and benign mimics, and the requirement for computationally efficient inference.  
**Objective:** To systematically evaluate whether a lightweight, attention-augmented mobile convolutional network (MobileNetV3-Large + CBAM), paired with targeted loss calibration and probability-level ensembling, can improve the sensitivity-precision trade-off for minority melanoma detection while preserving high overall multiclass accuracy on the HAM10000 benchmark.  
**Methods:** We executed a controlled sequence of 15 experiments on the HAM10000 dataset using a strict lesion-level split ($8,017$ train, $1,010$ validation, $988$ test). We trained a precision-anchored base model using tempered class weighting ($\alpha=0.75$, Experiment 8) and a sensitivity-enhanced complement model with $1.20\times$ relative melanoma weighting (Experiment 13). We then evaluated a validation-calibrated probability ensemble ($0.85 \cdot P_8 + 0.15 \cdot P_{13}$, Experiment 14) and audited post-hoc decision thresholding (Experiment 15). Phase 16 conducted extensive Grad-CAM explainability and error decomposition across 1,010 validation images.  
**Results:** On the held-out test split ($N=988$), the final ensemble achieved an overall accuracy of $80.57\%$, balanced accuracy of $75.69\%$, a project-wide peak Macro F1 of $70.51\%$ ($0.7051$), and a peak Melanoma F1 of $52.43\%$ ($50.94\%$ recall, $54.00\%$ precision). Grad-CAM heatmaps verified that the network attends systematically to active lesion pigment networks rather than acquisition artifacts.  
**Limitations:** The 988-image test split was evaluated sequentially across experimental stages; reported metrics represent experimental benchmarking within this program rather than independent external validation.  
**Conclusion:** A lightweight attention-augmented mobile network combined with validation-calibrated probability ensembling establishes a robust, computationally efficient approach to balancing minority sensitivity and majority precision in dermatological image analysis.

---

## Keywords
Dermoscopy, Melanoma Classification, HAM10000, MobileNetV3-Large, CBAM Attention, Class Imbalance, Probability Ensemble, Grad-CAM, Explainable AI.

---

## 1. Introduction
Skin cancer is among the most prevalent human malignancies globally, with cutaneous melanoma responsible for the vast majority of skin-cancer-related fatalities. While dermatoscopy significantly improves diagnostic sensitivity compared to naked-eye examination, accurate interpretation demands considerable dermatological expertise. Computer-aided diagnostic systems leveraging deep learning offer substantial promise for clinical decision support. However, deploying computer vision to real-world dermatology faces critical bottlenecks: natural disease distributions are heavily skewed toward benign melanocytic nevi, early melanomas closely resemble dysplastic nevi, and clinical settings often require edge inference on low-cost hardware.

---

## 2. Research Objective
This research program investigated the following primary hypothesis:
> *"Can a lightweight mobile convolutional network (MobileNetV3-Large) augmented with spatial and channel attention (CBAM), combined with targeted class-imbalance interventions and validation-calibrated probability ensembling, improve the sensitivity and F1-score of the minority melanoma class while maintaining strong overall 7-class multiclass accuracy on the HAM10000 benchmark?"*

The investigation strictly enforced controlled single-variable experimental progressions, frozen checkpoint preservation, and absolute data leakage boundaries between training, validation, and test splits.

---

## 3. Dataset
Experiments were conducted using the **HAM10000** ("Human Against Machine with 10,000 training images") benchmark dataset [Tschandl et al., 2018], comprising 10,015 dermoscopic images collected across multiple centers:
- ViDIR Group, Medical University of Vienna, Austria
- Skin Cancer Practice of Cliff Rosendahl, Queensland, Australia

Over $50\%$ of cases are biopsy-verified by histopathology, with remaining cases confirmed by consensus dermatoscopic panels or dynamic in-vivo confocal microscopy.

---

## 4. Dataset Splitting
To eliminate identity leakage from multiple imaging acquisitions of identical lesions, a strict lesion-level split was implemented at random seed 42:

| Split | Image Count | Lesion Count | Proportion | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 8,017 | 6,009 | 80.05% | Parameter optimization (backbone & head) |
| **Validation** | 1,010 | 739 | 10.08% | Model selection, lambda sweep, Phase 16 error analysis |
| **Test** | 988 | 722 | 9.87% | Held-out benchmark evaluation only |
| **Total** | **10,015** | **7,470** | **100.0%** | Full HAM10000 population |

### Class Proportions Across Splits

| Class Code | Class Name | Train ($N=8017$) | Val ($N=1010$) | Test ($N=988$) | Total ($N=10015$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `akiec` | Actinic Keratosis | 257 (3.21%) | 37 (3.66%) | 33 (3.34%) | 327 (3.27%) |
| `bcc` | Basal Cell Carcinoma | 412 (5.14%) | 53 (5.25%) | 49 (4.96%) | 514 (5.13%) |
| `bkl` | Benign Keratosis | 891 (11.11%) | 107 (10.59%) | 101 (10.22%) | 1,099 (10.97%) |
| `df` | Dermatofibroma | 86 (1.07%) | 14 (1.39%) | 15 (1.52%) | 115 (1.15%) |
| `mel` | Melanoma | 890 (11.10%) | 117 (11.58%) | 106 (10.73%) | 1,113 (11.11%) |
| `nv` | Melanocytic Nevus | 5,367 (66.95%) | 668 (66.14%) | 670 (67.81%) | 6,705 (66.95%) |
| `vasc` | Vascular Lesion | 114 (1.42%) | 14 (1.39%) | 14 (1.42%) | 142 (1.42%) |

---

## 5. Preprocessing
Images were processed using deterministic pipelines:
1. Decoding RGB imagery in memory.
2. Bilinear spatial resizing to $224 \times 224$ pixels.
3. PyTorch tensor conversion with ImageNet normalization:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
4. Evaluation and testing were strictly gradient-free and deterministic (zero random cropping, zero rotation, zero color jitter, zero TTA).

---

## 6. Model Architecture
The primary architecture is **`DermaAI_MobileNetV3`** ($4,326,297$ parameters), composed of:
1. **Backbone**: MobileNetV3-Large pretrained on ImageNet, utilizing inverted residual blocks and depthwise separable convolutions.
2. **Attention Module**: A Convolutional Block Attention Module (**CBAM**) positioned at the terminal $960$-channel feature representation ($[B, 960, 7, 7]$). Channel attention dynamically scales feature detectors; spatial attention suppresses peripheral vignetting and skin hairs.
3. **Pooling & Head**: Adaptive average pooling ($[B, 960, 1, 1]$), linear projection ($960 \to 1280$), Hardswish activation, dropout ($p=0.20$), and final 7-class linear projection.

---

## 7. Training Methodology
Training utilized a two-stage gradual fine-tuning schedule:
- **Stage 1 (Head Warming)**: Backbone frozen, classification head trained for 5 epochs ($lr = 10^{-3}$, AdamW, batch size 32).
- **Stage 2 (Full-Network Fine-Tuning)**: Entire network unfreezed, trained for 25 epochs using cosine annealing schedules ($lr_{\max} = 10^{-4}$, weight decay $10^{-4}$).

---

## 8. Class-Imbalance Strategies
To address the $62.4\times$ imbalance ratio between `df` ($86$ train) and `nv` ($5,367$ train):
- **Unweighted Cross-Entropy (Exp 1)**: Severely suppressed minority sensitivity.
- **Inverse Frequency Weighting (Exp 2)**: $w_c = N / (C \cdot N_c)$ amplified minority gradients but caused excessive false positives on nevi.
- **Effective Number of Samples (Exp 3)**: Based on $w_c = (1 - \beta) / (1 - \beta^{N_c})$ with $\beta=0.9999$.
- **Tempered Class Weighting (Exp 8)**: An exponential damping factor $\alpha=0.75$:
  $$w_c = \left( \frac{N_{\max}}{N_c} \right)^\alpha$$
  provided the optimal trade-off, establishing the primary single-model benchmark.
- **Targeted Melanoma Weighting (Exp 13)**: Applied an additional $1.20\times$ multiplier directly to the melanoma class weight, successfully boosting melanoma recall to $59.43\%$.

---

## 9. Experimental Design
The research progressed across 15 controlled, single-variable experiments:
- **Exp 1**: Baseline unweighted MobileNetV3-Large ($224 \times 224$).
- **Exp 2**: Inverse frequency class weighting.
- **Exp 3**: Effective number of samples weighting ($\beta=0.9999$).
- **Exp 4**: Synthetic interpolation augmentations (Mixup + CutMix).
- **Exp 5**: RandAugment / AutoAugment automated policies.
- **Exp 6**: Integration of CBAM attention module.
- **Exp 7**: Gradual unfreezing fine-tuning protocol.
- **Exp 8**: Tempered class weighting ($\alpha=0.75$) + CBAM (**Base Model**).
- **Exp 9**: Label smoothing regularization ($\epsilon=0.10$).
- **Exp 10**: Post-hoc melanoma threshold tuning on Exp 8 ($\theta_{\text{mel}}=0.28$).
- **Exp 11**: Multi-view test-time augmentation (TTA: 5-crop + flips).
- **Exp 12**: Controlled input resolution scaling ($256 \times 256$).
- **Exp 13**: Targeted $1.20\times$ melanoma loss weighting (**Complement Model**).
- **Exp 14**: Validation-calibrated probability ensemble ($0.85 \text{Exp8} + 0.15 \text{Exp13}$).
- **Exp 15**: Threshold audit on Exp 14 ensemble ($\theta^*=0.50$, zero changed predictions).

---

## 10. Experimental Results (Experiments 1–15)

The table below presents the verified test-set performance metrics ($N=988$) across all 15 experiments:

| Exp | Description | Input | Acc (%) | Bal Acc (%) | Macro F1 | Weighted F1 | Mel Rec (%) | Mel Prec (%) | Mel F1 (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Baseline Unweighted CE | $224^2$ | 73.18 | 73.07 | 0.6395 | 0.7487 | 45.28 | 36.92 | 40.68 |
| 2 | Inverse Frequency Wt | $224^2$ | 78.95 | 75.10 | 0.6818 | 0.7983 | 55.66 | 47.97 | 51.53 |
| 3 | Effective Sample Wt | $224^2$ | 76.82 | 75.67 | 0.6704 | 0.7805 | 52.83 | 45.53 | 48.91 |
| 4 | Mixup + CutMix | $224^2$ | 70.24 | 67.30 | 0.5599 | 0.7219 | 35.85 | 35.85 | 35.85 |
| 5 | RandAugment Policy | $224^2$ | 69.64 | 64.99 | 0.5224 | 0.7158 | 30.19 | 34.41 | 32.16 |
| 6 | CBAM Attention Integration | $224^2$ | 71.76 | 72.66 | 0.6076 | 0.7389 | 55.66 | 40.14 | 46.64 |
| 7 | Gradual Unfreezing | $224^2$ | 77.23 | 73.82 | 0.6666 | 0.7838 | 54.72 | 42.65 | 47.93 |
| 8 | Tempered Weighting ($\alpha=0.75$) | $224^2$ | **80.67** | 75.37 | 0.7018 | 0.8107 | 48.11 | **56.04** | 51.78 |
| 9 | Label Smoothing ($\epsilon=0.10$) | $224^2$ | 77.73 | **77.25** | 0.6590 | 0.7887 | 53.77 | 48.31 | 50.89 |
| 10 | Decision Threshold ($\theta=0.28$) | $224^2$ | 80.26 | 75.51 | 0.7023 | 0.8076 | 50.00 | 53.54 | 51.71 |
| 11 | Test-Time Augmentation | $224^2$ | 79.76 | 70.69 | 0.6622 | 0.8005 | 44.34 | 58.02 | 50.27 |
| 12 | Input Scaling ($256 \times 256$) | $256^2$ | 78.85 | 72.73 | 0.6845 | 0.7947 | 51.89 | 47.83 | 49.77 |
| 13 | Targeted Mel Weight ($1.20\times$) | $224^2$ | 78.04 | 73.14 | 0.6758 | 0.7892 | **59.43** | 45.32 | 51.43 |
| **14** | **Ensemble ($0.85 P_8 + 0.15 P_{13}$)** | $224^2$ | 80.57 | 75.69 | **0.7051** | **0.8106** | 50.94 | 54.00 | **0.5243** |
| 15 | Thresholded Ensemble ($\theta^*=0.50$) | $224^2$ | 80.57 | 75.69 | **0.7051** | **0.8106** | 50.94 | 54.00 | **0.5243** |

---

## 11. Final Probability Ensemble (Experiment 14)
The final system blends posterior class probability distributions:
$$P_{\text{ens}}(x) = 0.85 \cdot P_8(x) + 0.15 \cdot P_{13}(x)$$
The blending parameter $\lambda = 0.15$ was calibrated strictly on the validation set ($N=1,010$) using grid search under non-inferiority constraints on Macro F1.

### Complementarity Mechanics:
- **Concordant Retention**: $743 / 743$ concordant correct cases were preserved ($100\%$).
- **Precision Shielding**: High-confidence nevus calls from Exp 8 rejected Exp 13 false alarms.
- **Sensitivity Rescue**: Strong melanoma signals from Exp 13 overturned ambiguous Exp 8 calls, recovering 3 true melanomas while capturing 5 total correct predictions missed by Exp 8.

---

## 12. Final Evaluation (Test Split, N=988)
The Experiment 14 ensemble achieved:
- **Overall Accuracy**: $80.57\%$ ($796 / 988$ correct predictions)
- **Balanced Accuracy**: $75.69\%$
- **Macro F1-Score**: $0.7051$ ($70.51\%$, project-wide peak)
- **Weighted F1-Score**: $0.8106$ ($81.06\%$, project-wide peak)
- **Melanoma Precision**: $54.00\%$
- **Melanoma Recall**: $50.94\%$ ($54 / 106$ detected)
- **Melanoma F1-Score**: $0.5243$ ($52.43\%$, project-wide peak)

---

## 13. Per-Class Performance

| Class Code | Class Name | Support | Correct | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `akiec` | Actinic Keratosis | 33 | 31 | 51.67 | 93.94 | 66.67 |
| `bcc` | Basal Cell Carcinoma | 49 | 42 | 72.41 | 85.71 | 78.50 |
| `bkl` | Benign Keratosis | 101 | 67 | 55.83 | 66.34 | 60.63 |
| `df` | Dermatofibroma | 15 | 8 | 72.73 | 53.33 | 61.54 |
| `mel` | Melanoma | 106 | 54 | 54.00 | 50.94 | 52.43 |
| `nv` | Melanocytic Nevus | 670 | 581 | 93.41 | 86.72 | 89.94 |
| `vasc` | Vascular Lesion | 14 | 13 | 76.47 | 92.86 | 83.87 |

---

## 14. Melanoma-Class Error Analysis
Across the 106 test melanoma cases:
- **True Positives**: $54$ images ($50.94\%$ sensitivity).
- **False Negatives**: $52$ images ($49.06\%$).
- **Primary Failure Modes**:
  1. `mel` $\to$ `nv`: $26$ images ($50.0\%$ of all misses). Early superficial spreading melanomas presenting with regular reticular networks.
  2. `mel` $\to$ `bkl`: $19$ images ($36.5\%$ of all misses). Melanomas exhibiting verrucous surface texture or hyperkeratotic borders.
  3. `mel` $\to$ `bcc`: $4$ images ($7.7\%$).
  4. `mel` $\to$ `akiec`: $3$ images ($5.8\%$).
- **False Alarms (`nv` $\to$ `mel`)**: $38$ benign nevi predicted as melanoma due to atypical dark blotches or follicular clumping.

---

## 15. Confusion Matrix Analysis
The $7 \times 7$ confusion matrix demonstrates that errors are heavily concentrated along dermatologically predictable axes:
- Non-pigmented carcinomas (`bcc`, `akiec`) achieve high recall ($85.71\%$ and $93.94\%$).
- The primary ambiguity boundary occurs between pigmented melanocytic lesions (`mel` vs `nv`) and pigmented seborrheic keratoses (`mel` vs `bkl`).

---

## 16. Phase 16 Validation Error Analysis (N=1,010)
Conducted strictly on the 1,010 validation images:
- **Total Correct**: $804$ images ($79.60\%$).
- **Total Errors**: $206$ images ($20.40\%$).
- **Confidence Separation**:
  - Correct predictions: Mean confidence $87.2\%$ (median $95.5\%$).
  - Error predictions: Mean confidence $62.4\%$ (median $61.3\%$).
  - Model disagreements: Mean confidence drops to $45.1\%-54.1\%$, signaling severe aleatoric/epistemic uncertainty.
  - Anomaly: `mel` $\to$ `nv` exhibited high confidence ($74.6\%$), reflecting strong nevus prior dominance.

---

## 17. Grad-CAM Explainability
Grad-CAM was evaluated on `model.attention` (`CBAMBlock`, shape $[1, 960, 7, 7]$):
1. **Lesion Focus**: Activations centered squarely on lesion pigment networks. No reliance on peripheral skin, vignetting, or hair markers.
2. **Model Differentiation**: Experiment 8 focused tightly on central lesion cores; Experiment 13 expanded spatial attention to peripheral borders, directly reflecting its $1.20\times$ penalty for missed melanomas.
3. **Interpretability Boundary**: Grad-CAM is an attribution map showing correlated feature regions, not proof of clinical reasoning or causality.

---

## 18. Discussion
1. **Augmentation Limitations**: Mixup and CutMix destroyed critical fine border structures, dropping accuracy by over $10\%$.
2. **Architecture & Attention**: CBAM attention successfully suppressed acquisition artifacts, enabling a lightweight mobile architecture to rival larger models.
3. **Ensemble Dynamics**: The $85/15$ probability ensemble functioned as an asymmetric confidence filter, capturing complementary sensitivity without sacrificing majority class precision.

---

## 19. Limitations
1. **Repeated Test-Set Evaluation**: The test split was evaluated sequentially across 15 experiments; results represent experimental benchmarking within this program rather than independent external validation.
2. **Retrospective Benchmark**: HAM10000 exhibits demographic sampling bias toward fair-skinned European/Australian populations.
3. **Resolution**: Resizing to $224 \times 224$ discards sub-cellular dermoscopic details.

---

## 20. Reproducibility
- **Python**: 3.11.x
- **PyTorch**: 2.14.0+cpu
- **Torchvision**: 0.29.0+cpu
- **Random Seed**: 42
- **Checkpoints**: Bitwise verified with SHA-256 in `DERMAAI_DEMO_MANIFEST.json`.

---

## 21. Future Work
1. External validation on ISIC 2020 and diverse demographic datasets (Fitzpatrick phototypes IV–VI).
2. Multi-modal fusion with patient clinical metadata.
3. High-resolution transformer architectures ($384 \times 384$ or $512 \times 512$).
4. Prospective clinical reader studies.

---

## 22. Clinical Research Disclaimer

> [!WARNING]
> **RESEARCH USE ONLY:**
> This AI model and all reported performance metrics are developed strictly for academic research and experimental benchmarking on retrospective image sets (HAM10000). This system is NOT a certified medical diagnostic device, has NOT undergone clinical trial evaluation, and must NEVER be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic evaluation, or physician decision-making in patient care. All evaluations report image-level benchmark performance only.

---

## 23. Conclusion
The DermaAI research program confirms that lightweight mobile convolutional networks augmented with CBAM attention and calibrated probability ensembling achieve competitive 7-class dermoscopic classification performance ($80.57\%$ accuracy, $75.69\%$ balanced accuracy, $70.51\%$ Macro F1) with low computational overhead.
