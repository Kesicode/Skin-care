# Attention-Augmented Mobile Neural Networks and Validation-Calibrated Probability Ensembling for 7-Class Skin Lesion Classification on HAM10000

---

## Abstract

**Background:** Dermatological image classification presents significant clinical and machine-learning challenges, characterized by extreme class imbalance, subtle inter-class boundaries between malignant melanomas and benign mimics, and the requirement for computationally efficient inference at point-of-care.  
**Objective:** To systematically evaluate whether a lightweight convolutional architecture augmented with spatial-channel attention, coupled with targeted loss calibration and probability-level ensembling, can improve the sensitivity-precision trade-off for minority melanoma detection while preserving high overall multiclass accuracy on the HAM10000 benchmark.  
**Methods:** We implemented a controlled 15-stage experimental program utilizing `DermaAI_MobileNetV3` ($4,326,297$ parameters), integrating a MobileNetV3-Large backbone with a Convolutional Block Attention Module (CBAM). Using a lesion-level split of the HAM10000 dataset ($8,017$ train, $1,010$ validation, $988$ test), we trained a precision-anchored base model using tempered class weighting ($\alpha=0.75$, Experiment 8) and a sensitivity-enhanced complement model using targeted $1.20\times$ melanoma weighting (Experiment 13). A probability ensemble was calibrated strictly on validation data ($\lambda=0.15$, Experiment 14). Grad-CAM explainability and error decomposition were audited across 1,010 validation cases.  
**Results:** On the held-out test split ($N=988$), the final ensemble achieved an overall accuracy of $80.57\%$, balanced accuracy of $75.69\%$, and a project-wide peak Macro F1 of $70.51\%$ ($0.7051$). Melanoma sensitivity reached $50.94\%$ with $54.00\%$ precision (Melanoma F1 of $52.43\%$), outperforming both individual component models. Qualitative Grad-CAM confirmed attention localization on active lesion tissue rather than peripheral artifacts.  
**Limitations:** The project test split was evaluated sequentially across experimental stages. Results represent retrospective experimental benchmarking on HAM10000 without independent external or prospective clinical validation.  
**Conclusion:** A lightweight attention-augmented mobile network combined with validation-calibrated probability ensembling provides an effective, computationally efficient approach to balancing minority sensitivity and majority precision in dermatological image analysis.

---

## Index Terms
Dermoscopy, Skin Lesion Classification, Deep Learning, MobileNetV3, CBAM, Class Imbalance, Ensemble Learning, Grad-CAM, HAM10000.

---

## I. Introduction
Malignant melanoma accounts for approximately $75\%$ of all skin cancer deaths despite representing a minority of diagnoses [REFERENCE TO BE ADDED]. Early detection via dermoscopy dramatically improves survival rates, but clinical examination requires extensive specialized training and exhibits inter-observer variability [REFERENCE TO BE ADDED]. While deep convolutional neural networks (CNNs) have shown promise in automated dermoscopic image analysis, clinical translation is constrained by three fundamental hurdles:

1. **Extreme Class Imbalance**: In natural clinical prevalence and curated benchmarks such as HAM10000 [Tschandl et al., 2018], benign melanocytic nevi outnumber melanomas and rare dermatofibromas by orders of magnitude. Standard empirical risk minimization severely suppresses minority sensitivity.
2. **High Inter-Class Morphological Similarity**: Early superficial spreading melanomas frequently exhibit regular reticular pigment networks mimicking benign junctional nevi, while pigmented seborrheic keratoses overlap visually with nodular melanomas.
3. **Deployment Constraints**: Clinical point-of-care applications demand lightweight, CPU-compatible architectures capable of low-latency local execution without relying on multi-gigabyte server models.

This study presents a systematic investigation of `DermaAI_MobileNetV3`—a lightweight mobile architecture augmented with the Convolutional Block Attention Module (CBAM) [Woo et al., 2018]—combined with targeted class-imbalance strategies and probability-level ensembling on the HAM10000 benchmark.

---

## II. Dataset and Materials

### A. HAM10000 Benchmark
The HAM10000 dataset comprises $10,015$ high-resolution dermoscopic images collected from the Medical University of Vienna and Cliff Rosendahl's clinic in Queensland, Australia [Tschandl et al., 2018]. More than $50\%$ of lesions are validated by histopathology, with the remainder confirmed through expert dermatoscopic consensus or dynamic in-vivo confocal microscopy.

The dataset spans seven diagnostic categories:
- `akiec`: Actinic Keratoses and Intraepithelial Carcinoma ($n=327$, $3.27\%$)
- `bcc`: Basal Cell Carcinoma ($n=514$, $5.13\%$)
- `bkl`: Benign Keratosis ($n=1,099$, $10.97\%$)
- `df`: Dermatofibroma ($n=115$, $1.15\%$)
- `mel`: Melanoma ($n=1,113$, $11.11\%$)
- `nv`: Melanocytic Nevus ($n=6,705$, $66.95\%$)
- `vasc`: Vascular Lesion ($n=142$, $1.42\%$)

### B. Lesion-Level Data Partitioning
To prevent data contamination from multiple imaging angles of the same physical lesion, a strict lesion-level split was implemented at random seed 42:
- **Training Set**: $8,017$ images ($80.05\%$)
- **Validation Set**: $1,010$ images ($10.08\%$)
- **Test Set**: $988$ images ($9.87\%$)

The test split remained strictly sequestered during model training and hyperparameter optimization.

---

## III. Methodology

### A. Model Architecture: DermaAI_MobileNetV3
The model employs a MobileNetV3-Large backbone [Howard et al., 2019] pretrained on ImageNet, featuring inverted bottleneck residual blocks, linear bottlenecks, and Squeeze-and-Excitation channel gating. 

Immediately preceding global spatial pooling, a **Convolutional Block Attention Module (CBAM)** [Woo et al., 2018] is inserted into the 960-channel feature representation:
$$\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$$
$$\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \otimes \mathbf{F}'$$

- **Channel Sub-module**: Uses shared multi-layer perceptron (MLP) reduction ($r=16$) on dual max-pooled and average-pooled descriptors:
  $$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right)$$
- **Spatial Sub-module**: Concatenates inter-channel max and average projections across a $7 \times 7$ convolution:
  $$\mathbf{M}_s(\mathbf{F}') = \sigma\left(f^{7\times 7}\left([\text{AvgPool}(\mathbf{F}'); \text{MaxPool}(\mathbf{F}')]\right)\right)$$

The classification head consists of adaptive average pooling to $[1, 1]$, a fully connected linear layer ($960 \to 1280$), Hardswish activation, dropout ($p=0.20$), and a final 7-class linear projection. Total parameters: **4,326,297**.

### B. Preprocessing & Two-Stage Training
All images undergo deterministic bilinear resizing to $224 \times 224$ pixels and standard ImageNet channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

Training proceeds in two stages:
1. **Stage 1 (Head Warming)**: Backbone frozen, classification head trained for 5 epochs ($lr = 10^{-3}$, AdamW).
2. **Stage 2 (Fine-Tuning)**: Entire network unfreezed, trained for 25 epochs using cosine annealing ($lr_{\max} = 10^{-4}$, weight decay $10^{-4}$).

---

## IV. Experimental Design

Fifteen controlled experiments were executed sequentially:
1. **Exp 1**: Unweighted Cross-Entropy baseline.
2. **Exp 2**: Standard inverse class frequency weighting ($w_c = N / (C \cdot N_c)$).
3. **Exp 3**: Class-balanced focal weighting based on effective number of samples ($\beta=0.9999$).
4. **Exp 4**: Synthetic interpolation augmentations (Mixup + CutMix).
5. **Exp 5**: Automated augmentation policies (RandAugment).
6. **Exp 6**: Integration of CBAM attention module.
7. **Exp 7**: Gradual unfreezing fine-tuning protocol.
8. **Exp 8**: Tempered class weighting ($\alpha=0.75$) with CBAM and gradual unfreezing (**Base Model**).
9. **Exp 9**: Label smoothing regularization ($\epsilon=0.10$).
10. **Exp 10**: Post-hoc melanoma threshold tuning on Exp 8 ($\theta_{\text{mel}}=0.28$).
11. **Exp 11**: Multi-view test-time augmentation (TTA: 5-crop + flips).
12. **Exp 12**: Controlled input resolution scaling ($256 \times 256$).
13. **Exp 13**: Targeted $1.20\times$ relative melanoma weighting with $\alpha=0.75$ (**Complement Model**).
14. **Exp 14**: Validation-calibrated probability ensemble ($0.85 \text{Exp8} + 0.15 \text{Exp13}$).
15. **Exp 15**: Threshold audit on Exp 14 ensemble ($\theta^*=0.50$, zero changed predictions).

---

## V. Results

### A. Comparative Experimental Benchmarks
Table I reports the performance of all 15 experimental configurations evaluated on the held-out test split ($N=988$).

**TABLE I: Comprehensive Performance Summary (Experiments 1–15, N=988)**

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

### B. Final Per-Class Performance
Table II provides the exact per-class metrics for the final Experiment 14 ensemble.

**TABLE II: Final Per-Class Test Performance (Experiment 14 Ensemble, N=988)**

| Class Code | Class Name | Support | Correct | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `akiec` | Actinic Keratosis | 33 | 31 | 51.67 | 93.94 | 66.67 |
| `bcc` | Basal Cell Carcinoma | 49 | 42 | 72.41 | 85.71 | 78.50 |
| `bkl` | Benign Keratosis | 101 | 67 | 55.83 | 66.34 | 60.63 |
| `df` | Dermatofibroma | 15 | 8 | 72.73 | 53.33 | 61.54 |
| `mel` | Melanoma | 106 | 54 | 54.00 | 50.94 | 52.43 |
| `nv` | Melanocytic Nevus | 670 | 581 | 93.41 | 86.72 | 89.94 |
| `vasc` | Vascular Lesion | 14 | 13 | 76.47 | 92.86 | 83.87 |
| **Macro** | **Unweighted Average** | — | — | 68.07 | 75.69 | **70.51** |
| **Weighted** | **Prevalence Weighted** | — | — | 82.35 | 80.57 | **81.06** |

---

## VI. Error Analysis and Explainability

### A. Failure-Mode Decomposition
Detailed analysis of the 192 test errors (and the 1,010 validation cases in Phase 16) established three primary error clusters:
1. **`mel` $\to$ `nv` ($26$ test cases, $24.5\%$ of melanomas)**: Early superficial spreading melanomas presenting with regular reticular pigment networks, misclassified with high model confidence ($74.6\%$).
2. **`mel` $\to$ `bkl` ($19$ test cases, $17.9\%$ of melanomas)**: Melanomas exhibiting hyperkeratosis, verrucous borders, and pseudo-follicular plugs.
3. **`nv` $\to$ `mel` ($38$ test cases)**: Dysplastic nevi with prominent central blotches or atypical follicular structures triggering false alarms.

### B. Grad-CAM Qualitative Inspection
Grad-CAM was evaluated on the verified convolutional feature layer `model.attention` ($[1, 960, 7, 7]$). Across all categories, heatmaps placed peak attribution strictly on active lesion tissue and irregular pigment borders rather than peripheral skin, dark corners, or dermatoscope borders. Experiment 13 consistently exhibited wider attention spreads to lesion margins, directly reflecting its $1.20\times$ penalty on false negatives.

---

## VII. Discussion
Our findings demonstrate that aggressive augmentation strategies (Mixup/CutMix in Exp 4: $70.24\%$ accuracy) and heavy automated policies (Exp 5: $69.64\%$) are counterproductive in dermoscopy, where structural sharpness and color calibration represent critical diagnostic features. 

Conversely, combining gradual unfreezing, CBAM attention, and tempered weighting ($\alpha=0.75$) enabled MobileNetV3-Large to establish an $80.67\%$ baseline accuracy. Most notably, the probability ensemble ($85/15$) acted as an asymmetric confidence filter: when Experiment 8 was highly confident in a benign nevus, it rejected Experiment 13's mild sensitivity bias, whereas in borderline cases, strong melanoma activations from Experiment 13 successfully overturned ambiguous calls, driving peak overall Macro F1 ($70.51\%$) and peak Melanoma F1 ($52.43\%$).

---

## VIII. Limitations

1. **Repeated Test-Set Evaluation**: The 988-image test split was evaluated sequentially across Experiments 1–15. Although hyperparameters and blending weights were selected exclusively on the validation split, the benchmark reflects experimental development within this project rather than independent external validation.
2. **Retrospective HAM10000 Data**: The dataset exhibits demographic sampling bias toward fair-skinned European/Australian populations and standardized dermatoscope optics.
3. **Coarse Resolution**: $7 \times 7$ feature maps upsampled to $224 \times 224$ provide coarse localization rather than sub-cellular boundary segmentation.

---

## IX. Conclusion
The DermaAI research program confirms that lightweight mobile neural networks augmented with CBAM attention and calibrated probability ensembling achieve competitive 7-class dermoscopic classification performance ($80.57\%$ accuracy, $75.69\%$ balanced accuracy, $70.51\%$ Macro F1) with low computational overhead. Post-hoc decision thresholding confirmed that the ensemble functions optimally at its natural argmax decision boundary.

---

## References

1. P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, no. 1, p. 180161, 2018.
2. A. Howard et al., "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1314–1324.
3. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 3–19.
4. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.
5. [REFERENCE TO BE ADDED]
6. [REFERENCE TO BE ADDED]

---

> [!WARNING]
> **CLINICAL RESEARCH DISCLAIMER:**
> This AI model and all reported performance metrics are developed strictly for academic research and experimental benchmarking on retrospective image sets (HAM10000). This system is NOT a certified medical diagnostic device, has NOT undergone clinical trial evaluation, and must NEVER be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic evaluation, or physician decision-making in patient care. All evaluations report image-level benchmark performance only.
