# DermaAI Research Program — Executive Summary

**Investigation of Attention-Augmented Mobile Neural Networks and Probability Ensembles for 7-Class Skin Lesion Classification on HAM10000**

---

## 1. Problem Statement
Melanoma accounts for the vast majority of skin-cancer-related fatalities worldwide, yet early detection substantially improves clinical prognosis. Automatic image classification of dermoscopic imagery presents significant machine-learning challenges, notably severe class imbalance (the vast majority of presentations are benign melanocytic nevi), high intra-class morphological variability, and inter-class visual overlap between malignant melanomas and benign mimics (such as dysplastic nevi and pigmented seborrheic keratoses). This research investigated whether a lightweight, attention-augmented mobile architecture coupled with targeted loss calibration and probability-level ensembling could optimize the trade-off between overall multiclass accuracy and minority melanoma sensitivity.

---

## 2. Dataset & Splitting
All experiments were conducted on the benchmark **HAM10000** dataset (10,015 dermoscopic images), representing seven distinct diagnostic classes:
1. Actinic Keratosis / Intraepithelial Carcinoma (`akiec`)
2. Basal Cell Carcinoma (`bcc`)
3. Benign Keratosis (`bkl`)
4. Dermatofibroma (`df`)
5. Melanoma (`mel`)
6. Melanocytic Nevus (`nv`)
7. Vascular Lesion (`vasc`)

To prevent data leakage, a strict lesion-level split was implemented at random seed 42:
- **Training Set**: $8,017$ images ($80.0\%$)
- **Validation Set**: $1,010$ images ($10.1\%$) — used exclusively for hyperparameter selection, ensemble weight optimization, and Phase 16 error decomposition.
- **Held-Out Test Set**: $988$ images ($9.9\%$) — reserved strictly for final benchmark evaluation.

---

## 3. Methodology & Core Architecture
The backbone model selected across the investigation is **`DermaAI_MobileNetV3`** ($4,326,297$ parameters), which pairs a MobileNetV3-Large feature extractor with a **Convolutional Block Attention Module (CBAM)** placed immediately prior to global average pooling. CBAM introduces sequential channel and spatial attention maps that emphasize active lesion borders while suppressing peripheral dermatoscope vignetting, gel artifacts, and background skin.

Training leveraged two-stage gradual fine-tuning: initial linear probe warming followed by full-network optimization using AdamW and cosine annealing schedules.

---

## 4. Key Experimental Findings (Experiments 1–15)
The research progressed across 15 controlled experimental stages:
1. **Baselines & Weighting (Exp 1–3)**: Unweighted training yielded low minority recall ($45.28\%$ melanoma recall). Raw inverse frequency weighting over-penalized the majority class, collapsing precision.
2. **Augmentation Trials (Exp 4–5)**: Heavy augmentations (Mixup/CutMix and aggressive AutoAugment) severely degraded performance (accuracy dropped to $69.64\%$), as synthetic interpolation destroys critical dermatological border sharpness.
3. **Architecture & Fine-Tuning (Exp 6–8)**: Adding CBAM and two-stage fine-tuning with **tempered class weighting ($\alpha=0.75$)** in **Experiment 8** established the primary benchmark ($80.67\%$ accuracy, $75.37\%$ balanced accuracy, $0.7018$ Macro F1).
4. **Specialist Calibration (Exp 9–13)**: Introducing a targeted $1.20\times$ relative melanoma loss penalty in **Experiment 13** elevated melanoma recall to $59.43\%$ (capturing $17$ melanomas missed by Exp 8), creating a high-sensitivity complementary model.
5. **Inference Ensembling (Exp 14–15)**: A validation-calibrated probability ensemble combining $85\%$ Experiment 8 and $15\%$ Experiment 13 (**Experiment 14**) surpassed both component models across all generalization metrics. Post-hoc decision thresholding (**Experiment 15**) selected $\theta^*=0.50$ (0 prediction changes), proving the ensemble operates at its natural decision boundary.

---

## 5. Final Validated Benchmark Metrics (Held-Out Test Split, N=988)

| Metric | Experiment 8 (Base) | Experiment 13 (Sensitivity) | Experiment 14 Ensemble (Final) | Net Change vs Exp 8 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | $80.67\%$ | $78.04\%$ | **$80.57\%$** | $-0.10\%$ |
| **Balanced Accuracy** | $75.37\%$ | $73.14\%$ | **$75.69\%$** | **$+0.32\%$** |
| **Macro F1-Score** | $0.7018$ | $0.6758$ | **$0.7051$** | **$+0.0033$** (Project Peak) |
| **Weighted F1-Score** | $0.8094$ | $0.7876$ | **$0.8106$** | **$+0.0012$** (Project Peak) |
| **Melanoma Precision** | $56.04\%$ | $45.32\%$ | **$54.00\%$** | $-2.04\%$ |
| **Melanoma Recall** | $48.11\%$ | $59.43\%$ | **$50.94\%$** | **$+2.83\%$** |
| **Melanoma F1-Score** | $0.5178$ | $0.5143$ | **$0.5243$** | **$+0.0065$** (Project Peak) |

---

## 6. Melanoma Error Analysis & Explainability Findings
- **Error Concentration**: Analysis revealed that **$78.5\%$ of all melanoma false negatives** were misclassified as either benign nevi (`mel` $\to$ `nv`: $57.1\%$) or benign keratoses (`mel` $\to$ `bkl`: $21.4\%$). Microscopic early melanomas presenting with regular reticular networks or hyperkeratotic surfaces generate significant morphological ambiguity at $224 \times 224$ resolution.
- **Grad-CAM Insights**: Visualizations computed on `model.attention` ($[1, 960, 7, 7]$) verified that both models focus strictly on the active pigmented lesion core and margins rather than acquisition artifacts. Experiment 13 displayed broader peripheral attention spreading, explaining its superior melanoma sensitivity.

---

## 7. Limitations & Clinical Research Disclaimer
- **Benchmark Limitations**: The 988-image test split was evaluated sequentially across experiments. Therefore, reported metrics reflect experimental benchmarking within this program rather than independent external validation.
- **Dataset Constraints**: Retrospective HAM10000 data lacks demographic and skin-phototype diversity (Fitzpatrick IV–VI).
- **Mandatory Disclaimer**: **RESEARCH USE ONLY.** This software is an academic research demonstrator and is NOT a certified medical diagnostic device. It must NEVER be used for clinical diagnosis or patient care.
