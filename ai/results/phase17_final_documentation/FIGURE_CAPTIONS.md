# Phase 17 — Figure Captions & Visual Documentation

Comprehensive documentation of all figures, diagnostic matrices, explainability overlays, and comparative charts generated for the DermaAI HAM10000 7-class lesion classification research program.

---

## 1. System Architecture & Methodology

### Figure S1: DermaAI Dual-Model Probability Ensemble Architecture
- **Filename**: `system_architecture.png` (Described in `FINAL_RESEARCH_REPORT.md`)
- **Caption**:
  > **Figure S1. High-level architecture of the frozen DermaAI inference pipeline.** Preprocessed $224 \times 224 \times 3$ dermoscopic images undergo dual forward passes through two frozen MobileNetV3-Large + CBAM neural networks: Experiment 8 (precision-anchored base model, tempered loss weighting $\alpha=0.75$) and Experiment 13 (sensitivity-enhanced complement model, $1.20\times$ targeted melanoma weighting). Independent softmax probabilities $P_8$ and $P_{13}$ are linearly blended using validation-calibrated coefficients ($P_{\text{ens}} = 0.85 \cdot P_8 + 0.15 \cdot P_{13}$). Final predictions are obtained via standard argmax, followed by dual Grad-CAM extraction on the verified CBAM attention modules (`model.attention`, $[1, 960, 7, 7]$).

---

## 2. Benchmark Confusion Matrices (Held-Out Test Set, N=988)

### Figure 1: Final Test Confusion Matrix (Raw Counts)
- **Filename**: [`final_confusion_matrix.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_confusion_matrix.png)
- **Caption**:
  > **Figure 1. Raw multiclass confusion matrix for the Experiment 14 probability ensemble on the held-out HAM10000 test split ($N=988$).** The vertical axis denotes ground-truth biopsy/consensus labels, and the horizontal axis represents predicted classes. The diagonal elements represent correct image-level classifications ($796 / 988$, $80.57\%$ accuracy). Key confusion patterns include $26$ melanomas misclassified as melanocytic nevi (`mel` $\to$ `nv`), $19$ melanomas misclassified as benign keratoses (`mel` $\to$ `bkl`), and $38$ melanocytic nevi misclassified as melanomas (`nv` $\to$ `mel`).

### Figure 2: Final Normalized Test Confusion Matrix (Class Recall)
- **Filename**: [`final_confusion_matrix_normalized.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_confusion_matrix_normalized.png)
- **Caption**:
  > **Figure 2. Row-normalized confusion matrix depicting individual class sensitivity (recall) for the Experiment 14 probability ensemble ($N=988$).** Diagonal cells report empirical recall per class: `akiec` ($93.94\%$), `bcc` ($85.71\%$), `bkl` ($66.34\%$), `df` ($53.33\%$), `mel` ($50.94\%$), `nv` ($86.72\%$), and `vasc` ($92.86\%$). Balanced accuracy across all seven categories is $75.69\%$. Major off-diagonal error concentrations highlight morphological overlap between early melanomas and pigmented benign lesions (`mel` $\to$ `nv`: $24.53\%$; `mel` $\to$ `bkl`: $17.92\%$).

---

## 3. Experiment Comparison Figures (Experiments 1–15)

### Figure 3: Overall Test Accuracy across Experiments
- **Filename**: [`accuracy_by_experiment.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/accuracy_by_experiment.png)
- **Caption**:
  > **Figure 3. Progression of overall test classification accuracy across Experiments 1–15.** Evaluated on the fixed 988-image test split. Peak accuracy was established by Experiment 8 ($80.67\%$) and preserved by the final Experiment 14 probability ensemble ($80.57\%$). Heavy augmentations (Mixup/CutMix in Exp 4: $70.24\%$) and aggressive AutoAugment policies (Exp 5: $69.64\%$) caused significant accuracy degradation.

### Figure 4: Balanced Accuracy across Experiments
- **Filename**: [`balanced_accuracy_by_experiment.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/balanced_accuracy_by_experiment.png)
- **Caption**:
  > **Figure 4. Progression of unweighted balanced accuracy across Experiments 1–15.** Measures mean recall across all 7 categories. Experiment 9 (label smoothing) achieved $77.25\%$, while the final Experiment 14 ensemble achieved $75.69\%$ (a $+0.32\%$ increase over Experiment 8's $75.37\%$), demonstrating effective minority class preservation without degrading majority nevus representation.

### Figure 5: Macro F1-Score across Experiments
- **Filename**: [`macro_f1_by_experiment.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/macro_f1_by_experiment.png)
- **Caption**:
  > **Figure 5. Unweighted Macro F1-score across all 15 research configurations.** The final Experiment 14 ensemble attained the project-wide peak Macro F1 of $70.51\%$ ($0.7051$), successfully reconciling high precision on the dominant `nv` class with improved recall across minority classes (`mel`, `akiec`, `bcc`, `vasc`).

### Figure 6: Melanoma Recall across Experiments
- **Filename**: [`melanoma_recall_by_experiment.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/melanoma_recall_by_experiment.png)
- **Caption**:
  > **Figure 6. Test-set sensitivity (recall) for the melanoma class ($N=106$).** Demonstrates the operational trade-off between baseline precision and minority recall. Targeted melanoma weighting in Experiment 13 peaked at $59.43\%$ recall (at the expense of lower precision, $45.32\%$). The final Experiment 14 ensemble calibrated this trade-off to $50.94\%$ recall while recovering precision to $54.00\%$.

### Figure 7: Melanoma F1-Score across Experiments
- **Filename**: [`melanoma_f1_by_experiment.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/melanoma_f1_by_experiment.png)
- **Caption**:
  > **Figure 7. Harmonic mean of melanoma precision and recall (Melanoma F1) across Experiments 1–15.** Experiment 14 achieved the project-wide peak Melanoma F1 of $52.43\%$ ($0.5243$), outperforming both individual component models (Exp 8: $51.78\%$; Exp 13: $51.43\%$) through constructive probability blending.

---

## 4. Phase 16 Explainability & Error Analysis Figures (Validation Split, N=1,010)

*Note: The following publication figures were generated during Phase 16 on the 1,010-image validation split to ensure zero test-set contamination.*

### Figure 8: Correct Melanoma True Positives (Figure 1 in Phase 16)
- **Filename**: `ai/results/phase16_explainability/figures/figure1_correct_melanoma.png`
- **Caption**:
  > **Figure 8. Grad-CAM visual attributions for correctly classified melanoma true positives.** Displays four representative biopsy-confirmed cases (`ISIC_0026210`, `ISIC_0027732`, `ISIC_0026645`, `ISIC_0028214`). Both Experiment 8 and Experiment 13 focus strongly on central and eccentric atypical pigment networks. Experiment 13 demonstrates broader peripheral attention spreading to irregular borders, reflecting its higher training penalty for missed melanomas.

### Figure 9: Melanoma Misclassified as Nevus (Figure 2 in Phase 16)
- **Filename**: `ai/results/phase16_explainability/figures/figure2_melanoma_to_nv_errors.png`
- **Caption**:
  > **Figure 9. Qualitative inspection of the primary clinical risk mode: Melanoma $\to$ Nevus errors (`mel` $\to$ `nv`).** Displays four false-negative cases (`ISIC_0026857`, `ISIC_0024763`, `ISIC_0028852`, `ISIC_0027982`). Lesions exhibit regular, symmetric reticular pigment networks that visually mimic benign junctional nevi, causing high-confidence misclassifications (mean ensemble confidence: $74.6\%$).

### Figure 10: Melanoma Misclassified as Benign Keratosis (Figure 3 in Phase 16)
- **Filename**: `ai/results/phase16_explainability/figures/figure3_melanoma_to_bkl_errors.png`
- **Caption**:
  > **Figure 10. Morphological confusion in Melanoma $\to$ Benign Keratosis errors (`mel` $\to$ `bkl`).** Displays four cases (`ISIC_0025945`, `ISIC_0024948`, `ISIC_0027419`, `ISIC_0025680`). Spatial attention localizes on hyperkeratotic scales, verrucous borders, and pseudo-follicular plugs that overlap dermoscopically with solar lentigines and seborrheic keratoses.

### Figure 11: Nevus Misclassified as Melanoma (Figure 4 in Phase 16)
- **Filename**: `ai/results/phase16_explainability/figures/figure4_nv_to_melanoma_false_positives.png`
- **Caption**:
  > **Figure 11. False-alarm analysis: Benign Nevi predicted as Melanoma (`nv` $\to$ `mel`).** Displays four cases (`ISIC_0031804`, `ISIC_0024754`, `ISIC_0031383`, `ISIC_0024823`). Focal dark globules, localized atypical blotches, or hair follicles generate strong localized gradient attributions that push the ensemble probability above the decision boundary.

### Figure 12: Model Disagreement Dynamics (Figure 5 in Phase 16)
- **Filename**: `ai/results/phase16_explainability/figures/figure5_exp8_vs_exp13_disagreements.png`
- **Caption**:
  > **Figure 12. Spatial attribution divergence in model disagreement scenarios.** Displays four conflict cases (`ISIC_0025076`, `ISIC_0029687`, `ISIC_0029417`, `ISIC_0029274`). Demonstrates how Experiment 8's compact core focus contrasts with Experiment 13's peripheral margin sensitivity, illuminating the mechanisms behind the $85/15$ probability arbitration.
