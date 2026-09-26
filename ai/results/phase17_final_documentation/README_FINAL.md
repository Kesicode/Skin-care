# DermaAI: Final Research Documentation (HAM10000 7-Class Classification)

This directory (`ai/results/phase17_final_documentation/`) contains the complete scientific reports, data tables, confusion matrices, and explainability records for the DermaAI machine learning research program.

---

## 1. Project Overview & Research Question

**Research Question:**
> *"Can a lightweight attention-augmented mobile convolutional network (MobileNetV3-Large + CBAM), paired with targeted loss-weighting interventions and validation-calibrated probability ensembling, effectively mitigate severe dermatological class imbalance and improve minority melanoma sensitivity without compromising overall 7-class multiclass accuracy on the HAM10000 benchmark?"*

**Answer:**
**Yes.** Combining a precision-anchored base model (Experiment 8, tempered weighting $\alpha=0.75$) with a sensitivity-enhanced complement model (Experiment 13, $1.20\times$ melanoma loss weight) via a validation-calibrated probability ensemble ($0.85 \cdot P_8 + 0.15 \cdot P_{13}$, Experiment 14) established the project-wide peak Macro F1 ($70.51\%$), Balanced Accuracy ($75.69\%$), and Melanoma F1 ($52.43\%$) on the held-out test split ($N=988$).

---

## 2. Directory Contents

| Filename | Description |
| :--- | :--- |
| [`FINAL_RESEARCH_REPORT.md`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/FINAL_RESEARCH_REPORT.md) | Exhaustive 23-section comprehensive scientific report |
| [`FINAL_IEEE_PAPER.md`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/FINAL_IEEE_PAPER.md) | Formal academic conference paper formatted to IEEE standards |
| [`EXECUTIVE_SUMMARY.md`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/EXECUTIVE_SUMMARY.md) | High-level 2-page research brief for decision makers |
| [`VIVA_PREPARATION.md`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/VIVA_PREPARATION.md) | Technical defense guide covering 15 core architectural questions |
| [`FIGURE_CAPTIONS.md`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/FIGURE_CAPTIONS.md) | Detailed descriptions and citations for all generated figures |
| [`documentation_manifest.json`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/documentation_manifest.json) | Machine-readable metadata and verification records |
| [`dataset_distribution.csv`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/dataset_distribution.csv) | Exact class counts and proportions across train, val, and test splits |
| [`experiment_comparison_table.csv`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/experiment_comparison_table.csv) | Full metrics across all 15 controlled research experiments |
| [`final_results_table.csv`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_results_table.csv) | Focused benchmark comparison between Exp 8, Exp 13, and Exp 14 |
| [`final_per_class_metrics.csv`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_per_class_metrics.csv) | Precision, recall, and F1 across all 7 classes for the final ensemble |
| [`final_confusion_matrix.csv`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_confusion_matrix.csv) | Raw multiclass confusion matrix array ($7 \times 7$) on test split |
| [`melanoma_analysis.csv`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/melanoma_analysis.csv) | Quantitative tracking of melanoma detection and false-negative routes |
| [`final_confusion_matrix.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_confusion_matrix.png) | High-resolution raw test confusion matrix heatmap |
| [`final_confusion_matrix_normalized.png`](file:///c:/Users/kesi/Downloads/Skin-care/ai/results/phase17_final_documentation/final_confusion_matrix_normalized.png) | Normalized test confusion matrix heatmap (class recall) |
| `*_by_experiment.png` | 5 comparative trend plots across Experiments 1–15 |

---

## 3. Key Benchmark Metrics (Held-Out Test Split, N=988)

- **Accuracy**: $80.57\%$ ($796 / 988$ correct)
- **Balanced Accuracy**: $75.69\%$
- **Macro F1-Score**: $0.7051$ (Peak)
- **Weighted F1-Score**: $0.8106$ (Peak)
- **Melanoma Precision**: $54.00\%$
- **Melanoma Recall**: $50.94\%$ ($54 / 106$ detected)
- **Melanoma F1-Score**: $0.5243$ (Peak)

---

## 4. Population Separation Protocol

To prevent methodological confusion and data leakage:
- **Final Test Benchmark ($N=988$)**: The official performance metrics above reflect the project test split (`ai/dataset/splits/test.csv`).
- **Phase 16 Error & Explainability Analysis ($N=1,010$)**: Detailed Grad-CAM case selections, error category distributions, and confidence analyses were performed on the separate validation split (`ai/dataset/splits/val.csv`).
- **Zero Test Tuning**: The test set was never used for hyperparameter selection, threshold tuning, or ensemble weighting.

---

## 5. Clinical Research Disclaimer

> [!WARNING]
> **RESEARCH USE ONLY:**
> This AI model and all reported performance metrics are developed strictly for academic research and experimental benchmarking on retrospective image sets (HAM10000). This system is NOT a certified medical diagnostic device, has NOT undergone clinical trial evaluation, and must NEVER be used as a substitute for professional clinical diagnosis, biopsy, dermoscopic evaluation, or physician decision-making in patient care. All evaluations report image-level benchmark performance only.
