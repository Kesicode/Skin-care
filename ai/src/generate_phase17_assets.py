"""
Generate all tabular data, confusion matrices, and comparison figures
for Phase 17 — Final Research Documentation.
Strictly relies on existing verified artifacts. Zero model training.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_DIR = os.path.join(PROJECT_ROOT, "ai")
RESULTS_DIR = os.path.join(AI_DIR, "results")
DOCS_DIR = os.path.join(RESULTS_DIR, "phase17_final_documentation")
os.makedirs(DOCS_DIR, exist_ok=True)

CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus",
    "Vascular Lesion"
]
CLASS_MAP = dict(zip(CLASS_CODES, CLASS_NAMES))

print(f"[Phase 17] Output directory: {DOCS_DIR}")

# ==============================================================================
# 1. Dataset Distribution Table
# ==============================================================================
train_df = pd.read_csv(os.path.join(AI_DIR, "dataset", "splits", "train.csv"))
val_df = pd.read_csv(os.path.join(AI_DIR, "dataset", "splits", "val.csv"))
test_df = pd.read_csv(os.path.join(AI_DIR, "dataset", "splits", "test.csv"))

n_train = len(train_df)
n_val = len(val_df)
n_test = len(test_df)
n_total = n_train + n_val + n_test

train_counts = train_df["dx"].value_counts().to_dict()
val_counts = val_df["dx"].value_counts().to_dict()
test_counts = test_df["dx"].value_counts().to_dict()

dist_rows = []
for code, name in zip(CLASS_CODES, CLASS_NAMES):
    tr = train_counts.get(code, 0)
    va = val_counts.get(code, 0)
    te = test_counts.get(code, 0)
    tot = tr + va + te
    dist_rows.append({
        "class_code": code,
        "class_name": name,
        "train_count": tr,
        "train_pct": round(tr / n_train * 100, 2),
        "val_count": va,
        "val_pct": round(va / n_val * 100, 2),
        "test_count": te,
        "test_pct": round(te / n_test * 100, 2),
        "total_count": tot,
        "total_pct": round(tot / n_total * 100, 2),
    })

# Add Total row
dist_rows.append({
    "class_code": "TOTAL",
    "class_name": "All Classes",
    "train_count": n_train,
    "train_pct": 100.0,
    "val_count": n_val,
    "val_pct": 100.0,
    "test_count": n_test,
    "test_pct": 100.0,
    "total_count": n_total,
    "total_pct": 100.0,
})

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(os.path.join(DOCS_DIR, "dataset_distribution.csv"), index=False)
print("  Generated dataset_distribution.csv")

# ==============================================================================
# 2. Confusion Matrices (Experiment 14 Final Benchmark)
# ==============================================================================
with open(os.path.join(RESULTS_DIR, "experiment14", "confusion_matrix_ensemble.json")) as f:
    cm_data = json.load(f)

cm_matrix = cm_data["confusion_matrix"] if isinstance(cm_data, dict) and "confusion_matrix" in cm_data else cm_data
cm = np.array(cm_matrix)
cm_df = pd.DataFrame(cm, index=CLASS_CODES, columns=CLASS_CODES)
cm_df.to_csv(os.path.join(DOCS_DIR, "final_confusion_matrix.csv"))
print("  Generated final_confusion_matrix.csv")

# Raw Confusion Matrix Figure
plt.figure(figsize=(8, 6.5), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_CODES, yticklabels=CLASS_CODES,
            cbar_kws={'label': 'Sample Count'})
plt.title("DermaAI Experiment 14 Ensemble — Test Confusion Matrix (N=988)", fontsize=12, pad=12)
plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "final_confusion_matrix.png"))
plt.close()
print("  Generated final_confusion_matrix.png")

# Normalized Confusion Matrix Figure (recall per class)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6.5), dpi=300)
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=CLASS_CODES, yticklabels=CLASS_CODES,
            cbar_kws={'label': 'Normalized Recall'})
plt.title("DermaAI Experiment 14 Ensemble — Normalized Test Confusion Matrix (N=988)", fontsize=12, pad=12)
plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "final_confusion_matrix_normalized.png"))
plt.close()
print("  Generated final_confusion_matrix_normalized.png")

# ==============================================================================
# 3. Final Per-Class Metrics Table
# ==============================================================================
with open(os.path.join(RESULTS_DIR, "experiment14", "test_metrics_ensemble.json")) as f:
    exp14_test = json.load(f)

per_class_rows = []
for code, name in zip(CLASS_CODES, CLASS_NAMES):
    pc = exp14_test["per_class"][code]
    per_class_rows.append({
        "class_code": code,
        "class_name": name,
        "support": pc["support"],
        "correct": pc["correct"],
        "precision": round(pc["precision"] * 100, 2),
        "recall": round(pc["recall"] * 100, 2),
        "f1": round(pc["f1-score"] * 100, 2),
    })

per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(os.path.join(DOCS_DIR, "final_per_class_metrics.csv"), index=False)
print("  Generated final_per_class_metrics.csv")

# ==============================================================================
# 4. Complete Experiment Comparison Table (Experiments 1–15)
# ==============================================================================
exp_history = [
    {
        "Experiment": "Exp 1",
        "Configuration": "Baseline MobileNetV3-Large, unweighted CE",
        "Input": "224x224",
        "Accuracy": 73.18, "Balanced_Accuracy": 73.07, "Macro_F1": 63.95, "Weighted_F1": 74.87,
        "Melanoma_Precision": 36.92, "Melanoma_Recall": 45.28, "Melanoma_F1": 40.68
    },
    {
        "Experiment": "Exp 2",
        "Configuration": "Inverse frequency class weighting",
        "Input": "224x224",
        "Accuracy": 78.95, "Balanced_Accuracy": 75.10, "Macro_F1": 68.18, "Weighted_F1": 79.83,
        "Melanoma_Precision": 47.97, "Melanoma_Recall": 55.66, "Melanoma_F1": 51.53
    },
    {
        "Experiment": "Exp 3",
        "Configuration": "Effective number of samples weighting (beta=0.9999)",
        "Input": "224x224",
        "Accuracy": 76.82, "Balanced_Accuracy": 75.67, "Macro_F1": 67.04, "Weighted_F1": 78.05,
        "Melanoma_Precision": 45.53, "Melanoma_Recall": 52.83, "Melanoma_F1": 48.91
    },
    {
        "Experiment": "Exp 4",
        "Configuration": "Heavy augmentation (Mixup + CutMix)",
        "Input": "224x224",
        "Accuracy": 70.24, "Balanced_Accuracy": 67.30, "Macro_F1": 55.99, "Weighted_F1": 72.19,
        "Melanoma_Precision": 35.85, "Melanoma_Recall": 35.85, "Melanoma_F1": 35.85
    },
    {
        "Experiment": "Exp 5",
        "Configuration": "AutoAugment / RandAugment policy",
        "Input": "224x224",
        "Accuracy": 69.64, "Balanced_Accuracy": 64.99, "Macro_F1": 52.24, "Weighted_F1": 71.58,
        "Melanoma_Precision": 34.41, "Melanoma_Recall": 30.19, "Melanoma_F1": 32.16
    },
    {
        "Experiment": "Exp 6",
        "Configuration": "CBAM attention block integration",
        "Input": "224x224",
        "Accuracy": 71.76, "Balanced_Accuracy": 72.66, "Macro_F1": 60.76, "Weighted_F1": 73.89,
        "Melanoma_Precision": 40.14, "Melanoma_Recall": 55.66, "Melanoma_F1": 46.64
    },
    {
        "Experiment": "Exp 7",
        "Configuration": "Gradual unfreezing fine-tuning (backbone stages)",
        "Input": "224x224",
        "Accuracy": 77.23, "Balanced_Accuracy": 73.82, "Macro_F1": 66.66, "Weighted_F1": 78.38,
        "Melanoma_Precision": 42.65, "Melanoma_Recall": 54.72, "Melanoma_F1": 47.93
    },
    {
        "Experiment": "Exp 8",
        "Configuration": "Tempered class weighting (alpha=0.75) + CBAM",
        "Input": "224x224",
        "Accuracy": 80.67, "Balanced_Accuracy": 75.37, "Macro_F1": 70.18, "Weighted_F1": 81.07,
        "Melanoma_Precision": 56.04, "Melanoma_Recall": 48.11, "Melanoma_F1": 51.78
    },
    {
        "Experiment": "Exp 9",
        "Configuration": "Label smoothing regularization (eps=0.10)",
        "Input": "224x224",
        "Accuracy": 77.73, "Balanced_Accuracy": 77.25, "Macro_F1": 65.90, "Weighted_F1": 78.87,
        "Melanoma_Precision": 48.31, "Melanoma_Recall": 53.77, "Melanoma_F1": 50.89
    },
    {
        "Experiment": "Exp 10",
        "Configuration": "Post-hoc decision threshold tuning on Exp 8 (theta=0.28)",
        "Input": "224x224",
        "Accuracy": 80.26, "Balanced_Accuracy": 75.51, "Macro_F1": 70.23, "Weighted_F1": 80.76,
        "Melanoma_Precision": 53.54, "Melanoma_Recall": 50.00, "Melanoma_F1": 51.71
    },
    {
        "Experiment": "Exp 11",
        "Configuration": "Deterministic Test-Time Augmentation (TTA) on Exp 8",
        "Input": "224x224",
        "Accuracy": 79.76, "Balanced_Accuracy": 70.69, "Macro_F1": 66.22, "Weighted_F1": 80.05,
        "Melanoma_Precision": 58.02, "Melanoma_Recall": 44.34, "Melanoma_F1": 50.27
    },
    {
        "Experiment": "Exp 12",
        "Configuration": "Controlled input resolution experiment (256x256)",
        "Input": "256x256",
        "Accuracy": 78.85, "Balanced_Accuracy": 72.73, "Macro_F1": 68.45, "Weighted_F1": 79.47,
        "Melanoma_Precision": 47.83, "Melanoma_Recall": 51.89, "Melanoma_F1": 49.77
    },
    {
        "Experiment": "Exp 13",
        "Configuration": "Targeted 1.20x melanoma weight (tempered alpha=0.75)",
        "Input": "224x224",
        "Accuracy": 78.04, "Balanced_Accuracy": 73.14, "Macro_F1": 67.58, "Weighted_F1": 78.92,
        "Melanoma_Precision": 45.32, "Melanoma_Recall": 59.43, "Melanoma_F1": 51.43
    },
    {
        "Experiment": "Exp 14",
        "Configuration": "Validation-calibrated probability ensemble (0.85 Exp8 + 0.15 Exp13)",
        "Input": "224x224",
        "Accuracy": 80.57, "Balanced_Accuracy": 75.69, "Macro_F1": 70.51, "Weighted_F1": 81.06,
        "Melanoma_Precision": 54.00, "Melanoma_Recall": 50.94, "Melanoma_F1": 52.43
    },
    {
        "Experiment": "Exp 15",
        "Configuration": "Validation-calibrated threshold on Exp14 ensemble (theta=0.50, 0 changes)",
        "Input": "224x224",
        "Accuracy": 80.57, "Balanced_Accuracy": 75.69, "Macro_F1": 70.51, "Weighted_F1": 81.06,
        "Melanoma_Precision": 54.00, "Melanoma_Recall": 50.94, "Melanoma_F1": 52.43
    }
]

exp_comp_df = pd.DataFrame(exp_history)
exp_comp_df.to_csv(os.path.join(DOCS_DIR, "experiment_comparison_table.csv"), index=False)
print("  Generated experiment_comparison_table.csv")

# Final Results Table
final_results_rows = [
    {
        "Model": "Experiment 8 (Tempered alpha=0.75)",
        "Role": "Precision-anchored base model",
        "Accuracy": 80.67, "Balanced_Accuracy": 75.37, "Macro_F1": 70.18, "Weighted_F1": 81.07,
        "Melanoma_Recall": 48.11, "Melanoma_Precision": 56.04, "Melanoma_F1": 51.78
    },
    {
        "Model": "Experiment 13 (Targeted 1.20x Mel)",
        "Role": "Sensitivity-enhanced complement",
        "Accuracy": 78.04, "Balanced_Accuracy": 73.14, "Macro_F1": 67.58, "Weighted_F1": 78.92,
        "Melanoma_Recall": 59.43, "Melanoma_Precision": 45.32, "Melanoma_F1": 51.43
    },
    {
        "Model": "Experiment 14 (Ensemble 85/15)",
        "Role": "Final frozen probability ensemble",
        "Accuracy": 80.57, "Balanced_Accuracy": 75.69, "Macro_F1": 70.51, "Weighted_F1": 81.06,
        "Melanoma_Recall": 50.94, "Melanoma_Precision": 54.00, "Melanoma_F1": 52.43
    },
    {
        "Model": "Experiment 15 (Threshold Verification)",
        "Role": "Threshold-tuned ensemble audit",
        "Accuracy": 80.57, "Balanced_Accuracy": 75.69, "Macro_F1": 70.51, "Weighted_F1": 81.06,
        "Melanoma_Recall": 50.94, "Melanoma_Precision": 54.00, "Melanoma_F1": 52.43
    }
]
final_results_df = pd.DataFrame(final_results_rows)
final_results_df.to_csv(os.path.join(DOCS_DIR, "final_results_table.csv"), index=False)
print("  Generated final_results_table.csv")

# Melanoma Analysis Table
mel_analysis_rows = [
    {
        "Model": "Experiment 1 (Baseline)",
        "Melanoma_Support": 106,
        "Melanoma_Detected": 48,
        "Melanoma_Recall": 45.28,
        "Melanoma_Precision": 36.92,
        "Melanoma_F1": 40.68,
        "False_Negatives": 58,
        "Missed_as_NV": 35,
        "Missed_as_BKL": 18
    },
    {
        "Model": "Experiment 8 (Tempered)",
        "Melanoma_Support": 106,
        "Melanoma_Detected": 51,
        "Melanoma_Recall": 48.11,
        "Melanoma_Precision": 56.04,
        "Melanoma_F1": 51.78,
        "False_Negatives": 55,
        "Missed_as_NV": 28,
        "Missed_as_BKL": 21
    },
    {
        "Model": "Experiment 13 (Mel-Focused)",
        "Melanoma_Support": 106,
        "Melanoma_Detected": 63,
        "Melanoma_Recall": 59.43,
        "Melanoma_Precision": 45.32,
        "Melanoma_F1": 51.43,
        "False_Negatives": 43,
        "Missed_as_NV": 21,
        "Missed_as_BKL": 16
    },
    {
        "Model": "Experiment 14 (Final Ensemble)",
        "Melanoma_Support": 106,
        "Melanoma_Detected": 54,
        "Melanoma_Recall": 50.94,
        "Melanoma_Precision": 54.00,
        "Melanoma_F1": 52.43,
        "False_Negatives": 52,
        "Missed_as_NV": 26,
        "Missed_as_BKL": 19
    }
]
mel_analysis_df = pd.DataFrame(mel_analysis_rows)
mel_analysis_df.to_csv(os.path.join(DOCS_DIR, "melanoma_analysis.csv"), index=False)
print("  Generated melanoma_analysis.csv")

# ==============================================================================
# 5. Descriptive Experiment Comparison Figures
# ==============================================================================
exp_labels = [f"Exp {i}" for i in range(1, 16)]

metrics_to_plot = [
    ("Accuracy", "accuracy_by_experiment.png", "Overall Test Accuracy across Experiments (%)", 65, 85),
    ("Balanced_Accuracy", "balanced_accuracy_by_experiment.png", "Balanced Test Accuracy across Experiments (%)", 60, 80),
    ("Macro_F1", "macro_f1_by_experiment.png", "Macro F1-Score across Experiments (%)", 50, 75),
    ("Melanoma_Recall", "melanoma_recall_by_experiment.png", "Melanoma Test Recall across Experiments (%)", 25, 65),
    ("Melanoma_F1", "melanoma_f1_by_experiment.png", "Melanoma Test F1-Score across Experiments (%)", 30, 55),
]

for col, filename, title, ymin, ymax in metrics_to_plot:
    plt.figure(figsize=(10, 5), dpi=300)
    vals = exp_comp_df[col].values
    
    # Neutral corporate/academic color scheme
    bars = plt.bar(exp_labels, vals, color="#3b82f6", edgecolor="#1d4ed8", width=0.65)
    
    # Highlight Exp 8 and Exp 14 slightly
    bars[7].set_color("#059669")   # Exp 8 (emerald)
    bars[12].set_color("#d97706")  # Exp 13 (amber)
    bars[13].set_color("#7c3aed")  # Exp 14 (violet)
    bars[14].set_color("#6d28d9")  # Exp 15 (purple)
    
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Experiment", fontsize=10, labelpad=8)
    plt.ylabel(col.replace("_", " ") + " (%)", fontsize=10, labelpad=8)
    plt.ylim(ymin, ymax)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                 f"{h:.1f}%" if col != "Macro_F1" else f"{h:.1f}",
                 ha="center", va="bottom", fontsize=8, rotation=0)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, filename))
    plt.close()
    print(f"  Generated {filename}")

print("[Phase 17] All tables and figures successfully generated.")
