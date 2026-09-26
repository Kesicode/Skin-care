import os
import sys
import json
import time
import random
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import seaborn as sns

try:
    from .model import DermaAI_MobileNetV3
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING

# ── Reproducibility ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths & Config ──────────────────────────────────────────────────────────────
SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
AI_DIR        = os.path.dirname(SRC_DIR)
DATASET_DIR   = os.path.join(AI_DIR, "dataset")
SPLITS_DIR    = os.path.join(DATASET_DIR, "splits")
MODELS_DIR    = os.path.join(AI_DIR, "models")
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment15")

CHECKPOINT_EXP8  = os.path.join(MODELS_DIR, "experiment8_best_model.pt")
CHECKPOINT_EXP13 = os.path.join(MODELS_DIR, "experiment13_best_model.pt")

VAL_CSV_PATH  = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(SPLITS_DIR, "test.csv")

# Fixed Experiment 14 parameters
LAMBDA_ENSEMBLE = 0.15
WEIGHT_EXP8     = 0.85
WEIGHT_EXP13    = 0.15

# Predefined 17-point threshold search grid
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]
BATCH_SIZE     = 32

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX     = CLASS_MAPPING["mel"]
NV_IDX      = CLASS_MAPPING["nv"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Step 1: Model Loading & Freezing ───────────────────────────────────────────
def load_and_verify_models(device):
    print("="*70)
    print("STEP 1: LOAD & VERIFY EXP8 AND EXP13 CHECKPOINTS (FROZEN EVAL MODE)")
    print("="*70)
    assert os.path.exists(CHECKPOINT_EXP8), f"Exp8 checkpoint missing: {CHECKPOINT_EXP8}"
    assert os.path.exists(CHECKPOINT_EXP13), f"Exp13 checkpoint missing: {CHECKPOINT_EXP13}"
    
    model8 = DermaAI_MobileNetV3(num_classes=7)
    model8.load_state_dict(torch.load(CHECKPOINT_EXP8, map_location=device))
    model8 = model8.to(device)
    model8.eval()
    for p in model8.parameters():
        p.requires_grad = False
    
    model13 = DermaAI_MobileNetV3(num_classes=7)
    model13.load_state_dict(torch.load(CHECKPOINT_EXP13, map_location=device))
    model13 = model13.to(device)
    model13.eval()
    for p in model13.parameters():
        p.requires_grad = False
        
    p8 = sum(p.numel() for p in model8.parameters())
    p13 = sum(p.numel() for p in model13.parameters())
    
    print(f"  Model A (Exp8)  : DermaAI_MobileNetV3 | Params: {p8:,} | Requires Grad: False")
    print(f"  Model B (Exp13) : DermaAI_MobileNetV3 | Params: {p13:,} | Requires Grad: False")
    assert p8 == 4326297 and p13 == 4326297, "Parameter count mismatch! Expected 4,326,297 parameters."
    
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out8 = model8(dummy)
        out13 = model13(dummy)
        prob8 = torch.softmax(out8, dim=1).cpu().numpy()
        prob13 = torch.softmax(out13, dim=1).cpu().numpy()
        
    assert out8.shape == (2, 7) and out13.shape == (2, 7)
    assert np.allclose(prob8.sum(axis=1), 1.0, atol=1e-5), "Model 8 softmax must sum to 1.0"
    assert np.allclose(prob13.sum(axis=1), 1.0, atol=1e-5), "Model 13 softmax must sum to 1.0"
    print("  Forward pass and softmax checks: PASSED")
    print("="*70 + "\n")
    return model8, model13


# ── Step 2: Probability Extraction ─────────────────────────────────────────────
def extract_probabilities(model, dataloader, device, desc="Inference"):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    assert np.allclose(all_probs.sum(axis=1), 1.0, atol=1e-5), f"{desc} probabilities must sum to 1.0"
    return all_probs, all_labels


# ── Helper to Compute Comprehensive Metrics ────────────────────────────────────
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), zero_division=0
    )
    
    # Melanoma specific
    mel_recall = float(per_class_r[MEL_IDX])
    mel_precision = float(per_class_p[MEL_IDX])
    mel_f1 = float(per_class_f1[MEL_IDX])
    mel_support = int(per_class_supp[MEL_IDX])
    mel_tp = int(cm[MEL_IDX, MEL_IDX])
    mel_fp = int(cm[:, MEL_IDX].sum() - mel_tp)
    mel_fn = int(mel_support - mel_tp)
    mel_pred_total = int(cm[:, MEL_IDX].sum())
    
    mel_to_nv = int(cm[MEL_IDX, NV_IDX])
    mel_to_bkl = int(cm[MEL_IDX, CLASS_MAPPING["bkl"]])
    mel_to_akiec = int(cm[MEL_IDX, CLASS_MAPPING["akiec"]])
    mel_to_bcc = int(cm[MEL_IDX, CLASS_MAPPING["bcc"]])
    
    # Nevus specific
    nv_recall = float(per_class_r[NV_IDX])
    nv_precision = float(per_class_p[NV_IDX])
    nv_f1 = float(per_class_f1[NV_IDX])
    nv_support = int(per_class_supp[NV_IDX])
    nv_tp = int(cm[NV_IDX, NV_IDX])
    nv_to_mel = int(cm[NV_IDX, MEL_IDX])
    
    per_class_dict = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class_dict[cname] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1-score": float(per_class_f1[i]),
            "support": int(per_class_supp[i]),
            "correct": int(cm[i, i])
        }
        
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "melanoma_precision": mel_precision,
        "melanoma_recall": mel_recall,
        "melanoma_f1": mel_f1,
        "melanoma_correct_images": mel_tp,
        "melanoma_total_images": mel_support,
        "melanoma_predicted_total": mel_pred_total,
        "melanoma_tp": mel_tp,
        "melanoma_fp": mel_fp,
        "melanoma_fn": mel_fn,
        "melanoma_to_nv_errors": mel_to_nv,
        "melanoma_to_bkl_errors": mel_to_bkl,
        "melanoma_to_akiec_errors": mel_to_akiec,
        "melanoma_to_bcc_errors": mel_to_bcc,
        "nv_precision": nv_precision,
        "nv_recall": nv_recall,
        "nv_f1": nv_f1,
        "nv_correct_images": nv_tp,
        "nv_total_images": nv_support,
        "nv_to_mel_errors": nv_to_mel,
        "bcc_recall": float(per_class_r[CLASS_MAPPING["bcc"]]),
        "bkl_recall": float(per_class_r[CLASS_MAPPING["bkl"]]),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }


# ── Plot & Save Confusion Matrix ───────────────────────────────────────────────
def save_confusion_matrix_plot(cm, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar=True, annot_kws={"size": 11}
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ── Post-hoc Decision Override Rule ────────────────────────────────────────────
def apply_melanoma_threshold(probs, threshold):
    """
    If P(mel) >= threshold: predict mel (4)
    Else: predict argmax(probs)
    """
    argmax_preds = np.argmax(probs, axis=1)
    thresholded_preds = np.where(probs[:, MEL_IDX] >= threshold, MEL_IDX, argmax_preds)
    return thresholded_preds, argmax_preds


# ── Main Routine ───────────────────────────────────────────────────────────────
def main():
    start_total_time = time.time()
    device = torch.device("cpu")
    print("\n" + "="*70)
    print("EXPERIMENT 15: VALIDATION-CALIBRATED MELANOMA THRESHOLD ON EXP14 ENSEMBLE")
    print(f"Device : {device} | PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}")
    print("="*70)
    
    # Preprocessing
    eval_transform = get_transforms(is_train=False)
    
    # ── ZERO TRAINING SET LOADING ─────────────────────────────────────────────
    assert os.path.exists(VAL_CSV_PATH), f"Val split missing: {VAL_CSV_PATH}"
    assert os.path.exists(TEST_CSV_PATH), f"Test split missing: {TEST_CSV_PATH}"
    
    val_df  = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    print(f"Splits loaded: Validation = {len(val_df)} images | Test = {len(test_df)} images")
    print("TRAINING SET WAS NOT LOADED (Strict Leakage Boundary Maintained).")
    
    val_dataset  = HAM10000Dataset(val_df, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, transform=eval_transform)
    
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load and freeze models
    model8, model13 = load_and_verify_models(device)
    
    # ── VALIDATION ENSEMBLE INFERENCE ──────────────────────────────────────────
    print("="*70)
    print("STEP 2: VALIDATION PROBABILITY EXTRACTION & EXP14 ENSEMBLE REPRODUCTION")
    print("="*70)
    val_start_time = time.time()
    P8_val, y_val = extract_probabilities(model8, val_loader, device, desc="Val Exp8")
    P13_val, y_val_check = extract_probabilities(model13, val_loader, device, desc="Val Exp13")
    val_inference_duration = time.time() - val_start_time
    assert np.array_equal(y_val, y_val_check), "Validation label alignment error!"
    
    # Form fixed Exp 14 Ensemble probabilities
    P_ens_val = WEIGHT_EXP8 * P8_val + WEIGHT_EXP13 * P13_val
    assert np.allclose(P_ens_val.sum(axis=1), 1.0, atol=1e-5), "Validation ensemble probabilities must sum to 1.0"
    
    # Compute baseline argmax validation metrics
    val_preds_argmax = np.argmax(P_ens_val, axis=1)
    val_metrics_argmax = compute_metrics(y_val, val_preds_argmax)
    with open(os.path.join(RESULTS_DIR, "validation_metrics_argmax.json"), "w") as f:
        json.dump(val_metrics_argmax, f, indent=4)
        
    print(f"Validation Ensemble Argmax | Acc: {val_metrics_argmax['accuracy']*100:.2f}% | "
          f"BalAcc: {val_metrics_argmax['balanced_accuracy']*100:.2f}% | "
          f"Macro F1: {val_metrics_argmax['macro_f1']:.4f} | "
          f"Mel Rec: {val_metrics_argmax['melanoma_recall']*100:.2f}% | "
          f"Mel Prec: {val_metrics_argmax['melanoma_precision']*100:.2f}% | "
          f"Mel F1: {val_metrics_argmax['melanoma_f1']:.4f}")
    
    # ── 17-POINT THRESHOLD SWEEP ON VALIDATION ─────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: 17-POINT PREDEFINED VALIDATION THRESHOLD SWEEP (θ = 0.10 - 0.90)")
    print("="*70)
    sweep_results = []
    val_metrics_per_theta = {}
    
    for theta in THRESHOLD_GRID:
        preds_theta, _ = apply_melanoma_threshold(P_ens_val, theta)
        
        # Calculate changes vs argmax
        diff_mask = (preds_theta != val_preds_argmax)
        changed_count = int(diff_mask.sum())
        pred_mel_count = int((preds_theta == MEL_IDX).sum())
        add_mel_count = pred_mel_count - int((val_preds_argmax == MEL_IDX).sum())
        
        m = compute_metrics(y_val, preds_theta)
        val_metrics_per_theta[theta] = m
        
        sweep_results.append({
            "threshold": float(theta),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_precision": m["weighted_precision"],
            "weighted_recall": m["weighted_recall"],
            "weighted_f1": m["weighted_f1"],
            "melanoma_precision": m["melanoma_precision"],
            "melanoma_recall": m["melanoma_recall"],
            "melanoma_f1": m["melanoma_f1"],
            "melanoma_tp": m["melanoma_tp"],
            "melanoma_fp": m["melanoma_fp"],
            "melanoma_fn": m["melanoma_fn"],
            "mel_to_nv": m["melanoma_to_nv_errors"],
            "mel_to_bkl": m["melanoma_to_bkl_errors"],
            "nv_recall": m["nv_recall"],
            "nv_to_mel_fp": m["nv_to_mel_errors"],
            "predicted_melanoma_count": pred_mel_count,
            "changed_predictions_vs_argmax": changed_count,
            "additional_melanoma_predictions": add_mel_count
        })
        
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(os.path.join(RESULTS_DIR, "threshold_sweep_validation.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "threshold_sweep_validation.json"), "w") as f:
        json.dump(sweep_results, f, indent=4)
    print(f"Validation threshold sweep saved: {os.path.join(RESULTS_DIR, 'threshold_sweep_validation.csv')}")
    
    # ── THRESHOLD SELECTION RULE ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4: APPLY THRESHOLD SELECTION RULE (VALIDATION DATA ONLY)")
    print("="*70)
    f1_max = float(sweep_df["melanoma_f1"].max())
    print(f"  Maximum Validation Melanoma F1 (F1_max): {f1_max:.4f}")
    
    # Candidate thresholds: F1_max - F1(theta) <= 0.005
    candidate_mask = (f1_max - sweep_df["melanoma_f1"]) <= 0.005
    candidate_df = sweep_df[candidate_mask].copy()
    candidate_thetas = candidate_df["threshold"].tolist()
    print(f"  Candidate thresholds within 0.005 tolerance ({len(candidate_thetas)}): {candidate_thetas}")
    
    # Tie-break 1: Higher Validation Melanoma Recall
    max_mel_recall = candidate_df["melanoma_recall"].max()
    tb1_df = candidate_df[candidate_df["melanoma_recall"] == max_mel_recall]
    print(f"  Tie-break 1 (Max Mel Recall): {max_mel_recall*100:.2f}% (retained {len(tb1_df)} candidate(s))")
    
    selection_tie_breaker = "Primary (Max Melanoma F1)"
    if len(tb1_df) == 1:
        selected_row = tb1_df.iloc[0]
        selection_tie_breaker = "Tie-break 1 (Higher Validation Melanoma Recall)"
    else:
        # Tie-break 2: Higher Validation Balanced Accuracy
        max_bacc = tb1_df["balanced_accuracy"].max()
        tb2_df = tb1_df[tb1_df["balanced_accuracy"] == max_bacc]
        print(f"  Tie-break 2 (Max Balanced Accuracy): {max_bacc*100:.2f}% (retained {len(tb2_df)} candidate(s))")
        if len(tb2_df) == 1:
            selected_row = tb2_df.iloc[0]
            selection_tie_breaker = "Tie-break 2 (Higher Validation Balanced Accuracy)"
        else:
            # Tie-break 3: Higher Validation Overall Accuracy
            max_acc = tb2_df["accuracy"].max()
            tb3_df = tb2_df[tb2_df["accuracy"] == max_acc]
            print(f"  Tie-break 3 (Max Overall Accuracy): {max_acc*100:.2f}% (retained {len(tb3_df)} candidate(s))")
            if len(tb3_df) == 1:
                selected_row = tb3_df.iloc[0]
                selection_tie_breaker = "Tie-break 3 (Higher Validation Overall Accuracy)"
            else:
                # Tie-break 4: Smaller threshold
                min_theta = tb3_df["threshold"].min()
                selected_row = tb3_df[tb3_df["threshold"] == min_theta].iloc[0]
                selection_tie_breaker = "Tie-break 4 (Smaller Threshold)"
                
    selected_theta = float(selected_row["threshold"])
    
    print("\n" + "#"*70)
    print(f"FINAL SELECTED MELANOMA THRESHOLD (VAL ONLY): theta* = {selected_theta:.2f}")
    print(f"  Val Melanoma F1 : {selected_row['melanoma_f1']:.4f}")
    print(f"  Val Mel Recall  : {selected_row['melanoma_recall']*100:.2f}%")
    print(f"  Val Mel Prec    : {selected_row['melanoma_precision']*100:.2f}%")
    print(f"  Val Bal Acc     : {selected_row['balanced_accuracy']*100:.2f}%")
    print(f"  Val Accuracy    : {selected_row['accuracy']*100:.2f}%")
    print(f"  Selection Logic : {selection_tie_breaker}")
    print("#"*70 + "\n")
    
    # Save selected_threshold.json
    selected_threshold_data = {
        "selected_threshold": selected_theta,
        "selection_rule": "F1_max - MelF1 <= 0.005 -> Max Melanoma Recall -> Max Balanced Accuracy -> Max Accuracy -> Min Theta",
        "selection_tie_breaker": selection_tie_breaker,
        "f1_max": f1_max,
        "tolerance": 0.005,
        "candidate_thresholds": candidate_thetas,
        "validation_metrics_selected": {
            "accuracy": float(selected_row["accuracy"]),
            "balanced_accuracy": float(selected_row["balanced_accuracy"]),
            "macro_f1": float(selected_row["macro_f1"]),
            "melanoma_precision": float(selected_row["melanoma_precision"]),
            "melanoma_recall": float(selected_row["melanoma_recall"]),
            "melanoma_f1": float(selected_row["melanoma_f1"]),
            "melanoma_tp": int(selected_row["melanoma_tp"]),
            "melanoma_fp": int(selected_row["melanoma_fp"]),
            "nv_to_mel_fp": int(selected_row["nv_to_mel_fp"]),
            "changed_predictions_vs_argmax": int(selected_row["changed_predictions_vs_argmax"]),
            "additional_melanoma_predictions": int(selected_row["additional_melanoma_predictions"])
        },
        "validation_metrics_argmax": {
            "accuracy": val_metrics_argmax["accuracy"],
            "balanced_accuracy": val_metrics_argmax["balanced_accuracy"],
            "macro_f1": val_metrics_argmax["macro_f1"],
            "melanoma_precision": val_metrics_argmax["melanoma_precision"],
            "melanoma_recall": val_metrics_argmax["melanoma_recall"],
            "melanoma_f1": val_metrics_argmax["melanoma_f1"]
        },
        "status": "PERMANENTLY FROZEN (Pre-Test Evaluation)"
    }
    with open(os.path.join(RESULTS_DIR, "selected_threshold.json"), "w") as f:
        json.dump(selected_threshold_data, f, indent=4)
        
    val_selected_metrics = val_metrics_per_theta[selected_theta]
    with open(os.path.join(RESULTS_DIR, "validation_metrics_thresholded.json"), "w") as f:
        json.dump(val_selected_metrics, f, indent=4)
        
    print("="*70)
    print("LEAKAGE BOUNDARY VERIFICATION: THRESHOLD IS PERMANENTLY FROZEN.")
    print("No further threshold adjustment will occur.")
    print("Proceeding to evaluate held-out test set EXACTLY ONCE.")
    print("="*70 + "\n")
    
    # ── TEST INFERENCE ONCE ────────────────────────────────────────────────────
    test_start_time = time.time()
    P8_test, y_test = extract_probabilities(model8, test_loader, device, desc="Test Exp8")
    P13_test, y_test_check = extract_probabilities(model13, test_loader, device, desc="Test Exp13")
    test_inference_duration = time.time() - test_start_time
    assert np.array_equal(y_test, y_test_check), "Test label alignment error!"
    
    # Core stored ensemble probabilities for test set
    P_ens_test = WEIGHT_EXP8 * P8_test + WEIGHT_EXP13 * P13_test
    assert np.allclose(P_ens_test.sum(axis=1), 1.0, atol=1e-5), "Test ensemble probabilities must sum to 1.0"
    
    # Predictions derived from the EXACT SAME ensemble probabilities
    pred_exp14_argmax = np.argmax(P_ens_test, axis=1)
    pred_exp15_thresholded, _ = apply_melanoma_threshold(P_ens_test, selected_theta)
    
    # Compute Full Test Metrics
    m_exp14 = compute_metrics(y_test, pred_exp14_argmax)
    m_exp15 = compute_metrics(y_test, pred_exp15_thresholded)
    
    with open(os.path.join(RESULTS_DIR, "test_metrics_exp14.json"), "w") as f:
        json.dump(m_exp14, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "test_metrics_exp15.json"), "w") as f:
        json.dump(m_exp15, f, indent=4)
        
    # Classification Reports TXT
    rep_exp14 = classification_report(y_test, pred_exp14_argmax, target_names=CLASS_NAMES, digits=4, zero_division=0)
    rep_exp15 = classification_report(y_test, pred_exp15_thresholded, target_names=CLASS_NAMES, digits=4, zero_division=0)
    
    with open(os.path.join(RESULTS_DIR, "classification_report_exp14.txt"), "w") as f:
        f.write("EXPERIMENT 14 ENSEMBLE (ARGMAX) ON TEST SET (N=988):\n\n" + rep_exp14)
    with open(os.path.join(RESULTS_DIR, "classification_report_exp15.txt"), "w") as f:
        f.write(f"EXPERIMENT 15 THRESHOLDED ENSEMBLE (theta={selected_theta:.2f}) ON TEST SET (N=988):\n\n" + rep_exp15)
        
    # Confusion Matrices JSON & PNG
    cm_exp14 = np.array(m_exp14["confusion_matrix"])
    cm_exp15 = np.array(m_exp15["confusion_matrix"])
    
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_exp14.json"), "w") as f:
        json.dump({"classes": CLASS_NAMES, "confusion_matrix": cm_exp14.tolist()}, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_exp15.json"), "w") as f:
        json.dump({"classes": CLASS_NAMES, "confusion_matrix": cm_exp15.tolist()}, f, indent=4)
        
    save_confusion_matrix_plot(cm_exp14, os.path.join(RESULTS_DIR, "confusion_matrix_exp14.png"),
                               "Experiment 14: Ensemble Argmax Confusion Matrix (Test N=988)")
    save_confusion_matrix_plot(cm_exp15, os.path.join(RESULTS_DIR, "confusion_matrix_exp15.png"),
                               f"Experiment 15: Thresholded Ensemble (θ={selected_theta:.2f}) Confusion Matrix (Test N=988)")

    # ── PREDICTION CHANGE ANALYSIS ─────────────────────────────────────────────
    print("="*70)
    print("STEP 5: PREDICTION CHANGE & TRANSITION ANALYSIS (TEST N=988)")
    print("="*70)
    N = len(y_test)
    diff_mask = (pred_exp15_thresholded != pred_exp14_argmax)
    total_changed = int(diff_mask.sum())
    
    correct_exp14 = (pred_exp14_argmax == y_test)
    correct_exp15 = (pred_exp15_thresholded == y_test)
    
    inc_to_cor = int((~correct_exp14 & correct_exp15).sum())
    cor_to_inc = int((correct_exp14 & ~correct_exp15).sum())
    inc_to_inc = int((~correct_exp14 & ~correct_exp15 & diff_mask).sum())
    net_correct_change = inc_to_cor - cor_to_inc
    
    # Specific transition counts
    # Under melanoma threshold override, all changes originate from non-mel -> mel
    nv_to_mel_changes = int(((pred_exp14_argmax == NV_IDX) & (pred_exp15_thresholded == MEL_IDX)).sum())
    bkl_to_mel_changes = int(((pred_exp14_argmax == CLASS_MAPPING["bkl"]) & (pred_exp15_thresholded == MEL_IDX)).sum())
    other_to_mel_changes = total_changed - (nv_to_mel_changes + bkl_to_mel_changes)
    
    pred_changes_data = {
        "test_total_images": N,
        "selected_threshold": selected_theta,
        "total_changed_predictions": total_changed,
        "changed_percentage": float(total_changed / N * 100),
        "transitions_breakdown": {
            "incorrect_to_correct": inc_to_cor,
            "correct_to_incorrect": cor_to_inc,
            "incorrect_to_incorrect": inc_to_inc,
            "net_correct_change": net_correct_change
        },
        "source_class_of_melanoma_overrides": {
            "from_nv_to_mel": nv_to_mel_changes,
            "from_bkl_to_mel": bkl_to_mel_changes,
            "from_other_to_mel": other_to_mel_changes
        },
        "overall_correct_counts": {
            "exp14_correct": int(correct_exp14.sum()),
            "exp15_correct": int(correct_exp15.sum()),
            "net_difference": int(correct_exp15.sum() - correct_exp14.sum())
        }
    }
    with open(os.path.join(RESULTS_DIR, "threshold_prediction_changes.json"), "w") as f:
        json.dump(pred_changes_data, f, indent=4)

    # ── MELANOMA OPERATING-POINT ANALYSIS ──────────────────────────────────────
    mel_before = int(m_exp14["melanoma_predicted_total"])
    mel_after = int(m_exp15["melanoma_predicted_total"])
    add_mel_pred = mel_after - mel_before
    
    mel_tp_before = int(m_exp14["melanoma_tp"])
    mel_tp_after = int(m_exp15["melanoma_tp"])
    d_mel_tp = mel_tp_after - mel_tp_before
    
    mel_fp_before = int(m_exp14["melanoma_fp"])
    mel_fp_after = int(m_exp15["melanoma_fp"])
    d_mel_fp = mel_fp_after - mel_fp_before
    
    mel_fn_before = int(m_exp14["melanoma_fn"])
    mel_fn_after = int(m_exp15["melanoma_fn"])
    d_mel_fn = mel_fn_after - mel_fn_before
    
    primary_mechanism = "Both (recovering missed melanomas while increasing false alarms)"
    if d_mel_tp > 0 and d_mel_fp == 0:
        primary_mechanism = "Pure recovery of missed melanomas"
    elif d_mel_tp == 0 and d_mel_fp > 0:
        primary_mechanism = "Pure increase in false positives"
    elif d_mel_tp == 0 and d_mel_fp == 0:
        primary_mechanism = "No change in melanoma operating point"
        
    melanoma_analysis_data = {
        "melanoma_predictions_before_thresholding": mel_before,
        "melanoma_predictions_after_thresholding": mel_after,
        "additional_melanoma_predictions": add_mel_pred,
        "melanoma_tp_before": mel_tp_before,
        "melanoma_tp_after": mel_tp_after,
        "melanoma_tp_change": d_mel_tp,
        "melanoma_fp_before": mel_fp_before,
        "melanoma_fp_after": mel_fp_after,
        "melanoma_fp_change": d_mel_fp,
        "melanoma_fn_before": mel_fn_before,
        "melanoma_fn_after": mel_fn_after,
        "melanoma_fn_change": d_mel_fn,
        "melanoma_recall_before": m_exp14["melanoma_recall"],
        "melanoma_recall_after": m_exp15["melanoma_recall"],
        "melanoma_recall_change": m_exp15["melanoma_recall"] - m_exp14["melanoma_recall"],
        "melanoma_precision_before": m_exp14["melanoma_precision"],
        "melanoma_precision_after": m_exp15["melanoma_precision"],
        "melanoma_precision_change": m_exp15["melanoma_precision"] - m_exp14["melanoma_precision"],
        "melanoma_f1_before": m_exp14["melanoma_f1"],
        "melanoma_f1_after": m_exp15["melanoma_f1"],
        "melanoma_f1_change": m_exp15["melanoma_f1"] - m_exp14["melanoma_f1"],
        "mel_to_nv_before": m_exp14["melanoma_to_nv_errors"],
        "mel_to_nv_after": m_exp15["melanoma_to_nv_errors"],
        "mel_to_nv_change": m_exp15["melanoma_to_nv_errors"] - m_exp14["melanoma_to_nv_errors"],
        "mel_to_bkl_before": m_exp14["melanoma_to_bkl_errors"],
        "mel_to_bkl_after": m_exp15["melanoma_to_bkl_errors"],
        "mel_to_bkl_change": m_exp15["melanoma_to_bkl_errors"] - m_exp14["melanoma_to_bkl_errors"],
        "primary_mechanism": primary_mechanism
    }
    with open(os.path.join(RESULTS_DIR, "melanoma_analysis.json"), "w") as f:
        json.dump(melanoma_analysis_data, f, indent=4)

    # ── Comparison Artifacts ───────────────────────────────────────────────────
    # Comparison with Exp 14
    comp_exp14 = {
        "experiment14_argmax": {
            "accuracy": m_exp14["accuracy"],
            "balanced_accuracy": m_exp14["balanced_accuracy"],
            "macro_precision": m_exp14["macro_precision"],
            "macro_recall": m_exp14["macro_recall"],
            "macro_f1": m_exp14["macro_f1"],
            "weighted_f1": m_exp14["weighted_f1"],
            "melanoma_precision": m_exp14["melanoma_precision"],
            "melanoma_recall": m_exp14["melanoma_recall"],
            "melanoma_f1": m_exp14["melanoma_f1"],
            "mel_to_nv_errors": m_exp14["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": m_exp14["melanoma_to_bkl_errors"],
            "nv_recall": m_exp14["nv_recall"],
            "nv_to_mel_errors": m_exp14["nv_to_mel_errors"]
        },
        "experiment15_thresholded": {
            "accuracy": m_exp15["accuracy"],
            "balanced_accuracy": m_exp15["balanced_accuracy"],
            "macro_precision": m_exp15["macro_precision"],
            "macro_recall": m_exp15["macro_recall"],
            "macro_f1": m_exp15["macro_f1"],
            "weighted_f1": m_exp15["weighted_f1"],
            "melanoma_precision": m_exp15["melanoma_precision"],
            "melanoma_recall": m_exp15["melanoma_recall"],
            "melanoma_f1": m_exp15["melanoma_f1"],
            "mel_to_nv_errors": m_exp15["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": m_exp15["melanoma_to_bkl_errors"],
            "nv_recall": m_exp15["nv_recall"],
            "nv_to_mel_errors": m_exp15["nv_to_mel_errors"]
        },
        "delta_exp15_minus_exp14": {
            "accuracy_change": m_exp15["accuracy"] - m_exp14["accuracy"],
            "balanced_accuracy_change": m_exp15["balanced_accuracy"] - m_exp14["balanced_accuracy"],
            "macro_f1_change": m_exp15["macro_f1"] - m_exp14["macro_f1"],
            "weighted_f1_change": m_exp15["weighted_f1"] - m_exp14["weighted_f1"],
            "melanoma_recall_change": m_exp15["melanoma_recall"] - m_exp14["melanoma_recall"],
            "melanoma_precision_change": m_exp15["melanoma_precision"] - m_exp14["melanoma_precision"],
            "melanoma_f1_change": m_exp15["melanoma_f1"] - m_exp14["melanoma_f1"],
            "mel_to_nv_change": m_exp15["melanoma_to_nv_errors"] - m_exp14["melanoma_to_nv_errors"],
            "mel_to_bkl_change": m_exp15["melanoma_to_bkl_errors"] - m_exp14["melanoma_to_bkl_errors"],
            "nv_to_mel_change": m_exp15["nv_to_mel_errors"] - m_exp14["nv_to_mel_errors"]
        }
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment14.json"), "w") as f:
        json.dump(comp_exp14, f, indent=4)

    # Comparison with Exp 8
    exp8_path = os.path.join(AI_DIR, "results", "experiment8", "metrics.json")
    if os.path.exists(exp8_path):
        with open(exp8_path, "r") as f:
            e8_data = json.load(f)
        comp_exp8 = {
            "experiment8": {
                "accuracy": e8_data.get("accuracy"),
                "balanced_accuracy": e8_data.get("balanced_accuracy"),
                "macro_f1": e8_data.get("macro_f1"),
                "melanoma_recall": e8_data.get("melanoma_recall"),
                "melanoma_precision": e8_data.get("melanoma_precision"),
                "melanoma_f1": e8_data.get("melanoma_f1")
            },
            "experiment15": {
                "accuracy": m_exp15["accuracy"],
                "balanced_accuracy": m_exp15["balanced_accuracy"],
                "macro_f1": m_exp15["macro_f1"],
                "melanoma_recall": m_exp15["melanoma_recall"],
                "melanoma_precision": m_exp15["melanoma_precision"],
                "melanoma_f1": m_exp15["melanoma_f1"]
            },
            "delta_exp15_minus_exp8": {
                "accuracy_change": m_exp15["accuracy"] - e8_data.get("accuracy", 0),
                "balanced_accuracy_change": m_exp15["balanced_accuracy"] - e8_data.get("balanced_accuracy", 0),
                "macro_f1_change": m_exp15["macro_f1"] - e8_data.get("macro_f1", 0),
                "melanoma_recall_change": m_exp15["melanoma_recall"] - e8_data.get("melanoma_recall", 0),
                "melanoma_precision_change": m_exp15["melanoma_precision"] - e8_data.get("melanoma_precision", 0),
                "melanoma_f1_change": m_exp15["melanoma_f1"] - e8_data.get("melanoma_f1", 0)
            }
        }
        with open(os.path.join(RESULTS_DIR, "comparison_with_experiment8.json"), "w") as f:
            json.dump(comp_exp8, f, indent=4)

    # Comparison with Exp 13
    exp13_path = os.path.join(AI_DIR, "results", "experiment13", "metrics.json")
    if os.path.exists(exp13_path):
        with open(exp13_path, "r") as f:
            e13_data = json.load(f)
        comp_exp13 = {
            "experiment13": {
                "accuracy": e13_data.get("accuracy"),
                "balanced_accuracy": e13_data.get("balanced_accuracy"),
                "macro_f1": e13_data.get("macro_f1"),
                "melanoma_recall": e13_data.get("melanoma_recall"),
                "melanoma_precision": e13_data.get("melanoma_precision"),
                "melanoma_f1": e13_data.get("melanoma_f1")
            },
            "experiment15": {
                "accuracy": m_exp15["accuracy"],
                "balanced_accuracy": m_exp15["balanced_accuracy"],
                "macro_f1": m_exp15["macro_f1"],
                "melanoma_recall": m_exp15["melanoma_recall"],
                "melanoma_precision": m_exp15["melanoma_precision"],
                "melanoma_f1": m_exp15["melanoma_f1"]
            },
            "delta_exp15_minus_exp13": {
                "accuracy_change": m_exp15["accuracy"] - e13_data.get("accuracy", 0),
                "balanced_accuracy_change": m_exp15["balanced_accuracy"] - e13_data.get("balanced_accuracy", 0),
                "macro_f1_change": m_exp15["macro_f1"] - e13_data.get("macro_f1", 0),
                "melanoma_recall_change": m_exp15["melanoma_recall"] - e13_data.get("melanoma_recall", 0),
                "melanoma_precision_change": m_exp15["melanoma_precision"] - e13_data.get("melanoma_precision", 0),
                "melanoma_f1_change": m_exp15["melanoma_f1"] - e13_data.get("melanoma_f1", 0)
            }
        }
        with open(os.path.join(RESULTS_DIR, "comparison_with_experiment13.json"), "w") as f:
            json.dump(comp_exp13, f, indent=4)

    # ── Configuration & Metadata ───────────────────────────────────────────────
    total_duration = time.time() - start_total_time
    total_images_inferred = len(y_val) + len(y_test)
    avg_inf_time = (val_inference_duration + test_inference_duration) / total_images_inferred
    
    threshold_config = {
        "selected_threshold": selected_theta,
        "target_class": "melanoma (mel, class 4)",
        "decision_rule": "if P_ensemble[mel] >= theta then mel else argmax(P_ensemble)",
        "search_grid": THRESHOLD_GRID,
        "selection_dataset": "validation (N=1,010)",
        "selection_criterion": selected_threshold_data["selection_rule"],
        "selection_tie_breaker": selection_tie_breaker,
        "status": "frozen"
    }
    with open(os.path.join(RESULTS_DIR, "threshold_configuration.json"), "w") as f:
        json.dump(threshold_config, f, indent=4)
        
    inference_config = {
        "model_A": {
            "name": "Experiment 8 (Tempered Class Weights alpha=0.75)",
            "checkpoint": CHECKPOINT_EXP8,
            "architecture": "DermaAI_MobileNetV3",
            "weight": WEIGHT_EXP8
        },
        "model_B": {
            "name": "Experiment 13 (Melanoma-Focused Tempered Class Weighting 1.20x)",
            "checkpoint": CHECKPOINT_EXP13,
            "architecture": "DermaAI_MobileNetV3",
            "weight": WEIGHT_EXP13
        },
        "ensemble_lambda": LAMBDA_ENSEMBLE,
        "input_resolution": "224x224",
        "dataset_split": "HAM10000 (Val=1010, Test=988)",
        "retraining": False,
        "parameters_updated": 0,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "seed": SEED,
        "validation_inference_duration_seconds": val_inference_duration,
        "test_inference_duration_seconds": test_inference_duration,
        "average_inference_time_per_image_seconds": avg_inf_time,
        "total_runtime_seconds": total_duration
    }
    with open(os.path.join(RESULTS_DIR, "inference_configuration.json"), "w") as f:
        json.dump(inference_config, f, indent=4)
        
    experiment_summary = {
        "experiment_name": "Experiment 15: Validation-Calibrated Melanoma Threshold on Exp 14 Ensemble",
        "selected_threshold": selected_theta,
        "selection_tie_breaker": selection_tie_breaker,
        "validation_melanoma_f1": float(selected_row["melanoma_f1"]),
        "validation_melanoma_recall": float(selected_row["melanoma_recall"]),
        "test_accuracy": m_exp15["accuracy"],
        "test_balanced_accuracy": m_exp15["balanced_accuracy"],
        "test_macro_f1": m_exp15["macro_f1"],
        "test_weighted_f1": m_exp15["weighted_f1"],
        "test_melanoma_precision": m_exp15["melanoma_precision"],
        "test_melanoma_recall": m_exp15["melanoma_recall"],
        "test_melanoma_f1": m_exp15["melanoma_f1"],
        "test_melanoma_tp": m_exp15["melanoma_tp"],
        "test_melanoma_fp": m_exp15["melanoma_fp"],
        "test_mel_to_nv_errors": m_exp15["melanoma_to_nv_errors"],
        "test_mel_to_bkl_errors": m_exp15["melanoma_to_bkl_errors"],
        "test_nv_to_mel_errors": m_exp15["nv_to_mel_errors"],
        "total_changed_predictions_vs_exp14": total_changed,
        "net_correct_change_vs_exp14": net_correct_change,
        "computational_cost": {
            "validation_inference_seconds": val_inference_duration,
            "test_inference_seconds": test_inference_duration,
            "total_experiment_seconds": total_duration,
            "training_time_seconds": 0.0
        }
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(experiment_summary, f, indent=4)

    # ── Final Console Output ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION RESULTS: EXP14 ENSEMBLE vs EXP15 THRESHOLDED")
    print("="*70)
    print(f"{'Metric':<25} | {'Exp 14 (Argmax)':<15} | {'Exp 15 (θ=' + str(selected_theta) + ')':<15} | {'Delta (Exp15 - Exp14)'}")
    print("-"*75)
    metrics_to_print = [
        ("Accuracy", m_exp14["accuracy"]*100, m_exp15["accuracy"]*100, "%"),
        ("Balanced Accuracy", m_exp14["balanced_accuracy"]*100, m_exp15["balanced_accuracy"]*100, "%"),
        ("Macro Precision", m_exp14["macro_precision"], m_exp15["macro_precision"], "f"),
        ("Macro Recall", m_exp14["macro_recall"], m_exp15["macro_recall"], "f"),
        ("Macro F1", m_exp14["macro_f1"], m_exp15["macro_f1"], "f"),
        ("Weighted F1", m_exp14["weighted_f1"], m_exp15["weighted_f1"], "f"),
        ("Melanoma Recall", m_exp14["melanoma_recall"]*100, m_exp15["melanoma_recall"]*100, "%"),
        ("Melanoma Precision", m_exp14["melanoma_precision"]*100, m_exp15["melanoma_precision"]*100, "%"),
        ("Melanoma F1", m_exp14["melanoma_f1"], m_exp15["melanoma_f1"], "f"),
        ("Melanoma TP (/106)", m_exp14["melanoma_tp"], m_exp15["melanoma_tp"], "d"),
        ("Melanoma FP", m_exp14["melanoma_fp"], m_exp15["melanoma_fp"], "d"),
        ("Mel -> NV Errors", m_exp14["melanoma_to_nv_errors"], m_exp15["melanoma_to_nv_errors"], "d"),
        ("Mel -> BKL Errors", m_exp14["melanoma_to_bkl_errors"], m_exp15["melanoma_to_bkl_errors"], "d"),
        ("NV Recall", m_exp14["nv_recall"]*100, m_exp15["nv_recall"]*100, "%"),
        ("NV -> MEL False Pos", m_exp14["nv_to_mel_errors"], m_exp15["nv_to_mel_errors"], "d")
    ]
    for name, v14, v15, fmt in metrics_to_print:
        d = v15 - v14
        if fmt == "%":
            print(f"{name:<25} | {v14:>13.2f}% | {v15:>13.2f}% | {d:>+14.2f}%")
        elif fmt == "f":
            print(f"{name:<25} | {v14:>15.4f} | {v15:>15.4f} | {d:>+16.4f}")
        elif fmt == "d":
            print(f"{name:<25} | {int(v14):>15d} | {int(v15):>15d} | {int(d):>+16d}")
            
    print("="*70)
    print(f"Total Changed Predictions (Test N=988)     : {total_changed} ({total_changed/N*100:.2f}%)")
    print(f"  Incorrect -> Correct (Saved)             : {inc_to_cor}")
    print(f"  Correct -> Incorrect (Corrupted)         : {cor_to_inc}")
    print(f"  Incorrect -> Incorrect (Class shifted)   : {inc_to_inc}")
    print(f"  Net Correct Change                       : {net_correct_change:+d}")
    print(f"All 21 Experiment 15 artifacts saved to: {RESULTS_DIR}")
    print("="*70)
    print("Experiment 15 complete. STOPPING as required -- awaiting next instruction.")


if __name__ == "__main__":
    main()
