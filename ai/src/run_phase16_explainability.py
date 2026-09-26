import os
import sys
import json
import time
import random
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import seaborn as sns

try:
    from .model import DermaAI_MobileNetV3
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING, IMAGENET_MEAN, IMAGENET_STD

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
RESULTS_DIR   = os.path.join(AI_DIR, "results", "phase16_explainability")

CHECKPOINT_EXP8  = os.path.join(MODELS_DIR, "experiment8_best_model.pt")
CHECKPOINT_EXP13 = os.path.join(MODELS_DIR, "experiment13_best_model.pt")

VAL_CSV_PATH  = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(SPLITS_DIR, "test.csv")

WEIGHT_EXP8  = 0.85
WEIGHT_EXP13 = 0.15
BATCH_SIZE   = 32

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX     = CLASS_MAPPING["mel"]
NV_IDX      = CLASS_MAPPING["nv"]

# Subdirectories
DIR_GRADCAM      = os.path.join(RESULTS_DIR, "gradcam")
DIR_EXP8_CAM     = os.path.join(DIR_GRADCAM, "exp8")
DIR_EXP13_CAM    = os.path.join(DIR_GRADCAM, "exp13")
DIR_COMP_CAM     = os.path.join(DIR_GRADCAM, "comparisons")
DIR_SELECTED_CAM = os.path.join(DIR_GRADCAM, "selected_cases")
DIR_ERROR_ANAL   = os.path.join(RESULTS_DIR, "error_analysis")
DIR_REPORTS      = os.path.join(RESULTS_DIR, "reports")
DIR_FIGURES      = os.path.join(RESULTS_DIR, "figures")

for d in [RESULTS_DIR, DIR_GRADCAM, DIR_EXP8_CAM, DIR_EXP13_CAM, DIR_COMP_CAM,
          DIR_SELECTED_CAM, DIR_ERROR_ANAL, DIR_REPORTS, DIR_FIGURES]:
    os.makedirs(d, exist_ok=True)


# ── Grad-CAM Implementation ───────────────────────────────────────────────────
class GradCAM:
    """
    Standard Grad-CAM on target convolutional feature layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.f_hook = target_layer.register_forward_hook(self._forward_hook)
        self.b_hook = target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, inp, out):
        self.activations = out
        
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
        
    def generate_heatmap(self, input_tensor, target_class_idx):
        """
        input_tensor: [1, 3, 224, 224] with requires_grad=True
        target_class_idx: int (0 to 6)
        Returns: [H, W] normalized heatmap in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, target_class_idx]
        score.backward(retain_graph=False)
        
        # GAP of gradients over spatial dimensions (height, width)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, 960, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [1, 1, 7, 7]
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        # Quality check
        assert np.isfinite(cam).all(), "Heatmap contains NaN or Inf values!"
        return cam

    def cleanup(self):
        self.f_hook.remove()
        self.b_hook.remove()


def overlay_cam_on_image(img_pil, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    img_pil: PIL.Image
    heatmap: 2D numpy array [7, 7] in [0, 1]
    Returns: PIL.Image of overlay
    """
    img_np = np.array(img_pil.convert("RGB"))
    h, w, _ = img_np.shape
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (JET or TURBO)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend overlay
    overlay = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * img_np)
    return Image.fromarray(overlay), heatmap_resized


# ── Step 1: Model Loading & Freezing ───────────────────────────────────────────
def load_models(device):
    print("="*70)
    print("STEP 1: LOAD & VERIFY EXP8 AND EXP13 CHECKPOINTS (FROZEN EVAL MODE)")
    print("="*70)
    model8 = DermaAI_MobileNetV3(num_classes=7)
    model8.load_state_dict(torch.load(CHECKPOINT_EXP8, map_location=device))
    model8 = model8.to(device).eval()
    for p in model8.parameters():
        p.requires_grad = False
        
    model13 = DermaAI_MobileNetV3(num_classes=7)
    model13.load_state_dict(torch.load(CHECKPOINT_EXP13, map_location=device))
    model13 = model13.to(device).eval()
    for p in model13.parameters():
        p.requires_grad = False
        
    p8 = sum(p.numel() for p in model8.parameters())
    p13 = sum(p.numel() for p in model13.parameters())
    assert p8 == 4326297 and p13 == 4326297
    print(f"  Model A (Exp8)  : DermaAI_MobileNetV3 | Params: {p8:,} | Target: model.attention")
    print(f"  Model B (Exp13) : DermaAI_MobileNetV3 | Params: {p13:,} | Target: model.attention")
    print("="*70 + "\n")
    return model8, model13


# ── Step 2: Probability Extraction for Quantitative Error Analysis ─────────────
def extract_predictions(model8, model13, dataloader, device):
    all_P8 = []
    all_P13 = []
    all_y = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting Predictions", leave=False):
            imgs = imgs.to(device)
            p8 = torch.softmax(model8(imgs), dim=1).cpu().numpy()
            p13 = torch.softmax(model13(imgs), dim=1).cpu().numpy()
            all_P8.append(p8)
            all_P13.append(p13)
            all_y.append(lbls.numpy())
            
    all_P8 = np.vstack(all_P8)
    all_P13 = np.vstack(all_P13)
    all_y = np.concatenate(all_y)
    all_Pens = WEIGHT_EXP8 * all_P8 + WEIGHT_EXP13 * all_P13
    return all_P8, all_P13, all_Pens, all_y


# ── Step 3: Run Full Phase 16 Routine ──────────────────────────────────────────
def main():
    start_time = time.time()
    device = torch.device("cpu")
    print("\n" + "="*70)
    print("PHASE 16: GRAD-CAM EXPLAINABILITY + FINAL ERROR ANALYSIS")
    print(f"Device : {device} | PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}")
    print("="*70)
    
    eval_transform = get_transforms(is_train=False)
    
    val_df = pd.read_csv(VAL_CSV_PATH)
    val_dataset = HAM10000Dataset(val_df, transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model8, model13 = load_models(device)
    
    # ── Quantitative Inference on Validation Split ────────────────────────────
    print("Extracting full predictions on Validation Set (N=1,010)...")
    P8_val, P13_val, Pens_val, y_val = extract_predictions(model8, model13, val_loader, device)
    
    pred8_val = np.argmax(P8_val, axis=1)
    pred13_val = np.argmax(P13_val, axis=1)
    pred_ens_val = np.argmax(Pens_val, axis=1)
    
    # Compute error categories for all 1,010 validation images
    error_records = []
    for idx in range(len(val_df)):
        row = val_df.iloc[idx]
        img_id = row["image_id"]
        gt = int(y_val[idx])
        p8_class = int(pred8_val[idx])
        p13_class = int(pred13_val[idx])
        ens_class = int(pred_ens_val[idx])
        
        conf8 = float(P8_val[idx, p8_class])
        conf13 = float(P13_val[idx, p13_class])
        conf_ens = float(Pens_val[idx, ens_class])
        
        mel_p8 = float(P8_val[idx, MEL_IDX])
        mel_p13 = float(P13_val[idx, MEL_IDX])
        mel_pens = float(Pens_val[idx, MEL_IDX])
        
        # Categorize
        if ens_class == gt:
            if p8_class == gt and p13_class != gt:
                cat = "exp8_only_correct"
            elif p8_class != gt and p13_class == gt:
                cat = "exp13_only_correct"
            else:
                cat = "correct"
        else:
            if gt == MEL_IDX and ens_class == NV_IDX:
                cat = "mel_to_nv"
            elif gt == MEL_IDX and ens_class == CLASS_MAPPING["bkl"]:
                cat = "mel_to_bkl"
            elif gt == MEL_IDX and ens_class == CLASS_MAPPING["akiec"]:
                cat = "mel_to_akiec"
            elif gt == MEL_IDX and ens_class == CLASS_MAPPING["bcc"]:
                cat = "mel_to_bcc"
            elif gt == MEL_IDX and ens_class == CLASS_MAPPING["df"]:
                cat = "mel_to_df"
            elif gt == NV_IDX and ens_class == MEL_IDX:
                cat = "nv_to_mel"
            elif p8_class != p13_class and p8_class != gt and p13_class != gt:
                cat = "both_wrong_disagreement"
            else:
                cat = "other_error"
                
        error_records.append({
            "val_index": idx,
            "image_id": img_id,
            "lesion_id": row["lesion_id"],
            "image_path": row["image_path"],
            "ground_truth_label": CLASS_NAMES[gt],
            "ground_truth_idx": gt,
            "exp8_prediction": CLASS_NAMES[p8_class],
            "exp13_prediction": CLASS_NAMES[p13_class],
            "ensemble_prediction": CLASS_NAMES[ens_class],
            "exp8_confidence": conf8,
            "exp13_confidence": conf13,
            "ensemble_confidence": conf_ens,
            "gt_probability_exp8": float(P8_val[idx, gt]),
            "gt_probability_exp13": float(P13_val[idx, gt]),
            "gt_probability_ensemble": float(Pens_val[idx, gt]),
            "melanoma_probability_exp8": mel_p8,
            "melanoma_probability_exp13": mel_p13,
            "melanoma_probability_ensemble": mel_pens,
            "error_category": cat,
            "exp8_vs_exp13_disagreement": bool(p8_class != p13_class)
        })
        
    error_df = pd.DataFrame(error_records)
    error_df.to_csv(os.path.join(DIR_ERROR_ANAL, "error_summary.csv"), index=False)
    with open(os.path.join(DIR_ERROR_ANAL, "error_summary.json"), "w") as f:
        json.dump(error_records, f, indent=4)
        
    print(f"Error summary saved: {os.path.join(DIR_ERROR_ANAL, 'error_summary.csv')}")
    
    # ── Final Metrics Table ────────────────────────────────────────────────────
    # Load previously verified test benchmarks for Exp 8, Exp 13, Exp 14
    metrics_table = [
        {"Model": "Experiment 8 (Tempered alpha=0.75)", "Accuracy": 80.67, "Balanced_Accuracy": 75.37, "Macro_F1": 0.7018, "Weighted_F1": 0.8107, "Melanoma_Recall": 48.11, "Melanoma_Precision": 56.04, "Melanoma_F1": 51.78},
        {"Model": "Experiment 13 (Melanoma-Focused 1.20x)", "Accuracy": 78.04, "Balanced_Accuracy": 73.14, "Macro_F1": 0.6758, "Weighted_F1": 0.7892, "Melanoma_Recall": 59.43, "Melanoma_Precision": 45.32, "Melanoma_F1": 51.43},
        {"Model": "Experiment 14 (Probability Ensemble lambda=0.15)", "Accuracy": 80.57, "Balanced_Accuracy": 75.69, "Macro_F1": 0.7051, "Weighted_F1": 0.8106, "Melanoma_Recall": 50.94, "Melanoma_Precision": 54.00, "Melanoma_F1": 52.43}
    ]
    pd.DataFrame(metrics_table).to_csv(os.path.join(RESULTS_DIR, "final_model_metrics_table.csv"), index=False)

    # ── Step 4: Deterministic Case Selection for Grad-CAM ──────────────────────
    print("\nSelecting deterministic representative cases for Grad-CAM...")
    # Group 1: Correct Melanoma (GT=mel, Ens=mel) -> Select first 4
    g1_cases = error_df[(error_df["ground_truth_label"] == "mel") & (error_df["ensemble_prediction"] == "mel")].head(4).to_dict("records")
    # Group 2: Mel -> NV errors (GT=mel, Ens=nv) -> Select first 4
    g2_cases = error_df[error_df["error_category"] == "mel_to_nv"].head(4).to_dict("records")
    # Group 3: Mel -> BKL errors (GT=mel, Ens=bkl) -> Select first 4
    g3_cases = error_df[error_df["error_category"] == "mel_to_bkl"].head(4).to_dict("records")
    # Group 4: NV -> Mel false positives (GT=nv, Ens=mel) -> Select first 4
    g4_cases = error_df[error_df["error_category"] == "nv_to_mel"].head(4).to_dict("records")
    # Group 5: Exp8 vs Exp13 disagreements
    # 2 Exp8 correct & Exp13 wrong; 2 Exp8 wrong & Exp13 correct
    g5_cases_b = error_df[error_df["error_category"] == "exp8_only_correct"].head(2).to_dict("records")
    g5_cases_c = error_df[error_df["error_category"] == "exp13_only_correct"].head(2).to_dict("records")
    g5_cases = g5_cases_b + g5_cases_c
    
    print(f"Selected: Group 1 ({len(g1_cases)}), Group 2 ({len(g2_cases)}), Group 3 ({len(g3_cases)}), Group 4 ({len(g4_cases)}), Group 5 ({len(g5_cases)})")

    # ── Step 5: Generate Grad-CAM Heatmaps & Composite Figures ─────────────────
    print("\nGenerating Grad-CAM visualizations...")
    cam_model8 = GradCAM(model8, model8.attention)
    cam_model13 = GradCAM(model13, model13.attention)
    
    # Target Layer metadata
    target_layer_name = "model.attention (CBAMBlock)"
    target_layer_shape = "[B, 960, 7, 7]"
    
    case_groups = [
        ("Figure 1: Correct Melanoma Cases (GT=mel, Ens=mel)", g1_cases, "figure1_correct_melanoma.png", "correct_mel"),
        ("Figure 2: Melanoma → Nevus Errors (GT=mel, Ens=nv)", g2_cases, "figure2_melanoma_to_nv_errors.png", "mel_to_nv"),
        ("Figure 3: Melanoma → Benign Keratosis Errors (GT=mel, Ens=bkl)", g3_cases, "figure3_melanoma_to_bkl_errors.png", "mel_to_bkl"),
        ("Figure 4: Nevus → Melanoma False Positives (GT=nv, Ens=mel)", g4_cases, "figure4_nv_to_melanoma_false_positives.png", "nv_to_mel"),
        ("Figure 5: Exp8 vs Exp13 Disagreement Cases", g5_cases, "figure5_exp8_vs_exp13_disagreements.png", "disagreement")
    ]
    
    saved_cam_metadata = []
    
    for fig_title, cases, fig_filename, grp_tag in case_groups:
        n_cases = len(cases)
        fig, axes = plt.subplots(n_cases, 4, figsize=(18, 4.2 * n_cases))
        if n_cases == 1:
            axes = np.expand_dims(axes, 0)
            
        for row_idx, c in enumerate(cases):
            img_path = c["image_path"]
            img_id = c["image_id"]
            img_pil = Image.open(img_path).convert("RGB")
            
            # Prepare tensor for Grad-CAM
            t_input = eval_transform(img_pil).unsqueeze(0).to(device)
            t_input.requires_grad = True
            
            # Determine target classes to visualize
            gt_idx = c["ground_truth_idx"]
            pred_idx = CLASS_MAPPING[c["ensemble_prediction"]]
            
            # Primary target: for correct cases, target is pred; for errors, target is GT or competing
            target_class_8 = CLASS_MAPPING[c["exp8_prediction"]]
            target_class_13 = CLASS_MAPPING[c["exp13_prediction"]]
            
            # Generate CAMs
            cam8_pred = cam_model8.generate_heatmap(t_input, target_class_8)
            cam13_pred = cam_model13.generate_heatmap(t_input, target_class_13)
            
            # Also generate ground truth CAM if error case
            cam8_gt = cam_model8.generate_heatmap(t_input, gt_idx)
            cam13_gt = cam_model13.generate_heatmap(t_input, gt_idx)
            
            overlay8_pred, _ = overlay_cam_on_image(img_pil, cam8_pred)
            overlay13_pred, _ = overlay_cam_on_image(img_pil, cam13_pred)
            overlay8_gt, _ = overlay_cam_on_image(img_pil, cam8_gt)
            overlay13_gt, _ = overlay_cam_on_image(img_pil, cam13_gt)
            
            # Save individual image files
            orig_save_path = os.path.join(DIR_SELECTED_CAM, f"{grp_tag}_{row_idx+1}_{img_id}_original.png")
            cam8_save_path = os.path.join(DIR_EXP8_CAM, f"{grp_tag}_{row_idx+1}_{img_id}_exp8_cam.png")
            cam13_save_path = os.path.join(DIR_EXP13_CAM, f"{grp_tag}_{row_idx+1}_{img_id}_exp13_cam.png")
            img_pil.save(orig_save_path)
            overlay8_pred.save(cam8_save_path)
            overlay13_pred.save(cam13_save_path)
            
            # Plot in composite figure
            # Col 0: Original
            axes[row_idx, 0].imshow(img_pil)
            axes[row_idx, 0].set_title(f"Image: {img_id}\nGT: {c['ground_truth_label'].upper()}", fontsize=11, fontweight="bold")
            axes[row_idx, 0].axis("off")
            
            # Col 1: Exp 8 CAM (Predicted)
            axes[row_idx, 1].imshow(overlay8_pred)
            axes[row_idx, 1].set_title(f"Exp 8 CAM (Pred: {c['exp8_prediction'].upper()})\nConf: {c['exp8_confidence']*100:.1f}% | Mel: {c['melanoma_probability_exp8']*100:.1f}%", fontsize=10)
            axes[row_idx, 1].axis("off")
            
            # Col 2: Exp 13 CAM (Predicted)
            axes[row_idx, 2].imshow(overlay13_pred)
            axes[row_idx, 2].set_title(f"Exp 13 CAM (Pred: {c['exp13_prediction'].upper()})\nConf: {c['exp13_confidence']*100:.1f}% | Mel: {c['melanoma_probability_exp13']*100:.1f}%", fontsize=10)
            axes[row_idx, 2].axis("off")
            
            # Col 3: Exp13 CAM for GT or Competing Overlay
            axes[row_idx, 3].imshow(overlay13_gt if pred_idx != gt_idx else overlay13_pred)
            col3_title = f"Exp 13 CAM (GT: {CLASS_NAMES[gt_idx].upper()})\nGT Prob: {c['gt_probability_exp13']*100:.1f}%" if pred_idx != gt_idx else f"Ensemble Decision: {c['ensemble_prediction'].upper()}\nConf: {c['ensemble_confidence']*100:.1f}% | Mel: {c['melanoma_probability_ensemble']*100:.1f}%"
            axes[row_idx, 3].set_title(col3_title, fontsize=10, fontweight="bold")
            axes[row_idx, 3].axis("off")
            
            saved_cam_metadata.append({
                "case_group": grp_tag,
                "image_id": img_id,
                "ground_truth": c["ground_truth_label"],
                "exp8_prediction": c["exp8_prediction"],
                "exp13_prediction": c["exp13_prediction"],
                "ensemble_prediction": c["ensemble_prediction"],
                "exp8_confidence": c["exp8_confidence"],
                "exp13_confidence": c["exp13_confidence"],
                "ensemble_confidence": c["ensemble_confidence"]
            })
            
        plt.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        fig_save_path = os.path.join(DIR_FIGURES, fig_filename)
        fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig_save_path}")
        
    cam_model8.cleanup()
    cam_model13.cleanup()
    
    # ── Metadata Configuration ─────────────────────────────────────────────────
    gradcam_config = {
        "architecture": "DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)",
        "checkpoints": {
            "exp8": CHECKPOINT_EXP8,
            "exp13": CHECKPOINT_EXP13
        },
        "target_layer_name": target_layer_name,
        "target_layer_description": "Final refined convolutional feature map immediately preceding adaptive average pooling",
        "target_layer_shape": target_layer_shape,
        "input_resolution": "224x224",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "interpolation": "Bilinear",
        "cam_formulation": "Standard Grad-CAM: GAP gradients * activations -> ReLU -> min-max normalize -> resize overlay",
        "colormap": "JET",
        "overlay_alpha": 0.5,
        "verification_quality_checks": {
            "finite_values": True,
            "no_nan": True,
            "no_inf": True,
            "range": "[0, 1]",
            "model_weights_unmodified": True
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(RESULTS_DIR, "gradcam_configuration.json"), "w") as f:
        json.dump(gradcam_config, f, indent=4)
        
    # ── Summary JSON ───────────────────────────────────────────────────────────
    total_val = len(error_df)
    total_correct = int((error_df["ensemble_prediction"] == error_df["ground_truth_label"]).sum())
    total_incorrect = total_val - total_correct
    
    mel_df = error_df[error_df["ground_truth_label"] == "mel"]
    total_mel = len(mel_df)
    mel_correct = int((mel_df["ensemble_prediction"] == "mel").sum())
    mel_to_nv = int((mel_df["ensemble_prediction"] == "nv").sum())
    mel_to_bkl = int((mel_df["ensemble_prediction"] == "bkl").sum())
    mel_to_akiec = int((mel_df["ensemble_prediction"] == "akiec").sum())
    mel_to_bcc = int((mel_df["ensemble_prediction"] == "bcc").sum())
    mel_to_df = int((mel_df["ensemble_prediction"] == "df").sum())
    
    nv_df = error_df[error_df["ground_truth_label"] == "nv"]
    nv_to_mel = int((nv_df["ensemble_prediction"] == "mel").sum())
    
    exp8_correct_mask = (error_df["exp8_prediction"] == error_df["ground_truth_label"])
    exp13_correct_mask = (error_df["exp13_prediction"] == error_df["ground_truth_label"])
    disagreement_mask = (error_df["exp8_prediction"] != error_df["exp13_prediction"])
    
    exp_summary = {
        "phase": "Phase 16 — Grad-CAM Explainability + Final Error Analysis",
        "dataset_analyzed": "HAM10000 Validation Split (N=1,010)",
        "total_validation_samples": total_val,
        "ensemble_correct_samples": total_correct,
        "ensemble_incorrect_samples": total_incorrect,
        "validation_accuracy": float(total_correct / total_val),
        "melanoma_analysis": {
            "total_melanoma": total_mel,
            "melanoma_correct": mel_correct,
            "melanoma_recall": float(mel_correct / total_mel),
            "melanoma_to_nv": mel_to_nv,
            "melanoma_to_bkl": mel_to_bkl,
            "melanoma_to_akiec": mel_to_akiec,
            "melanoma_to_bcc": mel_to_bcc,
            "melanoma_to_df": mel_to_df,
            "nv_to_mel_false_positives": nv_to_mel
        },
        "model_disagreement_analysis": {
            "disagreement_count": int(disagreement_mask.sum()),
            "disagreement_rate": float(disagreement_mask.sum() / total_val),
            "exp8_only_correct": int((exp8_correct_mask & ~exp13_correct_mask).sum()),
            "exp13_only_correct": int((~exp8_correct_mask & exp13_correct_mask).sum()),
            "both_wrong_different_predictions": int((~exp8_correct_mask & ~exp13_correct_mask & disagreement_mask).sum())
        },
        "figures_generated": [
            "figure1_correct_melanoma.png",
            "figure2_melanoma_to_nv_errors.png",
            "figure3_melanoma_to_bkl_errors.png",
            "figure4_nv_to_melanoma_false_positives.png",
            "figure5_exp8_vs_exp13_disagreements.png"
        ],
        "runtime_seconds": time.time() - start_time
    }
    with open(os.path.join(RESULTS_DIR, "explainability_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=4)
        
    print("\n" + "="*70)
    print("PHASE 16 EXECUTION COMPLETE")
    print(f"Total Validation Analyzed : {total_val} images")
    print(f"Ensemble Accuracy (Val)   : {total_correct/total_val*100:.2f}% ({total_correct}/{total_val})")
    print(f"Melanoma Recall (Val)     : {mel_correct/total_mel*100:.2f}% ({mel_correct}/{total_mel})")
    print(f"Mel -> NV Errors (Val)    : {mel_to_nv}")
    print(f"Mel -> BKL Errors (Val)   : {mel_to_bkl}")
    print(f"NV -> Mel Alarms (Val)    : {nv_to_mel}")
    print(f"All 5 publication figures saved to: {DIR_FIGURES}")
    print(f"All error summaries saved to: {DIR_ERROR_ANAL}")
    print("="*70)


if __name__ == "__main__":
    main()
