# DermaAI: Hackathon & Live Demo Script (2.5 - 3 Minutes)

This script provides an exact, word-for-word timed demonstration for judges, conferences, or academic project reviews.

---

## Demo Overview Table

| Timestamp | Phase | Screen Action | Talking Point / Core Message |
|---|---|---|---|
| **0:00 – 0:30** | Introduction & Problem | Landing page hero & Research Notice | The severe class imbalance of skin lesions & need for accountable AI |
| **0:30 – 1:00** | Architecture & Integrity | Click "Model Specifications" modal | MobileNetV3 + CBAM, 85/15 frozen ensemble, zero retraining |
| **1:00 – 1:45** | Live Inference | Drag & drop dermoscopic lesion | Real-time prediction, probability breakdown across all 7 classes |
| **1:45 – 2:20** | Dual Grad-CAM Attention | Scroll to Grad-CAM heatmaps | Explainability: Comparing Exp8 (general) vs Exp13 (melanoma-weighted) |
| **2:20 – 2:45** | Research & Ethics Rigor | Point to Experiment 15 note & disclaimers | Leakage prevention, threshold verification, clinical safety boundaries |
| **2:45 – 3:00** | Conclusion & Q&A | Overview of complete pipeline | Production-ready, reproducible, transparent deep learning |

---

## Minute-by-Minute Script

### [0:00 - 0:30] Introduction: The Challenge in Dermoscopy
**Speaker:**
> "Hello everyone. Welcome to **DermaAI**.
>
> Skin cancer classification on dermoscopic images is notoriously challenging due to extreme class imbalance: benign melanocytic nevi outnumber lethal melanomas by more than 6 to 1 in standard clinical benchmarks like HAM10000.
>
> Traditional models often sacrifice melanoma sensitivity to achieve high overall accuracy, or suffer false alarms. DermaAI solves this with a **validation-calibrated, frozen probability ensemble** combined with **dual convolutional attention explainability**."

**Visual Action:**
- Show the DermaAI homepage.
- Point out the prominent top alert banner: *"Academic Research Demonstrator: Strictly for image-level classification on retrospective HAM10000 data. Not for clinical diagnosis."*

---

### [0:30 - 1:00] The Engineering: Complementary Frozen Models
**Speaker:**
> "Rather than an uninterpretable single black-box, DermaAI blends two complementary neural networks:
> 1. **Experiment 8**, our baseline MobileNetV3-Large with CBAM attention and tempered class weighting ($\alpha=0.75$), which achieves **80.67% test accuracy** and **70.18% Macro F1**.
> 2. **Experiment 13**, which incorporates a targeted $1.20\times$ melanoma weight, boosting melanoma recall from **48.11% up to 59.43%**.
>
> In Phase 14, we calibrated the optimal convex combination strictly on validation data: **85% Experiment 8 and 15% Experiment 13**.
> On the frozen 988-image test set, this ensemble achieves **80.57% accuracy**, **75.69% balanced accuracy**, and **52.43% melanoma F1**—our strongest overall generalization."

**Visual Action:**
- Click the **"Model Architecture & Specs"** button to reveal the modal.
- Highlight the SHA-256 checksums and parameter count: *4,326,297 parameters per model*.

---

### [1:00 - 1:45] Live Demo: Real-Time Multiclass Inference
**Speaker:**
> "Let's demonstrate live inference. I'll drag and drop a dermoscopic lesion into the upload dropzone.
>
> When I click 'Analyze Lesion Image', you will observe our multi-stage pipeline execute:
> First, image validation and resize to $224 \times 224$;
> Second, forward passes through both frozen PyTorch models;
> Third, probability blending and argmax classification;
> And fourth, backward gradient passes to compute dual Grad-CAM attention heatmaps."

**Visual Action:**
- Drop an image into the dropzone.
- Click **"Analyze Lesion Image"**.
- Point out the loading indicator states: *Loading image... $\to$ Running ensemble... $\to$ Generating Grad-CAM...*
- Display the result card: Predicted Class, Confidence percentage, and Top-3 distribution.

---

### [1:45 - 2:20] Explainability: Dual Grad-CAM Comparison
**Speaker:**
> "Crucially, DermaAI does not simply give a label—it explains *where* each model looked.
>
> Here we display side-by-side Grad-CAM visualizations targeting the CBAM convolutional attention layer (`model.attention`).
>
> Notice the comparison:
> - On the left, **Experiment 8** captures the broader lesion morphology and border symmetry.
> - On the right, **Experiment 13** focuses heavily on focal pigment networks and irregular peripheral areas characteristic of melanocytic lesions.
>
> This gives researchers instant visual insight into model agreement and divergence."

**Visual Action:**
- Scroll down to the **Dual Grad-CAM Explainability** section.
- Hover over the heatmaps showing the overlay with the original dermoscopic image.

---

### [2:20 - 2:45] Rigor, Safety & Disclaimers
**Speaker:**
> "We placed equal priority on scientific rigor and ethical boundaries:
> - In **Experiment 15**, we investigated decision threshold tuning ($\theta^*=0.50$) and proved that the natural argmax rule is mathematically optimal for this ensemble.
> - The 988-image test set was strictly locked during all tuning; zero test images were ever used for threshold selection or demo serving.
> - Every API response and UI screen enforces a strict **RESEARCH USE ONLY** notice. We make zero clinical efficacy claims."

**Visual Action:**
- Point to the **"Experiment 15 Verification Note"** and the footer research disclaimer.

---

### [2:45 - 3:00] Conclusion
**Speaker:**
> "DermaAI demonstrates that modern lightweight mobile architectures, principled tempered class weighting, and validation-calibrated ensembles can deliver high sensitivity, strong multiclass generalization, and transparent explainability in under 4.4 million parameters.
>
> Thank you, and we welcome your questions!"

---

## Common Questions & Quick Answers

- **Q: Why MobileNetV3-Large instead of ViT or ResNet-50?**
  *A: MobileNetV3-Large with CBAM achieves competitive representation (80.57% test accuracy) with only 4.3M parameters and ~15ms CPU inference time, making it deployable on edge and mobile devices.*

- **Q: Why 85% Exp8 and 15% Exp13?**
  *A: Phase 14 evaluated convex combinations $\lambda \in [0, 1]$ strictly on the 1,010 validation images. $\lambda=0.15$ maximized validation Macro F1 while boosting melanoma recall.*

- **Q: Did you retrain anything for the demo?**
  *A: Zero retraining. Both checkpoints are strictly frozen with SHA-256 verification.*
