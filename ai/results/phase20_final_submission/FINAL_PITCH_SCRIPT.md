# DermaAI: Final Hackathon Pitch Script (2.5 - 3 Minutes)

This spoken script is timed to deliver a complete, compelling, and scientifically rigorous hackathon presentation in 2 minutes and 50 seconds.

---

## Pitch Structure & Timing Guide

| Timestamp | Topic | Visual Focus |
|---|---|---|
| **0:00 – 0:20** | Introduction & Problem | Slide 1–2 / Problem |
| **0:20 – 0:45** | Dataset & Imbalance Challenge | Slide 3 / Dataset Distribution |
| **0:45 – 1:10** | 15-Experiment Research Methodology | Slide 4 / Progression Trend |
| **1:10 – 1:30** | Final 85/15 Probability Ensemble | Slide 6 / Ensemble Math |
| **1:30 – 2:10** | Live Website Demonstration | DermaAI Web Application |
| **2:10 – 2:35** | Dual Grad-CAM & Probability Distribution | Grad-CAM Panel & Result Card |
| **2:35 – 2:50** | Quantitative Error Analysis | Slide 9 / Confusion Matrix |
| **2:50 – 3:00** | Scientific Limitations & Conclusion | Slide 12–14 / Conclusion |

---

## Spoken Presentation Script

### [0:00 - 0:20] Introduction & Problem
> "Good morning, judges and fellow researchers. This is **DermaAI**.
>
> Dermoscopic image classification is a cornerstone problem in computer-aided triage. However, dermatological datasets face a severe structural bottleneck: extreme class imbalance and massive visual overlap between benign and malignant skin lesions.
>
> Our goal was not simply to train a black-box model, but to build a transparent, leakage-aware research demonstrator grounded in controlled experimentation."

---

### [0:20 - 0:45] Dataset & Imbalance Challenge
> "We studied the widely benchmarked HAM10000 dataset across seven diagnostic lesion categories.
>
> In HAM10000, benign melanocytic nevi outnumber lethal melanomas by more than six to one. Under standard cross-entropy training, standard neural networks easily achieve deceptive 80% accuracy simply by ignoring dangerous minority classes.
>
> To ensure rigorous evaluation, we enforced a strict lesion-level split with seed 42:
> 8,017 training images, 1,010 validation images, and a permanently locked benchmark split of 988 test images.
> Crucially, our validation and benchmark populations remained strictly separated throughout."

---

### [0:45 - 1:10] 15-Experiment Research Methodology
> "Rather than guessing hyperparameters, we conducted 15 controlled single-variable experiments:
>
> We started with uniform cross-entropy in Experiment 1, progressed through inverse-frequency weighting in Experiment 2, and introduced MobileNetV3-Large with CBAM attention and tempered class weighting in Experiment 8.
>
> We then tested learning rate schedules, heavy augmentations, higher spatial resolutions, and modern architectures like ConvNeXt-Tiny.
>
> In Experiment 13, we introduced targeted 1.20x melanoma class weighting, which boosted melanoma sensitivity from 48.11% up to 59.43%."

---

### [1:10 - 1:30] Final 85/15 Probability Ensemble
> "To capture the best of both worlds, Experiment 14 calibrated a convex probability ensemble:
> **85% Experiment 8 plus 15% Experiment 13**.
>
> The two frozen component models contribute probability distributions that are blended before the final argmax class decision.
>
> On our frozen 988-image test split, this ensemble achieved **80.57% accuracy**, **75.69% balanced accuracy**, and **52.43% melanoma F1**—our strongest overall generalization.
> In Experiment 15, we investigated threshold calibration and proved that the natural argmax rule remains mathematically optimal."

---

### [1:30 - 2:10] Live Website Demonstration
> "Now let's switch to the live DermaAI software demonstrator.
>
> As you can see on screen, DermaAI is built with a Next.js 16 frontend and a containerized FastAPI backend running in-memory inference.
>
> Notice the top banner: this system is strictly an academic research demonstrator.
>
> I'll drag and drop a non-test dermoscopic image into the dropzone and click 'Analyze Lesion Image'.
> Within 400 milliseconds on CPU, the backend executes image validation, parallel model evaluation, convex blending, and dual backward Grad-CAM attribution."

---

### [2:10 - 2:35] Dual Grad-CAM & Probability Distribution
> "Here are our live results:
> The system outputs the predicted class, confidence, and complete 7-class probability distribution, verifying that probabilities sum to 1.0.
>
> Below, we render side-by-side Grad-CAM heatmaps hooking into `model.attention`:
> Notice how Experiment 8 captures global border symmetry, while Experiment 13 concentrates intently on focal atypical pigment networks.
>
> Grad-CAM provides qualitative visualization of regions contributing to each score—it is an interpretability aid, not causal proof."

---

### [2:35 - 2:50] Quantitative Error Analysis
> "True engineering requires understanding failure modes.
> On our benchmark test split, remaining errors reveal clear morphological overlaps:
> - 26 melanomas were classified as nevi due to border regularity.
> - 19 melanomas were classified as benign keratoses due to keratotic surface texture.
> - 38 dysplastic benign nevi were predicted as melanoma.
> We document these confusions openly rather than hiding them."

---

### [2:50 - 3:00] Limitations & Conclusion
> "Finally, we maintain strict scientific transparency: DermaAI has not undergone clinical trials, was evaluated on a retrospective dataset, and makes zero clinical diagnostic claims.
>
> DermaAI demonstrates how controlled experimentation, complementary ensembling, visual explainability, and failure analysis can be integrated into an accountable research demonstrator.
>
> Thank you! We welcome your questions."
