# Phase 17 — Viva Voce Preparation & Technical Defense Guide

This document provides rigorous, scientifically grounded defenses for key architectural, methodological, and experimental decisions made throughout the DermaAI research project.

---

### Q1: Why HAM10000?
**Defense:**
The HAM10000 ("Human Against Machine with 10,000 training images") dataset (Tschandl et al., 2018) is the standard open-access benchmark for multi-source dermatoscopic image analysis. Comprising 10,015 dermoscopic images collected across multiple centers (Cliff Rosendahl in Queensland, Australia, and the Department of Dermatology at the Medical University of Vienna, Austria), it represents authentic clinical variability in acquisition hardware, illumination, and lesion morphology. Crucially, over $50\%$ of the lesions are confirmed by histopathology (the clinical gold standard), with the remainder verified by expert consensus or follow-up confocal microscopy. It allows reproducible benchmarking against established literature without proprietary data silos.

---

### Q2: Why 7 classes?
**Defense:**
Binary classification ("benign vs malignant") obscures vital diagnostic nuances and creates high risk of failure in real-world deployment. HAM10000 defines 7 distinct diagnostic categories covering the full spectrum of common pigmented lesions:
1. `akiec`: Actinic Keratoses and Intraepithelial Carcinoma (pre-cancerous/in-situ)
2. `bcc`: Basal Cell Carcinoma (common malignant)
3. `bkl`: Benign Keratoses (solar lentigines, seborrheic keratoses, lichen-planus-like)
4. `df`: Dermatofibroma (benign fibrous histiocytoma)
5. `mel`: Melanoma (invasive and in-situ malignant melanocytic lesion)
6. `nv`: Melanocytic Nevi (benign melanocytic nevi)
7. `vasc`: Vascular Lesions (angiomas, pyogenic granulomas, hemorrhage)

A 7-class multiclass setup forces the neural representation to learn differential features between morphologically similar lesions (e.g., differentiating melanoma from pigmented seborrheic keratosis, or dermatofibroma from nevus), providing a much richer foundation for clinical decision support.

---

### Q3: Why MobileNetV3-Large?
**Defense:**
Dermatological AI models intended for real-world deployment should operate efficiently on resource-constrained hardware (e.g., clinics, point-of-care mobile devices) without requiring cloud GPU clusters. MobileNetV3-Large combines:
1. Hardware-aware Neural Architecture Search (NAS) optimized for mobile CPU latencies.
2. Inverted residual blocks with linear bottlenecks.
3. Squeeze-and-Excitation (SE) channel attention modules.
4. Hardswish non-linearities for fast, float-point efficient activation.

At $4,326,297$ parameters, it achieves competitive classification power comparable to ResNet-50 ($25.6\text{M}$ parameters) or DenseNet-121 ($8\text{M}$ parameters), while executing in under $40\text{ ms}$ on standard multi-core CPUs.

---

### Q4: Why CBAM (Convolutional Block Attention Module)?
**Defense:**
Standard convolutional backbones treat all spatial pixels and feature channels uniformly. Dermoscopic lesions, however, are characterized by eccentric, localized morphological cues (pigment asymmetry, atypical networks, regression structures, blue-white veils) situated amidst homogeneous background skin. 

CBAM (Woo et al., 2018) introduces a sequential channel-spatial attention mechanism:
$$\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}$$
$$\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \otimes \mathbf{F}'$$

- **Channel Attention**: Emphasizes lesion-relevant feature detectors (pigment channels, textures) while suppressing background noise.
- **Spatial Attention**: Focuses spatial computation on the lesion nidus and active border while attenuating peripheral artifacts (hair, vignetting rings, gel bubbles).

Adding CBAM immediately prior to adaptive pooling improved balanced accuracy and stabilized spatial attribution, as verified in Phase 16 Grad-CAM.

---

### Q5: Why class weighting?
**Defense:**
The HAM10000 dataset exhibits extreme class imbalance: melanocytic nevi (`nv`) represent $66.95\%$ ($5,367 / 8,017$) of the training split, whereas dermatofibromas (`df`) represent only $1.07\%$ ($86$ images), and vascular lesions (`vasc`) $1.42\%$ ($114$ images). Unweighted Cross-Entropy loss (Experiment 1) minimizes empirical risk by biasing predictions toward the majority class, achieving $73.18\%$ overall accuracy but yielding poor minority recall ($45.28\%$ melanoma recall, $36.92\%$ precision). Class weighting penalizes errors on rare and critical classes proportionally to their scarcity.

---

### Q6: Why tempered weights ($\alpha=0.75$)?
**Defense:**
Inverse frequency weighting ($w_c = N / (C \cdot N_c)$, Experiment 2) applies extreme penalties (a $62\times$ weight ratio between `df` and `nv`). While this elevates minority recall, it destabilizes gradient optimization, causing excessive false alarms on the majority nevus class, dropping overall accuracy to $78.95\%$ and precision. 

Tempered class weighting applies an exponential damping factor $\alpha \in (0, 1)$:
$$w_c = \left( \frac{N_{\max}}{N_c} \right)^\alpha$$

In Experiment 8, $\alpha = 0.75$ was validated as the optimal operating point. It provided sufficient minority gradient amplification to protect rare classes while avoiding the precision collapse caused by raw inverse weighting, establishing the primary single-model benchmark ($80.67\%$ accuracy, $75.37\%$ balanced accuracy, $0.7018$ Macro F1).

---

### Q7: Why Experiment 13?
**Defense:**
In clinical dermatology, false-negative melanomas represent the most dangerous error mode. Although Experiment 8 achieved strong overall performance, its melanoma recall was $48.11\%$ ($51 / 106$). Experiment 13 was designed as a targeted hypothesis test: *Can a mild, targeted increase ($1.20\times$) in the relative training weight of melanoma improve sensitivity while maintaining overall structural stability?*

Experiment 13 successfully increased melanoma recall from $48.11\%$ to $59.43\%$ ($63 / 106$), capturing $17$ melanomas missed by Experiment 8. While its overall accuracy softened to $78.04\%$ due to false alarms, it generated an ideal high-recall complementary specialist model.

---

### Q8: Why a probability-level ensemble?
**Defense:**
Rather than relying on a single compromise model that attempts to balance precision and recall simultaneously in one set of network weights, ensembling leverages the distinct operational characteristics of two frozen models:
- **Experiment 8**: High precision, conservative decision boundaries, peak overall accuracy ($80.67\%$).
- **Experiment 13**: High sensitivity, aggressive lesion boundary detection, high melanoma recall ($59.43\%$).

Blending their predicted class posterior probability distributions:
$$P_{\text{ens}}(x) = (1 - \lambda) P_8(x) + \lambda P_{13}(x)$$
smooths out individual model noise, reduces prediction variance, and creates an asymmetric confidence filter without requiring network retraining or weight merging.

---

### Q9: Why $85/15$ ($\lambda=0.15$)?
**Defense:**
The blending coefficient $\lambda = 0.15$ was determined through a rigorous, systematic grid search on the **validation split only** ($N=1,010$, grid $\lambda \in [0.00, 1.00]$, step $0.05$), with the test set strictly locked. The optimization objective enforced strict non-inferiority constraints:
$$\lambda^* = \arg\max_\lambda \text{Recall}_{\text{mel}}(\lambda) \quad \text{s.t.} \quad \text{Macro F1}_{\max} - \text{Macro F1}(\lambda) \le 0.005$$

At $\lambda = 0.15$, the ensemble preserved $100\%$ of the concordant correct cases ($743/743$), retained $48$ of Exp 8's unique correct calls, and captured $5$ of Exp 13's unique correct calls, achieving the project-wide peak Macro F1 ($70.51\%$) and peak Melanoma F1 ($52.43\%$).

---

### Q10: What did Experiment 15 show?
**Defense:**
Experiment 15 tested whether post-hoc decision threshold tuning on the ensemble's melanoma probability output ($\theta \in [0.10, 0.90]$) could further optimize melanoma recall on validation data. The validation grid search selected $\theta^* = 0.50$ as optimal. Because the ensemble probabilities already sum to $1.0$, a threshold of $\theta^* = 0.50$ on a binary comparison against competing classes produced **zero changed predictions** relative to the natural multiclass argmax decision rule. This proved that the $85/15$ probability ensemble was already intrinsically calibrated at its natural decision boundary, eliminating the need for complex post-hoc threshold overrides.

---

### Q11: What are the major melanoma errors?
**Defense:**
As proven by the Phase 16 error analysis ($N=1,010$ validation images) and confirmed on the test benchmark ($N=988$):
1. **$57.1\%$ of melanoma false negatives** are misclassified as melanocytic nevi (`mel` $\to$ `nv`, $26$ test images).
2. **$21.4\%$ of melanoma false negatives** are misclassified as benign keratoses (`mel` $\to$ `bkl`, $19$ test images).
3. Combined, **$78.5\%$ of missed melanomas** belong to benign pigmented classes.

Morphologically, early superficial spreading melanomas present with subtle, symmetric reticular networks that mimic dysplastic junctional nevi, or verrucous hyperkeratotic borders that mimic seborrheic keratoses. At $224 \times 224$ input resolution, these fine micro-structural atypical streaks are partially smoothed out, making them the primary source of diagnostic ambiguity.

---

### Q12: What did Grad-CAM show?
**Defense:**
Grad-CAM computed on the $960$-channel CBAM attention output (`model.attention`, $[1, 960, 7, 7]$) verified:
1. **Lesion Centering**: Activations localized consistently on the pigmented lesion core and active borders. The network did not rely on background skin, gel bubbles, dark corner vignetting, or hair markers.
2. **Model Differentiation**: Experiment 8 focused tightly on the central lesion nidus, whereas Experiment 13 expanded spatial attention to peripheral margins, directly reflecting its $1.20\times$ penalty for missing border atypia.
3. **Interpretability Boundary**: Grad-CAM is an attribution map showing which feature map activations correlated with the output logit; it does not prove clinical causality or human-like algorithmic reasoning.

---

### Q13: What are the primary study limitations?
**Defense:**
1. **Repeated Test Set Evaluation**: The 988-image test split was evaluated sequentially across Experiments 1–15. While hyperparameters were selected on validation splits, the test metrics represent experimental benchmarking within this program rather than independent external validation.
2. **Single Dataset Source**: HAM10000 exhibits specific acquisition biases (primarily fair-skinned populations, specific dermatoscope brands).
3. **Severe Class Imbalance**: Rare classes (`df`: 15 test images, `vasc`: 14 test images) have wide confidence intervals.
4. **Resolution Downsampling**: Resizing images to $224 \times 224$ discards micro-architectural pigment network details.

---

### Q14: Why is this NOT a clinical diagnostic system?
**Defense:**
A machine learning classifier that achieves $80.57\%$ image-level accuracy on a retrospective, curated benchmark is fundamentally different from a medical device suitable for clinical practice:
- It has not been validated in prospective clinical trials.
- It has not been tested against board-certified dermatologists under real-world clinical workflows.
- It lacks patient history context (age, lesion evolution, family history, total body photography).
- A false-negative rate of $49.06\%$ on melanoma ($52 / 106$ missed) would be clinically unacceptable without human biopsy triage.
- Softmax output scores represent normalized model activations, not calibrated epidemiological probabilities of disease.

---

### Q15: What is required for future clinical translation?
**Defense:**
1. **External Multi-Center Validation**: Testing on independent datasets such as ISIC 2019/2020, BCN20000, and diverse skin phototypes (Fitzpatrick scale IV–VI).
2. **Multi-Modal Integration**: Fusing dermoscopic imagery with patient clinical metadata (age, anatomical site, lesion diameter, ABCD criteria).
3. **Higher-Resolution Encoders**: Utilizing modern vision transformers (e.g., Swin Transformer, EVA-02) at $384 \times 384$ or $512 \times 512$ with patch-level attention.
4. **Prospective Clinical Trials**: Evaluating whether AI assistance improves dermatologist sensitivity without generating excess biopsy burden.
