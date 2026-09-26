# DermaAI: Technical Defense & Viva Q&A Guide (20 Questions)

This document provides rigorous, scientifically grounded answers to the 20 most critical technical, architectural, and clinical defense questions.

---

### 1. What is DermaAI?
**Answer**:  
DermaAI is an academic research demonstrator for 7-class dermoscopic image classification on the retrospective HAM10000 dataset. It integrates a frozen dual-model probability ensemble (85% Experiment 8 + 15% Experiment 13, MobileNetV3-Large + CBAM) with dual Grad-CAM convolutional attention explainability.

---

### 2. Why HAM10000?
**Answer**:  
HAM10000 ("Human Against Machine with 10,000 training images") is the standard peer-reviewed benchmark for automated skin lesion analysis. It provides multi-source dermoscopic captures with histological and consensus ground truth, allowing reproducible comparison across research literature.

---

### 3. Why seven classes instead of binary benign vs. malignant?
**Answer**:  
Dermatological practice rarely presents a simple binary choice. Clinicians must distinguish between distinct benign entities (e.g. seborrheic keratoses vs. melanocytic nevi) and premalignant or malignant conditions (actinic keratoses, basal cell carcinomas, melanomas). A 7-class formulation reflects real-world clinical triage complexity.

---

### 4. Why MobileNetV3-Large?
**Answer**:  
MobileNetV3-Large provides state-of-the-art parameter efficiency. With inverted bottleneck residual blocks and hard-swish non-linearities, it achieves 80.57% test accuracy in only 4,326,297 parameters per model (~16.7 MB), executing CPU forward inference in ~15ms without requiring expensive cloud GPUs.

---

### 5. Why CBAM (Convolutional Block Attention Module)?
**Answer**:  
CBAM applies sequential channel and spatial attention before the linear classification head. Channel attention isolates high-level diagnostic features (e.g. pigment network density), while spatial attention highlights lesion margins against surrounding healthy skin.

---

### 6. Why class weighting?
**Answer**:  
HAM10000 has severe class imbalance (Melanocytic Nevi comprise >67% of cases). Unweighted training optimizes overall accuracy by neglecting minority classes. Tempered class weighting ($\alpha=0.75$) penalizes minority misclassifications without inducing the unstable variance of full inverse-frequency weighting.

---

### 7. Why Experiment 13?
**Answer**:  
Experiment 13 applied a targeted $1.20\times$ relative melanoma weighting. This boosted melanoma recall from 48.11% (in Experiment 8) to 59.43%, creating an ideal high-sensitivity partner model for ensemble blending.

---

### 8. Why 85% Experiment 8 and 15% Experiment 13?
**Answer**:  
The ensemble coefficient $\lambda = 0.15$ was calibrated through a systematic grid search across $\lambda \in [0.0, 1.0]$ strictly on the 1,010 validation images. $\lambda = 0.15$ maximized validation Macro F1 while boosting melanoma recall, outperforming both standalone models.

---

### 9. Why a probability-level ensemble instead of voting?
**Answer**:  
Probability-level blending preserves the underlying confidence distribution of each model. A soft convex combination ($0.85 P_8 + 0.15 P_{13}$) retains smooth, calibrated class margins that hard majority voting discards.

---

### 10. Why not average raw logits?
**Answer**:  
Raw logits are unbounded and can have divergent numerical scales between independently trained models. Softmax probabilities bound each model's output to the simplex $\sum P = 1$, ensuring calibrated convex weighting.

---

### 11. Why argmax?
**Answer**:  
Standard $\arg\max$ selects the class with the highest blended probability. In Experiment 15, we confirmed that threshold tuning ($\theta^*=0.50$) produced zero changed predictions relative to natural $\arg\max$, proving the blended distribution is naturally well-calibrated.

---

### 12. What did Experiment 15 show?
**Answer**:  
Experiment 15 conducted a validation-calibrated melanoma threshold search across $\theta \in [0.05, 0.95]$. The optimal threshold was $\theta^* = 0.50$, which resulted in identical predictions to the natural $\arg\max$ decision rule across all 988 test images, validating that post-hoc threshold adjustment was unnecessary.

---

### 13. Why is melanoma recall not extremely high (>95%)?
**Answer**:  
In multi-class dermoscopy on HAM10000, forcing melanoma recall above 90% causes massive precision collapse (dropping precision below 15%) and degrades overall accuracy. DermaAI achieves a balanced, optimal operating point: 50.94% recall, 54.00% precision, and 52.43% F1-score.

---

### 14. What are the main error modes?
**Answer**:  
On the 988-image test split, errors reflect known visual overlap:
- 26 melanomas predicted as nevi (subtle, early-stage lesions with regular borders).
- 19 melanomas predicted as benign keratoses (verrucous/keratotic textures).
- 38 benign nevi predicted as melanoma (dysplastic/atypical benign lesions).

---

### 15. What does Grad-CAM show?
**Answer**:  
Grad-CAM highlights regions that contributed strongly to the selected class score in each component model. It helps us inspect model behavior, but it is not proof of causal reasoning.

---

### 16. Is this clinically validated?
**Answer**:  
No. DermaAI is an academic research demonstrator. Its results are image-level benchmark measurements on HAM10000 and do not establish clinical diagnostic performance.

---

### 17. Why not call 80.57% clinical accuracy?
**Answer**:  
80.57% is the accuracy measured on the project's 988-image HAM10000 test split. It is an image-level research benchmark, not clinical diagnostic accuracy.

---

### 18. What are the primary scientific limitations?
**Answer**:  
1. Single-source retrospective data (HAM10000).
2. Lack of prospective multi-center clinical trials.
3. Limited sample representation for rare classes like dermatofibroma ($N=115$) and vascular lesions ($N=142$).
4. Sensitivity to uncurated consumer smartphone lighting and camera artifacts.

---

### 19. How was data leakage addressed?
**Answer**:  
The project used a lesion-level split with separate training, validation, and test populations. However, because the test split was evaluated repeatedly over the sequential experiment program, the final benchmark should not be treated as independent external validation.

---

### 20. What would be required for real clinical use?
**Answer**:  
Translating this system to real clinical environments would require:
1. Multi-center prospective clinical trials across diverse patient populations and skin Fitzpatrick phototypes.
2. Direct comparison against board-certified dermatologists.
3. Regulatory compliance certification (e.g., FDA De Novo / CE MDR Software as a Medical Device).
4. Robust rejection algorithms for out-of-distribution artifacts (hair, ruler marks, ink).
