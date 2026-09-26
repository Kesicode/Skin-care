"""
Inference pipeline for DermaAI.
Executes the frozen Experiment 14 probability ensemble (0.85 * P8 + 0.15 * P13),
standard argmax decision, top-3 distribution, and dual-model Grad-CAM generation.
"""

from typing import Dict, Any
from PIL import Image
import torch
import torch.nn.functional as F

from .config import (
    EXP8_WEIGHT,
    EXP13_WEIGHT,
    CLASS_CODES,
    CLASS_NAMES,
    RESEARCH_NOTICE,
    DEVICE
)
from .model_service import model_service
from .preprocessing import preprocess_image
from .gradcam import generate_component_gradcam
from .schemas import (
    PredictionResponse,
    PredictionDetail,
    TopPrediction,
    EnsembleWeights,
    GradCAMData
)


def run_dermaai_inference(rgb_image: Image.Image) -> PredictionResponse:
    """
    Executes the full frozen inference and explainability pipeline on an in-memory RGB image.
    """
    model8, model13 = model_service.get_models()
    device = torch.device(DEVICE)

    # 1. Preprocess image
    input_tensor = preprocess_image(rgb_image, device=device)

    # 2. Dual forward passes (inference_mode)
    with torch.inference_mode():
        logits8 = model8(input_tensor)
        logits13 = model13(input_tensor)

        p8 = F.softmax(logits8, dim=-1)
        p13 = F.softmax(logits13, dim=-1)

        # 3. Probability Ensemble
        p_final = EXP8_WEIGHT * p8 + EXP13_WEIGHT * p13

        # 4. Argmax Prediction & Model Confidence
        pred_idx = int(torch.argmax(p_final, dim=-1).item())
        confidence = float(p_final[0, pred_idx].item())

        # 5. Full 7-class probability mapping
        probabilities: Dict[str, float] = {}
        for idx, code in enumerate(CLASS_CODES):
            probabilities[code] = float(p_final[0, idx].item())

        # 6. Top-3 Ranked Predictions
        top3_indices = torch.topk(p_final, k=3, dim=-1).indices[0].tolist()
        top3_list = [
            TopPrediction(
                class_code=CLASS_CODES[idx],
                class_name=CLASS_NAMES[CLASS_CODES[idx]],
                probability=float(p_final[0, idx].item())
            )
            for idx in top3_indices
        ]

    # 7. Generate Grad-CAM for each frozen component targeting the predicted class
    pred_code = CLASS_CODES[pred_idx]
    cam_exp8_b64 = None
    cam_exp13_b64 = None
    cam_available = True

    try:
        cam_exp8_b64 = generate_component_gradcam(
            model=model8,
            input_tensor=input_tensor,
            original_img=rgb_image,
            target_class_idx=pred_idx
        )
        cam_exp13_b64 = generate_component_gradcam(
            model=model13,
            input_tensor=input_tensor,
            original_img=rgb_image,
            target_class_idx=pred_idx
        )
    except Exception as e:
        print(f"[DermaAI Warning] Grad-CAM generation failed: {e}")
        cam_available = False

    # 8. Assemble structured response
    response = PredictionResponse(
        success=True,
        prediction=PredictionDetail(
            class_index=pred_idx,
            class_code=pred_code,
            class_name=CLASS_NAMES[pred_code]
        ),
        confidence=confidence,
        probabilities=probabilities,
        top3=top3_list,
        ensemble=EnsembleWeights(
            experiment8_weight=EXP8_WEIGHT,
            experiment13_weight=EXP13_WEIGHT
        ),
        gradcam=GradCAMData(
            available=cam_available,
            target_class=pred_code,
            target_layer="model.attention",
            experiment8=cam_exp8_b64,
            experiment13=cam_exp13_b64,
            notice=(
                "Grad-CAM highlights image regions that contributed strongly to the selected class score "
                "in each component model. Grad-CAM is an interpretability aid, not proof of clinical reasoning or causality."
            )
        ),
        research_notice=RESEARCH_NOTICE,
        experiment15_note=(
            "Experiment 15 verified that the selected threshold did not change any predictions "
            "relative to the natural argmax decision rule."
        )
    )

    return response
