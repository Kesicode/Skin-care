"""
Pydantic schemas for the DermaAI FastAPI demo backend.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EnsembleWeights(BaseModel):
    experiment8_weight: float = Field(0.85, description="Weight assigned to Experiment 8 probabilities")
    experiment13_weight: float = Field(0.15, description="Weight assigned to Experiment 13 probabilities")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status")
    service: str = Field("DermaAI", description="Service name")
    models_loaded: bool = Field(..., description="Whether both frozen models are loaded")
    ensemble: EnsembleWeights = Field(..., description="Ensemble weighting configuration")
    input_size: str = Field("224x224", description="Expected input image dimensions")
    device: str = Field("cpu", description="Execution hardware device")


class ModelInfoResponse(BaseModel):
    architecture: str = Field("DermaAI_MobileNetV3", description="Neural network architecture")
    backbone: str = Field("MobileNetV3-Large", description="Feature extractor backbone")
    attention: str = Field("CBAM", description="Convolutional Block Attention Module")
    parameters_per_model: int = Field(4326297, description="Number of parameters per model")
    input_size: List[int] = Field([224, 224], description="Input spatial resolution [H, W]")
    num_classes: int = Field(7, description="Number of output lesion classes")
    classes: List[str] = Field(..., description="Ordered list of class codes")
    ensemble: Dict[str, float] = Field(..., description="Component probability weights")
    decision_rule: str = Field("argmax", description="Inference decision rule")
    experiment15_verification: str = Field(
        "Experiment 15 verified that threshold tuning (theta*=0.50) produced zero changed predictions "
        "relative to the natural argmax decision rule.",
        description="Note on decision threshold"
    )


class PredictionDetail(BaseModel):
    class_index: int = Field(..., description="Predicted class index (0-6)")
    class_code: str = Field(..., description="Dataset class mnemonic (e.g. mel, nv, bkl)")
    class_name: str = Field(..., description="Full descriptive name of lesion class")


class TopPrediction(BaseModel):
    class_code: str = Field(..., description="Class code")
    class_name: str = Field(..., description="Full class name")
    probability: float = Field(..., description="Ensemble output probability in [0, 1]")


class GradCAMData(BaseModel):
    available: bool = Field(True, description="Whether Grad-CAM generation succeeded")
    target_class: str = Field(..., description="Class targeted for gradient attribution")
    target_layer: str = Field("model.attention", description="Target convolutional attention layer")
    experiment8: Optional[str] = Field(None, description="Base64 encoded PNG overlay for Exp 8")
    experiment13: Optional[str] = Field(None, description="Base64 encoded PNG overlay for Exp 13")
    notice: str = Field(
        "Grad-CAM highlights image regions that contributed strongly to the selected class score in each component model. "
        "Grad-CAM is an interpretability aid, not proof of clinical reasoning or causality.",
        description="Interpretability notice"
    )


class PredictionResponse(BaseModel):
    success: bool = Field(True, description="Request success status")
    prediction: PredictionDetail = Field(..., description="Predicted lesion class")
    confidence: float = Field(..., description="Model ensemble confidence in [0, 1]")
    probabilities: Dict[str, float] = Field(..., description="Full 7-class probability distribution")
    top3: List[TopPrediction] = Field(..., description="Top 3 ranked predictions by probability")
    ensemble: EnsembleWeights = Field(..., description="Ensemble weights used")
    gradcam: GradCAMData = Field(..., description="Component Grad-CAM visualizations")
    research_notice: str = Field(..., description="Mandatory academic research disclaimer")
    experiment15_note: str = Field(
        "Experiment 15 verified that the selected threshold did not change any predictions relative to the natural argmax decision rule.",
        description="Experiment 15 verification disclosure"
    )


class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Request failure indicator")
    error: str = Field(..., description="Sanitized, human-readable error description")
    error_code: str = Field(..., description="Machine-readable error category")
