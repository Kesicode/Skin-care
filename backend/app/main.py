"""
FastAPI Application for DermaAI Demo Backend.
Serves health checks, model metadata, and frozen ensemble prediction with Grad-CAM explainability.
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    EXP8_WEIGHT,
    EXP13_WEIGHT,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_MIME_TYPES,
    CLASS_CODES,
    DEVICE
)
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ErrorResponse,
    EnsembleWeights
)
from .model_service import model_service
from .utils import validate_image_bytes
from .inference import run_dermaai_inference

app = FastAPI(
    title="DermaAI Research Demo Backend",
    description="FastAPI service for the frozen HAM10000 7-class ensemble with Grad-CAM explainability.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend or custom deployment origins
allowed_origins_env = os.environ.get("DERMAAI_ALLOWED_ORIGINS", "*")
if allowed_origins_env.strip() == "*":
    origins = ["*"]
else:
    origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Returns runtime service health, device information, and model loading state.
    """
    return HealthResponse(
        status="ok",
        service="DermaAI",
        models_loaded=model_service.loaded,
        ensemble=EnsembleWeights(
            experiment8_weight=EXP8_WEIGHT,
            experiment13_weight=EXP13_WEIGHT
        ),
        input_size="224x224",
        device=DEVICE
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
@app.get("/api/models/info", response_model=ModelInfoResponse, include_in_schema=False)
async def model_info():
    """
    Returns public technical metadata describing the frozen research architecture.
    """
    return ModelInfoResponse(
        architecture="DermaAI_MobileNetV3",
        backbone="MobileNetV3-Large",
        attention="CBAM",
        parameters_per_model=4_326_297,
        input_size=[224, 224],
        num_classes=7,
        classes=CLASS_CODES,
        ensemble={
            "experiment8": EXP8_WEIGHT,
            "experiment13": EXP13_WEIGHT
        },
        decision_rule="argmax",
        experiment15_verification=(
            "Experiment 15 verified that threshold tuning (theta*=0.50) produced zero changed predictions "
            "relative to the natural argmax decision rule."
        )
    )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or corrupted image"},
        413: {"model": ErrorResponse, "description": "Payload exceeds maximum allowed size"},
        500: {"model": ErrorResponse, "description": "Internal processing failure"}
    }
)
async def predict_dermoscopy(image: UploadFile = File(...)):
    """
    Processes an uploaded dermoscopic image and returns the ensemble classification
    along with Grad-CAM heatmaps from both frozen component models.
    """
    # 1. Validate content-type header
    if image.content_type and image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type '{image.content_type}'. Please upload a JPG, JPEG, or PNG dermoscopic image."
        )

    # 2. Read image content into memory
    try:
        content = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image stream: {str(e)}"
        )

    # 3. Validate image size & format in memory
    try:
        rgb_image = validate_image_bytes(content, max_bytes=MAX_UPLOAD_SIZE_BYTES)
    except ValueError as ve:
        err_msg = str(ve)
        if "exceeds maximum allowed size" in err_msg:
            status_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            raise HTTPException(status_code=status_413, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode uploaded image. Ensure file is a valid dermoscopic JPG or PNG."
        )

    # 4. Execute frozen inference pipeline
    try:
        response = run_dermaai_inference(rgb_image)
        return response
    except Exception as e:
        print(f"[DermaAI Server Error] Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during model inference. Please try another valid image."
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "error_code": f"HTTP_{exc.status_code}"}
    )


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=False)
