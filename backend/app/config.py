"""
DermaAI Demo Backend Configuration
Defines frozen model parameters, ensemble weights, class mappings, and runtime settings.
"""

import os
from typing import Dict, List

# Base directories
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
AI_DIR = os.path.join(PROJECT_ROOT, "ai")
MODELS_DIR = os.environ.get("DERMAAI_MODELS_DIR", os.path.join(AI_DIR, "models"))

# Frozen Checkpoints
EXP8_CHECKPOINT_PATH = os.environ.get("DERMAAI_EXP8_CHECKPOINT", os.path.join(MODELS_DIR, "experiment8_best_model.pt"))
EXP13_CHECKPOINT_PATH = os.environ.get("DERMAAI_EXP13_CHECKPOINT", os.path.join(MODELS_DIR, "experiment13_best_model.pt"))

# Integrity specifications
EXPECTED_PARAMS_COUNT = 4_326_297
EXPECTED_EXP8_SHA256 = "ccc579cb2087545fa1118b7551015bec17c0682ec04066268d3a46aa5d23be2c"
EXPECTED_EXP13_SHA256 = "9fab1b1218fddc86e060f322506bafa691429ef2620d7ff1c26041dc7b52ef21"
EXPECTED_EXP8_SIZE = 17512600
EXPECTED_EXP13_SIZE = 17512921

# Frozen Ensemble Specification (Experiment 14)
EXP8_WEIGHT = 0.85
EXP13_WEIGHT = 0.15

# Architecture metadata
ARCHITECTURE_NAME = "DermaAI_MobileNetV3"
BACKBONE_NAME = "MobileNetV3-Large"
ATTENTION_MODULE = "CBAM"
GRADCAM_TARGET_LAYER = "model.attention"
GRADCAM_FEATURE_SHAPE = (1, 960, 7, 7)

# Image & Preprocessing Specifications
INPUT_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_UPLOAD_SIZE_MB = int(os.environ.get("DERMAAI_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Hardware Device
DEVICE = os.environ.get("DERMAAI_DEVICE", "cpu")

# Dataset Class Definition (Strict Order: index 0 to 6)
CLASSES: List[Dict[str, str]] = [
    {"index": 0, "code": "akiec", "name": "Actinic Keratosis / Intraepithelial Carcinoma"},
    {"index": 1, "code": "bcc", "name": "Basal Cell Carcinoma"},
    {"index": 2, "code": "bkl", "name": "Benign Keratosis (Solar Lentigo / Seborrheic Keratosis)"},
    {"index": 3, "code": "df", "name": "Dermatofibroma"},
    {"index": 4, "code": "mel", "name": "Melanoma"},
    {"index": 5, "code": "nv", "name": "Melanocytic Nevus"},
    {"index": 6, "code": "vasc", "name": "Vascular Lesion"}
]

CLASS_CODES = [c["code"] for c in CLASSES]
CLASS_NAMES = {c["code"]: c["name"] for c in CLASSES}
CLASS_INDEX_TO_CODE = {c["index"]: c["code"] for c in CLASSES}
CLASS_CODE_TO_INDEX = {c["code"]: c["index"] for c in CLASSES}

# Mandatory Research Notice
RESEARCH_NOTICE = (
    "RESEARCH USE ONLY: This system is an academic research demonstrator for "
    "image-level classification of dermoscopic images on the retrospective HAM10000 dataset. "
    "It is NOT a certified medical diagnostic device, has NOT undergone clinical trial evaluation, "
    "and must NEVER be used as a substitute for professional clinical diagnosis, biopsy, "
    "dermoscopic examination, or physician decision-making in patient care."
)
