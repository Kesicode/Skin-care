"""
Singleton Model Service for DermaAI.
Loads, verifies, and manages the frozen Experiment 8 and Experiment 13 models.
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Tuple

from .config import (
    EXP8_CHECKPOINT_PATH,
    EXP13_CHECKPOINT_PATH,
    EXPECTED_PARAMS_COUNT,
    EXPECTED_EXP8_SHA256,
    EXPECTED_EXP13_SHA256,
    EXPECTED_EXP8_SIZE,
    EXPECTED_EXP13_SIZE,
    EXP8_WEIGHT,
    EXP13_WEIGHT,
    DEVICE,
    PROJECT_ROOT
)
from .utils import compute_sha256

# Ensure ai directory is importable
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ai.src.model import DermaAI_MobileNetV3
except ImportError:
    # Alternative direct import if inside ai directory
    from model import DermaAI_MobileNetV3


class ModelService:
    """
    Manages frozen Experiment 8 and Experiment 13 PyTorch models.
    Ensures single loading into memory, parameter freezing, and checksum verification.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = torch.device(DEVICE)
        self.model8: torch.nn.Module = None
        self.model13: torch.nn.Module = None
        self.loaded = False
        self._load_and_verify_models()
        self._initialized = True

    def _verify_file_integrity(self, path: str, expected_size: int, expected_sha256: str, name: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[DermaAI] Fatal: {name} checkpoint not found at: {path}")

        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise ValueError(
                f"[DermaAI] Fatal: {name} file size mismatch. Expected {expected_size} bytes, got {actual_size} bytes."
            )

        actual_sha256 = compute_sha256(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"[DermaAI] Fatal: {name} SHA-256 integrity check failed!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_sha256}"
            )

    def _load_and_verify_models(self):
        print("=" * 65)
        print("[DermaAI] Initializing Frozen Model Service...")
        print("=" * 65)

        # 1. Verify Checkpoint Files
        self._verify_file_integrity(EXP8_CHECKPOINT_PATH, EXPECTED_EXP8_SIZE, EXPECTED_EXP8_SHA256, "Experiment 8")
        self._verify_file_integrity(EXP13_CHECKPOINT_PATH, EXPECTED_EXP13_SIZE, EXPECTED_EXP13_SHA256, "Experiment 13")

        # 2. Load Model A (Experiment 8)
        print(f"[DermaAI] Loading Experiment 8 from {EXP8_CHECKPOINT_PATH}...")
        self.model8 = DermaAI_MobileNetV3(num_classes=7, use_attention=True)
        state_dict_8 = torch.load(EXP8_CHECKPOINT_PATH, map_location=self.device)
        self.model8.load_state_dict(state_dict_8)
        self.model8.to(self.device).eval()
        for p in self.model8.parameters():
            p.requires_grad = False

        params8 = sum(p.numel() for p in self.model8.parameters())
        if params8 != EXPECTED_PARAMS_COUNT:
            raise ValueError(f"[DermaAI] Exp8 parameter count mismatch: {params8} vs expected {EXPECTED_PARAMS_COUNT}")
        if not hasattr(self.model8, "attention"):
            raise AttributeError("[DermaAI] Exp8 is missing verified 'attention' (CBAM) module!")
        print(f"[DermaAI] Exp8 loaded successfully (Params: {params8:,})")

        # 3. Load Model B (Experiment 13)
        print(f"[DermaAI] Loading Experiment 13 from {EXP13_CHECKPOINT_PATH}...")
        self.model13 = DermaAI_MobileNetV3(num_classes=7, use_attention=True)
        state_dict_13 = torch.load(EXP13_CHECKPOINT_PATH, map_location=self.device)
        self.model13.load_state_dict(state_dict_13)
        self.model13.to(self.device).eval()
        for p in self.model13.parameters():
            p.requires_grad = False

        params13 = sum(p.numel() for p in self.model13.parameters())
        if params13 != EXPECTED_PARAMS_COUNT:
            raise ValueError(f"[DermaAI] Exp13 parameter count mismatch: {params13} vs expected {EXPECTED_PARAMS_COUNT}")
        if not hasattr(self.model13, "attention"):
            raise AttributeError("[DermaAI] Exp13 is missing verified 'attention' (CBAM) module!")
        print(f"[DermaAI] Exp13 loaded successfully (Params: {params13:,})")

        # 4. Sanity Check on dummy tensor
        with torch.inference_mode():
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            out8 = self.model8(dummy)
            out13 = self.model13(dummy)

            if out8.shape != (1, 7) or out13.shape != (1, 7):
                raise ValueError(f"[DermaAI] Output shape mismatch: out8={out8.shape}, out13={out13.shape}")

            p8 = F.softmax(out8, dim=-1)
            p13 = F.softmax(out13, dim=-1)
            p_ens = EXP8_WEIGHT * p8 + EXP13_WEIGHT * p13

            if not torch.isclose(torch.sum(p_ens), torch.tensor(1.0, device=self.device), atol=1e-4):
                raise ValueError("[DermaAI] Ensemble probabilities do not sum to 1.0 within tolerance!")

        print(f"[DermaAI] Ensemble = {EXP8_WEIGHT:.2f} (Exp8) / {EXP13_WEIGHT:.2f} (Exp13)")
        print(f"[DermaAI] Classes = 7 (akiec, bcc, bkl, df, mel, nv, vasc)")
        print(f"[DermaAI] Input = 224x224")
        print(f"[DermaAI] Device = {self.device.type.upper()}")
        print(f"[DermaAI] Ready for inference.")
        print("=" * 65)

        self.loaded = True

    def get_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if not self.loaded:
            raise RuntimeError("[DermaAI] Models have not been successfully loaded.")
        return self.model8, self.model13


# Global singleton instance
model_service = ModelService()
