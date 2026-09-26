"""
Automated Acceptance Verification Script for Phase 18 — DermaAI Demo Engineering.
Verifies all 13 critical technical requirements and reports PASS/FAIL status.
"""

import io
import os
import sys
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from backend.app.config import (
    EXP8_CHECKPOINT_PATH,
    EXP13_CHECKPOINT_PATH,
    EXPECTED_EXP8_SIZE,
    EXPECTED_EXP13_SIZE,
    EXPECTED_EXP8_SHA256,
    EXPECTED_EXP13_SHA256,
    EXPECTED_PARAMS_COUNT,
    CLASS_CODES,
    EXP8_WEIGHT,
    EXP13_WEIGHT,
    GRADCAM_TARGET_LAYER
)
from backend.app.utils import compute_sha256, pil_to_base64_png
from backend.app.model_service import model_service
from backend.app.preprocessing import preprocess_image
from backend.app.gradcam import GradCAMGenerator, overlay_cam
from backend.app.main import app


def run_acceptance_verification():
    results = {}
    print("=" * 65)
    print("DERMAAI DEMO ACCEPTANCE VERIFICATION (PHASE 18)")
    print("=" * 65)

    # 1. Model Loading
    try:
        assert model_service.loaded, "Model service loaded flag is False"
        assert model_service.model8 is not None, "Model 8 is None"
        assert model_service.model13 is not None, "Model 13 is None"
        results["Model Loading"] = "PASS"
    except Exception as e:
        results["Model Loading"] = f"FAIL ({e})"

    # 2. Exp8 Checkpoint
    try:
        assert os.path.exists(EXP8_CHECKPOINT_PATH), "Exp8 checkpoint not found"
        size8 = os.path.getsize(EXP8_CHECKPOINT_PATH)
        assert size8 == EXPECTED_EXP8_SIZE, f"Exp8 size {size8} != {EXPECTED_EXP8_SIZE}"
        sha8 = compute_sha256(EXP8_CHECKPOINT_PATH)
        assert sha8.lower() == EXPECTED_EXP8_SHA256.lower(), f"Exp8 sha256 mismatch"
        results["Exp8 Checkpoint"] = "PASS"
    except Exception as e:
        results["Exp8 Checkpoint"] = f"FAIL ({e})"

    # 3. Exp13 Checkpoint
    try:
        assert os.path.exists(EXP13_CHECKPOINT_PATH), "Exp13 checkpoint not found"
        size13 = os.path.getsize(EXP13_CHECKPOINT_PATH)
        assert size13 == EXPECTED_EXP13_SIZE, f"Exp13 size {size13} != {EXPECTED_EXP13_SIZE}"
        sha13 = compute_sha256(EXP13_CHECKPOINT_PATH)
        assert sha13.lower() == EXPECTED_EXP13_SHA256.lower(), f"Exp13 sha256 mismatch"
        results["Exp13 Checkpoint"] = "PASS"
    except Exception as e:
        results["Exp13 Checkpoint"] = f"FAIL ({e})"

    # 4. Architecture
    try:
        assert model_service.model8.__class__.__name__ == "DermaAI_MobileNetV3"
        assert model_service.model13.__class__.__name__ == "DermaAI_MobileNetV3"
        assert hasattr(model_service.model8, "attention")
        assert hasattr(model_service.model13, "attention")
        results["Architecture"] = "PASS"
    except Exception as e:
        results["Architecture"] = f"FAIL ({e})"

    # 5. Parameter Count
    try:
        p8 = sum(p.numel() for p in model_service.model8.parameters())
        p13 = sum(p.numel() for p in model_service.model13.parameters())
        assert p8 == EXPECTED_PARAMS_COUNT, f"Exp8 params {p8} != {EXPECTED_PARAMS_COUNT}"
        assert p13 == EXPECTED_PARAMS_COUNT, f"Exp13 params {p13} != {EXPECTED_PARAMS_COUNT}"
        results["Parameter Count"] = "PASS"
    except Exception as e:
        results["Parameter Count"] = f"FAIL ({e})"

    # 6. Class Mapping
    try:
        expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        assert CLASS_CODES == expected_classes, f"Class mapping order mismatch: {CLASS_CODES}"
        results["Class Mapping"] = "PASS"
    except Exception as e:
        results["Class Mapping"] = f"FAIL ({e})"

    # 7. Preprocessing
    # Generate synthetic dermoscopic test image
    synth_arr = np.zeros((256, 256, 3), dtype=np.uint8)
    synth_arr[:, :] = [215, 175, 150]
    y, x = np.ogrid[:256, :256]
    synth_arr[(x - 128)**2 + (y - 128)**2 <= 45**2] = [65, 35, 25]
    test_img = Image.fromarray(synth_arr)

    try:
        tensor = preprocess_image(test_img, model_service.device)
        assert tensor.shape == (1, 3, 224, 224), f"Preprocessed shape {tensor.shape} != (1, 3, 224, 224)"
        assert not torch.isnan(tensor).any(), "Preprocessed tensor contains NaN"
        results["Preprocessing"] = "PASS"
    except Exception as e:
        results["Preprocessing"] = f"FAIL ({e})"

    # 8. Ensemble Formula
    try:
        with torch.inference_mode():
            out8 = model_service.model8(tensor)
            out13 = model_service.model13(tensor)
            p8 = F.softmax(out8, dim=-1)
            p13 = F.softmax(out13, dim=-1)
            p_final = EXP8_WEIGHT * p8 + EXP13_WEIGHT * p13

        expected_blend = 0.85 * p8.cpu().numpy() + 0.15 * p13.cpu().numpy()
        np.testing.assert_allclose(p_final.cpu().numpy(), expected_blend, atol=1e-6)
        assert torch.isclose(torch.sum(p_final), torch.tensor(1.0, device=model_service.device), atol=1e-4)
        results["Ensemble Formula"] = "PASS"
    except Exception as e:
        results["Ensemble Formula"] = f"FAIL ({e})"

    # 9. Argmax Decision
    try:
        pred_idx = int(torch.argmax(p_final, dim=-1).item())
        assert 0 <= pred_idx <= 6, f"Invalid predicted index {pred_idx}"
        results["Argmax Decision"] = "PASS"
    except Exception as e:
        results["Argmax Decision"] = f"FAIL ({e})"

    # 10. Confidence
    try:
        confidence = float(p_final[0, pred_idx].item())
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"
        results["Confidence"] = "PASS"
    except Exception as e:
        results["Confidence"] = f"FAIL ({e})"

    # 11. Grad-CAM Exp8
    try:
        gen8 = GradCAMGenerator(model=model_service.model8, target_layer=model_service.model8.attention)
        cam8 = gen8.compute_cam(tensor, target_class_idx=pred_idx)
        assert cam8.shape == (7, 7), f"CAM shape {cam8.shape} != (7, 7)"
        assert np.isfinite(cam8).all(), "CAM8 contains NaN or Inf"
        assert 0.0 <= cam8.min() and cam8.max() <= 1.0, "CAM8 not in [0, 1]"
        _, b64_8 = overlay_cam(test_img, cam8)
        assert b64_8.startswith("data:image/png;base64,"), "CAM8 base64 prefix missing"
        gen8.cleanup()
        results["Grad-CAM Exp8"] = "PASS"
    except Exception as e:
        results["Grad-CAM Exp8"] = f"FAIL ({e})"

    # 12. Grad-CAM Exp13
    try:
        gen13 = GradCAMGenerator(model=model_service.model13, target_layer=model_service.model13.attention)
        cam13 = gen13.compute_cam(tensor, target_class_idx=pred_idx)
        assert cam13.shape == (7, 7), f"CAM shape {cam13.shape} != (7, 7)"
        assert np.isfinite(cam13).all(), "CAM13 contains NaN or Inf"
        assert 0.0 <= cam13.min() and cam13.max() <= 1.0, "CAM13 not in [0, 1]"
        _, b64_13 = overlay_cam(test_img, cam13)
        assert b64_13.startswith("data:image/png;base64,"), "CAM13 base64 prefix missing"
        gen13.cleanup()
        results["Grad-CAM Exp13"] = "PASS"
    except Exception as e:
        results["Grad-CAM Exp13"] = f"FAIL ({e})"

    # 13. API
    try:
        client = TestClient(app)
        h_resp = client.get("/api/health")
        assert h_resp.status_code == 200, f"Health check returned {h_resp.status_code}"

        info_resp = client.get("/api/model-info")
        assert info_resp.status_code == 200, f"Model info returned {info_resp.status_code}"

        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        files = {"image": ("synth_test.png", buf.getvalue(), "image/png")}
        p_resp = client.post("/api/predict", files=files)
        assert p_resp.status_code == 200, f"Predict returned {p_resp.status_code}"
        p_data = p_resp.json()
        assert p_data["success"] is True
        assert p_data["gradcam"]["available"] is True
        results["API"] = "PASS"
    except Exception as e:
        results["API"] = f"FAIL ({e})"

    # Summary Output
    print()
    all_pass = True
    for test_name, status in results.items():
        print(f"    {test_name}: {status}")
        if status != "PASS":
            all_pass = False

    print("=" * 65)
    final_status = "DEMO COMPLETE" if all_pass else "DEMO INCOMPLETE"
    print(f"FINAL STATUS: {final_status}")
    print("=" * 65)

    return all_pass


if __name__ == "__main__":
    success = run_acceptance_verification()
    sys.exit(0 if success else 1)
