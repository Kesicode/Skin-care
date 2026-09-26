"""
DermaAI Smoke Test Suite (Phase 19).
Fast end-to-end smoke verification using synthetic in-memory data.
Zero test-set leakage: only synthetic in-memory images are used.
"""

import sys
import io
import time
from PIL import Image
import numpy as np

# Ensure project root is on sys.path
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import (
    EXPECTED_EXP8_SHA256,
    EXPECTED_EXP13_SHA256,
    EXPECTED_EXP8_SIZE,
    EXPECTED_EXP13_SIZE,
    EXPECTED_PARAMS_COUNT,
    EXP8_WEIGHT,
    EXP13_WEIGHT,
    CLASS_CODES
)


def create_synthetic_image_bytes(format="JPEG", size=(256, 256), color=(180, 130, 110)):
    """Creates an in-memory synthetic image buffer."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def run_smoke_tests():
    print("=" * 70)
    print("DERMAAI PHASE 19: AUTOMATED SMOKE TESTS")
    print("=" * 70)

    client = TestClient(app)
    tests_passed = 0
    total_tests = 7

    # 1. Health Endpoint Test
    print("\n[Test 1/7] GET /api/health ...")
    start = time.time()
    res = client.get("/api/health")
    duration = (time.time() - start) * 1000
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "ok", "Status must be 'ok'"
    assert data["service"] == "DermaAI", "Service must be 'DermaAI'"
    assert data["models_loaded"] is True, "Models must be loaded"
    assert data["ensemble"]["experiment8_weight"] == EXP8_WEIGHT
    assert data["ensemble"]["experiment13_weight"] == EXP13_WEIGHT
    print(f"  --> PASSED ({duration:.1f}ms): Service healthy, models loaded={data['models_loaded']}")
    tests_passed += 1

    # 2. Model Metadata Endpoint Test
    print("\n[Test 2/7] GET /api/models/info ...")
    start = time.time()
    res = client.get("/api/models/info")
    duration = (time.time() - start) * 1000
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    info = res.json()
    assert info["architecture"] == "DermaAI_MobileNetV3"
    assert info["backbone"] == "MobileNetV3-Large"
    assert info["attention"] == "CBAM"
    assert info["parameters_per_model"] == EXPECTED_PARAMS_COUNT
    assert info["classes"] == CLASS_CODES
    assert info["ensemble"]["experiment8"] == EXP8_WEIGHT
    assert info["ensemble"]["experiment13"] == EXP13_WEIGHT
    assert info["decision_rule"] == "argmax"
    print(f"  --> PASSED ({duration:.1f}ms): Architecture verified (DermaAI_MobileNetV3, 4,326,297 params, 7 classes)")
    tests_passed += 1

    # 3. Valid Synthetic Prediction Test (JPEG)
    print("\n[Test 3/7] POST /api/predict (Synthetic JPEG) ...")
    img_bytes = create_synthetic_image_bytes(format="JPEG")
    start = time.time()
    res = client.post(
        "/api/predict",
        files={"image": ("synthetic_lesion.jpg", img_bytes, "image/jpeg")}
    )
    duration = (time.time() - start) * 1000
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    pred = res.json()
    assert pred["success"] is True
    assert pred["prediction"]["class_code"] in CLASS_CODES
    assert 0.0 <= pred["confidence"] <= 1.0
    probs = pred["probabilities"]
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities must sum to 1.0, got {prob_sum}"
    assert len(pred["top3"]) == 3
    assert pred["gradcam"]["available"] is True
    assert pred["gradcam"]["experiment8"].startswith("data:image/png;base64,")
    assert pred["gradcam"]["experiment13"].startswith("data:image/png;base64,")
    assert "RESEARCH USE ONLY" in pred["research_notice"]
    print(f"  --> PASSED ({duration:.1f}ms): Predicted={pred['prediction']['class_code']}, "
          f"Conf={pred['confidence']:.4f}, Grad-CAM generated, ProbSum={prob_sum:.4f}")
    tests_passed += 1

    # 4. Valid Synthetic Prediction Test (PNG)
    print("\n[Test 4/7] POST /api/predict (Synthetic PNG) ...")
    png_bytes = create_synthetic_image_bytes(format="PNG", color=(120, 90, 80))
    start = time.time()
    res = client.post(
        "/api/predict",
        files={"image": ("synthetic_lesion.png", png_bytes, "image/png")}
    )
    duration = (time.time() - start) * 1000
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    pred_png = res.json()
    assert pred_png["success"] is True
    print(f"  --> PASSED ({duration:.1f}ms): PNG handled successfully (Predicted={pred_png['prediction']['class_code']})")
    tests_passed += 1

    # 5. Invalid File Rejection (Text file)
    print("\n[Test 5/7] POST /api/predict (Rejection: invalid MIME text/plain) ...")
    res = client.post(
        "/api/predict",
        files={"image": ("notes.txt", b"This is not a dermoscopic image.", "text/plain")}
    )
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    data = res.json()
    assert data["success"] is False
    print(f"  --> PASSED: Non-image correctly rejected with HTTP 400: '{data['error']}'")
    tests_passed += 1

    # 6. Corrupt Image Rejection (Fake image header)
    print("\n[Test 6/7] POST /api/predict (Rejection: corrupt image bytes) ...")
    res = client.post(
        "/api/predict",
        files={"image": ("corrupted.jpg", b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupted_payload", "image/jpeg")}
    )
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    data = res.json()
    assert data["success"] is False
    print(f"  --> PASSED: Corrupted image bytes rejected with HTTP 400")
    tests_passed += 1

    # 7. Payload Too Large (>10 MB)
    print("\n[Test 7/7] POST /api/predict (Rejection: payload exceeding 10MB) ...")
    oversized_bytes = b"0" * (11 * 1024 * 1024)
    res = client.post(
        "/api/predict",
        files={"image": ("huge_image.jpg", oversized_bytes, "image/jpeg")}
    )
    assert res.status_code == 413, f"Expected 413, got {res.status_code}"
    data = res.json()
    assert data["success"] is False
    print(f"  --> PASSED: Oversized image (>10MB) rejected with HTTP 413: '{data['error']}'")
    tests_passed += 1

    print("\n" + "=" * 70)
    print(f"SMOKE TEST SUMMARY: {tests_passed}/{total_tests} TESTS PASSED (100%)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_tests()
    if not success:
        sys.exit(1)
