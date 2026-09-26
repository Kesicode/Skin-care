"""
Automated Backend Test Suite for DermaAI Demo.
Validates model loading, parameter count, ensemble arithmetic, Grad-CAM, and FastAPI endpoints.
Does NOT use the research test split.
"""

import io
import os
import sys
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi.testclient import TestClient

# Ensure backend and project root are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.config import (
    EXP8_WEIGHT,
    EXP13_WEIGHT,
    EXPECTED_PARAMS_COUNT,
    CLASS_CODES,
    CLASSES
)
from backend.app.model_service import model_service
from backend.app.preprocessing import preprocess_image
from backend.app.gradcam import GradCAMGenerator, overlay_cam
from backend.app.main import app


class TestDermaAIDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.model8, cls.model13 = model_service.get_models()
        cls.device = model_service.device

        # Create a synthetic dermoscopy-like test image in memory (256x256 RGB)
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Background skin tone
        arr[:, :, 0] = 210
        arr[:, :, 1] = 160
        arr[:, :, 2] = 140
        # Dark pigmented central lesion
        y, x = np.ogrid[:256, :256]
        mask = (x - 128) ** 2 + (y - 128) ** 2 <= 50 ** 2
        arr[mask, 0] = 60
        arr[mask, 1] = 40
        arr[mask, 2] = 30
        cls.sample_pil = Image.fromarray(arr)

        buf = io.BytesIO()
        cls.sample_pil.save(buf, format="PNG")
        cls.sample_bytes = buf.getvalue()

    def test_01_models_loaded(self):
        """Verify that both models are loaded and frozen."""
        self.assertTrue(model_service.loaded)
        self.assertIsNotNone(self.model8)
        self.assertIsNotNone(self.model13)

        p8 = sum(p.numel() for p in self.model8.parameters())
        p13 = sum(p.numel() for p in self.model13.parameters())
        self.assertEqual(p8, EXPECTED_PARAMS_COUNT)
        self.assertEqual(p13, EXPECTED_PARAMS_COUNT)

        for p in self.model8.parameters():
            self.assertFalse(p.requires_grad)
        for p in self.model13.parameters():
            self.assertFalse(p.requires_grad)

    def test_02_class_order(self):
        """Verify strict dataset class ordering."""
        expected = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        self.assertEqual(CLASS_CODES, expected)
        for idx, item in enumerate(CLASSES):
            self.assertEqual(item["index"], idx)
            self.assertEqual(item["code"], expected[idx])

    def test_03_preprocessing(self):
        """Verify deterministic preprocessing output shape and normalization."""
        tensor = preprocess_image(self.sample_pil, self.device)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertFalse(torch.isnan(tensor).any())

    def test_04_ensemble_arithmetic(self):
        """Verify that P_final == 0.85 * P8 + 0.15 * P13 exactly."""
        tensor = preprocess_image(self.sample_pil, self.device)
        with torch.inference_mode():
            out8 = self.model8(tensor)
            out13 = self.model13(tensor)
            p8 = F.softmax(out8, dim=-1)
            p13 = F.softmax(out13, dim=-1)
            p_final = EXP8_WEIGHT * p8 + EXP13_WEIGHT * p13

        # Check normalization
        sum_p = torch.sum(p_final).item()
        self.assertAlmostEqual(sum_p, 1.0, places=4)

        # Check explicit probability blending
        expected_blend = 0.85 * p8.cpu().numpy() + 0.15 * p13.cpu().numpy()
        np.testing.assert_allclose(p_final.cpu().numpy(), expected_blend, atol=1e-6)

        # Check argmax decision
        pred_idx = torch.argmax(p_final, dim=-1).item()
        self.assertIn(pred_idx, list(range(7)))

    def test_05_gradcam_exp8_and_exp13(self):
        """Verify Grad-CAM generation on model.attention ([1, 960, 7, 7])."""
        tensor = preprocess_image(self.sample_pil, self.device)

        for name, model in [("Exp8", self.model8), ("Exp13", self.model13)]:
            generator = GradCAMGenerator(model=model, target_layer=model.attention)
            try:
                heatmap = generator.compute_cam(tensor, target_class_idx=4)  # mel
                self.assertEqual(heatmap.shape, (7, 7))
                self.assertTrue(np.isfinite(heatmap).all())
                self.assertGreaterEqual(heatmap.min(), 0.0)
                self.assertLessEqual(heatmap.max(), 1.0)

                overlay, b64_str = overlay_cam(self.sample_pil, heatmap)
                self.assertEqual(overlay.size, self.sample_pil.size)
                self.assertTrue(b64_str.startswith("data:image/png;base64,"))
            finally:
                generator.cleanup()

    def test_06_api_health(self):
        """Verify GET /api/health endpoint."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "DermaAI")
        self.assertTrue(data["models_loaded"])
        self.assertEqual(data["ensemble"]["experiment8_weight"], 0.85)
        self.assertEqual(data["ensemble"]["experiment13_weight"], 0.15)
        self.assertEqual(data["input_size"], "224x224")

    def test_07_api_model_info(self):
        """Verify GET /api/model-info endpoint."""
        resp = self.client.get("/api/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["architecture"], "DermaAI_MobileNetV3")
        self.assertEqual(data["backbone"], "MobileNetV3-Large")
        self.assertEqual(data["attention"], "CBAM")
        self.assertEqual(data["parameters_per_model"], 4326297)
        self.assertEqual(data["num_classes"], 7)
        self.assertEqual(data["decision_rule"], "argmax")

    def test_08_api_predict_success(self):
        """Verify POST /api/predict with valid image."""
        files = {"image": ("test_lesion.png", self.sample_bytes, "image/png")}
        resp = self.client.post("/api/predict", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["success"])
        self.assertIn("class_code", data["prediction"])
        self.assertIn("class_name", data["prediction"])
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertEqual(len(data["probabilities"]), 7)
        self.assertEqual(len(data["top3"]), 3)
        self.assertTrue(data["gradcam"]["available"])
        self.assertTrue(data["gradcam"]["experiment8"].startswith("data:image/png;base64,"))
        self.assertTrue(data["gradcam"]["experiment13"].startswith("data:image/png;base64,"))
        self.assertIn("RESEARCH USE ONLY", data["research_notice"])

    def test_09_api_predict_invalid_image(self):
        """Verify POST /api/predict rejects non-image payload."""
        files = {"image": ("fake.txt", b"not an image", "image/png")}
        resp = self.client.post("/api/predict", files=files)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main()
