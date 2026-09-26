"""
Grad-CAM Implementation for DermaAI Frozen Models.
Targets the verified CBAM attention layer (model.attention) [1, 960, 7, 7].
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from typing import Tuple
from .utils import pil_to_base64_png


class GradCAMGenerator:
    """
    Computes class activation maps for a target convolutional layer using hooks.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.f_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.b_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute_cam(self, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        """
        Computes the normalized 2D Grad-CAM heatmap for target_class_idx.
        input_tensor: [1, 3, 224, 224]
        Returns: [7, 7] numpy float array in [0, 1].
        """
        # Enable grad temporarily for Grad-CAM
        with torch.enable_grad():
            x = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            logits = self.model(x)
            
            # Runtime validation of feature map shape
            if self.activations is None or self.activations.shape[1] != 960:
                raise RuntimeError(
                    f"Unexpected activation shape: {self.activations.shape if self.activations is not None else 'None'}. "
                    "Expected [1, 960, 7, 7] on model.attention."
                )

            score = logits[0, target_class_idx]
            score.backward(retain_graph=False)

            if self.gradients is None:
                raise RuntimeError("Failed to compute gradients on target layer.")

            # Channel-wise global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, 960, 1, 1]
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, 7, 7]
            cam = F.relu(cam)

            cam_np = cam.squeeze().cpu().detach().numpy()

            # Min-Max Normalization to [0, 1]
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max - c_min > 1e-8:
                cam_norm = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            assert np.isfinite(cam_norm).all(), "Grad-CAM output contains NaN or Inf values!"
            return cam_norm

    def cleanup(self):
        """Removes registered PyTorch hooks to avoid leaks."""
        self.f_hook.remove()
        self.b_hook.remove()


def overlay_cam(
    original_img: Image.Image,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[Image.Image, str]:
    """
    Overlays a [7, 7] float heatmap onto original_img.
    Returns: (PIL.Image of overlay, base64 PNG data URL string).
    """
    img_rgb = original_img.convert("RGB")
    w, h = img_rgb.size
    img_np = np.array(img_rgb)

    # Bilinear upsample heatmap to image spatial dimensions
    resized_cam = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255.0 * np.clip(resized_cam, 0.0, 1.0))

    # Apply colormap
    colored_cam = cv2.applyColorMap(cam_uint8, colormap)
    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

    # Alpha blend overlay
    blended = np.uint8(alpha * colored_cam + (1.0 - alpha) * img_np)
    overlay_pil = Image.fromarray(blended)
    b64_str = pil_to_base64_png(overlay_pil)

    return overlay_pil, b64_str


def generate_component_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_img: Image.Image,
    target_class_idx: int
) -> str:
    """
    Helper function to run Grad-CAM on model.attention and return the base64 overlay.
    """
    if not hasattr(model, "attention"):
        raise AttributeError("Model does not have verified 'attention' module for Grad-CAM.")

    generator = GradCAMGenerator(model=model, target_layer=model.attention)
    try:
        heatmap = generator.compute_cam(input_tensor, target_class_idx)
        _, b64_png = overlay_cam(original_img, heatmap)
        return b64_png
    finally:
        generator.cleanup()
