"""
Deterministic preprocessing for DermaAI demo inference.
Matches exactly the evaluation transform from Experiment 8, 13, and 14.
"""

from PIL import Image
import torch
import torchvision.transforms as transforms
from .config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


# Deterministic evaluation transform: Resize to 224x224 -> ToTensor -> Normalize
eval_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Takes a PIL RGB image, applies deterministic evaluation preprocessing,
    and returns a batch tensor of shape [1, 3, 224, 224] on the specified device.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = eval_transform(image)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor
