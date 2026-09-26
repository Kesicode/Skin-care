"""
Utility functions for hashing, image encoding, and validation.
"""

import base64
import hashlib
import io
import os
from PIL import Image


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def pil_to_base64_png(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 PNG data URL string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def validate_image_bytes(image_bytes: bytes, max_bytes: int) -> Image.Image:
    """
    Validates image bytes for size, format, and decodability.
    Converts to RGB in-memory.
    Raises ValueError with descriptive safe message if invalid.
    """
    if not image_bytes:
        raise ValueError("Empty image payload provided.")

    if len(image_bytes) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise ValueError(f"Uploaded file exceeds maximum allowed size of {max_mb} MB.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify header and integrity
    except Exception as e:
        raise ValueError(f"Corrupted or unsupported image file: {str(e)}")

    # Re-open after verify() because verify() invalidates the PIL buffer
    try:
        img = Image.open(io.BytesIO(image_bytes))
        rgb_img = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image as RGB: {str(e)}")

    w, h = rgb_img.size
    if w < 16 or h < 16:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for dermoscopic analysis.")
    if w > 8192 or h > 8192:
        raise ValueError(f"Image dimensions ({w}x{h}) exceed reasonable maximum bounds.")

    return rgb_img
