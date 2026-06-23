# image_utils.py — Image validation and foil detection utilities.

import logging
from typing import Optional

import cv2
import numpy as np


def ensure_valid_image(img_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Validate and normalize an input image to 3-channel BGR format."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[-1] == 4:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    if img_bgr.shape[-1] == 3:
        return img_bgr
    return None


def foil_score(img_bgr: np.ndarray) -> float:
    """
    Calculate a 0..1 score indicating how likely an image is to be a foil card.

    Uses bright-spot ratio (specular highlights) and local contrast (Laplacian).
    """
    img = ensure_valid_image(img_bgr)
    if img is None:
        return 0.0
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright_ratio = float((gray > 240).astype(np.uint8).mean())
        contrast = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))) / 255.0
        return float(np.clip(0.7 * bright_ratio + 0.3 * contrast, 0.0, 1.0))
    except Exception as e:
        logging.error(f"Error calculating foil score: {e}")
        return 0.0


def is_probably_foil(img_bgr: np.ndarray, threshold: float = 0.08) -> bool:
    """Return True if the image is likely a foil card (default threshold tuned empirically)."""
    return foil_score(img_bgr) >= threshold
