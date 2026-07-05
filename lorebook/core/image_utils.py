# image_utils.py — Image validation and foil detection utilities.

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

# Standard TCG card aspect ratio (63mm × 88mm portrait).
CARD_ASPECT = 63 / 88.0


def focus_rect(h: int, w: int, height_frac: float = 0.6) -> Tuple[int, int, int, int]:
    """
    Return (fx, fy, fw, fh) for a centered 63:88 portrait card box sized to
    height_frac of the frame height. Shared by the GUI focus overlay and the
    sorter's pre-match crop so both look at the same region.
    """
    fh = min(int(h * height_frac), h - 4)
    fw = min(int(fh * CARD_ASPECT), w - 4)
    fx = max((w - fw) // 2, 2)
    fy = max((h - fh) // 2, 2)
    return fx, fy, fw, fh


def crop_to_card(img_bgr: Optional[np.ndarray], height_frac: float = 0.6) -> Optional[np.ndarray]:
    """
    Crop a frame to the centered card focus box (see focus_rect).

    Returns the frame unchanged when it is missing or too small to crop
    meaningfully, so callers can apply it unconditionally.
    """
    if img_bgr is None or img_bgr.ndim < 2:
        return img_bgr
    h, w = img_bgr.shape[:2]
    fx, fy, fw, fh = focus_rect(h, w, height_frac)
    fx, fy = max(fx, 0), max(fy, 0)
    fw, fh = min(fw, w - fx), min(fh, h - fy)
    if fw > 10 and fh > 10:
        return img_bgr[fy:fy + fh, fx:fx + fw].copy()
    return img_bgr


class MotionGate:
    """
    Fires once per card placed under the camera.

    Feed one small grayscale frame per tick; update() returns True exactly
    once when the scene has changed (motion: a card being placed) and then
    stayed steady for steady_frames consecutive ticks. It re-arms only after
    motion is seen again, so a card left in place never re-triggers.

    min_std (0 = disabled) suppresses triggers on near-uniform frames, so
    removing a card from a plain background doesn't fire a wasted scan of the
    empty scene; a textured background degrades gracefully to firing anyway.

    Pure numpy — used by the GUI's auto-scan mode and reusable by the sorter
    for card-presence detection later.
    """

    def __init__(self, steady_frames: int = 10, diff_threshold: float = 4.0,
                 min_std: float = 0.0):
        self.steady_frames = max(1, steady_frames)
        self.diff_threshold = diff_threshold
        self.min_std = min_std
        self._prev: Optional[np.ndarray] = None
        self._steady = 0
        self._armed = False  # motion must be seen before a trigger

    def reset(self) -> None:
        self._prev, self._steady, self._armed = None, 0, False

    def update(self, gray: Optional[np.ndarray]) -> bool:
        """Feed the next frame; True exactly when a fresh card has settled."""
        if gray is None or np.size(gray) == 0:
            return False
        gray = np.asarray(gray, dtype=np.float32)
        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray
            return False
        diff = float(np.mean(np.abs(gray - self._prev)))
        self._prev = gray

        if diff >= self.diff_threshold:
            self._armed = True
            self._steady = 0
            return False
        if not self._armed:
            return False
        self._steady += 1
        if self._steady < self.steady_frames:
            return False
        self._armed = False
        self._steady = 0
        return float(np.std(gray)) >= self.min_std


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
