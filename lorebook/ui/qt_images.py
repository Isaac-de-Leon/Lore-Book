# qt_images.py — render OpenCV BGR images onto Qt labels.

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def crop_to_fill(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop img to the target aspect ratio (CSS "cover")."""
    if img_bgr.size == 0 or target_w <= 0 or target_h <= 0:
        return np.ascontiguousarray(img_bgr)
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.ascontiguousarray(img_bgr)
    if (w / float(h)) > (target_w / float(target_h)):
        new_w = int(h * target_w / target_h)
        x0 = max((w - new_w) // 2, 0)
        cropped = img_bgr[:, x0:x0 + new_w]
    else:
        new_h = int(w * target_h / target_w)
        y0 = max((h - new_h) // 2, 0)
        cropped = img_bgr[y0:y0 + new_h, :]
    return np.ascontiguousarray(cropped)


def show_bgr(label: QLabel, img_bgr: np.ndarray | None, fill: bool = False) -> None:
    """Render a BGR (or gray/BGRA) image onto label; fill=True crops to cover it."""
    if img_bgr is None or img_bgr.size == 0:
        return
    if fill:
        img_bgr = crop_to_fill(img_bgr, label.width(), label.height())
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    if img_bgr.dtype != np.uint8 or not img_bgr.flags["C_CONTIGUOUS"]:
        img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    qimg = QImage(img_bgr.data, w, h, img_bgr.strides[0], QImage.Format_BGR888)
    label.setPixmap(
        QPixmap.fromImage(qimg).scaled(
            label.size(),
            Qt.IgnoreAspectRatio if fill else Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    )
