"""Activation heatmap overlay (the debug-mode view of what the model looks at)."""

from collections.abc import Callable
from typing import Any, cast

import cv2
import numpy as np

from lorebook.core.features import keras_backend
from lorebook.core.features.onnx_backend import OnnxExtractor
from lorebook.core.features.preprocessing import INPUT_SIZE, mobilenet_preprocess
from lorebook.core.features.registry import Backend, default_backend, get_extractor


def visualize_activation_overlay(img_bgr: np.ndarray, model: Any = None, blocking: bool = True) -> np.ndarray:
    """Return the image with a jet-colormap heatmap of average last-conv activations blended in.

    blocking=False is for per-frame callers on the GUI thread: if the model
    isn't loaded yet or is busy (a scan or DB-build batch holds it), the
    image is returned unchanged instead of stalling the preview.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    if model is None and default_backend() == Backend.ONNX:
        onnx = cast(OnnxExtractor, get_extractor(Backend.ONNX))  # the only backend with activations()
        if not blocking and not onnx.loaded():
            return img_bgr
        return _activation_overlay(img_bgr, onnx.activations)
    if not blocking and model is None and not keras_backend.models_loaded():
        return img_bgr
    if not keras_backend.model_lock.acquire(blocking=blocking):
        return img_bgr
    try:
        use_model = model or keras_backend.get_models()[1]
        return _activation_overlay(img_bgr, lambda batch: keras_backend.infer(use_model, batch))
    finally:
        keras_backend.model_lock.release()


def _activation_overlay(
    img_bgr: np.ndarray, activations_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """activations_fn maps a preprocessed (1, 224, 224, 3) batch to (1, 7, 7, 1280)."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    activations = activations_fn(mobilenet_preprocess(batch))[0]  # (7,7,1280)

    heatmap = activations.mean(axis=-1)  # (7,7)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    hmin, hmax = float(np.min(heatmap)), float(np.max(heatmap))
    if hmax > hmin:
        heatmap = (255 * (heatmap - hmin) / (hmax - hmin)).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)

    return cv2.addWeighted(img_bgr, 0.6, cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), 0.4, 0)
