"""MobileNetV2 feature extraction and activation heatmap overlay.

Extraction is pluggable behind get_extractor(backend):
  - "onnx"   : MobileNetV2 on ONNX Runtime (desktop default when onnxruntime
               is installed). The model is built from the Keras weights without
               TensorFlow (core/onnx_model.py) — Windows Smart App Control
               blocks TensorFlow's unsigned DLL, but allows onnxruntime.
  - "keras"  : MobileNetV2 via TensorFlow/Keras (fallback, also used to
               build the .tflite model).
  - "tflite" : the same MobileNetV2 features via a .tflite Interpreter
               (tflite_runtime preferred, falls back to tensorflow.lite).
               Lightweight enough to run on a Raspberry Pi for the sorter.
All backends share identical preprocessing, so the 1280-dim vectors they
produce are interchangeable and cache-compatible (see scripts/check_parity.py).
"""

import os

# Set before anything can import TensorFlow; setdefault respects a user-set level.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from lorebook.core.features.extractor import FeatureExtractor, ImageInput
from lorebook.core.features.heatmap import visualize_activation_overlay
from lorebook.core.features.keras_backend import models_loaded
from lorebook.core.features.registry import (
    BACKENDS,
    Backend,
    default_backend,
    extract_features,
    get_extractor,
)

__all__ = [
    "BACKENDS",
    "Backend",
    "FeatureExtractor",
    "ImageInput",
    "default_backend",
    "extract_features",
    "get_extractor",
    "models_loaded",
    "visualize_activation_overlay",
]
