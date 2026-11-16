"""Image preprocessing and vector finalizing shared by every feature backend.

All backends go through these, so the 1280-dim vectors they produce are
interchangeable and cache-compatible (see scripts/check_parity.py).
"""

import logging

import cv2
import numpy as np

from lorebook.core.features.extractor import ImageInput
from lorebook.core.image_utils import ensure_valid_image
from lorebook.core.matching import l2_normalize

logger = logging.getLogger(__name__)

# MobileNetV2's input is INPUT_SIZE x INPUT_SIZE pixels and its pooled output has
# FEATURE_DIM values. Every backend, the ONNX model builder and the cache agree on these.
INPUT_SIZE = 224
FEATURE_DIM = 1280


def blank_batch() -> np.ndarray:
    """Return an all-zero (1, INPUT_SIZE, INPUT_SIZE, 3) batch, used to warm a model up."""
    return np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), np.float32)


def mobilenet_preprocess(batch: np.ndarray) -> np.ndarray:
    """
    MobileNetV2 'tf'-mode preprocessing: scale pixels to [-1, 1].

    Pure-numpy equivalent of keras.applications.mobilenet_v2.preprocess_input,
    so the keras and tflite backends share identical preprocessing without the
    tflite path needing TensorFlow installed.
    """
    return batch.astype(np.float32) / 127.5 - 1.0


def preprocess_to_batch(img_or_path: ImageInput) -> np.ndarray | None:
    """
    Load/validate an image and return a (1, 224, 224, 3) preprocessed float32 batch.

    Shared by every backend so their feature vectors stay interchangeable.
    Returns None if the image cannot be read or validated.
    """
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Could not read image file: %s", img_or_path)
            return None
    else:
        img = img_or_path

    valid = ensure_valid_image(img)
    if valid is None:
        return None

    img_rgb = cv2.cvtColor(valid, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    return mobilenet_preprocess(batch)


def finalize_vector(features: np.ndarray) -> np.ndarray | None:
    """Flatten, validate the 1280-dim shape, and L2-normalize a raw feature vector."""
    features = np.asarray(features).flatten().astype(np.float32)
    if features.size != FEATURE_DIM:
        logger.error("Unexpected feature dimension: %s", features.size)
        return None
    return l2_normalize(features)
