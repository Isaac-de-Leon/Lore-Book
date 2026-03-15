# features.py — MobileNetV2 feature extraction and activation heatmap overlay.

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)

from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model

from image_utils import ensure_valid_image
from matching import _l2_normalize

_base_model: Optional[Model] = None
_feat_model: Optional[Model] = None   # 1280-dim pooled features
_act_model: Optional[Model] = None    # last conv activations (for heatmaps)


def _get_models() -> Tuple[Model, Model]:
    """Return (feature_model, activation_model), lazy-initialized on first call."""
    global _base_model, _feat_model, _act_model
    if _feat_model is None or _act_model is None:
        _base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        _feat_model = Model(inputs=_base_model.input, outputs=_base_model.output)
        _act_model = Model(inputs=_base_model.input, outputs=_base_model.layers[-2].output)
    return _feat_model, _act_model


def extract_features(img_or_path: Union[np.ndarray, str]) -> Optional[np.ndarray]:
    """
    Extract a normalized 1280-dim feature vector from an image or file path.

    Uses MobileNetV2 pretrained on ImageNet. The model is loaded lazily on first call.
    """
    try:
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
            if img is None:
                logging.error(f"Could not read image file: {img_or_path}")
                return None
        else:
            img = img_or_path

        img = ensure_valid_image(img)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        processed = preprocess_input(batch)

        feat_model, _ = _get_models()
        features = feat_model.predict(processed, verbose=0).flatten().astype(np.float32)

        if features.size != 1280:
            logging.error(f"Unexpected feature dimension: {features.size}")
            return None

        return _l2_normalize(features)

    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None


def visualize_activation_overlay(img_bgr: np.ndarray, model: Optional[Model] = None) -> np.ndarray:
    """Return the image with a jet-colormap heatmap of average last-conv activations blended in."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    _, act_model = _get_models()
    use_model = model or act_model

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    activations = use_model.predict(preprocess_input(batch), verbose=0)[0]  # (7,7,1280)

    heatmap = activations.mean(axis=-1)  # (7,7)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    hmin, hmax = float(np.min(heatmap)), float(np.max(heatmap))
    if hmax > hmin:
        heatmap = (255 * (heatmap - hmin) / (hmax - hmin)).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)

    return cv2.addWeighted(img_bgr, 0.6, cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), 0.4, 0)
