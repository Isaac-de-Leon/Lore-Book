# features.py — MobileNetV2 feature extraction and activation heatmap overlay.
#
# Extraction is pluggable behind get_extractor(backend):
#   - "keras"  : MobileNetV2 via TensorFlow/Keras (desktop default, also used to
#                build the .tflite model).
#   - "tflite" : the same MobileNetV2 features via a .tflite Interpreter
#                (tflite_runtime preferred, falls back to tensorflow.lite).
#                Lightweight enough to run on a Raspberry Pi for the sorter.
# Both backends share identical preprocessing, so the 1280-dim vectors they
# produce are interchangeable and cache-compatible (see scripts/check_parity.py).

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

from lorebook.core.image_utils import ensure_valid_image
from lorebook.core.matching import _l2_normalize

_base_model: Optional[Model] = None
_feat_model: Optional[Model] = None   # 1280-dim pooled features
_act_model: Optional[Model] = None    # last conv activations (for heatmaps)

# Where the tflite feature model lives. Override with LOREBOOK_TFLITE_MODEL.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TFLITE_MODEL = os.environ.get(
    "LOREBOOK_TFLITE_MODEL", os.path.join(_REPO_ROOT, "mobilenetv2_features.tflite")
)


def _get_models() -> Tuple[Model, Model]:
    """Return (feature_model, activation_model), lazy-initialized on first call."""
    global _base_model, _feat_model, _act_model
    if _feat_model is None or _act_model is None:
        _base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        _feat_model = Model(inputs=_base_model.input, outputs=_base_model.output)
        _act_model = Model(inputs=_base_model.input, outputs=_base_model.layers[-2].output)
    return _feat_model, _act_model


def _preprocess_to_batch(img_or_path: Union[np.ndarray, str]) -> Optional[np.ndarray]:
    """
    Load/validate an image and return a (1, 224, 224, 3) preprocessed float32 batch.

    Shared by every backend so their feature vectors stay interchangeable.
    Returns None if the image cannot be read or validated.
    """
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
    return preprocess_input(batch)


def _finalize(features: np.ndarray) -> Optional[np.ndarray]:
    """Flatten, validate the 1280-dim shape, and L2-normalize a raw feature vector."""
    features = np.asarray(features).flatten().astype(np.float32)
    if features.size != 1280:
        logging.error(f"Unexpected feature dimension: {features.size}")
        return None
    return _l2_normalize(features)


class _KerasExtractor:
    """MobileNetV2 feature extractor backed by TensorFlow/Keras."""

    def extract(self, img_or_path: Union[np.ndarray, str]) -> Optional[np.ndarray]:
        try:
            batch = _preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            feat_model, _ = _get_models()
            features = feat_model.predict(batch, verbose=0)
            return _finalize(features)
        except Exception as e:
            logging.error(f"Error extracting features (keras): {e}")
            return None


class _TFLiteExtractor:
    """MobileNetV2 feature extractor backed by a .tflite Interpreter (lazy-loaded)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_TFLITE_MODEL
        self._interpreter = None
        self._in_index = None
        self._out_index = None
        self._in_dtype = None

    def _ensure_interpreter(self):
        if self._interpreter is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"tflite model not found at {self.model_path}. "
                "Generate it with: python scripts/convert_to_tflite.py"
            )
        try:
            from tflite_runtime.interpreter import Interpreter  # Pi-friendly, no full TF
        except Exception:
            try:
                from tensorflow.lite import Interpreter  # desktop fallback
            except Exception as e:
                raise ImportError(
                    "The tflite backend needs 'tflite-runtime' or 'tensorflow' installed."
                ) from e
        interp = Interpreter(model_path=self.model_path)
        interp.allocate_tensors()
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]
        self._interpreter = interp
        self._in_index = in_detail["index"]
        self._out_index = out_detail["index"]
        self._in_dtype = in_detail["dtype"]

    def extract(self, img_or_path: Union[np.ndarray, str]) -> Optional[np.ndarray]:
        try:
            self._ensure_interpreter()
            batch = _preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            self._interpreter.set_tensor(self._in_index, batch.astype(self._in_dtype))
            self._interpreter.invoke()
            features = self._interpreter.get_tensor(self._out_index)
            return _finalize(features)
        except Exception as e:
            logging.error(f"Error extracting features (tflite): {e}")
            return None


_extractors: dict = {}


def get_extractor(backend: str = "keras"):
    """
    Return a cached feature extractor for the given backend.

    The returned object exposes `.extract(img_or_path) -> Optional[np.ndarray]`,
    yielding a 1280-dim L2-normalized vector. Supported backends: "keras", "tflite".
    """
    key = (backend or "keras").lower()
    if key not in _extractors:
        if key == "keras":
            _extractors[key] = _KerasExtractor()
        elif key == "tflite":
            _extractors[key] = _TFLiteExtractor()
        else:
            raise ValueError(f"Unknown extractor backend: {backend!r} (use 'keras' or 'tflite').")
    return _extractors[key]


def extract_features(img_or_path: Union[np.ndarray, str]) -> Optional[np.ndarray]:
    """
    Extract a normalized 1280-dim feature vector from an image or file path.

    Uses MobileNetV2 (Keras backend) by default. The model is loaded lazily on
    first call. For other backends use get_extractor(backend).extract(...).
    """
    return get_extractor("keras").extract(img_or_path)


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
