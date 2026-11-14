# features.py — MobileNetV2 feature extraction and activation heatmap overlay.
#
# Extraction is pluggable behind get_extractor(backend):
#   - "onnx"   : MobileNetV2 on ONNX Runtime (desktop default when onnxruntime
#                is installed). The model is built from the Keras weights without
#                TensorFlow (core/onnx_model.py) — Windows Smart App Control
#                blocks TensorFlow's unsigned DLL, but allows onnxruntime.
#   - "keras"  : MobileNetV2 via TensorFlow/Keras (fallback, also used to
#                build the .tflite model).
#   - "tflite" : the same MobileNetV2 features via a .tflite Interpreter
#                (tflite_runtime preferred, falls back to tensorflow.lite).
#                Lightweight enough to run on a Raspberry Pi for the sorter.
# All backends share identical preprocessing, so the 1280-dim vectors they
# produce are interchangeable and cache-compatible (see scripts/check_parity.py).

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # respect a user-set level

import importlib.util
import logging
import threading
import time
import warnings
from collections.abc import Sequence

import cv2
import numpy as np

from lorebook.core.image_utils import ensure_valid_image
from lorebook.core.matching import l2_normalize

logger = logging.getLogger(__name__)

# TensorFlow/Keras are imported lazily (only when the "keras" backend or the
# heatmap overlay is actually used) so importing this module — and the "tflite"
# backend — works on a Raspberry Pi that has only tflite-runtime installed.
_base_model = None
_feat_model = None   # 1280-dim pooled features
_act_model = None    # last conv activations (for heatmaps)

# Serializes everything that touches the Keras models: the one-time load and
# every inference. The GUI scans while the DB-build worker extracts batches;
# concurrent model construction or inference on a shared model is not
# thread-safe, so both sides queue here. Reentrant because inference paths
# call _get_models() while holding it.
_model_lock = threading.RLock()

# Python locks aren't fair: a build looping batch after batch can re-take
# _model_lock before a waiting scan ever gets it, leaving the scan stuck for
# the whole build. Interactive work (a scan, the debug heatmap) raises this
# count while it waits; batch work yields to it before each batch.
_interactive_waiters = 0
_waiters_guard = threading.Lock()


class _interactive:
    """Context manager: hold _model_lock with priority over batch extraction."""

    def __enter__(self) -> None:
        global _interactive_waiters
        with _waiters_guard:
            _interactive_waiters += 1
        try:
            _model_lock.acquire()
        finally:
            with _waiters_guard:
                _interactive_waiters -= 1

    def __exit__(self, *exc) -> None:
        _model_lock.release()


def _yield_to_interactive(max_wait_s: float = 30.0) -> None:
    """Batch work: let any waiting interactive caller go first."""
    deadline = time.monotonic() + max_wait_s
    while _interactive_waiters and time.monotonic() < deadline:
        time.sleep(0.005)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_tflite_model_path() -> str:
    """
    Where the tflite feature model lives, resolved at call time (not import
    time) so LOREBOOK_TFLITE_MODEL takes effect whenever it is set.
    """
    return os.environ.get(
        "LOREBOOK_TFLITE_MODEL", os.path.join(_REPO_ROOT, "mobilenetv2_features.tflite")
    )


def _mobilenet_preprocess(batch: np.ndarray) -> np.ndarray:
    """
    MobileNetV2 'tf'-mode preprocessing: scale pixels to [-1, 1].

    Pure-numpy equivalent of keras.applications.mobilenet_v2.preprocess_input,
    so the keras and tflite backends share identical preprocessing without the
    tflite path needing TensorFlow installed.
    """
    return batch.astype(np.float32) / 127.5 - 1.0


def _get_models():
    """Return (feature_model, activation_model), lazy-initialized on first call.

    Imports TensorFlow/Keras on first use only (keras backend / heatmap overlay).
    """
    global _base_model, _feat_model, _act_model
    if _feat_model is not None and _act_model is not None:
        return _feat_model, _act_model
    with _model_lock:
        if _feat_model is not None and _act_model is not None:
            return _feat_model, _act_model  # another thread finished loading first
        # Silence TF/Keras import-time UserWarnings here only — never as a
        # process-wide filter at module import.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            import tensorflow as tf
            from keras.applications import MobileNetV2
            from keras.models import Model

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        tf.get_logger().setLevel("ERROR")

        base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        feat = Model(inputs=base.input, outputs=base.output)
        act = Model(inputs=base.input, outputs=base.layers[-2].output)
        # Publish only fully built models (the unlocked fast path reads these).
        _base_model, _act_model, _feat_model = base, act, feat
        return _feat_model, _act_model


def models_loaded() -> bool:
    """True once the Keras models are in memory (no load is triggered)."""
    return _feat_model is not None and _act_model is not None


def _preprocess_to_batch(img_or_path: np.ndarray | str) -> np.ndarray | None:
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
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    return _mobilenet_preprocess(batch)


def _infer(model, batch: np.ndarray) -> np.ndarray:
    """
    Run one small batch through a Keras model and return a NumPy array.

    Calls the model directly instead of predict(): predict() builds a tf.data
    pipeline and traces a tf.function on every call, which is slow for one
    frame and is where a scan crashed natively on a worker thread. Batch
    extraction (DB builds) keeps predict(), whose overhead is paid per batch.
    Callers hold _model_lock.
    """
    return np.asarray(model(batch, training=False))


def _finalize(features: np.ndarray) -> np.ndarray | None:
    """Flatten, validate the 1280-dim shape, and L2-normalize a raw feature vector."""
    features = np.asarray(features).flatten().astype(np.float32)
    if features.size != 1280:
        logger.error("Unexpected feature dimension: %s", features.size)
        return None
    return l2_normalize(features)


class _KerasExtractor:
    """MobileNetV2 feature extractor backed by TensorFlow/Keras."""

    def ensure_ready(self) -> None:
        """Load the model and run one warm-up inference now, raising on
        failure, so callers can fail fast and the first real scan doesn't pay
        TensorFlow's first-call setup."""
        with _model_lock:
            feat_model, _ = _get_models()
            _infer(feat_model, np.zeros((1, 224, 224, 3), np.float32))

    def extract(self, img_or_path: np.ndarray | str) -> np.ndarray | None:
        try:
            batch = _preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            with _interactive():
                feat_model, _ = _get_models()
                features = _infer(feat_model, batch)
            return _finalize(features)
        # Boundary: TF/Keras raise many error types; the contract is "None on
        # failure" so one bad frame never aborts a scan or sort run.
        except Exception as e:  # noqa: BLE001
            logger.error("Error extracting features (keras): %s", e)
            return None

    def extract_batch(self, images: Sequence) -> list[np.ndarray | None]:
        """
        Extract features for many images with a single predict call.

        Returns one vector (or None for unreadable/invalid inputs) per input,
        in order. This is the DB-build fast path: Keras predict overhead is
        paid once per batch instead of once per card, and inference stays on
        one thread (concurrent predicts on a shared model are not guaranteed
        thread-safe).
        """
        results: list[np.ndarray | None] = [None] * len(images)
        rows, positions = [], []
        for i, img in enumerate(images):
            batch = _preprocess_to_batch(img)
            if batch is not None:
                rows.append(batch[0])
                positions.append(i)
        if not rows:
            return results
        try:
            _yield_to_interactive()
            with _model_lock:
                feat_model, _ = _get_models()
                features = feat_model.predict(np.stack(rows), verbose=0)
            for pos, row in zip(positions, features):
                results[pos] = _finalize(row)
        except Exception as e:  # noqa: BLE001 — boundary, see extract()
            logger.error("Error extracting batch features (keras): %s", e)
        return results


def _check_tflite_input_dtype(dtype, model_path: str) -> None:
    """
    Reject non-float32 tflite models.

    The [-1, 1] float batch from _mobilenet_preprocess would be silently
    truncated to garbage by a blind cast to an int8/uint8 quantized input,
    producing vectors that pass the size check but match near-randomly.
    """
    if np.dtype(dtype) != np.float32:
        raise ValueError(
            f"tflite model at {model_path} expects {np.dtype(dtype).name} input; "
            "only float32 models are supported. Re-export without quantization "
            "(scripts/convert_to_tflite.py)."
        )


class _TFLiteExtractor:
    """MobileNetV2 feature extractor backed by a .tflite Interpreter (lazy-loaded)."""

    def __init__(self, model_path: str | None = None):
        self._explicit_model_path = model_path
        self._interpreter = None
        self._in_index = None
        self._out_index = None

    @property
    def model_path(self) -> str:
        return self._explicit_model_path or _default_tflite_model_path()

    def ensure_ready(self) -> None:
        """Load the interpreter now, raising on failure, so callers can fail fast."""
        self._ensure_interpreter()

    def _ensure_interpreter(self):
        """Load the interpreter on first use and return it."""
        if self._interpreter is not None:
            return self._interpreter
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"tflite model not found at {self.model_path}. "
                "Generate it with: python scripts/convert_to_tflite.py"
            )
        try:
            from tflite_runtime.interpreter import Interpreter  # Pi-friendly, no full TF
        except ImportError:
            try:
                from tensorflow.lite import Interpreter  # desktop fallback
            except ImportError as e:
                raise ImportError(
                    "The tflite backend needs 'tflite-runtime' or 'tensorflow' installed."
                ) from e
        interp = Interpreter(model_path=self.model_path)
        interp.allocate_tensors()
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]
        _check_tflite_input_dtype(in_detail["dtype"], self.model_path)
        self._interpreter = interp
        self._in_index = in_detail["index"]
        self._out_index = out_detail["index"]
        return interp

    def extract(self, img_or_path: np.ndarray | str) -> np.ndarray | None:
        try:
            interp = self._ensure_interpreter()
            batch = _preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            interp.set_tensor(self._in_index, batch)
            interp.invoke()
            features = interp.get_tensor(self._out_index)
            return _finalize(features)
        except Exception as e:  # noqa: BLE001 — boundary, see _KerasExtractor.extract()
            logger.error("Error extracting features (tflite): %s", e)
            return None

    def extract_batch(self, images: Sequence) -> list[np.ndarray | None]:
        """API parity with the keras extractor; the tflite interpreter is
        fixed at batch size 1, so this is a plain loop (no speedup)."""
        return [self.extract(img) for img in images]


class _OnnxExtractor:
    """MobileNetV2 feature extractor on ONNX Runtime (model built on first use).

    InferenceSession.run is thread-safe, so inference needs no lock; only the
    one-time model build + session load is serialized.
    """

    def __init__(self, model_path: str | None = None):
        self._explicit_model_path = model_path
        self._session = None
        self._load_lock = threading.Lock()

    @property
    def model_path(self) -> str:
        from lorebook.core.onnx_model import default_model_path
        return self._explicit_model_path or default_model_path()

    def loaded(self) -> bool:
        """True once the session is in memory (no load is triggered)."""
        return self._session is not None

    def ensure_ready(self) -> None:
        """Build/load the model and run one warm-up inference, raising on failure."""
        self._run(np.zeros((1, 224, 224, 3), np.float32))

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        with self._load_lock:
            if self._session is None:
                import onnxruntime as ort

                from lorebook.core.onnx_model import ensure_model

                path = ensure_model(self.model_path)
                opts = ort.SessionOptions()
                opts.log_severity_level = 3  # errors only
                self._session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        return self._session

    def _run(self, batch: np.ndarray, output: str = "features") -> np.ndarray:
        return self._ensure_session().run([output], {"input": batch})[0]

    def activations(self, batch: np.ndarray) -> np.ndarray:
        """(N, 7, 7, 1280) last-conv activations for a preprocessed batch."""
        return self._run(batch, "activations")

    def extract(self, img_or_path: np.ndarray | str) -> np.ndarray | None:
        try:
            batch = _preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            return _finalize(self._run(batch))
        except Exception as e:  # noqa: BLE001 — boundary, see _KerasExtractor.extract()
            logger.error("Error extracting features (onnx): %s", e)
            return None

    def extract_batch(self, images: Sequence) -> list[np.ndarray | None]:
        """One session run per call; same contract as _KerasExtractor.extract_batch."""
        results: list[np.ndarray | None] = [None] * len(images)
        rows, positions = [], []
        for i, img in enumerate(images):
            batch = _preprocess_to_batch(img)
            if batch is not None:
                rows.append(batch[0])
                positions.append(i)
        if not rows:
            return results
        try:
            features = self._run(np.stack(rows))
            for pos, row in zip(positions, features):
                results[pos] = _finalize(row)
        except Exception as e:  # noqa: BLE001 — boundary, see extract()
            logger.error("Error extracting batch features (onnx): %s", e)
        return results


BACKENDS = ("onnx", "keras", "tflite")


def default_backend() -> str:
    """
    The extractor used when none is named: LOREBOOK_BACKEND if set, else
    "onnx" when onnxruntime is installed, else "keras".
    """
    override = os.environ.get("LOREBOOK_BACKEND")
    if override:
        return override.lower()
    return "onnx" if importlib.util.find_spec("onnxruntime") is not None else "keras"


_extractors: dict = {}


def get_extractor(backend: str | None = None):
    """
    Return a cached feature extractor for the given backend (default_backend() if None).

    The returned object exposes `.extract(img_or_path) -> Optional[np.ndarray]`,
    yielding a 1280-dim L2-normalized vector. Supported backends: see BACKENDS.
    """
    key = (backend or default_backend()).lower()
    if key not in _extractors:
        if key == "onnx":
            _extractors[key] = _OnnxExtractor()
        elif key == "keras":
            _extractors[key] = _KerasExtractor()
        elif key == "tflite":
            _extractors[key] = _TFLiteExtractor()
        else:
            raise ValueError(f"Unknown extractor backend: {backend!r} (use one of {', '.join(BACKENDS)}).")
    return _extractors[key]


def extract_features(img_or_path: np.ndarray | str) -> np.ndarray | None:
    """
    Extract a normalized 1280-dim feature vector from an image or file path.

    Uses default_backend(); the model is loaded lazily on first call. For a
    specific backend use get_extractor(backend).extract(...).
    """
    return get_extractor().extract(img_or_path)


def visualize_activation_overlay(img_bgr: np.ndarray, model=None, blocking: bool = True) -> np.ndarray:
    """Return the image with a jet-colormap heatmap of average last-conv activations blended in.

    blocking=False is for per-frame callers on the GUI thread: if the model
    isn't loaded yet or is busy (a scan or DB-build batch holds it), the
    image is returned unchanged instead of stalling the preview.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    if model is None and default_backend() == "onnx":
        onnx = get_extractor("onnx")
        if not blocking and not onnx.loaded():
            return img_bgr
        return _activation_overlay(img_bgr, onnx.activations)
    if not blocking and model is None and not models_loaded():
        return img_bgr
    if not _model_lock.acquire(blocking=blocking):
        return img_bgr
    try:
        use_model = model or _get_models()[1]
        return _activation_overlay(img_bgr, lambda batch: _infer(use_model, batch))
    finally:
        _model_lock.release()


def _activation_overlay(img_bgr: np.ndarray, activations_fn) -> np.ndarray:
    """activations_fn maps a preprocessed (1, 224, 224, 3) batch to (1, 7, 7, 1280)."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    activations = activations_fn(_mobilenet_preprocess(batch))[0]  # (7,7,1280)

    heatmap = activations.mean(axis=-1)  # (7,7)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    hmin, hmax = float(np.min(heatmap)), float(np.max(heatmap))
    if hmax > hmin:
        heatmap = (255 * (heatmap - hmin) / (hmax - hmin)).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)

    return cv2.addWeighted(img_bgr, 0.6, cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), 0.4, 0)
