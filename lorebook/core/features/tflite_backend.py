"""MobileNetV2 feature extraction on a .tflite Interpreter.

Light enough for the Raspberry Pi sorter: tflite_runtime is preferred, with
tensorflow.lite as the desktop fallback.
"""

import logging
import os
from collections.abc import Sequence

import numpy as np

from lorebook.core.features.preprocessing import finalize_vector, preprocess_to_batch

logger = logging.getLogger(__name__)

# lorebook/core/features/tflite_backend.py -> four levels up is the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def default_tflite_model_path() -> str:
    """
    Where the tflite feature model lives, resolved at call time (not import
    time) so LOREBOOK_TFLITE_MODEL takes effect whenever it is set.
    """
    return os.environ.get(
        "LOREBOOK_TFLITE_MODEL", os.path.join(_REPO_ROOT, "mobilenetv2_features.tflite")
    )


def _check_tflite_input_dtype(dtype, model_path: str) -> None:
    """
    Reject non-float32 tflite models.

    The [-1, 1] float batch from mobilenet_preprocess would be silently
    truncated to garbage by a blind cast to an int8/uint8 quantized input,
    producing vectors that pass the size check but match near-randomly.
    """
    if np.dtype(dtype) != np.float32:
        raise ValueError(
            f"tflite model at {model_path} expects {np.dtype(dtype).name} input; "
            "only float32 models are supported. Re-export without quantization "
            "(scripts/convert_to_tflite.py)."
        )


class TFLiteExtractor:
    """MobileNetV2 feature extractor backed by a .tflite Interpreter (lazy-loaded)."""

    def __init__(self, model_path: str | None = None):
        """Remember the model path; the interpreter is loaded on first use."""
        self._explicit_model_path = model_path
        self._interpreter = None
        self._in_index = None
        self._out_index = None

    @property
    def model_path(self) -> str:
        """Return the .tflite file to load: the explicit path, else the default location."""
        return self._explicit_model_path or default_tflite_model_path()

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
        """Return the image's normalized 1280-dim vector, or None if it can't be extracted."""
        try:
            interp = self._ensure_interpreter()
            batch = preprocess_to_batch(img_or_path)
            if batch is None:
                return None
            interp.set_tensor(self._in_index, batch)
            interp.invoke()
            features = interp.get_tensor(self._out_index)
            return finalize_vector(features)
        except Exception as e:  # noqa: BLE001 — boundary, see KerasExtractor.extract()
            logger.error("Error extracting features (tflite): %s", e)
            return None

    def extract_batch(self, images: Sequence) -> list[np.ndarray | None]:
        """API parity with the keras extractor; the tflite interpreter is
        fixed at batch size 1, so this is a plain loop (no speedup)."""
        return [self.extract(img) for img in images]
