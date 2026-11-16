# tests/test_tflite_backend.py — the tflite extractor against a fake Interpreter (no tflite-runtime).

import sys
import types

import numpy as np
import pytest

from lorebook.core.features.tflite_backend import TFLiteExtractor


class FakeInterpreter:
    """Mimics the tflite Interpreter calls the extractor makes."""

    input_dtype = np.float32
    fail_on_invoke = False

    def __init__(self, model_path):
        self.model_path = model_path
        self.inputs = []

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"dtype": self.input_dtype, "index": 0}]

    def get_output_details(self):
        return [{"index": 7}]

    def set_tensor(self, index, batch):
        assert index == 0
        self.inputs.append(batch)

    def invoke(self):
        if self.fail_on_invoke:
            raise RuntimeError("interpreter crashed")

    def get_tensor(self, index):
        assert index == 7
        return np.ones((1, 1280), np.float32)


@pytest.fixture
def tflite_runtime(monkeypatch):
    """Install a fake tflite_runtime package and return its Interpreter class (a fresh subclass)."""
    interpreter = type("Interpreter", (FakeInterpreter,), {})
    package = types.ModuleType("tflite_runtime")
    module = types.ModuleType("tflite_runtime.interpreter")
    module.Interpreter = interpreter
    monkeypatch.setitem(sys.modules, "tflite_runtime", package)
    monkeypatch.setitem(sys.modules, "tflite_runtime.interpreter", module)
    return interpreter


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "mobilenetv2_features.tflite"
    path.write_bytes(b"model")
    return str(path)


def _card():
    return np.random.default_rng(3).integers(0, 255, (300, 220, 3), dtype=np.uint8)


def test_extract_returns_a_normalized_vector(tflite_runtime, model_file):
    extractor = TFLiteExtractor(model_file)
    assert extractor.model_path == model_file
    extractor.ensure_ready()

    vector = extractor.extract(_card())
    assert vector is not None and vector.shape == (1280,) and vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    assert extractor._interpreter.inputs[-1].shape == (1, 224, 224, 3)

    # The interpreter is fixed at batch size 1: a batch is one call per image.
    results = extractor.extract_batch([_card(), np.zeros((0, 0, 3), np.uint8)])
    assert results[0] is not None and results[1] is None


def test_missing_model_file_says_how_to_generate_it(tflite_runtime, tmp_path):
    extractor = TFLiteExtractor(str(tmp_path / "missing.tflite"))
    with pytest.raises(FileNotFoundError, match="convert_to_tflite"):
        extractor.ensure_ready()
    assert extractor.extract(_card()) is None  # a scan degrades to "no vector" instead of raising


def test_quantized_model_is_rejected(tflite_runtime, model_file):
    tflite_runtime.input_dtype = np.int8
    with pytest.raises(ValueError, match="float32"):
        TFLiteExtractor(model_file).ensure_ready()


def test_interpreter_failure_returns_none(tflite_runtime, model_file):
    tflite_runtime.fail_on_invoke = True
    assert TFLiteExtractor(model_file).extract(_card()) is None
