# tests/test_onnx_backend.py — ONNX model build + "onnx" extractor (no TensorFlow).
#
# The model build is checked on synthetic weights with the real MobileNetV2
# layer layout (needs h5py + onnx + onnxruntime, skipped otherwise). Parity with
# real Keras vectors is a manual check: scripts/check_parity.py.

import io

import numpy as np
import pytest

import lorebook.core.features as features
import lorebook.core.onnx_model as onnx_model


def _synthetic_weights() -> bytes:
    """An .h5 shaped like Keras' MobileNetV2 no-top weights, random values."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    layers: list[tuple[str, dict[str, np.ndarray]]] = []

    def conv(kind: str, k: int, cin: int, cout: int) -> None:
        n = len(layers)
        if kind == "dw":
            layers.append((f"dw{n}", {"depthwise_kernel": rng.normal(0, 0.1, (k, k, cin, 1))}))
        else:
            layers.append((f"conv{n}", {"kernel": rng.normal(0, 0.1, (k, k, cin, cout))}))
        layers.append((f"bn{n}", {
            "gamma": rng.uniform(0.5, 1.5, cout), "beta": rng.normal(0, 0.1, cout),
            "moving_mean": rng.normal(0, 0.1, cout), "moving_variance": rng.uniform(0.5, 1.5, cout),
        }))

    conv("conv", 3, 3, 32)
    ch = 32
    for block_id, (expansion, _stride, filters) in enumerate(onnx_model._BLOCKS):
        hidden = ch * expansion
        if block_id:
            conv("conv", 1, ch, hidden)
        conv("dw", 3, hidden, hidden)
        conv("conv", 1, hidden, filters)
        ch = filters
    conv("conv", 1, ch, 1280)

    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        names = ["input_1"] + [n for n, _ in layers] + ["out_relu"]
        f.attrs["layer_names"] = [n.encode() for n in names]
        for name in ("input_1", "out_relu"):
            f.create_group(name).attrs["weight_names"] = []
        for name, weights in layers:
            g = f.create_group(name)
            g.attrs["weight_names"] = [f"{name}/{w}:0".encode() for w in weights]
            for w, arr in weights.items():
                g.create_dataset(f"{name}/{w}:0", data=arr.astype(np.float32))
    return buf.getvalue()


@pytest.fixture
def built_model(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / "m.onnx"
    path.write_bytes(onnx_model.build_model_bytes(_synthetic_weights()))
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_built_model_outputs(built_model):
    batch = np.random.default_rng(1).uniform(-1, 1, (2, 224, 224, 3)).astype(np.float32)
    feats, acts = built_model.run(["features", "activations"], {"input": batch})
    assert feats.shape == (2, 1280) and acts.shape == (2, 7, 7, 1280)
    # features is the global average of the (ReLU6-clipped) activations.
    np.testing.assert_allclose(feats, acts.mean(axis=(1, 2)), rtol=1e-4, atol=1e-5)
    assert acts.min() >= 0.0 and acts.max() <= 6.0


def test_build_rejects_wrong_layer_count():
    pytest.importorskip("onnx")
    h5py = pytest.importorskip("h5py")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["layer_names"] = [b"Conv1"]
        g = f.create_group("Conv1")
        g.attrs["weight_names"] = [b"Conv1/kernel:0"]
        g.create_dataset("Conv1/kernel:0", data=np.zeros((3, 3, 3, 32), np.float32))
    with pytest.raises(ValueError, match="conv/BN layers"):
        onnx_model.build_model_bytes(buf.getvalue())


def test_weights_integrity_check(monkeypatch):
    monkeypatch.setattr(onnx_model, "_keras_cached_weights", lambda: None)
    monkeypatch.setattr(onnx_model, "fetch_bytes", lambda url, timeout: b"not the weights")
    with pytest.raises(ValueError, match="integrity"):
        onnx_model._load_weights_bytes()


def test_ensure_model_reuses_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"existing")
    monkeypatch.setattr(onnx_model, "build_model_bytes", lambda w: pytest.fail("rebuilt"))
    assert onnx_model.ensure_model(str(path)) == str(path)


def test_ensure_model_builds_once_when_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(onnx_model, "_load_weights_bytes", lambda: b"w")
    monkeypatch.setattr(onnx_model, "build_model_bytes", lambda w: calls.append(w) or b"model")
    path = str(tmp_path / "m.onnx")
    assert onnx_model.ensure_model(path) == path
    assert onnx_model.ensure_model(path) == path
    assert calls == [b"w"]
    assert (tmp_path / "m.onnx").read_bytes() == b"model"


def test_default_backend(monkeypatch):
    monkeypatch.setenv("LOREBOOK_BACKEND", "TFLite")
    assert features.default_backend() == "tflite"
    monkeypatch.delenv("LOREBOOK_BACKEND")
    has_ort = features.importlib.util.find_spec("onnxruntime") is not None
    assert features.default_backend() == ("onnx" if has_ort else "keras")


class _FakeSession:
    def __init__(self):
        self.runs: list[tuple[list[str], tuple[int, ...]]] = []

    def run(self, outputs, feeds):
        batch = feeds["input"]
        self.runs.append((outputs, batch.shape))
        if outputs == ["activations"]:
            return [np.ones((batch.shape[0], 7, 7, 1280), np.float32)]
        return [np.ones((batch.shape[0], 1280), np.float32)]


def _card() -> np.ndarray:
    return np.random.default_rng(2).integers(0, 255, (300, 220, 3), dtype=np.uint8)


def test_onnx_extract_and_batch():
    ext = features._OnnxExtractor()
    ext._session = _FakeSession()
    vec = ext.extract(_card())
    assert vec is not None and vec.shape == (1280,)
    assert np.isclose(np.linalg.norm(vec), 1.0)

    out = ext.extract_batch([_card(), np.zeros((0, 0, 3), np.uint8), _card()])
    assert out[1] is None and out[0] is not None and out[2] is not None
    assert ext._session.runs[-1] == (["features"], (2, 224, 224, 3))  # one run, valid rows only


def test_onnx_extract_failure_returns_none():
    ext = features._OnnxExtractor()

    class Boom:
        def run(self, *a):
            raise RuntimeError("bad session")

    ext._session = Boom()
    assert ext.extract(_card()) is None
    assert ext.extract_batch([_card()]) == [None]


def test_overlay_uses_onnx_and_never_loads_when_nonblocking(monkeypatch):
    monkeypatch.setenv("LOREBOOK_BACKEND", "onnx")
    ext = features._OnnxExtractor()
    monkeypatch.setitem(features._extractors, "onnx", ext)
    img = _card()
    assert features.visualize_activation_overlay(img, blocking=False) is img  # not loaded yet
    ext._session = _FakeSession()
    out = features.visualize_activation_overlay(img, blocking=False)
    assert out.shape == img.shape and out is not img
    assert ext._session.runs[-1][0] == ["activations"]
