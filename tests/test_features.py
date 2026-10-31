# tests/test_features.py — keras extractor call paths (fake models, no TensorFlow).
#
# Single-image inference must call the model directly, never predict():
# predict() builds a tf.data pipeline and traces a tf.function per call, which
# is slow for one frame and is where a scan crashed natively on a worker
# thread. Batch extraction (DB builds) keeps predict().

import threading

import numpy as np
import pytest

import lorebook.core.features as features


class _FakeTensor:
    """Stands in for a tf EagerTensor: array conversion only, no indexing or .mean()."""

    def __init__(self, value: np.ndarray):
        self._value = value

    def __array__(self, dtype=None, copy=None):
        return self._value if dtype is None else self._value.astype(dtype)


class FakeModel:
    """Records direct calls; predict() fails unless the test allows it.

    A direct call returns a _FakeTensor (real Keras returns a tensor, while
    predict() returns an ndarray), or raises `fail` when a test sets it.
    """

    def __init__(self, out_shape: tuple[int, ...], allow_predict: bool = False):
        self.out_shape = out_shape
        self.allow_predict = allow_predict
        self.fail: Exception | None = None
        self.calls: list[tuple[tuple[int, ...], object]] = []
        self.predict_calls: list[tuple[int, ...]] = []
        self.lock_free_during_call: list[bool] = []

    def _probe_lock(self) -> None:
        # True if another thread could take _model_lock right now.
        result: list[bool] = []

        def try_acquire() -> None:
            got = features._model_lock.acquire(blocking=False)
            if got:
                features._model_lock.release()
            result.append(got)

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join()
        self.lock_free_during_call.append(result[0])

    def __call__(self, batch, training=None):
        self._probe_lock()
        self.calls.append((batch.shape, training))
        if self.fail is not None:
            raise self.fail
        rng = np.random.default_rng(0)
        return _FakeTensor(rng.random((batch.shape[0], *self.out_shape), dtype=np.float32))

    def predict(self, batch, verbose=0):
        if not self.allow_predict:
            raise AssertionError("predict() must not be used for single-image inference")
        self.predict_calls.append(batch.shape)
        rng = np.random.default_rng(1)
        return rng.random((batch.shape[0], *self.out_shape), dtype=np.float32)


@pytest.fixture
def fake_models(monkeypatch):
    """Install fake feature/activation models as the loaded Keras models."""
    feat = FakeModel((1280,))
    act = FakeModel((7, 7, 1280))
    monkeypatch.setattr(features, "_feat_model", feat)
    monkeypatch.setattr(features, "_act_model", act)
    return feat, act


def _card(h: int = 300, w: int = 200) -> np.ndarray:
    return np.random.default_rng(2).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_extract_calls_model_directly(fake_models):
    feat, _ = fake_models
    vec = features.extract_features(_card())

    assert feat.calls == [((1, 224, 224, 3), False)]
    assert feat.predict_calls == []
    assert vec is not None and vec.shape == (1280,) and vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
    assert feat.lock_free_during_call == [False]  # serialized with builds/heatmap


def test_extract_invalid_image_skips_model(fake_models):
    feat, _ = fake_models
    assert features.extract_features(np.zeros((0, 0, 3), np.uint8)) is None
    assert feat.calls == []


def test_extract_model_error_returns_none(fake_models, caplog):
    # One bad frame must not abort a scan or sort run.
    feat, _ = fake_models
    feat.fail = RuntimeError("boom")

    assert features.extract_features(_card()) is None
    assert len(feat.calls) == 1
    assert "Error extracting features (keras): boom" in caplog.text


def test_ensure_ready_runs_one_warm_up_inference(fake_models):
    feat, _ = fake_models
    features.get_extractor("keras").ensure_ready()

    assert feat.calls == [((1, 224, 224, 3), False)]
    assert feat.predict_calls == []
    assert feat.lock_free_during_call == [False]


def test_ensure_ready_raises_when_warm_up_inference_fails(fake_models):
    # The sorter fails fast on this error and ScanWorker.warm_up logs it, so
    # the warm-up inference must not swallow it.
    feat, _ = fake_models
    feat.fail = RuntimeError("no model")

    with pytest.raises(RuntimeError, match="no model"):
        features.get_extractor("keras").ensure_ready()


def test_activation_overlay_calls_model_directly(fake_models):
    _, act = fake_models
    img = _card(240, 180)
    out = features.visualize_activation_overlay(img)

    assert act.calls == [((1, 224, 224, 3), False)]
    assert act.predict_calls == []
    assert out.shape == img.shape and out.dtype == img.dtype
    assert act.lock_free_during_call == [False]


def test_activation_overlay_non_blocking_skips_while_model_busy(fake_models):
    # The GUI thread draws the heatmap per frame; it must not wait for a scan
    # or DB-build batch that holds the model.
    _, act = fake_models
    img = _card(240, 180)
    held, release = threading.Event(), threading.Event()

    def hold_lock() -> None:  # another thread: _model_lock is reentrant
        with features._model_lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold_lock)
    t.start()
    try:
        assert held.wait(5)
        assert features.visualize_activation_overlay(img, blocking=False) is img
    finally:
        release.set()
        t.join()
    assert act.calls == []


def test_activation_overlay_non_blocking_never_loads_model(monkeypatch):
    monkeypatch.setattr(features, "_feat_model", None)
    monkeypatch.setattr(features, "_act_model", None)

    def no_load():
        raise AssertionError("a non-blocking overlay must not load the model")

    monkeypatch.setattr(features, "_get_models", no_load)
    img = _card(240, 180)

    assert features.visualize_activation_overlay(img, blocking=False) is img


def test_extract_batch_keeps_predict_and_none_slots(fake_models):
    feat, _ = fake_models
    feat.allow_predict = True
    results = features.get_extractor("keras").extract_batch(
        [_card(), np.zeros((0, 0, 3), np.uint8), _card(400, 280)]
    )

    assert feat.predict_calls == [(2, 224, 224, 3)]  # one call, invalid input dropped
    assert feat.calls == []
    assert results[1] is None
    assert all(r is not None and r.shape == (1280,) for r in (results[0], results[2]))
