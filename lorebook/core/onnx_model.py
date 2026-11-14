# onnx_model.py — build the MobileNetV2 feature model as ONNX, without TensorFlow.
#
# Windows Smart App Control blocks TensorFlow's unsigned native DLL, so the
# desktop app runs feature extraction on ONNX Runtime (Microsoft-signed)
# instead. The model is rebuilt here, layer by layer, from the same ImageNet
# weights file Keras downloads (MobileNetV2, alpha 1.0, 224px, no top), with
# each BatchNormalization folded into its convolution. Only `onnx` and `h5py`
# are needed, and only once: the result is cached at default_model_path().
#
# Outputs match the Keras models in features.py:
#   "features"    (N, 1280)        global-average-pooled features (pre-normalization)
#   "activations" (N, 7, 7, 1280)  last ReLU6 activations, NHWC (heatmap overlay)
# Input "input" is (N, 224, 224, 3) float32 NHWC, already scaled to [-1, 1].

import hashlib
import io
import logging
import os
import threading

import numpy as np

from lorebook.core.fileio import atomic_write_bytes
from lorebook.core.net import fetch_bytes
from lorebook.core.paths import data_path

logger = logging.getLogger(__name__)

MODEL_FILENAME = "mobilenetv2_features.onnx"
WEIGHTS_FILENAME = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
WEIGHTS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/" + WEIGHTS_FILENAME
WEIGHTS_SHA256 = "f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c"

INPUT_SIZE = 224
FEATURE_DIM = 1280
_BN_EPSILON = 1e-3   # keras.applications.MobileNetV2's BatchNormalization epsilon
_OPSET = 13

# Inverted-residual blocks of MobileNetV2 (alpha 1.0): (expansion, stride, output channels).
_BLOCKS = [
    (1, 1, 16),
    (6, 2, 24), (6, 1, 24),
    (6, 2, 32), (6, 1, 32), (6, 1, 32),
    (6, 2, 64), (6, 1, 64), (6, 1, 64), (6, 1, 64),
    (6, 1, 96), (6, 1, 96), (6, 1, 96),
    (6, 2, 160), (6, 1, 160), (6, 1, 160),
    (6, 1, 320),
]

_build_lock = threading.Lock()


def default_model_path() -> str:
    """Where the ONNX model lives; LOREBOOK_ONNX_MODEL overrides (read at call time)."""
    return os.environ.get("LOREBOOK_ONNX_MODEL") or data_path(MODEL_FILENAME)


def _keras_cached_weights() -> str | None:
    """The weights file Keras already downloaded, if any (KERAS_HOME or ~/.keras)."""
    keras_home = os.environ.get("KERAS_HOME") or os.path.join(os.path.expanduser("~"), ".keras")
    path = os.path.join(keras_home, "models", WEIGHTS_FILENAME)
    return path if os.path.isfile(path) else None


def _load_weights_bytes() -> bytes:
    """Weights from the Keras cache when present, else downloaded; hash-checked either way."""
    cached = _keras_cached_weights()
    if cached:
        with open(cached, "rb") as f:
            data = f.read()
        source = cached
    else:
        logger.info("Downloading MobileNetV2 weights (~9 MB)…")
        data = fetch_bytes(WEIGHTS_URL, timeout=120)
        source = WEIGHTS_URL
    digest = hashlib.sha256(data).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"MobileNetV2 weights from {source} failed the integrity check (sha256 {digest}).")
    return data


def _read_conv_bn_pairs(h5) -> list[tuple[str, np.ndarray, tuple[np.ndarray, ...]]]:
    """
    Return [(kind, kernel, (gamma, beta, mean, var)), ...] in model order.

    Parsed by weight type rather than layer name, so it doesn't depend on
    which Keras version named the layers. Every conv is followed by its BN.
    """
    def _s(v) -> str:
        return v.decode() if isinstance(v, bytes) else str(v)

    convs: list[tuple[str, np.ndarray]] = []
    bns: list[tuple[np.ndarray, ...]] = []
    for layer in (_s(n) for n in h5.attrs["layer_names"]):
        group = h5[layer]
        weights = {_s(w).rsplit("/", 1)[-1].split(":")[0]: np.asarray(group[_s(w)], np.float32)
                   for w in group.attrs["weight_names"]}
        if not weights:
            continue
        if "depthwise_kernel" in weights:
            convs.append(("dw", weights["depthwise_kernel"]))
        elif "kernel" in weights:
            convs.append(("conv", weights["kernel"]))
        elif "gamma" in weights:
            bns.append(tuple(weights[k] for k in ("gamma", "beta", "moving_mean", "moving_variance")))
        else:
            raise ValueError(f"Unexpected weights in layer {layer!r}: {sorted(weights)}")
    expected = 2 + sum(3 if i else 2 for i in range(len(_BLOCKS)))
    if len(convs) != expected or len(bns) != expected:
        raise ValueError(f"Expected {expected} conv/BN layers, found {len(convs)}/{len(bns)}.")
    return [(kind, kernel, bn) for (kind, kernel), bn in zip(convs, bns)]


def build_model_bytes(weights: bytes) -> bytes:
    """Build the serialized ONNX model from the Keras .h5 weights file contents."""
    import h5py
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    with h5py.File(io.BytesIO(weights), "r") as h5:
        layers = iter(_read_conv_bn_pairs(h5))

    nodes: list = []
    inits: list = []
    counter = [0]

    def fresh(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    inits += [numpy_helper.from_array(np.array(0.0, np.float32), "relu6_min"),
              numpy_helper.from_array(np.array(6.0, np.float32), "relu6_max")]

    def conv_bn(x: str, stride: int, relu6: bool) -> str:
        kind, kernel, (gamma, beta, mean, var) = next(layers)
        scale = gamma / np.sqrt(var + _BN_EPSILON)
        if kind == "dw":   # (kh, kw, C, 1) -> (C, 1, kh, kw)
            w = kernel.transpose(2, 3, 0, 1)
            group = w.shape[0]
        else:              # (kh, kw, in, out) -> (out, in, kh, kw)
            w = kernel.transpose(3, 2, 0, 1)
            group = 1
        w = (w * scale[:, None, None, None]).astype(np.float32)
        b = (beta - mean * scale).astype(np.float32)
        k = w.shape[-1]
        # Keras pads stride-2 3x3 convs bottom/right only (TF "same" / correct_pad
        # on even sizes); ONNX pads are [top, left, bottom, right].
        pads = [0, 0, 0, 0] if k == 1 else ([1, 1, 1, 1] if stride == 1 else [0, 0, 1, 1])
        wn, bn_, out = fresh("W"), fresh("B"), fresh("conv")
        inits.extend([numpy_helper.from_array(w, wn), numpy_helper.from_array(b, bn_)])
        nodes.append(helper.make_node("Conv", [x, wn, bn_], [out], kernel_shape=[k, k],
                                      strides=[stride, stride], pads=pads, group=group))
        if not relu6:
            return out
        act = fresh("relu6")
        nodes.append(helper.make_node("Clip", [out, "relu6_min", "relu6_max"], [act]))
        return act

    nodes.append(helper.make_node("Transpose", ["input"], ["nchw"], perm=[0, 3, 1, 2]))
    x = conv_bn("nchw", stride=2, relu6=True)
    channels = 32
    for block_id, (expansion, stride, filters) in enumerate(_BLOCKS):
        inp = x
        if block_id:   # block 0 has no expansion conv
            x = conv_bn(x, stride=1, relu6=True)
        x = conv_bn(x, stride=stride, relu6=True)       # depthwise
        x = conv_bn(x, stride=1, relu6=False)           # linear projection
        if stride == 1 and channels == filters:
            added = fresh("add")
            nodes.append(helper.make_node("Add", [inp, x], [added]))
            x = added
        channels = filters
    x = conv_bn(x, stride=1, relu6=True)                # Conv_1 -> 1280, out_relu

    nodes += [
        helper.make_node("Transpose", [x], ["activations"], perm=[0, 2, 3, 1]),
        helper.make_node("GlobalAveragePool", [x], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "mobilenetv2_features",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", INPUT_SIZE, INPUT_SIZE, 3])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", FEATURE_DIM]),
         helper.make_tensor_value_info("activations", TensorProto.FLOAT, ["N", 7, 7, FEATURE_DIM])],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="lorebook",
                              opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = 8   # loadable by any onnxruntime from 1.10 on
    onnx.checker.check_model(model)
    return model.SerializeToString()


def ensure_model(path: str | None = None) -> str:
    """Return the ONNX model path, building it first (one-time, ~seconds) if missing."""
    path = path or default_model_path()
    if os.path.isfile(path):
        return path
    with _build_lock:
        if not os.path.isfile(path):
            logger.info("Building ONNX feature model at %s", path)
            atomic_write_bytes(path, build_model_bytes(_load_weights_bytes()))
    return path
