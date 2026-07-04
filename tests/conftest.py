# tests/conftest.py — shared test setup.
#
# Stubs TensorFlow/Keras in sys.modules before any test module imports
# lorebook code, so the suite runs without those heavy packages installed
# (no model download, no GPU). Also puts the repo root on sys.path so
# `from PhotoMatching import ...` works from any pytest invocation dir.

import os
import sys
import unittest.mock

_tf_mock = unittest.mock.MagicMock()
for _mod in [
    "tensorflow", "tf",
    "keras", "keras.applications", "keras.applications.mobilenet_v2", "keras.models",
]:
    sys.modules.setdefault(_mod, _tf_mock)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
