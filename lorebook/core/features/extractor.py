"""The interface every feature backend implements."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

# An image already in memory (BGR, gray or BGRA) or the path of an image file.
ImageInput = np.ndarray | str


class FeatureExtractor(Protocol):
    """Turns card images into 1280-dim L2-normalized feature vectors.

    The ONNX, Keras and tflite backends all satisfy this, so callers (the
    cache build, the scan worker, the sorter) never depend on a particular
    one. Anything with these methods can stand in, which is how tests supply
    fakes.
    """

    def ensure_ready(self) -> None:
        """Load the model now, raising if it can't be loaded, so callers can fail fast."""

    def extract(self, img_or_path: ImageInput) -> np.ndarray | None:
        """Return the image's vector, or None if it can't be read or extracted."""

    def extract_batch(self, images: Sequence[ImageInput]) -> list[np.ndarray | None]:
        """Return one vector (or None) per image, in order."""
