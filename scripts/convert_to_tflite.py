#!/usr/bin/env python3
"""Convert the MobileNetV2 feature model to a .tflite file for the sorter.

Run once on a desktop (where full TensorFlow is installed); copy the resulting
.tflite to the Raspberry Pi, where it runs via tflite-runtime without TF.

    python scripts/convert_to_tflite.py
    python scripts/convert_to_tflite.py --output mobilenetv2_features.tflite
"""
import argparse
import os

from lorebook.core.features import _default_tflite_model_path, _get_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=_default_tflite_model_path(), help="Output .tflite path."
    )
    args = parser.parse_args()

    import tensorflow as tf

    # Convert the exact model instance the keras extractor uses, so the
    # exported .tflite can never diverge from it.
    feat_model, _ = _get_models()
    converter = tf.lite.TFLiteConverter.from_keras_model(feat_model)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {args.output} ({len(tflite_model) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
