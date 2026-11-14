#!/bin/bash
# Install the same light test deps CI uses (.github/workflows/tests.yml) so
# `ruff check .` and pytest work immediately in fresh cloud sessions.
# Local machines manage their own environment, so this is a no-op there.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

pip install --quiet --disable-pip-version-check pytest numpy opencv-python-headless ruff
