# ruff_check.py — PostToolUse hook: lint each Python file right after Claude edits it.
#
# Reads the hook payload from stdin. Exit code 2 feeds ruff's output back to
# Claude so it fixes the problem now instead of in CI. Silently does nothing
# when ruff isn't installed or the file isn't Python.

import json
import shutil
import subprocess
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    path = (payload.get("tool_input") or {}).get("file_path", "")
    if not path.endswith(".py") or shutil.which("ruff") is None:
        return 0
    result = subprocess.run(["ruff", "check", path], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout + result.stderr, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
