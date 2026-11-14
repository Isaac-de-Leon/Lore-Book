# fileio.py — atomic file writes shared by every writer in the app.
#
# Write to a uniquely named temp file in the target's directory, then
# os.replace() it over the target. Readers (and a crash mid-write) only ever
# see the old file or the complete new one; concurrent writers never share a
# temp file; a failure removes the temp file and re-raises.

import contextlib
import json
import os
import stat
import tempfile
from collections.abc import Callable
from typing import IO, Any


def _target_mode(path: str) -> int:
    """The existing file's permissions, or the umask default for a new file.

    mkstemp creates files 0600; without this, os.replace would silently
    tighten the target's permissions on every write.
    """
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def atomic_write(
    path: str,
    write: Callable[[IO[Any]], object],
    *,
    binary: bool = False,
    encoding: str = "utf-8",
    newline: str | None = None,
    suffix: str = ".tmp",
    fsync: bool = False,
) -> None:
    """Atomically replace ``path`` with what ``write(f)`` writes to ``f``.

    suffix names the temp file (``<name>.<random><suffix>``) so callers can
    recognise and clean up leftovers from a killed process. fsync=True also
    flushes to disk before the rename (for data that must survive power loss).
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=suffix, dir=directory)
    try:
        os.chmod(tmp, _target_mode(path))
        if binary:
            f: IO[Any] = os.fdopen(fd, "wb")
        else:
            f = os.fdopen(fd, "w", encoding=encoding, newline=newline)
        with f:
            write(f)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path: str, data: bytes, *, suffix: str = ".tmp") -> None:
    atomic_write(path, lambda f: f.write(data), binary=True, suffix=suffix)


def atomic_write_json(path: str, payload: Any, **dump_kwargs: Any) -> None:
    atomic_write(path, lambda f: json.dump(payload, f, **dump_kwargs))
