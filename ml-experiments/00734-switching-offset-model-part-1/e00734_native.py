"""Build and load the C kernel of experiment 734's Viterbi pass.

The kernel source is `e00734_kernels.c`. It is compiled with the system C compiler into a shared
library in a private folder of the user's own under the temporary folder, named after a hash of
the source and the flags, and loaded with `ctypes`. Each process compiles into a file named after
its process ID and renames the file into place, so concurrent workers never load a partial file.
No package is added: the build needs only `gcc` or `cc`. When the compiler is missing, or the
build or the load fails, `load` returns `None` and the callers use the NumPy path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

SOURCE: Final[Path] = Path(__file__).with_name("e00734_kernels.c")
FLAGS: Final[tuple[str, ...]] = ("-O3", "-march=native", "-fno-fast-math", "-ffp-contract=off")
"""No fast-math and no fused multiply-add, so every float32 addition is rounded as in NumPy."""
_F32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_I32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_I8 = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")


def _compiler() -> str | None:
    for name in ("gcc", "cc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    return None


@lru_cache(maxsize=1)
def load() -> ctypes.CDLL | None:
    """The compiled kernel library, or `None` when it cannot be built.

    Set `E00734_KERNEL=numpy` to skip the build and use the NumPy path.
    """
    if os.environ.get("E00734_KERNEL", "auto") == "numpy":
        return None
    cc = _compiler()
    if cc is None or not SOURCE.exists():
        return None
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(FLAGS).encode() + cc.encode())
    cache = Path(tempfile.gettempdir()) / f"e00734-kernels-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o077:
        return None
    lib_path = cache / f"e00734_kernels_{digest.hexdigest()[:16]}.so"
    if not lib_path.exists():
        tmp = lib_path.with_name(f".{lib_path.name}.{os.getpid()}.tmp")
        cmd = [cc, *FLAGS, "-shared", "-fPIC", str(SOURCE), "-o", str(tmp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except subprocess.SubprocessError, OSError:
            tmp.unlink(missing_ok=True)
            return None
        tmp.rename(lib_path)
    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError:
        return None
    lib.viterbi_chunk.restype = None
    lib.viterbi_chunk.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        _F32,
        _F32,
        _I32,
        _F32,
        _F32,
        _F32,
        _I8,
        _I8,
        _I8,
    ]
    return lib
