import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    library = Path(os.environ["LD_PRELOAD"]).resolve(strict=True)
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = os.environ.get("METRIC_NCCL_SHA256")
    if expected is not None and digest != expected:
        raise ValueError("Preloaded NCCL library does not match the frozen SHA-256")
    get_version = ctypes.CDLL(None).ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    result = get_version(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"NCCL version query failed: {result}")
    print(json.dumps(dict(
        schema="rulervla_metric_nccl_runtime_v1",
        library_path=str(library),
        sha256=digest,
        version=version.value,
        expected_sha256=expected,
        jax_imported="jax" in sys.modules,
    )), flush=True)


if __name__ == "__main__":
    main()
