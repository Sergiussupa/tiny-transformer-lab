from __future__ import annotations

import os
import numpy as _numpy

_requested = os.environ.get("TINY_TRANSFORMER_DEVICE", "cpu").strip().lower()

if _requested in {"gpu", "cuda", "cupy"}:
    try:
        import cupy as xp
    except Exception as exc:
        raise RuntimeError(
            "TINY_TRANSFORMER_DEVICE=gpu, but CuPy cannot be imported. "
            "Install a matching cupy-cudaXX package."
        ) from exc

    try:
        _device_count = int(xp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "CuPy is installed, but CUDA initialization failed. "
            "Check nvidia-smi, the NVIDIA driver, and the CuPy CUDA package."
        ) from exc

    if _device_count < 1:
        raise RuntimeError("CuPy found no CUDA-capable GPU")

    IS_GPU = True
    DEVICE = "gpu"

else:
    xp = _numpy
    IS_GPU = False
    DEVICE = "cpu"


def host_rng(seed=None):
    """NumPy RNG keeps initialization and dataset seeds CPU-compatible."""
    return _numpy.random.default_rng(seed)


def to_numpy(value, dtype=None):
    """Copy a GPU array to host; leave CPU values as NumPy arrays."""
    if IS_GPU and isinstance(value, xp.ndarray):
        arr = xp.asnumpy(value)
    else:
        arr = _numpy.asarray(value)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def scalar(value):
    """Return a Python scalar from a NumPy/CuPy scalar or 0-D array."""
    arr = to_numpy(value)
    return arr.item() if getattr(arr, "shape", None) == () else arr


def synchronize() -> None:
    if IS_GPU:
        xp.cuda.Stream.null.synchronize()


def device_summary() -> str:
    if not IS_GPU:
        return "CPU / NumPy"

    dev_id = int(xp.cuda.runtime.getDevice())
    props = xp.cuda.runtime.getDeviceProperties(dev_id)

    name = props.get("name", b"CUDA GPU")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")

    return f"GPU {dev_id}: {name} / CuPy {xp.__version__}"
