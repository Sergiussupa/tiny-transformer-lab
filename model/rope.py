from __future__ import annotations

from typing import Tuple

from model.backend import xp as np


def rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    *,
    base: float = 10000.0,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute cosine and sine tables for RoPE.

    Returns:
      cos: (max_seq_len, head_dim // 2)
      sin: (max_seq_len, head_dim // 2)
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    half = head_dim // 2

    idx = np.arange(half, dtype=dtype)
    inv_freq = base ** (-idx / half)

    positions = np.arange(max_seq_len, dtype=dtype)
    angles = np.outer(positions, inv_freq)

    cos = np.cos(angles)
    sin = np.sin(angles)

    return cos, sin


def apply_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """
    Apply RoPE along the last axis.

    x: (..., T, D)
    cos/sin: (T, D // 2)

    Returns an array with the same shape as x.
    """
    x = np.asarray(x)

    D = x.shape[-1]

    if D % 2 != 0:
        raise ValueError("Last dimension must be even for RoPE")

    T = x.shape[-2]

    if cos.shape[0] < T or sin.shape[0] < T:
        raise ValueError("cos/sin length < sequence length")

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    cos_t = cos[:T]
    sin_t = sin[:T]

    # Broadcast over batch/head dimensions.
    while cos_t.ndim < x_even.ndim:
        cos_t = cos_t[None, ...]
        sin_t = sin_t[None, ...]

    out_even = x_even * cos_t - x_odd * sin_t
    out_odd = x_even * sin_t + x_odd * cos_t

    out = np.empty_like(x)

    out[..., ::2] = out_even
    out[..., 1::2] = out_odd

    return out
