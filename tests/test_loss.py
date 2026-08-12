# tests/test_loss.py
from __future__ import annotations

import sys
from typing import Callable, List
import numpy as np

from _bootstrap import add_project_root
add_project_root()

from model.loss import causal_lm_cross_entropy


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"[OK]  {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        raise


def test_loss_zero_when_perfect_confidence() -> None:
    # 1 token, vocab=3, target=2
    logits = np.array([[[-10.0, -10.0, 10.0]]], dtype=np.float32)  # (1,1,3)
    tgt = np.array([[2]], dtype=np.int64)

    out = causal_lm_cross_entropy(logits, tgt)
    assert out.loss < 1e-3
    assert out.dlogits.shape == logits.shape

    # grad should sum to 0 over vocab for valid positions
    s = np.sum(out.dlogits, axis=-1)
    assert np.allclose(s, 0.0, atol=1e-6)


def test_loss_mask_and_ignore_index() -> None:
    logits = np.zeros((1, 3, 5), dtype=np.float32)
    tgt = np.array([[1, -100, 3]], dtype=np.int64)
    mask = np.array([[1, 1, 0]], dtype=np.int64)

    out = causal_lm_cross_entropy(logits, tgt, ignore_index=-100, mask=mask, reduction="mean")
    # valid positions: only t=0 (t=1 ignored_index, t=2 masked out)
    assert out.dlogits.shape == logits.shape

    # only first position has non-zero grad
    assert np.any(out.dlogits[0, 0] != 0)
    assert np.all(out.dlogits[0, 1] == 0)
    assert np.all(out.dlogits[0, 2] == 0)


def test_bad_shapes_raise() -> None:
    logits = np.zeros((2, 4, 7), dtype=np.float32)
    tgt = np.zeros((2, 5), dtype=np.int64)
    try:
        causal_lm_cross_entropy(logits, tgt)
        assert False, "expected ValueError"
    except ValueError:
        pass


def main() -> int:
    tests: List[tuple[str, Callable[[], None]]] = [
        ("loss_zero_when_perfect_confidence", test_loss_zero_when_perfect_confidence),
        ("loss_mask_and_ignore_index", test_loss_mask_and_ignore_index),
        ("bad_shapes_raise", test_bad_shapes_raise),
    ]
    for name, fn in tests:
        _run(name, fn)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(1)
