from __future__ import annotations

import sys
from typing import Callable, List

import numpy as np

from _bootstrap import add_project_root

add_project_root()

from model.embedding import Embedding


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"[OK]  {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        raise


def test_embedding_backward_accumulates_rows_1d() -> None:
    emb = Embedding.init(
        vocab_size=5,
        d_model=3,
        seed=0,
    )

    emb.dW = np.zeros_like(emb.W)

    ids = np.array(
        [1, 1, 3],
        dtype=np.int64,
    )

    out = emb.forward(ids)

    assert out.shape == (3, 3)

    dout = np.ones_like(
        out,
        dtype=np.float32,
    )

    emb.backward(dout)

    # Token 1 appears twice.
    assert np.allclose(
        emb.dW[1],
        np.array([2, 2, 2], dtype=np.float32),
    )

    # Token 3 appears once.
    assert np.allclose(
        emb.dW[3],
        np.array([1, 1, 1], dtype=np.float32),
    )

    # Other rows remain zero.
    assert np.allclose(
        emb.dW[0],
        0.0,
    )


def test_embedding_backward_2d_shape() -> None:
    emb = Embedding.init(
        vocab_size=7,
        d_model=4,
        seed=0,
    )

    emb.dW = np.zeros_like(emb.W)

    ids = np.array(
        [
            [0, 2],
            [2, 6],
        ],
        dtype=np.int64,
    )

    out = emb.forward(ids)

    dout = np.ones_like(
        out,
        dtype=np.float32,
    )

    emb.backward(dout)

    assert emb.dW.shape == (7, 4)

    # Token 2 appears twice.
    assert np.allclose(
        emb.dW[2],
        2.0,
    )


def main() -> int:
    tests: List[tuple[str, Callable[[], None]]] = [
        (
            "embedding_backward_accumulates_rows_1d",
            test_embedding_backward_accumulates_rows_1d,
        ),
        (
            "embedding_backward_2d_shape",
            test_embedding_backward_2d_shape,
        ),
    ]

    for name, fn in tests:
        _run(name, fn)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(1)
