from __future__ import annotations

import sys
from typing import Callable, List

import numpy as np

from _bootstrap import add_project_root

add_project_root()

from model.linear import Linear


def _run(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"[OK]  {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        raise


def test_linear_backward_2d_exact() -> None:
    # W: out=2, in=3
    W = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.float32,
    )

    b = np.array([0, 0], dtype=np.float32)

    lin = Linear(
        W=W.copy(),
        b=b.copy(),
    )

    x = np.array(
        [[1, 0, -1]],
        dtype=np.float32,
    )

    y = lin.forward(x)
    assert y.shape == (1, 2)

    dy = np.array(
        [[1.0, 2.0]],
        dtype=np.float32,
    )

    dx = lin.backward(dy)

    assert dx.shape == x.shape

    # dx = dy @ W
    expected_dx = dy @ W
    assert np.allclose(dx, expected_dx)

    # dW = dy^T @ x
    expected_dW = dy.T @ x
    assert np.allclose(lin.dW, expected_dW)

    # db = sum(dy)
    assert np.allclose(
        lin.db,
        np.array([1.0, 2.0], dtype=np.float32),
    )


def test_linear_backward_3d_shapes() -> None:
    lin = Linear.init(4, 6, seed=0)

    x = np.zeros(
        (2, 3, 4),
        dtype=np.float32,
    )

    y = lin.forward(x)
    dy = np.ones_like(y)

    dx = lin.backward(dy)

    assert dx.shape == x.shape
    assert lin.dW.shape == lin.W.shape
    assert lin.db.shape == lin.b.shape


def main() -> int:
    tests: List[tuple[str, Callable[[], None]]] = [
        (
            "linear_backward_2d_exact",
            test_linear_backward_2d_exact,
        ),
        (
            "linear_backward_3d_shapes",
            test_linear_backward_3d_shapes,
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
