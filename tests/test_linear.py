from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.linear import Linear


def test_linear_2d_exact():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    W = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    b = np.array(
        [10.0, 20.0, 30.0, 40.0],
        dtype=np.float32,
    )

    lin = Linear(W=W, b=b)
    y = lin.forward(x)

    expected = np.array(
        [
            [11.0, 22.0, 33.0, 46.0],
            [9.0, 20.0, 31.0, 40.0],
        ],
        dtype=np.float32,
    )

    assert y.shape == (2, 4)
    assert np.allclose(y, expected), (y, expected)


def test_linear_3d_shape_and_values():
    x = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ],
        dtype=np.float32,
    )

    W = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    lin = Linear(W=W, b=None)
    y = lin.forward(x)

    expected = np.array(
        [
            [
                [1.0, 2.0],
                [4.0, 5.0],
            ]
        ],
        dtype=np.float32,
    )

    assert y.shape == (1, 2, 2)
    assert np.allclose(y, expected), (y, expected)


def test_linear_bad_last_dim_raises():
    lin = Linear.init(3, 2, seed=0)
    bad = np.zeros((5, 4), dtype=np.float32)

    try:
        lin.forward(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def run():
    tests = [
        ("linear_2d_exact", test_linear_2d_exact),
        ("linear_3d_shape_and_values", test_linear_3d_shape_and_values),
        ("linear_bad_last_dim_raises", test_linear_bad_last_dim_raises),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
