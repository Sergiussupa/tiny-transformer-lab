from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.ops import RMSNorm, silu, softmax


def test_silu_basic_values():
    x = np.array(
        [-2.0, 0.0, 2.0],
        dtype=np.float32,
    )

    y = silu(x)

    assert abs(float(y[1])) < 1e-7

    # For these selected inputs the outputs are ordered.
    assert float(y[0]) < float(y[1]) < float(y[2])


def test_softmax_sums_to_one():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    y = softmax(x, axis=-1)

    sums = np.sum(y, axis=-1)

    assert np.allclose(
        sums,
        np.ones_like(sums),
    )

    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)


def test_softmax_shift_invariant():
    x = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    y1 = softmax(x)
    y2 = softmax(x + 1000.0)

    assert np.allclose(y1, y2), (y1, y2)


def test_rmsnorm_shape_and_scale():
    x = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    norm = RMSNorm.init(
        d_model=4,
        eps=1e-6,
    )

    y = norm.forward(x)

    assert y.shape == x.shape

    rms = np.sqrt(
        np.mean(
            y * y,
            axis=-1,
        )
    )

    assert np.allclose(
        rms,
        np.ones_like(rms),
        atol=1e-4,
    ), rms


def test_rmsnorm_bad_dim_raises():
    norm = RMSNorm.init(
        d_model=4,
    )

    bad = np.zeros(
        (2, 3),
        dtype=np.float32,
    )

    try:
        norm.forward(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rmsnorm_backward_matches_finite_difference():
    """
    Compare the manual RMSNorm backward pass with a numerical gradient.
    """
    x = np.array(
        [
            [0.7, -1.2, 0.3],
            [1.1, 0.2, -0.8],
        ],
        dtype=np.float64,
    )

    dy = np.array(
        [
            [0.4, -0.3, 0.8],
            [-0.5, 0.7, 0.2],
        ],
        dtype=np.float64,
    )

    norm = RMSNorm.init(
        d_model=3,
        eps=1e-6,
        dtype=np.float64,
    )

    y = norm.forward(x)
    dx = norm.backward(dy)

    analytic_dx = dx.copy()
    analytic_dw = norm.grad_weight.copy()

    eps = 1e-6

    # Numerical gradient with respect to x.
    numerical_dx = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[index] += eps
        x_minus[index] -= eps

        y_plus = norm.forward(x_plus)
        y_minus = norm.forward(x_minus)

        loss_plus = np.sum(y_plus * dy)
        loss_minus = np.sum(y_minus * dy)

        numerical_dx[index] = (
            loss_plus - loss_minus
        ) / (2.0 * eps)

    assert np.allclose(
        analytic_dx,
        numerical_dx,
        atol=1e-5,
        rtol=1e-5,
    ), (analytic_dx, numerical_dx)

    # Numerical gradient with respect to weight.
    original_weight = norm.weight.copy()
    numerical_dw = np.zeros_like(original_weight)

    for i in range(original_weight.shape[0]):
        norm.weight[:] = original_weight
        norm.weight[i] += eps
        loss_plus = np.sum(
            norm.forward(x) * dy
        )

        norm.weight[:] = original_weight
        norm.weight[i] -= eps
        loss_minus = np.sum(
            norm.forward(x) * dy
        )

        numerical_dw[i] = (
            loss_plus - loss_minus
        ) / (2.0 * eps)

    norm.weight[:] = original_weight

    assert np.allclose(
        analytic_dw,
        numerical_dw,
        atol=1e-5,
        rtol=1e-5,
    ), (analytic_dw, numerical_dw)


def run():
    tests = [
        (
            "silu_basic_values",
            test_silu_basic_values,
        ),
        (
            "softmax_sums_to_one",
            test_softmax_sums_to_one,
        ),
        (
            "softmax_shift_invariant",
            test_softmax_shift_invariant,
        ),
        (
            "rmsnorm_shape_and_scale",
            test_rmsnorm_shape_and_scale,
        ),
        (
            "rmsnorm_bad_dim_raises",
            test_rmsnorm_bad_dim_raises,
        ),
        (
            "rmsnorm_backward_matches_finite_difference",
            test_rmsnorm_backward_matches_finite_difference,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
