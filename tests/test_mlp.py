from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.linear import Linear
from model.mlp import MLP


def test_mlp_output_shape():
    mlp = MLP.init(
        d_model=4,
        d_ff=8,
        seed=0,
    )

    x = np.random.randn(
        2,
        5,
        4,
    ).astype(np.float32)

    y = mlp.forward(x)

    assert y.shape == (
        2,
        5,
        4,
    )


def test_mlp_bad_last_dim_raises():
    mlp = MLP.init(
        d_model=4,
        d_ff=8,
        seed=0,
    )

    bad = np.zeros(
        (2, 5, 3),
        dtype=np.float32,
    )

    try:
        mlp.forward(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_mlp_zero_weights_gives_zero():
    d_model = 4
    d_ff = 8

    z_gate = Linear(
        W=np.zeros(
            (d_ff, d_model),
            dtype=np.float32,
        ),
        b=np.zeros(
            (d_ff,),
            dtype=np.float32,
        ),
    )

    z_up = Linear(
        W=np.zeros(
            (d_ff, d_model),
            dtype=np.float32,
        ),
        b=np.zeros(
            (d_ff,),
            dtype=np.float32,
        ),
    )

    z_down = Linear(
        W=np.zeros(
            (d_model, d_ff),
            dtype=np.float32,
        ),
        b=np.zeros(
            (d_model,),
            dtype=np.float32,
        ),
    )

    mlp = MLP(
        d_model=d_model,
        d_ff=d_ff,
        gate_proj=z_gate,
        up_proj=z_up,
        down_proj=z_down,
    )

    x = np.random.randn(
        1,
        6,
        4,
    ).astype(np.float32)

    y = mlp.forward(x)

    assert np.allclose(
        y,
        0.0,
    ), y


def test_mlp_backward_matches_finite_difference():
    """
    Compare the manually implemented MLP backward pass
    against a numerical gradient with respect to x.
    """
    mlp = MLP.init(
        d_model=4,
        d_ff=6,
        seed=42,
        dtype=np.float64,
    )

    x = np.array(
        [
            [
                [0.2, -0.4, 0.6, 0.1],
                [-0.5, 0.3, 0.7, -0.2],
            ]
        ],
        dtype=np.float64,
    )

    dout = np.array(
        [
            [
                [0.4, -0.1, 0.2, 0.5],
                [-0.3, 0.6, -0.2, 0.1],
            ]
        ],
        dtype=np.float64,
    )

    mlp.forward(x)

    analytic_dx = mlp.backward(
        dout
    ).copy()

    numerical_dx = np.zeros_like(x)

    eps = 1e-6

    for index in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[index] += eps
        x_minus[index] -= eps

        y_plus = mlp.forward(
            x_plus
        )

        loss_plus = np.sum(
            y_plus * dout
        )

        y_minus = mlp.forward(
            x_minus
        )

        loss_minus = np.sum(
            y_minus * dout
        )

        numerical_dx[index] = (
            loss_plus - loss_minus
        ) / (2.0 * eps)

    assert np.allclose(
        analytic_dx,
        numerical_dx,
        atol=1e-5,
        rtol=1e-5,
    ), (
        analytic_dx,
        numerical_dx,
    )


def run():
    tests = [
        (
            "mlp_output_shape",
            test_mlp_output_shape,
        ),
        (
            "mlp_bad_last_dim_raises",
            test_mlp_bad_last_dim_raises,
        ),
        (
            "mlp_zero_weights_gives_zero",
            test_mlp_zero_weights_gives_zero,
        ),
        (
            "mlp_backward_matches_finite_difference",
            test_mlp_backward_matches_finite_difference,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
