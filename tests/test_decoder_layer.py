from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.decoder_layer import DecoderLayer


def test_decoder_layer_shape():
    layer = DecoderLayer.init(
        d_model=4,
        n_heads=2,
        d_ff=8,
        seed=0,
    )

    x = np.random.randn(
        2,
        5,
        4,
    ).astype(np.float32)

    y = layer.forward(x)

    assert y.shape == (
        2,
        5,
        4,
    )


def test_decoder_layer_zero_branches_preserve_input():
    """
    If both Attention and MLP branches produce zero,
    the residual connections must pass x through unchanged.
    """
    layer = DecoderLayer.init(
        d_model=4,
        n_heads=2,
        d_ff=8,
        seed=0,
    )

    # Zero all Attention projections.
    for proj in (
        layer.attn.q_proj,
        layer.attn.k_proj,
        layer.attn.v_proj,
        layer.attn.o_proj,
    ):
        proj.W[...] = 0.0

        if proj.b is not None:
            proj.b[...] = 0.0

    # Zero all MLP projections.
    for proj in (
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ):
        proj.W[...] = 0.0

        if proj.b is not None:
            proj.b[...] = 0.0

    x = np.random.randn(
        1,
        6,
        4,
    ).astype(np.float32)

    y = layer.forward(x)

    assert np.allclose(
        y,
        x,
        atol=1e-6,
    )


def test_decoder_layer_bad_dim_raises():
    layer = DecoderLayer.init(
        d_model=4,
        n_heads=2,
        d_ff=8,
        seed=0,
    )

    bad = np.zeros(
        (1, 5, 3),
        dtype=np.float32,
    )

    try:
        layer.forward(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_decoder_layer_backward_matches_finite_difference():
    """
    Compare the full manual DecoderLayer backward pass
    against a numerical gradient with respect to x.
    """
    layer = DecoderLayer.init(
        d_model=4,
        n_heads=2,
        d_ff=6,
        seed=42,
        dtype=np.float64,
    )

    x = np.array(
        [
            [
                [0.2, -0.3, 0.5, 0.7],
                [0.1, 0.4, -0.6, 0.2],
                [-0.5, 0.8, 0.3, -0.2],
            ]
        ],
        dtype=np.float64,
    )

    dout = np.array(
        [
            [
                [0.4, -0.2, 0.1, 0.3],
                [-0.1, 0.5, 0.2, -0.4],
                [0.3, 0.2, -0.5, 0.1],
            ]
        ],
        dtype=np.float64,
    )

    layer.forward(x)

    analytic_dx = layer.backward(
        dout
    ).copy()

    numerical_dx = np.zeros_like(x)

    eps = 1e-6

    for index in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[index] += eps
        x_minus[index] -= eps

        y_plus = layer.forward(
            x_plus
        )

        loss_plus = np.sum(
            y_plus * dout
        )

        y_minus = layer.forward(
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
            "decoder_layer_shape",
            test_decoder_layer_shape,
        ),
        (
            "decoder_layer_zero_branches_preserve_input",
            test_decoder_layer_zero_branches_preserve_input,
        ),
        (
            "decoder_layer_bad_dim_raises",
            test_decoder_layer_bad_dim_raises,
        ),
        (
            "decoder_layer_backward_matches_finite_difference",
            test_decoder_layer_backward_matches_finite_difference,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
