from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.attention import (
    MultiHeadSelfAttention,
    _causal_mask,
)


def test_causal_mask_structure():
    m = _causal_mask(4)

    # Diagonal and past positions are allowed.
    assert m[0, 0] == 0
    assert m[3, 0] == 0

    # Future positions are masked.
    assert m[0, 1] < -1e6
    assert m[1, 3] < -1e6


def test_attention_output_shape():
    attn = MultiHeadSelfAttention.init(
        d_model=4,
        n_heads=1,
        seed=0,
    )

    x = np.random.randn(
        2,
        5,
        4,
    ).astype(np.float32)

    y = attn.forward(x)

    assert y.shape == x.shape


def test_attention_causal_property_sanity():
    """
    Changing a future token must not change outputs
    at earlier positions.
    """
    attn = MultiHeadSelfAttention.init(
        d_model=4,
        n_heads=1,
        seed=42,
    )

    x1 = np.random.randn(
        1,
        6,
        4,
    ).astype(np.float32)

    x2 = x1.copy()

    # Strongly perturb the final token.
    x2[:, 5, :] += 100.0

    y1 = attn.forward(x1)
    y2 = attn.forward(x2)

    assert np.allclose(
        y1[:, :5, :],
        y2[:, :5, :],
        atol=1e-4,
    )


def test_attention_bad_shape_raises():
    attn = MultiHeadSelfAttention.init(
        d_model=4,
        n_heads=1,
        seed=0,
    )

    bad = np.zeros(
        (3, 4),
        dtype=np.float32,
    )

    try:
        attn.forward(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_attention_rope_position_zero_all_heads():
    """
    RoPE at sequence position zero must be the identity
    for every attention head.

    This also verifies that RoPE is applied along the
    sequence axis rather than along the head axis.
    """
    attn = MultiHeadSelfAttention.init(
        d_model=8,
        n_heads=2,
        seed=123,
    )

    x = np.random.randn(
        1,
        4,
        8,
    ).astype(np.float32)

    attn.forward(x)

    assert attn._q is not None
    assert attn._k is not None

    assert attn._q_rope is not None
    assert attn._k_rope is not None

    # Position zero has angle zero:
    # cos(0)=1, sin(0)=0.
    assert np.allclose(
        attn._q[:, 0, :, :],
        attn._q_rope[:, 0, :, :],
        atol=1e-6,
    )

    assert np.allclose(
        attn._k[:, 0, :, :],
        attn._k_rope[:, 0, :, :],
        atol=1e-6,
    )


def test_attention_backward_matches_finite_difference():
    """
    Compare the manually implemented attention backward
    pass against a numerical gradient with respect to x.
    """
    attn = MultiHeadSelfAttention.init(
        d_model=4,
        n_heads=1,
        seed=0,
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

    attn.forward(x)

    analytic_dx = attn.backward(
        dout
    ).copy()

    numerical_dx = np.zeros_like(x)

    eps = 1e-6

    for index in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[index] += eps
        x_minus[index] -= eps

        y_plus = attn.forward(
            x_plus
        )

        loss_plus = np.sum(
            y_plus * dout
        )

        y_minus = attn.forward(
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
           "causal_mask_structure",
            test_causal_mask_structure,
        ),
        (
            "attention_output_shape",
            test_attention_output_shape,
        ),
        (
            "attention_causal_property_sanity",
            test_attention_causal_property_sanity,
        ),
        (
            "attention_bad_shape_raises",
            test_attention_bad_shape_raises,
        ),
        (
            "attention_rope_position_zero_all_heads",
            test_attention_rope_position_zero_all_heads,
        ),
        (
            "attention_backward_matches_finite_difference",
            test_attention_backward_matches_finite_difference,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run() 
