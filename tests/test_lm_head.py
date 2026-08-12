from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.lm_head import LMHead


def test_lm_head_output_shape_3d():
    B, T, D, V = 2, 5, 4, 9

    head = LMHead.init(
        d_model=D,
        vocab_size=V,
        seed=0,
    )

    x = np.zeros(
        (B, T, D),
        dtype=np.float32,
    )

    y = head.forward(x)

    assert y.shape == (
        B,
        T,
        V,
    )


def test_lm_head_output_shape_2d_and_1d():
    T, D, V = 7, 4, 11

    head = LMHead.init(
        d_model=D,
        vocab_size=V,
        seed=1,
    )

    x2 = np.zeros(
        (T, D),
        dtype=np.float32,
    )

    y2 = head.forward(x2)

    assert y2.shape == (
        T,
        V,
    )

    x1 = np.zeros(
        (D,),
        dtype=np.float32,
    )

    y1 = head.forward(x1)

    assert y1.shape == (
        V,
    )


def test_lm_head_bad_last_dim_raises():
    head = LMHead.init(
        d_model=4,
        vocab_size=8,
        seed=0,
    )

    x = np.zeros(
        (2, 3, 5),
        dtype=np.float32,
    )

    try:
        head.forward(x)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_lm_head_zero_weights_gives_zero_logits():
    B, T, D, V = 2, 3, 4, 7

    head = LMHead.init(
        d_model=D,
        vocab_size=V,
        seed=0,
    )

    head.proj.W[...] = 0.0

    if head.proj.b is not None:
        head.proj.b[...] = 0.0

    x = np.random.default_rng(
        0
    ).standard_normal(
        (B, T, D)
    ).astype(np.float32)

    y = head.forward(x)

    assert y.shape == (
        B,
        T,
        V,
    )

    assert np.allclose(
        y,
        0.0,
    )


def test_lm_head_backward_shapes():
    head = LMHead.init(
        d_model=4,
        vocab_size=7,
        seed=0,
    )

    x = np.zeros(
        (2, 3, 4),
        dtype=np.float32,
    )

    logits = head.forward(x)

    dlogits = np.ones_like(
        logits,
        dtype=np.float32,
    )

    dx = head.backward(
        dlogits
    )

    assert dx.shape == x.shape

    params = head.params()

    assert len(params) == 2

    (W, dW), (b, db) = params

    assert dW.shape == W.shape
    assert db.shape == b.shape


def run():
    tests = [
        (
            "lm_head_output_shape_3d",
            test_lm_head_output_shape_3d,
        ),
        (
            "lm_head_output_shape_2d_and_1d",
            test_lm_head_output_shape_2d_and_1d,
        ),
        (
            "lm_head_bad_last_dim_raises",
            test_lm_head_bad_last_dim_raises,
        ),
        (
            "lm_head_zero_weights_gives_zero_logits",
            test_lm_head_zero_weights_gives_zero_logits,
        ),
        (
            "lm_head_backward_shapes",
            test_lm_head_backward_shapes,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
