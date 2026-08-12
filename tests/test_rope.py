from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.rope import rope_frequencies, apply_rope


def test_rope_identity_at_pos0():
    # Position 0 => cos=1, sin=0 => identity.
    D = 4
    cos, sin = rope_frequencies(D, max_seq_len=4)

    x = np.array(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=np.float32,
    )

    y = apply_rope(x, cos, sin)

    assert np.allclose(x, y), (x, y)


def test_rope_norm_preserved():
    # Rotation preserves the norm of each pair.
    D = 4
    cos, sin = rope_frequencies(D, max_seq_len=10)

    x = np.random.randn(5, D).astype(np.float32)
    y = apply_rope(x, cos, sin)

    x_pairs = x.reshape(5, 2, 2)
    y_pairs = y.reshape(5, 2, 2)

    x_norm = np.sum(x_pairs**2, axis=-1)
    y_norm = np.sum(y_pairs**2, axis=-1)

    assert np.allclose(x_norm, y_norm, atol=1e-5)


def test_rope_shape_3d():
    B, T, D = 2, 3, 4

    cos, sin = rope_frequencies(
        D,
        max_seq_len=10,
    )

    x = np.random.randn(
        B,
        T,
        D,
    ).astype(np.float32)

    y = apply_rope(x, cos, sin)

    assert y.shape == (B, T, D)


def test_rope_bad_dim_raises():
    cos, sin = rope_frequencies(
        4,
        max_seq_len=4,
    )

    bad = np.zeros(
        (2, 3),
        dtype=np.float32,
    )

    try:
        apply_rope(bad, cos, sin)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def run():
    tests = [
        (
            "rope_identity_at_pos0",
            test_rope_identity_at_pos0,
        ),
        (
            "rope_norm_preserved",
            test_rope_norm_preserved,
        ),
        (
            "rope_shape_3d",
            test_rope_shape_3d,
        ),
        (
            "rope_bad_dim_raises",
            test_rope_bad_dim_raises,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
