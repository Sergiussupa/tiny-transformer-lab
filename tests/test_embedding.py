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


def test_embedding_1d_shape_and_exact_values() -> None:
    w = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [-1.0, -2.0, -3.0, -4.0],
        ],
        dtype=np.float32,
    )

    emb = Embedding(weight=w)

    ids = np.array(
        [2, 0, 1],
        dtype=np.int64,
    )

    out = emb.forward(ids)

    assert out.shape == (3, 4)
    assert np.allclose(out[0], w[2])
    assert np.allclose(out[1], w[0])
    assert np.allclose(out[2], w[1])


def test_embedding_2d_shape() -> None:
    emb = Embedding.init(
        vocab_size=8,
        d_model=4,
        seed=123,
    )

    ids = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )

    out = emb.forward(ids)

    assert out.shape == (2, 3, 4)


def test_embedding_bad_ndim_raises() -> None:
    emb = Embedding.init(
        vocab_size=8,
        d_model=4,
        seed=0,
    )

    bad = np.zeros(
        (1, 1, 1),
        dtype=np.int64,
    )

    try:
        emb.forward(bad)
        assert False, "Expected ValueError for ndim != 1 or 2"
    except ValueError:
        pass


def test_embedding_non_integer_ids_raises() -> None:
    emb = Embedding.init(
        vocab_size=8,
        d_model=4,
        seed=0,
    )

    bad = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    try:
        emb.forward(bad)
        assert False, "Expected TypeError for non-integer ids"
    except TypeError:
        pass


def test_embedding_out_of_range_raises() -> None:
    emb = Embedding.init(
        vocab_size=3,
        d_model=4,
        seed=0,
    )

    try:
        emb.forward(
            np.array([0, 3], dtype=np.int64)
        )
        assert False, "Expected IndexError for id >= vocab_size"
    except IndexError:
        pass

    try:
        emb.forward(
            np.array([-1, 1], dtype=np.int64)
        )
        assert False, "Expected IndexError for negative id"
    except IndexError:
        pass


def main() -> int:
    tests: List[tuple[str, Callable[[], None]]] = [
        (
            "embedding_1d_shape_and_exact_values",
            test_embedding_1d_shape_and_exact_values,
        ),
        (
            "embedding_2d_shape",
            test_embedding_2d_shape,
        ),
        (
            "embedding_bad_ndim_raises",
            test_embedding_bad_ndim_raises,
        ),
        (
            "embedding_non_integer_ids_raises",
            test_embedding_non_integer_ids_raises,
        ),
        (
            "embedding_out_of_range_raises",
            test_embedding_out_of_range_raises,
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
