from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.loss import causal_lm_cross_entropy


def test_no_valid_tokens_preserves_2d_shape():
    logits = np.zeros((3, 5), dtype=np.float32)
    targets = np.full((3,), -100, dtype=np.int64)

    out = causal_lm_cross_entropy(
        logits,
        targets,
        ignore_index=-100,
    )

    assert out.loss == 0.0
    assert out.dlogits.shape == logits.shape
    assert np.all(out.dlogits == 0.0)


def test_loss_gradient_matches_finite_difference():
    logits = np.array(
        [[[0.2, -0.4, 0.7],
          [0.5, 0.1, -0.3]]],
        dtype=np.float64,
    )

    targets = np.array([[2, 0]], dtype=np.int64)

    out = causal_lm_cross_entropy(
        logits,
        targets,
        reduction="mean",
    )

    analytic = out.dlogits.copy()
    numerical = np.zeros_like(logits)
    eps = 1e-6

    for index in np.ndindex(logits.shape):
        plus = logits.copy()
        minus = logits.copy()

        plus[index] += eps
        minus[index] -= eps

        loss_plus = causal_lm_cross_entropy(
            plus,
            targets,
            reduction="mean",
        ).loss

        loss_minus = causal_lm_cross_entropy(
            minus,
            targets,
            reduction="mean",
        ).loss

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2.0 * eps)

    assert np.allclose(
        analytic,
        numerical,
        atol=1e-6,
        rtol=1e-6,
    ), (analytic, numerical)
