from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

import numpy as np

from model.config import TransformerConfig
from model.model import CausalLM


def _make_config(
    *,
    n_layers: int = 2,
) -> TransformerConfig:

    return TransformerConfig(
        vocab_size=11,
        d_model=4,
        n_layers=n_layers,
        n_heads=2,
        d_ff=8,
        max_seq_len=16,
        seed=42,
    )


def test_model_output_shape():
    cfg = _make_config(
        n_layers=2
    )

    model = CausalLM.init(cfg)

    input_ids = np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
        ],
        dtype=np.int64,
    )

    logits = model.forward(
        input_ids
    )

    assert logits.shape == (
        2,
        5,
        cfg.vocab_size,
    )


def test_model_hidden_shape():
    cfg = _make_config(
        n_layers=2
    )

    model = CausalLM.init(cfg)

    input_ids = np.array(
        [
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    hidden = model.forward_hidden(
        input_ids
    )

    assert hidden.shape == (
        1,
        3,
        cfg.d_model,
    )


def test_model_causal_property():
    """
    Changing the final token must not affect logits
    at earlier positions.
    """
    cfg = _make_config(
        n_layers=2
    )

    model = CausalLM.init(cfg)

    ids1 = np.array(
        [
            [1, 2, 3, 4],
        ],
        dtype=np.int64,
    )

    ids2 = ids1.copy()

    # Change only the final token.
    ids2[0, 3] = 8

    logits1 = model.forward(
        ids1
    )

    logits2 = model.forward(
        ids2
    )

    assert np.allclose(
        logits1[:, :3, :],
        logits2[:, :3, :],
        atol=1e-6,
    )


def test_model_backward_populates_gradients():
    cfg = _make_config(
        n_layers=2
    )

    model = CausalLM.init(cfg)

    input_ids = np.array(
        [
            [1, 2, 1, 3],
        ],
        dtype=np.int64,
    )

    logits = model.forward(
        input_ids
    )

    dlogits = np.ones_like(
        logits
    )

    model.backward(
        dlogits
    )

    assert (
        model.token_embeddings.grad_weight
        is not None
    )

    assert (
        model.token_embeddings.grad_weight.shape
        ==
        model.token_embeddings.weight.shape
    )

    params = model.params()

    assert len(params) > 0

    for param, grad in params:
        assert param.shape == grad.shape
        assert np.isfinite(grad).all()


def test_model_embedding_gradient_matches_finite_difference():
    """
    Verify one gradient through the complete model:

        logits
          ->
        LM Head
          ->
        final norm
          ->
        decoder layer
          ->
        embedding table

    The manual gradient for one embedding parameter is
    compared against a numerical finite difference.
    """

    cfg = TransformerConfig(
        vocab_size=7,
        d_model=4,
        n_layers=1,
        n_heads=2,
        d_ff=6,
        max_seq_len=8,
        seed=7,
    )

    model = CausalLM.init(
        cfg,
        dtype=np.float64,
    )

    input_ids = np.array(
        [
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    logits = model.forward(
        input_ids
    )

    dlogits = np.array(
        [
            [
                [
                    0.2,
                    -0.1,
                    0.3,
                    0.4,
                    -0.2,
                    0.1,
                    -0.3,
                ],
                [
                    -0.4,
                    0.2,
                    0.1,
                    -0.2,
                    0.3,
                    0.5,
                    -0.1,
                ],
                [
                    0.1,
                    0.3,
                    -0.5,
                    0.2,
                    -0.1,
                    0.4,
                    0.2,
                ],
            ]
        ],
        dtype=np.float64,
    )

    assert logits.shape == dlogits.shape

    model.backward(
        dlogits
    )

    assert (
        model.token_embeddings.grad_weight
        is not None
    )

    token_id = 2
    dim = 1

    analytic_grad = float(
        model.token_embeddings.grad_weight[
            token_id,
            dim,
        ]
    )

    eps = 1e-6

    original = float(
        model.token_embeddings.weight[
            token_id,
            dim,
        ]
    )

    model.token_embeddings.weight[
        token_id,
        dim,
    ] = original + eps

    logits_plus = model.forward(
        input_ids
    )

    loss_plus = np.sum(
        logits_plus * dlogits
    )

    model.token_embeddings.weight[
        token_id,
        dim,
    ] = original - eps

    logits_minus = model.forward(
        input_ids
    )

    loss_minus = np.sum(
        logits_minus * dlogits
    )

    model.token_embeddings.weight[
        token_id,
        dim,
    ] = original

    numerical_grad = float(
        (
            loss_plus
            - loss_minus
        )
        / (2.0 * eps)
    )

    assert np.allclose(
        analytic_grad,
        numerical_grad,
        atol=1e-5,
        rtol=1e-5,
    ), (
        analytic_grad,
        numerical_grad,
    )


def run():
    tests = [
        (
            "model_output_shape",
            test_model_output_shape,
        ),
        (
            "model_hidden_shape",
            test_model_hidden_shape,
        ),
        (
            "model_causal_property",
            test_model_causal_property,
        ),
        (
            "model_backward_populates_gradients",
            test_model_backward_populates_gradients,
        ),
        (
            "model_embedding_gradient_matches_finite_difference",
            test_model_embedding_gradient_matches_finite_difference,
        ),
    ]

    for name, fn in tests:
        fn()
        print(f"[OK]  {name}")


if __name__ == "__main__":
    run()
