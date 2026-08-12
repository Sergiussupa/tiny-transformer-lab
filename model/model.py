from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np
from model.config import TransformerConfig
from model.decoder_layer import DecoderLayer
from model.embedding import Embedding
from model.lm_head import LMHead
from model.ops import RMSNorm


@dataclass
class CausalLM:
    """
    Small decoder-only causal language model.

    Pipeline:

        token ids
            ->
        token embeddings
            ->
        DecoderLayer x N
            ->
        final RMSNorm
            ->
        LM Head
            ->
        vocabulary logits
    """

    config: TransformerConfig

    token_embeddings: Embedding
    layers: list[DecoderLayer]

    final_norm: RMSNorm
    lm_head: LMHead

    _cache_input_ids: np.ndarray | None = None
    _cache_hidden: np.ndarray | None = None

    @staticmethod
    def init(
        cfg: TransformerConfig,
        *,
        dtype=np.float32,
    ) -> "CausalLM":

        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        head_dim = (
            cfg.d_model // cfg.n_heads
        )

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE"
            )

        token_embeddings = Embedding.init(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            seed=cfg.seed,
            dtype=dtype,
        )

        layers: list[DecoderLayer] = []

        for i in range(cfg.n_layers):
            layer = DecoderLayer.init(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                rope_base=cfg.rope_base,
                seed=cfg.seed + 1000 * (i + 1),
                dtype=dtype,
            )

            layers.append(layer)

        final_norm = RMSNorm.init(
            cfg.d_model,
            dtype=dtype,
        )

        lm_head = LMHead.init(
            d_model=cfg.d_model,
            vocab_size=cfg.vocab_size,
            seed=cfg.seed + 9999,
            dtype=dtype,
        )

        return CausalLM(
            config=cfg,
            token_embeddings=token_embeddings,
            layers=layers,
            final_norm=final_norm,
            lm_head=lm_head,
        )

    def forward_hidden(
        self,
        input_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Convert token IDs into final hidden states.

        input_ids:
            (B, T)

        returns:
            (B, T, d_model)
        """

        x = self.token_embeddings.forward(
            input_ids
        )

        for layer in self.layers:
            x = layer.forward(x)

        x = self.final_norm.forward(x)

        return x

    def forward(
        self,
        input_ids: np.ndarray,
    ) -> np.ndarray:
        """
        input_ids:
            (B, T)

        returns:
            logits with shape (B, T, vocab_size)
        """

        self._cache_input_ids = np.asarray(
            input_ids
        )

        hidden = self.forward_hidden(
            input_ids
        )

        self._cache_hidden = hidden

        logits = self.lm_head.forward(
            hidden
        )

        return logits

    def backward(
        self,
        dlogits: np.ndarray,
    ) -> None:
        """
        Backpropagate manually through the complete model.

        Gradient path:

            LM Head
                ->
            final RMSNorm
                ->
            DecoderLayer x N in reverse order
                ->
            token embedding table
        """

        dh = self.lm_head.backward(
            dlogits
        )

        dh = self.final_norm.backward(
            dh
        )

        for layer in reversed(
            self.layers
        ):
            dh = layer.backward(dh)

        self.token_embeddings.backward(
            dh
        )

    def params(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return all trainable parameter/gradient pairs.
        """

        out: list[
            tuple[np.ndarray, np.ndarray]
        ] = []

        out.extend(
            self.token_embeddings.params()
        )

        for layer in self.layers:
            out.extend(
                layer.params()
            )

        out.extend(
            self.final_norm.params()
        )

        out.extend(
            self.lm_head.params()
        )

        return out
