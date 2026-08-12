from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np
from model.ops import RMSNorm
from model.attention import MultiHeadSelfAttention
from model.mlp import MLP


@dataclass
class DecoderLayer:
    d_model: int

    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    attn: MultiHeadSelfAttention
    mlp: MLP

    # Caches for backward.
    _x_in: np.ndarray | None = None
    _x_after_attn_res: np.ndarray | None = None

    @staticmethod
    def init(
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        rope_base: float = 10000.0,
        seed: int = 0,
        dtype=np.float32,
    ) -> "DecoderLayer":

        input_ln = RMSNorm.init(
            d_model,
            dtype=dtype,
        )

        post_attn_ln = RMSNorm.init(
            d_model,
            dtype=dtype,
        )

        attn = MultiHeadSelfAttention.init(
            d_model=d_model,
            n_heads=n_heads,
            rope_base=rope_base,
            seed=seed,
            dtype=dtype,
        )

        mlp = MLP.init(
            d_model=d_model,
            d_ff=d_ff,
            seed=seed + 1,
            dtype=dtype,
        )

        return DecoderLayer(
            d_model=d_model,
            input_layernorm=input_ln,
            post_attention_layernorm=post_attn_ln,
            attn=attn,
            mlp=mlp,
        )

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        x:
            (B, T, d_model)

        returns:
            (B, T, d_model)
        """
        x = np.asarray(x)

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, "
                f"got {x.shape[-1]}"
            )

        self._x_in = x

        # Attention block: Pre-Norm + residual.
        h = self.input_layernorm.forward(x)
        h = self.attn.forward(h)

        x = x + h

        self._x_after_attn_res = x

        # MLP block: Pre-Norm + residual.
        h = self.post_attention_layernorm.forward(x)
        h = self.mlp.forward(h)

        x = x + h

        return x

    def backward(
        self,
        dx: np.ndarray,
    ) -> np.ndarray:
        """
        dx:
            gradient with respect to layer output

        returns:
            gradient with respect to layer input
        """
        if (
            self._x_in is None
            or self._x_after_attn_res is None
        ):
            raise RuntimeError(
                "DecoderLayer.backward called before forward"
            )

        dx = np.asarray(dx)

        # --------------------------------------------------
        # Second residual:
        #
        # x_out = x1 + mlp(norm2(x1))
        # --------------------------------------------------

        # Direct residual path.
        d_x1 = dx

        # MLP branch.
        d_mlp_out = dx

        d_post_ln_out = self.mlp.backward(
            d_mlp_out
        )

        d_x1_from_post_ln = (
            self.post_attention_layernorm.backward(
                d_post_ln_out
            )
        )

        # Sum both paths into x1.
        d_x1 = (
            d_x1
            + d_x1_from_post_ln
        )

        # --------------------------------------------------
        # First residual:
        #
        # x1 = x0 + attention(norm1(x0))
        # --------------------------------------------------

        # Direct residual path.
        d_x0 = d_x1

        # Attention branch.
        d_attn_out = d_x1

        d_pre_ln_out = self.attn.backward(
            d_attn_out
        )

        d_x0_from_ln = (
            self.input_layernorm.backward(
                d_pre_ln_out
            )
        )

        # Sum both paths into x0.
        d_x0 = (
            d_x0
            + d_x0_from_ln
        )

        return d_x0

    def params(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        ps: list[
            tuple[np.ndarray, np.ndarray]
        ] = []

        ps.extend(
           self.input_layernorm.params()
        )

        ps.extend(
            self.post_attention_layernorm.params()
        )

        ps.extend(
            self.attn.params()
        )

        ps.extend(
            self.mlp.params()
        )

        return ps 
