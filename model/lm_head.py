from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np
from model.linear import Linear


@dataclass
class LMHead:
    """
    Project hidden states into vocabulary logits.

    Input:
        (..., d_model)

    Output:
        (..., vocab_size)
    """

    d_model: int
    vocab_size: int
    proj: Linear

    @staticmethod
    def init(
        d_model: int,
        vocab_size: int,
        *,
        seed: int = 0,
        dtype=np.float32,
    ) -> "LMHead":

        proj = Linear.init(
            d_model,
            vocab_size,
            seed=seed,
            dtype=dtype,
        )

        return LMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            proj=proj,
        )

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        x = np.asarray(x)

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"LMHead.forward: expected last dim "
                f"{self.d_model}, got {x.shape[-1]}"
            )

        return self.proj.forward(x)

    def backward(
        self,
        dlogits: np.ndarray,
    ) -> np.ndarray:

        return self.proj.backward(
            dlogits
        )

    def params(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        return self.proj.params()
