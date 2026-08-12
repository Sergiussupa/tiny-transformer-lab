from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np, host_rng


@dataclass
class Embedding:
    """
    Embedding table:
    weight: (vocab_size, d_model)

    Compatibility aliases:
      W  <-> weight
      dW <-> grad_weight
    """

    weight: np.ndarray
    grad_weight: np.ndarray | None = None
    _last_ids: np.ndarray | None = None

    @staticmethod
    def init(
        vocab_size: int,
        d_model: int,
        *,
        seed: int = 0,
        dtype=np.float32,
        scale: float = 0.02,
    ) -> "Embedding":
        rng = host_rng(seed)
        w = np.asarray(
            (rng.standard_normal((vocab_size, d_model)) * scale).astype(dtype)
        )
        return Embedding(weight=w)

    # Compatibility aliases for existing code/checkpoints/tests.
    @property
    def W(self) -> np.ndarray:
        return self.weight

    @W.setter
    def W(self, value: np.ndarray) -> None:
        self.weight = value

    @property
    def dW(self) -> np.ndarray | None:
        return self.grad_weight

    @dW.setter
    def dW(self, value: np.ndarray | None) -> None:
        self.grad_weight = value

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.weight.shape[1])

    def forward(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)

        if ids.ndim not in (1, 2):
            raise ValueError("Embedding.forward: ids must have ndim 1 or 2")

        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("Embedding.forward: ids must be integer type")

        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise IndexError("Embedding.forward: token id out of range")

        self._last_ids = ids
        return self.weight[ids]

    def zero_grad(self) -> None:
        self.grad_weight = np.zeros_like(self.weight)

    def backward(self, d_out: np.ndarray) -> None:
        """
        d_out:
          (T, D) if ids were (T,)
          (B, T, D) if ids were (B, T)

        Gradients are accumulated by embedding row, including repeated IDs.
        """
        if self._last_ids is None:
            raise RuntimeError("Embedding.backward: call forward() first")

        ids = self._last_ids
        d_out = np.asarray(d_out)

        expected = ids.shape + (self.d_model,)
        if d_out.shape != expected:
            raise ValueError(
                f"Embedding.backward: expected d_out shape {expected}, "
                f"got {d_out.shape}"
            )

        if self.grad_weight is None:
            self.zero_grad()

        flat_ids = ids.reshape(-1)
        flat_grad = d_out.reshape(-1, self.d_model)

        np.add.at(self.grad_weight, flat_ids, flat_grad)

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (parameter, gradient) pairs for the optimizer."""
        if self.grad_weight is None:
            self.zero_grad()

        return [(self.weight, self.grad_weight)]

    def clear_cache(self) -> None:
        self._last_ids = None
