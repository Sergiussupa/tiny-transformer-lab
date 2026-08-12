from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np, host_rng


@dataclass
class Linear:
    """
    Linear layer: y = x @ W^T + b

    W: (out_features, in_features)
    b: (out_features,) or None

    Supports x with shapes:
      - (in_features,)
      - (T, in_features)
      - (B, T, in_features)

    backward accepts dY with the same shape as y and returns dX.
    """

    W: np.ndarray
    b: np.ndarray | None = None

    # Gradients
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

    # Cache for backward
    _x_cache: np.ndarray | None = None

    @staticmethod
    def init(
        in_features: int,
        out_features: int,
        *,
        seed: int = 0,
        dtype=np.float32,
        bias: bool = True,
        scale: float = 0.02,
    ) -> "Linear":
        rng = host_rng(seed)

        W = np.asarray(
            (rng.standard_normal((out_features, in_features)) * scale).astype(dtype)
        )

        b = np.zeros((out_features,), dtype=dtype) if bias else None

        lin = Linear(W=W, b=b)
        lin.zero_grad()
        return lin

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if x.shape[-1] != self.W.shape[1]:
            raise ValueError(
                f"Linear.forward: expected last dim {self.W.shape[1]}, "
                f"got {x.shape[-1]}"
            )

        self._x_cache = x

        y = np.matmul(x, self.W.T)

        if self.b is not None:
            y = y + self.b

        return y

    def zero_grad(self) -> None:
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if self.b is not None else None

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("Linear.backward called before forward")

        x = self._x_cache
        dy = np.asarray(dy)

        # x:  (..., in)
        # dy: (..., out)
        if dy.shape[-1] != self.W.shape[0]:
            raise ValueError(
                f"Linear.backward: expected last dim {self.W.shape[0]}, "
                f"got {dy.shape[-1]}"
            )

        # dX = dY @ W
        dx = np.matmul(dy, self.W)

        # Collapse batch/time dimensions:
        # dW = dY^T @ X -> (out, in)
        x2 = x.reshape(-1, self.W.shape[1])
        dy2 = dy.reshape(-1, self.W.shape[0])

        if self.dW is None:
            self.dW = np.zeros_like(self.W)

        self.dW += dy2.T @ x2

        # db = sum dY over batch/time dimensions
        if self.b is not None:
            if self.db is None:
                self.db = np.zeros_like(self.b)

            self.db += np.sum(dy2, axis=0)

        return dx

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return (parameter, gradient) pairs for the optimizer.
        """
        if self.dW is None:
            self.dW = np.zeros_like(self.W)

        out: list[tuple[np.ndarray, np.ndarray]] = [
            (self.W, self.dW),
        ]

        if self.b is not None:
            if self.db is None:
                self.db = np.zeros_like(self.b)

            out.append((self.b, self.db))

        return out
