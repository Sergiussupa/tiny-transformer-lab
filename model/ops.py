from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)

    # Softmax is invariant to a constant shift.
    # Subtracting the maximum prevents exp() overflow.
    x = x - np.max(x, axis=axis, keepdims=True)

    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


@dataclass
class RMSNorm:
    d_model: int
    eps: float
    weight: np.ndarray
    grad_weight: np.ndarray

    # Cache for backward.
    _x: np.ndarray | None = None
    _inv_rms: np.ndarray | None = None

    @staticmethod
    def init(
        d_model: int,
        *,
        eps: float = 1e-6,
        dtype=np.float32,
    ) -> "RMSNorm":
        w = np.ones((d_model,), dtype=dtype)
        gw = np.zeros_like(w)

        return RMSNorm(
            d_model=d_model,
            eps=eps,
            weight=w,
            grad_weight=gw,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (..., D)

        y = x * inv_rms(x) * weight
        """
        x = np.asarray(x)

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"RMSNorm.forward: expected last dim {self.d_model}, "
                f"got {x.shape[-1]}"
            )

        mean_sq = np.mean(
            x * x,
            axis=-1,
            keepdims=True,
        )

        inv_rms = 1.0 / np.sqrt(
            mean_sq + self.eps
        )

        y = x * inv_rms * self.weight

        self._x = x
        self._inv_rms = inv_rms

        return y.astype(
            x.dtype,
            copy=False,
        )

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        dy: (..., D)
        returns dx: (..., D)

        Let:

            r = 1 / sqrt(mean(x^2) + eps)
            z = x * r
            y = z * weight

        Then:

            dz = dy * weight

        and:

            dr/dx_i = -(r^3) * x_i / D

        which gives:

            dx_i =
                dz_i * r
                - (r^3 / D) * x_i * sum_j(dz_j * x_j)
        """
        if self._x is None or self._inv_rms is None:
            raise RuntimeError(
                "RMSNorm.backward called before forward"
            )

        x = self._x
        r = self._inv_rms
        dy = np.asarray(dy)

        if dy.shape != x.shape:
            raise ValueError(
                f"RMSNorm.backward: dy shape {dy.shape} "
                f"!= x shape {x.shape}"
            )

        # z = normalized input
        z = x * r

        # Gradient for the learnable scale.
        # Sum over all dimensions except the last one.
        axes = tuple(range(dy.ndim - 1))

        self.grad_weight[...] += np.sum(
            dy * z,
            axis=axes,
        )

        # Gradient through the learned scale.
        dz = dy * self.weight

        D = self.d_model

        s = np.sum(
            dz * x,
            axis=-1,
            keepdims=True,
        )

        dx = dz * r - (
            s * (r ** 3) * x
        ) / D

        return dx.astype(
            x.dtype,
            copy=False,
        )

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (self.weight, self.grad_weight),
        ]
