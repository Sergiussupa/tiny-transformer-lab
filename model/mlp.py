from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np, host_rng
from model.linear import Linear
from model.ops import silu


def _silu_backward(
    x: np.ndarray,
    dout: np.ndarray,
) -> np.ndarray:
    """
    y = silu(x) = x * sigmoid(x)

    dy/dx =
        sigmoid(x)
        + x * sigmoid(x) * (1 - sigmoid(x))
    """
    x = np.asarray(x)
    dout = np.asarray(dout)

    sig = 1.0 / (1.0 + np.exp(-x))

    dy = (
        sig
        + x * sig * (1.0 - sig)
    )

    return dout * dy


@dataclass
class MLP:
    d_model: int
    d_ff: int

    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear

    # Caches for backward.
    _x: np.ndarray | None = None
    _g: np.ndarray | None = None
    _u: np.ndarray | None = None
    _a: np.ndarray | None = None
    _h: np.ndarray | None = None

    @staticmethod
    def init(
        d_model: int,
        d_ff: int,
        *,
        seed: int = 0,
        dtype=np.float32,
    ) -> "MLP":

        rng = host_rng(seed)

        def init_lin(
            in_f: int,
            out_f: int,
        ) -> Linear:

            scale = 0.02

            W = np.asarray(
                (
                    rng.standard_normal(
                        (out_f, in_f)
                    ) * scale
                ).astype(dtype)
            )

            b = np.zeros(
                (out_f,),
                dtype=dtype,
            )

            return Linear(
                W=W,
                b=b,
            )

        return MLP(
            d_model=d_model,
            d_ff=d_ff,
            gate_proj=init_lin(
                d_model,
                d_ff,
            ),
            up_proj=init_lin(
                d_model,
                d_ff,
            ),
            down_proj=init_lin(
                d_ff,
                d_model,
            ),
        )

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        x:
            (..., d_model)

        output:
            (..., d_model)
        """
        x = np.asarray(x)

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, "
                f"got {x.shape[-1]}"
            )

        g = self.gate_proj.forward(x)
        u = self.up_proj.forward(x)

        a = silu(g)

        h = a * u

        out = self.down_proj.forward(h)

        self._x = x
        self._g = g
        self._u = u
        self._a = a
        self._h = h

        return out

    def backward(
        self,
        dout: np.ndarray,
    ) -> np.ndarray:
        """
        dout:
            (..., d_model)

        returns:
            dx with shape (..., d_model)
        """
        if (
            self._x is None
            or self._g is None
            or self._u is None
            or self._a is None
            or self._h is None
        ):
            raise RuntimeError(
                "MLP.backward called before forward"
            )

        dout = np.asarray(dout)

        # out = down_proj(h)
        dh = self.down_proj.backward(
            dout
        )

        # h = a * u
        da = dh * self._u
        du = dh * self._a

        # a = silu(g)
        dg = _silu_backward(
            self._g,
            da,
        )

        # g = gate_proj(x)
        # u = up_proj(x)
        dx_g = self.gate_proj.backward(
            dg
        )

        dx_u = self.up_proj.backward(
            du
        )

        # x feeds both branches.
        dx = dx_g + dx_u

        return dx

    def params(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        ps: list[
            tuple[np.ndarray, np.ndarray]
        ] = []

        ps.extend(
            self.gate_proj.params()
        )

        ps.extend(
            self.up_proj.params()
        )

        ps.extend(
            self.down_proj.params()
        )

        return ps
