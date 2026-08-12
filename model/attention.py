from __future__ import annotations

from dataclasses import dataclass

from model.backend import xp as np, host_rng
from model.linear import Linear
from model.ops import softmax
from model.rope import rope_frequencies, apply_rope


def _causal_mask(T: int, *, dtype=np.float32) -> np.ndarray:
    """
    Causal mask of shape (T, T).

    0 where attention is allowed,
    a large negative value where looking into the future is forbidden.
    """
    m = np.triu(
        np.ones((T, T), dtype=dtype),
        k=1,
    )

    return np.where(
        m == 1,
        -1e9,
        0.0,
    ).astype(dtype)


def _softmax_backward(
    probs: np.ndarray,
    dprobs: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Compute dL/dx where probs = softmax(x).
    """
    s = np.sum(
        dprobs * probs,
        axis=axis,
        keepdims=True,
    )

    return probs * (dprobs - s)


def _apply_rope_inverse(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse RoPE rotation.

    x shape:
        (B, T, H, Hd)

    cos/sin:
        (T, Hd // 2)
    """
    x = np.asarray(x)

    Hd = x.shape[-1]

    if Hd % 2 != 0:
        raise ValueError(
            "RoPE inverse: last dim must be even"
        )

    x0 = x[..., 0::2]
    x1 = x[..., 1::2]

    cosb = cos[None, :, None, :]
    sinb = sin[None, :, None, :]

    y0 = x0 * cosb + x1 * sinb
    y1 = -x0 * sinb + x1 * cosb

    y = np.empty_like(x)

    y[..., 0::2] = y0
    y[..., 1::2] = y1

    return y


@dataclass
class MultiHeadSelfAttention:
    d_model: int
    n_heads: int

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear

    rope_base: float = 10000.0

    # Caches for backward.
    _x: np.ndarray | None = None
    _q: np.ndarray | None = None
    _k: np.ndarray | None = None
    _v: np.ndarray | None = None

    _q_rope: np.ndarray | None = None
    _k_rope: np.ndarray | None = None

    _vh: np.ndarray | None = None
    _probs: np.ndarray | None = None

    _Hd: int | None = None

    _cos: np.ndarray | None = None
    _sin: np.ndarray | None = None

    @staticmethod
    def init(
        d_model: int,
        n_heads: int,
        *,
        seed: int = 0,
        rope_base: float = 10000.0,
        dtype=np.float32,
    ) -> "MultiHeadSelfAttention":

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        head_dim = d_model // n_heads

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even (needed for RoPE)"
            )

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

        return MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            q_proj=init_lin(d_model, d_model),
            k_proj=init_lin(d_model, d_model),
            v_proj=init_lin(d_model, d_model),
            o_proj=init_lin(d_model, d_model),
            rope_base=rope_base,
        )

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        x:
            (B, T, D)

        output:
            (B, T, D)
        """
        x = np.asarray(x)

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape (B, T, D)"
            )

        B, T, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {D}"
            )

        H = self.n_heads
        Hd = D // H

        # Q, K, V projections.
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        # Split model dimension into attention heads.
        q = q.reshape(B, T, H, Hd)
        k = k.reshape(B, T, H, Hd)
        v = v.reshape(B, T, H, Hd)

        cos, sin = rope_frequencies(
            Hd,
            max_seq_len=T,
            base=self.rope_base,
            dtype=x.dtype,
        )

        # apply_rope expects (..., T, D).
        # Move heads before the sequence dimension:
        #
        # (B, T, H, Hd)
        #       ->
        # (B, H, T, Hd)
        qh = np.transpose(
            q,
            (0, 2, 1, 3),
        )

        kh = np.transpose(
            k,
            (0, 2, 1, 3),
        )

        qh = apply_rope(
            qh,
            cos,
            sin,
        )

        kh = apply_rope(
            kh,
            cos,
            sin,
        )

        # Keep a (B, T, H, Hd) representation for backward.
        q_rope = np.transpose(
            qh,
            (0, 2, 1, 3),
        )

        k_rope = np.transpose(
            kh,
            (0, 2, 1, 3),
        )

        vh = np.transpose(
            v,
            (0, 2, 1, 3),
        )

        # Scaled dot-product attention.
        scores = np.matmul(
            qh,
            np.transpose(
                kh,
                (0, 1, 3, 2),
            ),
        )

        scores = scores / np.sqrt(Hd).astype(
            x.dtype
        )

        scores = scores + _causal_mask(
            T,
            dtype=x.dtype,
        )[None, None, :, :]

        probs = softmax(
            scores,
            axis=-1,
        )

        out_h = np.matmul(
            probs,
            vh,
        )

        # Merge attention heads.
        out = np.transpose(
            out_h,
            (0, 2, 1, 3),
        ).reshape(
            B,
            T,
            D,
        )

        out = self.o_proj.forward(out)

        # Cache intermediate values for backward.
        self._x = x

        self._q = q
        self._k = k
        self._v = v

        self._q_rope = q_rope
        self._k_rope = k_rope

        self._vh = vh
        self._probs = probs

        self._Hd = Hd

        self._cos = cos
        self._sin = sin

        return out

    def backward(
        self,
        dout: np.ndarray,
    ) -> np.ndarray:
        """
        dout:
            (B, T, D)

        returns:
            dx with shape (B, T, D)
        """
        if (
            self._x is None
            or self._probs is None
            or self._vh is None
            or self._Hd is None
            or self._q_rope is None
            or self._k_rope is None
        ):
            raise RuntimeError(
                "Attention.backward called before forward"
            )

        x = self._x
        probs = self._probs
        vh = self._vh

        Hd = self._Hd

        cos = self._cos
        sin = self._sin

        assert cos is not None
        assert sin is not None

        dout = np.asarray(dout)

        if dout.shape != x.shape:
            raise ValueError(
                f"Attention.backward: dout shape {dout.shape} "
                f"!= x shape {x.shape}"
            )

        B, T, D = x.shape

        H = self.n_heads

        # 1. Backprop through output projection.
        d_out_before_o = self.o_proj.backward(
            dout
        )

        # (B, T, D)
        # ->
        # (B, H, T, Hd)
        d_out_h = d_out_before_o.reshape(
            B,
            T,
            H,
            Hd,
        ).transpose(
            0,
            2,
            1,
            3,
        )

        # out_h = probs @ vh
        dprobs = np.matmul(
            d_out_h,
            np.transpose(
                vh,
                (0, 1, 3, 2),
            ),
        )

        dvh = np.matmul(
            np.transpose(
                probs,
                (0, 1, 3, 2),
            ),
            d_out_h,
        )

        # 2. Backprop through softmax.
        dscores = _softmax_backward(
            probs,
            dprobs,
            axis=-1,
        )

        # scores = raw_scores / sqrt(Hd)
        draw = dscores / np.sqrt(Hd).astype(
            x.dtype
        )

        # raw_scores = Q @ K^T
        qh = np.transpose(
            self._q_rope,
            (0, 2, 1, 3),
        )

        kh = np.transpose(
            self._k_rope,
            (0, 2, 1, 3),
        )

        dq_h = np.matmul(
            draw,
            kh,
        )

        dk_h = np.matmul(
            np.transpose(
                draw,
                (0, 1, 3, 2),
            ),
            qh,
        )

        # Back to (B, T, H, Hd).
        dq_rope = np.transpose(
            dq_h,
            (0, 2, 1, 3),
        )

        dk_rope = np.transpose(
            dk_h,
            (0, 2, 1, 3),
        )

        # 3. Backprop through RoPE.
        dq = _apply_rope_inverse(
            dq_rope,
            cos,
            sin,
        )

        dk = _apply_rope_inverse(
            dk_rope,
            cos,
            sin,
        )

        # V path.
        dv = np.transpose(
            dvh,
            (0, 2, 1, 3),
        )

        # 4. Merge heads.
        dq = dq.reshape(B, T, D)
        dk = dk.reshape(B, T, D)
        dv = dv.reshape(B, T, D)

        # 5. Backprop through Q/K/V projections.
        dx_q = self.q_proj.backward(dq)
        dx_k = self.k_proj.backward(dk)
        dx_v = self.v_proj.backward(dv)

        # x feeds all three projections,
        # therefore the gradients are summed.
        dx = dx_q + dx_k + dx_v

        return dx

    def params(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        ps: list[
            tuple[np.ndarray, np.ndarray]
        ] = []

        ps.extend(
            self.q_proj.params()
        )

        ps.extend(
            self.k_proj.params()
        )

        ps.extend(
            self.v_proj.params()
        )

        ps.extend(
            self.o_proj.params()
        )

        return pso
