# loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from model.backend import xp as np, scalar


@dataclass
class CrossEntropyOut:
    loss: float
    dlogits: np.ndarray  # same shape as logits


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if keepdims:
        return y
    return np.squeeze(y, axis=axis)


def causal_lm_cross_entropy(
    logits: np.ndarray,
    target_ids: np.ndarray,
    *,
    ignore_index: int = -100,
    mask: Optional[np.ndarray] = None,
    reduction: str = "mean",
) -> CrossEntropyOut:
    """
    logits: (B, T, V) or (T, V)
    target_ids: (B, T) or (T,)
    mask: optional (B, T) or (T,) {0/1} or bool
    Returns:
      loss scalar + dlogits of same shape as logits

    NOTE: Эта функция НЕ делает shift. Предполагается, что target_ids уже сдвинуты правильно
          (например, data.py уже делает teacher forcing target).
    """
    logits = np.asarray(logits)
    target_ids = np.asarray(target_ids)

    if logits.ndim == 2:
        logits_3 = logits[None, ...]  # (1, T, V)
    elif logits.ndim == 3:
        logits_3 = logits
    else:
        raise ValueError(f"logits must be 2D or 3D, got shape {logits.shape}")

    if target_ids.ndim == 1:
        tgt_2 = target_ids[None, ...]  # (1, T)
    elif target_ids.ndim == 2:
        tgt_2 = target_ids
    else:
        raise ValueError(f"target_ids must be 1D or 2D, got shape {target_ids.shape}")

    B, T, V = logits_3.shape
    if tgt_2.shape != (B, T):
        raise ValueError(f"target_ids shape {tgt_2.shape} must match (B,T)=({B},{T})")

    if mask is None:
        m = np.ones((B, T), dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.ndim == 1:
            m = mask[None, ...].astype(bool)
        elif mask.ndim == 2:
            m = mask.astype(bool)
        else:
            raise ValueError("mask must be 1D or 2D")
        if m.shape != (B, T):
            raise ValueError(f"mask shape {m.shape} must match (B,T)=({B},{T})")

    valid = m & (tgt_2 != ignore_index)
    n_valid = int(scalar(np.sum(valid)))
    if n_valid == 0:
        out = np.zeros_like(logits_3)
        if logits.ndim == 2:
            out = out[0]
        return CrossEntropyOut(loss=0.0, dlogits=out)

    # log-probs
    lse = _logsumexp(logits_3, axis=-1, keepdims=True)  # (B,T,1)
    log_probs = logits_3 - lse  # (B,T,V)

    # loss: -log p(target)
    # gather
    flat_log_probs = log_probs.reshape(B * T, V)
    flat_tgt = tgt_2.reshape(B * T)
    flat_valid = valid.reshape(B * T)

    idx = np.where(flat_valid)[0]
    chosen = flat_log_probs[idx, flat_tgt[idx]]
    loss_sum = -float(scalar(np.sum(chosen)))

    if reduction == "mean":
        loss = loss_sum / n_valid
        scale = 1.0 / n_valid
    elif reduction == "sum":
        loss = loss_sum
        scale = 1.0
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # gradient: softmax - onehot
    # softmax = exp(log_probs)
    probs = np.exp(log_probs)  # (B,T,V)
    dlogits = probs.copy()

    # subtract 1 at target positions (only valid)
    # do it in flat space for simplicity
    flat_d = dlogits.reshape(B * T, V)
    flat_d[idx, flat_tgt[idx]] -= 1.0

    # zero-out invalid positions
    flat_d[~flat_valid, :] = 0.0

    # scale (mean reduction)
    flat_d *= scale

    out = flat_d.reshape(B, T, V)
    if logits.ndim == 2:
        out = out[0]
    return CrossEntropyOut(loss=loss, dlogits=out)
