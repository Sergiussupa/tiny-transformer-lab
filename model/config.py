from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import json


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int

    max_seq_len: int = 128
    rope_base: float = 10000.0
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def from_dict(
        d: Dict[str, Any],
    ) -> "TransformerConfig":
        return TransformerConfig(**d)

    @staticmethod
    def from_json(
        s: str,
    ) -> "TransformerConfig":
        return TransformerConfig.from_dict(
            json.loads(s)
        )
