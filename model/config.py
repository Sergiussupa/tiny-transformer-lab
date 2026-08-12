from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import json


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int = 128
    rope_base: float = 10000.0
    seed: int = 0
    tie_embeddings: bool = False  # optional

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LlamaConfig":
        return LlamaConfig(**d)

    @staticmethod
    def from_json(s: str) -> "LlamaConfig":
        return LlamaConfig.from_dict(json.loads(s))

