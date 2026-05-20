from dataclasses import dataclass, field
from typing import List


@dataclass
class EngramConfig:
    max_ngram_size: int = 2
    n_embed_per_ngram: int = 1024
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [2, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    hidden_size: int = 2560
    engram_vocab_size: List[int] = field(default_factory=lambda: [151936 * 5, 151936 * 5])
