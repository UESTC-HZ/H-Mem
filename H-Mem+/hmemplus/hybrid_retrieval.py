"""Dense and lexical episode retrieval and cross-encoder reranking."""
from __future__ import annotations

import collections
import math
from typing import Any

DEFAULT_DENSE = 50
DEFAULT_LEXICAL = 50


def _normalise(text: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in (text or "")).split())


class HybridIndex:
    """Dense + lexical candidate generation over the stored episodes."""

    def __init__(self, nodes: list[dict], vectors: dict[int, Any]):
        import numpy as np

        self.nodes = [n for n in nodes if n.get("pos") in vectors]
        self.positions = [n["pos"] for n in self.nodes]
        self.texts = [n.get("text") or "" for n in self.nodes]
        self.tokens = [set(_normalise(t).split()) for t in self.texts]
        self.matrix = (np.asarray([vectors[p] for p in self.positions], dtype=float)
                       if self.nodes else np.zeros((0, 1)))
        df = collections.Counter()
        for tokens in self.tokens:
            df.update(tokens)
        n = max(len(self.texts), 1)
        self.idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def __len__(self) -> int:
        return len(self.nodes)

    def candidates(self, question: str, query_vector, k_dense: int = DEFAULT_DENSE,
                   k_lexical: int = DEFAULT_LEXICAL) -> list[int]:
        """Return indices into `self.nodes` worth reranking."""
        import numpy as np

        if not self.nodes:
            return []
        picked: set[int] = set()
        if query_vector is not None and self.matrix.size:
            picked.update(np.argsort(-(self.matrix @ query_vector), kind="stable")[:k_dense].tolist())
        q_tokens = set(_normalise(question).split())
        if q_tokens:
            lexical = [sum(self.idf[t] for t in sorted(tokens & q_tokens))
                       for tokens in self.tokens]
            picked.update(sorted(range(len(self.nodes)), key=lambda i: -lexical[i])[:k_lexical])
        return sorted(picked)

    def passage(self, index: int) -> str:
        """The text handed to the reranker: the summary plus its verbatim provenance."""
        node = self.nodes[index]
        metadata = node.get("metadata") or {}
        parts = [self.texts[index]]
        for turn in metadata.get("source_turns") or []:
            text = (turn or {}).get("text") or ""
            if text:
                parts.append(text)
        return " ".join(parts)


class CrossEncoderReranker:
    """Lazily loaded cross encoder; the model is shared across calls."""

    _model = None
    _name = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def _get(self):
        if CrossEncoderReranker._model is None or CrossEncoderReranker._name != self.model_name:
            from sentence_transformers import CrossEncoder

            CrossEncoderReranker._model = CrossEncoder(self.model_name)
            CrossEncoderReranker._name = self.model_name
        return CrossEncoderReranker._model

    def rerank(self, question: str, indices: list[int], passages: list[str],
               top_k: int) -> list[tuple[int, float]]:
        if not indices:
            return []
        model = self._get()
        scores = model.predict([(question, p) for p in passages])
        ranked = sorted(zip(indices, (float(s) for s in scores)), key=lambda x: -x[1])
        return ranked[:top_k]
