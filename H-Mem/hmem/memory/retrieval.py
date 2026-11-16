"""Hierarchical, index-based memory retrieval (Section 3.2).

Retrieval is a *top-down traversal*: the query is embedded, compared with the
semantic vectors of the coarsest layer, and the selected position indices route
the search into the next layer. This replaces the flat O(N·D) scan of
traditional vector memory with an O((a + k·300)·D) routing (see the paper's
computational-efficiency analysis).
"""

from __future__ import annotations

import numpy as np

from ..config import HMemConfig
from ..schemas import LAYER_ORDER, MemoryLayer, RetrievalResult
from ..utils import get_logger
from .store import HierarchicalMemoryStore

log = get_logger("retrieval")


class HierarchicalRetriever:
    """Executes layer-by-layer top-k routing over a :class:`HierarchicalMemoryStore`."""

    def __init__(self, store: HierarchicalMemoryStore, config: HMemConfig):
        self.store = store
        self.config = config

    # ------------------------------------------------------------------ #
    def retrieve(self, query_vector: np.ndarray) -> tuple[list[RetrievalResult], int]:
        """Retrieve the top episodes for a query vector.

        Returns ``(results, n_compared)`` where ``n_compared`` is the total
        number of candidate vectors scored (efficiency bookkeeping).
        """
        mode = self.config.retrieval_mode
        if mode == "none":
            return [], 0
        if mode == "flat":
            return self._retrieve_flat(query_vector)
        if mode == "hierarchical":
            return self._retrieve_hierarchical(query_vector)
        raise ValueError(f"Unknown retrieval_mode {mode!r}")

    # ------------------------------------------------------------------ #
    def _retrieve_hierarchical(
        self, query_vector: np.ndarray
    ) -> tuple[list[RetrievalResult], int]:
        store = self.store
        total_compared = 0
        selected: list[int] = []

        # Level 1 -- Domain: the only *global* scan.
        domains = store.nodes_of_layer(MemoryLayer.DOMAIN)
        if not domains:
            return [], 0
        scores, ids, n = store.engine.search(
            MemoryLayer.DOMAIN.level, query_vector, self.config.top_k
        )
        total_compared += n
        current = [int(i) for i in ids]

        # Levels 2..L-1 -- route through the position indices.
        for layer in LAYER_ORDER[1:-1]:  # category, trace
            candidates: list[int] = []
            for parent in current:
                candidates.extend(store.children_of(parent))
            if not candidates:
                return [], total_compared
            scores, ids, n = store.engine.search(
                layer.level, query_vector, self.config.top_k, candidate_ids=candidates
            )
            total_compared += n
            current = [int(i) for i in ids]

        # Level L -- Episode: gather children of selected traces, then top-k.
        candidates = []
        for parent in current:
            candidates.extend(store.children_of(parent))
        if not candidates:
            return [], total_compared
        scores, ids, n = store.engine.search(
            MemoryLayer.EPISODE.level,
            query_vector,
            self.config.final_top_k,
            candidate_ids=candidates,
        )
        total_compared += n

        selected = [int(i) for i in ids]
        return self._to_results(selected, scores), total_compared

    # ------------------------------------------------------------------ #
    def _retrieve_flat(self, query_vector: np.ndarray) -> tuple[list[RetrievalResult], int]:
        """Brute-force over every Episode node (MemoryBank-style ablation)."""
        scores, ids, n = self.store.engine.search(
            MemoryLayer.EPISODE.level, query_vector, self.config.final_top_k
        )
        return self._to_results([int(i) for i in ids], scores), n

    # ------------------------------------------------------------------ #
    def _to_results(self, positions: list[int], scores: np.ndarray) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for idx, pos in enumerate(positions):
            node = self.store.get(pos)
            if node is None:
                continue
            score = float(scores[idx])
            results.append(RetrievalResult(node=node, score=score, weight=node.weight))
        return results
