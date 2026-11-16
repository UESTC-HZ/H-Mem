"""Vector-similarity engine used for memory retrieval.

The engine implements the *index-based routing* primitive of H-MEM
(Section 3.2). Two search shapes matter:

* **Global scan**  -- search *all* units of a layer. This happens at the
  coarsest (Domain) layer and for the flat MemoryBank-style ablation. When
  FAISS is available it runs on a persistent ``IndexFlatIP`` (inner product
  over L2-normalized vectors == cosine similarity).

* **Restricted scan** -- search only a small *candidate set* (the children of
  previously selected parents). This is the core efficiency win: H-MEM never
  compares the query against the full memory, only against the few hundred
  sub-memories reachable through the position indices.

Both shapes are exact; only the candidate sets differ. Compute cost is tracked
so the efficiency experiment can report FLOPs / latency fairly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import get_logger, l2_normalize, top_k_indices

log = get_logger("index")


class SimilarityEngine:
    """Cosine-similarity search over memory vectors, keyed by position index."""

    def __init__(self, dim: int, backend: str = "auto"):
        self.dim = int(dim)
        self.backend = self._resolve_backend(backend)
        # Source of truth: layer -> {position_index: normalized vector}.
        self._vectors: dict[int, dict[int, np.ndarray]] = {}
        self._faiss: dict[int, Any] = {}
        self._dirty: dict[int, bool] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "auto":
            try:
                import faiss  # noqa: F401

                return "faiss"
            except Exception:
                return "numpy"
        return backend

    @staticmethod
    def faiss_available() -> bool:
        try:
            import faiss  # noqa: F401

            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def add(self, layer: int, pos: int, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vec.shape[0]} != engine dim {self.dim}")
        self._vectors.setdefault(layer, {})[int(pos)] = l2_normalize(vec.reshape(1, -1))[0]
        self._dirty[layer] = True

    def remove(self, layer: int, pos: int) -> None:
        bucket = self._vectors.get(layer)
        if bucket is not None:
            bucket.pop(int(pos), None)
            self._dirty[layer] = True

    def reset(self, layer: int) -> None:
        self._vectors.pop(layer, None)
        self._faiss.pop(layer, None)
        self._dirty.pop(layer, None)

    def clear(self) -> None:
        """Remove every vector and index (used by ``store.clear``)."""
        self._vectors.clear()
        self._faiss.clear()
        self._dirty.clear()

    def vector(self, layer: int, pos: int) -> np.ndarray | None:
        """Return the stored (normalized) vector for ``pos``, if present."""
        bucket = self._vectors.get(layer)
        if bucket is None:
            return None
        return bucket.get(int(pos))

    def count(self, layer: int) -> int:
        return len(self._vectors.get(layer, {}))

    def total_count(self) -> int:
        return sum(len(bucket) for bucket in self._vectors.values())

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        layer: int,
        query: np.ndarray,
        k: int,
        candidate_ids: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(scores, ids, n_compared)``.

        ``n_compared`` is the number of candidate vectors the query was scored
        against — the figure used for the computational-efficiency analysis.
        """
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        bucket = self._vectors.get(layer, {})
        if not bucket:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), 0

        if candidate_ids is None:
            return self._search_global(layer, q, k)
        return self._search_restricted(layer, q, k, candidate_ids)

    def _search_global(
        self, layer: int, q: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        bucket = self._vectors[layer]
        ids = list(bucket.keys())
        n = len(ids)

        if self.backend == "faiss":
            index = self._get_faiss(layer)
            scores, out_ids = self._faiss_search(index, q, k, len(ids))
            return scores, out_ids, n

        vecs = np.stack([bucket[i] for i in ids])
        scores, out_ids = self._numpy_search(q, vecs, ids, k)
        return scores, out_ids, n

    def _search_restricted(
        self, layer: int, q: np.ndarray, k: int, candidate_ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray, int]:
        bucket = self._vectors[layer]
        ids = [i for i in candidate_ids if i in bucket]
        if not ids:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), 0
        vecs = np.stack([bucket[i] for i in ids])
        scores = (q @ vecs.T).astype(np.float32)
        top = top_k_indices(scores, k)
        return scores[top], np.asarray([ids[i] for i in top], dtype=np.int64), len(ids)

    @staticmethod
    def _numpy_search(
        q: np.ndarray, vecs: np.ndarray, ids: list[int], k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = (q @ vecs.T).astype(np.float32)
        top = top_k_indices(scores, k)
        out_ids = np.asarray([ids[i] for i in top], dtype=np.int64)
        return scores[top], out_ids

    # ------------------------------------------------------------------ #
    # FAISS plumbing
    # ------------------------------------------------------------------ #
    def _get_faiss(self, layer: int) -> Any:
        import faiss

        cached = self._faiss.get(layer)
        if cached is not None and not self._dirty.get(layer, False):
            return cached

        bucket = self._vectors[layer]
        ids = np.asarray(list(bucket.keys()), dtype=np.int64)
        vecs = np.stack([bucket[i] for i in bucket.keys()]).astype(np.float32)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        if ids.size:
            index.add_with_ids(vecs, ids)
        self._faiss[layer] = index
        self._dirty[layer] = False
        return index

    @staticmethod
    def _faiss_search(
        index: Any, q: np.ndarray, k: int, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = index.search(np.asarray([q], dtype=np.float32), min(k, n))
        return scores[0].astype(np.float32), ids[0].astype(np.int64)
