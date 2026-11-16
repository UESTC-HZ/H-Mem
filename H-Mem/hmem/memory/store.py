"""Hierarchical memory store (Section 3.1).

The store keeps four layers -- Domain, Category, Memory Trace, Episode -- and
wires every unit to its parent via a *position index*. The first three layers
are progressively refined indices; the Episode layer holds the concrete event
text and the inferred user profile.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import HMemConfig
from ..schemas import LAYER_ORDER, MemoryLayer, MemoryNode
from ..utils import get_logger, now
from .index import SimilarityEngine

log = get_logger("store")


class HierarchicalMemoryStore:
    """Stores and links memory nodes across the abstraction hierarchy."""

    def __init__(self, config: HMemConfig, dim: int):
        self.config = config
        self.dim = int(dim)
        self.storage_mode = config.storage_mode
        self.engine = SimilarityEngine(dim=self.dim, backend=config.backend)
        self._nodes: dict[int, MemoryNode] = {}
        self._children: dict[int, list[int]] = {}
        self._roots: list[int] = []
        self._next_pos = 1

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    def get(self, pos: int) -> MemoryNode | None:
        return self._nodes.get(int(pos))

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, pos: int) -> bool:
        return int(pos) in self._nodes

    def children_of(self, pos: int) -> list[int]:
        return list(self._children.get(int(pos), []))

    def roots(self) -> list[int]:
        return list(self._roots)

    def nodes_of_layer(self, layer: MemoryLayer) -> list[MemoryNode]:
        return [n for n in self._nodes.values() if n.layer is layer]

    def layer_positions(self, layer: MemoryLayer) -> list[int]:
        return [n.pos for n in self._nodes.values() if n.layer is layer]

    @property
    def next_pos(self) -> int:
        return self._next_pos

    def stats(self) -> dict[str, int]:
        return {
            "total_nodes": len(self._nodes),
            "domain": len(self.nodes_of_layer(MemoryLayer.DOMAIN)),
            "category": len(self.nodes_of_layer(MemoryLayer.CATEGORY)),
            "trace": len(self.nodes_of_layer(MemoryLayer.TRACE)),
            "episode": len(self.nodes_of_layer(MemoryLayer.EPISODE)),
        }

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        texts: Sequence[str],
        vectors: Sequence[np.ndarray],
        metadata: dict | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert one interaction into the hierarchy.

        ``texts`` / ``vectors`` follow the layer order (domain, category,
        trace, episode). Returns the position index of the new Episode node.
        ``storage_mode`` "flat" collapses everything into Episode nodes.
        """
        if self.storage_mode == "none":
            return -1

        timestamp = now() if timestamp is None else float(timestamp)
        metadata = metadata or {}

        if self.storage_mode == "flat":
            return self._create_node(
                parent_pos=None,
                layer=MemoryLayer.EPISODE,
                text=" ".join(texts),
                vector=vectors[-1],
                timestamp=timestamp,
                metadata=metadata,
            )

        n = min(len(texts), len(vectors), self.config.num_layers)
        parent_pos: int | None = None
        for level in range(n - 1):  # domain .. trace (all but the episode)
            layer = LAYER_ORDER[level]
            parent_pos = self._find_or_create(
                parent_pos, layer, texts[level], vectors[level], timestamp
            )

        episode_text = texts[-1]
        episode_vec = vectors[-1]
        profile = metadata.get("user_profile") or {}
        if profile:
            episode_text = f"{episode_text} {self._profile_text(profile)}".strip()
            episode_vec = vectors[-1]  # profile already embedded via extraction

        return self._create_node(
            parent_pos=parent_pos,
            layer=MemoryLayer.EPISODE,
            text=episode_text,
            vector=episode_vec,
            timestamp=timestamp,
            metadata=metadata,
        )

    @staticmethod
    def _profile_text(profile: dict) -> str:
        return " ".join(f"{k}: {v}" for k, v in profile.items() if v)

    def _find_or_create(
        self,
        parent_pos: int | None,
        layer: MemoryLayer,
        text: str,
        vector: np.ndarray,
        timestamp: float,
    ) -> int:
        """Reuse an existing sibling node when semantically close, else create.

        This keeps the first three layers as a stable, deduplicated index while
        episodes always fan out beneath the matching trace.
        """
        candidates = self.roots() if parent_pos is None else self.children_of(parent_pos)
        if candidates:
            existing = self._best_match(candidates, vector)
            if existing is not None and existing[1] >= self.config.dedup_threshold:
                return existing[0]
        return self._create_node(parent_pos, layer, text, vector, timestamp)

    def _best_match(
        self, candidate_pos: list[int], vector: np.ndarray
    ) -> tuple[int, float] | None:
        """Cosine-similarity nearest sibling above ``dedup_threshold``."""
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        query = query / (np.linalg.norm(query) or 1.0)
        best_pos, best_score = None, -1.0
        for pos in candidate_pos:
            cand_vec = self.engine.vector(self._layer_level(self.get(pos).layer), pos)
            if cand_vec is None:
                continue
            score = float(cand_vec @ query)
            if score > best_score:
                best_pos, best_score = pos, score
        if best_pos is None:
            return None
        return best_pos, best_score

    def _create_node(
        self,
        parent_pos: int | None,
        layer: MemoryLayer,
        text: str,
        vector: np.ndarray,
        timestamp: float,
        metadata: dict | None = None,
    ) -> int:
        pos = self._next_pos
        self._next_pos += 1

        node = MemoryNode(
            pos=pos,
            layer=layer,
            text=text,
            embedding=np.asarray(vector, dtype=np.float32).reshape(-1),
            weight=1.0,
            strength=self.config.forgetting.initial_strength,
            created_at=timestamp,
            last_access=timestamp,
            parent_pos=parent_pos,
            metadata=metadata or {},
        )

        self._nodes[pos] = node
        self.engine.add(self._layer_level(layer), pos, node.embedding)

        if parent_pos is None:
            self._roots.append(pos)
        else:
            self._children.setdefault(int(parent_pos), []).append(pos)

        # Position-index encoding: the parent points at its sub-memory.
        if parent_pos is not None:
            parent = self._nodes[int(parent_pos)]
            if pos not in parent.child_pos:
                parent.child_pos.append(pos)

        return pos

    def add_node(
        self,
        layer: MemoryLayer,
        text: str,
        vector: np.ndarray,
        parent_pos: int | None = None,
        timestamp: float | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Directly insert a node (used by synthetic builders / efficiency runs)."""
        timestamp = now() if timestamp is None else float(timestamp)
        return self._create_node(parent_pos, layer, text, vector, timestamp, metadata)

    def clear(self) -> None:
        """Drop all memory nodes and vectors, leaving a fresh, empty store."""
        self._nodes.clear()
        self._children.clear()
        self._roots.clear()
        self._next_pos = 1
        self.engine.clear()

    @staticmethod
    def _layer_level(layer: MemoryLayer) -> int:
        return layer.level

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        node_records = []
        vectors: dict[str, np.ndarray] = {}
        for pos, node in self._nodes.items():
            record = node.to_dict()
            record.pop("embedding", None)
            node_records.append(record)
            vectors[f"pos_{pos}"] = node.embedding.astype(np.float32)

        with open(path / "nodes.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "next_pos": self._next_pos,
                    "storage_mode": self.storage_mode,
                    "nodes": node_records,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        np.savez_compressed(path / "embeddings.npz", **vectors)
        log.info("Saved %d nodes to %s", len(self._nodes), path)

    def load(self, directory: str | Path) -> None:
        path = Path(directory)
        with open(path / "nodes.json", encoding="utf-8") as fh:
            meta = json.load(fh)

        embeddings = np.load(path / "embeddings.npz")

        self._nodes.clear()
        self._children.clear()
        self._roots.clear()
        self._next_pos = int(meta.get("next_pos", 1))
        self.storage_mode = meta.get("storage_mode", self.storage_mode)

        for record in meta["nodes"]:
            pos = int(record["pos"])
            embedding = embeddings[f"pos_{pos}"]
            node = MemoryNode.from_dict({**record, "embedding": embedding.tolist()})
            self._nodes[pos] = node
            self.engine.add(self._layer_level(node.layer), pos, node.embedding)
            if node.parent_pos is None:
                self._roots.append(pos)
            else:
                self._children.setdefault(int(node.parent_pos), []).append(pos)

        log.info("Loaded %d nodes from %s", len(self._nodes), path)
