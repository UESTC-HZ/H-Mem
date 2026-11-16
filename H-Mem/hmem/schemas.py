"""Core data models for H-MEM.

The central concept (Section 3.2) is that every memory vector at layer L is
*embedded* with a positional index encoding:

    v_i^(L) = [ e_i^(L)      # dense semantic vector (used for similarity)
              | p_{i-1}x    # self index (its own position in the memory)
              | p_i1..p_iK  # discrete indices of sub-memories in layer L+1 ]

We keep the semantic vector in a NumPy array (indexed by FAISS) and store the
discrete position indices as attributes so that routing never requires an
exhaustive similarity computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MemoryLayer(str, Enum):
    """The four memory layers, from most to least abstract."""

    DOMAIN = "domain"  # L1: high-level domain of interest
    CATEGORY = "category"  # L2: category / sub-domain
    TRACE = "trace"  # L3: memory trace (keyword summary)
    EPISODE = "episode"  # L4: concrete episode + user profile

    @property
    def level(self) -> int:
        """1-based abstraction level (1 == most abstract)."""
        return LAYER_ORDER.index(self) + 1

    @property
    def next_layer(self) -> MemoryLayer | None:
        idx = LAYER_ORDER.index(self) + 1
        return LAYER_ORDER[idx] if idx < len(LAYER_ORDER) else None

    @property
    def prev_layer(self) -> MemoryLayer | None:
        idx = LAYER_ORDER.index(self) - 1
        return LAYER_ORDER[idx] if idx >= 0 else None


LAYER_ORDER: list[MemoryLayer] = [
    MemoryLayer.DOMAIN,
    MemoryLayer.CATEGORY,
    MemoryLayer.TRACE,
    MemoryLayer.EPISODE,
]

LAYER_NAMES: list[str] = [layer.value for layer in LAYER_ORDER]


class FeedbackLabel(str, Enum):
    """User feedback used by the dynamic memory-regulation mechanism."""

    APPROVE = "approve"
    NEUTRAL = "neutral"
    REBUT = "rebut"


@dataclass
class MemoryNode:
    """A single memory unit at some layer of the hierarchy.

    ``pos`` is the *position index* (a dense, stable integer) embedded after
    the semantic vector. ``parent_pos`` and ``child_pos`` implement the index
    pointers to the layer above / below respectively.
    """

    pos: int
    layer: MemoryLayer
    text: str  # the text that was encoded into the semantic vector
    embedding: np.ndarray  # dense semantic vector (float32, D,)
    weight: float = 1.0  # memory strength / confidence level
    strength: float = 100.0  # S in the Ebbinghaus retention R = exp(-t / S)
    created_at: float = 0.0
    last_access: float = 0.0
    parent_pos: int | None = None  # self index of the parent (layer L-1)
    child_pos: list[int] = field(default_factory=list)  # sub-memory indices
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Position-index encoding helpers
    # ------------------------------------------------------------------ #
    @property
    def self_index(self) -> int:
        """p_{i-1}x — the memory's own position index."""
        return self.pos

    @property
    def sub_memory_indices(self) -> list[int]:
        """p_i1..p_iK — indices of sub-memories in the next layer."""
        return self.child_pos

    def encoded_vector(self) -> np.ndarray:
        """Return the *semantic* vector consumed by FAISS.

        The full encoded vector is ``[semantic | self index | sub indices]``;
        the discrete index tail is stored as attributes (not in the float
        array) because similarity is computed over semantic vectors only.
        """
        return np.asarray(self.embedding, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos,
            "layer": self.layer.value,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "weight": self.weight,
            "strength": self.strength,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "parent_pos": self.parent_pos,
            "child_pos": self.child_pos,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryNode:
        return cls(
            pos=int(data["pos"]),
            layer=MemoryLayer(data["layer"]),
            text=data["text"],
            embedding=np.asarray(data["embedding"], dtype=np.float32),
            weight=float(data.get("weight", 1.0)),
            strength=float(data.get("strength", 100.0)),
            created_at=float(data.get("created_at", 0.0)),
            last_access=float(data.get("last_access", 0.0)),
            parent_pos=data.get("parent_pos"),
            child_pos=[int(x) for x in data.get("child_pos", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractedMemory:
    """The four-layer memory extracted from a single interaction (Section 3.1)."""

    domain: str
    category: str
    trace: str  # keyword / gist summary
    event: str  # concrete episode content
    user_profile: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None

    def as_text(self) -> str:
        """The concatenation that gets embedded as the episode semantic vector."""
        profile = " ".join(f"{k}: {v}" for k, v in self.user_profile.items())
        parts = [self.event]
        if profile:
            parts.append(profile)
        return " ".join(parts)


@dataclass
class RetrievalResult:
    """One retrieved memory unit, with its confidence level."""

    node: MemoryNode
    score: float  # cosine similarity of the semantic vectors
    weight: float  # memory weight / confidence level for the LLM

    @property
    def confidence(self) -> float:
        """Confidence level shown to the LLM (normalized weight x similarity)."""
        return float(self.weight * max(self.score, 0.0))


@dataclass
class AnswerResult:
    """The outcome of ``HMemAgent.answer``."""

    question: str
    answer: str
    retrieved: list[RetrievalResult] = field(default_factory=list)
    prompt: str = ""
    n_candidates_searched: int = 0  # total vectors compared (efficiency bookkeeping)

    @property
    def n_retrieved(self) -> int:
        return len(self.retrieved)
