"""Shared utilities: logging, time, JSON parsing, similarity helpers."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import numpy as np

_LOGGER_NAME = "hmem"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger without forcing global configuration."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def setup_logging(level: str = "INFO") -> None:
    """Configure the ``hmem`` logger once (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False


def now() -> float:
    """Monotonic-clock timestamp used for forgetting-curve decay."""
    return time.monotonic()


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #


def l2_normalize(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """L2-normalize rows so that dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query vector and a stack of candidates.

    ``query``: (D,) or (1, D); ``candidates``: (N, D). Returns (N,).
    """
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    c = np.asarray(candidates, dtype=np.float32)
    q = l2_normalize(q)
    c = l2_normalize(c)
    return (q @ c.T).ravel()


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k highest scores (ascending rank order returned)."""
    k = min(k, scores.size)
    if k <= 0:
        return np.array([], dtype=np.int64)
    return np.argsort(scores)[::-1][:k]


# --------------------------------------------------------------------------- #
# Robust JSON parsing (LLM outputs are not always clean JSON)
# --------------------------------------------------------------------------- #

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from free-form LLM output."""
    if not text:
        return None
    candidate = text.strip()

    # 1. Direct parse.
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Strip markdown code fences.
    match = _CODE_FENCE_RE.search(candidate)
    if match:
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Grab the outermost balanced {...} block.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(candidate[start : end + 1])
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def parse_json_list(text: str) -> list[dict[str, Any]]:
    """Best-effort extraction of a JSON list of objects from LLM output."""
    if not text:
        return []
    candidate = text.strip()
    try:
        value = json.loads(candidate)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    match = _CODE_FENCE_RE.search(candidate)
    if match:
        try:
            value = json.loads(match.group(1))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []
