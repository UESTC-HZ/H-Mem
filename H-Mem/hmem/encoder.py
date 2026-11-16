"""Text encoders for H-MEM memory vectors.

The paper uses BERT as the memory encoder. We support three backends and pick
one automatically so the package works with or without model downloads:

* ``sentence``     -- sentence-transformers (best sentence-level quality)
* ``transformers`` -- raw Hugging Face BERT with mean pooling (paper default)
* ``hash``         -- deterministic hashing vectorizer (offline / unit tests)
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .config import EncoderConfig
from .utils import get_logger, l2_normalize

log = get_logger("encoder")

_TOKEN_RE = re.compile(r"\w+")


class TextEncoder:
    """Interface for encoders: ``encode(texts) -> (N, D) float32 matrix``."""

    dim: int

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        raise NotImplementedError


class HashEncoder(TextEncoder):
    """Deterministic, dependency-free bag-of-char/word-n-gram hashing.

    Not intended for production quality — it exists so the library and its
    tests run in fully offline environments. Embeddings are L2-normalized.
    """

    def __init__(self, dim: int = 768, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = _TOKEN_RE.findall(text)
        features = list(words)
        features.extend(w[:5] for w in words)  # prefix sub-word
        # Character 3-grams for typo tolerance.
        compact = "".join(words)
        features.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
        return features

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.md5(f"{self.seed}:{feature}".encode()).digest()
                idx = int.from_bytes(digest[:8], "big") % self.dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                matrix[row, idx] += sign
        return l2_normalize(matrix) if normalize else matrix


class TransformersEncoder(TextEncoder):
    """BERT-style encoder via Hugging Face ``transformers`` with mean pooling."""

    def __init__(self, config: EncoderConfig):
        import torch  # noqa: F401 -- imported lazily to keep the base import light
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.dim = config.dim or self.model.config.hidden_size
        self._device = _resolve_device(config.device)
        self.model.to(self._device)
        self.model.eval()

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        import torch

        outputs = []
        bs = self.config.batch_size
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state  # (B, L, H)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            outputs.append(pooled.cpu().numpy())
        matrix = np.vstack(outputs).astype(np.float32)
        return l2_normalize(matrix) if normalize else matrix


class SentenceEncoder(TextEncoder):
    """Sentence-transformers encoder (better sentence embeddings than raw BERT)."""

    def __init__(self, config: EncoderConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._model = SentenceTransformer(config.model_name, device=_resolve_device(config.device))
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        matrix = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return matrix


def _resolve_device(device: str) -> str:
    if device not in ("auto", "cpu", "cuda"):
        return device
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch absent
            return "cpu"
    return device


def build_encoder(config: EncoderConfig) -> TextEncoder:
    """Construct the encoder selected by ``config.backend`` ("auto" resolves)."""
    backend = config.backend
    if backend == "auto":
        backend = _auto_backend()

    if backend == "hash":
        log.info("Using deterministic HashEncoder (offline).")
        return HashEncoder(dim=config.dim)
    if backend == "sentence":
        log.info("Using sentence-transformers encoder: %s", config.model_name)
        return SentenceEncoder(config)
    if backend == "transformers":
        log.info("Using transformers encoder: %s", config.model_name)
        return TransformersEncoder(config)
    raise ValueError(
        f"Unknown encoder backend {backend!r}; expected sentence|transformers|hash|auto"
    )


def _auto_backend() -> str:
    try:
        import sentence_transformers  # noqa: F401

        return "sentence"
    except Exception:
        pass
    try:
        import transformers  # noqa: F401

        return "transformers"
    except Exception:
        return "hash"
