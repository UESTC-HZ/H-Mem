"""Memory extraction: dialogue -> four-layer structured memory (Section 3.1).

A dedicated *memory extraction model* (``memory_llm``) parses each interaction
into domain / category / trace / episode and an inferred user profile, guided
by :mod:`hmem.prompts`.
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMClient
from ..prompts import EXTRACTION_SYSTEM, extraction_user
from ..schemas import ExtractedMemory
from ..utils import get_logger, parse_json

log = get_logger("extraction")


class MemoryExtractor:
    """Turns a dialogue turn into an :class:`ExtractedMemory`."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(
        self, dialogue: str, timestamp: str | None = None
    ) -> ExtractedMemory:
        """Run the memory-analysis LLM and coerce its JSON into a memory.

        If the LLM returns unparseable JSON, we degrade gracefully to a
        dialogue-derived fallback so a single bad extraction never crashes the
        agent.
        """
        raw = ""
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": extraction_user(dialogue)},
                ],
                json_mode=True,
            )
        except Exception as exc:  # pragma: no cover - network/model errors
            log.warning("Memory extraction failed (%s); using fallback.", exc)

        data = parse_json(raw) or {}
        if not any(data.get(k) for k in ("domain", "category", "trace", "event")):
            log.warning("Memory extraction returned no structure; using fallback.")
            data = _fallback_extraction(dialogue)

        return ExtractedMemory(
            domain=_as_str(data.get("domain")),
            category=_as_str(data.get("category")),
            trace=_as_str(data.get("trace")),
            event=_as_str(data.get("event")),
            user_profile=_profile(data.get("user_profile")),
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ #
    def level_texts(self, memory: ExtractedMemory) -> list:
        """Return the four layer texts in layer order."""
        profile = self._profile_text(memory.user_profile)
        return [memory.domain, memory.category, memory.trace, memory.event + (" " + profile if profile else "")]

    @staticmethod
    def _profile_text(profile: dict[str, Any]) -> str:
        return " ".join(f"{k}: {v}" for k, v in profile.items() if v)


def _fallback_extraction(dialogue: str) -> dict[str, Any]:
    """Derive a minimal extraction from the raw dialogue text.

    Keeps the pipeline running when the memory-analysis LLM misbehaves; the
    hierarchy still forms, just with coarser (text-derived) semantics.
    """
    text = (dialogue or "").strip() or "unknown"
    return {
        "domain": "general",
        "category": "conversation",
        "trace": "dialogue",
        "event": text[:200],
        "user_profile": {},
    }


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value).strip()


def _profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items() if v not in (None, "")}
