"""The H-MEM agent: the orchestration layer that ties storage, retrieval and
updating together into a long-term-reasoning loop (Figure 2 of the paper).
"""

from __future__ import annotations

from typing import Any

from .config import HMemConfig
from .encoder import TextEncoder, build_encoder
from .llm import LLMClient
from .memory import (
    HierarchicalMemoryStore,
    HierarchicalRetriever,
    MemoryExtractor,
    MemoryUpdater,
    analyze_feedback,
)
from .prompts import QA_SYSTEM, qa_user
from .schemas import AnswerResult, FeedbackLabel, RetrievalResult
from .utils import get_logger

log = get_logger("agent")


class HMemAgent:
    """A long-term LLM agent backed by hierarchical memory."""

    def __init__(self, config: HMemConfig):
        self.config = config
        self.encoder: TextEncoder = build_encoder(config.encoder)
        self.llm: LLMClient = LLMClient(config.llm)
        self.memory_llm: LLMClient = LLMClient(config.memory_llm)

        self.store = HierarchicalMemoryStore(config, dim=self.encoder.dim)
        self.retriever = HierarchicalRetriever(self.store, config)
        self.extractor = MemoryExtractor(self.memory_llm)
        self.updater = MemoryUpdater(config)

        self._last_retrieved: list[RetrievalResult] = []
        self._last_n_compared: int = 0

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        user_text: str,
        assistant_text: str = "",
        timestamp: float | None = None,
    ) -> int:
        """Store one user/assistant interaction into the memory hierarchy.

        Returns the position index of the created Episode node (or -1 when
        ``storage_mode`` is "none").
        """
        if not user_text.strip():
            return -1
        dialogue = self._render_dialogue(user_text, assistant_text)
        memory = self.extractor.extract(dialogue)

        texts = self.extractor.level_texts(memory)
        vectors = list(self.encoder.encode(texts))

        metadata = {
            "user_profile": memory.user_profile,
            "event": memory.event,
            "timestamp": memory.timestamp,
        }
        return self.store.ingest(texts, vectors, metadata=metadata, timestamp=timestamp)

    def ingest_session(self, turns: list[dict[str, str]], timestamp: float | None = None) -> int:
        """Ingest a whole session (list of ``{"speaker","text"}`` turns)."""
        last_pos = -1
        for i, turn in enumerate(turns):
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            if not text:
                continue
            if speaker.lower() in ("user", "speaker_a"):
                # Pair with the following assistant turn when available.
                nxt = turns[i + 1] if i + 1 < len(turns) else None
                assistant = nxt.get("text", "") if nxt else ""
                last_pos = self.ingest(text, assistant, timestamp=timestamp)
        return last_pos

    @staticmethod
    def _render_dialogue(user_text: str, assistant_text: str) -> str:
        dialogue = f"User: {user_text}"
        if assistant_text:
            dialogue += f"\nAssistant: {assistant_text}"
        return dialogue

    # ------------------------------------------------------------------ #
    # Retrieval + answer
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str) -> list[RetrievalResult]:
        """Return the memories retrieved for ``question`` (no LLM answer).

        Useful for debugging the index-based routing and inspecting confidence
        levels.
        """
        query_vector = self.encoder.encode([question])[0]
        retrieved, n_compared = self.retriever.retrieve(query_vector)
        self._last_retrieved = retrieved
        self._last_n_compared = n_compared
        return retrieved

    def answer(self, question: str) -> AnswerResult:
        """Retrieve relevant memories and produce a grounded answer."""
        retrieved = self.retrieve(question)

        prompt = qa_user(question, retrieved)
        answer = self.llm.chat(
            [
                {"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )

        return AnswerResult(
            question=question,
            answer=answer.strip(),
            retrieved=retrieved,
            prompt=prompt,
            n_candidates_searched=self._last_n_compared,
        )

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def observe_feedback(
        self,
        user_feedback: str,
        answer: str | None = None,
        retrieved: list[RetrievalResult] | None = None,
    ) -> None:
        """Adjust memory weights from user feedback on the last answer.

        The feedback label / multiplier is produced *once* by the LLM (falling
        back to a keyword heuristic) and applied on top of the forgetting-curve
        decay to every retrieved memory.
        """
        retrieved = retrieved if retrieved is not None else self._last_retrieved
        answer = answer if answer is not None else ""
        if not retrieved:
            return

        # Analyse feedback once (one LLM call), then apply to all memories.
        analysis = analyze_feedback(
            self.memory_llm,
            answer=answer,
            user_feedback=user_feedback,
            memory_text=" | ".join(r.node.text[:120] for r in retrieved),
        )
        label: FeedbackLabel = analysis["label"]
        weight: float | None = (
            analysis.get("weight") if self.config.feedback.use_llm_weight else None
        )
        for result in retrieved:
            self.updater.update(result.node, label, weight)

    # ------------------------------------------------------------------ #
    # Persistence / introspection
    # ------------------------------------------------------------------ #
    def save(self, directory: str | None = None) -> None:
        self.store.save(directory or self.config.persist_dir)

    def load(self, directory: str | None = None) -> None:
        self.store.load(directory or self.config.persist_dir)

    def reset(self) -> None:
        """Clear all stored memory and retrieval state (fresh agent)."""
        self.store.clear()
        self._last_retrieved = []
        self._last_n_compared = 0

    @property
    def last_retrieved(self) -> list[RetrievalResult]:
        return list(self._last_retrieved)

    @property
    def last_n_compared(self) -> int:
        return self._last_n_compared

    def stats(self) -> dict[str, Any]:
        return self.store.stats()
