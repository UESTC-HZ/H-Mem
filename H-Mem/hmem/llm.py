"""LLM backends for H-MEM.

Three providers are supported:

* ``ollama`` -- local Ollama server (default; matches the paper's deployment)
* ``openai`` -- any OpenAI-compatible ``/v1/chat/completions`` endpoint
* ``mock``   -- deterministic stub used by unit tests and the offline demo
"""

from __future__ import annotations

import json
from typing import Any

import requests

from .config import LLMConfig
from .utils import get_logger

log = get_logger("llm")


class LLMClient:
    """A thin chat-completions wrapper around Ollama / OpenAI / mock backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        # Token accounting (journal harness patch, 2026-10-20). The API returns a `usage` block on
        # every reply and the original code discarded it, so a run could report how many CALLS it
        # made but not what it COST. Cost cannot be estimated from call counts when prompts range
        # from a two-line extraction to a long retrieval-conditioned answer.
        self.usage: dict[str, int] = {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "cached_tokens": 0,
        }

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat request and return the assistant text."""
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens

        if self.config.provider == "ollama":
            return self._chat_ollama(messages, json_mode, temperature, max_tokens)
        if self.config.provider == "openai":
            return self._chat_openai(messages, json_mode, temperature, max_tokens)
        if self.config.provider == "mock":
            return self._chat_mock(messages, json_mode)
        raise ValueError(f"Unknown LLM provider {self.config.provider!r}")

    def complete_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any] | None:
        """Convenience: chat + best-effort JSON parse."""
        from .utils import parse_json

        return parse_json(self.chat(messages, json_mode=True, **kwargs))

    # ------------------------------------------------------------------ #
    def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.config.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "") or ""

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(
            f"{self.config.base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._record_usage(data.get("usage") or {})
        return data["choices"][0]["message"].get("content", "") or ""

    def _record_usage(self, usage: dict[str, Any]) -> None:
        """Accumulate token usage; tolerate providers that omit or reshape the block."""
        u = self.usage
        u["calls"] += 1
        u["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        u["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        u["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
        u["cached_tokens"] += int(usage.get("prompt_cache_hit_tokens")
                                 or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                                 or 0)

    def _chat_mock(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        """Deterministic stub.

        Returns a canned JSON object for extraction/feedback prompts and a
        canned sentence otherwise, so the whole agent loop is runnable offline.
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        lower = last_user.lower()

        if json_mode and "extract the hierarchical memory" in lower:
            # Echo the dialogue snippet so the offline demo produces distinct
            # memories per interaction.
            snippet = _dialogue_snippet(last_user)
            return json.dumps(
                {
                    "domain": "entertainment",
                    "category": "action movies",
                    "trace": "action movie, kung fu, jackie chan",
                    "event": f"The user said: {snippet}",
                    "user_profile": {
                        "preferences": "action movies",
                        "interests": "martial arts films",
                        "emotional_state": "curious",
                        "behavioral_patterns": "asks for recommendations",
                    },
                }
            )
        if json_mode and "respond with a single json object matching" in lower:
            # Feedback-regulation prompt.
            return json.dumps({"label": "neutral", "weight": 1.0})
        if json_mode:
            return json.dumps({"answer": "mock response"})

        return "I recall from our previous conversations that you like action movies."


def _dialogue_snippet(text: str, limit: int = 60) -> str:
    """Extract the first user utterance from an extraction prompt."""
    marker = "User:"
    idx = text.find(marker)
    if idx == -1:
        snippet = text
    else:
        snippet = text[idx + len(marker):]
    snippet = snippet.split("Assistant:")[0].split("\n")[0].strip()
    return snippet[:limit] or "something"
