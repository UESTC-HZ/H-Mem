"""Prompt templates used by H-MEM.

The memory-extraction prompt follows Section 3.1 of the paper and decomposes a
dialogue into the four hierarchy levels (domain / category / trace / episode).
"""

from __future__ import annotations

from typing import Any

from .schemas import RetrievalResult

# --------------------------------------------------------------------------- #
# Memory extraction (Section 3.1)
# --------------------------------------------------------------------------- #

EXTRACTION_SYSTEM = (
    "You are an information analyze agent for a long-term LLM system. "
    "Given a dialogue, you must extract and structure the information into a "
    "hierarchical memory format. Follow this hierarchy strictly:\n"
    "1. Identify the high-level domain of interest.\n"
    "2. Extract specific categories or subdomains related to the topic.\n"
    "3. Summarize the keywords of the dialogue.\n"
    "4. Extract specific events and user profile.\n"
    "Output the result as structured JSON only, with no extra commentary."
)

EXTRACTION_JSON_SCHEMA = """{
  "domain": "high-level domain of interest (short phrase)",
  "category": "specific category or subdomain (short phrase)",
  "trace": "comma-separated keywords summarizing the dialogue",
  "event": "one-sentence summary of the specific event or content of this turn",
  "user_profile": {
    "preferences": "preferences implied by the user",
    "interests": "interests implied by the user",
    "emotional_state": "current emotional state",
    "behavioral_patterns": "behavioral patterns observed"
  }
}"""


def extraction_user(dialogue: str) -> str:
    return (
        f"Analyze the following dialogue and extract the hierarchical memory.\n\n"
        f"Dialogue:\n{dialogue}\n\n"
        f"Respond with a single JSON object matching this schema:\n"
        f"{EXTRACTION_JSON_SCHEMA}"
    )


# --------------------------------------------------------------------------- #
# Question answering (Section 3.2)
# --------------------------------------------------------------------------- #

QA_SYSTEM = (
    "You are a helpful assistant with a long-term memory. Answer the user's "
    "question using the retrieved memories below. Each memory is followed by "
    "its confidence level (higher is more reliable). If the memories do not "
    "contain the answer, say that you do not have that information rather than "
    "guessing. Answer concisely with only the information asked for."
)


def _format_memory(result: RetrievalResult, index: int) -> str:
    node = result.node
    return (
        f"[Memory {index}] (confidence: {result.confidence:.3f})\n"
        f"  {node.text}\n"
    )


def qa_user(question: str, retrieved: list[RetrievalResult]) -> str:
    if not retrieved:
        memories = "(no relevant memories retrieved)"
    else:
        memories = "\n".join(_format_memory(r, i) for i, r in enumerate(retrieved, 1))
    return f"Retrieved memories:\n{memories}\n\nQuestion: {question}"


# --------------------------------------------------------------------------- #
# Feedback analysis (Section 3.3)
# --------------------------------------------------------------------------- #

FEEDBACK_SYSTEM = (
    "You are a memory-weight regulator for a long-term LLM system. Given a "
    "user's feedback to an answer that was grounded in a particular memory, "
    "classify the feedback and produce a weight multiplier. The multiplier "
    "must be > 1 to strengthen the memory (approval), < 1 to weaken it "
    "(rebuttal), or exactly 1 for no signal. Respond with JSON only."
)

FEEDBACK_JSON_SCHEMA = """{
  "label": "approve" | "neutral" | "rebut",
  "weight": 1.5
}"""


def feedback_user(answer: str, feedback: str, memory_text: str) -> str:
    return (
        f"Memory content: {memory_text}\n"
        f"Assistant answer: {answer}\n"
        f"User feedback: {feedback}\n\n"
        f"Respond with a single JSON object matching this schema:\n"
        f"{FEEDBACK_JSON_SCHEMA}"
    )


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def dict_to_json_block(data: dict[str, Any]) -> str:
    """Render a dict as a JSON block for inclusion in a prompt."""
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)
