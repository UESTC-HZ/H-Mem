"""Dynamic memory regulation (Section 3.3).

H-MEM combines the Ebbinghaus forgetting curve with an LLM-driven feedback
mechanism so memory strength tracks *changing* user psychology (e.g. a user who
once loved skiing but later soured on it). No gradient/parameter update of the
LLM is involved -- only external memory weights change.
"""

from __future__ import annotations

import math

from ..config import FeedbackConfig, ForgettingConfig, HMemConfig
from ..schemas import FeedbackLabel, MemoryNode
from ..utils import get_logger, now

log = get_logger("update")


def retention(strength: float, delta_t: float) -> float:
    """Ebbinghaus retention ``R = exp(-delta_t / S)``."""
    if strength <= 0:
        return 0.0
    return float(math.exp(-max(delta_t, 0.0) / strength))


class MemoryUpdater:
    """Applies forgetting-curve decay and feedback modulation to memory weights."""

    def __init__(self, config: HMemConfig):
        self.forgetting: ForgettingConfig = config.forgetting
        self.feedback: FeedbackConfig = config.feedback

    # ------------------------------------------------------------------ #
    def update(
        self,
        node: MemoryNode,
        feedback: FeedbackLabel = FeedbackLabel.NEUTRAL,
        feedback_weight: float | None = None,
        timestamp: float | None = None,
    ) -> MemoryNode:
        """Decay + feedback-regulate a single memory node, in place.

        * ``approve`` -- reinforce (multiply weight up, boost strength)
        * ``neutral`` -- natural forgetting-curve shrink
        * ``rebut``   -- weaken (multiply weight down, reduce strength)
        """
        timestamp = now() if timestamp is None else float(timestamp)
        delta_t = max(timestamp - node.last_access, 0.0)

        decay = retention(node.strength, delta_t)
        multiplier = feedback_weight if feedback_weight is not None else self._default_multiplier(feedback)

        new_weight = node.weight * decay * multiplier
        node.weight = _clamp(new_weight, self.forgetting.min_weight, self.forgetting.max_weight)

        # Strength (S) is reinforced on approval and eroded on rebuttal so the
        # forgetting curve itself becomes adaptive.
        if feedback is FeedbackLabel.APPROVE:
            node.strength = _clamp(node.strength * 1.1, self.forgetting.min_strength, self.forgetting.max_strength)
        elif feedback is FeedbackLabel.REBUT:
            node.strength = _clamp(node.strength * 0.9, self.forgetting.min_strength, self.forgetting.max_strength)

        node.last_access = timestamp
        return node

    def update_many(
        self,
        nodes: list[MemoryNode],
        feedback: FeedbackLabel,
        feedback_weight: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        timestamp = now() if timestamp is None else float(timestamp)
        for node in nodes:
            self.update(node, feedback, feedback_weight, timestamp)

    # ------------------------------------------------------------------ #
    def _default_multiplier(self, feedback: FeedbackLabel) -> float:
        if feedback is FeedbackLabel.APPROVE:
            return self.feedback.approve_weight
        if feedback is FeedbackLabel.REBUT:
            return self.feedback.rebut_weight
        return self.feedback.neutral_weight


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# LLM-based feedback analysis
# --------------------------------------------------------------------------- #


def analyze_feedback(
    llm,
    answer: str,
    user_feedback: str,
    memory_text: str,
) -> dict:
    """Ask the LLM to classify feedback and produce a weight multiplier.

    Falls back to a simple keyword heuristic if the LLM is unavailable or the
    JSON cannot be parsed.
    """
    from ..prompts import FEEDBACK_SYSTEM, feedback_user
    from ..utils import parse_json

    try:
        raw = llm.chat(
            [
                {"role": "system", "content": FEEDBACK_SYSTEM},
                {"role": "user", "content": feedback_user(answer, user_feedback, memory_text)},
            ],
            json_mode=True,
        )
        data = parse_json(raw) or {}
        label = _coerce_label(data.get("label"))
        weight = data.get("weight")
        if label is not None:
            return {
                "label": label,
                "weight": float(weight) if isinstance(weight, (int, float)) else None,
            }
    except Exception as exc:  # pragma: no cover - network/model errors
        log.warning("LLM feedback analysis failed, using heuristic: %s", exc)

    return _heuristic_feedback(user_feedback)


def _coerce_label(value) -> FeedbackLabel | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "approve": FeedbackLabel.APPROVE,
        "approval": FeedbackLabel.APPROVE,
        "positive": FeedbackLabel.APPROVE,
        "neutral": FeedbackLabel.NEUTRAL,
        "none": FeedbackLabel.NEUTRAL,
        "rebut": FeedbackLabel.REBUT,
        "rebuttal": FeedbackLabel.REBUT,
        "negative": FeedbackLabel.REBUT,
        "disagree": FeedbackLabel.REBUT,
    }
    return mapping.get(text)


def _heuristic_feedback(user_feedback: str) -> dict:
    text = (user_feedback or "").strip().lower()
    approve_terms = ("correct", "yes", "thanks", "great", "good", "right", "exactly", "love", "approve")
    rebut_terms = ("wrong", "no", "incorrect", "not", "disagree", "hate", "bad", "never")
    if any(t in text for t in rebut_terms):
        return {"label": FeedbackLabel.REBUT, "weight": None}
    if any(t in text for t in approve_terms):
        return {"label": FeedbackLabel.APPROVE, "weight": None}
    return {"label": FeedbackLabel.NEUTRAL, "weight": None}
