"""H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents.

H-MEM organizes an agent's long-term memory into a multi-level hierarchy
(Domain -> Category -> Memory Trace -> Episode) and attaches a *position index*
to every memory vector that points at its sub-memories in the next layer.
Retrieval then routes the query through the hierarchy layer by layer instead of
computing similarity against every stored memory (see the EACL 2026 paper).

Typical usage::

    from hmem import HMemAgent
    from hmem.config import HMemConfig

    agent = HMemAgent(HMemConfig())
    agent.ingest("I love action movies.", "You might enjoy Kung Fu films!", ts=0.0)
    answer = agent.answer("What movies do I like?")
    print(answer.answer)
"""

from .agent import HMemAgent
from .config import HMemConfig, load_config
from .schemas import (
    AnswerResult,
    FeedbackLabel,
    MemoryLayer,
    MemoryNode,
    RetrievalResult,
)

__all__ = [
    "HMemAgent",
    "HMemConfig",
    "load_config",
    "FeedbackLabel",
    "MemoryLayer",
    "MemoryNode",
    "RetrievalResult",
    "AnswerResult",
]

__version__ = "0.1.0"
