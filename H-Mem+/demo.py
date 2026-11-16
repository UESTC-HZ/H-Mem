"""Run source-augmented retrieval and budgeted packing on synthetic dialogue."""

import argparse
from collections import Counter
import re

import numpy as np

from hmemplus import HMemPlus, isolate, source_records

CONVERSATION = {
    "session_1_date_time": "2 November, 2026",
    "session_1": [
        {"speaker": "Alex", "text": "I'm taking a ceramics class."},
        {"speaker": "Sam", "text": "The studio provides clay."},
    ],
    "session_2_date_time": "2026-11-03",
    "session_2": [
        {"speaker": "Sam", "text": "Can we stay late on Friday?"},
        {"speaker": "Alex", "text": "The ceramics studio closes at 6 p.m. on Friday."},
    ],
    "session_3_date_time": "2026-11-05",
    "session_3": [
        {"speaker": "Alex", "text": "I planted basil by the kitchen window."},
        {"speaker": "Sam", "text": "I'll water it this weekend."},
    ],
}

# The closing time is available only through the original-turn windows.
EPISODES = [
    {
        "pos": 1, "layer": "episode", "text": "Ceramics studio supplies",
        "metadata": {"source_turns": CONVERSATION["session_1"]},
    },
    {
        "pos": 2, "layer": "episode", "text": "Kitchen gardening",
        "metadata": {"source_turns": CONVERSATION["session_3"]},
    },
]

WORD = re.compile(r"[a-z0-9]+")
BUDGET_TOKEN = re.compile(r"\w+|[^\w\s]")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def count_tokens(text: str) -> int:
    return len(BUDGET_TOKEN.findall(text))


def overlap_scores(question: str, documents: list[str]) -> list[float]:
    query = set(words(question))
    return [float(len(query.intersection(words(document)))) for document in documents]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=64, help="Budget in demo tokens")
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be non-negative")

    sources = source_records(CONVERSATION)
    nodes = isolate(EPISODES, sources)
    vocabulary = sorted({word for node in nodes for word in words(node["text"])})

    def encode(text: str) -> np.ndarray:
        counts = Counter(words(text))
        vector = np.asarray([counts[word] for word in vocabulary], dtype=float)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    vectors = {node["pos"]: encode(node["text"]) for node in nodes}
    pipeline = HMemPlus(
        nodes, vectors, sources, overlap_scores, count_tokens,
        context_budget=args.budget, budget_tokenizer="demo regex tokens",
    )
    question = "When does the ceramics studio close on Friday?"
    result = pipeline.prepare(question, encode(question))["retrieval"]

    print("H-Mem+ | offline demo with bag-of-words vectors and overlap scores")
    print("Question:", question)
    print("Reranked candidates:")
    for candidate in sorted(result["candidate_documents"], key=lambda item: -item["score"]):
        print(f"  {candidate['id']} (score {candidate['score']:.0f}): {candidate['source_ids']}")
    print("Packed sources:", result["source_ids"])
    print("Skipped for budget:", result["skipped_for_budget"])
    print(f"Context budget used: {count_tokens(result['context'])}/{args.budget} demo tokens")
    print("Reader evidence:")
    print(result["context"] or "(none retrieved)")


if __name__ == "__main__":
    main()
