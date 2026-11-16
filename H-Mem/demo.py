"""Run the conference H-Mem write and retrieval loop on synthetic dialogue."""

import json

from hmem import HMemAgent, HMemConfig, MemoryLayer
from hmem.llm import LLMClient


class DemoMemoryClient(LLMClient):
    """Add a second topic to the built-in mock extraction responses."""

    def _chat_mock(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        if json_mode and "User: I also enjoy skiing competitions." in messages[-1]["content"]:
            return json.dumps({
                "domain": "outdoor sports",
                "category": "skiing",
                "trace": "skiing competitions",
                "event": "The user enjoys watching skiing competitions.",
            })
        return super()._chat_mock(messages, json_mode)


def main() -> None:
    config = HMemConfig()
    config.encoder.backend = "hash"
    config.encoder.dim = 256
    config.backend = "numpy"
    config.top_k = 1
    config.llm.provider = "mock"
    config.memory_llm.provider = "mock"
    agent = HMemAgent(config)
    agent.extractor.llm = DemoMemoryClient(config.memory_llm)

    interactions = [
        ("I love action movies!", "Try a Jackie Chan action movie."),
        ("I also enjoy skiing competitions.", "Let's look for a ski race."),
        ("I prefer to watch films with subtitles.", "I'll keep that in mind."),
    ]
    for index, (user, assistant) in enumerate(interactions):
        agent.ingest(user, assistant, timestamp=1_700_000_000 + index)

    print("H-Mem | offline demo with a hash encoder and mock LLMs")
    print("Memory nodes:", agent.stats())
    result = agent.answer("What action movies do I like for entertainment?")
    print("Question:", result.question)
    print("Vectors scored during hierarchical retrieval:", result.n_candidates_searched)
    for match in result.retrieved:
        path = [match.node]
        while path[-1].parent_pos is not None:
            parent = agent.store.get(path[-1].parent_pos)
            if parent is None:
                raise RuntimeError("Missing parent in the memory hierarchy")
            path.append(parent)
        if len(path) != len(MemoryLayer):
            raise RuntimeError("Expected a four-layer path to the episode")
        route = " -> ".join(f"{node.layer.value}[{node.pos}]" for node in reversed(path))
        print(f"  {route} (similarity {match.score:.3f})")
        print(f"    {match.node.text[:100]}")
    print("Mock answer:", result.answer)


if __name__ == "__main__":
    main()
