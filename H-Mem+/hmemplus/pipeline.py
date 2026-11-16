"""Reader-independent H-Mem+ interface."""

from .answer_prompt import SYSTEM
from .retrieval import DialogueEvidenceRetriever


class HMemPlus:
    """Prepare reader messages from a single dialogue's episodes and source turns."""

    def __init__(
        self,
        nodes,
        vectors,
        sources,
        rerank,
        count_tokens,
        reader=None,
        candidate_depth=50,
        reader_windows=10,
        context_budget=2048,
        *,
        budget_tokenizer="local BERT wordpieces; not API usage",
    ):
        self.retriever = DialogueEvidenceRetriever(
            nodes,
            vectors,
            sources,
            rerank,
            count_tokens,
            candidate_depth=candidate_depth,
            reader_windows=reader_windows,
            context_budget=context_budget,
            budget_tokenizer=budget_tokenizer,
        )
        self.reader = reader

    def prepare(self, question, query_vector):
        """Retrieve evidence and render messages without invoking a reader."""
        result = self.retriever.retrieve(question, query_vector)
        context = result["context"] or "(none retrieved)"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Memories:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ]
        return {"messages": messages, "retrieval": result}

    def answer(self, question, query_vector):
        """Pass the prepared messages to an explicitly configured reader."""
        if self.reader is None:
            raise ValueError("Provide an explicit reader callable")
        prepared = self.prepare(question, query_vector)
        return {**prepared, "answer": self.reader(prepared["messages"])}
