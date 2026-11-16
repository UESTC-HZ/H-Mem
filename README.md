**English** | [简体中文](README.zh-CN.md)

# H-Mem and H-Mem+

Core implementations and runnable examples for **H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents** and **H-Mem+: Source-Augmented Hierarchical Memory for Budgeted Evidence Matching**.

## Runtime and API Credentials

| Execution mode | API key requirement | Configuration |
| --- | --- | --- |
| Offline examples | None | Python dependencies. The examples use synthetic dialogues and lightweight model substitutes. |
| Custom dialogue data with local models | None for unauthenticated local services | Prepare the dialogue inputs and configure the local models. H-Mem supports a local Ollama server. H-Mem+ requires episode summaries and embeddings. Answer generation additionally requires a local reader. |
| Custom dialogue data with authenticated cloud services | Required for each authenticated service | Configure the API key for each service used for memory extraction, embedding, reranking, or answer generation. |

## Installation and Examples

Use Python 3.12. From this directory, create a virtual environment and install the demo dependencies.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r H-Mem/requirements.txt -r 'H-Mem+/requirements.txt'
python -B H-Mem/demo.py
python -B 'H-Mem+/demo.py'
```

Each directory can also run on its own with its local `requirements.txt` and `python -B demo.py`. The bundled demos run on CPU after dependency installation and require no model or dataset downloads. Reproducing the paper's benchmark tables requires its datasets, model configuration, and evaluation protocol.

### H-Mem

The demo writes three interactions across two domains using a hash encoder and mock language models. With a routing width of one, retrieval follows the entertainment branch through the domain, category, and trace layers to the matching episodes. The output includes node counts for each layer, retrieval paths, the number of vectors scored, and a simulated answer.

### H-Mem+

The demo uses two indexed episode summaries and three sessions of original dialogue. Retrieval recovers the studio's closing time from the original turns in session 2, identified as `D2:1` and `D2:2`. The evidence context lists each selected source once, in dialogue order.

The demo uses bag-of-words vectors, lexical overlap scores, and a budget of 64 tokens counted by a regular expression. It reports the reranked candidates, packed sources, candidates skipped because of the budget, budget usage, and the evidence text prepared for the reader. Run `python -B 'H-Mem+/demo.py' --budget 0` to inspect the output with a zero-token budget.

## Core Implementation

| Algorithm | Source files | Functionality |
| --- | --- | --- |
| H-Mem | [extraction.py](H-Mem/hmem/memory/extraction.py), [store.py](H-Mem/hmem/memory/store.py) | Memory extraction, four-layer storage, and position indices |
| H-Mem | [retrieval.py](H-Mem/hmem/memory/retrieval.py), [update.py](H-Mem/hmem/memory/update.py) | Top-down retrieval and feedback-based memory updates |
| H-Mem+ | [retrieval.py](H-Mem%2B/hmemplus/retrieval.py), [hybrid_retrieval.py](H-Mem%2B/hmemplus/hybrid_retrieval.py) | Dialogue-local episode links and dense/lexical summary retrieval |
| H-Mem+ | [evidence_budget.py](H-Mem%2B/hmemplus/evidence_budget.py), [pipeline.py](H-Mem%2B/hmemplus/pipeline.py) | Session-local source windows, joint reranking, budgeted packing, and reader messages |

## Custom Data and Model Configuration

- **H-Mem** uses `HMemAgent.ingest(user_text, assistant_text)` to write a turn pair and `retrieve(question)` or `answer(question)` to query memory. Configure the memory extraction model under `memory_llm` and the answer model under `llm`. With `load_config()`, set `HMEM_MEMORY_LLM_API_KEY` or `HMEM_LLM_API_KEY` for the corresponding cloud endpoint if it requires authentication. See [config.py](H-Mem/hmem/config.py) for encoder and endpoint configuration.
- **H-Mem+** uses the H-Mem hierarchy as its memory-construction stage. Its retrieval API takes stored episode records (`pos`, `layer`, `text`, and `metadata.source_turns`), episode vectors keyed by position, and ordered source turns for one dialogue. Each episode's source turns must form a unique contiguous match in that dialogue. See [demo.py](H-Mem%2B/demo.py) for an input example. `prepare(question, query_vector)` returns the selected evidence and the reader prompt. `answer(question, query_vector)` invokes the configured `reader(messages)` callback to generate an answer. If the callback calls an authenticated cloud service, configure its API key in the client used by the callback.
- In the paper's H-Mem+ configuration, each summary search and the raw-window lexical search have depth 50. The reranker uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Packing considers the highest-scoring ten candidates under a 2,048-wordpiece evidence budget measured with the local BERT tokenizer. To use these components, provide matching BERT-based episode and query embeddings, a `rerank(question, documents)` callback that returns one score per document in input order, and a `count_tokens(text)` callback that counts BERT wordpieces without truncation or special tokens. The budget covers the rendered evidence context.

## License

See [LICENSE](LICENSE) for the original MIT license and copyright notice.
