"""Final dialogue-isolated H-Mem+ retrieval used in the journal experiments."""
from .evidence_budget import EvidenceBudgetRetriever, LexicalIndex
from .hybrid_retrieval import HybridIndex

def episode_source_ids(node, sources):
    """Resolve repeated utterances by exact contiguous source-turn sequence."""
    turns = node.get('metadata', {}).get('source_turns', [])
    if not turns: raise ValueError('Episode without provenance')
    keys = [(t['speaker'], t['text']) for t in turns]
    allkeys = [(s.speaker, s.text) for s in sources]
    matches = [i for i in range(len(sources)-len(keys)+1) if allkeys[i:i+len(keys)] == keys]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous contiguous source provenance')
    i = matches[0]
    return [s.source_id for s in sources[i:i+len(keys)]]

def isolate(nodes, sources):
    """Strict provenance import; never include an episode from another dialogue."""
    keys = {(s.speaker, s.text) for s in sources}; result = []
    for n in nodes:
        if n['layer'] != 'episode': continue
        turns = n.get('metadata', {}).get('source_turns', [])
        if not turns: raise ValueError('Episode without provenance')
        if not all((t['speaker'], t['text']) in keys for t in turns): continue
        episode_source_ids(n, sources)
        result.append(n)
    if not result: raise ValueError('No dialogue-local episodes found')
    return result

class DialogueEvidenceRetriever(EvidenceBudgetRetriever):
    """Same frozen retrieval/packing; import provenance supports repeated phrases.

    The original constructor resolves each turn independently, so repeated
    goodbyes are ambiguous. Resolve an entire stored pair by exact adjacency.
    No text, scores, candidates, budget or inherited retrieval method changes.
    """
    def __init__(self, nodes, vectors, sources, rerank, count_tokens,
                 candidate_depth=50, reader_windows=10, context_budget=2048,
                 *, budget_tokenizer="local BERT wordpieces; not API usage"):
        self.nodes = {n['pos']:n for n in nodes if n['layer']=='episode'}
        self.summary = HybridIndex(list(self.nodes.values()), vectors)
        self.sources = sources
        self.by_id = {s.source_id:s for s in sources}
        self.rerank = rerank; self.count_tokens = count_tokens
        self.depth = candidate_depth; self.reader_windows = reader_windows
        self.context_budget = context_budget
        self.budget_tokenizer = budget_tokenizer
        self.episode_sources = {p:episode_source_ids(n,sources) for p,n in self.nodes.items()}
        self.windows = [[t.source_id for t in sources[max(0,i-1):i+2] if t.session==s.session]
                        for i,s in enumerate(sources)]
        self.raw_lexical = LexicalIndex([self.render_sources(ids) for ids in self.windows])
