"""Versioned source-preserving H-Mem+ retrieval, without model API calls.

The API accepts conversation turns, stored nodes and question text only. Gold
answers, categories and annotated evidence IDs cannot influence selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import collections
import math
import re

from .hybrid_retrieval import HybridIndex, _normalise


@dataclass(frozen=True)
class Source:
    source_id: str
    speaker: str
    text: str
    observed: str
    session: int
    order: int

    def render(self):
        return (f"[Source {self.source_id}; observed {self.observed}; "
                f"speaker {self.speaker}] {self.text}")


def _observed_date(raw: str) -> str:
    """Normalize stored session date formats."""
    if not raw:
        return raw
    for date_format in ("%I:%M %p on %d %B, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(raw.strip(), date_format).date().isoformat()
        except ValueError:
            continue
    return raw


def source_records(conversation):
    """Retain every turn, including sessions opened by the second speaker.

    Text-only protocol: never silently add captions that baselines did not see.
    Session-local sequence IDs are not gold annotations; they address raw inputs.
    """
    sessions = sorted((int(k.split('_')[1]), v) for k,v in conversation.items()
                      if re.fullmatch(r'session_\d+', k))
    sources = []
    for session, turns in sessions:
        date = conversation.get(f'session_{session}_date_time', '')
        observed = _observed_date(date)
        for i,t in enumerate(turns, 1):
            sources.append(Source(f'D{session}:{i}', t['speaker'], t.get('text') or '',
                                  observed, session, len(sources)))
    return sources


class LexicalIndex:
    def __init__(self, texts):
        self.tokens = [set(_normalise(t).split()) for t in texts]
        counts = collections.Counter(t for ts in self.tokens for t in ts)
        self.idf = {t: math.log(1+len(texts)/(1+c)) for t,c in counts.items()}

    def top(self, question, k):
        q = set(_normalise(question).split())
        scores = [sum(self.idf[t] for t in sorted(q & ts)) for ts in self.tokens]
        return sorted(range(len(scores)),key=lambda i: -scores[i])[:k]


class EvidenceBudgetRetriever:
    """Summary hybrid + raw lexical candidates, local reranking and source packing."""
    def __init__(self, nodes, vectors, sources, rerank, count_tokens,
                 candidate_depth=50, reader_windows=10, context_budget=2048,
                 *, budget_tokenizer="local BERT wordpieces; not API usage"):
        self.nodes = {n['pos']:n for n in nodes if n['layer']=='episode'}
        self.summary = HybridIndex(list(self.nodes.values()), vectors)
        self.sources = sources
        self.by_id = {s.source_id:s for s in sources}
        self.rerank = rerank
        self.count_tokens = count_tokens
        self.depth = candidate_depth
        self.reader_windows = reader_windows
        self.context_budget = context_budget
        self.budget_tokenizer = budget_tokenizer
        self.episode_sources = {}
        matches = collections.defaultdict(list)
        for s in sources:
            matches[s.speaker,s.text].append(s.source_id)
        for pos,n in self.nodes.items():
            ids = []
            for t in n.get('metadata',{}).get('source_turns',[]):
                found = matches[t['speaker'],t['text']]
                if len(found)!=1:
                    raise ValueError(f'Ambiguous or missing provenance for episode {pos}')
                ids.extend(found)
            self.episode_sources[pos] = ids
        # Include the preceding and following turn for questions / pronouns.
        # Never cross a session boundary; indexing covers all original turns.
        self.windows = []
        for i,s in enumerate(sources):
            self.windows.append([t.source_id for t in sources[max(0,i-1):i+2]
                                 if t.session==s.session])
        self.raw_lexical = LexicalIndex([self.render_sources(ids) for ids in self.windows])

    def render_sources(self, ids):
        return '\n'.join(self.by_id[i].render() for i in ids)

    def legacy_passage(self,pos):
        n = self.nodes[pos]
        return ' '.join([n['text']] + [t['text'] for t in n['metadata']['source_turns'] if t['text']])

    def replay(self, selected_positions, labeled=False):
        """Original selected episodes; only rendering differs in the labeled arm."""
        parts = []
        ids = []
        for i,pos in enumerate(selected_positions,1):
            ids.extend(self.episode_sources[pos])
            text = self.legacy_passage(pos)
            if labeled:
                text = self.nodes[pos]['text']+'\n'+self.render_sources(self.episode_sources[pos])
            parts.append(f'[Memory {i}] {text}')
        return dict(context='\n'.join(parts), source_ids=list(dict.fromkeys(ids)),
                    selected_episodes=selected_positions)

    def retrieve(self, question, query_vector):
        candidate_ids = self.summary.candidates(question, query_vector,
                         k_dense=self.depth,k_lexical=self.depth)
        docs, doc_ids, doc_sources = [], [], []
        for i in candidate_ids:
            n = self.summary.nodes[i]
            ids = self.episode_sources[n['pos']]
            docs.append(n['text']+'\n'+self.render_sources(ids))
            doc_ids.append(f"episode:{n['pos']}"); doc_sources.append(ids)
        for i in self.raw_lexical.top(question,self.depth):
            ids = self.windows[i]
            docs.append(self.render_sources(ids))
            doc_ids.append(f'window:{self.sources[i].source_id}'); doc_sources.append(ids)
        scores = list(map(float,self.rerank(question,docs)))
        if len(scores)!=len(docs) or not all(math.isfinite(s) for s in scores):
            raise ValueError('Invalid reranker output')
        order = sorted(range(len(docs)),key=lambda i: -scores[i])
        selected, selected_docs, rejected = [], [], []
        for i in order[:self.reader_windows]:
            new = [sid for sid in doc_sources[i] if sid not in selected]
            proposal = sorted(selected+new,key=lambda sid:self.by_id[sid].order)
            if self.count_tokens(self.render_sources(proposal)) <= self.context_budget:
                selected = proposal
                selected_docs.append(doc_ids[i])
            else:
                rejected.append(doc_ids[i])
        context = self.render_sources(selected)
        assert self.count_tokens(context)<=self.context_budget
        return dict(context=context,source_ids=selected,selected_documents=selected_docs,
                    candidate_documents=[dict(id=d,score=s,source_ids=ids)
                                         for d,s,ids in zip(doc_ids,scores,doc_sources)],
                    skipped_for_budget=rejected,context_budget=self.context_budget,
                    budget_tokenizer=self.budget_tokenizer)
