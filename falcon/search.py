"""
Search one query against the static gallery (answer #38: the gallery is a fixed base that can be
built in advance; each query is processed ALONE, like one frame from a camera stream).

For a single query:
  1. cosine similarity to the gallery; keep the top-K candidates
  2. k-reciprocal re-ranking of those candidates (allowed: answers #28, #38; not timed: #31)
  3. refusal signal "cos+gap" = cosine of the re-ranked top-1 + (that cosine - the best cosine among
     the other gallery items). Chosen in experiments/threshold_study.py for the official objective
     0.7*F1 + 0.3*TNR (answer #15).
Nothing here ever looks at another query.

rank_candidates() is shared by the batch run (Gallery, in memory) and the web service
(candidates come from the pgvector database), so both give identical answers.
"""
import numpy as np

from reid.rerank import re_ranking


def rank_candidates(q, cand_vecs, cand_cos, k1=6, k2=2, lam=0.3, rerank=True):
    """q: (D,) unit query; cand_vecs: (K, D) unit vectors of the cosine top-K, sorted by cosine
    (best first); cand_cos: (K,) their cosines. Returns (order into the candidates, cos+gap confidence).
    The gap uses only candidates: the best 'other' gallery item is always inside the cosine top-K."""
    k = len(cand_cos)
    order = np.arange(k)
    if rerank and k > 1:
        sim = re_ranking(q[None], cand_vecs, k1, k2, lam)[0]
        order = np.argsort(-sim, kind="stable")
    top = order[0]
    others = np.delete(cand_cos, top)
    gap = cand_cos[top] - (others.max() if len(others) else -1.0)
    return order, float(cand_cos[top] + gap)


class Gallery:
    def __init__(self, vectors, ids, k1=6, k2=2, lam=0.3, topk=100, rerank=True):
        g = np.asarray(vectors, dtype=np.float32)
        self.g = g / np.linalg.norm(g, axis=1, keepdims=True)
        self.ids = list(ids)
        self.k1, self.k2, self.lam, self.topk, self.rerank = k1, k2, lam, topk, rerank

    def search(self, q, n=10):
        """q: (D,) vector of ONE query. Returns (ranked gallery indices [n], top-1 index, cos+gap)."""
        q = np.asarray(q, dtype=np.float32)
        q = q / np.linalg.norm(q)
        cos = self.g @ q                                           # (G,)
        cand = np.argsort(-cos, kind="stable")[:max(min(self.topk, len(cos)), min(n, len(cos)))]
        order, conf = rank_candidates(q, self.g[cand], cos[cand], self.k1, self.k2, self.lam, self.rerank)
        ranked = cand[order][:n]
        return ranked, int(ranked[0]), conf
