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
    """In-memory gallery for the batch run.

    rescore_vectors (optional) = TWO-STAGE search (answer #28): the fast vectors pick the cosine top-K
    candidates; the heavier rescore vectors (Base+ViT ensemble) re-order ONLY those candidates, and the
    refusal signal is computed from them. Still one query at a time (answer #38)."""

    def __init__(self, vectors, ids, k1=6, k2=2, lam=0.3, topk=100, rerank=True, rescore_vectors=None):
        self.g = _unit(vectors)
        self.g2 = None if rescore_vectors is None else _unit(rescore_vectors)
        self.ids = list(ids)
        self.k1, self.k2, self.lam, self.topk, self.rerank = k1, k2, lam, topk, rerank

    def search(self, q, n=10, q_rescore=None):
        """q: (D,) fast vector of ONE query (+ q_rescore for two-stage).
        Returns (ranked gallery indices [n], top-1 index, cos+gap)."""
        q = _unit(q[None])[0]
        cos = self.g @ q                                           # (G,)
        cand = np.argsort(-cos, kind="stable")[:max(min(self.topk, len(cos)), min(n, len(cos)))]
        if self.g2 is None:
            q2, g2, c2 = q, self.g[cand], cos[cand]
        else:                                                      # stage 2: re-order the candidates
            q2 = _unit(q_rescore[None])[0]
            c2 = self.g2[cand] @ q2
            pre = np.argsort(-c2, kind="stable")                   # rank_candidates expects cosine order
            cand, c2 = cand[pre], c2[pre]
            g2 = self.g2[cand]
        order, conf = rank_candidates(q2, g2, c2, self.k1, self.k2, self.lam, self.rerank)
        ranked = cand[order][:n]
        return ranked, int(ranked[0]), conf


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)
