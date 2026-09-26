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

order_candidates() (steps 2-3, plus the optional two-stage re-ordering) is shared by the batch run
(Gallery, in memory) and the web service (candidates come from the pgvector database), so both give
identical answers.
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


def order_candidates(q, cand_vecs, cand_cos, q_rescore=None, cand_rescore=None,
                     k1=6, k2=2, lam=0.3, rerank=True):
    """Order ONE query's cosine top-K candidates (sorted by fast cosine, best first).
    Single stage: k-reciprocal re-ranking of the fast vectors.
    Two-stage (answer #28): the heavier rescore vectors (q_rescore, cand_rescore) re-order ONLY these
    candidates, and re-ranking + the refusal signal are computed from them.
    Returns (order into the candidates, cos+gap confidence, the cosines shown to the user)."""
    if cand_rescore is None:
        order, conf = rank_candidates(q, cand_vecs, cand_cos, k1, k2, lam, rerank)
        return order, conf, cand_cos
    q2 = _unit(q_rescore[None])[0]
    g2 = _unit(cand_rescore)
    c2 = g2 @ q2
    pre = np.argsort(-c2, kind="stable")                       # rank_candidates expects cosine order
    order, conf = rank_candidates(q2, g2[pre], c2[pre], k1, k2, lam, rerank)
    return pre[order], conf, c2


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
        order, conf, _ = order_candidates(q, self.g[cand], cos[cand], q_rescore,
                                          None if self.g2 is None else self.g2[cand],
                                          self.k1, self.k2, self.lam, self.rerank)
        ranked = cand[order][:n]
        return ranked, int(ranked[0]), conf


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)
