"""
Search one query against the static gallery (answer #38: the gallery is a fixed base that can be
built in advance; each query is processed ALONE, like one frame from a camera stream).

Gallery.search(q) does, for this single query only:
  1. cosine similarity to every gallery vector
  2. k-reciprocal re-ranking of its top-K cosine candidates (allowed: answers #28, #38; not timed: #31)
  3. refusal signal "cos+gap" = cosine of the re-ranked top-1 + (that cosine - best cosine among the
     other gallery items). Chosen in experiments/threshold_study.py for the official objective
     0.7*F1 + 0.3*TNR (answer #15).
Nothing here ever looks at another query.
"""
import numpy as np

from reid.rerank import re_ranking


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
        k = min(self.topk, len(cos))
        cand = np.argsort(-cos, kind="stable")[:k]                 # cosine top-K
        if self.rerank and k > 1:
            sim = re_ranking(q[None], self.g[cand], self.k1, self.k2, self.lam)[0]
            cand = cand[np.argsort(-sim, kind="stable")]           # re-ranked top-K
        ranked = cand[:n]
        if len(ranked) < n:                                        # tiny gallery: fill by cosine order
            rest = [j for j in np.argsort(-cos, kind="stable") if j not in set(ranked)]
            ranked = np.concatenate([ranked, rest[:n - len(ranked)]]).astype(int)
        top = int(ranked[0])
        others = np.delete(cos, top)
        gap = cos[top] - (others.max() if len(others) else -1.0)
        return ranked, top, float(cos[top] + gap)
