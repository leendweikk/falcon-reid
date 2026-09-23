import numpy as np


def re_ranking(q, g, k1=6, k2=2, lam=0.3):
    """k-reciprocal re-ranking (Zhong et al. 2017). Returns a similarity (higher = better)."""
    feats = np.concatenate([q, g])
    dist = np.clip(2 - 2 * feats @ feats.T, 0, None)          # squared euclidean for unit vectors
    dist = (dist / dist.max(axis=0)).T.astype(np.float32)
    n = len(feats)
    rank = np.argsort(dist, axis=1)
    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        fwd = rank[i, :k1 + 1]
        recip = fwd[np.where(rank[fwd, :k1 + 1] == i)[0]]
        expansion = recip
        for c in recip:
            half = int(np.around(k1 / 2)) + 1
            c_fwd = rank[c, :half]
            c_recip = c_fwd[np.where(rank[c_fwd, :half] == c)[0]]
            if len(np.intersect1d(c_recip, recip)) > 2 / 3 * len(c_recip):
                expansion = np.append(expansion, c_recip)
        expansion = np.unique(expansion)
        w = np.exp(-dist[i, expansion])
        V[i, expansion] = w / w.sum()
    if k2 > 1:
        V = np.stack([V[rank[i, :k2]].mean(axis=0) for i in range(n)])
    jaccard = np.zeros((len(q), n), dtype=np.float32)
    for i in range(len(q)):
        jaccard[i] = 1 - np.minimum(V[i], V).sum(axis=1) / np.maximum(V[i], V).sum(axis=1)
    final = jaccard * (1 - lam) + dist[:len(q)] * lam
    return -final[:, len(q):]                                  # minus distance = similarity


def re_ranking_streaming(q, g, k1=6, k2=2, lam=0.3, topk=None):
    """ALLOWED version (organizers' answer 23.09): each query is re-ranked alone
    against the static gallery and never sees other queries.
    topk: re-rank only the top-k cosine candidates (fast, scalable); None = whole gallery."""
    cos = q @ g.T
    out = np.full_like(cos, -np.inf)
    for i in range(len(q)):
        idx = np.arange(len(g)) if topk is None else np.argsort(-cos[i])[:topk]
        out[i, idx] = re_ranking(q[i:i + 1], g[idx], k1, k2, lam)[0]
        rest = np.setdiff1d(np.arange(len(g)), idx)
        out[i, rest] = cos[i, rest] - 10.0          # below all re-ranked items, keep cosine order
        if (i + 1) % 200 == 0:
            print(f"  re-ranking: {i + 1}/{len(q)} queries")
    return out