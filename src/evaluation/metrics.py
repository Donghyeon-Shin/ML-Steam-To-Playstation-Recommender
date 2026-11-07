"""
Shared Top-K ranking metrics for recommendation experiments.
"""

from math import log2

def ranking_metrics(recommended: list[str], heldout: set[str], k: int = 10) -> dict:
    """
    Evaluate an ordered Top-K list against one user's hidden games.
    """

    # Recall and ideal DCG require at least one relevant game for this user.
    if not heldout:
        raise ValueError("A user must have at least one held-out game")
    if k <= 0:
        raise ValueError("k must be positive")

    # Apply the same cutoff to the actual ranking and the ideal ranking below.
    ranked = recommended[:k]
    hits = sum(game in heldout for game in ranked)
    # Earlier hits receive more credit. With zero-based rank, rank + 2 gives
    # a weight of 1 at the first position and decreasing weights after it.
    dcg = sum(1.0 / log2(rank + 2) for rank, game in enumerate(ranked) if game in heldout)
    # The ideal list puts every available hit first, up to the cutoff k.
    idcg = sum(1.0 / log2(rank + 2) for rank in range(min(len(heldout), k)))

    # Recall measures held-out coverage, NDCG divides by the ideal score, and
    # hit rate is a per-user 0/1 indicator that callers average across users.
    return {
        f"recall@{k}": hits / len(heldout),
        f"ndcg@{k}": dcg / idcg,
        f"hit_rate@{k}": float(hits > 0)
    }
