"""
Shared Top-K ranking metrics for recommendation experiments.
"""

from math import log2

def ranking_metrics(recommended: list[str], heldout: set[str], k: int = 10) -> dict:
    """
    Evaluate an ordered Top-K list against one user's hidden games.
    """

    if not heldout:
        raise ValueError("A user must have at least one held-out game")
    if k <= 0:
        raise ValueError("k must be positive")

    ranked = recommended[:k]
    hits = sum(game in heldout for game in ranked) # correct recommendations
    dcg = sum(1.0 / log2(rank + 2) for rank, game in enumerate(ranked) if game in heldout) # Discounted Cumulative Gain
    idcg = sum(1.0 / log2(rank + 2) for rank in range(min(len(heldout), k))) # ideal Discounted Cumulative Gain

    return {
        f"recall@{k}": hits / len(heldout),
        f"ndcg@{k}": dcg / idcg,
        f"hit_rate@{k}": float(hits > 0)
    }
