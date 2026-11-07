"""Steam item cosine similarity; binary histories for unseen evaluation users."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ItemBasedCF:
    def __init__(self, weighting="binary", neighbors=0):
        if weighting not in {"binary", "playtime"}:
            raise ValueError("weighting must be binary or playtime")
        if neighbors < 0:
            raise ValueError("neighbors must be nonnegative; 0 means all")
        self.weighting = weighting
        self.neighbors = neighbors

    def fit(self, train):
        required = {"user_id", "unified_game_id"}
        if self.weighting == "playtime":
            required.add("playtime_forever")
        if not required.issubset(train.columns):
            raise ValueError(f"Missing columns: {required - set(train.columns)}")
        if train.empty or train[list(required)].isna().any().any():
            raise ValueError("Training data is empty or has null required values")
        if train.duplicated(["user_id", "unified_game_id"]).any():
            raise ValueError("Duplicate user-game pairs; use the cleaned training data")

        # One shared, deterministic item index for both training signals.
        self.items = np.array(sorted(train["unified_game_id"].astype(str).unique()))
        self.item_to_index = {item: i for i, item in enumerate(self.items)}
        users, user_ids = pd.factorize(train["user_id"].astype(str), sort=True)
        games = train["unified_game_id"].astype(str).map(self.item_to_index).to_numpy()
        weights = np.ones(len(train), dtype=np.float32)
        if self.weighting == "playtime":
            minutes = pd.to_numeric(train["playtime_forever"], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(minutes).all() or (minutes < 0).any():
                raise ValueError("Playtime must be finite, nonnegative minutes")
            # Retain zero-minute ownership with weight 1; unowned cells stay 0.
            weights = (1.0 + np.log1p(minutes)).astype(np.float32)

        # Rows are Steam users and columns are games. Missing ownership stays zero.
        x = csr_matrix((weights, (users, games)), shape=(len(user_ids), len(self.items)))
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel())
        normalized = x.multiply(1.0 / norms).tocsr()
        # Dot products of unit-length game columns give pairwise cosine similarities.
        self.similarity = (normalized.T @ normalized).toarray().astype(np.float32)
        np.fill_diagonal(self.similarity, 0.0)

        # Row i keeps the strongest outgoing neighbors of observed game i.
        if self.neighbors:
            for i, row in enumerate(self.similarity):
                order = np.argsort(-row, kind="stable")
                row[order[self.neighbors:]] = 0.0
        return self

    def recommend_many(self, histories, k=10):
        """Score a bounded batch; ties follow ascending unified game ID."""
        if not hasattr(self, "similarity"):
            raise ValueError("Call fit before recommend")
        if k <= 0:
            raise ValueError("k must be positive")
        if not histories:
            return []
        rows, cols, observed = [], [], []
        for r, history in enumerate(histories):
            unique = sorted(set(map(str, history)))
            unknown = set(unique) - self.item_to_index.keys()
            if unknown:
                raise ValueError(f"Games outside training catalog: {sorted(unknown)[:5]}")
            if not unique:
                raise ValueError("An evaluation user must have observed games")
            indices = [self.item_to_index[item] for item in unique]
            observed.append(indices)
            rows.extend([r] * len(indices))
            cols.extend(indices)
        # Use binary ownership at inference so PlayStation users need no playtime data.
        histories_matrix = csr_matrix(
            (np.ones(len(cols), dtype=np.float32), (rows, cols)),
            shape=(len(histories), len(self.items)),
        )
        # Sum each candidate's similarity to the games in the user's history.
        scores = histories_matrix @ self.similarity
        recommendations = []
        for r, seen in enumerate(observed):
            # Mask observed games so they cannot appear in the top-k results.
            scores[r, seen] = -np.inf
            order = np.argsort(-scores[r], kind="stable")
            order = order[np.isfinite(scores[r, order])][:k]
            recommendations.append(self.items[order].tolist())
        return recommendations

    def recommend(self, observed_games, k=10):
        return self.recommend_many([observed_games], k=k)[0]
