"""Optional training-only extensions; the original ItemBasedCF is unchanged."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.item_based.model import ItemBasedCF


class TunedItemBasedCF(ItemBasedCF):
    def __init__(self, weighting='binary', neighbors=0, shrinkage=0,
                 beta=1, cap_quantile=None, time_unit='hours',
                 min_common_users=0, min_similarity=0):
        # The base class validates shared options and provides the ranking logic.
        super().__init__(weighting, neighbors)
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError('shrinkage must be finite and nonnegative')
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if cap_quantile is not None and not 0 < cap_quantile <= 1:
            raise ValueError('cap_quantile must be in (0, 1]')
        if time_unit not in {'hours', 'minutes'}:
            raise ValueError('time_unit must be hours or minutes')
        self.shrinkage, self.beta = shrinkage, beta
        self.cap_quantile, self.time_unit = cap_quantile, time_unit
        # Support is an integer co-owner count; similarity is a normalized score.
        if not isinstance(min_common_users, (int, np.integer)) or min_common_users < 0:
            raise ValueError('min_common_users must be a nonnegative integer')
        if not np.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
            raise ValueError('min_similarity must be in [0, 1]')
        self.min_common_users = min_common_users
        self.min_similarity = min_similarity

    def fit(self, train):
        # Validate with the original model before applying any transform.
        original_neighbors = self.neighbors
        # Defer neighbor pruning until after weighting and support corrections.
        self.neighbors = 0
        try:
            # Reuse base validation and catalog indexing. Binary mode also keeps
            # the base cosine matrix as the starting point for its corrections.
            super().fit(train)
            self.cap_value_ = None
            if self.weighting == 'playtime':
                times = pd.to_numeric(train.playtime_forever).to_numpy(dtype=float)
                # The input duration is recorded in minutes. Choose the unit before
                # calculating the cap and applying log1p so both use the same scale.
                if self.time_unit == 'hours':
                    times = times / 60
                # Estimate the cap from played games only, using Steam training data.
                positive = times[times > 0]
                if self.cap_quantile is not None and positive.size:
                    self.cap_value_ = float(np.quantile(positive, self.cap_quantile))
                    # Cap unusually long durations without removing ownership rows.
                    times = np.minimum(times, self.cap_value_)
                # Beta scales the time contribution. When beta is zero, every owned
                # pair has weight 1 and yields the same cosine as binary ownership.
                # Construct the weighted matrix directly; zero-minute owners stay 1.
                values = (1 + self.beta * np.log1p(times)).astype(np.float32)
                rows, users = pd.factorize(train.user_id.astype(str), sort=True)
                cols = train.unified_game_id.astype(str).map(self.item_to_index).to_numpy()
                x = csr_matrix((values, (rows, cols)), shape=(len(users), len(self.items)))
                # Normalize each weighted game column before computing cosine.
                norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel())
                normalized = x.multiply(1 / norms).tocsr()
                self.similarity = (normalized.T @ normalized).toarray().astype(np.float32)
                np.fill_diagonal(self.similarity, 0)
            # Count nonzero similarities before applying filters or limiting neighbors.
            self.raw_edges_ = int(np.count_nonzero(self.similarity))
            # Build co-ownership support only when one of the support-based options is active.
            if self.shrinkage or self.min_common_users:
                rows, users = pd.factorize(train.user_id.astype(str), sort=True)
                cols = train.unified_game_id.astype(str).map(self.item_to_index).to_numpy()
                owned = csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)),
                                   shape=(len(users), len(self.items)))
                # Count co-owners independently of the playtime weights.
                support = (owned.T @ owned).toarray()
                # Remove game pairs with too few shared owners outright.
                if self.min_common_users:
                    self.similarity[support < self.min_common_users] = 0
                if self.shrinkage:
                    # Give less weight to similarities based on few co-owners.
                    # The multiplier approaches 1 as support grows relative to shrinkage.
                    self.similarity *= support / (support + self.shrinkage)
            # Threshold refers to the corrected similarity, before top-N pruning.
            self.similarity[self.similarity < self.min_similarity] = 0
            # Keep the strongest corrected similarities per observed game.
            if original_neighbors:
                for row in self.similarity:
                    order = np.argsort(-row, kind='stable')
                    row[order[original_neighbors:]] = 0
            # These diagnostics show how much pruning reduced candidate connections.
            self.retained_edges_ = int(np.count_nonzero(self.similarity))
            self.zero_neighbor_games_ = int((np.count_nonzero(self.similarity, axis=1) == 0).sum())
        finally:
            # Restore the configured limit even if validation or fitting fails.
            self.neighbors = original_neighbors
        return self
