import unittest
import numpy as np
import pandas as pd
from src.models.item_based.model import ItemBasedCF
from src.models.item_based.tuned_model import TunedItemBasedCF

class TuningTests(unittest.TestCase):
    def setUp(self):
        # Give A one co-owner with B and one with C. Include an owned game with
        # zero playtime so the weighting tests distinguish it from a missing pair.
        self.train=pd.DataFrame({'user_id':['u','u','v','v','w'],
          'unified_game_id':['A','B','A','C','B'],'playtime_forever':[0,60,120,600,60]})
    def test_legacy_and_beta_zero(self):
        # Minute-based tuning with default weights and no cap should reproduce the
        # original playtime cosine matrix.
        a=ItemBasedCF('playtime').fit(self.train)
        b=TunedItemBasedCF('playtime',time_unit='minutes').fit(self.train)
        np.testing.assert_allclose(a.similarity,b.similarity)
        # Beta zero makes every owned interaction weigh 1, matching binary training.
        a=ItemBasedCF().fit(self.train)
        b=TunedItemBasedCF('playtime',beta=0).fit(self.train)
        np.testing.assert_allclose(a.similarity,b.similarity)
    def test_shrinkage(self):
        a=ItemBasedCF().fit(self.train)
        # Each nonzero pair has one co-owner, so the shrinkage multiplier is 1/(1+3).
        b=TunedItemBasedCF(shrinkage=3).fit(self.train)
        np.testing.assert_allclose(b.similarity,a.similarity/4)
    def test_cap_and_pruning(self):
        m=TunedItemBasedCF('playtime',cap_quantile=.5,neighbors=1).fit(self.train)
        # Positive hours are 1, 2, 10, and 1. Their median is 1.5 hours.
        self.assertEqual(m.cap_value_,1.5)
        # The neighbor limit applies per row after the weighted cosine calculation.
        self.assertTrue((np.count_nonzero(m.similarity,axis=1)<=1).all())
    def test_invalid_weight_parameters(self):
        # Reject a negative weight, non-finite shrinkage, and a zero cap quantile.
        for kw in [{'beta':-1},{'shrinkage':float('nan')},{'cap_quantile':0}]:
            with self.assertRaises(ValueError):TunedItemBasedCF(**kw)

class FilterTests(unittest.TestCase):
    # Use the same support counts and cosine values for the filtering tests.
    setUp = TuningTests.setUp
    def test_min_common_users(self):
        # No game pair reaches two distinct co-owners, so every row loses its neighbors.
        m=TunedItemBasedCF(min_common_users=2).fit(self.train)
        self.assertEqual(m.retained_edges_,0)
        self.assertEqual(m.zero_neighbor_games_,3)
    def test_corrected_threshold(self):
        # Shrinkage lowers the strongest cosine below 0.2, so a post-correction
        # threshold should remove every edge.
        m=TunedItemBasedCF(shrinkage=3,min_similarity=.2).fit(self.train)
        self.assertEqual(m.retained_edges_,0)
        # Without shrinkage, only A and C clear 0.6. Count both directions and
        # verify that the raw graph still reports all four directed edges.
        m=TunedItemBasedCF(min_similarity=.6).fit(self.train)
        self.assertEqual(m.retained_edges_,2)
        self.assertEqual(m.raw_edges_,4)
    def test_filter_validation(self):
        # Reject negative or fractional counts and a similarity threshold above 1.
        for kw in [{'min_common_users':-1},{'min_common_users':1.5},{'min_similarity':1.1}]:
            with self.assertRaises(ValueError):TunedItemBasedCF(**kw)

if __name__=='__main__':unittest.main()
