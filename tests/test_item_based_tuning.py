import unittest
import numpy as np
import pandas as pd
from src.models.item_based.model import ItemBasedCF
from src.models.item_based.tuned_model import TunedItemBasedCF

class TuningTests(unittest.TestCase):
    def setUp(self):
        self.train=pd.DataFrame({'user_id':['u','u','v','v','w'],
          'unified_game_id':['A','B','A','C','B'],'playtime_forever':[0,60,120,600,60]})
    def test_legacy_and_beta_zero(self):
        a=ItemBasedCF('playtime').fit(self.train)
        b=TunedItemBasedCF('playtime',time_unit='minutes').fit(self.train)
        np.testing.assert_allclose(a.similarity,b.similarity)
        a=ItemBasedCF().fit(self.train)
        b=TunedItemBasedCF('playtime',beta=0).fit(self.train)
        np.testing.assert_allclose(a.similarity,b.similarity)
    def test_shrinkage(self):
        a=ItemBasedCF().fit(self.train)
        b=TunedItemBasedCF(shrinkage=3).fit(self.train)
        np.testing.assert_allclose(b.similarity,a.similarity/4)
    def test_cap_and_pruning(self):
        m=TunedItemBasedCF('playtime',cap_quantile=.5,neighbors=1).fit(self.train)
        self.assertEqual(m.cap_value_,1.5)
        self.assertTrue((np.count_nonzero(m.similarity,axis=1)<=1).all())
    def test_invalid_weight_parameters(self):
        for kw in [{'beta':-1},{'shrinkage':float('nan')},{'cap_quantile':0}]:
            with self.assertRaises(ValueError):TunedItemBasedCF(**kw)

class FilterTests(unittest.TestCase):
    setUp = TuningTests.setUp
    def test_min_common_users(self):
        m=TunedItemBasedCF(min_common_users=2).fit(self.train)
        self.assertEqual(m.retained_edges_,0)
        self.assertEqual(m.zero_neighbor_games_,3)
    def test_corrected_threshold(self):
        m=TunedItemBasedCF(shrinkage=3,min_similarity=.2).fit(self.train)
        self.assertEqual(m.retained_edges_,0)
        m=TunedItemBasedCF(min_similarity=.6).fit(self.train)
        self.assertEqual(m.retained_edges_,2)
        self.assertEqual(m.raw_edges_,4)
    def test_filter_validation(self):
        for kw in [{'min_common_users':-1},{'min_common_users':1.5},{'min_similarity':1.1}]:
            with self.assertRaises(ValueError):TunedItemBasedCF(**kw)

if __name__=='__main__':unittest.main()
