import unittest

import numpy as np
import pandas as pd

from src.models.item_based.model import ItemBasedCF
from src.models.item_based.evaluate import evaluate


class ItemBasedTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2", "u2", "u3"],
            "unified_game_id": ["A", "B", "A", "B", "C", "C"],
            "playtime_forever": [0, 10, 20, 0, 30, 0],
        })

    def test_cosine_and_exclusion(self):
        model = ItemBasedCF().fit(self.train)
        np.testing.assert_allclose(model.similarity, [[0, 1, .5], [1, 0, .5], [.5, .5, 0]], atol=1e-6)
        self.assertEqual(model.recommend(["A"], 10), ["B", "C"])
        self.assertEqual(model.recommend(["C"], 1), ["A"])

    def test_playtime_and_zero_ownership(self):
        model = ItemBasedCF("playtime").fit(self.train)
        x = np.array([[1, 1 + np.log1p(10), 0],
                      [1 + np.log1p(20), 1, 1 + np.log1p(30)], [0, 0, 1]])
        x /= np.linalg.norm(x, axis=0)
        expected = x.T @ x
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.similarity, expected, atol=1e-6)

    def test_evaluation_known_answer(self):
        model = ItemBasedCF().fit(self.train)
        observed = pd.DataFrame({"user_id": ["v"], "unified_game_id": ["A"]})
        heldout = pd.DataFrame({"user_id": ["v"], "unified_game_id": ["B"]})
        summary, per_user = evaluate(model, observed, heldout, k=1)
        self.assertEqual(summary["recall@1"], 1)
        self.assertEqual(summary["ndcg@1"], 1)
        self.assertEqual(summary["hit_rate@1"], 1)
        with self.assertRaises(ValueError):
            evaluate(model, observed, observed)

    def test_batch_and_neighbors(self):
        model = ItemBasedCF(neighbors=1).fit(self.train)
        self.assertTrue((np.count_nonzero(model.similarity, axis=1) <= 1).all())
        histories = [["A"], ["C"], ["A", "B"]]
        self.assertEqual(model.recommend_many(histories), [model.recommend(x) for x in histories])

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            ItemBasedCF().fit(pd.concat([self.train, self.train.iloc[:1]]))
        model = ItemBasedCF().fit(self.train)
        with self.assertRaises(ValueError):
            model.recommend(["unknown"])
        with self.assertRaises(ValueError):
            model.recommend([])


if __name__ == "__main__":
    unittest.main()
