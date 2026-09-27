"""Reciprocity, ratio intersection and input preservation."""
import unittest
import numpy as np
from src.matching.mutual_matching import mutual_nearest_neighbors, filter_mutual_ratio


class MutualTests(unittest.TestCase):
    def test_reciprocal_indices_distances(self):
        a = np.array([[0], [2], [10]], np.float32)
        b = np.array([[10.5], [.1]], np.float32)
        result = mutual_nearest_neighbors(a, b)
        self.assertEqual([(m.query_index, m.train_index) for m in result.accepted], [(0, 1), (2, 0)])
        np.testing.assert_allclose([m.distance for m in result.accepted], [.1, .5])

    def test_empty(self):
        empty, full = np.empty((0, 128), np.float32), np.ones((2, 128), np.float32)
        for a, b in ((empty, full), (full, empty), (empty, empty)):
            result = mutual_nearest_neighbors(a, b)
            self.assertEqual(result.accepted, ())
            self.assertEqual(filter_mutual_ratio(result).accepted, ())

    def test_one_sided(self):
        result = mutual_nearest_neighbors(np.array([[0], [1], [2]], np.float32), np.array([[.1]], np.float32))
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].query_index, 0)
        self.assertEqual(filter_mutual_ratio(result).accepted, ())

    def test_repeatable_ties_and_preservation(self):
        a = np.zeros((4, 128), np.float32)
        b = a.copy()
        a.flags.writeable = False
        first = mutual_nearest_neighbors(a, b)
        self.assertEqual(first.accepted, mutual_nearest_neighbors(a, b).accepted)
        self.assertEqual(len(first.accepted), 1)
        self.assertFalse(filter_mutual_ratio(first).accepted)
        np.testing.assert_array_equal(a, b)

    def test_combined(self):
        a = np.array([[0], [1], [10]], np.float32)
        b = np.array([[.1], [12]], np.float32)
        result = mutual_nearest_neighbors(a, b)
        before = result.accepted
        combined = filter_mutual_ratio(result, .75)
        self.assertEqual([(m.query_index, m.train_index) for m in combined.accepted], [(0, 0), (2, 1)])
        self.assertEqual(combined.rejected_count, 1)
        self.assertEqual(result.accepted, before)

    def test_invalid(self):
        good = np.zeros((2, 128), np.float32)
        for bad in (np.full((2, 128), np.nan, np.float32), np.zeros((2, 3), np.float32)):
            with self.assertRaises(ValueError):
                mutual_nearest_neighbors(good, bad)
        with self.assertRaises(TypeError):
            mutual_nearest_neighbors(good.astype(np.float64), good)
        with self.assertRaises(ValueError):
            filter_mutual_ratio(mutual_nearest_neighbors(good, good), 1)
