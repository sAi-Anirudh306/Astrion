"""Unfiltered KNN matching contracts and numerical checks."""

import unittest
import numpy as np

from src.matching.descriptor_matching import match_descriptors
from src.features.rootsift import rootsift_descriptors


class DescriptorMatchingTests(unittest.TestCase):
    def test_known_neighbors_distances_indices(self) -> None:
        query = np.array([[0, 0], [9, 0]], np.float32)
        train = np.array([[3, 4], [10, 0], [0, 0]], np.float32)
        result = match_descriptors(query, train)
        self.assertEqual([(m.query_index, m.train_index, m.distance, m.neighbor_rank)
                          for m in result.neighbors[0]], [(0, 2, 0, 1), (0, 0, 5, 2)])
        self.assertEqual(result.neighbors[1][0].train_index, 1)
        self.assertAlmostEqual(result.neighbors[1][1].distance, np.sqrt(52), places=5)
        self.assertEqual(result.k, 2)
        self.assertEqual(result.metric, "L2")

    def test_empty_and_insufficient_train(self) -> None:
        empty, two = np.empty((0, 128), np.float32), np.ones((2, 128), np.float32)
        self.assertEqual(match_descriptors(empty, two).neighbors, ())
        self.assertEqual(match_descriptors(two, empty).neighbors, ((), ()))
        self.assertEqual(match_descriptors(empty, empty).neighbors, ())
        result = match_descriptors(two, two[:1])
        self.assertEqual([len(row) for row in result.neighbors], [1, 1])
        self.assertEqual(result.k, 2)

    def test_k_and_tie_repeatability(self) -> None:
        source = np.zeros((3, 128), np.float32)
        first = match_descriptors(source, source)
        self.assertEqual(first.neighbors, match_descriptors(source, source).neighbors)
        self.assertEqual([m.train_index for m in first.neighbors[0]], [0, 1])
        self.assertEqual(len(match_descriptors(source, source, 9).neighbors[0]), 3)
        self.assertEqual(len(match_descriptors(source, source, 1).neighbors[0]), 1)
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                match_descriptors(source, source, k)

    def test_invalid_arrays(self) -> None:
        valid = np.zeros((1, 128), np.float32)
        for invalid in (None, [], np.zeros(128, np.float32), np.zeros((2, 0), np.float32),
                        np.zeros((1, 128), np.float64), np.zeros((1, 128), np.uint8),
                        np.full((1, 128), np.nan, np.float32), np.full((1, 128), np.inf, np.float32),
                        np.zeros((1, 127), np.float32)):
            for query, train in ((invalid, valid), (valid, invalid)):
                with self.assertRaises((TypeError, ValueError)):
                    match_descriptors(query, train)

    def test_preservation_and_rootsift(self) -> None:
        source = np.random.default_rng(42).uniform(0, 10, (5, 128)).astype(np.float32)
        root = rootsift_descriptors(source)
        train = root[::-1]
        before = root.copy()
        root.flags.writeable = False
        train.flags.writeable = False
        result = match_descriptors(root, train)
        self.assertEqual([row[0].train_index for row in result.neighbors], [4, 3, 2, 1, 0])
        self.assertTrue(all(row[0].distance == 0 for row in result.neighbors))
        np.testing.assert_array_equal(root, before)

    def test_extreme_finite_values(self) -> None:
        limit = float(np.finfo(np.float32).max)
        query = np.array([[limit, limit]], np.float32)
        train = np.array([[-limit, -limit], [limit, 0]], np.float32)
        with np.errstate(all="raise"):
            result = match_descriptors(query, train)
        self.assertEqual(result.neighbors[0][0].train_index, 1)
        self.assertAlmostEqual(result.neighbors[0][0].distance / limit, 1)
        self.assertAlmostEqual(result.neighbors[0][1].distance / limit, 2 * np.sqrt(2))
        self.assertIn("float64", result.backend)

    def test_subnormal_distances(self) -> None:
        tiny = np.nextafter(np.float32(0), np.float32(1))
        query = np.array([[0]], np.float32)
        train = np.array([[2 * tiny], [tiny]], np.float32)
        result = match_descriptors(query, train)
        self.assertEqual(result.neighbors[0][0].train_index, 1)
        self.assertEqual(result.neighbors[0][0].distance, float(tiny))

    def test_random_against_float64_reference(self) -> None:
        rng = np.random.default_rng(8)
        query = rng.normal(size=(7, 128)).astype(np.float32)
        train = rng.normal(size=(12, 128)).astype(np.float32)
        result = match_descriptors(query, train)
        for qi, row in enumerate(result.neighbors):
            distances = np.linalg.norm(train.astype(np.float64) - query[qi], axis=1)
            expected = np.argsort(distances)[:2]
            self.assertEqual([m.train_index for m in row], expected.tolist())
            np.testing.assert_allclose([m.distance for m in row], distances[expected], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
