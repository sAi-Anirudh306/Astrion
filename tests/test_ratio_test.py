"""Strict Lowe ratio semantics and malformed-group handling."""
from dataclasses import replace
import unittest
import numpy as np

from src.matching.descriptor_matching import MatchCandidate, MatchingResult, match_descriptors
from src.matching.ratio_test import filter_ratio


def candidates(d1: float, d2: float) -> MatchingResult:
    return MatchingResult(((MatchCandidate(0, 0, d1, 1), MatchCandidate(0, 1, d2, 2)),),
                          1, 2, 128, 2, "test", 0)


class RatioTests(unittest.TestCase):
    def test_accept_reject_boundary(self) -> None:
        for d1, expected in ((2, 1), (3, 0), (3.5, 0)):
            result = filter_ratio(candidates(d1, 4))
            self.assertEqual(len(result.accepted), expected)
            self.assertEqual(result.rejected_count, 1 - expected)
        match = filter_ratio(candidates(2, 4)).accepted[0]
        self.assertEqual((match.query_index, match.train_index, match.nearest_distance,
                          match.second_distance, match.ratio), (0, 0, 2, 4, 0.5))

    def test_zero_distances(self) -> None:
        self.assertEqual(filter_ratio(candidates(0, 1)).accepted[0].ratio, 0)
        result = filter_ratio(candidates(0, 0))
        self.assertEqual(result.zero_second_count, 1)
        self.assertEqual(result.accepted, ())

    def test_empty_and_missing_neighbor(self) -> None:
        empty = np.empty((0, 128), np.float32)
        one = np.ones((1, 128), np.float32)
        self.assertEqual(filter_ratio(match_descriptors(empty, one)).query_count, 0)
        for train in (empty, one):
            result = filter_ratio(match_descriptors(one, train))
            self.assertEqual(result.missing_second_count, 1)
            self.assertEqual(result.rejected_count, 1)

    def test_invalid_thresholds(self) -> None:
        for threshold in (0, 1, -1, 2, np.nan, np.inf, True, "0.75", None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                filter_ratio(candidates(1, 2), threshold)

    def test_invalid_distances(self) -> None:
        for d1, d2 in ((np.nan, 2), (1, np.inf), (-1, 2), (3, 2), (1, 0), (True, 2)):
            with self.subTest(distances=(d1, d2)), self.assertRaises(ValueError):
                filter_ratio(candidates(d1, d2))

    def test_malformed_groups(self) -> None:
        base = candidates(1, 2)
        first, second = base.neighbors[0]
        invalid = (replace(base, neighbors=()), replace(base, query_count=-1),
                   replace(base, metric="Hamming"), replace(base, neighbors=([first, second],)),
                   replace(base, neighbors=((first, replace(second, train_index=0)),)),
                   replace(base, neighbors=((replace(first, query_index=1), second),)),
                   replace(base, neighbors=((first, replace(second, neighbor_rank=1)),)),
                   replace(base, neighbors=((first, replace(second, train_index=2)),)),
                   replace(base, neighbors=((first, "bad"),)))
        for item in invalid:
            with self.assertRaises(ValueError):
                filter_ratio(item)
        with self.assertRaises(TypeError):
            filter_ratio([])

    def test_matcher_compatibility_preservation(self) -> None:
        query = np.array([[0, 0], [10, 0]], np.float32)
        train = np.array([[0, 0], [10, 0], [20, 0]], np.float32)
        matches = match_descriptors(query, train)
        before = repr(matches)
        result = filter_ratio(matches)
        self.assertEqual([m.train_index for m in result.accepted], [0, 1])
        self.assertEqual(repr(matches), before)

    def test_extreme_distances(self) -> None:
        self.assertAlmostEqual(filter_ratio(candidates(1e300, 2e300)).accepted[0].ratio, 0.5)
        self.assertAlmostEqual(filter_ratio(candidates(1e-300, 2e-300)).accepted[0].ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
