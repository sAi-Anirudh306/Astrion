"""Affine recovery, outliers and input contracts."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from src.geometry.ransac import RANSACConfig, verify_affine
from src.matching.ratio_test import RatioMatch, RatioTestResult, filter_ratio
from src.matching.descriptor_matching import match_descriptors


def correspondences(n: int) -> RatioTestResult:
    return RatioTestResult(tuple(RatioMatch(i, i, 1, 4, 0.25) for i in range(n)), n, 0, 0, 0, .75, 0)


class RANSACTests(unittest.TestCase):
    def test_known_transforms_and_outliers(self) -> None:
        theta = .2
        transforms = (np.array([[1, 0, 8], [0, 1, 12]]),
            np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]]),
            np.array([[1.1, 0, 0], [0, 1.1, 0]]),
            np.array([[1.1, .12, 8], [-.08, .95, 12]]))
        a = np.random.default_rng(12).uniform(0, 500, (100, 2))
        for matrix in transforms:
            with self.subTest(matrix=matrix):
                b = a @ matrix[:, :2].T + matrix[:, 2]
                b[-20:] += [500, -400]
                before_a, before_b = a.copy(), b.copy()
                result = verify_affine(correspondences(100), a, b, RANSACConfig(reprojection_threshold=.1))
                np.testing.assert_allclose(result.matrix, matrix, atol=1e-4)
                np.testing.assert_array_equal(result.inlier_mask, np.arange(100) < 80)
                self.assertEqual(result.inlier_count, 80)
                self.assertEqual(result.outlier_count, 20)
                self.assertEqual(result.inlier_ratio, .8)
                np.testing.assert_array_equal(a, before_a)
                np.testing.assert_array_equal(b, before_b)

    def test_insufficient_points(self) -> None:
        for n in (0, 1, 2):
            with self.assertRaises(ValueError):
                verify_affine(correspondences(n), np.zeros((n, 2)), np.zeros((n, 2)))

    def test_degenerate(self) -> None:
        for points in (np.zeros((10, 2)), np.column_stack((np.arange(10), np.arange(10)))):
            with self.assertRaises(ValueError):
                verify_affine(correspondences(10), points, points)

    def test_duplicates_and_repeatability(self) -> None:
        a = np.tile(np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float), (3, 1))
        b = a + [8, 12]
        a.flags.writeable = b.flags.writeable = False
        matches = correspondences(len(a))
        before = repr(matches)
        first = verify_affine(matches, a, b)
        second = verify_affine(matches, a, b)
        np.testing.assert_array_equal(first.matrix, second.matrix)
        self.assertEqual(first.inlier_count, 12)
        self.assertEqual(repr(matches), before)

    def test_invalid_points_indices(self) -> None:
        a = np.array([[0, 0], [1, 0], [0, 1]], float)
        for bad in (np.zeros((3, 3)), np.full((3, 2), np.nan), np.full((3, 2), np.inf), np.full((3, 2), 1e30)):
            with self.assertRaises(ValueError):
                verify_affine(correspondences(3), a, bad)
        for index in (-1, 3, True, 1.5):
            match = replace(correspondences(3).accepted[0], query_index=index)
            with self.assertRaises(ValueError):
                verify_affine(replace(correspondences(3), accepted=(match,) + correspondences(3).accepted[1:]), a, a)

    def test_estimator_failure(self) -> None:
        a = np.array([[0, 0], [1, 0], [0, 1]], float)
        with patch("src.geometry.ransac.cv2.estimateAffine2D", return_value=(None, None)):
            with self.assertRaises(RuntimeError):
                verify_affine(correspondences(3), a, a)

    def test_ratio_compatibility(self) -> None:
        descriptors = np.eye(10, 128, dtype=np.float32)
        ratio = filter_ratio(match_descriptors(descriptors, descriptors))
        points = np.random.default_rng(3).uniform(0, 100, (10, 2))
        result = verify_affine(ratio, points, points + [8, 12])
        self.assertEqual(result.inliers, ratio.accepted)
        self.assertLess(result.inlier_error_statistics["max"], 1e-4)

    def test_invalid_config(self) -> None:
        for kwargs in ({"confidence": 1}, {"confidence": np.nan}, {"max_iterations": 0},
                       {"refinement_iterations": -1}, {"seed": True}, {"reprojection_threshold": 0}):
            with self.assertRaises(ValueError):
                RANSACConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
