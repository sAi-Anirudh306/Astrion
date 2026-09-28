"""Synthetic evaluation contracts; no learned model or inference dependency."""
import json
import unittest
import numpy as np

from src.evaluation.metrics import (affine_diagnostics, correspondence_statistics,
    coverage_statistics, image_similarity, pixel_to_map_distance,
    reprojection_residuals, residual_statistics, residual_thresholds, spatial_distribution)


class EvaluationTests(unittest.TestCase):
    def test_known_forward_residuals(self):
        moving = np.array([[0, 0], [1, 2]], float)
        matrix = np.array([[2, 0, 10], [0, 3, -2]], float)
        reference = np.array([[10, -2], [15, 8]], float)
        np.testing.assert_array_equal(reprojection_residuals(moving, reference, matrix), [0, 5])
        np.testing.assert_array_equal(moving, [[0, 0], [1, 2]])

    def test_exact_statistics(self):
        stats = residual_statistics([0, 3, 4])
        self.assertEqual(stats['rmse'], np.sqrt(25/3))
        self.assertEqual(stats['mean'], 7/3)
        self.assertEqual(stats['median'], 3)
        self.assertEqual(stats['minimum'], 0)
        self.assertEqual(stats['maximum'], 4)
        self.assertAlmostEqual(stats['std'], np.sqrt(26/9))
        self.assertAlmostEqual(stats['p90'], 3.8)
        self.assertAlmostEqual(stats['p95'], 3.9)

    def test_thresholds_inclusive(self):
        result = residual_thresholds([0, .5, 1, 2, 3, 4], 100)
        self.assertEqual([r['count'] for r in result], [2, 3, 4, 5])
        self.assertEqual([r['nominal_map_metres'] for r in result], [50, 100, 200, 300])
        self.assertEqual(result[1]['percentage'], 50)

    def test_map_conversion(self):
        np.testing.assert_allclose(pixel_to_map_distance([.5, 1.164, 3], 100), [50, 116.4, 300], rtol=1e-14)
        for scale in (0, -1, np.nan, np.inf, True, [100]):
            with self.assertRaises(ValueError):
                pixel_to_map_distance([1], scale)

    def test_correspondence_counts(self):
        self.assertEqual(correspondence_statistics([1, 0, 1]),
                         dict(candidate_count=3, inlier_count=2, outlier_count=1, inlier_ratio=2/3))

    def test_grid_boundaries_and_occupancy(self):
        result = spatial_distribution([[0, 0], [4, 0], [7.99, 7.99]], (8, 8), (2, 2))
        self.assertEqual(result['counts'], [[1, 1], [0, 1]])
        self.assertEqual(result['occupied_cells'], 3)
        self.assertEqual(result['occupancy_percentage'], 75)
        self.assertEqual(result['maximum_cell_count'], 1)
        self.assertEqual(result['minimum_occupied_count'], 1)
        self.assertEqual(result['mean_occupied_count'], 1)
        self.assertEqual(result['std_occupied_count'], 0)
        self.assertAlmostEqual(result['normalized_entropy'], np.log(3)/np.log(4))

    def test_entropy_uniform_and_concentrated(self):
        xy = np.array([(x, y) for x in range(4) for y in range(4)])
        self.assertAlmostEqual(spatial_distribution(xy, (4, 4))['normalized_entropy'], 1)
        self.assertEqual(spatial_distribution(np.zeros((16, 2)), (4, 4))['normalized_entropy'], 0)

    def test_coverage(self):
        result = coverage_statistics([[1, 1], [0, 0]], [[1, 0], [1, 1]])
        self.assertEqual(result['total_output_pixels'], 4)
        self.assertEqual(result['valid_registered_pixels'], 2)
        self.assertEqual(result['valid_reference_pixels'], 3)
        self.assertEqual(result['common_valid_pixels'], 1)
        self.assertEqual(result['registered_coverage_percentage'], 50)
        self.assertEqual(result['common_output_percentage'], 25)
        self.assertAlmostEqual(result['common_reference_percentage'], 100/3)
        self.assertIsNone(coverage_statistics([[0]], [[0]])['common_reference_percentage'])

    def test_affine_diagnostics(self):
        result = affine_diagnostics([[0, -2, 20], [3, 0, 50]])
        self.assertAlmostEqual(result['determinant'], 6)
        np.testing.assert_allclose(result['singular_values'], [3, 2], rtol=1e-14)
        self.assertAlmostEqual(result['condition_number'], 1.5)
        self.assertAlmostEqual(result['area_scale_factor'], 6)
        self.assertLess(affine_diagnostics([[-1, 0, 0], [0, 1, 0]])['determinant'], 0)
        self.assertIsNone(affine_diagnostics([[0, 0, 0], [0, 1, 0]])['condition_number'])

    def test_empty_correspondences(self):
        residuals = reprojection_residuals(np.empty((0, 2)), np.empty((0, 2)), [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(residuals.size, 0)
        self.assertTrue(all(x is None for x in residual_statistics(residuals).values()))
        self.assertIsNone(correspondence_statistics([])['inlier_ratio'])
        self.assertIsNone(residual_thresholds([], 100)[0]['percentage'])
        spatial = spatial_distribution(np.empty((0, 2)), (8, 8))
        self.assertEqual(spatial['occupied_cells'], 0)
        self.assertIsNone(spatial['normalized_entropy'])
        json.dumps(spatial, allow_nan=False)

    def test_invalid_residuals_and_points(self):
        for bad in ([np.nan], [np.inf], [-1], [[1]], ['1']):
            with self.assertRaises(ValueError):
                residual_statistics(bad)
        for points in ([[np.nan, 0]], [[0, np.inf]], [[-1, 0]], [[8, 0]], [[0, 8]], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                spatial_distribution(points, (8, 8))
        with self.assertRaises(ValueError):
            reprojection_residuals([[0, 0]], [], [[1, 0, 0], [0, 1, 0]])
        with self.assertRaises(ValueError):
            reprojection_residuals([[np.nan, 0]], [[0, 0]], [[1, 0, 0], [0, 1, 0]])

    def test_invalid_masks_shapes_and_affine(self):
        for mask in ([[np.nan]], [[2]], [[-1]]):
            with self.assertRaises(ValueError):
                coverage_statistics(mask, [[1]])
        for shape in ((0, 2), (2.5, 3), (True, 3)):
            with self.assertRaises(ValueError):
                spatial_distribution(np.empty((0, 2)), shape)
        for affine in (np.eye(3), [[np.nan, 0, 0], [0, 1, 0]]):
            with self.assertRaises(ValueError):
                affine_diagnostics(affine)
        with self.assertRaises(ValueError):
            correspondence_statistics([1, np.nan])

    def test_similarity_common_mask_and_finite(self):
        a = np.array([[0, 1, 2], [np.nan, 999, 4]], float)
        b = np.array([[0, 2, 4], [6, -999, np.inf]], float)
        mask = np.array([[1, 1, 1], [1, 0, 1]], bool)
        result = image_similarity(a, b, mask, np.ones_like(mask))
        self.assertEqual(result['common_valid_pixels'], 3)
        self.assertAlmostEqual(result['correlation'], 1)
        self.assertEqual(result['percentile_bounds_moving_reference'], [[.02, 1.98], [.04, 3.96]])
        b[1, 1] = 1e30
        self.assertEqual(result, image_similarity(a, b, mask, np.ones_like(mask)))
        self.assertTrue(np.isnan(a[1, 0]))

    def test_similarity_undefined(self):
        a = np.ones((3, 3))
        for mask in (np.ones((3, 3), bool), np.zeros((3, 3), bool), np.eye(3, dtype=bool)):
            result = image_similarity(a, a, mask, mask)
            self.assertIsNone(result['correlation'])
            self.assertIsNotNone(result['correlation_undefined_reason'])
            json.dumps(result, allow_nan=False)

    def test_similarity_negative_and_shape_errors(self):
        a = np.arange(9).reshape(3, 3)
        mask = np.ones_like(a, bool)
        self.assertAlmostEqual(image_similarity(a, -a, mask, mask)['correlation'], -1)
        for b, m in ((a[:2], mask), (a, mask[:2]), (a, np.full((3, 3), np.nan))):
            with self.assertRaises(ValueError):
                image_similarity(a, b, m, mask)


if __name__ == '__main__':
    unittest.main()
