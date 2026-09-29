"""Synthetic spatial-control tests; no scientific files, fitting or inference."""
import json
import unittest
import numpy as np

from src.spatial.spatial_distribution import (assign_grid_cells, distribution_statistics,
    spatial_spread, select_spatially_balanced_matches)
from src.evaluation.metrics import spatial_distribution


class SpatialDistributionTests(unittest.TestCase):
    def select(self, points, **kwargs):
        return select_spatially_balanced_matches(points, points, (8, 8), grid_shape=(2, 2), **kwargs)

    def test_regular_assignment(self):
        result = assign_grid_cells([[0, 0], [4, 0], [0, 4], [7, 7]], (8, 8), (2, 2))
        np.testing.assert_array_equal(result.rows, [0, 0, 1, 1])
        np.testing.assert_array_equal(result.columns, [0, 1, 0, 1])
        np.testing.assert_array_equal(result.cells, [0, 1, 2, 3])

    def test_internal_boundaries(self):
        edge = 4.
        points = [[np.nextafter(edge, 0), 0], [edge, 0], [np.nextafter(edge, 8), 0], [0, edge]]
        np.testing.assert_array_equal(assign_grid_cells(points, (8, 8), (2, 2)).cells, [0, 1, 1, 2])

    def test_final_valid_pixel_and_fractional_boundary(self):
        points = [[320, 312], [np.nextafter(321., 0), np.nextafter(313., 0)]]
        np.testing.assert_array_equal(assign_grid_cells(points, (313, 321)).cells, [15, 15])

    def test_outside_and_nonfinite_rejected(self):
        for point in ([8, 0], [0, 8], [-.01, 0], [0, -.01], [np.nan, 1], [1, np.inf]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                assign_grid_cells([point], (8, 8))

    def test_empty_input(self):
        points = np.empty((0, 2))
        assignment = assign_grid_cells(points, (8, 8))
        self.assertEqual(assignment.cells.size, 0)
        result = self.select(points, confidence=[], residuals=[])
        self.assertEqual(result.source.shape, (0, 2))
        self.assertEqual(result.original_indices.dtype.kind, 'i')
        stats = distribution_statistics(points, (8, 8))
        self.assertEqual(stats['occupied_cells'], 0)
        self.assertEqual(stats['maximum_cell_count'], 0)
        self.assertIsNone(stats['normalized_entropy'])
        self.assertIsNone(stats['minimum_occupied_count'])
        json.dumps(stats, allow_nan=False)

    def test_invalid_dimensions(self):
        for shape in ((0, 2), (-1, 2), (2.5, 3), (True, 2), (2,), None):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                assign_grid_cells([[0, 0]], shape)
        for grid in ((0, 4), (2, False), (1.2, 4)):
            with self.assertRaises(ValueError):
                assign_grid_cells([[0, 0]], (8, 8), grid)
        with self.assertRaises(ValueError):
            distribution_statistics([[0, 0]], (8, 8), (1, 1))

    def test_counts_and_occupied_statistics(self):
        stats = distribution_statistics([[0, 0], [1, 1], [4, 4]], (8, 8), (2, 2))
        self.assertEqual(stats['counts'], [[2, 0], [0, 1]])
        self.assertEqual(stats['occupied_cells'], 2)
        self.assertEqual(stats['occupancy_percentage'], 50)
        self.assertEqual(stats['minimum_occupied_count'], 1)
        self.assertEqual(stats['maximum_cell_count'], 2)
        self.assertEqual(stats['mean_occupied_count'], 1.5)
        self.assertEqual(stats['std_occupied_count'], .5)

    def test_entropy(self):
        uniform = [[0, 0], [4, 0], [0, 4], [4, 4]]
        self.assertAlmostEqual(distribution_statistics(uniform, (8, 8), (2, 2))['normalized_entropy'], 1)
        self.assertEqual(distribution_statistics([[1, 1]]*4, (8, 8))['normalized_entropy'], 0)
        stats = distribution_statistics(uniform[:2], (8, 8), (2, 2))
        self.assertAlmostEqual(stats['normalized_entropy'], .5)

    def test_per_cell_cap(self):
        points = np.repeat([[0., 0.], [4., 0.], [0., 4.], [4., 4.]], 8, axis=0)
        result = self.select(points, max_per_cell=3)
        np.testing.assert_array_equal(np.bincount(result.cell_indices), [3]*4)
        self.assertEqual(len(result.original_indices), 12)

    def test_confidence_ordering(self):
        points = [[0, 0], [1, 0], [2, 0]]
        result = self.select(points, max_per_cell=1, confidence=[.1, .9, .5])
        np.testing.assert_array_equal(result.original_indices, [1])
        np.testing.assert_allclose(result.confidence, [.9])

    def test_residual_ordering(self):
        result = self.select([[0, 0], [1, 0], [2, 0]], max_per_cell=1, residuals=[2, 1, 3])
        np.testing.assert_array_equal(result.original_indices, [1])
        np.testing.assert_array_equal(result.residuals, [1])

    def test_lexicographic_priority_and_ties(self):
        points = [[0, 0], [1, 0], [2, 0], [3, 0]]
        result = self.select(points, max_per_cell=2, residuals=[1, 1, .5, 1], confidence=[.8, .9, .1, .9])
        np.testing.assert_array_equal(result.original_indices, [1, 2])
        tied = self.select(points, max_per_cell=2, confidence=[.5]*4, residuals=[1]*4)
        np.testing.assert_array_equal(tied.original_indices, [0, 1])

    def test_original_index_alignment_and_no_mutation(self):
        destination = np.array([[0., 0.], [4., 4.], [1., 1.], [5., 5.]])
        source = destination+10
        saved = source.copy()
        source.flags.writeable = destination.flags.writeable = False
        result = select_spatially_balanced_matches(source, destination, (8, 8), (2, 2),
            max_per_cell=1, residuals=[2, 3, 1, 2], confidence=[.1, .2, .3, .4], source_shape=(20, 20))
        np.testing.assert_array_equal(result.original_indices, [2, 3])
        np.testing.assert_array_equal(result.source, source[result.original_indices])
        np.testing.assert_array_equal(result.destination, destination[result.original_indices])
        np.testing.assert_array_equal(result.cell_indices, [0, 3])
        np.testing.assert_array_equal(result.cell_rows, [0, 1])
        np.testing.assert_array_equal(result.cell_columns, [0, 1])
        result.source[0] = 999
        np.testing.assert_array_equal(source, saved)

    def test_round_robin_global_limit(self):
        points = np.repeat([[0., 0.], [4., 0.], [0., 4.], [4., 4.]], 4, axis=0)
        result = self.select(points, max_per_cell=4, max_matches=6)
        np.testing.assert_array_equal(np.bincount(result.cell_indices, minlength=4), [2, 2, 1, 1])
        np.testing.assert_array_equal(result.original_indices, [0, 1, 4, 5, 8, 12])
        small = self.select(points, max_per_cell=4, max_matches=2)
        np.testing.assert_array_equal(small.cell_indices, [0, 1])

    def test_round_robin_skips_exhausted_cells(self):
        points = [[0, 0]]+[[4, 0]]*4+[[0, 4]]*4
        result = self.select(points, max_per_cell=3, max_matches=6)
        np.testing.assert_array_equal(np.bincount(result.cell_indices), [1, 3, 2])

    def test_zero_and_large_limits(self):
        points = [[0, 0], [1, 1]]
        for options in ({'max_per_cell': 0}, {'max_matches': 0}):
            self.assertEqual(len(self.select(points, **options).source), 0)
        np.testing.assert_array_equal(self.select(points, max_matches=999).original_indices, [0, 1])

    def test_reproducibility_and_no_score_order(self):
        points = np.array([[4, 4], [0, 0], [5, 5], [1, 1]])
        a, b = self.select(points, max_per_cell=1), self.select(points, max_per_cell=1)
        np.testing.assert_array_equal(a.original_indices, [0, 1])
        np.testing.assert_array_equal(a.original_indices, b.original_indices)
        self.assertIsNone(a.confidence)
        self.assertIsNone(a.residuals)

    def test_spread_bbox(self):
        result = spatial_spread([[1, 2], [7, 6]], (8, 10))
        self.assertEqual(result['width_fraction'], .6)
        self.assertEqual(result['height_fraction'], .5)
        self.assertEqual(result['bounding_box_min_xy'], [1, 2])
        self.assertEqual(spatial_spread([[2, 2]], (8, 8))['width_fraction'], 0)
        self.assertIsNone(spatial_spread(np.empty((0, 2)), (8, 8))['width_fraction'])

    def test_input_and_score_validation(self):
        points = [[0, 0], [1, 1]]
        for options in ({'confidence': [1]}, {'confidence': [0, 2]}, {'confidence': [np.nan, 0]},
                        {'residuals': [1]}, {'residuals': [0, -1]}, {'residuals': [np.inf, 0]},
                        {'max_per_cell': -1}, {'max_per_cell': True}, {'max_matches': 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.select(points, **options)
        for bad in ([[0, 0]], [[1, 2, 3]], [[np.nan, 0], [1, 1]], [['x', 'y'], ['x', 'y']]):
            with self.assertRaises(ValueError):
                select_spatially_balanced_matches(bad, points, (8, 8))
        with self.assertRaises(ValueError):
            select_spatially_balanced_matches([[9, 0]], [[0, 0]], (8, 8), source_shape=(8, 8))

    def test_evaluation_api_compatibility(self):
        points = [[1, 1], [5, 5], [6, 6]]
        self.assertEqual(spatial_distribution(points, (8, 8)), distribution_statistics(points, (8, 8)))


if __name__ == '__main__':
    unittest.main()
