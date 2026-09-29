"""Visualization contracts using small synthetic arrays; no model inference."""
import tempfile
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from src.evaluation.visualization import (display_image, show_display, plot_image,
    plot_keypoints, select_match_indices, plot_matches, overlay_image, checkerboard_image,
    normalized_difference, plot_registration, plot_residuals, plot_spatial_distribution,
    plot_metrics_panel)
from scripts.tycho_visualization import load_artifacts, run


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(64, dtype=np.float32).reshape(8, 8)
        self.mask = np.ones((8, 8), bool)
        self.points = np.array([[0., 0.], [3., 4.], [7., 7.]])
        self.identity = np.array([[1., 0., 0.], [0., 1., 0.]])

    def tearDown(self):
        plt.close("all")

    def test_uint8_display(self):
        image = self.image.astype(np.uint8)
        result = display_image(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 1)
        self.assertFalse(np.shares_memory(result.data, image))

    def test_uint16_display(self):
        image = (self.image*1000).astype(np.uint16)
        saved = image.copy()
        result = display_image(image)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.max(), 1)
        np.testing.assert_array_equal(image, saved)

    def test_float32_scientific_display(self):
        image = self.image*25-500
        image.flags.writeable = False
        result = display_image(image)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)
        np.testing.assert_array_equal(image, self.image*25-500)

    def test_nodata_nan_mask_and_valid_black(self):
        image = np.array([[0., 1., 2.], [-999., np.nan, np.inf]], dtype=np.float32)
        result = display_image(image, nodata=-999)
        np.testing.assert_array_equal(result.mask, [[False]*3, [True]*3])
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], .5)
        self.assertEqual(result[0, 2], 1)
        masked = np.ma.array(image, mask=[[1, 0, 0], [1, 1, 1]])
        self.assertTrue(display_image(masked).mask[0, 0])
        self.assertTrue(np.isnan(image[1, 1]))

    def test_empty_valid_and_constant_display(self):
        self.assertTrue(display_image(self.image, np.zeros_like(self.mask)).mask.all())
        self.assertEqual(display_image(np.ones((3, 3))).max(), 0)
        fig, ax = plt.subplots()
        plot_image(ax, self.image, np.zeros_like(self.mask))
        fig.canvas.draw()

    def test_display_input_validation(self):
        for image in ([], np.zeros((1, 1, 3)), np.array([['x']]), np.array([[1j]])):
            with self.assertRaises(ValueError):
                display_image(image)
        for bounds in ((99, 1), (-1, 99), (1, 101), (1, np.nan)):
            with self.assertRaises(ValueError):
                display_image(self.image, percentiles=bounds)
        for mask in (np.ones((2, 2)), np.full((8, 8), 2), np.full((8, 8), np.nan)):
            with self.assertRaises(ValueError):
                display_image(self.image, mask)

    def test_deterministic_sampling(self):
        np.testing.assert_array_equal(select_match_indices(10, 4), [0, 3, 6, 9])
        np.testing.assert_array_equal(select_match_indices(10, 4), select_match_indices(10, 4))
        np.testing.assert_array_equal(select_match_indices(3, None), [0, 1, 2])
        self.assertEqual(select_match_indices(0).size, 0)
        for cap in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                select_match_indices(10, cap)

    def test_matches_preserve_coordinates_and_total_counts(self):
        fig, ax = plt.subplots()
        original = self.points.copy()
        chosen = plot_matches(ax, self.image, self.image, self.points, self.points,
                              inlier_mask=[1, 0, 1], max_matches=2)
        np.testing.assert_array_equal(chosen, [0, 2])
        np.testing.assert_array_equal(self.points, original)
        lines = next(item for item in ax.collections if isinstance(item, LineCollection))
        np.testing.assert_array_equal(lines.get_segments()[1], [[7, 7], [15, 7]])
        self.assertIn('3 candidates', ax.get_title(loc='left'))
        self.assertIn('2 inliers', ax.get_title(loc='left'))
        fig.canvas.draw()

    def test_empty_matches(self):
        empty = np.empty((0, 2))
        fig, ax = plt.subplots()
        indices = plot_matches(ax, self.image, self.image, empty, empty,
                               inlier_mask=np.empty(0, bool))
        self.assertEqual(indices.size, 0)
        fig.canvas.draw()

    def test_match_mask_and_confidence_validation(self):
        _, ax = plt.subplots()
        for mask in ([1], [1, 2, 0], [1, np.nan, 0]):
            with self.assertRaises(ValueError):
                plot_matches(ax, self.image, self.image, self.points, self.points, inlier_mask=mask)
        for scores in ([.5], [0, 1, np.nan], [-1, .5, 1], [1, 2, 3]):
            with self.assertRaises(ValueError):
                plot_matches(ax, self.image, self.image, self.points, self.points, confidence=scores)
        with self.assertRaises(ValueError):
            plot_matches(ax, self.image, self.image, self.points, self.points[:2])

    def test_overlay_dimensions_and_mask(self):
        mask = self.mask.copy()
        mask[0, 0] = False
        a, b = self.image.copy(), self.image[::-1].copy()
        result = overlay_image(a, b, mask, self.mask)
        self.assertEqual(result.shape, self.image.shape)
        self.assertTrue(result.mask[0, 0])
        expected = .5*display_image(a, mask)+.5*display_image(b, mask)
        np.testing.assert_allclose(result.compressed(), expected.compressed())
        np.testing.assert_array_equal(a, self.image)
        with self.assertRaises(ValueError):
            overlay_image(a, b[:2])
        with self.assertRaises(ValueError):
            overlay_image(a, b, alpha=1.5)

    def test_checkerboard_blocks_and_dimensions(self):
        reference = self.image[::-1]
        result = checkerboard_image(self.image, reference, block_size=2)
        a, b = display_image(self.image), display_image(reference)
        self.assertEqual(result.shape, (8, 8))
        np.testing.assert_allclose(result[:2, :2], a[:2, :2])
        np.testing.assert_allclose(result[:2, 2:4], b[:2, 2:4])
        np.testing.assert_allclose(result[2:4, :2], b[2:4, :2])
        mask = self.mask.copy()
        mask[2, 2] = False
        self.assertTrue(checkerboard_image(self.image, reference, mask, self.mask).mask[2, 2])
        for size in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                checkerboard_image(self.image, reference, block_size=size)

    def test_difference_common_normalization(self):
        mask = self.mask.copy()
        mask[0, 0] = False
        a, b = self.image.copy(), self.image*3+20
        a[0, 0] = 1e8
        result = normalized_difference(a, b, mask, self.mask)
        self.assertTrue(result.mask[0, 0])
        np.testing.assert_allclose(result.compressed(), 0, atol=1e-14)

    def test_keypoint_scale_and_orientation(self):
        fig, ax = plt.subplots()
        plot_keypoints(ax, self.image, self.points, scales=[2, 4, 6], orientations=[0, 90, 180])
        self.assertEqual(len(ax.patches), 3)
        np.testing.assert_array_equal(ax.patches[1].center, self.points[1])
        self.assertEqual(ax.patches[1].radius, 2)
        fig.canvas.draw()
        with self.assertRaises(ValueError):
            plot_keypoints(ax, self.image, self.points, scales=[1, -1, 1])
        with self.assertRaises(ValueError):
            plot_keypoints(ax, self.image, self.points, orientations=[0, -1, 90])

    def test_residual_values_direction_and_scaling(self):
        src = np.array([[1., 2.], [2., 3.]])
        dst = np.array([[4., 6.], [2., 3.]])
        fig, ax = plt.subplots()
        residuals = plot_residuals(ax, self.image, src, dst, self.identity, vectors=True, arrow_scale=2)
        np.testing.assert_array_equal(residuals, [5, 0])
        arrows = ax.collections[0]
        np.testing.assert_array_equal(arrows.U, [6, 0])
        np.testing.assert_array_equal(arrows.V, [8, 0])
        np.testing.assert_array_equal(arrows.X, src[:, 0])
        np.testing.assert_array_equal(arrows.get_array(), residuals)
        fig.canvas.draw()

    def test_residual_input_validation_and_empty(self):
        _, ax = plt.subplots()
        for points in ([[np.nan, 0]], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                plot_residuals(ax, self.image, points, [[1, 2]], self.identity)
        for scale in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                plot_residuals(ax, self.image, self.points, self.points, self.identity, arrow_scale=scale)
        with self.assertRaises(ValueError):
            plot_residuals(ax, self.image, self.points, self.points, np.eye(3))
        result = plot_residuals(ax, self.image, np.empty((0, 2)), np.empty((0, 2)), self.identity)
        self.assertEqual(result.size, 0)

    def test_registration_spatial_and_metrics_render(self):
        fig, axes = plt.subplots(2, 2)
        plot_registration(axes, self.image, self.image, self.image, self.mask, self.mask, self.mask)
        self.assertTrue(all(len(ax.images) == 1 for ax in axes.ravel()))
        fig.canvas.draw()
        fig, axes = plt.subplots(1, 2)
        stats = plot_spatial_distribution(axes[0], self.image, self.points)
        self.assertEqual(sum(map(sum, stats['counts'])), 3)
        plot_metrics_panel(axes[1], {'Feature RMSE': '1.164 px'})
        fig.canvas.draw()

    def test_missing_artifacts_and_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, 'evaluation report missing'):
                load_artifacts(Path(temp))
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(FileExistsError, 'overwrite'):
            run(root/'results/milestone_09_evaluation')
        with self.assertRaisesRegex(ValueError, 'under project results'):
            run(root/'data/raw/forbidden_output')


if __name__ == '__main__':
    unittest.main()
