"""Small CPU-only tests exercising the actual affine USAC backend."""
import unittest
from unittest.mock import patch

import numpy as np

from src.geometry.magsac import MAGSACConfig, affine_backend_capability, verify_affine


class MAGSACTests(unittest.TestCase):
    def setUp(self):
        self.a = np.random.default_rng(20).uniform(0, 100, (80, 2))
        self.matrix = np.array([[1.1, .13, 8], [-.07, .94, 12]])
        self.b = self.a @ self.matrix[:, :2].T + self.matrix[:, 2]

    def test_affine(self):
        fit = verify_affine(self.a, self.b)
        np.testing.assert_allclose(fit.matrix, self.matrix, atol=.001)

    def test_direction(self):
        fit = verify_affine(self.a, self.b)
        np.testing.assert_allclose(self.a @ fit.matrix[:, :2].T+fit.matrix[:, 2], self.b, atol=.001)

    def test_translation(self):
        fit = verify_affine(self.a, self.a+[7, -4])
        np.testing.assert_allclose(fit.matrix, [[1, 0, 7], [0, 1, -4]], atol=.001)

    def test_rotation_scale(self):
        angle = .3
        linear = 1.2*np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        fit = verify_affine(self.a, self.a@linear.T)
        np.testing.assert_allclose(fit.matrix[:, :2], linear, atol=.001)

    def test_outliers(self):
        self.b[-20:] += 100
        fit = verify_affine(self.a, self.b)
        self.assertEqual(fit.inlier_mask.sum(), 60)
        np.testing.assert_allclose(fit.matrix, self.matrix, atol=.001)

    def test_mask_length(self):
        fit = verify_affine(self.a, self.b)
        self.assertEqual(fit.inlier_mask.shape, (80,))
        self.assertEqual(fit.backend_inlier_mask.shape, (80,))

    def test_residuals(self):
        self.b[-1] += [3, 4]
        fit = verify_affine(self.a, self.b)
        manual = np.linalg.norm(self.b-self.a@fit.matrix[:, :2].T-fit.matrix[:, 2], axis=1)
        np.testing.assert_allclose(fit.errors, manual, atol=1e-12)
        self.assertAlmostEqual(fit.errors[-1], 5, places=3)

    def test_insufficient(self):
        for n in (0, 1, 2):
            with self.assertRaises(ValueError): verify_affine(self.a[:n], self.b[:n])

    def test_mismatched(self):
        with self.assertRaises(ValueError): verify_affine(self.a, self.b[:-1])

    def test_nan(self):
        self.a[0, 0] = np.nan
        with self.assertRaises(ValueError): verify_affine(self.a, self.b)

    def test_infinity(self):
        self.b[0, 0] = np.inf
        with self.assertRaises(ValueError): verify_affine(self.a, self.b)

    def test_threshold(self):
        for value in (0, -1, np.nan, np.inf, True):
            with self.assertRaises(ValueError): MAGSACConfig(reprojection_threshold=value)

    def test_confidence(self):
        for value in (0, 1, -1, np.nan, True):
            with self.assertRaises(ValueError): MAGSACConfig(confidence=value)

    def test_iterations(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): MAGSACConfig(max_iterations=value)

    def test_failure(self):
        with patch('src.geometry.magsac.cv2.estimateAffine2D', return_value=(None, None)):
            with self.assertRaises(RuntimeError): verify_affine(self.a, self.b)

    def test_original_indices_duplicates(self):
        self.b[1] += 100
        a, b = np.vstack([self.a, self.a[0]]), np.vstack([self.b, self.b[0]])
        fit = verify_affine(a, b)
        np.testing.assert_array_equal(fit.inlier_indices, np.r_[0, np.arange(2, 81)])
        self.assertEqual(fit.metadata['unique_fit_pairs'], 80)

    def test_support_consistency(self):
        self.b[-20:] += np.random.default_rng(5).normal(0, 2, (20, 2))
        fit = verify_affine(self.a, self.b)
        np.testing.assert_array_equal(fit.inlier_mask, fit.errors <= 3)

    def test_repeated(self):
        self.b[-20:] += 100
        first = verify_affine(self.a, self.b)
        for _ in range(3):
            other = verify_affine(self.a, self.b)
            np.testing.assert_array_equal(first.matrix, other.matrix)
            np.testing.assert_array_equal(first.inlier_mask, other.inlier_mask)

    def test_metadata(self):
        fit = verify_affine(self.a, self.b)
        self.assertEqual(fit.metadata['score'], 'SCORE_METHOD_MAGSAC')
        self.assertEqual(fit.metadata['local_optimization'], 'LOCAL_OPTIM_SIGMA')
        self.assertEqual(fit.metadata['model'], 'affine_2d_six_parameter')

    def test_capability(self):
        self.assertTrue(affine_backend_capability()['supported'])

    def test_degenerate(self):
        line = np.column_stack([np.arange(10), np.arange(10)])
        with self.assertRaises(ValueError): verify_affine(line, line)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError): verify_affine(np.ones((8, 3)), self.b[:8])

    def test_singular_result(self):
        with patch('src.geometry.magsac.cv2.estimateAffine2D', return_value=(np.zeros((2, 3)), np.ones((80, 1)))):
            with self.assertRaises(RuntimeError): verify_affine(self.a, self.b)

    def test_unchanged_inputs(self):
        a, b = self.a.copy(), self.b.copy()
        verify_affine(self.a, self.b)
        np.testing.assert_array_equal(a, self.a)
        np.testing.assert_array_equal(b, self.b)


if __name__ == '__main__':
    unittest.main()
