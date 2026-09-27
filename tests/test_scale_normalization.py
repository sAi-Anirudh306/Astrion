"""Focused physical-resolution grid and area-averaging tests."""
import unittest
import numpy as np
from src.preprocessing.scale_normalization import normalize_resolution


class ScaleTests(unittest.TestCase):
    def test_area_average_and_preservation(self):
        image = (np.indices((12, 8)).sum(axis=0) % 2).astype(np.float32)
        original = image.copy()
        image.flags.writeable = False
        result = normalize_resolution(image, 5, 10)
        np.testing.assert_allclose(result.data, .5)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(result.data.shape, (6, 4))
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(result.effective_resolution_xy, (10, 10))

    def test_rounding_and_identity(self):
        image = np.full((11, 7), .3, np.float32)
        result = normalize_resolution(image, 5.15, 10)
        self.assertEqual(result.data.shape, (6, 4))
        self.assertAlmostEqual(result.nominal_factor, .515)
        self.assertAlmostEqual(result.effective_resolution_xy[0], 5.15 * 7 / 4)
        identity = normalize_resolution(image, 5, 5)
        self.assertFalse(np.shares_memory(identity.data, image))
        np.testing.assert_array_equal(identity.data, image)

    def test_invalid_images(self):
        for image in (np.empty((0, 2), np.float32), np.zeros((2, 2, 2), np.float32),
                      np.array([[np.nan]], np.float32), np.array([[np.inf]], np.float32),
                      np.array([[-.1]], np.float32), np.array([[1.1]], np.float32)):
            with self.assertRaises(ValueError):
                normalize_resolution(image, 5, 10)
        with self.assertRaises(TypeError):
            normalize_resolution(np.zeros((2, 2), np.uint16), 5, 10)

    def test_invalid_resolutions(self):
        image = np.ones((8, 8), np.float32)
        for source, target in ((0, 10), (5, -1), (True, 10), (5, np.nan),
                               (np.inf, 10), (10, 5), (5, 1000)):
            with self.assertRaises(ValueError):
                normalize_resolution(image, source, target)
