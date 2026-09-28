"""Independent synthetic checks of scientific affine registration."""
import unittest
import numpy as np
from src.geometry.registration import warp_affine, transform_points


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(49, dtype=np.float32).reshape(7, 7)
        self.identity = np.array([[1., 0., 0.], [0., 1., 0.]])

    def test_identity_and_preservation(self):
        original = self.image.copy()
        result = warp_affine(self.image, self.identity, (7, 7))
        np.testing.assert_array_equal(result.image, original)
        self.assertEqual(result.image.dtype, np.float32)
        self.assertEqual(result.valid_mask.dtype, bool)
        self.assertTrue(result.valid_mask.all())
        result.image[0, 0] = -1
        np.testing.assert_array_equal(self.image, original)

    def test_translation_direction_and_dimensions(self):
        matrix = np.array([[1., 0., 2.], [0., 1., 3.]])
        result = warp_affine(self.image, matrix, (12, 11))
        np.testing.assert_array_equal(transform_points([[2, 1]], matrix), [[4, 4]])
        self.assertEqual(result.image[4, 4], self.image[1, 2])
        np.testing.assert_array_equal(result.image[3:10, 2:9], self.image)
        self.assertEqual(result.image.shape, (12, 11))
        self.assertEqual(result.valid_mask.sum(), 49)
        self.assertTrue(np.isnan(result.image[~result.valid_mask]).all())

    def test_rotation_and_scale(self):
        result = warp_affine(self.image, [[0, -1, 6], [1, 0, 0]], (7, 7))
        np.testing.assert_array_equal(result.image, np.rot90(self.image, -1))
        scaled = warp_affine(self.image, [[2, 0, 0], [0, 2, 0]], (13, 13))
        np.testing.assert_array_equal(scaled.image[::2, ::2], self.image)
        self.assertAlmostEqual(float(scaled.image[3, 3]), 12.)

    def test_nodata_and_nearest_mask(self):
        image = np.full((7, 7), 100, np.float32)
        image[3, 3] = np.nan
        mask = np.ones((7, 7), bool)
        mask[1, 1] = False
        original = mask.copy()
        result = warp_affine(image, [[1, 0, .25], [0, 1, .25]], (8, 8), validity_mask=mask)
        self.assertFalse(result.valid_mask[3, 3])
        self.assertFalse(result.valid_mask[1, 1])
        self.assertFalse(result.valid_mask[7, :].any())
        np.testing.assert_allclose(result.image[result.valid_mask], 100)
        self.assertTrue(np.isnan(result.image[~result.valid_mask]).all())
        np.testing.assert_array_equal(mask, original)
        self.assertTrue(np.isnan(image[3, 3]))

    def test_finite_nodata_and_empty_coverage(self):
        image = self.image.copy()
        image[2, 2] = -999
        result = warp_affine(image, self.identity, (7, 7), nodata=-999)
        self.assertFalse(result.valid_mask[2, 2])
        self.assertEqual(result.image[2, 2], -999)
        empty = warp_affine(image, self.identity, (7, 7), validity_mask=np.zeros((7, 7), bool))
        self.assertFalse(empty.valid_mask.any())
        self.assertTrue(np.isnan(empty.image).all())

    def test_invalid_transform_and_points(self):
        for matrix in (np.eye(3), [[1, 0, np.nan], [0, 1, 0]], [[0, 0, 0], [0, 1, 0]]):
            with self.assertRaises(ValueError):
                warp_affine(self.image, matrix, (7, 7))
        with self.assertRaises(ValueError):
            transform_points([[np.inf, 0]], self.identity)

    def test_invalid_images_options(self):
        with self.assertRaises(TypeError):
            warp_affine(self.image.astype(np.uint16), self.identity, (7, 7))
        for options in ({'output_shape': (0, 7)}, {'interpolation': 'cubic'},
                        {'nodata': np.inf}, {'validity_mask': np.full((7, 7), 2)}):
            args = dict(output_shape=(7, 7))
            args.update(options)
            with self.assertRaises(ValueError):
                warp_affine(self.image, self.identity, **args)


if __name__ == '__main__':
    unittest.main()
