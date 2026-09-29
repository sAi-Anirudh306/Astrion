"""Synthetic scale grids, reversible coordinates and classical matching."""
import unittest
import cv2
import numpy as np
from src.preprocessing.scale_normalization import normalize_resolution, build_scale_pyramid
from src.matching.correspondence import match_scale_pyramid


def structured_image(size=192):
    image = np.zeros((size, size), np.float32)
    rng = np.random.default_rng(15)
    for _ in range(100):
        x, y = rng.integers(10, size-10, 2)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), float(rng.uniform(.2, 1)), -1)
    return image


class MultiscaleTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((11, 17), np.float32)
        self.level = normalize_resolution(self.image, 1, 2)

    def test_identity(self):
        level = normalize_resolution(self.image, 1, 1)
        np.testing.assert_array_equal(level.forward_matrix, np.eye(3))
        np.testing.assert_array_equal(level.data, self.image)

    def test_known_downsampling(self):
        level = normalize_resolution(np.ones((12, 20), np.float32), 1, 2)
        self.assertEqual(level.data.shape, (6, 10))
        self.assertEqual(level.actual_factors_xy, (.5, .5))

    def test_odd_dimensions_effective_gsd(self):
        self.assertEqual(self.level.data.shape, (6, 9))
        self.assertEqual(self.level.actual_factors_xy, (9/17, 6/11))
        np.testing.assert_allclose(self.level.effective_resolution_xy, [17/9, 11/6])

    def test_noninteger(self):
        level = normalize_resolution(self.image, 5.15, 7.5)
        self.assertEqual(level.data.shape, (8, 12))

    def test_forward(self):
        np.testing.assert_allclose(self.level.original_to_scaled(np.array([[4., 6.]])),
                                   [[4.5*9/17-.5, 6.5*6/11-.5]])

    def test_inverse(self):
        np.testing.assert_allclose(self.level.scaled_to_original(np.array([[3., 4.]])),
                                   [[3.5*17/9-.5, 4.5*11/6-.5]])

    def test_roundtrip_edges(self):
        points = np.array([[0., 0.], [16, 10], [8.23, 4.72]])
        scaled = self.level.original_to_scaled(points)
        np.testing.assert_allclose(self.level.scaled_to_original(scaled, pixel_footprint=True), points, atol=1e-14)

    def test_last_scaled_pixel(self):
        original = self.level.scaled_to_original(np.array([[8., 5.]]))
        np.testing.assert_allclose(self.level.original_to_scaled(original), [[8, 5]], atol=1e-14)

    def test_invalid_negative(self):
        for fn in (self.level.original_to_scaled, self.level.scaled_to_original):
            with self.assertRaises(ValueError): fn(np.array([[-.01, 0]]))

    def test_explicit_footprint(self):
        p = np.array([[-.5, -.5], [16.5, 10.5]])
        q = self.level.original_to_scaled(p, pixel_footprint=True)
        np.testing.assert_allclose(q, [[-.5, -.5], [8.5, 5.5]])
        with self.assertRaises(ValueError):
            self.level.original_to_scaled(np.array([[-.51, 0]]), pixel_footprint=True)

    def test_outside(self):
        with self.assertRaises(ValueError): self.level.original_to_scaled(np.array([[17., 5]]))
        with self.assertRaises(ValueError): self.level.scaled_to_original(np.array([[9., 5]]))

    def test_nonfinite_shape(self):
        for p in (np.array([[np.nan, 0]]), np.array([[0, np.inf]]), np.ones((2, 3))):
            with self.assertRaises(ValueError): self.level.original_to_scaled(p)

    def test_empty_points(self):
        self.assertEqual(self.level.scaled_to_original(np.empty((0, 2))).shape, (0, 2))

    def test_invalid_gsd(self):
        for gsd in (0, -1, np.inf, np.nan, True):
            with self.assertRaises(ValueError): build_scale_pyramid(self.image, gsd, [2])
            with self.assertRaises(ValueError): build_scale_pyramid(self.image, 1, [gsd])

    def test_no_upsampling(self):
        with self.assertRaises(ValueError): build_scale_pyramid(self.image, 2, [1])

    def test_dtype_and_source_preserved(self):
        self.image.flags.writeable = False
        p = build_scale_pyramid(self.image, 1, [1, 2])
        for level in p.levels:
            self.assertEqual(level.data.dtype, np.float32)
            self.assertFalse(np.shares_memory(self.image, level.data))

    def test_duplicates_and_order(self):
        p = build_scale_pyramid(self.image, 1, [2, 1, 1.99, 3])
        self.assertEqual([l.target_resolution for l in p.levels], [2, 1, 3])
        self.assertEqual(p.request_to_level, (0, 1, 0, 2))
        self.assertEqual(p.requested_resolutions, (2, 1, 1.99, 3))

    def test_metadata(self):
        meta = self.level.metadata()
        self.assertEqual(meta['original_shape'], (11, 17))
        self.assertEqual(meta['source_gsd'], 1)
        np.testing.assert_allclose(self.level.forward_matrix@self.level.inverse_matrix, np.eye(3), atol=1e-14)

    def test_empty_levels_rejected(self):
        with self.assertRaises(ValueError): build_scale_pyramid(self.image, 1, [])

    def test_deterministic_pyramid(self):
        a = build_scale_pyramid(self.image, 1, [1, 2, 3])
        b = build_scale_pyramid(self.image, 1, [1, 2, 3])
        for first, second in zip(a.levels, b.levels):
            self.assertEqual(first.metadata(), second.metadata())
            np.testing.assert_array_equal(first.data, second.data)

    def test_empty_features(self):
        records = match_scale_pyramid(build_scale_pyramid(self.image, 1, [1, 2]), self.image)
        for record in records:
            self.assertEqual(record.source.shape, (0, 2))
            self.assertEqual(record.destination.shape, (0, 2))
            self.assertEqual(len(record.valid_matches.accepted), 0)

    def test_candidate_mapping_and_reproducibility(self):
        image = structured_image()
        reference = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        pyramid = build_scale_pyramid(image, 1, [2])
        a = match_scale_pyramid(pyramid, reference)[0]
        b = match_scale_pyramid(pyramid, reference)[0]
        self.assertGreater(len(a.valid_matches.accepted), 10)
        np.testing.assert_allclose(a.source, (a.destination+.5)*2-.5, atol=1e-5)
        np.testing.assert_array_equal(a.source, b.source)
        self.assertEqual(a.valid_matches.accepted, b.valid_matches.accepted)
        for i, m in zip(a.accepted_indices, a.valid_matches.accepted):
            self.assertEqual(a.ratio.accepted[i], m)

    def test_validity_mask(self):
        image = structured_image()
        pyramid = build_scale_pyramid(image, 1, [1])
        records = match_scale_pyramid(pyramid, image, source_mask=np.zeros(image.shape, bool))
        self.assertGreater(len(records[0].ratio.accepted), 10)
        self.assertEqual(len(records[0].valid_matches.accepted), 0)

    def test_invalid_mask(self):
        p = build_scale_pyramid(self.image, 1, [1])
        with self.assertRaises(ValueError): match_scale_pyramid(p, self.image, reference_mask=np.ones((2, 2), bool))

    def test_reference_reuse_and_level_identity(self):
        image = structured_image()
        pyramid = build_scale_pyramid(image, 1, [1, 2])
        records = match_scale_pyramid(pyramid, image)
        self.assertIs(records[0].reference_features, records[1].reference_features)
        for index, record in enumerate(records):
            self.assertEqual(record.level_index, index)
            self.assertIs(record.level, pyramid.levels[index])

    def test_affine_composition(self):
        matrix = np.array([[1.1, .2, 3.], [-.1, .9, 4.]])
        points = np.array([[1., 2.], [10., 8.]])
        composed = (np.vstack([matrix, [0, 0, 1]])@self.level.forward_matrix)[:2]
        expected = self.level.original_to_scaled(points)@matrix[:, :2].T+matrix[:, 2]
        np.testing.assert_allclose(points@composed[:, :2].T+composed[:, 2], expected, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
