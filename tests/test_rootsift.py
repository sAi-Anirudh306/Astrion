"""Numerical and integration contracts for RootSIFT."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from src.features.rootsift import rootsift_descriptors, transform_sift
from src.features.sift import extract_sift
from src.preprocessing.preprocessing import preprocess_image


class RootSIFTTests(unittest.TestCase):
    def test_known_transformation(self) -> None:
        source = np.zeros((2, 128), dtype=np.float32)
        source[0, :2] = [1, 3]
        source[1, :4] = 2
        result = rootsift_descriptors(source)
        np.testing.assert_allclose(result[0, :2], [0.5, np.sqrt(0.75)], rtol=1e-7)
        np.testing.assert_array_equal(result[1, :4], 0.5)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_no_mutation_noncontiguous_readonly(self) -> None:
        source = np.arange(512, dtype=np.float32).reshape(2, 256)[:, ::2]
        before = source.copy()
        source.flags.writeable = False
        result = rootsift_descriptors(source)
        np.testing.assert_array_equal(source, before)
        self.assertFalse(np.shares_memory(source, result))

    def test_empty(self) -> None:
        result = rootsift_descriptors(np.empty((0, 128), np.float32))
        self.assertEqual(result.shape, (0, 128))
        self.assertEqual(result.dtype, np.float32)
        transformed = transform_sift(extract_sift(np.zeros((16, 16), np.float32)))
        self.assertEqual(transformed.points.shape, (0, 2))

    def test_zero_rows(self) -> None:
        source = np.zeros((3, 128), np.float32)
        source[1, 10] = 7
        result = rootsift_descriptors(source)
        np.testing.assert_array_equal(result[[0, 2]], 0)
        self.assertEqual(result[1, 10], 1)
        self.assertTrue(np.isfinite(result).all())

    def test_normalization_property_and_scaling(self) -> None:
        source = np.random.default_rng(42).uniform(0, 100, (100, 128)).astype(np.float32)
        result = rootsift_descriptors(source)
        np.testing.assert_allclose(np.linalg.norm(result.astype(np.float64), axis=1), 1, atol=1e-7)
        np.testing.assert_allclose(result.astype(np.float64) ** 2,
                                   source / source.sum(axis=1, keepdims=True, dtype=np.float64), atol=1e-8)
        np.testing.assert_allclose(result, rootsift_descriptors(source * 2), atol=1e-7)

    def test_extreme_float32_values(self) -> None:
        source = np.zeros((3, 128), np.float32)
        source[0] = np.finfo(np.float32).max
        source[1] = np.nextafter(np.float32(0), np.float32(1))
        source[2, :2] = [np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))]
        with np.errstate(all="raise"):
            result = rootsift_descriptors(source)
        np.testing.assert_allclose(np.linalg.norm(result.astype(np.float64), axis=1), 1, atol=1e-7)
        self.assertGreater(result[2, 1], 0)

    def test_invalid_descriptors(self) -> None:
        invalid = (None, [], np.zeros(128, np.float32), np.zeros((2, 127), np.float32),
                   np.zeros((1, 1, 128), np.float32), np.zeros((0, 0), np.float32),
                   np.zeros((1, 128), np.float64), np.zeros((1, 128), np.uint8),
                   np.zeros((1, 128), bool), np.zeros((1, 128), complex))
        for source in invalid:
            with self.subTest(source=str(source)), self.assertRaises((TypeError, ValueError)):
                rootsift_descriptors(source)
        for value in (-1, np.nan, np.inf, -np.inf):
            source = np.zeros((1, 128), np.float32)
            source[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                rootsift_descriptors(source)

    def test_sift_compatibility_and_order(self) -> None:
        image = np.zeros((128, 128), np.uint8)
        cv2.circle(image, (40, 50), 15, 255, 3)
        cv2.rectangle(image, (75, 70), (105, 105), 200, -1)
        normalized = preprocess_image(image).data
        before_image = normalized.copy()
        sift = extract_sift(normalized)
        before = sift.descriptors.copy()
        self.assertGreater(len(sift.keypoints), 0)
        root = transform_sift(sift)
        self.assertIs(root.keypoints, sift.keypoints)
        self.assertEqual(root.descriptors.shape, sift.descriptors.shape)
        np.testing.assert_array_equal(root.points, sift.points)
        np.testing.assert_array_equal(sift.descriptors, before)
        np.testing.assert_array_equal(normalized, before_image)
        self.assertEqual(root.descriptor_type, "RootSIFT")
        self.assertEqual(root.sift_config, sift.config)
        with self.assertRaises(ValueError):
            transform_sift(replace(sift, keypoints=()))
        with self.assertRaises(TypeError):
            transform_sift(root)


if __name__ == "__main__":
    unittest.main()
