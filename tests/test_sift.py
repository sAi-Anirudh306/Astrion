"""SIFT baseline behavior and compatibility with preprocessing."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.features.sift import SIFTConfig, extract_sift, prepare_sift_image
from src.preprocessing.preprocessing import preprocess_image


def structured_image() -> np.ndarray:
    """Deterministic geometric targets at several sizes and orientations."""
    image = np.zeros((256, 256), dtype=np.uint8)
    for x, y, radius in ((45, 50, 15), (120, 70, 25), (190, 170, 30)):
        cv2.circle(image, (x, y), radius, 220, 3)
    cv2.rectangle(image, (30, 140), (80, 200), 180, -1)
    cv2.putText(image, "TMC", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    return image


class SIFTTests(unittest.TestCase):
    def test_detection_descriptors_and_coordinates(self) -> None:
        result = extract_sift(preprocess_image(structured_image()).data)
        self.assertGreater(len(result.keypoints), 0)
        self.assertEqual(result.descriptors.shape, (len(result.keypoints), 128))
        self.assertEqual(result.descriptors.dtype, np.float32)
        self.assertTrue(np.isfinite(result.descriptors).all())
        self.assertEqual(result.points.shape, (len(result.keypoints), 2))
        self.assertTrue((result.points >= 0).all())
        self.assertTrue((result.points < 256).all())
        for point, xy in zip(result.keypoints, result.points):
            np.testing.assert_allclose(xy, point.pt)

    def test_repeatability_and_preservation(self) -> None:
        image = preprocess_image(structured_image()).data.T
        before = image.copy()
        image.flags.writeable = False
        first, second = extract_sift(image), extract_sift(image)
        np.testing.assert_array_equal(first.points, second.points)
        np.testing.assert_array_equal(first.descriptors, second.descriptors)
        np.testing.assert_array_equal(image, before)

    def test_fixed_scaling_no_contrast_stretch(self) -> None:
        image = np.array([[0, 0.1, 0.5, 1]], dtype=np.float32)
        prepared = prepare_sift_image(image)
        np.testing.assert_array_equal(prepared, [[0, 26, 128, 255]])
        self.assertFalse(np.shares_memory(image, prepared))
        self.assertTrue(prepared.flags.c_contiguous)

    def test_empty_and_small(self) -> None:
        for image in (np.zeros((64, 64), np.float32), np.full((64, 64), 0.4, np.float32),
                      np.zeros((1, 1), np.float32), np.eye(7, dtype=np.float32),
                      np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))):
            with self.subTest(shape=image.shape):
                result = extract_sift(image)
                self.assertEqual(result.keypoints, ())
                self.assertEqual(result.descriptors.shape, (0, 128))
                self.assertEqual(result.points.shape, (0, 2))

    def test_invalid_input(self) -> None:
        for image in (None, [[0]], np.zeros((0, 3), np.float32), np.zeros(3, np.float32),
                      np.zeros((3, 3, 3), np.float32), np.zeros((3, 3), np.uint16),
                      np.zeros((3, 3), np.uint8), np.zeros((3, 3), np.float64),
                      np.array([[np.nan]], np.float32), np.array([[np.inf]], np.float32),
                      np.array([[-0.01]], np.float32), np.array([[1.01]], np.float32)):
            with self.subTest(image=str(image)):
                with self.assertRaises((TypeError, ValueError)):
                    extract_sift(image)

    def test_invalid_config(self) -> None:
        for params in ({"nfeatures": -1}, {"nfeatures": True}, {"nOctaveLayers": 0},
                       {"nfeatures": 1.5}, {"sigma": 0}, {"edgeThreshold": float("nan")},
                       {"contrastThreshold": -1}, {"sigma": float("inf")},
                       {"enable_precise_upscale": 1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SIFTConfig(**params)
        with self.assertRaises(TypeError):
            extract_sift(np.zeros((8, 8), np.float32), {})

    def test_parameters_passed_and_recorded(self) -> None:
        config = SIFTConfig(nfeatures=25, nOctaveLayers=4, contrastThreshold=0.05,
                            edgeThreshold=12, sigma=1.7)
        with patch("src.features.sift.cv2.SIFT_create", wraps=cv2.SIFT_create) as factory:
            result = extract_sift(preprocess_image(structured_image()).data, config)
        self.assertEqual(result.config, config)
        self.assertEqual(factory.call_args.kwargs["nfeatures"], 25)
        self.assertEqual(factory.call_args.kwargs["sigma"], 1.7)
        self.assertEqual(result.opencv_version, cv2.__version__)

    def test_uint16_preprocessing_compatibility(self) -> None:
        image = structured_image().astype(np.uint16) * 200
        result = extract_sift(preprocess_image(image).data)
        self.assertGreater(len(result.keypoints), 0)

    def test_opencv_failure_not_silenced(self) -> None:
        with patch("src.features.sift.cv2.SIFT_create", side_effect=cv2.error("failure")):
            with self.assertRaisesRegex(RuntimeError, "OpenCV SIFT failed"):
                extract_sift(np.zeros((8, 8), np.float32))


if __name__ == "__main__":
    unittest.main()
