"""Preprocessing unit tests and full-resolution Borrow K integration checks."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.ingestion.image_loader import load_image
from src.preprocessing.normalization import normalize_intensity
from src.preprocessing.denoising import denoise
from src.preprocessing.illumination import normalize_illumination
from src.preprocessing.preprocessing import PreprocessingConfig, preprocess_image


class PreprocessingTests(unittest.TestCase):
    """Check numerical behavior, source preservation and failure handling."""

    def test_uint_inputs_shape_dtype_range_and_preservation(self) -> None:
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                source = np.arange(256, dtype=dtype).reshape(16, 16)
                before = source.copy()
                source.flags.writeable = False
                result = preprocess_image(source)
                self.assertEqual(result.data.shape, source.shape)
                self.assertEqual(result.data.dtype, np.float32)
                self.assertEqual(result.output_range[0], 0.0)
                expected_max = 1.0 if dtype == np.uint8 else 249.9 / 655.35
                self.assertAlmostEqual(result.output_range[1], expected_max, places=6)
                self.assertTrue(np.isfinite(result.data).all())
                self.assertLessEqual(result.data.max(), 1)
                self.assertFalse(np.shares_memory(result.data, source))
                np.testing.assert_array_equal(source, before)

    def test_known_percentile_mapping(self) -> None:
        source = np.arange(101, dtype=np.uint16).reshape(1, 101)
        output = normalize_intensity(source, 10, 90, min_span=0.0001)
        self.assertEqual(output[0, 10], 0)
        self.assertEqual(output[0, 90], 1)
        self.assertAlmostEqual(float(output[0, 50]), 0.5)
        self.assertTrue(np.all(np.diff(output[0]) >= 0))

    def test_constant_images_all_operations(self) -> None:
        for dtype, value in ((np.uint8, 0), (np.uint8, 127), (np.uint16, 65535)):
            with self.subTest(dtype=dtype, value=value):
                result = preprocess_image(np.full((17, 19), value, dtype=dtype),
                    PreprocessingConfig(denoise_sigma=0.5, illumination_strength=0.25))
                np.testing.assert_array_equal(result.data, np.zeros((17, 19)))

    def test_low_contrast_gain_is_limited(self) -> None:
        image = np.full((20, 20), 30000, dtype=np.uint16)
        image[:, 10:] += 1
        output = normalize_intensity(image)
        self.assertGreater(output.max(), 0)
        self.assertLess(output.max(), 0.002)

    def test_sparse_structure_not_erased(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint16)
        image[50, 50] = 65535
        output = normalize_intensity(image)
        self.assertEqual(output[50, 50], 1)
        self.assertEqual(np.count_nonzero(output), 1)

    def test_percentile_clipping_with_gain_limit(self) -> None:
        image = np.arange(101, dtype=np.uint16).reshape(1, 101)
        output = normalize_intensity(image, 10, 90)
        self.assertEqual(output[0, 90], output[0, 100])
        self.assertAlmostEqual(float(output.max()), 80 / 655.35, places=6)

    def test_float_input(self) -> None:
        image = np.linspace(0, 1, 100, dtype=np.float64).reshape(10, 10)
        output = preprocess_image(image).data
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual((output.min(), output.max()), (0, 1))

    def test_invalid_images(self) -> None:
        invalid = (None, [[1]], np.zeros((0, 2)), np.zeros(3), np.zeros((2, 2, 3)),
                   np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=np.int16),
                   np.ones((2, 2), dtype=complex), np.array([[np.nan]]),
                   np.array([[np.inf]]), np.array([[-0.1]]), np.array([[1.1]]))
        for source in invalid:
            with self.subTest(source=str(source)):
                with self.assertRaises((TypeError, ValueError)):
                    preprocess_image(source)

    def test_invalid_configuration(self) -> None:
        invalid = ({"lower_percentile": 99}, {"upper_percentile": 101},
                   {"min_span": 0}, {"denoise_sigma": -1}, {"denoise_sigma": 2},
                   {"illumination_sigma": 0}, {"illumination_strength": 0.6},
                   {"illumination_strength": float("nan")}, {"denoise_sigma": True},
                   {"min_span": "bad"})
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    PreprocessingConfig(**settings)
        with self.assertRaises(TypeError):
            preprocess_image(np.zeros((2, 2), dtype=np.uint8), {})

    def test_denoising_reduces_synthetic_noise(self) -> None:
        rng = np.random.default_rng(42)
        source = (0.5 + rng.normal(0, 0.03, (100, 100))).astype(np.float32)
        before = source.copy()
        output = denoise(source, sigma=0.5)
        self.assertLess(output.var(), source.var())
        np.testing.assert_array_equal(source, before)

    def test_illumination_reduces_synthetic_gradient(self) -> None:
        source = np.tile(np.linspace(0.2, 0.8, 128, dtype=np.float32), (64, 1))
        before = source.copy()
        output = normalize_illumination(source, sigma=16, strength=0.25)
        self.assertLess(np.ptp(output), np.ptp(source))
        self.assertAlmostEqual(float(output.mean()), float(source.mean()), places=6)
        np.testing.assert_array_equal(source, before)

    def test_standalone_operations_validate_and_copy(self) -> None:
        source = np.full((10, 10), 0.4, dtype=np.float32)
        for function in (denoise, normalize_illumination):
            output = function(source)
            self.assertFalse(np.shares_memory(source, output))
            np.testing.assert_array_equal(output, source)
            with self.assertRaises(ValueError):
                function(np.array([[2.0]], dtype=np.float32))
        np.testing.assert_allclose(normalize_illumination(source, strength=0.25), source)
        with self.assertRaises(ValueError):
            denoise(source, sigma=float("inf"))

    def test_operation_order_and_repeatability(self) -> None:
        source = np.arange(400, dtype=np.uint16).reshape(20, 20)
        before = source.copy()
        config = PreprocessingConfig(denoise_sigma=0.5, illumination_strength=0.25)
        result = preprocess_image(source, config)
        expected = normalize_illumination(denoise(normalize_intensity(source), 0.5), 16, 0.25)
        np.testing.assert_array_equal(result.data, expected)
        np.testing.assert_array_equal(result.data, preprocess_image(source, config).data)
        np.testing.assert_array_equal(source, before)
        self.assertEqual(result.operations, ("percentile_normalization", "gaussian_denoising",
                                             "additive_illumination_normalization"))

    def test_small_and_noncontiguous_images(self) -> None:
        for source in (np.ones((1, 1), dtype=np.uint8),
                       np.arange(100, dtype=np.uint16).reshape(10, 10).T[::2]):
            result = preprocess_image(source, PreprocessingConfig(
                denoise_sigma=0.5, illumination_strength=0.25))
            self.assertEqual(result.data.shape, source.shape)
            self.assertTrue(np.isfinite(result.data).all())


class BorrowKPreprocessingTests(unittest.TestCase):
    """Exercise all selected operations on both complete real images."""

    def test_full_resolution_products(self) -> None:
        root = Path(__file__).resolve().parents[1] / "data/test/borrow_k"
        paths = (
            root / "browse/raw/20260629/ch2_tmc_nrf_20260629T2059373111_b_brw_d18.png",
            root / "data/raw/20260629/ch2_tmc_nrf_20260629T2059373111_d_img_d18.img",
        )
        config = PreprocessingConfig(denoise_sigma=0.5, illumination_strength=0.25)
        for path in paths:
            with self.subTest(product=path.suffix):
                source = load_image(path)
                before = hashlib.sha256(source.data.tobytes()).digest()
                source.data.flags.writeable = False
                result = preprocess_image(source, config)
                self.assertEqual(result.data.shape, source.data.shape)
                self.assertEqual(result.data.dtype, np.float32)
                self.assertGreaterEqual(result.data.min(), 0)
                self.assertLessEqual(result.data.max(), 1)
                self.assertTrue(np.isfinite(result.data).all())
                self.assertEqual(hashlib.sha256(source.data.tobytes()).digest(), before)
                self.assertFalse(np.shares_memory(source.data, result.data))
                print("\nBorrow K preprocessing: " + json.dumps({
                    "input": str(path), "input_shape": result.input_shape,
                    "input_dtype": result.input_dtype, "input_range": result.input_range,
                    "output_shape": result.data.shape, "output_dtype": str(result.data.dtype),
                    "output_range": result.output_range, "operations": result.operations,
                    "config": asdict(config), "runtime_seconds": result.runtime_seconds,
                }))
                del result, source


if __name__ == "__main__":
    unittest.main()
