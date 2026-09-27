"""Phase-energy structure, numerical safety and reproducibility tests."""
import unittest
import numpy as np
from src.features.phase_congruency import phase_congruency, PhaseCongruencyConfig


class PhaseCongruencyTests(unittest.TestCase):
    def test_dimensions_dtype_range_and_preservation(self) -> None:
        image = np.random.default_rng(1).integers(0, 400, (47, 53), dtype=np.uint16)
        before = image.copy()
        image.flags.writeable = False
        result = phase_congruency(image)
        self.assertEqual(result.data.shape, image.shape)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertTrue(np.isfinite(result.data).all())
        self.assertTrue(np.all((result.data >= 0) & (result.data <= 1)))
        np.testing.assert_array_equal(image, before)

    def test_flat_and_nearly_flat(self) -> None:
        for image in (np.zeros((32, 32)), np.full((32, 32), 65535, np.uint16),
                      1 + np.eye(32) * 1e-8, np.ones((1, 1))):
            self.assertFalse(phase_congruency(image).data.any())

    def test_step_edge(self) -> None:
        image = np.zeros((96, 96), np.float32)
        image[:, 48:] = 1
        result = phase_congruency(image).data
        self.assertGreater(result[20:76, 46:50].mean(), .1)
        self.assertGreater(result[20:76, 46:50].mean(), 5 * result[20:76, 15:25].mean())

    def test_line_ridge(self) -> None:
        image = np.zeros((96, 96), np.float32)
        image[:, 48] = 1
        result = phase_congruency(image).data
        self.assertGreater(result[20:76, 48].mean(), .1)
        self.assertGreater(result[20:76, 48].mean(), 5 * result[20:76, 15:25].mean())

    def test_deterministic_and_gain_offset(self) -> None:
        image = np.zeros((65, 67), np.float32)
        image[20:40, 15:45] = 1
        first = phase_congruency(image).data
        np.testing.assert_array_equal(first, phase_congruency(image).data)
        np.testing.assert_allclose(first, phase_congruency(image * 12 + 5).data, atol=2e-6)

    def test_invalid_input(self) -> None:
        for image in (np.zeros((2, 2, 2)), np.empty((0, 4)), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.assertRaises(ValueError):
                phase_congruency(image)
        for image in ([[1]], np.ones((2, 2), complex), np.ones((2, 2), bool)):
            with self.assertRaises(TypeError):
                phase_congruency(image)

    def test_parameters(self) -> None:
        for kwargs in ({"n_scales": 1}, {"n_orientations": 0}, {"n_scales": True},
                       {"sigma_on_f": 1}, {"wavelength_multiplier": 1}, {"min_wavelength": 0},
                       {"noise_k": -1}, {"noise_floor": np.inf}, {"spread_gain": np.nan},
                       {"spread_cutoff": 2}, {"padding": -1}):
            with self.assertRaises(ValueError):
                PhaseCongruencyConfig(**kwargs)

    def test_large_finite_values_and_noise_threshold(self) -> None:
        image = np.zeros((32, 32))
        image[:, 16:] = np.finfo(np.float64).max
        self.assertTrue(np.isfinite(phase_congruency(image).data).all())
        result = phase_congruency(image, PhaseCongruencyConfig(noise_floor=100))
        self.assertFalse(result.data.any())
