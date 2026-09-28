"""Offline LoFTR wrapper tests; no checkpoint downloads."""
import unittest
from unittest.mock import patch
import numpy as np
from src.features.learned import prepare_input, postprocess, LoFTRMatcher


class LearnedTests(unittest.TestCase):
    def test_padding_preserves_input(self):
        a = np.ones((13, 19), np.float32)
        padded, mask, padding = prepare_input(a)
        self.assertEqual(padded.shape, (16, 24))
        self.assertEqual(padding, (3, 5))
        self.assertEqual(mask.sum(), a.size)
        np.testing.assert_array_equal(a, padded[:13, :19])
        self.assertFalse(np.shares_memory(a, padded))

    def test_validation(self):
        for a in (np.zeros((3, 8), np.float32), np.zeros((8, 8, 1), np.float32),
                  np.full((8, 8), np.nan, np.float32), np.full((8, 8), 2, np.float32)):
            with self.assertRaises(ValueError): prepare_input(a)
        with self.assertRaises(TypeError): prepare_input(np.zeros((8, 8), np.uint8))

    def test_coordinates_confidence_invalid_padding(self):
        a = np.array([[1, 2], [19, 2], [-1, 2], [2, 2], [3, 3], [4, 4]], float)
        b = np.array([[3, 4], [3, 4], [3, 4], [np.nan, 4], [3, 3], [4, 4]])
        c = np.array([.4, .8, .9, .9, np.inf, .8])
        result = postprocess(a, b, c, (13, 19), (16, 16))
        np.testing.assert_array_equal(result.source, [[1, 2], [4, 4]])
        np.testing.assert_allclose(result.confidence, [.4, .8])
        self.assertEqual(len(result.filter_confidence(.4).source), 2)
        self.assertEqual(len(result.filter_confidence(.6).source), 1)
        np.testing.assert_array_equal(result.source, postprocess(a, b, c, (13, 19), (16, 16)).source)
        with self.assertRaises(ValueError): result.filter_confidence(np.nan)

    def test_empty_and_malformed(self):
        result = postprocess(np.empty((0, 2)), np.empty((0, 2)), np.empty(0), (8, 8), (8, 8))
        self.assertEqual(result.source.shape, (0, 2))
        with self.assertRaises(ValueError): postprocess(np.zeros((2, 2)), np.zeros((1, 2)), np.zeros(2), (8, 8), (8, 8))

    def test_mock_inference(self):
        import torch
        class Mock(torch.nn.Module):
            def forward(self, data):
                assert not torch.is_grad_enabled()
                assert data["image0"].shape == (1, 1, 16, 24)
                return {"keypoints0": torch.tensor([[1., 2.], [20., 3.]]),
                        "keypoints1": torch.tensor([[3., 4.], [3., 4.]]), "confidence": torch.tensor([.8, .9])}
        matcher = LoFTRMatcher(model=Mock(), device="cpu")
        result = matcher.match(np.ones((13, 19), np.float32), np.ones((16, 16), np.float32))
        self.assertEqual(len(result.source), 1)
        self.assertEqual(result.metadata["invalid_or_padding_rejected"], 1)

    def test_oom_retries_identical_arrays_on_cpu(self):
        import torch
        matcher = LoFTRMatcher(model=torch.nn.Identity(), device="cpu")
        matcher.device = "cuda"
        values = (np.array([[1., 2.]]), np.array([[3., 4.]]), np.array([.8]))
        with patch.object(matcher, "_infer", side_effect=[torch.cuda.OutOfMemoryError("test"), values]) as infer, \
             patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.empty_cache"), \
             patch("torch.cuda.max_memory_allocated", return_value=1024):
            result = matcher.match(np.ones((8, 8), np.float32), np.ones((8, 8), np.float32))
        self.assertEqual(result.metadata["device"], "cpu")
        self.assertIn("OOM", result.metadata["fallback"])
        self.assertEqual(infer.call_count, 2)
        self.assertIs(infer.call_args_list[0].args[0], infer.call_args_list[1].args[0])
