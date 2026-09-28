"""Optional Kornia LoFTR correspondence path; classical imports stay independent."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from numbers import Real
import gc
import numpy as np


@dataclass(frozen=True)
class LearnedMatches:
    """Coordinates are (x,y) in original supplied grids, before bottom/right padding."""
    source: np.ndarray
    destination: np.ndarray
    confidence: np.ndarray
    metadata: dict

    def filter_confidence(self, threshold: float) -> "LearnedMatches":
        """Keep confidence >= threshold; deterministic order and independent arrays."""
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("confidence threshold must be finite in [0,1]")
        keep = self.confidence >= threshold
        return LearnedMatches(self.source[keep], self.destination[keep], self.confidence[keep],
                              self.metadata | {"confidence_threshold": float(threshold)})


def prepare_input(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Validate normalized float32; zero-pad bottom/right to multiples of eight.

    Returns padded pixels, valid-pixel mask, (bottom,right) padding. No resize,
    offset, intensity stretch or source mutation. Tiny inputs are rejected.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise TypeError("LoFTR requires float32 preprocessing output")
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("LoFTR requires a 2D image at least 8x8")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("LoFTR input must be finite in [0,1]")
    padding = tuple((-n) % 8 for n in image.shape)
    pads = ((0, padding[0]), (0, padding[1]))
    return np.pad(image, pads), np.pad(np.ones(image.shape, bool), pads), padding


def postprocess(source: np.ndarray, destination: np.ndarray, confidence: np.ndarray,
                source_shape: tuple, destination_shape: tuple) -> LearnedMatches:
    """Remove invalid confidences/coordinates and padding-only correspondences."""
    for shape in (source_shape, destination_shape):
        if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0 for n in shape):
            raise ValueError("Original image shapes must contain two positive integers")
    a, b, c = np.asarray(source), np.asarray(destination), np.asarray(confidence)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape or c.shape != (len(a),):
        raise ValueError("Expected aligned (N,2), (N,2), (N,) model outputs")
    if any(x.dtype.kind not in "fiu" for x in (a, b, c)):
        raise TypeError("Model outputs must be real numeric arrays")
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(c) & (c >= 0) & (c <= 1)
    for points, shape in ((a, source_shape), (b, destination_shape)):
        keep &= (points >= 0).all(axis=1) & (points[:, 0] < shape[1]) & (points[:, 1] < shape[0])
    return LearnedMatches(a[keep].astype(np.float32), b[keep].astype(np.float32), c[keep].astype(np.float32),
        {"model_output_count": len(a), "invalid_or_padding_rejected": int((~keep).sum())})


class LoFTRMatcher:
    """Float32 pretrained inference, one pair at a time, with explicit CPU fallback.

    A local documented checkpoint avoids implicit downloads during tests/import.
    Kornia defaults (including its internal coarse confidence gate) are retained.
    Returned raw counts are model-emitted correspondences, not all dense pairs.
    Inject a torch module for offline tests. CUDA OOM retries the SAME inputs and
    parameters once on CPU, records the reason, and stays on CPU thereafter.
    """
    def __init__(self, checkpoint: str | Path | None = None, device: str = "auto", model=None):
        import torch
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.info = {"torch": torch.__version__, "dtype": "float32", "checkpoint": str(checkpoint)}
        if model is None:
            import kornia
            from kornia.feature import LoFTR
            if checkpoint is None or not Path(checkpoint).is_file():
                raise FileNotFoundError("A local LoFTR outdoor checkpoint is required")
            model = LoFTR(pretrained=None)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(weights["state_dict"])
            self.info.update(library="kornia", library_version=kornia.__version__, pretrained="outdoor (MegaDepth)", model_config=model.config)
        self.model = model.eval().float()
        self.initial_fallback = None
        try:
            self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            self.device = "cpu"
            self.model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self.initial_fallback = "CUDA OOM during model transfer: using CPU float32"

    def _infer(self, a, b, ma, mb):
        import torch
        with torch.inference_mode():
            output = self.model({"image0": torch.from_numpy(a)[None, None].to(self.device),
                "image1": torch.from_numpy(b)[None, None].to(self.device),
                "mask0": torch.from_numpy(ma)[None].to(self.device, dtype=torch.float32),
                "mask1": torch.from_numpy(mb)[None].to(self.device, dtype=torch.float32)})
            return tuple(output[key].detach().cpu().numpy() for key in ("keypoints0", "keypoints1", "confidence"))

    def match(self, source: np.ndarray, destination: np.ndarray) -> LearnedMatches:
        """Return valid correspondences; release inference tensors between calls."""
        import torch
        tick = perf_counter()
        # Bound the dense coarse correlation matrix before allocating GPU tensors.
        # This guard never silently resizes scientific processing grids.
        for image in (source, destination):
            if isinstance(image, np.ndarray) and image.size > 1_000_000:
                raise ValueError("LoFTR input exceeds 1M-pixel safety limit; normalize physical scale first")
        a, ma, pa = prepare_input(source)
        b, mb, pb = prepare_input(destination)
        if (a.size // 64) * (b.size // 64) > 16_000_000:
            raise ValueError("LoFTR coarse correlation exceeds 16M-entry safety limit")
        fallback = None
        attempted_peak = None
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            values = self._infer(a, b, ma, mb)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            fallback = "CUDA OOM: retry identical arrays/model in float32 on CPU"
            attempted_peak = int(torch.cuda.max_memory_allocated())
        if fallback:
            gc.collect()
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.initial_fallback = fallback
            values = self._infer(a, b, ma, mb)
        peak = int(torch.cuda.max_memory_allocated()) if self.device == "cuda" else attempted_peak
        result = postprocess(*values, source.shape, destination.shape)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return LearnedMatches(result.source, result.destination, result.confidence,
            self.info | result.metadata | {"device": self.device, "padding_source_bottom_right": pa,
                "padding_destination_bottom_right": pb, "fallback": fallback or self.initial_fallback,
                "runtime_seconds": perf_counter()-tick, "peak_gpu_allocated_bytes": peak})
