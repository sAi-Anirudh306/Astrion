"""Multi-orientation log-Gabor phase congruency (no gradient-edge substitute).

Algorithm reference: P. Kovesi, Image Features From Phase Congruency (1999),
and https://www.peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html.
Independent NumPy implementation of quadrature filtering and phase-deviation
energy. The scalar output pools noise-compensated energy over orientations,
divided by pooled amplitude; it is not the phasecong3 maximum-moment output.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhaseCongruencyConfig:
    """Wavelengths in input pixels; noise_k is Rayleigh standard deviations.

    noise_floor is an additional energy threshold after max-absolute input
    scaling. Spread cutoff/gain penalize narrow-band responses. Reflection
    padding reduces FFT wraparound but does not remove all boundary effects.
    Parameter bounds exclude unstable filters and excessive filter banks.
    """
    n_scales: int = 4
    n_orientations: int = 6
    min_wavelength: float = 3.0
    wavelength_multiplier: float = 2.1
    sigma_on_f: float = 0.55
    noise_k: float = 2.0
    noise_floor: float = 0.0
    spread_cutoff: float = 0.5
    spread_gain: float = 10.0
    padding: int = 32

    def __post_init__(self) -> None:
        for name, lower, upper in (("n_scales", 2, 12), ("n_orientations", 1, 32), ("padding", 0, 512)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        for name, lower, upper in (("min_wavelength", 2., 1024.),
                ("wavelength_multiplier", 1.01, 10.), ("sigma_on_f", .05, .95),
                ("noise_k", 0., 100.), ("noise_floor", 0., 100.),
                ("spread_cutoff", 0., 1.), ("spread_gain", .01, 100.)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite in [{lower}, {upper}]")


@dataclass(frozen=True)
class PhaseCongruencyResult:
    """Independent float32 [0,1] map, config and per-orientation thresholds."""
    data: NDArray[np.float32]
    config: PhaseCongruencyConfig
    noise_thresholds: tuple[float, ...]
    runtime_seconds: float
    representation: str = "orientation-pooled weighted phase energy / pooled amplitude"


def phase_congruency(image: np.ndarray,
                     config: PhaseCongruencyConfig | None = None) -> PhaseCongruencyResult:
    """Compute local phase agreement across log-Gabor scales and orientations.

    Accept finite real 2-D numeric arrays, including uint16 and floating point.
    Max-absolute scaling before float32 conversion prevents overflow; no input
    mutation occurs. Relative contrast <= 1e-6 returns zeros (near-flat policy).
    FFT and accumulation use float64 precision. DC-free radial log-Gabors are
    multiplied by raised-cosine angular lobes and a radial low-pass envelope
    (cutoff .45 cycles/pixel, order 15). Real/imaginary inverse FFT components
    are the even/odd quadrature responses.

    At each orientation energy is sum(E*meanE + O*meanO -
    abs(E*meanO - O*meanE)). Median smallest-scale amplitude estimates Rayleigh
    tau; the scale geometric sum estimates total noise, whose mean + k*std
    is subtracted before rectification. A sigmoid frequency-spread weight
    penalizes single-scale responses. Pooled weighted energy / pooled amplitude
    lies in [0,1]; clipping only protects rounding. No percentile stretching.
    Assumes approximately white noise; shadows and nodata edges remain features.
    This representation is designed to reduce intensity dependence, not proof
    of cross-sensor or illumination invariance.
    """
    started = perf_counter()
    config = PhaseCongruencyConfig() if config is None else config
    if not isinstance(config, PhaseCongruencyConfig):
        raise TypeError("config must be PhaseCongruencyConfig")
    if not isinstance(image, np.ndarray) or image.dtype.kind not in "uif":
        raise TypeError("image must be a real numeric NumPy array")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be nonempty and two-dimensional")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    work = image.astype(np.float64)
    magnitude = float(np.max(np.abs(work)))
    if magnitude:
        work /= magnitude
    if float(np.ptp(work)) <= 1e-6:
        return PhaseCongruencyResult(np.zeros(image.shape, np.float32), config,
                                    (0.,) * config.n_orientations, perf_counter()-started)
    work = work.astype(np.float32).astype(np.float64)
    work -= work.mean()
    pad = config.padding
    if pad:
        work = np.pad(work, pad, mode="reflect")
    spectrum = np.fft.fft2(work)
    fy, fx = np.meshgrid(np.fft.fftfreq(work.shape[0]), np.fft.fftfreq(work.shape[1]), indexing="ij")
    radius = np.hypot(fx, fy)
    theta = np.arctan2(-fy, fx)
    lowpass = 1 / (1 + (radius / .45) ** 30)
    radius[0, 0] = 1
    filters = []
    for scale in range(config.n_scales):
        wavelength = config.min_wavelength * config.wavelength_multiplier ** scale
        radial = np.exp(-np.log(radius * wavelength) ** 2 / (2 * np.log(config.sigma_on_f) ** 2)) * lowpass
        radial[0, 0] = 0
        filters.append(radial)
    numerator = np.zeros_like(work)
    denominator = np.zeros_like(work)
    thresholds = []
    epsilon = 1e-12
    region = (slice(pad, -pad), slice(pad, -pad)) if pad else (slice(None), slice(None))
    for orientation in range(config.n_orientations):
        angle = orientation * np.pi / config.n_orientations
        delta = np.abs(np.arctan2(np.sin(theta-angle), np.cos(theta-angle)))
        angular = (np.cos(np.minimum(delta * config.n_orientations / 2, np.pi)) + 1) / 2
        responses = [np.fft.ifft2(spectrum * radial * angular) for radial in filters]
        amplitudes = [np.abs(response) for response in responses]
        amplitude_sum = np.sum(amplitudes, axis=0)
        total = np.sum(responses, axis=0)
        mean = total / np.maximum(np.abs(total), epsilon)
        energy = sum(response.real * mean.real + response.imag * mean.imag -
                     np.abs(response.real * mean.imag - response.imag * mean.real)
                     for response in responses)
        tau = float(np.median(amplitudes[0][region])) / np.sqrt(np.log(4))
        total_tau = tau * sum(config.wavelength_multiplier ** -s for s in range(config.n_scales))
        threshold = total_tau * (np.sqrt(np.pi / 2) + config.noise_k * np.sqrt((4-np.pi)/2)) + config.noise_floor
        thresholds.append(float(threshold))
        width = (amplitude_sum / (np.maximum.reduce(amplitudes) + epsilon) - 1) / (config.n_scales-1)
        weight = 1 / (1 + np.exp(config.spread_gain * (config.spread_cutoff-width)))
        numerator += weight * np.maximum(energy-threshold, 0)
        denominator += amplitude_sum
    output = np.clip(numerator / (denominator + epsilon), 0, 1)[region].astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite phase congruency output")
    return PhaseCongruencyResult(output, config, tuple(thresholds), perf_counter()-started)
