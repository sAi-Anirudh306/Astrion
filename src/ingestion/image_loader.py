"""Common ingestion result and sensor dispatch (TMC only for Milestone 1)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ImageData:
    """Original decoded pixels and provenance, without scaling or normalization.

    ``raw`` denotes scientific samples; ``browse`` denotes a supplied preview.
    Pixels are an independent in-memory array, never a writable source mapping.
    """

    data: NDArray[Any]
    metadata: dict[str, Any]
    source_path: Path
    metadata_path: Path
    sensor: str
    product_kind: Literal["raw", "browse"]


def load_image(
    image_path: str | Path,
    sensor: str = "TMC",
    metadata_path: str | Path | None = None,
) -> ImageData:
    """Dispatch a TMC .img or browse .png with its PDS4 label.

    The label defaults to the image path with an .xml suffix. Unsupported
    sensors raise ValueError; no fallback to another sensor is performed.
    """
    if sensor.strip().upper() not in {"TMC", "TMC-2", "TMC2"}:
        raise ValueError(f"Unsupported sensor: {sensor!r}; only TMC is implemented")
    from .tmc_loader import load_tmc_image

    return load_tmc_image(image_path, metadata_path)
