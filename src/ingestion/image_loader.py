"""Common scientific ingestion result and TMC/OHRC sensor dispatch."""

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
    *, window: tuple[int, int, int, int] | None = None,
) -> ImageData:
    """Dispatch TMC or OHRC with label-validated identity; AUTO reads XML.

    The label defaults to the image path with an .xml suffix. Unsupported
    sensors raise ValueError; no fallback to another sensor is performed.
    """
    selected = sensor.strip().upper()
    if selected == 'AUTO':
        from .metadata import read_metadata
        label = metadata_path or Path(image_path).with_suffix('.xml')
        instrument = str(read_metadata(label).get('instrument', '')).lower()
        selected = {'orbiter high resolution camera': 'OHRC', 'ohrc': 'OHRC',
                    'terrain mapping camera': 'TMC'}.get(instrument, instrument.upper())
    if selected == 'OHRC':
        from .ohrc_loader import load_ohrc_image
        return load_ohrc_image(image_path, metadata_path, window=window)
    if selected not in {"TMC", "TMC-2", "TMC2"}:
        raise ValueError(f"Unsupported sensor: {sensor!r}")
    if window is not None:
        raise ValueError('Generic window argument currently supported for OHRC only')
    from .tmc_loader import load_tmc_image

    return load_tmc_image(image_path, metadata_path)
