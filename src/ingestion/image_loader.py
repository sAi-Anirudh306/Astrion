"""Common scientific ingestion result and TMC/OHRC/IIRS sensor dispatch."""

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
    bands: int | list[int] | None = None,
    allow_validated_signedness_exception: bool = False,
) -> ImageData:
    """Dispatch label-validated sensors; AUTO reads instrument independently of axes.

    IIRS integer bands return 2D samples; sequences return (band,row,column).
    No implicit hyperspectral-to-grayscale reduction is performed. The opt-in
    signedness exception is IIRS-specific and fully verified by that loader.

    The label defaults to the image path with an .xml suffix. Unsupported
    sensors raise ValueError; no fallback to another sensor is performed.
    """
    selected = sensor.strip().upper()
    if selected == 'AUTO':
        import xml.etree.ElementTree as ET
        from .metadata import NS, get_text
        label = metadata_path or Path(image_path).with_suffix('.xml')
        root = ET.parse(label).getroot()
        instruments = [get_text(c, 'pds:name') for c in root.findall('.//pds:Observing_System_Component', NS)
                       if get_text(c, 'pds:type') == 'Instrument']
        if len(instruments) != 1 or instruments[0] is None:
            raise ValueError('Expected one labelled instrument')
        instrument = instruments[0].lower()
        selected = {'orbiter high resolution camera': 'OHRC', 'ohrc': 'OHRC',
                    'terrain mapping camera': 'TMC', 'imaging infrared spectrometer': 'IIRS'}.get(instrument, instrument.upper())
    if selected == 'IIRS':
        from .iirs_loader import load_iirs_image
        return load_iirs_image(image_path, metadata_path, bands=bands, window=window,
            allow_validated_signedness_exception=allow_validated_signedness_exception)
    if bands is not None or allow_validated_signedness_exception:
        raise ValueError('Band selection and signedness exception are IIRS-only options')
    if selected == 'OHRC':
        from .ohrc_loader import load_ohrc_image
        return load_ohrc_image(image_path, metadata_path, window=window)
    if selected not in {"TMC", "TMC-2", "TMC2"}:
        raise ValueError(f"Unsupported sensor: {sensor!r}")
    if window is not None:
        raise ValueError('Generic window argument currently supported for OHRC only')
    from .tmc_loader import load_tmc_image

    return load_tmc_image(image_path, metadata_path)
