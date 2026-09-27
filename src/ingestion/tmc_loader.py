"""Read metadata-labelled TMC scientific arrays and browse PNG products."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .image_loader import ImageData
from .metadata import read_metadata


def _dtype(data_type: str) -> np.dtype[Any]:
    """Resolve explicitly supported PDS4 types without native-endian guesses."""
    supported = {"UnsignedLSB2": "<u2", "UnsignedByte": "u1"}
    if data_type not in supported:
        raise ValueError(f"Unsupported data type: {data_type!r}")
    return np.dtype(supported[data_type])


def _read_raw(path: Path, metadata: dict[str, Any], dtype: np.dtype[Any]) -> NDArray[Any]:
    """Validate the complete binary layout before reading unchanged samples."""
    height, width = metadata["image_height"], metadata["image_width"]
    offset = metadata["offset_bytes"]
    expected = offset + height * width * dtype.itemsize
    if expected != metadata["file_size_bytes"]:
        raise ValueError(
            f"Dimension/file-size mismatch: dimensions {(height, width)}, dtype {dtype.str} "
            f"and offset {offset} require {expected} bytes; label declares "
            f"{metadata['file_size_bytes']} bytes"
        )
    samples = np.fromfile(path, dtype=dtype, count=height * width, offset=offset)
    if samples.size != height * width:
        raise ValueError(f"File-size mismatch while reading {path}: incomplete image data")
    sequence = metadata["axis_sequence"]
    shape = (height, width) if sequence["Line"] == 1 else (width, height)
    order = "C" if metadata["axis_index_order"] == "Last Index Fastest" else "F"
    array = samples.reshape(shape, order=order)
    return array if sequence["Line"] == 1 else array.T


def load_tmc_image(
    image_path: str | Path, metadata_path: str | Path | None = None,
) -> ImageData:
    """Load TMC .img samples or a browse .png, preserving dtype and values.

    Requires a PDS4 label (adjacent .xml by default). FileNotFoundError
    identifies absent input; ValueError identifies malformed metadata,
    unsupported types/layouts, corrupt PNGs, and size/dimension mismatches.
    PNG dimensions/dtype describe decoded pixels, while file_size describes
    the compressed file. No calibration or visualization conversion is applied.
    """
    path = Path(image_path).resolve()
    label = Path(metadata_path).resolve() if metadata_path is not None else path.with_suffix(".xml")
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in {".img", ".png"}:
        raise ValueError(f"Unsupported TMC image format: {path.suffix}")
    metadata = read_metadata(label)
    instrument = (metadata.get("instrument") or "").strip().lower()
    if instrument not in {"terrain mapping camera", "tmc", "tmc-2", "tmc2"}:
        raise ValueError(f"Unsupported sensor in TMC metadata: {instrument!r}")
    if metadata["file_name"] != path.name:
        raise ValueError(f"Metadata file_name mismatch: {metadata['file_name']!r} != {path.name!r}")
    actual_size = path.stat().st_size
    if actual_size != metadata["file_size_bytes"]:
        raise ValueError(
            f"File-size mismatch for {path}: expected {metadata['file_size_bytes']} bytes, "
            f"found {actual_size} bytes"
        )
    dtype = _dtype(metadata["data_type"])
    if path.suffix.lower() == ".img":
        data = _read_raw(path, metadata, dtype)
        kind = "raw"
    else:
        if metadata["offset_bytes"] != 0:
            raise ValueError("Unsupported PNG offset: expected zero")
        encoded = np.fromfile(path, dtype=np.uint8)
        try:
            data = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise ValueError(f"Invalid PNG image data: {path}") from exc
        if data is None:
            raise ValueError(f"Invalid PNG image data: {path}")
        kind = "browse"
    expected_shape = (metadata["image_height"], metadata["image_width"])
    if data.shape != expected_shape:
        raise ValueError(f"Dimension mismatch: label specifies {expected_shape}, decoded {data.shape}")
    if data.dtype != dtype:
        raise ValueError(f"Data type mismatch: label specifies {dtype}, decoded {data.dtype}")
    return ImageData(data, metadata, path, label, "TMC", kind)
