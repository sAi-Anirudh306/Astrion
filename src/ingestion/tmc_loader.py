"""Read metadata-labelled TMC scientific arrays and browse PNG products."""

from pathlib import Path
from numbers import Integral, Real
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


def _read_raw(path: Path, metadata: dict[str, Any], dtype: np.dtype[Any],
              row_range: tuple[int, int] | None = None, *, mmap: bool = False) -> NDArray[Any]:
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
    if row_range is not None:
        start, stop = row_range
        sequence = metadata["axis_sequence"]
        contiguous_rows = ((sequence["Line"] == 1 and metadata["axis_index_order"] == "Last Index Fastest")
                           or (sequence["Line"] == 2 and metadata["axis_index_order"] == "First Index Fastest"))
        if not contiguous_rows:
            raise ValueError("Row extraction requires contiguous Sample values within each Line")
        offset += start * width * dtype.itemsize
        height = stop - start
    samples = (np.memmap(path, dtype=dtype, mode="r", shape=(height * width,), offset=offset)
               if mmap else np.fromfile(path, dtype=dtype, count=height * width, offset=offset))
    if samples.size != height * width:
        raise ValueError(f"File-size mismatch while reading {path}: incomplete image data")
    sequence = metadata["axis_sequence"]
    shape = (height, width) if sequence["Line"] == 1 else (width, height)
    order = "C" if metadata["axis_index_order"] == "Last Index Fastest" else "F"
    array = samples.reshape(shape, order=order)
    return array if sequence["Line"] == 1 else array.T


def load_tmc_image(
    image_path: str | Path, metadata_path: str | Path | None = None,
    *, row_range: tuple[int, int] | None = None, mmap: bool = False,
) -> ImageData:
    """Load TMC .img samples or a browse .png, preserving dtype and values.

    Requires a PDS4 label (adjacent .xml by default). FileNotFoundError
    identifies absent input; ValueError identifies malformed metadata,
    unsupported types/layouts, corrupt PNGs, and size/dimension mismatches.
    Optional row_range is a zero-based [start, stop) raw row interval; only
    those rows are read. Source metadata stays intact, with a subset record.
    PNG dimensions/dtype describe decoded pixels, while file_size describes
    the compressed file. No calibration or visualization conversion is applied.
    mmap=True returns a read-only mapping for raw products, after all checks.
    """
    path = Path(image_path).resolve()
    label = Path(metadata_path).resolve() if metadata_path is not None else path.with_suffix(".xml")
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in {".img", ".png"}:
        raise ValueError(f"Unsupported TMC image format: {path.suffix}")
    if not isinstance(mmap, bool) or (mmap and path.suffix.lower() != ".img"):
        raise ValueError("mmap must be boolean and is supported only for raw .img products")
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
    if row_range is not None:
        if (not isinstance(row_range, tuple) or len(row_range) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in row_range)
                or not 0 <= row_range[0] < row_range[1] <= metadata["image_height"]):
            raise ValueError("row_range must be integer (start, stop) within image height, start < stop")
        if path.suffix.lower() != ".img":
            raise ValueError("Row extraction is supported only for raw .img products")
    if path.suffix.lower() == ".img":
        data = _read_raw(path, metadata, dtype, row_range, mmap=mmap)
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
    if row_range is not None:
        expected_shape = (row_range[1] - row_range[0], metadata["image_width"])
        metadata["subset"] = {"row_start": int(row_range[0]), "row_stop_exclusive": int(row_range[1]),
                              "image_height": expected_shape[0], "image_width": expected_shape[1]}
    if data.shape != expected_shape:
        raise ValueError(f"Dimension mismatch: label specifies {expected_shape}, decoded {data.shape}")
    if data.dtype != dtype:
        raise ValueError(f"Data type mismatch: label specifies {dtype}, decoded {data.dtype}")
    return ImageData(data, metadata, path, label, "TMC", kind)


def latitude_row_range(metadata: dict[str, Any], latitude_min: float,
                       latitude_max: float) -> tuple[int, int]:
    """Approximate a full-width latitude strip using linear footprint edges.

    Corner latitudes are treated as first/last row centers. Each side is
    interpolated independently; take the envelope and round outward. Supports
    either along-track direction. Rejects out-of-footprint requests, constant
    latitude edges and edges running in opposing directions. This four-corner
    approximation is not orthorectification or a per-pixel geolocation model.
    Returned indices are zero-based [start, stop), clipped to image bounds.
    """
    for value in (latitude_min, latitude_max):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not -90 <= value <= 90:
            raise ValueError("Latitude bounds must be finite degrees in [-90, 90]")
    if latitude_min >= latitude_max:
        raise ValueError("latitude_min must be less than latitude_max")
    height = metadata.get("image_height")
    if isinstance(height, bool) or not isinstance(height, Integral) or height < 2:
        raise ValueError("Image height must be an integer >= 2")
    endpoints = []
    directions = []
    for side in ("left", "right"):
        top, bottom = metadata.get(f"upper_{side}_latitude"), metadata.get(f"lower_{side}_latitude")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or not -90 <= v <= 90 for v in (top, bottom)):
            raise ValueError("Missing or invalid footprint corner latitude")
        if abs(bottom - top) < 1e-12:
            raise ValueError("Cannot infer rows from a constant latitude edge")
        if latitude_min < min(top, bottom) or latitude_max > max(top, bottom):
            raise ValueError("Requested latitude interval is outside the footprint edge coverage")
        directions.append(np.sign(bottom - top))
        endpoints.extend((lat - top) / (bottom - top) * (height - 1) for lat in (latitude_min, latitude_max))
    if directions[0] != directions[1]:
        raise ValueError("Footprint edges have opposing latitude directions")
    return max(0, int(np.floor(min(endpoints)))), min(height, int(np.ceil(max(endpoints))) + 1)


def load_tmc_latitude_region(image_path: str | Path, latitude_min: float,
                             latitude_max: float, metadata_path: str | Path | None = None) -> ImageData:
    """Read only the approximate latitude strip, preserving original sample values."""
    label = Path(metadata_path) if metadata_path is not None else Path(image_path).with_suffix(".xml")
    metadata = read_metadata(label)
    rows = latitude_row_range(metadata, latitude_min, latitude_max)
    result = load_tmc_image(image_path, label, row_range=rows)
    result.metadata["subset"].update({"requested_latitude_min": latitude_min,
        "requested_latitude_max": latitude_max,
        "geolocation_method": "linear interpolation of both footprint edges; outward-rounded row envelope"})
    return result


def row_subregion_footprint(metadata: dict[str, Any], start_row: int,
                            stop_row: int) -> dict[str, dict[str, float]]:
    """Approximate four corners for [start_row, stop_row), without reading pixels.

    Consistent with latitude_row_range: product corners represent first/last
    row centers, so fractions are start/(height-1) and (stop-1)/(height-1).
    Left and right edges interpolate independently, including longitude along
    the shortest arc. Output longitudes use 0..360 degrees east. This is only
    product-corner interpolation, NOT precise per-pixel geolocation.
    """
    height = metadata.get("image_height")
    if (isinstance(height, bool) or not isinstance(height, Integral) or height < 2
            or any(isinstance(v, bool) or not isinstance(v, Integral) for v in (start_row, stop_row))
            or not 0 <= start_row < stop_row <= height):
        raise ValueError("Invalid row interval or full image height")
    corners = {}
    for side in ("left", "right"):
        values = [metadata.get(f"{edge}_{side}_{axis}") for edge in ("upper", "lower")
                  for axis in ("latitude", "longitude")]
        if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in values):
            raise ValueError("Missing or invalid footprint coordinates")
        lat0, lon0, lat1, lon1 = values
        if not (-90 <= lat0 <= 90 and -90 <= lat1 <= 90 and -180 <= lon0 <= 360 and -180 <= lon1 <= 360):
            raise ValueError("Footprint coordinates outside supported degree ranges")
        delta_lon = (lon1 - lon0 + 180) % 360 - 180
        if abs(delta_lon) == 180:
            raise ValueError("Ambiguous longitude interpolation across 180 degrees")
        for edge, row in (("upper", start_row), ("lower", stop_row - 1)):
            fraction = row / (height - 1)
            corners[f"{edge}_{side}"] = {"latitude": float(lat0 + fraction * (lat1 - lat0)),
                                         "longitude": float((lon0 + fraction * delta_lon) % 360)}
    return corners
