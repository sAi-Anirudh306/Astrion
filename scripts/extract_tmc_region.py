"""CLI: python -m scripts.extract_tmc_region IMAGE --lat-min -44 --lat-max -42.6 --output data/processed/tycho

Writes unchanged scientific samples as .npy and a separate full-resolution
8-bit percentile preview. Refuses to overwrite any existing output.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.ingestion.tmc_loader import load_tmc_latitude_region
from src.preprocessing.normalization import normalize_intensity


def main() -> None:
    """Extract a latitude strip and save source provenance and measured statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to((root / "data/processed").resolve()):
        raise ValueError("Outputs must be under the project's data/processed directory")
    stem = f"{args.image.stem}_lat_{args.lat_min:g}_{args.lat_max:g}"
    paths = {"scientific": output / f"{stem}.npy", "preview": output / f"{stem}_preview.png",
             "report": output / f"{stem}.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Output already exists; choose another output directory")
    started = perf_counter()
    result = load_tmc_latitude_region(args.image, args.lat_min, args.lat_max, args.metadata)
    read_seconds = perf_counter() - started
    percentiles = np.percentile(result.data, [0, 1, 5, 50, 95, 99, 100])
    output.mkdir(parents=True, exist_ok=True)
    with paths["scientific"].open("xb") as stream:
        np.save(stream, result.data, allow_pickle=False)
    # Check the saved product without materializing a second full array.
    saved = np.load(paths["scientific"], mmap_mode="r", allow_pickle=False)
    for row in range(0, len(saved), 256):
        np.testing.assert_array_equal(saved[row:row + 256], result.data[row:row + 256])
    normalized = normalize_intensity(result.data)
    normalized *= 255
    np.rint(normalized, out=normalized)
    preview = normalized.astype(np.uint8)
    del normalized
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError("Preview PNG encoding failed")
    with paths["preview"].open("xb") as stream:
        stream.write(encoded.tobytes())
    report = {"source_image": str(result.source_path), "source_label": str(result.metadata_path),
              "source_metadata": result.metadata, "shape": result.data.shape, "dtype": result.data.dtype.str,
              "bytes_read": result.data.nbytes, "read_seconds": read_seconds,
              "percentiles": dict(zip(("min", "p1", "p5", "median", "p95", "p99", "max"), percentiles.tolist())),
              "preview": "full-resolution uint8, existing default 1-99 percentile normalization; display only",
              "limitation": "latitude rows approximated from linear footprint edges, not precise geolocation",
              "outputs": {key: str(path) for key, path in paths.items()}}
    with paths["report"].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
