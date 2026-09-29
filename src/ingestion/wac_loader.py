"""Windowed lunar WAC extraction from approximate TMC quadrilateral footprints."""
import json
from pathlib import Path
from numbers import Real
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window, bounds

from .metadata import read_metadata
from .tmc_loader import row_subregion_footprint


def signed_longitude(longitude: float) -> float:
    """Convert finite degrees east (including 0..360) to [-180, 180)."""
    if isinstance(longitude, bool) or not isinstance(longitude, Real) or not np.isfinite(longitude):
        raise ValueError("Longitude must be finite degrees")
    return float((longitude + 180) % 360 - 180)


def extract_geolocated_wac(wac_path: str | Path, geographic_points: np.ndarray,
                           output_path: str | Path) -> dict:
    """Extract a covered bounding window using supplied lon/lat controls.

    Uses all supplied control/boundary points, never substitutes TMC corners.
    One raster pixel of padding; full footprint must lie inside the raster and
    visible lunar orthographic domain. Values, nodata, CRS and mask preserved.
    """
    from src.geometry.georeferencing import project_coordinates, lunar_geographic_crs
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    points = np.asarray(geographic_points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx2 geolocation controls')
    with rasterio.open(wac_path) as source:
        lunar_geographic_crs(source.crs)
        projected = project_coordinates(points, source.crs)
        if not np.isfinite(projected).all():
            raise ValueError('Geolocation outside visible WAC projection')
        inverse = ~source.transform
        xy = np.array([inverse @ tuple(p) for p in projected])
        lo, hi = np.floor(xy.min(0)).astype(int)-1, np.ceil(xy.max(0)).astype(int)+1
        if np.any(lo < 0) or np.any(hi > [source.width, source.height]):
            raise ValueError('Complete OHRC footprint not covered by WAC bounds')
        window = Window(int(lo[0]), int(lo[1]), int(hi[0]-lo[0]), int(hi[1]-lo[1]))
        data, mask = source.read(1, window=window), source.read_masks(1, window=window)
        if not (mask > 0).any():
            raise ValueError('WAC crop has no valid reference pixels')
        profile = source.profile.copy()
        profile.update(width=data.shape[1], height=data.shape[0], transform=source.window_transform(window))
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(path, 'w', **profile) as target:
                target.write(data, 1)
                target.write_mask(mask)
        left, bottom, right, top = bounds(window, source.transform)
        lon, lat = transform(source.crs, lunar_geographic_crs(source.crs),
                             [left, right, right, left], [bottom, bottom, top, top])
        result = dict(covered=True, source_path=str(wac_path), output_path=str(path),
            source_shape=list(source.shape), source_bounds=list(source.bounds),
            input_geographic_bounds=[*points.min(0), *points.max(0)],
            geographic_bounds=[min(lon), min(lat), max(lon), max(lat)],
            projected_bounds=[left, bottom, right, top],
            transform=list(profile['transform']), crs=source.crs.to_wkt(), resolution=list(source.res),
            nodata=source.nodata, shape=list(data.shape), valid_pixels=int((mask > 0).sum()),
            window=dict(column_start=int(lo[0]), row_start=int(lo[1]), column_stop=int(hi[0]), row_stop=int(hi[1])))
    with rasterio.open(path) as saved:
        np.testing.assert_array_equal(saved.read(1), data)
        np.testing.assert_array_equal(saved.read_masks(1), mask)
    return result


def extract_wac_region(label_path: str | Path, wac_path: str | Path,
                       start_row: int, stop_row: int, output_directory: str | Path) -> dict[str, Any]:
    """Extract a bounding raster window, without resampling or polygon masking.

    Uses Moon sphere R=1737400 m, never an Earth geographic CRS. Four TMC
    corners are approximate row-center coordinates from product-level edges,
    NOT precise per-pixel geolocation. Projected edges are sampled as well as
    corners to account for projection curvature; one source pixel of padding
    covers sampling/row-center uncertainty (not unknown geolocation error).
    Refuses partial raster coverage and invisible orthographic coordinates.
    Only the requested raster window is read; GDAL may read enclosing blocks.
    Output values, CRS, pixel spacing and nodata are preserved; the affine
    origin changes to the window origin. Existing outputs are never replaced.
    """
    metadata = read_metadata(label_path)
    corners = row_subregion_footprint(metadata, start_row, stop_row)
    directory = Path(output_directory).resolve()
    stem = f"wac_tmc_rows_{start_row}_{stop_row}"
    tif, png, report_path = (directory / (stem + suffix) for suffix in (".tif", "_preview.png", ".json"))
    if any(path.exists() for path in (tif, png, report_path)):
        raise FileExistsError("Extraction outputs already exist")
    geographic = CRS.from_string("+proj=longlat +R=1737400 +no_defs")
    names = ("upper_left", "upper_right", "lower_right", "lower_left")
    lons = [signed_longitude(corners[name]["longitude"]) for name in names]
    lats = [corners[name]["latitude"] for name in names]
    # Unwrap to avoid interpolating the long way around a longitude seam.
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lons)))
    edge_lons, edge_lats = [], []
    for i in range(4):
        j = (i + 1) % 4
        end_lon = unwrapped[i] + signed_longitude(unwrapped[j] - unwrapped[i])
        edge_lons.extend(signed_longitude(v) for v in np.linspace(unwrapped[i], end_lon, 257))
        edge_lats.extend(np.linspace(lats[i], lats[j], 257).tolist())
    with rasterio.open(wac_path) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("WAC must have a projected lunar CRS")
        params = source.crs.to_dict()
        radius = params.get("R", params.get("a"))
        if params.get("proj") != "ortho" or radius is None or abs(float(radius) - 1737400) > .01:
            raise ValueError("Expected orthographic Moon CRS with radius 1737400 m")
        if source.count != 1:
            raise ValueError("Expected a single-band WAC reference")
        x, y = transform(geographic, source.crs, lons, lats)
        ex, ey = transform(geographic, source.crs, edge_lons, edge_lats)
        if not np.isfinite([*x, *y, *ex, *ey]).all():
            raise ValueError("Footprint is outside the visible projection domain")
        inverse = ~source.transform
        pixels = np.array([inverse @ (a, b) for a, b in zip(ex, ey)])
        col0, row0 = np.floor(pixels.min(axis=0)).astype(int) - 1
        col1, row1 = np.ceil(pixels.max(axis=0)).astype(int) + 1
        if not (0 <= col0 < col1 <= source.width and 0 <= row0 < row1 <= source.height):
            raise ValueError("Complete footprint window is not covered by the WAC raster")
        window = Window(int(col0), int(row0), int(col1-col0), int(row1-row0))
        data = source.read(1, window=window)
        mask = source.read_masks(1, window=window)
        valid = (mask > 0) & np.isfinite(data)
        if not valid.any():
            raise ValueError("WAC window contains no valid pixels")
        profile = source.profile.copy()
        profile.update(width=int(window.width), height=int(window.height), transform=source.window_transform(window))
        directory.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(tif, "w", **profile) as target:
                target.write(data, 1)
                target.write_mask(mask)
                target.update_tags(**source.tags())
        report = {"tmc_label": str(Path(label_path).resolve()), "wac_source": str(Path(wac_path).resolve()),
            "tmc_rows": [start_row, stop_row], "corners": corners,
            "geographic_bounds_unwrapped_degrees": {"min_longitude": float(min(unwrapped)), "max_longitude": float(max(unwrapped)),
                                                      "min_latitude": min(lats), "max_latitude": max(lats)},
            "projected_corners": {name: [a, b] for name, a, b in zip(names, x, y)},
            "projected_footprint_bounds": [min(ex), min(ey), max(ex), max(ey)],
            "crop_projected_bounds": list(bounds(window, source.transform)),
            "window": {"row_start": int(row0), "row_stop": int(row1), "column_start": int(col0), "column_stop": int(col1)},
            "shape": list(data.shape), "valid_pixel_count": int(valid.sum()),
            "crs": source.crs.to_wkt(), "resolution": list(source.res), "nodata": source.nodata,
            "output_tif": str(tif), "output_png": str(png),
            "limitation": "Approximate product-corner interpolation; not precise per-pixel geolocation. Bounding crop includes pixels outside the quadrilateral."}
    low, high = np.percentile(data[valid], [1, 99])
    preview = np.zeros(data.shape, np.uint8)
    if high > low:
        preview[valid] = np.rint(np.clip((data[valid].astype(np.float64)-low)/(high-low), 0, 1)*255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError("PNG preview encoding failed")
    with png.open("xb") as stream:
        stream.write(encoded.tobytes())
    report["preview_percentiles"] = [float(low), float(high)]
    with rasterio.open(tif) as saved:
        np.testing.assert_array_equal(saved.read(1), data)
        assert saved.crs == profile["crs"] and saved.transform == profile["transform"]
        assert saved.nodata == profile["nodata"]
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report
