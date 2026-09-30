"""Footprint-based approximate map projection, NOT rigorous orthorectification.

No camera, orbit, terrain/DEM, control points or image registration are used.
The geographic model interpolates four approximate pixel-center corners.
"""
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from src.ingestion.wac_loader import signed_longitude


class FootprintMapping:
    """Bilinear lon/lat model: P=A+B*u+C*v+D*u*v.

    Input pixel (x,y) centers run 0..width-1 and 0..height-1. u=x/(width-1),
    v=y/(height-1). Geographic arrays are (longitude,latitude) degrees east.
    Longitudes unwrap about UL along shortest arcs; output uses [-180,180).
    Local, non-folding quadrilaterals only (longitude extent <180 degrees).
    """
    def __init__(self, corners: dict, height: int, width: int):
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 2 for n in (height, width)):
            raise ValueError("Image dimensions must be integers >=2")
        try:
            points = np.array([[corners[k]["longitude"], corners[k]["latitude"]]
                for k in ("upper_left", "upper_right", "lower_left", "lower_right")], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Four named geographic corners are required") from exc
        if not np.isfinite(points).all() or np.any(np.abs(points[:, 1]) > 90):
            raise ValueError("Corners must have finite coordinates and valid latitude")
        anchor = signed_longitude(float(points[0, 0]))
        points[:, 0] = [anchor+signed_longitude(float(lon-anchor)) for lon in points[:, 0]]
        if np.ptp(points[:, 0]) >= 180:
            raise ValueError("Footprint must have unambiguous longitude extent <180 degrees")
        self.height, self.width = int(height), int(width)
        self.a = points[0].copy()
        self.b, self.c = points[1]-points[0], points[2]-points[0]
        self.d = points[3]-points[1]-points[2]+points[0]
        determinants = [np.linalg.det(np.column_stack((self.b+self.d*v, self.c+self.d*u)))
                        for u,v in ((0,0), (0,1), (1,0), (1,1))]
        scale = max(np.linalg.norm(self.b), np.linalg.norm(self.c), np.linalg.norm(self.d))**2
        if (scale == 0 or min(abs(x) for x in determinants) <= 1e-10*scale
                or not (all(x > 0 for x in determinants) or all(x < 0 for x in determinants))):
            raise ValueError("Degenerate, folded or numerically singular footprint")

    def _evaluate(self, uv: np.ndarray) -> np.ndarray:
        return self.a + uv[:, :1]*self.b + uv[:, 1:]*self.c + np.prod(uv, axis=1)[:, None]*self.d

    def boundary(self, samples_per_edge: int = 257) -> np.ndarray:
        """Densely sample the closed image-center perimeter in lunar lon/lat."""
        if (isinstance(samples_per_edge, bool) or not isinstance(samples_per_edge, Integral)
                or samples_per_edge < 2):
            raise ValueError('Boundary needs at least two integer samples per edge')
        corners = np.array([[0, 0], [self.width-1, 0],
                            [self.width-1, self.height-1], [0, self.height-1], [0, 0]])
        pixels = np.concatenate([np.linspace(a, b, samples_per_edge)
                                 for a, b in zip(corners[:-1], corners[1:])])
        return self.forward(pixels)

    def forward(self, pixels: np.ndarray) -> np.ndarray:
        """Map an (N,2) pixel-center array to geographic lon/lat degrees."""
        pixels = np.asarray(pixels, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Pixels must be finite (N,2) coordinates")
        uv = pixels / [self.width-1, self.height-1]
        if np.any((uv < 0) | (uv > 1)):
            raise ValueError("Pixel coordinates outside image-center footprint")
        points = self._evaluate(uv)
        points[:, 0] = (points[:, 0]+180) % 360-180
        return points

    def inverse(self, geographic: np.ndarray, tolerance: float = 1e-10,
                max_iterations: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vector Newton inverse; return pixels, valid mask, geographic residual.

        Nonconvergent/outside/nonfinite queries have NaN pixels and false mask.
        Tolerance is degrees; bounds tolerance is 1e-9 in normalized coordinates.
        Degenerate/folding models are rejected at construction, not repaired.
        """
        if not np.isfinite(tolerance) or tolerance <= 0 or isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
            raise ValueError("Invalid inverse tolerance or iteration count")
        points = np.array(geographic, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Geographic points must have shape (N,2)")
        finite = np.isfinite(points).all(axis=1)
        points[~finite] = self.a
        points[:, 0] = self.a[0] + (points[:, 0]-self.a[0]+180) % 360-180
        uv = (points-self.a) @ np.linalg.inv(np.column_stack((self.b, self.c))).T
        uv = np.clip(uv, -1, 2)
        active = finite.copy()
        for _ in range(max_iterations):
            error = self._evaluate(uv)-points
            pending = active & (np.linalg.norm(error, axis=1) > tolerance)
            if not pending.any():
                break
            ju = self.b+uv[:, 1:]*self.d
            jv = self.c+uv[:, :1]*self.d
            det = ju[:, 0]*jv[:, 1]-ju[:, 1]*jv[:, 0]
            safe = np.abs(det) > 1e-14
            active &= safe
            pending &= safe
            delta = np.zeros_like(uv)
            delta[pending, 0] = (error[pending, 0]*jv[pending, 1]-error[pending, 1]*jv[pending, 0])/det[pending]
            delta[pending, 1] = (ju[pending, 0]*error[pending, 1]-ju[pending, 1]*error[pending, 0])/det[pending]
            uv -= np.clip(delta, -1, 1)
        residual = np.linalg.norm(self._evaluate(uv)-points, axis=1)
        valid = active & (residual <= tolerance) & np.all((uv >= -1e-9) & (uv <= 1+1e-9), axis=1)
        pixels = np.clip(uv, 0, 1)*[self.width-1, self.height-1]
        pixels[~valid] = np.nan
        residual[~finite] = np.nan
        return pixels, valid, residual


def lunar_geographic_crs(projected: CRS) -> CRS:
    """Derive the geographic sphere from WAC CRS; reject Earth/nonlunar grids."""
    params = projected.to_dict()
    radius = params.get("R", params.get("a"))
    if (params.get("proj") != "ortho" or radius is None or not 1_000_000 < float(radius) < 2_000_000
            or abs(float(params.get("b", radius))-float(radius)) > .01):
        raise ValueError("Expected a lunar-sphere orthographic reference CRS")
    return CRS.from_dict(proj="longlat", R=float(radius))


def project_coordinates(geographic: np.ndarray, projected: CRS) -> np.ndarray:
    """Transform lon/lat to the supplied WAC lunar CRS using Rasterio/PROJ."""
    source = lunar_geographic_crs(projected)
    points = np.asarray(geographic, dtype=float)
    x, y = transform(source, projected, points[:, 0].tolist(), points[:, 1].tolist())
    return np.column_stack((x, y))


def sample_bilinear(source: np.ndarray, pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Read four native samples per valid output center, retaining fractional DN.

    Supports read-only memory-mapped arrays. No antialiasing/area averaging is
    added: bilinear sampling is the requested baseline and can alias at 100 m.
    """
    output = np.full(len(pixels), np.nan, np.float32)
    xy = pixels[valid]
    x0, y0 = np.floor(xy).astype(int).T
    x1, y1 = np.minimum(x0+1, source.shape[1]-1), np.minimum(y0+1, source.shape[0]-1)
    dx, dy = (xy-np.column_stack((x0, y0))).T
    output[valid] = ((1-dx)*(1-dy)*source[y0,x0] + dx*(1-dy)*source[y0,x1]
                     + (1-dx)*dy*source[y1,x0] + dx*dy*source[y1,x1])
    return output


@dataclass(frozen=True)
class MapProjectionResult:
    """Scientific float32 DN, mask, inverse coordinates and exact reference grid."""
    data: np.ndarray
    valid: np.ndarray
    source_pixels: np.ndarray
    profile: dict
    maximum_inverse_residual_degrees: float


def project_to_reference(source: np.ndarray, mapping: FootprintMapping,
                         reference_path: str | Path, output_path: str | Path) -> MapProjectionResult:
    """Inverse-map WAC centers onto the TMC footprint; write a new float32 TIFF.

    Output CRS/affine/dimensions match WAC exactly. NaN nodata and an internal
    GDAL mask distinguish absence of TMC coverage from real zero intensity.
    WAC radiometric nodata does not remove otherwise valid TMC coverage.
    This is footprint-based approximate map projection, NOT orthorectification.
    """
    if not isinstance(source, np.ndarray) or source.shape != (mapping.height, mapping.width) or source.dtype.kind not in "uif":
        raise ValueError("Source must be a real 2D array matching footprint dimensions")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    with rasterio.open(reference_path) as reference:
        geographic = lunar_geographic_crs(reference.crs)
        profile = reference.profile.copy()
        rows, cols = np.indices((reference.height, reference.width), dtype=float)
        x, y = reference.transform @ (cols.ravel()+.5, rows.ravel()+.5)
        lon, lat = transform(reference.crs, geographic, x.tolist(), y.tolist())
        pixels, valid, residual = mapping.inverse(np.column_stack((lon, lat)))
        sampled = sample_bilinear(source, pixels, valid)
        valid &= np.isfinite(sampled)
        sampled[~valid] = np.nan
        shape = (reference.height, reference.width)
    data, mask = sampled.reshape(shape), valid.reshape(shape)
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(destination, "w", **profile) as target:
            target.write(data, 1)
            target.write_mask(mask.astype(np.uint8)*255)
            target.update_tags(geolocation_method="footprint-based approximate map projection",
                limitation="Not rigorous orthorectification; no camera model or DEM", resampling="bilinear at output pixel centers")
    return MapProjectionResult(data, mask, pixels, profile,
        float(residual[valid].max()) if valid.any() else 0.)
