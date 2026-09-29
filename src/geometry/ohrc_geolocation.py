"""OHRC geolocation from all supplied rectilinear control records, not corners."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


class OHRCGeolocation:
    """Bilinear lon/lat interpolation in full-image pixel-center coordinates.

    Controls columns: longitude, latitude, pixel, scan. A complete Cartesian
    control grid is required; invalid records/holes are never silently filled.
    Longitude is unwrapped continuously, outputs may cross 180/360. No forward
    extrapolation. Inverse is local Newton inversion initialized at the nearest
    control, returns explicit validity and NaN for unsupported solutions.
    """
    def __init__(self, controls: np.ndarray):
        a = np.asarray(controls, dtype=float)
        if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
            raise ValueError('Expected finite Nx4 geolocation controls')
        if np.any(a[:, 2:] < 0) or np.any(np.abs(a[:, 1]) > 90):
            raise ValueError('Invalid geolocation controls')
        self.x, self.y = np.unique(a[:, 2]), np.unique(a[:, 3])
        if min(len(self.x), len(self.y)) < 2 or len(a) != len(self.x)*len(self.y):
            raise ValueError('Complete control grid with at least 2x2 samples required')
        if len(np.unique(a[:, 2:], axis=0)) != len(a):
            raise ValueError('Duplicate control positions')
        a = a[np.lexsort((a[:, 2], a[:, 3]))]
        grid = a[:, :2].reshape(len(self.y), len(self.x), 2).copy()
        grid[:, :, 0] = np.rad2deg(np.unwrap(np.unwrap(np.deg2rad(grid[:, :, 0]), axis=1), axis=0))
        self.grid = grid
        self._interp = RegularGridInterpolator((self.y, self.x), grid, bounds_error=True)
        self._tree = cKDTree(grid.reshape(-1, 2))
        self._pixels = a[:, 2:]

    def forward(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError('Expected finite Nx2 pixel/scan coordinates')
        return self._interp(p[:, ::-1])

    def inverse(self, geographic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(geographic, dtype=float).copy()
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError('Expected finite Nx2 longitude/latitude')
        if len(q) == 0:
            return np.empty((0, 2)), np.empty(0, bool)
        center = float(self.grid[:, :, 0].mean())
        q[:, 0] = center+(q[:, 0]-center+180)%360-180
        p = self._pixels[self._tree.query(q)[1]].copy()
        lower, upper = np.array([self.x[0], self.y[0]]), np.array([self.x[-1], self.y[-1]])
        active = np.ones(len(p), bool)
        converged = np.zeros(len(p), bool)
        for _ in range(12):
            indices = np.flatnonzero(active & ~converged)
            if not len(indices):
                break
            v = p[indices]
            current = self.forward(v)
            converged[indices] = np.linalg.norm(current-q[indices], axis=1) < 1e-12
            indices = indices[~converged[indices]]
            if not len(indices):
                continue
            v = p[indices]
            current = self.forward(v)
            derivatives = []
            for axis in range(2):
                lo, hi = v.copy(), v.copy()
                lo[:, axis] = np.maximum(v[:, axis]-.1, lower[axis])
                hi[:, axis] = np.minimum(v[:, axis]+.1, upper[axis])
                derivatives.append((self.forward(hi)-self.forward(lo))/(hi[:, axis]-lo[:, axis])[:, None])
            jac = np.stack(derivatives, axis=2)
            invertible = np.abs(np.linalg.det(jac)) > 1e-18
            active[indices[~invertible]] = False
            selected = indices[invertible]
            if len(selected):
                p[selected] -= np.linalg.solve(jac[invertible], (current[invertible]-q[selected])[..., None])[..., 0]
            active &= (p >= lower).all(axis=1) & (p <= upper).all(axis=1)
        valid = active.copy()
        if active.any():
            valid[active] &= np.linalg.norm(self.forward(p[active])-q[active], axis=1) < 1e-8
        p[~valid] = np.nan
        return p, valid


def held_out_validation(controls: np.ndarray) -> dict:
    """Withhold every fifth interior scan row; retain complete remaining grid.

    Validation checks interpolation against supplied product geometry, not
    external ground control. Angular errors and inverse pixel errors separate.
    """
    from src.evaluation.metrics import residual_statistics
    y = np.unique(controls[:, 3])
    withheld = y[2:-1:5]
    if not len(withheld):
        raise ValueError('Insufficient scan rows for holdout')
    keep = ~np.isin(controls[:, 3], withheld)
    model = OHRCGeolocation(controls[keep])
    truth = controls[~keep]
    prediction = model.forward(truth[:, 2:])
    delta = prediction-truth[:, :2]
    delta[:, 0] = (delta[:, 0]+180)%360-180
    recovered, valid = model.inverse(truth[:, :2])
    pixel_errors = np.linalg.norm(recovered[valid]-truth[valid, 2:], axis=1)
    return dict(sample_count=len(truth), held_out_scan_rows=withheld.tolist(), training_count=int(keep.sum()),
        longitude_absolute_degrees=residual_statistics(np.abs(delta[:, 0])),
        latitude_absolute_degrees=residual_statistics(np.abs(delta[:, 1])),
        inverse_pixel_displacement=residual_statistics(pixel_errors), inverse_valid_count=int(valid.sum()),
        inverse_invalid_count=int((~valid).sum()),
        predicted_lonlat=prediction.tolist(), truth_lonlat=truth[:, :2].tolist(),
        note='Withheld product geometry consistency, not independently measured geographic accuracy')
