"""Milestone 8: saved-transform scientific registration; no matching or tuning.

Run: python -m scripts.tycho_registration
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import rasterio
from src.geometry.registration import warp_affine, transform_points
from scripts.real_tmc_wac import save_png


def preview(image: np.ndarray, valid: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Display-only P1/P99 mapping; scientific raster remains unchanged."""
    out = np.zeros(image.shape, np.uint8)
    out[valid] = np.rint(np.clip((image[valid]-bounds[0])/max(float(bounds[1]-bounds[0]), 1e-12), 0, 1)*255).astype(np.uint8)
    return out


def main() -> None:
    """Write, reopen and validate the exact WAC-grid registration products."""
    tick = perf_counter()
    root = Path(__file__).resolve().parents[1]
    moving_path = root/'data/processed/tycho/map_projected/tmc_on_wac_grid.tif'
    reference_path = root/'data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.tif'
    previous = root/'results/experiment_tycho_map_projected'
    report_path = previous/'report.json'
    points_path = previous/'loftr_0.20_matches.npz'
    output = root/'data/processed/tycho/registered'
    results = root/'results/milestone_08_registration'
    for directory in (output, results):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Refusing to overwrite existing experiment: {directory}')
    sources = (moving_path, reference_path, report_path, points_path)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    saved = json.loads(report_path.read_text())
    row = next(r for r in saved['matching'] if r['name'] == 'loftr_0.20')
    matrix = np.asarray(row['affine_matrix'], np.float64)
    with rasterio.open(moving_path) as src, rasterio.open(reference_path) as ref:
        moving, source_mask = src.read(1), src.read_masks(1) > 0
        fixed, fixed_mask = ref.read(1), ref.read_masks(1) > 0
        profile = ref.profile.copy()
        if src.crs != ref.crs or src.transform != ref.transform or src.shape != ref.shape:
            raise ValueError('Expected source and reference on identical initial grid')
        crs, grid, shape = ref.crs, ref.transform, ref.shape
    source_mask &= np.isfinite(moving)
    warp_tick = perf_counter()
    registered = warp_affine(moving, matrix, shape, validity_mask=source_mask)
    warp_seconds = perf_counter()-warp_tick
    with np.load(points_path) as points:
        keep = points['inlier_mask']
        if len(keep) != row['candidates'] or int(keep.sum()) != row['inliers']:
            raise ValueError('Saved correspondence counts disagree with report')
        src_xy, dst_xy = points['source'][keep], points['destination'][keep]
    residuals = np.linalg.norm(transform_points(src_xy, matrix)-dst_xy, axis=1)
    stats = {name: float(function(residuals)) for name, function in
             [('min', np.min), ('median', np.median), ('mean', np.mean), ('max', np.max)]}
    stats['rmse'] = float(np.sqrt(np.mean(residuals**2)))
    if not np.isclose(stats['rmse'], row['rmse'], atol=1e-9):
        raise ValueError('Saved-transform residuals differ from experiment')
    for directory in (output, results):
        directory.mkdir(parents=True, exist_ok=True)
    limitation = ('Underlying TMC georeferencing is footprint-based approximate map projection, '
                  'not rigorous orthorectification. Correspondence residuals are not external '
                  'geographic accuracy; no subpixel geographic accuracy or radiometric equivalence is claimed.')
    profile.update(driver='GTiff', dtype='float32', count=1, nodata=np.nan, compress='deflate')
    tif = output/'tmc_registered_to_wac.tif'
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(tif, 'w', **profile) as dst:
            dst.write(registered.image, 1)
            dst.write_mask(registered.valid_mask.astype(np.uint8)*255)
            dst.update_tags(scientific_limitation=limitation, interpolation='nodata-aware bilinear',
                            transform_direction='source pixel to destination pixel',
                            affine_matrix=json.dumps(matrix.tolist()))
    with rasterio.open(tif) as check:
        readback = check.read(1)
        assert check.crs == crs and check.transform == grid and check.shape == shape
        assert check.dtypes == ('float32',) and np.isnan(check.nodata)
        np.testing.assert_array_equal(check.read_masks(1)>0, registered.valid_mask)
        assert np.isfinite(readback[registered.valid_mask]).all()
        assert np.isnan(readback[~registered.valid_mask]).all()
        np.testing.assert_array_equal(readback, registered.image)
    tmc_bounds = np.percentile(moving[source_mask], [1, 99])
    wac_bounds = np.percentile(fixed[fixed_mask], [1, 99])
    before = preview(moving, source_mask, tmc_bounds)
    after = preview(registered.image, registered.valid_mask, tmc_bounds)
    reference = preview(fixed, fixed_mask, wac_bounds)
    save_png(output/'tmc_registered_preview.png', after)
    save_png(output/'tmc_registered_valid_mask.png', registered.valid_mask.astype(np.uint8)*255)
    overlay = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    overlap = registered.valid_mask & fixed_mask
    combined = np.rint(.5*reference.astype(float)+.5*after).astype(np.uint8)
    overlay[overlap] = combined[overlap, None]
    save_png(results/'overlay.png', overlay)
    panels = []
    for title, array in [('A: WAC reference', reference), ('B: TMC before', before),
                         ('C: TMC registered', after), ('D: Actual 50% overlay', overlay)]:
        panel = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR) if array.ndim == 2 else array
        panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        panels.append(panel)
    save_png(results/'side_by_side.png', np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
    difference = np.zeros(shape, np.uint8)
    difference[overlap] = np.abs(after.astype(np.int16)-reference.astype(np.int16))[overlap].astype(np.uint8)
    save_png(results/'difference_diagnostic.png', difference)
    sample = np.array([[0, 0], [320, 0], [0, 312], [320, 312], [160, 156]], float)
    for path in sources:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path.relative_to(root))]
    metadata = dict(source_image=str(moving_path), reference_image=str(reference_path),
        transform_source=str(report_path), correspondence_source=str(points_path),
        affine_matrix=matrix.tolist(), transform_direction='source (x,y) pixel -> destination (x,y) pixel',
        interpolation='bilinear, excluding nodata contributions with normalized valid weights',
        mask_interpolation='nearest neighbor', source_valid_pixels=int(source_mask.sum()),
        registered_valid_pixels=int(registered.valid_mask.sum()),
        registered_valid_percentage=float(100*registered.valid_mask.mean()),
        output_dimensions=list(shape), dtype='float32', crs=crs.to_wkt(),
        geotransform=list(grid), resolution=[abs(grid.a), abs(grid.e)], nodata='NaN',
        ransac_candidate_count=row['candidates'], ransac_inlier_count=row['inliers'],
        inlier_ratio=row['inlier_ratio'], pre_registration_correspondence_rmse=row['rmse'],
        post_transform_correspondence_residuals=stats, spatial_occupancy=row['occupied_wac_cells'],
        coordinate_checks={'source_xy': sample.tolist(), 'destination_xy': transform_points(sample, matrix).tolist()},
        scientific_limitation=limitation,
        difference_diagnostic='Absolute difference of independently display-normalized images on valid overlap only; not radiometric error.',
        preview_percentiles={'tmc_shared_before_after': tmc_bounds.tolist(), 'wac': wac_bounds.tolist()},
        input_sha256=hashes, validation='Exact grid, CRS, shape, float32, binary mask, finite valid and NaN invalid readback passed',
        software={'opencv': cv2.__version__, 'numpy': np.__version__, 'rasterio': rasterio.__version__},
        warp_runtime_seconds=warp_seconds, total_runtime_seconds=perf_counter()-tick)
    (results/'registration_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
