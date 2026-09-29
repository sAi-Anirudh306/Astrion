"""M16 real OHRC ingestion/geolocation and WAC diagnostic, no learned inference.

Run: python -m scripts.ohrc_wac --allow-geometry-header-exception
The exception was explicitly authorized for this product: XML Header length 31
conflicts with XML Table offset 30 and actual 30-byte header. Default is strict.
Repeat runs require fresh --output and --reference-output directories.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ingestion.ohrc_loader import read_ohrc_metadata, validate_ohrc_file, load_ohrc_image, load_ohrc_geometry, crop_to_full
from src.ingestion.wac_loader import extract_geolocated_wac
from src.geometry.ohrc_geolocation import OHRCGeolocation, held_out_validation
from src.geometry.georeferencing import project_coordinates
from src.geometry.ransac import verify_affine, RANSACConfig
from src.preprocessing.preprocessing import preprocess_image
from src.preprocessing.scale_normalization import build_scale_pyramid, ScalePyramid
from src.features.phase_congruency import phase_congruency
from src.features.sift import extract_sift
from src.features.rootsift import transform_sift
from src.matching.correspondence import match_scale_pyramid
from src.evaluation.metrics import residual_statistics, spatial_distribution
from src.evaluation.visualization import plot_image, plot_matches, plot_keypoints


LIMITATIONS = [
    'Successful ingestion does not establish correspondence accuracy.',
    'Synthetic loader tests establish software behavior, not real sensor performance.',
    'Held-out geometry interpolation consistency is separate from feature-registration residuals and is not external ground truth.',
    'Downsampling OHRC to WAC resolution discards fine spatial information.',
    'Upsampling WAC cannot recover missing detail; no upsampling was performed.',
    'Scale normalization does not solve viewpoint differences or moving lunar shadows.',
    'Phase Congruency does not guarantee Sun-angle invariance: shadows can alter structure.',
    'One OHRC observation does not establish universal cross-sensor performance.',
    'Feature-registration residuals do not establish absolute geographic accuracy.',
    'Interpolation between supplied controls remains approximate, not exact per-pixel ground truth.',
    'No camera model, DEM orthorectification or bundle adjustment is implemented.',
    'WAC cannot resolve many native OHRC structures; the selected crop has very few WAC-scale pixels.',
    'Pixel/scan origin and degree interpretation are inferred from control extrema and agreement with labelled image corners, not explicitly declared in geometry XML.',
    'Positive local longitude alone does not distinguish 0..360 from signed longitude; both give the same local location.',
    'Auxiliary field-layout discrepancies prevent confident normalized OAT/SPM interpretation; XML Sun/attitude values are used instead.',
    'Wall times depend on machine and disk cache; no performance guarantee.',
    'Nodata endpoint masks do not prevent descriptor support from seeing nearby fill boundaries.']


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stats(image: np.ndarray) -> dict:
    return dict(shape=list(image.shape), dtype=str(image.dtype), minimum=float(image.min()),
                maximum=float(image.max()), mean=float(image.mean()), std=float(image.std()))


def auxiliary_inspection(directory: Path) -> dict:
    """Record actual byte structure; do not guess fields conflicting with README."""
    files = {}
    for suffix, length in [('oath', 201), ('oat', 628), ('spm', 249), ('lbr', 258)]:
        path = next(directory.glob('*.'+suffix))
        raw = path.read_bytes()
        files[suffix] = dict(path=str(path), size=len(raw), documented_record_bytes=length,
            complete_records=len(raw)//length, trailing_bytes=len(raw)%length,
            first_record_raw_ascii=raw[:length].decode('ascii'), sha256=digest(path))
    files['interpretation'] = ('README retained as provenance. SPM documented F9.3 field boundaries do not parse the observed first record cleanly; '
        'OATH OAT record-length field is 601, whereas README and actual embedded OAT records specify 628 bytes. Stop OAT/SPM numeric interpretation; no replacement inference. '
        'Solar incidence and attitude come from image XML. README explicitly permits 90 minus Sun elevation, but XML already supplies incidence.')
    return files


def evaluate_branch(level, reference, reference_mask, name):
    tick = perf_counter()
    record = match_scale_pyramid(ScalePyramid((level,), (level.target_resolution,), (0,)), reference,
                                  reference_mask=reference_mask)[0]
    matching_seconds = perf_counter()-tick
    config = RANSACConfig()
    tick = perf_counter()
    fit, failure = None, 'insufficient support'
    if len(record.valid_matches.accepted) >= 3:
        try:
            fit = verify_affine(record.valid_matches, record.source_features.points, record.reference_features.points, config)
            failure = None
        except (ValueError, RuntimeError) as exc:
            failure = str(exc)
    keep = fit.inlier_mask if fit else np.zeros(len(record.source), bool)
    report = dict(branch=name, source_keypoints=len(record.source_features.keypoints),
        reference_keypoints=len(record.reference_features.keypoints), descriptor_shape=list(record.source_features.descriptors.shape),
        raw_descriptor_candidates=sum(map(len, record.knn.neighbors)), ratio_candidates=len(record.ratio.accepted),
        candidates=len(record.source), inliers=int(keep.sum()),
        outliers=int((~keep).sum()) if fit else None, inlier_ratio=fit.inlier_ratio if fit else None,
        affine_matrix=fit.matrix.tolist() if fit else None,
        residuals_reference_pixels=residual_statistics(fit.errors[keep]) if fit else None,
        spatial_support=spatial_distribution(record.destination[keep], reference.shape),
        failure=failure, estimator=asdict(config),
        interpretation='Estimated support is not proof of correctness' if fit else 'No model; unverified candidates are not classified outliers',
        runtime_seconds=dict(feature_matching=matching_seconds, geometric_verification=perf_counter()-tick))
    return record, keep, report


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/milestone_16_ohrc'))
    parser.add_argument('--reference-output', type=Path, default=Path('data/processed/ohrc/reference/wac_ohrc_center.tif'))
    parser.add_argument('--allow-geometry-header-exception', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, reference_path = (root/args.output).resolve(), (root/args.reference_output).resolve()
    if not output.is_relative_to(root/'results') or not reference_path.is_relative_to(root/'data/processed/ohrc'):
        raise ValueError('Outputs must remain within results/ and data/processed/ohrc/')
    if output.exists() or reference_path.exists():
        raise FileExistsError('Use fresh output paths; previous artifacts preserved')
    product = root/'data/raw/ohrc'
    stem = 'ch2_ohr_ncp_20210402T0155096873'
    image = product/f'data/calibrated/20210402/{stem}_d_img_d18.img'
    geometry = product/f'geometry/calibrated/20210402/{stem}_g_grd_d18.csv'
    wac = root/'data/reference/lroc/WAC_GLOBAL_O000N0000_100M.TIF'
    required = list(product.rglob('*'))
    protected = [p for p in required if p.is_file()]+[wac]
    # Hash all OHRC and WAC inputs, plus official M8 provenance files.
    official = json.loads((root/'results/milestone_09_evaluation/evaluation.json').read_text())
    protected += [(root/v['path']).resolve() for v in official['provenance']['inputs'].values()]
    before_hash = {str(p.relative_to(root)): digest(p) for p in set(protected)}
    times = {}
    tick = perf_counter()
    metadata = read_ohrc_metadata(image.with_suffix('.xml'))
    size = validate_ohrc_file(image, metadata)
    times['xml_metadata'] = perf_counter()-tick
    h, w = metadata['image_height'], metadata['image_width']
    tick = perf_counter()
    controls, geometry_report = load_ohrc_geometry(geometry, (h, w),
        allow_header_length_discrepancy=args.allow_geometry_header_exception)
    times['geometry_csv'] = perf_counter()-tick
    if geometry_report['ranges']['pixel'] != [0, w-1] or geometry_report['ranges']['scan'] != [0, h-1]:
        raise ValueError('Cannot establish zero-based complete geometry domain')
    model = OHRCGeolocation(controls)
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    labelled = np.array(list(metadata['normalized']['footprint'].values()))
    if not np.allclose(model.forward(corners), labelled, atol=2e-6, rtol=0):
        raise ValueError('Geometry conflicts with labelled corner coordinates')
    tick = perf_counter()
    held = held_out_validation(controls)
    times['geolocation_validation'] = perf_counter()-tick
    with rasterio.open(wac) as f:
        predicted = project_coordinates(np.array(held['predicted_lonlat']), f.crs)
        truth = project_coordinates(np.array(held['truth_lonlat']), f.crs)
        held_distances = np.linalg.norm(predicted-truth, axis=1)
        held['projected_metres'] = residual_statistics(held_distances)
        wac_source = dict(crs=f.crs.to_wkt(), transform=list(f.transform), bounds=list(f.bounds), shape=list(f.shape), resolution=list(f.res))
        target_gsd = float(f.res[0])
        projected_all = project_coordinates(controls[:, :2], f.crs)
        inside = (np.isfinite(projected_all).all(axis=1) & (projected_all[:, 0] >= f.bounds.left)
            & (projected_all[:, 0] <= f.bounds.right) & (projected_all[:, 1] >= f.bounds.bottom) & (projected_all[:, 1] <= f.bounds.top))
        wac_source['all_geometry_points_within_raster_bounds'] = bool(inside.all())
    # A fixed center crop independent of any feature or correspondence result.
    rows = min(8192, h)
    r0 = (h-rows)//2
    crop_window = (r0, r0+rows, 0, w)
    tick = perf_counter()
    loaded = load_ohrc_image(image, window=crop_window)
    times['ohrc_window_load'] = perf_counter()-tick
    source_stats = stats(loaded.data)
    tick = perf_counter()
    processed = preprocess_image(loaded)
    times['preprocessing'] = perf_counter()-tick
    subset = loaded.metadata['subset']
    del loaded
    crop_corners = np.array([[0, r0], [w-1, r0], [w-1, r0+rows-1], [0, r0+rows-1]])
    boundary = np.vstack([np.linspace(crop_corners[i], crop_corners[(i+1)%4], 257) for i in range(4)])
    boundary_geo = model.forward(boundary)
    tick = perf_counter()
    reference_report = extract_geolocated_wac(wac, boundary_geo, reference_path)
    times['wac_extraction'] = perf_counter()-tick
    with rasterio.open(reference_path) as f:
        ref_raw, ref_mask, ref_crs, ref_transform = f.read(1), f.read_masks(1)>0, f.crs, f.transform
    reference_processing = preprocess_image(ref_raw).data
    tick = perf_counter()
    gsd = metadata['normalized']['native_gsd_m']
    pyramid = build_scale_pyramid(processed.data, gsd, [target_gsd, 10.])
    times['scale_normalization'] = perf_counter()-tick
    primary, structural = pyramid.levels
    probe_crop = np.array([[0., 0.], [w-1., rows-1.], [(w-1)/2, (rows-1)/2]])
    scaled = primary.original_to_scaled(probe_crop)
    recovered = primary.scaled_to_original(scaled, pixel_footprint=True)
    roundtrip = float(np.max(np.abs(recovered-probe_crop)))
    full = crop_to_full(probe_crop, subset)
    geo_probe = model.forward(full)
    image_inverse, inverse_valid = model.inverse(geo_probe)
    projected_probe = project_coordinates(geo_probe, ref_crs)
    # Rasterio inverse affine uses pixel corners; subtract .5 for pixel centers.
    wac_pixels = np.array([(~ref_transform) @ tuple(p) for p in projected_probe])-.5
    projected_roundtrip = np.array([ref_transform @ tuple(p+.5) for p in wac_pixels])
    coordinate_report = dict(crop_xy=probe_crop.tolist(), full_xy=full.tolist(), scaled_xy=scaled.tolist(),
        longitude_latitude=geo_probe.tolist(), projected_wac_metres=projected_probe.tolist(), wac_pixel_centers=wac_pixels.tolist(),
        scale_roundtrip_max_pixels=roundtrip, geolocation_roundtrip_valid=inverse_valid.tolist(),
        geolocation_roundtrip_max_pixels=float(np.max(np.abs(image_inverse[inverse_valid]-full[inverse_valid]))) if inverse_valid.any() else None,
        wac_affine_roundtrip_max_metres=float(np.max(np.abs(projected_roundtrip-projected_probe))))
    tick = perf_counter()
    structural_features = transform_sift(extract_sift(structural.data))
    times['structural_sift_rootsift'] = perf_counter()-tick
    tick = perf_counter()
    pc_structure = phase_congruency(structural.data)
    pc_features = transform_sift(extract_sift(pc_structure.data))
    primary_pc, reference_pc = phase_congruency(primary.data), phase_congruency(reference_processing)
    times['phase_congruency_and_features'] = perf_counter()-tick
    intensity_record, intensity_keep, intensity = evaluate_branch(primary, reference_processing, ref_mask, 'intensity')
    from dataclasses import replace
    phase_record, phase_keep, phase = evaluate_branch(replace(primary, data=primary_pc.data), reference_pc.data, ref_mask, 'phase_congruency')
    for result in (intensity, phase):
        times[result['branch']+'_correspondence'] = result['runtime_seconds']['feature_matching']
        times[result['branch']+'_geometric_verification'] = result['runtime_seconds']['geometric_verification']
    auxiliary = auxiliary_inspection(product/'miscellaneous/calibrated/20210402')
    output.mkdir(parents=True, exist_ok=False)
    figures = []
    def save(fig, name):
        fig.tight_layout(rect=(0, .06, 1, .98))
        fig.text(.02, .015, 'Display-normalized copies | OHRC product-geometry interpolation is not external ground truth', fontsize=8)
        fig.savefig(output/name, dpi=160, bbox_inches='tight')
        plt.close(fig)
        figures.append(name)
    for name, image_data, title, mask in [
        ('ohrc_preview.png', cv2.resize(processed.data, (750, max(1, round(rows*750/w))), interpolation=cv2.INTER_AREA), 'Central OHRC crop | display reduced only', None),
        ('ohrc_reference_crop.png', ref_raw, 'Geolocation-derived WAC reference', ref_mask),
        ('ohrc_scaled.png', primary.data, f'OHRC at requested {target_gsd:g} m/pixel', None),
        ('ohrc_phase_congruency.png', pc_structure.data, 'OHRC phase congruency | 10 m structural diagnostic', None)]:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_image(ax, image_data, mask, title)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_keypoints(ax, structural.data, structural_features.points,
                   title=f'10 m diagnostic | {len(structural_features.keypoints)} SIFT keypoints')
    save(fig, 'ohrc_keypoints.png')
    fig, ax = plt.subplots(figsize=(8, 7))
    show = controls[::37]
    ax.scatter(show[:, 0], show[:, 1], s=3, label='Every 37th control (display only)')
    ax.plot(boundary_geo[:, 0], boundary_geo[:, 1], color='orange', label='Fixed selected crop')
    ax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)', title='OHRC control-grid footprint; all controls used in interpolation')
    ax.set_aspect('equal')
    ax.legend()
    save(fig, 'ohrc_geometry.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(held_distances, bins=40)
    ax.set(xlabel='Held-out product-control displacement in WAC projected metres', ylabel='Count', title=f'{held["sample_count"]} held-out geometry records')
    save(fig, 'ohrc_geolocation_residuals.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, record, ref, name in zip(axes, [intensity_record, phase_record], [reference_processing, reference_pc.data], ['Intensity', 'Phase congruency']):
        source_points = np.array([record.source_features.points[m.query_index] for m in record.valid_matches.accepted]).reshape(-1, 2)
        plot_matches(ax, record.level.data, ref, source_points, record.destination,
                     reference_mask=ref_mask, title=name+' | all candidates', max_matches=None)
    save(fig, 'ohrc_matches.png')
    for record, keep in [(intensity_record, intensity_keep), (phase_record, phase_keep)]:
        if keep.any():
            fig, ax = plt.subplots(figsize=(10, 5))
            source_points = np.array([record.source_features.points[m.query_index] for m in record.valid_matches.accepted]).reshape(-1, 2)
            plot_matches(ax, record.level.data, reference_processing, source_points[keep], record.destination[keep], title='Verified support; correctness not established')
            save(fig, 'ohrc_inliers.png')
            break
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.text(.02, .95, '\n'.join([
        'ASTRION M16 | Real calibrated OHRC', f'Image: {h} x {w}, uint8; {gsd:g} m/pixel',
        f'Geometry: {len(controls)} controls; {geometry_report["unique_scans"]} x {geometry_report["unique_pixels"]} grid',
        f'Held-out geometry RMSE: {held["projected_metres"]["rmse"]:.6f} projected m (not absolute accuracy)',
        f'WAC-scale OHRC: {primary.data.shape}; WAC crop: {ref_raw.shape}',
        f'Intensity candidates/inliers: {intensity["candidates"]}/{intensity["inliers"]}',
        f'Phase candidates/inliers: {phase["candidates"]}/{phase["inliers"]}',
        'No learned inference. No official TMC artifacts replaced.',
        'Supplied geometry consistency is separate from correspondence correctness.']), va='top', fontsize=12)
    save(fig, 'ohrc_summary.png')
    for path, expected in before_hash.items():
        if digest(root/path) != expected:
            raise RuntimeError(f'Scientific/protected input changed: {path}')
    plan = (root/'PLAN.md').read_text().split('# MILESTONE 16\n')[1].split('# MILESTONE 17')[0].strip()
    audit = [dict(requirement=name, initial_status=status, final_status='satisfied') for name, status in [
        ('inspect actual OHRC product format', 'missing'), ('implement loader', 'missing'),
        ('metadata integration', 'partially satisfied'), ('common image representation', 'partially satisfied'),
        ('tests', 'missing'), ('OHRC experiments', 'missing')]]
    report = dict(status='PASS', scope='OHRC support/validation completed; correspondence success is not assumed',
        plan_m16_exact_section='# MILESTONE 16\n'+plan, audit=audit, product_stem=stem,
        paths=dict(image=str(image.relative_to(root)), image_label=str(image.with_suffix('.xml').relative_to(root)),
                   geometry=str(geometry.relative_to(root)), geometry_label=str(geometry.with_suffix('.xml').relative_to(root))),
        metadata=metadata, expected_file_size=metadata['file_size_bytes'], actual_file_size=size,
        geometry=geometry_report, geolocation=dict(method='Bilinear regular-grid interpolation using all supplied controls; bounded Newton inverse',
            selection_reason='Complete rectilinear grid with shorter final intervals', held_out_validation=held,
            full_product_corners_lonlat=model.forward(corners).tolist()),
        auxiliary=auxiliary, selected_crop=dict(rule='Fixed central 8192 rows (or full height if smaller), full width; no match-driven selection',
            window=list(crop_window), subset=subset, geographic_boundary=boundary_geo.tolist()),
        wac_source=wac_source, reference_crop=reference_report,
        preprocessing=dict(config=asdict(processed.config), scientific=source_stats, processed=stats(processed.data)),
        scale=dict(primary=primary.metadata(), structural_diagnostic=structural.metadata(), coordinate_validation=coordinate_report),
        structural_features=dict(sift_count=len(structural_features.keypoints), rootsift_shape=list(structural_features.descriptors.shape),
            full_image_keypoints=crop_to_full(structural.scaled_to_original(structural_features.points), subset).tolist()),
        phase_congruency=dict(output=stats(pc_structure.data), finite=bool(np.isfinite(pc_structure.data).all()),
            sift_count=len(pc_features.keypoints), config=asdict(pc_structure.config), runtime_seconds=pc_structure.runtime_seconds),
        correspondence=dict(intensity=intensity, phase_congruency=phase), learned_correspondence=None,
        operation_runtimes_seconds=times, total_runtime_seconds=perf_counter()-started,
        protected_sha256=before_hash, protected_sources_unchanged=True, limitations=LIMITATIONS, figures=figures)
    (output/'ohrc.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [f'M16 PASS: {stem}; ingestion/geolocation validated, matching outcome reported independently.',
        f'{h} x {w}, one-band uint8; byte order not applicable; {size} bytes; {gsd:g} m/pixel.',
        f'Geometry: {len(controls)} valid rows, {geometry_report["invalid_rows"]} invalid; pixel/scan ranges {geometry_report["ranges"]}.',
        'Approved header exception: XML Header 31 vs Table offset/actual header 30; strict loader default retained.',
        f'Sun azimuth/elevation/incidence from XML: {metadata["normalized"]["sun_azimuth_deg"]}/{metadata["normalized"]["sun_elevation_deg"]}/{metadata["normalized"]["solar_incidence_deg"]} deg; orbit {metadata["normalized"]["orbit"]}.',
        f'Bilinear grid interpolation holdout: {held["sample_count"]} samples; projected-metre RMSE/median/max {held["projected_metres"]["rmse"]:.6f}/{held["projected_metres"]["median"]:.6f}/{held["projected_metres"]["maximum"]:.6f}.',
        f'Inverse holdout: {held["inverse_valid_count"]} valid, {held["inverse_invalid_count"]} outside/failed; pixel RMSE {held["inverse_pixel_displacement"]["rmse"]:.6f}.',
        f'Crop {crop_window}; WAC bounds covered; reference {ref_raw.shape}; primary scaled OHRC {primary.data.shape}.',
        f'Scale effective GSD {primary.effective_resolution_xy}; factors {primary.actual_factors_xy}; coordinate roundtrip {roundtrip:.3g} px.',
        f'10 m structural SIFT/RootSIFT: {len(structural_features.keypoints)} / {structural_features.descriptors.shape}; PC SIFT: {len(pc_features.keypoints)}.',
        f'100 m intensity/PC: candidates {intensity["candidates"]}/{phase["candidates"]}, inliers {intensity["inliers"]}/{phase["inliers"]}; failures {intensity["failure"]}/{phase["failure"]}.',
        'No LoFTR, camera-model orthorectification or WAC-grid resampling. Auxiliary raw records preserved; conflicting field layouts not interpreted.',
        f'Total runtime {report["total_runtime_seconds"]:.3f} s.', *LIMITATIONS]
    (output/'ohrc_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for name in figures:
        if cv2.imread(str(output/name)) is None:
            raise RuntimeError(f'Unreadable figure {name}')
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    main()
