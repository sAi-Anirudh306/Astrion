"""Experiment 1A: original Borrow K + large WAC -> metadata ROI -> NEW LoFTR.

Run: .venv/Scripts/python.exe -m scripts.auto_localization
Outputs are confined to new directories under results/v2_auto_localization.
The historical Tycho crop is opened only by the final evaluation stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import rasterio
from rasterio.enums import Resampling
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle

from src.ingestion.metadata import read_metadata
from src.ingestion.tmc_loader import load_tmc_image
from src.ingestion.wac_loader import localize_tmc_reference
from src.geometry.georeferencing import FootprintMapping, project_to_reference, sample_bilinear
from src.geometry.ransac import verify_affine, RANSACConfig
from src.geometry.registration import warp_affine
from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.learned import LoFTRMatcher, LearnedMatches
from src.matching.ratio_test import RatioMatch, RatioTestResult
from src.matching.cross_modal import support_metrics, ReliabilityPolicy
from src.evaluation.metrics import localization_overlap
from src.evaluation.experiments import write_json
from src.evaluation.final_package import digest, protected_snapshot, check_protected, point_rows, write_csv
from src.evaluation.visualization import (plot_image, plot_matches, display_image, show_display,
                                          overlay_image, checkerboard_image)


def git(root: Path, *args: str) -> str:
    """Read repository provenance without changing Git state."""
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def new_output(root: Path, output: Path) -> Path:
    """Constrain writes to the authorized experiment tree; never reuse a run."""
    output = output.resolve()
    if not output.is_relative_to((root/'results/v2_auto_localization').resolve()):
        raise ValueError('Output must be under results/v2_auto_localization')
    output.mkdir(parents=True, exist_ok=False)
    return output


def provenance(root: Path, inputs: dict[str, Path]) -> dict:
    """Hash actual inputs and current code, including uncommitted experiment code."""
    paths = sorted(set(root.glob('src/**/*.py')) | set(root.glob('scripts/*.py')))
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'opencv-python', 'rasterio', 'torch', 'kornia', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(branch=git(root, 'branch', '--show-current'), commit=git(root, 'rev-parse', 'HEAD'),
        v1_tag_object=git(root, 'rev-parse', 'astrion-v1-sih-submission'),
        v1_tag_commit=git(root, 'rev-parse', 'astrion-v1-sih-submission^{commit}'),
        working_tree_status=git(root, 'status', '--short'), software=versions,
        inputs={name: dict(path=str(path.resolve()), size=path.stat().st_size, sha256=digest(path))
                for name, path in inputs.items()},
        code_sha256={p.relative_to(root).as_posix(): digest(p) for p in paths})


def figure(rows: int = 1, cols: int = 1, size: tuple = (8, 6)) -> tuple:
    fig = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(fig)
    return fig, fig.subplots(rows, cols, squeeze=False)


def save_view(path: Path, pixels: np.ndarray, mask: np.ndarray, title: str) -> None:
    fig, axes = figure()
    plot_image(axes[0, 0], pixels, mask, title)
    fig.savefig(path, dpi=150)
    fig.clear()


def reference_overview(wac: Path, localization: dict, output: Path) -> np.ma.MaskedArray:
    """Read only a decimated display; science uses the full-resolution ROI window."""
    with rasterio.open(wac) as source:
        ratio = min(1., 1200/max(source.shape))
        shape = tuple(max(1, round(n*ratio)) for n in source.shape)
        overview = source.read(1, out_shape=shape, masked=True, resampling=Resampling.nearest)
    fig, axes = figure(size=(10, 10))
    ax = axes[0, 0]
    shown = display_image(overview)
    ax.imshow(shown, cmap='gray', extent=(0, localization['source_shape'][1],
                                         localization['source_shape'][0], 0))
    draw_location(ax, localization)
    ax.set(title='Large WAC orthographic reference: visible hemisphere centered at 0 N, 0 E',
           xlabel='WAC column (native pixels)', ylabel='WAC row (native pixels)')
    ax.legend(loc='lower left')
    fig.savefig(output/'reference_overview.png', dpi=150)
    fig.clear()
    return shown


def draw_location(ax, localization: dict) -> None:
    boundary = np.asarray(localization['localization']['pixel_boundary'])
    ax.plot(*boundary.T, color='#ffb347', linewidth=1.5, label='Metadata footprint')
    left, top, right, bottom = localization['localization']['automatic_pixel_bounds']
    ax.add_patch(Rectangle((left, top), right-left, bottom-top, fill=False,
                          edgecolor='#00d5c7', linewidth=1.5, label='Automatic ROI + safety margin'))


def prepare_pair(raw: np.ndarray, localization: dict, output: Path) -> tuple:
    """Reuse v1 native preprocessing then bilinear map sampling on the new ROI.

    A single float32 native processing copy is released after sampling. Native
    scalar GSD normalization is retained as a diagnostic; matching uses actual
    projected WAC grid spacing, avoiding a second 6.27/100 downsample.
    """
    mapping = FootprintMapping(localization['corners'], *raw.shape)
    roi = output/'automatic_roi.tif'
    projected = project_to_reference(raw, mapping, roi, output/'source_projected.tif')
    with rasterio.open(roi) as source:
        reference, reference_mask = source.read(1), source.read_masks(1) > 0
    processed = preprocess_image(raw)
    gsd = localization['metadata']['pixel_resolution_m_per_pixel']
    target = float(localization['resolution'][0])
    nominal = normalize_resolution(processed.data, gsd, target)
    save_view(output/'source_native_scale.png', nominal.data, np.ones(nominal.data.shape, bool),
              'Original full TMC at nominal 100 m sampling / display only')
    source_processing = sample_bilinear(processed.data, projected.source_pixels,
                                       projected.valid.ravel()).reshape(reference.shape)
    del processed
    common = projected.valid & reference_mask
    if not common.any():
        raise ValueError('Projected observation has no valid WAC overlap')
    # Exact established v1 policy: independently normalize WAC, black common-invalid
    # processing pixels, then require all four neighboring endpoint pixels valid.
    reference_processing = preprocess_image(reference).data
    source_level = normalize_resolution(np.where(common, source_processing, 0).astype(np.float32), target, target)
    reference_level = normalize_resolution(np.where(common, reference_processing, 0).astype(np.float32), target, target)
    scale = dict(native_source_gsd_m=gsd, reference_map_spacing_m=target,
        native_nominal_normalization=nominal.metadata(),
        native_nominal_factor=nominal.nominal_factor,
        matching_source_scale=source_level.metadata(), matching_reference_scale=reference_level.metadata(),
        projection_output_shape=list(reference.shape),
        policy='v1 bilinear native-to-WAC projection handles geometry and scale; normalize_resolution is identity on matching map grids',
        limitation='100 m is projected map spacing, not uniform surface GSD; high-latitude orthographic foreshortening. Bilinear projection can alias.')
    save_view(output/'automatic_roi.png', reference, reference_mask, 'Automatic WAC ROI from full product metadata')
    save_view(output/'source_prepared.png', source_level.data, common, 'TMC / native preprocessing then approximate map projection')
    save_view(output/'reference_prepared.png', reference_level.data, common, 'WAC / established common-mask processing policy')
    np.savez_compressed(output/'prepared_pair.npz', source=source_level.data, reference=reference_level.data, common_mask=common)
    return projected, reference, reference_mask, source_level.data, reference_level.data, common, scale


def infer_and_verify(a: np.ndarray, b: np.ndarray, common: np.ndarray,
                     checkpoint: Path, output: Path) -> tuple:
    """One new inference, confidence >= .20, unchanged affine and spatial gates."""
    import torch
    from scripts.tycho_map_projected import mask_membership
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tick = perf_counter()
    matcher = LoFTRMatcher(checkpoint)
    raw = matcher.match(a, b)
    np.savez_compressed(output/'new_inference_raw.npz', source=raw.source,
                        destination=raw.destination, confidence=raw.confidence)
    valid = mask_membership(raw.source, common) & mask_membership(raw.destination, common)
    matches = LearnedMatches(raw.source[valid], raw.destination[valid], raw.confidence[valid], raw.metadata).filter_confidence(.2)
    pairs = tuple(RatioMatch(i, i, None, None, None) for i in range(len(matches.source)))
    adapter = RatioTestResult(pairs, len(pairs), 0, 0, 0, None, 0.)
    fit, failure = None, None
    try:
        fit = verify_affine(adapter, matches.source, matches.destination, RANSACConfig())
    except (ValueError, RuntimeError) as exc:
        failure = str(exc)
    keep = fit.inlier_mask if fit else np.zeros(len(pairs), bool)
    metrics = support_metrics(matches.source, matches.destination, fit.matrix if fit else None,
        keep, b.shape, raw_matches=len(raw.source), runtime_seconds=perf_counter()-tick)
    metrics.update(evidence_origin='NEW LoFTR inference on automatically generated Borrow K/WAC pair',
        fit_failure=failure, invalid_endpoint_rejections=int((~valid).sum()),
        confidence_threshold=.2, inference=raw.metadata,
        residual_interpretation='Fitted correspondence residuals in WAC ROI pixels, not localization error')
    np.savez_compressed(output/'matches.npz', source=matches.source, destination=matches.destination,
                        confidence=matches.confidence, inlier_mask=keep)
    rows, fields = point_rows(matches.source, matches.destination, matches.confidence, keep)
    write_csv(output/'match_points.csv', rows, fields)
    for filename, subset, title in [('correspondences.png', np.ones(len(keep), bool), 'NEW LoFTR candidates'),
                                     ('inliers.png', keep, 'Geometrically verified pairs')]:
        fig, axes = figure(size=(12, 6))
        plot_matches(axes[0, 0], a, b, matches.source[subset], matches.destination[subset],
                     source_mask=common, reference_mask=common, inlier_mask=keep[subset], title=title)
        fig.savefig(output/filename, dpi=150)
        fig.clear()
    return metrics, matches, keep


def register_if_reliable(metrics: dict, projected, reference: np.ndarray,
                         reference_mask: np.ndarray, output: Path):
    """Only the unchanged shared gate can permit scientific registration artifacts."""
    if metrics['reliability']['status'] != 'RELIABLE':
        write_json(output/'registration_withheld.json', metrics['reliability'])
        return None
    registered = warp_affine(projected.data, np.asarray(metrics['transform']), reference.shape,
                             validity_mask=projected.valid)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(output/'registered.tif', 'w', **projected.profile) as target:
            target.write(registered.image, 1)
            target.write_mask(registered.valid_mask.astype(np.uint8)*255)
            target.update_tags(affine_matrix=json.dumps(metrics['transform']),
                               limitation='Approximate footprint geolocation; fitted residuals are not absolute accuracy')
    with rasterio.open(output/'registered.tif') as saved:
        np.testing.assert_array_equal(saved.read(1), registered.image)
        np.testing.assert_array_equal(saved.read_masks(1) > 0, registered.valid_mask)
        if saved.transform != projected.profile['transform'] or saved.crs != projected.profile['crs']:
            raise ValueError('Registration grid changed during export')
    save_view(output/'registered.png', registered.image, registered.valid_mask, 'Accepted registration / interpolated TMC DN')
    for name, make in [('overlay', overlay_image), ('checkerboard', checkerboard_image)]:
        fig, axes = figure()
        show_display(axes[0, 0], make(registered.image, reference, registered.valid_mask, reference_mask),
                     f'{name.title()} / common valid coverage')
        fig.savefig(output/f'{name}.png', dpi=150)
        fig.clear()
    return registered


def evaluate_old_crop(root: Path, localization: dict) -> dict:
    """Post-experiment evaluation only. Identity mismatch disqualifies old truth."""
    receipt_path = root/'data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    label = Path(receipt['tmc_label'])
    old_metadata = read_metadata(label)
    old_roi = Path(receipt['output_tif'])
    with rasterio.open(old_roi) as old:
        if old.crs != rasterio.crs.CRS.from_wkt(localization['crs']):
            raise ValueError('Evaluation crop CRS differs from automatic ROI')
    w = receipt['window']
    old_bounds = [w['column_start'], w['row_start'], w['column_stop'], w['row_stop']]
    same = old_metadata['logical_identifier'] == localization['source_product_id']
    return dict(receipt_path=str(receipt_path), receipt_sha256=digest(receipt_path),
        comparison_product_id=old_metadata['logical_identifier'], comparison_crop=str(old_roi),
        comparison_crop_sha256=digest(old_roi), comparison_pixel_bounds=old_bounds,
        comparison_use='Evaluation only, opened after localization, new inference and registration decision',
        **localization_overlap(localization['localization']['automatic_pixel_bounds'], old_bounds,
                               localization['resolution'][0], same_observation=same))


def summary_figure(output: Path, overview, localization, projected, reference, reference_mask,
                   matches, keep, metrics, registered) -> None:
    fig, axes = figure(2, 3, size=(18, 11))
    ax = axes[0, 0]
    ax.imshow(overview, cmap='gray', extent=(0, localization['source_shape'][1], localization['source_shape'][0], 0))
    draw_location(ax, localization)
    ax.set_title('1 | Large WAC / visible lunar hemisphere')
    ax.legend(fontsize=7, loc='lower left')
    plot_image(axes[0, 1], reference, reference_mask, '2 | Automatically extracted ROI')
    plot_image(axes[0, 2], projected.data, projected.valid, '3 | Original TMC projected onto ROI')
    plot_matches(axes[1, 0], projected.data, reference, matches.source, matches.destination,
                 source_mask=projected.valid, reference_mask=reference_mask, inlier_mask=keep,
                 max_matches=60, title='4 | New correspondence evidence')
    if registered is None:
        axes[1, 1].set_axis_off()
        axes[1, 1].text(.1, .5, '5 | REGISTRATION WITHHELD\nExisting reliability gate failed', fontsize=16)
    else:
        plot_image(axes[1, 1], registered.image, registered.valid_mask, '5 | Accepted registered TMC')
    axes[1, 2].set_axis_off()
    axes[1, 2].text(.02, .95,
        f"Borrow K / NEW inference\nCandidates: {metrics['candidate_matches']}\nInliers: {metrics['verified_inliers']}\n"
        f"Fitted RMSE: {metrics['rmse']} px\nSpatial cells: {metrics['spatial_occupied_cells']}/16\n"
        f"Gate: {metrics['reliability']['status']}\n\nMetadata-guided localization only.\n"
        'Tycho crop is a different observation.\nAbsolute localization error unavailable.\nNo global image-only localization claim.',
        va='top', fontsize=12)
    fig.suptitle('ASTRION V2 / Experiment 1A / Original observation + large reference', fontsize=18)
    fig.savefig(output/'experiment_overview.png', dpi=150)
    fig.clear()


def run(root: Path, output: Path, margin_km: float = 5.) -> dict:
    started = perf_counter()
    if git(root, 'branch', '--show-current') != 'astrion-v2-auto-localization':
        raise ValueError('Experiment must run on astrion-v2-auto-localization')
    source = root/'data/test/borrow_k/data/raw/20260629/ch2_tmc_nrf_20260629T2059373111_d_img_d18.img'
    wac = root/'data/reference/lroc/WAC_GLOBAL_O000N0000_100M.TIF'
    checkpoint = root/'models/loftr_outdoor_kornia.ckpt'
    output = new_output(root, output)
    protected = protected_snapshot(root, output)
    write_json(output/'protection_before.json', protected)
    origin = provenance(root, dict(source=source, label=source.with_suffix('.xml'), reference=wac, checkpoint=checkpoint))
    write_json(output/'provenance.json', origin)
    (output/'implementation.patch').write_text(git(root, 'diff', '--no-ext-diff'), encoding='utf-8')
    print('Validated branch; inputs and preserved evidence inventoried.', flush=True)
    raw = load_tmc_image(source, mmap=True)
    tick = perf_counter()
    localization = localize_tmc_reference(raw.metadata_path, wac, output/'automatic_roi.tif', margin_km=margin_km)
    localization_seconds = perf_counter()-tick
    write_json(output/'localization.json', localization)
    write_json(output/'predicted_footprint.json', dict(coordinate_system='Lunar selenographic lon/lat degrees; NOT Earth GeoJSON',
        corners=localization['corners'], sampled_boundary=localization['geographic_boundary']))
    print('Automatic ROI:', localization['localization']['automatic_pixel_bounds'], flush=True)
    overview = reference_overview(wac, localization, output)
    tick = perf_counter()
    projected, reference, reference_mask, a, b, common, scale = prepare_pair(raw.data, localization, output)
    preparation_seconds = perf_counter()-tick
    write_json(output/'scale.json', scale)
    print('Prepared matching grid:', a.shape, '; running NEW LoFTR inference.', flush=True)
    metrics, matches, keep = infer_and_verify(a, b, common, checkpoint, output)
    write_json(output/'metrics.json', metrics)
    registered = register_if_reliable(metrics, projected, reference, reference_mask, output)
    # First semantic access to any historical crop is here, after the result exists.
    comparison = evaluate_old_crop(root, localization)
    write_json(output/'localization_evaluation.json', comparison)
    summary_figure(output, overview, localization, projected, reference, reference_mask, matches, keep, metrics, registered)
    check_protected(root, protected)
    for identity in origin['inputs'].values():
        if digest(Path(identity['path'])) != identity['sha256']:
            raise ValueError('Input content changed')
    if git(root, 'rev-parse', 'astrion-v1-sih-submission') != origin['v1_tag_object']:
        raise ValueError('Preserved v1 tag changed')
    result = dict(experiment='ASTRION V2 Experiment 1A', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        evidence_origin='NEW inference, no v1 points imported', source_product_id=localization['source_product_id'],
        reference_product_id=wac.stem, mode='A: metadata-guided', mode_b='Not implemented; future work',
        metadata_roi_extraction='PASS',
        automatic_localization=('PASS' if metrics['reliability']['status'] == 'RELIABLE' else 'FAIL'),
        localization_assessment_basis='Metadata traceability and new correspondence gate; independent same-observation ROI truth unavailable',
        independently_validated_localization_error=None,
        configuration=dict(margin_map_km=margin_km,
            margin_rationale='Fixed 5 km default chosen before inference as a modest map-plane safety buffer; not a calibrated uncertainty or tuned inlier optimum',
            preprocessing=asdict(PreprocessingConfig()), ransac=asdict(RANSACConfig()),
            reliability=asdict(ReliabilityPolicy()), confidence_threshold=.2, seed=0),
        scale=scale, localization_comparison=comparison, metrics=metrics,
        registration='AVAILABLE' if registered is not None else 'WITHHELD',
        runtime_seconds=dict(localization=localization_seconds, preparation=preparation_seconds,
            inference=metrics['inference']['runtime_seconds'], matching_and_verification=metrics['runtime_seconds'],
            total_including_protection=perf_counter()-started),
        preservation=dict(existing_data_and_results_unchanged=True, used_inputs_sha256_unchanged=True,
            v1_tag_unchanged=True, scope='All existing data/results size+mtime; <=8 MB SHA256; used raw, WAC, label, checkpoint fully hashed'),
        limitations=['No same-observation validated crop: old Tycho crop cannot measure Borrow K localization error.',
            'Four-corner approximation, not rigorous orthorectification; no camera/DEM used.',
            'Orthographic reference covers visible hemisphere, not full lunar globe; map GSD differs from surface GSD.',
            'Native-to-map bilinear sampling can alias; inherited v1 common-mask policy can miss offsets outside predicted footprint.',
            'One observation and one fixed margin; no robustness or global image-only localization evidence.',
            'Terrestrial pretrained LoFTR; engineering reliability is not independent geographic truth.'])
    write_json(output/'experiment.json', result)
    report = (f"AUTOMATIC LOCALIZATION: {result['automatic_localization']} (independent absolute accuracy unmeasured)\n"
        f"Metadata ROI extraction: PASS\nSource: {source}\nReference: {wac}\n"
        f"Automatic WAC bounds [left,top,right,bottom]: {localization['localization']['automatic_pixel_bounds']}\n"
        f"Margin: {margin_km} map km\nNative TMC GSD: {scale['native_source_gsd_m']} m; WAC map spacing: {scale['reference_map_spacing_m']} m\n"
        f"NEW LoFTR candidates: {metrics['candidate_matches']}; inliers: {metrics['verified_inliers']}\n"
        f"Inlier ratio: {metrics['inlier_ratio']}; fitted RMSE: {metrics['rmse']} WAC pixels\n"
        f"Spatial cells: {metrics['spatial_occupied_cells']}/16; gate: {metrics['reliability']['status']}\n"
        f"Registration: {result['registration']}\n"
        'Validated Tycho ROI is a different observation. Its displacement/IoU are not localization error.\n'
        f"Descriptive old-ROI IoU: {comparison['iou']}; center displacement: {comparison['center_displacement_pixels']} map pixels\n"
        f"Runtime: {result['runtime_seconds']}\nPreservation: PASS\n\n"+'\n'.join(result['limitations'])+'\n')
    (output/'report.txt').write_text(report, encoding='utf-8')
    write_json(output/'output_hashes.json', {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(report, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/v2_auto_localization'))
    parser.add_argument('--margin-km', type=float, default=5.)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output, args.margin_km)


if __name__ == '__main__':
    main()
