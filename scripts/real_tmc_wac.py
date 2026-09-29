"""Run the fixed-baseline real Tycho scale experiment: python -m scripts.real_tmc_wac."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np
import rasterio

from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.sift import extract_sift, prepare_sift_image, SIFTConfig
from src.features.rootsift import transform_sift
from src.matching.descriptor_matching import match_descriptors
from src.matching.ratio_test import filter_ratio
from src.geometry.ransac import verify_affine, RANSACConfig


def save_png(path: Path, image: np.ndarray) -> None:
    """Write derived visualization, checking encoder success."""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def draw_matches(path: Path, a: np.ndarray, b: np.ndarray, fa, fb, matches) -> None:
    """Show at most 60 evenly selected matches, without image resizing."""
    selected = np.linspace(0, len(matches)-1, min(60, len(matches)), dtype=int)
    lines = [cv2.DMatch(matches[i].query_index, matches[i].train_index,
                       float(matches[i].nearest_distance)) for i in selected]
    canvas = cv2.drawMatches(a, fa.keypoints, b, fb.keypoints, lines, None,
                            matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_png(path, canvas)


def main() -> None:
    """Reuse all validated stages; failures remain explicit null model results.

    Residuals are fitted-model consistency in WAC pixels, not ground-truth
    accuracy. Nominal physical scale does not correct projection differences.
    WAC nodata is black for this unmasked baseline, recorded as a limitation.
    """
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    moving = root / "data/processed/tycho/ch2_tmc_nrf_20211122T2123225722_d_img_d18_lat_-44_-42.6.npy"
    fixed = root / "data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.tif"
    output = root / "results/experiment_tycho_real_multiscale"
    output.mkdir(parents=True, exist_ok=False)
    source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (moving, fixed)}
    config, sift_config, ransac_config = PreprocessingConfig(), SIFTConfig(), RANSACConfig()
    raw = np.load(moving, mmap_mode="r", allow_pickle=False)
    tmc = preprocess_image(raw, config)
    with rasterio.open(fixed) as dataset:
        wac_raw = dataset.read(1)
        valid = dataset.read_masks(1) > 0
        wac_resolution = dataset.res
    wac = preprocess_image(wac_raw, config)
    wac_sift = extract_sift(wac.data, sift_config)
    wac_root = transform_sift(wac_sift)
    wac_display = prepare_sift_image(wac.data)
    report = {"moving": str(moving), "fixed": str(fixed),
              "source_resolution_m": 5.15, "wac_resolution_xy_m": wac_resolution,
              "tmc_input_shape": list(raw.shape), "tmc_input_dtype": str(raw.dtype),
              "wac_shape": list(wac_raw.shape), "wac_valid_pixels": int(valid.sum()),
              "preprocessing": asdict(config), "operations": tmc.operations,
              "sift": asdict(sift_config), "ransac": asdict(ransac_config),
              "ratio_threshold": .75, "matching": "RootSIFT L2 KNN k=2; TMC query, WAC train",
              "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                           "numpy": np.__version__, "rasterio": rasterio.__version__},
              "shared_runtime_seconds": {"tmc_preprocessing": tmc.runtime_seconds,
                  "wac_preprocessing": wac.runtime_seconds, "wac_sift": wac_sift.runtime_seconds,
                  "wac_rootsift": wac_root.runtime_seconds},
              "limitations": ["No ground truth: residuals measure fitted-model consistency only.",
                  "Approximate footprint and nominal GSD do not correct projection or viewing geometry.",
                  "WAC nodata remains black; features near nodata boundaries may be spurious.",
                  "Three-point affine support can fit by construction; a returned model alone is not success."],
              "scales": []}
    for resolution in (75., 100., 125., 150.):
        tick = perf_counter()
        folder = output / f"{int(resolution)}m"
        folder.mkdir()
        scale = normalize_resolution(tmc.data, 5.15, resolution)
        resize_runtime = perf_counter() - tick
        features = extract_sift(scale.data, sift_config)
        roots = transform_sift(features)
        knn = match_descriptors(roots.descriptors, wac_root.descriptors)
        ratio = filter_ratio(knn)
        ransac_start = perf_counter()
        geometry = None
        failure = None
        try:
            geometry = verify_affine(ratio, roots.points, wac_root.points, ransac_config)
        except (ValueError, RuntimeError) as exc:
            failure = str(exc)
        ransac_runtime = perf_counter() - ransac_start
        record = {"target_resolution_m": resolution, "resize_factor": scale.nominal_factor,
                  "actual_factors_xy": scale.actual_factors_xy,
                  "effective_resolution_xy_m": scale.effective_resolution_xy,
                  "shape": list(scale.data.shape), "tmc_keypoints": len(features.keypoints),
                  "wac_keypoints": len(wac_sift.keypoints),
                  "rootsift_shape": list(roots.descriptors.shape), "descriptor_dtype": str(roots.descriptors.dtype),
                  "wac_rootsift_shape": list(wac_root.descriptors.shape),
                  "knn_query_groups": len(knn.neighbors),
                  "knn_two_neighbor_groups": sum(len(g) == 2 for g in knn.neighbors),
                  "knn_candidate_count": sum(map(len, knn.neighbors)),
                  "ratio_matches": len(ratio.accepted),
                  "ransac_inliers": geometry.inlier_count if geometry else 0,
                  "inlier_ratio": geometry.inlier_ratio if geometry else None,
                  "inlier_residual_px": geometry.inlier_error_statistics if geometry else None,
                  "inlier_rmse_px": float(np.sqrt(np.mean(geometry.errors[geometry.inlier_mask] ** 2))) if geometry else None,
                  "transform_scaled_tmc_to_wac": geometry.matrix.tolist() if geometry else None,
                  "ransac_failure": failure,
                  "runtime_seconds": {"downsampling": resize_runtime, "sift": features.runtime_seconds,
                      "rootsift": roots.runtime_seconds, "matching": knn.runtime_seconds,
                      "ratio": ratio.runtime_seconds, "ransac": ransac_runtime,
                      "scale_pipeline_excluding_shared_preprocessing": perf_counter()-tick}}
        # Save coordinates and decisions so every correspondence remains auditable.
        record["accepted_matches"] = [asdict(m) | {"source_xy": roots.points[m.query_index].tolist(),
            "destination_xy": wac_root.points[m.train_index].tolist(),
            "inlier": bool(geometry.inlier_mask[i]) if geometry else False}
            for i, m in enumerate(ratio.accepted)]
        display = prepare_sift_image(scale.data)
        save_png(folder / "tmc_scaled.png", display)
        save_png(folder / "wac.png", wac_display)
        draw_matches(folder / "ratio_matches.png", display, wac_display, features, wac_sift, ratio.accepted)
        draw_matches(folder / "ransac_inliers.png", display, wac_display, features, wac_sift,
                     geometry.inliers if geometry else ())
        record["visualization_directory"] = str(folder)
        report["scales"].append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "accepted_matches"}), flush=True)
    maximum = max(r["ransac_inliers"] for r in report["scales"])
    report["best_scales_by_inlier_count_m"] = [r["target_resolution_m"] for r in report["scales"]
                                               if r["ransac_inliers"] == maximum] if maximum else []
    report["total_runtime_seconds_including_io"] = perf_counter()-started
    if any((p.stat().st_size, p.stat().st_mtime_ns) != source_stats[str(p)] for p in (moving, fixed)):
        raise RuntimeError("Source file size or modification time changed")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def _multiscale_records(records: tuple, config: RANSACConfig) -> tuple[list, dict]:
    """Verify in scaled coordinates; compose models into one original-source frame."""
    from src.evaluation.metrics import residual_statistics
    rows, models = [], []
    max_roundtrip = 0.
    for record in records:
        level = record.level
        matches = record.valid_matches
        tick = perf_counter()
        geometry, failure = None, None
        if len(matches.accepted) < 3:
            failure = 'insufficient support'
        else:
            try:
                geometry = verify_affine(matches, record.source_features.points,
                                         record.reference_features.points, config)
            except (ValueError, RuntimeError) as exc:
                failure = str(exc)
        verification_seconds = perf_counter()-tick
        original_candidates, reference_candidates = record.source, record.destination
        matrix_original = None
        if geometry is not None:
            matrix_original = (np.vstack((geometry.matrix, [0, 0, 1])) @ level.forward_matrix)[:2]
            models.append((record.level_index, matrix_original))
        h, w = level.original_shape
        probes = np.array([[0., 0.], [w-1, h-1], [(w-1)/2, (h-1)/2]])
        recovered = level.scaled_to_original(level.original_to_scaled(probes), pixel_footprint=True)
        error = float(np.max(np.abs(recovered-probes)))
        max_roundtrip = max(max_roundtrip, error)
        rows.append(dict(level_index=record.level_index, **level.metadata(),
            source_keypoints=len(record.source_features.keypoints), reference_keypoints=len(record.reference_features.keypoints),
            raw_knn_candidates=sum(map(len, record.knn.neighbors)), ratio_matches=len(record.ratio.accepted),
            filtered_matches=len(matches.accepted), mask_rejected=len(record.ratio.accepted)-len(matches.accepted),
            accepted_ratio_indices=record.accepted_indices.tolist(),
            candidates=[dict(query_index=m.query_index, train_index=m.train_index, ratio=m.ratio,
                descriptor_distance=m.nearest_distance, source_original=original_candidates[i].tolist(),
                source_scaled=record.source_features.points[m.query_index].tolist(),
                reference=reference_candidates[i].tolist()) for i, m in enumerate(matches.accepted)],
            verified_inliers=geometry.inlier_count if geometry else 0,
            inlier_ratio=geometry.inlier_ratio if geometry else None,
            residuals_reference_pixels=residual_statistics(geometry.errors[geometry.inlier_mask]) if geometry else None,
            transform_scaled_to_reference=geometry.matrix.tolist() if geometry else None,
            transform_original_to_reference=matrix_original.tolist() if geometry else None,
            inlier_mask=geometry.inlier_mask.tolist() if geometry else None,
            verification_status='model estimated; correctness unproven' if geometry else failure,
            mapping_roundtrip_max_error_pixels=error,
            runtime_seconds=dict(source_feature_matching=record.runtime_seconds, verification=verification_seconds)))
    comparisons = []
    for i, (first, a) in enumerate(models):
        for second, b in models[i+1:]:
            h, w = records[0].level.original_shape
            points = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1], [(w-1)/2, (h-1)/2, 1]])
            distances = np.linalg.norm(points@(a-b).T, axis=1)
            comparisons.append(dict(levels=[first, second], corner_center_disagreement_reference_px=distances.tolist(),
                                    mean=float(distances.mean()), maximum=float(distances.max())))
    return rows, dict(valid_model_count=len(models), comparisons=comparisons,
                     note='Estimator disagreement, not ground-truth error; fewer than two models precludes comparison.',
                     maximum_coordinate_roundtrip_error_pixels=max_roundtrip)


def _multiscale_figures(output: Path, records: tuple, reference: np.ndarray, groups: dict) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.visualization import plot_image, plot_matches
    files = []
    def save(fig, name):
        fig.tight_layout(rect=(0, .05, 1, .96))
        fig.text(.02, .01, 'M15 scale diagnostics | Display normalization only | Feature support is not geographic accuracy', fontsize=8)
        fig.savefig(output/name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        files.append(name)
    fig, axes = plt.subplots(1, len(records), figsize=(13, 6))
    for ax, record in zip(axes, records):
        plot_image(ax, record.level.data, title=f'{record.level.target_resolution:g} m/px\n{record.level.data.shape}')
    save(fig, 'scale_levels.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(r.level.target_resolution) for r in records], [len(r.source_features.keypoints) for r in records])
    ax.set(xlabel='Requested TMC GSD (m/pixel)', ylabel='SIFT keypoint count', title='All detected source keypoints by scale')
    save(fig, 'keypoints_by_scale.png')
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, record in zip(axes.ravel(), records):
        matches = record.valid_matches.accepted
        source = np.array([record.source_features.points[m.query_index] for m in matches]).reshape(-1, 2)
        plot_matches(ax, record.level.data, reference, source, record.destination,
                     max_matches=None, title=f'{record.level.target_resolution:g} m/px | {len(matches)} ratio matches; all shown')
    save(fig, 'matches_by_scale.png')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, rows) in zip(axes, groups.items()):
        labels = [str(r['target_gsd']) for r in rows]
        x = np.arange(len(rows))
        ax.bar(x-.18, [r['filtered_matches'] for r in rows], .36, label='Candidates')
        ax.bar(x+.18, [r['verified_inliers'] for r in rows], .36, label='Affine support')
        ax.set(xticks=x, xticklabels=labels, xlabel='Requested GSD', title=name)
        ax.legend(fontsize=8)
    save(fig, 'multiscale_summary.png')
    return files


def run_milestone15(output: Path) -> None:
    """Audit and extend the earlier experiment using reusable scale/match APIs.

    Run: python -m scripts.real_tmc_wac --milestone15
    Prior default experiment entry point remains unchanged.
    """
    from src.preprocessing.scale_normalization import build_scale_pyramid
    from src.matching.correspondence import match_scale_pyramid
    from scripts.tycho_phase_congruency import digest
    from scripts.tycho_visualization import load_artifacts
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    output = (root/output).resolve()
    if not output.is_relative_to(root/'results') or output.exists():
        raise ValueError('Use a new output directory under results/')
    baseline_path = root/'results/experiment_tycho_real_multiscale/report.json'
    baseline = json.loads(baseline_path.read_text())
    # The early PC report recorded hashes of this baseline and its scientific inputs.
    historical = json.loads((root/'results/experiment_tycho_phase_congruency/report.json').read_text())
    for path, expected in historical['sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Historical artifact changed: {path}')
    official = load_artifacts(root)  # validates M8/M9 masks, matrix and artifact hashes
    plan = (root/'PLAN.md').read_text(encoding='utf-8').split('# MILESTONE 15\n')[1].split('# MILESTONE 16')[0].strip()
    audit = [dict(requirement=name, initial_status=status, evidence=evidence, final_status='satisfied')
        for name, status, evidence in [
            ('image pyramids', 'partially satisfied', 'Independent resampling loop existed; reusable pyramid and rounded-grid deduplication missing.'),
            ('scale metadata', 'partially satisfied', 'Actual factors/effective GSD existed; original grid and explicit mapping metadata missing.'),
            ('coordinate conversion', 'partially satisfied', 'Pixel-center formula documented; validated reusable forward/inverse functions missing.'),
            ('multi-scale feature extraction', 'partially satisfied', 'SIFT/RootSIFT per-scale experiment existed; reusable candidate recovery API missing.'),
            ('experiments', 'satisfied', 'd20a018 saved all four negative pre-map-projection outcomes; M15 adds controlled synthetic and same-grid diagnostics.')]]
    config, sift, geometry_config = PreprocessingConfig(**baseline['preprocessing']), SIFTConfig(**baseline['sift']), RANSACConfig(**baseline['ransac'])
    raw = np.load(baseline['moving'], mmap_mode='r', allow_pickle=False)
    moving = preprocess_image(raw, config).data
    with rasterio.open(baseline['fixed']) as dataset:
        reference = preprocess_image(dataset.read(1), config).data
    pyramid = build_scale_pyramid(moving, baseline['source_resolution_m'], [75., 100., 125., 150.])
    records = match_scale_pyramid(pyramid, reference, sift, baseline['ratio_threshold'])
    before, before_consistency = _multiscale_records(records, geometry_config)
    for row, old, record in zip(before, baseline['scales'], records):
        for current, previous in [('source_keypoints', 'tmc_keypoints'), ('reference_keypoints', 'wac_keypoints'),
                                  ('filtered_matches', 'ratio_matches'), ('verified_inliers', 'ransac_inliers')]:
            if row[current] != old[previous]:
                raise ValueError(f'Early result not reproduced: {current} at {row["target_gsd"]}')
        for match, previous in zip(record.valid_matches.accepted, old['accepted_matches']):
            if (match.query_index, match.train_index) != (previous['query_index'], previous['train_index']):
                raise ValueError('Saved match indices differ')
            np.testing.assert_array_equal(record.source_features.points[match.query_index], previous['source_xy'])
            np.testing.assert_array_equal(record.reference_features.points[match.train_index], previous['destination_xy'])
        row['saved_baseline_reproduced'] = True

    # Current diagnostic: valid-only normalization of scientific DN copies.
    # The old unmasked result above intentionally retains historical preprocessing.
    def valid_processing(image, mask):
        values = image[mask].astype(np.float32)
        values = values / max(float(np.abs(values).max()), 1.)
        processed = preprocess_image(values.reshape(-1, 1), config).data.ravel()
        result = np.zeros(image.shape, np.float32)
        result[mask] = processed
        return result
    projected = valid_processing(official['moving'], official['moving_mask'])
    fixed = valid_processing(official['reference'], official['reference_mask'])
    identity_pyramid = build_scale_pyramid(projected, 100., [100.])
    np.testing.assert_array_equal(projected, identity_pyramid.levels[0].data)
    projected_records = match_scale_pyramid(identity_pyramid, fixed, sift, .75,
        official['moving_mask'], official['reference_mask'])
    projected_rows, projected_consistency = _multiscale_records(projected_records, geometry_config)

    # Deterministic synthetic source with independently generated coarse observation.
    synthetic = np.zeros((384, 384), np.float32)
    rng = np.random.default_rng(15)
    for _ in range(180):
        x, y = rng.integers(12, 372, 2)
        cv2.circle(synthetic, (int(x), int(y)), int(rng.integers(3, 12)), float(rng.uniform(.2, 1)), -1)
    coarse = cv2.resize(synthetic, (192, 192), interpolation=cv2.INTER_AREA)
    synthetic_pyramid = build_scale_pyramid(synthetic, 1., [1., 2., 3.])
    synthetic_records = match_scale_pyramid(synthetic_pyramid, coarse, sift, .75)
    synthetic_rows, synthetic_consistency = _multiscale_records(synthetic_records, geometry_config)
    truth = np.array([[.5, 0, -.25], [0, .5, -.25]])
    for row, record in zip(synthetic_rows, synthetic_records):
        errors = np.linalg.norm(record.source@truth[:, :2].T+truth[:, 2]-record.destination, axis=1)
        row['known_synthetic_mapping_candidate_rmse_px'] = float(np.sqrt(np.mean(errors**2))) if len(errors) else None
    if synthetic_rows[1]['verified_inliers'] < 10:
        raise RuntimeError('Controlled correct-scale test has insufficient support')
    output.mkdir(parents=True, exist_ok=False)
    figures = _multiscale_figures(output, records, reference,
        {'Pre-map-projection': before, 'Map-projected, same grid': projected_rows, 'Synthetic, reference GSD=2': synthetic_rows})
    limitations = [
        'Resolution normalization does not solve geometric misregistration.',
        'A scale performing well on Tycho is not universally optimal.',
        'Downsampling loses high-frequency information.',
        'Upsampling is rejected; it would not create new spatial information.',
        'Scale normalization does not solve Sun-angle shadow changes.',
        'Scale normalization alone does not solve cross-sensor spectral differences.',
        'Pre-map-projection failure remains evidence that geometry was a dominant bottleneck, not proof geometry was the only issue.',
        'TMC uses approximate footprint-based georeferencing, not rigorous camera/DEM orthorectification.',
        'Feature residuals do not establish absolute geographic accuracy.',
        'Scalar nominal GSD assumes square source pixels; rounded effective x/y GSD is reported separately.',
        'Historical pre-map-projection WAC nodata is black to reproduce the early baseline; do not interpret boundary features as terrain.',
        'Map-projected diagnostic excludes invalid match endpoints but descriptors may still be affected by zero-filled mask boundaries.',
        'Synthetic reference is derived by known resampling of the same image, not an independent sensor observation.',
        'Per-level outputs are separate; match counts across levels are not independent or additive support.']
    # Recheck protected official artifacts and historical input content after work.
    for path, expected in historical['sha256'].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f'Historical input modified: {path}')
    for item in official['report']['provenance']['inputs'].values():
        if digest(root/item['path']) != item['sha256']:
            raise RuntimeError('Official artifact modified')
    report = dict(plan_m15_exact_section='# MILESTONE 15\n'+plan, audit=audit,
        source_reference_gsd_m=[5.15, 100.], nominal_native_ratio=100/5.15,
        requested_gsd=pyramid.requested_resolutions, request_to_level=pyramid.request_to_level,
        preprocessing=asdict(config), sift=asdict(sift), ransac=asdict(geometry_config), ratio_threshold=.75,
        verification_policy='Existing affine RANSAC in scaled source -> original reference coordinates; threshold 3 reference pixels at every level. Models composed to original-source frame for comparison.',
        timing_policy='Per-level source feature/matching and verification timings exclude pyramid resampling and shared reference extraction; total runner time includes these, validation and figure writing.',
        candidate_policy='Independent levels; no merge, cross-level deduplication, score or best-scale selection.',
        pre_map_projection=dict(levels=before, consistency=before_consistency, baseline_path=str(baseline_path)),
        map_projected=dict(levels=projected_rows, consistency=projected_consistency, identity_pixels_exact=True,
            preprocessing='Scientific DN normalized on valid pixels only; scaled by max magnitude then existing preprocessing, invalid fill zero. Separate diagnostic, not reproduction of M8 LoFTR.'),
        synthetic=dict(levels=synthetic_rows, consistency=synthetic_consistency,
            source_gsd=1., reference_gsd=2., known_transform=truth.tolist(), seed=15),
        provenance=dict(historical_sha256=historical['sha256'], official_inputs=official['report']['provenance']['inputs']),
        versions=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__),
        limitations=limitations, figures=figures, runtime_seconds=perf_counter()-started)
    (output/'multiscale.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['M15 complete: pyramid, metadata, coordinate conversion, reusable SIFT/RootSIFT matching, experiments.',
        'Native TMC/WAC ratio 19.42; TMC target GSDs 75/100/125/150 m. No upsampling.',
        'Pixel centers map with actual rounded x/y dimensions; (p+.5)*factor-.5, invertible with explicit footprint boundary domain.',
        'Pre-map-projection baseline reproduced exactly: keypoints 1310/718/479/325; matches 2/2/0/0; zero inliers.',
        f'Map-projected 100 m identity diagnostic: {projected_rows[0]["filtered_matches"]} candidates; {projected_rows[0]["verified_inliers"]} affine support.',
        'Synthetic GSD 1/2/3: '+str([(r['filtered_matches'], r['verified_inliers']) for r in synthetic_rows])+' (candidates, inliers).',
        f'Maximum real coordinate round-trip error: {before_consistency["maximum_coordinate_roundtrip_error_pixels"]:.3g} px.',
        'No earlier results, M8 model or registered image replaced; no LoFTR inference.', *limitations,
        f'Runner runtime: {report["runtime_seconds"]:.3f} seconds.']
    (output/'multiscale_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for name in figures+['multiscale.json', 'multiscale_summary.txt']:
        if (output/name).stat().st_size == 0:
            raise RuntimeError(f'Empty output: {name}')
    print('\n'.join(lines[:7]))
    print(lines[-1])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone15', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/milestone_15_multiscale'))
    args = parser.parse_args()
    if args.milestone15:
        run_milestone15(args.output)
    else:
        main()
