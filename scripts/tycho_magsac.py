"""M13: saved full-candidate affine estimator comparison; no inference or refitting.

Run with .venv/Scripts/python.exe -m scripts.tycho_magsac.
Use --output for a new directory on repeat runs; existing results are preserved.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.tycho_visualization import load_artifacts
from src.geometry.magsac import MAGSACConfig, affine_backend_capability, verify_affine
from src.geometry.registration import transform_points, warp_affine
from src.evaluation.metrics import (affine_diagnostics, correspondence_statistics,
    image_similarity, reprojection_residuals, residual_statistics, spatial_distribution)
from src.evaluation.visualization import plot_image, plot_matches, plot_spatial_distribution


LIMITATIONS = [
    "Inputs are image-derived LoFTR correspondences, not external ground-control truth.",
    "Lower residual RMSE does not prove higher absolute lunar geographic accuracy.",
    "Threshold changes alter support definitions; every tested threshold is reported.",
    "Estimator agreement is not ground truth.",
    "TMC uses approximate footprint-based georeferencing, not rigorous camera/DEM orthorectification.",
    "One Tycho scene cannot establish universal estimator superiority.",
    "Raw estimator-returned models are compared: saved RANSAC includes 10 LM refinement iterations; "
    "USAC includes sigma-consensus optimization and backend polishing. Neither receives external "
    "refinement. Differences include the estimators' internal optimization, not only scoring.",
    "USAC threshold controls support/termination; internal maximum sigma is not independently exposed.",
    "Pearson is a cross-image similarity diagnostic on independently normalized common-valid pixels."]


def evaluate(data: dict, matrix: np.ndarray, keep: np.ndarray, threshold: float) -> dict:
    errors = reprojection_residuals(data['source'], data['destination'], matrix)
    return dict(threshold_pixels=threshold, matrix=matrix.tolist(),
        correspondence=correspondence_statistics(keep), inlier_indices=np.flatnonzero(keep).tolist(),
        residuals_pixels=residual_statistics(errors[keep]), all_residuals_pixels=errors.tolist(),
        spatial=spatial_distribution(data['destination'][keep], data['reference'].shape),
        affine=affine_diagnostics(matrix))


def disagreement(baseline: np.ndarray, matrix: np.ndarray, shape: tuple) -> dict:
    h, w = shape
    points = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1], [(w-1)/2, (h-1)/2]])
    delta = matrix-baseline
    distances = np.linalg.norm(transform_points(points, matrix)-transform_points(points, baseline), axis=1)
    return dict(matrix_difference=delta.tolist(), linear_frobenius=float(np.linalg.norm(delta[:, :2])),
        translation_difference_pixels=float(np.linalg.norm(delta[:, 2])), source_locations=points.tolist(),
        location_labels=['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'],
        displacement_pixels=distances.tolist(), mean_pixels=float(distances.mean()),
        maximum_pixels=float(distances.max()))


def write_figures(output: Path, data: dict, primary, baseline: dict, records: list, warped) -> list[str]:
    paths = []
    def save(fig, name):
        fig.text(.02, .015, 'Image-derived features; no external geographic truth | Display-only P1/P99 normalization', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .96))
        fig.savefig(output/name, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths.append(name)
    for name, mask in [('ransac', data['keep']), ('magsac', primary.inlier_mask)]:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_matches(ax, data['moving'], data['reference'], data['source'][mask], data['destination'][mask],
            data['moving_mask'], data['reference_mask'], max_matches=150,
            title=f'{name.upper()} | {mask.sum()}/677 inliers; at most 150 deterministic display samples')
        save(fig, name+'_inliers.png')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows = [baseline]+records
    labels = ['Saved RANSAC 3']+[f'MAGSAC {r["threshold_pixels"]:g}' for r in records]
    for ax, values, title in zip(axes,
        [[r['correspondence']['inlier_count'] for r in rows], [r['residuals_pixels']['rmse'] for r in rows],
         [r['spatial']['normalized_entropy'] for r in rows]],
        ['Inlier support', 'Inlier RMSE (px)', 'Equal-area 4x4 entropy']):
        ax.bar(labels, values); ax.set_title(title); ax.tick_params(axis='x', rotation=30)
    save(fig, 'estimator_comparison.png')
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_image(ax, data['reference'], data['reference_mask'], 'Support comparison | 3-pixel thresholds')
    r, m = data['keep'], primary.inlier_mask
    for mask, label, color in [(r&m, 'Both', '#16a085'), (r&~m, 'RANSAC only', '#e67e22'),
                               (~r&m, 'MAGSAC only', '#8e44ad'), (~r&~m, 'Neither', '#666666')]:
        points = data['destination'][mask]
        ax.scatter(points[:, 0], points[:, 1], s=12, c=color, label=f'{label}: {mask.sum()}')
    ax.legend(fontsize=8); save(fig, 'inlier_set_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, matrix, mask in [('RANSAC', data['matrix'], r), ('MAGSAC', primary.matrix, m)]:
        errors = reprojection_residuals(data['source'], data['destination'], matrix)
        axes[0].hist(errors[mask], bins=np.linspace(0, 3, 25), histtype='step', label=name)
        ordered = np.sort(errors)
        axes[1].plot(ordered, np.arange(1, len(ordered)+1)/len(ordered), label=name)
    axes[0].set(title='Own inlier residuals at 3 px', xlabel='Residual (px)', ylabel='Count')
    axes[1].set(title='All 677 candidates (ECDF)', xlabel='Residual (px)', ylabel='Fraction')
    for ax in axes: ax.legend()
    save(fig, 'residual_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, mask, name in zip(axes, [r, m], ['RANSAC', 'MAGSAC']):
        plot_spatial_distribution(ax, data['reference'], data['destination'][mask], data['reference_mask'])
        ax.set_title(name+' | '+ax.get_title(loc='left'), loc='left', fontsize=10)
    save(fig, 'spatial_support_comparison.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, image, mask, name in zip(axes, [data['reference'], data['registered'], warped.image],
            [data['reference_mask'], data['registered_mask'], warped.valid_mask],
            ['WAC reference', 'Official M8 RANSAC', 'Experimental MAGSAC']):
        plot_image(ax, image, mask, name)
    save(fig, 'registration_comparison.png')
    return paths


def main() -> None:
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/milestone_13_magsac'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root/args.output).resolve()
    if not output.is_relative_to(root/'results'):
        raise ValueError('Output must be a new directory under results/')
    if output.exists():
        raise FileExistsError(f'Preserving existing results: {output}; use --output')
    capability = affine_backend_capability()
    if not capability['supported']:
        raise RuntimeError(f'No verified full affine MAGSAC backend: {capability}')
    data = load_artifacts(root)
    source, destination = data['source'], data['destination']
    baseline = evaluate(data, data['matrix'], data['keep'], 3.)
    if not np.array_equal(data['keep'], np.asarray(baseline['all_residuals_pixels']) <= 3):
        raise ValueError('Saved RANSAC support does not match its 3-pixel classification')
    if len(source) != 677 or int(data['keep'].sum()) != 569:
        raise ValueError('Saved full-candidate baseline differs from expected Tycho experiment')
    baseline['saved_ransac_configuration'] = data['report']['provenance']['ransac']
    records, fits = [], []
    for threshold in (2., 3., 4.):
        config = MAGSACConfig(reprojection_threshold=threshold)
        repeats = [verify_affine(source, destination, config) for _ in range(5)]
        fit = repeats[0]
        record = evaluate(data, fit.matrix, fit.inlier_mask, threshold)
        record.update(metadata=fit.metadata, backend_inlier_indices=np.flatnonzero(fit.backend_inlier_mask).tolist(),
            backend_inlier_count=int(fit.backend_inlier_mask.sum()),
            backend_vs_explicit_mask_disagreement=int(np.count_nonzero(fit.backend_inlier_mask != fit.inlier_mask)),
            transform_disagreement=disagreement(data['matrix'], fit.matrix, data['reference'].shape),
            reproducibility=dict(runs=5, identical_transform=all(np.array_equal(fit.matrix, f.matrix) for f in repeats),
                identical_final_mask=all(np.array_equal(fit.inlier_mask, f.inlier_mask) for f in repeats),
                identical_backend_mask=all(np.array_equal(fit.backend_inlier_mask, f.backend_inlier_mask) for f in repeats),
                maximum_matrix_difference=max(float(np.max(np.abs(f.matrix-fit.matrix))) for f in repeats),
                runtimes_seconds=[f.runtime_seconds for f in repeats]))
        records.append(record); fits.append(fit)
    primary = fits[1]
    r, m = data['keep'], primary.inlier_mask
    overlap = dict(common=int((r&m).sum()), ransac_only=int((r&~m).sum()), magsac_only=int((~r&m).sum()),
                   neither=int((~r&~m).sum()), jaccard=float((r&m).sum()/(r|m).sum()))
    warped = warp_affine(data['moving'], primary.matrix, data['reference'].shape, validity_mask=data['moving_mask'])
    shared = data['registered_mask'] & warped.valid_mask & data['reference_mask']
    similarity = {}
    for name, image, mask in [('ransac', data['registered'], data['registered_mask']),
                              ('magsac', warped.image, warped.valid_mask)]:
        similarity[name] = dict(own_overlap=image_similarity(image, data['reference'], mask, data['reference_mask']),
                                shared_overlap=image_similarity(image, data['reference'], shared, shared))
    output.mkdir(parents=True, exist_ok=False)
    figures = write_figures(output, data, primary, baseline, records, warped)
    report = dict(opencv_version=capability['opencv_version'], backend_capability=capability,
        model='affine_2d_six_parameter', input_candidate_count=len(source),
        provenance=dict(evaluation_path=data['evaluation_path'], evaluation_sha256=data['evaluation_sha256'],
                        inputs=data['report']['provenance']['inputs']),
        comparison_mode='Raw estimator-returned models; no external refinement',
        ransac_baseline=baseline, magsac_configurations=records, primary_threshold_pixels=3.,
        support_overlap=overlap, registration_similarity=similarity, limitations=LIMITATIONS,
        documentation='https://docs.opencv.org/4.13.0/de/d3e/tutorial_usac.html',
        figures=figures, display_subsampling='At most 150 inliers; evenly spaced original subset indices',
        runtime_seconds=perf_counter()-started)
    (output/'magsac.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [f"OpenCV {capability['opencv_version']}: true six-parameter affine UsacParams MAGSAC score + sigma LO.",
        'Raw backend outputs including internal refinement; no external refit. All 677 saved candidates.',
        f"Saved RANSAC: 569 inliers; RMSE {baseline['residuals_pixels']['rmse']:.6f} px."]
    for record in records:
        c, e, s = record['correspondence'], record['residuals_pixels'], record['spatial']
        lines.append(f"MAGSAC {record['threshold_pixels']:g} px: {c['inlier_count']} inliers "
                     f"({100*c['inlier_ratio']:.2f}%); RMSE {e['rmse']:.6f}; median {e['median']:.6f}; "
                     f"occupancy {s['occupied_cells']}/16; entropy {s['normalized_entropy']:.6f}.")
    delta = records[1]['transform_disagreement']
    lines += [f'Support overlap: {overlap}',
              f"Primary transform disagreement: linear Frobenius {delta['linear_frobenius']:.6f}; "
              f"translation {delta['translation_difference_pixels']:.6f} px; corners/center "
              f"mean {delta['mean_pixels']:.6f}, maximum {delta['maximum_pixels']:.6f} px.",
              'Pearson RANSAC / MAGSAC: '+ ' / '.join(f"{similarity[k]['own_overlap']['correlation']:.6f}" for k in ('ransac', 'magsac')),
              'Pearson on identical shared coverage: '+ ' / '.join(f"{similarity[k]['shared_overlap']['correlation']:.6f}" for k in ('ransac', 'magsac')),
              'Five repeats per threshold: identical matrices and masks = '+str(all(
                  rec['reproducibility']['identical_transform'] and rec['reproducibility']['identical_final_mask']
                  and rec['reproducibility']['identical_backend_mask'] for rec in records))+'.',
              *LIMITATIONS, f"Runner runtime: {report['runtime_seconds']:.3f} seconds"]
    (output/'magsac_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for name in figures+['magsac.json', 'magsac_summary.txt']:
        if (output/name).stat().st_size == 0:
            raise RuntimeError(f'Empty output: {name}')
    print('\n'.join(lines[:6]))
    print(f"Outputs: {output}; runtime {report['runtime_seconds']:.3f}s")


if __name__ == '__main__':
    main()
