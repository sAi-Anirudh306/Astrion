"""Milestone 11: spatial selection from saved verified matches, fixed M8 affine.

Run: .venv/Scripts/python.exe -m scripts.tycho_spatial_distribution
Repeat runs require --output results/<new-directory>. No inference/refit/warp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.tycho_visualization import load_artifacts
from src.evaluation.metrics import residual_statistics, reprojection_residuals
from src.evaluation.visualization import plot_spatial_distribution
from src.spatial.spatial_distribution import (assign_grid_cells, distribution_statistics,
    spatial_spread, select_spatially_balanced_matches)


def describe_subset(source: np.ndarray, destination: np.ndarray, matrix: np.ndarray,
                    shape: tuple[int, int], original_count: int) -> dict:
    """Calculate distribution and fixed-model residuals for precisely this subset."""
    return dict(selected_count=len(source), retained_percentage=100*len(source)/original_count,
                distribution=distribution_statistics(destination, shape),
                residuals_wac_pixels=residual_statistics(reprojection_residuals(source, destination, matrix)),
                spread=spatial_spread(destination, shape))


def write_plots(output: Path, reference: np.ndarray, valid: np.ndarray,
                subsets: list[tuple[str, np.ndarray, dict]]) -> None:
    """Use the established M10 grid renderer with all selected points, no subsampling."""
    for name, points, record in subsets:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_spatial_distribution(ax, reference, points, valid)
        fig.suptitle(f"{name.replace('_', ' ')} | {len(points)} verified correspondences", fontsize=13)
        fig.text(.08, .04, "All subset points shown unchanged. Fixed affine; no registration refit.", fontsize=9)
        fig.subplots_adjust(top=.85, bottom=.12)
        fig.savefig(output/f"{name}.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for ax, (name, points, record) in zip(axes.ravel(), subsets):
        plot_spatial_distribution(ax, reference, points, valid)
        ax.set_title(f"{name.replace('_', ' ')} | n={len(points)}\n"
                     f"Cells {record['distribution']['occupied_cells']}/16 | entropy {record['distribution']['normalized_entropy']:.6f}\n"
                     f"Fixed-model RMSE {record['residuals_wac_pixels']['rmse']:.6f} px", fontsize=10, loc="left")
    fig.suptitle("ASTRION | Deterministic spatial balancing of verified Tycho correspondences", fontsize=15)
    fig.text(.07, .025, "Residual-first selection favors the existing affine fit. Lower subset residuals do not demonstrate improved registration.", fontsize=10)
    fig.subplots_adjust(top=.89, bottom=.09, hspace=.40, wspace=.22)
    fig.savefig(output/"distribution_comparison.png", dpi=160)
    plt.close(fig)


def summary(report: dict) -> str:
    """Readable comparison with full cell counts and explicit interpretation limits."""
    lines = ["ASTRION Milestone 11: spatial distribution control", report['selection_ordering'],
             "Affine transform remains fixed at the Milestone 8 result."]
    for name, record in [("Original", report['original'])]+[(f"Cap {r['max_per_cell']}/cell", r) for r in report['configurations']]:
        d, residual, spread = record['distribution'], record['residuals_wac_pixels'], record['spread']
        lines.append(f"{name}: n={record['selected_count']} ({record['retained_percentage']:.2f}% retained); "
                     f"occupancy={d['occupied_cells']}/{d['total_cells']} ({d['occupancy_percentage']:.2f}%); "
                     f"entropy={d['normalized_entropy']:.6f}; RMSE={residual['rmse']:.6f} px; "
                     f"bbox width/height={spread['width_fraction']:.6f}/{spread['height_fraction']:.6f}")
        lines.append("  Cell counts: "+str(d['counts']))
    lines += ["Original occupied-cell counts range from "
              f"{report['original']['distribution']['minimum_occupied_count']} to {report['original']['distribution']['maximum_cell_count']}.",
              report['recommendation']['reason'], *report['limitations'],
              f"Runner runtime: {report['runtime_seconds']:.3f} s"]
    return '\n'.join(lines)+'\n'


def run(output: Path) -> dict:
    """Verify saved provenance, select caps 5/10/20, then write a fresh experiment."""
    tick = perf_counter()
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(root/'results'):
        raise ValueError("Experiment output must be under project results/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    data = load_artifacts(root)
    saved = data['report']
    candidate_indices = np.flatnonzero(data['keep'])
    source, destination = data['source'][data['keep']], data['destination'][data['keep']]
    confidence = data['confidence'][data['keep']]
    matrix, shape = data['matrix'], data['reference'].shape
    residuals = reprojection_residuals(source, destination, matrix)
    original = describe_subset(source, destination, matrix, shape, len(source))
    if original['distribution'] != saved['spatial_distribution']:
        raise ValueError("Baseline spatial distribution disagrees with Milestone 9")
    if not np.isclose(original['residuals_wac_pixels']['rmse'], saved['reprojection_residuals']['wac_pixels']['rmse'], rtol=0, atol=1e-9):
        raise ValueError("Baseline residuals disagree with Milestone 9")
    original['original_candidate_indices'] = candidate_indices.tolist()
    assignment = assign_grid_cells(destination, shape)
    original['cell_indices'] = assignment.cells.tolist()
    configurations, subsets = [], [('original_distribution', destination, original)]
    for cap in (5, 10, 20):
        selected = select_spatially_balanced_matches(source, destination, shape, max_per_cell=cap,
            confidence=confidence, residuals=residuals, source_shape=data['moving'].shape)
        record = describe_subset(selected.source, selected.destination, matrix, shape, len(source))
        record.update(max_per_cell=cap, global_maximum=None,
            selected_input_inlier_indices=selected.original_indices.tolist(),
            selected_original_candidate_indices=candidate_indices[selected.original_indices].tolist(),
            cell_indices=selected.cell_indices.tolist(), cell_rows=selected.cell_rows.tolist(),
            cell_columns=selected.cell_columns.tolist(), confidence=selected.confidence.tolist(),
            fixed_transform_residuals=selected.residuals.tolist())
        configurations.append(record)
        subsets.append((f'balanced_{cap}_per_cell', selected.destination, record))
    report = dict(input_correspondence_count=len(source), grid_shape=[4, 4], reference_shape=list(shape),
        original=original, configurations=configurations,
        selection_ordering="Per cell: lower fixed-affine residual, then higher saved LoFTR confidence, then original input index.",
        available_scores=["saved LoFTR confidence", "residuals recomputed with the unchanged saved M8 affine"],
        global_allocation="Round-robin in ascending row-major occupied cell ID, skipping exhausted queues. Partial final round favors earlier cell IDs. No global cap used in Tycho experiments.",
        output_order="Original input order. Input indices address the 569-inlier array; original candidate indices address the saved 677-pair archive.",
        recommendation=dict(max_per_cell=None, reason="No unique best cap is established: 5 retains fewer correspondences with more even cell counts, while 10 and 20 retain more support within populated cells. Report all predeclared caps; choosing a downstream operating point requires an explicit match budget or independent validation. No composite score or transform optimization was used."),
        limitations=[
            "Grid-based balancing improves correspondence distribution control but does not prove geometric correctness.",
            "Selection operates on already RANSAC-verified correspondences and does not replace geometric verification.",
            "Residual-based ranking uses the existing fitted affine model and can favor points already consistent with that model. Lower selected-set residuals are not evidence of improved registration.",
            "Spatial entropy measures distribution among grid cells and is not a complete measure of geometric observability or proof of uniform spatial coverage.",
            "The TMC map projection remains footprint-based approximate georeferencing rather than rigorous camera/DEM orthorectification.",
            "Bounding-box fractions measure correspondence spatial footprint, not registered image coverage; no new registered coverage is calculated.",
            "Feature reprojection residuals are not independently verified absolute lunar geographic accuracy; no external ground-control truth is available."],
        provenance=dict(evaluation_path=data['evaluation_path'], evaluation_sha256=data['evaluation_sha256'],
                        inputs=saved['provenance']['inputs'], affine_matrix=matrix.tolist(),
                        transform_refitted=False, loftr_rerun=False, registration_rerun=False),
        software=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__))
    output.mkdir(parents=True, exist_ok=False)
    write_plots(output, data['reference'], data['reference_mask'], subsets)
    for record in saved['provenance']['inputs'].values():
        if hashlib.sha256((root/record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError("Input artifact changed during spatial selection")
    if hashlib.sha256((root/data['evaluation_path']).read_bytes()).hexdigest() != data['evaluation_sha256']:
        raise RuntimeError("Milestone 9 evaluation changed during spatial selection")
    report['runtime_seconds'] = perf_counter()-tick
    (output/'spatial_distribution.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'spatial_distribution_summary.txt').write_text(summary(report), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results/milestone_11_spatial_distribution')
    print(summary(run(parser.parse_args().output)))


if __name__ == '__main__':
    main()
