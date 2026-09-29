"""M12 saved-inlier quadtree experiment: no inference, affine fitting or warping.

Run: .venv/Scripts/python.exe -m scripts.tycho_quadtree
Use --output results/<fresh-directory> for repeat runs. Existing results are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.tycho_visualization import load_artifacts
from scripts.tycho_spatial_distribution import describe_subset
from src.evaluation.metrics import reprojection_residuals
from src.evaluation.visualization import plot_quadtree, plot_spatial_distribution
from src.spatial.quadtree import build_quadtree, select_quadtree_matches, tree_statistics


def load_m11(root: Path, data: dict) -> tuple[dict, str]:
    """Require the same saved input lineage and verify M11 subsets without reselecting."""
    path = root/'results/milestone_11_spatial_distribution/spatial_distribution.json'
    if not path.is_file():
        raise FileNotFoundError(f'Required M11 comparison report missing: {path}')
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report['provenance']['evaluation_sha256'] != data['evaluation_sha256'] or
            report['provenance']['inputs'] != data['report']['provenance']['inputs'] or
            not np.array_equal(report['provenance']['affine_matrix'], data['matrix'])):
        raise ValueError('M11 report does not share the saved evaluation inputs/transform')
    inliers = np.flatnonzero(data['keep'])
    if report['input_correspondence_count'] != len(inliers):
        raise ValueError('M11 original count disagrees with saved inliers')
    original = describe_subset(data['source'][inliers], data['destination'][inliers],
                               data['matrix'], data['reference'].shape, len(inliers))
    if any(original[key] != report['original'][key] for key in original):
        raise ValueError('M11 original metrics disagree with saved inliers')
    if sorted(row['max_per_cell'] for row in report['configurations']) != [5, 10, 20]:
        raise ValueError('Expected saved M11 caps 5, 10 and 20')
    for row in report['configurations']:
        indices = np.asarray(row['selected_input_inlier_indices'])
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or
                (indices < 0).any() or (indices >= len(inliers)).any() or
                len(np.unique(indices)) != len(indices)):
            raise ValueError('M11 selected indices are invalid')
        candidates = inliers[indices]
        if not np.array_equal(candidates, row['selected_original_candidate_indices']):
            raise ValueError('M11 inlier/candidate index mappings disagree')
        actual = describe_subset(data['source'][candidates], data['destination'][candidates],
                                  data['matrix'], data['reference'].shape, len(inliers))
        if any(actual[key] != row[key] for key in actual):
            raise ValueError('M11 selected-subset metrics disagree with saved pairs')
    return report, hashlib.sha256(raw).hexdigest()


def comparison_row(name: str, record: dict) -> dict:
    """Same equal-area metrics for original, fixed-grid and adaptive subsets."""
    return dict(name=name, **{key: record[key] for key in
        ('selected_count', 'retained_percentage', 'distribution', 'residuals_wac_pixels', 'spread')})


def write_plots(output: Path, data: dict, experiments: list, comparisons: list) -> None:
    """Show partitions and subsets at unchanged coordinates, without display sampling."""
    reference, valid = data['reference'], data['reference_mask']
    for tree, selected, record in experiments:
        fig, ax = plt.subplots(figsize=(9, 8))
        plot_quadtree(ax, reference, tree, selected.original_indices, valid,
            f"Capacity {tree.capacity}; depth limit {tree.max_depth}; select 1/occupied leaf\n"
            f"{len(selected.source)} selected / {len(tree.destination)} verified inliers")
        fig.text(.08, .035, 'Subdivision capacity and selection cap are distinct. Residual-first selection; fixed affine.', fontsize=9)
        fig.subplots_adjust(top=.87, bottom=.12)
        fig.savefig(output/f'quadtree_capacity_{tree.capacity}.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (tree, selected, record) in zip(axes, experiments):
        plot_quadtree(ax, reference, tree, selected.original_indices, valid,
                      f'Capacity {tree.capacity}; 1 per occupied leaf\nSelected {len(selected.source)}')
    fig.suptitle('ASTRION | Adaptive leaf partitions and selected correspondences', fontsize=16)
    fig.text(.04, .03, 'All 569 inputs shown in gray; selected points in cyan. No coordinates changed. No affine refit.', fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.13, wspace=.32)
    fig.savefig(output/'quadtree_selected_matches.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    for ax, (row, points) in zip(axes.ravel(), comparisons):
        plot_spatial_distribution(ax, reference, points, valid)
        ax.set_title(f"{row['name']} | n={row['selected_count']}\n"
                     f"4x4 cells {row['distribution']['occupied_cells']}/16; H={row['distribution']['normalized_entropy']:.4f}\n"
                     f"Fixed-affine RMSE {row['residuals_wac_pixels']['rmse']:.4f} px", fontsize=10, loc='left')
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].text(.02, .8, 'Common evaluation: equal-area 4x4 grid\n\n'
        'M11: equal caps per fixed cell\nM12: one match per adaptive leaf\n\n'
        'Dense regions receive more leaves.\nNo composite ranking of methods.\n\n'
        'Residual-first ranking favors the fitted model.\nLower RMSE does not prove better registration.', fontsize=11, va='top')
    fig.suptitle('ASTRION | Original, fixed-grid and quadtree spatial selection', fontsize=16)
    fig.subplots_adjust(top=.86, bottom=.08, hspace=.4, wspace=.3)
    fig.savefig(output/'quadtree_comparison.png', dpi=160)
    plt.close(fig)


def summary(report: dict) -> str:
    """Concise tree and equal-grid comparison with scientific caveats."""
    lines = ['ASTRION Milestone 12: adaptive correspondence selection', report['selection_ordering']]
    for row in report['configurations']:
        stats = row['tree_statistics']
        lines.append(f"Capacity {row['tree_parameters']['capacity']}; select {row['max_selected_per_leaf']}/leaf: "
                     f"nodes={stats['total_node_count']}, leaves={stats['leaf_count']}, occupied={stats['occupied_leaf_count']}, "
                     f"depth range={stats['minimum_leaf_depth']}..{stats['maximum_reached_depth']}; "
                     f"occupied leaves by depth={stats['occupied_leaves_by_depth']}")
    for row in report['comparison']:
        d, s = row['distribution'], row['spread']
        lines.append(f"{row['name']}: n={row['selected_count']} ({row['retained_percentage']:.2f}%); "
            f"4x4 occupancy={d['occupied_cells']}/{d['total_cells']} ({d['occupancy_percentage']:.2f}%); "
            f"entropy={d['normalized_entropy']:.6f}; RMSE={row['residuals_wac_pixels']['rmse']:.6f} px; "
            f"bbox width/height={s['width_fraction']:.6f}/{s['height_fraction']:.6f}")
    lines += [report['interpretation'], *report['limitations'], f"Runner runtime: {report['runtime_seconds']:.3f} s"]
    return '\n'.join(lines)+'\n'


def run(output: Path) -> dict:
    """Evaluate exactly the predeclared capacities 5/10/20, depth 4, one per leaf."""
    tick = perf_counter()
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(root/'results'):
        raise ValueError('Output must be under project results/')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {output}')
    data = load_artifacts(root)
    m11, m11_hash = load_m11(root, data)
    candidate_indices = np.flatnonzero(data['keep'])
    source, destination = data['source'][data['keep']], data['destination'][data['keep']]
    confidence = data['confidence'][data['keep']]
    residuals = reprojection_residuals(source, destination, data['matrix'])
    experiments = []
    comparisons = [(comparison_row('Original', m11['original']), destination)]
    for row in m11['configurations']:
        comparisons.append((comparison_row(f"M11 cap {row['max_per_cell']}", row),
                            destination[row['selected_input_inlier_indices']]))
    for capacity in (5, 10, 20):
        tree = build_quadtree(destination, data['reference'].shape, capacity=capacity, max_depth=4)
        selected = select_quadtree_matches(tree, source, max_per_leaf=1, confidence=confidence,
                                           residuals=residuals, source_shape=data['moving'].shape)
        record = describe_subset(selected.source, selected.destination, data['matrix'], tree.image_shape, len(source))
        nodes = [dict(node_id=node.node_id, bounds=list(node.bounds), depth=node.depth,
                      original_input_indices=list(node.original_indices), is_leaf=node.is_leaf,
                      child_ids=[child.node_id for child in node.children]) for node in tree.nodes]
        leaves = [dict(leaf_id=leaf.node_id, bounds=list(leaf.bounds), depth=leaf.depth,
                       points_before_selection=len(leaf.original_indices),
                       selected_count=int(np.count_nonzero(selected.leaf_ids == leaf.node_id))) for leaf in tree.leaves]
        record.update(tree_parameters=dict(capacity=capacity, max_depth=4, min_cell_size=list(tree.min_cell_size)),
            max_selected_per_leaf=1, global_maximum=None, tree_statistics=tree_statistics(tree), nodes=nodes, leaves=leaves,
            selected_input_inlier_indices=selected.original_indices.tolist(),
            selected_original_candidate_indices=candidate_indices[selected.original_indices].tolist(),
            selected_leaf_ids=selected.leaf_ids.tolist(), confidence=selected.confidence.tolist(),
            fixed_transform_residuals=selected.residuals.tolist())
        experiments.append((tree, selected, record))
        comparisons.append((comparison_row(f'M12 capacity {capacity}', record), selected.destination))
    report = dict(input_count=len(source), image_dimensions=list(data['reference'].shape),
        image_dimensions_order='height,width', evaluation_grid=[4, 4], configurations=[r for _, _, r in experiments],
        comparison=[r for r, _ in comparisons],
        selection_ordering='Per leaf: lower fixed-affine residual, then higher saved LoFTR confidence, then original input index. Returned in original input order.',
        global_allocation='Round-robin by ascending depth-first occupied leaf ID; exhausted leaves skipped; partial round favors earlier IDs. No global cap used in primary run.',
        boundary_policy='Half-open [xmin,xmax) x [ymin,ymax); split midpoints go right/below; child order TL,TR,BL,BR; final image extent excluded; final integer pixel valid.',
        interpretation='Fixed-grid selection caps counts in equal-area regions. Quadtree selection allocates one point per occupied adaptive leaf, giving denser regions more leaves. Compare all predeclared settings on the same 4x4 grid; no method is declared superior and no composite score is used.',
        limitations=[
            'Quadtree selection controls spatial redundancy but does not establish correspondence correctness.',
            'Input correspondences were already RANSAC verified; selection does not replace geometric verification.',
            'Residual-first selection is biased toward the existing fitted affine model.',
            'Adaptive quadtree leaves have unequal areas, so leaf occupancy is not directly equivalent to equal-area 4x4 occupancy. No adaptive-leaf entropy is compared with fixed-grid entropy.',
            'Lower selected-subset RMSE does not prove improved registration; the M8 transform remains fixed.',
            'The TMC map projection remains approximate footprint-based georeferencing rather than rigorous camera/DEM orthorectification.',
            'Feature reprojection residuals are not independently verified absolute lunar geographic accuracy. Bounding-box spread is correspondence footprint, not registered image coverage.',
            'Capacity is a subdivision trigger, not a hard bound at terminal leaves; depth/size limits may leave over-capacity leaves. Duplicate coordinates are retained as distinct input indices.'],
        provenance=dict(inputs=data['report']['provenance']['inputs'], evaluation_path=data['evaluation_path'],
            evaluation_sha256=data['evaluation_sha256'], m11_path='results/milestone_11_spatial_distribution/spatial_distribution.json',
            m11_sha256=m11_hash, affine_matrix=data['matrix'].tolist(), transform_refitted=False,
            loftr_rerun=False, ransac_rerun=False, registration_rerun=False),
        software=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__))
    output.mkdir(parents=True, exist_ok=False)
    write_plots(output, data, experiments, comparisons)
    paths = {record['path']: record['sha256'] for record in report['provenance']['inputs'].values()}
    paths[data['evaluation_path']] = data['evaluation_sha256']
    paths[report['provenance']['m11_path']] = m11_hash
    for path, expected in paths.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Saved artifact changed during experiment: {path}')
    report['runtime_seconds'] = perf_counter()-tick
    (output/'quadtree.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'quadtree_summary.txt').write_text(summary(report), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results/milestone_12_quadtree')
    print(summary(run(parser.parse_args().output)))


if __name__ == '__main__':
    main()
