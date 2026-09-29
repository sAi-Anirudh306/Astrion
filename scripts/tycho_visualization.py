"""Milestone 10: visualize saved Tycho artifacts without inference, fitting or warping.

Run: .venv/Scripts/python.exe -m scripts.tycho_visualization
Repeat runs require --output results/<new-directory>; existing results are preserved.
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

from scripts.tycho_evaluation import read_raster, check_saved_mask
from src.evaluation.metrics import correspondence_statistics, reprojection_residuals, spatial_distribution
from src.evaluation.visualization import (plot_image, plot_matches, plot_registration,
    overlay_image, checkerboard_image, normalized_difference, show_display,
    plot_residuals, plot_spatial_distribution, plot_metrics_panel)


def load_artifacts(root: Path) -> dict:
    """Validate M9 provenance hashes, masks, grid and M8 feature-fit consistency."""
    report_path = root/"results/milestone_09_evaluation/evaluation.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Required evaluation report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs, paths = report["provenance"]["inputs"], {}
    for name, record in inputs.items():
        path = (root/record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact must be inside project: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required saved artifact missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Artifact changed since evaluation: {path}")
        paths[name] = path
    for required in ("before", "after", "reference", "before_mask", "after_mask", "points", "metadata"):
        if required not in paths:
            raise ValueError(f"Evaluation provenance is missing {required}")
    moving, moving_mask, grid_a = read_raster(paths["before"])
    registered, registered_mask, grid_b = read_raster(paths["after"])
    reference, reference_mask, grid = read_raster(paths["reference"])
    if grid_a != grid or grid_b != grid or list(grid[0]) != report["provenance"]["shape"]:
        raise ValueError("Saved images must share the evaluated reference grid")
    check_saved_mask(paths["before_mask"], moving_mask)
    check_saved_mask(paths["after_mask"], registered_mask)
    with np.load(paths["points"], allow_pickle=False) as archive:
        source, destination, confidence, keep = (archive[key].copy() for key in
            ("source", "destination", "confidence", "inlier_mask"))
    if keep.dtype != bool or keep.shape != (len(source),):
        raise ValueError("Saved inlier mask must be boolean and aligned")
    if correspondence_statistics(keep) != report["correspondence"]:
        raise ValueError("Saved classifications disagree with evaluation")
    matrix = np.asarray(report["provenance"]["affine_matrix"], float)
    metadata = json.loads(paths["metadata"].read_text())
    if not np.array_equal(matrix, metadata["affine_matrix"]):
        raise ValueError("Transform differs from Milestone 8")
    residuals = reprojection_residuals(source, destination, matrix)[keep]
    if not np.isclose(np.sqrt(np.mean(residuals**2)), report["reprojection_residuals"]["wac_pixels"]["rmse"], rtol=0, atol=1e-9):
        raise ValueError("Feature residuals disagree with evaluation")
    if spatial_distribution(destination[keep], reference.shape) != report["spatial_distribution"]:
        raise ValueError("Spatial counts disagree with evaluation")
    return dict(report=report, moving=moving, reference=reference, registered=registered,
                moving_mask=moving_mask, reference_mask=reference_mask, registered_mask=registered_mask,
                source=source, destination=destination, confidence=confidence, keep=keep, matrix=matrix,
                evaluation_path=str(report_path.relative_to(root)),
                evaluation_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())


def metric_labels(report: dict) -> dict[str, str]:
    """Format saved measurements with units; no fabricated geographic accuracy."""
    c, r, s = (report["correspondence"], report["reprojection_residuals"]["wac_pixels"], report["spatial_distribution"])
    pair = report["before_vs_after"]["own_common_coverage"]
    return {"Candidates": str(c["candidate_count"]), "Inliers": str(c["inlier_count"]),
            "Inlier ratio": f"{100*c['inlier_ratio']:.2f}%", "Feature RMSE": f"{r['rmse']:.3f} px",
            "Median residual": f"{r['median']:.3f} px", "Spatial coverage": f"{s['occupied_cells']}/{s['total_cells']} cells",
            "Registered coverage": f"{report['coverage']['registered_coverage_percentage']:.2f}%",
            "Pearson before": f"{pair['before']['correlation']:.3f}",
            "Pearson after": f"{pair['after']['correlation']:.3f}"}


def write_figures(output: Path, data: dict, block_size: int, arrow_scale: float) -> dict:
    """Compose reusable axes plots into the twelve requested presentation outputs."""
    a, b, registered = (data[key] for key in ("moving", "reference", "registered"))
    ma, mb, mr = (data[key] for key in ("moving_mask", "reference_mask", "registered_mask"))
    src, dst, keep = (data[key] for key in ("source", "destination", "keep"))
    metrics = metric_labels(data["report"])
    displayed = {}

    def save(fig: plt.Figure, name: str, note: str = "") -> None:
        fig.text(.015, .018, "Display-only P1/P99 normalization | Lavender = nodata / excluded"+
                 (" | "+note if note else ""), fontsize=8, color="#526171")
        fig.subplots_adjust(left=.07, right=.96, top=.88, bottom=.11, wspace=.30, hspace=.35)
        fig.savefig(output/name, dpi=180, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_image(axes[0], a, ma, "INPUT | Map-projected Chandrayaan-2 TMC-2")
    plot_image(axes[1], b, mb, "REFERENCE | LRO WAC, Tycho")
    save(fig, "input_pair.png", "100 m/pixel map grid")

    for name, selected, cap in (("verified_matches.png", keep, 150),
                                ("inliers_outliers.png", np.ones(len(keep), bool), None)):
        fig, ax = plt.subplots(figsize=(13, 7))
        indices = plot_matches(ax, a, b, src[selected], dst[selected], ma, mb,
            inlier_mask=keep[selected] if cap is None else None,
            confidence=data["confidence"][selected] if cap is not None else None,
            max_matches=cap, title="GEOMETRIC VERIFICATION | Inliers and outliers" if cap is None
            else "VERIFIED LoFTR PAIRS | TMC (left) to WAC (right)")
        displayed[name] = np.flatnonzero(selected)[indices].tolist()
        save(fig, name, "All candidates shown" if cap is None else "150/569 inliers shown; evenly spaced original indices")

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    plot_registration(axes, a, b, registered, ma, mb, mr)
    save(fig, "registration_comparison.png", "Common-valid overlay")

    composites = {
        "registration_overlay.png": (overlay_image(registered, b, mr, mb), "REGISTERED | 50% TMC / 50% WAC overlay", "gray"),
        "registration_checkerboard.png": (checkerboard_image(registered, b, mr, mb, block_size),
            f"REGISTERED | Checkerboard, {block_size}-pixel blocks", "gray"),
        "normalized_difference.png": (normalized_difference(registered, b, mr, mb),
            "CROSS-IMAGE DIAGNOSTIC | Absolute normalized difference", "magma")}
    for name, (display, title, cmap) in composites.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        artist = show_display(ax, display, title, cmap)
        if name == "normalized_difference.png":
            fig.colorbar(artist, ax=ax, label="Normalized intensity difference (display units)", fraction=.045)
        save(fig, name, "Common coverage only; not an accuracy metric")

    for vectors in (False, True):
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_residuals(ax, b, src[keep], dst[keep], data["matrix"], mb, vectors, arrow_scale)
        save(fig, "residual_vectors.png" if vectors else "residual_map.png",
             "Predicted to matched WAC position; color unscaled" if vectors else "569 inliers; not geographic ground truth")
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_spatial_distribution(ax, b, dst[keep], mb)
    save(fig, "spatial_distribution.png", "Entropy is not proof of uniformity")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_metrics_panel(ax, metrics, "ASTRION | Tycho registration evaluation",
        "Feature residuals are not absolute geographic accuracy.\nPearson: cross-image similarity on respective overlaps.")
    save(fig, "metrics_panel.png", "Saved Milestone 9 measurements")

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    fig.suptitle("ASTRION / LunarMatch   |   Tycho registration", fontsize=23, fontweight="bold", x=.07, ha="left", y=.975)
    fig.text(.07, .925, "INPUT  >  CORRESPONDENCE  >  GEOMETRIC VERIFICATION  >  REGISTRATION  >  EVALUATION",
             fontsize=11, color="#406179")
    plot_image(axes[0, 0], a, ma, "01 | TMC-2: approximate map projection")
    plot_image(axes[0, 1], b, mb, "02 | LRO WAC reference: 100 m/pixel")
    shown = plot_matches(axes[0, 2], a, b, src[keep], dst[keep], ma, mb,
                        max_matches=45, title="03 | Verified LoFTR correspondences")
    displayed["astrion_result_summary.png"] = np.flatnonzero(keep)[shown].tolist()
    plot_image(axes[1, 0], registered, mr, "04 | Registered TMC-2: saved affine result")
    show_display(axes[1, 1], composites["registration_checkerboard.png"][0], "05 | Checkerboard: common valid coverage")
    plot_metrics_panel(axes[1, 2], metrics, "06 | Evaluation / saved measurements",
        "Feature residuals, not absolute geographic accuracy.\nPearson uses each image's valid WAC overlap.")
    fig.text(.07, .065, "No external ground-control truth or rigorous TMC camera/DEM orthorectification. Images are independently display-normalized.",
             fontsize=10, color="#66464a")
    save(fig, "astrion_result_summary.png", "45/569 verified pairs shown; deterministic display sampling")
    return displayed


def run(output: Path, block_size: int = 32, arrow_scale: float = 5.) -> dict:
    """Write a fresh experiment directory and record inputs, display choices and runtime."""
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(root/"results"):
        raise ValueError("Visualization output must be under project results/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")
    if not np.isscalar(arrow_scale) or not np.isfinite(arrow_scale) or arrow_scale <= 0:
        raise ValueError("Arrow scale must be finite and positive")
    data = load_artifacts(root)
    output.mkdir(parents=True, exist_ok=False)
    displayed = write_figures(output, data, block_size, arrow_scale)
    files = sorted(output.glob("*.png"))
    if len(files) != 12 or any(path.stat().st_size == 0 for path in files):
        raise RuntimeError("Expected twelve nonempty visualization PNGs")
    # Read-only visualization must leave every scientific/evaluation artifact intact.
    for record in data["report"]["provenance"]["inputs"].values():
        if hashlib.sha256((root/record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("Scientific input changed during visualization")
    manifest = dict(evaluation_path=data["evaluation_path"], evaluation_sha256=data["evaluation_sha256"],
        inputs=data["report"]["provenance"]["inputs"], outputs=[str(path.relative_to(root)) for path in files],
        display_normalization="Independent valid-pixel P1/P99; min/max fallback for collapsed bounds; comparison composites normalize on common support",
        nodata_color="#d7cfe3", checkerboard_block_size=block_size, residual_arrow_magnification=arrow_scale,
        residual_vector_direction="Transformed TMC position to matched WAC position; colors use unscaled residuals",
        sampling="Evenly spaced original indices, display only; full candidate set shown in inliers_outliers.png",
        displayed_original_candidate_indices=displayed, limitations=data["report"]["limitations"],
        runtime_seconds=perf_counter()-started,
        software=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__))
    (output/"visualization_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"results/milestone_10_visualization")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--arrow-scale", type=float, default=5.)
    args = parser.parse_args()
    result = run(args.output, args.block_size, args.arrow_scale)
    print(f"Created {len(result['outputs'])} figures in {result['runtime_seconds']:.3f} s: {args.output}")


if __name__ == "__main__":
    main()
