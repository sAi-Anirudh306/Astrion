"""Evaluate saved Milestone 8 artifacts without inference or geometric fitting.

Run: .venv/Scripts/python.exe -m scripts.tycho_evaluation
Use --output results/<new-name> for a repeat run; existing outputs are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.evaluation.metrics import (affine_diagnostics, correspondence_statistics,
    coverage_statistics, image_similarity, pixel_to_map_distance,
    reprojection_residuals, residual_statistics, residual_thresholds, spatial_distribution)


def read_raster(path: Path) -> tuple:
    """Read scientific values and combine GDAL validity, nodata and finite checks."""
    with rasterio.open(path) as dataset:
        image = dataset.read(1)
        valid = (dataset.read_masks(1) > 0) & np.isfinite(image)
        if dataset.nodata is not None:
            valid &= image != dataset.nodata
        return image, valid, (dataset.shape, dataset.crs, dataset.transform)


def check_saved_mask(path: Path, valid: np.ndarray) -> None:
    """Require the published PNG mask to agree exactly with scientific coverage."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.shape != valid.shape or not np.isin(mask, [0, 255]).all():
        raise ValueError(f"Invalid saved binary mask: {path}")
    if not np.array_equal(mask > 0, valid):
        raise ValueError(f"Saved PNG and scientific raster validity disagree: {path}")


def visual_diagnostics(output: Path, reference: np.ndarray, valid: np.ndarray,
                       common: np.ndarray, points: np.ndarray,
                       residuals: np.ndarray, counts: list) -> None:
    """Four evaluation plots; correspondence coordinates are used unchanged."""
    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output/name, dpi=150)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=30, color="steelblue", edgecolor="white")
    plt.xlabel("Inlier feature reprojection residual (WAC pixels)")
    plt.ylabel("Inlier count")
    plt.title("Affine fit residuals (not geographic ground truth)")
    save("residual_histogram.png")
    lo, hi = np.percentile(reference[valid], [1, 99])
    background = np.ma.array(reference, mask=~valid)
    height, width = reference.shape
    for kind in ("residual", "grid"):
        plt.figure(figsize=(7, 6))
        plt.imshow(background, cmap="gray", vmin=lo, vmax=hi)
        if kind == "residual":
            scatter = plt.scatter(points[:, 0], points[:, 1], c=residuals,
                                  s=12, cmap="plasma", vmin=0)
            plt.colorbar(scatter, label="Feature residual (WAC pixels)")
            plt.title("Verified reference feature locations")
        else:
            plt.scatter(points[:, 0], points[:, 1], s=3, c="cyan", alpha=.4)
            for x in np.linspace(0, width, 5):
                plt.axvline(x, color="yellow", linewidth=.8)
            for y in np.linspace(0, height, 5):
                plt.axhline(y, color="yellow", linewidth=.8)
            for row in range(4):
                for col in range(4):
                    plt.text((col+.5)*width/4, (row+.5)*height/4, str(counts[row][col]),
                             ha="center", va="center", color="white", bbox=dict(facecolor="black", alpha=.65))
            plt.title("4 x 4 reference grid: verified inlier counts")
        plt.xlim(-.5, width-.5)
        plt.ylim(height-.5, -.5)
        plt.xlabel("Reference column (pixels)")
        plt.ylabel("Reference row (pixels)")
        save("residual_spatial_map.png" if kind == "residual" else "inlier_grid_distribution.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(common, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Common valid coverage: {int(common.sum()):,} pixels\nWhite = valid TMC and WAC; black = excluded")
    plt.xlabel("Reference column (pixels)")
    plt.ylabel("Reference row (pixels)")
    save("coverage_mask.png")


def summary(report: dict) -> str:
    """Concise human-readable values with the essential scientific limitations."""
    r = report["reprojection_residuals"]["wac_pixels"]
    c, s, a = (report[key] for key in ("coverage", "spatial_distribution", "transform_diagnostics"))
    lines = ["ASTRION Milestone 9 registration evaluation", str(report["correspondence"]),
             "Inlier residuals (WAC pixels): " + json.dumps(r),
             "Nominal map distances (m): " + json.dumps(report["reprojection_residuals"]["nominal_map_metres"])]
    lines += [f"Residual <= {t['pixels']:g} px ({t['nominal_map_metres']:g} m): {t['count']} ({t['percentage']:.2f}%)"
              for t in report["residual_thresholds"]]
    lines += [f"Spatial occupancy: {s['occupied_cells']}/{s['total_cells']} ({s['occupancy_percentage']:.2f}%); entropy {s['normalized_entropy']:.6f}",
              "Grid counts: " + str(s["counts"]), "Coverage: " + json.dumps(c),
              "Affine diagnostics: " + json.dumps(a)]
    for support in ("own_common_coverage", "shared_before_after_coverage"):
        pair = report["before_vs_after"][support]
        lines.append(f"Similarity ({support}): " + "; ".join(
            f"{key}: n={value['common_valid_pixels']}, Pearson={value['correlation']}" for key, value in pair.items()))
    lines += ["SSIM omitted: irregular nodata requires masked local windows; no masked-window SSIM implemented.",
              f"Evaluation runtime: {report['runtime_seconds']:.3f} s", *report["limitations"]]
    return "\n".join(lines)+"\n"


def run(output: Path) -> dict:
    """Validate artifact provenance and common grid, then write a new evaluation."""
    tick = perf_counter()
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(root/"results"):
        raise ValueError("Evaluation outputs must be under project results/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    paths = dict(
        metadata=root/"results/milestone_08_registration/registration_metadata.json",
        experiment=root/"results/experiment_tycho_map_projected/report.json",
        points=root/"results/experiment_tycho_map_projected/loftr_0.20_matches.npz",
        before=root/"data/processed/tycho/map_projected/tmc_on_wac_grid.tif",
        before_mask=root/"data/processed/tycho/map_projected/tmc_on_wac_grid_valid_mask.png",
        after=root/"data/processed/tycho/registered/tmc_registered_to_wac.tif",
        after_mask=root/"data/processed/tycho/registered/tmc_registered_valid_mask.png",
        reference=root/"data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.tif")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required saved artifact missing: {path}")
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    metadata = json.loads(paths["metadata"].read_text())
    experiment = json.loads(paths["experiment"].read_text())
    for key in ("before", "reference", "experiment", "points"):
        relative = str(paths[key].relative_to(root))
        if hashes[key] != metadata["input_sha256"][relative]:
            raise ValueError(f"Artifact differs from Milestone 8 input: {paths[key]}")
    matrix = np.asarray(metadata["affine_matrix"], dtype=float)
    row = next(r for r in experiment["matching"] if r["name"] == "loftr_0.20")
    if not np.array_equal(matrix, row["affine_matrix"]):
        raise ValueError("Saved affine matrices disagree")
    with np.load(paths["points"], allow_pickle=False) as archive:
        source, destination, keep = (archive[key] for key in ("source", "destination", "inlier_mask"))
    correspondence = correspondence_statistics(keep)
    if keep.dtype != bool or keep.shape != (len(source),):
        raise ValueError("Saved inlier mask must be boolean and aligned with correspondences")
    all_residuals = reprojection_residuals(source, destination, matrix)
    residuals = all_residuals[keep]
    if (correspondence["candidate_count"] != metadata["ransac_candidate_count"] or
            correspondence["inlier_count"] != metadata["ransac_inlier_count"]):
        raise ValueError("Saved correspondence counts disagree with Milestone 8")
    stats = residual_statistics(residuals)
    if not np.isclose(stats["rmse"], metadata["post_transform_correspondence_residuals"]["rmse"], rtol=0, atol=1e-9):
        raise ValueError("Residual RMSE disagrees with Milestone 8")
    before, before_valid, before_grid = read_raster(paths["before"])
    after, after_valid, after_grid = read_raster(paths["after"])
    reference, reference_valid, grid = read_raster(paths["reference"])
    if before_grid != grid or after_grid != grid:
        raise ValueError("All scientific rasters must share shape, CRS and geotransform")
    shape, crs, affine = grid
    if crs is None or not crs.is_projected or crs.linear_units != "metre":
        raise ValueError("Expected a projected reference grid in metres")
    basis = np.array([[affine.a, affine.b], [affine.d, affine.e]])
    scale = float(np.linalg.norm(basis[:, 0]))
    if not np.allclose(basis.T@basis, np.eye(2)*scale**2) or not np.isclose(scale, 100):
        raise ValueError("Expected an isotropic 100 m/pixel WAC grid")
    check_saved_mask(paths["before_mask"], before_valid)
    check_saved_mask(paths["after_mask"], after_valid)
    if int(after_valid.sum()) != metadata["registered_valid_pixels"]:
        raise ValueError("Registered coverage disagrees with Milestone 8")
    own_before = image_similarity(before, reference, before_valid, reference_valid)
    own_after = image_similarity(after, reference, after_valid, reference_valid)
    shared = before_valid & after_valid & reference_valid
    spatial = spatial_distribution(destination[keep], shape)
    report = dict(
        correspondence=correspondence,
        reprojection_residuals=dict(wac_pixels=stats, nominal_map_metres=residual_statistics(pixel_to_map_distance(residuals, scale)), metres_per_pixel=scale),
        residual_thresholds=residual_thresholds(residuals, scale), spatial_distribution=spatial,
        coverage=coverage_statistics(after_valid, reference_valid),
        transform_diagnostics=affine_diagnostics(matrix), image_similarity=own_after,
        before_vs_after=dict(own_common_coverage=dict(before=own_before, after=own_after),
            shared_before_after_coverage=dict(before=image_similarity(before, reference, shared, shared),
                                             after=image_similarity(after, reference, shared, shared))),
        limitations=[
            f"The {stats['rmse']:.6f}-pixel correspondence RMSE measures residual agreement of image-derived matched features after affine fitting. At 100 m/pixel it corresponds numerically to {stats['rmse']*scale:.1f} m of map-grid displacement, NOT independently verified absolute lunar geolocation accuracy.",
            "No external ground-control truth or rigorous TMC camera/DEM orthorectification is available in this experiment.",
            "Residuals use the same RANSAC inliers used for affine fitting; they are not held-out accuracy or proof of correct matches.",
            "TMC and WAC are not radiometrically equivalent. Correlation is a cross-image similarity diagnostic, not geometric accuracy; lighting, resampling and independent percentile clipping affect it.",
            "Own-coverage comparisons use different supports; shared-coverage comparison controls that footprint difference. Neither establishes causality or external accuracy.",
            "Grid entropy measures count concentration at the chosen grid resolution, not proof of uniform correspondence."],
        provenance=dict(inputs={key: dict(path=str(path.relative_to(root)), sha256=hashes[key]) for key, path in paths.items()},
            sensor_pair=["Chandrayaan-2 TMC-2", "LRO WAC"], affine_matrix=matrix.tolist(),
            transform_direction=metadata["transform_direction"], shape=list(shape), crs_wkt=crs.to_wkt(), geotransform=list(affine),
            ransac=experiment["ransac"], loftr_confidence_threshold=.2, preprocessing=experiment["preprocessing"],
            software=dict(python=platform.python_version(), numpy=np.__version__, rasterio=rasterio.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__)))
    output.mkdir(parents=True, exist_ok=False)
    visual_diagnostics(output, reference, reference_valid, after_valid & reference_valid,
                       destination[keep], residuals, spatial["counts"])
    report["runtime_seconds"] = perf_counter()-tick
    (output/"evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"evaluation_summary.txt").write_text(summary(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"results/milestone_09_evaluation")
    print(summary(run(parser.parse_args().output)))


if __name__ == "__main__":
    main()
