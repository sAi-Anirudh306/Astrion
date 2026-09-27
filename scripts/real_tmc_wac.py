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


if __name__ == "__main__":
    main()
