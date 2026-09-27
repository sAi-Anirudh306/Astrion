"""Controlled phase-congruency branch: python -m scripts.tycho_phase_congruency.

Reads the saved intensity baseline; never writes to that experiment. No
downstream parameter search or geometric fallback is performed.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np
import rasterio

from scripts.real_tmc_wac import save_png, draw_matches
from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.phase_congruency import phase_congruency, PhaseCongruencyConfig
from src.features.sift import extract_sift, prepare_sift_image, SIFTConfig
from src.features.rootsift import transform_sift
from src.matching.descriptor_matching import match_descriptors
from src.matching.ratio_test import filter_ratio
from src.geometry.ransac import verify_affine, RANSACConfig


def digest(path: Path) -> str:
    """Stream a content hash without loading the scientific array into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pc_statistics(result) -> dict:
    """Record actual structural response range and estimated noise thresholds."""
    return {"min": float(result.data.min()), "max": float(result.data.max()),
            "mean": float(result.data.mean()), "std": float(result.data.std()),
            "noise_thresholds": result.noise_thresholds,
            "runtime_seconds": result.runtime_seconds}


def main() -> None:
    """Replace only the feature-input representation relative to the baseline."""
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "results/experiment_tycho_real_multiscale/report.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    moving, fixed = Path(baseline["moving"]), Path(baseline["fixed"])
    config = PreprocessingConfig(**baseline["preprocessing"])
    sift_config = SIFTConfig(**baseline["sift"])
    ransac_config = RANSACConfig(**baseline["ransac"])
    if (baseline["ratio_threshold"] != .75 or asdict(ransac_config) != asdict(RANSACConfig())
            or asdict(sift_config) != asdict(SIFTConfig())):
        raise ValueError("Saved baseline differs from requested fixed parameters")
    pc_config = PhaseCongruencyConfig()
    output = root / "results/experiment_tycho_phase_congruency"
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): digest(p) for p in (moving, fixed, baseline_path)}
    tmc = preprocess_image(np.load(moving, mmap_mode="r", allow_pickle=False), config)
    with rasterio.open(fixed) as dataset:
        wac = preprocess_image(dataset.read(1), config)
        valid_count = int((dataset.read_masks(1) > 0).sum())
    wac_pc = phase_congruency(wac.data, pc_config)
    wac_sift = extract_sift(wac_pc.data, sift_config)
    wac_root = transform_sift(wac_sift)
    wac_display = prepare_sift_image(wac_pc.data)
    report = {"baseline_report": str(baseline_path), "sha256": hashes,
              "moving": str(moving), "fixed": str(fixed),
              "preprocessing": asdict(config), "phase_congruency": asdict(pc_config),
              "pc_aggregation": wac_pc.representation,
              "sift": asdict(sift_config), "ratio_threshold": .75,
              "ransac": asdict(ransac_config), "matching": baseline["matching"],
              "wac_shape": list(wac.data.shape), "wac_resolution_m": 100,
              "wac_valid_pixels": valid_count, "wac_pc_statistics": pc_statistics(wac_pc),
              "shared_runtime_seconds": {"tmc_preprocessing": tmc.runtime_seconds,
                  "wac_preprocessing": wac.runtime_seconds, "wac_pc": wac_pc.runtime_seconds,
                  "wac_sift": wac_sift.runtime_seconds, "wac_rootsift": wac_root.runtime_seconds},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "opencv": cv2.__version__, "rasterio": rasterio.__version__},
              "limitations": baseline["limitations"] + [
                  "Phase congruency uses a white-noise approximation; shadow boundaries remain structural responses.",
                  "No per-image contrast stretch of phase-congruency maps; downstream SIFT thresholds unchanged.",
                  "Reflection padding reduces but cannot eliminate FFT boundary effects."],
              "scales": [], "comparison": []}
    for base in baseline["scales"]:
        tick = perf_counter()
        resolution = base["target_resolution_m"]
        scale = normalize_resolution(tmc.data, baseline["source_resolution_m"], resolution)
        downsample_time = perf_counter()-tick
        if list(scale.data.shape) != base["shape"]:
            raise ValueError("Scale grid differs from saved baseline")
        pc = phase_congruency(scale.data, pc_config)
        features = extract_sift(pc.data, sift_config)
        roots = transform_sift(features)
        knn = match_descriptors(roots.descriptors, wac_root.descriptors, k=2)
        ratio = filter_ratio(knn, threshold=.75)
        geometry, failure = None, None
        geometry_start = perf_counter()
        try:
            geometry = verify_affine(ratio, roots.points, wac_root.points, ransac_config)
        except (ValueError, RuntimeError) as exc:
            failure = str(exc)
        geometry_time = perf_counter()-geometry_start
        record = {"target_resolution_m": resolution, "resize_factor": scale.nominal_factor,
                  "actual_factors_xy": scale.actual_factors_xy,
                  "effective_resolution_xy_m": scale.effective_resolution_xy,
                  "shape": list(scale.data.shape), "phase_congruency": asdict(pc_config),
                  "pc_statistics": pc_statistics(pc), "tmc_keypoints": len(features.keypoints),
                  "wac_keypoints": len(wac_sift.keypoints),
                  "tmc_rootsift_shape": list(roots.descriptors.shape),
                  "wac_rootsift_shape": list(wac_root.descriptors.shape), "descriptor_dtype": "float32",
                  "knn_groups": len(knn.neighbors), "knn_candidates": sum(map(len, knn.neighbors)),
                  "ratio_matches": len(ratio.accepted),
                  "ransac_inliers": geometry.inlier_count if geometry else 0,
                  "inlier_ratio": geometry.inlier_ratio if geometry else None,
                  "transform_scaled_tmc_to_wac": geometry.matrix.tolist() if geometry else None,
                  "rmse_wac_px": float(np.sqrt(np.mean(geometry.errors[geometry.inlier_mask]**2))) if geometry else None,
                  "residual_wac_px": geometry.inlier_error_statistics if geometry else None,
                  "ransac_failure": failure,
                  "runtime_seconds": {"downsampling": downsample_time, "phase_congruency": pc.runtime_seconds,
                      "sift": features.runtime_seconds, "rootsift": roots.runtime_seconds,
                      "matching": knn.runtime_seconds, "ratio": ratio.runtime_seconds,
                      "ransac": geometry_time, "pipeline_excluding_shared": perf_counter()-tick}}
        source_points, destination_points = roots.points, wac_root.points
        record["accepted_matches"] = [asdict(m) | {
            "source_xy": source_points[m.query_index].tolist(),
            "destination_xy": destination_points[m.train_index].tolist(),
            "inlier": bool(geometry.inlier_mask[i]) if geometry else None,
            "residual_wac_px": float(geometry.errors[i]) if geometry else None}
            for i, m in enumerate(ratio.accepted)]
        folder = output / f"{int(resolution)}m"
        folder.mkdir()
        display = prepare_sift_image(pc.data)
        save_png(folder / "tmc_scaled_intensity.png", prepare_sift_image(scale.data))
        save_png(folder / "wac_intensity.png", prepare_sift_image(wac.data))
        save_png(folder / "tmc_phase_congruency.png", display)
        save_png(folder / "wac_phase_congruency.png", wac_display)
        for name, image, result in (("tmc", display, features), ("wac", wac_display, wac_sift)):
            save_png(folder / f"{name}_pc_keypoints.png", cv2.drawKeypoints(
                image, result.keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        draw_matches(folder / "ratio_matches.png", display, wac_display, features, wac_sift, ratio.accepted)
        draw_matches(folder / "ransac_inliers.png", display, wac_display, features, wac_sift,
                     geometry.inliers if geometry else ())
        np.save(folder / "tmc_phase_congruency.npy", pc.data, allow_pickle=False)
        record["visualization_directory"] = str(folder)
        report["scales"].append(record)
        report["comparison"].append({"target_resolution_m": resolution,
            "baseline_ratio_matches": base["ratio_matches"], "pc_ratio_matches": len(ratio.accepted),
            "baseline_inliers": base["ransac_inliers"], "pc_inliers": record["ransac_inliers"]})
        print(json.dumps({k: v for k, v in record.items() if k != "accepted_matches"}), flush=True)
    np.save(output / "wac_phase_congruency.npy", wac_pc.data, allow_pickle=False)
    maximum = max(r["ransac_inliers"] for r in report["scales"])
    report["best_scales_by_inlier_count_m"] = [r["target_resolution_m"] for r in report["scales"]
                                               if r["ransac_inliers"] == maximum] if maximum else []
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError("Source or baseline content changed")
    report["total_runtime_seconds_including_io"] = perf_counter()-started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
