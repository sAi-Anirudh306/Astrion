"""72 fixed candidate-generation diagnostics; no registration or tuning.

Run: python -m scripts.tycho_candidate_generation
"""
from dataclasses import asdict
import csv
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import rasterio

from scripts.real_tmc_wac import save_png, draw_matches
from scripts.tycho_phase_congruency import digest
from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.phase_congruency import phase_congruency, PhaseCongruencyConfig
from src.features.sift import extract_sift, prepare_sift_image, SIFTConfig
from src.features.rootsift import transform_sift
from src.matching.mutual_matching import mutual_nearest_neighbors, filter_mutual_ratio
from src.matching.ratio_test import filter_ratio, RatioMatch, RatioTestResult
from src.geometry.ransac import verify_affine, RANSACConfig


def statistics(values) -> dict | None:
    """Finite scalar summary; empty measurements are absent, not zero error."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return dict(min=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), max=float(values.max()))


def geometry_diagnostics(result, points: np.ndarray, shape: tuple, resolution: float) -> dict:
    """Measurement-only diagnostics; never alter fitted models or inlier masks.

    Broad scale warnings compare singular values with nominal resolution/100;
    map projection and viewing geometry make these heuristics, not ground truth.
    """
    destination = points[[m.train_index for m in result.inliers]]
    cells = np.clip((destination / np.array(shape[::-1]) * 4).astype(int), 0, 3)
    counts = np.bincount(cells[:, 1]*4+cells[:, 0], minlength=16)
    occupied = int(np.count_nonzero(counts))
    fraction = float(counts.max()/len(destination))
    singular = np.linalg.svd(result.matrix[:, :2], compute_uv=False)
    condition = float(singular[0]/singular[-1])
    relative = singular / (resolution/100)
    rmse = float(np.sqrt(np.mean(result.errors[result.inlier_mask]**2)))
    warnings = []
    if result.inlier_count < 8:
        warnings.append("fewer_than_8_inliers")
    if occupied < 4 or fraction > .6:
        warnings.append("spatially_concentrated")
    if relative.min() < .25 or relative.max() > 4 or condition > 5:
        warnings.append("implausible_scale_or_anisotropy_under_nominal_GSD")
    if rmse > 2:
        warnings.append("poor_residual_rmse_above_2px")
    unique = len(np.unique(destination, axis=0))
    if unique < 8:
        warnings.append("fewer_than_8_unique_destination_positions")
    return {"occupied_wac_cells": occupied, "wac_grid_counts": counts.reshape(4, 4).tolist(),
            "max_cell_fraction": fraction, "unique_inlier_destination_positions": unique,
            "linear_singular_values": singular.tolist(), "singular_values_relative_nominal": relative.tolist(),
            "linear_condition_number": condition, "determinant": float(np.linalg.det(result.matrix[:, :2])),
            "rmse": rmse, "warnings": warnings}


def mutual_geometry_container(mutual) -> RatioTestResult:
    """Adapt indices to the existing geometry API without applying a ratio test.

    RatioTestResult is the required legacy input container. Threshold is None
    here to explicitly mean 'not applied'; available ratios are real measured
    values, not acceptance decisions. The estimator consumes only indices.
    """
    accepted = []
    for match in mutual.accepted:
        group = mutual.forward.neighbors[match.query_index]
        second = group[1].distance if len(group) > 1 else None
        ratio = match.distance/second if second else None
        accepted.append(RatioMatch(match.query_index, match.train_index, match.distance,
                                   second, ratio))  # type: ignore[arg-type]
    return RatioTestResult(tuple(accepted), mutual.forward.query_count,
        mutual.forward.query_count-len(accepted), 0, 0, None, 0.)  # type: ignore[arg-type]


def evaluate(candidates, a, b, shape: tuple, resolution: float) -> tuple[dict, object]:
    """Use unchanged affine RANSAC; keep failures explicit and measurements null."""
    tick = perf_counter()
    result, reason = None, None
    try:
        result = verify_affine(candidates, a.points, b.points, RANSACConfig())
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
    elapsed = perf_counter()-tick
    record = {"candidate_count": len(candidates.accepted),
        "distance_statistics": statistics([m.nearest_distance for m in candidates.accepted]),
        "inliers": result.inlier_count if result else 0,
        "inlier_ratio": result.inlier_ratio if result else None,
        "transform": result.matrix.tolist() if result else None,
        "residual_statistics": result.inlier_error_statistics if result else None,
        "rmse": None, "occupied_wac_cells": 0, "warnings": [reason] if reason else [],
        "failure_reason": reason, "ransac_runtime_seconds": elapsed}
    if result:
        record.update(geometry_diagnostics(result, b.points, shape, resolution))
    record["candidates"] = [asdict(m) | {"source_xy": a.points[m.query_index].tolist(),
        "destination_xy": b.points[m.train_index].tolist(),
        "inlier": bool(result.inlier_mask[i]) if result else None,
        "residual_px": float(result.errors[i]) if result else None}
        for i, m in enumerate(candidates.accepted)]
    return record, result


def main() -> None:
    """Cache features per representation/scale and test exactly nine strategies."""
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    base_path = root / "results/experiment_tycho_real_multiscale/report.json"
    pc_path = root / "results/experiment_tycho_phase_congruency/report.json"
    baseline = json.loads(base_path.read_text())
    previous_pc = json.loads(pc_path.read_text())
    moving, fixed = Path(baseline["moving"]), Path(baseline["fixed"])
    output = root / "results/experiment_tycho_candidate_generation"
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): digest(p) for p in (moving, fixed, base_path, pc_path)}
    config = PreprocessingConfig(**baseline["preprocessing"])
    sift = SIFTConfig(**baseline["sift"])
    pc_config = PhaseCongruencyConfig(**previous_pc["phase_congruency"])
    if baseline["ransac"] != asdict(RANSACConfig()):
        raise ValueError("Baseline geometric parameters differ")
    tmc = preprocess_image(np.load(moving, mmap_mode="r"), config)
    with rasterio.open(fixed) as dataset:
        wac = preprocess_image(dataset.read(1), config)
    scaled = {r: normalize_resolution(tmc.data, 5.15, r) for r in (75., 100., 125., 150.)}
    report = {"sha256": hashes, "preprocessing": asdict(config), "sift": asdict(sift),
        "phase_congruency": asdict(pc_config), "ransac": asdict(RANSACConfig()),
        "thresholds": [.75, .8, .85, .9], "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
        "warning_policy": {"minimum_inliers": 8, "minimum_occupied_cells": 4,
            "maximum_cell_fraction": .6, "maximum_rmse_px": 2,
            "nominal_relative_singular_value_range": [.25, 4], "maximum_condition_number": 5},
        "limitations": ["No ground truth; fitted residuals are not accuracy.",
            "All strategies other than ratio 0.75 are candidate-generation diagnostics.",
            "Nominal-GSD physical plausibility warnings are heuristic, not a success score.",
            "Nodata and shadow boundaries remain in both representations.",
            "The same points fit and evaluate the model; three-point fits can have near-zero residual.",
            "2000 RANSAC iterations may miss models at very low true-inlier fractions."],
        "visualization_selection": "highest inlier count per representation/scale; tie: occupied cells then lower RMSE; stable strategy order",
        "shared_runtime_seconds": {"tmc_preprocessing": tmc.runtime_seconds, "wac_preprocessing": wac.runtime_seconds},
        "feature_runs": [], "configurations": []}
    for representation in ("intensity", "phase_congruency"):
        feature_tick = perf_counter()
        fixed_image = wac.data if representation == "intensity" else phase_congruency(wac.data, pc_config).data
        fb = extract_sift(fixed_image, sift)
        rb = transform_sift(fb)
        report["shared_runtime_seconds"][representation+"_wac_representation_features"] = perf_counter()-feature_tick
        for resolution, scale in scaled.items():
            tick = perf_counter()
            image = scale.data if representation == "intensity" else phase_congruency(scale.data, pc_config).data
            fa = extract_sift(image, sift)
            ra = transform_sift(fa)
            feature_time = perf_counter()-tick
            mutual = mutual_nearest_neighbors(ra.descriptors, rb.descriptors)
            choices = [("mutual", None, mutual_geometry_container(mutual))]
            for threshold in (.75, .8, .85, .9):
                choices.append(("ratio", threshold, filter_ratio(mutual.forward, threshold)))
                choices.append(("mutual_ratio", threshold, filter_mutual_ratio(mutual, threshold)))
            report["feature_runs"].append({"representation": representation, "resolution": resolution,
                "shape": list(image.shape), "factor": scale.nominal_factor,
                "tmc_keypoints": len(fa.keypoints), "wac_keypoints": len(fb.keypoints),
                "feature_runtime_seconds": feature_time, "bidirectional_matching_seconds": mutual.runtime_seconds})
            records = []
            for strategy, threshold, candidates in choices:
                record, result = evaluate(candidates, ra, rb, fixed_image.shape, resolution)
                record.update(representation=representation, resolution=resolution, strategy=strategy,
                    threshold=threshold, diagnostic=not(strategy == "ratio" and threshold == .75),
                    tmc_keypoints=len(fa.keypoints), wac_keypoints=len(fb.keypoints),
                    candidate_runtime_seconds=(mutual.runtime_seconds if strategy != "ratio" else mutual.forward.runtime_seconds)+candidates.runtime_seconds)
                record["runtime_seconds"] = record["candidate_runtime_seconds"]+record["ransac_runtime_seconds"]
                report["configurations"].append(record)
                records.append((record, candidates, result))
                if strategy == "ratio" and threshold == .75:
                    prior = baseline if representation == "intensity" else previous_pc
                    previous = next(s for s in prior["scales"] if s["target_resolution_m"] == resolution)
                    if record["candidate_count"] != previous["ratio_matches"]:
                        raise RuntimeError("Baseline 0.75 counts did not reproduce")
            best, candidates, geometry = max(records, key=lambda item: (item[0]["inliers"],
                item[0]["occupied_wac_cells"], -(item[0]["rmse"] if item[0]["rmse"] is not None else float("inf"))))
            folder = output / f"{representation}_{int(resolution)}m"
            folder.mkdir()
            display, fixed_display = prepare_sift_image(image), prepare_sift_image(fixed_image)
            draw_matches(folder / "selected_candidates.png", display, fixed_display, fa, fb, candidates.accepted)
            draw_matches(folder / "selected_inliers.png", display, fixed_display, fa, fb, geometry.inliers if geometry else ())
            best["visualization_directory"] = str(folder)
            print(json.dumps({k:v for k,v in best.items() if k != "candidates"}), flush=True)
    if any(digest(Path(p)) != value for p,value in hashes.items()):
        raise RuntimeError("Input or prior report content changed")
    report["total_runtime_seconds"] = perf_counter()-started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["representation", "resolution", "strategy", "threshold", "candidate_count", "inliers",
              "inlier_ratio", "rmse", "occupied_wac_cells", "distance_statistics", "residual_statistics",
              "transform", "warnings", "runtime_seconds"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in report["configurations"]:
            writer.writerow({key: json.dumps(record[key]) if isinstance(record[key], (list, dict)) else record[key] for key in fields})
    lines = ["# Candidate-generation diagnostics", "", "All residuals in WAC pixels. No ground-truth accuracy or registration claim.", "",
             "|Representation|m/px|Strategy|Ratio|Candidates|Inliers|Inlier fraction|RMSE|Cells /16|Warnings|",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in report["configurations"]:
        lines.append("|"+"|".join(str(r[k]) for k in ("representation", "resolution", "strategy", "threshold",
            "candidate_count", "inliers", "inlier_ratio", "rmse", "occupied_wac_cells", "warnings"))+"|")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
