"""Real Tycho LoFTR experiment: python -m scripts.tycho_loftr. No registration."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import rasterio
import torch
from scripts.real_tmc_wac import save_png
from scripts.tycho_phase_congruency import digest
from scripts.tycho_candidate_generation import geometry_diagnostics
from src.preprocessing.preprocessing import preprocess_image, PreprocessingConfig
from src.preprocessing.scale_normalization import normalize_resolution
from src.features.learned import LoFTRMatcher
from src.features.sift import prepare_sift_image
from src.matching.ratio_test import RatioMatch, RatioTestResult
from src.geometry.ransac import verify_affine, RANSACConfig


def draw(path, a, b, matches) -> None:
    """At most 60 evenly spaced output rows, with no manual point selection."""
    left = [cv2.KeyPoint(float(x), float(y), 1) for x, y in matches.source]
    right = [cv2.KeyPoint(float(x), float(y), 1) for x, y in matches.destination]
    indices = np.linspace(0, len(left)-1, min(len(left), 60), dtype=int)
    lines = [cv2.DMatch(int(i), int(i), 0.) for i in indices]
    save_png(path, cv2.drawMatches(a, left, b, right, lines, None,
        matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))


def main() -> None:
    """Fixed four-scale/four-confidence inference; failures have null metrics."""
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "results/experiment_tycho_real_multiscale/report.json"
    baseline = json.loads(baseline_path.read_text())
    moving, fixed = Path(baseline["moving"]), Path(baseline["fixed"])
    checkpoint = root / "models/loftr_outdoor_kornia.ckpt"
    output = root / "results/experiment_tycho_loftr"
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): digest(p) for p in (moving, fixed, baseline_path, checkpoint)}
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = PreprocessingConfig(**baseline["preprocessing"])
    tmc = preprocess_image(np.load(moving, mmap_mode="r"), config)
    with rasterio.open(fixed) as dataset:
        wac = preprocess_image(dataset.read(1), config)
    init_start = perf_counter()
    matcher = LoFTRMatcher(checkpoint)
    report = {"sha256": hashes, "model": matcher.info, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_runtime": torch.version.cuda, "preprocessing": asdict(config),
        "ransac": asdict(RANSACConfig()), "confidence_thresholds": [.2, .4, .6, .8],
        "checkpoint_url": "https://huggingface.co/kornia/loftr/resolve/main/loftr_outdoor.ckpt",
        "initialization_seconds": perf_counter()-init_start,
        "initial_environment": {"torch_installed": False, "torchvision_installed": False, "kornia_installed": False},
        "torchvision": "not installed; not required by Kornia LoFTR",
        "shared_preprocessing_seconds": tmc.runtime_seconds+wac.runtime_seconds,
        "warnings": ["Outdoor MegaDepth weights are not trained on lunar data.",
            "LoFTR confidence is not a calibrated probability of lunar correspondence correctness.",
            "Raw counts follow the unchanged built-in coarse 0.20 gate.",
            "WAC nodata remains black, as in the intensity baseline.",
            "Fitted residuals do not establish ground-truth accuracy."], "scales": []}
    for resolution in (75., 100., 125., 150.):
        tick = perf_counter()
        scale = normalize_resolution(tmc.data, 5.15, resolution)
        raw = matcher.match(scale.data, wac.data)
        record = {"resolution": resolution, "factor": scale.nominal_factor,
            "shape": list(scale.data.shape), "wac_shape": list(wac.data.shape),
            "raw_matches": len(raw.source), "inference": raw.metadata, "configurations": []}
        folder = output / f"{int(resolution)}m"
        folder.mkdir()
        a, b = prepare_sift_image(scale.data), prepare_sift_image(wac.data)
        save_png(folder / "tmc_scaled.png", a)
        save_png(folder / "wac.png", b)
        draw(folder / "raw_matches.png", a, b, raw)
        np.savez(folder / "raw_matches.npz", source=raw.source, destination=raw.destination, confidence=raw.confidence)
        visualizations = {}
        for threshold in (.2, .4, .6, .8):
            filtered = raw.filter_confidence(threshold)
            # Legacy RANSAC container adapter: only row indices are consumed.
            # Descriptor distances/ratio do not exist for LoFTR; explicitly None.
            pairs = tuple(RatioMatch(i, i, None, None, None) for i in range(len(filtered.source)))
            candidates = RatioTestResult(pairs, len(pairs), 0, 0, 0, None, 0.)
            geometry, failure = None, None
            geom_tick = perf_counter()
            try:
                geometry = verify_affine(candidates, filtered.source, filtered.destination, RANSACConfig())
            except (ValueError, RuntimeError) as exc:
                failure = str(exc)
            row = {"confidence_threshold": threshold, "filtered_matches": len(pairs),
                "inliers": geometry.inlier_count if geometry else 0,
                "inlier_ratio": geometry.inlier_ratio if geometry else None,
                "transform": geometry.matrix.tolist() if geometry else None,
                "residual_statistics": geometry.inlier_error_statistics if geometry else None,
                "rmse": None, "failure_reason": failure, "occupied_wac_cells": 0,
                "unique_source_positions": None, "unique_inlier_destination_positions": None,
                "determinant": None, "linear_singular_values": None, "linear_condition_number": None,
                "ransac_runtime_seconds": perf_counter()-geom_tick, "warnings": [failure] if failure else []}
            if geometry:
                row.update(geometry_diagnostics(geometry, filtered.destination, wac.data.shape, resolution))
                row["unique_source_positions"] = len(np.unique(filtered.source[geometry.inlier_mask], axis=0))
                if row["unique_source_positions"] < 8:
                    row["warnings"].append("fewer_than_8_unique_source_positions")
                if row["linear_condition_number"] > 100 or min(row["singular_values_relative_nominal"]) < .01:
                    row["warnings"].append("near_singular_linear_transform")
                row["inlier_indices_in_filtered"] = np.flatnonzero(geometry.inlier_mask).tolist()
            record["configurations"].append(row)
            from src.features.learned import LearnedMatches
            keep = geometry.inlier_mask if geometry else np.zeros(len(pairs), bool)
            signature = (len(pairs), tuple(np.flatnonzero(keep)))
            if signature not in visualizations:
                visualizations[signature] = f"{threshold:.2f}"
                draw(folder / f"confidence_{threshold:.2f}.png", a, b, filtered)
                draw(folder / f"inliers_{threshold:.2f}.png", a, b,
                     LearnedMatches(filtered.source[keep], filtered.destination[keep], filtered.confidence[keep], {}))
            row["visualization_threshold_suffix"] = visualizations[signature]
        record["runtime_seconds"] = perf_counter()-tick
        report["scales"].append(record)
        print(json.dumps(record), flush=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise RuntimeError("Source or checkpoint changed")
    report["total_runtime_seconds"] = perf_counter()-started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
