"""Content validation for M21 deliverables, including independent saved-data checks."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from src.evaluation.experiments import ExperimentConfig
from src.evaluation.final_package import (EXCLUDED, check_protected, comparison, digest,
                                           point_rows, resolve, strict_json)
from src.geometry.registration import transform_points, warp_affine
from src.spatial.spatial_distribution import distribution_statistics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_metrics(record: dict) -> None:
    """Validate counts, ratios, spatial totals and saved reliability decisions."""
    m = record["metrics"]
    n, k = m["candidate_matches"], m["verified_inliers"]
    require(type(n) is int and type(k) is int and 0 <= k <= n, "Invalid candidate/inlier counts")
    ratio = m["inlier_ratio"]
    require((ratio is None if n == 0 else isinstance(ratio, (int, float)) and math.isclose(ratio, k/n, abs_tol=1e-12)), "Inlier ratio disagrees with counts")
    require(m["rmse"] is None or (math.isfinite(m["rmse"]) and m["rmse"] >= 0), "Invalid fitted residual RMSE")
    spatial = m["spatial_distribution"]
    counts = np.asarray(spatial["counts"])
    require(counts.shape == tuple(spatial["grid_shape"]) and np.isfinite(counts).all()
            and (counts >= 0).all() and (counts == counts.astype(int)).all(), "Invalid spatial grid")
    require(counts.sum() == k and counts.size == spatial["total_cells"], "Spatial support total mismatch")
    occupied = np.count_nonzero(counts)
    require(occupied == m["spatial_occupied_cells"] == spatial["occupied_cells"], "Spatial occupied count mismatch")
    require(math.isclose(m["spatial_occupancy_ratio"], occupied/counts.size, abs_tol=1e-12), "Spatial ratio mismatch")
    require(record["status"] == record["reliability"]["status"] == m["reliability"]["status"], "Reliability disagreement")
    if record["status"] == "RELIABLE":
        p = record["reliability"]["policy"]
        require(n >= p["minimum_candidates"] and k >= p["minimum_inliers"]
                and occupied >= p["minimum_occupied_cells"] and ratio >= p["minimum_inlier_ratio"]
                and m["maximum_residual"] <= p["maximum_residual"]
                and m["affine_diagnostics"]["condition_number"] <= p["maximum_condition_number"]
                and m["transform_available"] and m["rmse"] is not None, "Reliable record violates saved policy")
    else:
        require(bool(record["reliability"]["reasons"]), "Rejection reason required")


def read_points(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), reader.fieldnames
    require(fields is not None and fields[:4] == ["source_x", "source_y", "reference_x", "reference_y"], "Invalid point CSV header")
    for row in rows:
        require(set(row) == set(fields) and all(v is not None and math.isfinite(float(v)) for v in row.values()), "Invalid/nonfinite point CSV")
        if "inlier" in row:
            require(row["inlier"] in {"0", "1"}, "Invalid inlier membership")
    return rows, fields


def validate_headline(package: Path, record: dict) -> dict:
    """Recompute residual/spatial diagnostics and scientific raster resampling, no refit."""
    import rasterio
    a, m = record["final_artifacts"], record["metrics"]
    required = ("authoritative_points", "match_points", "verified_inlier_points", "registered_product",
                "moving_product", "reference_product", "registration_receipt", "metrics",
                "candidate_matches", "verified_inliers", "before_after", "overlay", "checkerboard")
    for key in required:
        require(key in a, f"Missing TMC deliverable: {key}")
        resolve(package, a[key]["path"])
    receipt = strict_json(resolve(package, a["registration_receipt"]["path"]))
    require(m["transform"] == receipt["affine_matrix"], "TMC transform differs from M8")
    require(m["candidate_matches"] == receipt["ransac_candidate_count"] and
            m["verified_inliers"] == receipt["ransac_inlier_count"], "TMC counts differ from M8")
    require(math.isclose(m["rmse"], receipt["post_transform_correspondence_residuals"]["rmse"], abs_tol=1e-9), "TMC RMSE differs from M8")
    with np.load(resolve(package, a["authoritative_points"]["path"]), allow_pickle=False) as saved:
        source, reference = saved["source"], saved["destination"]
        mask = saved["inlier_mask"]
        expected, fields = point_rows(source, reference, saved["confidence"], mask)
    rows, columns = read_points(resolve(package, a["match_points"]["path"]))
    require(columns == fields and len(rows) == len(expected) == m["candidate_matches"], "TMC candidate export mismatch")
    require(np.array_equal([[float(r[k]) for k in fields] for r in rows], [[float(r[k]) for k in fields] for r in expected]), "TMC point values changed")
    verified, _ = read_points(resolve(package, a["verified_inlier_points"]["path"]))
    require(verified == [r for r in rows if r["inlier"] == "1"] and len(verified) == m["verified_inliers"], "TMC inlier export mismatch")
    residual = transform_points(source[mask], np.asarray(m["transform"])) - reference[mask]
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    require(math.isclose(rmse, m["rmse"], rel_tol=1e-7, abs_tol=1e-7), "Recomputed TMC fitted residual RMSE mismatch")
    arrays = {}
    profiles = {}
    for key in ("moving_product", "reference_product", "registered_product"):
        path = resolve(package, a[key]["path"])
        with rasterio.open(path) as dataset:
            require(dataset.count == 1 and dataset.width * dataset.height <= 1_000_000, "Expected bounded scientific crop")
            data = dataset.read(1)
            valid = dataset.read_masks(1) > 0
            require(valid.any() and np.isfinite(data[valid]).all(), "Raster lacks finite valid pixels")
            arrays[key] = (data, valid)
            profiles[key] = (dataset.crs, dataset.transform, dataset.shape, dataset.dtypes)
    moving, mask_source = arrays["moving_product"]
    fixed, _ = arrays["reference_product"]
    registered, registered_mask = arrays["registered_product"]
    require(profiles["registered_product"][:3] == profiles["reference_product"][:3], "Registered raster grid/CRS mismatch")
    require(registered.dtype == np.float32 and np.isnan(registered[~registered_mask]).all(), "Registered data must preserve float32/nodata")
    replay = warp_affine(moving.astype(np.float32), np.asarray(m["transform"]), fixed.shape, validity_mask=mask_source)
    require(np.array_equal(replay.valid_mask, registered_mask) and np.array_equal(replay.image, registered, equal_nan=True), "Registered pixels differ from saved-transform resampling")
    require(int(registered_mask.sum()) == receipt["registered_valid_pixels"], "Registered coverage differs from M8")
    spatial = distribution_statistics(reference[mask], fixed.shape)
    require(spatial["counts"] == m["spatial_distribution"]["counts"], "Recomputed TMC spatial support differs")
    for key, old in (("moving_product", "source_image"), ("reference_product", "reference_image"), ("authoritative_points", "correspondence_source")):
        suffix = receipt[old].replace("\\", "/")
        hashes = [v for k, v in receipt["input_sha256"].items() if suffix.endswith(k.replace("\\", "/"))]
        require(len(hashes) == 1 and digest(resolve(package, a[key]["path"])) == hashes[0], "Prepared evidence differs from M8 input digest")
    return {key: "PASS" for key in ("SOFTWARE RESULT RECORD", "CORRESPONDING MATCH POINTS", "REGISTERED PRODUCT",
        "RMSE", "INLIER MATCH COUNT", "INLIER RATIO", "SPATIAL COVERAGE", "RELIABILITY STATUS", "PROVENANCE")}


def compare_packages(baseline: Path, reproduced: Path) -> dict:
    """Compare scientific records and exported products, excluding run bookkeeping.

    Validate both sealed packages first. Exact byte comparisons cover the saved
    rasters, coordinates and figures; no numerical tolerance hides changed evidence.
    """
    validate_package(baseline)
    validate_package(reproduced)
    old = {r["name"]: r for r in strict_json(baseline / "experiments.json")["experiments"]}
    new = {r["name"]: r for r in strict_json(reproduced / "experiments.json")["experiments"]}
    require(old.keys() == new.keys(), "Reproduction experiment matrix changed")
    stable = ("experiment_id", "configuration", "status", "reliability", "metrics", "parameters",
              "metadata", "registration_status", "coordinate_frame", "evidence_mode")
    compared = 0
    for name, before in old.items():
        after = new[name]
        for field in stable:
            require(before[field] == after[field], f"Reproduction changed {name}: {field}")
        require(before["timing"]["historical_seconds"] == after["timing"]["historical_seconds"],
                f"Historical runtime changed: {name}")
        require(before["final_artifacts"].keys() == after["final_artifacts"].keys(), f"Artifact set changed: {name}")
        for key, artifact in before["final_artifacts"].items():
            require(digest(resolve(baseline, artifact["path"])) ==
                    digest(resolve(reproduced, after["final_artifacts"][key]["path"])),
                    f"Reproduction artifact changed: {name}/{key}")
            compared += 1
    require((baseline / "comparison.csv").read_bytes() == (reproduced / "comparison.csv").read_bytes(),
            "Reproduction comparison CSV changed")
    return dict(status="PASS", experiment_count=len(old), artifacts_byte_identical=compared,
                baseline=str(baseline), reproduced=str(reproduced),
                expected_differences=["Execution/run IDs and timestamps", "Import/event/packaging durations",
                    "Repository revision and absolute provenance paths", "Protection inventory of existing local files",
                    "Added prepared-input manifest", "Manifest and ZIP hashes reflecting those changes"])


def validate_package(package: Path, *, root: Path | None = None, sealed: bool = True) -> dict:
    """Fail closed on inconsistent evidence. root additionally audits protected originals."""
    package = package.resolve()
    for filename in ("experiments.json", "run_config.json", "summary.json", "summary.txt", "comparison.csv", "protection.json"):
        resolve(package, filename)
    for path in package.rglob("*.json"):
        strict_json(path)
    records = strict_json(package / "experiments.json")["experiments"]
    configs = strict_json(package / "run_config.json")["experiments"]
    require(len(records) == len(configs) > 0 and len({r["experiment_id"] for r in records}) == len(records), "Experiment collection incomplete/duplicated")
    require({r["name"] for r in records} == {c["name"] for c in configs}, "Final matrix differs from configuration")
    m20 = {r["name"]: r for r in strict_json(package / "evidence/m20.json")["experiments"]}
    m18 = {r["name"]: r for r in strict_json(package / "evidence/m18.json")["experiments"]}
    require(set(m20) == {r["name"] for r in records}, "Final matrix omits established M20 comparisons")
    if root:
        require(strict_json(root / "configs/milestone_20.json") == strict_json(package / "run_config.json"), "Configuration differs from authoritative M20")
        check_protected(root, strict_json(package / "protection.json")["files"])
    if (package / "prepared_inputs.json").is_file():
        inputs = strict_json(package / "prepared_inputs.json")
        require(inputs["schema_version"] == 1 and
                {p["sensor"] for p in inputs["pairs"]} == {r["parameters"]["config"]["source_sensor"] for r in records},
                "Prepared-input sensor inventory differs")
        identities = inputs["evidence"] + [i for p in inputs["pairs"] for i in p["inputs"]]
        if root:
            check_protected(root, {i["path"]: i for i in identities})
    for r in records:
        validate_metrics(r)
        require(r == strict_json(resolve(package, r["record_path"])), "Per-experiment record disagreement")
        old = m20[r["name"]]
        require(r["configuration"] in [ExperimentConfig.from_dict(c).to_dict() for c in configs] and r["metrics"] == old["metrics"] and r["parameters"] == old["parameters"]
                and r["metadata"] == old["metadata"] and r["status"] == old["status"], "Experiment disagrees with authoritative M20")
        prior = m18[r["configuration"]["experiment_name"]]
        require(r["metrics"] == prior["metrics"], "Experiment differs from M18")
        require(r["evidence_mode"] == "imported_historical_evidence" and r["mode"] == "import"
                and bool(r["provenance"]["report_sha256"]) and bool(r["execution_id"]), "Missing or misleading import provenance")
        if root:
            require(digest(Path(r["provenance"]["report_path"])) == r["provenance"]["report_sha256"], "Source report digest mismatch")
        for artifact in r["final_artifacts"].values():
            path = resolve(package, artifact["path"])
            if path.suffix == ".png":
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
        a = r["final_artifacts"]
        exported = strict_json(resolve(package, a["metrics"]["path"]))
        require(all(exported[k] == v for k, v in r["metrics"].items()), "Metrics product mismatch")
        rows, fields = read_points(resolve(package, a["match_points"]["path"]))
        require(len(rows) == r["metrics"]["candidate_matches"], "Point count disagrees with metrics")
        if r["name"] != "TMC saved LoFTR":
            require("inlier" not in fields and "confidence" not in fields, "Fabricated unavailable per-point fields")
            expected, _ = point_rows(r["metadata"].get("source_points_original", []), r["metadata"].get("reference_points_original", []))
            require([[float(row[f]) for f in fields] for row in rows] == [[float(row[f]) for f in fields] for row in expected], "Candidate coordinates changed")
        if r["status"] == "RELIABLE":
            require(r["registration_status"] == "AVAILABLE" and "registered_product" in a, "Reliable result missing registered product")
        else:
            require(r["registration_status"] == "WITHHELD" and not any(k in a for k in ("registered_product", "overlay", "checkerboard")), "Rejected result claims registration")
            withheld = strict_json(resolve(package, a["registration_withheld"]["path"]))
            require(withheld == {"registration_status": "WITHHELD", "reliability": r["reliability"], "metrics": r["metrics"]}, "Withheld receipt mismatch")
    expected_csv = io.StringIO(newline="")
    expected_rows = comparison(records)
    writer = csv.DictWriter(expected_csv, fieldnames=list(expected_rows[0]))
    writer.writeheader()
    writer.writerows(expected_rows)
    with (package / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        actual = list(csv.DictReader(stream))
    require(actual == list(csv.DictReader(io.StringIO(expected_csv.getvalue()))), "Comparison CSV disagreement")
    summary = strict_json(package / "summary.json")
    reliable = sum(r["status"] == "RELIABLE" for r in records)
    require(summary["experiment_count"] == len(records) and summary["reliable_experiment_count"] == reliable
            and summary["withheld_experiment_count"] == len(records)-reliable
            and summary["imported_historical_evidence_count"] == len(records)
            and summary["freshly_executed_count"] == 0 and summary["failures"] == 0, "Summary counts disagree")
    headlines = [r for r in records if r["experiment_id"] == summary["headline_experiment"]]
    require(len(headlines) == 1 and headlines[0]["name"] == "TMC saved LoFTR", "Missing headline TMC")
    require(summary["metrics"] == headlines[0]["metrics"] and summary["artifact_references"] == headlines[0]["final_artifacts"], "Headline summary mismatch")
    checks = validate_headline(package, headlines[0])
    if sealed:
        for filename in ("validation_report.json", "validation_report.txt", "manifest.json", "ASTRION_Final_Deliverable.zip"):
            resolve(package, filename)
        manifest = strict_json(package / "manifest.json")
        inventory = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file() and p.name not in EXCLUDED}
        require(len(manifest["files"]) == len(inventory) and {e["path"] for e in manifest["files"]} == inventory, "Manifest inventory differs")
        for entry in manifest["files"]:
            path = resolve(package, entry["path"])
            require(path.stat().st_size == entry["size"] and digest(path) == entry["sha256"], f"Manifest hash mismatch: {entry['path']}")
        with zipfile.ZipFile(package / "ASTRION_Final_Deliverable.zip") as archive:
            expected_files = inventory | {"manifest.json"}
            require(len(archive.namelist()) == len(expected_files) and set(archive.namelist()) == expected_files and archive.testzip() is None, "ZIP inventory/CRC invalid")
            for name in archive.namelist():
                require(archive.read(name) == resolve(package, name).read_bytes(), f"ZIP content mismatch: {name}")
    return dict(status="PASS", experiment_count=len(records), reliable_count=reliable,
                withheld_count=len(records)-reliable, failures=0, sih_deliverables=checks,
                protected_originals="PASS (size/mtime for large files; SHA256 for files <=8 MB)" if root else "NOT CHECKED: repository unavailable",
                content_validation="Records, point values, rasters, residuals, spatial support, CSV and summaries verified",
                seal_validation="Manifest and ZIP verified" if sealed else "Content audit before manifest/ZIP sealing")
