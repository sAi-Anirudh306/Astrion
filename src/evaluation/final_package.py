"""M21 packaging of the complete M20 import matrix; no new scientific fitting."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
import uuid
import zipfile

import numpy as np

from src.evaluation.experiments import ExperimentRunner, load_configs, write_json, json_safe
from src.ui.service import DEMOS, historical_artifacts

LIMITATIONS = [
    "Fitted residual RMSE is not absolute geographic accuracy or ground-truth positional error.",
    "Approximate TMC footprint mapping is not precise orthorectification.",
    "Three affine-defining points provide no redundant validation.",
    "Physical scale normalization does not guarantee modality invariance.",
    "Phase Congruency does not guarantee correspondence.",
    "Pretrained LoFTR has terrestrial-to-lunar domain shift; inference was not repeated.",
    "LRO WAC is an external lunar reference; one TMC success does not prove universal sensor invariance.",
    "Insufficient-support cases are deliberately rejected; IIRS values are RAW DN, not calibrated radiance.",
    "OHRC native sampling is 0.26 m versus WAC 100 m: severe scale gap.",
    "Missing historical per-point inlier masks are not reconstructed or fabricated.",
    "Large protected files are checked by size/mtime, not rescanned for a new full-content digest.",
]
EXCLUDED = {"manifest.json", "ASTRION_Final_Deliverable.zip"}


def digest(path: Path) -> str:
    """Stream a SHA256 digest without loading a complete file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(path: Path) -> dict:
    """Reject nonfinite numbers (including exponent overflow) and duplicate keys."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=constant, object_pairs_hook=pairs)
    if json_safe(value) != value:
        raise ValueError("Nonfinite JSON number")
    return value


def resolve(package: Path, relative: str) -> Path:
    """Resolve only package-local artifacts, including protection against symlinks."""
    p = Path(relative)
    target = (package / p).resolve()
    if p.is_absolute() or p.drive or ".." in p.parts or not target.is_relative_to(package.resolve()):
        raise ValueError(f"Artifact escapes package: {relative}")
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError(f"Missing/empty artifact: {relative}")
    return target


def protected_snapshot(root: Path, output: Path) -> dict:
    """Inventory originals, prepared products and existing results without giant scans."""
    records = {}
    for area in (root / "data", root / "results"):
        for path in sorted(area.rglob("*")):
            if (not path.is_file() or path.resolve().is_relative_to(output.resolve())
                    or "edge-profile" in path.parts or "__pycache__" in path.parts):
                continue
            stat = path.stat()
            records[path.relative_to(root).as_posix()] = {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": digest(path) if stat.st_size <= 8_000_000 else None,
            }
    return records


def check_protected(root: Path, records: dict) -> None:
    """Check every previously inventoried file; new UI runs are permitted."""
    for relative, expected in records.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Protected file missing or outside repository: {relative}")
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Protected file changed: {relative}")
        if expected["sha256"] and digest(path) != expected["sha256"]:
            raise ValueError(f"Protected checksum changed: {relative}")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write a new CSV, including a header for an explicitly empty point set."""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_rows(source, reference, confidence=None, inliers=None) -> tuple[list[dict], list[str]]:
    """Export only available fields; never infer individual inlier membership."""
    source, reference = np.asarray(source, float).reshape(-1, 2), np.asarray(reference, float).reshape(-1, 2)
    if source.shape != reference.shape or not np.isfinite([source, reference]).all():
        raise ValueError("Invalid corresponding coordinates")
    fields = ["source_x", "source_y", "reference_x", "reference_y"]
    arrays = [source[:, 0], source[:, 1], reference[:, 0], reference[:, 1]]
    for name, values in (("confidence", confidence), ("inlier", inliers)):
        if values is not None:
            values = np.asarray(values)
            if values.shape != (len(source),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name}")
            if name == "inlier" and not np.isin(values, [0, 1]).all():
                raise ValueError("Invalid inlier mask")
            fields.append(name)
            arrays.append(values.astype(int) if name == "inlier" else values)
    return json_safe([dict(zip(fields, row)) for row in zip(*arrays)]), fields


def comparison(records: list[dict]) -> list[dict]:
    """One stable comparison row per configured experiment."""
    rows = []
    for r in records:
        m, p = r["metrics"], r["parameters"]
        rows.append(dict(experiment_id=r["experiment_id"], name=r["name"],
            source_sensor=p["config"]["source_sensor"], reference_sensor=p["config"]["reference_sensor"],
            representation=p["config"]["representation"], method=p["feature"] or p["matcher"],
            candidate_matches=m["candidate_matches"], inlier_matches=m["verified_inliers"],
            inlier_ratio=m["inlier_ratio"], fitted_rmse_px=m["rmse"],
            spatial_coverage=m["spatial_occupancy_ratio"], reliability=r["status"],
            registration_status=r["registration_status"], runtime=r["timing"]["historical_seconds"],
            evidence_mode=r["evidence_mode"]))
    return rows


def plot_points(path: Path, source, reference, title: str) -> None:
    """Paired coordinate plots; numbers identify actual corresponding points."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(fig)
    for ax, points, label in zip(fig.subplots(1, 2), (source, reference), ("Source", "Reference")):
        xy = np.asarray(points).reshape(-1, 2)
        ax.scatter(xy[:, 0], xy[:, 1], s=10)
        if len(xy) <= 30:
            for n, (x, y) in enumerate(xy):
                ax.annotate(str(n + 1), (x, y))
        ax.set(title=label, xlabel="x / column", ylabel="y / row")
        ax.invert_yaxis()
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    fig.clear()


def package_experiment(root: Path, output: Path, result: dict) -> dict:
    """Preserve M20 contract and add portable final deliverable references."""
    r = json_safe(result)
    sensor = r["parameters"]["config"]["source_sensor"].lower()
    folder = output / f"{sensor}_wac" / r["name"].lower().replace(" ", "_")
    folder.mkdir(parents=True)
    r.update(evidence_mode="imported_historical_evidence", registration_status=(
        "AVAILABLE" if r["status"] == "RELIABLE" else "WITHHELD"), final_artifacts={},
        runtime_scope="M18 recorded classical execution or saved-support replay; not full raw processing or LoFTR inference")
    def link(key, path, origin):
        r["final_artifacts"][key] = {"path": path.relative_to(output).as_posix(), "origin": origin}
    def copy(key, path, filename, origin):
        target = folder / filename
        shutil.copyfile(path, target)
        link(key, target, origin)
    m = r["metrics"]
    if sensor == "tmc":
        receipt = strict_json(root / "results/milestone_08_registration/registration_metadata.json")
        r["source_product"], r["reference_product"] = receipt["source_image"], receipt["reference_image"]
    else:
        receipt = strict_json(output / "evidence" / f"{sensor}_preparation.json")
        r["source_product"] = (receipt["product_stem"] if sensor == "ohrc" else receipt["product"]["product_id"])
        r["reference_product"] = receipt["reference_crop" if sensor == "ohrc" else "reference"]["output_path"]
        r["preparation_evidence"] = f"evidence/{sensor}_preparation.json"
    if sensor == "iirs":
        r["scientific_values"] = "RAW DN; predetermined spectral reduction, not calibrated radiance"
        r["spectral_band_indices_zero_based"] = r["metadata"]["spectral_representation"]
    if r["name"] == "TMC saved LoFTR":
        path = root / "results/experiment_tycho_map_projected/loftr_0.20_matches.npz"
        with np.load(path, allow_pickle=False) as points:
            source, reference = points["source"], points["destination"]
            rows, fields = point_rows(source, reference, points["confidence"], points["inlier_mask"])
        copy("authoritative_points", path, "authoritative_matches.npz", "Saved M19/M8 point evidence")
        write_csv(folder / "verified_inlier_points.csv", [row for row in rows if row["inlier"]], fields)
        link("verified_inlier_points", folder / "verified_inlier_points.csv", "Saved inlier mask; no refit")
        r["coordinate_frame"] = "Zero-based (x=column,y=row) prepared TMC/WAC 100 m grids"
    else:
        source = r["metadata"].get("source_points_original", [])
        reference = r["metadata"].get("reference_points_original", [])
        rows, fields = point_rows(source, reference)
        r["coordinate_frame"] = "Zero-based original prepared crop grids, before common-GSD scaling; not full raw strip coordinates"
        r["point_limitation"] = "No saved per-point inlier mask or confidence; omitted. OHRC has zero accepted candidates."
    if len(rows) != m["candidate_matches"]:
        raise ValueError(f"Candidate evidence mismatch: {r['name']}")
    write_csv(folder / "match_points.csv", rows, fields)
    link("match_points", folder / "match_points.csv", "Recorded candidate coordinates; header-only if zero candidates")
    plot_points(folder / "candidate_coordinates.png", source, reference, r["name"] + " recorded candidate coordinates")
    link("candidate_coordinates", folder / "candidate_coordinates.png", "Regenerated plot of saved coordinates")
    demo = next((d for d in DEMOS if d.experiment == r["name"]), None)
    if demo:
        for artifact in historical_artifacts(root, demo, r):
            key = artifact.title.lower().replace(" / ", "_").replace(" ", "_")
            copy(key, artifact.path, key + artifact.path.suffix, artifact.origin)
    if r["status"] == "RELIABLE":
        if r["name"] != "TMC saved LoFTR":
            raise ValueError("Unexpected reliable experiment: explicit product association required")
        for key, relative in {
            "registered_product": "data/processed/tycho/registered/tmc_registered_to_wac.tif",
            "moving_product": "data/processed/tycho/map_projected/tmc_on_wac_grid.tif",
            "reference_product": "data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.tif",
        }.items():
            copy(key, root / relative, key + ".tif", "Imported scientific raster; unmodified pixels")
        copy("registration_receipt", root / "results/milestone_08_registration/registration_metadata.json",
             "registration_receipt.json", "Authoritative M8 registration receipt")
    else:
        write_json(folder / "registration_withheld.json", {
            "registration_status": "WITHHELD", "reliability": r["reliability"], "metrics": m})
        link("registration_withheld", folder / "registration_withheld.json", "Preserved M18 rejection")
    write_json(folder / "metrics.json", {"rmse_label": "fitted residual RMSE (reference processing pixels)", **m})
    link("metrics", folder / "metrics.json", "Imported authoritative M18 metrics")
    r["record_path"] = (folder / "result.json").relative_to(output).as_posix()
    write_json(folder / "result.json", r)
    (folder / "README.txt").write_text("\n".join([
        r["name"], r["status"], "Registration: " + r["registration_status"],
        *r["reliability"]["reasons"], r["coordinate_frame"], r.get("point_limitation", ""),
        "All scientific experiment evidence is imported. Figures/CSV are packaging operations.",
        *[f"{key}: {Path(value['path']).name} ({value['origin']})" for key, value in r["final_artifacts"].items()],
        *LIMITATIONS]) + "\n", encoding="utf-8")
    return r


def create_manifest(output: Path, records: list[dict]) -> dict:
    """Inventory every payload file; explicitly exclude self and enclosing ZIP."""
    associations = {a["path"]: (r["experiment_id"], key, a["origin"])
                    for r in records for key, a in r["final_artifacts"].items()}
    entries = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED:
            continue
        relative = path.relative_to(output).as_posix()
        exp, kind, description = associations.get(relative, (None, path.suffix.lstrip("."), path.stem.replace("_", " ")))
        entries.append(dict(path=relative, artifact_type=kind, experiment_id=exp,
                            description=description, size=path.stat().st_size, sha256=digest(path)))
    return dict(schema_version=1, files=entries, exclusions=sorted(EXCLUDED),
                note="Manifest cannot hash itself; ZIP includes manifest and all payload files.")


def export_zip(output: Path) -> None:
    """Stable ordering, timestamps and permissions for a fixed package snapshot."""
    target = output / "ASTRION_Final_Deliverable.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != target:
                info = zipfile.ZipInfo(path.relative_to(output).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())


def prepared_inputs(root: Path, records: list[dict]) -> dict:
    """Describe the exact prepared inputs and original lineage without recropping.

    Historical OHRC/IIRS source arrays were not persisted. Their preparation
    receipts and saved M18 evidence are explicit inputs, never reconstructed
    from display images. Large originals are inventoried without rereading them.
    """
    root = root.resolve()

    def identity(value: str | Path, role: str) -> dict:
        # Historical receipts contain workstation-absolute paths. Rebase only
        # known repository data/results suffixes, not arbitrary external paths.
        parts = str(value).replace("\\", "/").split("/")
        start = next((i for i, part in enumerate(parts) if part in {"data", "results", "configs"}), None)
        if start is None:
            raise ValueError(f"Unrecognized prepared input path: {value}")
        relative = Path(*parts[start:])
        path = resolve(root, relative.as_posix())
        stat = path.stat()
        small = stat.st_size <= 8_000_000
        return dict(path=relative.as_posix(), role=role, size=stat.st_size,
                    sha256=digest(path) if small else None,
                    mtime_ns=stat.st_mtime_ns,
                    assurance="SHA256" if small else "Size/mtime; large original or intermediate not rehashed")

    def receipt(relative: str) -> dict:
        return json.loads((root / relative).read_text(encoding="utf-8"))

    tmc_path = "data/processed/tycho/ch2_tmc_nrf_20211122T2123225722_d_img_d18_lat_-44_-42.6.json"
    map_path = "data/processed/tycho/map_projected/tmc_on_wac_grid_metadata.json"
    ref_path = "data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.json"
    tmc, mapping, reference = receipt(tmc_path), receipt(map_path), receipt(ref_path)
    groups = [dict(sensor="TMC", source_product=tmc["source_metadata"]["logical_identifier"],
        source_window=tmc["source_metadata"]["subset"], reference_window=reference["window"],
        native_gsd_m=tmc["source_metadata"]["pixel_resolution_m_per_pixel"],
        prepared_gsd_m=mapping["resolution"], reference_gsd_m=reference["resolution"],
        processing=mapping["processing_policy"],
        inputs=[identity(value, role) for value, role in (
            (tmc["source_image"], "original sensor product"), (tmc["source_label"], "original PDS4 label"),
            (reference["wac_source"], "original LRO reference"),
            (tmc["outputs"]["scientific"], "historical native scientific crop"),
            ("data/processed/tycho/map_projected/tmc_on_wac_grid.tif", "prepared scientific input"),
            (reference["output_tif"], "prepared WAC input"),
            (tmc_path, "crop provenance"), (map_path, "mapping provenance"), (ref_path, "reference provenance"))])]
    for sensor, relative, paths_key, selection_key, reference_key in (
            ("OHRC", "results/milestone_16_ohrc/ohrc.json", "paths", "selected_crop", "reference_crop"),
            ("IIRS", "results/milestone_17_iirs/iirs.json", "source_paths", "selected_region", "reference")):
        prior = receipt(relative)
        ref = prior[reference_key]
        selection = prior[selection_key]
        window = selection["subset"] if sensor == "OHRC" else selection
        group = dict(sensor=sensor,
            source_product=prior["product_stem"] if sensor == "OHRC" else prior["product"]["product_id"],
            source_window=window, reference_window=ref["window"],
            native_gsd_m=(prior["metadata"]["pixel_resolution_m_per_pixel"] if sensor == "OHRC"
                          else prior["product"]["native_gsd_m"]), reference_gsd_m=ref["resolution"],
            processing="Saved M16/M17 preparation and M18 correspondence evidence; no new scientific execution",
            source_array_policy="Native prepared source arrays were not persisted; display PNGs are not scientific inputs",
            inputs=[identity(value, "original " + key) for key, value in prior[paths_key].items()]
                + [identity(ref["source_path"], "original LRO reference"),
                   identity(ref["output_path"], "prepared WAC input"), identity(relative, "preparation provenance")])
        if sensor == "IIRS":
            group["spectral_reductions"] = [{k: rep[k] for k in ("name", "band_indices", "official_band_numbers", "wavelengths")}
                                             for rep in prior["representations"]]
            group["scientific_values"] = "RAW DN, not calibrated radiance"
        groups.append(group)
    evidence_paths = {"configs/milestone_20.json", "results/milestone_20_experiments/experiments.json",
        "results/milestone_08_registration/registration_metadata.json",
        "results/experiment_tycho_map_projected/loftr_0.20_matches.npz",
        "data/processed/tycho/registered/tmc_registered_to_wac.tif"}
    evidence_paths.update(receipt("results/milestone_08_registration/registration_metadata.json")["input_sha256"])
    for record in records:
        evidence_paths.add(record["configuration"]["report_path"])
        evidence_paths.update(p["path"] for p in (record["provenance"].get("prior_reports") or {}).values())
        evidence_paths.update(a["path"] for a in record.get("artifacts", []))
        demo = next((d for d in DEMOS if d.experiment == record["name"]), None)
        if demo:
            evidence_paths.update(str(a.path) for a in historical_artifacts(root, demo, record))
    return dict(schema_version=1, mode="validated_prepared_inputs_and_historical_evidence",
                note="Logical input set; no duplicated imagery, region search, threshold tuning or inference",
                pairs=groups, evidence=[identity(p, "required saved evidence or final product") for p in sorted(evidence_paths)])


def build_package(root: Path, output: Path) -> dict:
    """Run the complete established import matrix into a fresh results directory."""
    from src.evaluation.final_validation import validate_package
    started = perf_counter()
    root, output = root.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(root / "results"):
        raise ValueError("Choose a fresh directory under results; existing packages are never overwritten")
    config_path = root / "configs/milestone_20.json"
    configs = load_configs(config_path)
    if any(c.mode != "import" for c in configs):
        raise ValueError("Final prepared matrix must preserve M20 import architecture")
    protected = protected_snapshot(root, output)
    output.mkdir(parents=True)
    evidence = output / "evidence"
    evidence.mkdir()
    shutil.copyfile(config_path, output / "run_config.json")
    # Historical reports may contain NaN; preserve originals and sanitize copies.
    for name, path in (("m18.json", root / configs[0].report_path),
                       ("m20.json", root / "results/milestone_20_experiments/experiments.json"),
                       ("ohrc_preparation.json", root / "results/milestone_16_ohrc/ohrc.json"),
                       ("iirs_preparation.json", root / "results/milestone_17_iirs/iirs.json")):
        write_json(evidence / name, json.loads(path.read_text(encoding="utf-8")))
    runner = ExperimentRunner(root)
    records = [package_experiment(root, output, runner.run(c)) for c in configs]
    write_json(output / "prepared_inputs.json", prepared_inputs(root, records))
    write_json(output / "experiments.json", dict(schema_version=1, experiments=records))
    write_csv(output / "comparison.csv", comparison(records))
    headline = next(r for r in records if r["name"] == "TMC saved LoFTR")
    reliable = sum(r["status"] == "RELIABLE" for r in records)
    summary = dict(project="ASTRION / LunarMatch", milestone=21, run_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(), experiment_count=len(records),
        freshly_executed_count=0, deterministically_regenerated_experiment_count=0,
        imported_historical_evidence_count=len(records), reliable_experiment_count=reliable,
        withheld_experiment_count=len(records)-reliable, failures=0,
        sensor_pairs=sorted({f"{r['parameters']['config']['source_sensor']}-WAC" for r in records}),
        headline_experiment=headline["experiment_id"], metrics=headline["metrics"],
        artifact_references=headline["final_artifacts"], validation_state="PASS",
        runtime_seconds=perf_counter()-started, runtime_scope="Packaging and import before serialization/final audit",
        software=dict(python=platform.python_version(), numpy=np.__version__,
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()),
        limitations=LIMITATIONS)
    write_json(output / "summary.json", summary)
    lines = ["ASTRION final validation / SIH26166", f"{len(records)} imported experiments; {reliable} reliable; {len(records)-reliable} withheld.",
        "Pairs: " + ", ".join(summary["sensor_pairs"]), "No new matching/inference, band search, or threshold tuning.",
        "Headline: TMC saved LoFTR", f"Metrics: {json.dumps(headline['metrics'])}",
        "Seven nonempty candidate CSVs; two zero-candidate OHRC header-only CSVs. TMC saved inlier CSV included.",
        "Registered scientific float32 GeoTIFF, source/reference crops and presentation comparisons in tmc_wac/tmc_saved_loftr/.",
        *[f"{r['name']}: {r['status']}; registration {r['registration_status']}" for r in records],
        "Software: repository commit and versions recorded in summary.json; source code is not duplicated in this results-only ZIP.",
        *LIMITATIONS]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "protection.json", dict(files=protected, assurance=LIMITATIONS[-1]))
    report = validate_package(output, root=root, sealed=False)
    write_json(output / "validation_report.json", report)
    (output / "validation_report.txt").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_json(output / "manifest.json", create_manifest(output, records))
    export_zip(output)
    validate_package(output, root=root)
    return summary
