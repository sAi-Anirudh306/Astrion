"""Read-only presentation of the authoritative final package and stage evidence.

This module deliberately imports no scientific execution modules. The catalog
binds named experiments to known artifacts, never to arbitrary uploaded paths.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any


REPORT = "results/final/experiments.json"


@dataclass(frozen=True)
class Demo:
    key: str
    title: str
    experiment: str
    description: str


DEMOS = (
    Demo("tmc", "TMC-2 ↔ WAC", "TMC saved LoFTR", "Optical correspondence · positive control"),
    Demo("ohrc", "OHRC ↔ WAC", "OHRC saved intensity", "Extreme scale difference · support withheld"),
    Demo("iirs", "IIRS ↔ WAC", "IIRS mean intensity", "Cross-spectral RAW DN · support withheld"),
)


@dataclass(frozen=True)
class Artifact:
    title: str
    path: Path
    origin: str


@dataclass(frozen=True)
class Stage:
    title: str
    status: str
    description: str
    detail: Any
    artifact: str | None = None
    runtime: str = "N/A — no stage duration recorded"


def safe_path(root: Path, relative: str, *, area: str = "results") -> Path:
    """Confine access to a project area, including symlink resolution."""
    root = root.resolve()
    requested = Path(relative)
    if requested.is_absolute() or requested.drive or ".." in requested.parts:
        raise ValueError("Use a relative path within the prepared artifact area")
    path = (root / requested).resolve()
    boundary = (root / area).resolve()
    if not boundary.is_relative_to(root) or not path.is_relative_to(boundary):
        raise ValueError("Artifact path escapes the permitted area")
    return path


@lru_cache(maxsize=32)
def _json_snapshot(path: str, modified: int, size: int) -> str:
    del modified, size
    return Path(path).read_text(encoding="utf-8")


def read_json(root: Path, relative: str) -> dict:
    """Cache small JSON text while returning independent mutable dictionaries."""
    path = safe_path(root, relative)
    stat = path.stat()
    if stat.st_size > 8_000_000:
        raise ValueError("UI reports must be smaller than 8 MB")
    value = json.loads(_json_snapshot(str(path), stat.st_mtime_ns, stat.st_size))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON report object")
    return value


def load_results(root: Path) -> dict[str, dict]:
    """Validate the presentation contract without re-assessing scientific gates."""
    report = read_json(root, REPORT)
    if report.get("schema_version") != 1 or not isinstance(report.get("experiments"), list):
        raise ValueError("Expected a version-1 final experiments report")
    records = {}
    for result in report["experiments"]:
        if not isinstance(result, dict) or not isinstance(result.get("name"), str):
            raise ValueError("Invalid experiment record")
        name = result["name"]
        if name in records:
            raise ValueError("Duplicate experiment name")
        for field in ("metrics", "parameters", "metadata", "reliability", "timing"):
            if not isinstance(result.get(field), dict):
                raise ValueError(f"Missing experiment {field}")
        status = result.get("status")
        if status not in {"RELIABLE", "INSUFFICIENT_SUPPORT", "NO_MODEL", "FAILED"}:
            raise ValueError("Unknown reliability status")
        if status != result["reliability"].get("status"):
            raise ValueError("Conflicting reliability status")
        nested = result["metrics"].get("reliability", {})
        if nested.get("status", status) != status:
            raise ValueError("Conflicting metric reliability status")
        records[name] = result
    return records


def discover_demos(root: Path) -> tuple[Demo, ...]:
    """Fixed deterministic presentation order; no recursive filesystem scan."""
    records = load_results(root)
    return tuple(demo for demo in DEMOS if demo.experiment in records)


def format_metric(value: Any, *, digits: int = 2, suffix: str = "", percent: bool = False) -> str:
    """Unavailable/nonfinite values are never displayed as numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "N/A"
    value = value * 100 if percent else value
    number = f"{value:,.{digits}f}" if digits else f"{value:,.0f}"
    if value and abs(value) < 10 ** (-digits):
        number = f"{value:.2e}"
    return number + ("%" if percent else suffix)


def reliability(result: dict) -> tuple[str, str]:
    """Return the saved decision and a restrained semantic color class."""
    status = result.get("status")
    return {
        "RELIABLE": ("RELIABLE", "good"),
        "INSUFFICIENT_SUPPORT": ("INSUFFICIENT SUPPORT", "warning"),
        "NO_MODEL": ("NO MODEL", "warning"),
        "FAILED": ("FAILED", "failed"),
    }.get(status, ("UNAVAILABLE", "warning"))


def metric_cards(result: dict) -> dict[str, str]:
    m = result["metrics"]
    spatial = m.get("spatial_distribution") or {}
    occupied, total = m.get("spatial_occupied_cells"), spatial.get("total_cells")
    coverage = "N/A" if occupied is None or total is None else (
        f"{format_metric(occupied, digits=0)} / {format_metric(total, digits=0)}")
    return {
        "Candidate matches": format_metric(m.get("candidate_matches"), digits=0),
        "Verified inliers": format_metric(m.get("verified_inliers"), digits=0),
        "Inlier ratio": format_metric(m.get("inlier_ratio"), percent=True),
        "Fitted residual RMSE": format_metric(m.get("rmse"), digits=3, suffix=" px"),
        "Spatial cells": coverage,
        "Recorded runtime": format_metric(result["timing"].get("historical_seconds"), digits=4, suffix=" s"),
    }


def registration_allowed(result: dict) -> bool:
    return result.get("status") == "RELIABLE" and result["metrics"].get("transform_available") is True


def final_product(root: Path, result: dict, key: str) -> Path | None:
    """Resolve an exported product only within the final package."""
    item = result.get("final_artifacts", {}).get(key)
    if item is None:
        return None
    return safe_path(root, "results/final/" + item["path"], area="results/final")


def artifacts(root: Path, demo: Demo, result: dict) -> tuple[Artifact, ...]:
    """Use packaged final figures; historical lookup is only for unpackaged records."""
    if result.get("name") != demo.experiment:
        raise ValueError("Demo/result association mismatch")
    if "final_artifacts" not in result:
        return historical_artifacts(root, demo, result)
    titles = {
        "source": "Source", "reference": "Reference",
        "candidate_matches": "Candidate matches", "candidate_coordinates": "Candidate coordinates",
        "verified_inliers": "Verified inliers", "spatial_distribution": "Spatial distribution",
        "residuals": "Residuals", "before_after": "Before / after", "overlay": "Overlay",
        "checkerboard": "Checkerboard", "preprocessed_scaled": "Preprocessed / scaled",
        "features_(prior_preparation)": "Features (prior preparation)", "spectrum": "Spectrum",
    }
    found = []
    for key, title in titles.items():
        if key in {"before_after", "overlay", "checkerboard"} and not registration_allowed(result):
            continue
        path = final_product(root, result, key)
        if path is not None and path.is_file():
            found.append(Artifact(title, path, result["final_artifacts"][key]["origin"]))
    return tuple(found)


def historical_artifacts(root: Path, demo: Demo, result: dict) -> tuple[Artifact, ...]:
    """Explicit figure associations prevent mixing similar experiments.

M17 figures describe prior preparation, not the new M18 correspondence run.
Registration figures are exposed only when the M8 transform and counts agree.
"""
    if result.get("name") != demo.experiment:
        raise ValueError("Demo/result association mismatch")
    specs = []
    if demo.key == "tmc":
        base = "results/experiment_tycho_map_projected/"
        specs += [("Source", base + "tmc_map_projected.png", "M19 prepared map-projected TMC display"),
                  ("Reference", base + "wac_reference.png", "M19 external LRO WAC display")]
        # These artifacts share M8/M9 evidence, unlike the separate LoFTR scale sweep.
        try:
            receipt = read_json(root, "results/milestone_08_registration/registration_metadata.json")
            m = result["metrics"]
            linked = (receipt.get("affine_matrix") == m.get("transform")
                      and receipt.get("ransac_candidate_count") == m.get("candidate_matches")
                      and receipt.get("ransac_inlier_count") == m.get("verified_inliers"))
        except (OSError, ValueError):
            linked = False
        if linked:
            specs += [("Candidate matches", base + "loftr_0.20_candidates.png", "M19 saved LoFTR support"),
                      ("Verified inliers", "results/milestone_10_visualization/verified_matches.png", "M10 · same M8/M9 support; display subsampled"),
                      ("Spatial distribution", "results/milestone_10_visualization/spatial_distribution.png", "M10 · same M8/M9 support"),
                      ("Residuals", "results/milestone_10_visualization/residual_map.png", "M10 · fitted reference-pixel residuals")]
            if registration_allowed(result):
                specs += [(title, "results/milestone_10_visualization/" + filename, "M10 · saved M8 registration")
                          for title, filename in (("Before / after", "registration_comparison.png"),
                                                  ("Overlay", "registration_overlay.png"),
                                                  ("Checkerboard", "registration_checkerboard.png"))]
    elif demo.key in {"ohrc", "iirs"}:
        sensor = demo.key
        base = f"results/milestone_{'16_ohrc' if sensor == 'ohrc' else '17_iirs'}/"
        origin = "M16 saved control" if sensor == "ohrc" else "M17 preparation context; M18 metrics shown above"
        specs += [(title, base + sensor + ending, origin) for title, ending in (
            ("Source", "_preview.png" if sensor == "ohrc" else "_band_mean.png"),
            ("Reference", "_reference_crop.png"), ("Preprocessed / scaled", "_scaled.png"),
            ("Features (prior preparation)", "_keypoints.png"))]
        if sensor == "ohrc":
            specs.append(("Candidate matches", base + "ohrc_matches.png", origin))
        else:
            specs.append(("Spectrum", base + "iirs_spectrum.png", origin))
    found = []
    for title, relative, origin in specs:
        path = safe_path(root, relative)
        if path.is_file():
            found.append(Artifact(title, path, origin))
    return tuple(found)


def build_stages(result: dict, available: tuple[Artifact, ...]) -> tuple[Stage, ...]:
    """Reconstruct only recorded evidence; never simulate live execution."""
    p, m, meta = result["parameters"], result["metrics"], result["metadata"]
    names = {a.title for a in available}
    def stage(title: str, detail: Any, description: str, artifact: str | None = None,
              status: str | None = None) -> Stage:
        return Stage(title, status or ("COMPLETE" if detail else "NOT RECORDED"), description,
                     detail or "No stage-specific evidence in this saved record.",
                     artifact if artifact in names else None)
    reason = "; ".join(result["reliability"].get("reasons", []))
    registered = registration_allowed(result) and "Before / after" in names
    return (
        stage("Input", p.get("config"), "Recorded sensor pair and representation", "Source"),
        stage("Ingestion", None, "Prepared artifacts; raw data is not read in this session", "Source", "SKIPPED"),
        stage("Metadata & geolocation", meta.get("input_provenance"), "Inspect saved provenance; no new geolocation", "Reference"),
        stage("Preprocessing", p.get("preprocessing"), "Recorded normalization and illumination settings", "Preprocessed / scaled"),
        stage("Scale normalization", meta.get("source_scale") or meta.get("scale"), "Physical sampling on the recorded processing grid", "Preprocessed / scaled"),
        stage("Feature / representation", {"feature": p.get("feature"), "matcher": p.get("matcher"),
                                          "spectral_indices": meta.get("spectral_representation")},
              "Saved learned support or classical descriptors", "Features (prior preparation)"),
        stage("Correspondence generation", {"method": p.get("matcher"), "candidates": m.get("candidate_matches")},
              f"{format_metric(m.get('candidate_matches'), digits=0)} candidate matches", "Candidate matches"),
        stage("Geometric verification", {"parameters": p.get("verifier"), "inliers": m.get("verified_inliers"),
                                         "residuals": m.get("residual_statistics")},
              f"{format_metric(m.get('verified_inliers'), digits=0)} verified inliers", "Verified inliers",
              "COMPLETE" if m.get("transform_available") else "WARNING"),
        stage("Spatial validation", m.get("spatial_distribution"), "Reference-grid evidence and reliability gates", "Spatial distribution",
              "COMPLETE" if result["status"] == "RELIABLE" else "WARNING"),
        stage("Registration", "Saved M8 registration is available." if registered else reason or "No linked registration artifact available.",
              "Saved alignment available" if registered else "Registration withheld" if not registration_allowed(result) else "Artifact unavailable",
              "Before / after", "COMPLETE" if registered else "SKIPPED"),
        stage("Evaluation", result["reliability"], "Engineering reliability; no external ground truth", "Residuals",
              "COMPLETE" if result["status"] == "RELIABLE" else "FAILED" if result["status"] == "FAILED" else "WARNING"),
    )


def supporting_metadata(root: Path, demo: Demo) -> dict:
    """Small prior receipts only; missing optional metadata is harmless."""
    relative = {"iirs": "results/final/evidence/iirs_preparation.json",
                "ohrc": "results/final/evidence/ohrc_preparation.json"}.get(demo.key)
    if relative is None:
        return {}
    try:
        report = read_json(root, relative)
        if demo.key == "iirs":
            product = report.get("product", {})
            return {"product": product.get("product_id"), "bands": product.get("bands"),
                    "wavelength_min": product.get("wavelength_min"), "wavelength_max": product.get("wavelength_max"),
                    "wavelength_units": product.get("wavelength_units"),
                    "reductions": [{k: rep.get(k) for k in ("name", "band_indices", "official_band_numbers", "wavelengths")}
                                   for rep in report.get("representations", [])]}
        return {"product": report.get("product_stem"), "metadata": report.get("metadata")}
    except (OSError, ValueError):
        return {}


def downloads(root: Path) -> dict[str, tuple[str, bytes, str]]:
    """Serve existing compact reports without changing or bundling raw data."""
    found = {}
    for title, filename, mime in (("Experiment JSON", "experiments.json", "application/json"),
                                  ("Human summary", "summary.txt", "text/plain"),
                                  ("Comparison CSV", "comparison.csv", "text/csv")):
        path = safe_path(root, "results/final/" + filename, area="results/final")
        if path.is_file() and path.stat().st_size <= 8_000_000:
            found[title] = (filename, path.read_bytes(), mime)
    return found
