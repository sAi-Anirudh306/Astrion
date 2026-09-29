"""Small prepared-data walkthrough controller; scientific functions stay upstream.

Each advance performs one real operation or loads explicitly recorded evidence.
Runs are session-owned, write only to unique results directories, and never
replace the M20 scientific decision with a presentation computation.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from src.ui.service import (Artifact, Demo, artifacts, metric_cards, read_json,
                            registration_allowed, safe_path)

TITLES = ("Input", "Ingestion", "Metadata & geolocation", "Preprocessing",
          "Scale normalization", "Feature / representation", "Correspondence generation",
          "Geometric verification", "Spatial validation", "Registration", "Evaluation",
          "Final ASTRION Output")
SENSORS = {"tmc": "Chandrayaan-2 TMC-2", "ohrc": "Chandrayaan-2 OHRC", "iirs": "Chandrayaan-2 IIRS"}


@dataclass
class Output:
    title: str
    status: str = "PENDING"
    origin: str = ""
    explanation: str = ""
    method: str = ""
    facts: dict = field(default_factory=dict)
    images: list[Artifact] = field(default_factory=list)
    seconds: float | None = None


@dataclass
class PipelineRun:
    root: Path
    directory: Path
    demo: Demo
    result: dict
    outputs: list[Output]
    events: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict, repr=False)
    provenance: dict = field(default_factory=dict)


def build_pipeline_run(root: Path, demo: Demo, result: dict) -> PipelineRun:
    """Create a unique run and import selected evidence through the M20 runner."""
    from src.evaluation.experiments import ExperimentConfig, ExperimentRunner, json_safe
    if result["name"] != demo.experiment or result["parameters"]["config"]["source_sensor"] != {"tmc":"TMC", "ohrc":"OHRC", "iirs":"IIRS"}[demo.key]:
        raise ValueError("Selected demo and sensor evidence disagree")
    directory = safe_path(root, "results/ui_pipeline/" + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    events = []
    def event_callback(event):
        events.append({"phase": event.phase, "seconds": event.elapsed_seconds,
                       "status": event.detail.get("status")})
    config = result["configuration"]
    safe_path(root, config["report_path"])
    imported = ExperimentRunner(root, event_callback).run(ExperimentConfig.from_dict(config))
    if imported["metrics"] != result["metrics"] or imported["status"] != result["status"]:
        raise ValueError("M20 evidence differs from the selected saved result")
    run = PipelineRun(root.resolve(), directory, demo, result, [Output(t) for t in TITLES], events)
    run.provenance = {"configuration": config, "m20_import": imported["provenance"],
                      "recorded_metrics": json_safe(result["metrics"]), "inputs": {},
                      "note": "New presentation operations do not replace historical scientific results."}
    return run


def _images(run: PipelineRun, *names: str) -> list[Artifact]:
    return [a for a in artifacts(run.root, run.demo, run.result) if a.title in names]


def _record_input(run: PipelineRun, relative: str, area: str = "results") -> Path:
    path = safe_path(run.root, relative, area=area)
    if path.stat().st_size > 8_000_000:
        raise ValueError("Prepared walkthrough inputs must be smaller than 8 MB")
    run.provenance["inputs"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def _figure(run: PipelineRun, name: str, draw: Callable, title: str) -> Artifact:
    """Render numerical evidence through existing plotting APIs into a new file."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    path = safe_path(run.root, str((run.directory / name).relative_to(run.root)))
    if path.exists():
        raise FileExistsError("Presentation artifact already exists")
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    draw(fig)
    fig.savefig(path, dpi=130)
    fig.clear()
    return Artifact(title, path, "Generated this run from real data; display-only normalization")


def _panels(run: PipelineRun, name: str, panels: list, title: str) -> Artifact:
    from src.evaluation.visualization import plot_image
    def draw(fig):
        axes = fig.subplots(1, len(panels), squeeze=False)[0]
        for ax, (label, image, mask) in zip(axes, panels):
            plot_image(ax, image, mask, title=label)
    return _figure(run, name, draw, title)


def _tmc_ingestion(run: PipelineRun, output: Output) -> None:
    import numpy as np
    import rasterio
    receipt = read_json(run.root, "results/milestone_08_registration/registration_metadata.json")
    for key, relative in (("source", "data/processed/tycho/map_projected/tmc_on_wac_grid.tif"),
                          ("reference", "data/processed/tycho/common_reference/wac_tmc_rows_115734_124259.tif")):
        path = _record_input(run, relative, "data/processed")
        expected = receipt["input_sha256"].get(relative.replace("/", "\\"))
        if expected != run.provenance["inputs"][relative]:
            raise ValueError("Prepared raster differs from the validated M8 input")
        with rasterio.open(path) as dataset:
            if dataset.width * dataset.height > 1_000_000 or dataset.count != 1:
                raise ValueError("Expected a small, single-band prepared crop")
            data = dataset.read(1)
            mask = (dataset.read_masks(1) > 0) & np.isfinite(data)
            run.context[key], run.context[key + "_mask"] = data, mask
            output.facts[key.title()] = f"{data.shape[0]} × {data.shape[1]} prepared pixels · {data.dtype} · {abs(dataset.transform.a):g} m/pixel"
    output.origin = "Computed now: validated prepared raster read"
    output.images = [_panels(run, "ingestion.png", [("TMC prepared scientific crop", run.context["source"], run.context["source_mask"]),
                            ("WAC prepared reference", run.context["reference"], run.context["reference_mask"])], "Ingested prepared crops")]


def _execute(run: PipelineRun, index: int, o: Output) -> None:
    r, m, meta = run.result, run.result["metrics"], run.result["metadata"]
    config = r["parameters"]["config"]
    c = run.context
    o.status = "COMPLETE"
    o.origin = "Recorded evidence: historical processing is not rerun"
    if index == 0:
        o.explanation = "The selected demo defines the source and external lunar reference."
        o.method = "Prepared observation pair"
        o.facts = {"Source": SENSORS[run.demo.key], "Reference": "LRO WAC", "Representation": config["representation"]}
        o.images = _images(run, "Source", "Reference")
    elif index == 1:
        o.explanation = "Inspect the prepared input; these dimensions do not describe the full raw product."
        o.method = "Prepared crop ingestion; original scientific products remain unchanged"
        o.images = _images(run, "Source", "Reference")
        if run.demo.key == "tmc":
            _tmc_ingestion(run, o)
        else:
            scale = meta.get("source_scale") or meta.get("scale") or {}
            o.facts = {"Source": SENSORS[run.demo.key], "Prepared crop (rows × columns)": str(scale.get("original_shape", "N/A")),
                       "Native GSD (m/pixel)": config["source_gsd"], "Product representation": "RAW DN reduction" if run.demo.key == "iirs" else "Optical crop"}
            try:
                prior = read_json(run.root, "results/milestone_17_iirs/iirs.json" if run.demo.key == "iirs" else "results/milestone_16_ohrc/ohrc.json")
                product = prior.get("product", prior.get("metadata", {}))
                o.facts["Product identity"] = product.get("product_id", prior.get("product_stem", "N/A"))
                o.facts["Recorded raw dtype"] = product.get("dtype", product.get("data_type", "N/A"))
            except (OSError, ValueError):
                o.facts["Product identity"] = "Optional metadata unavailable"
    elif index == 2:
        o.explanation = "Relate the source region to its localized reference; footprint mapping is approximate, not precise orthorectification."
        o.method = "Recorded footprint-guided localization"
        o.images = _images(run, "Source", "Reference")
        o.facts = {"Source GSD (m/pixel)": config["source_gsd"], "Reference GSD (m/pixel)": config["reference_gsd"]}
        if run.demo.key == "tmc":
            path = _record_input(run, "data/processed/tycho/map_projected/tmc_on_wac_grid_metadata.json", "data/processed")
            data = json.loads(path.read_text())
            corners = data["corners"]
            o.facts["Mapping"] = data["method"]
            def draw(fig):
                ax = fig.subplots()
                points = [corners[k] for k in ("upper_left", "upper_right", "lower_right", "lower_left", "upper_left")]
                ax.plot([p["longitude"] for p in points], [p["latitude"] for p in points], "o-", color="#8070b8")
                ax.set(xlabel="Lunar longitude (degrees)", ylabel="Lunar latitude (degrees)", title="Recorded TMC source footprint localized to the WAC grid")
            o.images.append(_figure(run, "footprint.png", draw, "Recorded footprint"))
    elif index == 3:
        o.explanation = "Compare scientific input with the real normalized processing representation."
        o.method = "Recorded preparation"
        o.images = _images(run, "Source", "Preprocessed / scaled")
        if run.demo.key == "tmc":
            from src.matching.cross_modal import prepare_representation
            for key in ("source", "reference"):
                c[key + "_processed"], _ = prepare_representation(c[key], c[key + "_mask"])
            o.origin = "Computed now: M18 prepared-crop preprocessing demonstration"
            o.method = "Valid-sample DN mapping and default percentile normalization; no denoising or illumination correction"
            o.facts["Evidence boundary"] = "Saved LoFTR used native preprocessing before map sampling. This crop demonstration is not its original intermediate or a new matching run."
            from src.evaluation.visualization import display_image, show_display
            import numpy as np
            def draw(fig):
                axes = fig.subplots(1, 2)
                show_display(axes[0], display_image(c["source"], c["source_mask"], percentiles=(0, 100)), "Before: scientific DN (min/max display)")
                show_display(axes[1], np.ma.array(c["source_processed"], mask=~c["source_mask"]), "After: actual normalized values [0, 1]")
            o.images = [_figure(run, "preprocessing.png", draw, "Preprocessing before / after")]
        else:
            o.facts["Evidence limit"] = "Saved source and prepared/scaled views; an isolated before/after preprocessing array is unavailable. No scientific processing of display PNGs."
    elif index == 4:
        o.explanation = "Inspect physical sampling and the resulting processing dimensions."
        o.method = "M15/common physical scale normalization"
        o.images = _images(run, "Source", "Preprocessed / scaled")
        scale = meta.get("source_scale") or meta.get("scale") or {}
        if run.demo.key == "tmc":
            from src.preprocessing.scale_normalization import normalize_resolution
            level = normalize_resolution(c["source_processed"], config["source_gsd"], config["reference_gsd"])
            scale = level.metadata()
            o.origin = "Computed now: existing scale-normalization function"
            o.images = [_panels(run, "scale.png", [("Before: prepared 100 m grid", c["source_processed"], c["source_mask"]),
                            ("After: same common grid; no resizing required", level.data, c["source_mask"])], "Scale normalization")]
        o.facts = {"Input GSD": scale.get("source_gsd", "N/A"), "Target GSD": scale.get("target_gsd", "N/A"),
                   "Effective GSD x/y": str(scale.get("effective_gsd_xy", "N/A")), "Output dimensions": str(scale.get("scaled_shape", "N/A"))}
    elif index == 5:
        o.explanation = "Inspect the representation used for correspondence; learned matching does not produce SIFT keypoints."
        o.method = str(r["parameters"].get("matcher") or "Recorded representation")
        o.images = _images(run, "Source", "Reference") if run.demo.key == "tmc" else _images(run, "Features (prior preparation)", "Spectrum")
        o.facts = {"Source features": m.get("source_keypoints") if m.get("source_keypoints") is not None else "N/A — no separate keypoints recorded",
                   "Reference features": m.get("reference_keypoints") if m.get("reference_keypoints") is not None else "N/A — no separate keypoints recorded"}
        if run.demo.key == "tmc":
            o.facts["Output"] = "Saved LoFTR input/representation displays; no new inference"
    elif index == 6:
        o.explanation = "Inspect proposed source-to-reference relationships before geometric rejection."
        o.method = str(r["parameters"].get("matcher"))
        o.facts = {"Candidate correspondences": m["candidate_matches"]}
        o.images = _images(run, "Candidate matches")
        if run.demo.key == "iirs":
            # Real M18 coordinate pairs, not the different M17 inlier classification.
            points = meta.get("source_points_original", [])
            destination = meta.get("reference_points_original", [])
            def draw(fig):
                axes = fig.subplots(1, 2)
                for ax, data, title in zip(axes, (points, destination), ("IIRS source candidate locations", "WAC reference candidate locations")):
                    for n, (x, y) in enumerate(data):
                        ax.plot(x, y, "o", color="#8070b8")
                        ax.annotate(str(n+1), (x, y))
                    ax.set(title=title, xlabel="Column (original processing grid)", ylabel="Row")
                    ax.invert_yaxis()
                fig.suptitle("Recorded M18 candidate pairs — matching numbers identify correspondence")
            o.images = [_figure(run, "candidates.png", draw, "Actual M18 candidate coordinates")]
    elif index == 7:
        o.explanation = "Geometric verification separates consistent support from rejected candidate relationships."
        o.method = "Recorded affine verification; no new model fitting"
        o.facts = {k: v for k, v in metric_cards(r).items() if k in {"Candidate matches", "Verified inliers", "Inlier ratio"}}
        o.images = _images(run, "Candidate matches", "Verified inliers")
        if not o.images:
            o.images = run.outputs[6].images.copy()
            o.facts["Evidence limit"] = "Candidate locations shown; per-point M18 inlier mask is unavailable. Inlier count is recorded evidence."
        o.status = "COMPLETE" if registration_allowed(r) else "WARNING"
    elif index == 8:
        o.explanation = "Inspect occupied reference cells and how widely the verified support is distributed."
        o.method = "Recorded reference-grid inlier counts"
        o.facts = {"Spatial cells": metric_cards(r)["Spatial cells"]}
        counts = m.get("spatial_distribution", {}).get("counts")
        if counts is not None:
            def draw(fig):
                ax = fig.subplots()
                ax.imshow(counts, cmap="Purples", vmin=0)
                for row, values in enumerate(counts):
                    for col, value in enumerate(values):
                        ax.text(col, row, str(value), ha="center", va="center", color="white" if value > max(map(max, counts))/2 else "black")
                ax.set(title="Actual verified inlier counts per reference cell", xlabel="Grid column", ylabel="Grid row")
            o.images = [_figure(run, "spatial.png", draw, "Spatial support from recorded counts")]
        o.status = "COMPLETE" if registration_allowed(r) else "WARNING"
    elif index == 9:
        o.explanation = "Apply the supported transform to produce the registered moving image."
        o.method = "Existing mask-aware affine resampling with the saved transform"
        if not registration_allowed(r):
            o.status = "SKIPPED"
            o.explanation = "REGISTRATION WITHHELD — " + "; ".join(r["reliability"]["reasons"])
            o.facts = {"Registered product": "WITHHELD"}
        elif run.demo.key == "tmc":
            import numpy as np
            from src.geometry.registration import warp_affine
            from src.evaluation.visualization import plot_registration
            if "Before / after" not in {a.title for a in artifacts(run.root, run.demo, r)}:
                raise ValueError("Cannot link saved transform to validated registration evidence")
            reg = warp_affine(c["source"].astype(np.float32), np.asarray(m["transform"]), c["reference"].shape, validity_mask=c["source_mask"])
            c["registered"], c["registered_mask"] = reg.image, reg.valid_mask
            np.savez_compressed(run.directory / "registered_product.npz", image=reg.image, valid_mask=reg.valid_mask, transform=np.asarray(m["transform"]))
            def draw(fig):
                axes = fig.subplots(1, 4)
                plot_registration(axes, c["source"], c["reference"], reg.image, c["source_mask"], c["reference_mask"], reg.valid_mask)
                for ax, title in zip(axes, ("Moving before", "Fixed reference", "Registered moving", "50% overlay\nCommon valid support")):
                    ax.set_title(title, loc="left", fontsize=10)
            o.images = [_figure(run, "registered.png", draw, "Moving before · fixed reference · registered moving · overlay")]
            o.origin = "Computed now: resampling with recorded reliable transform; no model refit"
            o.facts = {"Registered shape": str(reg.image.shape), "Valid pixels": int(reg.valid_mask.sum())}
    elif index == 10:
        o.explanation = "Evaluate correspondence evidence; fitted residual RMSE is not absolute lunar geolocation accuracy."
        o.method = "M20 saved metrics; new display comparisons where registration is supported"
        o.facts = metric_cards(r)
        if "registered" in c and registration_allowed(r):
            from src.evaluation.visualization import overlay_image, checkerboard_image, show_display
            def draw(fig):
                for ax, fn, title in zip(fig.subplots(1, 2), (overlay_image, checkerboard_image), ("Overlay", "Checkerboard")):
                    show_display(ax, fn(c["registered"], c["reference"], c["registered_mask"], c["reference_mask"]), title)
            o.images = [_figure(run, "evaluation.png", draw, "Overlay / Checkerboard")]
            o.origin = "Recorded metrics + newly generated comparison of the recomputed registered crop"
        else:
            o.images = run.outputs[8].images.copy()
            o.status = "WARNING"
    else:
        o.explanation = ("Corresponding match points, the supported registered product, and evaluation in one result."
                         if registration_allowed(r) else "Actual correspondence evidence and evaluation; registration is withheld because support is insufficient.")
        o.method = "Final evidence summary"
        o.facts = metric_cards(r)
        o.images = run.outputs[7].images + run.outputs[9].images + run.outputs[10].images
        o.status = "COMPLETE" if registration_allowed(r) else "WARNING"
        o.origin = "Mixed execution: each preceding stage identifies new computation or recorded evidence"


def advance_pipeline(run: PipelineRun, target: int, callback: Callable | None = None) -> Output:
    """Execute prerequisites once, publish honest running/failure states, persist receipts."""
    if not 0 <= target < len(TITLES):
        raise ValueError("Invalid stage")
    for index in range(target + 1):
        output = run.outputs[index]
        if output.status == "FAILED":
            raise ValueError("This run failed; start a new run after correcting the input")
        if output.status != "PENDING":
            continue
        output.status = "RUNNING"
        if callback:
            callback(output)
        tick = perf_counter()
        try:
            _execute(run, index, output)
        except Exception as exc:
            output.status, output.explanation = "FAILED", str(exc)
            raise
        finally:
            output.seconds = perf_counter() - tick
            receipt = {"stage": output.title, "status": output.status, "origin": output.origin,
                       "method": output.method, "facts": output.facts, "explanation": output.explanation,
                       "operation_seconds": output.seconds,
                       "artifacts": [{"title": a.title, "path": str(a.path.relative_to(run.root)),
                                      "sha256": hashlib.sha256(a.path.read_bytes()).hexdigest()} for a in output.images],
                       "m20_events": run.events, "provenance": run.provenance}
            with (run.directory / f"{index:02d}_receipt.json").open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
        if callback:
            callback(output)
    return run.outputs[target]
