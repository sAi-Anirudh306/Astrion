"""ASTRION prepared-results demo. Launch with python -m streamlit run app.py."""
import base64
from html import escape
from io import BytesIO
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Install the optional UI: .venv\\Scripts\\python.exe -m pip install -r requirements-ui.txt")
from PIL import Image, UnidentifiedImageError

from src.ui.service import (DEMOS, Artifact, Demo, artifacts, build_stages, discover_demos,
                            downloads, final_product, format_metric, load_results, metric_cards,
                            registration_allowed, reliability, safe_path, supporting_metadata)

ROOT = Path(__file__).resolve().parent


@st.cache_data(max_entries=48, show_spinner=False)
def display_image(path: str, modified: int, width: int = 1500) -> bytes:
    """Cache bounded display derivatives in memory; originals remain unchanged."""
    del modified
    with Image.open(path) as source:
        if source.width * source.height > 24_000_000:
            raise ValueError("Prepared UI image exceeds the display size limit")
        source.thumbnail((width, 1100))
        output = BytesIO()
        source.convert("RGB").save(output, format="WEBP", quality=88)
        return output.getvalue()


def show_artifact(artifact: Artifact, *, width: int = 1500) -> None:
    try:
        data = display_image(str(artifact.path), artifact.path.stat().st_mtime_ns, width)
        st.image(data, caption=artifact.origin, width="stretch")
    except (OSError, ValueError, UnidentifiedImageError):
        st.caption("This optional visualization is unavailable. Recorded metrics remain accessible.")


@st.cache_data(max_entries=4, show_spinner=False)
def registered_preview(path: str, modified: int) -> bytes:
    """Render the actual masked scientific GeoTIFF without changing its pixels."""
    del modified
    import numpy as np
    import rasterio
    from src.evaluation.visualization import display_image as scientific_display
    with rasterio.open(path) as dataset:
        if dataset.count != 1 or dataset.width * dataset.height > 1_000_000:
            raise ValueError("Expected a small prepared single-band registered product")
        values = scientific_display(dataset.read(1), dataset.read_masks(1) > 0)
    pixels = np.asarray(np.ma.filled(values, 0) * 255, dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).convert("RGB").save(output, format="WEBP", quality=88)
    return output.getvalue()


def final_products(result: dict, *, expand_registered: bool = True) -> None:
    """Expose real correspondence exports and the reliability-gated final raster."""
    st.caption("Validated final package · results/final")
    for key, title in (("match_points", "Download corresponding match points (CSV)"),
                       ("verified_inlier_points", "Download verified inliers (CSV)")):
        path = final_product(ROOT, result, key)
        if path is not None and path.is_file():
            st.download_button(title, path.read_bytes(), path.name, "text/csv", key=key)
    if registration_allowed(result):
        path = final_product(ROOT, result, "registered_product")
        if path is not None and path.is_file():
            panel = (st.expander("Registered scientific product · float32 GeoTIFF")
                     if expand_registered else st.container())
            with panel:
                try:
                    st.image(registered_preview(str(path), path.stat().st_mtime_ns),
                             caption="Display normalization of the actual final registered raster", width="stretch")
                    st.download_button("Download registered GeoTIFF", path.read_bytes(), path.name, "image/tiff")
                except (OSError, ValueError):
                    st.caption("Registered product preview unavailable.")


def html(value: str) -> None:
    st.markdown(value, unsafe_allow_html=True)


def hero() -> None:
    html('<div class="nav"><span class="wordmark">ASTRION</span>'
         '<span class="nav-note">LUNARMATCH &nbsp; / &nbsp; PLANETARY VISION</span></div>')
    image_html = ""
    try:
        path = safe_path(ROOT, "assets/astrion_hero.png", area="assets")
        encoded = base64.b64encode(display_image(str(path), path.stat().st_mtime_ns, 1672)).decode("ascii")
        image_html = f'<img alt="Moon and orbiting satellite illuminated by a restrained violet orbital arc" src="data:image/webp;base64,{encoded}">'
    except (OSError, ValueError):
        st.caption("Hero artwork unavailable. The scientific demo is still accessible below.")
    html('<section class="hero">' + image_html + '<div class="hero-copy">'
         '<div class="eyebrow">From orbital imagery to measured evidence</div>'
         '<h1>Different views.<br>Common ground.</h1>'
         '<p class="tagline">Planetary Image Correspondence<br>&amp; Registration</p>'
         '<p>Explore how ASTRION connects lunar observations across sensors, '
         'illumination and scale — and recognizes when the evidence is insufficient.</p>'
         '<div class="hero-foot">CHANDRAYAAN-2 &nbsp; / &nbsp; EXTERNAL LRO WAC REFERENCE</div>'
         '</div></section>')


def section(title: str, note: str = "") -> None:
    html(f'<div class="section"><h2>{escape(title)}</h2><span class="muted">{escape(note)}</span></div>')


def input_cards(demo: Demo, result: dict, images: tuple[Artifact, ...]) -> None:
    config = result["parameters"].get("config") or {}
    meta = result["metadata"]
    supplemental = supporting_metadata(ROOT, demo)
    by_name = {a.title: a for a in images}
    for col, side, sensor, gsd in zip(st.columns(2), ("Source", "Reference"),
                                     ({"tmc": "Chandrayaan-2 TMC-2", "ohrc": "Chandrayaan-2 OHRC", "iirs": "Chandrayaan-2 IIRS"}[demo.key], "LRO WAC"),
                                     (config.get("source_gsd"), config.get("reference_gsd"))):
        with col, st.container(border=True):
            html(f'<div class="eyebrow">{side}</div>')
            st.markdown(f"**{sensor}**")
            if side in by_name:
                show_artifact(by_name[side], width=660)
            else:
                st.caption("Prepared preview unavailable")
            st.caption(f"Recorded input GSD · {format_metric(gsd, suffix=' m/pixel')}")
            scale = meta.get("source_scale", meta.get("scale", {})) if side == "Source" else meta.get("reference_scale", {})
            if scale.get("original_shape"):
                st.caption("Input dimensions · " + " × ".join(str(x) for x in scale["original_shape"]) + " (rows × columns)")
            if side == "Source":
                st.caption(meta.get("modality", "Modality not recorded"))
                if supplemental.get("product"):
                    st.caption("Product · " + supplemental["product"])
            else:
                st.caption("External lunar reference · not Chandrayaan-2 imagery")
    if demo.key == "iirs":
        p = supplemental
        st.caption(f"Hyperspectral · {format_metric(p.get('bands'), digits=0)} bands · "
                   f"{format_metric(p.get('wavelength_min'), digits=1)}–{format_metric(p.get('wavelength_max'), digits=1)} "
                   f"{p.get('wavelength_units') or ''} · RAW DN, not calibrated radiance")
        indices = meta.get("spectral_representation", [])
        st.caption("Selected M18 representation: predetermined band mean, zero-based indices " + ", ".join(map(str, indices)))
        with st.expander("Predetermined M17 spectral reductions"):
            st.json(p.get("reductions", []))


def result_summary(result: dict) -> None:
    label, color = reliability(result)
    html(f'<div class="section"><h2>Evidence, assessed.</h2><span class="status {color}">● {label}</span></div>')
    config = result["parameters"].get("config") or {}
    st.caption(f"{config.get('source_sensor', 'N/A')} ↔ {config.get('reference_sensor', 'N/A')}  /  "
               f"{config.get('representation', 'N/A')}  /  {result['parameters'].get('matcher') or 'N/A'}")
    html('<div class="metric-grid">' + "".join(
        f'<div class="metric-card"><div class="metric-label">{escape(k)}</div><div class="metric-value">{escape(v)}</div></div>'
        for k, v in metric_cards(result).items()) + '</div>')
    origin = result["metadata"].get("evidence_origin", "Historical experiment")
    st.caption("Runtime scope: " + origin + ". M20 import time and walkthrough operations are separate from these historical measurements.")
    st.caption("RMSE is a fitted correspondence residual in reference processing pixels, not absolute geographic accuracy. Spatial cells measure support, not image overlap.")
    if not registration_allowed(result):
        reason = "; ".join(result["reliability"].get("reasons", [])) or "No reliable transform is recorded."
        html('<div class="withheld"><strong>REGISTRATION WITHHELD</strong>'
             '<p>Verified evidence does not satisfy the recorded reliability criteria.</p>'
             f'<p>{escape(reason)}</p></div>')
    if result["metrics"].get("verified_inliers") == 3:
        st.caption("Three affine-defining points provide no redundant validation; a small fitted residual does not establish a reliable alignment.")


def watch_pipeline(demo: Demo, result: dict, images: tuple[Artifact, ...]) -> None:
    from src.ui.pipeline import TITLES, advance_pipeline, build_pipeline_run
    section("Follow the evidence", "PREPARED-DATA WALKTHROUGH")
    st.caption("Advance through real outputs. Lightweight prepared-crop operations execute on demand; historical stages are explicitly labelled recorded evidence.")
    run_key, stage_key = "run_" + demo.key, "stage_" + demo.key
    if run_key not in st.session_state:
        input_cards(demo, result, images)
        if not st.button("Begin walkthrough", key="begin_" + demo.key, type="primary"):
            return
        try:
            st.session_state[run_key] = build_pipeline_run(ROOT, demo, result)
        except Exception as exc:
            st.error("Unable to open this prepared-data run: " + str(exc))
            return
        st.session_state[stage_key] = 0
    run = st.session_state[run_key]
    selected = st.session_state.get(stage_key, 0)
    previous, following, finish = st.columns([1, 1, 2])
    if previous.button("Previous", disabled=selected == 0, key="previous_" + demo.key):
        st.session_state[stage_key] = selected - 1
    if following.button("Next", disabled=selected == len(TITLES)-1, key="next_" + demo.key):
        st.session_state[stage_key] = selected + 1
    if finish.button("Show final output", key="final_" + demo.key):
        st.session_state[stage_key] = len(TITLES)-1
    selected = st.select_slider("Inspect pipeline stage", options=list(range(len(TITLES))),
                                format_func=lambda i: TITLES[i], key=stage_key)
    progress = st.empty()
    try:
        current = advance_pipeline(run, selected, lambda output: progress.caption(f"{output.title} · {output.status}"))
    except Exception as exc:
        st.error("Walkthrough stopped: " + str(exc))
        st.caption("Historical Quick Result remains available. No unsupported registration has been created.")
        if st.button("Reset failed walkthrough", key="reset_" + demo.key):
            del st.session_state[run_key]
            st.rerun()
        return
    progress.empty()
    st.caption("Evidence import complete · " + result["status"].replace("_", " ") if any(e["phase"] == "completed" for e in run.events) else "Evidence import pending")
    icon = '<svg class="line-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="3" y="3" width="14" height="14" rx="3"/><path d="M6 10h8M10 6v8"/></svg>'
    cards = []
    for i, stage in enumerate(run.outputs):
        color = "good" if stage.status == "COMPLETE" else "warning" if stage.status == "WARNING" else "failed" if stage.status == "FAILED" else "muted"
        cards.append(f'<div class="stage {"active" if selected == i else ""}"><div class="stage-top"><span>{icon}{i+1:02d}</span>'
                     f'<span class="{color}">{stage.status}</span></div><h3>{escape(stage.title)}</h3><p>{escape(stage.origin or "Advance to inspect this stage")}</p></div>')
    html('<div class="stage-grid">' + "".join(cards) + '</div>')
    with st.container(border=True):
        st.markdown(f"**{selected+1:02d} / {current.title}** · {current.status}")
        st.write(current.explanation)
        st.caption(current.origin)
        st.caption("Method · " + current.method)
        st.caption("This run's stage operation (including presentation): " + format_metric(current.seconds, digits=3, suffix=" s"))
        if selected not in (10, 11):
            for key, value in current.facts.items():
                st.markdown(f"**{escape(key)}** · {escape(str(value))}")
        if selected in (10, 11):
            result_summary(result)
        if selected == 11:
            st.markdown("**CORRESPONDING MATCH POINTS · REGISTERED PRODUCT · EVALUATION**" if registration_allowed(result)
                        else "**CORRESPONDENCE EVIDENCE · REGISTRATION WITHHELD · EVALUATION**")
        for artifact in current.images:
            st.markdown("**" + artifact.title + "**")
            show_artifact(artifact)
        if not current.images:
            st.caption("No optional visualization is available for this stage. The recorded evidence above remains available.")
        if selected == 11 and registration_allowed(result):
            path = run.directory / "registered_product.npz"
            if path.is_file():
                st.download_button("Download registered crop + mask + transform", path.read_bytes(), "registered_product.npz", "application/octet-stream")
        if selected == 11:
            final_products(result, expand_registered=False)
        st.caption("Fitted residuals are not geographic ground truth. Three-point fits lack redundancy. Scale normalization and Phase Congruency do not guarantee modality invariance; pretrained LoFTR has domain shift. LRO WAC is an external reference.")


def workspace(demo: Demo, images: tuple[Artifact, ...]) -> None:
    section("Observation workspace", "PREPARED SCIENTIFIC VISUALIZATIONS")
    if not images:
        st.caption("No prepared images are available for this case. Metrics and reports remain accessible.")
        return
    titles = [a.title for a in images]
    default = titles.index("Verified inliers") if "Verified inliers" in titles else 0
    chosen = st.selectbox("Visualization", titles, index=default, key="view_" + demo.key)
    show_artifact(next(a for a in images if a.title == chosen))
    if "Before / after" in titles:
        with st.expander("Registration comparison · before / after, overlay, checkerboard"):
            options = [a for a in images if a.title in {"Before / after", "Overlay", "Checkerboard"}]
            tabs = st.tabs([a.title for a in options])
            for tab, artifact in zip(tabs, options):
                with tab:
                    show_artifact(artifact)


def education() -> None:
    with st.expander("How ASTRION works"):
        st.markdown("""**Metadata-guided localization** narrows the search to an approximate shared region.
**Illumination preprocessing** normalizes contrast using recorded settings.
**Physical scale normalization** brings sampling distances onto comparable grids.

**SIFT / RootSIFT** detect locations and describe their local structure.
**Phase Congruency** represents structural information; it does not guarantee matches.
**LoFTR** predicts learned correspondences; this demo reuses saved output.

**RANSAC / MAGSAC++** test geometric consistency; each experiment records its actual verifier.
**Spatial validation** checks whether evidence is distributed across the reference.
**Registration** applies a supported transform.
**Reliability assessment** checks explicit engineering criteria and withholds unsupported results.""")
    with st.expander("Scientific limitations · essential context"):
        st.markdown("""- Fitted residual RMSE is not absolute geographic accuracy.
- Three affine-defining points are not redundant validation.
- Physical scale normalization does not guarantee modality invariance.
- Phase Congruency does not guarantee correspondence.
- Pretrained LoFTR has terrestrial-to-lunar domain shift.
- LRO WAC is an external lunar reference, not Chandrayaan-2 imagery.
- One successful TMC case does not establish universal invariance.
- Insufficient-support results are deliberately rejected.
- IIRS representations are predetermined reductions of RAW DN, not calibrated radiance.""")


def main() -> None:
    st.set_page_config(page_title="ASTRION · Planetary correspondence", page_icon="◌", layout="wide")
    html("<style>" + (ROOT / "src/ui/style.css").read_text(encoding="utf-8") + "</style>")
    hero()
    section("Explore the observations", "DEMO / SAVED EXPERIMENTS")
    try:
        demos = discover_demos(ROOT)
        records = load_results(ROOT)
    except (OSError, ValueError) as exc:
        st.error("Prepared experiment records are unavailable. Restore results/final to open the demo.")
        with st.expander("Technical diagnostics"):
            st.text(str(exc))
        education()
        return
    if not demos:
        st.info("No supported demo experiments were found in the saved report.")
        return
    if len(demos) != len(DEMOS):
        st.caption("Some demo records are missing. Available cases remain selectable.")
    choice = st.segmented_control("Demo case", [d.key for d in demos], default=demos[0].key,
                                  format_func=lambda k: next(d.title for d in demos if d.key == k), key="demo")
    demo = next(d for d in demos if d.key == (choice or demos[0].key))
    result = records[demo.experiment]
    st.caption(demo.description)
    mode = st.segmented_control("Operating mode", ["Quick Result", "Watch Pipeline"], default="Quick Result", key="mode")
    images = artifacts(ROOT, demo, result)
    if mode == "Watch Pipeline":
        watch_pipeline(demo, result, images)
    else:
        result_summary(result)
        workspace(demo, images)
        final_products(result)
    if mode != "Watch Pipeline":
        quick_details(demo, result, images)
    html('<div class="footer"><span>ASTRION / LunarMatch</span><span>SIH26166 · Research demonstration · No endorsement implied</span></div>')


def quick_details(demo: Demo, result: dict, images: tuple[Artifact, ...]) -> None:
    """Keep reproducibility and educational material in Quick Result."""
    with st.expander("Source & reference · observation details"):
        input_cards(demo, result, images)
    with st.expander("Technical details & reproducibility"):
        st.caption("Read-only parameters from the selected record. Changing historical settings would require a new experiment.")
        st.json({"experiment_id": result.get("experiment_id"), "parameters": result["parameters"],
                 "provenance": result.get("provenance"), "timing": result["timing"],
                 "reliability": result["reliability"], "metrics": result["metrics"],
                 "recorded_events": result.get("events", [])})
    with st.expander("Custom input · requirements"):
        st.write("This presentation supports prepared demo results. Custom execution is not enabled.")
        st.caption("Scientific processing requires a supported sensor product, matching metadata, valid-pixel masks, physical sampling information and an appropriate reference. Arbitrary image uploads cannot supply those requirements.")
    education()
    section("Take the evidence with you", "VALIDATED FINAL REPORTS / ALL NINE EXPERIMENTS")
    for col, (title, (name, payload, mime)) in zip(st.columns(3), downloads(ROOT).items()):
        col.download_button(title, payload, name, mime, width="stretch")


if __name__ == "__main__":
    main()
