"""ASTRION pitch-oriented prepared-results UI.

Launch from the repository root with:
    .venv\Scripts\python.exe -m streamlit run app.py

This UI does not change the validated scientific pipeline. It presents the
existing prepared evidence as a cleaner, video-friendly workflow.
"""
import base64
from html import escape
from io import BytesIO
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    raise SystemExit(
        "Install the optional UI: .venv\\Scripts\\python.exe -m pip install -r requirements-ui.txt"
    )

from PIL import Image, UnidentifiedImageError

from src.ui.service import (
    Artifact,
    Demo,
    artifacts,
    discover_demos,
    downloads,
    final_product,
    format_metric,
    load_results,
    metric_cards,
    registration_allowed,
    reliability,
    safe_path,
    supporting_metadata,
)

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
    """Display one generated/saved visualization without modifying its source."""
    try:
        data = display_image(
            str(artifact.path), artifact.path.stat().st_mtime_ns, width
        )
        st.image(data, caption=artifact.origin, width="stretch")
    except (OSError, ValueError, UnidentifiedImageError):
        st.caption(
            "This optional visualization is unavailable. Recorded metrics remain accessible."
        )


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


@st.cache_data(max_entries=4, show_spinner=False)
def comparison_views(
    products: tuple[tuple[str, int], ...],
) -> tuple[Image.Image, Image.Image]:
    """Blend saved moving/registered rasters with their reference in memory only."""
    import numpy as np
    import rasterio
    from src.evaluation.visualization import display_image as scientific_display

    if len(products) != 3:
        raise ValueError("Comparison requires moving, reference and registered products")
    displays = []
    grid = None
    for path, _modified in products:
        with rasterio.open(path) as dataset:
            if dataset.count != 1 or dataset.width * dataset.height > 1_000_000:
                raise ValueError("Expected a small prepared single-band comparison product")
            current_grid = (dataset.shape, dataset.transform, dataset.crs)
            if grid is not None and current_grid != grid:
                raise ValueError("Comparison products must share the saved reference grid")
            grid = current_grid
            displays.append(scientific_display(dataset.read(1), dataset.read_masks(1) > 0))

    moving, reference, registered = displays
    # Identical support in both views prevents nodata edges from implying improvement.
    valid = ~np.logical_or.reduce([np.ma.getmaskarray(view) for view in displays])
    if not valid.any():
        raise ValueError("Comparison products have no common valid pixels")
    views = []
    for view in (moving, registered):
        blend = np.ma.filled((view + reference) * 0.5, 0)
        pixels = np.repeat((blend * 255).astype(np.uint8)[..., None], 3, axis=2)
        pixels[~valid] = (215, 207, 227)
        views.append(Image.fromarray(pixels))
    return views[0], views[1]


def show_overlay_comparison(result: dict) -> None:
    """Compare saved registration evidence, falling back when the UI extra is absent."""
    if not registration_allowed(result):
        return
    try:
        paths = [final_product(ROOT, result, key) for key in (
            "moving_product", "reference_product", "registered_product"
        )]
        if any(path is None for path in paths):
            raise ValueError("Prepared comparison products are missing")
        before, after = comparison_views(tuple(
            (str(path), path.stat().st_mtime_ns) for path in paths
        ))
    except (OSError, ValueError):
        st.caption("Interactive comparison unavailable. See the saved overlay below.")
        return

    try:
        from streamlit_image_comparison import image_comparison
    except ImportError:
        st.caption("Interactive slider unavailable; showing both views side by side.")
        for column, view, label in zip(st.columns(2), (before, after), (
            "Before Registration", "After Registration"
        )):
            column.image(view, caption=label, width="stretch")
    else:
        st.caption("Drag left or right to compare alignment against the same LRO WAC reference.")
        # Juxtapose uses intrinsic image dimensions: enlarge only the RGB display
        # copies, with a height cap for unusually tall inputs. Saved rasters stay intact.
        scale = min(900 / before.width, 900 / before.height)
        display_size = (max(1, round(before.width * scale)),
                        max(1, round(before.height * scale)))
        comparison_before = before.resize(display_size, Image.Resampling.LANCZOS)
        comparison_after = after.resize(display_size, Image.Resampling.LANCZOS)
        # Match both the Streamlit element and iframe to the display aspect ratio.
        # The package's fixed height otherwise leaves gaps on narrow screens.
        html(
            "<style>.st-key-registration_slider "
            '[data-testid="stElementContainer"]:has(iframe) {'
            "position:relative; width:100% !important; height:auto !important;"
            "flex:0 0 auto !important; min-height:0;"
            f"aspect-ratio:{display_size[0]} / {display_size[1]};"
            "}.st-key-registration_slider iframe {"
            "position:absolute; inset:0; display:block; width:100%; height:100% !important; border:0;"
            "}</style>"
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            with st.container(key="registration_slider", width=display_size[0]):
                image_comparison(
                    comparison_before, comparison_after,
                    label1="Before Registration", label2="After Registration",
                    width=display_size[0], starting_position=50,
                    make_responsive=True, in_memory=True,
                )
        # Juxtapose can initialize at thumbnail size while its tab is hidden.
        # Scope fluid sizing to this iframe, including its initial load/reloads.
        st.html("""
            <script>
            (() => {
                const fit = frame => {
                    const doc = frame.contentDocument;
                    if (!doc?.head || doc.getElementById('astrion-slider-fit')) return;
                    const style = doc.createElement('style');
                    style.id = 'astrion-slider-fit';
                    style.textContent = `
                        html, body { margin:0; width:100%; height:100%; overflow:hidden; }
                        #foo { width:100% !important; height:100% !important; }
                        .jx-slider { width:100%; height:100%; }
                        .jx-image img { height:100% !important; width:auto !important; }
                    `;
                    doc.head.appendChild(style);
                };
                const bind = () => {
                    const frame = document.querySelector('.st-key-registration_slider iframe');
                    if (!frame) return;
                    frame.addEventListener('load', () => fit(frame));
                    fit(frame);
                    observer.disconnect();
                };
                const observer = new MutationObserver(bind);
                observer.observe(document.body, {childList:true, subtree:true});
                bind();
            })();
            </script>
        """, unsafe_allow_javascript=True)
    st.caption(
        "50% reference blends of the saved source (before) and registered source (after). "
        "Display normalization only; lavender marks pixels outside common valid support."
    )


def html(value: str) -> None:
    st.markdown(value, unsafe_allow_html=True)


def hero() -> None:
    """Keep the existing ASTRION hero artwork and visual identity."""
    html(
        '<div class="nav"><span class="wordmark">ASTRION</span>'
        '<span class="nav-note">LUNARMATCH &nbsp; / &nbsp; PLANETARY VISION</span></div>'
    )

    image_html = ""
    try:
        path = safe_path(ROOT, "assets/astrion_hero.png", area="assets")
        encoded = base64.b64encode(
            display_image(str(path), path.stat().st_mtime_ns, 1672)
        ).decode("ascii")
        image_html = (
            '<img alt="Moon and orbiting satellite illuminated by a restrained '
            f'violet orbital arc" src="data:image/webp;base64,{encoded}">'
        )
    except (OSError, ValueError):
        st.caption(
            "Hero artwork unavailable. The scientific demo is still accessible below."
        )

    html(
        '<section class="hero">'
        + image_html
        + '<div class="hero-copy">'
        '<div class="eyebrow">From orbital imagery to measured evidence</div>'
        '<h1>Different views.<br>Common ground.</h1>'
        '<p class="tagline">Planetary Image Correspondence<br>&amp; Registration</p>'
        '<p>Explore how ASTRION connects lunar observations across sensors, '
        'illumination and scale, and recognizes when the evidence is insufficient.</p>'
        '<div class="hero-foot">CHANDRAYAAN-2 &nbsp; / &nbsp; EXTERNAL LRO WAC REFERENCE</div>'
        "</div></section>"
    )


def section(title: str, note: str = "") -> None:
    html(
        f'<div class="section"><h2>{escape(title)}</h2>'
        f'<span class="muted">{escape(note)}</span></div>'
    )


def input_cards(demo: Demo, result: dict, images: tuple[Artifact, ...]) -> None:
    """Show the actual source/reference pair before pipeline execution."""
    config = result["parameters"].get("config") or {}
    meta = result["metadata"]
    supplemental = supporting_metadata(ROOT, demo)
    by_name = {a.title: a for a in images}

    sensors = {
        "tmc": "Chandrayaan-2 TMC-2",
        "ohrc": "Chandrayaan-2 OHRC",
        "iirs": "Chandrayaan-2 IIRS",
    }

    for col, side, sensor, gsd in zip(
        st.columns(2),
        ("Source", "Reference"),
        (sensors[demo.key], "LRO WAC"),
        (config.get("source_gsd"), config.get("reference_gsd")),
    ):
        with col, st.container(border=True):
            html(f'<div class="eyebrow">{side}</div>')
            st.markdown(f"**{sensor}**")

            if side in by_name:
                show_artifact(by_name[side], width=660)
            else:
                st.caption("Prepared preview unavailable")

            st.caption(
                f"Recorded input GSD · {format_metric(gsd, suffix=' m/pixel')}"
            )

            scale = (
                meta.get("source_scale", meta.get("scale", {}))
                if side == "Source"
                else meta.get("reference_scale", {})
            )
            if scale.get("original_shape"):
                st.caption(
                    "Input dimensions · "
                    + " × ".join(str(x) for x in scale["original_shape"])
                    + " (rows × columns)"
                )

            if side == "Source":
                st.caption(meta.get("modality", "Modality not recorded"))
                if supplemental.get("product"):
                    st.caption("Product · " + supplemental["product"])
            else:
                st.caption("External lunar reference · not Chandrayaan-2 imagery")

    if demo.key == "iirs":
        p = supplemental
        st.caption(
            f"Hyperspectral · {format_metric(p.get('bands'), digits=0)} bands · "
            f"{format_metric(p.get('wavelength_min'), digits=1)}–"
            f"{format_metric(p.get('wavelength_max'), digits=1)} "
            f"{p.get('wavelength_units') or ''} · RAW DN, not calibrated radiance"
        )


def result_summary(result: dict) -> None:
    """Final reliability and engineering metrics."""
    label, color = reliability(result)
    html(
        f'<div class="section"><h2>Final result</h2>'
        f'<span class="status {color}">● {escape(label)}</span></div>'
    )

    config = result["parameters"].get("config") or {}
    st.caption(
        f"{config.get('source_sensor', 'N/A')} ↔ "
        f"{config.get('reference_sensor', 'N/A')}  /  "
        f"{config.get('representation', 'N/A')}  /  "
        f"{result['parameters'].get('matcher') or 'N/A'}"
    )

    html(
        '<div class="metric-grid">'
        + "".join(
            f'<div class="metric-card"><div class="metric-label">{escape(k)}</div>'
            f'<div class="metric-value">{escape(v)}</div></div>'
            for k, v in metric_cards(result).items()
        )
        + "</div>"
    )

    st.caption(
        "RMSE is a fitted correspondence residual in reference processing pixels, "
        "not absolute geographic accuracy."
    )

    if not registration_allowed(result):
        reason = (
            "; ".join(result["reliability"].get("reasons", []))
            or "No reliable transform is recorded."
        )
        html(
            '<div class="withheld"><strong>REGISTRATION WITHHELD</strong>'
            "<p>Verified evidence does not satisfy the recorded reliability criteria.</p>"
            f"<p>{escape(reason)}</p></div>"
        )

    if result["metrics"].get("verified_inliers") == 3:
        st.caption(
            "Three affine-defining points provide no redundant validation; "
            "a small fitted residual does not establish a reliable alignment."
        )


def final_products(result: dict, *, expand_registered: bool = True) -> None:
    """Expose the corresponding-point exports and reliability-gated raster."""
    for key, title in (
        ("match_points", "Download corresponding match points (CSV)"),
        ("verified_inlier_points", "Download verified inliers (CSV)"),
    ):
        path = final_product(ROOT, result, key)
        if path is not None and path.is_file():
            st.download_button(
                title, path.read_bytes(), path.name, "text/csv", key="final_" + key
            )

    if registration_allowed(result):
        path = final_product(ROOT, result, "registered_product")
        if path is not None and path.is_file():
            panel = (
                st.expander("Registered scientific product · float32 GeoTIFF")
                if expand_registered
                else st.container()
            )
            with panel:
                try:
                    st.image(
                        registered_preview(str(path), path.stat().st_mtime_ns),
                        caption="Display normalization of the actual final registered raster",
                        width="stretch",
                    )
                    st.download_button(
                        "Download registered GeoTIFF",
                        path.read_bytes(),
                        path.name,
                        "image/tiff",
                        key="registered_tif",
                    )
                except (OSError, ValueError):
                    st.caption("Registered product preview unavailable.")


def stage_header(title: str, subtitle: str, status: str = "PROCESSING") -> None:
    """Large pitch-friendly status panel. No synthetic step numbers."""
    html(
        '<div class="process-shell">'
        f'<div class="process-kicker">{escape(status)}</div>'
        f'<div class="process-title">{escape(title)}</div>'
        f'<div class="process-subtitle">{escape(subtitle)}</div>'
        '<div class="process-line"><span></span></div>'
        "</div>"
    )


def watch_pipeline(demo: Demo, result: dict, images: tuple[Artifact, ...]) -> None:
    """One-click pitch pipeline using the existing prepared-data controller."""
    from src.ui.pipeline import TITLES, advance_pipeline, build_pipeline_run

    section("Run ASTRION", "PITCH PIPELINE")

    run_key = "pitch_run_" + demo.key
    done_key = "pitch_done_" + demo.key

    if run_key not in st.session_state:
        st.caption(
            "ASTRION will move through the validated prepared pipeline and save "
            "presentation artifacts under results/ui_pipeline/<run-id>/."
        )
        input_cards(demo, result, images)

        left, right = st.columns([1, 2])
        if left.button(
            "Run ASTRION",
            type="primary",
            width="stretch",
            key="start_pipeline_" + demo.key,
        ):
            try:
                st.session_state[run_key] = build_pipeline_run(ROOT, demo, result)
                st.session_state[done_key] = False
                st.rerun()
            except Exception as exc:
                st.error("Unable to start ASTRION: " + str(exc))

        with right:
            st.caption(
                "The active process replaces the previous one on screen. "
                "At completion, all generated evidence is collected into a final gallery."
            )
        return

    run = st.session_state[run_key]
    live = st.empty()
    progress = st.empty()

    if not st.session_state.get(done_key, False):
        try:
            for index, title in enumerate(TITLES):
                percent = int((index / max(1, len(TITLES))) * 100)
                progress.progress(
                    index / len(TITLES),
                    text=f"{percent}% · {title}",
                )

                current = advance_pipeline(run, index)

                with live.container():
                    stage_header(
                        current.title,
                        current.explanation,
                        "ASTRION PROCESSING",
                    )

                    if current.images:
                        featured = current.images[0]
                        show_artifact(featured)

                    if current.facts:
                        facts = list(current.facts.items())[:4]
                        cols = st.columns(len(facts))
                        for col, (key, value) in zip(cols, facts):
                            with col:
                                st.metric(str(key), str(value))

                    st.caption(current.origin)
                    if current.method:
                        st.caption("Method · " + current.method)

            st.session_state[done_key] = True
            progress.progress(1.0, text="100% · Pipeline complete")

        except Exception as exc:
            progress.empty()
            st.error("Pipeline stopped: " + str(exc))
            st.caption("No unsupported registration has been created.")
            if st.button("Reset pipeline", key="reset_failed_" + demo.key):
                st.session_state.pop(run_key, None)
                st.session_state.pop(done_key, None)
                st.rerun()
            return

    if st.session_state.get(done_key, False):
        live.empty()
        progress.empty()

        html(
            '<div class="pipeline-done">'
            '<span>PIPELINE COMPLETE</span>'
            "<strong>ASTRION evidence ready</strong>"
            "<p>Correspondence, verification, registration and evaluation artifacts "
            "have been collected below.</p></div>"
        )

        result_summary(result)

        final_stage = run.outputs[-1]
        if final_stage.images:
            section(
                "Final visual evidence",
                "CORRESPONDENCE / REGISTRATION / EVALUATION",
            )
            tabs = st.tabs([a.title for a in final_stage.images])
            for tab, artifact in zip(tabs, final_stage.images):
                with tab:
                    show_artifact(artifact)

        final_products(result, expand_registered=False)

        with st.expander("View all pipeline stages and saved visuals"):
            for index, output in enumerate(run.outputs):
                status_class = (
                    "good"
                    if output.status == "COMPLETE"
                    else "warning"
                    if output.status in {"WARNING", "SKIPPED"}
                    else "failed"
                )
                html(
                    '<div class="timeline-row">'
                    f'<div class="timeline-dot {status_class}"></div>'
                    "<div>"
                    f'<span class="timeline-index">{index + 1:02d}</span>'
                    f"<strong>{escape(output.title)}</strong>"
                    f"<p>{escape(output.explanation)}</p>"
                    f"<small>{escape(output.origin)} · "
                    f'{escape(format_metric(output.seconds, digits=3, suffix=" s"))}'
                    "</small></div></div>"
                )

                if output.images:
                    tabs = st.tabs([a.title for a in output.images])
                    for tab, artifact in zip(tabs, output.images):
                        with tab:
                            show_artifact(artifact, width=1100)

        run_path = run.directory.relative_to(ROOT)
        st.caption(f"Presentation artifacts saved to · {run_path}")

        a, b = st.columns(2)
        if a.button("Run again", key="run_again_" + demo.key, width="stretch"):
            st.session_state.pop(run_key, None)
            st.session_state.pop(done_key, None)
            st.rerun()

        if b.button(
            "Open Quick Result", key="open_quick_" + demo.key, width="stretch"
        ):
            st.session_state["mode"] = "Quick Result"
            st.rerun()

        st.caption(
            "Fitted residuals are not geographic ground truth. "
            "Registration remains reliability-gated."
        )


def workspace(demo: Demo, images: tuple[Artifact, ...], result: dict) -> None:
    section("Visual evidence", "PREPARED SCIENTIFIC OUTPUTS")

    if not images:
        st.caption(
            "No prepared images are available for this case. Metrics remain accessible."
        )
        return

    titles = [a.title for a in images]
    default = titles.index("Verified inliers") if "Verified inliers" in titles else 0
    chosen = st.selectbox(
        "Visualization", titles, index=default, key="quick_view_" + demo.key
    )
    show_artifact(next(a for a in images if a.title == chosen))

    comparison = [
        a
        for a in images
        if a.title in {"Before / after", "Overlay", "Checkerboard"}
    ]
    if comparison:
        with st.expander("Registration comparison"):
            tabs = st.tabs([a.title for a in comparison])
            for tab, artifact in zip(tabs, comparison):
                with tab:
                    if artifact.title == "Overlay":
                        show_overlay_comparison(result)
                        st.markdown("**Final Registration Overlay**")
                    show_artifact(artifact)


def discover_local_products() -> list[Path]:
    """Discover user-added products without recursively scanning giant files on startup."""
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    products: list[Path] = []
    for path in sorted(raw.iterdir(), key=lambda p: p.name.lower()):
        if path.is_dir():
            products.append(path)
        elif path.is_file() and path.suffix.lower() in {
            ".img",
            ".xml",
            ".tif",
            ".tiff",
            ".hdr",
        }:
            products.append(path)
    return products


def local_data_panel() -> None:
    """Read-only discovery panel for new local products."""
    section("Local scientific data", "DATA/RAW")

    st.caption(
        "For large Chandrayaan products, place the product folder inside data/raw "
        "instead of uploading multi-GB files through the browser."
    )

    a, b = st.columns([1, 3])
    if a.button("Rescan data/raw", key="rescan_raw", width="stretch"):
        st.rerun()

    with b:
        st.code(str(ROOT / "data" / "raw"), language=None)

    products = discover_local_products()

    if not products:
        st.info(
            "No local products are visible under data/raw yet. "
            "Add a product folder and press Rescan data/raw."
        )
        return

    selected = st.selectbox(
        "Detected local product",
        products,
        format_func=lambda p: p.name,
        key="local_product",
    )

    with st.container(border=True):
        st.markdown("**" + selected.name + "**")

        if selected.is_dir():
            scientific = [
                p
                for p in selected.rglob("*")
                if p.is_file()
                and p.suffix.lower()
                in {".img", ".xml", ".tif", ".tiff", ".hdr"}
            ]
            st.caption(f"{len(scientific)} scientific / metadata files detected")

            if scientific:
                st.code(
                    "\n".join(
                        str(p.relative_to(ROOT)) for p in scientific[:10]
                    ),
                    language=None,
                )
                if len(scientific) > 10:
                    st.caption(f"+ {len(scientific) - 10} additional files")
        else:
            st.caption(str(selected.relative_to(ROOT)))

    reference = ROOT / "data" / "reference" / "lroc" / "WAC_GLOBAL_O000N0000_100M.TIF"
    if reference.is_file():
        st.success("LRO WAC reference detected")
    else:
        st.warning("Expected LRO WAC reference was not found.")

    st.warning(
        "New-product execution is not connected in this UI pass. "
        "This panel discovers local data without pretending that an arbitrary product "
        "has been scientifically processed. Use Prepared Demo for the current pitch run."
    )


def education() -> None:
    with st.expander("How ASTRION works"):
        st.markdown(
            """**Metadata-guided localization** narrows the search to an approximate shared region.

**Preprocessing and physical scale normalization** prepare comparable processing grids.

**SIFT / RootSIFT / LoFTR** generate correspondence evidence, depending on the recorded experiment.

**RANSAC / MAGSAC++** test geometric consistency.

**Spatial validation** checks whether verified evidence is distributed across the reference.

**Registration** is produced only when the recorded reliability gate permits it."""
        )

    with st.expander("Scientific limitations"):
        st.markdown(
            """- Fitted residual RMSE is not absolute geographic accuracy.
- Three affine-defining points are not redundant validation.
- Scale normalization does not guarantee modality invariance.
- Pretrained LoFTR has terrestrial-to-lunar domain shift.
- LRO WAC is an external lunar reference.
- One successful TMC case does not establish universal invariance.
- Insufficient-support results are deliberately rejected."""
        )


def quick_details(demo: Demo, result: dict, images: tuple[Artifact, ...]) -> None:
    with st.expander("Source & reference"):
        input_cards(demo, result, images)

    with st.expander("Technical details & reproducibility"):
        st.caption(
            "Read-only parameters from the selected validated experiment."
        )
        st.json(
            {
                "experiment_id": result.get("experiment_id"),
                "parameters": result["parameters"],
                "provenance": result.get("provenance"),
                "timing": result["timing"],
                "reliability": result["reliability"],
                "metrics": result["metrics"],
                "recorded_events": result.get("events", []),
            }
        )

    education()

    section("Downloads", "VALIDATED FINAL REPORTS")
    for col, (title, (name, payload, mime)) in zip(
        st.columns(3), downloads(ROOT).items()
    ):
        col.download_button(title, payload, name, mime, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="ASTRION · Planetary correspondence",
        page_icon="◌",
        layout="wide",
    )

    html(
        "<style>"
        + (ROOT / "src/ui/style.css").read_text(encoding="utf-8")
        + "</style>"
    )

    hero()

    workspace_mode = st.segmented_control(
        "Workspace",
        ["Prepared Demo", "Local data/raw"],
        default="Prepared Demo",
        key="workspace_mode",
    )

    if workspace_mode == "Local data/raw":
        local_data_panel()
        html(
            '<div class="footer"><span>ASTRION / LunarMatch</span>'
            "<span>Local scientific workspace · discovery only</span></div>"
        )
        return

    section("Chandrayaan-2 TMC-2 ↔ LRO WAC", "VALIDATED PREPARED EVIDENCE")

    try:
        demos = discover_demos(ROOT)
        records = load_results(ROOT)
    except (OSError, ValueError) as exc:
        st.error(
            "Prepared experiment records are unavailable. "
            "Restore results/final to open the demo."
        )
        with st.expander("Technical diagnostics"):
            st.text(str(exc))
        return

    demo = next((d for d in demos if d.key == "tmc"), None)
    if demo is None:
        st.info("The validated TMC-2 ↔ LRO WAC case is unavailable in the saved report.")
        return

    result = records[demo.experiment]
    images = artifacts(ROOT, demo, result)

    st.caption(demo.description)

    mode = st.segmented_control(
        "View",
        ["Quick Result", "Show Pipeline"],
        default="Quick Result",
        key="mode",
    )

    if mode == "Show Pipeline":
        watch_pipeline(demo, result, images)
    else:
        result_summary(result)
        workspace(demo, images, result)
        final_products(result)
        quick_details(demo, result, images)

    html(
        '<div class="footer"><span>ASTRION / LunarMatch</span>'
        "<span>SIH26166 · Research demonstration · No endorsement implied</span></div>"
    )


if __name__ == "__main__":
    main()
