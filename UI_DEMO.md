# ASTRION prepared-results UI

Launch from `D:\Codes\SIH`:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://127.0.0.1:8501>. The server binds to localhost. If the optional
dependency is absent, install it into the existing environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

## Scope and architecture

The UI originated after checkpoint `0e4e74e`; the submission cleanup connects
its final presentation to the validated M21 package.
`app.py` renders Streamlit controls and figures. `src/ui/service.py` reads the
final package's version-1 result contract and packaged figures/exports.
Final records preserve M18/M20 metrics, reliability policy, parameters,
provenance and orchestration events. Quick Result reads those records. Watch
Pipeline uses the M20 import runner and a prepared-data controller in
`src/ui/pipeline.py`; it never performs new matching or learned inference.
Scientific modules and historical results are unchanged.

Quick Result and the last Watch Pipeline stage expose corresponding-point CSVs.
The reliable TMC case also exposes verified-inlier CSVs and the actual float32
registered GeoTIFF, with a separate normalized display preview. Negative cases
do not expose registration downloads. Missing final records do not fall back
to M20 metrics. The complete final package remains unchanged on disk.

The interface uses dark neutral surfaces, small violet accents, restrained
semantic status indicators, compact metric cards and a selectable pipeline
grid. It uses system fonts and the existing `assets/astrion_hero.png`.
Bounded WebP derivatives are generated and cached in memory; the original hero
is unchanged. JSON snapshots are cached by path, modification time and size.
No raw strips, IIRS cube or neural model are loaded on startup.

## Demonstration

The default case is **TMC-2 ↔ WAC / TMC saved LoFTR**. `results/final` is the metric source
of truth. Its saved decision is RELIABLE. OHRC saved intensity and IIRS mean
intensity are INSUFFICIENT SUPPORT; registration is withheld, even when a
minimal affine fit exists.

**Quick Result** shows the sensor pair, representation, reliability, six metrics,
runtime scope and an artifact selector. TMC views include source, reference,
candidates, verified inliers, spatial distribution, residuals, before/after,
overlay and checkerboard. Registration figures are linked only when the M8
receipt's transform and support counts agree with the selected result.

OHRC includes prepared input/reference, scale-normalized image, prior features
and saved candidate visualization. IIRS includes the predetermined band-mean
preview, reference, scaled image, prior features and spectrum. M17 preparation
figures are explicitly distinguished from M18 correspondence metrics; the UI
does not mislabel M17 inlier images as M18 results.

**Watch Pipeline** starts with actual source/reference cards and a Begin button.
Previous, Next, stage selection and Show final output expose twelve stages.
The controller imports evidence through M20, verifies agreement with the saved
result and translates import completion into a compact evidence status. Each
stage publishes RUNNING only during real work, then its resulting state.
Completed stages are cached in the session and revisiting them performs no work.

For TMC, prepared raster hashes are checked against M8 before reading the small
crops. Existing M18 `prepare_representation`, common `normalize_resolution`,
and `warp_affine` functions perform actual lightweight operations. The TMC crop
is already on a 100 m grid, so scale normalization truthfully produces an
unchanged grid. Registration uses the recorded reliable affine without refitting.
The registered scientific float32 crop, validity mask and transform are available
as an NPZ download. Saved final-package metrics remain the scientific source of truth.

The newly computed preprocessing demonstration is **not** the historical LoFTR
intermediate: that experiment normalized native imagery before map sampling.
This boundary is stated in the stage. LoFTR input displays, candidates, verified
inliers and their classification remain explicitly recorded evidence; inference
and model fitting are not repeated. OHRC/IIRS preparation stages use recorded
evidence because isolated scientific intermediate arrays are unavailable without
additional raw-product access. PNG previews are never scientific pipeline inputs.

New per-run visual artifacts include TMC ingestion, footprint, preprocessing
before/after, scale comparison, registration and overlay/checkerboard; a spatial
count grid for every case; and IIRS M18 paired candidate coordinate plots.
The IIRS per-point M18 inlier mask is unavailable: geometric verification shows
recorded counts and candidate locations with that limitation, never a fabricated
inlier classification. Historical M17 feature figures retain their prior-context
labels. All figures are generated from real arrays, metadata or numeric evidence.

Unique directories under `results/ui_pipeline/<run-id>/` contain figures,
registered output where permitted, and stage JSON receipts with methods, inputs,
hashes, M20 provenance/events and measured operation durations. These durations
include visualization and are separate from historical benchmark runtimes.
No historical files are overwritten. Failures stop advancement; reset creates
a fresh run. Raw products, training and expensive inference are excluded.

Final ASTRION Output combines corresponding match evidence, the registered
product, overlay/checkerboard and metrics. Negative cases end with their actual
support evidence and registration WITHHELD. Watch Pipeline has no bottom
accordion stack or raw JSON. Brief scientific caveats remain visible. Quick
Result retains the approved presentation, observation details, educational
sections and consolidated Technical details & reproducibility including raw
metrics, provenance and historical events.

Observation details contain two source/reference cards. IIRS metadata includes
the 256-band RAW DN product, recorded wavelength range, and predetermined M17
single-band and band-mean definitions with explicit index conventions.
Advanced parameters, provenance, educational explanations and scientific
limitations are expandable. Downloads serve the final-package JSON, text summary
and comparison CSV covering all nine experiments.

Custom execution/upload is intentionally unavailable in this presentation.
The requirements section explains the sensor metadata, masks, sampling and
reference information needed for scientific processing. There is no algorithm
selection that silently changes a historical experiment.

## Data and limitations

The demo needs the prepared local `results/` directories, which are ignored by
Git. Cloning code alone does not supply scientific results. The main required
file is `results/final/experiments.json`; final figures and exported products live
inside that package. Watch Pipeline still uses M8/M16/M17/M18 evidence and the
prepared scientific TMC crops for its intermediate stages.
Missing optional figures are omitted; corrupt/unreadable images produce a
compact fallback. Missing/invalid final records produce a friendly error with
expandable diagnostics. The UI exposes only its fixed supported catalog.

TMC and OHRC historical runtime values measure saved-support replay, not full
LoFTR inference or raw processing. IIRS runtime is the recorded M18 classical
execution. M20 import duration remains separately available in provenance.
Fitted residual RMSE is not absolute geographic accuracy. Three affine-defining
points provide no redundant validation. Neither physical scale normalization,
Phase Congruency nor one positive learned control establishes modality or
planetary invariance. The UI preserves these limitations prominently.

The figure association checks are lightweight and are not a new complete hash
audit of every historical artifact. No external ground truth is introduced.
Unavailable historical timings are never inferred from the measured durations
of new presentation operations. This is a hybrid prepared-data walkthrough,
not a new end-to-end inference experiment.

## Validation

### Watch Pipeline revision

The focused UI suite passed **39 tests** (24 service, six Streamlit interaction,
nine prepared-run integration tests). The full suite was run once for this
revision: **427 passed in 72.547 seconds**. A final presentation-only adjustment
removed duplicate final metrics and shortened registration panel titles; the
six interaction tests were then repeated. Rendered Edge validation walks all
twelve TMC stages, checks loaded images, tests the IIRS insufficient-support
final state, verifies no Watch accordions and confirms Quick Result technical
details remain. New screenshots and the browser receipt live in
`results/ui_watch_validation/`.

Tests cover output descriptors/order, generated paths and receipts, M20 events,
no duplicate execution on revisiting stages, explicit historical origin,
sensor/demo consistency, reliable registered output, negative gating, final
metrics, bounded scientific input sizes and the revised information hierarchy.

### Initial UI validation

Focused tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_ui.py -v
.venv\Scripts\python.exe -m unittest discover -s tests -p test_ui_app.py -v
```

The initial 24 service tests cover discovery, ordering, loading, malformed records,
formatting, nonfinite values, reliability, registration suppression, safe paths
(including simulated symlink escape), downloads, missing artifacts, real
prepared cases and bounded startup reads. Five optional Streamlit AppTest
tests cover all case/mode combinations, stage selection, hero and image output,
missing results/artifacts and corrupt image fallback. They do not compare pixels.
AppTest tests skip when Streamlit or local prepared results are unavailable.

Completion run (once):
`.venv\Scripts\python.exe -m unittest discover -s tests -q`
passed **417 tests in 32.326 seconds** (388 existing + 29 added), with no
failures or skips. Existing Rasterio pending-deprecation notices and expected
AppTest context notices were non-fatal. Classification: **READY** for the
prepared-results presentation; custom/live execution is outside this UI scope.

The actual server was launched and returned HTTP 200. A hidden Edge session
verified rendered image loading, case/mode interactions, zero rendered
exceptions and no page-level horizontal overflow at 1920, 1366 and 390 pixels.
Manual visual inspection used the captured landing, pipeline and success
screenshots. Evidence is in ignored `results/ui_validation/`, including
`browser_validation.json` and screenshots. Browser tooling is validation-only;
it adds no application dependency.

## Dependencies installed

Direct optional UI dependency: `streamlit==1.64.0`, recorded in
`requirements-ui.txt`, following the existing optional requirements-file style.
Pillow was already installed. No existing dependency was upgraded or removed.
The installer added these required transitive packages:

```text
altair==6.3.0                 anyio==4.15.1
charset_normalizer==3.5.1     h11==0.16.0
httptools==0.8.0              idna==3.20
itsdangerous==2.2.0           jsonschema==4.26.0
jsonschema-specifications==2025.9.1
narwhals==2.26.0              pandas==3.0.6
protobuf==7.36.2              pyarrow==25.0.1
pydeck==0.9.3                 python-multipart==0.0.32
referencing==0.37.0           requests==2.34.2
rpds-py==2026.6.3             starlette==1.7.0
toml==0.10.2                 tzdata==2026.4
urllib3==2.8.0                uvicorn==0.54.0
watchdog==6.0.0               websockets==16.1.1
```

`pip check` passes. No packaging migration was performed.

## Files for review and commit

- `app.py`
- `src/ui/service.py` and `src/ui/style.css`
- `src/ui/pipeline.py`
- `.streamlit/config.toml`
- `requirements-ui.txt`
- `tests/test_ui.py` and `tests/test_ui_app.py`
- `tests/test_ui_pipeline.py`
- `UI_DEMO.md`
- User-provided `assets/astrion_hero.png` (previously untracked; unchanged)

No pre-existing tracked source file was modified. Do not commit the temporary
Edge profile or validation artifacts under `results/ui_validation/`.
