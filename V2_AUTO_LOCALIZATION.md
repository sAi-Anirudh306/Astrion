# ASTRION V2 Experiment 1A

**Automatic metadata-guided localization: PASS, supported by new correspondence
evidence. Registration: AVAILABLE.** The original Borrow K TMC product and the
large WAC mosaic produced an automatic reference window, 989 candidates and 736
inliers. No prepared reference crop or saved v1 correspondences entered this run.
Independent absolute localization accuracy remains unmeasured.

Work is confined to branch `astrion-v2-auto-localization`. The annotated tag
`astrion-v1-sih-submission` still resolves to commit
`8148206668028e79b5ed179bb941b086828b34be`. The nine-experiment v1 matrix and
`results/final` remain authoritative for the original submission.

## Audit findings

The audit read AGENTS.md, README.md, PROJECT_SPEC.md, PLAN.md,
FINAL_VALIDATION.md and REPOSITORY_CLEANUP.md, and inspected ingestion,
georeferencing, preprocessing, features, matching, evaluation and preparation
scripts. PLAN.md retains the original milestone checklist; the later final
validation and cleanup documents describe the completed v1 implementation.

The requested observation differs from the v1 positive control:

| Item | Requested Borrow K | Validated v1 Tycho |
|---|---|---|
| TMC product | `ch2_tmc_nrf_20260629T2059373111_d_img_d18` | `ch2_tmc_nrf_20211122T2123225722_d_img_d18` |
| Geographic location | Approximately 66.22–69.77 degrees north, 10.44–13.76 degrees east | Approximately 42.60–44.00 degrees south, 348.37–349.59 degrees east |
| Source selection | Entire original 21513 x 4000 product | Prepared rows `[115734,124259)` of a different original product |
| Old reference bounds | No validated Borrow K reference crop found | WAC columns `[14797,15118)`, rows `[29132,29445)` |

The Tycho source was previously selected by latitude `[-44,-42.6]` using
`load_tmc_latitude_region` / `scripts/extract_tmc_region.py`. Its matching WAC
window came from interpolated corners for those selected rows, sampled along
edges and padded by one pixel in `extract_wac_region`. An earlier, broader
`crop_lroc_reference.py` uses manually specified Tycho geographic bounds; that
broader crop is not the 313 x 321 reference used by the 569-inlier result.
The v1 map script consumed the already prepared source and reference. Its result
did not demonstrate automatic localization from the entire Borrow K observation.

The original Borrow K label is authoritative: raw unsigned little-endian
16-bit, 172104000 bytes, 6.27 m nominal native GSD, orbit 30533,
Sun azimuth 192.648459 degrees and elevation 25.906933 degrees. Its
`Selenographic` field supplies geographic footprint context; the raw array has
no raster affine geotransform. The four geographic corners are interpreted as
pixel centers by the existing approximate bilinear model.

The actual WAC header declares 34748 x 34748, uint8, nodata 0, 100 m square map
pixels, lunar orthographic projection centered at 0 N / 0 E with sphere radius
1737400 m, and raster bounds `[-1737400,-1737400,1737400,1737400]` metres.
This orthographic product displays one visible hemisphere, not the full globe.
Its scanline storage can cause GDAL to read enclosing blocks; the application
does not allocate the full mosaic. Overview decimation is display only.

## Implementation and method

`localize_tmc_reference` accepts a TMC label, WAC mosaic, new output path and
margin. There is no old-crop, latitude-selection or manual-pixel-bounds argument.
It uses the full product's four corners and 257 samples per boundary edge,
projects through the existing lunar Rasterio/PROJ implementation, rounds the
bounds outward and reads the resulting native-resolution WAC window. The
footprint must be visible and fully inside the raster. Only excess safety
margin may be clipped; partial footprint coverage is an error.

The fixed default margin is **5 km in the WAC map plane (50 pixels per side)**.
It was selected before inference as a modest buffer, not calibrated from
spacecraft uncertainty or tuned for inlier count. No margin sweep was run.
Surface distances differ from map distances near the orthographic limb.

The existing TMC loader now offers validated, read-only memory mapping.
One native float32 processing copy uses unchanged default preprocessing.
The existing footprint mapper samples both scientific DN and its separate
processing representation onto the new WAC window, preserving float32 DN,
NaN nodata and an internal mask. Native scientific values are never quantized
to uint8 or replaced by a resized display.

The nominal scalar GSD factor is `6.27/100 = 0.0627`. Reusing
`normalize_resolution` gives a 1349 x 251 native-scale diagnostic with effective
x/y sampling 99.920319 / 99.99 m. This is not the matching geometry. The v1
native-to-WAC bilinear projection accounts for map geometry and scale; both
matching arrays are 505 x 473 at 100 m map spacing. Subsequent normalization
on those grids has factor 1. No second native-GSD downsample or arbitrary
aspect-ratio resize is applied. Bilinear point sampling can alias, as in v1.

One **new** local pretrained outdoor LoFTR inference was executed, with confidence
threshold 0.20. The existing v1 common-valid-mask policy and four-neighbor
endpoint validity check were retained. That policy limits correspondence to the
predicted overlap even though the reference extraction includes a margin.
Mode B neighborhood refinement is future work; this is not global image-only
localization. No training, weight download, parameter sweep or threshold change
was performed.

Affine RANSAC retains threshold 3 reference pixels, 2000 iterations,
confidence 0.99, 10 refinement iterations and seed 0. The existing M18 gate
requires at least 12 candidates, 12 inliers, 6/16 occupied cells, ratio 0.25,
condition number at most 10 and maximum residual at most 3 pixels, with a
nondegenerate model. Scientific registration and comparison figures are written
only after that shared gate accepts.

## Measured result

| Measurement | Result |
|---|---|
| Predicted WAC bounds before margin, `[left,top,right,bottom]` | `[18539.096097,1071.864413,18911.232643,1475.326546]` |
| Automatic integer ROI | `[18489,1021,18962,1526]` |
| ROI shape, height x width | 505 x 473 |
| ROI projected bounds, `[left,bottom,right,top]` m | `[111500,1584800,158800,1635300]` |
| Margin clipped | No |
| Valid WAC pixels | 234818 |
| Raw model-emitted matches | 1202 |
| Invalid endpoint rejections | 213 |
| Candidates after mask and confidence filtering | 989 |
| Verified inliers | 736 |
| Inlier ratio | 74.4186046512% |
| Fitted residual RMSE | 1.3594343839587484 WAC ROI pixels |
| Occupied reference grid cells | 13/16 |
| Shared reliability decision | RELIABLE |
| Registration | AVAILABLE |
| Localization / preparation | 0.708 s / 9.353 s |
| New CUDA inference | 3.666 s |
| Matching including model setup and verification | 9.169 s |
| Total including protection and visualization | 64.504 s |
| Peak GPU allocation reported by matcher | 886934016 bytes; no CPU fallback |

The predicted geographic corner coordinates (longitude, latitude) are:

| Corner | Coordinates in degrees |
|---|---|
| Upper left | `(11.181851,69.769058)` |
| Upper right | `(13.760836,69.675647)` |
| Lower left | `(10.437497,66.298497)` |
| Lower right | `(12.674241,66.217692)` |

The Tycho crop was opened for evaluation only after the automatic result and
registration decision existed. Its IoU with the new ROI is 0, coverage is 0,
and center separation is 28267.261081 map pixels (2826.726108 map km).
Those are descriptive differences between unrelated locations, **not Borrow K
localization error** or a lunar surface distance. Independent localization
error in pixels/km is explicitly null. The engineering PASS is supported by
traceable metadata, new distributed inliers and accepted registration; it does
not satisfy an unavailable same-observation crop ground-truth comparison.
Pre-run preservation checks hash historical files but do not feed their
coordinates, pixels or saved correspondence evidence to localization/matching.

## Outputs and reproduction

All scientific experiment outputs are in `results/v2_auto_localization/`:

- `experiment.json`, `localization.json`, `predicted_footprint.json`, `scale.json`
- `automatic_roi.tif`, `source_projected.tif`, `prepared_pair.npz`
- `new_inference_raw.npz`, `matches.npz`, `match_points.csv`, `metrics.json`
- `registered.tif`, `registered.png`, `overlay.png`, `checkerboard.png`
- `reference_overview.png`, `automatic_roi.png`, `source_native_scale.png`,
  `source_prepared.png`, `reference_prepared.png`, `correspondences.png`,
  `inliers.png`, `experiment_overview.png`
- `localization_evaluation.json`, `provenance.json`, `implementation.patch`,
  `protection_before.json`, `output_hashes.json`, `report.txt`

The presentation overview was visually inspected along with the actual ROI,
prepared source, hemisphere overview and checkerboard. The GeoTIFF preserves
interpolated scientific DN; the PNGs are display figures. Correspondence
visualizations use deterministic subsets with full counts labeled.

Run from the repository root with its existing virtual environment and local
scientific bundle / model checkpoint. The completed directory is protected
against overwrite; a subsequent authorized reproduction needs a new child name:

```powershell
.venv\Scripts\python.exe -m scripts.auto_localization --output results/v2_auto_localization/reproduction_01
```

The default invocation used for this experiment was
`.venv\Scripts\python.exe -m scripts.auto_localization`.
Configuration, complete parsed source metadata, reference CRS, footprints,
window bounds, native/map GSD, software versions, branch/commit, actual input
SHA256 hashes, source-code hashes, thresholds, runtimes and output hashes are
recorded. GPU results can vary across software/hardware despite fixed seeds.

## Validation

Focused tests: 16 localization tests, 25 TMC ingestion tests (including three
new mapping tests), and 4 existing WAC tests passed. The initial localization
test run exposed GDAL's separate projection-domain exception type; it was
wrapped with a clear ingestion error and the focused suite passed afterward.
Tests cover lunar coordinates, longitude seam, densified boundaries, margin
rounding/clipping, invisible/out-of-bounds failure, unchanged native window
values/masks/grid, original-data preservation, metadata-only selection, scale
handling, unrelated-crop metric semantics, provenance hashes, overwrite refusal
and accepted/withheld registration paths. No existing tests were weakened.

The full existing suite was run **once: 474 tests passed in 51.368 seconds**,
with zero failures, errors or skips (455 existing plus 19 new). The preserved v1
package validator passed all nine experiments, including points, rasters,
scientific metrics, manifest and ZIP. Existing Rasterio deprecation and
Streamlit context notices were non-fatal.

The final audit passed: all **1959 pre-existing scientific/evidence files**,
including **95 data files**, retained size/mtime and applicable small-file hashes.
All **92 files in `results/final` are byte-identical** to their pre-implementation
SHA256 inventory. The actual TMC raw product, XML, full WAC mosaic and model
checkpoint were fully SHA256-checked again. Other giant archival data were not
fully rehashed. The original annotated v1 tag and HEAD commit remain unchanged.

The audit independently checked the metadata-derived window against the full
WAC pixels, mask and geotransform; rebuilt the exact endpoint/confidence filter
from the new raw inference; recomputed fitted residuals and the unchanged
reliability gate; and replayed the saved affine warp to verify the registered
float32 raster and mask exactly. This validation performed no new inference or
model fitting. Execution source-code hashes and all run output hashes agree.
An initial audit assertion compared Windows backslash inventory keys with
forward-slash keys; normalizing path separators fixed the audit comparison.
No final-package bytes had changed.

The machine-readable receipt is
`results/v2_auto_localization/validation/completion_audit.json`; focused/full
test logs, v1 validator output, copied baseline inventories and the audit script
are alongside it. The pre-implementation inventory is also retained in
`results/v2_auto_localization_audit/`; it predates the run and is itself included
in the run's preservation inventory.

Tracked changes are README.md, the existing metrics, georeferencing,
image-loader documentation, TMC loader and WAC loader modules, and TMC ingestion
tests. New source files are this report, `scripts/auto_localization.py` and
`tests/test_auto_localization.py`. Existing outputs and crops were not deleted,
renamed or overwritten. Git status shows seven modified files and three new
files, all unstaged. No commit or tag was created.

Scientific limits remain: approximate four-corner geolocation, map projection
foreshortening, bilinear aliasing, inherited mask constraints, terrestrial
pretraining, no independent geographic controls, one observation and one margin.
No claim of global image-only localization, universal robustness or absolute
subpixel accuracy follows from this experiment. No next experiment, commit or
tag is authorized by this result.
