# ASTRION Milestone 21 final deliverable

The final package is `results/final/`. It contains all nine configurations from
`configs/milestone_20.json`, imported through the existing M20 runner from M18.
There is one reliable TMC/WAC registration and eight deliberately withheld
registrations. No new inference, matching, band selection or threshold tuning
was performed. The completed package was copied unchanged from the validated
M21 staging directory; copying did not create a new scientific execution.

The package and datasets are ignored by Git. A source checkout alone does not
include the scientific evidence. The ZIP contains results, not the repository,
raw datasets or model weights. Keep the repository and its Python environment
with the results for executable software delivery.

## Validate or reproduce

Run from `D:\Codes\SIH` with the existing virtual environment:

```powershell
.venv\Scripts\python.exe scripts/final_validation.py --output results/final
.venv\Scripts\python.exe -m unittest discover -s tests -p test_final_validation.py -v
```

To reproduce packaging, choose a **new** output directory. Existing packages
and historical results are never overwritten:

```powershell
.venv\Scripts\python.exe scripts/final_validation.py --build --output results/final_reproduction
```

This imports all nine experiments, exports recorded points, copies the scientific
rasters and linked historical figures, generates coordinate plots, and audits
the package. It does not rerun the historical scientific experiments.

The full-suite completion command is:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The UI remains on its validated M20 contract, which the final validator checks
against every final record. Quick Result and Watch Pipeline are unchanged:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Completion validation: all **20 focused M21 tests passed in 36.641 s**.
The full suite was run once after implementation: **447 tests passed in
56.307 s**, with no failures or skips. Existing Rasterio deprecation notices and
Streamlit AppTest context notices were non-fatal.

The UI smoke check launched the real Streamlit server, confirmed HTTP 200 for
the landing page and health endpoint, and exercised Quick Result and Watch
Pipeline through Streamlit AppTest for TMC, OHRC and IIRS. It verified TMC
candidate/inlier, registered comparison, overlay, checkerboard and metric views;
both negative cases retained withheld registration at final output. This was
an interaction smoke check, not browser pixel validation. Its receipt is
`results/m21_completion/ui_smoke.json`.

## Package guide

```text
results/final/
  experiments.json          # all standardized M20 records plus final references
  run_config.json           # complete established matrix
  comparison.csv            # one row for each of nine experiments
  summary.json
  summary.txt
  manifest.json             # payload sizes and SHA256 checksums
  protection.json           # pre-run original/history inventory
  validation_report.json
  validation_report.txt
  ASTRION_Final_Deliverable.zip
  evidence/                 # strict-JSON copies of M18/M20 and sensor receipts
  tmc_wac/
    tmc_saved_loftr/         # headline: scientific GeoTIFFs, points, figures
    tmc_intensity/
    tmc_phase/
  ohrc_wac/
    ohrc_saved_intensity/
    ohrc_saved_phase/
  iirs_wac/
    iirs_single_intensity/
    iirs_single_phase/
    iirs_mean_intensity/
    iirs_mean_phase/
```

Every experiment has `result.json`, `metrics.json`, `match_points.csv`, a
candidate-coordinate plot and a folder guide. Rejected cases additionally have
`registration_withheld.json`. A zero-candidate OHRC CSV contains its header only.
The six classical TMC/IIRS CSVs omit unavailable confidence and inlier labels.
The TMC LoFTR CSV preserves both, with a separate verified-inlier CSV and the
authoritative saved NPZ. Coordinate conventions and IIRS spectral indices are
recorded in each result.

The headline folder includes `moving_product.tif`, `reference_product.tif`,
`registered_product.tif`, `before_after.png`, `overlay.png`, `checkerboard.png`,
candidate/inlier figures and the M8 registration receipt. The registered product
is the actual lossless float32 raster, not a Matplotlib figure.

## Evidence and validation boundaries

The headline saved evidence has 677 candidates, 569 inliers, an inlier ratio of
0.8404726735598228, fitted residual RMSE of 1.1642065290124644 reference pixels,
and 15 occupied cells out of 16. These are read from authoritative records,
not hard-coded into validation. The validator recomputes TMC residuals and
spatial counts from saved points and checks against M8, M18 and M20. It also
replays the existing mask-aware affine warp and compares the registered raster
pixels, mask, shape, grid and CRS. It performs no model refit.

JSON validation rejects nonfinite numbers and duplicate keys. Original reports
remain untouched; nonfinite values in their packaged copies become null. CSVs,
per-experiment records, summaries, artifact paths, readable images, manifest
hashes, and ZIP names/content are checked. The manifest excludes itself and its
enclosing ZIP to avoid recursive hashes. ZIP ordering, timestamps and permissions
are deterministic for a fixed package; new runs have new IDs and measured times.

The archived validation report records the content audit performed before
sealing. The final validation command additionally validates the completed
manifest and ZIP without mutating the package.

Original data, prepared inputs and pre-existing result files are checked using
size and modification time. Files at most 8 MB additionally receive SHA256
checks. Giant raw products are not rescanned, so this is not a new cryptographic
verification of every byte of the raw archive. Browser profiles and Python cache
files are excluded from scientific protection. New UI runs use new directories.

Historical runtime has its original scope: classical execution or saved-support
replay, not newly measured LoFTR inference. Packaging runtime is separate.

Scientific limits remain: fitted residual RMSE is not absolute geographic
accuracy; TMC footprint mapping is approximate, not precise orthorectification;
three affine-defining points lack redundant validation; physical scale
normalization and Phase Congruency do not guarantee correspondence or modality
invariance; terrestrial pretrained LoFTR has lunar domain shift; LRO WAC is an
external reference; one positive case does not establish universal sensor
invariance. OHRC has a severe native scale gap. IIRS remains RAW DN, not
calibrated radiance. Insufficient support is a rejection, not registration success.
