# ASTRION / LunarMatch

ASTRION is the SIH26166 planetary image correspondence and registration
submission. It ingests Chandrayaan-2 TMC-2, OHRC and IIRS products and evaluates
their correspondence with an external LRO WAC lunar reference. The architecture
separates sensor ingestion from processing, matching and evaluation; the
validated experiments are lunar, not evidence of general planetary invariance.

## Validated result

The final matrix contains **9 experiments: 1 reliable registration and 8 withheld
registrations**. The TMC saved-LoFTR positive control has 677 candidate matches,
569 verified inliers, an 84.0473% inlier ratio, 1.164206529 px fitted residual
RMSE and support in 15 of 16 spatial cells. Its final package contains an actual
float32 registered GeoTIFF, candidate and verified-inlier CSVs, overlays and
checkerboards. These headline numbers summarize the saved evidence; application
logic reads the package rather than hard-coding them.

**Fitted residual RMSE is not absolute geographic accuracy.** There are no
independent geographic control points for these experiments. Reliability is an
engineering support decision. The OHRC and IIRS demos retain
`INSUFFICIENT_SUPPORT` and withhold registration, even where a minimal fitted
model has small residuals. The full matrix also retains the negative TMC cases.

## Architecture and repository

```text
Sensor ingestion -> Metadata/geolocation -> Preprocessing -> Scale representation
 -> Feature detection/description -> Correspondence candidates -> Filtering
 -> Geometric verification -> Spatial support -> Registration -> Evaluation
 -> Visualization
```

SIFT/RootSIFT, descriptor and ratio matching, RANSAC/MAGSAC++, spatial control,
Phase Congruency and optional LoFTR are separate components. Classical processing
does not require PyTorch or a GPU. Sub-pixel refinement remains an unimplemented
placeholder; no sub-pixel accuracy claim is made.

```text
app.py                         Streamlit entry point
README.md                      Setup and reproduction
PROJECT_SPEC.md / PLAN.md       Scientific specification and milestone history
FINAL_VALIDATION.md             M21 validation and package contract
UI_DEMO.md                      UI behavior and historical validation
REPOSITORY_CLEANUP.md           Audit, deletion reasons and cleanup verification
AGENTS.md                      Repository development/data rules
requirements*.txt              Core, optional UI, optional learned dependencies
.streamlit/  assets/            UI configuration and hero artwork
configs/                       Established nine-experiment import matrix
src/
  ingestion/ preprocessing/ features/ matching/
  geometry/ spatial/ evaluation/ ui/
scripts/                       Final/experiment runners and preparation utilities
tests/                         Scientific, packaging and UI regression tests
data/                          Ignored local scientific inputs and derivatives
models/                        Ignored optional saved model weights
results/                       Ignored evidence, packages and generated UI runs
```

Historical scientific scripts are retained because they explain preparation or
are imported by later scripts/tests. They are not the submission launch path.

## Installation

The validated workstation uses Windows and **Python 3.11.9**, with the environment
at `D:\Codes\SIH\.venv`. Run commands from the repository root. For a new checkout:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.venv\Scripts\python.exe -m pip check
```

Use the existing environment when it is already installed. `requirements.txt`
records the tested core versions, including `opencv-contrib-python`; avoid
installing a second competing OpenCV distribution. Streamlit is declared in
`requirements-ui.txt`. `requirements-loftr.txt` is only needed for fresh learned
inference. Its CUDA installation note describes the original workstation;
another machine must choose an appropriate PyTorch installation. Saved final
evidence, package reproduction and the prepared UI do not rerun LoFTR or require
its weights. No packaging migration or environment rebuild was performed.

## Local data and evidence

Git does **not** contain the scientific datasets, prepared inputs or final results.
Obtain the validated local data/evidence bundle from the project maintainer and
retain these paths. Do not overwrite, rename or delete original products.

| Location | Role |
|---|---|
| `data/raw/tycho/` | Original TMC product from 2021-11-22, image and PDS4 label |
| `data/raw/ohrc/` | Original 2021-04-02 product, label, geometry and auxiliary files |
| `data/raw/iirs/` | Original 2021-06-28 cube, HDR, PDS4 label and auxiliary files |
| `data/reference/lroc/` | Original `WAC_GLOBAL_O000N0000_100M.TIF` |
| `data/test/` | Original Borrow K and other retained ingestion/integration data |
| `data/processed/tycho/` | Native crop, common WAC grid, map-projected TMC and registration |
| `data/processed/ohrc/reference/` | Validated OHRC-region WAC crop |
| `data/processed/iirs/reference/` | Validated IIRS-region WAC crop |

The clean prepared set is a **logical manifest**, `prepared_inputs.json` in each
new final package. It records original products, exact windows, GSD, processing
receipts and file identities without duplicating or recropping imagery. TMC uses
rows `[115734,124259)`; OHRC uses rows `[40978,49170)` and all 12000 columns;
IIRS uses rows `[12571,13083)` and all 250 columns. IIRS keeps zero-based band 17
and the predetermined mean of bands 12–23. These selections are unchanged.
Small inputs have freshly checked SHA256 values. Large originals/intermediates
use explicitly labelled size/mtime inventory; this is not a new full-content
hash audit of those files.

OHRC/IIRS native prepared source arrays were not saved as independent scientific
files. Their validated preparation receipts and M18 correspondence evidence are
explicit reproduction inputs. Display PNGs are not scientific inputs.

Required historical evidence includes M8 registration, M9 evaluation, M10
figures, M16 OHRC preparation, M17 IIRS preparation, M18 cross-modal results,
M20 experiment records and `experiment_tycho_map_projected` saved points/figures.
The manifest enumerates the exact paths. Keep the retained historical results:
the original M21 `protection.json` also checks earlier files by size/mtime and,
for small files, SHA256. Preserve file timestamps when restoring that archival
package. For relocated/restored files with different timestamps, build a fresh
package and compare its science to the preserved M21 package.

## Reproduce and validate

The existing final builder imports the complete established nine-experiment
matrix, exports saved coordinates, copies unchanged scientific rasters and
figures, generates coordinate plots, inventories the prepared input set and
seals a validated manifest/ZIP. It performs no new inference or parameter sweep.

One command builds, validates and compares against the authoritative M21 package:

```powershell
.venv\Scripts\python.exe scripts/final_validation.py --build --output results/final_clean_reproduction --compare-to results/final
```

The output directory must not exist; select a new name for later reproductions.
`results/final/` remains the authoritative, preserved package used by the UI.
`results/final_m21_preserved/` is the additional unchanged local backup created
during cleanup. No automatic package promotion or overwrite occurs.

Compare exact scientific records, historical runtimes, match-point exports,
rasters, figures and comparison CSV. Execution IDs, timestamps, measured import
durations, absolute provenance paths, local protection inventory and the added
input manifest can differ; consequently manifest/ZIP hashes can differ too.
Each ZIP is checked against its own manifest and payload.

```powershell
.venv\Scripts\python.exe scripts/final_validation.py --output results/final
.venv\Scripts\python.exe scripts/final_validation.py --output results/final_clean_reproduction --compare-to results/final
```

See [FINAL_VALIDATION.md](FINAL_VALIDATION.md) for the package layout and validator
checks. Archive delivery includes `ASTRION_Final_Deliverable.zip`; the code,
environment and historical evidence required for rebuilding are separate.

## Run the demonstration

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Select a prepared TMC, OHRC or IIRS demo, then **Quick Result** or **Watch
Pipeline**. Demo selection controls sensor identity. Quick Result reads final
metrics, figures and downloads from `results/final`, including the actual TMC
registered GeoTIFF and corresponding-point CSVs. A missing final package is an
explicit error; M20 metrics are not silently substituted.

Watch Pipeline retains its twelve stages. It checks the historical M18/M20
import against the selected final record and executes lightweight TMC prepared
crop operations where supported. Intermediate evidence stays in its proper
historical location; computed walkthrough products are labelled separately.
The recorded final metrics and final exports remain authoritative. OHRC/IIRS
end with registration withheld. Arbitrary custom-file execution is not enabled.

## Tests and scientific limits

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The original 447 tests are retained. Cleanup adds eight regression tests for
reproduction, input traceability and final-package UI authority (455 total).
Some integration tests need the local scientific bundle; UI tests require
Streamlit. See the cleanup report for the actual completion run and skip count.

The footprint mapping is approximate, not rigorous orthorectification. Three
affine-defining matches provide no redundant validation. Physical scale
normalization and Phase Congruency do not guarantee cross-modal correspondence.
Pretrained LoFTR has terrestrial-to-lunar domain shift. OHRC's 0.26 m sampling
and WAC's 100 m sampling have a severe scale gap. IIRS reductions contain RAW DN,
not calibrated radiance. One successful TMC example establishes neither
sensor invariance nor universal robustness. Missing historical per-point inlier
masks are not invented; unsupported registrations remain withheld.
