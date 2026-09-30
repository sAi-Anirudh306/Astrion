# ASTRION / LunarMatch - Codex Instructions

## PROJECT

Project name: ASTRION
System name: LunarMatch
SIH Problem Statement: SIH26166

Goal:
Build a modular planetary image correspondence system for Chandrayaan-2
OHRC, TMC-2 and IIRS imagery.

The original SIH problem focuses on lunar imagery, but the architecture
must be extensible to planetary imagery beyond the Moon.

---

# 1. DEVELOPMENT MODE

Work milestone-by-milestone.

DO NOT attempt to implement the entire project in one operation.

For every milestone:

1. Inspect the existing implementation.
2. Implement only the requested milestone.
3. Run relevant tests.
4. Fix errors caused by your implementation.
5. Report exactly what changed.
6. Report tests and their results.
7. STOP.

Do not begin the next major milestone automatically.

The human developer will explicitly authorize the next milestone.

---

# 2. PROJECT ROOT

Project root:

D:\Codes\SIH

Python:

Python 3.11.9

Virtual environment:

D:\Codes\SIH\.venv

Use the project virtual environment when running Python commands.

---

# 3. EXISTING STRUCTURE

The project currently uses:

src/
├── evaluation/
│   ├── metrics.py
│   └── visualization.py
├── features/
│   ├── learned.py
│   ├── phase_congruency.py
│   ├── rootsift.py
│   └── sift.py
├── geometry/
│   ├── magsac.py
│   ├── ransac.py
│   ├── registration.py
│   └── subpixel.py
├── ingestion/
│   ├── iirs_loader.py
│   ├── image_loader.py
│   ├── metadata.py
│   ├── ohrc_loader.py
│   └── tmc_loader.py
├── matching/
│   ├── correspondence.py
│   ├── descriptor_matching.py
│   └── ratio_test.py
├── preprocessing/
│   ├── denoising.py
│   ├── illumination.py
│   ├── normalization.py
│   └── preprocessing.py
└── spatial/
    ├── quadtree.py
    └── spatial_distribution.py

Do not restructure this project unless there is a concrete technical reason.

Do not create duplicate modules with alternative names.

---

# 4. IMPORTANT DATA RULE

Original scientific data must NEVER be modified, overwritten, renamed,
or deleted by the implementation.

Original data exists under:

data/raw/
data/test/

Derived products may be written to:

data/processed/

Experiment outputs belong under:

results/

---

# 5. SENSOR ARCHITECTURE

Keep sensor-specific ingestion separate.

TMC-specific logic belongs in:

src/ingestion/tmc_loader.py

OHRC-specific logic belongs in:

src/ingestion/ohrc_loader.py

IIRS-specific logic belongs in:

src/ingestion/iirs_loader.py

Generic dispatch belongs in:

src/ingestion/image_loader.py

PDS4 metadata parsing belongs in:

src/ingestion/metadata.py

Do not put sensor-specific binary parsing into generic matching code.

---

# 6. CURRENT METADATA IMPLEMENTATION

src/ingestion/metadata.py already contains a PDS4 XML metadata parser.

Preserve its useful functionality.

Do not replace working metadata extraction unnecessarily.

The parser handles fields including:

- logical_identifier
- version_id
- title
- start_date_time
- stop_date_time
- purpose
- processing_level
- target
- instrument
- job_id
- imaging_orbit_number
- dumping_orbit_number
- spacecraft_altitude_km
- pixel_resolution_m_per_pixel
- roll_deg
- pitch_deg
- yaw_deg
- sun_azimuth_deg
- sun_elevation_deg
- solar_incidence_deg
- projection
- area
- geographic footprint
- file_name
- file_size_bytes
- md5_checksum
- data_type
- image_height
- image_width

Improve it only when required by testing or integration.

---

# 7. BORROW K TEST DATA

Real Chandrayaan-2 TMC Borrow K data is available.

Browse image:

data/test/borrow_k/browse/raw/20260629/
ch2_tmc_nrf_20260629T2059373111_b_brw_d18.png

Browse metadata:

data/test/borrow_k/browse/raw/20260629/
ch2_tmc_nrf_20260629T2059373111_b_brw_d18.xml

Raw image:

data/test/borrow_k/data/raw/20260629/
ch2_tmc_nrf_20260629T2059373111_d_img_d18.img

Raw metadata:

data/test/borrow_k/data/raw/20260629/
ch2_tmc_nrf_20260629T2059373111_d_img_d18.xml

---

# 8. BORROW K RAW IMAGE FACTS

The PDS4 metadata declares:

Height:
21513

Width:
4000

Data type:
UnsignedLSB2

Expected byte size:
172104000 bytes

Pixel resolution:
6.27 m/pixel

Projection:
Selenographic

The raw file is little-endian unsigned 16-bit data.

It is NOT a normal PNG/JPEG/TIFF image.

Do not use cv2.imread() to load the raw .img file.

The expected NumPy interpretation is conceptually:

dtype = little-endian unsigned 16-bit

The actual implementation must derive dimensions and data type
from metadata rather than hard-coding them.

Validate the actual file size before reading.

Expected size:

21513 * 4000 * 2 = 172104000 bytes

If the file size does not match metadata, raise a clear error.

---

# 9. DATA REPRESENTATION

Maintain a distinction between:

1. Raw scientific data
2. Processing representation
3. Visualization representation

Do not silently convert scientific raw data to uint8.

A uint8 visualization may be generated separately.

Do not use a display-resized image as a substitute for scientific input.

---

# 10. COMMON ARCHITECTURE

The intended pipeline is:

INGESTION
    ↓
METADATA
    ↓
PREPROCESSING
    ↓
MULTI-SCALE REPRESENTATION
    ↓
FEATURE EXTRACTION
    ↓
CORRESPONDENCE GENERATION
    ↓
CANDIDATE FILTERING
    ↓
GEOMETRIC VERIFICATION
    ↓
SPATIAL MATCH CONTROL
    ↓
REGISTRATION
    ↓
SUB-PIXEL REFINEMENT
    ↓
EVALUATION
    ↓
VISUALIZATION

Keep these stages conceptually separate.

---

# 11. SCIENTIFIC TERMINOLOGY

Do not confuse:

Feature detection:
Finding interesting image locations.

Feature description:
Representing local image structure numerically.

Correspondence matching:
Finding candidate relationships between image features.

Geometric verification:
Determining whether candidate correspondences agree with a geometric model.

Spatial match control:
Determining whether reliable correspondences are adequately distributed.

Registration:
Estimating/applying the transformation between images.

Sub-pixel refinement:
Improving localization beyond integer-pixel coordinates.

Evaluation:
Quantifying performance.

---

# 12. CLASSICAL BASELINE

The first complete baseline should use:

- SIFT
- RootSIFT
- descriptor matching
- Lowe ratio test
- RANSAC
- registration
- metrics
- visualization

The classical pipeline must work without GPU or PyTorch.

---

# 13. ADVANCED METHODS

Later milestones may include:

- MAGSAC++
- Phase Congruency
- multi-scale processing
- spatial distribution
- quadtree
- OHRC
- IIRS
- LoFTR
- SuperPoint
- GPU acceleration

Do not implement these before the classical baseline is functional unless explicitly instructed.

---

# 14. SCIENTIFIC CLAIMS

Do not claim that an algorithm is:

- scale invariant
- illumination invariant
- sensor invariant
- cross-modal
- sub-pixel accurate
- planetary-general
- robust

unless there is experimental evidence supporting the claim.

Use language such as:

"designed to improve robustness"

when experimental validation is not yet available.

---

# 15. GROUND TRUTH

Never invent ground truth.

If ground truth is unavailable:

- clearly label metrics as relative or unsupervised
- do not report fake accuracy
- do not fabricate registration error
- do not fabricate known transformations

---

# 16. ERROR HANDLING

Use clear errors for:

- missing files
- invalid XML
- unsupported data types
- dimension mismatch
- file-size mismatch
- unsupported sensors
- invalid image data

Avoid silent failure.

---

# 17. CODE QUALITY

Use:

- Python type hints
- dataclasses where appropriate
- docstrings
- meaningful names
- small functions
- modular code
- configurable parameters

Avoid:

- global mutable state
- duplicated implementations
- unexplained magic numbers
- giant functions
- giant modules
- unnecessary abstraction

---

# 18. TESTING

Every major module must eventually have tests.

Use synthetic data for algorithmic unit tests when appropriate.

Use real Borrow K data for integration testing.

Do not consider a module complete merely because it imports successfully.

---

# 19. EXPERIMENT OUTPUTS

Experiments should eventually produce outputs such as:

results/
└── experiment_x/
    ├── metadata.json
    ├── run_config.json
    ├── keypoints_a.png
    ├── keypoints_b.png
    ├── candidate_matches.png
    ├── inlier_matches.png
    ├── registered.png
    ├── overlay.png
    └── metrics.json

Do not overwrite previous experiments.

---

# 20. REPRODUCIBILITY

Record:

- input files
- sensor
- metadata
- preprocessing parameters
- feature parameters
- matching parameters
- geometric model
- thresholds
- runtime
- match counts
- inlier counts
- metrics
- software versions when practical

Use deterministic seeds where appropriate.

---

# 21. IMPLEMENTATION PRIORITY

Follow this order unless the human developer explicitly changes it:

1. TMC ingestion
2. generic image loading
3. preprocessing
4. SIFT
5. RootSIFT
6. descriptor matching
7. ratio test
8. RANSAC
9. registration
10. metrics
11. visualization
12. spatial distribution
13. MAGSAC++
14. Phase Congruency
15. multi-scale processing
16. OHRC ingestion
17. IIRS ingestion
18. learned matching
19. advanced experiments

---

# 22. MILESTONE BEHAVIOR

At the end of every task report:

FILES CHANGED:
- ...

IMPLEMENTATION:
- ...

TESTS RUN:
- ...

RESULT:
- PASS / FAIL

REMAINING ISSUES:
- ...

NEXT MILESTONE:
- ...

Then STOP.

Do not automatically continue.

---

# 23. CURRENT TASK

M21 is complete. The latest authorized task is final repository cleanup and
clean reproduction, preserving the nine-experiment validated scientific matrix.
See README.md, FINAL_VALIDATION.md and REPOSITORY_CLEANUP.md for the current
layout, reproduction command and audit. The structure above records the original
development baseline; the implementation has since advanced through M21.

Do not start another scientific milestone, arbitrary custom-file execution or
ASTRION v2 without explicit authorization. Do not commit or tag automatically.
