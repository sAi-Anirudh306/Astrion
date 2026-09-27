# ASTRION / LunarMatch Technical Specification

## 1. PROJECT IDENTITY

Project:
ASTRION

Correspondence system:
LunarMatch

SIH:
Smart India Hackathon 2026

Problem Statement:
SIH26166

Original problem:
Multi-modal, Sun angle and scale invariant image correspondence
using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS).

---

# 2. CORE PROBLEM

Given two planetary remote-sensing images that may differ in:

- illumination
- Sun angle
- spatial resolution
- scale
- viewing geometry
- sensor characteristics
- modality

identify reliable corresponding locations and estimate the geometric
relationship between the images.

The initial target is Chandrayaan-2 lunar imagery.

The software architecture should be extensible to other planetary bodies.

---

# 3. SENSOR TYPES

## OHRC

High-resolution optical imagery.

Expected role:
Detailed surface correspondence.

Challenges:
Large scale differences and potentially different illumination.

---

## TMC-2

Terrain Mapping Camera.

Expected role:
Coarser optical/terrain mapping imagery.

Challenges:
Resolution differences, viewing geometry, illumination.

---

## IIRS

Imaging Infrared Spectrometer.

Expected role:
Cross-modal correspondence.

Challenges:
Spectral representation and large modality differences.

---

# 4. SYSTEM PIPELINE

Input
↓
Metadata
↓
Sensor-specific ingestion
↓
Common image representation
↓
Preprocessing
↓
Multi-scale representation
↓
Feature extraction
↓
Candidate correspondence generation
↓
Candidate filtering
↓
Geometric verification
↓
Spatial distribution analysis
↓
Registration
↓
Sub-pixel refinement
↓
Evaluation
↓
Visualization

---

# 5. INGESTION

The ingestion layer must:

- understand sensor-specific data formats
- read PDS4 metadata
- validate dimensions
- validate data types
- validate file sizes
- preserve raw scientific values
- provide common interfaces to downstream stages

Sensor-specific loaders:

tmc_loader.py
ohrc_loader.py
iirs_loader.py

Generic dispatcher:

image_loader.py

Metadata parser:

metadata.py

---

# 6. TMC RAW FORMAT

Borrow K raw TMC:

Height: 21513
Width: 4000
Data type: UnsignedLSB2
Bytes: 172104000
Resolution: 6.27 m/pixel

UnsignedLSB2 means little-endian unsigned 16-bit.

Expected size:

21513 × 4000 × 2
= 172104000 bytes

The loader must validate this.

The raw file must not be interpreted through cv2.imread().

---

# 7. PREPROCESSING

Preprocessing must distinguish:

raw data
from
algorithm representation
from
visualization representation.

Potential operations:

- percentile normalization
- contrast normalization
- denoising
- local normalization
- gradient representations
- illumination-aware transformations

Preprocessing must be configurable.

No irreversible transformation should be silently applied to the raw data.

---

# 8. FEATURE EXTRACTION

## SIFT

Initial classical feature method.

Outputs:

- keypoints
- descriptors

Parameters should be configurable.

---

## RootSIFT

RootSIFT is derived from SIFT descriptors through:

1. L1 normalization
2. element-wise square-root transformation

It is a descriptor representation, not a separate detector.

---

## Phase Congruency

Structural representation intended to reduce dependence on raw intensity.

Use as an additional branch.

---

## Learned Features

Potential later methods:

- SuperPoint
- LoFTR

These must integrate into the common correspondence representation.

---

# 9. MATCHING

Descriptor matching:

- BFMatcher initially
- optional FLANN later

Filtering:

- KNN matching
- Lowe ratio test

Correspondence results should contain:

- coordinates in image A
- coordinates in image B
- descriptor distances
- match indices
- filtering statistics

---

# 10. GEOMETRIC VERIFICATION

Initial method:

RANSAC.

Potential models:

- affine
- homography

Later:

MAGSAC++.

Output:

- transformation matrix
- inlier mask
- candidate count
- inlier count
- inlier ratio

An inlier is a geometrically consistent correspondence, not automatically
a scientifically confirmed correct correspondence.

---

# 11. REGISTRATION

Use the verified transformation to align images.

Outputs may include:

- transformation matrix
- registered image
- overlay
- difference visualization

Never overwrite source imagery.

---

# 12. SPATIAL CONTROL

Analyze whether reliable correspondences are distributed throughout
the useful overlap.

Important distinction:

Many matches concentrated in one small region

versus

reliable matches distributed across the scene.

Potential techniques:

- regular grids
- spatial coverage
- clustering analysis
- quadtree subdivision

---

# 13. SUB-PIXEL REFINEMENT

Performed after coarse correspondence and geometric verification.

Must use an actual refinement method.

Do not claim sub-pixel accuracy merely because coordinates are stored
as floating-point numbers.

---

# 14. EVALUATION

Metrics may include:

- keypoint count
- candidate match count
- ratio-test match count
- inlier count
- inlier ratio
- reprojection error
- transformation error when ground truth exists
- runtime
- spatial coverage
- image dimensions
- processing scale

No fabricated ground truth.

---

# 15. EXPERIMENT LEVELS

Level 1:
same/similar sensor and similar conditions

Level 2:
same sensor with different illumination

Level 3:
large scale difference

Level 4:
OHRC ↔ TMC-2

Level 5:
optical ↔ IIRS

---

# 16. MVP

The first complete working system is:

TMC input
↓
preprocessing
↓
SIFT
↓
RootSIFT
↓
descriptor matching
↓
ratio test
↓
RANSAC
↓
registration
↓
metrics
↓
visualization

This is the minimum end-to-end baseline.

---

# 17. SCIENTIFIC VALIDITY

The system should distinguish:

Implementation:
What the software does.

Experiment:
What was tested.

Observation:
What was measured.

Interpretation:
What the measurements may indicate.

Do not convert an implementation feature into a scientific claim.

For example:

Correct:
"Phase Congruency was implemented as an illumination-robust structural
representation."

Not automatically correct:
"Phase Congruency makes the system illumination invariant."

The latter requires experimental evidence.

---

# 18. OUTPUTS

Experiment outputs should be stored under:

results/experiment_x/

Potential files:

metadata.json
run_config.json
keypoints_a.png
keypoints_b.png
candidate_matches.png
inlier_matches.png
registered.png
overlay.png
metrics.json

---

# 19. SOFTWARE PRINCIPLES

Prefer:

modular
testable
reproducible
sensor-aware
scientifically defensible
configurable

Avoid:

hard-coded assumptions
duplicate implementations
silent data conversion
silent failure
fake metrics
unverified scientific claims