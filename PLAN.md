# ASTRION Development Plan

Status:
IN PROGRESS

Development rule:

Complete one milestone.

Run its tests.

Inspect the result.

Only then authorize the next milestone.

---

# MILESTONE 1
## TMC INGESTION

Status:
CURRENT

Goal:

Reliably load real Chandrayaan-2 TMC Borrow K data.

Tasks:

[ ] Verify existing metadata parser
[ ] Implement TMC raw loader
[ ] Support UnsignedLSB2
[ ] Validate file size
[ ] Validate dimensions
[ ] Validate dtype
[ ] Load browse PNG
[ ] Create common image representation
[ ] Implement generic image loader
[ ] Add unit tests
[ ] Add Borrow K integration test

Input:

data/test/borrow_k/

Expected raw image:

21513 × 4000
little-endian uint16

Success criteria:

- metadata parses
- raw file size matches metadata
- image loads
- shape equals metadata
- dtype is correct
- browse image loads
- tests pass

DO NOT START MILESTONE 2 UNTIL THIS MILESTONE PASSES.

---

# MILESTONE 2
## PREPROCESSING

Goal:

Create configurable preprocessing.

Tasks:

[ ] normalization
[ ] percentile clipping
[ ] denoising
[ ] illumination representation
[ ] preprocessing pipeline
[ ] tests

Success criteria:

- preprocessing does not modify original data
- parameters are configurable
- outputs are reproducible
- tests pass

---

# MILESTONE 3
## SIFT

Goal:

Extract SIFT keypoints and descriptors.

Tasks:

[ ] implement SIFT wrapper
[ ] configurable parameters
[ ] result dataclass
[ ] unit tests
[ ] visualization

Success criteria:

- SIFT works on real TMC imagery
- keypoints are returned
- descriptors are returned
- visualization is produced

---

# MILESTONE 4
## ROOTSIFT

Goal:

Implement RootSIFT descriptor transformation.

Tasks:

[ ] L1 normalization
[ ] square-root transformation
[ ] zero handling
[ ] tests

Success criteria:

- valid descriptors
- no NaN/Inf
- deterministic output

---

# MILESTONE 5
## DESCRIPTOR MATCHING

Goal:

Generate candidate correspondences.

Tasks:

[ ] BFMatcher
[ ] KNN matching
[ ] structured match result
[ ] tests

---

# MILESTONE 6
## RATIO TEST

Goal:

Remove ambiguous descriptor matches.

Tasks:

[ ] Lowe ratio test
[ ] configurable threshold
[ ] statistics
[ ] tests
[ ] visualization

---

# MILESTONE 7
## RANSAC

Goal:

Perform robust geometric verification.

Tasks:

[ ] point extraction
[ ] affine model
[ ] homography model
[ ] RANSAC
[ ] inlier mask
[ ] transformation matrix
[ ] metrics
[ ] tests

Success criteria:

- transformation is estimated
- inliers are identified
- output is reproducible

---

# MILESTONE 8
## REGISTRATION

Goal:

Register one image to the other.

Tasks:

[ ] image warping
[ ] registered output
[ ] overlay
[ ] tests
[ ] visualization

---

# MILESTONE 9
## EVALUATION

Goal:

Produce quantitative experiment results.

Tasks:

[ ] keypoint metrics
[ ] match metrics
[ ] inlier ratio
[ ] reprojection error
[ ] runtime
[ ] JSON output

---

# MILESTONE 10
## VISUALIZATION

Goal:

Produce presentation-quality experiment outputs.

Tasks:

[ ] keypoints
[ ] candidate matches
[ ] inlier matches
[ ] registered image
[ ] overlay
[ ] metrics visualization

---

# MILESTONE 11
## SPATIAL DISTRIBUTION

Goal:

Determine whether reliable matches are spatially distributed.

Tasks:

[ ] grid coverage
[ ] clustering analysis
[ ] spatial statistics
[ ] visualization

---

# MILESTONE 12
## QUADTREE

Goal:

Implement adaptive spatial subdivision.

Tasks:

[ ] quadtree
[ ] match insertion
[ ] subdivision
[ ] spatial coverage analysis
[ ] tests

---

# MILESTONE 13
## MAGSAC++

Goal:

Add MAGSAC++ geometric verification where supported.

Tasks:

[ ] check OpenCV support
[ ] implementation
[ ] fallback handling
[ ] comparison against RANSAC
[ ] experiment

Never label ordinary RANSAC as MAGSAC++.

---

# MILESTONE 14
## PHASE CONGRUENCY

Goal:

Add structural representation for illumination robustness.

Tasks:

[ ] implementation
[ ] preprocessing integration
[ ] computational benchmarking
[ ] comparison with SIFT baseline

---

# MILESTONE 15
## MULTI-SCALE

Goal:

Handle significant scale differences.

Tasks:

[ ] image pyramids
[ ] scale metadata
[ ] coordinate conversion
[ ] multi-scale feature extraction
[ ] experiments

---

# MILESTONE 16
## OHRC

Goal:

Integrate Chandrayaan-2 OHRC imagery.

Tasks:

[ ] inspect actual OHRC product format
[ ] implement loader
[ ] metadata integration
[ ] common image representation
[ ] tests
[ ] OHRC experiments

Do not assume the OHRC format before inspecting real data.

---

# MILESTONE 17
## IIRS

Goal:

Integrate IIRS data.

Tasks:

[ ] inspect actual IIRS product
[ ] understand dimensionality
[ ] wavelength metadata
[ ] spectral representation
[ ] band selection/reduction
[ ] common interface
[ ] tests

Do not treat IIRS as an ordinary grayscale image without documenting
the reduction.

---

# MILESTONE 18
## CROSS-MODAL CORRESPONDENCE

Goal:

Evaluate optical ↔ IIRS correspondence.

Tasks:

[ ] representation experiments
[ ] phase-based representation
[ ] feature experiments
[ ] geometric verification
[ ] failure analysis

Do not claim success without measurable evidence.

---

# MILESTONE 19
## LEARNED CORRESPONDENCE

Goal:

Evaluate learned methods.

Potential methods:

- LoFTR
- SuperPoint

Tasks:

[ ] dependency setup
[ ] CPU fallback where possible
[ ] GPU support
[ ] common output representation
[ ] benchmark against classical baseline

Classical pipeline must remain functional independently.

---

# MILESTONE 20
## EXPERIMENT FRAMEWORK

Goal:

Make experiments reproducible.

Tasks:

[ ] configuration files
[ ] experiment IDs
[ ] parameter logging
[ ] metadata logging
[ ] runtime logging
[ ] result JSON
[ ] visualization outputs

---

# MILESTONE 21
## ASTRION DEMO PIPELINE

Goal:

Create a clean end-to-end demonstration.

Input:

Image A
Image B

Output:

- detected features
- candidate correspondences
- verified correspondences
- spatial distribution
- transformation
- registered image
- confidence/quality information
- metrics

---

# FINAL VALIDATION

Before claiming the system is complete:

[ ] TMC ingestion works
[ ] metadata works
[ ] preprocessing works
[ ] SIFT works
[ ] RootSIFT works
[ ] matching works
[ ] ratio test works
[ ] RANSAC works
[ ] registration works
[ ] metrics work
[ ] visualization works
[ ] spatial analysis works
[ ] MAGSAC++ status documented
[ ] Phase Congruency evaluated
[ ] multi-scale evaluated
[ ] OHRC integration evaluated
[ ] IIRS integration evaluated
[ ] learned methods evaluated if feasible
[ ] experiments reproducible
[ ] no original data modified
[ ] no unsupported scientific claims made