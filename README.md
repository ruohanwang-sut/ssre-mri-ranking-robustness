# SSRE MRI ranking robustness

Supporting code and data for **Pairwise Ranking Robustness of Deep Learning MRI
Reconstruction Models Across Sensitivity Support Settings**.

SSRE evaluates whether pairwise model advantages persist across a prespecified
sensitivity-support grid while acquisition data and frozen checkpoints remain fixed.
The study uses 68 reused fastMRI brain AXT2 validation volumes, three checkpoints
from two model families, two acquisition conditions and four support fractions.
Three pair-by-condition curves retain dominance and three show strict crossing.
Results concern this validation subset and tested discrete grid; no clinical,
independent-test or leaderboard performance is claimed.

## Reproduce the statistical results

Use Python 3.10 with NumPy 2.2.6 (the recorded analysis versions).

```bash
python -m pip install -r requirements.txt
python code/reproduce_statistics.py
```

The script reads 1,632 frozen volume-level metric rows, reconstructs six paired
68x4 matrices, applies the original studentized max-t bootstrap functions with
10,000 draws and seed 20260901, and checks means, standard errors, confidence-band
endpoints, critical values and classifications against frozen statistics. It
requires neither original MRI files nor GPUs. This is a read-only statistical
reproduction, not a rerun of the original reconstruction experiment.

## Reproduce the qualitative comparison

```bash
python code/render_real_case.py
```

Outputs are written to outputs/qualitative/. Reference and reconstructions use
one gray window, identical ROI coordinates and magnification, and error maps
share one blue scale. The arrays are a one-case regeneration, not the lost
historical qualitative file. PSNR annotations refer to all 16 slices of the case.
See docs/data_dictionary.md for the distinction between the supplied main-figure
assembly and this reproducible supplementary display, including colour-scale limits.

## Contents

- data/: volume metrics, volume ordering, frozen paired statistics, plotting data,
  regenerated reconstruction arrays and selected raw coil-sensitivity magnitudes.
- code/: maintained read-only statistical and qualitative-display entry points.
- figures/: exact main images embedded in the final manuscript and supplementary S1.
- archive/: original experiment, qualitative re-export and panel-generation source.
- docs/: field definitions, frozen environment/configuration, external model sources
  and access/rights information.

Raw fastMRI acquisitions and third-party weights are not included. Their access
routes and frozen identities are documented in docs/model_sources.md and
docs/frozen_configuration.json. Historical archive scripts retain original paths
and internal dependencies; they are provided for inspection, not as portable runners.
The internal analysis plan was not publicly registered before the experiment;
this publication-time repository must not be treated as prospective public registration.

## Contact and citation

Ruohan Wang, Shenyang University of Technology; ruohan58@gmail.com.
ORCID: https://orcid.org/0009-0006-6069-9177.
The manuscript is unpublished at the time of repository preparation. No article
DOI or repository DOI has been assigned here. Cite this repository and its exact
commit; use an archived DOI if a fixed version is later deposited.
Public availability does not imply a blanket reuse licence; see docs/rights_and_access.md.
