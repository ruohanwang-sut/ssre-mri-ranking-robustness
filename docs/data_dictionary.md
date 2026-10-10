# Data dictionary

`volume_metrics.csv`: one row per volume/checkpoint/acquisition condition/support fraction.
The 1,632 rows are repeated measurements, not independent subjects. `filename` is the
original fastMRI dataset file identifier, not a patient name. `coil` is the coil count.
Model keys: VN = VarNet, PMR = PromptMR v1, PMRP = PromptMR+. The latter two are
checkpoints from one model family. C1: R=4, center fraction 0.08, L_nom=32;
C2: R=8, center fraction 0.04, L_nom=16. rho = L/L_nom; L is the sensitivity
estimator support width, not a change to acquired data. PSNR is in dB; NMSE and
SSIM are dimensionless. Metrics are computed over 16 slices per volume, at 320x320.

`volumes.csv` preserves the original 68-volume ordering used in paired bootstrap
resampling and records the acquisition mask offsets and sampled-column counts.
`mean_metrics.csv` supports manuscript tables 1 and 3, without interval estimates.
NMSE in the manuscript table is displayed in units of 10^-3; here it is unscaled.
`pairwise_bands.csv` supports table 2 and figure 3; a positive difference favours
the first model. The bands cover four tested points within one curve, not all six curves.
`frozen_statistics.json` retains full precision, classifications, descriptive strata
and historical ancillary fields; the manuscript's primary scope is the six D curves.
`figure_source_data.csv` is the original plotting-data export for figures 2 and 3.

`qualitative_regenerated.npz`: 26 float32 320x320 arrays, containing targets for C1
and C2 and 24 checkpoint/condition/support reconstructions, for slice index 7
(slice 8) of file_brain_AXT2_209_2090204. These are REGENERATED arrays. The 24
volume-level metric cells matched frozen records exactly; bitwise equality to the
lost historical qualitative file is not claimed. PSNR labels use the full 16-slice
volume, not the displayed slice. `qualitative_source_data.csv` records comparison
metrics and display parameters. `qualitative_reexport.json` is a path-sanitized
copy of the re-export report; its historical gate wording is not an external validation.

`varnet_sensitivity_magnitudes.npz`: eight raw float32 768x396 coil-magnitude
arrays exported by the original VarNet sensitivity network, coils 3, 6, 9 and 12,
at C1 rho=0.25 and 1.00. They are a separate later panel export, not contents of
the 26-array reconstruction NPZ. No full-population mechanism claim follows from
these examples. The original generating code is in archive/export_ssre_real_case_panels_001.py.

The three main PNGs are the exact images supplied for the final Word manuscript.
The supplied figure-1 error-map colour bar is labelled 0-1. Its assembly source
is not supplied here, so that colour-scale mapping is not independently certified.
For a fully reproducible display convention use supplementary figure S1 and
code/render_real_case.py: abs(prediction-reference)/max(reference), with one
common blue scale over all six maps (maximum approximately 0.1297).
