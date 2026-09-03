# CLIP ViT-L/14 matched-JPEG formal results with AIGI-Det-Calib

This directory extends the immutable 2026-09-02 main release with the independently validated AIGI-Det-Calib baseline. The original numeric artifacts remain unchanged in `results/clip_vlm_bias_controlled_matched_jpeg_20260902/`. The formal seeds are `0`, `2`, and `3`; seed `1` was used only for the disclosed development selection of the final residual scale (`0.75`) and is excluded from every reported aggregate.

## Acceptance scope

- Input profile: `matched_jpeg` (`2e8d51e705134ba6dd1245af72f36b30c1635d1d00c39f925357698352f796bf`).
- Four suites, 39 targets, 17 main-table methods, and 1,989 method-target-seed units.
- Every target contains 1,500 locked samples per seed; labels are consumed only by the evaluator.
- The protocol audit covers 33 Arrow bundles and 164,000 transformed samples.
- Main-table cells are target-wise means over seeds; each Mean column is the target-macro mean with standard deviation across the three seed-level macro means.
- Source-trained, CLIP-native, and method-specific source setups are separate comparison blocks. No global best is claimed across incompatible source states.

## Independent AIGI-Det-Calib baseline

The table row applies the official AIGI-Det-Calib scalar correction to the shared `source_ft` detector, not to Ours-Static. For every target, the first 100 samples in the locked stream estimate one label-free offset. Their predictions remain the original Source predictions; the offset is then frozen for the following 1,400 samples. Hidden labels enter only the evaluator, and the reported table slice is therefore the complete causal prequential stream rather than a retroactive recalibration.

| Dataset | Source AUC | AIGI AUC | Source Acc. | AIGI Acc. | Δ Acc. (pp) |
|---|---:|---:|---:|---:|---:|
| GenImage | 66.32 | 64.85 | 51.56 | 56.11 | +4.56 |
| AIGCDetectionBenchmark | 60.25 | 59.11 | 50.98 | 53.57 | +2.59 |
| AIGI-Holmes P3 | 61.87 | 60.24 | 50.32 | 54.10 | +3.77 |
| OpenSDID Global | 65.76 | 63.63 | 50.83 | 55.30 | +4.48 |

Across all 39 targets, the source detector changes from 50.90% to 54.38% causal Accuracy and from 62.46% to 61.01% causal AUC. On the 1,400 held-out samples, where a single fixed offset is applied uniformly, AUC is invariant (62.47% before and after calibration), while Accuracy improves from 50.94% to 54.67%, ECE from 0.4900 to 0.1339, NLL from 5.4912 to 0.7794, and Brier score from 0.4882 to 0.2655. Thus this row is interpreted only as a threshold-calibration baseline, not as a ranking-adaptation method. The `ours_static` branch in `aigi_det_calib_summary.json` is retained solely as a paired diagnostic and is not a table method.

## Paired effect of Ours

| Dataset | Static AUC | Ours AUC | Δ AUC (pp) | Static Acc. | Ours Acc. | Δ Acc. (pp) |
|---|---:|---:|---:|---:|---:|---:|
| GenImage | 82.08 | 83.46 | +1.38 | 67.97 | 78.66 | +10.69 |
| AIGCDetectionBenchmark | 81.96 | 83.07 | +1.12 | 64.61 | 77.41 | +12.79 |
| AIGI-Holmes P3 | 90.67 | 91.61 | +0.93 | 67.10 | 84.85 | +17.75 |
| OpenSDID Global | 96.79 | 96.79 | +0.00 | 80.32 | 92.03 | +11.71 |

The final readout is the fixed R47 rule. R37 is retained only as `ours_no_calibrated_readout` in the base release's `readout_ablation_summary.json`; it is not a second main-table method.

## Best result inside shared-source blocks (AUC, %)

| Dataset | Public-detector method | AUC | CLIP-native method | AUC |
|---|---|---:|---|---:|
| GenImage | `rotta` | 66.32 ± 0.74 | `cliptta` | 67.68 ± 0.22 |
| AIGCDetectionBenchmark | `eata` | 60.25 ± 0.08 | `dynaprompt` | 71.46 ± 0.15 |
| AIGI-Holmes P3 | `eata` | 61.87 ± 0.32 | `cliptta` | 70.60 ± 0.54 |
| OpenSDID Global | `eata` | 65.76 ± 0.19 | `cliptta` | 61.53 ± 0.14 |

## Unchanged companion evidence

The matched-JPEG bundle audit, Ours readout ablation, boundary-blind recurrent stream, and hardware-stratified efficiency results are unchanged by this table extension. Their confirmed artifacts remain in `results/clip_vlm_bias_controlled_matched_jpeg_20260902/` and are deliberately not duplicated here.

## Files

- `clip_vitl14_summary.json`: full cross-seed, per-target main result.
- `aigi_det_calib_summary.json`: strict-causal protocol, identities, validation, Source baseline, and the explicitly diagnostic Ours-Static branch.
- `per_seed_summary.json`: dataset-level metrics for every formal seed.
- `clip_vitl14_auc_table.csv` and `clip_vitl14_*_table.tex`: final aggregate tables.
- `calibration_summary.json`: Brier, NLL, and ECE for all main-table methods.
- `source_models.json`: source setup, checkpoint digest, code revision, and development/validation split.
- `artifact_manifest.json`: SHA-256 and byte size of every release artifact.
