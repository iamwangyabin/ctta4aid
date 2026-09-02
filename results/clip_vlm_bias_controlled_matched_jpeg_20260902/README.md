# CLIP ViT-L/14 matched-JPEG formal results

This directory contains the validated main campaign for the paper. The formal seeds are `0`, `2`, and `3`; seed `1` was used only for the disclosed development selection of the final residual scale (`0.75`) and is excluded from every reported aggregate.

## Acceptance scope

- Input profile: `matched_jpeg` (`2e8d51e705134ba6dd1245af72f36b30c1635d1d00c39f925357698352f796bf`).
- Four suites, 39 targets, 16 main-table methods, and 1,872 method-target-seed units.
- Every target contains 1,500 locked samples per seed; labels are consumed only by the evaluator.
- The protocol audit covers 33 Arrow bundles and 164,000 transformed samples.
- Main-table cells are target-wise means over seeds; each Mean column is the target-macro mean with standard deviation across the three seed-level macro means.
- Source-trained, CLIP-native, and method-specific source setups are separate comparison blocks. No global best is claimed across incompatible source states.

## Paired effect of Ours

| Dataset | Static AUC | Ours AUC | Δ AUC (pp) | Static Acc. | Ours Acc. | Δ Acc. (pp) |
|---|---:|---:|---:|---:|---:|---:|
| GenImage | 82.08 | 83.46 | +1.38 | 67.97 | 78.66 | +10.69 |
| AIGCDetectionBenchmark | 81.96 | 83.07 | +1.12 | 64.61 | 77.41 | +12.79 |
| AIGI-Holmes P3 | 90.67 | 91.61 | +0.93 | 67.10 | 84.85 | +17.75 |
| OpenSDID Global | 96.79 | 96.79 | +0.00 | 80.32 | 92.03 | +11.71 |

The final readout is the fixed R47 rule. R37 is retained only as `ours_no_calibrated_readout` in `readout_ablation_summary.json`; it is not a second main-table method.

## Best result inside shared-source blocks (AUC, %)

| Dataset | Public-detector method | AUC | CLIP-native method | AUC |
|---|---|---:|---|---:|
| GenImage | `rotta` | 66.32 ± 0.74 | `cliptta` | 67.68 ± 0.22 |
| AIGCDetectionBenchmark | `eata` | 60.25 ± 0.08 | `dynaprompt` | 71.46 ± 0.15 |
| AIGI-Holmes P3 | `eata` | 61.87 ± 0.32 | `cliptta` | 70.60 ± 0.54 |
| OpenSDID Global | `eata` | 65.76 ± 0.19 | `cliptta` | 61.53 ± 0.14 |

## Recurrent stream

The boundary-blind stream is `SD1.5_first → BigGAN → ADM → SD1.5_return`, with 1,792 online samples and a fixed disjoint 896-sample holdout. Ours obtains online AUC 83.88 $\pm$ 0.57, final AUC 87.02 $\pm$ 1.14, and forgetting 0.09 $\pm$ 0.08. Removing Detect, Route, or Update yields final AUC 82.96 $\pm$ 0.80, 83.84 $\pm$ 1.08, and 84.48 $\pm$ 1.36, respectively.

## Efficiency

Latency is never averaged across GPU models:
- seed 0, NVIDIA GeForce RTX 4090 (4090-2): Ours-Static 7.92 ms/sample; Ours 22.24 ms/sample.
- seed 2, NVIDIA GeForce RTX 4090 (4090-1): Ours-Static 6.59 ms/sample; Ours 14.13 ms/sample.
- seed 3, NVIDIA GeForce RTX 3090: Ours-Static 12.30 ms/sample; Ours 19.11 ms/sample.

Ours uses 49,281 trainable parameters for one active expert and at most 98,562 in the observed two-expert case. The maximum recorded peak memory is 1448.8 MB.

## Files

- `clip_vitl14_summary.json`: full cross-seed, per-target main result.
- `per_seed_summary.json`: dataset-level metrics for every formal seed.
- `clip_vitl14_auc_table.csv` and `clip_vitl14_*_table.tex`: final aggregate tables.
- `calibration_summary.json`: Brier, NLL, and ECE for all main-table methods.
- `readout_ablation_summary.json`: Ours-Static, R37, and final Ours.
- `recurrence_summary.json`: full recurrent-stream comparison and Detect/Route/Update ablations.
- `efficiency_summary.json`: hardware-stratified latency, memory, and trainable-parameter counts.
- `matched_jpeg_protocol_audit.json`: bundle-level byte, format, quality, geometry, and profile verification.
- `source_models.json`: source setup, checkpoint digest, code revision, and development/validation split.
- `artifact_manifest.json`: SHA-256 and byte size of every release artifact.
