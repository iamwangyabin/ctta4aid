# CLIP ViT-L/14 matched-JPEG target-prior sensitivity

This directory contains the validated paired sensitivity result for final **Ours** and its
same-checkpoint frozen source control (`ours_static`). It is a supplementary analysis and
does not replace or modify the balanced `matched_jpeg` main results.

## Protocol

- Input profile: `matched_jpeg`
  (`2e8d51e705134ba6dd1245af72f36b30c1635d1d00c39f925357698352f796bf`).
- Formal seeds: `0`, `2`, and `3`; threshold: `0.5`; batch size: `16`.
- Four datasets and 39 targets; every target stream contains 800 samples.
- Fake prevalence is 10%, 25%, 50%, 75%, or 90%, corresponding to real/fake counts
  720/80, 600/200, 400/400, 200/600, and 80/720.
- Each prevalence stream is a locked, order-preserving subset of the corresponding formal
  main manifest. Target labels are evaluator-only.
- `ours_static` and `ours` use byte-identical sample manifests for all 585 paired
  target/seed/prevalence units.

This release covers the completed **Ours versus frozen source control** paired slice only.
`ours_no_calibrated_readout` was not run here, so the separately configured three-method
prior-shift scope is not claimed complete.

## Four-dataset macro results

Values are percentages. Each value first averages targets within a dataset, then averages
the four datasets within each seed, and finally averages the three formal seeds. Full sample
standard deviations and dataset-level values are retained in `cross_seed_summary.json` and
`prior_shift_table.csv`.

| Fake prevalence | Static Acc. | Ours Acc. | Δ Acc. | Static BAcc. | Ours BAcc. | Δ BAcc. | Static Macro-F1 | Ours Macro-F1 | Δ Macro-F1 | Δ AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 93.35 | 82.70 | -10.65 | 69.83 | 81.84 | +12.02 | 72.13 | 71.20 | -0.92 | -0.84 |
| 25% | 84.55 | 84.61 | +0.06 | 69.87 | 82.63 | +12.76 | 70.17 | 80.85 | +10.68 | -0.20 |
| 50% | 69.94 | 82.52 | +12.57 | 69.94 | 82.52 | +12.57 | 64.15 | 82.23 | +18.08 | +0.19 |
| 75% | 55.41 | 75.62 | +20.21 | 70.03 | 81.04 | +11.01 | 54.10 | 74.25 | +20.15 | -0.15 |
| 90% | 46.59 | 63.98 | +17.39 | 69.95 | 77.42 | +7.47 | 43.62 | 58.10 | +14.49 | -0.19 |

## Interpretation

Ours improves Balanced Accuracy at every tested prevalence by 7.47–12.76 percentage
points. Ordinary Accuracy changes strongly with the class prior: at 10% fake, the frozen
control's real-biased decisions produce higher raw Accuracy even though Ours is 12.02 points
better in Balanced Accuracy. At 25–90% fake, Ours is non-negative or clearly better in raw
Accuracy. Macro-F1 improves at 25–90% fake but is 0.92 points lower at 10% fake.

AUC changes by only -0.84 to +0.19 points. The observed benefit is therefore a thresholded
decision and class-balance improvement, not a ranking improvement. `fake_f1` treats fake as
the positive class; `macro_f1` is the unweighted mean of real-class and fake-class F1.

## Acceptance audit

- 120 final summary files and 1,170 method-target-seed result units were imported.
- All metrics are finite and all confusion counts reconstruct exactly from the evaluator
  summaries.
- All 585 paired sample manifests are byte-identical, contain 800 unique sample identities,
  and have the expected class counts.
- Both runs use project commit `ff0b5255eb036d93aeab31c1ac1d7d9a0c59be55`, the fixed
  OpenAI CLIP ViT-L/14 initialization, the same rank-4 LoRA source checkpoint, and final
  Ours readout scale `0.75`.

## Files

- `per_target_summary.json`: every final target result, confusion matrix, and recorded
  efficiency value for both methods.
- `per_seed_summary.json`: dataset target-macro metrics for each formal seed.
- `cross_seed_summary.json`: dataset and four-dataset macro means, sample standard
  deviations, and paired Ours-minus-static deltas.
- `prior_shift_table.csv`: compact paired table for AUC, Accuracy, Balanced Accuracy,
  fake-class F1, and Macro-F1.
- `audit_summary.json`: completeness, identity, metric, and paired-manifest checks.
- `source_models.json`: model, checkpoint, source setup, and code identities.
- `artifact_manifest.json`: SHA-256 and byte size of every release artifact.
