# TDA batch-one matched-JPEG rerun

This directory contains the accepted replacement for the TDA row in the CLIP ViT-L/14
`matched_jpeg` experiment. It does not modify the earlier result releases. The rerun uses
TDA commit `e697fb0c8078cdeff93daa56bcf8860702542069` through project commit
`e5f24b457fcdbf1871398403c67f252b9f859dea`.

## Corrected protocol

The author clean/custom-dataset contract is now enforced as `batch_size=1`: each online
step contains one target sample and one deterministic global view, and TDA updates its
positive/negative caches before returning that sample's prediction. The previous run used
a batch of 16 distinct target samples, which entered the official multi-view selection and
feature-averaging branch. Those old TDA values remain available for audit but are not valid
for the paper row.

The binary class prompts, target identities and order, evaluator, threshold, formal seeds
`0/2/3`, fixed OpenAI CLIP ViT-L/14 checkpoint, and `matched_jpeg` bytes are otherwise
unchanged. This rerun does not alter Ours or its fixed `0.75` readout scale.

## Accepted results

Values are percentages. The uncertainty is the sample standard deviation across the three
seed-level target-macro values. Deltas compare against the retained batch-16 TDA audit.

| Dataset | Batch-one AUC | Δ AUC (pp) | Batch-one Accuracy | Δ Accuracy (pp) |
|---|---:|---:|---:|---:|
| GenImage | 65.11 ± 0.41 | +1.89 | 53.69 ± 0.15 | +1.86 |
| AIGCDetectionBenchmark | 69.51 ± 0.13 | +2.14 | 53.28 ± 0.05 | +1.91 |
| AIGI-Holmes P3 | 57.74 ± 0.30 | +1.11 | 53.05 ± 0.13 | +1.19 |
| OpenSDID Global | 56.94 ± 0.25 | +0.96 | 52.78 ± 0.14 | +1.24 |
| Four-dataset macro | 62.32 | +1.53 | 53.20 | +1.55 |

## Acceptance audit

- Four datasets, 39 targets, and formal seeds `0/2/3` completed: 117 target-seed runs.
- All 175,500 locked samples were delivered in 175,500 one-sample online batches.
- Every batch selected exactly its one sample for the TDA cache update.
- Every target contains exactly 1,500 unique locked sample identities.
- All reported AUC, Accuracy, and Balanced Accuracy values are finite.
- Every run records the `matched_jpeg` specification hash, pinned TDA commit, and CLIP
  checkpoint SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- The two-GPU run completed on NVIDIA GeForce RTX 3070 Ti devices after an actual CUDA
  smoke test of the batch-one wrapper.

## Files

- `clip_vitl14_summary.json`: full per-seed, per-target and cross-seed TDA result.
- `per_seed_summary.json`: dataset-level target-macro metrics for seeds `0/2/3`.
- `previous_tda_comparison.json`: exact old/new values and deltas.
- `audit_summary.json`: protocol identities and acceptance counts.
- `source_models.json`: source setup, prompts, commits, and checkpoint digest.
- `clip_vitl14_auc_table.csv` and `clip_vitl14_*_table.tex`: generated tables.
- `artifact_manifest.json`: SHA-256 and byte size of every release artifact.
