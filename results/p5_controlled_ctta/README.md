# P5 controlled CTTA table

P5 began only after the final P4 audit was committed and pushed. The controlled
table will compare Source, TENT, EATA, CoTTA, RoTTA, LAME, and T2A on three
evaluation seeds in both independent single-target and continual-stream modes.
IAPL remains a separate end-to-end protocol reference.

## 2026-08-05 source-training preflight

- A6000 was clean at 17 MiB and 0% utilization.
- The server's requested `cl` environment is at
  `/home/home/yabin/miniconda3/envs/cl`; the initial non-login path probe at
  `/home/yabin/miniconda3/envs/cl` failed and is intentionally recorded.
- The real environment has Python 3.12.12, PyTorch 2.9.1+cu128, torchvision
  0.24.1+cu128, datasets 4.4.2, and a working RTX A6000 CUDA device.
- All 57 tests and bytecode compilation passed in that environment.
- A duplicate local config-test attempt failed because the bundled local Python
  has no PyYAML (`ModuleNotFoundError: yaml`). This is retained as an environment
  failure; the authoritative `cl` run above exercised the full suite successfully.
- The 90 GiB ForenSynths Arrow root exposes 720,119 disjoint training rows and
  8,000 validation rows. First and last records from both splits decoded to
  normalized `3x224x224` tensors with valid binary labels.
- No compatible Fisher-bearing common CNN checkpoint exists on A6000, 3090, or
  4090-2. P5 must therefore train the declared shared source checkpoint before
  any method comparison.
- The torchvision IMAGENET1K_V2 ResNet-50 initialization was downloaded and
  verified as SHA-256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

The next ordered step is the bounded Arrow training smoke test, followed by the
full ten-epoch source run and 2,000-sample EATA Fisher estimation. Every table
run must then record the exact resulting checkpoint SHA-256.

The smoke test completed cleanly in 18.18 seconds. It trained on 64 rows,
validated on a disjoint 64 rows at 0.722656 AUC / 0.625 Accuracy, saved a
94,851,517-byte checkpoint with SHA-256 `8867bf2d...14f7`, and contained 106
finite Fisher tensors. GPU memory returned to 17 MiB with no surviving process.
This result validates the complete Arrow decode, optimizer, validation, Fisher,
serialization, and cleanup chain; it is not a performance result.

At 20:52:55 CST the full ten-epoch run launched on the clean A6000 as launcher
PID 2265579. The formal config hash is `c315aead...e7641` and launcher hash is
`ae180c88...032e`. At the one-minute check the training process plus four data
workers were alive, the GPU held 3,647 MiB at 100% utilization, and no startup
error was present. The log is intentionally expected to remain empty until the
first epoch summary. No seven-method table job will start before this run exits
cleanly and its full Fisher-bearing checkpoint is audited.

At 21:27:40 the first full epoch completed after 34:45 elapsed time. Training
loss was 0.00978 and the disjoint 8,000-row validation result was 0.99983 AUC /
0.998 Accuracy. The main process and four workers remained alive; the A6000 was
at 3,665 MiB, 89% utilization, 85 C, and 284.87 W with no logged error. The
recipe is unchanged and training continues toward the ten-epoch/Fisher gate.

At 23:27:51 the run reached 5/10 epochs after 2:34:56. Loss fell through
0.00327, 0.00255, 0.00217, and 0.00177. Validation AUC was 1.0 in epochs 2-5;
Accuracy was 1.0 except for a retained epoch-4 dip to 0.99325. Since checkpoint
selection updates only on a strictly greater AUC, the current in-memory best is
epoch 2, whose AUC and Accuracy are both 1.0. The GPU and five Python processes
remain healthy, so the same run continues without early stopping.

## Full source completion

The formal run exited cleanly after 4:31:57. All ten epochs are preserved in
`source_train_final_20260806.json`, including threshold-Accuracy dips in epochs
4, 6, 7, 8, 9, and 10. The strict AUC selection rule chose epoch 2 at 1.0 AUC
and 1.0 Accuracy on all 8,000 validation rows.

The final 94,851,517-byte checkpoint has SHA-256
`57d3e1ea43b914226449ecb5d4267d86324002f5b0210bad5a7667673acd3840`.
All 320 model tensors are finite. All 106 Fisher tensors are finite and nonzero.
The process exited with status 0, no worker survived, and A6000 returned to
17 MiB / 0%. Identical read-only copies were verified on A6000, 3090, and
4090-2; the 3090 copy stayed entirely under `/home` and did not touch its broken
`/data` mount.

The next ordered work is checkpoint-backed method smoke testing and freezing
three-seed configs. Single-target jobs must finish before continual-stream jobs.

The formal-checkpoint smoke then ran all seven methods on the same 32 real SAN
samples. All methods emitted finite metrics and complete artifacts, every result
recorded checkpoint `57d3e1ea...d3840`, all ordered manifests were exactly
equal, and EATA reported Fisher enabled on every batch. The tiny metrics are
stored only as an execution gate, not as comparative evidence. The first audit
script incorrectly expected `path`/`label` manifest columns and raised
`KeyError: path`; that failure is retained. The corrected audit used the real
`batch/domain/position/sample_id` schema and passed.

At 01:48:36--01:49:20 the three frozen all-method single-target jobs launched:
seed 0 on A6000, seed 1 on 3090, and seed 2 on 4090-2. Every host verified the
same source checkpoint and the same ForenSynths/Ojha metadata hashes before
launch. The 3090 uses only its read-only `/home` SSHFS and never touched broken
`/data`; 4090-2 uses the isolated 580.159.03 driver library. Initial checks found
all three main processes plus workers alive and producing domain results.

Performance will aggregate all three seeds. Raw latency will not be averaged
across unlike GPUs: seed 0/A6000 is the controlled efficiency row, while 3090
and 4090-2 timing remains hardware-qualified diagnostic data. Continual jobs
remain blocked until all single-target outputs and manifests are audited.

## Single-target completion

All three jobs exited with status 0: seed 2 on 4090-2 in 15:00.57, seed 0 on
A6000 in 27:28.91, and seed 1 on the read-only 3090 SSHFS in 29:24.40. Each run
contains all 126 method-target results, uses the same source checkpoint, and
evaluates 32,798 samples per method. The formal auditor checked every metric,
batch table, and ordered sample manifest. Within each seed, all seven method
manifests are byte-identical for every target. No result was restarted or
selected after inspection.

The three-seed, unweighted 18-domain macro results are:

| Method | mAUC (%) | mAcc (%) | A6000 ms/batch | Peak MiB |
|---|---:|---:|---:|---:|
| Source | 88.820 +/- 0.045 | 77.524 +/- 0.026 | 12.57 | 367 |
| TENT | 87.223 +/- 0.312 | 81.778 +/- 0.400 | 44.98 | 1,645 |
| EATA | 86.643 +/- 0.104 | 80.912 +/- 0.180 | 33.79 | 1,651 |
| CoTTA | 83.897 +/- 0.509 | 78.271 +/- 0.606 | 89.63 | 1,945 |
| RoTTA | 88.094 +/- 0.064 | 82.053 +/- 0.084 | 108.90 | 10,972 |
| LAME | 87.769 +/- 0.078 | 73.225 +/- 0.122 | 14.39 | 378 |
| T2A | 85.065 +/- 0.226 | 78.972 +/- 0.133 | 129.87 | 3,233 |

The spread is the population standard deviation over seeds 0, 1, and 2. Source
has the best mAUC, while RoTTA has the best thresholded mAcc; adapting does not
uniformly improve ranking quality. The latency and memory columns are only the
controlled seed-0/A6000 measurement. Full per-domain values and hardware-tagged
diagnostics are in `single_target_three_seed_summary_20260806.json`.

The single-target gate is now complete. The next ordered step is a fresh
three-host preflight followed by the frozen continual-stream matrix.

## Continual-stream launch

After commit `c163a07` was pushed, a fresh three-host preflight found no project
job, no existing continual output/log path, and clean GPUs. Code, the read-only
checkpoint, and all four Arrow state/mapping hashes matched. The first A6000
metadata command incorrectly assumed `dataset_dict.json` and failed on both
roots; inspection found the actual `mapping.json`, and the corrected identity
check passed. This preflight error is preserved in `continual_launch_20260806.json`.

Seeds 0, 1, and 2 launched at 02:28:41--02:29:34 on A6000, 3090, and 4090-2.
The first health check found every launcher and worker alive, all effective
configs written, and 1/1/2 completed methods respectively. The 3090 SSHFS
remains read-only and broken `/data` was not accessed. Final aggregation stays
blocked until all seven methods on all three seeds pass the continual audit.
