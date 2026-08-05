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
