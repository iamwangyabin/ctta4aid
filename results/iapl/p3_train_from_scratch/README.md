# P3 IAPL training reproduction

P3 trains IAPL from CLIP initialization rather than evaluating only the
authors' released checkpoints. UFD and GenImage each use seeds 100, 101, and
102. Every trained checkpoint will subsequently run the corresponding complete
official TTA evaluation, and the report will include the mean and sample
standard deviation across seeds.

The UFD training data is ready on all three compute hosts in the verified Arrow
format. The four official ProGAN categories contain 36,006 images each, for
144,024 training images total. Three single-rank jobs use the authors'
`run_universalfake.sh` settings and the existing `cl` environments.

All three UFD seeds completed training. Seed 100 took 1h53m04s on A6000; seeds
101 and 102 took 41m18s and 40m52s on 4090-1 and 4090-2. Their immediate
non-TTA full-domain Accuracy / AP results were 89.88% / 98.15%, 89.70% /
96.78%, and 86.44% / 97.13%, respectively. These numbers are training-time
static evaluations and must not be compared directly with the paper's TTA
table. The best-Accuracy checkpoints are retained for the full official TTA
runs.

Seed 100 initially stopped before model construction because A6000's `cl`
environment lacked `pytorch-wavelets`. The failure log was retained. The
environment was completed with `pytorch-wavelets` 1.3.0 and PyWavelets 1.8.0,
the full IAPL model import passed, and seed 100 was restarted from scratch and
completed successfully. Seeds 101 and 102 used PyWavelets 1.9.0; the version
difference is recorded for reproducibility.

The first seed-100 full-TTA attempt used an emergency two-node 5+3 rank layout
because 4090-1 was unreachable and 3090 was occupied. The three ranks on the
24 GiB 4090 failed on the first domain with two CUDA out-of-memory errors and
one cuBLAS allocation failure. No domain-level metric is reported from the
partial attempt. All eight rank logs and the exact failure layout are retained;
the run will restart from the beginning with the previously validated 4+2+2
layout when the third 24 GiB node is available.

The GenImage source on 3090 contains only 985 of 1,214 Arrow shards and is not
being treated as complete. Instead, the 96.4 GB, 30-part official SD1.4 archive
mirror was downloaded on 4090-2. The first transfer stopped after 43 GB when
the mirror closed a response early; its cache was retained and the
lower-concurrency retry completed. No partial-data result is reported as the
official training reproduction.

The conversion path is fixed in `scripts/import_genimage_to_hf_arrow.py`. It
requires a JSON plan with explicit subset, split, label, source directory or
ZIP prefix, and optional expected counts/bytes; writes `save_to_disk` Arrow shards plus
`mapping.json` and split metadata; and checks representative source/Arrow byte
hashes before atomically publishing the output. GenImage training then uses the
authors' `run_genimage.sh` settings through
`scripts/run_iapl_genimage_train_single.sh` and refuses non-Arrow input.
The two-record end-to-end smoke test passed in 4090-2's `cl` environment with
`datasets` 5.0.0: conversion, `save_to_disk` reload, representative byte hashes,
the framework reader and the training-launcher preflight all succeeded.
The resumed official SD1.4 download also completed: 30 files totaling exactly
96,413,397,770 bytes. Per-file size and SHA256 manifests are archived.

Multipart join and the full CRC test subsequently passed. The joined archive
contains 336,000 files; the official training split is exactly 162,000 AI plus
162,000 nature images totaling 93,016,744,367 uncompressed bytes. The converter
now reads those members directly from the verified ZIP into Arrow, so no
intermediate image-tree extraction is needed. A second two-record end-to-end
test confirmed ZIP member enumeration, byte-preserving Arrow storage and reload.

The first full ZIP-to-Arrow attempt exposed a performance bug after 1,176 rows:
the cache expression reopened the 336,000-entry ZIP directory for every image,
limiting conversion to 1.18 rows/s and projecting 76 hours. The run was stopped
without publishing an output, its log and incomplete 126 MiB cache were kept,
and the iterator was fixed to open each archive once. A regression test now
checks the handle count; the retry uses a fresh cache and the unchanged source
plan.

The fixed full conversion completed and atomically published 324,000 SD1.4
training rows (162,000 real and 162,000 fake) in 94 Arrow shards. The source
images total 93,016,744,367 bytes. Generation took 3m42s and the final
`save_to_disk` pass took 1m26s. Four boundary samples passed both byte-hash and
PIL decoding checks through the framework reader. Those checks were
representative rather than exhaustive.

The byte-verified P2 test extraction was then converted separately. Its eight
domains contain exactly 100,000 rows and 25,451,020,754 image bytes in 26
Arrow shards; every domain has the expected balanced real/fake counts. The
framework loaded and decoded all eight domains, including the 16,000-row
SD1.5 domain, and reported 100,000 rows in total. A later exhaustive pass fully
decoded all 100,000 payloads (50,000 JPEG and 50,000 PNG) with zero failures.

The final preflight found one more integration defect before training: the
compatibility patch added Arrow loading only to IAPL's generic
`Dataset_Creator`, but GenImage uses `Dataset_Creator_GenImage`. The pinned
checkout patch and verifier now require Arrow injection in both creators. The
fixed GenImage creator sees all 324,000 physical training rows and the expected
12,000 ADM / 16,000 SD1.5 test rows.

Seed 100 then started from a separate clean worktree at the pinned IAPL commit.
It reached reported batch 500 of 10,125 before PIL raised
`UnidentifiedImageError`; the attempt produced no checkpoint or metric and is
retained as a failed run. A subsequent exhaustive decode of all 324,000 Arrow
rows found exactly three invalid samples, all zero-byte SD1.4 fake PNGs:
`033_sdv4_00134.png`, `033_sdv4_00137.png`, and `033_sdv4_00152.png`. The ZIP
CRC had passed because zero-length members are structurally valid. An
independent GenImage cleaning appendix lists the same three SD1.4 files among
its removals, so this is a source-data defect rather than Arrow corruption.

The immutable 324,000-row conversion remains untouched. A separate atomic
metadata view hard-links the same 94 Arrow shards while selecting 323,997 rows
(162,000 real and 161,997 fake) and excluding only the three audited empty
payloads. Its row counts, excluded paths, and hard-link identities passed; a
second exhaustive decode then validated all 323,997 selected payloads (162,000
JPEG and 161,997 PNG) with zero failures. The 100,000-row test Arrow also
passed its exhaustive decode with zero failures. Seed 100 attempt 2 restarted
from scratch on 4090-2 at 10:54:36 +08:00, entered its 10,124-batch training
loop, and is running from the filtered training root plus the verified test
root. The retry completed all 10,124 batches in 1h41m13s and produced a
1,693,607,629-byte checkpoint with SHA256
`aa0d8ab805f5c4fc846154e7da25ffae8cea32cbab9a8eb5ab3203ea27387096`.
Its immediate single-view, non-TTA eight-domain evaluation was 84.03% mAcc /
98.86% mAP, with 99.98% real accuracy but only 68.08% fake accuracy. The high
AP and low thresholded fake accuracy show that this static result is strongly
real-biased on ADM, BigGAN, GLIDE, and Midjourney; it is retained as observed
and is not substituted for the pending official TTA evaluation.

Seed 101 started from scratch on the same clean pinned worktree and filtered
data at 13:08:46 +08:00. It completed the 10,124-batch loop in 1h41m10s and
the eight-domain static evaluation at 15:06:01. Its result was 95.44% mAcc /
99.62% mAP, with 99.79% real accuracy and 91.09% fake accuracy. The
1,693,607,565-byte checkpoint has SHA256
`4cbaec6c15a9a0219d68cb5ef947585b5362ed4a5befe3e359b3152e84eaf2d9`.
The 11.41-point mAcc difference from seed 100 is retained as an observed seed
sensitivity rather than hidden by selecting the better run.

Seed 102 started from scratch at 15:08:33 +08:00 using the same pinned
worktree, verified data, `cl` environment, and single-GPU protocol. It
completed the final 10,124-batch loop in 1h41m10s and the eight-domain static
evaluation at 17:05:28. Its result was 82.78% mAcc / 99.14% mAP, with 99.96%
real accuracy and 65.59% fake accuracy. The 1,693,607,565-byte checkpoint has
SHA256
`5d94e1159367ec8b4f6fc70cf6e5b8856430cc9f676c60117349b0b92bf3f18f`.

Across GenImage seeds 100/101/102, the non-TTA static result is 87.42 +/- 6.98%
mAcc and 99.21 +/- 0.38% mAP (mean +/- sample standard deviation). Real
accuracy is stable at 99.91 +/- 0.10%, while fake accuracy is only
74.92 +/- 14.06%. The instability is therefore a thresholded fake-class
failure on ADM, BigGAN, GLIDE, and Midjourney, not a loss of ranking quality.
The complete per-domain aggregate is retained in
`genimage/static_three_seed_summary.json`; no seed was discarded.

The UFD three-seed static aggregate is 88.67 +/- 1.94% Accuracy and
97.35 +/- 0.71% AP. Both UFD and GenImage training chains are now complete,
but P3 remains open until every trained checkpoint receives the full official
eight-rank TTA evaluation. The validated 4+2+2 layout still requires
4090-1 or an equivalent third 24 GiB node; 4090-1 remained unreachable at
17:18 +08:00. The 3070x2 fallback was also ruled out: its second GPU is
visible to `nvidia-smi`, but the `cl` PyTorch runtime reports CUDA unavailable
and zero devices even with `CUDA_VISIBLE_DEVICES=1`. The exact host audit is
stored in `resource_audit_20260724_1718.json`.

At 17:26 +08:00 the unrelated A6000 allocation had disappeared, leaving both
A6000 and 4090-2 idle. This makes a 6+2 rank layout technically possible
without changing the eight-rank data or TTA protocol, but six concurrent IAPL
processes have not passed a 48 GiB A6000 memory preflight. A paired launch was
not authorized on A6000; only ranks 6 and 7 started on 4090-2, waited before
the distributed world formed, and were terminated. No batch or domain ran,
and no metric is reported. The logs are preserved under
`ufd/seed100/tta_attempt2_partial_2node_6plus2_expandable`.

Later the 3070x2 driver recovered and both GPUs became visible. The exact
compatibility environment, runtime code, weights, and 91 GiB UFD Arrow tree
were copied to that host; all 206 files match the source under the ordered
per-file SHA256 tree hash
`66e2628c676f43b82d2d5b2f92989525463845cb7f28b47b7d52c7f59dba4132`.
This did not yield another valid layout. One official-batch rank was tested
independently on each 3070 Ti. GPU0 failed while requesting another 18 MiB at
a sampled 7857 MiB peak; GPU1 still failed while requesting 50 MiB at 7845
MiB after enabling expandable segments, lazy CUDA module loading, and
disabling the cuDNN plan cache. Both failures occurred before the first batch
completed, so no metric is reported and the proposed 4+1+1+2 layout is
rejected. Full setup and failure evidence is under
`ufd/seed100/tta_preflight_3070x2`.

At 06:03 +08:00 on 2026-07-25, 4090-2 developed an independent driver
failure: the running kernel still had NVIDIA 580.159.03 loaded after the
user-space packages and on-disk module had been upgraded to 580.173.02.
System `nvidia-smi` failed with an NVML version mismatch and both PyTorch
environments reported zero CUDA devices. No reboot, module reload, package
downgrade, or other system change was made. Matching 580.159.03
`libnvidia-compute-580` and `nvidia-utils-580` packages were downloaded from
Canonical Launchpad and extracted under the user-owned asset tree. With that
library directory, `nvidia-smi`, `cl` torch 2.12.0, and the official
`caid-gemini-compat` torch 2.2.2 all passed CUDA tensor tests. Both launchers
now accept `IAPL_NVIDIA_COMPAT_LIB_DIR`; identical script hashes were deployed
to A6000, 4090-2, and 3070x2. This repairs 4090-2 without hiding the host
failure, but does not remove the missing-third-node blocker.

At 12:04 +08:00, 3090 recovered with an idle 24 GiB GPU. Its existing
`/data/DF-arrow-data/{ForenSynths,Ojha}` tree was checked against 4090-2:
all 205 dataset files have identical names, sizes, and ordered per-file
content tree SHA256
`735262849f09c586f9f12beb778aec6a0e78f89b42b7961d02824c13f7deacc0`.
The exact compatibility environment and runtime assets were installed, then
CUDA, NCCL 23007, dataset row counts, launcher hashes, checkpoint, and CLIP
hashes all passed preflight. UFD seed100 official TTA started at 12:34 in the
protocol-faithful A6000 4 + 3090 2 + 4090-2 2 layout. All eight ranks passed
the distributed barrier and entered `crn`; the initial host allocations were
36,279 / 18,208 / 18,710 MiB with no traceback. This is a running result, not
a completed metric.
