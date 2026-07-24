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
data at 13:08:46 +08:00. It entered the 10,124-batch training loop in the
`cl` environment on 4090-2. Seed 102 remains queued behind it so the three
GenImage seeds do not compete for GPU memory or disk bandwidth.
