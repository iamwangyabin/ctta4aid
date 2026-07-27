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

The first three domains completed without a runtime error. Their official
Acc / AP values are `crn` 59.18% / 56.08%, `cyclegan` 96.79% / 94.02%, and
`dalle` 99.05% / 98.50%. The same prediction files were independently
recalculated with `scripts/compare_iapl_ufd_runs.py`; the three-domain macro
average is 85.01% Acc / 82.87% AP. The unexpectedly weak `crn` result is kept
as observed rather than filtered or restarted. The snapshot also records the
4 and 6 padded DistributedSampler indices in `crn` and `cyclegan`.

`biggan` and `deepfake` then completed at 95.83% / 92.76% and
95.28% / 96.15% Acc / AP. The five-domain independently recalculated macro
average is 89.23% / 87.50%, 7.68 and 11.99 points below the paper on the same
domains. The job continues into `gaugan` without selecting away the weak
domains.

At the ten-domain checkpoint, `gaugan` completed at 96.82% / 94.23%, the
three GLIDE domains ranged from 97.95% to 98.30% Acc and 97.89% to 98.20% AP,
and `guided` reached 82.25% / 92.88%. The ten-domain macro average is
91.95% Acc / 91.86% AP, 3.17 and 7.41 points below the paper on the same
domains. All ten prediction files and the independent recalculation are
retained while the unchanged run proceeds through `imle`.

`imle` reproduced the same severe pattern as `crn`: 59.54% Acc, 56.17% AP,
19.10% real accuracy, and 100.00% fake accuracy. It is retained unchanged.
`ldm_100` then completed at 98.80% / 98.27% Acc / AP. Across the first twelve
domains the macro result is 89.82% / 89.42%, 5.44 and 9.95 points below the
paper on those domains. Two A6000 Tailscale probes timed out during this
snapshot, but the LAN path, all ranks, and GPU utilization remained healthy;
the experiment was unaffected and continued into `ldm_200`.

UFD seed100 official TTA completed all 19 domains at 21:01 after 8h26m49s.
The official log reports 91.69% Acc / 90.82% AP; independent recalculation
from all 19 prediction files gives 91.6895% / 90.8209%, 3.92 and 8.50 points
below the paper. TTA raises Acc by 1.81 points over this checkpoint's static
evaluation but lowers AP by 7.33 points. The AP deficit is dominated by
`crn`, `imle`, `seeingdark`, and `stargan`. All eight ranks and all three
launchers exited normally, 88,353 unique indices are covered, and the 23
DistributedSampler padding records are explicitly retained.

4090-1 recovered after seed100 completed, allowing the otherwise unique
seed101 checkpoint to be copied and SHA256-verified on A6000, 3090, and
4090-2. Seed101 is next in the fixed execution order. It has not been started
because an unrelated active CAIDBench process owns 6,703 MiB on A6000; that
allocation is not interrupted, and seed102 is not allowed to overtake it.
At 02:03 on 2026-07-26 the same PID was still writing samples, owned 6,680 MiB,
and reported 5:01:25 remaining. The wait is therefore active rather than a
stale launcher state. At 02:54 it still owned 6,680 MiB and reported 4:13:52
remaining.

While seed101 waits, the GenImage multi-rank launcher was audited against the
Arrow-only training path. It had still required the old extracted ImageFolder
layout even though the pinned IAPL checkout already supports Arrow. The
launcher now gives `IAPL_DATASET_PATH` priority, validates one or two
`hf_arrow://` roots, defaults Arrow to `num_workers=0`, and retains the old
ImageFolder checks and `num_workers=8` fallback. The one-root form is
intentional: upstream `testtime_main` builds only the `tta` split and never
accesses `train_selected_subsets`, so only the verified 100,000-row test Arrow
is needed on evaluation nodes.

The final launcher SHA256 is
`894f0bfb21d77ffdab11208e64eb35300e186d2637917449fe64c797f10fa137`
on A6000, 3090, 4090-1, and 4090-2. A two-rank preflight on 4090-2 selected
NCCL 23007, the test-only Arrow URI, and `num_workers=0`; all 14 remote tests
passed. Two zero-byte transfer attempts are retained: macOS rsync
rejected unsupported `--append-verify`, and a detached managed-session child
did not persist. The tracked `--partial` retry completed local staging, then
the dataset was copied to A6000 and 3090. All three nodes contain 31 files and
25,467,557,463 bytes with the same ordered per-file SHA256 tree
`904464e62f1525f1deecfe85a5c64064ff7b0914a557275af0d7226c3b799b9f`.
The initial 3090 Tailscale transfer was explicitly interrupted at an observed
2,701,236,095 bytes and resumed over its faster `192.168.10.52` LAN route;
the partial bytes were retained. The UFD order guard remains unchanged. The
first 3090 launcher
preflight also failed before launch because its default released checkpoint
was absent; this was not hidden or bypassed. The trained seed100 checkpoint
was then copied to A6000 and 3090. Its 1,693,607,629 bytes and SHA256
`aa0d8ab805f5c4fc846154e7da25ffae8cea32cbab9a8eb5ab3203ea27387096`
now match the 4090-2 source. A6000 ranks 0-3, 3090 ranks 4-5, and 4090-2
ranks 6-7 all pass their exact Arrow/NCCL launcher preflight with that trained
weight. Seed101 and seed102 were then copied too; their SHA256 values
`4cbaec6c15a9a0219d68cb5ef947585b5362ed4a5befe3e359b3152e84eaf2d9`
and `5d94e1159367ec8b4f6fc70cf6e5b8856430cc9f676c60117349b0b92bf3f18f`
match on A6000, 3090, and 4090-2. All GenImage TTA inputs are therefore ready,
but they remain behind UFD seeds 101 and 102. Full evidence is in
`genimage/tta_arrow_preflight_20260726_0154`.

The unrelated A6000 PID exited by 06:43 on 2026-07-26. All three target GPUs
were free, every seed101 output directory was absent, and the exact checkpoint,
Arrow roots, NCCL 23007, seed, rank groups, and 4090-2 compatibility library
passed a fresh preflight on all nodes. UFD seed101 official TTA started at
06:45:05 in the same A6000 4 + 3090 2 + 4090-2 2 layout. All eight ranks
crossed the distributed barrier and entered `crn`; initial allocations were
36,903 / 18,652 / 19,154 MiB at 100% utilization. Rank0 reached 0/1596 with
an observed 8,437 MiB peak and no traceback, OOM, runtime, or collective error.
This is a running result, and seed102 remains blocked behind it.

The first completed seed101 domain is a material seed-sensitivity result:
`crn` reached 99.79% Acc / 99.61% AP, versus seed100's 59.18% / 56.08% on
the identical protocol. `cyclegan` then completed at 99.28% / 98.73%. The
two-domain independently recalculated macro result is 99.54% Acc / 99.17% AP,
4.00 / -0.81 points relative to the paper on the same domains and
21.55 / 24.12 points above seed100. Four and six sampler-padding duplicates
are retained for `crn` and `cyclegan`. Two 4090-2 Tailscale SSH monitoring
probes timed out, but rank0 continued across the `cyclegan` boundary into
`dalle`, proving that the established distributed job was not interrupted.
Both the monitoring failure and the successful continuation are preserved.

UFD seed101 official TTA completed all 19 domains at 15:18 on 2026-07-26
after 8h32m55s. The official log reports 89.54% Acc / 95.48% AP; independent
recalculation gives 89.5358% / 95.4825%, 6.07 / 3.83 points below the paper.
Compared with seed100, seed101 is 2.15 points lower in Accuracy but 4.66
points higher in AP, with extreme opposite outcomes on `crn` and `imle`.
All eight ranks and three launchers exited normally, all 88,353 unique indices
are covered, and 23 DistributedSampler padding records remain in the official
metrics. The first final-summary attempt failed before reading predictions
because the summarizer was not deployed on A6000; the tracked local script
then completed the audit. Both attempts are recorded. Seed102 is now next and
must pass a fresh three-host preflight before it starts.

The 15:30 fresh seed102 preflight passed on 3090 and 4090-2, but failed the
idle-GPU requirement on A6000 after an unrelated CoDA-Prompt CAIDBench process
acquired 5,036 MiB. Its PID remained active after stage-1 protocol evaluation.
No partial distributed world was launched and the unrelated process was not
interrupted. The failed preflight is retained; all three node preflights must
be repeated after A6000 is released.

PID 32143 completed stage 20, saved its protocol metrics at 20:51, and exited.
Before the three-host seed102 preflight could be repeated, a separate S-Prompt
CAIDBench representative10 run acquired the A6000. The new PID 50094 started
at 21:02:32 and owned 5,120 MiB when observed at 21:13. Seed102 still has not
launched, no partial world exists, and the two previously passing node checks
remain non-reusable. This blocker transition is preserved in
`ufd/seed102/resource_monitor_20260726_2113`.

PID 50094 finished stage 10, saved its metrics at 22:31, and exited. A6000 was
actually idle at 22:43. Fresh, non-reused preflights then passed on all three
hosts between 22:44:59 and 22:45:18, covering idle GPUs, absent output
directories, checkpoint and launcher hashes, Arrow state, NCCL 23007, seed102,
port 29642, rank groups, and the isolated 4090-2 compatibility libraries. UFD
seed102 launched at 22:46:13 in the fixed 4+2+2 layout. All eight ranks crossed
the distributed barrier and entered `crn`; rank0 reached 0/1596 with an 8,437
MiB peak, and no traceback, OOM, runtime, or collective error was found. The
run remains in progress and all GenImage TTA runs remain queued behind it.

At 00:13, seed102 completed `crn` and entered `cyclegan` at 250/331. Independent
recalculation from the completed prediction file gives 51.62% Acc / 59.27% AP
and 3.27% / 100% real/fake accuracy, retaining four sampler-padding records.
This is 7.56 points lower in Accuracy and 3.19 points higher in AP than seed100,
but 48.17 / 40.34 points below seed101 on the same domain. The large weak
result is preserved without rerunning or selecting a seed. All ranks and
launchers remain healthy.

At 01:43, seed102 had completed `crn`, `cyclegan`, `dalle`, `biggan`, and
`deepfake`, then reached 500/1250 on `gaugan`. Independent recalculation across
the five prediction files gives 88.27% macro Accuracy / 90.52% macro AP and
78.92% / 97.62% real/fake accuracy. The other four domains exceed 94% on both
metrics, so the retained `crn` bias remains the main cause of the lower macro
result rather than an execution failure. All eight ranks remain healthy.

At 03:14, seed102 completed ten domains and entered `imle` at 200/1596. The
independently recalculated ten-domain macro result is 90.01% Accuracy / 93.24%
AP with 88.10% / 91.92% real/fake accuracy. `guided` is the weakest new domain
at 83.35% Accuracy but 95.84% AP; its 98.80% real versus 67.90% fake accuracy
shows a threshold bias opposite to `crn`. The three GLIDE variants reach
91.05%--94.85% Accuracy and 95.02%--97.45% AP. These weak and opposing domain
behaviors are preserved, and the distributed run remains healthy.

At 05:43, seed102 completed seventeen domains and entered `stargan` at 50/500.
The independently recalculated macro result fell to 87.68% Accuracy / 90.30%
AP with 82.47% / 92.88% real/fake accuracy. `imle` is only 51.60% / 59.58%
Acc/AP with 3.23% real accuracy, closely matching this seed's `crn` failure;
`seeingdark` reaches 61.39% / 57.69% with 22.78% real and 100% fake accuracy.
In contrast, `progan` reaches 99.89% / 100.00%. Every weak and strong result is
retained. Only `stargan` and `stylegan` remain, with all ranks healthy.

UFD seed102 completed all 19 domains at 07:12:52 after 8h26m39s. The official
log reports 88.51% Acc / 91.16% AP; independent recalculation gives 88.5146% /
91.1639%, 7.09 / 8.15 points below the paper. TTA raises Accuracy by 2.07
points but lowers AP by 5.97 points versus this checkpoint's static evaluation.
All 88,353 unique indices are covered, 23 padding records are retained, all
ranks and launchers exited, and every target GPU was released. The first local
cross-seed comparison failed at import because system Python lacked numpy; the
same tracked script then passed in A6000's `cl` environment. UFD three-seed
official TTA now averages 89.91% +/- 1.62 Accuracy and 92.49% +/- 2.60 AP.
`crn` and `imle` each span more than 48 Accuracy points across seeds, so the
domain behavior is not stable. GenImage seed100 is the next P3 run.

GenImage seed100 attempt 1 started on the same three hosts at 07:22:55, but
failed before the first batch. A6000 ranks 0--3 treated the literal
`hf_arrow://` URI as an ImageFolder path and raised `FileNotFoundError` for
`test/ADM`; ranks 4--7 then exited with secondary NCCL errors after their peers
disappeared. The earlier Arrow preflight had checked state files and arguments,
not the actual `Dataset_Creator_GenImage` construction path, so it produced a
false positive after the A6000 runtime drifted behind the other two hosts. All
eleven logs and the zero-prediction outcome are preserved under
`genimage/seed100/tta_attempt1_missing_a6000_genimage_arrow`.

The stale A6000 runtime was backed up and replaced with the byte-identical
3090/4090-2 implementation (`25f09044...`). The tracked launcher now performs
a real first-domain GenImage Arrow dataset construction before a preflight can
pass. Seed100 remains next; it must pass this stronger check on all three hosts
and restart from an empty official output directory.

At 07:36, all three hosts passed that strengthened preflight. Each imported the
actual pinned GenImage creator and constructed `ADM` as a non-empty 12,000-row
Arrow dataset; launcher/checkpoint hashes, NCCL 23007, idle GPUs, and absent
official output directories were also rechecked. The A6000 `cl` environment
passed all 12 focused protocol tests. The clean seed100 retry is now cleared to
launch, while seed101/102 remain blocked by execution order.

The clean seed100 retry launched at 07:39:27. A6000 runs ranks 0--3, 3090
runs 4--5, and 4090-2 runs 6--7 on port 29643. Every launcher repeated the
12,000-row ADM runtime smoke, all eight ranks crossed the distributed barrier,
and rank0 entered `ADM` at 0/1500 with an 8,437 MiB peak. The three GPUs are
active and no traceback, runtime, OOM, or collective error is present. This
attempt is running; seed101 remains queued.

At 08:50, seed100 completed `ADM` in 1h09m32s and entered `BigGAN`; by 09:15
it had reached 500/1500. Independent recalculation over all 12,000 unique ADM
indices exactly matches the rounded official result: 62.06% Acc / 92.26% AP,
99.98% real accuracy, and only 24.13% fake accuracy, with no sampler padding.
Compared with this checkpoint's static ADM result, TTA raises Accuracy by 0.71
points but lowers AP by 2.79 points. It is also 23.48 / 6.04 points below the
released-checkpoint P2 ADM Acc/AP. This weak, strongly real-biased result is
preserved without changing the seed; all ranks remain healthy.

`BigGAN` completed at 10:00 after 1h09m39s. Independent recalculation gives
83.91% Acc / 99.24% AP and 99.98% / 67.83% real/fake accuracy over 12,000
unique indices with no padding. Unlike ADM, TTA improves this checkpoint's
static BigGAN result by 23.64 Accuracy points and 1.45 AP points, although
Accuracy remains 14.78 points below the released-checkpoint P2 result. The
two completed domains average 72.98% Acc / 95.75% AP. `glide` reached 300/1500
at 10:15, with all ranks healthy; the opposing ADM and BigGAN responses are
both retained.

`glide` completed at 11:09 after 1h09m18s. Its 12,000 unique predictions give
83.09% Acc / 98.92% AP and 99.98% / 66.20% real/fake accuracy without padding.
TTA improves the trained checkpoint's static glide Accuracy by 2.06 points but
lowers AP by 0.25 points; it remains 12.86 Accuracy points below P2's released
checkpoint. The three-domain mean is now 76.35% Acc / 96.80% AP, with 99.98%
mean real versus 52.72% fake accuracy. `Midjourney` reached 100/1500 at 11:15,
and all ranks remain healthy.

`Midjourney` completed at 12:18 after 1h08m51s. Independent metrics are 75.53%
Acc / 92.21% AP and 99.97% / 51.10% real/fake accuracy over all 12,000 unique
indices. TTA raises Accuracy only 0.73 points while reducing AP by 6.78 points
versus this checkpoint's static result, and remains 20.21 / 6.91 points below
P2 Acc/AP. The four-domain mean is 76.15% Acc / 95.66% AP, with the real/fake
mean still split at 99.98% / 52.32%. `stable_diffusion_v_1_4` reached 600/1500
at 12:46; no execution error is present.

`stable_diffusion_v_1_4` completed at 13:27 after 1h08m40s. It is effectively
perfect: 99.99% Acc / 99.99% AP, 100.00% real, and 99.98% fake accuracy over
12,000 unique samples. The changes from this checkpoint's 100% static result
are below 0.01 points, and it is slightly above P2 on both metrics. The five
completed domains average 80.92% Acc / 96.52% AP, while mean fake accuracy is
still only 61.85% because of the earlier cross-generator domains. SD1.5 reached
450/2000 at 13:48 with all ranks healthy.

`stable_diffusion_v_1_5` completed at 14:58 after 1h31m16s. Independent
recalculation over all 16,000 unique predictions gives 99.9125% Acc / 99.9128%
AP and 99.95% / 99.875% real/fake accuracy without padding. This is within
0.03 points of the trained checkpoint's static result and 0.18 / 0.04 points
above P2 Acc/AP. The six-domain mean is now 84.08% Acc / 97.09% AP, but mean
fake accuracy remains 68.19% because the weak cross-generator domains are
retained. At 15:20, `VQDM` reached 450/1500 and all ranks remained healthy.

`VQDM` completed at 16:06 after 1h08m28s. All 12,000 unique predictions give
95.6667% Acc / 99.6831% AP and 99.9667% / 91.3667% real/fake accuracy without
padding. TTA raises this checkpoint's static Accuracy by 0.57 points but lowers
AP by 0.24 points; it remains 3.13 Accuracy points below P2 while AP is tied to
within 0.01 points. The seven-domain mean is 85.74% Acc / 97.46% AP. At 16:19,
`wukong` reached 200/1500; all three launchers and GPUs remained active.

GenImage seed100 completed all eight domains at 17:15:11 after 9h35m44s.
`wukong` reached 99.8583% Acc / 99.9775% AP. The independently recalculated
macro result is 87.5026% Acc / 97.7740% AP, 9.20 / 1.73 points below the paper
and 9.27 / 1.72 points below P2's released checkpoint. Relative to this trained
checkpoint's static evaluation, TTA raises Accuracy by 3.47 points but lowers
AP by 1.09 points. Mean real/fake accuracy remains split at 99.9792% / 75.0260%,
with ADM, Midjourney, BigGAN, and glide driving the deficit. All 100,000 unique
indices are covered without padding, every rank and launcher exited, all GPUs
were released, and no execution error was found. Seed101 is next after a fresh
three-host preflight; no weak result is discarded.

Fresh seed101 preflights passed on all three hosts between 17:22 and 17:23.
The GPUs were idle, port 29644 was free, all official output directories were
absent, and the checkpoint (`4cbaec6...`), launcher, and runtime hashes matched.
Every host also constructed the actual 12,000-row ADM Arrow dataset. Seed101
then launched at 17:24:02 in the fixed A6000 4 + 3090 2 + 4090-2 2 layout.
All eight ranks crossed the barrier, rank0 entered ADM at 0/1500 with an 8,437
MiB peak, and no startup error was found. Seed102 remains queued.
