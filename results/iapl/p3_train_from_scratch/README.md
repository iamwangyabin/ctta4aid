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

The GenImage source on 3090 currently contains 985 of 1,214 Arrow shards. It is
not being treated as complete. Instead, the 96.4 GB, 30-part official SD1.4
archive mirror is being downloaded on 4090-2. The first transfer stopped after
43 GB when the mirror closed a response early; the Hugging Face cache was
retained and a lower-concurrency resume was started. The archive parts will be
verified and converted to the project's Hugging Face Arrow format before any
GenImage training begins. No partial-data result will be reported as the
official training reproduction.

The conversion path is fixed in `scripts/import_genimage_to_hf_arrow.py`. It
requires a JSON plan with explicit subset, split, label, source directory and
optional expected counts/bytes; writes `save_to_disk` Arrow shards plus
`mapping.json` and split metadata; and checks representative source/Arrow byte
hashes before atomically publishing the output. GenImage training then uses the
authors' `run_genimage.sh` settings through
`scripts/run_iapl_genimage_train_single.sh` and refuses non-Arrow input.
