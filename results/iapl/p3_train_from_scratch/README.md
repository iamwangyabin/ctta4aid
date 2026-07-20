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

The GenImage source on 3090 currently contains 985 of 1,214 Arrow shards. It is
not being treated as complete. The missing shards and the SD1.4 subset must be
verified before GenImage training starts; no partial-data result will be
reported as the official training reproduction.
