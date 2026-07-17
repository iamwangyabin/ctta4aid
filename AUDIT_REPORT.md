# Baseline reproduction audit

Audit date: 2026-07-17

## Outcome

| Method | Code audit result | Numerical reproduction result | Main-table role |
| --- | --- | --- | --- |
| Source-only | correct common control | requires project checkpoint/data | required |
| TENT | vendored authors' official core plus protocol wrapper | not run: official data/weights unavailable | required general TTA baseline |
| EATA | vendored authors' official core; binary wrapper and full Fisher enforced | not run: official data/weights unavailable | required reliability/anti-forgetting baseline |
| CoTTA | patched vendored official ImageNet/ResNet-50 core | not run: official data/weights unavailable | required continual TTA baseline |
| RoTTA | patched vendored official RobustBN/CSTU/EMA core | not run: CIFAR PTTA data/weights unavailable | important dynamic-stream baseline |
| LAME | vendored official affinity/Laplacian core; parameter-free wrapper | not run: official benchmark models/data unavailable | important output-adaptation control |
| T²A | patched vendored public adapter core; release cannot execute unmodified | not run: original Deepfake data/weights unavailable | required closest task baseline, with repair qualifier |
| IAPL | pinned official checkout and compatibility-audited runner | UFD Arrow 19-domain run complete; Accuracy gate passed, AP gate failed | official AIGC-specific end-to-end baseline; not paper-number reproduced |

“Code audit passed” does not mean paper numbers have been reproduced. The machine-readable status written by CNN methods remains `not_run_requires_official_data_and_weights` until the original benchmark sanity runs are completed.

## Official source verification

The fetcher was executed from a clean destination and checked out all seven expected revisions:

- TENT `e9e926a668d85244c66a6d5c006efbd2b82e83e8`;
- EATA `f739b3668cc7617e9b9f1979c1a358497a3472c3`;
- CoTTA `c212a204b32be4005092e4323105a24a29ad2952`;
- RoTTA `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9`;
- LAME `d2e5f63090bc1c8129bf7cbd781029a5955e1a67`;
- T²A `33c8ccc64afdda260564123d6c790d030a89ff81`;
- IAPL `a173e7783bbafaa00d60e6e31774a0bc14411a23`.

The IAPL verifier additionally confirmed that the checkout differs from the pinned commit only by `patches/iapl-a173e77-compat.patch`: the hard-coded CLIP path uses the authors' existing argument, PyTorch checkpoint loading remains compatible with 2.6+, and the dataset creator accepts the framework's byte-preserving Arrow adapter.

## Corrections made by this audit

1. TENT wrapper now calls the vendored authors' `Tent` class and its official configuration, parameter collection, forward-and-adapt, state copy and reset.
2. EATA wrapper calls the vendored authors' `EATA` class; both filters, probability EMA, entropy weighting and Fisher/EWC remain in that core. Fisher estimation uses clean source-validation images, batch size 64 and 2,000 samples. Missing Fisher fails closed.
3. CoTTA's earlier independent implementation was replaced by the vendored official ImageNet class and transforms. Compatibility patches are limited to torchvision/device support, normalization bridging and protocol separation.
4. T²A's independent reconstruction was replaced by vendored `BaseAdapter`/`T2AAdapter`/loss code. Its public release is non-executable, so every required repair is explicit in `official/PROVENANCE.md`.
5. IAPL now runs the pinned authors' source, preserves the released per-image episodic Adapt-Then-Predict protocol, and fails if domains/mean are missing or reported mean Accuracy/AP misses the configured authors' reference tolerance.
6. Online streams now use a seeded global shuffle instead of path/class-blocked order. Every run saves `sample_manifest.csv` and source-checkpoint SHA-256.
7. Continual final average AUC is now the arithmetic mean of domain AUCs; pooled AUC is reported separately. Prediction, adaptation and total latency are also separated.
8. Method parameters are now centralized under `configs/methods/`; TENT is corrected to the authors' released Adam `lr=1e-3` profile, EATA uses the authors' `e_margin`/`d_margin`/Fisher names, and T²A's unreported executable repairs are isolated from released YAML values.
9. A complete 28-file CNN matrix now covers GenImage and UniversalFakeDetect, single-target and continual settings, and all seven CNN methods. IAPL retains two separate native-protocol configs.
10. RoTTA now uses the pinned authors' RobustBN, CSTU eviction/memory, timeliness weighting, strong augmentation, EMA teacher, and official update scheduling. Binary class count, 224 resolution, device/torchvision compatibility, reset, and protocol split are explicit patches.
11. LAME now uses the pinned authors' affinity and Laplacian optimization core. It adapts outputs in `predict`, keeps `adapt` stateless, and records the singleton RBF guard; its CC BY-NC-SA license is explicit.
12. RoTTA's strong augmentation now receives pixel-space tensors through the same normalization bridge used for CoTTA, instead of clipping ImageNet-normalized input directly.
13. T²A prediction now restores BatchNorm running buffers, and its repaired Bernoulli branch guarantees a non-pseudo complementary class.
14. Continual forgetting now comes from repeated evaluation on fixed, seeded, disjoint holdouts after every domain. The old online-prefix/final-holdout difference and label-blocked final loader were removed.

## Verification run

The final suite completed independently on both RTX 4090 hosts: 45 tests and 36 subtests passed on each host, with only four PyTorch deprecation warnings. The merged UFD artifact contains exactly 19 finite domain records, uses one model hash, one CLIP hash and the same two Arrow fingerprints throughout, and has SHA-256 `a9cbce00a1deb89a174a078eb3e2e3f3c3bb4bdadeba4ac89aa3c7a1756d4deb`.

Full numerical certification remains blocked: the completed IAPL UFD run misses the AP gate, while the other methods still lack their original benchmark runs. No result should be labeled “fully reproduced” until the gates in `REPRODUCIBILITY.md` are satisfied.
