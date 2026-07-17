# Vendored official method cores

This directory is the algorithm boundary. Files under `methods/` are protocol
wrappers; they may move tensors, expose `predict/adapt/reset`, record metrics,
and translate configuration, but must not reimplement the published method.

## TENT

- Upstream: `DequanWang/tent`, commit `e9e926a668d85244c66a6d5c006efbd2b82e83e8`
- Upstream file: `tent.py`
- Vendored file: `tent.py`
- Algorithm patches: none
- Wrapper change: prediction and the authors' forward-and-adapt call are invoked separately.

## EATA

- Upstream: `mr-eggplant/EATA`, commit `f739b3668cc7617e9b9f1979c1a358497a3472c3`
- Upstream file: `eata.py`
- Vendored file: `eata.py`
- Algorithm patches: none
- Wrapper changes: binary entropy margin, checkpoint-supplied Fisher, protocol split, and full reset of probability EMA/counters.

## CoTTA

- Upstream: `qinenergy/cotta`, commit `c212a204b32be4005092e4323105a24a29ad2952`
- Upstream files: `imagenet/cotta.py`, `imagenet/my_transforms.py`
- Vendored files: `cotta.py`, `cotta_transforms.py`
- Compatibility patches:
  - package-relative transform import;
  - torchvision `resample/fillcolor` replaced by `interpolation/fill`;
  - hard-coded `.cuda()` restore mask replaced by the parameter device;
  - hard-coded constants exposed as constructor arguments with unchanged ImageNet defaults;
  - official teacher prediction extracted and optionally supplied to adaptation so the framework can enforce Predict-Then-Adapt without drawing a second augmentation set.
- Wrapper change: ImageNet-normalized tensors are converted to pixel space only around the authors' augmentation.

## RoTTA

- Upstream: `BIT-DA/RoTTA`, commit `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9`
- Upstream files: `core/adapter/rotta.py`, `core/adapter/base_adapter.py`, `core/utils/memory.py`, `core/utils/bn_layers.py`, `core/utils/custom_transforms.py`
- Vendored files: `rotta.py`, `rotta_transforms.py`
- Preserved algorithm components: RobustBN source/target statistics, CSTU class-balanced memory, uncertainty/timeliness eviction, timeliness-weighted teacher loss, strong augmentation, EMA teacher, and update-frequency scheduling.
- Compatibility patches:
  - upstream modules are consolidated behind a package-local core;
  - deprecated torchvision `resample/fillcolor` arguments are replaced by `interpolation/fill`;
  - hard-coded `.cuda()` timeliness weights use the current tensor device;
  - source snapshot, optimizer snapshot, memory, teacher, counters, and transforms are all reset by the framework reset contract;
  - image size is exposed with the original value represented in configuration.
- Task/protocol changes: `NUM_CLASS` is 2 instead of CIFAR-10/100; input size is 224 instead of 32; the EMA output is computed and cached before the unchanged memory/update path to enforce Predict-Then-Adapt.
- Wrapper change: ImageNet-normalized tensors are converted to pixel space around the authors' strong augmentation and normalized again before the student forward.

## LAME

- Upstream: `fiveai/LAME`, commit `d2e5f63090bc1c8129bf7cbd781029a5955e1a67`
- Upstream file: `src/adaptation/lame.py`
- Vendored file: `lame.py`
- Algorithm patches: none to the RBF/kNN/linear affinities, unary construction, fixed-point Laplacian update, energy, or convergence rule.
- Wrapper changes: Detectron2 model/result plumbing is replaced by the framework detector's same-pass penultimate features and logits; LAME output adaptation executes in `predict`, while `adapt` is a state-free no-op; a singleton batch returns the source probability because the released RBF bandwidth is undefined at `N=1`.
- License boundary: the vendored core is CC BY-NC-SA 4.0. See `THIRD_PARTY_NOTICES.md`; it must not be treated as MIT project code.

## T²A

- Upstream: `HongHanh2104/T2A-Think-Twice-Before-Adaptation`, commit `33c8ccc64afdda260564123d6c790d030a89ff81`
- Upstream files: `adapters/base_adapter.py`, `adapters/T2A.py`, `losses/__init__.py`, `utils.py`
- Vendored files: `t2a.py`, `t2a_losses.py`
- Required executable repairs:
  - initialize `entropy_fn`, `e_margin`, `filter_grad`, and `cosine_strategy`;
  - sample one non-pseudo complementary class index from the released `1-p` weights instead of passing a `B×C` Bernoulli matrix to `gather`; binary detection deterministically selects the other class;
  - pass logits to normalized log-softmax losses;
  - deep-copy reset state;
  - identify BN parameters by identity for gradient masking;
  - guard degenerate normalized-loss denominators;
  - extract `adapt` and `predict` from the authors' combined forward.
- Unresolved paper/code discrepancy: public `entropy_minimization` applies
  `1 / exp(H - e_margin)` weighting, while the paper describes ordinary
  entropy minimization and the released YAML does not define `e_margin`.
  The port retains the public-code weighting and exposes the binary margin
  only under `release_repairs`; it requires author clarification/sensitivity
  analysis before numerical fidelity can be certified.
- Wrapper change: tensor-output detector is presented to the authors' adapter as its expected `{"cls": logits}` model.
- Protocol wrapper change: the pre-adaptation prediction restores BatchNorm running buffers so it cannot update persistent state before `adapt`.

## IAPL

IAPL is intentionally not vendored because its repository has no software
license. `run_iapl_official.py` fetches and executes the pinned authors'
checkout. The verifier permits only the exact changes recorded in
`patches/iapl-a173e77-compat.patch`.

The same audited compatibility patch also opts out of PyTorch 2.6+
`weights_only=True` checkpoint loading and adds an ImageFolder-compatible
adapter for memory-mapped Hugging Face Arrow datasets. It does not change IAPL's
transforms, sample labels, TTA views, optimizer, or adapt-then-predict protocol.

P0 diagnostic runs apply two additional, separately recorded patches after the
compatibility patch. `iapl-a173e77-prediction-capture.patch` only saves sampler
indices, labels, and probabilities. `iapl-a173e77-bn-buffer-ablation.patch`
adds opt-in environment switches for restoring checkpoint BatchNorm buffers
before every sample and disabling DDP buffer broadcasts. Both switches default
to the authors' released behavior. Results produced with either ablation must be
reported as diagnostics, not as unmodified official-protocol reproduction.
