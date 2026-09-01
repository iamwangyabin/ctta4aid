# AIGI-Det-Calib Audit on Static No-JPEG Baselines, Seeds 0–1

本目录记录 AAAI 2026 方法 [AIGI-Det-Calib](https://github.com/muliyangm/AIGI-Det-Calib) 在
既有 UnivFD、RINE、NPR `matched_jpeg` logits 上的后处理审计。它不重新训练、不更新
backbone，只给二分类 logit 加一个标量偏置，等价于移动决策阈值。因此 **AUC 不会改变**；
它只能修复 score scale/threshold，不能修复排序能力。论文见
[arXiv:2602.01973](https://arxiv.org/abs/2602.01973)。

## Primary label-free audit

主审计对每个 `method × seed × target` 无标签随机抽取 100 张，只把 logits 送入作者
unsupervised core；抽样不读取标签，并在排除这 100 张后的 holdout 上评估。运行 10 个
确定性重复。下表为两个 detector seed 的 target-macro 均值 ± 样本标准差（%）：
每格依次为 `AUC / 原始 Acc@0.5 → 校准 Accuracy`；前两项使用完整 target，校准
Accuracy 使用排除 100 张 calibration subset 后的 holdout。

| Method | GenImage | AIGCDetectionBenchmark | AIGI-Holmes P3 | OpenSDID Global |
| --- | ---: | ---: | ---: | ---: |
| UnivFD / Ojha | 79.934±0.107 / 62.583±0.034 → **73.326±0.071** | 77.794±0.107 / 61.928±0.027 → **71.743±0.068** | 87.857±0.085 / 68.172±0.025 → **80.472±0.030** | 92.252±0.115 / 72.935±0.078 → **84.085±0.117** |
| RINE | 85.893±1.749 / 55.136±3.220 → **79.179±1.552** | 82.946±1.683 / 54.050±2.641 → **76.089±1.327** | 86.818±1.103 / 51.117±1.121 → **79.261±1.223** | 93.007±0.689 / 54.875±4.023 → **85.158±0.554** |
| NPR | 49.633±3.628 / 50.967±0.278 → **49.092±2.860** | 51.065±2.663 / 50.704±0.272 → **50.555±1.868** | 56.611±6.212 / 50.345±0.198 → **54.128±4.655** | 61.770±5.450 / 50.710±0.368 → **58.254±4.301** |

## What the released method actually does

- Supervised calibration uses per-class KDEs from labeled target samples and chooses one additive
  threshold offset that maximizes class-balanced accuracy.
- Unsupervised calibration sets the boundary from the target logit distribution’s first moment. With
  the released `real_ratio=0.5`, this is effectively the mean target logit.
- The official `run.py` path is **not label-free end to end**: it first calls
  `balanced_sample(label)` and selects equal real/fake counts, then passes those logits to the
  label-agnostic unsupervised core. This violates this project’s target-label embargo.
- The paper states 100 validation images and 10 runs. The README says 50 shots/20 runs; the CLI
  defaults to 100/20; but `run_n_experiment` fails to forward `n_shot`, so the released code always
  uses the sampler default of **20 shots (10 real + 10 fake)** and 20 runs.
- The released benchmark standard-deviation expression also reduces the wrong array axis. These
  issues are recorded in `source_models.json`; no upstream code is vendored because the fixed commit
  has no license file.

## Findings

1. **RINE is the strongest calibration beneficiary.** Its two-seed macro Accuracy rises from
   55.136→79.179 on GenImage, 54.050→76.089 on AIGCDetectionBenchmark, 51.117→79.261 on
   P3, and 54.875→85.158 on OpenSDID. The previously collapsed Accuracy was largely threshold
   misalignment rather than lost ranking ability.
2. UnivFD also gains 9.8–12.3 accuracy points, but remains below calibrated RINE on GenImage and
   AIGCDetectionBenchmark.
3. NPR is the necessary negative control. Its AUC is near chance on the older benchmarks, so
   unsupervised scalar calibration cannot rescue it; on GenImage it even drops from 50.967 to
   49.092 Accuracy. Calibration is not a substitute for discriminative features.
4. Label-balanced and strictly label-free sampling are close for UnivFD/RINE because these target
   sets are balanced, but only the latter is admissible as a no-label diagnostic here.

## Files

- `per_seed_summary.json`: 两个 seed、39 个 target 的完整 offset/Accuracy 审计及身份哈希。
- `seeds01_summary.json`: 逐 target 与数据集 target-macro 的两 seed 均值/样本标准差。
- `table.csv`: 可直接制表的逐 target 与 Mean 扁平汇总。
- `source_models.json`: 官方 commit/文件哈希、发布代码问题、环境与基础结果验收记录。

本目录不包含 raw predictions、checkpoint、日志、PID、monitor 或失败过程文件，也不把
有标签 oracle/官方 label-balanced 路径冒充无标签正式结果。
