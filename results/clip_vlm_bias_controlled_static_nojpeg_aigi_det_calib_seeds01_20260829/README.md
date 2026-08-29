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

## Does it threaten Ours?

为避免样本差异，`same_sample_r47_seed1.json` 把三个静态 detector 限定到 R47/Ours
seed 1 的锁定在线清单（每 target 1500 张）。更严格的因果版本先用源阈值预测前 100
张，再从这 100 张无标签 logits 估计 offset，只校准后 1400 张。下表 Accuracy 是完整
1500 张流的组合值；R47 是同一 1500 张上的在线 target-macro 值。

| Dataset | RINE + causal calibration AUC / Acc. (%) | R47/Ours online AUC / Acc. (%) | Δ RINE−Ours (pp) |
| --- | ---: | ---: | ---: |
| GenImage | 87.384 / 78.705 | 84.563 / 78.838 | +2.821 / -0.133 |
| AIGCDetectionBenchmark | 84.029 / 75.455 | 83.435 / 76.302 | +0.594 / -0.847 |
| AIGI-Holmes P3 | 87.708 / 78.073 | 90.777 / 83.447 | -3.069 / -5.373 |
| OpenSDID Global | 92.863 / 82.560 | 96.224 / 90.853 | -3.360 / -8.293 |

结论是：**它对“Accuracy 提升来自复杂 TTA”这种宽泛叙述构成明显威胁，但对我们完整
在线方法只构成部分威胁。** RINE 的静态排序在 GenImage/AIGC 上本来就很强；若允许
先看每个 target 的校准集并回溯应用阈值，Accuracy 会进一步提高。但在因果完整流设置
中，R47/Ours 的 Accuracy 四套数据都更高，尤其 P3 与 OpenSDID 分别高 5.373 和
8.293 个百分点；Ours 还在两个新数据集上保持更高 AUC。

不过这里的 R47 只是 `commit=3b73f51` 的单 seed exploratory candidate，本身明确
不具备正式表格资格；上表只能用于威胁诊断，不能作为最终论文结论。AIGI-Det-Calib
知道 target 边界并先收集校准样本，而 R47 是不使用 generator ID 的
Predict-Then-Adapt continual stream，因此二者也不能混成同协议 SOTA 排名。

## Files

- `per_seed_summary.json`: 两个 seed、39 个 target 的完整 offset/Accuracy 审计及身份哈希。
- `seeds01_summary.json`: 逐 target 与数据集 target-macro 的两 seed 均值/样本标准差。
- `same_sample_r47_seed1.json`: 与 R47 锁定清单同样本的随机、因果和回溯校准诊断。
- `table.csv`: 可直接制表的逐 target 与 Mean 扁平汇总。
- `source_models.json`: 官方 commit/文件哈希、发布代码问题、环境与基础结果验收记录。

本目录不包含 raw predictions、checkpoint、日志、PID、monitor 或失败过程文件，也不把
有标签 oracle/官方 label-balanced 路径冒充无标签正式结果。
