# Static No-JPEG Baselines on `matched_jpeg`, Seeds 0–1

本目录保存 UnivFD/Ojha、RINE 和 NPR 三个**非 TTA 静态检测器**的两 seed 补充结果。
三者都从 GenImage SD v1.4 的同一 Arrow 源训练集重新训练，源训练时关闭 JPEG 扰动
分支；随后冻结各自 checkpoint，直接在四套 `matched_jpeg` target 上推理。这里没有测试时
参数更新、teacher、memory 或 prompt adaptation，也没有用 target label 选择 checkpoint、
阈值或超参数。

本轮按用户要求运行 `seed=0,1`，没有继续运行 seed 2。因此它用于观察静态方法的初步
稳定性，但不是 CLIP 主实验规定的三 seed 正文结果。不同方法保留各自原生 backbone 和
训练 recipe，不能把本表解释为同 backbone 的受控消融。

## Results

AUC 和 Accuracy 均先在数据集内部对 target 等权宏平均，再报告两个 seed 的均值 ± **样本
标准差**；Accuracy 阈值固定为 `0.5`。GenImage 排除作为源训练域的 SD v1.4。每个
`method × seed` 共评估 39 个 target、152,000 张图像。

| Method | GenImage AUC / Acc. (%) | AIGCDetectionBenchmark AUC / Acc. (%) | AIGI-Holmes P3 AUC / Acc. (%) | OpenSDID Global AUC / Acc. (%) |
| --- | ---: | ---: | ---: | ---: |
| UnivFD / Ojha | 79.934±0.107 / 62.583±0.034 | 77.794±0.107 / 61.928±0.027 | **87.857±0.085** / 68.172±0.025 | 92.252±0.115 / 72.935±0.078 |
| RINE | **85.893±1.749** / 55.136±3.220 | **82.946±1.683** / 54.050±2.641 | 86.818±1.103 / 51.117±1.121 | **93.007±0.689** / 54.875±4.023 |
| NPR | 49.633±3.628 / 50.967±0.278 | 51.065±2.663 / 50.704±0.272 | 56.611±6.212 / 50.345±0.198 | 61.770±5.450 / 50.710±0.368 |

粗体只标记同一数据集内最高的两 seed 平均 AUC；Accuracy 未加粗。`table.csv` 包含全部
39 个 target 的 AUC、Accuracy 和 AP 均值/标准差。

## Findings

1. 两 seed 后结论与 seed 0 一致：RINE 在 GenImage、AIGCDetectionBenchmark 和
   OpenSDID Global 上取得最高平均 AUC，UnivFD 在 AIGI-Holmes P3 上最高。
2. UnivFD 最稳定，四套数据的宏平均 AUC 标准差仅为 0.085–0.115 个百分点。RINE 的
   AUC 标准差为 0.689–1.749 个百分点，仍保持较强排序能力，但固定阈值 Accuracy 的
   标准差达到 1.121–4.023 个百分点，说明跨 seed 分数标定更敏感。
3. NPR 的两个源验证 AUC 都超过 99.94%，但 target 平均 AUC 较低且标准差为
   2.663–6.212 个百分点。训练均已完成；结果表明其残差线索在当前编码/几何匹配的跨生成器
   设置下不仅泛化有限，而且对训练随机性较敏感。
4. 固定 `0.5` 阈值下，UnivFD 的 Accuracy 明显高于另外两种方法。本轮没有读取 target
   label 做阈值重标定，因此 Accuracy 同时反映排序能力和源域分数尺度的迁移情况。

## Protocol

- Seeds 固定为 0 和 1。UnivFD 每个 seed 训练 10 epochs；RINE 和 NPR 每个 seed 训练
  1 epoch。checkpoint 只按 12,000 张 SD v1.4 源验证图的 AUC 选择。
- 源训练 Arrow 共 323,997 行。两台服务器上的 100 个文件清单聚合哈希一致；源
  inventory、过滤索引、实际 checkpoint SHA-256、公开上游 commit 和实现文件哈希见
  `source_models.json`。
- 训练时 `jpg_prob=0`，没有 `RandomCompress` 或随机 JPEG 重编码。普通几何增强保留；
  UnivFD 和 RINE 仍按各自 recipe 保留 `p=0.5` blur，NPR 不使用 blur。
- UnivFD 与 RINE 从同一个固定 OpenAI CLIP ViT-L/14 权重开始，SHA-256 为
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`；
  NPR 使用官方 `resnet50(pretrained=False, num_classes=1)` 残差检测器。
- RINE 对 GenImage 中短边小于 224 的自然图使用 `Resize(short edge=256)` 几何桥接后
  再 crop；该桥接不包含 JPEG 扰动。
- 正式 target profile 固定为 `matched_jpeg`。验收器验证了六个运行中全部 39 个 target
  的 `sample_id_sha256`、`label_sha256` 和 profile manifest 身份一致；模型始终不读取
  target hidden labels。
- 评测均为冻结推理，无 TTA。六个实际 best checkpoint 文件的 SHA-256 均与训练和评测
  JSON 记录一致。
- AIDE 仍未产生数值：其固定公开版本的完整 ConvNeXt-XXL/双分支 20-epoch 路径未满足
  可审计启动条件，没有用缩减 recipe 或项目自写替代实现冒充官方复现。

## Files

- `per_seed_summary.json`：六个运行的逐 seed、逐数据集、逐 target 指标及身份哈希。
- `seeds01_summary.json`：两个 seed 的逐 target 均值/样本标准差及数据集宏平均。
- `table.csv`：论文表格可直接读取的扁平汇总。
- `source_models.json`：源数据、训练 recipe、实现版本、checkpoint 和验收记录。

本目录只包含最终且可解释的汇总，不包含 checkpoint、raw predictions、日志、PID、monitor
或失败过程文件。原有
`results/clip_vlm_bias_controlled_static_nojpeg_seed0_20260829/` 单 seed 快照保持不变。
