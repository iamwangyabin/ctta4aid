# Static No-JPEG Baselines on `matched_jpeg`, Seed 0

本目录保存 UnivFD/Ojha、RINE 和 NPR 三个**非 TTA 静态检测器**的一轮补充结果。
三者都从 GenImage SD v1.4 的同一 Arrow 源训练集重新训练，源训练时关闭 JPEG
扰动分支；随后冻结 checkpoint，直接在四套 `matched_jpeg` target 上推理。这里没有
测试时参数更新、teacher、memory、prompt adaptation，也没有用 target label 选择
checkpoint 或超参数。

这是一轮按用户要求只运行 `seed=0` 的补充比较，不是 CLIP 主实验的三 seed 正文结果。
不同方法保留各自原生 backbone 和训练 recipe，因此表格用于比较 method-native 静态检测
能力，不应解释为同 backbone 的受控消融。

## Results

AUC 和 Accuracy 均为数据集内部逐 target 的等权宏平均；Accuracy 阈值固定为 `0.5`。
GenImage 排除作为源训练域的 SD v1.4，共评估其余 7 个 target。每种方法合计评估 39 个
target、152,000 张图像。

| Method | GenImage AUC / Acc. (%) | AIGCDetectionBenchmark AUC / Acc. (%) | AIGI-Holmes P3 AUC / Acc. (%) | OpenSDID Global AUC / Acc. (%) |
| --- | ---: | ---: | ---: | ---: |
| UnivFD / Ojha | 80.010 / 62.559 | 77.869 / 61.909 | **87.918 / 68.190** | 92.334 / 72.990 |
| RINE | **84.656 / 57.412** | **81.756 / 55.918** | 86.038 / 51.910 | **93.494 / 57.720** |
| NPR | 47.068 / 51.164 | 49.182 / 50.897 | 52.218 / 50.485 | 57.917 / 50.970 |

粗体只标记同一数据集内最高 AUC；Accuracy 未加粗。完整 39 个 target 的 AUC、Accuracy、
AP、样本 identity 哈希和标签哈希位于 `seed0_summary.json`，论文表格可直接读取的扁平版本
位于 `table.csv`。

## Findings

1. RINE 在 GenImage、AIGCDetectionBenchmark 和 OpenSDID Global 上取得最高的
   target-macro AUC；UnivFD 在 AIGI-Holmes P3 上最高。
2. 固定 `0.5` 阈值下，UnivFD 的四套 Accuracy 都明显高于 RINE。RINE 的排序能力较强，
   但跨域分数尺度没有校准到同一个固定阈值；本轮没有使用 target label 做阈值重标定。
3. NPR 在干净 SD v1.4 源验证集上的 AUC 为 99.960%，但在 `matched_jpeg` target 上四套
   宏平均 AUC 仅为 47.068%--57.917%。这说明源训练已经收敛，而其残差线索在编码与几何
   匹配后的跨生成器设置中没有稳定泛化，不能把低 target 数值归因于没有完成训练。

## Protocol

- Seed 固定为 0。UnivFD 训练 10 epochs；RINE 和 NPR 各训练 1 epoch。checkpoint 只按
  12,000 张 SD v1.4 源验证图的 AUC 选择，最佳源验证 AUC 分别为 98.756%、99.999% 和
  99.960%。
- 源训练 Arrow 共 323,997 行。两台服务器上的 100 个文件逐文件 SHA-256 清单聚合哈希
  一致；源 inventory、过滤索引、模型 checkpoint、公开上游 commit 和实现文件哈希见
  `source_models.json`。
- 训练时 `jpg_prob=0`，没有 `RandomCompress` 或随机 JPEG 重编码。普通几何增强保留；
  UnivFD 和 RINE 仍按 recipe 保留 `p=0.5` blur，NPR 不使用 blur。
- UnivFD 与 RINE 从同一个固定 OpenAI CLIP ViT-L/14 权重开始，SHA-256 为
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。
  NPR 使用官方 `resnet50(pretrained=False, num_classes=1)` 残差检测器。
- RINE 上游源图默认足够大，但 GenImage 中存在短边小于 224 的自然图；本轮仅增加
  `Resize(short edge=256)` 的几何桥接，再执行其 crop。该桥接不包含 JPEG 扰动。
- 正式 target profile 固定为 `matched_jpeg`；三种方法共享 target identity、标签和
  target 顺序。汇总器已验证全部方法的 `sample_id_sha256` 与 `label_sha256` 一致。
- AIDE 没有产生数值：其固定公开版本的完整 ConvNeXt-XXL/双分支 20-epoch 训练路径未满足
  本轮可审计启动条件，因此没有用缩减 recipe 或项目自写替代实现冒充官方复现。

本目录仅提交最终汇总、复现身份和结论，不包含 checkpoint、raw predictions、日志、PID、
monitor 或失败过程文件。单 seed 的标准差字段为结构性 `0`，不得解释为跨 seed 稳定性。
