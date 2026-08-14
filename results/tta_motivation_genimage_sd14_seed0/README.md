# GenImage SD v1.4 TTA Motivation Study, Seed 0

本目录保存一轮完整的动机实验最终结果。Controlled 与 IAPL 使用以 SD v1.4 为训练
来源的模型；OST 使用作者发布的 MetaXception 初始化，并在 Full 模式读取固定的
SD v1.4 源模板。三条轨道都在其余七个 GenImage 生成器上做逐目标独立评估。每个
variant 共评估 88,000 张测试图；表中 AUC、BAcc 和差值均为七个 target 的等权宏平均。

这不是跨 backbone 排名。Controlled ResNet-50、IAPL CLIP ViT-L/14 和 OST
MetaXception 只允许在各自轨道内部做配对比较。

## Results

| Track | Variant | AUC (%) | BAcc (%) | Paired delta AUC (pp) | Paired delta BAcc (pp) | Negative AUC targets | Time (ms/image) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled | Source | 69.169 | 65.113 | - | - | - | 0.368 |
| Controlled | TENT | 52.276 | 52.874 | -16.893 | -12.239 | 7/7 | 1.245 |
| Controlled | EATA | 65.802 | 63.843 | -3.367 | -1.270 | 5/7 | 0.961 |
| Controlled | T2A | 58.072 | 55.296 | -11.097 | -9.817 | 6/7 | 3.315 |
| IAPL | Static | 96.566 | 89.987 | - | - | - | 10.896 |
| IAPL | Views-only | 99.606 | 96.010 | +3.040 | +6.023 | 0/7 | 76.220 |
| IAPL | Full | 99.509 | 95.920 | -0.097 | -0.090 | 4/7 | 369.310 |
| OST | Static | 50.444 | 49.894 | - | - | - | 2.532 |
| OST | Full | 49.835 | 49.851 | -0.609 | -0.043 | 6/7 | 12.472 |

Paired baseline 分别为 Source、IAPL Static、IAPL Views-only 和 OST Static。
IAPL Full 相对 Static 的总变化仍为 `+2.943` AUC pp、`+5.932` BAcc pp；上表把
Full 配对到 Views-only，是为了单独测量两步 prompt adaptation 的增量。

## Findings

1. 公共 ResNet-50 上，TENT、EATA 和 T2A 的宏平均都低于 Source。TENT 在七个
   target 上全部负迁移，T2A 在六个 target 上负迁移；EATA 损失较小，但仍有五个
   target 的 AUC 下降。
2. IAPL 完整管线明显优于单视图 Static，但收益已经由 32 views 与 OIS 全部实现。
   在 Views-only 之上加入两步 prompt update 后，AUC 反而下降 0.097 pp，BAcc 下降
   0.090 pp，且运行时间由 76.2 增至 369.3 ms/image。
3. OST 的一步 fast-weight update 没有改善公开 MetaXception 初始化：宏平均 AUC
   下降 0.609 pp，六个 target 负迁移。Static 与 Full 都接近随机水平。

因此，这轮结果不支持“测试时参数更新本身在 GenImage 伪造检测上普遍有效”。它支持的
更精确结论是：专用测试时管线可能有效，但必须把多视图/选择收益与真正的参数适应收益
拆开。IAPL 的提升来自前者，而本设置中的 prompt update 和 OST fast-weight update
都没有带来正增量。

## Protocol

- Seed 为 0；SD v1.4 的 12,000 张源域测试图不进入目标评估。
- Controlled 组共享同一个 ResNet-50 checkpoint、Fisher、样本、顺序和阈值。
- IAPL 的 Static、Views-only 和 Full 共享同一个作者 checkpoint。Full 保留 32 views、
  OIS、每图两步适应以及每图 prompt/optimizer reset。
- OST 的 Static 与 Full 共享作者发布的 `xception_meta.pth`。Full 使用固定的 SD v1.4
  源模板 Arrow 子集和已披露的 full-frame alpha blending；这不是作者人脸 benchmark
  的原数值复现。
- 每个 target 都重新加载模型，target 间不继承状态。IAPL Full 按 target 分发到两台
  相同型号的 RTX 4090；`ADM`、`BigGAN` 在 4090-1，其余五域在 4090-2。
- 代码版本为 `1a1add0cbcb9d455a24a632bc38991f57f67d746`。数据及 checkpoint
  哈希见 `source_models.json`。

本目录只报告一个正式 seed，不能替代三 seed 的最终主实验。完整逐域指标和全部配对
差值位于 `seed0_summary.json`，论文表格的扁平版本位于 `table.csv`。
