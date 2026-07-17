# IAPL and CTTA P0-P5 research plan

启动日期：2026-07-17。所有阶段都保留失败结果、逐域指标、配置、资产哈希、日志和运行环境。
阶段只按 P0 到 P5 顺序推进；诊断失败不会被静默跳过。

## P0: five-domain protocol diagnosis

诊断域固定为 `crn`、`guided`、`imle`、`san`、`seeingdark`。对照论文逐域
Accuracy/AP，并依次运行：

1. 已完成的单域、单 GPU、每域重启进程结果；
2. 单 GPU、五域同进程顺序运行；
3. 两节点真实 DDP，用于验证 sampler、rank seed 和 buffer 同步的影响；
4. 八进程、八 rank 真实 DDP，使用同一数据和官方域顺序；
5. Arrow 与由同一原始字节导出的 ImageFolder 后端一致性检查。

除域级 Accuracy/AP 外，instrumented run 保存 sampler index、label 和 probability，
用于比较样本级排序、重复 padding 样本和错误交集。

### Critical protocol finding

作者 TTA 代码只为每张图片恢复 prompt tensor 和 optimizer state。推理适应前调用
`model.train()`，Conditional Information Learner 中的 BatchNorm running buffers 没有
恢复；DDP 还会按默认行为广播 buffers。因此“按域拆成互相独立的进程”并不严格等价
于作者的整表单次分布式运行。P0 必须先量化这个差异，P1 才能给出可信结论。

当前实验室只有三张可容纳完整 IAPL 进程的空闲 GPU。NCCL 2.19 会明确拒绝
同卡多 rank；因此八 rank 运行使用 NVIDIA 从 NCCL 2.30 起提供的
`NCCL_MULTI_RANK_GPU_ENABLE=1` 实验功能，映射为 A6000 四 rank、两张 4090
各两 rank，并以 `NCCL_MAX_CTAS=2` 限制每个 rank 的 channel 资源。三机八 rank
CUDA all-reduce 已通过，所有 rank 得到同一和值 36。这个设置忠实保留八进程的
sampler padding、rank seed、DDP gradient/buffer 同步语义，但物理拓扑不是论文常见的
一卡一 rank；结果中必须单独披露，不能写成硬件拓扑完全一致。

## P1: UFD official full rerun

使用官方 19 域顺序、32 views、top-6 confidence selection、2 TTA steps、学习率
`5e-3`、OIS 和 8 ranks。以论文 95.61% mAcc / 99.32% mAP 为参考，两个指标
均需进入 1 个百分点门限；未进入门限也保存完整结果并继续报告。

八 rank 采用上节的 rank-faithful NCCL 2.30 同卡多 rank 映射。若后续获得八张
单独 GPU，再补一轮一卡一 rank 审计；在此之前不把物理拓扑标记为完全复现。

## P2: original GenImage

使用 SD1.4 checkpoint 和原版 8 测试域：ADM、BigGAN、GLIDE、Midjourney、
SD1.4、SD1.5、VQDM、Wukong。目标是核对论文 96.7% mAcc，不使用 CAIDBench
同名映射替代。

## P3: training reproduction

分别从 ProGAN 四类和 SD1.4 开始训练 IAPL，冻结完整配置和 checkpoint 哈希。
每条训练至少运行三个 seeds；每个 seed 都完成对应完整测试集，报告均值、样本标准差
和官方 checkpoint 对照。

## P4: inference ablations

按推理代价从低到高运行 TTA steps、views、confidence selection 数量、entropy loss、
OIS 开关。每项报告 mAcc、mAP、real/fake accuracy、单图延迟、峰值显存和相对完整
IAPL 的变化。

## P5: controlled CTTA table

在相同 CNN checkpoint、样本、顺序和 Predict-Then-Adapt 协议下运行 Source、TENT、
EATA、CoTTA、RoTTA、LAME、T2A。包含 independent single-target 与 continual stream，
每项三个 seeds，报告 online、final、forgetting、延迟和显存。IAPL 继续作为不同
backbone/协议的端到端参考，不能混入控制变量排名。
