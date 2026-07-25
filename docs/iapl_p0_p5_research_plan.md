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

### P0 conclusion

P0 于 2026-07-17 完成。三机八 rank 五域均值为 87.99% Accuracy / 92.72% AP，
相对论文五域均值分别低 0.53 / 5.61 个百分点。Accuracy 已进入一个百分点，AP
差距仍明显。11 个 CNNDetection 域和 8 个公开 diffusion 域共 88,353 张均已逐字节
匹配官方归档；Arrow/ImageFolder、单进程/多进程、sampler padding、rank seed、
BatchNorm 每样本恢复和 DDP buffer broadcast 均已做对照。

四 rank SAN 中，关闭 buffer broadcast 只改变 AP +0.006 个百分点，每样本恢复
BatchNorm 只改变 -0.045 个百分点。单 rank SAN/SeeingDark 的 BatchNorm 效应方向
相反，两域平均 AP 反而降低 0.63 个百分点。因此主要差距不是数据副本、Arrow 后端、
随机视图的 rank 划分或简单 buffer 状态，而是论文 AP 与公开 checkpoint、公开数据和
公开推理路径能够产生的结果之间的 artifact-level reproduction gap。P1 继续完整报告
公开协议实测结果，不通过未披露的协议修改追数值。

## P1: UFD official full rerun

使用官方 19 域顺序、32 views、top-6 confidence selection、2 TTA steps、学习率
`5e-3`、OIS 和 8 ranks。以论文 95.61% mAcc / 99.32% mAP 为参考，两个指标
均需进入 1 个百分点门限；未进入门限也保存完整结果并继续报告。

八 rank 采用上节的 rank-faithful NCCL 2.30 同卡多 rank 映射。若后续获得八张
单独 GPU，再补一轮一卡一 rank 审计；在此之前不把物理拓扑标记为完全复现。

数据侧已对当前公开 UFD 19 域做完整官方归档核验：11 个 CNNDetection 域
72,353 张和 8 个 diffusion 域 16,000 张，共 88,353 张全部逐字节一致。UFD
仓库明确说明其论文曾在每个 diffusion 域随机评估 10,000 张，但公开包只发布
1,000 real + 1,000 fake；因此本项目的“官方协议复跑”严格指公开可复现的
2,000 张/域版本，不把无法取得的 10,000 张/域原始抽样宣称为已复现。

### P1 result

P1 于 2026-07-18 完成。19 域、88,353 个唯一公开样本全部得到样本级预测；
8-rank padding 后共 88,376 条记录，其中 23 条为可核验的 sampler 重复项。最终
mAcc 为 95.49%，相对论文 95.61% 低 0.12 个百分点，达到一个百分点标准；mAP
为 97.22%，相对论文 99.32% 低 2.10 个百分点，未达到标准。

AP 超出一个百分点的域为 CycleGAN、BigGAN、StarGAN、SeeingDark、SAN、CRN
和 IMLE。完整逐域 Accuracy/AP、real/fake Accuracy、indices、labels、probabilities
及日志均已归档。按研究计划保留这个负结果并进入 P2，不修改协议追数值。

## P2: original GenImage

使用 SD1.4 checkpoint 和原版 8 测试域：ADM、BigGAN、GLIDE、Midjourney、
SD1.4、SD1.5、VQDM、Wukong。目标是核对论文 96.7% mAcc，不使用 CAIDBench
同名映射替代。

### P2 result

P2 于 2026-07-19 完成。原版公开 `genimage_test.zip` 的 100,000 张图片全部
通过 ZIP CRC 和提取清单核验，三台计算节点的数据、代码及 SD1.4 checkpoint
哈希一致。官方 8-rank、32 views、2 TTA steps、OIS、smooth 协议得到 96.77%
mAcc / 99.49% mAP，相对论文 96.7% / 99.5% 分别为 +0.07 / -0.01 个百分点，
两个指标均达到一个百分点复现标准。所有样本级预测、8 个 rank 日志和 3 个
launcher 日志已归档；100,000 个 sampler index 全部唯一，无分布式 padding 重复。

## P3: training reproduction

分别从 ProGAN 四类和 SD1.4 开始训练 IAPL，冻结完整配置和 checkpoint 哈希。
每条训练至少运行三个 seeds；每个 seed 都完成对应完整测试集，报告均值、样本标准差
和官方 checkpoint 对照。

### P3 progress

截至 2026-07-24，UFD 三个 seed 的训练和非 TTA 静态评测已完成，完整官方
TTA 因 4090-1 离线而等待已验证的 4+2+2 rank 布局。GenImage 的官方 SD1.4
训练数据和 8 域测试数据已完成逐样本解码审计；源归档中的 3 个零字节 PNG
作为失败证据保留，并从训练元数据视图中精确排除。

GenImage seed100 的修复后训练已完成，静态结果为 84.03% mAcc / 98.86% mAP，
检查点哈希为
`aa0d8ab805f5c4fc846154e7da25ffae8cea32cbab9a8eb5ab3203ea27387096`。
该结果的 real/fake Accuracy 为 99.98% / 68.08%，说明固定阈值明显偏向 real；
这项负结果不做协议外校准。Seed101 随后得到 95.44% mAcc / 99.62% mAP，
seed102 得到 82.78% mAcc / 99.14% mAP。三 seed 静态汇总为
87.42 +/- 6.98% mAcc、99.21 +/- 0.38% mAP；real/fake Accuracy 分别为
99.91 +/- 0.10% 和 74.92 +/- 14.06%。这证明主要不稳定性来自未见生成器上的
固定阈值 fake 召回，而不是排序 AP；不挑选更优 seed，也不做协议外校准。

UFD 与 GenImage 的三 seed 从头训练链路和静态全测试集评测均已完成。P3 尚未
结束：六个训练检查点都还需要官方 8-rank TTA。已验证布局为
A6000 `4 ranks` + 4090-1 `2 ranks` + 4090-2 `2 ranks`，但 4090-1
截至 2026-07-24 17:18 仍离线，因此不使用失败过的 5+3 布局代替。3070x2
也不能作为临时节点：GPU 1 虽可被 `nvidia-smi` 单独看到，但 `cl` 环境的
PyTorch 在 `CUDA_VISIBLE_DEVICES=1` 下仍报告 CUDA 不可用且设备数为 0。

17:26 再检查时 A6000 已完全空闲，因此 6+2 rank 是不改变 8-rank 数据分片与
TTA 超参数的候选布局；但 48 GiB 上并发 6 个 IAPL 进程尚无显存预检证据。
一次成对启动只有 4090-2 的 ranks 6-7 实际运行，它们在分布式 world 形成前
被终止，完成 0 个 batch、0 个 domain，日志按失败尝试保留。后续优先等待
4090-1 恢复 4+2+2；若要试 6+2，先取得明确授权并做受监控的显存预检。

晚间 3070x2 的驱动恢复，两张卡均重新被 PyTorch 识别。为排除环境与数据问题，
已复制完全一致的兼容环境、IAPL 代码、权重和 91 GiB UFD Arrow 数据；206 个
文件的逐文件内容树哈希在源端与目标端同为
`66e2628c676f43b82d2d5b2f92989525463845cb7f28b47b7d52c7f59dba4132`。
但两张 8 GiB 卡分别运行单个官方 batch-size 32 rank 时，均在首 batch 完成前
OOM，采样峰值为 7857/7845 MiB；第二张卡已加入 expandable segments、CUDA
延迟加载和关闭 cuDNN 计划缓存，仍然失败。因此 4+1+1+2 布局被实验证伪，
不能通过偷偷降低 batch size 代替官方协议。

2026-07-25 06:03，4090-2 又出现独立故障：系统用户态 NVIDIA 库已自动升级到
580.173.02，但运行中的内核模块仍为 580.159.03，导致 NVML 版本不匹配，
两个 PyTorch 环境均看不到 CUDA。未重启、未重载模块、未降级系统包；从
Canonical Launchpad 下载匹配的 580.159.03 用户态包，仅解压到用户资产目录。
通过该隔离目录后，`nvidia-smi`、`cl` 和 `caid-gemini-compat` 均通过 CUDA
张量测试。两个评测启动器新增 `IAPL_NVIDIA_COMPAT_LIB_DIR`，并以一致哈希
同步至三台现有节点。4090-2 已无系统侵入地恢复，但第三个大显存节点仍缺失。

12:04 复查时 3090 恢复且 24 GiB GPU 空闲。其现有 UFD Arrow 的 205 个数据
文件与 4090-2 在文件名、大小和逐文件内容上完全一致，内容树 SHA256 为
`735262849f09c586f9f12beb778aec6a0e78f89b42b7961d02824c13f7deacc0`。
同步并核验兼容环境、代码和权重后，12:34 已按 A6000 `4 ranks` + 3090
`2 ranks` + 4090-2 `2 ranks` 启动 UFD seed100 官方 TTA。8 个 ranks
全部通过分布式 barrier 并进入 `crn`，三台主机初始显存为
36,279 / 18,208 / 18,710 MiB，尚无 traceback。当前只记录“运行中”，不把
首 batch 当成完成结果。

截至 14:16，前三个域已完整结束且预测文件已同步保存：`crn` Acc/AP 为
59.18%/56.08%，`cyclegan` 为 96.79%/94.02%，`dalle` 为
99.05%/98.50%。独立复算与运行日志一致，三域宏平均为 85.01%/82.87%。
`crn` 的明显低值不做筛选或重启，保留到 19 域全部结束后统一分析；当前已进入
`biggan`，P3 仍为运行中。

截至 15:16，`biggan` 和 `deepfake` 也已完成，Acc/AP 分别为
95.83%/92.76% 和 95.28%/96.15%。五域独立复算宏平均为
89.23%/87.50%，相同域上较论文低 7.68/11.99 个百分点；预测、日志和
DistributedSampler padding 审计已保存，当前运行 `gaugan`。

截至 17:46，已完成 10/19 域。`gaugan` 为 96.82%/94.23%，三个 GLIDE
域的 Acc 为 97.95%-98.30%、AP 为 97.89%-98.20%，`guided` 为
82.25%/92.88%。十域宏平均 91.95%/91.86%，相同域上较论文低
3.17/7.41 个百分点。十个预测文件和独立复算已保存，当前 `imle`
运行到 1150/1596。

截至 18:16，`imle` 又出现与 `crn` 相同的严重低值：Acc/AP
59.54%/56.17%，real/fake accuracy 为 19.10%/100.00%；不做筛选或
重启。`ldm_100` 为 98.80%/98.27%，十二域宏平均降至
89.82%/89.42%，较论文同域低 5.44/9.95 个百分点。A6000 的 Tailscale
监控连接连续两次超时，但局域网、GPU 和八 ranks 均正常，实验未受影响并已进入
`ldm_200`。

## P4: inference ablations

按推理代价从低到高运行 TTA steps、views、confidence selection 数量、entropy loss、
OIS 开关。每项报告 mAcc、mAP、real/fake accuracy、单图延迟、峰值显存和相对完整
IAPL 的变化。

## P5: controlled CTTA table

在相同 CNN checkpoint、样本、顺序和 Predict-Then-Adapt 协议下运行 Source、TENT、
EATA、CoTTA、RoTTA、LAME、T2A。包含 independent single-target 与 continual stream，
每项三个 seeds，报告 online、final、forgetting、延迟和显存。IAPL 继续作为不同
backbone/协议的端到端参考，不能混入控制变量排名。
