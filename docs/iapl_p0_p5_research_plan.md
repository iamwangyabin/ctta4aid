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

UFD 与 GenImage 的三 seed 从头训练链路、静态全测试集评测和后续官方
8-rank TTA 均已完成。执行阶段最终使用 A6000 `4 ranks` + 3090 `2 ranks` +
4090-2 `2 ranks`；此前 4090-1 离线、5+3 OOM、3070x2 单 rank OOM、A6000
Arrow runtime 漂移和 4090-2 驱动用户态不匹配均作为失败记录保留。P3 的完整
结论见本节末尾，而不是用更优 seed 覆盖这些失败与弱结果。

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

UFD seed100 官方 TTA 于 21:01 完成全部 19 域，耗时 8:26:49。官方日志
Acc/AP 为 91.69%/90.82%，19 个预测文件独立复算为
91.6895%/90.8209%，较论文低 3.92/8.50 个百分点。相对同一权重的静态
评测，TTA 令 Acc 提升 1.81 点但 AP 降低 7.33 点，主要由 `crn`、
`imle`、`seeingdark`、`stargan` 拉低。八 ranks 和三 launcher 均正常
退出，88,353 个唯一索引完整覆盖，23 个分布式 padding 样本保留在正式指标中。

4090-1 随后恢复，seed101 的 1,693,616,351 字节权重已复制到 A6000、
3090 和 4090-2，三机 SHA256 均为
`f81a0a9d69e57acea79ee8dbb3b00e39e4b5395a084884ec0f99999722f4bb14`。
当前 A6000 有无关 CAIDBench 进程占用 6,703 MiB，暂不冒险叠加 4 个
IAPL ranks；21:47 该任务仍持续写入训练日志并报告约 9:55 剩余时间。按顺序
等待 seed101，不能跳到 seed102。

等待期间只做后续去单点准备：seed102 权重已从 4090-2 复制并在 A6000、
3090、4090-2 三机核验同一 SHA256
`d324bac298ddd827ee17b688e932da03cf864d0e0933320c269a2d343811313d`。
这不改变执行顺序；seed102 必须等 seed101 完成并审计后才能启动。01:22
A6000 无关任务仍报告约 5:46 剩余时间。02:03 该 PID 仍持续写入样本，
占用 6,680 MiB，并报告 5:01:25 剩余时间；02:54 仍占用 6,680 MiB，
最新 ETA 为 4:13:52，因此 seed101 等待状态不是陈旧锁。

等待期间也完成了 GenImage 官方 TTA 启动器的 Arrow 修复。旧启动器仍强制检查
ImageFolder 的 `test/` 和 `extract_manifest.json`，现在改为优先读取
`IAPL_DATASET_PATH`，支持一个测试 Arrow 根或训练/测试两个 Arrow 根；Arrow
默认 `num_workers=0`，ImageFolder 回退仍保持 8。检查固定版 IAPL 后确认
`testtime_main` 只构造 `tta` split，不访问训练集，因此评测节点只需复制
25,467,557,463 字节、100,000 行的 `GenImage_test`，不需要重复复制 93 GB
SD1.4 训练 payload。最终启动器 SHA256
`894f0bfb21d77ffdab11208e64eb35300e186d2637917449fe64c797f10fa137`
已在 A6000、3090、4090-1、4090-2 四机核验一致；4090-2 上单测试根、
NCCL 23007、两 rank 预检和 14 项测试全部通过。A6000 与 3090 的测试 Arrow
复制随后完成；A6000、3090、4090-2 三机均为 31 个文件、
25,467,557,463 字节，有序逐文件 SHA256 树一致为
`904464e62f1525f1deecfe85a5c64064ff7b0914a557275af0d7226c3b799b9f`。
三个 GenImage 训练权重随后也全部复制并完成三机哈希核验；这只是后续准备，
不改变 UFD seed101、seed102 优先顺序。
首次本机 rsync 因 macOS 2.6.9 不支持 `--append-verify` 在传输前失败，第二次
脱离托管会话也未传输任何字节；两次失败均保留。3090 的第一次正式复制经
Tailscale 传到观测值 2,701,236,095 字节后，为切换更快的
`192.168.10.52` 局域网路由而以 rsync code 20 中断，partial 数据保留并成功
续传，不能把该中断从审计记录中删掉。

3090 的第一次节点启动预检还暴露了一个独立缺项：启动器默认的作者发布
GenImage 权重在该机不存在，因此在 0 batch 处失败。随后复制的是本阶段真正要评测
的 seed100 训练权重，而不是用别的权重掩盖错误。A6000、3090、4090-2 上该
权重均为 1,693,607,629 字节，SHA256 一致为
`aa0d8ab805f5c4fc846154e7da25ffae8cea32cbab9a8eb5ab3203ea27387096`。
三台节点分别按 ranks 0-3、4-5、6-7 使用测试 Arrow、NCCL 23007 完成启动
预检。GenImage seed101/102 权重 SHA256 分别为
`4cbaec6c15a9a0219d68cb5ef947585b5362ed4a5befe3e359b3152e84eaf2d9`、
`5d94e1159367ec8b4f6fc70cf6e5b8856430cc9f676c60117349b0b92bf3f18f`，
均在 A6000、3090、4090-2 一致。GenImage 三 seed 输入已全部就绪，但仍必须
等待 UFD seed101、seed102。

06:43 复查时 A6000 的无关 PID 16500 已退出，三台目标 GPU 均空闲，seed101
输出目录均不存在。随后在三机重新核对 checkpoint、Arrow 根、NCCL 23007、
seed、rank 分组和 4090-2 隔离驱动库，预检全部通过。06:45:05 按 A6000
`4 ranks` + 3090 `2 ranks` + 4090-2 `2 ranks` 正式启动 UFD seed101。
八 ranks 已全部通过分布式 barrier 并进入 `crn`，三机初始显存为
36,903 / 18,652 / 19,154 MiB，利用率均为 100%。rank0 到达 0/1596，
已观测峰值 8,437 MiB，未见 traceback、OOM、runtime 或 collective 错误。
当前只记为运行中，seed102 继续等待，不能提前启动。

seed101 首个完成域 `crn` 得到 99.79% Acc / 99.61% AP，而相同协议的
seed100 为 59.18% / 56.08%，显示出非常大的训练 seed 敏感性；不因高分或
低分筛选权重。`cyclegan` 随后得到 99.28% / 98.73%。两域独立复算宏平均
为 99.54% Acc / 99.17% AP，相对论文同域为 +4.00 / -0.81 个百分点，
相对 seed100 同域为 +21.55 / +24.12 个百分点。`crn`、`cyclegan` 的 4、6
个 sampler padding 重复样本均保留。期间 4090-2 的 Tailscale SSH 监控连续
两次超时，但 rank0 持续越过 `cyclegan` 边界进入 `dalle`，说明已有分布式
作业未中断；监控失败与成功续跑证据同时保存。

截至 09:43，seed101 已完成 5/19 域。`dalle` 为 97.20%/98.86%，
`biggan` 为 99.45%/99.59%；`deepfake` 则降至 66.12%/87.22%，其
real/fake accuracy 为 100.00%/32.10%。五域独立复算宏平均为
92.37% Acc / 96.80% AP，较论文同域低 4.53/2.69 个百分点，较 seed100
同域高 3.14/9.30 个百分点。该弱域不筛选、不重启，五个预测文件和 sampler
padding 审计均已保存。此前间歇超时的 4090-2 SSH 监控已恢复，八 ranks 均
在位，三台 GPU 利用率均为 100%，当前继续运行 `gaugan`。首次同步预测时远端
brace 路径被当作字面文件名而失败，随后改用远端通配符成功；失败与重试均入档。

截至 10:44，seed101 已完成 8/19 域。`gaugan` 达到 99.81%/99.66%，但
`glide_50_27` 与 `glide_100_10` 分别只有 86.15%/94.61% 和
85.75%/94.07% Acc/AP；两域 real accuracy 接近 100%，fake accuracy 仅
72.60%/71.70%，再次显示固定阈值的 fake 召回不稳定。八域宏平均为
91.69%/96.54%，较论文同域低 5.87/3.09 个百分点；相对 seed100 同域，
Acc 低 0.72 点而 AP 高 5.56 点。全部弱结果原样保留，当前进入
`glide_100_27`。4090-2 的 Tailscale 别名再次超时，但局域网连接立即确认
两个 ranks 均在、GPU 利用率 100%，因此记为监控链路故障而非实验失败。

截至 11:13，seed101 已完成 10/19 域。第三个 GLIDE 域
`glide_100_27` 为 83.50%/92.32%，`guided` 进一步降至
64.45%/88.38%，后者 real/fake accuracy 为 99.80%/29.10%。十域宏平均
为 88.15% Acc / 95.30% AP，较论文同域低 6.97/3.96 个百分点；相对
seed100 同域，Acc 低 3.80 点但 AP 高 3.44 点。该组合继续说明排序性能与固定
阈值分类性能明显脱钩。十个预测文件、sampler padding 和三机八 rank 状态均已
审计，未发现 traceback、OOM、NCCL 或 collective 错误，当前运行 `imle`。

截至 12:44，seed101 已完成 13/19 域。`imle` 达到 99.78%/99.58%，与
seed100 的 59.54%/56.17% 形成反向极端差异，进一步确认训练 seed 对域级
适应轨迹有决定性影响；不因该高分选择 seed。`ldm_100` 和 `ldm_200` 均为
97.70% Acc，AP 分别为 98.82%/98.77%。十三域宏平均为 90.51%/96.17%，
较论文同域低 5.07/3.24 个百分点；相对 seed100 同域，Acc 几乎相同
（-0.01 点），AP 高 6.06 点。十三个预测文件和完整复算已保存，当前进入
`ldm_200_cfg`，三机八 ranks 未见运行错误。

截至 13:46，seed101 已完成 17/19 域。`ldm_200_cfg` 为 89.90%/94.52%，
`progan` 为 99.86%/99.99%；`san` 和 `seeingdark` 则只有
70.45%/90.03% 与 75.83%/82.85% Acc/AP，fake accuracy 分别为
41.10%/52.22%。十七域宏平均为 88.98%/95.15%，较论文同域低
6.60/4.10 个百分点；相对 seed100 同域，Acc 低 2.34 点、AP 高 4.66 点。
SAN 的两个 sampler padding 重复样本和全部弱结果均原样保留。当前进入
`stargan`，之后只剩 `stylegan`，三机八 ranks 未见运行错误。

UFD seed101 官方 TTA 于 15:18 完成全部 19 域，耗时 8:32:55。官方日志
Acc/AP 为 89.54%/95.48%，19 个预测文件独立复算为 89.5358%/95.4825%，
较论文低 6.07/3.83 个百分点。相对同一权重的静态评测，TTA 令 Acc 降低
0.16 点、AP 降低 1.30 点；相对 seed100，Acc 低 2.15 点而 AP 高 4.66 点。
`deepfake`、`san`、`seeingdark`、`guided` 和三个 GLIDE 域构成主要低值，
而 `crn`、`imle` 又相对 seed100 出现反向高分，全部保留且不挑 seed。八 ranks
和三 launcher 均正常退出，88,353 个唯一索引完整覆盖，23 个 padding 样本
进入正式指标。首次最终汇总尝试因 A6000 runtime 未部署
`summarize_iapl_predictions.py` 在读取预测前失败，随后使用仓库内同一脚本在
本机成功复算；失败和重试均归档。seed102 现在按顺序成为下一项，启动前仍需
重新执行三机精确预检。

15:30 的 seed102 新鲜预检中，3090 与 4090-2 均通过 checkpoint、启动器、
Arrow、NCCL 23007、seed102 和空输出目录检查；但 seed101 释放 A6000 后，另一项
CoDA-Prompt CAIDBench 作业已启动，PID 32143 占用 5,036 MiB，且日志仍在完成
stage 1 协议评测。因此 A6000 的“GPU 必须空闲”检查明确失败，seed102 没有启动，
也没有只启动部分 ranks。该作业不被中断；失败预检已保存，待其退出后必须重新
执行三台节点的完整预检，而不是复用这次两台通过结果。

20:51，PID 32143 完成 stage 20、写出协议指标并退出；但在能够执行三机重检前，
A6000 又由同一服务器上的另一项 S-Prompt CAIDBench representative10 作业占用。
新 PID 50094 于 21:02:32 启动，21:13 观测占用 5,120 MiB 且仍在运行。因此
seed102 依旧没有启动，之前两台通过的预检仍不得复用。PID 切换、旧作业正常结束
以及新阻塞均单独归档；继续等待 A6000 真正空闲后重做三机完整预检。

22:31，PID 50094 完成 stage 10、保存协议指标并退出；22:43 已确认 A6000
真正空闲。22:44:59--22:45:18 重新执行而非复用三台精确预检，三机均通过
空闲 GPU、空输出目录、checkpoint/launcher SHA256、Arrow、NCCL 23007、
seed102、端口 29642 和 rank 分组检查。22:46:13 起按 A6000 `4 ranks` +
3090 `2 ranks` + 4090-2 `2 ranks` 启动 UFD seed102。八 ranks 均已跨过
distributed barrier 并进入 `crn`；三机显存为 37,167/18,656/19,154 MiB，
rank0 到达 0/1596、峰值 8,437 MiB，未见 traceback、OOM、runtime 或 collective
错误。当前只记为运行中，不提前启动 GenImage。

00:13，seed102 完成首域 `crn` 并进入 `cyclegan` 250/331。预测文件独立复算为
51.62% Acc / 59.27% AP，real/fake accuracy 为 3.27%/100%；4 个 sampler padding
重复样本保留。该结果相对 seed100 的同域 Acc 低 7.56 点、AP 高 3.19 点，
相对 seed101 则低 48.17/40.34 点，进一步确认训练 seed 会造成方向相反的巨大
域级波动。弱结果不剔除、不重跑挑 seed；八 ranks 与三 launcher 仍正常。

01:43，seed102 已完成 `crn`、`cyclegan`、`dalle`、`biggan`、`deepfake`
五域并进入 `gaugan` 500/1250。五个预测文件独立复算宏平均为 88.27% Acc /
90.52% AP，real/fake accuracy 为 78.92%/97.62%。除 `crn` 外，其余四域
Acc/AP 均超过 94%，说明当前宏平均的主要下拉仍是已保留的 `crn` 极端偏置，
而不是运行故障；八 ranks 未见异常。

03:14，seed102 完成十域并进入 `imle` 200/1596。十个预测文件独立复算宏平均
为 90.01% Acc / 93.24% AP，real/fake accuracy 为 88.10%/91.92%。新增五域中
`guided` 只有 83.35% Acc，但 AP 仍为 95.84%，其 98.80% real / 67.90% fake
accuracy 表明阈值偏向真实类；三个 GLIDE 域 Acc 为 91.05%--94.85%、AP 为
95.02%--97.45%。这些域与 `crn` 的相反类别偏置全部保留，运行仍无故障。

05:43，seed102 已完成十七域并进入 `stargan` 50/500。十七域宏平均降至
87.68% Acc / 90.30% AP，real/fake accuracy 为 82.47%/92.88%。新出现的
`imle` 只有 51.60%/59.58% Acc/AP，real accuracy 3.23%，与该 seed 的 `crn`
几乎同型；`seeingdark` 为 61.39%/57.69%，real/fake accuracy 22.78%/100%。
相反，`progan` 达 99.89%/100.00%。这些极端弱域与强域均原样保留，仅剩
`stargan`、`stylegan`，八 ranks 无故障。

07:12:52，seed102 完成全部 19 域，耗时 8:26:39。官方日志为 88.51% Acc /
91.16% AP，独立复算精确值为 88.5146%/91.1639%，较论文低 7.09/8.15 点。
相对静态评测，TTA 令 Acc 提高 2.07 点、AP 降低 5.97 点。19 个预测文件覆盖
88,353 个唯一索引，23 个 padding 样本保留；八 ranks、三 launcher 全部退出，
三机 GPU 均释放且无运行错误。本机首次跨 seed 比较因系统 Python 缺少 numpy
在导入阶段失败，随后在 A6000 `cl` 环境用同一跟踪脚本成功重试，失败与重试均
归档。UFD 三 seed 官方 TTA 至此完成，汇总为 89.91% +/- 1.62 Acc、
92.49% +/- 2.60 AP；`crn`、`imle` 的跨 seed Acc 范围均超过 48 点，不能视为
域级稳定。P3 尚未完成，下一项严格为 GenImage seed100 官方 TTA。

07:22:55 启动 GenImage seed100 后，第一次尝试在首 batch 前失败。A6000 的
`Dataset_Creator_GenImage` 缺少 Arrow 分支，ranks 0--3 将 `hf_arrow://` URI
误当成 ImageFolder 路径并在 `test/ADM` 抛出 `FileNotFoundError`；其余四个
ranks 随后因对端退出产生次生 NCCL 错误。此前预检只检查 Arrow `state.json`
和参数，没有真正实例化 GenImage creator，因此未发现 A6000 运行时漂移。零预测、
十一份日志和三机退出/GPU 释放状态均已归档，没有把失败隐藏成模型结果。

A6000 旧文件已备份，并替换为与 3090、4090-2 相同的
`25f0904428...` 实现。GenImage launcher 也增加了真实的首域 Arrow 构建 smoke，
该检查在 `IAPL_PREFLIGHT_ONLY=1` 退出前执行。P3 顺序不变：重新部署后，seed100
必须在三机通过增强预检，再从空输出目录完整重跑；seed101/102 继续等待。

07:36，三机增强预检全部通过：实际 pinned `Dataset_Creator_GenImage` 均成功将
`ADM` 构造成 12,000 行非空 Arrow dataset；同时复核了 launcher/checkpoint
哈希、NCCL 23007、GPU 空闲与正式输出目录不存在。A6000 的 `cl` 环境还通过
12 项聚焦协议测试。seed100 已具备从 rank 0 干净重启条件，仍不提前启动后续 seed。

07:39:27，seed100 第二次尝试按 A6000 `4 ranks` + 3090 `2 ranks` +
4090-2 `2 ranks` 在端口 29643 干净启动。三个 launcher 均再次通过 12,000 行
`ADM` runtime smoke，八 ranks 全部越过分布式 barrier；rank0 已进入 `ADM`
0/1500，峰值 8,437 MiB，三机 GPU 均在计算，未见 traceback、runtime、OOM 或
collective 错误。当前只记录为运行中，seed101 继续等待本次完成并审计。

08:50，seed100 用 1:09:32 完成 `ADM` 并进入 `BigGAN`，09:15 已到
500/1500。对 12,000 个唯一索引独立复算得到 62.0583% Acc / 92.2576% AP，
real/fake accuracy 为 99.9833%/24.1333%，无 sampler padding，与官方四舍五入
日志 62.06%/92.26% 一致。相对该权重静态 ADM，TTA 的 Acc 提高 0.71 点但 AP
下降 2.79 点；相对 P2 官方权重 ADM 又低 23.48/6.04 点。该强烈偏真实类的弱结果
原样保留，不换 seed；八 ranks 仍健康。

10:00，`BigGAN` 用 1:09:39 完成；12,000 个唯一索引独立复算为 83.9083%
Acc / 99.2389% AP，real/fake accuracy 为 99.9833%/67.8333%，无 padding。
与 ADM 相反，TTA 相对该权重静态 BigGAN 提高 23.64 Acc 点和 1.45 AP 点，
但 Acc 仍比 P2 官方权重低 14.78 点。前两域宏平均为 72.98% Acc / 95.75%
AP。10:15 已进入 `glide` 300/1500，八 ranks 健康；两种相反响应都保留。

11:09，`glide` 用 1:09:18 完成；12,000 个唯一预测独立复算为 83.0917%
Acc / 98.9172% AP，real/fake accuracy 为 99.9833%/66.20%，无 padding。
相对静态评测，TTA 令 Acc 提高 2.06 点、AP 降低 0.25 点；Acc 仍比 P2 官方
权重低 12.86 点。前三域宏平均为 76.35% Acc / 96.80% AP，平均 real/fake
accuracy 为 99.98%/52.72%，阈值偏置仍明显。11:15 `Midjourney` 已到
100/1500，八 ranks 健康。

12:18，`Midjourney` 用 1:08:51 完成；12,000 个唯一索引独立复算为
75.5333% Acc / 92.2107% AP，real/fake accuracy 为 99.9667%/51.10%。
TTA 相对该权重静态结果仅提高 0.73 Acc 点，却降低 6.78 AP 点；相对 P2
Acc/AP 仍低 20.21/6.91 点。前四域宏平均为 76.15% Acc / 95.66% AP，
平均 real/fake accuracy 为 99.98%/52.32%。该 AP 退化原样保留；12:46
`stable_diffusion_v_1_4` 已到 600/1500，运行无故障。

13:27，`stable_diffusion_v_1_4` 用 1:08:40 完成，独立复算为 99.9917%
Acc / 99.9942% AP，real/fake accuracy 为 100%/99.9833%，覆盖 12,000 个
唯一样本。相对静态 100% 的变化不足 0.01 点，且两指标略高于 P2。前五域宏平均
升至 80.92% Acc / 96.52% AP，但此前跨生成器域令平均 fake accuracy 仍只有
61.85%。13:48 SD1.5 已到 450/2000，八 ranks 健康。

14:58，`stable_diffusion_v_1_5` 用 1:31:16 完成。16,000 个唯一预测独立
复算为 99.9125% Acc / 99.9128% AP，real/fake accuracy 为
99.95%/99.875%，无 padding。相对该权重静态结果的 Acc/AP 变化均不足
0.03 点，并比 P2 官方权重高 0.18/0.04 点。前六域宏平均升至 84.08% Acc /
97.09% AP，但前面弱域令平均 fake accuracy 仍只有 68.19%。15:20 `VQDM`
已到 450/1500，八 ranks 健康，失败和弱结果继续原样保留。

16:06，`VQDM` 用 1:08:28 完成。12,000 个唯一预测独立复算为 95.6667%
Acc / 99.6831% AP，real/fake accuracy 为 99.9667%/91.3667%，无
padding。TTA 相对该权重静态结果提高 0.57 Acc 点、降低 0.24 AP 点；相对
P2 官方权重低 3.13 Acc 点，AP 差不足 0.01 点。前七域宏平均为 85.74% Acc /
97.46% AP。16:19 `wukong` 已到 200/1500，三 launcher 与三 GPU 均健康。

17:15:11，GenImage seed100 完成全部八域，总耗时 9:35:44；`wukong` 为
99.8583% Acc / 99.9775% AP。独立复算最终宏平均为 87.5026% Acc /
97.7740% AP，较论文低 9.20/1.73 点，较 P2 官方权重低 9.27/1.72 点。
相对该训练权重静态评测，TTA 提高 3.47 Acc 点、降低 1.09 AP 点。平均
real/fake accuracy 仍分裂为 99.9792%/75.0260%，主要差距来自 ADM、
Midjourney、BigGAN 和 glide。100,000 个唯一索引全部覆盖且无 padding，
八 ranks、三 launcher 全部退出，三 GPU 释放，无执行错误。弱结果完整保留；
P3 下一项严格为 GenImage seed101，启动前必须重新通过三机预检。

17:22--17:23，seed101 在三机完成全新、未复用的预检：GPU 空闲、端口 29644
未占用、正式输出目录不存在，checkpoint `4cbaec6...`、launcher 与 runtime
哈希一致，并且每台机器都用实际代码成功构造 12,000 行 ADM Arrow dataset。
17:24:02 按 A6000 4 + 3090 2 + 4090-2 2 ranks 启动；八 ranks 全部通过
barrier，rank0 进入 ADM 0/1500，峰值 8,437 MiB，无启动错误。seed102 继续
按顺序等待。

18:33，seed101 用 1:08:23 完成 `ADM`。12,000 个唯一预测独立复算为
84.3917% Acc / 94.1741% AP，real/fake accuracy 为 99.4167%/69.3667%，
无 padding。TTA 相对该权重静态结果提高 2.12 Acc 点、降低 3.75 AP 点；相对
P2 官方权重低 1.14/4.12 点，但相对相同协议的训练 seed100 高 22.33/1.92 点，
再次显示明显 seed 敏感性。两者均保留。18:49 BigGAN 到 300/1500，八 ranks
健康。

19:41，seed101 `BigGAN` 用 1:08:19 完成。12,000 个唯一预测独立复算为
99.60% Acc / 99.2773% AP，real/fake accuracy 为 99.25%/99.95%，无
padding。TTA 相对该权重静态评测提高 11.83 Acc 点、降低 0.35 AP 点；Acc
比 P2 高 0.91 点、比训练 seed100 高 15.69 点，AP 比 P2 低 0.38 点但与
seed100 基本持平。前两域宏平均为 92.00% Acc / 96.73% AP。19:49 `glide`
到 100/1500，八 ranks 健康。

20:50，seed101 `glide` 用 1:08:14 完成。12,000 个唯一预测独立复算为
97.4833% Acc / 98.6287% AP，real/fake accuracy 为 99.4333%/95.5333%，
无 padding。TTA 相对静态 Acc 仅提高 0.16 点，却降低 1.09 AP 点；Acc 比
P2 高 1.53 点、比训练 seed100 高 14.39 点，AP 则低 0.83/0.29 点。前三域
宏平均为 93.83% Acc / 97.36% AP。21:18 `Midjourney` 到 600/1500，八
ranks 健康，AP 退化原样保留。

21:58，seed101 `Midjourney` 用 1:08:17 完成。12,000 个唯一预测独立复算
为 97.8750% Acc / 98.5034% AP，real/fake accuracy 为
99.4333%/96.3167%，无 padding。TTA 相对静态提高 0.73 Acc 点、降低
1.29 AP 点；Acc 比 P2 高 2.13 点、比训练 seed100 高 22.34 点，AP 比 P2
低 0.61 点但比 seed100 高 6.29 点。前四域宏平均为 94.84% Acc / 97.65%
AP。22:19 SD1.4 到 400/1500，八 ranks 健康。

23:06，seed101 `stable_diffusion_v_1_4` 用 1:08:12 完成。12,000 个唯一
预测独立复算为 99.6750% Acc / 99.6338% AP，real/fake accuracy 为
99.35%/100%，无 padding。绝对结果仍强，但 TTA 相对该权重静态 Acc/AP 低
0.26/0.37 点，也略低于 P2 和训练 seed100。前五域宏平均为 95.81% Acc /
98.04% AP。23:19 SD1.5 到 250/2000，八 ranks 健康，小幅退化原样保留。

00:37，seed101 `stable_diffusion_v_1_5` 用 1:31:06 完成。16,000 个唯一
预测独立复算为 99.5125% Acc / 99.3998% AP，real/fake accuracy 为
99.1375%/99.8875%，无 padding。TTA 相对静态低 0.32/0.54 Acc/AP 点，
相对 P2 低 0.23/0.47 点，相对训练 seed100 低 0.40/0.51 点。前六域宏平均
为 96.42% Acc / 98.27% AP。00:49 `VQDM` 到 200/1500，八 ranks 健康；
两个 diffusion 域连续出现的小幅退化均保留。

01:46，seed101 `VQDM` 用 1:08:16 完成。12,000 个唯一预测独立复算为
99.3583% Acc / 99.3414% AP，real/fake accuracy 为 99.2167%/99.50%，
无 padding。TTA 相对静态 Acc 几乎不变但 AP 低 0.65 点；Acc 比 P2 高
0.57 点、比训练 seed100 高 3.69 点，AP 则低 0.35/0.34 点。前七域宏平均
为 96.84% Acc / 98.42% AP。`wukong` 随后进入 0/1500，八 ranks 健康，
排序退化继续原样保留。

02:54:32，GenImage seed101 完成全部八域，总耗时 9:30:30；`wukong` 为
99.6000% Acc / 99.6095% AP。独立复算最终宏平均为 97.1870% Acc /
98.5710% AP，real/fake accuracy 为 99.3047%/95.0693%。相对论文，Acc
高 0.49 点、AP 低 0.93 点；相对 P2 为 +0.42/-0.92 点。相对该权重静态
评测，TTA 提高 1.75 Acc 点、降低 1.05 AP 点；相对训练 seed100 则提高
9.68/0.80 Acc/AP 点，证明明显 seed 敏感性而非执行故障。100,000 个唯一
索引全部覆盖且无 padding，八 ranks、三 launcher 全部退出，三 GPU 释放，
未发现 traceback、OOM、runtime 或 collective 错误。P3 下一项严格为
GenImage seed102，必须先重新通过三机预检。

03:26，seed102 全新三机预检通过。A6000/3090 空闲显存占用为 17/106 MiB，
端口 29645 空闲、输出目录不存在、checkpoint/launcher/runtime 哈希一致，实际
ADM Arrow creator 均返回 12,000 行。4090-2 首次裸 NVML 探测因系统 580.173
用户态库与已加载驱动不匹配而失败；该失败保留，随后仅使用既定隔离 580.159.03
兼容库重新执行整套节点检查并通过。03:27:12 按 4+2+2 ranks 启动 seed102；
八 ranks 均通过 barrier，rank0 进入 ADM 0/1500，峰值 8,437 MiB，三机显存
占用 37,179/18,656/19,162 MiB，未发现启动执行错误。这是 P3 最后一个 TTA
run；完成及跨 seed 审计前不得进入 P4。

04:37，seed102 完成 `ADM`，耗时 1:08:51。12,000 个唯一预测独立复算为
67.1583% Acc / 95.6877% AP，real/fake accuracy 为 99.9667%/34.3500%，
无 padding。TTA 相对静态仅提高 0.69 Acc 点，却降低 2.16 AP 点；相对 P2
低 18.38/2.61 Acc/AP 点，相对训练 seed101 Acc 低 17.23 点而 AP 高 1.51
点，相对 seed100 则高 5.10/3.43 点。这个明显受阈值和 seed 影响的弱结果
完整保留。04:48，BigGAN 到 200/1500，八 ranks 健康且无执行错误。

05:46，seed102 `BigGAN` 用 1:09:17 完成。12,000 个唯一预测独立复算为
79.2250% Acc / 99.9018% AP，real/fake accuracy 为 99.9833%/58.4667%，
无 padding。TTA 相对该权重静态 Acc/AP 提高 27.42/3.21 点，但 Acc 仍比
P2、训练 seed100、训练 seed101 分别低 19.47、4.68、20.38 点；AP 则分别
高 0.25、0.66、0.62 点。前两域宏平均为 73.19% Acc / 97.79% AP，再次
表明排序很强而固定阈值不稳定。`glide` 随后进入 0/1500，八 ranks 健康。

06:56，seed102 `glide` 用 1:09:35 完成。12,000 个唯一预测独立复算为
69.5417% Acc / 99.5674% AP，real/fake accuracy 为 100%/39.0833%，无
padding。TTA 相对静态 Acc/AP 为 +2.00/-0.00 点；AP 比 P2、训练 seed100、
训练 seed101 分别高 0.11、0.65、0.94 点，Acc 却分别低 26.41、13.55、
27.94 点。前三域宏平均为 71.98% Acc / 98.39% AP，进一步支持固定阈值
诊断。07:18，`Midjourney` 到 450/1500，八 ranks 健康且无执行错误。

08:05，seed102 `Midjourney` 用 1:09:38 完成。12,000 个唯一预测独立复算
为 85.7000% Acc / 94.2166% AP，real/fake accuracy 为
99.9667%/71.4333%，无 padding。TTA 相对静态提高 6.03 Acc 点、降低
4.85 AP 点；相对 P2 低 10.04/4.90 点，相对训练 seed101 低 12.18/4.29
点，但相对训练 seed100 高 10.17/2.01 点。前四域宏平均为 75.41% Acc /
97.34% AP，阈值和排序退化都完整保留。08:18，SD1.4 到 250/1500，八
ranks 健康。

09:15，seed102 `stable_diffusion_v_1_4` 用 1:09:23 完成，在 12,000 个
唯一样本上 Acc/AP/real/fake 全部为 100%，无 padding。该结果与静态评测、
P2、训练 seed100、训练 seed101 持平或略高。前五域宏平均升至 80.33% Acc /
97.87% AP，但此前未见生成器的弱结果不被掩盖。SD1.5 随后进入 50/2000，
八 ranks 健康且无执行错误。

10:47，seed102 `stable_diffusion_v_1_5` 用 1:31:46 完成。16,000 个唯一
预测独立复算为 99.9125% Acc / 99.9571% AP，real/fake accuracy 为
99.9375%/99.8875%，无 padding。该结果与静态 Acc 和训练 seed100 持平，
比 P2 高 0.18/0.09 Acc/AP 点，比训练 seed101 高 0.40/0.56 点。前六域
宏平均为 83.59% Acc / 98.22% AP；两个 diffusion 域稳定且强，此前未见
生成器仍是宏平均瓶颈。`VQDM` 随后进入 0/1500，八 ranks 健康。

11:55，seed102 `VQDM` 用 1:08:35 完成。12,000 个唯一样本独立复算为
97.5833% Acc / 99.8646% AP，real/fake accuracy 为 99.9667%/95.20%，
无 padding。TTA 相对静态 Acc/AP 为 +0.66/-0.11 点；Acc 比 P2、训练
seed101 分别低 1.21、1.78 点，但比训练 seed100 高 1.92 点，AP 则高于
三个发布/训练参考。前七域宏平均为 85.59% Acc / 98.46% AP。12:18，最后
一个 `wukong` 到 500/1500，八 ranks 健康且无执行错误。

13:04:11，GenImage seed102 完成全部八域，总耗时 9:36:59；`wukong` 为
99.9667% Acc / 99.9911% AP。独立复算最终宏平均为 87.3859% Acc /
98.6483% AP，real/fake Accuracy 为 99.9755%/74.7964%，相对论文低
9.31/0.85 点，相对 P2 低 9.39/0.84 点。相对该权重静态评测，TTA 提高
4.61 Acc 点、降低 0.49 AP 点。100,000 个唯一索引全部覆盖且无 padding，
八 ranks、三 launcher 全部退出，三 GPU 释放，未发现执行错误。

跨 seed 样本审计第一次在本机因系统 Python 缺少 NumPy 而在读取预测前失败，
第一次远端重试又因猜测的 `cl` 路径不存在而失败；随后使用 A6000 上实际的
`/home/home/yabin/miniconda3/envs/cl/bin/python` 成功。三 seed 的索引和标签
序列完全相同。ADM、BigGAN、glide、Midjourney 的阈值分歧最大，两个 diffusion
域和 wukong 稳定。

至此 P3 六个官方 TTA run 全部完成并审计。GenImage 三 seed 官方 TTA 为
90.69 +/- 5.63% Acc、98.33 +/- 0.48% AP，real/fake Accuracy 为
99.75 +/- 0.39%/81.63 +/- 11.64%；Accuracy 跨 seed 极差 9.80 点，fake
Accuracy 极差 20.27 点。因此 P3 的训练和评测链路完成，但预设稳定性标准未
达到，负结果不做事后 seed 筛选。严格顺序的下一阶段为 P4 推理消融。

## P4: inference ablations

按推理代价从低到高运行 TTA steps、views、confidence selection 数量、entropy loss、
OIS 开关。每项报告 mAcc、mAP、real/fake accuracy、单图延迟、峰值显存和相对完整
IAPL 的变化。

P4 固定使用发布的 ProGAN checkpoint、公开 UFD 19 域 88,353 张图、seed100 和
官方 8-rank 的 A6000 `4` + 3090 `2` + 4090-2 `2` 布局，避免把 P3 的训练
seed 不稳定性混入推理模块判断。基线为 32 views、2 steps、精确选择 6 views、
averaged entropy 和 OIS。11 个 run 每次只改变一个因素：views `8/16/32`、
steps `1/2/3`、选择数 `2/4/6/8/12`、pointwise entropy、关闭 OIS；完整基线
会重跑以补齐计时和显存。论文 Table 8 的 averaged/pointwise 结果和 Table 9 的
`T=1/2/3` 结果作为参考，但公开数据差异仍按 P1 口径说明。

新增 runtime patch 只开放精确选择数和 averaged/pointwise loss，并记录各域八
ranks 最大 wall time、rank 内单图均值延迟、PyTorch allocated/reserved 峰值；
launcher 另以 1 秒间隔采样每台物理 GPU 的总显存与利用率。运行按预计 view-forward
代价排序，所有失败和弱结果保留。三机真实 Arrow creator、loss 和静态参数预检已经
通过；第一项 `views8` 已于 2026-07-28 15:09 CST 按 8-rank 布局启动。八个 rank
均越过 distributed barrier，rank0 已进入首域 `crn`，三台 GPU 均达到 100% 利用率，
启动审计未发现 traceback、NCCL/CUDA 或 OOM 错误。`views8` 已于 17:55 CST
完成 19 域：95.6007% mAcc、98.1885% mAP、8.9692 images/s，最终产物和三机资源
曲线均已同步。第二项 `views16` 通过三机 Arrow 预检后于 18:00:55 CST 启动，八个
rank 均已越过 barrier，并于 22:42:40 CST 完成 19 域：95.5999% mAcc、97.6885%
mAP、5.2513 images/s。其 Acc 与 views8 基本相同，但 AP 低 0.4999 个百分点，吞吐
仅为 58.5%，显存更高。第三项 `steps1` 通过三机 32-view Arrow 预检后于
22:50:27 CST 启动，八个 rank 已越过 barrier。

该次 `steps1` attempt1 在完成前十域后遭遇硬件故障。3090 的 `/data` 所在
NVMe 于 01:24 开始 READ timeout，控制器 reset 失败后在 01:26:52 被内核禁用，
EXT4 journal 随即中止；从该盘 mmap Arrow 的 ranks 4-5 同时 SIGBUS，rank7
随后因远端 peer 消失报告 `ncclRemoteError`。当时主机仍有 14 GiB 可用内存、
7.1 GiB swap 和 117 GiB 文件系统余量，且无 CUDA OOM，因此根因不是资源参数。
十域预测仍有效；`imle` 虽在 rank0 完成本地循环，却未完成八 rank gather，不能
计入结果。完整日志、三机显存曲线和内核证据均保留。

为避免在域中途重启导致 augmentation RNG 流变化，attempt1 不拼接续跑。
02:29:04 CST 从 seed100 重新运行全部 19 域：逻辑 8-rank 协议不变，4090-1
替代故障 3090 承载 ranks 4-5，A6000 继续 ranks 0-3，4090-2 继续 ranks 6-7。
4090-1 已用隔离的 580.159.03 用户态库修复驱动版本不匹配，三机真实 Arrow
preflight 全部通过，八个 ranks 已越过 barrier 并进入 `crn`。不得提前启动
后续变体或 P5。

03:46 attempt2 已完成 `crn`、`cyclegan`、`dalle`、`biggan` 四域并进入
`deepfake`，宏平均为 97.2057% Acc / 97.9478% AP。与故障 attempt1 的相同
四域相比仅差 -0.0141/-0.0080 个百分点，说明替换物理节点只产生很小数值漂移，
不做事后选择。三机吞吐保持 5.50-5.51 images/s，八 rank 日志无执行错误；
4090-1 的 23,731 MiB 物理读数含启动前已有的 5,277 MiB 驱动残留记账，跨 run
显存比较以 rank-local PyTorch 峰值为准。

04:46 attempt2 已完成前十域并进入 `imle`，宏平均为 95.1660% Acc /
98.3126% AP。与故障 attempt1 同十域相比为 +0.0141/-0.0007 个百分点，恢复
运行与原运行实质一致；`guided` 的 72.90% Acc / 95.93% AP 及 46.90% fake
Accuracy 继续作为弱结果保留。吞吐、显存和八 rank 错误审计均稳定。

05:48 attempt2 已完成十四域并进入 `progan`，宏平均为 95.6894% Acc /
98.3107% AP。相对 P1 相同十四域为 +0.0141/+0.5465 个百分点；新完成的
`imle` 为 91.9408% Acc / 94.9137% AP，其中 real/fake Accuracy 为
83.8866%/100%。预测覆盖、吞吐、显存和八 rank 日志审计继续通过，后续变体
仍未提前启动。

随后 `progan` 以 100% Acc/AP 完成，attempt2 的有效域达到十五个，部分宏平均
为 95.9767% Acc / 98.4234% AP。但在 `san` 阶段 4090-1 同时从 LAN 和
Tailscale 失联；A6000 与 4090-2 均观测到 ARP 邻居不完整及 no route to host。
rank0 虽完成本地循环，八 rank gather 未完成且没有 `san` 预测文件，因此该域
不计入结果。当前保留进程等待 7,200 秒 collective timeout 或链路恢复；若最终
失败，将完整归档 attempt2 并从头启动 attempt3，不拼接新的 RNG 流。

已准备无破坏性的 attempt3 备用路径：3090 GPU 与系统盘仍正常，但故障 `/data`
仍是 0-byte NVMe 和 ext4 `shutdown`，不会再读取。3090 通过用户态 SSHFS 从
3070x2 只读挂载完整 UFD Arrow；真实 loader 已通过 19 域、88,353 样本、标签、
row mapping 与首图解码预检。若 attempt2 超时，可恢复原 A6000 `4`、3090 `2`、
4090-2 `2` GPU 布局从头运行；远程存储差异将单独记录，并审计是否改变 A6000
四 rank 的关键路径瓶颈。

08:08-08:09 attempt2 最终触发预期的 NCCL watchdog：A6000 与 4090-2 的
存活 ranks 均记录 `ALLREDUCE` sequence 193 超过 7,200 秒，进程全部退出并释放
GPU。最终日志和资源曲线已归档；`san` 仍不计入，前十五域只作为失败前有效部分。

08:23 attempt3 从 seed100 和全 19 域重新启动，恢复 A6000 `4`、3090 `2`、
4090-2 `2` 的原物理/逻辑布局。3090 只读使用 3070x2 的 SSHFS Arrow 副本，
不访问故障 `/data`。三机真实 32-view creator 与代码哈希预检通过，八 ranks 已
越过 barrier，rank0 进入 `crn`；初始显存为 37,179/18,652/19,162 MiB，三机
均 100% 利用率且无启动错误。后续 P4 变体仍未提前启动。

09:19 attempt3 已完成 `crn`、`cyclegan`、`dalle` 三域并进入 `biggan`，宏平均
为 96.6717% Acc / 97.7579% AP。相对 attempts 1/2 相同三域，Acc 最大只差
0.0131 点、AP 最大只差 0.0100 点，SSHFS 恢复未引入实质预测漂移。吞吐为
5.40-5.43 images/s，三个域的关键路径 rank 全在 A6000 而非 3090 SSHFS；挂载、
显存和八 rank 错误审计均正常。

09:48 attempt3 又完成 `biggan`、`deepfake` 并进入 `gaugan`；前五域宏平均为
96.9494% Acc / 97.8619% AP。相对 attempts 1/2 相同五域，Acc 最大差 0.0166
点、AP 最大差 0.0232 点，仍属于运行微小波动而非远程存储回归。五域关键路径
全部位于 A6000，吞吐为 5.40-5.46 images/s；SSHFS 挂载、三机满载和八 rank
错误审计继续正常。

10:30 attempt3 已继续完成 `gaugan`、`glide_50_27`、`glide_100_10` 并进入
`glide_100_27`。前八域 one-step TTA 为 97.5734% Acc / 98.4958% AP；相对 P1
同域 two-step，Acc 仅低 0.0264 点而 AP 高 0.5547 点。相对 attempts 1/2 的
最大复现偏差也仅 0.0279 Acc 点、0.0114 AP 点。八域关键路径继续全部位于
A6000，SSHFS、吞吐、三机 GPU 和八 rank 错误审计均正常。

11:16 远程监控恢复，期间实验未中断。11:21 attempt3 已完成
`glide_100_27`、`guided`、`imle` 并进入 `ldm_100`，越过 attempt1 的十域
故障边界。前十一域为 94.8494% Acc / 98.0033% AP；相对 P1 同域 two-step，
Acc 仅低 0.0009 点，AP 高 0.6836 点。相对 attempt1 共享前十域最大偏差为
0.0133/0.0082 Acc/AP 点，相对 attempt2 十一域为 0.0235/0.0004 点。三机、
SSHFS、A6000 关键路径和八 rank 错误审计继续正常。

12:03 attempt3 已完成并成功 gather `san`，因此不是只重复 attempt2 的 rank0
本地循环，而是正式越过其 4090-1 掉线边界。12:05 已累计完成十七域，新增
`ldm_100`、`ldm_200`、`ldm_200_cfg`、`progan`、`san`、`seeingdark`，只剩
`stargan`、`stylegan`。当前为 95.4605% Acc / 97.7813% AP；相对 P1 同域
two-step 高 0.0113/0.6171 Acc/AP 点，相对 attempt2 共享前十五域仅差
0.0205/0.0008 点。三机、SSHFS、样本数和八 rank 错误审计全部正常。

12:17 `stargan` 完成，attempt3 已进入最后一个 `stylegan` 域。前十八域为
95.5460% Acc / 97.7375% AP，相对 P1 同域 two-step 高 0.0135/0.6550
Acc/AP 点。最终域启动时三机 GPU 均正常工作，3090 SSHFS 与八 rank 日志
继续正常；`ois_off` 尚未提前启动。

12:53:06 `stylegan` 完成，`steps1` attempt3 以 4:29:48 完成全部 19 域。
最终为 95.5064% Acc / 97.8391% AP，real/fake Accuracy 为
96.2234%/94.7879%。相对论文 Table 9 的 T=1 高 1.0364 Acc 点、低 0.6909
AP 点；相对本地 P1 two-step 几乎相同 Acc（+0.0141 点），AP 高 0.6217 点。
与 P1 的 88,353 个相同索引和标签逐样本比较有 431 个阈值分歧（0.4878%），
平均绝对概率差 0.00558。八 rank 无错误，全部进程退出，三机 GPU 已空闲；
attempt1/2 失败记录完整保留。P4 下一项严格为 `ois_off`。

`steps1` 最终审计并推送后，`ois_off` 在三机重新通过空输出、脚本哈希、NCCL、
真实 `crn` Arrow、端口、GPU 与 SSHFS 预检，并于 12:59:58 使用端口 29662
启动。该实验只把 OIS 从 true 改为 false，其余 views、TTA steps、选择数、熵、
seed、rank 布局、域顺序、checkpoint 和数据完全固定。八 ranks 已越过 barrier，
rank0 进入 `crn`，A6000/3090/4090-2 初始显存为 37,175/18,652/19,154 MiB，
均为 100% 利用率且无错误。后续变体未提前启动。

21:14:52 `ois_off` 完成全部 19 域，最终为 94.6641% Acc / 98.2870% AP，
real/fake Accuracy 为 96.8499%/92.4680%。相对本地 P1 的 OIS-on 对照，关闭
OIS 使 Acc 和 fake Accuracy 分别下降 0.8282/2.2402 点，但 AP 和 real
Accuracy 分别上升 1.0696/0.5749 点；`san` 单域下降 18.4091 Acc 点。由此保留
“OIS 主要改善阈值校准和异常域稳定性，而非普遍提升排序 AP”的结论。

结果审计和提交 `4e5d40c` 推送后，下一项 `select2` 已在三机通过脚本、NCCL、
checkpoint、空输出、端口和真实 12,764 行 `crn` Arrow 预检；3090 只读 SSHFS
与 4090-2 兼容驱动均正常。但 A6000 上另一用户的训练进程正使用 2,986 MiB，
连续 SM 样本为 46-49%，此时启动会破坏 P4 耗时和物理显存曲线的公平性。未修改
该进程，`select2` 保持未启动，后续心跳先复核干净 GPU 门槛再按顺序启动。

03:17:47 共享训练自行退出，未作任何干预；A6000 恢复到 17 MiB、0% 利用率。
三机重新通过空输出、端口、GPU 与 3090 只读 SSHFS 检查后，`select2` 于
03:18:49 先启动 worker，并在 03:19:19 完成八 rank 启动，使用端口 29663。
所有 ranks 已越过 barrier，rank0 进入 `crn`；A6000/3090/4090-2 初始显存为
37,167/18,652/19,158 MiB，均为 100% 利用率，八份日志无错误。后续变体未提前
启动。

04:47 `select2` 已完成 `crn`、`cyclegan` 并进入 `dalle`，部分宏平均为
94.1064% Acc / 94.7412% AP，相对 P1 相同两域低 1.5573/0.6449 点。差距主要
来自 `crn`：89.9123% Acc / 92.0658% AP，Acc 低 2.6237 点，其中 fake
Accuracy 仍为 100%，real Accuracy 低 5.2459 点。15,406 个唯一索引和标签与
P1 完全一致，共 360 个阈值分歧。三机、3090 只读 SSHFS、吞吐、显存和八 rank
日志均正常，弱结果原样保留，后续变体未提前启动。

06:19 `select2` 已扩展到五域并进入 `gaugan`，部分宏平均回升到 96.3388% Acc /
96.8035% AP，但相对 P1 相同五域仍低 0.6768/0.2990 点；real Accuracy 低
1.4599 点，fake Accuracy 高 0.1071 点。差距仍主要由 `crn` 贡献，`dalle` 的
Acc 与 P1 持平且 AP 高 0.0999 点，`deepfake` Acc 高 0.0555 点。26,811 个
唯一样本索引和标签完全一致，吞吐稳定在 2.9520-2.9632 images/s，三机、SSHFS、
显存和八 rank 日志继续正常。

07:18 `select2` 已完成八域并进入 `glide_100_27`，部分宏平均为 97.2005% Acc /
97.7374% AP，相对 P1 相同八域低 0.3993/0.2037 点。real Accuracy 仍低
1.0149 点，fake Accuracy 高 0.2169 点；新增域中 `glide_100_10` 的 Acc/AP
高 0.3500/0.1251 点，部分抵消了 `crn` 的持续损失。40,811 个索引和标签完全
一致，吞吐、三机 GPU、SSHFS 和八 rank 日志均稳定。

07:48 `select2` 已完成十域并进入 `imle`，部分宏平均为 95.0004% Acc /
97.6275% AP，相对 P1 相同十域仅低 0.0994/0.1602 点。Acc 差距收窄主要来自
`guided`：其绝对结果仍弱（74.40% Acc / 95.40% AP），但比 P1 高 2.00 Acc
点、4.30 fake-accuracy 点。real Accuracy 仍低 0.8519 点，fake Accuracy 高
0.6535 点。44,811 个索引和标签一致，吞吐、显存、SSHFS 与八 rank 日志正常。

09:48 `select2` 已完成十四域并运行 `progan`，部分宏平均为 95.4638% Acc /
97.5420% AP，相对 P1 相同十四域低 0.2115/0.2223 点。新增差距主要来自
`imle`：89.8888% Acc / 91.3541% AP，低 2.4671/1.2863 点，real Accuracy
低 4.9327 点；三个 LDM 域与 P1 接近且 Acc 略高。63,575 个索引和标签一致，
三机、SSHFS、吞吐、显存与八 rank 日志继续正常。

10:18 `select2` 已完成十七域并运行 `stargan`，部分宏平均为 95.0484% Acc /
96.7281% AP，相对 P1 相同十七域低 0.4007/0.4361 点。`progan` 基本不变，
`san` Acc 仅低 0.2273 点；但 `seeingdark` 降至 86.3889% Acc / 84.0614% AP，
低 3.6111/3.3731 点，real Accuracy 低 7.2222 点。该弱结果原样保留。72,373
个索引和标签一致，三机、SSHFS、显存与八 rank 日志正常。

11:39:06 `select2` 完成全部 19 域，耗时 8:19:47。最终为 95.1162% Acc /
96.8206% AP，real/fake Accuracy 为 95.0867%/95.1451%。相对完全匹配的 P1
六视图对照，选择两视图使 Acc/AP/real Accuracy 分别下降
0.3761/0.3968/1.1884 点，fake Accuracy 上升 0.4369 点；主要损失集中在
`seeingdark`、`crn` 和 `imle`，而 `guided` Acc 上升 2.00 点。88,353 个唯一
索引和标签完全一致，共 1,015 个阈值分歧，平均绝对概率差 0.01151。

关键路径总计 29,925.58 s，吞吐为 2.9524 unique images/s，加权瓶颈 rank
延迟为 2,708.93 ms/image；rank 峰值显存为 8,437.55/8,776 MiB，三机干净物理
峰值为 37,167/18,652/19,158 MiB。八 rank、三 launcher、只读 SSHFS 与退出
状态全部审计通过。因此保留“选择两视图并非无损加速，六视图是更均衡默认值”的
结论。该最终结果提交并推送前不启动 `select4`。

提交 `f3f1506` 推送后，`select4` 在三机重新通过空输出、代码、NCCL、checkpoint、
端口、GPU 和真实 12,764 行 `crn` Arrow 预检；3090 只读 SSHFS 正常且未访问失效
`/data`，4090-2 使用已审计驱动兼容库。rank 4/5 于 11:58:42 启动，rank 6/7
于 11:59:00 启动，rank 0-3 于 11:59:15 完成 worker-first 八 rank 启动，端口
29664。全部 ranks 已越过 barrier 并进入 `crn`，rank0 首个 iteration 为
4.9484 s；三机初始显存为 37,175/18,652/19,162 MiB，均为 100% 利用率且无
启动错误。后续变体未提前启动。

13:18 `select4` 已完成 `crn` 并进入 `cyclegan`。`crn` 为 91.6432% Acc /
92.1722% AP，相对 P1 六视图低 0.8929/0.6030 点，但相对 `select2` 高
1.7309/0.1064 点；real Accuracy 分别低 1.7852 点、高 3.4607 点，fake
Accuracy 仍为 100%。12,764 个唯一索引和标签与两组对照完全一致；相对 P1 有
154 个阈值分歧，平均绝对概率差 0.01240。该域吞吐为 2.9483 unique images/s，
瓶颈 rank 延迟 2,712.53 ms/image。三机 GPU、3090 只读 SSHFS 与八 rank 日志
均正常，暂时退化原样保留，后续变体未提前启动。

13:48 `select4` 已完成 `crn`、`cyclegan`、`dalle` 并运行 `biggan`。三域
部分平均为 96.3738% Acc / 96.5628% AP，相对 P1 低 0.4187/0.2168 点，
但相对 `select2` 高 0.6195/0.1798 点。real Accuracy 相对两者分别低
0.7371 点、高 1.4381 点；fake Accuracy 低 0.10/0.20 点。`cyclegan` 相对
`select2` 恢复 0.3776 Acc 点，`dalle` 则相对两组对照均低 0.25 点。17,406
个唯一索引和标签完全一致，相对 P1 有 166 个阈值分歧，平均绝对概率差
0.00981。域吞吐保持 2.9472-2.9565 images/s，三机、SSHFS、显存和八 rank
日志继续正常，后续变体未提前启动。

14:48 `select4` 已完成五域并运行 `gaugan`，部分平均为 96.7531% Acc /
96.9622% AP，相对 P1 低 0.2625/0.1403 点，但相对 `select2` 高
0.4143/0.1586 点。real Accuracy 相对两者分别低 0.4796 点、高 0.9802 点；
fake Accuracy 低 0.0452/0.1522 点。`biggan` 与 P1 几乎相同且比 `select2`
高 0.25 Acc 点，`deepfake` 与两组对照的 Acc 差均小于 0.04 点。26,811 个
索引和标签完全一致，相对 P1 有 180 个阈值分歧，平均绝对概率差 0.00703。
吞吐、rank 显存、三机物理显存、SSHFS 和八 rank 日志继续稳定。

15:48 `select4` 已完成七域并进入 `glide_100_10`，部分平均升至
97.3408% Acc / 97.5981% AP，相对 P1 低 0.1875/0.1194 点，相对
`select2` 高 0.3188/0.1312 点。real Accuracy 相对两者分别低 0.3712 点、
高 0.7745 点；fake Accuracy 与 P1 基本持平，较 `select2` 低 0.1373 点。
`gaugan` 相对 `select2` 恢复 0.16 Acc 点，`glide_50_27` Acc 持平且 AP 高
0.1055 点。38,811 个索引和标签完全一致，相对 P1 有 194 个阈值分歧，平均
绝对概率差 0.00527。吞吐、显存、只读 SSHFS 和全部 rank 日志继续正常。

16:18 `select4` 已完成十域并进入 `imle`，部分平均为 95.0385% Acc /
97.6713% AP，相对 P1 低 0.0613/0.1163 点，相对 `select2` 高
0.0382/0.0439 点。`guided` 绝对结果仍弱（73.00% Acc / 94.96% AP），相对
P1 Acc 高 0.60 点，但相对 `select2` 低 1.40 点。44,811 个索引和标签与两组
对照一致，相对 P1 有 234 个阈值分歧，平均绝对概率差 0.00547。

A6000 原始监控在 `glide_100_27` 期间从 37,175 MiB 短暂升至 39,240 MiB：
15:55:43 开始、15:55:54 达峰、15:55:55 恢复。rank 内部峰值仍为
8,437.55/8,776 MiB，八 rank 无错误；16:21 仅见四个 P4 进程且物理显存为
37,175 MiB。因事件当时没有进程快照，额外 2,065 MiB 原样记录为“归属不明的
瞬时峰值”，不武断归因于 P4 或其他用户。该域耗时未出现离群但保守标记，不重启、
不丢弃结果，后续使用 rank 内部显存作为可归因曲线。

`imle` 期间又记录到三段 A6000 共享显存。第一段在 16:57:50 首次观察时已为
39,448 MiB，16:58:16 达到本次原始峰值 41,788 MiB，16:59:07 恢复；第二段
从 16:59:43 升至 41,398 MiB 并持续到 17:09:14。两段均没有同步进程快照，
因此保留为归属不明，不事后猜测。第三段从 17:09:19 开始，17:21:48 快照确认
另一用户的 `train_sharepara_moe_0406_loss.py` 进程占用 4,530 MiB，四个 P4 rank
仍合计 37,144 MiB；该进程只观察、未修改或终止。当前预测有效，但 `imle` 的
耗时与吞吐标记为共享 GPU 污染。截至该时点分别报告 41,788 MiB 原始主机峰值、
37,175 MiB 干净 P4 平台值及 rank 内部 CUDA 显存，不重启、不丢弃原始结果。

17:49 `select4` 已完成十三域并运行 `ldm_200_cfg`，部分平均为 95.4505% Acc /
97.5280% AP，相对 P1 低 0.0882/0.1257 点，相对 `select2` 高
0.1626/0.1025 点。`imle` 为 91.7215% Acc / 92.1722% AP，相对 P1 低
0.6344/0.4683 点，但相对 `select2` 恢复 1.8327/0.8180 点；`ldm_100` Acc
与 P1 相同，`ldm_200` Acc 高 0.10 点。61,575 个唯一索引和标签与两组对照
一致，相对 P1 有 365 个阈值分歧，加权平均绝对概率差为 0.00624。

外部 A6000 进程仍在：17:49:18 起显存配置发生变化，17:52:38 该进程占用降至
1,554 MiB，主机原始显存为 38,734 MiB，四个 P4 rank 仍合计 37,144 MiB。
`imle`、`ldm_100`、`ldm_200` 吞吐分别为 2.9303/2.9163/2.9151 images/s，
低于之前 2.9472-2.9601 的范围，因此预测保留但耗时明确标为共享 GPU 污染；
未干预外部进程。4090-2 默认 NVML 仍有已知库版本不匹配，使用已审计
580.159.03 兼容库可正常读取 19,162 MiB/100%，两 rank 和全部日志正常。
18:00:01 同一外部进程又扩回 4,514 MiB，A6000 原始显存回到 41,694 MiB，
说明其占用会动态变化，`ldm_200_cfg` 耗时同样标记为共享 GPU 污染。

18:18 `select4` 新增完成 `ldm_200_cfg` 并进入 `progan`。十四域部分平均为
95.6005% Acc / 97.6475% AP，相对 P1 低 0.0748/0.1167 点，相对
`select2` 高 0.1367/0.1055 点。`ldm_200_cfg` 为 97.55% Acc / 99.20% AP，
Acc 相对 P1 高 0.10 点、相对 `select2` 低 0.20 点。63,575 个唯一索引和标签
继续完全一致，相对 P1 有 371 个阈值分歧，加权平均绝对概率差为 0.00615。
外部 A6000 进程仍占 4,514 MiB，该域吞吐降至 2.8307 images/s、瓶颈延迟升至
2,826.17 ms，成为目前最明显的共享 GPU 耗时影响；预测指标有效并原样保留。
八 rank、3090 只读 SSHFS 和 4090-2 兼容库路径均正常，后续变体未提前启动。

18:49 `select4` 已完成十六域并进入 `seeingdark`。部分平均为 95.7953% Acc /
97.7355% AP；同域 Acc 首次比 P1 高 0.0056 点，AP 仍低 0.0368 点；相对
`select2` 的 Acc/AP 高 0.2056/0.2157 点。`progan` 四项指标均为 100%，与 P1
完全一致；`san` 为 94.32% Acc / 96.70% AP，相对 P1 高 1.1364/1.0449 点，
相对 `select2` 高 1.3636/1.9747 点。72,013 个唯一索引和标签完全一致，相对
P1 仅有 378 个阈值分歧，加权平均绝对概率差为 0.00552。正向结果不筛选、原样
保留。外部 A6000 进程仍在，因此 `progan`、`san` 及运行中的 `seeingdark` 耗时
均标为共享 GPU 污染；两个已完成域吞吐为 2.8790/2.8421 images/s，但预测有效。
八 rank 和两条数据路径正常，`pointwise` 未提前启动。

18:52 `seeingdark` 完成，17 域快照在进入 `stargan` 前保留其偏弱结果：
88.61% Acc / 86.61% AP，real/fake Accuracy 为 77.22%/100%。相对 P1
Acc/AP 低 1.3889/0.8267 点，但仍比 `select2` 高 2.2222/2.5464 点。17 域
部分平均为 95.3727% Acc / 97.0810% AP，相对 P1 低 0.0764/0.0833 点，
相对 `select2` 高 0.3243/0.3528 点。72,373 个唯一索引和标签完全一致，弱结果
不重跑、不筛选。外部 A6000 任务仍在，该域吞吐降至 2.3972 images/s、瓶颈
延迟升至 3,337.23 ms，因此耗时不作为干净 P4 测量，预测结果仍有效。

19:18 `stargan` 完成并进入最后的 `stylegan`。18 域部分平均为 95.4548% Acc /
97.0038% AP，相对 P1 低 0.0778/0.0787 点，相对 `select2` 高
0.3285/0.3409 点。`stargan` 为 96.85% Acc / 95.69% AP，相对 P1 Acc 低
0.10 点且 AP 相同，相对 `select2` 高 0.40/0.1372 点。76,371 个唯一索引和标签
完全一致，相对 P1 有 407 个阈值分歧，加权平均绝对概率差为 0.00560。

此前外部 PID 已退出，但相同用户和命令的新 PID 832548 于 19:06:54 启动并占
4,530 MiB；19:08:02 A6000 原始显存出现新的 42,386 MiB 峰值。五次 pmon 采样
中直接观察到该进程一次占用 33% SM/28% memory，仍只观察、不干预。因此
`stylegan` 耗时明确属于共享计算污染，预测仍可用于性能指标。

20:23:14 `select4` 完成全部 19 域，总耗时 8:23:59。最终为 95.4230% Acc /
97.1425% AP，real/fake Accuracy 为 95.9466%/94.8983%。相对匹配的 P1 六视图
对照仅低 0.0693/0.0749 Acc/AP 点，real Accuracy 低 0.3284 点、fake Accuracy
高 0.1900 点；相对 `select2` 恢复 0.3068 Acc、0.3219 AP 和 0.8600 real
Accuracy 点，fake Accuracy 低 0.2468 点。剩余 P1 差距集中在 `seeingdark`、
`crn` 和 `imle`，而 `san` Acc 高 1.1364 点。88,353 个唯一索引和标签完全
一致，相对 P1 有 455 个阈值分歧，加权平均绝对概率差为 0.00538。

关键路径总计 30,181.21 s，吞吐为 2.9274 unique images/s，加权瓶颈延迟为
2,732.07 ms/image；rank 内部峰值显存仍为 8,437.55/8,776 MiB。三机原始峰值
为 42,910/18,652/19,162 MiB，但 A6000 峰值及 `imle` 至 `stylegan` 耗时受
共享进程污染，干净 P4 平台值为 37,175 MiB；最终原始峰值是 19:59:44 的单个
采样。八 rank 和三 launcher 无错误，3090 SSHFS 全程只读，所有 P4 进程退出。

结论是四视图为目前最佳降成本设置，六视图仍是最均衡准确率默认值。`pointwise`
没有提前启动：20:27 外部 A6000 进程仍占 4,538 MiB 且使用 48% GPU，故其干净
profiling 预检进入等待，避免继续污染下一项有序实验。

7 月 31 日 05:48，外部训练序列完全退出，A6000 恢复为 17 MiB、0% 利用率且无
计算进程；等待期间观察到 PID 832548、862256、867149、871170，但从未修改或
终止。首次三机预检命令因 SSH 远端未保留数据路径中管道符的引号而在三机均以
126 退出，首次端口探测也因 Python `-c` 引号丢失以 2 退出；两次均未启动 rank、
未创建实验结果，失败原样记录后修正命令并通过复检。

修正后的预检确认真实 12,764 行 `crn` Arrow、NCCL 23007、六视图选择、pointwise
entropy、3090 只读 SSHFS、4090-2 隔离驱动库、三机脚本哈希、空输出目录及空闲
29665 端口。05:52:47/05:53:09 先启动 3090 和 4090-2 workers，05:53:43 再启动
A6000 ranks。八 rank 均越过 barrier，rank0 已进入 `crn`，首 iteration 为
6.6779 s；三机初始显存为 37,167/18,656/19,162 MiB、利用率均为 100%，关键日志
无错误。有序第 7 项 `pointwise` 已开始运行。

07:19 首域 `crn` 完成并进入 `cyclegan`。`crn` 为 89.0821% Acc / 91.8670%
AP，real/fake Accuracy 为 78.1710%/100%；相对同为六视图的 averaged P1
对照低 3.4539 Acc、0.9081 AP 和 6.9057 real-accuracy 点，相对 `select4`
低 2.5611/0.3052 Acc/AP 点。12,764 个唯一索引和标签与两组对照完全一致，
相对 P1 有 443 个阈值分歧、概率 MAD 为 0.03206，差异全部集中在 real 类。
该偏弱结果不重跑、不筛选。

该域耗时 4,446.67 s，吞吐 2.8705 images/s、瓶颈延迟 2,786.13 ms，比
`select4` 慢约 2.71%；rank 内部显存仍为 8,437.55/8,776 MiB，三机原始峰值
为干净的 37,167/18,656/19,162 MiB。八 rank 均存活且 3090 SSHFS 只读。
一次 A6000 SSH 检查超时、首次带格式的远端文件列表因引号丢失失败，均原样记录
并立即只读重试成功，对实验无影响。

07:49 已完成 `crn`、`cyclegan`、`dalle` 并运行 `biggan`。三域部分平均为
95.6242% Acc / 96.3819% AP，相对匹配 P1 低 1.1683/0.3977 点，相对
`select4` 低 0.7496/0.1809 点。`cyclegan` 为 98.6405%/97.7071%，相对 P1
低 0.1511/0.2899 点；`dalle` 为 99.15%/99.5716%，Acc 高 0.10 点且 AP
基本不变。17,406 个唯一样本和标签与两组对照一致，相对 P1 有 455 个阈值分歧、
加权 MAD 为 0.02422，正负结果均保留。

三域关键路径共 6,066.76 s、吞吐 2.8691 images/s，比 `select4` 慢约 2.79%；
rank 内部显存维持 8,437.55/8,776 MiB，三机原始峰值仍为干净的
37,167/18,656/19,162 MiB。八 rank 正常，A6000 未出现外部共享进程。

08:49 已完成 `biggan`、`deepfake` 并运行 `gaugan`。五域部分平均为
96.2831% Acc / 96.7975% AP，相对匹配 P1 低 0.7325/0.3049 点，相对
`select4` 低 0.4700/0.1646 点。`biggan` 为 98.50%/97.7603%，相对 P1
低 0.25/0.1981 点；`deepfake` 为 96.0429%/97.0817%，Acc 高 0.0925 点、
AP 低 0.1334 点。26,811 个唯一样本和标签与两组对照一致，相对 P1 有 484
个阈值分歧、加权 MAD 为 0.01684。

五域关键路径共 9,342.31 s、吞吐 2.8698 images/s，比 `select4` 慢约 2.87%；
rank 内部显存仍为 8,437.55/8,776 MiB，三机原始峰值保持
37,167/18,656/19,162 MiB。A6000 仅有四个 P4 rank，八 rank 全部存活，3090
SSHFS 仍为只读。首次本地 P1 对比误用了不存在的结果路径并以 1 退出，失败已记录，
随后使用正确路径完成复算，对实验无影响。

09:44 已完成 `gaugan`、`glide_50_27` 并运行 `glide_100_10`。七域部分平均为
97.0208% Acc / 97.4702% AP，相对匹配 P1 低 0.5075/0.2473 点，相对
`select4` 低 0.3200/0.1279 点。`gaugan` 为 99.23%/98.8725%，相对 P1
低 0.19/0.2942 点；`glide_50_27` 为 98.50%/99.4314%，相对 P1 高
0.30/0.0877 点。38,811 个唯一样本和标签与两组对照一致，相对 P1 有 517
个阈值分歧、加权 MAD 为 0.01250。

七域关键路径共 13,477.68 s、吞吐 2.8797 images/s，比 `select4` 慢约 2.56%；
rank 内部显存保持 8,437.55/8,776 MiB，三机原始峰值仍为
37,167/18,656/19,162 MiB。A6000 仅有四个 P4 rank，八 rank 均存活，3090
SSHFS 保持只读。

09:48 一次经配置 SSH 别名执行的 3090 只读 `nvidia-smi` 监控连接超时并以
255 退出；同一轮并行检查中该主机的 `findmnt` 成功。立即复查得到 18,656 MiB、
100% 利用率，3090 两 rank 与 A6000 四 rank 均保持活跃，rank0 继续更新
`glide_100_10`；A6000 和 4090-2 到 3090 LAN 的三次探测均零丢包。该瞬时监控
失败已独立保存，不是节点故障、rank 停滞或数据挂载故障，对实验无影响。

10:19 已完成 `glide_100_10`、`glide_100_27`、`guided` 并运行 `imle`。十域
部分平均为 94.8895% Acc / 97.5010% AP，相对匹配 P1 低 0.2103/0.2866 点，
相对 `select4` 低 0.1490/0.1703 点。`glide_100_10` 相对 P1 高
0.40/0.0149 点，`glide_100_27` 基本持平。`guided` 是混合结果：绝对 Acc
仅 73.50%，但仍比 P1 高 1.10 点；AP 为 94.2165%，比 P1 低 1.1452 点。
该结果不重跑、不筛选。

44,811 个唯一样本和标签与两组对照一致，相对 P1 有 582 个阈值分歧、加权
MAD 为 0.01224。十域关键路径共 15,539.40 s、吞吐 2.8837 images/s，比
`select4` 慢约 2.43%；rank 内部与三机显存峰值均未变化，八 rank 活跃且日志
无关键错误。

10:18 并行状态检查再次出现同样的 3090 SSH 超时：GPU 查询超时时，并发的挂载
查询成功；串行重试仅用 0.37 s 即返回 18,656 MiB、100% 利用率。两次超时都只
出现在同机并发 SSH 连接时，且全部 rank 持续推进，因此后续 3090 挂载与 GPU
检查改为串行。重复监控失败和修正方案均已独立保存。

11:32 已完成大域 `imle` 并运行 `ldm_100`。十一域部分平均为 94.3830% Acc /
96.9756% AP，相对匹配 P1 低 0.4674/0.3441 点，相对 `select4` 低
0.3540/0.1958 点。`imle` 为 89.3170% Acc / 91.7218% AP，real/fake
Accuracy 为 78.6408%/100%；相对 P1 低 3.0388 Acc、0.9187 AP 和 6.0758
real-accuracy 点，成为 pointwise 当前主要损失来源。该弱结果原样保留。

57,575 个唯一样本和标签与两组对照一致，相对 P1 有 974 个阈值分歧、加权
MAD 为 0.01588。十一域关键路径共 19,917.74 s、吞吐 2.8906 images/s，比
`select4` 慢约 2.00%；rank 内部与三机显存峰值未变化，八 rank 活跃且日志无
关键错误。

11:31 两个并发 A6000 SSH 连接中，日志查询在 banner exchange 阶段超时，预测
列表连接成功；串行重试 0.81 s 即返回并确认正常推进。监控修正因此推广到所有
主机：同一服务器上的 SSH/SCP 操作全部串行。随后日志/profile 与预测的快照传输
已按此规则依次完成，均成功；该监控失败与修正也已独立保存。

12:05 已完成 `ldm_100`、`ldm_200`、`ldm_200_cfg` 并运行 `progan`。十四域
部分平均为 95.3366% Acc / 97.4498% AP，相对匹配 P1 低 0.3387/0.3145 点，
相对 `select4` 低 0.2639/0.1977 点。`ldm_100` Acc 高 0.05 点、AP 低
0.1814 点；`ldm_200` Acc 持平、AP 低 0.2039 点；`ldm_200_cfg` Acc 高
0.35 点、AP 低 0.2324 点。

63,575 个唯一样本和标签与两组对照一致，相对 P1 有 998 个阈值分歧、加权
MAD 为 0.01476。rank 内部显存仍为 8,437.55/8,776 MiB，日志无关键错误。

A6000 上另一用户的 PID 1012081 于 11:33:59 启动并占用 4,348 MiB；原始监控
在 11:34:14 越过 40,000 MiB，11:35:13 达到 42,364 MiB。五次 pmon 采样未
直接看到其 SM 使用，但短窗口不足以证明受影响域全程无竞争。因此 A6000 原始显存
从该时刻起标记为共享进程污染，`ldm_100` 后段及之后域的计时保守标记为可能受
共享计算影响；预测和 rank 内部 CUDA 显存仍有效。外部进程仅观察，未作修改。

12:20 复查发现先前外部 PID 1012081 已消失，同一独立 A6000 用户在 12:15:59
启动替代 PID 1023924，占用 4,476 MiB；此时 A6000 总显存 41,648 MiB，
`progan` 运行至 rank 0 的 350/1000。五次 pmon 采样中两次直接观察到该进程
使用 9%/19% SM，确认 `progan` 存在共享计算，而不只是共享显存。八个实验 rank
均存活，3090 SSHFS 仍为只读，另两台主机显存保持预期平台。预测继续有效；
`progan` 计时保留但不作为干净计时对照，外部进程未作任何修改。

12:53 已完成 `progan`、`san`、`seeingdark` 并开始 `stargan`。十七域部分平均
为 94.9786% Acc / 96.6477% AP，相对匹配 P1 低 0.4705/0.5166 点，相对
`select4` 低 0.3941/0.4333 点。`progan` 为 100%/100% 并与两组对照一致；
`san` 为 94.0909%/94.9032%，相对 P1 Acc 高 0.9091 点而 AP 低 0.7556 点；
`seeingdark` 仅 85.8333%/83.8105%，相对 P1 低 4.1667/3.6240 点，其中
real Accuracy 仅 71.6667%，fake Accuracy 仍为 100%。混合与弱结果均原样保留。

72,373 个唯一样本和标签与两组对照一致，相对 P1 有 1,031 个阈值分歧、加权
MAD 为 0.01342。rank 内部显存仍为 8,437.55/8,776 MiB，日志无关键错误。
十七域 profile 比 `select4` 慢 1.52%，但 `progan`、`san`、`seeingdark` 期间
已直接确认共享计算，因此该计时仅保存审计，不作为干净计时结论。

13:20 已完成 `stargan` 并开始最后一个 `stylegan` 域。十八域部分平均为
95.0757% Acc / 96.5869% AP，相对匹配 P1 低 0.4569/0.4955 点，相对
`select4` 低 0.3791/0.4169 点。`stargan` 为 96.7250%/95.5545%，real
Accuracy 93.4533%、fake Accuracy 100%，相对 P1 低 0.2250/0.1372 点。

76,371 个唯一样本和标签与两组对照一致，相对 P1 有 1,056 个阈值分歧、加权
MAD 为 0.01304。rank 内部显存和日志继续正常。外部 A6000 进程仍占 4,496 MiB，
因此十八域 profile 继续仅作审计保存，不作为干净计时结论。

14:25 `pointwise` 完成全部 19 域并通过结果审计。最终为 95.0718% Acc /
96.7480% AP，相对匹配 averaged P1 低 0.4205/0.4694 点，相对 `select4`
低 0.3513/0.3945 点。平均 real Accuracy 相对 P1 低 1.3897 点，fake Accuracy
反而高 0.5502 点；差距主要集中在 `crn`、`imle`、`seeingdark`。最后的
`stylegan` 为 95.0017%/99.6471%，Acc 相对 P1 高 0.2336 点。

88,353 个唯一样本和标签与两组对照一致，相对 P1 有 1,132 个阈值分歧、加权
MAD 为 0.01213。八个实验 rank 均正常退出，3090 SSHFS 全程只读，日志无关键
错误。预测和 rank 内部显存有效；A6000 原始显存及 `ldm_100` 后段起的计时继续
按共享进程污染处理；`progan` 期间直接观测到外部 SM 使用，之后该进程持续驻留。

14:34 有序第 8 项 profiled averaged `baseline` 的输出路径和数据挂载检查通过，
但 A6000 上另一用户 PID 1023924 仍占 4,514 MiB 并使用 GPU，因此未启动。
外部进程未作修改；待 A6000 回到干净空闲平台后重新执行三主机完整预检并启动。

14:50 再次按顺序检查三台主机。`baseline` 输出目录在三机仍均不存在，3090
只读 SSHFS 正常且未访问故障 `/data`，3090/4090-2 分别为 106/252 MiB、0%。
A6000 的同一外部 PID 1023924 已运行 2:34:29，仍占 4,514 MiB，检查瞬间 GPU
利用率为 71%，并有 9 个同命令 worker。该进程仅观察、未修改；为保持 baseline
计时和显存曲线可比，实验继续处于预检等待，未越过到 `select8`。

15:19 复查发现 PID 1023924 已自行退出，但同一外部用户的训练序列随即换成
PID 1063576/1068634，分别占 1,816/4,710 MiB；A6000 合计 6,554 MiB、70%
利用率，共有 18 个同命令 worker。3090/4090-2 仍空闲，SSHFS 与三机空输出
检查继续通过。外部任务仅观察、未修改；`baseline` 仍不启动，顺序未前移。

17:19 两个外部 PID 已分别持续 2:18:31/2:05:24，A6000 为 6,556 MiB、98%；
另外两机、只读 SSHFS 和三机空输出仍正常。等待状态已再次归档，未提前启动实验。

19:19 同两个 PID 已分别持续 4:18:53/4:05:46，A6000 仍为 6,556 MiB、99%；
其他前置条件无漂移。持续等待已归档并保留严格实验顺序。

21:19 同两个 PID 已分别持续 6:18:51/6:05:44，A6000 仍为 6,556 MiB、99%；
3090/4090-2、只读 SSHFS 和三机空输出保持正常。周期审计已同步，未越序。

23:19 同两个 PID 已分别持续 8:19:14/8:06:07，A6000 仍为 6,556 MiB、77%；
其他前置条件保持正常。持续等待已归档，`baseline` 未启动。

8 月 1 日 01:19，同两个 PID 已分别持续 10:19:10/10:06:03，A6000
仍为 6,556 MiB、99%。3090 瞬时利用率 1% 且无计算进程，4090-2 为 0%，只读
SSHFS 和三机空输出保持正常。跨日等待已归档，顺序未改变。

03:19 同两个 PID 已分别持续 12:18:49/12:05:42，A6000 仍为 6,556 MiB、99%；
其他两机、只读 SSHFS 和三机空输出无漂移。十二小时等待已归档并同步。

03:49 PID 1063576 已自行退出，但 PID 1068634 仍占 4,712 MiB，共 9 个
worker，A6000 为 4,737 MiB、20%。该变化已归档；由于关键节点尚未回到
干净空闲平台，`baseline` 仍未启动。

04:19 剩余外部 PID 1068634 自行退出，A6000 回到 17 MiB、0% 且无计算
进程。三机随后完成代码/权重哈希、NCCL 23007、真实 12,764 行 `crn`
Arrow、averaged entropy、六视图精确选择、只读 SSHFS、4090-2 隔离驱动库、
空输出、空闲 GPU 和 29666 端口检查。

04:22:13/04:22:29 先启动 3090 ranks 4-5 和 4090-2 ranks 6-7，04:22:49
启动 A6000 ranks 0-3。八 ranks 均越过 barrier，rank0 已进入 `crn`，首迭代
5.4718 秒、rank 峰值 8,437 MiB；三机为 37,175/18,652/19,158 MiB 且
100% 利用率。有序第 8 项 profiled averaged `baseline` 已开始运行。

05:49 `baseline` 完成 `crn` 和 `dalle`，平均为 95.7117% Acc / 96.1881% AP，
相对匹配 P1 仅低 0.0813 Acc 点、高 0.0172 AP 点。`crn` 为 92.4734% /
92.8021%，`dalle` 为 98.9500% / 99.5741%。14,764 个唯一样本的 index/
label 序列与 P1 完全一致，但随机视图导致 56 个阈值分歧、加权 MAD
0.00425，说明微小指标差异不是协议漂移。

两域净计时吞吐为 2.9320 images/s，同域比干净 `select4` 慢 0.59%，比干净
`pointwise` 快 2.12%；rank 峰值为 8,437.55/8,776 MiB。三机无关键错误，
A6000 仅有本实验四个 rank，3090 SSHFS 仍为只读。后处理中两次本地
Python 路径/依赖失败与最终成功重试均已原样记录。`biggan` 正在运行。

07:19 `baseline` 已完成前五域，平均为 96.9907% Acc / 97.1246% AP，
相对匹配 P1 仅低 0.0249 Acc 点、高 0.0221 AP 点。新完成的 `biggan`
为 98.8250%/98.1444%，`cyclegan` 为 98.7915%/97.9244%，`deepfake`
为 95.9135%/97.1781%，均与 P1 在 0.2 点内。

26,811 个唯一样本的 index/label 序列与 P1 一致，共 81 个阈值分歧、
加权 MAD 0.00344。同域净吞吐为 2.9308 images/s，比 `select4` 慢 0.73%、
比 `pointwise` 快 2.12%。三机、八 ranks、A6000 独占性和只读 SSHFS 无异常，
`gaugan` 已进入中段。

08:49 `baseline` 完成前十域，平均为 95.1343% Acc / 97.7847% AP，
相对 P1 高 0.0345 Acc 点、仅低 0.0029 AP 点。`gaugan` 为
99.4400%/99.1667%，三个 GLIDE 为 98.10%/99.40%、98.00%/98.96%、
98.45%/99.43%。`guided` 为 72.40%/95.27%，real/fake Accuracy 为
98.90%/45.90%；该低 fake 结果与 P1 一致并原样保留。

44,811 个唯一样本序列与 P1 一致，118 个阈值分歧，加权 MAD 0.00297。
净吞吐 2.9313 images/s，同域比 `select4` 慢 0.76%、比 `pointwise` 快
1.65%。三机仍保持干净运行，`imle` 已开始。

10:19 `baseline` 完成前十三域并进入 `ldm_200_cfg`，平均为
95.5728% Acc / 97.6573% AP，相对匹配 P1 仅高 0.0340/0.0035 点。
新完成的 `imle` 为 92.4029%/92.6001%，`ldm_100` 为
99.20%/99.44%，`ldm_200` 为 99.50%/99.66%，三域均与 P1 在 0.10 点内。

61,575 个唯一样本的 index/label 序列与 P1 一致，共 161 个阈值分歧，
加权 MAD 0.00300。同域净吞吐为 2.9324 images/s，比 `select4` 慢 0.47%、
比 `pointwise` 快 1.45%。rank 峰值保持 8,437.55/8,776 MiB，3090 SSHFS
仍为只读，八 ranks、A6000 独占性和关键日志检查均正常。

10:49 前十四域已完成，`progan` 运行至 500/1000。十四域平均为
95.7140% Acc / 97.7751% AP，相对 P1 高 0.0387/0.0109 点；新完成的
`ldm_200_cfg` 为 97.55%/99.31%。63,575 个唯一样本序列与 P1 一致，
167 个阈值分歧，加权 MAD 0.00300。

同次审计发现 A6000 上另一用户 PID 1255877 于 10:46:42 启动，占
4,348 MiB 并带 8 个子进程。该进程只观察、未修改。原始监控从 10:46:51
开始超过本实验 37,175 MiB 干净平台，随后达到 41,528 MiB；13 秒 `pmon`
抽样未直接看到外部 SM 使用，但不能证明整个受影响区间均无竞争。前十四域
仍构成 2.9322 images/s 的干净边界；`progan` 约 450 iter 后的计时及之后
A6000 原始显存作保守限定，预测和 rank 内部显存仍有效，实验不中止、不选结果。

11:19 `baseline` 完成前十七域并进入 `stargan`，平均为
95.4677% Acc / 97.2005% AP，相对 P1 高 0.0185/0.0362 点。`progan`
为 100%/100%，`san` 为 92.9545%/95.6206%，`seeingdark` 为
90.00%/87.9364%。72,373 个唯一样本序列与 P1 一致，172 个阈值分歧，
加权 MAD 0.00270。

第二个外部 PID 1268262 于 11:04:31 启动，另占 1,974 MiB 并带 8 个
子进程。约 11:19:45 的 `pmon` 在 `stargan` 期间直接观测到两个外部 PID
分别使用 14%/28% SM；两进程均未修改。十七域观测吞吐 2.9283 images/s
原样保留但不作为干净结论，前十四域继续作为干净计时边界；预测指标和 rank
内部显存有效，实验继续运行。

11:49 `stargan` 完成并进入最后的 `stylegan`。十八域平均为
95.5445% Acc / 97.1116% AP，相对 P1 高 0.0119/0.0291 点。`stargan`
为 96.8500%/95.6002%，real/fake Accuracy 为 93.7031%/100%，仅比 P1
低 0.10/0.09 点。76,371 个唯一样本序列与 P1 一致，182 个阈值分歧，
加权 MAD 0.00270。

`stylegan` 期间仍直接观测到外部 SM 使用；A6000 原始峰值 43,869 MiB
不归因于本实验。十八域观测吞吐 2.9243 images/s 原样保留，但前十四域仍是
唯一干净计时边界。另两节点、八 ranks、只读 SSHFS、rank 内部显存和关键
错误检查均正常。

`baseline` 于 12:49:12 完成全部 19 域，最终 Acc/AP 为
95.5023%/97.2455%，相对 P1 仅高 0.0100/0.0281 点；real/fake Accuracy
仅高 0.0064/0.0135 点。88,353 个唯一样本及标签序列完全一致，197 个阈值
分歧、加权 MAD 0.00250，说明差异来自随机视图而非协议漂移。

相对 `pointwise`，averaged entropy 的 Acc/AP 高 0.4305/0.4975 点，主要来自
real Accuracy 高 1.3961 点，fake Accuracy 低 0.5367 点。`stylegan` 最终为
94.7430%/99.6564%，与 P1 在 0.03/0.02 点内；`guided` 的 72.40% Acc
弱结果继续保留。

全程观测吞吐为 2.9134 images/s，rank 内部分配/保留峰值为
8,437.55/8,776 MiB。两个外部进程持续至结束，因此 A6000 43,869 MiB
原始峰值和全 19 域计时不作为干净结论；前十四域提供 2.9322 images/s
干净边界。八个实验 rank 均已退出，3090/4090-2 回到 106/252 MiB，SSHFS
全程只读，rank 和 launcher 日志均无关键错误。有序第 9 项 `select8` 下一步
等待 A6000 回到干净空闲平台后再启动。

13:03:15 的 `select8` 首次预检确认严格顺序已满足：`baseline` 最终产物已由
`f47d306` 提交。三台主机均无 `select8` 输出，3090 的只读 SSHFS Arrow 挂载
和 4090-2 的隔离驱动均正常且 GPU 空闲；但 A6000 上两个外部训练进程仍在
运行，占 6,324 MiB、利用率 99%。未修改这些外部进程，`select8` 保持
`preflight_wait`，待 A6000 恢复 17 MiB/0% 干净基线后重做完整三机预检。

15:19:36 复查时，同一组两个外部任务仍以 6,458 MiB/99% 占用 A6000，
`select8` 输出仍不存在且外部进程未作修改；第二份等待审计已保留，实验不越序。

17:19:38 周期复查时，两个外部 PID 已分别持续 6:32:56/6:15:07，A6000
为 6,458 MiB/85%，`select8` 输出仍为空。外部任务未修改，也未越序启动后续项。

19:19 复查发现上一组 PID 已自行退出，但同一外部训练序列已于 19:13/19:14
换成 PID 1338443/1339204，分别占 1,988/4,232 MiB 且各有 8 个 worker；
A6000 为 6,250 MiB/80%，未观测到可安全启动的干净窗口。3090/4090-2 复核
仍空闲、只读且三机输出为空。一次 `awk` 计数转义失败也已保留，不影响进程
完整列表给出的判断。

21:20:14 时替换后的两个 PID 已持续 2:06:21/2:05:26，A6000 仍为
6,250 MiB/99%，`select8` 输出继续为空。外部任务未修改，有序等待继续保留。

23:20:13 时同两个替换 PID 已持续 4:06:20/4:05:25，A6000 为
6,254 MiB/99%；`select8` 仍未启动且输出为空，四小时等待已归档且未修改外部任务。

8 月 2 日 01:20:46 时，替换后的两个 PID 已持续 6:06:52/6:05:57，A6000
仍为 6,254 MiB/93%；`select8` 未启动且输出为空，跨夜等待已归档且未修改外部任务。

03:20:55 时两个替换 PID 已持续 8:07:02/8:06:07，A6000 仍占 6,254 MiB；
瞬时 65% 利用率不代表干净平台。`select8` 仍无输出且未启动，外部任务未修改。

05:20:44 时两个替换 PID 已持续 10:06:50/10:05:55，A6000 为
6,254 MiB/99%；`select8` 未启动且输出为空，十小时等待已归档且未修改外部任务。

07:20:47 时两个替换 PID 已持续 12:06:53/12:05:58，A6000 为
6,254 MiB/95%；`select8` 未启动且输出为空，十二小时等待已归档且未修改外部任务。

09:21 时占 4,234 MiB 的外部 PID 已自行退出，剩余 1,990 MiB PID 于 09:22:13
前退出，A6000 恢复 17 MiB/0% 干净平台。一次延迟 SSH 观察无输出，立即重试
确认转换且失败已保留。随后三机重新通过代码/权重哈希、NCCL 23007、真实
12,764 行 `crn` Arrow、精确八视图选择、averaged entropy、3090 只读 SSHFS、
4090-2 隔离驱动、三机空输出及空闲端口预检。

`select8` 分别于 09:26:14、09:26:33、09:27:01 启动 3090 ranks 4-5、
4090-2 ranks 6-7 和 A6000 ranks 0-3，端口为 29667。八 ranks 均跨过 barrier，
rank 0 已进入 `crn`，首 iter 为 3.8346 秒。A6000 最终 shell guard 将 `17,0`
误解析成单字段并产生两条整数检查警告，但启动前正确查询和 guard 自身输出均
确认 17 MiB/0%、无 compute process、端口空闲，启动后 GPU 进程又精确匹配四个
rank PID，因此运行平台仍是干净的。另一次启动后 SSH banner timeout 重试成功；
两项失败均已保留在 launch audit，未跳过。

11:20 首份干净快照含 `crn`、`dalle`、`biggan`，均值 Acc/AP 为
96.8253%/96.9201%，相对匹配 P1 高 0.0466/0.1533 点，相对 profiled
selection6 高 0.0758/0.0799 点。18,764 个唯一索引和标签一致，相对 P1 有
114 个阈值分歧、加权 MAD 0.00658。干净吞吐 2.8814 images/s，同域相对
selection6 低 1.75%。

实时复制监控期间，两个外部 A6000 进程于 11:21:13/11:21:17 启动；此时前三域
已完成，`cyclegan` 正在运行。11:24 分别占 2,112/4,348 MiB。一次 pmon 样本
未见其直接 SM 使用，但从 11:21:25 起的 A6000 原始显存及 `cyclegan` 后计时
保守按共享平台限定；预测与 rank 内部显存仍有效。尝试复制运行中并不存在的
`metrics.json` 失败已保留，随后仅从已完成预测文件生成本地指标，未修改预测。

11:51 `cyclegan` 已完成并进入 `deepfake`，结果为 98.90% Acc / 98.14% AP，
real/fake Accuracy 为 97.81%/100%。11:52:45 连续五次 `pmon` 中，外部 PID
1461586 有一次直接使用 12% SM，故竞争已被直接确认；另一个外部 PID 在这五次
样本中未显示 SM 使用。两个外部进程均未修改。前三域仍是唯一干净计时边界，
`cyclegan` 起的计时原样保留但不作为干净性能结论；预测、标签和 rank 内部显存
继续有效，`select8` 不重启、不筛选结果并按顺序继续。

12:20 复查发现 attempt1 已发生不可恢复故障。为 3090 提供只读 Arrow 的
3070x2 在 11:29 关机并于 11:33:41 重启；连接中断期间，3090 ranks 4-5
在 11:32:06 对内存映射 Arrow 取页时同时 SIGBUS，rank7 随后记录
`ncclRemoteError`。`deepfake` 虽在 rank0 完成本地 676/676 iter，但八 rank
gather 未完成且没有预测文件，不能计入。已精确终止其余失去 peers 的本实验 ranks，
两个外部 A6000 任务未修改；三机日志、监控、profile 和四个有效预测已完整归档。

失败前有效的 `crn`、`dalle`、`biggan`、`cyclegan` 四域均值为
97.3452% Acc / 97.2257% AP，相对匹配 P1 高 0.0633/0.1514 点，相对
selection6 baseline 高 0.0852/0.1145 点。21,406 个唯一索引和标签与两份参考
完全一致。前三域仍是干净计时边界；`cyclegan` 的共享平台计时原样保留。

根因不是 3090 的故障 `/data`、CUDA OOM 或数据损坏：attempt1 只使用 `/home`
下的 SSHFS URI；源机恢复后，同一挂载已重新通过 19 域、88,353 行、标签、映射和
每域首图解码。首个复检 SSH 调用异常无输出，显式 marker 重试以 exit 0 完成；
随后精确 IAPL launcher smoke 也通过 `crn` 12,764 行、32 views、selection count 8、
averaged entropy 与 OIS 检查。
本地比较工具的缺 numpy、参数误用、workspace dependency lookup 卡住、4090-1
超时和一次 `awk` 引号错误也均保留。attempt2 不拼接旧预测，将从 seed100 重跑
全部 19 域；当前 A6000 上两个外部任务仍占 6,498 MiB/99%，因此 attempt2 输出
保持为空并等待干净平台，不提前运行 `select12`。

## P5: controlled CTTA table

在相同 CNN checkpoint、样本、顺序和 Predict-Then-Adapt 协议下运行 Source、TENT、
EATA、CoTTA、RoTTA、LAME、T2A。包含 independent single-target 与 continual stream，
每项三个 seeds，报告 online、final、forgetting、延迟和显存。IAPL 继续作为不同
backbone/协议的端到端参考，不能混入控制变量排名。
