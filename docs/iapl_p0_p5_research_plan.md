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

## P4: inference ablations

按推理代价从低到高运行 TTA steps、views、confidence selection 数量、entropy loss、
OIS 开关。每项报告 mAcc、mAP、real/fake accuracy、单图延迟、峰值显存和相对完整
IAPL 的变化。

## P5: controlled CTTA table

在相同 CNN checkpoint、样本、顺序和 Predict-Then-Adapt 协议下运行 Source、TENT、
EATA、CoTTA、RoTTA、LAME、T2A。包含 independent single-target 与 continual stream，
每项三个 seeds，报告 online、final、forgetting、延迟和显存。IAPL 继续作为不同
backbone/协议的端到端参考，不能混入控制变量排名。
