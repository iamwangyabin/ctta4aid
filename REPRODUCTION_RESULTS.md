# CAIDBench reproduction results

更新时间：2026-07-17。当前公共 CNN 轨道已经完成 2 个数据轨道、7 个方法、
单目标与连续流、3 个随机种子。表中均为 `mean +- sample std`，单位为 AUC。

## 1. Source checkpoints

| 轨道 | 来源 | 源域验证 | SHA-256 |
| --- | --- | --- | --- |
| CAIDBench-GenImage 映射轨道 | SD1.4 训练 10 epochs 的公共 ResNet-50 | AUC 0.99462 | `5678f9f33bad7a162a8fa2119c5838a9050bbd391e37a1e75ca4586e878b7686` |
| CAIDBench-UFD 代理轨道 | 官方 CNNDetection `blur_jpg_prob0.5.pth` 无损映射成二分类 ResNet-50 | ProGAN AUC 1.00000 | `82c1e645da47e18643fa8ea9c0863c471f2bdc0d741a119053a4573bf8f4e684` |

两个 checkpoint 都包含 EATA 所需的 BN Fisher。所有方法、目标和 seed 都从同一
轨道 checkpoint 重新开始；结果 JSON 记录 checkpoint 路径和 SHA-256。

## 2. Independent single-target

这里每个 `method x target x seed` 独立重置。数值是各目标 AUC 的宏平均。

| Method | CAIDBench-GenImage 映射 | CAIDBench-UFD 代理 |
| --- | ---: | ---: |
| Source | 0.6394 +- 0.0000 | **0.8670 +- 0.0013** |
| TENT | 0.6205 +- 0.0325 | 0.8149 +- 0.0159 |
| EATA | 0.6310 +- 0.0011 | 0.8491 +- 0.0018 |
| CoTTA | 0.6330 +- 0.0041 | 0.8308 +- 0.0085 |
| RoTTA | 0.6292 +- 0.0004 | 0.8503 +- 0.0021 |
| LAME | **0.6394 +- 0.0006** | 0.8532 +- 0.0013 |
| T2A | 0.6214 +- 0.0037 | 0.8388 +- 0.0015 |

主要域级现象：CAIDBench-GenImage 映射轨道上 EATA 改善 Midjourney，CoTTA 改善 GLIDE/BigGAN，
LAME 小幅改善 ADM/VQDM；但没有适配方法稳定超过 Source 的整体宏平均。CAIDBench-UFD 代理轨道
上 Source 在多数域仍最强，适配的局部收益主要出现在 StarGAN、GauGAN、CRN、IMLE
和 LDM。TENT 在 GenImage GLIDE 上对流顺序非常敏感，造成明显方差。

## 3. Continual stream

`Online` 是适应流 pooled AUC；`Final` 是最后 checkpoint 在各已见域固定 holdout
上的域 AUC 宏平均；`Forget` 是除最后域外的 `best holdout AUC - final AUC` 平均。

### CAIDBench-GenImage 映射轨道

| Method | Online | Final | Forget |
| --- | ---: | ---: | ---: |
| Source | 0.6402 +- 0.0012 | **0.6376 +- 0.0038** | 0.0000 |
| TENT | 0.5485 +- 0.0692 | 0.5235 +- 0.0473 | 0.0477 +- 0.0104 |
| EATA | **0.6447 +- 0.0018** | 0.6315 +- 0.0044 | 0.0000 |
| CoTTA | 0.5720 +- 0.0187 | 0.5754 +- 0.0163 | 0.0138 +- 0.0035 |
| RoTTA | 0.6291 +- 0.0022 | 0.6306 +- 0.0041 | 0.0397 +- 0.0072 |
| LAME | 0.6196 +- 0.0011 | 0.6375 +- 0.0034 | 0.0000 |
| T2A | 0.5485 +- 0.0146 | 0.4779 +- 0.0673 | 0.1431 +- 0.0690 |

### CAIDBench-UFD 代理轨道

| Method | Online | Final | Forget |
| --- | ---: | ---: | ---: |
| Source | **0.8506 +- 0.0010** | **0.8431 +- 0.0055** | 0.0000 |
| TENT | 0.6650 +- 0.0206 | 0.6263 +- 0.0341 | 0.0968 +- 0.0512 |
| EATA | 0.8431 +- 0.0028 | 0.8177 +- 0.0058 | 0.0000 |
| CoTTA | 0.6361 +- 0.0276 | 0.5070 +- 0.0151 | 0.1159 +- 0.0359 |
| RoTTA | 0.8435 +- 0.0012 | 0.7320 +- 0.0079 | 0.1210 +- 0.0079 |
| LAME | 0.8166 +- 0.0021 | 0.8258 +- 0.0056 | 0.0000 |
| T2A | 0.8099 +- 0.0170 | 0.6668 +- 0.0794 | 0.1088 +- 0.0693 |

LAME 的零 forgetting 是无跨 batch 参数状态的结构性结果，不应解释成记忆能力。
EATA 在这两个 checkpoint 上更新较保守，在线结果稳定，但没有稳定提高最终宏平均。
RoTTA 在 ProGAN-like 在线阶段接近 Source，最终 holdout 下降则显示出明显历史域遗忘。

## 4. IAPL official track

固定官方 commit `a173e7783bbafaa00d60e6e31774a0bc14411a23`，只应用仓库内批准的
CLIP 路径、PyTorch checkpoint 加载和 Arrow 数据适配补丁。ModelScope 的 SD1.4/ProGAN 权重和 OpenAI CLIP
ViT-L/14 已按发布 SHA-256 校验；CAIDBench 的 8 个 GenImage test 域按原始编码字节
无损导出，每域 1000 real + 1000 fake，并保存逐样本 manifest 和 SHA-256。这里是按
生成器名称映射的 CAIDBench 评估，不等同于作者原始 GenImage 文件集合。

官方冒烟和正式全量链路均已完成。单 GPU DDP、32 views、2 TTA steps、OIS、
逐图 prompt/optimizer reset 和 Adapt-Then-Predict 均保持不变。8 域按域拆到
4090-1/4090-2 各 4 域。后续审计发现 Conditional Information Learner 的 BatchNorm
running buffers 不会逐图恢复，因此这种分域执行会重置 buffers，不能再声称与作者整表
单次分布式运行具有完全相同的状态轨迹。

| Domain | Accuracy | AP |
| --- | ---: | ---: |
| ADM | 86.90% | 98.89% |
| BigGAN | 96.45% | 94.09% |
| glide | 95.90% | 99.46% |
| Midjourney | 95.65% | 99.04% |
| stable_diffusion_v_1_4 | 100.00% | 100.00% |
| stable_diffusion_v_1_5 | 50.45% | 54.73% |
| VQDM | 99.10% | 99.65% |
| wukong | 99.70% | 99.84% |
| **Mean** | **90.52%** | **93.21%** |

合并结果没有通过作者 `96.7% / 99.5%`、容差 1% 的门禁，绝对差分别为
6.18 和 6.29 个百分点，因此状态是 `reference_gate_failed`，不能写成论文数值复现成功。
异常主要集中在 CAIDBench 的 SD1.5；去掉该域后的七域均值约为 96.24% / 98.71%，
已进入相同容差。进一步核对 Arrow 元数据后确认：CAIDBench `SD1.5` 来自 DFLIP，
`BigGAN` 来自 ForenSynths，而不是作者 GenImage 中的同名目录；另一个
`StableDiffusion-v1-5-local` 候选用每类 100 张诊断也只有 50.00% / 55.25%。因此这次
门禁失败首先是数据源不等价，不能用于否定 IAPL，也不能写成作者 GenImage 数值复现。

精确 UniversalFakeDetect 官方 19 域轨道不能由 CAIDBench 的 10 个聚合域替代；
项目仍不会把 proxy 数值写成作者 benchmark 复现。

原始 UFD 19 域 Arrow 轨道已于 2026-07-17 完成。ForenSynths 11 域与 Ojha 8 域
共 88,353 张；路径到 Arrow 行号、真假标签及首张图片字节已在两台机器分别校验。
4090-1 完成 9 域、44,967 张，4090-2 完成 10 域、43,386 张，所有域均独立重置。

| Domain | Accuracy | AP | Real Acc. | Fake Acc. |
| --- | ---: | ---: | ---: | ---: |
| BigGAN | 98.75% | 97.91% | 97.75% | 99.75% |
| CRN | 92.31% | 92.47% | 84.61% | 100.00% |
| CycleGAN | 98.75% | 98.00% | 97.50% | 100.00% |
| DALL-E | 99.05% | 99.58% | 99.60% | 98.50% |
| DeepFake | 95.99% | 97.20% | 99.00% | 92.96% |
| GauGAN | 99.42% | 99.17% | 98.86% | 99.98% |
| GLIDE 100/10 | 98.20% | 99.50% | 99.60% | 96.80% |
| GLIDE 100/27 | 97.75% | 98.98% | 99.60% | 95.90% |
| GLIDE 50/27 | 98.25% | 99.30% | 99.60% | 96.90% |
| Guided | 72.50% | 94.86% | 98.60% | 46.40% |
| IMLE | 92.37% | 92.45% | 84.74% | 100.00% |
| LDM 100 | 99.20% | 99.35% | 99.60% | 98.80% |
| LDM 200 | 99.45% | 99.46% | 99.60% | 99.30% |
| LDM 200 cfg | 97.45% | 99.17% | 99.60% | 95.30% |
| ProGAN | 99.99% | 100.00% | 100.00% | 99.98% |
| SAN | 92.47% | 95.92% | 96.35% | 88.58% |
| SeeingDark | 89.44% | 88.30% | 78.89% | 100.00% |
| StarGAN | 96.67% | 95.69% | 93.35% | 100.00% |
| StyleGAN | 94.71% | 99.67% | 99.98% | 89.43% |
| **Mean** | **95.41%** | **97.21%** | **96.15%** | **94.66%** |

作者参考值为 Accuracy 95.61%、AP 99.32%。Accuracy 绝对差 0.20 个百分点，通过
1 个百分点容差；AP 绝对差 2.11 个百分点，未通过。因此完整评估状态是
`reference_gate_failed`：19 域工程链路已跑完，但不能表述为论文数值复现成功。
主要异常是 `guided` 的 fake accuracy 46.40%，以及 `seeingdark`、`crn`、`imle`
的 AP 明显偏低。

本次按域使用单 GPU，保持 32 views、2 TTA steps、OIS、逐图 reset 和
Adapt-Then-Predict；作者发布脚本则以 8 个进程运行，并按 rank 使用不同随机种子。
另外，这份 Arrow 发布的若干域样本数与常见 UFD 目录版不完全相同。两者都可能影响
随机视图及 AP，因而当前结果应称为“已验证 Arrow 版本的完整 19 域复现”，而不是
“作者原始数据与 8 进程随机轨迹的逐项等价复现”。两台机器使用的 ProGAN checkpoint
SHA-256 为 `1e04047b74d287ba2f3682cde84246688dfa486354a5677b5147e677bc2a3f81`，
CLIP ViT-L/14 SHA-256 为
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。
最终合并 JSON 的 SHA-256 为
`a9cbce00a1deb89a174a078eb3e2e3f3c3bb4bdadeba4ac89aa3c7a1756d4deb`。

## 5. Artifacts

- 本地三 seed 汇总：`results/caidbench/genimage/`、`results/caidbench/progan/`。
- 4090-1 原始 GenImage 结果：`/home/yabin/ctta4aid_20260716/outputs/caidbench/genimage/`。
- 4090-2 原始 ProGAN 结果：`/home/yabin/ctta4aid_20260716/outputs/caidbench/progan/`。
- 服务器日志：`/home/yabin/outputs/caidbench/{genimage,progan}/logs/`。
- IAPL 数据 manifest：`/home/yabin/ctta4aid_assets/data/iapl_genimage_shard_{a,b}/`。
- 本地 IAPL 官方日志与合并门禁：`results/iapl/`。
- 本地 UFD Arrow 19 域逐域指标、合并日志与最终门禁：
  `results/iapl/ufd_arrow_dual4090/`。
- 双 4090 原始 UFD Arrow 分域结果：
  `/home/yabin/ctta4aid-arrow/outputs/iapl_official/universalfake_arrow_1gpu/`。
- 3090 自动合并目录：
  `/home/yabin/projects/ctta4aid-arrow/outputs/iapl_official/universalfake_arrow_dual4090/`。

每个 CNN 原始目录保留 `metrics.json`、`online_curve.csv`、`batch_stats.csv`、
`sample_manifest.csv`；连续流额外保留 `holdout_matrix.csv`、
`final_holdout_manifest.csv` 和冻结后的 `effective_config.json`。
