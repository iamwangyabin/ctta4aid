# Online TTA for AI-Generated Image Detection

这是一个用于 AI 生成图像检测的在线测试时适应项目。论文专项只纳入有作者公开实现的方法，当前保留四条方法轨道，并将 `matched_jpeg` 固定为正文 target 输入协议：

- **CLIP ViT-L/14 论文主实验**：唯一预训练模型固定为 OpenAI CLIP ViT-L/14，正式 target 输入固定为 `matched_jpeg`，在 GenImage、AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID Global 上按方法原生训练及适配方式比较通用 TTA、CLIP-native 与任务专用方法，只做接入 ViT-L/14 和二分类数据所必需的最小修改。
- **Controlled CTTA 补充实验**：Source、TENT、EATA、CoTTA、RoTTA、LAME 和 T2A 共用同一个 ResNet-50 源模型，保留为 CNN 对照与补充材料。
- **IAPL 独立能力**：从同一 CLIP ViT-L/14 预训练底座按 IAPL 原生源训练流程得到任务 checkpoint，再按逐图 Adapt-Then-Predict 协议运行；主表中必须显式标出其 source setup 与未做任务训练的 zero-shot CLIP state 不同。
- **OST 补充实验**：使用作者的 MetaXception、AM-Softmax 和单步 fast weights；每张测试图从源训练集抽取带标签模板，合成伪样本后 Adapt-Then-Predict。

已确认的 ResNet-50 结果仍按独立目录保存在 `results/`，只提交最终汇总、复现身份和结论，不提交运行日志或中间产物。新的 CLIP 正文结果必须写入明确标识 `matched_jpeg` 的 `results/clip_vlm_bias_controlled_*` 目录，绝不覆盖这些补充材料。

## 项目结构

```text
configs/
  datasets/       数据集、checkpoint 和目标域
  protocols/      Single-target 与 Continual 协议
  methods/        各方法参数
  experiments/    CLIP 主实验、Controlled CTTA、IAPL 与 OST 实验入口
  train/          公共源模型训练配置
src/
  cli/            公共命令行辅助逻辑
  data/           数据集、变换和数据流
  evaluation/     指标与在线评估
  methods/        统一 TTA 方法接口
  models/         检测器及 IAPL、OST loader
  official/       固定版本的第三方算法核心
results/          仅保存重新完成后的最终实验结果
scripts/          数据检查和必要下载脚本
tests/            配置、协议、数据与方法测试
```

配置采用组合方式：`datasets/`、`protocols/` 和 `methods/` 分别描述独立职责，`experiments/` 只负责组合它们并指定 seed 与输出目录。固定的第三方来源和 commit 记录在 `configs/official_sources.yaml`。

## 数据格式

框架只接受 `data.format: arrow`，不直接读取 ImageFolder、ZIP、Parquet 或其他 Arrow 组织方式。这里的 `arrow` 是本项目固定的数据契约，而不是任意 Apache Arrow 文件：

- 根目录可以是一个 Hugging Face `Dataset.save_to_disk` bundle，也可以按
  `data/<split>/<generator>/` 递归包含多个 bundle；
- 必须包含 `mapping.json`，记录 `image_path -> row index`；
- 数据列必须包含原始编码字节 `image` 和逻辑路径 `image_path`；
- 逻辑路径必须包含 generator 和 `0_real`/`1_fake` 或 `nature`/`ai` 标签目录；split 由路径组件或 `<split>.json` 索引明确给出。

原始数据只作为离线转换输入，不进入训练、适应或评估调用链。所有数据集都必须先转换并校验为上述统一 Arrow bundle。

## 安装

需要 Python 3.10+ 和 PyTorch 2.2+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLIP ViT-L/14 论文主实验

主实验唯一预训练模型固定为 OpenAI CLIP ViT-L/14，本地 checkpoint 的 SHA-256 为
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。这里统一的是
预训练底座、目标样本、顺序、seed 和 evaluator，不是强迫所有方法共享一个分类头、
一句固定 prompt 或相同的测试前任务状态。每个方法保留论文原生的 source training、
batch/views、状态转移、预测/适应顺序和 prompt 构造，只做接入固定 ViT-L/14 与二分类
数据所必需的最小修改。

正文的正式 target 输入采用 `matched_jpeg` profile：real 与 fake 都执行相同的
256x256 center-crop/resize，再从 75/80/85/90/95 中按不含类别目录的逻辑路径
确定性选择 JPEG 质量。原始编码与 `all_jpeg_q90` 只作为独立补充审计，不能与
`matched_jpeg` 主结果混表。四个数据集的逐 target AUC 与 Accuracy 详细表均保留在正文；
原始 16-profile campaign 已验收并冻结在
`results/clip_vlm_bias_controlled_matched_jpeg_20260902/`；加入独立
AIGI-Det-Calib baseline 后、包含 17 种运行配置的完整验收版本保存在
`results/clip_vlm_bias_controlled_matched_jpeg_aigi_det_calib_20260903/`，
不会回写或覆盖前一目录。正文从该完整结果中报告 15 行；其中
`frozen_clip` 和 `ours_static` 各自的 117 个 `target x seed` 单元分别只作为补充诊断和
冻结源状态对照保留，不进入正文主表。

三类 source setup 仍按下表完整披露，但不再作为八张正文表中的重复分组标题：

| 区块 | 起点 | 方法 | 比较规则 |
|---|---|---|---|
| 公共源域 CLIP detector | 固定 ViT-L/14 初始化后，在同一源数据上训练的公共二分类 checkpoint | Source、AIGI-Det-Calib、TENT、EATA、SAR、CoTTA、RoTTA-LN、LAME、T2A | 共享源 checkpoint，可作严格配对比较 |
| CLIP-native adaptation | 未做任务微调的固定 ViT-L/14 checkpoint | TDA、DynaPrompt、CLIPTTA、BATCLIP | 共享二分类类别语义，各自保留原生 template、文本分类器或 prompt learner |
| Method-specific source training | 固定 ViT-L/14 初始化后，按方法自己的源训练流程得到的 checkpoint | IAPL、Ours | source state 不同，跨方法数值只作描述性比较 |

正文表将 Source 至 IAPL 连续列为外部基线，只在最终 `Ours` 前保留一条横线。内部
`ours_static` 运行不是独立方法，只在消融与配对分析中以 `Frozen source control` 出现。
每列在全部正文方法中将最佳结果加粗、
次佳的不同结果加下划线；按
表中报告到小数点后两位的数值确定名次，同精度并列时共享标记。该排名只用于描述结果，
严格的适应增益仍只从相同 source state 的配对结果推断。

`frozen_clip` 仍保留配置、实现和完整结果，用固定 real/fake 文本原型衡量预训练
CLIP 中偶然存在的零样本语义信号。它没有任务检测器训练，也没有测试时适应机制，且其
prompt ensemble 不是各 CLIP-native 方法的严格配对静态版本。因此它只在补充材料中说明，
不作为正文方法行，也不参与任何最佳结果排名。

各方法的冻结运行约定如下。`BN -> LN` 只表示把公开实现中的归一化参数枚举映射到
CLIP LayerNorm affine；不得改写目标函数、筛选规则、teacher、Fisher、gradient masking
或在线状态。

| Method | 判别能力来源 | 保留的原生机制 | ViT-L/14 必要改动与状态 |
|---|---|---|---|
| AIGI-Det-Calib | 公共 Source detector 的 logits | 作者的无标签 scalar offset | 固定作者 commit；每个 target 前 100 个锁定样本只估计 offset 且保留 Source prediction，后 1,400 个样本使用固定 offset；主表报告完整因果 prequential 结果 |
| TENT | 公共源域二分类 detector | 熵最小化及原生在线顺序 | `BN -> LN`，表格加脚注 |
| EATA | 公共源域二分类 detector | 可靠/非冗余筛选、熵最小化、Fisher 防遗忘 | `BN -> LN`；Fisher 必须由同一源 checkpoint 与源数据计算 |
| SAR | 公共源域二分类 detector | 可靠样本筛选、SAM 与恢复机制 | 使用作者公开的 ViT/LayerNorm 路径 |
| CoTTA | 公共源域二分类 detector | student/EMA teacher、增强平均、随机恢复 | 保留作者全参数更新；只将像素增强的归一化桥接为 CLIP 输入归一化 |
| RoTTA-LN | 公共源域二分类 detector | CSTU memory、teacher/student、EMA 与熵目标 | 显式以 CLIP visual LayerNorm affine 替代 RobustBN；无统计插值等价物；24 GB GPU 上 stream/update microbatch 均为 2，完整 64-sample 加权均值只 step/EMA 一次，表格不得写成原版 RoTTA |
| LAME | 公共源域 detector 的特征与 logits | 参数无关的 Laplacian 输出适配 | 仅接入 CLIP 特征；保留其 batch contract |
| T2A | 公共源域二分类 detector | 不确定性选择、negative learning、gradient masking | 归一化梯度参照由 BN 最小映射为 LayerNorm，其他逻辑不动 |
| IAPL | IAPL 原生源训练得到的 CLIP detector | 32 views、2 steps、OIS、逐图 prompt/optimizer reset | 只替换统一数据接口，不改为公共 binary head |
| TDA | CLIP 原生文本分类器 | 正负 cache、无反向传播 | 严格 `batch_size=1`；custom 数据集每步只输入一个样本的一张确定性 global view，先更新 cache 再预测；仅将作者文本构造桥接到二分类语义 |
| DynaPrompt | CLIP 类别名与在线 context | 多视图 prompt tuning、动态 prompt buffer | 换成 ViT-L/14；不固定最终 prompt |
| CLIPTTA | CLIP 文本原型 | 官方 closed-set 对比适配及 batch 机制 | 使用二分类类别语义与作者原生文本构造 |
| BATCLIP | CLIP 图像与文本两端 | 双模态目标及原生在线更新 | 换成固定 ViT-L/14；不得退化成只更新视觉端 |
| Ours | rank-4 LoRA 二分类 CLIP detector 与 source score anchors | BIC/GMM 分段与伪监督、CLIP 特征专家路由、类条件 Gaussian replay、每专家 residual MLP，以及固定 `0.75` residual shrinkage 和解析截距重拟合 | 最终方法固定为内部版本 R47；严格 Predict-Then-Adapt；同 checkpoint 的 `ours_static` 运行只作 Frozen source control，R37 只作消融 |

所有 target hidden labels 始终只进入 evaluator。CLIP-native 方法的类别语义属于任务定义，
但 template 或 prompt 不得使用目标标签选择。主结果不再把一个数据集压缩成单个数值：
每个数据集分别给出逐 target 的 AUC 表和 Accuracy 表。所有单元格均报告正式验证
seed `0/2/3` 的均值及跨 seed 标准差，其中标准差以较小下标显示；Mean 对每个 seed
先计算 target-macro，再报告三个 seed 的均值及标准差。Accuracy 固定使用 0.5 阈值。
正式结论中的 AUC 和 Accuracy 都以 target-macro 为准。把所有 target 样本混合计算的 pooled
指标只用于诊断域间分数尺度，不用于方法晋级、最佳结果选择或论文主结论。

| 数据集 | 固定 target 列顺序 |
|---|---|
| GenImage | BigGAN、ADM、GLIDE、SD v1.5、VQDM、Wukong、Midjourney |
| AIGCDetectionBenchmark | ProGAN、StyleGAN、BigGAN、CycleGAN、StarGAN、GauGAN、StyleGAN2、WFIR、ADM、GLIDE、Midjourney、SD v1.4、SD v1.5、VQDM、Wukong、DALL-E2、SDXL |
| AIGI-Holmes P3 | Janus、Janus-Pro-1B、Janus-Pro-7B、Show-o、LlamaGen、Infinity、VAR、PixArt-XL、SD3.5-L、FLUX |
| OpenSDID Global | SD1.5、SD2.1、SDXL、SD3、Flux.1 |

汇总器仍可用 `--template-only` 生成八张空白 LaTeX 模板；正式目录中的八张详细表则由
三个已验收 seed 的结果生成，同时保留 CSV、JSON 和 LaTeX：

```bash
python scripts/summarize_clip_vlm_results.py \
  --template-only \
  --output-dir /tmp/clip_vitl14_paper_table
```

公共 detector 的 source checkpoint 由 `configs/train/genimage_sd14_clip_vitl14.yaml`
从完整 GenImage SD v1.4 训练集微调一次：3 epoch、全 visual tower 和二分类 head、
AdamW、CLIP 训练增强，并在相同 source checkpoint 的干净 source holdout 上计算 EATA
所需的 LayerNorm Fisher。这个 checkpoint 在全部三个正式 online seed `0/2/3` 和所有公共
detector 方法之间共享。每种方法的分类器或 prompt 构造、可训练参数、batch/views、更新步数、
状态重置和预测/适应顺序均由其方法配置锁定。`configs/experiments/clip_vlm/` 提供这些
方法原生基础配置，正文正式运行入口是
`configs/experiments/clip_vlm_bias_controlled/matched_jpeg_<dataset>_seed<seed>.yaml`。
TDA 另使用
`configs/experiments/clip_vlm_bias_controlled/matched_jpeg_tda_<dataset>_seed<seed>.yaml`
做严格 batch-one 复跑，并独立写入 `matched_jpeg/tda_batch1/`。早期完整 campaign 中将
16 个不同 target 样本当作 TDA 多视图 batch 的数值只保留作审计，不再进入正文；只有这组
`batch_size=1`、单样本单 global view 的 seed `0/2/3` 结果可替换正文 TDA 行。
新增验证 seed 3 的锁定样本身份与顺序保存在
`configs/datasets/manifests/clip_vlm/`；生成时先逐数据集复现既有 seed-0 manifest，确认
采样与 DataLoader 顺序完全一致后再冻结 seed-3 manifest。
训练配置还固定排除了 preflight 检出的三条零字节 SD v1.4 源图逻辑路径；不会用空白或
合成像素替代损坏样本。
三个正式 seed 的 1,989 个完整 campaign `method x target x seed` 单元均已通过样本身份、输入
profile、checkpoint、方法协议和有限指标检查；排除 117 个补充 Frozen CLIP 诊断单元和
117 个 `ours_static` 冻结源状态对照单元后，八张正文表报告其余 1,755 个单元。此前完成的
ResNet-50 数值表仍原样保留在论文补充材料中。RoTTA-LN 数值来自独立运行并保留必要迁移
说明。AIGI-Det-Calib 的表格行只使用公共 `source_ft` checkpoint 上的严格无标签因果结果；
`ours_static + AIGI-Det-Calib` 仅保存在校验汇总中作诊断。TTC 因没有可固定的作者公开实现而
从定量表删除，只在 related work 中讨论。

最终结果、校准指标、读出消融、动态复现流、效率统计、协议审计和 source/checkpoint 身份均
保存在 `results/clip_vlm_bias_controlled_matched_jpeg_aigi_det_calib_20260903/`。以下第一条命令
可从完整运行目录重新生成原始 16-method 汇总；第二条命令把已验收的 AIGI-Det-Calib 正式
campaign 作为独立 baseline 加入新目录：

```bash
python scripts/summarize_clip_vlm_results.py \
  --dataset genimage=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/genimage \
  --dataset aigc_detection_benchmark=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/aigc_detection_benchmark \
  --dataset aigi_holmes_p3=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/aigi_holmes_p3 \
  --dataset opensdid_global=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/opensdid_global \
  --output-dir /data/results/clip_vlm_bias_controlled_matched_jpeg_vitl14

python scripts/summarize_clip_vlm_results.py \
  --base-summary /data/results/clip_vlm_bias_controlled_matched_jpeg_vitl14/clip_vitl14_summary.json \
  --aigi-det-calib-results /data/experiments/aigi_det_calib/runs \
  --output-dir /data/results/clip_vlm_bias_controlled_matched_jpeg_with_aigi
```

## Matched-JPEG 正文协议与编码审计

`matched_jpeg` 是 CLIP ViT-L/14 正文主实验的正式输入协议。原始编码的 CLIP campaign
不覆盖、不改名，降为补充对照；`all_jpeg_q90` 也是独立敏感性审计。各 profile 只替换
target Arrow 中的图像字节，模型 checkpoint、source setup、方法原生配置、锁定样本身份与
顺序、阈值和 evaluator 都继承 `configs/experiments/clip_vlm/`。正文 `matched_jpeg`
固定使用验证 seed `0/2/3`；seed 1 仅保留为开发运行，不进入正文正式均值。

| Profile | 所有 real/fake 共同处理 | 用途 |
|---|---|---|
| `all_jpeg_q90` | EXIF orientation、RGB、JPEG Q90；保留视觉尺寸 | 单一质量的编码格式敏感性审计 |
| `matched_jpeg` | EXIF orientation、RGB、center-crop/resize 到 256x256；从 75/80/85/90/95 中按不含类别目录的逻辑路径确定性取值 | 正文主实验的编码与几何匹配协议 |

两套处理都不能只压缩 fake。`matched_jpeg` 的质量选择不读取二分类标签；同名 real/fake
逻辑样本会得到相同质量。原始 JPEG 会被再次编码，因此仍可能有 double-compression，
因此正文将它准确描述为编码与几何匹配协议，不声称所有数据偏差已经消失。

每个原始 Arrow 根分别离线转换，输出路径必须显式包含 profile 名。转换不覆盖输入，并在
根目录和每个 bundle 写入 `bias_control_manifest.json`：

```bash
python scripts/build_bias_controlled_arrow.py \
  --profile all_jpeg_q90 \
  --input-root /data/arrow/raw/genimage \
  --output-root /data/arrow/bias_controlled/all_jpeg_q90/genimage

python scripts/build_bias_controlled_arrow.py \
  --profile matched_jpeg \
  --input-root /data/arrow/raw/genimage \
  --output-root /data/arrow/bias_controlled/matched_jpeg/genimage

python scripts/check_arrow_datasets.py genimage \
  /data/arrow/bias_controlled/matched_jpeg/genimage \
  --bias-control-profile matched_jpeg
```

四个数据集分别设置独立环境变量。其他三个数据集使用同样的后缀规则：

```bash
export GENIMAGE_ALL_JPEG_Q90_ARROW_ROOT=/data/arrow/bias_controlled/all_jpeg_q90/genimage
export AIGC_DETECTION_BENCHMARK_ALL_JPEG_Q90_ARROW_ROOT=/data/arrow/bias_controlled/all_jpeg_q90/aigc_detection_benchmark
export AIGI_HOLMES_P3_ALL_JPEG_Q90_ARROW_ROOT=/data/arrow/bias_controlled/all_jpeg_q90/aigi_holmes_p3
export OPENSDID_GLOBAL_ALL_JPEG_Q90_ARROW_ROOT=/data/arrow/bias_controlled/all_jpeg_q90/opensdid_global

export GENIMAGE_MATCHED_JPEG_ARROW_ROOT=/data/arrow/bias_controlled/matched_jpeg/genimage
export AIGC_DETECTION_BENCHMARK_MATCHED_JPEG_ARROW_ROOT=/data/arrow/bias_controlled/matched_jpeg/aigc_detection_benchmark
export AIGI_HOLMES_P3_MATCHED_JPEG_ARROW_ROOT=/data/arrow/bias_controlled/matched_jpeg/aigi_holmes_p3
export OPENSDID_GLOBAL_MATCHED_JPEG_ARROW_ROOT=/data/arrow/bias_controlled/matched_jpeg/opensdid_global
```

以 `matched_jpeg`、GenImage seed 0 为例：

```bash
python run_single_target.py \
  --config configs/experiments/clip_vlm_bias_controlled/matched_jpeg_genimage_seed0.yaml
```

运行前会同时校验配置 profile、bundle 规范哈希和输出路径。输出固定写到
`${CTTA4AID_EXPERIMENT_ROOT}/clip_vlm_bias_controlled/<profile>/<dataset>/seed<seed>`；
任何缺失、错配或试图写回原始 `clip_vlm/` 目录的运行都会直接失败。四数据集三正式 seed
已经完整验收，正文八张详细表只填入 `matched_jpeg` 汇总。原始编码与 `all_jpeg_q90`
仍须按 profile 分别汇总并仅作为补充结果。

## Controlled CTTA 补充实验

这条 CNN 补充轨道读取符合项目数据契约的 UniversalFakeDetect Arrow bundle。运行前设置：

```bash
export UFD_FORENSYNTHS_ARROW_ROOT=/data/DF-arrow-data/ForenSynths
export UFD_OJHA_ARROW_ROOT=/data/DF-arrow-data/Ojha
export UFD_SOURCE_CHECKPOINT=/weights/ufd_progan_resnet50.pt
```

训练共同源模型：

```bash
python train_source.py --config configs/train/source.yaml
```

这一步也就是 **T2A 的离线训练阶段**：训练 T2A 随后要适应的源检测器。
T2A 官方发布只包含测试时 adapter，没有另一个可训练的 T2A 专属网络；因此这里不能
为 T2A 私自训练更强 checkpoint。Source、TENT、EATA、CoTTA、RoTTA、LAME 和
T2A 必须读取这次训练产生的同一个 `source.pt`，否则 Controlled CTTA 比较失去公平性。
为了让调度任务可以明确写成 T2A 训练，也提供了不改变训练算法的显式入口：

```bash
python train_source.py --config configs/train/t2a_ufd_source.yaml
python train_source.py --config configs/train/t2a_genimage_source.yaml
```

使用其中任一产物时，必须让对应数据轨道的全部七种方法共享该 `source.pt`，不能只给
T2A 使用。

运行 seed 0：

```bash
python run_single_target.py \
  --config configs/experiments/controlled_ctta/single_target_seed0.yaml
python run_continual_stream.py \
  --config configs/experiments/controlled_ctta/continual_seed0.yaml
```

将配置名中的 `seed0` 换成 `seed1` 或 `seed2` 可运行其余 seed。

Single-target 会为每个 `method x target` 重新加载源 checkpoint。Continual stream 只在每个方法开始时加载一次，域切换时保留方法状态；每个域结束后会在与适应样本不重叠的固定 holdout 上计算当前域收益、过去域 forgetting，以及尚未进入适应流的 future-generator transfer。未来域样本仅用于 evaluator 的只读预测，标签只进入指标计算；二者都不进入方法的适应调用。

GenImage CNN Controlled CTTA 使用 SD v1.4 训练公共 ResNet-50 源模型，并将其从
目标域中排除。七个方法共享该 checkpoint、Fisher、样本顺序和
Predict-Then-Adapt 协议；IAPL 仍作为独立的 CLIP 补充轨道汇报。设置：

```bash
export GENIMAGE_SD14_TRAIN_ARROW_ROOT=/data/DF-arrow/SDv14_train
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
export GENIMAGE_SOURCE_CHECKPOINT=/outputs/source_train/genimage_sd14_resnet50_arrow/source.pt
```

依次训练源模型并运行 seed 0：

```bash
python train_source.py --config configs/train/genimage_sd14_source.yaml
python run_single_target.py \
  --config configs/experiments/controlled_ctta/genimage_single_target_seed0.yaml
python run_continual_stream.py \
  --config configs/experiments/controlled_ctta/genimage_continual_seed0.yaml
```

实验输出包括 `metrics.json`、`online_curve.csv`、`batch_stats.csv`、`sample_manifest.csv` 和汇总 JSON；Continual 额外输出完整的 `holdout_matrix.csv` 与 `final_holdout_manifest.csv`。完整矩阵将每个 checkpoint 的评估标记为 `past`、`current` 或 `future`，并在 summary 中报告相对同一方法适应前初始状态的 current gain、future transfer 和 future negative-transfer rate。

## 外部基准连续评估

AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID 都只作为 target-only 测试集，重复使用 GenImage SD v1.4 源检测器，不将外部图像用于源训练。原始数据必须先离线转换为项目 Arrow bundle：

```bash
python scripts/prepare_external_arrow.py aigc_detection_benchmark \
  --input-root /data/AIGCDetectionBenchmark/test_set \
  --output-root /data/arrow/aigc_detection_benchmark
python scripts/prepare_external_arrow.py aigi_holmes_p3 \
  --input-root /data/AIGI-Holmes/TestSet.zip \
  --output-root /data/arrow/aigi_holmes_p3
python scripts/prepare_external_arrow.py opensdid_global \
  --input-root /data/OpenSDI_test \
  --output-root /data/arrow/opensdid_global
```

转换默认为每个 generator 和二类标签保留 1,000 张图片；运行时再按 seed 在其中定义 750 张适应样本和 250 张互不重叠的 final holdout。AIGI-Holmes 可直接从 `TestSet.zip` 流式读取入选图像，无需完整解压原始集。每个入选图像都会完整解码；可恢复的截断图像会以其解码像素重编码为 PNG，并在对应 `bundle_manifest.json` 的 `recovered_images` 中披露，无法恢复的图像不会进入 Arrow bundle。三个外部基准的正式 continual seed 配置还会读取已提交的 online 与 final-holdout manifest；只要 Arrow 数据缺少任何样本、图片顺序变化或 batch 划分不同，运行会在首个不一致的 batch 进入方法前报错。OpenSDID 正式设置只使用 global (`entire/`) 操作范围。转换后必须校验：

```bash
python scripts/check_arrow_datasets.py aigc_detection_benchmark /data/arrow/aigc_detection_benchmark
python scripts/check_arrow_datasets.py aigi_holmes_p3 /data/arrow/aigi_holmes_p3
python scripts/check_arrow_datasets.py opensdid_global /data/arrow/opensdid_global
```

设置 Arrow 根目录和已训练源 checkpoint 后，以 AIGCDetectionBenchmark seed 0 为例：

```bash
export GENIMAGE_SOURCE_CHECKPOINT=/outputs/source_train/genimage_sd14_resnet50_arrow/source.pt
export AIGC_DETECTION_BENCHMARK_ARROW_ROOT=/data/arrow/aigc_detection_benchmark
python run_continual_stream.py \
  --config configs/experiments/controlled_ctta/aigc_detection_benchmark_continual_seed0.yaml
```

将配置名替换为 `aigi_holmes_p3_continual_seed{0,1,2}` 或
`opensdid_global_continual_seed{0,1,2}`，并设置对应的
`AIGI_HOLMES_P3_ARROW_ROOT` 或 `OPENSDID_GLOBAL_ARROW_ROOT`，即可运行其余正式流。三 seed 完成后可以汇总 AUC 与准确率：

```bash
python scripts/summarize_continual_results.py \
  --results-root /data/outputs/controlled_ctta/aigc_detection_benchmark/continual \
  --output-dir /data/results/aigc_detection_benchmark
```

## IAPL

IAPL 是独立的逐图协议。它对每张图生成 32 个视图，重置 prompt 和优化器，执行 2 步适应后再预测。BatchNorm buffers 只在同一个目标域内部跨图片保留；切换目标域时重新加载模型，因此各目标结果相互独立。CLIP 主表可报告其数值，但必须将作者任务 checkpoint 标为不同于未做任务训练的 zero-shot CLIP state。

IAPL 与 CLIP VLM 主轨和 CNN Controlled CTTA 共用 `src.data` 中的 dataset factory、domain loader 和样本三元组接口。IAPL 使用全局/局部多视图变换；CLIP VLM 其余方法分别使用单视图或其作者的原生多视图变换。

安装额外依赖：

```bash
pip install -e ".[iapl]"
```

重新运行 IAPL 还需要：

- 作者发布的 IAPL checkpoint；
- OpenAI CLIP ViT-L/14 checkpoint；
- GenImage 或 UniversalFakeDetect 的项目标准 Arrow bundle。

固定 commit `a173e7783bbafaa00d60e6e31774a0bc14411a23` 的 IAPL 模型核心已内置在 `src/official/iapl/`。上游未声明软件许可证，相关来源和再分发状态记录在 `THIRD_PARTY_NOTICES.md`。

GenImage 使用统一的数据级环境变量，不再读取 IAPL 专用原始图片目录：

```bash
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
```

设置对应配置中的环境变量后运行：

```bash
python run_single_target.py --config configs/experiments/iapl/genimage_static.yaml
python run_single_target.py --config configs/experiments/iapl/genimage_views_only.yaml
python run_single_target.py --config configs/experiments/iapl/genimage.yaml
python run_single_target.py --config configs/experiments/iapl/ufd.yaml
```

The three target-only external suites reuse the committed seed-0 online
manifests from the Controlled CTTA evaluation, so IAPL receives the same image
identities and per-domain order without using external images for source
training:

```bash
python run_single_target.py --config configs/experiments/iapl/aigc_detection_benchmark.yaml
python run_single_target.py --config configs/experiments/iapl/aigi_holmes_p3.yaml
python run_single_target.py --config configs/experiments/iapl/opensdid_global.yaml
```

The paired non-adaptive controls use the same locked samples and order, but
evaluate the frozen checkpoint from one global view without OIS or prompt
updates:

```bash
python run_single_target.py --config configs/experiments/iapl/aigc_detection_benchmark_static.yaml
python run_single_target.py --config configs/experiments/iapl/aigi_holmes_p3_static.yaml
python run_single_target.py --config configs/experiments/iapl/opensdid_global_static.yaml
```

GenImage 的三个入口读取同一个作者 checkpoint。`static` 只预测标准全局视图，
`views_only` 使用 32 views 和 OIS 但不更新 prompt，默认入口再加入两步 prompt
adaptation。三者用于拆分多视图选择和参数更新各自带来的收益，不用于与其他 backbone
比较绝对分数。

## Ours

仓库只保留一套 Ours 实现，位于 `src/methods/ours.py`。方法从固定 OpenAI CLIP
ViT-L/14 初始化的 rank-4 LoRA 二分类 source checkpoint 出发；部署时 source detector
完全冻结。对已经到达的无标签测试流，方法依次执行：

1. 在 source score 上用 BIC 选择一维 GMM，并进行因果分段；
2. 用所选专家的等先验 GMM posterior 产生伪监督与连续可靠度；
3. 用冻结 CLIP feature 在当前专家与历史专家之间路由；
4. 为每个专家维护 real/fake 类条件对角 Gaussian 充分统计；
5. 每批生成 128 个 real 与 128 个 fake 伪特征，以平衡 BCE 更新该专家的 residual MLP。

最终方法与唯一保留的 readout 消融共享以上全部状态、路由、监督、replay 和优化轨迹，
二者由同一个 `Ours` 类直接构建，不再通过研究版本类互相继承：

- `ours`：论文最终设置。预测时把 feature-dependent residual 固定缩放为 `0.75`，然后在
  当前已生成的同一批平衡 replay 上，用确定性一维二分法重新拟合一个专家截距；不增加
  optimizer、学习率、replay 或 target 阈值。
- `ours_no_calibrated_readout`：唯一保留的 readout 消融。直接使用完整 residual，不执行缩放和截距重拟合。
- `ours_static`：同一 source checkpoint 的冻结非适应对照，不是第三个方法版本。

`0.75` 来自四数据集 seed-1 开发筛选，方法固定后不得继续调整；seed 0、2 和 3 用于独立
验证，正文汇总器只接受 `seed0/seed2/seed3`。论文正文统一写作 **Ours**，消融表写作
**Ours w/o calibrated readout**；R37/R47 只作为内部追踪编号，不作为两个方法名。

一次性训练并标定 source checkpoint：

```bash
export CLIP_VIT_L14_CHECKPOINT=/weights/ViT-L-14.pt
export GENIMAGE_SD14_TRAIN_ARROW_ROOT=/data/DF-arrow/SDv14_train
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
python train_source.py \
  --config configs/train/genimage_sd14_clip_vitl14_lora_ours.yaml
```

正式运行时设置：

```bash
export CLIP_VIT_L14_CHECKPOINT=/weights/ViT-L-14.pt
export OURS_SOURCE_CHECKPOINT=/outputs/clip_vitl14_lora_ours_source_train/source.pt
python run_single_target.py \
  --config configs/experiments/clip_vlm_bias_controlled/matched_jpeg_ours_genimage_seed0.yaml
```

四个数据集、三个正式 seed `0/2/3` 均有最终设置和 readout 消融的独立入口，输出分别写入
`matched_jpeg/ours_single_target/` 与 `matched_jpeg/ours_no_calibrated_readout_ablation/`。
三 seed 已完成独立审计；正文主表只报告最终 Ours，内部 `ours_static` 运行作为同 checkpoint
的 Frozen source control 保留在读出消融和复现流分析中，读出消融保存在最终结果目录的
`readout_ablation_summary.json`。

为验证 Detect、Route 和历史专家复用确实发生在未知边界的数据流中，补充实验固定使用
GenImage `SD1.5_first -> BigGAN -> ADM -> SD1.5_return`。四个 episode 的名字只属于
evaluator；方法只接收图像，不接收 generator identity、切换点或 label。每个 online episode
从对应正式 seed 的锁定主 manifest 中取每类 224 个样本，每个独立 holdout episode 取每类
112 个样本；online/holdout 以及两段 SD1.5 均互不重叠。比较集合固定为 Frozen source control、
Ours w/o calibrated readout、Ours、CoTTA 和 RoTTA-LN。不同方法保留其原生 batch size，
manifest 因而锁定完全相同的样本身份与全局顺序，但不把 delivery batch 划分强加给方法。
输出同时报告 online AUC/Accuracy、返回域 holdout 恢复、检测延迟、误切分、历史专家路由率
和最终专家数。

```bash
python run_continual_stream.py \
  --config configs/experiments/clip_vlm_bias_controlled/matched_jpeg_recurrence_genimage_seed0.yaml
```

将 `seed0` 换为 `seed2` 或 `seed3` 即得到其余正式运行；结果独立写入
`clip_vlm_bias_controlled/matched_jpeg/genimage_recurrence/seed<seed>`，不进入正文单目标主表。
Detect、Route 和 Update 的因果消融复用完全相同的 R47 实现、checkpoint、manifest 与
固定读出，仅以 `ablation_mode` 关闭一个模块，不新增方法别名。例如：

```bash
python run_continual_stream.py \
  --config configs/experiments/clip_vlm_bias_controlled/matched_jpeg_recurrence_genimage_ablation_no_detect_seed0.yaml
```

另外两项将文件名中的 `no_detect` 换成 `no_route` 或 `no_update`；三个正式 seed 均有锁定
入口，输出位于相应 `seed<seed>/ablations/<mode>` 子目录。

历史 ASCAL、PoundTTA 以及其他 Rxx 研究接口、配置和诊断入口已从当前框架删除；如需审计
早期探索，只通过 Git 历史追溯，不在当前运行注册表中保留兼容别名。

## OST

OST 是独立的逐样本 Adapt-Then-Predict 协议，不加入公共 ResNet-50 的 Controlled CTTA 表。它对每张测试图执行作者论文 Algorithm 1 的核心步骤：从源训练集随机抽取一个带标签模板，生成已知为假的伪样本，用 `{伪样本, 模板}` 的 AM-Softmax loss 做一次 fast-weight 更新，再预测原图。目标 hidden label 始终只进入 evaluator。

固定 commit `1e4518b9e560baf9c5693f13a402fa5d7104190f` 的 MetaXception、内循环优化器和 AM-Softmax 已内置在 `src/official/ost/`。作者发布的推理代码依赖未随仓库提供的 SimSwap 运行时和人脸 landmarks；为让相同目标可用于通用伪造图像，本框架明确改用 full-frame alpha blending。该设置是公开核心的跨任务数据适配，不能当作论文人脸 benchmark 的原数值复现。上游 OST 仓库未声明软件许可证，来源和再分发状态记录在 `THIRD_PARTY_NOTICES.md`。

OST 需要区分两个 checkpoint：作者发布的 `xception_meta.pth` 是元训练初始化，
`train_source.py` 运行作者的一步二阶 support/query 目标后产出本数据轨道的
`ost_meta.pt`。通用图像轨道的合成仍使用前述明确披露的 full-frame alpha blending，
所以它是 OST 目标在本任务上的训练与迁移，不是 FF++ 人脸数值复现。

先设置初始化权重和 Arrow 数据，在 ProGAN 或 SD v1.4 源域上训练：

```bash
export OST_XCEPTION_INITIALIZATION=/weights/xception_meta.pth
export UFD_FORENSYNTHS_ARROW_ROOT=/data/DF-arrow-data/ForenSynths
python train_source.py --config configs/train/ost_ufd_meta.yaml

export GENIMAGE_SD14_TRAIN_ARROW_ROOT=/data/DF-arrow/SDv14_train
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
python train_source.py --config configs/train/ost_genimage_meta.yaml
```

再把训练产物作为 OST 测试 checkpoint：

```bash
export OST_CHECKPOINT=outputs/source_train/ost_ufd_progan_meta/ost_meta.pt
export UFD_FORENSYNTHS_ARROW_ROOT=/data/DF-arrow-data/ForenSynths
export UFD_OJHA_ARROW_ROOT=/data/DF-arrow-data/Ojha
python run_single_target.py --config configs/experiments/ost/ufd.yaml

export OST_CHECKPOINT=outputs/source_train/ost_genimage_sd14_meta/ost_meta.pt
export GENIMAGE_SD14_TEMPLATE_ARROW_ROOT=/data/DF-arrow/SDv14_train_templates_seed0
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
python run_single_target.py --config configs/experiments/ost/genimage_static.yaml
python run_single_target.py --config configs/experiments/ost/genimage.yaml
```

OST uses the same fixed external manifests and the SD v1.4 source-template
bundle for its target-only external runs:

```bash
python run_single_target.py --config configs/experiments/ost/aigc_detection_benchmark.yaml
python run_single_target.py --config configs/experiments/ost/aigi_holmes_p3.yaml
python run_single_target.py --config configs/experiments/ost/opensdid_global.yaml
```

OST 的 `static` 与默认入口加载同一个 MetaXception checkpoint；前者不抽取源模板、
不构造 fast weights，后者执行完整的一步 OST 适应。二者的配对差值才是 OST 的 TTA
收益。

## TTA 动机实验

GenImage 动机实验固定 SD v1.4 为各方法自己的训练来源，并在其余七个生成器上做
逐目标独立评估。T2A 使用公共 ResNet-50 的 Source 作为关闭 TTA 的配对对照；IAPL
比较 `static -> views_only -> full`；OST 比较 `static -> full`。每组内部固定 checkpoint、
目标样本、顺序、seed 和阈值，但不要求不同方法共享 backbone，也不把跨方法绝对分数
解释为公平排名。

本轮 OST 动机配对直接加载作者发布的 `xception_meta.pth`，不把额外 SD v1.4
meta-training 混入 TTA 收益。完整 OST 模式仍按方法要求读取带标签的源训练模板；为
避免复制整个 87GB 训练集，离线固定 seed 0、每类 1000 张的标准 Arrow 子集。Static
模式不读取这些模板。该设置验证的是公开 OST 初始化在通用伪造检测数据上的配对变化，
不是作者人脸 benchmark 的数值复现。

最终报告 AUC、AP、Balanced Accuracy、real/fake accuracy、ECE、Brier score、NLL、
负迁移目标比例以及每样本时间、吞吐和峰值显存。这个实验只回答专用 TTA 是否有效、
收益是否稳定以及代价是什么；公共 checkpoint 下的公平排名仍只由 Controlled CTTA
主实验回答。

公共 ResNet-50 配对组的 seed 0 入口为：

```bash
export CTTA4AID_EXPERIMENT_ROOT=/data/experiments/tta_motivation_genimage_sd14_seed0
python run_single_target.py \
  --config configs/experiments/controlled_ctta/genimage_tta_motivation_seed0.yaml
```

三个方法轨道的正式输出都写入这个仓库外目录；仓库中的 `outputs/`、日志和 PID 不作为
正式结果保存。实验确认后只将汇总表、复现身份和结论导入新的 `results/` 目录。

已完成的 GenImage SD v1.4 seed 0 动机实验见
[`results/tta_motivation_genimage_sd14_seed0/`](results/tta_motivation_genimage_sd14_seed0/README.md)。
该结果显示 IAPL 的主要提升来自多视图与 OIS，而不是两步 prompt 参数更新；TENT、
EATA、T2A 和 OST 的参数适应在这轮配对实验中均未提高宏平均 AUC。

## 静态无 TTA 基线补充实验

UnivFD/Ojha、RINE 和 NPR 已在同一 GenImage SD v1.4 Arrow 源训练集上关闭 JPEG 扰动后
重新训练，并以冻结 checkpoint 在四套正式 `matched_jpeg` target 上完成 seed 0 和 1
评估。两 seed 的逐 target 均值/标准差、固定阈值 Accuracy、source/checkpoint 身份和完整
协议披露见
[`results/clip_vlm_bias_controlled_static_nojpeg_seeds01_20260829/`](results/clip_vlm_bias_controlled_static_nojpeg_seeds01_20260829/README.md)。
该实验是 method-native 静态检测器的两 seed 补充比较，不替代 CLIP 主实验的三 seed
正文表。

## 实验边界

- CLIP 主结果只使用固定 OpenAI CLIP ViT-L/14 预训练权重，并锁定目标样本 identity、目标内顺序和 seed；每种方法保留原生 source training、分类器或 prompt 构造、batch/views、在线状态与预测/适应顺序。每个数据集分别报告逐 target AUC 和阈值 0.5 的 Accuracy；每个 target 均报告三 seed 均值及标准差，Mean 对每个 seed 先计算 target-macro 后再报告跨 seed 均值及标准差。
- TENT、EATA 和 T2A 只允许把公开实现中的 BN 参数参照最小映射到 CLIP visual LayerNorm affine；其余方法逻辑不得重写。CoTTA 保留作者 ImageNet 分支的全参数 student/teacher 更新，只将其像素空间增强桥接到 CLIP 输入归一化。EATA 必须包含匹配公共源域 CLIP detector 的 Fisher。RoTTA-LN 显式以 visual LayerNorm affine 替代 RobustBN，但保留 CSTU、teacher/student EMA、熵目标和在线更新频率；由于没有 RobustBN 统计插值，它只能按迁移版本名称披露。
- CLIP-native 方法只共享类别语义，不共享人为固定的一句最终 prompt。Frozen CLIP 只保留为补充零样本语义诊断，不进入正文表格；IAPL 与 Ours 使用各自原生源训练并单独披露 source setup；TTC 在作者公开实现可固定前仅保留在 related work，不进入定量表。
- CNN Controlled CTTA 方法共享 backbone、源 checkpoint、输入顺序、batch size 和 Predict-Then-Adapt 协议；既有数值表原样保留为补充材料，不得被新 CLIP 空表覆盖或删除。
- OST 使用自己的 MetaXception checkpoint、源训练模板和 Adapt-Then-Predict 协议，只能单独披露；当前通用 alpha 合成结果不等价于作者的人脸实验。
- T2A 使用经过必要运行修复的作者公开核心，不能描述为未经修改的官方实现。
- LAME 核心采用 CC BY-NC-SA 4.0，仅限非商业使用；完整第三方授权见 `THIRD_PARTY_NOTICES.md`。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests train_source.py run_single_target.py \
  run_continual_stream.py scripts
```

代码测试通过只说明实现能够执行，不等同于已经复现论文中的全部官方数值。
