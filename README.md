# Online TTA for AI-Generated Image Detection

这是一个用于 AI 生成图像检测的在线测试时适应项目。论文专项只纳入有作者公开实现的方法，当前保留四条方法轨道，并将 `matched_jpeg` 固定为正文 target 输入协议：

- **CLIP ViT-L/14 论文主实验**：唯一预训练模型固定为 OpenAI CLIP ViT-L/14，正式 target 输入固定为 `matched_jpeg`，在 GenImage、AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID Global 上按方法原生训练及适配方式比较通用 TTA、CLIP-native 与任务专用方法，只做接入 ViT-L/14 和二分类数据所必需的最小修改。
- **Controlled CTTA 补充实验**：Source、TENT、EATA、CoTTA、RoTTA、LAME 和 T2A 共用同一个 ResNet-50 源模型，保留为 CNN 对照与补充材料。
- **IAPL 独立能力**：从同一 CLIP ViT-L/14 预训练底座按 IAPL 原生源训练流程得到任务 checkpoint，再按逐图 Adapt-Then-Predict 协议运行；主表中必须显式标出其 source setup 与 Frozen CLIP 不同。
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
  models/         检测器及 IAPL、OST、PoundNet loader
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
版面压缩在结果完整后再处理。

主表按 source setup 分为三个区块：

| 区块 | 起点 | 方法 | 比较规则 |
|---|---|---|---|
| 公共源域 CLIP detector | 固定 ViT-L/14 初始化后，在同一源数据上训练的公共二分类 checkpoint | Source、TENT、EATA、SAR、CoTTA、RoTTA-LN、LAME、T2A | 共享源 checkpoint，可在块内比较最佳结果 |
| CLIP-native | 未做任务微调的固定 ViT-L/14 checkpoint | Frozen CLIP、TDA、DynaPrompt、CLIPTTA、BATCLIP | 共享二分类类别语义，各自保留原生 template、文本分类器或 prompt learner |
| Method-specific source training | 固定 ViT-L/14 初始化后，按方法自己的源训练流程得到的 checkpoint | IAPL、Ours | source state 不同，只披露数值，不做跨块最佳排名 |

各方法的冻结运行约定如下。`BN -> LN` 只表示把公开实现中的归一化参数枚举映射到
CLIP LayerNorm affine；不得改写目标函数、筛选规则、teacher、Fisher、gradient masking
或在线状态。

| Method | 判别能力来源 | 保留的原生机制 | ViT-L/14 必要改动与状态 |
|---|---|---|---|
| TENT | 公共源域二分类 detector | 熵最小化及原生在线顺序 | `BN -> LN`，表格加脚注 |
| EATA | 公共源域二分类 detector | 可靠/非冗余筛选、熵最小化、Fisher 防遗忘 | `BN -> LN`；Fisher 必须由同一源 checkpoint 与源数据计算 |
| SAR | 公共源域二分类 detector | 可靠样本筛选、SAM 与恢复机制 | 使用作者公开的 ViT/LayerNorm 路径 |
| CoTTA | 公共源域二分类 detector | student/EMA teacher、增强平均、随机恢复 | 保留作者全参数更新；只将像素增强的归一化桥接为 CLIP 输入归一化 |
| RoTTA-LN | 公共源域二分类 detector | CSTU memory、teacher/student、EMA 与熵目标 | 显式以 CLIP visual LayerNorm affine 替代 RobustBN；无统计插值等价物；24 GB GPU 上 stream/update microbatch 均为 2，完整 64-sample 加权均值只 step/EMA 一次，表格不得写成原版 RoTTA |
| LAME | 公共源域 detector 的特征与 logits | 参数无关的 Laplacian 输出适配 | 仅接入 CLIP 特征；保留其 batch contract |
| T2A | 公共源域二分类 detector | 不确定性选择、negative learning、gradient masking | 归一化梯度参照由 BN 最小映射为 LayerNorm，其他逻辑不动 |
| IAPL | IAPL 原生源训练得到的 CLIP detector | 32 views、2 steps、OIS、逐图 prompt/optimizer reset | 只替换统一数据接口，不改为公共 binary head |
| TDA | CLIP 原生文本分类器 | 正负 cache、无反向传播 | 使用二分类类别语义与作者的 template 构造 |
| DynaPrompt | CLIP 类别名与在线 context | 多视图 prompt tuning、动态 prompt buffer | 换成 ViT-L/14；不固定最终 prompt |
| CLIPTTA | CLIP 文本原型 | 官方 closed-set 对比适配及 batch 机制 | 使用二分类类别语义与作者原生文本构造 |
| BATCLIP | CLIP 图像与文本两端 | 双模态目标及原生在线更新 | 换成固定 ViT-L/14；不得退化成只更新视觉端 |
| Ours | PoundNet 的配对真实/伪造 prompt detector | 冻结语义路由、延迟残差记忆、条件原型与受保护低秩适配 | 严格 Predict-Then-Adapt；同时报告同 checkpoint 的 Ours-Static |

所有 target hidden labels 始终只进入 evaluator。CLIP-native 方法的类别语义属于任务定义，
但 template 或 prompt 不得使用目标标签选择。主结果不再把一个数据集压缩成单个数值：
每个数据集分别给出逐 target 的 AUC 表和 Accuracy 表。target 单元格报告三个正式 seed
的均值，Mean 列报告 target-macro 均值及跨 seed 标准差；Accuracy 固定使用 0.5 阈值。
正式结论中的 AUC 和 Accuracy 都以 target-macro 为准。把所有 target 样本混合计算的 pooled
指标只用于诊断域间分数尺度，不用于方法晋级、最佳结果选择或论文主结论。

| 数据集 | 固定 target 列顺序 |
|---|---|
| GenImage | BigGAN、ADM、GLIDE、SD v1.5、VQDM、Wukong、Midjourney |
| AIGCDetectionBenchmark | ProGAN、StyleGAN、BigGAN、CycleGAN、StarGAN、GauGAN、StyleGAN2、WFIR、ADM、GLIDE、Midjourney、SD v1.4、SD v1.5、VQDM、Wukong、DALL-E2、SDXL |
| AIGI-Holmes P3 | Janus、Janus-Pro-1B、Janus-Pro-7B、Show-o、LlamaGen、Infinity、VAR、PixArt-XL、SD3.5-L、FLUX |
| OpenSDID Global | SD1.5、SD2.1、SDXL、SD3、Flux.1 |

可先生成八张数值为空的 LaTeX 详细表；全量运行完成后对同一命令去掉
`--template-only`，汇总器会自动填入逐 target 数值，并导出 CSV、JSON 和 LaTeX：

```bash
python scripts/summarize_clip_vlm_results.py \
  --template-only \
  --output-dir /tmp/clip_vitl14_paper_table
```

公共 detector 的 source checkpoint 由 `configs/train/genimage_sd14_clip_vitl14.yaml`
从完整 GenImage SD v1.4 训练集微调一次：3 epoch、全 visual tower 和二分类 head、
AdamW、CLIP 训练增强，并在相同 source checkpoint 的干净 source holdout 上计算 EATA
所需的 LayerNorm Fisher。这个 checkpoint 在全部三个 online seed 和所有公共 detector
方法之间共享。每种方法的分类器或 prompt 构造、可训练参数、batch/views、更新步数、
状态重置和预测/适应顺序均由其方法配置锁定。`configs/experiments/clip_vlm/` 提供这些
方法原生基础配置，正文正式运行入口是
`configs/experiments/clip_vlm_bias_controlled/matched_jpeg_<dataset>_seed<seed>.yaml`。
训练配置还固定排除了 preflight 检出的三条零字节 SD v1.4 源图逻辑路径；不会用空白或
合成像素替代损坏样本。
八张新表在完整三 seed campaign 验收前仍全部保持空白；此前完成的 ResNet-50 数值表原样
保留在论文补充材料中。RoTTA-LN 只有在独立运行并通过结果身份核验后才可填值；TTC
因没有可固定的作者公开实现而从定量表删除，只在 related work 中讨论。

四个数据集的三 seed 全部完成后，写入最终结果目录并生成论文主表：

```bash
python scripts/summarize_clip_vlm_results.py \
  --dataset genimage=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/genimage \
  --dataset aigc_detection_benchmark=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/aigc_detection_benchmark \
  --dataset aigi_holmes_p3=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/aigi_holmes_p3 \
  --dataset opensdid_global=/data/experiments/clip_vlm_bias_controlled/matched_jpeg/opensdid_global \
  --output-dir /data/results/clip_vlm_bias_controlled_matched_jpeg_vitl14
```

## Matched-JPEG 正文协议与编码审计

`matched_jpeg` 是 CLIP ViT-L/14 正文主实验的正式输入协议。原始编码的 CLIP campaign
不覆盖、不改名，降为补充对照；`all_jpeg_q90` 也是独立敏感性审计。各 profile 只替换
target Arrow 中的图像字节，模型 checkpoint、source setup、方法原生配置、锁定样本身份与
顺序、三个 seed、阈值和 evaluator 都继承 `configs/experiments/clip_vlm/`。

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
任何缺失、错配或试图写回原始 `clip_vlm/` 目录的运行都会直接失败。四数据集三 seed
完整验收前，正文八张详细表仍保持空白；验收后只填入 `matched_jpeg` 汇总。原始编码与
`all_jpeg_q90` 必须按 profile 分别汇总并仅作为补充结果。

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

IAPL 是独立的逐图协议。它对每张图生成 32 个视图，重置 prompt 和优化器，执行 2 步适应后再预测。BatchNorm buffers 只在同一个目标域内部跨图片保留；切换目标域时重新加载模型，因此各目标结果相互独立。CLIP 主表可报告其数值，但必须将作者任务 checkpoint 标为不同于 Frozen CLIP 的 source setup。

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

## PoundTTA（Ours）

PoundTTA 使用已训练 PoundNet 的成对真实/伪造 prompts。对每个语义类别，两个单位文本
特征的中点作为冻结语义索引，其差向量作为真伪方向；测试时只在这些真伪方向张成的
残差空间中建立记忆和更新低秩 adapter，不更新 CLIP backbone 或 PoundNet prompts。
新样本必须先由旧状态完成预测，之后才经过原图、水平翻转和尺度视图投票进入候选队列；
候选还需经过至少一个 batch 的延迟确认才能进入按 real/fake 预留容量的稳定记忆。只有
双类记忆支持、语义覆盖、残差证据和适应需求同时充分时才启用校正，否则输出严格退回
PoundNet source prediction。

正式配置总是成对运行 `ours_static` 与 `ours`。前者加载完全相同的 PoundNet 和 OpenAI
CLIP ViT-L/14 checkpoint，但关闭候选队列、记忆和 adapter；两行的差值才表示在线适应
收益。设置：

```bash
export CLIP_VIT_L14_CHECKPOINT=/weights/ViT-L-14.pt
export POUNDNET_CHECKPOINT=/weights/poundnet_ViTL_Progan_20240506_23_30_25.ckpt
python run_single_target.py \
  --config configs/experiments/clip_vlm/genimage_seed0.yaml
```

当前主版本只使用 PoundNet 内部的 CLIP 语义类别库，不调用额外 captioner。目标标签、
generator identity 和未来样本都不会进入候选筛选、记忆、门控或 adapter loss。完整三 seed
结果验收前，论文表格仍不得填入 Ours 数值。

## ASCAL（锚定分数校准，独立新方法）

ASCAL 是与 PoundTTA 并存的独立方法轨道：base 为固定 OpenAI CLIP ViT-L/14 + 视觉
MLP 投影上的小 rank LoRA + 二分类 head（注意力 `out_proj` 不注入 LoRA，因为
`nn.MultiheadAttention` 走 functional 路径直接消费其权重）。部署时模型参数完全冻结，
方法只在线维护真/假分数分布的锚定记忆：real 单高斯、fake 多分量 GMM；窗口更新永远以
source 锚点为 MAP 先验，并由双峰性、入库率和 3σ 漂移保险丝守门。`ascal_static` 与
`ascal` 加载同一 checkpoint 成对运行，前者关闭入库与更新，两行差值才是在线适应收益。

训练与标定（一次离线完成，温度、锚点、入库阈值全部写入 checkpoint 的
`score_anchors`；缺失锚点的 checkpoint 会被拒绝启动）：

```bash
export CLIP_VIT_L14_CHECKPOINT=/weights/ViT-L-14.pt
export GENIMAGE_SD14_TRAIN_ARROW_ROOT=/data/DF-arrow/SDv14_train
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
python train_source.py --config configs/train/genimage_sd14_clip_vitl14_lora_ascal.yaml
```

运行时把产物路径暴露给方法配置（`configs/methods/ascal.yaml`），再以任意
single-target 或 continual 实验配置调用 `ascal_static` / `ascal`：

```bash
export ASCAL_LORA_SOURCE_CHECKPOINT=/outputs/clip_vitl14_lora_ascal_source_train/source.pt
```

完整三 seed 结果验收前，论文表格不得填入 ASCAL 数值；其 source setup 属于
method-specific source training 区块，不与公共源域 detector 区块跨块比较。

### ASCAL-GMM（简化的非对称在线校准）

`ascal_gmm` 是在同一 LoRA source detector 上独立保留的简化方案，不改写上面的原始
ASCAL 消融。它使用单个标准 CLIP 视图，并把每个已经到达的 target 原始分数全部加入
一维 GMM；没有置信度入库阈值、窗口、类别容量、连续通过次数、漂移门控或融合系数。
每次预测严格使用前一批结束后得到的状态，当前 batch 只在预测完成后进入拟合。

分量数由 BIC 在 `1..(1 + K_source_fake)` 中自动选择，其中 source fake-GMM 的分量数
只提供一个不读取 target label 的复杂度上限。BIC 只选出一个分量时，证据不足，输出
严格退回 source 温度校准概率；选出两个或更多分量时，按均值排序，将最低均值分量视为
单峰 real，其余所有分量的责任度相加作为 fake 概率。因此方法会自然从“一块 real +
一块 fake”开始，只有无标签分数本身支持时 fake 才增加更多模式。source 锚点只用于
温度回退、分数方向确认和上述复杂度上限，不参与 target MAP 锁定。

`ascal_gmm_static` 是同 checkpoint、同单视图输入的精确非适应对照。当前用于诊断的
`matched_jpeg` seed1 入口位于
`configs/experiments/clip_vlm_bias_controlled/matched_jpeg_ascal_gmm_*_seed1.yaml`；它们
不是完整三 seed 正式结果，不能回填论文主表。

`ascal_gmm_shift` 进一步把 GMM 限制为一个在线阈值估计器，不再直接输出 mixture
posterior。BIC 仍拟合全部历史无标签分数，但按相邻分量均值的最大间隔把低分连续块视为
real、高分连续块视为 fake，并取该间隔两端均值的中点作为决策边界。这样 real 的单个
紧凑分数块可以由多个高斯近似，而不会把其第二个分量强行判成 fake。最终输出固定为
`sigmoid((source_score - target_boundary) / source_temperature)`，所以每个因果状态内严格
保留 source score 排序；BIC 只有一个分量时仍精确退回 source。该变体同样没有新增窗口、
阈值或融合超参数，并与 `ascal_gmm_shift_static` 成对运行。seed1 诊断入口位于
`matched_jpeg_ascal_gmm_shift_*_seed1.yaml`，完整三 seed 验收前仍不得进入正文主表。

`ascal_gmm_median_shift` 保留同一个最大间隔候选边界，但不允许最新一次拟合直接替换部署
边界。每次多分量 GMM 拟合成功后，只把因果候选边界追加到历史中；下一批预测使用全部
历史候选边界的累计中位数。它不需要 EMA 系数、稳定窗口或截断阈值，并能抑制 BIC 分量
重排造成的单批边界跳变。当前 GMM 退回单分量时仍精确使用 source 输出。该变体与
`ascal_gmm_median_shift_static` 成对运行，seed1 诊断入口位于
`matched_jpeg_ascal_gmm_median_shift_*_seed1.yaml`，同样不能在三 seed 验收前进入正文
主表。

`ascal_gmm_density_shift` 是不使用语义特征的边界规则消融。它保留最大间隔给出的低分
real 块与高分 fake 块划分，也保留候选边界的累计中位数，但分别归一化两侧 GMM 权重，
以等先验 real/fake 条件密度在最大间隔内的交点替代分量均值中点。若间隔内不存在交点，
则确定性退回原最大间隔中点；BIC 只选出一个分量时仍精确退回 source。该变体不引入
target 阈值、窗口、融合系数或语义输入，并与 `ascal_gmm_density_shift_static` 成对运行；
seed1 诊断入口为 `matched_jpeg_ascal_gmm_density_shift_*_seed1.yaml`。

`ascal_gmm_segmented_shift` 在中位数边界版本上加入无标签的因果分段，专门处理 continual
stream 中旧域分数长期拖住新域边界的问题。每批仍先用旧状态预测；预测后，方法用 BIC
比较“当前整段一个 GMM”和“一个因果后缀段 GMM”。后缀由固定二进制调度在 2、4、8、
... 批尺度间选择：2 批最频繁，更长尺度周期性检查，每个偶数批只拟合一个尺度。单个
batch 只是一次时间观测，至少连续两批才构成最小的持续变化证据；分段模型还要支付一个
变点参数的 BIC 惩罚，只有总 BIC 严格更低才触发，因此没有可调窗口长度或人为漂移阈值。
候选新段还必须由 BIC 选出至少两个分量，且自身不支持再次二分；单分量仍按原规则视为
证据不足，避免把局部类别波动或一个尚未稳定的过渡批次误当成完整新域。
触发后只丢弃旧段的分数与边界历史，冻结 detector、LoRA 和 source 温度均不重置，也不
读取 target label、生成器边界或语义特征。single-target seed1 入口为
`matched_jpeg_ascal_gmm_segmented_shift_*_seed1.yaml`；与未分段中位数版本的 continual
对照入口为 `matched_jpeg_ascal_gmm_segmented_shift_continual_*_seed1.yaml`，两者均属于
诊断实验，完整三 seed 验收前不能进入正文主表。

`ascal_gmm_segmented_handoff_shift` 保留上述 BIC 分段判据，只改变变点后的边界交接。
触发分段时，方法记录变点前最后一次实际输出的边界作为锚点；新段得到第 `j` 个有效
GMM 边界后，以 `1/j` 的锚点权重和 `(j-1)/j` 的新段累计中位数权重产生部署边界。
因此新段第一次预测仍使用旧边界，随后旧边界权重按证据计数自然消失，不需要 EMA、
平滑率或交接长度。分段后的 GMM 若暂时退回单分量，则保持最后一次已输出边界而不发生
第二次跳变；初始阶段仍保持精确 source fallback。该方法仍冻结 detector、LoRA、分类头
与 source 温度，不使用 target label、生成器边界或语义信息。与未分段和硬分段版本的
continual seed1 三方法对照入口为
`matched_jpeg_ascal_gmm_segmented_handoff_shift_continual_*_seed1.yaml`，仍只属于诊断实验。

`ascal_gmm_segmented_memory_shift` 保留硬分段的因果 BIC 变点判据，但不再永久删除每个
完成段学到的目标分布。变点触发时，旧段被压缩为 GMM、最大间隔边界和样本计数，不保存
target 图片、标签或生成器身份；若当前段曾由历史记忆召回，则用它最近一次完成的访问更新
同一条记忆，而不是不断产生重复条目。对新的稳定后缀，方法比较“直接使用某个固定历史
GMM”的负二倍预测对数似然与“在当前后缀重新拟合 GMM”的 BIC，并为历史条目身份支付
`2 log M` 描述长度；固定历史模型得分严格更低时才召回，否则创建新的分布状态。被召回
边界只作为一个证据票：新段第一个有效边界仍使用历史边界，之后其权重按有效边界数量的
倒数自然消失。该规则没有 memory capacity、相似度阈值或手工召回权重，能在
`A -> B -> A` 流中复用 A 的校准，同时让未匹配的新段保持原硬分段行为。与硬分段的
continual seed1 对照入口为
`matched_jpeg_ascal_gmm_segmented_memory_shift_continual_*_seed1.yaml`；它仍是诊断变体，
完整三 seed 验收前不得写入正文主表。

`ascal_gmm_segmented_memory_posterior` 在同一分段与 episodic memory 上把“只移动一个
边界”进一步改成联合密度 posterior。最大分量间隔仍只负责把有序 GMM 分成连续的低分
real 块和高分 fake 块；两块内部的分量权重分别重新归一化，所以 real 与 fake 都允许由
多个高斯描述，尤其不会假设所有 fake 只有一个模式。预测采用等类别先验的 Bayes 规则
`p(fake|s) = p(s|fake) / (p(s|real) + p(s|fake))`，GMM 在 target stream 中观测到的两块
总质量不会被偷换成类别先验。若描述长度判据召回历史段，方法不平均两个概率，而是在
log-odds 空间把历史 likelihood ratio 当作一票证据：第一次使用历史 posterior，之后其
权重随当前段有效 GMM 证据数按倒数自然衰减。该版本没有新增 class prior、融合系数、
posterior temperature 或 target 阈值；单分量时仍精确退回 source。与边界版的 continual
seed1 对照入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_continual_*_seed1.yaml`。由于密度比
不强制保持 source score 排序，这一版本同时检验多峰建模能否在 Accuracy 优先的前提下
进一步改善 AUC，仍属于三 seed 正式验收前的诊断实验。

`ascal_gmm_segmented_memory_posterior_projection` 的研究版本名为
**ASCAL-JMP-Median（R01）**。它保留上述联合密度对 real/fake 多峰块的
建模，但只取等先验 posterior 为 0.5 时、位于最大分量间隔内的 Bayes 分界；若该间隔内
不存在密度交点，则确定性退回间隔中点。最终概率仍由冻结 detector 的原始 score 减去
这个分界后，经 source temperature 的 sigmoid 得到，因此同一因果状态内严格保留源模型
排序。历史段召回同样只贡献一票已投影的 Bayes 分界，不混合 posterior，也不引入融合
系数、posterior temperature、目标阈值或额外先验。与直接 posterior 的 continual seed1
成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_projection_continual_*_seed1.yaml`；它用于
检验能否保留联合密度带来的 Accuracy 校准，同时避免尾部密度比破坏 target-macro AUC，
仍不属于完整三 seed 正式结果。

`ascal_gmm_segmented_memory_posterior_current_projection` 的研究版本名为
**ASCAL-JMP-Current（R02）**。它注意到当前段的每次 GMM 重拟合已经包含该段全部因果历史
分数，因此不再对一串彼此嵌套的累计拟合边界重复取中位数，而直接使用最新累计段 GMM 的
等密度分界。若召回历史 episode，历史分界仍只作为一票证据，并按当前段有效拟合次数自然
衰减；分段、描述长度召回、source fallback、单调 score 投影和冻结范围均保持 R01 不变。
该版本删除一种滞后来源且不增加阈值、窗口、温度、融合权重或其他 target 超参数。与 R01
的 seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_current_projection_continual_*_seed1.yaml`。
R02 的四数据集 seed1 诊断没有通过预注册的 Accuracy 与 target-macro AUC Pareto 晋级门槛，
因此不会进入 seed2/seed3，R01 继续作为当前候选基线。该负结果保留独立提交、源码归档和
结构化运行记录，不通过覆盖配置或针对单数据集修改规则来回收。

`ascal_gmm_segmented_memory_posterior_guarded_projection` 的研究版本名为
**ASCAL-JMP-GuardedScan（R03）**。它保留 R01 的累计中位数、联合密度边界、分段记忆和
单调 source-score 投影；普通时刻仍只执行原有的一个二进制调度后缀检查。只有已经部署的
历史/召回融合边界离开当前 GMM 最大分量间隔时，才把这次 MDL 变点搜索扩展为从 2 batch
到当前段一半长度的全部二次幂后缀。扩展搜索中的每个候选仍必须通过原有 segmented BIC、
多分量新段和后缀内部稳定性规则，边界越界本身不会直接重置或裁剪预测。这样把“历史状态
与当前无标签密度不相容”只当作一次更完整的变点检验触发器，而不是新增阈值或直接相信
最新拟合；target 超参数仍为零。与 R01 的 seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_guarded_projection_continual_*_seed1.yaml`。
R03 的四数据集 seed1 诊断和复现审计均完整结束，但相对 R01 的平均 Accuracy 下降
0.2068 个百分点，target-macro online AUC 下降 0.1618 个百分点，因此同样不进入
seed2/seed3。无标签状态显示守卫实际触发 235 次，却只有 4 个非原调度尺度的变点；少数
额外变点会改变后续分段与记忆轨迹，而没有形成稳定收益。R01 继续作为候选基线。

`ascal_gmm_segmented_memory_posterior_support_projection` 的研究版本名为
**ASCAL-JMP-SupportMedian（R04）**。它回到 R01 的原始分段、记忆、联合密度边界和
单调投影，只改变嵌套 GMM 边界的中位数计票：每次拟合的票重等于该拟合实际汇总的当前段
因果样本数，再取加权中位数；累计支持恰好平分时取相邻两个边界的中点。这样较晚、覆盖
更多已到达样本的拟合拥有更高证据权重，但任何一个最新拟合仍不能像 R02 那样直接覆盖
历史。R01 的 episodic recall 仍是一票并按有效拟合次数衰减，其他逻辑完全不变。该权重
来自在线已观测样本数，不新增幂指数、平滑率、窗口或阈值，target 超参数仍为零。离线重放
R01 的无标签状态时，它把边界落出当前 GMM 主间隔的次数从 462 降到 260；正式判断仍只看
独立 seed1 成对运行。入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_support_projection_continual_*_seed1.yaml`。
R04 的四数据集运行与因果审计均已完成：相对 R01，target-macro online AUC 提升 0.0133
个百分点，但平均 Accuracy 下降 0.1355 个百分点，fake Accuracy 下降 0.9616 个百分点；
因此越界次数减少并没有形成 Accuracy 优先的 Pareto 收益，R04 不进入 seed2/seed3。

`ascal_gmm_segmented_memory_posterior_global_residual` 的研究版本名为
**ASCAL-JMP-GlobalResidual（R05）**。它完整保留 R01 的原始 source score、GMM 分段、
episode memory、累计中位数和 Bayes 分界轨迹，只增加一个贯穿整条 continual stream 的
全局 residual。历史 GMM 仍只记录冻结 detector 的 `fake_logit-real_logit`，adaptive score
永远不会写回或改写历史。每批同时提取冻结视觉特征，并去除 source 二分类头方向后做 L2
归一化；预测完成且当前 source-score GMM 支持 real/fake 两块后，以等先验 posterior 的
`abs(2p-1)` 作为连续可靠度，分别累计 soft real/fake 特征原型。residual 是当前特征对 fake
原型与 real 原型的余弦相似度差，最终唯一 logit 为 R01 的单调边界投影 logit 加这个 residual。
它从零开始，只由上一批及更早的状态影响预测；GMM 单峰时不更新，但已经学到的一个全局
residual 继续保留。该闭式原型更新不需要 optimizer、learning rate、硬置信阈值、loss 权重、
memory capacity 或逐域 residual，且 residual 不参与自己的伪标签生成，避免移动 score 坐标
和自证循环。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_global_residual_continual_*_seed1.yaml`。
R05 的四数据集 seed1 因果审计全部通过：相对 R01，平均 Accuracy 提升 0.1128 个百分点，
target-macro online AUC 提升 0.0579 个百分点，四个数据集的 Accuracy 均未下降，online AUC
在三个数据集上升、一个数据集近似持平。该结果通过原定 Pareto 继续门槛，证明 feature
residual 可以在不破坏分类边界的情况下改变排序；但增幅仍小，因此它保留为可复现的正向
proof-of-concept，不据此声明最终方法。

`ascal_gmm_segmented_memory_posterior_mixture_residual` 的研究版本名为
**ASCAL-JMP-MixtureResidual（R06）**。它只修改 R05 的 fake 特征读出，R01 的 source score、
GMM 分段、episode memory、累计中位数和边界投影轨迹仍逐批保持不变。R05 把完整 fake
后验压成一个全局均值，可能使不同生成机制的互补特征彼此抵消；R06 保留一个汇总整个 real
GMM block 的稳定原型，同时把 BIC 选出的每个 fake score 分量分别累计为一个特征原型。
分量数直接继承原有 target BIC 和 source anchor complexity cap，不新增手调 `K`；分量在
排序后的 fake block 内按 rank 获得跨 batch 的确定性身份。每张图仍只产生一个 residual，
其值为与最相似 fake 原型的余弦相似度减去与 real 原型的余弦相似度，范围自然限制在
`[-2, 2]`，最终唯一 logit 仍为 R01 base logit 加该 residual。更新继续使用等先验 source-score
posterior、连续可靠度和 Predict-Then-Adapt 顺序，不使用 target label、generator id、optimizer、
learning rate、置信阈值、fusion weight 或 memory capacity。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_mixture_residual_continual_*_seed1.yaml`。
R06 的四数据集 seed1 运行和因果审计全部通过，多 fake 机制在每个数据集都实际触发，最多
保留 3 个 fake 原型；但相对 R01 的平均 Accuracy 只提升 0.1365 个百分点，target-macro
online AUC 只提升 0.0589 个百分点，相对 R05 的 AUC 仅高 0.0010 个百分点。它没有通过
运行前固定的 0.1 个百分点 AUC 优越门槛，因此不进入 seed2/seed3。这个结果说明 fake
分量确实存在，但继续细分由 source score 教出的 fake 原型仍主要复述源分类结构。

`ascal_gmm_segmented_memory_posterior_real_deviation_residual` 的研究版本名为
**ASCAL-JMP-RealDeviation（R07）**。它保留 R01 的全部 score/GMM/分段/记忆/边界轨迹和
R05 的冻结正交特征入口，但完全删除 fake 特征原型，只累计等先验 source-score posterior
给出的 soft-real 特征均值 `mu = R u`。对单位特征 `h`，唯一 residual 为
`R(R - cos(h, u))`；它等于样本到 `mu` 的平方距离减去 soft-real 自身平均平方距离后的一半，
因此在累计 soft-real 测度下期望恰好为零，不会依靠常数偏移制造 Accuracy，且由
`R in [0,1]` 可知范围自然落在 `[-0.25, 2]`。越偏离稳定 real 流形的样本得到越高 fake
分数，而 heterogeneous fake 不再被要求共享伪标签原型。该版本仍只有一个 residual，不用
target label、generator id、fake 模式数、optimizer、learning rate、阈值、fusion weight 或
memory capacity；seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_real_deviation_residual_continual_*_seed1.yaml`。
R07 的四数据集 seed1 运行和全部因果审计通过，real-deviation 机制在四套数据上都持续生效；
但相对 R01 的平均 Accuracy 虽提升 0.1342 个百分点，target-macro online AUC 却下降
0.0380 个百分点，相对 R06 下降 0.0969 个百分点。因此 R07 不进入 seed2/seed3。该结果
说明偏离 real 均值确实能移动分类边界，但 atypical real 与 heterogeneous fake 会共享较大
偏离，不能把未经条件化的几何异常直接当成排序证据。

`ascal_gmm_segmented_memory_posterior_conditional_residual` 的研究版本名为
**ASCAL-JMP-ConditionalResidual（R08）**。它回到 R05 唯一产生正 AUC 信号的全局原型分数
`q`，保留 `q` 本身，并从历史 pre-update 的 `q` 中减去可由不可变 source margin `s`
线性解释的部分。新增 innovation 为
`rho_+ * sigma_s / (T sigma_q) * [(q-mu_q) - cov(q,s)/var(s) * (s-mu_s)]`，其中
`rho_+=max(corr(q,s),0)`。因此与 source 不一致的残差自动得到零信任；正相关但不完全重复
source 的部分才进入排序。其 innovation 在累计历史下均值为零、与 `s` 协方差为零，RMS
最多是 source margin RMS 的一半，所有均值、方差、协方差都由无窗口的因果 Welford 状态
精确累计。该版本仍只有一个 residual，不使用 target label、generator id、optimizer、学习率、
阈值、窗口、fusion weight、memory capacity 或 shrinkage 参数；seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_conditional_residual_continual_*_seed1.yaml`。

`ascal_gmm_segmented_memory_posterior_preroute` 的研究版本名为
**ASCAL-JMP-PreRoute（R09）**。它把 R01 的 episodic memory 从“变点发生后才检索的档案”改成
真正参与当前预测与后续学习归属的专家库。每个专家仍不是一套神经网络，而是一个只在不可变
source margin 坐标上建模的冻结 GMM 及其投影 Bayes 分界；detector、LoRA 和分类头全程冻结。
每个当前无标签 mini-batch 到达后，方法先计算 active learning GMM 和所有已完成 memory GMM
对该批 score 的固定预测 deviance `-2 sum log p_e(s_i)`，不加相似度阈值或融合系数，直接选
deviance 最小的专家；平局时保留 active state，再按最早 memory index 确定性打破平局。当前批
立即用所选专家的分界预测，因此在 `A -> B -> A` 中，返回 A 的第一批就能使用 A，而不是等
adapt 后再惠及下一批。路由使用整批而非单个 scalar score，是因为单图 score 同时混合类别与
域信息；这是明确披露的 batch-transductive Predict，但没有利用标签，也没有用当前批更新后的
状态回头重算自身预测。

Predict 完成后，所选 memory 同时成为 adapt 的候选归属，但不会被无条件修改。方法复用原有
无标签描述长度原则，比较“该固定 memory 的 deviance 加 `2 log M` 身份码”和“只对当前批
重新拟合 GMM 的 BIC”；旧专家严格胜出时，才结束此前 active visit、把学习状态立即切到该
memory，并以其历史分界启动一次新访问。后续批次累计拟合这次访问，流离开时再用最新访问
替换该 memory；若当前批更像新状态，旧 memory 完全不动，继续由 R01 分段器学习并创建新
专家。这样同一个路由决定同时服务预测和持续适应，又避免“任何新域都被迫选中最近旧专家”
造成记忆污染。它没有新增网络、阈值、窗口、学习率、memory capacity 或 target 超参数；
seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_preroute_continual_*_seed1.yaml`。
R09 的四数据集 seed1 正式运行和全部路由因果审计通过，证明旧专家能够在当前批预测前被
实际选中，并能在确认后接管后续学习状态；四套流共发生 1175 批 memory prediction，494 次
学习归属检查中 141 次确认、353 次拒绝。相对 R01，平均 Accuracy 提升 0.3696 个百分点，
但 target-macro online AUC 下降 0.7813 个百分点，average forgetting 增加 1.2247 个百分点；
它也比当前排序最佳候选 R06 的 AUC 低 0.8402 个百分点，因此不进入 seed2/seed3。失败原因
不是专家库没有工作，而是预测阶段在 MDL 新颖性判定之前就让最大似然旧专家接管：随后被
adapt 拒绝的 353 次旧专家归属已经改变了当前批排序。下一版只能让 Predict 与 Adapt 共享
同一个无参数归属结论，不能继续把“最像的历史专家”和“足以属于该专家”混为一谈。

`ascal_gmm_segmented_memory_posterior_mdl_route` 的研究版本名为
**ASCAL-JMP-MDLRoute（R10）**。它只修正 R09 暴露出的 Predict/Adapt 归属分裂，不改变 R01
专家的构造、分界或更新规则。当前无标签批先以固定预测 deviance 找到最像的历史 GMM；这时
它还只是 proposal。若 proposal 不是当前 active expert，方法在内存中临时拟合一个当前批
GMM，并比较“旧专家 deviance 加均匀专家身份码 `2 log M`”与“新 GMM 的 BIC”。只有旧专家
描述长度严格更短时，它才成为该批唯一的正式归属：立即负责当前预测，并在 Predict 完成后
原样交给 Adapt，启动或继续该专家的一次 active visit。否则旧专家在预测前即被拒绝，当前批
从一开始就使用 R01 active readout（无 active 时使用 source），随后也只进入同一 active/new
state 学习路径。临时 GMM 只充当“这是新状态”的无标签描述长度基线，不参与分类、不写入
memory，也不会在 Adapt 中重新拟合或重新投票。因而每批严格满足 one-batch-one-expert，且
没有新增阈值、窗口、融合权重、学习率、memory capacity 或 target 超参数；seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_mdl_route_continual_*_seed1.yaml`。
R10 的四数据集 seed1 正式运行、MDL 算术和 Predict/Adapt 归属审计全部通过。相对 R09，
Accuracy 再提升 0.0978 个百分点，target-macro online AUC 回升 0.0271 个百分点，average
forgetting 减少 0.6901 个百分点，证明在预测前拒绝错误 proposal 是必要修正；但相对 R01，
Accuracy 虽提升 0.4674 个百分点，target-macro online AUC 仍下降 0.7542 个百分点，且
forgetting 增加 0.5346 个百分点，因此仍不进入 seed2/seed3。四套流的 1175 次 memory
proposal 中，353 次错误跨专家 proposal 已在预测前被拒绝，141 次通过 MDL 并完成一致的
预测与学习交接；余下 681 次却是在 active expert 和它自己的 archived snapshot 之间选择。
这两者名义上属于同一专家，但预测读取旧快照、Adapt 更新 live visit，仍然违反
one-expert-one-live-state。下一版应在 active GMM 可用时从候选中去掉同一 identity 的历史
快照，只允许 archived memory 代表非当前专家或在 live state 暂不可用时充当回退。

`ascal_gmm_segmented_memory_posterior_live_route` 的研究版本名为
**ASCAL-JMP-LiveRoute（R11）**。它完整保留 R10 的 likelihood proposal、MDL admission、
Predict/Adapt 唯一归属和后续 visit 更新，只修正同一专家被重复暴露的问题。若当前 active
expert 已有双峰 live GMM，路由候选中只保留这个持续更新的 live state，并隐藏该 identity
在 episodic memory 中尚未替换的 archived snapshot；其他历史专家照常参与路由。若 active
visit 刚启动、live GMM 尚未形成双峰，则其 archived snapshot 仍可作为同 identity 的冷启动
回退，避免突然退回 source。这个规则不删除 memory，也不改变任何 GMM、BIC、身份码、分界
或更新公式，只让每个 expert 在任一时刻至多暴露一个可路由状态；因此仍没有新增 target
超参数。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_live_route_continual_*_seed1.yaml`。
R11 的四数据集 seed1 正式运行和全部配对、协议、MDL 算术及专家归属审计通过。R10 中
681 批“active expert 与自身 archived snapshot 竞争”的同身份复用，在 R11 中降为 39 批，
且全部只发生在 live GMM 尚不可用的冷启动阶段；live GMM 可用时没有一次旧快照重复参与
路由。四套流共提出 640 次 memory proposal，其中 601 次属于跨专家切换：171 次通过 MDL
并由同一专家完成当前批预测和后续适应，430 次在预测前拒绝。结构修正使 average forgetting
相对 R10 减少 0.1849 个百分点，但没有形成排序收益：相对 R01，平均 Accuracy 提升 0.4318
个百分点，target-macro online AUC 下降 0.7737 个百分点；相对 R10，Accuracy 下降 0.0356
个百分点，AUC 下降 0.0195 个百分点。R11 因而不进入 seed2/seed3，R06 继续作为当前指标
领先候选。这个结果排除了“同专家旧快照竞争”作为主要 AUC 损失来源，并把问题进一步定位为：
不同批次由不同专家边界产生的可变 logit 平移虽然能改善 0.5 阈值分类，却破坏了跨批样本的
全局可比排序；下一版不能再只改路由，应让专家知识改变排序证据而不是反复改写 score 原点。

`ascal_gmm_segmented_memory_posterior_ordinal_route` 的研究版本名为
**ASCAL-JMP-OrdinalRoute（R12）**。它不修改 R11 的专家候选、MDL admission、唯一 live
state、硬分类决定或 Adapt 归属，只改变最终一个标量的有序读出。通过路由的专家仍先决定
当前样本属于 0.5 以下的 real 区间还是 0.5 以上的 fake 区间；区间内部不再使用会随专家变化
的 `source margin - expert boundary`，而统一使用不可变 source probability 排序。具体地，
source probability 为 `q` 时，专家判 real 输出 `q/2`，判 fake 输出 `1/2 + q/2`。因此它逐样本
严格保留 R11 的 Accuracy，同时在每个预测类内部严格恢复冻结 source margin 的全局次序；
没有专家可用时则完全保留原 source probability，不做重映射。该读出仍只有一个最终概率，
不增加网络、残差、温度、阈值、融合权重或 target 超参数，用来直接检验 R11 的 AUC 损失是否
来自跨专家边界平移。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_ordinal_route_continual_*_seed1.yaml`。
运行前固定的晋级条件是 Accuracy 相对 R11 最多下降 0.2 个百分点且 target-macro online AUC
超过 R06；否则只作为排序坐标诊断，不进入 seed2/seed3。

R12 的四数据集 seed1 成对运行和协议、路由状态、逐样本硬决定审计均已通过；3546 个部署
专家的 batch、56584 个样本全部逐样本保持 R11 的分类结果。四数据集宏平均 Accuracy 因而
与 R11 完全相同，为 80.7969%；target-macro online AUC 从 R11 的 86.8854% 恢复到
87.6831%，同时 average forgetting 从 0.3588% 降到 0.0564%。相对 R01，Accuracy 提升
0.4318 个百分点且 AUC 提升 0.0240 个百分点；相对 R06，Accuracy 提升 0.2953 个百分点，
但 AUC 仍低 0.0349 个百分点，因此没有通过“必须超过 R06 AUC”的预注册 seed2/seed3 门槛。
此外，这个区间序数值是决策分数而非校准 posterior，NLL 明显变差；下一版应把因果 feature
residual 放入统一的排序坐标，而不能靠调整区间映射回收结果。

`ascal_gmm_segmented_memory_posterior_routed_residual` 的研究版本名为
**ASCAL-JMP-RoutedResidual（R13）**。它吸收 R09-R12 的路由诊断，但不再让所选专家的边界
平移最终 score：R01 的 source-score GMM、分段、episode memory、累计中位数和连续边界投影
仍作为唯一共享校准底座。当前无标签 batch 只用冻结 source margin，在一个 eligible live GMM
与所有非 active 历史 GMM 中产生最小 predictive deviance proposal；非 active memory 仍须以
`fixed deviance + 2 log M` 严格击败当前 batch GMM 的 BIC 才能成为 residual 专家。active
expert 已有 live GMM 时隐藏其同 identity archived snapshot，因此每个 identity 仍至多暴露
一个可选状态。

每个 residual 专家只保存一个 soft-real 特征和由该专家 BIC GMM 自然给出的若干 ordered fake
特征原型。冻结视觉特征先去除 source 二分类头方向并 L2 归一化；所选专家用预测时的等先验
source-score posterior 及 `abs(2p-1)` 连续可靠度，在当前 batch 正式预测完成后闭式累计这些
充分统计。最终唯一 logit 是 R01 连续 base logit 加“最相似 fake 原型余弦减 real 原型余弦”；
同一批在 Predict 中读取哪个专家，Adapt 就只更新哪个专家，但这一 residual 路由永远不改变
R01 score calibrator 的学习归属或 score 原点。方法不保存图片或逐样本 raw feature，不使用
target label、generator id、optimizer、学习率、硬置信阈值、fusion weight、memory capacity
或新增 target 超参数。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_routed_residual_continual_*_seed1.yaml`；运行前
固定晋级条件仍是 Accuracy 相对 R11 最多下降 0.2 个百分点且 target-macro online AUC 超过
R06。

`ascal_gmm_segmented_memory_posterior_routed_ridge_residual` 的研究版本名为
**ASCAL-JMP-RoutedRidge（R14）**。它冻结 R13 的 source-score 校准、唯一 live GMM 候选、
MDL admission 和 residual 专家归属，只把每专家的 real/fake 余弦原型替换为一个零初始化的
在线线性 head。输入仍是去除 source 分类头方向并 L2 归一化的冻结 CLIP 特征；预测时只有
R13 已选中的专家输出 `h^T w_e`，最终唯一 logit 为 R01 base logit 加该值，新专家尚未学习时
严格等于 R13 的零 residual 路径。线性 head 不含额外 bias，因为统一平移仍由 R01 base
负责；residual 只学习正交特征中的样本级差异，也不再人为裁剪其读出。

预测完成后，预测时所选 GMM 产生等先验 posterior `p`、有界 soft target `u=2p-1` 和连续
可靠度 `c=|2p-1|`。同一专家随后以固定 identity prior 解
`sum_i c_i(h_i^T w-u_i)^2 + ||w||^2`；实现用 Woodbury 递推精确更新 inverse Gram 和权重，
所以不需要 epoch、optimizer、学习率、硬阈值，也不保存图片或逐样本 feature。完整岭回归的
每专家充分统计包含一个 `768 x 768` inverse Gram，而不只是 768 个权重；这是用内存换取无
学习率的精确在线解，必须在效率统计中如实报告。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_routed_ridge_residual_continual_*_seed1.yaml`；
运行前门槛为 Accuracy 相对 R12 最多下降 0.2 个百分点且 target-macro online AUC 超过 R06。

`ascal_gmm_segmented_memory_posterior_ordinal_ridge` 的研究版本名为
**ASCAL-JMP-OrdinalRidge（R15）**。它以 R12 而不是 R01/R13 为不可变主干：当前 batch 仍由
R12 唯一 live GMM 候选和 MDL admission 选择同一个专家，该专家的投影边界仍产生逐样本
real/fake 硬决定，Predict 后的 GMM 分段、memory handoff 和专家归属也完全沿用 R12。唯一新增
状态是每个 R12 专家携带一个零初始化在线线性 rank residual；冻结 CLIP 特征先去除 source
分类头方向并 L2 归一化，新专家未学习或没有可用专家时严格退化为 R12。

预测时，所选专家只把其旧 Ridge 输出加到不可变 source logit，并减去当前 batch 的 residual
均值，避免重新引入跨专家 score 原点平移；修正后的 sigmoid 只替换 R12 的类内排序坐标，
R12 判 real 仍映射到 0.5 以下、判 fake 仍映射到 0.5 以上，因此任何 residual 都不能改变
R12 硬预测。预测完成后，同一预测时 GMM 产生等先验 posterior `p`、连续可靠度
`c=abs(2p-1)` 和 `GMM teacher logit - immutable source logit` 目标；目标先减去可靠度加权批均值，
再以固定 identity prior 做精确 Woodbury 岭回归更新。方法没有 epoch、optimizer、学习率、
硬置信阈值、residual 融合系数或第二套路由，也不保存图片或逐样本 feature。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_ordinal_ridge_continual_*_seed1.yaml`；运行前
门槛固定为逐样本硬决定与 R12 完全一致，且 target-macro online AUC 超过 R06。

`ascal_gmm_segmented_memory_posterior_joint_ridge` 的研究版本名为
**ASCAL-JMP-JointRidge（R16）**。它保留 R12 的不可变 source-score 坐标、unique-live MDL
专家路由、GMM 分段、memory handoff 和 Predict-Then-Adapt 归属，但不再冻结 R12 的最终硬
决定。每个专家携带一个零初始化的在线 Ridge；输入是去除 source 分类头方向并 L2 归一化的
冻结 CLIP 特征，再附加一个常数 bias 坐标。零状态严格输出 R12 概率，历史专家恢复时同时
恢复其 GMM、inverse Gram、feature residual 和 bias。

预测时先取 R12 概率的 logit，再直接加上旧专家的 `phi(h)^T w_e`，最终 sigmoid 不做半区间
锁定或 batch residual 中心化。因此 feature 权重可以改变样本排序，bias 可以移动有效分类
边界，样本也可以跨越 0.5。其联合概率解释为 `final odds = R12 odds * exp(r_e(h))`。预测完成
后，预测时旧 GMM 给出等先验 posterior `p`、可靠度 `c=abs(2p-1)` 和严格 residual 目标
`logit(p) - logit(p_R12)`，同一专家通过固定 identity prior 的精确 Woodbury RLS 更新。
修正输出永不回写路由或 GMM，二者始终保留不可变 source score，避免在线权重变化破坏历史
专家坐标。方法不引入 epoch、optimizer、学习率、硬置信阈值、融合系数或 memory 容量。
seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_joint_ridge_continual_*_seed1.yaml`；预注册晋级
门槛为四数据集宏平均 Accuracy 超过 R12，且 target-macro online AUC 超过 R06。

`ascal_gmm_segmented_memory_posterior_pairwise_ridge` 的研究版本名为
**ASCAL-JMP-PairwiseRidge（R17）**。它针对 R16 的固定 seed1 失败诊断，只替换 feature
residual 的监督目标和作用域：R12 的 immutable source score、unique-live MDL 路由、GMM
分段、memory handoff 与 Predict-Then-Adapt 归属全部保持不变，但不再让每个 GMM 专家携带
一套不同刻度的 log-odds residual。整个持续流只保存一个零初始化、无 bias 的线性 ranker；
冻结 CLIP 特征仍先去除 source 分类头方向并 L2 归一化，新 ranker 未学习或当前没有 R12
专家时严格退化为 R12。

预测完成后，R12 的单调 routed decision 给每个样本确定 real/fake 方向，预测时所选旧 GMM
只给出连续 posterior 可靠度。posterior 先投影到 R12 决策所在的半区间；若 GMM 密度尾部与
R12 单调决定矛盾，则投影为 0.5 并自动得到零权重。当前 batch 的所有 soft fake-real 特征差
以单位排序差为目标，形成一个无截距 pairwise Ridge；实现通过 soft-pair graph Laplacian 将
全部样本对精确压缩到至多 `batch_size - 1` 个更新方向，再用 Woodbury RLS 累计到同一个全局
充分统计。最终概率为 `sigmoid(logit(p_R12) + h^T w)`，因此允许样本级排序和阈值改变，但不再
继承无界 GMM logit、专家 bias 或专家间 score scale。方法不保存图片、逐样本特征或样本对，
也不引入 epoch、optimizer、学习率、硬置信阈值、residual 系数或第二套路由。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_pairwise_ridge_continual_*_seed1.yaml`；预注册
晋级门槛仍为四数据集宏平均 Accuracy 超过 R12，且 target-macro online AUC 超过 R06。

`ascal_gmm_segmented_memory_posterior_analytic_expert` 的研究版本名为
**ASCAL-JMP-AnalyticExpert（R18）**。它继续完整保留 R12 的冻结 source-score 坐标、
unique-live MDL 路由、GMM 分段、memory handoff 和 Predict-Then-Adapt 专家归属，但把此前的
“base logit 加 residual”改成每专家一个直接学习最终有符号类别坐标的在线 Ridge。输入为
R12 概率的有界坐标 `b=2p_R12-1`、去除 source 分类头方向并 L2 归一化的冻结 CLIP 特征以及
常数 bias；固定先验均值为 `[1, 0, ..., 0]`，因此任何新专家或尚未更新的历史专家都精确输出
R12 概率。学习后唯一判别式为 `d_e=alpha_e b+h_perp^T v_e+beta_e`：`alpha_e` 学习该专家应
保留多少 R12 证据，`v_e` 提供能够改变 AUC 的样本级排序修正，`beta_e` 学习分类边界。

当前 batch 先用不可变 source scores 选择一个 R12 专家，再用该专家的旧 Ridge 状态正式预测；
预测完成后，仍由预测时旧 GMM 给出等先验 posterior。posterior 被投影到 R12 已决定的类别
半区间，矛盾证据落到 0.5 并自动获得零权重；其余样本使用 `u=2p-1` 作为有界软标签、
`c=|u|` 作为连续可靠度，精确递推求解
`sum_i c_i(phi_i^T w_e-u_i)^2+||w_e-[1,0,...,0]||^2`。每个专家只保留 inverse Gram 与权重，
不保存图片或逐样本特征，不使用 epoch、optimizer、学习率、硬置信阈值、融合系数或新增 target
超参数；Ridge 输出也永不回写路由和 GMM。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_analytic_expert_continual_*_seed1.yaml`；预注册
晋级门槛仍为四数据集宏平均 Accuracy 超过 R12，且 target-macro online AUC 超过 R06。

R18 的四数据集 seed1 成对运行以及协议、R12 replay、路由状态和因果更新审计均已通过；3546
个专家更新覆盖 56584 个在线样本，解析求解失败为 0。四数据集宏平均 Accuracy 为 80.7962%，
相对 R12 的 80.7969% 基本不变；pooled online AUC 从 87.8513% 降到 87.4988%，
target-macro online AUC 从 87.6831% 降到 86.3476%，且四个数据集的 target-macro AUC 均下降。
相对当前 AUC 最优候选 R06，R18 的 Accuracy 高 0.2946 个百分点，但 target-macro AUC 低
1.3704 个百分点；average forgetting 也从 R12 的 0.0564% 增至 1.6773%。R18 的 Brier、NLL
和 ECE 均明显改善，说明有界 soft target 确实学到了概率校准，但 56584 个在线专家样本中只
改变了 2 个硬决定，同时所有在线与固定 holdout 前向累计产生 70892 次越界截断。该结果表明
直接把 Ridge 有符号回归值线性映射并裁剪为最终概率会压缩大量排序信息，不能作为 R12 的统一
判别读出。R18 不进入 seed2/seed3；R12 继续作为 Accuracy 锚点，R06 继续作为 AUC 领先候选。

`ascal_gmm_segmented_memory_posterior_rms_ridge_expert` 的研究版本名为
**ASCAL-JMP-RMSRidgeExpert（R19）**。它保留 R12 的冻结 source-score 坐标、unique-live MDL
路由、GMM 分段、memory handoff 和 Predict-Then-Adapt 归属；每个路由专家只新增一个直接的
二输出解析 Ridge 分类器。输入是完整的 L2 归一化冻结 CLIP 特征和常数 bias，监督目标是 R12
在当前批已经作出的 real/fake one-hot 伪标签，而不是 R12 分数、GMM posterior 或 residual。
预测时旧 GMM 只给连续权重 `c=|2p_GMM-1|`；若 posterior 与 R12 类别方向矛盾，该样本权重
自然归零，不再另设硬置信阈值。

每个专家因果累计 inverse regularized Gram、feature/one-hot cross-covariance、两类可靠质量和
R12 logit 平方和，并用 Woodbury RLS 在预测后精确更新。推理时 Ridge margin 为
`r=h^T(W_fake-W_real)`；R12 margin 与 Ridge margin 分别除以该专家历史可靠样本上的 RMS，再用
`sigmoid(z_R12/rms_R12+r/rms_Ridge)` 产生统一排序分数。Ridge 未更新、任一伪类尚无可靠质量或
历史 RMS 无效时，输出逐值精确回退 R12；当前 batch 的统计绝不参与自身预测。这里 sigmoid 只
是统一 evaluator 接口的单调映射，不宣称校准 posterior。方法没有 epoch、optimizer、学习率、
置信阈值、融合系数或新增 target 超参数，也不保存图片和逐样本特征。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_rms_ridge_expert_continual_*_seed1.yaml`；运行前
固定的晋级门槛为四数据集宏平均 Accuracy 严格超过 R12，且 target-macro online AUC 超过 R06。

R19 的四数据集 seed1 成对运行以及协议、R12 replay、路由状态、充分统计和因果更新审计均已
通过；3546 次 Ridge 更新覆盖 56584 个候选样本，3530 个后续 batch 实际使用历史 Ridge 预测，
解析求解失败为 0。四数据集宏平均 Accuracy 从 R12 的 80.7969% 提高到 81.1025%，四个数据集
均有提升；但 target-macro online AUC 从 87.6831% 降到 87.4692%，仍低于 R06 的 87.7180%，
pooled online AUC 也从 87.8513% 降到 87.3964%。39 个 target 中只有 16 个 AUC 提升，AIGC
Detection Benchmark 和 AIGI-Holmes P3 的 target-macro AUC 分别下降 0.4833 和 0.8643 个百分
点。与此同时，final target-macro AUC 相对 R12 提高 0.3858 个百分点，说明解析分类器确实学到
了可迁移的最终判别状态，但在线学习过程中的排序扰动尚未受控。

R19 共改变 R12 的 2474 个在线硬决定，其中 2357 个为 fake 到 real，只有 117 个为 real 到
fake；宏平均 real Accuracy 提高 4.1196 个百分点，fake Accuracy 下降 3.5083 个百分点，最终
只留下 0.3056 个百分点的 Accuracy 净增益。该非对称变化说明失败不在 Ridge 是否运行，而在
监督与读出：直接拟合 R12 硬伪标签会继承困难域的 real 偏置，而分别做 RMS 归一化后又会把
尚未证明可靠的 Ridge margin 强制提升到与 R12 margin 相同的能量尺度。average forgetting 也
从 R12 的 0.0564% 增至 0.7769%。因此 R19 不进入 seed2/seed3；R12 继续作为 Accuracy 锚点，
R06 继续作为 online AUC 领先候选。下一版若继续使用解析分类器，必须让新增判别证据的作用强度
由其自身因果可靠性决定，而不能再做无条件等能量融合。

`ascal_gmm_segmented_memory_posterior_equal_prior_ridge_expert` 的研究版本名为
**ASCAL-JMP-EqualPriorRidge（R20）**。它先隔离验证 R19 暴露出的类别先验问题，不同时改动
Ridge 的介入时机：R12 路由、GMM、分段、memory handoff、硬伪标签、连续可靠度、RLS 更新、
专家状态和单位 RMS 融合均与 R19 完全相同。唯一变化发生在旧专家的预测读出。对专家已有的
Ridge 方向 `d_e=W_fake-W_real`，利用 R19 已保存的每类 cross-covariance `q_e,y` 和可靠质量
`M_e,y` 解析得到两个历史 margin 质心：

```text
mu_e,y = d_e^T q_e,y / M_e,y
center_e = (mu_e,real + mu_e,fake) / 2
r_equal-prior(h) = h^T d_e - center_e
```

该中心把两个伪类的历史 Ridge margin 质心放到零点两侧等距离的位置，因此不会把伪标签流中
`real:fake` 的经验质量比例直接当成部署先验。centered Ridge RMS 也由 inverse Gram、
cross-covariance 和 class mass 精确推导，不回放样本；最终仍使用
`sigmoid(z_R12/rms_R12+r_equal-prior/rms_Ridge)`。冷启动和无效状态逐值精确回退 R12，方法不
新增持久状态、阈值、融合系数、epoch、optimizer 或学习率。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_equal_prior_ridge_expert_continual_*_seed1.yaml`；
预注册晋级门槛仍为四数据集宏平均 Accuracy 严格超过 R12，且 target-macro online AUC 超过
R06。R20 只回答“等先验中心能否消除 R19 的 fake→real 偏置”；inverse-Gram 证据门控保留为
下一轮独立结构变化，不能与本轮结果混在一起解释。

R20 的四数据集 seed1 成对运行以及协议、R12 replay、R19 路由与学习状态、等先验解析式和
Predict-Then-Adapt 因果更新审计均已通过；3546 次更新覆盖 56584 个候选样本，解析求解失败为
0。四数据集宏平均 Accuracy 为 81.2227%，相对 R12 提高 0.4258 个百分点、相对 R19 提高
0.1202 个百分点；pooled online AUC 为 87.5473%，相对 R19 提高 0.1510 个百分点。等先验中心
把 fake 到 real 的决定变化从 R19 的 2357 次降到 2169 次，同时 real 到 fake 从 117 次增至
156 次，说明它确实减轻了 Ridge 读出的类别先验偏置。

但 R20 的 target-macro online AUC 为 87.4655%，相对 R19 仍低 0.0037 个百分点，也低于 R12
的 87.6831% 和 R06 的 87.7180%；average forgetting 为 0.7782%，同样没有改善。由于同一专家
内减去一个固定 center 不会改变样本排序，R20 只能修正边界和专家间偏移，不能消除 Ridge 在
证据不足时造成的样本级排序扰动。因此 R20 不进入 seed2/seed3。下一版只允许把无条件单位 RMS
融合改为由旧 inverse Gram 解析得到的样本级证据门控，R20 的路由、监督、状态更新和等先验
中心必须保持不变。

`ascal_gmm_segmented_memory_posterior_evidence_gated_ridge_expert` 的研究版本名为
**ASCAL-JMP-EvidenceGatedRidge（R21）**。它完整继承 R20 的 source-score 路由、GMM、分段、
memory handoff、伪标签、连续可靠度、每专家 RLS 状态、等先验中心和历史 RMS；唯一变化是把
无条件相加的两个单位 RMS 分类信号改成样本级证据门控的分类器接管。对当前归一化 CLIP 特征
`h`，使用预测时旧专家的 inverse regularized Gram `P_e` 计算

```text
g_e(h) = clip(1 - h^T P_e h / h^T h, 0, 1)
z_ridge(h) = r_equal-prior(h) * rms_R12 / rms_Ridge
z_final(h) = (1 - g_e(h)) * z_R12(h) + g_e(h) * z_ridge(h)
```

计算证据时把 design vector 的常数 bias 坐标置零，避免仅学会专家截距就获得样本级判别信任。
因为单位 Ridge 先验对应 `P_e=I`，从未被历史可靠样本覆盖的特征方向有 `g=0`，逐样本精确返回
R12；该方向的 posterior variance 随证据累积而下降时，`g` 才无阈值地趋近 1。Ridge 是直接
二分类器而不是 residual，所以这里让它在有证据时逐步替代 R12，而不再重复相加同一伪类别
信号。当前 batch 只使用更新前的 `P_e`，预测后仍按 R20 原规则更新同一个专家。方法不新增
持久状态、阈值、融合系数、epoch、optimizer、学习率或 target 超参数。seed1 成对入口为
`matched_jpeg_ascal_gmm_segmented_memory_posterior_evidence_gated_ridge_expert_continual_*_seed1.yaml`；
预注册晋级门槛仍为四数据集宏平均 Accuracy 严格超过 R12，且 target-macro online AUC 超过
R06。

R21 的 seed1 首轮按用户要求在 GenImage 和 AIGCDetectionBenchmark 完整结束后停止，AIGI-Holmes
P3 与 OpenSDID Global 未纳入结果，已开始但未完成的 AIGI-Holmes P3 临时输出已删除。因此本轮
只能记为两数据集 pilot，不能执行或声称通过四数据集预注册晋级。两个完整运行的样本、holdout、
R12 replay、R20 路由与学习状态、Predict-Then-Adapt 因果更新和解析状态维度审计均通过；2156 次
更新覆盖 34244 个应用样本，求解失败为 0。

相对配对 R12，R21 在 GenImage 上的 target-macro Accuracy、target-macro online AUC 和诊断性
pooled online AUC 分别提高 0.4190、0.4419 和 1.2010 个百分点；但 final target-macro AUC
下降 1.2433 个百分点，
average forgetting 增加 2.3172 个百分点。在 AIGCDetectionBenchmark 上，target-macro
Accuracy 和诊断性 pooled online AUC 分别提高 0.8784 和 0.7252 个百分点，target-macro online
AUC 却下降 0.3474 个百分点，average forgetting 增加 1.4383 个百分点。两个已完成数据集的
简单平均 target-macro Accuracy、target-macro online AUC 和诊断性 pooled online AUC 分别提高
0.6487、0.0473 和 0.9631 个百分点。
这里的 Accuracy 数值因所有 target 均为 1500 个在线样本而与 target-macro Accuracy 相等。正式
结论只认可 target-macro：R21 改善了分类边界，但 target-macro AUC 提升不稳定，pooled AUC 的
变化不能作为排序能力提升的证据；GenImage 可用于快速筛选，不能单独作为跨数据集有效性的证明。

`ascal_gmm_segmented_memory_posterior_feature_routed_trusted_ridge` 的研究版本名为
**ASCAL-JMP-FeatureRoutedTrustedRidge（R22）**。它把 R21 中互相纠缠的专家选择、无标签可信度
和最终判别拆成三个单一职责。当前 batch 的冻结 CLIP 特征先减去 source 二分类头的真假方向，
再做 L2 归一化，并以对各专家历史特征原型的平均余弦相似度选择一个专家；这样路由主要比较域
特征而不是当前 batch 的真假比例，冻结 source score 也不再参与历史专家排序或召回准入。
所选专家的旧 GMM 仍保留一个 real block 和由 BIC 自然确定的若干 fake components，但只在
当前预测完成后产生等先验伪类别和连续可靠度 `abs(2p_GMM-1)`。Ridge 直接拟合 real/fake
one-hot 标签，不回归 GMM posterior，也不把自己的输出反馈给 GMM。

最终唯一评测分数为冻结 Base fake probability 与所选专家 direct Ridge softmax probability 的
等权平均，GMM posterior 和 GMM 分界均不进入最终输出。新专家的 Ridge 权重为零，因此其
probability 为 0.5；与 Base 平均后仍严格保留 Base 的 0.5 决策和样本次序。随着 GMM 认可的
历史样本通过解析 RLS 累积，Ridge 才能利用完整 CLIP 特征改变后续样本排序。当前 batch 始终
读取旧路由原型、旧 GMM 和旧 Ridge，预测完成后才更新同一个专家，不保存图片或逐样本特征，
也不新增 confidence threshold、routing threshold、epoch、optimizer、学习率或 target 参数。
首轮只在 GenImage matched-JPEG seed1 上作为 pilot 筛选，结果完成前不得写成正式三 seed 结论。

R22 的 GenImage seed1 pilot 已在 4090-2 上由固定提交 `77a9f01` 完成。10500 个在线样本与
Source、R12、R21 的身份和顺序逐项一致，CLIP、LoRA source checkpoint 与 matched-JPEG
profile 哈希也完全一致。R22 的 target-macro online Accuracy/AUC 为 69.2286%/82.3997%；
相对 Source 为 +0.9143/-0.1034 个百分点，相对 R12 为 -7.5810/+0.1659 个百分点，相对 R21
为 -8.0000/-0.2759 个百分点，因此未通过晋级条件，不进入 seed2/seed3。

这次失败不是 Ridge 没有更新：657 次解析更新覆盖 10484 个候选样本，三个专家的有效支持为
9709.29，求解失败为 0。事后只读分解把最终读出前的 feature-routed internal decision 单独计分；
target labels 仅由外部 evaluator 使用，不进入方法。其 target-macro Accuracy/AUC 为
75.8762%/82.1228%，相对 Source 为 +7.5619/-0.3803 个百分点，相对 R12 为
-0.9333/-0.1110 个百分点。因此特征路由保留了大部分 R12 Accuracy 收益，但并非稳定完成：
658 个 batch 中发生 166 次 handoff，同一 target 内有 165 次专家身份切换，glide 和 Midjourney
的主导专家样本占比仅为 49.87% 和 52.27%。

主导失败仍在最终读出。one-hot Ridge 的 batch 平均绝对 margin 只有 0.7863，而 Base 在困难
生成器上强烈偏向 real；把两个 probability 固定平均后，Ridge 往往无法把 Base 的假图拉回
0.5 以上。最终读出相对内部路由决定损失 6.6476 个 Accuracy 百分点：它修正 553 个内部错误，
却破坏 1251 个内部正确决定，并把 1762 个 fake 决定改回 real。也就是说，R22 同时存在路由
抖动和读出冲突，但约 6.65 个百分点的直接坍塌来自固定平均。这证明“都变成 `[0, 1]` 概率”
只保证数值范围一致，并不保证两套分类证据的决策强度可比；下一版应先修复读出，再单独判断
是否需要稳定路由，同时保持 GMM 只负责伪标签与训练可靠度。

同一固定流上的 Ridge-only 只读反事实进一步确认了这一点。由于 R22 的最终 probability 不反馈
给路由、GMM 或 Ridge 更新，移除 Base average 不改变任何在线状态轨迹；64 个 cold-start 样本
严格输出 0.5，其余样本只使用所选专家的 direct Ridge softmax probability。该读出的
target-macro Accuracy/AUC 为 76.7238%/82.2767%，相对原 R22 为 +7.4952/-0.1230 个百分点，
相对 Source 为 +8.4095/-0.2264 个百分点，相对 R12 为 -0.0857/+0.0430 个百分点。它证明
direct Ridge 本身已经基本恢复 R12 的分类能力，固定平均才是 R22 Accuracy 坍塌的主因；但其
AUC 仍低于 Source 和 R21，因此 Ridge-only 可作为下一版直接读出的基线，尚不能宣称已经解决
持续排序适应。这里移除的只是 Base probability 的最终融合；冻结 Base score 仍仅在内部供 GMM
产生伪类别和可靠度，不进入最终预测。

`ascal_gmm_segmented_memory_posterior_feature_routed_source_ridge` 的研究版本名为
**ASCAL-JMP-SourceRidgeInheritance（R23）**。它先把 R22 中形式不一致的神经网络 Base 头和
在线 Ridge 专家统一成同一种分类器：源阶段仍用三轮监督训练 LoRA 特征，但部署前在完整源训练
集的 L2 归一化 CLIP-LoRA 特征与常数 bias 坐标上拟合二输出解析 Ridge。checkpoint 同时保存
regularized Gram、inverse Gram、cross-covariance、权重、类别质量和源样本数；部署的 Base
分类头就是这组 Ridge 权重，校准锚点也在替换后的 Base 上重新计算。

每个新专家不再从零开始，而是深拷贝完整源 Ridge 统计，因此诞生时与 Base 的逐样本 margin
完全一致，并天然继承源域监督知识。当前 batch 仍只用去除真假方向后的冻结 CLIP 特征选择历史
专家，所选旧 GMM 仍只提供等先验伪标签和连续可靠度。预测完成后，可靠度加权的 target
one-hot 伪样本通过精确 Woodbury RLS 加到该专家继承的源统计上；不保存图片或逐样本特征，
也不新增阈值、学习率、epoch、融合系数或 target 参数。

最终输出只使用所选专家的 Ridge margin，并沿用源阶段冻结 temperature；不再把 Base 与 Ridge
做 probability 平均，也不让 GMM posterior 直接参与预测。未适应的新专家严格等于新的 Source
Ridge，持续学习只是在同一特征坐标和同一解析目标上从源充分统计继续累积。R23 首轮固定为
GenImage matched-JPEG seed1 的 Source-Ridge static 与完整 CTTA 成对 pilot，不据此作正式
三 seed 或跨数据集结论。

R23 pilot 已在 4090-2 上由固定提交 `2ec3fc6` 完成。新的 Source-Ridge checkpoint 使用
323997 个有效源样本，real/fake 统计质量为 162000/161997；解析统计哈希为
`074c08d4f52657e3a0fb1143f530741e6bf77dd439689cddefb38f6bdcf42c79`，部署头与保存权重的
最大绝对误差为 `2.47e-8`。static 与 CTTA 均完整评测相同的 10500 个在线样本，样本 manifest
哈希一致。Source-Ridge static 的 target-macro online Accuracy/AUC 为
65.0667%/75.4417%，R23 为 65.4000%/78.8682%，即 Accuracy 提高 0.3333 个百分点，AUC
提高 **3.4266 个百分点**。逐 target AUC 在 BigGAN、ADM、VQDM 和 Midjourney 分别提高
9.5504、5.2692、5.3235 和 3.6037 个百分点；wukong 仅下降 0.0014 个百分点。

这次 AUC 增益确实来自持续解析更新而不是静态重映射：4 个专家完整继承各自的 323997 个源
样本统计，657 次预测后更新吸收 10274.22 的 target 有效质量，解析求解失败和冷启动批次均为
0；固定过去域上的 mean current AUC gain 为 3.3642 个百分点，mean future AUC gain 为
1.3457 个百分点。不过新 Source-Ridge 本身明显弱于此前 LoRA 神经网络头 Source 的
68.3143%/82.5031%，所以 R23 绝对值仍低约 2.91/3.63 个百分点。当前实验同时改变了 LoRA
特征训练时的分类坐标和最终分类头，不能把 static 退化单独归因于 Ridge。下一轮应保持此前
成功的 LoRA 特征训练协议不变，只在源 Ridge 拟合时切换到统一归一化坐标，再原样复用 R23
在线过程；这不增加任何 target 超参数，也能隔离真正的源阶段瓶颈。

R23 随后由固定提交 `78d0438` 在其余三个 matched-JPEG 数据集上完成同 checkpoint、同 seed1、
同锁定流的 Source-Ridge static/R23 成对运行；AIGCDetectionBenchmark 使用 4090-1，
AIGI-Holmes P3 与 OpenSDID Global 使用 4090-2。运行前没有按数据集修改方法参数，三组运行的
Static/R23 在线 manifest 与最终 holdout manifest 哈希分别完全一致。target-macro online 结果为：

| 数据集 | Static Acc. | R23 Acc. | Delta Acc. (pp) | Static AUC | R23 AUC | Delta AUC (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GenImage | 65.0667 | 65.4000 | +0.3333 | 75.4417 | 78.8682 | +3.4266 |
| AIGCDetectionBenchmark | 61.1059 | 60.9098 | -0.1961 | 69.9596 | 71.6960 | +1.7364 |
| AIGI-Holmes P3 | 59.9000 | 57.9733 | -1.9267 | 78.5407 | 72.5748 | -5.9659 |
| OpenSDID Global | 76.1333 | 77.4133 | +1.2800 | 91.2638 | 90.4394 | -0.8244 |
| 四数据集宏平均 | 65.5515 | 65.4241 | -0.1274 | 78.8014 | 78.3946 | -0.4068 |

因此 R23 不通过四数据集在线 Accuracy/AUC 联合晋级条件。它在 GenImage 与
AIGCDetectionBenchmark 上确实提高排序，在 OpenSDID 上提高 Accuracy，但 AIGI-Holmes P3 的
负迁移抵消了这些收益。这个失败又不是“Ridge 没学到”：四数据集最终固定 holdout 的宏平均
Accuracy 从 65.6048% 提高到 65.7509%，AUC 从 78.6947% 提高到 79.9287%，分别增加
0.1461 和 1.2340 个百分点；仅 AIGI 的最终 holdout AUC 也从 78.9462% 增至 79.3221%，与其
在线 AUC 下降 5.9659 个百分点形成明显反差。也就是说，当前解析分类器最终可以形成有用方向，
但其因果在线轨迹不稳定，不能把流结束后的改善冒充在线 CTTA 成功。

四组运行共执行 3662 次预测后解析更新，覆盖 58436 个候选样本，吸收 56975.90 的 target
有效质量，求解失败和 cold start 都为 0。事后用隐藏标签以外部 evaluator 只读诊断发现，GMM
赋予 fake 的有效质量占比在 GenImage、AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID
上仅为 18.16%、14.36%、11.20% 和 27.54%；合计只有 15.93%，而四条评测流实际均为平衡
二分类。R23 相对 Source-Ridge 共改变 2804 个在线硬决定，其中 2400 个是 fake 到 real，只有
404 个是 real 到 fake。AIGI 的 Janus-Pro-7B 与 Show-o AUC 分别下降 20.9631 和 19.3228 个
百分点；AIGC 的首个 ProGAN target 也下降 10.8837 个百分点，所以问题不能只归咎于历史专家
路由。真正的薄弱处是：连续可靠度可以压低边界附近样本，却无法阻止一个错误但尖锐、类别质量
严重失衡的 GMM posterior 持续用偏向 real 的 cross-covariance 更新 target Gram，并改变 Ridge
的样本级排序方向。

在改写 Ridge 监督前，先用
`ascal_gmm_segmented_memory_posterior_feature_routed_source_ridge_gmm_readout` 做了一个只回答
“当前专家 GMM 自己能分到什么程度”的 **R24 / ASCAL-JMP-SourceRidgeGMMReadout** 诊断。
R24 完整保留 R23 的 source checkpoint、CLIP 特征路由、GMM、分段、记忆、伪监督和 Ridge
状态更新；Ridge 只在后台影子更新，绝不进入最终 logit。所选历史 GMM 仍先产生 hard real/fake
决定，最终概率恢复 R12 的 ordinal readout：GMM 决定落在 `[0, 0.5)` 还是 `[0.5, 1]`，冻结
Source-Ridge 概率只负责各区间内部排序。因此阈值 0.5 的 Accuracy 直接衡量 GMM 决策，而不是
专家 Ridge 分类。

R24 已由固定提交 `aaa2584` 在 4090-1 上完成 GenImage matched-JPEG seed1。Static、R23 与
R24 的 target-macro 在线结果为：

| 读出 | Accuracy | AUC | 相对 Static Accuracy (pp) | 相对 Static AUC (pp) |
| --- | ---: | ---: | ---: | ---: |
| Source-Ridge Static | 65.0667 | 75.4417 | 0.0000 | 0.0000 |
| R23 expert Ridge | 65.4000 | 78.8682 | +0.3333 | +3.4266 |
| R24 expert GMM | **66.7714** | 75.3538 | **+1.7048** | -0.0879 |

R24 在 10500 个在线样本中有 10484 个使用旧 GMM，只有首批 16 个走 source fallback；四个
专家完成 658 次 GMM refit 且没有拟合失败。GMM 相对 Source-Ridge 改变 371 个硬决定，其中
275 个修正了 Source 错误，96 个破坏了 Source 正确决定，净增 179 个正确样本。逐 target
Accuracy 在 BigGAN、ADM、glide、VQDM、wukong 和 Midjourney 分别提高 0.47、0.47、1.40、
2.20、0.80 和 6.80 个百分点，仅 SD v1.5 下降 0.20；但 target-macro AUC 基本不变。这说明
当前 **Source-Ridge score 坐标上的 GMM** 学到了一定的阈值/类别分界能力，却没有提供更好的
样本级排序；这个结论不能外推成“此前成功的 GMM 专家只能提升 1.70 个百分点”。

旧的神经网络 Source checkpoint `f7a351...` 上，GenImage Source Accuracy/AUC 为
68.3143%/82.5031%，R01 GMM projection 为 76.2667%/82.1521%，R12 GMM ordinal route 为
76.8095%/82.2337%；因此 R01、R12 的 Accuracy 分别真正提高 **7.9524** 和 **8.4952** 个
百分点。即便改用 R22 的 CLIP 特征路由，其最终 Ridge 融合前的内部 GMM 决策仍有
75.8762% Accuracy，相对旧 Source 提高 **7.5619** 个百分点。R24 换成了新的解析
Source-Ridge checkpoint `ccef3f...`，其 Static Accuracy/AUC 已降至 65.0667%/75.4417%；更
关键的是 GMM 拟合坐标也从旧神经网络 margin 换成了这个较弱的 Source-Ridge margin。R12
在平衡 GenImage 流上的 predicted-fake rate 为 39.3429%，R24 只有 18.2762%；R24 的宏平均
fake recall 也只有 35.0476%，同时 real recall 达到 98.4952%，说明主要退化是新 score 坐标
让 GMM 严重塌向 real，而不是 GMM 专家结构本身失效。

这个对照只改变最终读出：R23 与 R24 的在线、holdout manifest 哈希完全一致；对真实 658 个
batch 的 172 个因果状态字段逐项比较，91073 个数值在 `rtol=atol=1e-12` 下无一不一致，类别
状态也零差异，最大绝对差仅为累计能量上的 `2.91e-11` 跨机浮点误差。R24 比 R23 的 Accuracy
高 1.3714 个百分点、AUC 低 3.5144 个百分点，因而把两个模块的作用分开了：GMM 分界更利于
当前阈值分类，Ridge 学到了额外排序方向，但现有 Ridge 读出会牺牲部分 GMM 的正确硬决定。
R24 的最终 holdout Accuracy/AUC 相对 Static 为 -1.0571/-0.0075 个百分点，所以它仍只是
seed1 诊断，不进入正式三 seed 主结果。R24 只适合回答“R23 当前状态中的 GMM 还能剩下多少
能力”，不能替代旧 R12/R22 对 GMM 专家上限的结论；当前首要问题应定位为 Source-Ridge
替换后 GMM score coordinate 的类别可分性下降。

`ascal_gmm_segmented_memory_posterior_feature_routed_gaussian_replay_mlp` 的研究版本名为
**ASCAL-JMP-GaussianReplayMLP（R25）**。它回到旧的神经网络 LoRA Source checkpoint 与
R22 的 CLIP 特征路由，不再强制 Ridge 充当专家分类器。每个专家仍用预测时选中的旧 GMM
posterior 产生 hard 伪类和连续可靠度，但最终预测不读取 GMM posterior；可靠目标 CLIP 特征
只用于在线更新 real/fake 两个类别条件对角高斯的样本数、质量、均值和二阶中心矩，随后立即
丢弃。每批预测完成后，从两类累计高斯中等量采样与当前 batch 同量的伪特征，对该专家独立的
`768 -> 64 -> 1` GELU MLP 做一次 Adam 更新。MLP 输出层从零初始化，正式预测统一为冻结
Source logit 加所选专家 MLP residual logit，因此新专家和未就绪专家严格等于旧 Source，且不
存在概率平均、融合系数或 GMM/Ridge/Source 三路读出。

R25 是固定的 GenImage matched-JPEG seed1 诊断而非正式三 seed 结果。它只保存每类对角高斯
充分统计、路由原型、MLP 与 optimizer 状态，不保存图片、逐样本特征或固定容量 replay bank；
平衡生成回放也不继承当前 stream 的 observed class ratio。首版固定 hidden dimension 64、
Adam learning rate `1e-3`、每个已预测 batch 一步更新，回放量直接由当前 batch size 决定；
不使用置信度阈值、epoch、memory capacity 或 target label 选择这些参数。首版故意只验证
“分布式特征回放能否训练出提升排序的专家头”，若单高斯的伪特征流形质量不足，再把 fake
分布扩成多分量，而不在同一版本同时增加结构。

R25 已由固定提交 `f9e3927` 在 4090-2 上完成 GenImage matched-JPEG seed1。旧神经
Source static、R12 与 R25 的 online manifest 哈希均为 `891a3eba...`，最终 holdout
manifest 哈希均为 `20beba40...`，因此下表使用完全相同的样本身份和顺序。指标均为
target-macro，不使用 pooled AUC：

| 方法 | Online Accuracy | Online AUC | Final-holdout Accuracy | Final-holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| 旧神经 Source static | 68.3143 | 82.5031 | 67.8000 | 81.6334 |
| R12 GMM ordinal route | **76.8095** | 82.2337 | 74.2571 | 81.4473 |
| R25 Gaussian replay MLP | 74.8667 | **82.7725** | **77.4571** | **83.5714** |

R25 相对 Source 的因果 online Accuracy 提高 **6.5524** 个百分点，target-macro AUC
提高 **0.2694** 个百分点，说明“高置信伪特征分布 + 生成回放 + 样本级 residual”的
确能在保留大部分 Accuracy 收益的同时改变排序。但它尚未取代 R12：相对 R12，R25
online AUC 提高 0.5388 个百分点，Accuracy 却下降 1.9429 个百分点。最终固定 holdout
上 R25 相对 Source 的 Accuracy/AUC 分别提高 9.6571/1.9381 个百分点，相对 R12
分别提高 3.2000/2.1241 个百分点；这证明最终学到的专家状态有效，但不能用这个
流结束后的结果代替更严格的 online 结论。

逐 target 看，R25 相对 Source 的 AUC 在 ADM、VQDM 和 BigGAN 上分别提高
6.2705、4.7301 和 0.6378 个百分点，但在 glide 和 Midjourney 上分别下降 6.4060 和
3.2764 个百分点，表明单个对角高斯对部分 fake 域的多模态特征仍然过于粗糙。整条流创建
3 个特征路由专家，658 次 GMM refit 无失败；三个 cold-start batch 后有 654 个 batch
实际用生成伪特征更新 MLP，共生成 10484 个临时伪特征，最终三个专家全部就绪。
因此 R25 是值得保留的排序学习候选，但仍只是 seed1 诊断，不进入正文正式表。

`ascal_gmm_segmented_memory_posterior_feature_routed_expanded_gaussian_replay_mlp`
的研究版本名为 **ASCAL-JMP-ExpandedGaussianReplay（R26）**。它只修改 R25 的生成
回放量：每次专家适应从当前累计 real/fake 对角高斯中分别独立采样 128 个伪特征，
得到 256 个当次新生成且平衡的训练样本。它们按当前 stream batch size 分批，只遍历
一次，每个生成特征恰好用于一次 minibatch；不是对 R25 的同一组 16 个特征重复训练。
冻结 Source、CLIP 特征路由、GMM 伪监督、类条件高斯统计、`768 -> 64 -> 1` 专家
MLP 及 `Base logit + residual logit` 预测全部不变。每个专家只在创建时以零 residual
出生，不复制 Base 参数；后续只累积更新当前路由专家，其他专家与 Base 均不受梯度修改。
256 是运行 seed1 前固定的唯一新计算预算，不根据 target label 或结果调整。

R26 已由最终固定提交 `ae29fa3` 在 4090-2 上完成 GenImage matched-JPEG seed1。
Source、R12、R25 和 R26 使用相同的 online manifest `891a3eba...` 与 holdout
manifest `20beba40...`。下表仍只报告 target-macro 指标，不用 pooled AUC：

| 方法 | Online Accuracy | Online AUC | Final-holdout Accuracy | Final-holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| 旧神经 Source static | 68.3143 | 82.5031 | 67.8000 | 81.6334 |
| R12 GMM ordinal route | 76.8095 | 82.2337 | 74.2571 | 81.4473 |
| R25 16-sample Gaussian replay | 74.8667 | 82.7725 | 77.4571 | **83.5714** |
| R26 256-sample distinct replay | **78.8095** | **84.4869** | **78.6286** | 83.4781 |

R26 相对 Source 的 online Accuracy/AUC 分别提高 **10.4952/1.9838** 个百分点；
相对 R12 分别提高 **2.0000/2.2531** 个百分点，相对 R25 分别提高
**3.9429/1.7143** 个百分点。它因而是当前 GenImage seed1 上同时兼顾 Accuracy 和
AUC 的最强在线候选。最终 holdout 相对 R25 的 Accuracy 还提高 1.1714 个百分点，
但 AUC 微降 0.0934 个百分点；这个很小的流结束后差异不改变 R26 的主要 online 结论。

R25 与 R26 的 658 个批次上，84 个路由、分段、GMM 和特征统计字段逐项完全一致，
不一致字段数为 0。因此这个改善可以归因于唯一改动的专家训练量，而不是路由或 GMM
轨迹巧合变化。R26 共完成 657 次专家回放更新，生成 168192 个当次独立伪特征，
分成 10554 个只遍历一次的 minibatch optimizer step；三个专家全部就绪。它的宏平均
fake/real recall 为 68.9905%/88.6286%，同时高于 R12 的 66.1524%/87.4667%，
说明大量新采样不只是移动阈值，而是训练出了更有效的样本级 residual。R26 仍只是
seed1 候选诊断，在其他数据集和 seed2/seed3 验收前不进入正文正式表。

R26 之后的凝练采用逐项向后消融，而不是同时删除多个组件。首个候选
**ASCAL-JMP-SharedResidualHead（R27）**只把三个专家各自的 residual MLP 与 Adam
状态替换为一个全局共享的 `768 -> 64 -> 1` residual MLP；每专家的分段、历史 GMM、
CLIP 路由原型、real/fake 对角高斯充分统计、256 个独立平衡回放样本以及
Predict-Then-Adapt 顺序全部保持 R26 不变。当前 batch 仍先路由并由选中专家的 GMM
与特征分布产生伪监督，但生成样本统一更新共享 head，推理为 `Base logit + shared
residual logit`。R27 只回答“多个分类 head 是否必要”，只有 online target-macro
Accuracy 不低于 78.8095 且 AUC 不低于 84.4869 时才接受；否则恢复 R26 的专家专属
head，再消融下一项。该门槛在运行前固定，不读取 target label 调参。

R27 已由固定提交 `cdddf3b` 在 4090-1 完成：online target-macro Accuracy/AUC 为
**77.8095/84.4293**，final-holdout 为 **76.6571/82.1957**。相对 R26，online
Accuracy/AUC 分别下降 **1.0000/0.0575** 个百分点，final-holdout 分别下降
**1.9714/1.2824** 个百分点，因此严格拒绝“共享一个 residual head”，R26 的每专家
独立 head 与 optimizer 必须保留。跨服务器按 `1e-10` 数值容差审计的 658 个 batch、
154 个共同非 head 语义字段没有不一致，说明下降来自 head 共享，而不是路由、分段、GMM
或 feature memory 轨迹改变。该结论仍只属于 GenImage matched-JPEG seed1 诊断。

与 R27 并行的第二个单变量候选 **ASCAL-JMP-LinearResidualHead（R28）**从 R26
出发，只把每专家 `768 -> 64 -> 1` GELU residual MLP 改为零初始化的 `768 -> 1`
线性 residual；专家数量、路由、GMM、高斯统计、回放量、Adam 学习率和最终加法均不变。
为避免网络结构改变连带改变后续伪特征，线性 head 出生时会消耗并丢弃与 R26 隐层初始化
完全相同数量的随机数，后续 Gaussian replay 抽样保持对齐。R28 只回答“非线性隐藏层是否
必要”，使用与 R27 相同的 R26 双指标非劣门槛，不能与共享 head 的效果混为一次改动。

R28 已由固定提交 `b287e04` 在 4090-2 完成：online target-macro Accuracy/AUC 为
**77.3238/83.8303**，final-holdout 为 **78.3714/84.1525**。虽然 holdout AUC
比 R26 高 0.6735 个百分点，但主要 online Accuracy/AUC 分别下降 **1.4857/0.6565**
个百分点，因而仍按预注册门槛拒绝；不能用流结束后的单个次要提升覆盖 online 双指标失败。
R26 与 R28 的 658 个 batch 上，154 个共同非 head 语义字段逐项精确一致，说明线性 head
确实欠拟合在线样本级修正，`768 -> 64 -> 1` GELU 非线性层应保留。

第三个单变量候选 **ASCAL-JMP-UniformConfidence（R29）**仍从 R26 出发，保留完全
相同的 GMM hard pseudo-label，但把 `|2p-1|` 连续可靠度替换为每个样本权重 1；其余
路由、分段、每专家统计、256 回放和 MLP 均不变。它用于直接判断 GMM posterior 的
连续置信度是否是必要组件，而不是把 GMM 整体删掉。R29 同样按 R26 双指标非劣门槛
独立判定；若下降，就保留连续可靠度并继续从最近被接受的骨架做下一项消融。

R29 已由固定提交 `f1f18ef` 在 4090-1 完成：online target-macro Accuracy/AUC 为
**78.8095/84.4713**，final-holdout 为 **78.6000/83.3663**。其 online Accuracy
与 R26 完全相同，但 AUC 下降 **0.0156** 个百分点；final-holdout Accuracy/AUC 也分别
下降 **0.0286/0.1118** 个百分点，因此按运行前固定的双指标非劣门槛拒绝。按 `1e-10`
容差比较 658 个 batch 的 131 个非置信度、非 head 语义字段，不一致数为 0；变化确实来自
把连续 GMM reliability 改成单位权重。虽然差距很小，也不能在看到结果后放宽门槛，故
`|2p-1|` 连续可靠度保留为核心组件。

第四个候选 **ASCAL-JMP-ActiveOnly（R30）**原本计划保留 R26 的分段与 shadow 档案，
只允许当前 active candidate 进入预测和适应，从而检验历史专家路由是否必要。完成后做逐批
状态审计才发现：R30 确实删除了每个 batch 的 CLIP feature memory candidates，但继承的
segment-change callback 仍会按 source-score GMM 描述长度召回旧 memory，再把旧 head
挂到名为 `active_learning_state` 的候选上。因此它实际只是“无逐批 feature router”，不是
“无历史召回”，不能用来决定删除路由或 memory。

R30 的固定提交 `d391e46` 在 4090-2 得到 online target-macro Accuracy/AUC
**79.0095/84.6232**，表面上相对 R26 提高 **0.2000/0.1363** 个百分点；但 658 个
batch 中发生了 3 次 score-based memory recall，并有 **280** 个 batch 的 active candidate
实际绑定历史 memory state。其 final-holdout Accuracy/AUC 还下降 **2.8571/1.0776**
个百分点。故 `run_record.json` 将“指标门槛通过”和“消融有效”分开记录：前者为真，后者为
假，R30 不晋级，也不能作为路由可删的证据。

修正后的 **ASCAL-JMP-NoHistoricalRecall（R34）**同时关闭两个历史入口：每批候选列表
不含 archive，segment-change callback 也永远不能选回旧 score memory。完成段仍只写入
shadow archive 供审计，随后启动全新的 current-segment GMM/feature distribution/MLP；
旧状态不再被预测或适应读取。R34 不增加阈值或相似度，其他 GMM 置信度、256 平衡 Gaussian
replay、非线性 head 与 Predict-Then-Adapt 顺序均保持 R26。只有 R34 通过 R26 的 online
Accuracy/AUC 双非劣门槛，才能继续做“连 shadow archive 也删除”的下一步消融。

R34 已由固定提交 `8501b3d` 在 4090-2 完成：online target-macro Accuracy/AUC 为
**78.8571/84.5472**，相对 R26 分别提高 **0.0476/0.0604** 个百分点，因而通过预注册
的 forward-online 双非劣门槛。逐批审计确认 658 个 batch 的 score-based recall、memory
selection、active memory index 和 prediction memory index 全部为 0。与此同时，
final-holdout Accuracy/AUC 降至 **73.6857/82.1075**，相对 R26 下降
**4.9429/1.3705** 个百分点；这说明历史召回不是标准单向 online 指标的必要组件，但对流结束
后的旧域保持很有价值。主方法凝练按预注册 online 门槛接受删除，历史专家库则保留为明确的
retention 扩展，而不能声称它对 forgetting 没有作用。

通过后继续运行 **ASCAL-JMP-CurrentSegmentCore（R35）**：R34 从不选中的五个 shadow
archive 也完全删除，段切换时直接丢弃旧 GMM/feature distribution/MLP，只保留一个当前段
状态；候选构造只计算当前 active GMM，不再遍历任何历史 mixture。GMM 软监督、连续可靠度、
256 个平衡 Gaussian replay、每段非线性 residual 和 Base 冻结均保持不变。由于 R34 的
archive 从未影响预测，R35 预期与 R34 逐样本完全等价；只有在线指标非劣且轨迹审计成立，
才能把它确定为不含历史路由与 memory 的最简 forward-online 核心。

R35 已由固定提交 `3139506` 在 4090-2 完成，online Accuracy/AUC 仍为
**78.8571/84.5472**，final-holdout 仍为 **73.6857/82.1075**，四个聚合值与 R34
逐位完全相同。排除有意删除的 archive 聚合统计、计时和方法名后，658 个 batch 的 148 个
共同语义字段按 `1e-10` 容差无任何不一致；流末 `memory_size=0`，五个完成段状态均已丢弃，
历史 selection/recall 都是 0。因此 shadow archive 被正式删除，R35 在当时仅含
online Accuracy/AUC 的门槛下晋级为最简 forward-online 消融；加入 forgetting 后的最终
三指标判定见下文，不能再把该结论外推为完整 CTTA 核心。

若 R35 验证通过，最后一个结构消融 **ASCAL-JMP-GlobalStreamCore（R36）**再删除参数
无关的 BIC change-point scan 与 current-state reset：一个 GMM、一个 real/fake Gaussian
feature distribution 和一个 residual MLP 累积整个因果流，其他监督、可靠度、平衡回放和
训练预算完全不变。它回答“分段本身是否必要”。R36 只有同时不低于 R35 的 online
target-macro Accuracy/AUC 才能继续简化；否则 R35 的因果分段 reset 就是 forward-only 核心的必要
组成，而不是为了路由历史专家留下的冗余模块。

R36 已由固定提交 `b6c8e06` 在 4090-2 完成：online target-macro Accuracy/AUC 为
**77.8857/84.5346**，final-holdout 为 **75.9714/82.6038**。相对 R35，online
Accuracy/AUC 分别下降 **0.9714/0.0126** 个百分点，因此即使 AUC 仍略高于 R26，也因
Accuracy 明确失败而拒绝；不能用 final-holdout 的 **+2.2857/+0.4962** 覆盖主 online
门槛。状态审计确认 658 个 batch 中 segment check/change 都为 0，始终只有一个 global
expert 且没有 memory，故失败确实回答了分段问题：参数无关的因果 BIC reset 必须保留。

第一阶段按 forward-online 双指标完成的逐项消融仍保留如下；它用于判断 GMM 教师、连续
reliability、Gaussian memory、平衡回放和非线性专家 head 是否可删，但不能单独决定完整
CTTA 核心：

| 版本 | 唯一消融 | Online Accuracy | Online AUC | 当时的前向判定 |
| --- | --- | ---: | ---: | --- |
| R26 | 完整参考 | 78.8095 | 84.4869 | 参考 |
| R27 | 每专家 head 改为共享 head | 77.8095 | 84.4293 | 拒绝 |
| R28 | 非线性 head 改为线性 head | 77.3238 | 83.8303 | 拒绝 |
| R29 | 连续 reliability 改为单位权重 | 78.8095 | 84.4713 | 拒绝 |
| R30 | 仅删逐批 feature candidates | 79.0095 | 84.6232 | 实现审计无效 |
| R31 | 累计 Gaussian replay 改为当前批重采样 | 77.1524 | 82.1093 | 拒绝 |
| R32 | 平衡 replay 改为伪类别 prior replay | 78.7524 | 84.4383 | 拒绝 |
| R33 | GMM 教师改为 Source 自举 | 73.2095 | 81.3834 | 拒绝 |
| R34 | 完全关闭历史召回 | 78.8571 | 84.5472 | 前向通过，retention 待审 |
| R35 | 删除未使用的 shadow archive | 78.8571 | 84.5472 | 前向通过，retention 待审 |
| R36 | 删除因果分段 reset | 77.8857 | 84.5346 | 拒绝 |

前述 R27--R36 最初只用 online Accuracy/AUC 做前向门槛；在把 continual retention
明确加入目标后，R35 不能再作为完整 CTTA 主方法。它虽然有 78.8571/84.5472 的 online
Accuracy/AUC，但 final-holdout 只有 73.6857/82.1075，average AUC forgetting 为
**2.9163 个百分点**。历史专家不能再被称为可选扩展，而必须纳入三指标核心。重新按
“online Accuracy、online AUC 均不下降，average AUC forgetting 不增加”的门槛凝练后，
新增了两个只改变一项机制的版本。

**ASCAL-JMP-CLIPExpertMemory（R37）**从 R26 出发，只关闭 BIC 段变化时继承的
source-score/GMM 历史 memory 搜索。当前 batch 的历史身份选择只比较冻结 CLIP 特征：
先将归一化特征投影到 source real/fake 分类方向的正交子空间，再选择 batch 平均余弦相似度
最大的 active 或历史专家。GMM 仍负责无标签 BIC 分段、所选专家的 hard pseudo-label 和
连续 `|2p-1|` reliability，但不参与历史身份路由，也不进入最终预测；正式输出仍为
`sigmoid(Base logit + selected expert residual logit)`。预测结束后才用同一个 CLIP 选中
专家更新类条件 Gaussian 统计和 MLP，保持 Predict-Then-Adapt。

R37 由固定提交 `437aeb1` 在 4090-2 完成。它与 R26 使用完全相同的 online/holdout
manifest，score-based segment-change memory comparison 在全部 658 个 batch 中均为 0。
除方法名和被关闭入口的四个诊断字段外，其余非计时预测、路由、GMM、回放和专家状态轨迹
逐项一致；所有 Accuracy、AUC、final holdout 和 forgetting 数值也逐位一致。流末仍只有
**3 个专家身份**，657 次 batch 路由中有 184 次选择历史 memory，产生 166 次历史专家
visit；这些 visit 是 CLIP 特征路由，不是 source-score 路由。

**ASCAL-JMP-SegmentExpertMemory（R38）**进一步检验能否把“CLIP 选中哪个专家”与
“当前 BIC 分段学习状态”完全解耦：历史专家仍可预测并接收当前 batch 更新，但逐批路由不再
切换 active score state，只有 BIC change-point 才能创建新状态。这个改动没有增加阈值、
hysteresis、cooldown 或 memory cap，却没有通过三指标门槛。R38 抑制了 314 次路由状态
handoff，最终形成 5 个已完成 memory 加 1 个 active expert，6 个 head 全部就绪，专家参数
从 R37 的 147843 增至 295686；online Accuracy/AUC 降至 78.6095/84.4782，
final-holdout 降至 77.6000/83.1221，average AUC forgetting 增至 **1.2489 个百分点**。
因此“路由专家也是当前持续学习状态”不是可删除的偶然耦合：它使再次到来的相似 batch
继续访问同一专家，而不是按每个 BIC 段重复创建相近专家。

重新按三指标审视后的关键结果如下；AUC 和 Accuracy 均为 target-macro，forgetting 是固定
holdout 上的 average AUC forgetting：

| 版本 | 唯一变化 | Online Accuracy | Online AUC | Final Accuracy | Final AUC | Forgetting | 三指标判定 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Source | 冻结源模型 | 68.3143 | 82.5031 | 67.8000 | 81.6334 | 0.0000 | 无学习对照 |
| R26 | 原完整历史版本 | 78.8095 | 84.4869 | 78.6286 | 83.4781 | 0.7752 | 参考 |
| R35 | 删除全部历史专家 | **78.8571** | **84.5472** | 73.6857 | 82.1075 | 2.9163 | 拒绝：严重遗忘 |
| R36 | R35 再删除 BIC 分段 | 77.8857 | 84.5346 | 75.9714 | 82.6038 | 1.6696 | 拒绝 |
| **R37** | **删除 score-based 历史召回** | **78.8095** | **84.4869** | **78.6286** | **83.4781** | **0.7752** | **接受：当前核心** |
| R38 | 路由与 active BIC 状态解耦 | 78.6095 | 84.4782 | 77.6000 | 83.1221 | 1.2489 | 拒绝 |

因此当前三指标核心确定为 **R37**，不是 R35。它可以凝练为四个有因果关系的模块：

1. 冻结 Base，只提供不可漂移的 source logit 与 CLIP feature；
2. 在线 GMM/BIC，只负责发现新段，并为到达样本产生 pseudo-label 与连续可靠度；
3. CLIP feature prototype memory，只负责在 active 与历史专家间无标签路由；
4. 每专家 real/fake 对角 Gaussian 充分统计与零出生 `768 -> 64 -> 1` residual MLP，使用
   每类 128 个当次新采样伪特征持续更新，最终只输出 `Base logit + one residual logit`。

专家数不是预先指定的，也没有 capacity 超参数：BIC 只在无标签流中提出新状态，CLIP 路由
负责复用已有身份；本次锁定流最终增量得到 3 个专家。方法不保存原图或逐样本特征，不读取
target label，不使用 route threshold、confidence threshold、融合系数或 memory cap。方法级
可见量仍只有 hidden dim 64、学习率 `1e-3` 和每批 256 个新 replay 样本，`1e-6` 只是
方差数值下界。以上仍是 GenImage matched-JPEG seed1 候选诊断；四数据集、三个正式 seed
全部验收前，R37 不进入论文正文数值表。

针对 R37 在跨数据集检查中暴露出的排序瓶颈，后续候选已在查看新结果前固定为三个小改动，
不再改路由、GMM、分段、专家记忆或最终 `Base + one expert residual` 结构。
**ASCAL-JMP-DecoupledRank（R39）**把一个专家输出拆成标量校准 bias 与特征相关 residual：
平衡 BCE 在 residual `detach` 后只更新 bias，所有 replay minibatch 内的 pseudo-fake/real
样本对则用无 margin 的 pairwise logistic loss 只更新特征权重；两项 mean loss 固定等权，
不增加可调混合系数。**ASCAL-JMP-ConservativeRank（R40）**只把 R39 学习率从 `1e-3`
降为 `3e-4`；**ASCAL-JMP-CompactRank（R41）**只把 R39 隐藏宽度从 64 降为 32。
四个 matched-JPEG 数据集共享同一组三版本 seed1 screen 配置；它们是预注册的探索候选，
不得根据正式 target label 继续细扫并把最优行伪装成无偏主结果。

直接 pseudo-pair 排序不再继续扩展；后续最小候选回到 R37 的平衡 replay BCE，并只在
Predict 阶段约束不可靠的样本级修正。**ASCAL-JMP-GMMConfidenceGate（R42）**将专家
输出拆成标量 bias 与特征 residual，使用所选专家已有 GMM 的连续可靠度
`c(s)=abs(2p_GMM(s)-1)`，最终输出
`source logit + bias + c(s) * feature residual`。GMM 不提供额外分类分数，bias 始终保留，
只有会改变排序的特征 residual 被无阈值门控；当前 batch 仍先预测、后更新。
**ASCAL-JMP-GMMConfidenceGateSquared（R43）**只把门从 `c` 改为 `c^2`，作为更保守的
单变量敏感性检查。两者的 R37 训练目标、CLIP 路由、BIC 分段、Gaussian replay、学习率、
隐藏宽度和专家记忆完全一致，不增加可学习融合系数或读取 target label。

进一步的单变量候选 **ASCAL-JMP-OrthogonalResidual（R44）**不改 R37 的训练目标和最终
`Base + residual` 结构，只把专家 MLP 的输入投影到冻结 Source 二分类头方向的正交补空间，
并重新归一化；Base logit 仍完整保留原真假方向。该坐标与 R37 已用于专家路由的 CLIP
坐标完全相同，因此不增加投影维度、阈值或融合超参数。它检验当前 residual 是否主要在
重复拟合 Base score；若有效，新增排序只能来自 Source 尚未使用的 CLIP 特征信息。

**ASCAL-JMP-WithinClassOrderGuard（R45）**保留 R44 的完整预测式，只在 replay 训练损失中
加入 real 与 fake 两个伪类别各自的 residual 方差均值。该项中的标量 expert bias 自动抵消，
两类 residual 均值之间的距离也不受约束，因此阈值校准和类间分离仍由原平衡 BCE 完整学习；
受抑制的只有同一伪类别内部、没有额外证据支持的样本重排。方差项固定单位系数且不暴露为
配置，不引入 pairwise 伪排序、margin、confidence threshold 或 target label。

四数据集检查表明 R45 的方差约束过强：它能保护 AIGI-Holmes P3 与 OpenSDID 中原本较强的
Source 排序，却也惩罚了 GenImage 与 AIGCDetectionBenchmark 中有用的类内 residual 变化。
因此 **ASCAL-JMP-NonInversionGuard（R46）**回到 R37 的完整特征输入，只在同一 replay
伪类别内最终 logit 真正翻转 Source 两两顺序时施加零 margin hinge；顺序未翻转时损失严格为
零。expert bias 和整类平移仍在 pair difference 中自动抵消，BCE、路由、GMM、回放量与
预测式均不改变，也不新增阈值、margin 或可调 loss 权重。

第五个单变量候选 **ASCAL-JMP-CurrentBatchReplay（R31）**保留 R26 的 256 样本平衡
训练预算、路由、GMM 伪标签、连续可靠度和专家 MLP，但训练特征不再从累计 real/fake
Gaussian memory 生成，而只在当前 batch 的两类伪特征中按可靠度有放回采样；当前批缺少
任一伪类别就跳过该次 head 更新，样本随后立即丢弃。类条件高斯统计仍以 shadow 状态更新，
只为严格审计其他轨迹，不进入训练。它单独检验累计分布式 feature replay 是否必要；若
R31 下降，则 Gaussian memory 与生成回放应保留在凝练后的核心方法中。

R31 的固定提交 `ed10158` 因 4090-1 中途离线而由同一源码归档在 4090-2 完成重跑：
online target-macro Accuracy/AUC 为 **77.1524/82.1093**，final-holdout 为
**75.8571/82.1958**。相对 R26，online 分别下降 **1.6571/2.3775** 个百分点，
final-holdout 分别下降 **2.7714/1.2823** 个百分点；online AUC 甚至比 Source 低
0.3938 个百分点，因此明确拒绝 current-batch resampling。该流有 44 次因当前批缺少一个
伪类别而跳过 head 更新；排除这三个有意变化的更新计数字段后，658 个 batch 的 151 个共同
非训练语义字段按 `1e-10` 容差完全一致。累计 class-conditional Gaussian 充分统计与从中
持续生成新伪特征，是克服小 batch 覆盖不足的必要核心，而不是可删缓存。

第六个单变量候选 **ASCAL-JMP-PriorReplay（R32）**继续使用 R26 的累计 Gaussian
feature memory，但每次 256 个回放样本不再 real/fake 各半，而按该专家累计的可靠度加权
伪类别质量分配，并只保留每类至少一个样本以维持二分类目标。它检验 equal-class replay
是否是抵抗在线伪类别塌缩的必要组件；不读取真实 target class prior，也不改变 GMM、路由、
head 或总训练预算。R32 若下降，就证明平衡回放应作为核心设计而不是可删实现细节。

R32 已由固定提交 `0338d1a` 在 4090-2 完成：online target-macro Accuracy/AUC 为
**78.7524/84.4383**，final-holdout 为 **78.5429/83.3038**。相对 R26，online
分别下降 **0.0571/0.0486** 个百分点，final-holdout 分别下降 **0.0857/0.1743**
个百分点，故严格拒绝按累计伪类别质量分配 replay。R26 与 R32 的 658 个 batch 上，154 个
非 replay-readout 语义字段按 `1e-10` 容差完全一致；下降可归因于唯一改变的类采样比例。
虽然差距不大，结果仍说明显式 real/fake 等量回放在当前不平衡伪标签流中是必要保护，必须
保留在凝练核心中。

第七个也是本轮最后一个基础组件消融 **ASCAL-JMP-SourceSupervision（R33）**保留
R26 的 GMM 分段与专家身份，但不再让选中专家 GMM 给 feature memory 提供 posterior；
伪标签与连续可靠度都改为冻结 Source probability 的 `0.5` 决策和 `|2p_source-1|`。
路由、Gaussian replay、平衡训练和每专家 MLP 全部不变。它直接检验 GMM 教师是否只是
可替换的实现细节。R27--R33 完成后，才依据逐项结果组合被证明可删的组件并复跑最终凝练版。

R33 的固定提交 `251158d` 同样在 4090-2 完成：online target-macro Accuracy/AUC 为
**73.2095/81.3834**，final-holdout 为 **72.7143/80.1045**。相对 R26，online
分别下降 **5.6000/3.1035** 个百分点，final-holdout 分别下降 **5.9143/3.3736**
个百分点；online AUC 还比 Source 低 1.1197 个百分点，因而是明确失败而非边界波动。
R26 与 R33 的 658 个 batch 上，131 个非 feature-supervision/head 语义字段按 `1e-10`
容差完全一致，说明退化来自冻结 Source 自举标签，而不是 GMM、分段或路由轨迹变化。专家
GMM 的 equal-prior posterior 与连续可靠度是纠正 Source 整体偏移、建立两类 feature
distribution 的必要教师，不能被 Source 自己的概率循环替代。

ASCAL 诊断迭代采用不可复用的 `Rxx + 研究名 + method id`：每轮只允许一个结构变化，设计
依据只读取无标签在线状态，seed1 指标只用于候选晋级，不能据此添加逐数据集规则；每版必须
固定配置、commit、源码归档和 `run_record.json`。边界校准版本沿用 Accuracy 与
target-macro AUC 均不下降的严格 Pareto 门槛；从 R06 开始，排序 residual 分支在运行前固定为
Accuracy 非劣、AUC 优越门槛：四数据集宏平均 Accuracy 相对 R01 最多下降 0.2 个百分点，
target-macro online AUC 相对 R01 至少提升 0.1 个百分点且必须超过此前 residual 候选中的
最高值，R06 的直接比较对象是 R05，R07 至 R21 的直接比较对象均是当前最佳 R06；R12 至
R21 还额外使用 R11/R12 的高 Accuracy 作为锚点。R16 至 R21 不再要求逐样本硬决定不变，而是预先
要求四数据集宏平均 Accuracy 严格超过 R12，同时 target-macro online AUC 超过 R06。
该门槛只决定是否进入 seed2/seed3，不进入方法推理，也不能在结果产生后按数据集修改；完整
确认前仍不得写入正文正式表。

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

## 实验边界

- CLIP 主结果只使用固定 OpenAI CLIP ViT-L/14 预训练权重，并锁定目标样本 identity、目标内顺序和 seed；每种方法保留原生 source training、分类器或 prompt 构造、batch/views、在线状态与预测/适应顺序。每个数据集分别报告逐 target AUC 和阈值 0.5 的 Accuracy；target 单元格为三 seed 均值，Mean 为 target-macro 均值及跨 seed 标准差。
- TENT、EATA 和 T2A 只允许把公开实现中的 BN 参数参照最小映射到 CLIP visual LayerNorm affine；其余方法逻辑不得重写。CoTTA 保留作者 ImageNet 分支的全参数 student/teacher 更新，只将其像素空间增强桥接到 CLIP 输入归一化。EATA 必须包含匹配公共源域 CLIP detector 的 Fisher。RoTTA-LN 显式以 visual LayerNorm affine 替代 RobustBN，但保留 CSTU、teacher/student EMA、熵目标和在线更新频率；由于没有 RobustBN 统计插值，它只能按迁移版本名称披露。
- CLIP-native 方法只共享类别语义，不共享人为固定的一句最终 prompt。IAPL 与 Ours 使用各自原生源训练并单独披露 source setup；TTC 在作者公开实现可固定前仅保留在 related work，不进入定量表。
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
