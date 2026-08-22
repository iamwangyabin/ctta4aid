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
| 公共源域 CLIP detector | 固定 ViT-L/14 初始化后，在同一源数据上训练的公共二分类 checkpoint | Source、TENT、EATA、SAR、CoTTA、LAME、T2A | 共享源 checkpoint，可在块内比较最佳结果 |
| CLIP-native | 未做任务微调的固定 ViT-L/14 checkpoint | Frozen CLIP、TDA、DynaPrompt、CLIPTTA、BATCLIP | 共享二分类类别语义，各自保留原生 template、文本分类器或 prompt learner |
| Method-specific source training | 固定 ViT-L/14 初始化后，按方法自己的源训练流程得到的 checkpoint | IAPL、TTC、Ours | source state 不同，只披露数值，不做跨块最佳排名 |

各方法的冻结运行约定如下。`BN -> LN` 只表示把公开实现中的归一化参数枚举映射到
CLIP LayerNorm affine；不得改写目标函数、筛选规则、teacher、Fisher、gradient masking
或在线状态。

| Method | 判别能力来源 | 保留的原生机制 | ViT-L/14 必要改动与状态 |
|---|---|---|---|
| TENT | 公共源域二分类 detector | 熵最小化及原生在线顺序 | `BN -> LN`，表格加脚注 |
| EATA | 公共源域二分类 detector | 可靠/非冗余筛选、熵最小化、Fisher 防遗忘 | `BN -> LN`；Fisher 必须由同一源 checkpoint 与源数据计算 |
| SAR | 公共源域二分类 detector | 可靠样本筛选、SAM 与恢复机制 | 使用作者公开的 ViT/LayerNorm 路径 |
| CoTTA | 公共源域二分类 detector | student/EMA teacher、增强平均、随机恢复 | 保留作者全参数更新；只将像素增强的归一化桥接为 CLIP 输入归一化 |
| RoTTA | 需要源域 detector | robust BatchNorm、memory 与 teacher | robust BatchNorm 是方法核心，不能最小迁移；结果单元格留空并加脚注 |
| LAME | 公共源域 detector 的特征与 logits | 参数无关的 Laplacian 输出适配 | 仅接入 CLIP 特征；保留其 batch contract |
| T2A | 公共源域二分类 detector | 不确定性选择、negative learning、gradient masking | 归一化梯度参照由 BN 最小映射为 LayerNorm，其他逻辑不动 |
| IAPL | IAPL 原生源训练得到的 CLIP detector | 32 views、2 steps、OIS、逐图 prompt/optimizer reset | 只替换统一数据接口，不改为公共 binary head |
| TDA | CLIP 原生文本分类器 | 正负 cache、无反向传播 | 使用二分类类别语义与作者的 template 构造 |
| DynaPrompt | CLIP 类别名与在线 context | 多视图 prompt tuning、动态 prompt buffer | 换成 ViT-L/14；不固定最终 prompt |
| CLIPTTA | CLIP 文本原型 | 官方 closed-set 对比适配及 batch 机制 | 使用二分类类别语义与作者原生文本构造 |
| BATCLIP | CLIP 图像与文本两端 | 双模态目标及原生在线更新 | 换成固定 ViT-L/14；不得退化成只更新视觉端 |
| TTC | 已训练 AIGC detector | 课程式伪标签适配 | 作者公开实现可固定前结果单元格留空并加脚注，不得自写后冒充复现 |
| Ours | PoundNet 的配对真实/伪造 prompt detector | 冻结语义路由、延迟残差记忆、条件原型与受保护低秩适配 | 严格 Predict-Then-Adapt；同时报告同 checkpoint 的 Ours-Static |

所有 target hidden labels 始终只进入 evaluator。CLIP-native 方法的类别语义属于任务定义，
但 template 或 prompt 不得使用目标标签选择。主结果不再把一个数据集压缩成单个数值：
每个数据集分别给出逐 target 的 AUC 表和 Accuracy 表。target 单元格报告三个正式 seed
的均值，Mean 列报告 target-macro 均值及跨 seed 标准差；Accuracy 固定使用 0.5 阈值。

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
保留在论文补充材料中。RoTTA 与 TTC 的结果单元格保持空白并由脚注解释，不得为了填表
放宽上述约束。

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
- TENT、EATA 和 T2A 只允许把公开实现中的 BN 参数参照最小映射到 CLIP visual LayerNorm affine；其余方法逻辑不得重写。CoTTA 保留作者 ImageNet 分支的全参数 student/teacher 更新，只将其像素空间增强桥接到 CLIP 输入归一化。EATA 必须包含匹配公共源域 CLIP detector 的 Fisher。RoTTA 因 robust BatchNorm 是核心而不生成结果，原因只在脚注披露。
- CLIP-native 方法只共享类别语义，不共享人为固定的一句最终 prompt。IAPL 与 Ours 使用各自原生源训练并单独披露 source setup；TTC 在作者公开实现可固定前不生成结果，原因只在脚注披露。
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
