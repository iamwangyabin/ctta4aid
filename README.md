# Online TTA for AI-Generated Image Detection

这是一个用于 AI 生成图像检测的在线测试时适应项目。论文专项只纳入有作者公开实现的方法，当前保留四条实验轨道：

- **CLIP VLM 主实验**：以 OpenAI CLIP ViT-L/14 为固定预训练起点，在 GenImage、AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID Global 上比较 Frozen CLIP、Tent-LN、SAR、LAME、TDA、DynaPrompt、CLIPTTA、BATCLIP 与 IAPL。
- **Controlled CTTA 补充实验**：Source、TENT、EATA、CoTTA、RoTTA、LAME 和 T2A 共用同一个 ResNet-50 源模型，保留为 CNN 对照与补充材料。
- **IAPL 独立能力**：使用作者发布的 IAPL checkpoint 和 CLIP ViT-L/14，按逐图 Adapt-Then-Predict 协议运行；主表中必须显式标出其 source setup 与 Frozen CLIP 不同。
- **OST 补充实验**：使用作者的 MetaXception、AM-Softmax 和单步 fast weights；每张测试图从源训练集抽取带标签模板，合成伪样本后 Adapt-Then-Predict。

已确认的 ResNet-50 结果仍按独立目录保存在 `results/`，只提交最终汇总、复现身份和结论，不提交运行日志或中间产物。新的 CLIP 主表结果必须写入新的 `results/clip_vlm_*` 目录，绝不覆盖这些补充材料。

## 项目结构

```text
configs/
  datasets/       数据集、checkpoint 和目标域
  protocols/      Single-target 与 Continual 协议
  methods/        各方法参数
  experiments/    Controlled CTTA、IAPL 与 OST 实验入口
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

## CLIP VLM 主表

主表固定使用 OpenAI CLIP ViT-L/14 的本地 checkpoint，并验证 SHA-256 为
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。
Frozen CLIP、Tent-LN、SAR、LAME、TDA、CLIPTTA 和 BATCLIP 使用相同的预注册二分类
文本原型 `a real photograph` / `an AI-generated image`。DynaPrompt 从相同的 real/fake
类名开始，但按其原生协议在线更新 context；IAPL 必须使用作者发布的任务 checkpoint，
因此主表会把它标为独立 source setup，而不是声称全部方法共享同一任务训练状态。

当前量化入口固定每个 target 的 Arrow sample identity 和顺序，三 seed 分别复用已经
确认的 manifest。主表报告每个数据集内 generator-macro AUROC 的三 seed 均值与标准差；
GenImage、AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID Global 分别含 7、17、10、5
个 target。方法保留论文原生的 batch contract：Tent-LN、SAR、LAME、TDA、CLIPTTA 和 BATCLIP
使用 batch 16；DynaPrompt 与 IAPL 使用 batch 1 的逐图多视图适应。表格明确标注
Predict-Then-Adapt 或 Adapt-Then-Predict，不将不同预测/适应顺序伪装为同一协议。

尚无论文源文件时，可先生成数值为空的 LaTex 主表；全量运行完成后对同一命令去掉
`--template-only`，汇总器会自动填入数值并导出 CSV、JSON 和 LaTex：

```bash
python scripts/summarize_clip_vlm_results.py \
  --template-only \
  --output-dir /tmp/clip_vitl14_paper_table
```

先放置 CLIP checkpoint 与 IAPL checkpoint，并设置四个 Arrow 根目录：

```bash
scripts/fetch_clip_vitl14.sh /data/weights/clip/ViT-L-14.pt
export CLIP_VIT_L14_CHECKPOINT=/data/weights/clip/ViT-L-14.pt
export IAPL_GENIMAGE_CHECKPOINT=/data/weights/iapl/GenImage.pth
export GENIMAGE_ARROW_ROOT=/data/DF-arrow/GenImage_test
export AIGC_DETECTION_BENCHMARK_ARROW_ROOT=/data/arrow/aigc_detection_benchmark
export AIGI_HOLMES_P3_ARROW_ROOT=/data/arrow/aigi_holmes_p3
export OPENSDID_GLOBAL_ARROW_ROOT=/data/arrow/opensdid_global
export CTTA4AID_EXPERIMENT_ROOT=/data/experiments
```

例如运行 GenImage 的三个 seed：

```bash
python run_single_target.py --config configs/experiments/clip_vlm/genimage_seed0.yaml
python run_single_target.py --config configs/experiments/clip_vlm/genimage_seed1.yaml
python run_single_target.py --config configs/experiments/clip_vlm/genimage_seed2.yaml
```

将配置名替换为 `aigc_detection_benchmark_seed{0,1,2}`、
`aigi_holmes_p3_seed{0,1,2}` 或 `opensdid_global_seed{0,1,2}` 可运行其余三个
数据集。TTC 尚未发现可固定的作者公开实现，因此只作为 related work；RoTTA、T2A 和
当前 CoTTA 公开核心依赖 BatchNorm，不能用项目自写的 LayerNorm 改版放入 ViT-L/14
主表。Tent-LN 明确使用 SAR 发布代码的 LayerNorm 路径，不冒充 BatchNorm-only 的原始
Tent；SAR 使用其官方 ViT LayerNorm 路径。ResNet-101
作为第二个预训练 backbone 的补充实验将在 CLIP 主表稳定后单独加入。

四个数据集的三 seed 全部完成后，写入最终结果目录并生成论文主表：

```bash
python scripts/summarize_clip_vlm_results.py \
  --dataset genimage=${CTTA4AID_EXPERIMENT_ROOT}/clip_vlm/genimage \
  --dataset aigc_detection_benchmark=${CTTA4AID_EXPERIMENT_ROOT}/clip_vlm/aigc_detection_benchmark \
  --dataset aigi_holmes_p3=${CTTA4AID_EXPERIMENT_ROOT}/clip_vlm/aigi_holmes_p3 \
  --dataset opensdid_global=${CTTA4AID_EXPERIMENT_ROOT}/clip_vlm/opensdid_global \
  --output-dir /data/results/clip_vlm_vitl14
```

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

- CLIP 主表固定 OpenAI CLIP ViT-L/14 的预训练 checkpoint、目标样本 identity、目标内顺序和 seed；每行都披露 source setup、batch contract 与预测/适应顺序。表格以 generator-macro AUROC 为主指标，报告三个锁定 seed 的均值与标准差。
- EATA checkpoint 必须包含匹配的源域 Fisher；缺少 Fisher 时默认拒绝运行。RoTTA、T2A 和当前 CoTTA 公开实现不能在没有 BatchNorm 的 ViT-L/14 上伪装为官方复现。
- IAPL 使用作者任务 checkpoint 和 Adapt-Then-Predict 协议，主表中只能作为单独披露的 source setup 比较，不能写成与 Frozen CLIP 同一检测器状态。
- CNN Controlled CTTA 方法共享 backbone、源 checkpoint、输入顺序、batch size 和 Predict-Then-Adapt 协议，并保留为补充材料。
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
