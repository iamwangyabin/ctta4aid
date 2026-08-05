# Online TTA for AI-Generated Image Detection

这是一个先固定实验协议、再逐步补齐方法的最小 Python 工程。项目包含两条不能混为一谈的实验轨道：

1. **CNN Online TTA 主轨道**：同一 ResNet-50 源 checkpoint，严格执行 Predict-Then-Adapt，支持单目标生成器与连续生成器流。
2. **IAPL 官方轨道**：运行作者的 CLIP ViT-L/14 代码，保留逐图重置、先适应再预测的原始协议，用于任务专用补充比较。

第一条轨道实现 Source、TENT、EATA、CoTTA、RoTTA、LAME、T²A；第二条轨道运行固定版本的官方 IAPL，并用参考值门禁判断是否达到论文数值。IAPL 可以作为完整方法进入端到端主比较，但表格必须同时披露 backbone、checkpoint、评价指标和 Adapt/Predict 顺序；只有“控制变量比较适应算法”时才要求另列公共 CNN 轨道。

## 1. 方法实现状态

| 方法 | 当前实现 | 可以如何表述 |
| --- | --- | --- |
| Source-only | 本项目统一实现 | 静态源模型基线 |
| TENT | 包内 vendored 作者核心＋薄 wrapper | wrapper 只拆分 predict/adapt，不重写算法 |
| EATA | 包内 vendored 作者核心＋二分类 wrapper | 官方筛选、EMA、熵加权和 Fisher/EWC 均由作者核心执行 |
| CoTTA | 包内 vendored 官方 ImageNet 核心＋兼容 wrapper | 官方 teacher、anchor、32-view、EMA、恢复和 loss |
| RoTTA | 包内 vendored 官方核心＋二分类/protocol wrapper | 官方 RobustBN、CSTU、timeliness/uncertainty、EMA teacher 和更新频率 |
| LAME | 包内 vendored 官方核心＋feature I/O wrapper | 官方 parameter-free affinity 与 Laplacian output adaptation；不更新模型 |
| T²A | 包内 vendored 作者公开核心＋必要修复 | 公开代码无法原样运行；补丁逐项列出 |
| IAPL | 固定作者官方仓库与 commit 直接运行 | 官方代码轨道；只应用路径、checkpoint 加载与 Arrow 输入兼容补丁 |

CNN 方法现在采用严格的两层结构：`src/online_aig_tta/official/` 保存固定 commit 中抽取的作者算法核心，`src/online_aig_tta/methods/` 只负责框架接口、张量搬运、配置翻译和统计。TENT/EATA wrapper 直接调用作者的 `configure_model`、`collect_params` 和官方 forward-and-adapt；CoTTA/T²A 也调用包内作者核心，不再在 wrapper 里重新写方法公式。

EATA 固定到作者仓库 commit `f739b3668cc7617e9b9f1979c1a358497a3472c3`。作者核心原代码面向 ImageNet；wrapper 只把类别相关阈值从 `log(1000) × 0.4` 改成二分类对应值 `log(2) × 0.4`，从共同 checkpoint 提供 Fisher，并将作者一次 forward 内的预测与更新拆开。两级筛选、概率 EMA、重加权熵和 Fisher 正则均在 vendored 作者核心中执行。

完整 EATA 必须有源域 Fisher。`train_source.py` 会和作者代码一样，在干净源验证图像及 evaluation transform 上，以 batch size 64、伪标签交叉熵和梯度平方，用最多 2,000 个样本计算并写入 checkpoint；缺少 Fisher 时默认报错。只有显式设置 `require_fisher: false` 才会运行作者所称的 ETA 消融，不能继续标成完整 EATA。

IAPL 固定到作者仓库 commit `a173e7783bbafaa00d60e6e31774a0bc14411a23`。作者仓库目前未附软件许可证，因此本项目不重新分发其源码，而是在运行前从官方仓库抓取精确 commit。批准的兼容补丁把写死的 CLIP 路径换成已有的 `--clip_path` 参数，为新版 PyTorch 显式关闭 `weights_only`，并接入不重编码图片的 Arrow dataset adapter；模型、变换、标签、优化器和适应协议不改。

CoTTA 固定到 commit `c212a204b32be4005092e4323105a24a29ad2952`，直接使用 vendored 的作者 ImageNet 分支：32 次增强、`AP=0.1`、`MT=0.999`、`RST=0.001`、对称一致性损失和 SGD `0.01`。兼容补丁仅处理新版 torchvision 参数、硬编码 CUDA、归一化输入桥接，以及为 Predict-Then-Adapt 抽出并缓存作者的 teacher prediction。

RoTTA 固定到 commit `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9`。项目保留作者 RobustBN、CSTU memory、类别平衡淘汰、uncertainty/timeliness 打分、强增强、EMA teacher 和每 64 个样本更新一次的路径。任务迁移把类别数改为 2、分辨率改为公共 224，并把 EMA prediction 缓存到 `adapt` 前；官方像素空间强增强外包有与 CoTTA 相同的 ImageNet 反归一化/再归一化桥接。Adam `1e-3`、`NU=.001`、memory/update frequency 64、`ALPHA=.05` 均保留官方 release 值。

LAME 固定到 commit `d2e5f63090bc1c8129bf7cbd781029a5955e1a67`。它是 parameter-free online inference：每个 batch 使用公共 CNN 的 penultimate features 和 source logits，执行作者 RBF affinity（`k=5`）与最多 100 步 Laplacian optimization，直接返回校准概率；`adapt()` 不修改任何参数或跨 batch 状态。单样本 batch 的官方 RBF bandwidth 为零，wrapper 明确退回 source probability。LAME 源码是 CC BY-NC-SA 4.0，仅限非商业并要求相同方式共享；详见 `THIRD_PARTY_NOTICES.md`。

T²A 固定对照 commit `33c8ccc64afdda260564123d6c790d030a89ff81`，vendored 其 `BaseAdapter`、`T2AAdapter`、loss 和 cosine utility。公开版本不能原样运行：adapter 使用了未初始化属性，Bernoulli 分支产生错误 target 形状，并把概率再次送入 `log_softmax`。包内核心保留作者类结构和控制流，修补为每样本一个且不等于伪标签的 complementary target，并在公共协议的 `predict` 阶段恢复 BN running buffers；因此仍必须表述为“patched authors' public core”，不能声称未经修改。

逐方法的固定来源、官方组件、协议包装和有意差异见 `REPRODUCIBILITY.md`，逐文件上游来源及补丁见 `src/online_aig_tta/official/PROVENANCE.md`，本轮逐项核查结果见 `AUDIT_REPORT.md`。完整配置入口见 `CONFIG_MATRIX.md`，baseline 取舍和风险等级见 `BASELINE_AUDIT.md`。每次 CNN 实验也会把同一份复现等级与差异写进结果 JSON。

## 2. 目录

```text
online-aig-tta/
├── configs/
│   ├── base/             # 数据与协议
│   ├── methods/          # 官方含义的方法参数
│   ├── experiments/      # 2 数据 × 2 设定 × 7 CNN 方法
│   ├── train/            # 两个源模型训练入口
│   ├── single_target.yaml
│   ├── continual_stream.yaml
│   ├── official_sources.yaml
│   ├── iapl_official_genimage.yaml
│   └── iapl_official_ufd.yaml
├── envs/iapl-official.yaml
├── patches/iapl-a173e77-compat.patch
├── src/online_aig_tta/
│   ├── official/         # 固定作者核心及补丁来源
│   └── methods/          # 项目 predict/adapt/reset wrapper
├── tests/
├── AUDIT_REPORT.md
├── BASELINE_AUDIT.md
├── CONFIG_MATRIX.md
├── REPRODUCIBILITY.md
├── fetch_official_baselines.py
├── train_source.py
├── run_single_target.py
├── run_continual_stream.py
└── run_iapl_official.py
```

`external/` 和 `weights/` 已加入 `.gitignore`，不会把第三方源码或大模型权重打进项目包。

## 3. CNN Online TTA 主轨道

### 安装

建议 Python 3.10+、PyTorch 2.2+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 数据布局

GenImage：

```text
/data/GenImage/
├── ADM/
│   ├── train/
│   │   ├── nature/
│   │   └── ai/
│   └── val/
│       ├── nature/
│       └── ai/
└── GLIDE/...
```

UniversalFakeDetect：

```text
/data/UFD/
├── train/progan/<object-class>/{0_real,1_fake}/...
└── test/biggan/{0_real,1_fake}/...
```

索引器会递归查找 `0_real`、`1_fake`，可兼容 ProGAN 的类别子目录。

也可以直接读取 Hugging Face `Dataset.save_to_disk` 生成的 Arrow 数据，图片不会
解包或重编码。DF-Arrow 数据根需要包含 `state.json`、`mapping.json` 和 Arrow
分片；UFD 的 19 个域可以跨多个根组合：

```bash
export UFD_FORENSYNTHS_ARROW_ROOT=/data/DF-arrow-data/ForenSynths
export UFD_OJHA_ARROW_ROOT=/data/DF-arrow-data/Ojha
# The local GenImage Arrow bundle is useful for six-domain diagnostics only.
export GENIMAGE_TEST_ARROW_ROOT=/data/DF-arrow-data/GenImage_test

python run_single_target.py --config configs/universalfake_arrow_single_target.yaml
python run_continual_stream.py --config configs/universalfake_arrow_continual_stream.yaml
```

Arrow 后端使用 `mapping.json` 做原始路径到全局行号的校验，并按需内存映射图片
字节；`ForenSynths/test.json` 与 `Ojha/test.json` 提供 UFD 域和二分类标签。
当前 `GenImage_test` Arrow 只含 6 域；完整 8 域复现仍使用官方 ZIP 或已整理的
ImageFolder，不能把缺少 BigGAN/glide 的 Arrow 子集写成完整 GenImage 结果。

### 运行

先选择数据对应的源模型配置并设置环境变量，然后训练共同源模型：

```bash
python train_source.py --config configs/train/genimage_sd14_resnet50.yaml
python train_source.py --config configs/train/universalfake_progan_resnet50.yaml
python train_source.py --config configs/train/universalfake_progan_resnet50_arrow.yaml
```

实验配置已按数据、设定和方法拆开，例如：

```bash
python run_single_target.py \
  --config configs/experiments/genimage/single_target/eata.yaml
python run_continual_stream.py \
  --config configs/experiments/universalfake/continual/cotta.yaml
```

全部 28 个 CNN 入口、路径变量和方法原始参数来源见 `CONFIG_MATRIX.md`。`configs/single_target.yaml` 与 `configs/continual_stream.yaml` 可用于一次运行 GenImage 七个 CNN 方法。

单目标入口会对每个 `method × target` 重新加载同一个 checkpoint。连续流入口只在每个方法开始前重置一次，生成器切换时保留参数。

每次实验输出 `metrics.json`、`online_curve.csv`、`batch_stats.csv`、`sample_manifest.csv` 和汇总 JSON。连续流额外输出 `holdout_matrix.csv` 与 `final_holdout_manifest.csv`：每个域结束后都在相同、与适应流不重叠的固定 holdout 上重评已见域，forgetting 由同一域历史最好 holdout AUC 与最终 holdout AUC 的差计算，平均值排除尚无后续阶段可发生遗忘的最后一个域。online 与 holdout 都使用独立、固定 seed 的全局 shuffle，避免类别目录排序形成全 real 后全 fake 的伪在线流；阶段性评估会恢复随机数状态，不改变后续适应轨迹。

## 4. IAPL 官方代码轨道

IAPL 不是 CNN 主轨道里的可插拔方法。作者代码使用 CLIP ViT-L/14 和作者训练的 prompt/adapter checkpoint；对每张测试图生成 32 个视图，恢复初始 prompt 和优化器，做 2 步适应，然后在选定视图上预测。也就是说它是 **episodic、per-image、Adapt-Then-Predict**，可训练参数不会在图像之间累积。需要注意，作者实现会在适应时进入 train mode，但没有恢复 Conditional Information Learner 的 BatchNorm running buffers；这些 buffers 会跨图片保留，并在 DDP 中广播。因此按域启停独立进程不是严格等价的官方执行轨迹。

### 获取固定版本

```bash
python fetch_official_baselines.py iapl
```

脚本会检查精确 commit 并应用路径补丁。也可以用 `all` 同时取回 EATA 官方仓库用于人工对照：

```bash
python fetch_official_baselines.py all
```

### 环境和权重

```bash
conda env create -f envs/iapl-official.yaml
conda activate iapl-official
```

还需要：

- 作者发布的 IAPL checkpoint：[ModelScope IAPL_pretrain](https://modelscope.cn/models/yihengli/IAPL_pretrain)；
- OpenAI CLIP ViT-L/14 的 `.pt` checkpoint；
- 与作者 `ImageFolder` 一致的数据目录，或保留原始路径和标签的 DF-Arrow 数据。

IAPL 的数据根目录与 CNN 主轨道不同，必须整理为：

```text
/data/IAPL-layout/GenImage/
├── train/SDv14/{0_real,1_fake}/...
└── test/
    ├── ADM/{0_real,1_fake}/...
    ├── BigGAN/{0_real,1_fake}/...
    └── ...

/data/IAPL-layout/UniversalFakeDetect/
├── train/{car,cat,chair,horse}/.../{0_real,1_fake}/...
└── test/{crn,cyclegan,dalle,...}/{0_real,1_fake}/...
```

修改对应 YAML 的三条路径：`dataset_path`、`pretrained_model`、`clip_path`，然后运行：

```bash
python run_iapl_official.py --config configs/iapl_official_genimage.yaml
python run_iapl_official.py --config configs/iapl_official_ufd.yaml
```

UFD 不需要导出成图片目录。设置 Arrow 根、IAPL checkout 和权重后可直接运行：

```bash
export IAPL_REPO_PATH=/path/to/IAPL
export UFD_FORENSYNTHS_ARROW_ROOT=/data/DF-arrow-data/ForenSynths
export UFD_OJHA_ARROW_ROOT=/data/DF-arrow-data/Ojha
python run_iapl_official.py --config configs/iapl_official_ufd_arrow_1gpu.yaml
```

此配置使用 `hf_arrow://root1|root2` 数据 URI，并保留作者的图像变换、32 视图、
2 个 TTA step 和逐图重置逻辑。配置名中的 `1gpu` 明确表示它不是作者发布的
8 进程执行形态。

两个配置逐项对齐作者发布的 `tta_genimage.sh` 与 `tta_universalfake.sh`：8 个进程、32 个视图、学习率 0.005、2 个 TTA step、OIS 开启；GenImage 额外开启 smooth。GPU 少于 8 张时可调整 `nproc_per_node` 做工程测试，但这已经偏离作者发布设置，正式复现必须记录。

运行器保存原始日志和 `official_iapl_metrics.json`。配置内含作者 README 报告的参考均值与容差；缺域、缺 mean 或偏差超过容差都会失败，防止“代码跑通”被误写成复现成功。作者代码报告的是 Accuracy、AP、real accuracy 和 fake accuracy，不是 AUC。不要把 AP 写成 AUC。

## 5. 协议与公平性边界

1. CNN 主表只使用 Predict-Then-Adapt，并共享 backbone、源 checkpoint、batch size、图像顺序与随机种子。
2. EATA 的 Fisher 只在源训练结束时计算；部署测试期间仍然 source-free。
3. IAPL 可以作为官方 AIGC-specific baseline 进入端到端方法表；同时标明其 CLIP、per-image reset、Adapt-Then-Predict 和 Accuracy/AP 协议。若另做“相同 checkpoint 的适应算法消融表”，再将它与公共 CNN 轨道分列。
4. CoTTA 使用官方 ImageNet/ResNet-50 算法分支，但统一二分类源 checkpoint 和 batch size 16 属于 AIGC 任务协议；原始 ImageNet-C 数值 sanity check 与 AIGC 公平主表是两项不同验证。
5. T²A 是作者公开代码的必要修复版；必须在原 Deepfake 数据与 checkpoint 上通过 sanity check 后，才可声称数值复现一致。
6. RoTTA 的 CSTU 按伪标签做类别平衡，不读取隐藏标签；二分类 memory 行为必须报告。LAME 没有连续状态，因此其 final/forgetting 结果不能解释成参数记忆或遗忘。
7. BatchNorm TTA 对 batch size 和类别组成敏感；不能使用按隐藏标签平衡的 batch 而不披露。
8. 超参数只能在独立 development generator 上选择。

## 6. 测试

无 PyTorch 环境也能验证配置、数据索引、官方核心路径、指标、协议顺序和 IAPL 命令；方法张量测试会明确标记为 skip：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests train_source.py run_single_target.py \
  run_continual_stream.py run_iapl_official.py fetch_official_baselines.py
```

此前审计曾在临时 PyTorch 2.5.1 环境中完成 34 项测试。本轮又加入 RoTTA 归一化桥接、T²A predict 状态不变、真实 complementary target、固定 holdout 顺序和阶段性 forgetting 矩阵测试；当前缺少完整 PyTorch/torchvision 依赖的环境只能执行轻量测试与字节码编译。有 CUDA、数据和权重后，仍须完成 `REPRODUCIBILITY.md` 的论文数值门槛；代码执行通过不等于论文数值已经复现。

## 7. 官方来源

- [EATA 论文](https://arxiv.org/abs/2204.02610) / [官方代码](https://github.com/mr-eggplant/EATA)
- [IAPL 论文](https://arxiv.org/abs/2508.01603) / [官方代码](https://github.com/liyih/IAPL)
- [T²A 论文](https://arxiv.org/abs/2505.18787) / [官方代码](https://github.com/HongHanh2104/T2A-Think-Twice-Before-Adaptation)
- [TENT 官方代码](https://github.com/DequanWang/tent)
- [CoTTA 官方代码](https://github.com/qinenergy/cotta)
- [RoTTA 论文](https://arxiv.org/abs/2303.13899) / [官方代码](https://github.com/BIT-DA/RoTTA)
- [LAME 论文](https://arxiv.org/abs/2201.05718) / [官方代码](https://github.com/fiveai/LAME)
- [GenImage 官方仓库](https://github.com/GenImage-Dataset/GenImage)
- [UniversalFakeDetect 官方仓库](https://github.com/WisconsinAIVision/UniversalFakeDetect)

第三方授权和固定 commit 见 `THIRD_PARTY_NOTICES.md`。
