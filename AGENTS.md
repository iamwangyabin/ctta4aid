# Project Guardrails

本文件约束所有在本仓库中工作的 AI 和自动化工具。目标是让项目保持可维护、可复现和可审计，而不是每次任务都重新设计目录、协议或实验范围。

除非用户在当前任务中明确要求，否则下列标记为“冻结”的内容不得删除、重命名、移动、合并或改变语义。发现问题时应先报告影响，再做最小修复；不能把“看起来没被直接引用”当作删除依据。

## 1. 项目范围

项目必须同时保留五条实验轨道：

1. **CLIP ViT-L/14 论文主实验**：唯一预训练模型固定为 OpenAI CLIP ViT-L/14，正式 target 输入固定为 `matched_jpeg` profile，在 GenImage、AIGCDetectionBenchmark、AIGI-Holmes P3 和 OpenSDID Global 上报告方法原生的在线结果。候选方法为 Source、TENT、EATA、SAR、CoTTA、RoTTA-LN、LAME、T2A、Frozen CLIP、TDA、DynaPrompt、CLIPTTA、BATCLIP、IAPL 和 Ours；只有满足下述最小迁移与公开实现约束的方法才产生数值。四个数据集的逐 target AUC 与 Accuracy 详细表均属于正文主结果，版面压缩问题后续单独处理。
2. **Controlled CTTA 补充实验**：公共 ResNet-50 checkpoint 下的 Source、TENT、EATA、CoTTA、RoTTA、LAME 和 T2A。已确认结果保持不可变，作为 CNN 对照与补充材料，不再作为论文主表。
3. **IAPL 补充轨道**：CLIP ViT-L/14、逐图 Adapt-Then-Predict 的独立方法能力；主表若列出 IAPL，必须标注其作者发布的任务 checkpoint，不能写成与 Frozen CLIP 相同的 source state。
4. **OST 补充实验**：MetaXception、源训练模板、逐图一次 fast-weight 更新的独立方法轨道。
5. **JPEG 输入协议与补充审计**：`matched_jpeg` 是 CLIP ViT-L/14 正文主实验的正式输入协议；原始编码和 `all_jpeg_q90` 仅作为分离的补充敏感性审计。三者必须保留独立 Arrow、运行目录、metadata 和汇总，任何结果不得互相覆盖或混入。

IAPL 和 OST 都是正式项目能力。不得因为当前机器缺少权重或当前没有结果，就把其配置、核心、loader、方法、测试、依赖或说明判定为垃圾。论文专项方法只加入有作者公开实现且来源可固定的方法；没有作者公开实现的方法不得用项目自写实现冒充复现。TTC 在作者公开实现可固定前只能作为 related work，不得生成量化复现行。

## 2. 冻结目录结构

`src/` 直接包含以下一级模块，不得重新引入 `src/<project_name>/...` 这一层：

```text
src/
  cli/
  data/
  evaluation/
  methods/
  models/
  official/
  __init__.py
  config.py
  types.py
```

各目录职责固定：

- `src/official/`：固定上游 commit 的第三方算法核心。
- `src/methods/`：公共框架 wrapper，只负责协议适配、配置翻译和统计。
- `src/models/`：模型构建、checkpoint 及 IAPL、OST 上游模型加载。
- `src/data/`：数据集、预处理和 stream。
- `src/evaluation/`：指标、在线评估和结果写入。
- `configs/datasets/`：数据、checkpoint 和目标域。
- `configs/protocols/`：Single-target 与 Continual 协议。
- `configs/methods/`：方法参数。
- `configs/experiments/`：组合配置、seed 和输出目录。
- `configs/experiments/clip_vlm_bias_controlled/`：正文 `matched_jpeg` 主实验及 JPEG 补充审计的独立入口。
- `configs/train/`：源模型训练配置。

这些层级是职责分离，不得为了减少目录数量而混放或合并。

数据输入格式冻结为项目标准 `arrow`，唯一实现位于 `src/data/arrow.py`：

- 所有训练、适应和评估配置必须使用 `data.format: arrow`。
- 不得重新引入 ImageFolder、ZIP、Parquet、CAIDBench Arrow 或其他并行数据读取后端。
- Arrow 根目录必须是 Hugging Face `Dataset.save_to_disk` 产物，并包含 `state.json`、`mapping.json`、`image` 与 `image_path`。
- generator、split 和二分类标签必须能由逻辑路径及可选的 `<split>.json` 索引唯一确定。
- 新数据集必须先离线转换并通过统一 Arrow 检查，再接入正式配置；原始数据格式不得进入实验调用链。

## 3. 冻结实验协议

### CLIP VLM 主实验

- backbone 固定为 OpenAI CLIP ViT-L/14，checkpoint SHA-256 固定为 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。
- 正文主结果固定使用 `matched_jpeg` target profile：real 与 fake 执行相同的 256x256 center-crop/resize，并从 `[75, 80, 85, 90, 95]` 按不含类别目录的逻辑路径确定性选择 JPEG 质量。原始编码与 `all_jpeg_q90` 不得替换或混入正文主表。
- 所有方法共享目标样本身份、目标内顺序、三个正式 seed 与 target-label embargo；各方法保留论文原生的 source training、batch/views、状态转移、预测/适应顺序和 prompt 构造，只做接入固定 CLIP、二分类数据与统一 evaluator 所必需的改动。
- 通用检测器适配方法 Source、TENT、EATA、SAR、CoTTA、RoTTA-LN、LAME 与 T2A 必须共享一个从固定 ViT-L/14 权重开始训练的源域二分类 checkpoint。它们不得被强行改成固定文本 prompt 分类器。
- CLIP-native 方法 Frozen CLIP、TDA、DynaPrompt、CLIPTTA 与 BATCLIP 直接从相同预训练 checkpoint 出发，保留各自论文的文本分类器、template 或 prompt learner。它们只共享二分类类别语义，不共享一条人为固定的最终 prompt；不得使用目标标签选择文本或超参数。
- IAPL 与 Ours 保留各自方法要求的源训练，但底层初始化仍必须来自同一固定 ViT-L/14 权重。主表必须把这类 method-specific source training 与前两组分块披露，不得跨 source setup 加粗全局最佳。
- TENT、EATA 与 T2A 的公开实现将 BatchNorm 作为关键参数参照时，可最小映射到 CLIP visual LayerNorm affine 参数；目标函数、样本筛选、teacher、Fisher、gradient masking、更新顺序及在线状态不得随之重写，表格与 metadata 必须加脚注披露该必要迁移。CoTTA 保留作者 ImageNet 分支的全参数 student/teacher 更新，只把像素空间增强的归一化桥接为 CLIP 的输入归一化。
- SAR 使用其官方 ViT LayerNorm 路径；其 ViT-B 最后三块过滤映射到 ViT-L/14 的最后三块必须在 metadata 中披露。
- RoTTA 在纯 ViT-L/14 主实验中使用经用户明确批准的 `RoTTA-LN` 迁移：只将 RobustBN 的可适应归一化参数替换为 CLIP visual LayerNorm affine 参数，保留 CSTU memory、teacher/student、EMA、entropy objective、optimizer、64-instance 更新频率和在线顺序。RobustBN 的源/目标统计插值没有 LayerNorm 等价物，因此明确缺失；FP32 ViT-L/14 student/teacher 在 24 GB GPU 上固定 batch size 2。表格、配置和 metadata 必须披露这些差异并标为 `RoTTA-LN`，不得冒充原版 RobustBN RoTTA。
- EATA 必须有与公共源域 CLIP detector 匹配的 source Fisher 才可标为 EATA；没有 Fisher 的运行只能标为 ETA 消融。
- TTC 在作者公开实现可固定前只保留在 related work，不得出现在定量表中，也不得用项目自写实现生成复现数值。
- 主表按“公共源域 CLIP detector”“CLIP-native”“method-specific source training”分块；只有前两块可分别在块内比较最佳结果。
- 四个数据集必须分别生成逐 target 的 AUC 表和 Accuracy 表，不能只报告数据集级平均值。target 列及顺序固定为现有数据配置；Accuracy 使用阈值 0.5。每个 target 单元格报告三个正式 seed 的均值，Mean 报告 target-macro 均值及跨 seed 标准差。
- 新 CLIP ViT-L/14 表格在完整三 seed campaign 验收前不得出现任何实验数值。既有 ResNet-50 数值表必须原样保留在论文补充材料中，并明确标为上一轮 CNN controlled results，不得与新 CLIP 结果混写。
- 每个 `method x target x seed` 必须重新构建方法；单目标结果使用已确认的 online manifest 锁定样本身份，批大小变化不得改变样本顺序。
- `configs/experiments/clip_vlm/` 定义方法原生的公共基础配置；经审定的正文正式入口为 `configs/experiments/clip_vlm_bias_controlled/matched_jpeg_<dataset>_seed<seed>.yaml`。公共 detector 组加载同一 SD v1.4 源训练 checkpoint；CLIP-native 组加载各自声明的文本分类器或 prompt profile；IAPL 加载作者任务 checkpoint。启动前必须完成对应 GPU smoke test，并在结果 metadata 中记录 `matched_jpeg` profile identity。

### Controlled CTTA 补充实验

- 方法集合固定为 `source, tent, eata, cotta, rotta, lame, t2a`。
- 所有方法共享同一 backbone、源 checkpoint、输入样本、顺序、batch size 和 seed。
- 方法不得读取 target hidden labels；labels 只能进入 evaluator。
- Single-target 必须为每个 `method x target` 重新构建模型并加载同一个源 checkpoint。
- Continual stream 每个方法只初始化一次，域切换时保留该方法的在线状态。
- Continual final holdout 必须与适应样本不重叠，并保持固定采样和固定顺序。
- EATA 默认必须加载源域 Fisher；没有 Fisher 的运行只能标作 ETA 消融。
- 不得静默修改目标域列表、stream 顺序、采样数量、指标定义、阈值或三个正式 seed。

### JPEG 输入协议与补充审计

- `matched_jpeg` campaign 是论文正文主实验；只能写入 `clip_vlm_bias_controlled/matched_jpeg/<dataset>/seed<seed>`。原始 `clip_vlm` campaign 保持不变并降为补充对照，不得写入、读取或补齐正文主实验目录。
- `all_jpeg_q90` 对 real 和 fake 都执行 EXIF orientation、RGB canonicalization 和单次 JPEG Q90 编码，保留视觉尺寸；不得只压缩 fake。
- `matched_jpeg` 对 real 和 fake 都执行相同的 256x256 center-crop/resize，再从固定 `[75, 80, 85, 90, 95]` 质量集合按不含类别目录的逻辑路径确定性取值；质量选择不得读取二分类标签。
- 两个 profile 都只改变 target image bytes。模型 checkpoint、source setup、方法配置、target/sample identity、锁定顺序、seed、阈值和 evaluator 必须继承对应的 `configs/experiments/clip_vlm/` 正式入口。
- 每个转换后 bundle 必须包含 `bias_control_manifest.json`，记录 profile 规范哈希、逻辑路径哈希、输入/输出字节哈希、格式、质量和几何统计；运行配置和 loader 必须验证 profile，缺失或不匹配时拒绝启动。
- 原始 JPEG 再编码会产生 double-compression；正文必须将 `matched_jpeg` 准确描述为编码与几何匹配协议，不得声称它完全消除了所有数据集偏差。
- `matched_jpeg` 是正式正文设置，`all_jpeg_q90` 与原始编码仅作为补充敏感性对照。正文表格仍须在四个数据集三个 seed 全量验收前保持空白；全量验收后只填入 `matched_jpeg` 汇总，其他 profile 不得混表。

### IAPL

- 固定上游 commit 为 `a173e7783bbafaa00d60e6e31774a0bc14411a23`。
- 保留 32 views、2 个适应 step、逐图 prompt/optimizer reset、entropy tuning 和 OIS。
- IAPL 是 Adapt-Then-Predict，不得伪装成主实验的 Predict-Then-Adapt。
- 每个 target 必须重新加载模型；不同 target 之间不得继承 BatchNorm buffers。
- 同一 target 内允许 Conditional Information Learner 的 BatchNorm buffers 跨图片保留。
- IAPL 固定版本的最小运行核心位于 `src/official/iapl/`，模型加载必须使用仓库内包导入。
- 上游没有声明软件许可证；必须保留源码头、`configs/official_sources.yaml` 和 `THIRD_PARTY_NOTICES.md` 中的来源及未授权状态，不得将其描述为 MIT 或其他开源许可证代码。

### OST

- 固定上游 commit 为 `1e4518b9e560baf9c5693f13a402fa5d7104190f`。
- 保留 MetaXception、AM-Softmax、每张测试图一次 fast-weight 更新和 Adapt-Then-Predict 顺序。
- OST 在测试时读取源训练集随机模板及其标签，但不得读取 target hidden labels。
- 每个 target 必须重新加载模型；不同 target 不得继承模型或 BatchNorm 状态，fast weights 不得跨测试图保留。
- 通用图像轨道使用明确披露的 full-frame alpha blending 数据适配，不得描述为作者人脸合成管线或论文原数值复现。
- OST 固定版本核心位于 `src/official/ost/`，数据输入仍只允许项目标准 Arrow。
- 上游没有声明软件许可证；必须保留源码头、`configs/official_sources.yaml` 和 `THIRD_PARTY_NOTICES.md` 中的来源及未授权状态。

任何协议变化都必须由用户明确批准，并同步修改配置、测试和 README。不得只改代码而保留旧协议说明。

## 4. 第三方核心与授权

`src/official/` 与 `src/methods/` 的两层结构是有意设计，不是重复代码：

- 不得把 wrapper 逻辑复制进 `src/official/`。
- 不得为了“统一风格”重写 vendored 官方算法核心。
- 兼容性或正确性修复必须保持最小范围，并在源文件头、`configs/official_sources.yaml` 和测试中记录。
- `THIRD_PARTY_NOTICES.md` 是授权文件，必须保留。
- LAME 仍受 CC BY-NC-SA 4.0 约束，不得当作普通 MIT 项目代码处理。

## 5. 结果文件

当前没有冻结的正式结果。新的实验完成后，`results/` 只接收最终结果：

- `results/controlled_ctta_genimage_20260812/` 与 `results/controlled_ctta_external_20260816/` 是已确认的 ResNet-50 补充材料；不得移动、覆盖、重算后回写或改动其 JSON 数值。
- CLIP VLM 正文主表必须写入新的、明确命名的 `results/clip_vlm_bias_controlled_*` 目录，并保留 `matched_jpeg` profile 名称与规范哈希、per-seed summary、跨 seed summary、CLIP checkpoint 身份、每种方法的分类器或 prompt 构造、source setup 与最终总表。
- 原始编码与 `all_jpeg_q90` 补充结果必须写入各自明确命名的结果目录；不同 profile 不得合并汇总，也不得回填正文 `matched_jpeg` 表。
- 正式结果至少保留 per-seed summary、跨 seed summary、源模型记录和最终总表。
- 结果文件一旦确认，不得修改已有 JSON 数值来匹配后续代码变化。
- 不得覆盖已经确认的结果；新实验必须写入新的、明确命名的结果文件。
- 不保留中间 checkpoint、日志、PID、GPU monitor、progress、snapshot、preflight 或失败过程文件。
- 只有最终且可解释的实验结果可以进入 Git。

## 6. 文档政策

长期维护的 Markdown 仅包括：

- `AGENTS.md`
- `README.md`
- `THIRD_PARTY_NOTICES.md`

完成新的正式实验后，可以在对应结果目录增加一个最终结果 `README.md`。

除非用户明确要求，不得新增 audit、plan、status、progress、notes、matrix 或 reproduction 类 Markdown。实现来源写入结构化 YAML 和源文件头；最终结论写入结果 README，不再创建重复说明文档。

## 7. 文件与脚本卫生

- 根目录实行白名单，只允许 `.gitignore`、`AGENTS.md`、`README.md`、`THIRD_PARTY_NOTICES.md`、`pyproject.toml`、`train_source.py`、`run_single_target.py` 和 `run_continual_stream.py`。新增任何根目录文件前必须得到用户明确授权并说明其长期职责。
- 可复用脚本放在 `scripts/`；一次性迁移、服务器探测、临时诊断和带绝对机器路径的脚本不得长期保留。
- 不得提交 `external/`、`weights/`、`outputs/`、虚拟环境、缓存或生成日志。
- 不得创建 `envs/`、`patches/` 或新的临时结果树，除非用户明确要求长期维护它们。
- 空的 `__init__.py` 可能是 Python 包边界，不得仅因为内容少就删除。
- 删除文件前必须检查配置引用、动态 import、测试、README、许可证和结果来源，不能只做文本引用计数。
- 本机一次性草稿、下载包、解压目录、转换产物、渲染文件、截图和诊断输出统一放到 `/private/tmp/ctta4aid/<task-name>/`，不得放在仓库根目录或长期目录中。
- 远程服务器上的临时日志、PID、监控文件、探测结果和中间产物必须放到仓库外的 `/tmp/ctta4aid/<task-name>/`；长时间实验确需持久化时，使用用户指定的仓库外任务目录。完成后只把最终且可解释的结果导入 `results/`。
- 临时文件和一次性脚本不得进入 Git。任务结束后应清理本任务创建且不再需要的临时内容，但不得删除用户已有文件或其他任务产物。
- 不得把临时方案改名后伪装成长期工具；只有被正式入口或维护流程复用、具有测试且不含机器绝对路径的脚本才能保留在 `scripts/`。

## 8. 修改流程

每次修改应遵循以下顺序：

1. 阅读 `AGENTS.md`、相关配置、实现和测试。
2. 检查 `git status -sb`，保留用户已有修改，不得回滚、覆盖或顺手提交无关内容。
3. 运行 `git fetch origin` 检查远端状态。工作区干净且远端仅领先时只能使用 `git pull --ff-only`；出现分叉、冲突或本地未提交修改时不得擅自 merge、rebase、stash、reset 或覆盖远端。
4. 判断是否触及冻结协议、结果、第三方核心或目录结构。
5. 采用最小改动，只编辑当前任务必需的文件和代码，不做无关重构、批量格式化、全仓机械改写或“顺便清理”。
6. 行为变化必须补充或更新测试。
7. 更新现有 README，而不是新建说明文件。
8. 执行适用验证并如实报告未运行的部分。
9. 使用 `git diff --check` 和 `git diff` 审核本任务差异，只暂存本任务文件。
10. 按第 9 节提交并同步 GitHub；验证本地提交与目标远端分支一致后，任务才算完成。

基础验证命令：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests train_source.py run_single_target.py \
  run_continual_stream.py scripts
git diff --check
```

涉及配置时至少运行配置测试；涉及方法时运行对应方法测试；涉及数据或协议时运行相应数据集或 online protocol 测试。缺少 PyTorch、数据、权重或 CUDA 时必须明确说明，不能把 skip 当作完整验证通过。

## 9. GitHub 同步

除非用户在当前任务中明确要求暂不提交或暂不推送，否则每个完成的项目修改都必须执行：

```text
修改 -> 验证 -> 只暂存本任务文件 -> git commit -> git push -> 验证远端同步
```

- 不得只修改本地后声称任务完成，也不得积攒多个无关任务后一次性提交。
- 工作区存在其他改动时必须显式列出并只暂存当前任务文件；禁止用 `git add -A`、`git add .` 或全仓提交把不明改动混入提交。
- 默认推送当前跟踪分支；不得为了方便新建分支、改默认分支或改变远端地址。
- 推送前必须确认提交中没有密码、token、SSH key、机器私有路径、数据、模型、日志、PID、缓存或其他秘密与生成物。
- 推送被拒绝时先 `git fetch origin` 并检查差异。不得用 force push、历史改写或覆盖远端解决冲突。
- 网络、权限、远端分叉或验证失败导致无法同步时，必须保留现场并明确报告实际状态；不得谎称已经提交或推送。
- 常规任务完成后的提交和推送属于本节要求，不再逐次请求额外授权；创建分支、force push、覆盖远端和其他高风险 Git 操作仍需用户在当前任务中明确授权。

## 10. 变更权限

以下操作必须得到用户在当前任务中的明确授权：

- 删除或替换 Controlled CTTA、IAPL 任一实验轨道。
- 改变冻结协议、方法集合、数据域、指标或正式结果。
- 修改或删除 `src/official/`、第三方授权或最终结果。
- 改变顶层目录和配置分层。
- 新增长期文档、环境目录、patch 集合或大规模实验脚本。
- 创建分支、force push、改写历史、覆盖远端状态或修改远端地址。

当用户的指令与本文件冲突时，以用户当前明确指令为准，但应先指出会被改变的冻结内容和影响范围。
