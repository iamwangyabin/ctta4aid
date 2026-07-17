# Baseline selection and fidelity audit

Audit date: 2026-07-17

## 1. 第一版主表应该保留什么

| 方法 | 是否进入 CNN 公平主表 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| Source-only | 必须 | 判断适应是否真的优于静态模型 | 项目控制组 |
| TENT | 必须 | 最小熵、BN-only 的基本 TTA 下界 | 官方核心移植；未做原 benchmark 数值复现 |
| EATA | 必须 | 样本筛选与 Fisher 约束，直接检验错误更新 | 官方核心移植；完整 EATA 强制 Fisher |
| CoTTA | 必须 | teacher/augmentation/restoration 的 continual TTA 基线 | 官方 ImageNet 核心兼容移植；未做原 benchmark 数值复现 |
| RoTTA | 重要补充 | 专门面向动态/相关连续流的 RobustBN＋类别平衡 memory 基线 | 官方核心二分类移植；未做 CIFAR PTTA 数值复现 |
| LAME | 重要补充 | 零参数输出适配，区分“模型更新”与“batch 校准”的收益 | 官方核心移植；无跨 batch 状态；未做原 benchmark 数值复现 |
| T²A | 必须但加限定 | 最接近 Deepfake Online TTA 的任务基线 | 公开代码需修复才能运行；风险最高 |
| IAPL | 端到端主比较 | 直接面向 AIGC detection 的测试时适应 | 固定官方代码；以参考门禁核验论文数值，需披露 CLIP、episodic、Adapt-Then-Predict |

七个 CNN 方法构成控制变量表：静态下界、最简单适应、可靠性约束、teacher continual、动态流 memory、零参数输出适配和最近任务方法各占一个位置。IAPL 是重要的官方 AIGC-specific baseline，可以进入完整方法的端到端主比较；不同 backbone 本身不是排除理由，但必须同时报告原生协议和指标。

## 2. 最值得追加的方法

优先级按“对论文结论的增量”排序：

1. **ATTSD**：IAPL 论文直接比较的 AIGC-specific test-time adaptation/debiasing 方法，任务匹配度高。当前未确认到可固定 commit 的作者官方实现；在官方代码可得之前，不应自行写一个近似版后标为 ATTSD。
2. **Insertable Adaptation Module**：直接研究 face forgery detector 的测试时可插拔适应，论文给出 memory、prototype、近邻和增强相关超参数。当前没有确认到作者官方仓库；在官方实现可得前不加入近似替代品。
3. **MEMO**：单样本增强边缘熵最小化，与 BN-only TENT 的机制互补，也是 T²A 论文采用的通用基线。若预算只允许再加一个通用方法，优先 MEMO。
4. **SAR 或 COME**：二选一即可，用于验证长流中的崩溃/稳定性；只有当前方法确实出现长期退化后再加，没必要一次堆满。

RoTTA 与 LAME 已因动态流稳定性和零参数对照的独立作用加入。STAMP、TAST、T3A、RDumb 等不是无意义，但第一版继续加入会扩大工程和调参空间，却不一定改变核心结论。T3A/TAST 更适合放在后续 prototype/memory 专题，而不是冒充 Insertable module。

## 3. 不应作为同表 Online TTA baseline 的方法

- **AIGI-Det-Calib** 使用带标签的 few-shot target calibration；它回答的是目标域少样本校准，不满足 target-label-free Online TTA。
- **IAPL 原生结果** 可以进入端到端方法主表；但不能拿它单独证明某个适应组件优于公共 CNN 方法，因为 backbone、源 checkpoint 和 Adapt/Predict 顺序同时发生了变化。
- 需要重新训练源模型、meta-training 或访问源图像的方法，除非单独定义新的轨道，否则不满足当前 source-free 部署条件。
- 仅更换 AIGC detector backbone 的检测方法属于源检测器 baseline，不属于 TTA baseline。

## 4. 现有复现需要特别注意什么

### TENT

- 算法核心可信度较高；wrapper 直接调用官方配置模型、参数收集和更新函数。
- 这次已将默认值纠正为 TENT 官方示例的 Adam `1e-3`，不再把 EATA 对比实验中的 SGD `2.5e-4` 当作 TENT 官方默认。
- 官方 YAML 是特定 benchmark 示例，不保证迁移到二分类 ResNet-50 最优；最终学习率必须只在 development generator 上选择。

### EATA

- 筛选、概率 EMA、熵重加权和 EWC 均来自作者核心。
- 最大风险是 Fisher：必须由与源训练数据不重叠的干净 source validation 集计算，并把样本清单和 checkpoint hash 一起保存。`require_fisher: false` 只能称 ETA 消融。
- `e_margin` 按作者规则从 ImageNet 的 `0.4*log(1000)` 转成二分类 `0.4*log(2)`；这是有公式依据的任务迁移，不是作者报告过的 AIGC 数值。

### CoTTA

- 使用官方 ImageNet 分支的 student、EMA teacher、anchor、32-view augmentation 和 stochastic restoration。
- 归一化桥接、torchvision/device 兼容和 Predict-Then-Adapt 的 teacher-target 缓存属于框架补丁。
- 原论文 batch size 64，公共主表为 16；32 次增强使它远慢于其他方法，正式效率表必须单独报告更新时间和显存。

### T²A

- 当前是最需要谨慎表述的方法。固定的公开 release 引用了未初始化字段，Bernoulli target 形状和 loss 输入也存在阻断问题。
- 项目把必要修复集中记录在 `release_repairs` 和 provenance 中，但公开代码的 entropy-minimization 路径与论文文字/公式仍存在疑点。
- 在作者原 Deepfake 数据和 checkpoint 上通过数值 sanity check 前，只能称 `patched authors' public core`，不能称“完整官方复现”。建议向作者确认缺失字段和 entropy 分支。

### RoTTA

- RobustBN、CSTU memory、伪标签类别平衡、timeliness/uncertainty 淘汰、强增强、teacher loss 和 EMA 均来自固定官方核心。
- 原 release 只提供 CIFAR-10-C/100-C 配置；`NUM_CLASS=2`、224 输入和公共 batch 16 是任务/协议迁移，不是作者报告过的 AIGC 设置。
- 公共 loader 输出 ImageNet-normalized tensor；wrapper 会在官方强增强前反归一化到像素空间，并在 student forward 前重新归一化。
- 官方 memory 和 update frequency 都是 64，因此公共 batch 16 下通常每四个完整 batch 更新一次；不要为了“每 batch 一步”私自改成 16 后仍称官方默认。

### LAME

- LAME 不训练参数，也不在 batch 间积累状态；这是方法定义，不是缺少 `adapt` 实现。
- RBF affinity、`k=5`、`lambda=1` 和 100 步上限来自官方 release。二分类在数学上有效，但效果强依赖 batch 内特征邻域和组成。
- `N=1` 时官方 RBF bandwidth 为 0；wrapper 返回 source probability 并记录 guard。正式运行应保持 batch 16，并披露最后一个不完整 batch。
- 官方代码为 CC BY-NC-SA 4.0，非商业/相同方式共享限制必须传递。

### IAPL

- 运行的是固定 commit 的作者仓库，配置名直接对应作者 shell/argparse。
- 官方实现每张图恢复 prompt 和 optimizer，生成 32 views、选约 6 个视图、适应 2 步后预测，不是 online parameter accumulation。
- 作者仓库未附可确认的软件许可证，所以项目只拉取固定源码，不把源码重新打包分发。

## 5. 目前能声称到什么程度

当前已完成“固定来源 + 算法核心级移植 + 接口/张量测试”，以及 IAPL 在 UFD Arrow 19 域上的完整运行。IAPL Accuracy 95.41% 进入作者参考值的 1 个百分点容差，AP 97.21% 未进入容差，因此仍不能声称“复现论文结果”。TENT、EATA、CoTTA、RoTTA、LAME、T²A 尚未拿作者原始数据和 checkpoint 跑通论文数值。每个方法的准确措辞、固定 commit 和补丁边界见 `REPRODUCIBILITY.md` 与 `src/online_aig_tta/official/PROVENANCE.md`。
