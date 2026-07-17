# Reproduction and fairness contract

这份文件是结果表的复现边界，不是宣传性说明。项目把“算法是否忠于作者实现”和“不同方法是否在相同 AIGC 协议下比较”拆成两个独立条件。两者都满足，才允许把结果写进 CNN 公平主表。

## 1. 两级验证

1. **官方 sanity track**：在作者的原 backbone、数据、协议、checkpoint 和超参数上运行固定官方代码或移植实现，检查是否落在作者数值容差内。它回答“复现是否正确”。
2. **AIGC common track**：所有 CNN 方法使用同一源 checkpoint、图像、顺序、batch size、seed、每 batch 一步和 Predict-Then-Adapt。它回答“迁移到本任务后谁更有效”。

不能用 common track 的相对排名代替官方数值 sanity check。IAPL 的官方 CLIP/episodic/Adapt-Then-Predict 结果可以进入端到端方法比较，但必须与“相同 checkpoint 的适应算法控制表”区分，并完整披露协议。

## 2. 固定来源与差异

| 方法 | 固定 commit | 复现等级 | 保留的官方组件 | 必要或有意差异 |
| --- | --- | --- | --- | --- |
| TENT | `e9e926a668d85244c66a6d5c006efbd2b82e83e8` | vendored official core | 作者 `configure_model`、`collect_params`、entropy、forward-and-adapt、reset | framework 拆分 Predict-Then-Adapt；统一 ResNet-50 |
| EATA | `f739b3668cc7617e9b9f1979c1a358497a3472c3` | vendored official core | 作者两级筛选、probability EMA、entropy weights、Fisher/EWC、forward-and-adapt | 二分类 margin；checkpoint Fisher；协议拆分；reset 额外清历史概率 |
| CoTTA | `c212a204b32be4005092e4323105a24a29ad2952` | patched vendored official ImageNet core | 作者 student、EMA teacher、anchor、32-view、对称 loss、EMA、RST、官方增强 | torchvision/device 兼容；归一化桥接；抽出并缓存 teacher target |
| RoTTA | `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9` | patched vendored official core | 作者 RobustBN、CSTU、类别平衡淘汰、uncertainty/timeliness、强增强、EMA teacher、频率更新 | 二分类/224 输入；torchvision/device 兼容；像素增强归一化桥接；缓存 EMA prediction 后再适配 |
| LAME | `d2e5f63090bc1c8129bf7cbd781029a5955e1a67` | vendored official core | 作者 affinity、unary、Laplacian fixed-point update、energy/convergence | Detectron I/O 换成公共 detector features/logits；singleton identity guard；无跨 batch 状态 |
| T²A | `33c8ccc64afdda260564123d6c790d030a89ff81` | patched vendored authors' public core | 作者 BaseAdapter/T2AAdapter、entropy、negative losses、全模型更新、gradient masking 控制流 | 仅修复公开版本阻断错误并拆分协议；详见下节 |
| IAPL | `a173e7783bbafaa00d60e6e31774a0bc14411a23` | pinned official checkout | 作者 `main.py`、CLIP、prompt adaptation、OIS、gate、condition、32 views、2 steps、官方协议 | CLIP 路径、PyTorch checkpoint 加载及 Arrow 输入兼容；源码不随项目分发 |

精确仓库 URL、上游文件、vendored core 和 wrapper 路径保存在 `configs/official_sources.yaml`。逐文件补丁边界见 `src/online_aig_tta/official/PROVENANCE.md`。运行 `python fetch_official_baselines.py all` 会 checkout 对照 commit；IAPL checkout 额外校验只存在 `patches/iapl-a173e77-compat.patch` 记录的批准 diff。

## 3. T²A 公开代码为何不能逐行执行

固定 commit 的公开 adapter 存在以下阻断问题：

- `e_margin`、`entropy_fn`、`filter_grad`、`cosine_strategy` 被调用但未初始化；
- Bernoulli 分支从 `B×C` 的 `1-p` 权重直接采样出 `B×C` target，后续 `gather` 要求每样本一个 target；修复版每样本采一个 class index，并显式排除当前伪标签；二分类时直接取另一类；
- `compute_noise_tolerant_negative_loss` 先 softmax，随后 normalized losses 又执行 log-softmax；
- model/optimizer reset 保存的不是 deep copy；
- 参数名包含 BN module name 的判断方向错误，不能可靠区分 BN 与非 BN 参数。
- 公开 `entropy_minimization` 额外使用 `1/exp(H-e_margin)` 重加权，但论文公式描述的是普通熵最小化，同时公开 YAML 没有给出 `e_margin`。

原样复制会得到运行错误或不符合论文公式的 loss。当前实现优先保留公开代码的控制流和熵重加权，并以二分类 `0.4*log(2)` 作为显式、可替换的 `release_repairs.e_margin`；其余阻断问题采用论文公式导向的最小修复。该选择必须做 `e_margin`/普通熵敏感性分析并向作者确认。这是可审计的修复版，不是未经修改的官方代码。

## 4. 公平主表的强制条件

- 所有 CNN 方法从同一 checkpoint 的哈希开始；结果 JSON 自动记录 SHA-256；
- 每个 `method × target × seed` 重新加载 checkpoint；
- 测试标签只进入 evaluator；
- 使用保存的 `sample_manifest.csv` 核对完全相同的样本顺序；
- 流顺序由 seed 控制但不按标签构造 batch；
- final holdout 与适应流使用同一采样 seed 的互斥切片，并以独立固定 seed 全局 shuffle；
- 每个域结束后在相同的已见域 holdout 上评估，形成 `holdout_matrix.csv`；forgetting 不再混用 online prefix 与 final holdout；
- 阶段性 holdout 评估固定并恢复 Python/NumPy/PyTorch RNG 状态，不能改变后续在线轨迹；
- T²A 的公共 `predict` 恢复 BN running buffers，所有持久状态更新只能发生在 `adapt`；
- batch size、分辨率、update steps 和输入预处理一致；
- `adapt` 调用机会每 batch 一次，但方法内部官方调度保留：RoTTA 每 64 个样本更新，LAME 零 optimizer step；
- 方法专属学习率只在独立 development generator 上选择；
- EATA 缺 Fisher 时必须失败；`require_fisher: false` 只能标作 ETA ablation；
- RoTTA 只能用预测类别维护 CSTU，禁止用隐藏标签平衡 memory；
- LAME 的适配输出在 `predict` 内完成且 `adapt` 不更新状态；其 final/forgetting 不能解释为跨域参数记忆；
- 连续流的 Final Average AUC 是域 AUC 的算术平均；pooled AUC 单独报告；
- IAPL 单独报告 Accuracy/AP，绝不把 AP 改名为 AUC。

效率表必须比较 `mean_total_ms_per_batch`。LAME 的优化发生在输出推断，所以主要记入 `predict_ms`；只比较 `adapt_ms` 会错误地把它显示为零开销。

## 5. 数值复现门槛

代码测试只证明接口和关键方程能执行。论文中使用“reproduced”一词前，还必须保存：

1. 官方仓库 commit 与工作区 diff；
2. 原始数据版本、checkpoint SHA-256、环境 lock、GPU 和 seed；
3. 作者原协议下的逐域数值及与论文/README 的绝对偏差；
4. 当前 common track 的完整配置与 `sample_manifest.csv`；
5. 至少 3 seeds 的均值与标准差。

IAPL runner 已自动检查所有配置域、mean 行以及 GenImage/UniversalFakeDetect 的作者参考 Accuracy/AP。UFD Arrow 19 域已完整运行，得到 95.41% Accuracy、97.21% AP：Accuracy 进入 1 个百分点容差，AP 相差 2.11 个百分点，因此状态为 `reference_gate_failed`，不能标记为论文数值复现成功。TENT、EATA、CoTTA、RoTTA、LAME、T²A 因本项目包内不含作者数据和权重，目前只有代码级测试；在对应原始 benchmark 数值跑完前，结果状态保持为 `not_run_requires_official_data_and_weights`。

## 6. 允许的论文表述

- TENT / EATA：`vendored authors' official core under a framework-owned AIGC Predict-Then-Adapt protocol`。
- CoTTA：`compatibility-patched vendored official ImageNet core under the common protocol`。
- RoTTA：`compatibility-patched vendored official core transferred from CIFAR PTTA to binary AIGC under the common protocol`。
- LAME：`vendored official parameter-free output-adaptation core using the common detector features and batches`。
- T²A：`patched vendored authors' public core, adapted to binary AIGC detection`。
- IAPL：`authors' official code at a pinned commit under its original episodic protocol`。

在官方数值门槛通过前，不应写“完全复现论文结果”。
