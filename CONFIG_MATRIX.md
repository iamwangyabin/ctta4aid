# Configuration matrix

项目把数据/协议、方法超参数和单次实验入口拆成三层 YAML，避免为了改一个路径复制整份方法配置。

## 1. 已提供的配置

CNN 公平主轨道共有 `2 datasets × 2 settings × 7 methods = 28` 个可直接运行的实验文件：

| 数据 | 单目标独立重置 | 多生成器连续流 |
| --- | --- | --- |
| GenImage | `configs/experiments/genimage/single_target/*.yaml` | `configs/experiments/genimage/continual/*.yaml` |
| UniversalFakeDetect | `configs/experiments/universalfake/single_target/*.yaml` | `configs/experiments/universalfake/continual/*.yaml` |

每个目录均包含：`source.yaml`、`tent.yaml`、`eata.yaml`、`cotta.yaml`、`rotta.yaml`、`lame.yaml` 和 `t2a.yaml`。项目不再包含未发表的自定义 prototype/memory 方法。

另外提供：

- `configs/iapl_official_genimage.yaml`：IAPL 作者的 GenImage/CLIP 原生协议；
- `configs/iapl_official_ufd.yaml`：IAPL 作者的 UniversalFakeDetect/CLIP 原生协议；
- `configs/iapl_official_ufd_arrow_1gpu.yaml`：直接读取 UFD Arrow、按单 GPU 独立域运行的已披露工程协议；
- `configs/train/genimage_sd14_resnet50.yaml`：公共 CNN 源模型训练；
- `configs/train/universalfake_progan_resnet50.yaml`：公共 CNN 源模型训练；
- `configs/single_target.yaml`、`configs/continual_stream.yaml`：一次运行 GenImage 七个 CNN 方法的便利配置；
- `configs/universalfake_single_target.yaml`、`configs/universalfake_continual_stream.yaml`：对应的 UniversalFakeDetect 便利配置。

IAPL 没有伪造 `continual` 配置：作者公开实现是逐图 reset、图像间不累积参数，强行放进 continual 流会成为新的非官方算法。若以后做 IAPL-Online，必须另命名为协议对齐的项目实现，并与作者原生结果分开。

## 2. 继承关系

实验文件只包含两条继承：

```yaml
extends:
  - ../../../base/genimage_single_target.yaml
  - ../../../methods/eata.yaml
```

- `configs/base/` 固定数据域、batch size、分辨率、流顺序和 Predict-Then-Adapt 协议；
- `configs/methods/` 固定方法参数、官方参数含义、原始 batch size、上游配置和 commit；
- `configs/experiments/` 组合前两层，确保每个数据/设定/方法都有独立入口。

后出现的父配置覆盖前面的同名字段；本地实验文件可再覆盖开发域上选定的值。加载后的结果会记录 `_config_path` 和完整 `_config_sources`，便于审计。

## 3. 方法参数来源

| 方法 | 当前默认配置 | 原始写法/含义的保留方式 | 公共协议改动 |
| --- | --- | --- | --- |
| TENT | Adam, `lr=1e-3`, `beta=0.9`, `wd=0`, 1 step | 对齐官方 `cfgs/tent.yaml` 的 BETA/LR/WD | batch size 改为公共 16；拆为 Predict-Then-Adapt |
| EATA | SGD, `lr=2.5e-4`, `fisher_alpha=2000`, `d_margin=0.05` | 名称对齐作者 `main.py`；`e_margin=0.4*log(C)`，二分类取 `C=2` | 公共 checkpoint 提供源域 Fisher；batch size 16 |
| CoTTA | SGD, `lr=0.01`, `MT=.999`, `RST=.001`, `AP=.1`, `N=32` | YAML 同时写出语义名称和论文/代码符号 | 公共 batch size 16；缓存 teacher target 以拆分协议 |
| RoTTA | Adam, `lr=1e-3`, `ν=.001`, memory/update frequency `64`, `α=.05` | 对齐官方 `configs/adapter/rotta.yaml` 的 NU/MEMORY_SIZE/UPDATE_FREQUENCY/LAMBDA/ALPHA | 类别数改为 2、输入 224、batch 16；像素增强归一化桥接；缓存 EMA 输出后再入 memory |
| LAME | RBF, `k=5`, `λ=1`, 最多 100 步 | 对齐官方 defaults、overall-best YAML 及 `laplacian_optimization` 默认值 | 取公共 CNN 的 penultimate feature/logits；无参数更新；单样本 batch 退回 source 输出 |
| T²A | Adam, `lr=1e-4`, `psi=.01`, `gamma=2`, `alpha=1`, `beta=1` | 对齐公开 `configs/T2A.yaml` | 公开代码缺失的字段单独放在 `release_repairs`，不能当官方默认 |
| IAPL | 32 views, `selection_p=.2`, `tta_steps=2`, `lr=.005` | 名称直接沿用作者 shell/argparse | 不进入 CNN 公平主表；保留 per-image reset 和 Adapt-Then-Predict |

公共 CNN batch size 16 是研究协议，不是所有论文的原始 batch size。原始值保存在每个方法的 `reference.original_batch_size`。LAME 原始默认恰好也是 16；这不代表其他设置也与其原 benchmark 完全相同。

基础配置的 `updates_per_batch: 1` 表示框架对每个测试 batch 调用一次 `adapt`，不是强迫每种算法执行一次 optimizer step。RoTTA 按官方 `UPDATE_FREQUENCY=64` 累积样本后更新，公共 batch 16 时通常每四个完整 batch 更新一次；LAME 完全不执行 optimizer step。为了保留官方算法，不能把这两者偷偷改成“每 batch 梯度一步”。

## 4. 路径环境变量

运行前至少设置对应变量：

```bash
export GENIMAGE_ROOT=/data/GenImage
export GENIMAGE_SOURCE_CHECKPOINT=/weights/genimage_sd14_resnet50.pt
export UFD_TEST_ROOT=/data/UniversalFakeDetect/test
export UFD_SOURCE_CHECKPOINT=/weights/ufd_progan_resnet50.pt
```

不解包直接读取 Arrow 时使用：

```bash
# This local bundle has six domains only; do not use it for the full 8-domain mean.
export GENIMAGE_TEST_ARROW_ROOT=/data/DF-arrow-data/GenImage_test
export UFD_FORENSYNTHS_ARROW_ROOT=/data/DF-arrow-data/ForenSynths
export UFD_OJHA_ARROW_ROOT=/data/DF-arrow-data/Ojha
```

训练 UniversalFakeDetect 源模型还需要：

```bash
export UFD_TRAIN_ROOT=/data/UniversalFakeDetect/train
export UFD_SOURCE_VAL_ROOT=/data/UniversalFakeDetect/source_val
```

两个 UFD root 都应以 `progan/` 作为下一层目录；训练与 Fisher/validation root 必须互不重叠。

IAPL 原生轨道使用独立的数据布局和变量：

```bash
export IAPL_GENIMAGE_ROOT=/data/IAPL-layout/GenImage
export IAPL_UFD_ROOT=/data/IAPL-layout/UniversalFakeDetect
export IAPL_GENIMAGE_CHECKPOINT=/weights/iapl_genimage.pth
export IAPL_UFD_CHECKPOINT=/weights/iapl_ufd.pth
export CLIP_VIT_L14_CHECKPOINT=/weights/ViT-L-14.pt
```

IAPL 的 Arrow UFD 配置还需要 `IAPL_REPO_PATH`，入口是
`configs/iapl_official_ufd_arrow_1gpu.yaml`。

## 5. 运行示例

```bash
python run_single_target.py \
  --config configs/experiments/genimage/single_target/eata.yaml

python run_continual_stream.py \
  --config configs/experiments/universalfake/continual/cotta.yaml

python run_continual_stream.py \
  --config configs/experiments/genimage/continual/rotta.yaml

python run_single_target.py \
  --config configs/experiments/universalfake/single_target/lame.yaml

python run_iapl_official.py --config configs/iapl_official_genimage.yaml
```

正式表格应运行 3 个 seed。当前 YAML 的 `seed` 是单次实验 seed；为每个 seed 保存一份冻结配置和独立输出目录，不要在同一目录覆盖结果。超参数只能在预先声明且不进入最终测试表的 development generator 上选择。
