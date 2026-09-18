# EMDAInvNet 论文复现

这是对 Kong et al., **EMDAInvNet: A Deep Learning Framework for 3-D
Permittivity Inversion of Underground Pipelines**, IEEE Sensors Journal,
2026, DOI: `10.1109/JSEN.2026.3687430` 的 clean-room PyTorch 复现。

作者没有在论文中给出可确认的 EMDAInvNet 代码仓库。本工程依据公开的 Fig. 2、
式 (3)-(17) 和 Section III 进行 clean-room PyTorch 复现。图、公式和可训练性冲突时，
整体网络拓扑以 Fig. 2 为依据，模块细节同时记录公式字面版和在公开 Dataset II 上验证
可训练的解释。论文未披露的
基础通道、attention reduction、spatial kernel、上采样实现、归一化方式及 LR decay
执行频率采用显式工程默认值。因此本工程可用于复现实验流程和消融趋势，但不宣称等同于
作者未公开的原始实现，也不承诺逐位复现 Table II/III。

## 已实现内容

- `64 x 64 x 64` 单通道 C-scan 到三维介电常数体的端到端映射；
- Fig. 2 的通道比例 `C -> 3C -> 6C -> 12C -> 24C -> 32C`；
- EMSFA：三层串行 `3x3x3 Conv + InstanceNorm + LeakyReLU`、多尺度拼接、
  `1x1x1` 融合、channel attention、spatial attention；
- DilatedBlock：膨胀率 `1/2/4/8` 的纯 3-D 卷积分支，统一融合、IN、LReLU 后加残差；
- 论文消融开关 `use_emsfa`、`use_dilated`；
- Adam、MAE、初始学习率 `1e-3`、每 epoch 乘 `0.98`、batch size 4、100 epochs；
- MAE、MSE、SSIM、MAPE、IoU、Dice；
- `.mat`（3DInvNet 的 `clean_data`/`mask`）、`.npy`、`.npz`；
- Dataset III 的预训练权重微调、单体积推理与 JSON 评估报告。

## 环境

本机已经验证可用的环境是：

```powershell
conda activate py310
cd C:\Users\Koyuu\PycharmProjects\RS\GPR\model\EMDAInvNet
python -m pip install -e .
python smoke_test.py
```

本次验证使用 PyTorch `2.9.1+cu128` 和 CUDA。严格按式（4）-（11）恢复各分支完整
通道后，`base_channels=8` 的模型为 16,637,852 参数；单个 `64^3` 体前向已通过。
训练显存还包含梯度、优化器状态和中间激活。

## 2026-09-16 严格核对修正

首版 423 万参数实现不能作为论文复现，已废止。塌缩归档保留为反例，但其 checkpoint
与新架构不兼容。修正内容如下：

- EMSFA 的三层卷积现在各自保留完整紧凑通道宽度，再拼接为 `3C`；旧版错误地又除以 3；
- 式（10）的 dilation `1/2/4/8` 分支现在各自保留完整通道；旧版错误地缩为 1/4；
- Fig. 2 的 bridge 改为 `Conv+IN+LReLU -> DilatedBlock -> Channel -> Spatial`，不再误用 EMSFA；
- `InstanceNorm3d` 按 PyTorch 标准实现使用 `affine=False`；论文未说明 learnable affine；
- 保留 Fig. 2 明确画出的输出 Sigmoid；
- Dataset II 改用正确物理范围 `[4,27]` 做 min-max，不再错误混入 Dataset III 的 49.92。

## 2026-09-18 公式字面版回退

曾按式（9）—（11）的字面形式同时做过三项修改：dilation branch 改成裸卷积、
transition 改成裸卷积、residual 改为激活后相加。真实 Dataset II/noisy_data 的 100-epoch
任务在前 23 轮完全塌缩，IoU 始终为 0，`pred_std` 降到约 `5.6e-9`。三项单变量短跑
也没有任何一项能单独恢复学习，其中 transition 的 IN/LReLU 对稳定性影响最大。

因此默认实现恢复为已在相同数据、seed 和训练配置上验证成功的组合：

```text
dilation branch: Conv + IN + LReLU
transition:       1x1 Conv + IN + LReLU
residual:         LReLU(IN(Fusion(...)) + T)
```

该组合在 40 epochs 的公开 Dataset II noisy 输入实验中达到 IoU `0.8012`、Dice `0.8862`；
其最佳 checkpoint 已在当前代码中重新加载并得到相同指标。配置仍保留三个显式开关，
可复现实验过的公式字面版，但它不再作为默认 baseline。

完整训练前先在服务器用真实 Dataset II 做严格预检：

```bash
python strict_preflight.py --config configs/paper.yaml \
  --manifest data/manifests/std_train.csv \
  --output outputs/paper_default/preflight.json
```

预检锁定 64³ 输入、16,637,852 个参数、输入/目标范围、初始化输出方差、输入依赖和
非零梯度。训练日志另记录 `pred_std`、`pred_input_delta`、目标区及背景区预测均值，
并在 epoch 1/5/10/20/30/40/60/80/100 保存独立 checkpoint。损失仍是论文的
plain voxel-wise MAE。

论文没有写出标签归一化公式，因此不能宣称 min-max 是作者代码原式；它是连接 Fig. 2
的 Sigmoid 与 Dataset II 物理范围时最常规、最少附加假设的实现。旧版
`(eps_r-4)/(49.92-4)` 明确把 Dataset III 上限混入 Dataset II，已经废止。若严格数值
仍无法对齐，必须把这一未披露项作为受控敏感性实验，而不能暗中改损失。

## 数据准备

论文共有三套数据：

1. Dataset I：作者自建的 1000 组实心圆柱 gprMax 数据，900/100；论文未发布下载地址。
2. Dataset II：3DInvNet 公开数据，共 6000 组，5850/150。原项目给出的下载地址：
   <https://drive.google.com/drive/folders/12mG7lA8g8KX55KPHmP9y3n4XxFZjLp4W?usp=sharing>
3. Dataset III：同一公开资源中的 232 组实测数据，220 组微调、12 组测试。

当前本机镜像只有少量 Dataset II 样本；服务器保存完整 5850/150 数据。下载并解压后，
若结构为 `dataset/train/mask1`（输入）和 `dataset/train/mask2`（标签），执行：

```powershell
python prepare_dataset.py `
  --inputs ..\3DInvNet\dataset\train\mask1 `
  --targets ..\3DInvNet\dataset\train\mask2 `
  --manifest data\manifests\dataset_ii_train.csv

python prepare_dataset.py `
  --inputs ..\3DInvNet\dataset\test\mask1 `
  --targets ..\3DInvNet\dataset\test\mask2 `
  --manifest data\manifests\dataset_ii_test.csv
```

建议从训练集固定划出验证集，用验证集选最优权重，最后只在测试集报告一次。论文原文使用
test set 选最优权重，这会产生测试集信息泄漏；若追求论文流程完全一致，可让
`val_manifest` 指向测试 manifest，但报告时必须注明。

输入约定为 `[D,H,W]`。论文对 Dataset I 先做 time-zero calibration、mean filtering，
再把原始 `11 x 10 x T` 插值为 `64^3`；Dataset III 用 trilinear interpolation 将
`21 x 88 x 512` 对齐为 `64^3`。本工程可在配置中启用 `direct_wave_suppression: true`
做均值道相减，但 time-zero 的拾取规则未披露，不能凭空固定。

## 训练与微调

```powershell
# Dataset II / 论文默认实验
python train.py --config configs\paper.yaml

# Dataset III：用 Dataset I 的权重初始化后微调
python train.py `
  --config configs\finetune_measured.yaml `
  --pretrained outputs\dataset_i\best.pt
```

输出包含 `best.pt`、`last.pt`、`metrics.csv` 和完整解析后的配置。AMP 默认开启，可用
`--no-amp` 关闭。

## 推理和评估

```powershell
python infer.py `
  --config configs\paper.yaml `
  --checkpoint outputs\paper_default\best.pt `
  --input sample.mat `
  --output outputs\sample_prediction.npz

python evaluate.py `
  --config configs\paper.yaml `
  --checkpoint outputs\paper_default\best.pt `
  --manifest data\manifests\dataset_ii_test.csv `
  --output outputs\paper_default\test_metrics.json
```

推理文件同时保存归一化结果 `normalized_permittivity` 和还原量纲后的
`permittivity`。

## 消融实验

`configs/ablation_*.yaml` 记录三个变体的关键开关；可直接从完整论文配置通过命令行运行：

```powershell
python train.py --config configs\paper.yaml --disable-emsfa --disable-dilated --output-dir outputs\ablation_baseline
python train.py --config configs\paper.yaml --disable-dilated --output-dir outputs\ablation_emsfa_only
python train.py --config configs\paper.yaml --disable-emsfa --output-dir outputs\ablation_dilated_only
python train.py --config configs\paper.yaml --output-dir outputs\ablation_full
```

| DilatedBlock | EMSFA | 论文 Table III IoU / Dice |
|---|---|---|
| 关闭 | 关闭 | 15.36% / 26.26% |
| 关闭 | 开启 | 86.97% / 92.92% |
| 开启 | 关闭 | 84.97% / 91.61% |
| 开启 | 开启 | 87.31% / 93.16% |

## 未披露与复现边界

- 基础通道 `C`、attention reduction、spatial attention kernel 未报告；
- 权重初始化、padding、精确 LeakyReLU slope 未报告；
- 标签归一化公式和二值化阈值未报告；
- Dataset III 的“reduced learning rate”没有数值；
- Dataset I 的模型随机放置、方向分布、time-zero 算法和 gprMax `.in` 文件未公开；
- Fig. 2 与文字给出通道比例和模块次序，但没有逐层参数表。
- `3D Upsample` 的具体算子未报告；当前 `ConvTranspose3d` 是工程默认值。
- 学习率衰减只报告了 0.98，未报告执行频率；当前按每 epoch 衰减解释。

因此，当前最可验证的复现目标是：代码拓扑与论文一致、Dataset II 上优于 3DInvNet
基线、两个模块的消融趋势与 Table III 一致。精确数值复现需要作者代码或补充材料。
