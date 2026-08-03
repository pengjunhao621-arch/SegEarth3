# Role-aware Prompt Test-Time Adaptation

## 1. 实验性质与边界

这是一个**冻结模型、无标签、按图重置的测试时优化（TTA）实验**，不是训练
方法，也不是 optimization-free 推理。SAM3 与 RemoteCLIP 权重均不更新；每个
视图只临时优化少量提示组合系数，处理下一张图或下一滑窗时重新从固定先验
开始。

本实验一次读取一张评测图并复用一次 SAM3 image backbone 与一次 RemoteCLIP
image encoder 结果，但并非“整个方法只有一次 decoder forward”：五条类别描述
需要分别经过冻结的 SAM3 grounding，融合后的类别提示也需要重新 grounding。
“同一次 forward 收集全部变体”在这里严格指同一次评测遍历、同一份图像特征和
同一套中间证据，而不是把不同提示错误地当成一次条件解码。

## 2. 保留的 SAM3 / baseline 结构

新路径默认关闭。关闭 `use_role_prompt_tta` 后，仍调用原始
`_inference_single_view_native`。开启后也保留：

1. SAM3 原始图像编码；
2. 同一条 contextual language condition 同时服务 semantic head、instance head
   与 Presence head；
3. 原始 object-score × Presence 候选过滤；
4. SegEarth-OV3 的 semantic/instance max fusion；
5. 最终 Presence 调制、滑窗顺序、类别聚合和阈值处理。

“按角色拆证据”只发生在提示调整依据中：semantic raw logits 用于不确定性，
Presence 用于限制是否允许调整，RemoteCLIP 只在其自身视觉—文本对齐空间评价
描述。不会把 RemoteCLIP 视觉向量直接加到 SAM3 的 256 维语言向量中，也不会
分别给 SAM3 三个 head 注入互不一致的提示。

## 3. 数据集提示池

每个数据集、每个类别固定五条描述：第 0 条是原始类别名，其余四条是由 Codex
参照标签定义和遥感俯视外观生成的受约束描述。所有描述都保留字面类别名，防止
描述脱离类别身份。提示池没有使用验证集标签或预测结果筛选：

- `configs/prompt_banks/udd5.json`
- `configs/prompt_banks/vdd.json`
- `configs/prompt_banks/vaihingen.json`
- `configs/prompt_banks/potsdam.json`
- `configs/prompt_banks/openearthmap.json`
- `configs/prompt_banks/loveda.json`

## 4. 方法链路

对类别 `c` 的五条描述，固定初始权重为：原始类别名 0.5，其余四条均分 0.5。

1. 用原始 baseline semantic raw logits 形成类别概率和归一化熵。
2. 每类只保留 baseline top-1 且低于图像平均熵的像素，再取最低熵的 20% 作为
   seed。此处不使用 final max-fusion 输出，避免 semantic 与 instance 自反馈。
3. 在 RemoteCLIP 的 16×16 patch 特征上汇聚 seed 原型；用同一个 RemoteCLIP
   文本塔评价五条描述，并对类内 affinity 去均值。它只决定“同一类别的哪条
   描述更符合当前图”，不负责跨类别判别。
4. baseline Presence 经 0.05 阈值映射为门控。无有效 seed 或低 Presence 的类
   不能接受视觉修正，也不能使用熵优化自由度。
5. 便宜的 surrogate 路径在预计算的五组 semantic raw maps 上做三步优化：

   `L_sur = class-balanced conditional entropy + 0.05 * KL(w || w_anchor)`

6. 主路径从 surrogate 权重开始，对最多四个 Presence/seed 合格的非背景类别
   各做一步真正穿过冻结 SAM3 grounding 的梯度更新：

   `L_e2e = binary entropy(class vs fixed strongest competitor on seed)
             + 0.05 * KL(w || w_start)`

   权重变化使用 `tanh` 限制，最大 logit 残差为 1.0；没有任何模型参数梯度。
7. 按有效 token mask 对五条完整 contextual language feature 序列做凸组合，
   再把融合后的**同一条**语言条件送回 SAM3 原生 grounding，让 semantic、
   instance、Presence 与 object score 按原流程共同重新计算。

当前第一轮不使用 focal loss 或 prototype diversity loss。中间统计会告诉我们：
类别/描述权重是否塌缩、seed 是否纯净、梯度是否有效、Presence 是否过度关门，
从而决定下一轮是否有证据加入这些约束。

## 5. 同次评测收集的 15 个变体

| 变体 | 回答的问题 |
|---|---|
| `baseline` | 受保护的原始查询路径 |
| `pool_max`, `pool_mean` | 多描述池直接聚合是否已有收益/噪声 |
| `anchor_output` | 固定 0.5 类别名锚点在输出空间是否有效 |
| `uniform_output` | 不区分描述的均匀输出组合 |
| `visual_output` | RemoteCLIP 视觉偏好本身是否有效 |
| `entropy_output` | 只有预测熵优化是否会自我强化 |
| `full_output` | 视觉 + Presence + 熵 + KL 的输出空间版本 |
| `full_no_anchor_output` | 去掉类别名质量先验与 KL 后是否失稳 |
| `full_no_presence_gate_output` | Presence 门控是否帮助或过度压制 |
| `anchor_regrounded` | 固定数据集语义先验重新进入 SAM3 是否有效 |
| `uniform_regrounded` | 均匀融合提示重新进入 SAM3 的效果 |
| `visual_regrounded` | 视觉加权提示重新 grounding 的效果 |
| `full_regrounded_surrogate` | surrogate 完整路径 |
| `full_regrounded_e2e` | 主方法：有限端到端 TTA 后原生重新 grounding |

所有变体共享同一图像特征、提示池和 baseline 证据；最终以精确 confusion matrix
重新汇总，不依赖逐图 mIoU 的简单平均。

## 6. 保存内容与判断方式

每张图的 rank-specific JSONL 保存：

- 每变体 confusion matrix、mIoU/aAcc、改变/修正/伤害/错到错像素；
- 每视图、每类别 seed 数量、seed entropy、GT seed purity（只做事后分析）；
- Presence gate、RemoteCLIP 类内 affinity、原型范数；
- anchor/visual/full/e2e 权重、winner、最大权重、effective prompt count；
- surrogate 与 e2e 每步 loss、KL、entropy、gradient norm；
- semantic/instance/Presence/final 相对 baseline 的变化；
- 融合语言相对字面类别锚点的残差范数；
- 少量降采样预测 NPZ，便于定位典型成功/失败图像。

汇总生成：

- `dataset_variants.csv`：数据集 × 变体；
- `per_class.csv`：类别 IoU 与变化；
- `class_diagnostics.csv`：类别效果与 seed/prompt/gate 机制量；
- `mechanism.csv`：优化与梯度健康度；
- `component_attribution.csv`：成对组件净贡献；
- `summary.json`：完整机器可读结果；
- `decision_report.md`：4/6 数据集正提升门槛与变体排名。

判断不只看主方法 mIoU：

- `anchor_regrounded` 好而 adaptive 不好：静态池有用，在线更新有问题；
- `visual_regrounded > uniform_regrounded`：RemoteCLIP 选择有正信息；
- `full_surrogate < visual` 且 entropy 下降：发生“更自信但更错”的自强化；
- `e2e > surrogate`：SAM3 内部重计算提供了 surrogate 缺失的信息；
- `no_presence > full`：Presence 门控可能把可恢复类别过早关掉；
- seed purity 与类别 ΔIoU 同向：优先改 seed；无关则优先查语言融合/损失；
- effective prompt count 接近 1 且收益下降：才有证据考虑 diversity；
- 稀有类改变少且错像素仍占主导：再考虑 focal/class reweighting，而不是预先加入。

## 7. 服务器命令

先检查代码、配置、RemoteCLIP 源码与权重：

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
python tools/preflight_role_prompt_tta.py --check-runtime-assets
```

单卡单图 smoke（会真实执行 1 个 e2e 类，主要用于发现接口/显存错误）：

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
ROOT=logs/role_prompt_tta_smoke \
GPU_LIST=0 NPROC=1 SMOKE_SAMPLES=1 \
bash tools/run_role_prompt_tta_v1.sh smoke
```

六数据集正式实验（不含 iSAID）：

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
ROOT=logs/role_prompt_tta_v1 \
GPU_LIST=0,1 NPROC=2 \
E2E_STEPS=1 E2E_MAX_CLASSES=4 \
bash tools/run_role_prompt_tta_v1.sh all
```

仅重新汇总已存在的 JSONL：

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
ROOT=logs/role_prompt_tta_v1 \
bash tools/run_role_prompt_tta_v1.sh summarize
```

若显存不足，首先保持算法不变，只把 `E2E_MAX_CLASSES=4` 调成 `2`；不要先改
损失或提示池。新的 `ROOT` 必须为空，脚本会拒绝向旧 JSONL 追加重复样本。
