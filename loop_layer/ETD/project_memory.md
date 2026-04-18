# ETD / BoolQ

- `aps/super_glue` 的 BoolQ **test** 划分 `label=-1`，无法算准确率；评测请用 **validation**。
- ETD 前向在 `k=1` 且 `N_E+N_T+N_D=L` 时与标准 `Qwen3Model` 前向数值一致（已用 max abs diff 校验）。
- `experiments/hf_hub_network_env.sh`：先 `unset` 代理后探测 `hf-mirror.com`；可达则用镜像并**保持无代理**；不可达则 `source /etc/network_turbo`，再试镜像，仍失败则 `HF_ENDPOINT=https://huggingface.co`。扫参入口脚本已 `source` 该文件。
- LogiQA：`datasets>=3` 不再加载带 `logiqa.py` 的 `EleutherAI/logiqa`；`hard_mc_benchmark_loaders.load_logiqa` 在 `RuntimeError` 时回退到 **parquet** 数据集 `fireworks-ai/logiqa`（选项去掉 `A.` 前缀后与 lm-eval 模板一致）。

## R33 信号失败根因（避免重蹈）

- **FFN Gini / 激活熵 / boundary_frac**：测量激活分布形状，由 `gate_proj` 权重矩阵决定，与输入几乎无关。layer8 处跨所有 benchmark CV=1.7%，曲线完全重叠。**不要再用**。
- **Attn Spectral Gap (max/2nd)**：被 layers 5–8 的 BOS sink token 主导，之后降至 ~1，对 T-block（layer 8+）毫无区分力。**不要再用**。
- 教训：**方向 > 分布形状**；**跨模块关系 > 单模块统计**；**有限差分 > 间接推断**。

## R34 核心发现（2026-04-14）

- **`cross_cos_a_m = cos(a_l, m_l)`**（attention 输出与 FFN 输出的方向余弦）是迄今最强的 T-block 定位信号：
  - 早期层（0–5）：≈0（独立写入）
  - 中层（8–22）：持续负值（竞争区），负向峰值约 -0.2 到 -0.4
  - 后期层（22+）：回升至 0 附近
  - **所有 8 个 benchmark 的 T-block 完全落在竞争区（cos < 0）内**
- **`cross_attn_to_ffn_sens`**（有限差分：移除 attention 贡献后 FFN 变化比例）在 Qwen3-8B 上普遍在 layer 16–22 达到峰值（~0.65），与 benchmark 种类关系不大，但峰值幅度与任务难度负相关（GPQA ~0.52，简单任务 ~0.66）。
- **ETD 机制新解**：第二遍循环的价值在于在 attention-FFN **竞争区**（cos < 0）用改写后的 hidden 重新协调两类记忆的对抗方向，而非简单"增加计算深度"。
- **AGIEval 异常**：中层（15–19）`cross_cos_a_m` 为正（协同），ETD 在协同区循环可能反而过度放大已有偏置，这解释了 AGIEval ETD 增益较小的现象。
- **t_start 判据（新）**：从 `cos(a_l, m_l)` 进入稳定负值区域的层开始；**t_stop 判据（新）**：在 cos 从负值回升至 ~0 之前结束。
- Qwen3-8B 的 Decoder Layer 结构：`residual1 = h; h = LN1(h); h,_ = self_attn(h,...); h = residual1 + h; residual2 = h; h = LN2(h); h = mlp(h); h = residual2 + h`。子层 hook 用 pre-hook（层输入） + self_attn output hook + mlp output hook + 层 output hook。
- R34 派生图：`experiments/plot_r34_derived_signals.py` 读 `results/r34_cross_memory_data_full.json`，对 12 个信号各画 **去全层均值**、**相邻层差分（对层均值）**、**样本方差**；输出到 `figures/r34_cross_memory/derived/`。`run_r34.sh` 主实验成功后会自动跑该脚本。

## R35 精确交换子实验（2026-04-14）

### 核心发现（3 条必记）

1. **交换子 norm 在后期层（28–35）爆炸性增长（~7倍于中层）**，与 T-block（通常层 8–22）完全不对齐。H1 被**证伪**：commutator_norm 不能定位 T-block。根本原因：交换子 norm 与残差流 hidden state norm 正相关，而 norm 随深度单调增长（Pre-LN 架构的固有特性）。

2. **T-block 内 cos(T1, T2) 持续负值（-0.09 到 -0.18）**：Term1（context→knowledge）和 Term2（knowledge→context）在 T-block 区域方向反向，约 30% 的向量量消（cancellation_ratio ≈ 0.65–0.71）。这说明 ETD 第二遍确实激活了与第一遍**反向耦合**的 Attention-FFN 交互，但这个反向耦合在所有 benchmark 上几乎相同，无法区分不同 T-block 边界。

3. **GPQA-Diamond 的 T2/(T1+T2) = 0.427，是所有 benchmark 中最高的**（其他约 0.35–0.40）；同时 cos(T1,T2) 最负（-0.184）。GPQA 中 "FFN 知识改变 Attention 上下文检索" 的比例最高，与其 ETD 增益最小（+2%）一致——知识→上下文的逆向干扰最强时，ETD 的协调效益反而受限。

### 信号失效原因总结（避免重蹈）

- **绝对 commutator norm**：由深度决定（hidden state norm 随层增大），不是 T-block 信号。**不要再用**。
- **累积 Σ||C_l|| vs ETD delta 相关性**：r≈-0.054，零相关，被 T-block 宽度差异掩盖。**不要再用**。

### 下一步最有价值的方向

- **传播增益实验**：对 $h_l$ 加归一化 $C_l$，重跑后续层，测 logit JSD 变化。这是交换子理论唯一未验证的部分（实验代码结构在 exp_r35 的 Phase 2 代理 A 中已设计）。
- **cos(T1,T2) 作为循环停止条件**：在量化"第二遍 Attention/FFN 耦合方向差异"上有一定意义，可试用作早停信号（阈值约 -0.05）。

### 技术笔记（Attention 重跑）

- `self_attn` pre-hook 需使用 `with_kwargs=True`（PyTorch >= 2.0），捕获 `position_embeddings`（RoPE cos/sin 张量）等参数；这些参数仅依赖 `position_ids`（序列位置），不依赖 hidden state 内容，可安全复用。
- 每层额外：1 次全序列 MLP forward + 1 次全序列 Attention forward，总额外耗时约 39s（8 bench × 20 样本）。

## R36 方向特异性传播增益实验（2026-04-14，**N=100**/bench 重跑）

### 核心发现（5 条必记）

1. **prop_sens 随层数单调递减**（逆直觉）：单位扰动 JSD 早期层更大、后期更小（logit 已锐化）。N=100 与 N=20 形态一致。

2. **DA = prop/rand：均值不可信；中位数≈1（N=100 强力结论）**  
   全部 7 bench 上 **DA 的 T-block 与 late 的中位数均在 ~0.97–1.11**，与 1 无系统偏离 → **典型样本上交换子方向并不优于随机方向**（H2 在 median 意义下证伪）。均值仍会因 `rand_sens→0` 爆炸（例 MMLU late mean≈85）。`r36_propagation_stats.json` 已写 `directional_advantage_*_median` 等字段。

3. **etd_effective 中位数：T-block > late 仍成立（6/7；GPQA late median 略负）**  
   复合信号在扩容样本后仍是最稳的“T-block vs 后期”标量之一。

4. **comm_persist**：**6/7** bench 上 T-block **均值**仍 > late；**AGIEval** 例外——均值几乎重合（0.077 vs 0.076），且 **median late(0.100) > median tblock(0.078)**，不要用 comm_persist 单信号套该任务。其余如 ARC-C tblock median 0.189 vs late 0.059。L18 的 `comm_persist@L18_mean` 峰值：MMLU **0.44**，ARC-C **0.34**，TruthfulQA **0.33**。

5. **H5 重表述**：**median 下 late 与 tblock 的 DA 均≈1**（支持“随机与交换子等效”的去混淆故事）；**均值下 H5 仍不成立**（离群）。

### 有效信号总结（R34–R36 跨轮次）

| 信号 | 来源 | 状态 | 说明 |
|------|------|------|------|
| `cos(a_l, m_l)` | R34 | ✅ 推荐 | T-block 内持续负值（竞争区），最稳定 |
| `cos(C_l, Δh_l)` | R35 | ✅ 可用 | T-block 内高于后期层，配合 DA 使用 |
| `comm_persist` | R36 | ✅ 新推荐 | T-block 内持续 > late，廉价（无额外 forward） |
| `etd_effective` | R36 | ✅ 组合信号 | = cos_res × DA，余弦项去噪，区分力最优 |
| `commutator_norm` | R35 | ❌ 禁用 | 与深度正相关，非 T-block 信号 |
| `DA = prop/rand` | R36 | ⚠️ 均值禁用 / **median 可用** | N=100：median≈1；报告须带 median |

### 技术笔记（Hook Injection）

- `register_forward_pre_hook` 注入扰动到 `h[:, -1, :] += perturb.to(t.device, dtype=t.dtype)`（仅最后 token），perturb 以 CPU float32 存储在 closure 中，在 hook 内转为 t 的设备和 dtype。
- `safe_cos` 必须在内部 `.cpu()` 两个张量（commutator 在 CUDA，prev_commutator 已 `.cpu()`），否则 device mismatch。
- 默认 **N_SAMPLES=100**；每样本后 `torch.cuda.empty_cache()` 降碎片 OOM 风险。总耗时约 **12min**（7 bench × 100 × 主 forward + 22 扰动 forward）。

## R37 信号引导 ETD 选层实验（2026-04-14，N=100，硬推理 benchmarks）

### 实验目标

验证 `cos(Term1, Δh_l)` 信号（Term1 近似交换子）能否在 test-time 自动选择 ETD 循环区间，免除人工扫参。

### 关键结果（两轮 R37a + R37b）

| Benchmark | Baseline | Oracle | **最佳信号方法** | **Δ vs Baseline** | 方法名 |
|-----------|---------|--------|----------------|-----------------|--------|
| MMLU-HS-Math | 0.40 | 0.43 | **0.43** | **+3pp** | onset_cos8 |
| GPQA-Diamond | 0.38 | 0.33* | **0.40** | **+2pp** | persample_cos8 |
| AGIEval | 0.52 | 0.54 | **0.58** | **+6pp** | persample_variable |

*GPQA oracle 手工推导有误（实际低于 baseline 5pp），信号方法反超。

### 三条必记发现

1. **onset_cos8（阈值 0.28, n_t=8）在 MMLU 上免扫参即得 oracle 水平（100%）**  
   onset 准则自然选出 [12,20]，与 R30 扫参最优 [10,18] 几乎重合。cos_res 首次超阈值对应 ETD t_start 甜点。

2. **persample_variable（n_t ∈ {4,6,8} 全搜索）在 AGIEval 超越 oracle +4pp**  
   选出 [15,19]（n_t=4），比 oracle [13,20] 更短更晚，适配了 AGIEval 的高质量短窗口特性。

3. **每样本选层（persample）始终优于全局固定窗口（global）**：H2 对所有 3 个 benchmark 成立。

### 信号失效场景

- **onset_cos8 对 AGIEval 失败（-5pp）**：AGIEval 样本的 cos_res 在 L12-L17 普遍低于 0.28，onset 落到 L18（晚期），与最优 L15 区偏离。固定阈值 0.28 对不同 benchmark 不通用。
- **persample_variable 对 MMLU 失败（-3pp）**：n_t=4 短窗口在 MMLU 上无效，全搜索偏向高 cos_res 短窗口但 MMLU 需要 n_t=8 宽覆盖。

### 技术要点（Term1 快速计算）

```python
# 探针前向后，每个探针层:
m_l0 = layer.mlp(layer.post_attention_layernorm(h_i))  # [1,1,D]: 交换 FFN/Attn 顺序的近似
term1 = m_l_actual - m_l0                              # 注意力对 FFN 输出方向的影响
delta_h = a_l + m_l_actual                             # 实际残差更新
cos_res_l = safe_cos(term1, delta_h)                   # 对齐度（越高越应循环）
```

探针层每隔 2 层（L6, L8, ..., L28），总开销 < 1.5x baseline forward。

### 下一步（方向 A）

用 ≤20 样本校准每 benchmark 阈值（calibrated_onset），预期 MMLU/GPQA/AGIEval 全部达 oracle 水平。或两阶段：onset 确定 t_start → variable_nt 确定 n_t。

## R38 全 Benchmark 信号引导 ETD（2026-04-15，N=100，8 benchmarks）

### 实验结论（三轮迭代）

**普适最佳信号：`persample_var`（逐样本 n_t∈{4,6,8}，min_start=9）**
- 宏平均 Δacc=+0.014（vs baseline；扫参最优=+0.054）
- 4/8 benchmark 上优于 baseline，4/8 上是所有信号方法赢家
- 在 AGIEval（+6pp）、LogiQA（+6pp）、MMLU（通过 calib_onset8 +5pp）表现最优

### 关键结果表

| Benchmark | Baseline | 扫参最优 | 最佳信号 | %扫参 | 方法 |
|-----------|---------|--------|--------|------|------|
| BoolQ | 0.82 | 0.87 | **0.84** | 96.6% | 逐样本-变长 |
| ARC-C | 0.56 | 0.58 | **0.56** | 96.6% | 标定全局-8 |
| TruthfulQA | 0.32 | 0.38 | **0.36** | 94.7% | 逐样本-变长 |
| CSQA | 0.64 | 0.69 | **0.68** | 98.6% | 逐样本-8层 |
| MMLU-HS-Math | 0.40 | 0.43 | **0.45** | 104.7% | 标定Onset-8 |
| GPQA-Diamond | 0.38 | 0.44 | **0.40** | 90.9% | 逐样本-8层 |
| AGIEval | 0.52 | 0.54 | **0.58** | 107.4% | 逐样本-变长 |
| LogiQA | 0.36 | 0.50 | **0.42** | 84.0% | 逐样本-变长 |

### 三条必记教训

1. **早期层（L6-L8）存在高 cos_res 假阳性峰（0.40-0.52）**  
   这些峰是模型初始化特性，不是 T-block 信号。ETD 在 L6-L8 循环会导致准确率崩溃（ARC-C 0.56→0.29！）。**min_start 必须 ≥ 9**，不能扩展到更早的层。

2. **n_t 扩展到 {4,6,8,10,12,14} 被早期假阳性占据**  
   R38b 验证：即使早期层 cos_res 最高，ETD 窗口包含这些层也是有害的。宽 n_t 候选与低 min_start 组合是双重错误。

3. **MMLU 和 AGIEval 上信号方法超越扫参最优**  
   `calib_onset8` 在 MMLU 达到 0.45（扫参最优 0.43，超越 +2pp）；`persample_var` 在 AGIEval 达到 0.58（扫参最优 0.54，超越 +4pp）。这证明信号方法不只是"接近"扫参，在某些任务上可以超越。

### 信号失效场景（补充 R37）

- **BoolQ**：最优窗口 [8,22]（n_t=14），超出 n_t∈{4,6,8} 搜索空间，且 t_start=8 低于 min_start=9。本质上需要非常宽的早层循环，当前信号框架无法支持（扩展 min_start 又引入假阳性）。
- **LogiQA**：最优窗口 [14,19]（n_t=5），介于 n_t=4 和 n_t=6 之间，候选集无精确匹配。
- **GPQA**：最优窗口 [18,20]（n_t=2），比最小候选 n_t=4 还窄。

### 技术要点

- LogiQA 加载：`EleutherAI/logiqa` 在 offline 模式下抛出 `HfHubHTTPError`（不是 `RuntimeError`），需直接使用 `fireworks-ai/logiqa`（已缓存）。
- 标定阶段 N_CALIB=20 即可稳定推导 mean cos_res profile，比全量评测节省 80% 计算。
- 5 种可视化：全 benchmark 条形图、热力图、Δacc 散点图、标定 profile（含早期假阳性标注）、t_start violin 图。

## R38-Multimodel：Llama3-8B / Gemma2-2B（与 R38 同条件、8 benchmarks）

- **脚本**：`experiments/exp_r38_multimodel_signal.py`，`--preset llama3-8b | gemma2-2b`；一键 `run_r38_multimodel.sh`。
- **sweep_best**：从各模型已有 `etd_layer_sweep_r30style.json` + `hard_mc/...` 按任务取**准确率最高**的 `(t_start,t_stop)`，不沿用 Qwen 表。
- **Gemma2 Term1**：`mlp(pre_feedforward_layernorm(h_i))`；Llama 与 Qwen：`mlp(post_attention_layernorm(h_i))`。
- **探针范围**：Llama `min_start=8,max_start=20`，probe 6..26 步长 2；Gemma `min_start=5,max_start=16`，probe 4..22 步长 2。
- **输出**：`results/r38_multimodel_llama3_signal.json`、`results/r38_multimodel_gemma2_signal.json`，图在 `figures/r38_multimodel_*/summary_multimodel.png`。

## R40 BBH+GSM8K（lm-eval 任务 + Hub 数据）

- **`httpx.ConnectTimeout` / SSL 握手超时**：常见于 `http_proxy`/`https_proxy` 指向不可达或慢速代理；`datasets` 经 `huggingface_hub` 下载 parquet 时会走该代理。**使用 `HF_ENDPOINT=https://hf-mirror.com` 时应在 shell 与脚本中 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY`**，勿再依赖 `source /etc/network_turbo` 强塞代理。
- **`run_r40_bbh_gsm8k.sh`** 已在开头 unset 上述变量，并设置 `HF_HUB_DOWNLOAD_TIMEOUT`（默认 300s）；**`exp_r40_bbh_gsm8k_etd.py`** 在 import 重型库前同样 pop 代理并 `setdefault HF_HUB_DOWNLOAD_TIMEOUT`。
- **小样本冒烟**：`--bbh-limit` 很小时原先 `len(items) < 5` 会跳过整个 BBH；已改为仅 `not items` 时跳过，便于 2 条文档仍跑通 MC+ETD 路径（标定子集自动缩短到可用条数）。
