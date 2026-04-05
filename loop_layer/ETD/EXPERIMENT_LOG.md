# ETD 实验记录（Qwen3-8B）

本文档汇总在 `/root/autodl-tmp/loop_layer/ETD` 中完成的 **Encode–Think–Decode（ETD）** 复现实验：角距离采集、Kneedle 分层、固定 \(k\in\{2,3\}\) 的中间块循环，以及在 BoolQ / ARC 上的准确率评测。

---

## 1. 环境与模型

| 项目 | 取值 |
|------|------|
| 模型 | Qwen3-8B（`Qwen3ForCausalLM`） |
| 本地权重路径 | `/root/autodl-tmp/model_qwen` |
| 层数 \(L\) | 36（`num_hidden_layers`） |
| 代码根目录 | `/root/autodl-tmp/loop_layer/ETD` |
| 依赖 | 见 `requirements.txt`；角距离 Kneedle 使用 `kneed` |
| 数据 | 离线优先：`HF_DATASETS_OFFLINE=1`，使用本机 HuggingFace 缓存 |

---

## 2. 算法要点（固定 \(k\)，无 ACT）

- **角距离**（最后一 token，相邻隐状态）：\(d^{(l)}=\frac{1}{\pi}\arccos\frac{\langle x^{(l)},x^{(l+1)}\rangle}{\|x^{(l)}\|\|x^{(l+1)}\|}\)，对样本加权平均得到每层一段距离序列。
- **Kneedle**：对正向距离序列与**反向**距离序列分别找拐点，得到 \(N_E\)、\(N_D\)，再令 \(N_T=L-N_E-N_D\)。
- **前向**：`Embed → E（前 \(N_E\) 层）→ T 重复 \(k\) 次（中间 \(N_T\) 层）→ D（后 \(N_D\) 层）→ norm → lm_head`。
- **评测**：多选题用 **continuation 的 token 对数似然之和** 比较选项（BoolQ 二选一，ARC 四选一）。

---

## 3. 两次角距离采集分别选了哪些层

角距离与分层 **按数据集各做了一次**（数据源与文本不同，参数一致）：

| 采集任务 | 数据与采样 | 输出文件 |
|----------|------------|----------|
| **第 1 次：BoolQ** | `aps/super_glue` / `boolq`，`train` 随机 **128** 条（`seed=0`），提示格式 `passage + question + Answer:` | `artifacts/boolq_layers.json` |
| **第 2 次：ARC** | `allenai/ai2_arc` / `ARC-Challenge`，`train` 随机 **128** 条（`seed=0`），提示格式 `Question: ...\nAnswer:` | `artifacts/arc_layers.json` |

### 3.1 分层结果（两次数值一致）

两次 Kneedle 得到的 **块大小** 相同：

| 符号 | 层数 | 说明 |
|------|------|------|
| \(N_E\) | **3** | Encoder |
| \(N_T\) | **30** | Thinking（可被循环 \(k\) 次） |
| \(N_D\) | **3** | Decoder |
| 合计 | \(3+30+3=36\) | 与 Qwen3-8B 总层数一致 |

### 3.2 与 `model.model.layers` 下标的对应关系（Python 0-based）

以下与 `transformers` 中 **单层一块** 的编号一致（`layers[0]` 为最底层）：

| 块 | 层下标范围（含端点） | 层编号（1-based，便于对照论文） |
|----|----------------------|----------------------------------|
| **E（Encoder）** | **`0`–`2`** | 第 1–3 层 |
| **T（Thinking）** | **`3`–`32`** | 第 4–33 层 |
| **D（Decoder）** | **`33`–`35`** | 第 34–36 层 |

**结论：**  
- **BoolQ 角距离采集** 与 **ARC 角距离采集** 在本次实验中 **选中的层划分相同**：均为 **E=3 层 + T=30 层 + D=3 层**，上表中的 **0-based 索引** 即为两次实验共同采用的划分。  
- 两次 JSON 中的 **`distances` 数组**（36 个相邻层平均角距离）因 **训练样本文本不同** 而略有差异，但 Kneedle 拐点仍给出相同的 \(N_E,N_T,N_D\)。

---

## 4. 评测设置

| 数据集 | 划分 | 说明 |
|--------|------|------|
| **BoolQ** | `validation` | 官方 **`test` 无 gold（label=-1）**，无法用准确率；故用 validation 作带标签评测。 |
| **ARC-Challenge** | `test` | 使用 `answerKey` 与选项文本。 |

固定 **\(k=2\)** 与 **\(k=3\)** 各跑一遍；并增加 **baseline**（标准单次前向，无 T 块循环）。

---

## 5. 已跑子集结果（存档 JSON）

以下为实际写入 `artifacts/` 的数值（子集规模见 `n_examples`）。

### 5.1 BoolQ（`validation`，800 条）

| 模式 | 准确率 |
|------|--------|
| baseline | 0.8575 |
| ETD \(k=2\) | 0.860 |
| ETD \(k=3\) | 0.7675 |

文件：`artifacts/eval_boolq_validation_800.json`

### 5.2 ARC-Challenge（`test`，400 条）

| 模式 | 准确率 |
|------|--------|
| baseline | 0.525 |
| ETD \(k=2\) | 0.3225 |
| ETD \(k=3\) | 0.295 |

文件：`artifacts/eval_arc_test_400.json`

**备注：** 中间块未训练、仅重复计算时，ARC 上 ETD 相对 baseline 明显下降属预期；BoolQ 上 \(k=3\) 亦低于 baseline，可作消融参考。

---

## 6. 复现实验命令摘要

```bash
cd /root/autodl-tmp/loop_layer/ETD
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 角距离 + 分层（各 128 样本）
python3 collect_layers.py --dataset boolq --max-samples 128 --out artifacts/boolq_layers.json
python3 collect_layers.py --dataset arc   --max-samples 128 --out artifacts/arc_layers.json

# 评测（示例：BoolQ 全量 validation 将 --limit 设为 0）
python3 evaluate_etd.py --dataset boolq --layer-json artifacts/boolq_layers.json --k 2 3 --baseline --limit 0
python3 evaluate_etd.py --dataset arc --split test --layer-json artifacts/arc_layers.json --k 2 3 --baseline --limit 0
```

一键脚本：`./run_all.sh`（可通过环境变量 `MODEL`、`MAX_SAMPLES`、`EVAL_LIMIT` 覆盖默认）。

---

## 7. 实现与校验

- **ETD 与 baseline 一致性**：在 \(k=1\) 且 \(N_E+N_T+N_D=L\) 时，`etd_forward_logits` 与模型原生前向 logits **最大绝对误差为 0**（曾用随机分层做过校验）。
- **主要脚本**：`angle_distance.py`、`etd_forward.py`、`collect_layers.py`、`evaluate_etd.py`。

---

*记录生成对应仓库路径：`/root/autodl-tmp/loop_layer/ETD`。若重新采集角距离或更换随机种子，\(N_E,N_T,N_D\) 可能变化，请以最新 `artifacts/*_layers.json` 为准。*
