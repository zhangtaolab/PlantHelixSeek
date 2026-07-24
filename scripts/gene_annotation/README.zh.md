# PlantHelixSeek-Anno — 基因结构预测

PlantHelixSeek-Anno 是从 PlantHelixSeek 基座微调的 token 分类 checkpoint。它对 FASTA 序列中的每个碱基标注 **17 类 BILOU 标签**（CDS / intron / 5′UTR / 3′UTR / intergenic），再将逐碱基预测解码为标准 GFF3 基因模型。

> 📌 English: [README.md](README.md)。

## 流程

```
下载                          预测                           GFF3 解码
huggingface-cli download      →  predict_genome.py          →  pred_to_gff3.py
     (模型权重)                    (.npz 逐碱基标签)             (.gff3 基因模型)
```

## 模型

| Checkpoint | 基座 | 任务 | 标签数 | 下载 |
|------------|------|------|:------:|------|
| `PlantHelixSeek-Anno` | PlantHelixSeek 470M | Token 分类 | 17 (BILOU) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-Anno) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-Anno) |

权重位于 `models/PlantHelixSeek-Anno/`，加载时需 `trust_remote_code=True`。

> 💡 若 `models/PlantHelixSeek-Anno/` 已下载，可跳过[下载](#1-下载)，直接进入[预测](#2-预测-predict_genomepy)。

### 标签体系（17 类 BILOU）

| 分组 | 标签 |
|------|------|
| intergenic | `O` |
| CDS | `B-CDS` `I-CDS` `L-CDS` `U-CDS` |
| Intron | `B-INTRON` `I-INTRON` `L-INTRON` `U-INTRON` |
| 5′ UTR | `B-UTR5` `I-UTR5` `L-UTR5` `U-UTR5` |
| 3′ UTR | `B-UTR3` `I-UTR3` `L-UTR3` `U-UTR3` |

---

## 1. 下载

```bash
huggingface-cli download zhangtaolab/PlantHelixSeek-Anno \
    --local-dir models/PlantHelixSeek-Anno
```

**Via ModelScope** (China-friendly):

```bash
modelscope download --model zhangtaolab/PlantHelixSeek-Anno \
    --local_dir models/PlantHelixSeek-Anno

> 💡 国内镜像：`export HF_ENDPOINT=https://hf-mirror.com` 加速下载。

---

## 2. 预测 (`predict_genome.py`)

滑窗基因组扫描。每个 8 192 bp 窗口在正反链各推理一次，有效中间区域直接拼接（不做概率平均）。注意力层 patch 为 SDPA 以避免 O(N²) 显存。

### 用法

```bash
python scripts/gene_annotation/predict_genome.py \
    --genome-fasta data/gene_annotation/test_sequence.fa \
    --model-path models/PlantHelixSeek-Anno \
    --output-dir tmp/anno_predict \
    --batch-size 4 --device cuda:0 \
    --use-crf
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--genome-fasta` | —（必选）| 输入 FASTA |
| `--model-path` | —（必选）| 模型目录（含 `config.json` + `model.safetensors`）|
| `--output-dir` | —（必选）| `.npz` 输出目录 |
| `--batch-size` | 8 | 推理 batch size（RTX 5090 在 seq 8 192 下推荐 4）|
| `--device` | 自动 | `cuda:0`、`cpu` 等 |
| `--window-size` | 8192 | 窗口大小（≤ 模型 max_position_embeddings）|
| `--stride` | 4096 | 窗口步长（= window/2，50% 重叠）|
| `--use-crf` | 关闭 | 启用 CRF Viterbi 解码 |
| `--crf-path` | None | 训练好的 CRF 权重路径（省略则用默认约束）|
| `--genic-bias` | 0.0 | intergenic 偏置（正值 → 更少基因）|
| `--chromosome` | None | 仅处理指定染色体 |

### 输出

每条染色体一个 `{chrom}.npz`，包含：

| 键 | 形状 | 说明 |
|----|------|------|
| `fwd_predictions` / `rev_predictions` | `[seq_len]` | 17 类 BILOU 标签（启用 `--use-crf` 时为 CRF 解码）|
| `fwd_probabilities` / `rev_probabilities` | `[seq_len, 17]` | 每碱基 17 类 softmax 概率 |
| `fwd_feature_probs` / `rev_feature_probs` | `[seq_len, 5]` | 聚合后的 5 类特征概率 |

---

## 3. GFF3 解码 (`pred_to_gff3.py`)

将 NPZ 预测解码为标准 GFF3 基因模型，含 ID/Parent 层级关系。

### 解码方法

| 方法 | 说明 |
|------|------|
| `viterbi` | 5 类特征概率 Viterbi 解码 + CRF 标签定界（推荐）|
| `viterbi+orf` | 同上 + ORF 校正 |
| `heuristic` | 概率阈值 + 启发式聚类（向后兼容）|

### 用法

```bash
python scripts/gene_annotation/pred_to_gff3.py \
    --npz-dir tmp/anno_predict \
    --genome-fasta data/gene_annotation/test_sequence.fa \
    --decoding-method viterbi \
    --workers 4
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--npz-dir` | —（必选）| `.npz` 文件目录 |
| `--genome-fasta` | —（必选）| 基因组 FASTA（用于序列 / 染色体大小）|
| `--decoding-method` | `viterbi+orf` | `viterbi` / `viterbi+orf` / `heuristic` |
| `--workers` | 1 | 并行工作线程 |
| `--domain` | `plant` | 转移矩阵类型（`plant` / `animal`）|
| `--source` | `PlantHelixSeek-anno` | GFF3 source 列 |
| `--species-prefix` | `""` | 物种简写（如 `LP`）；传入后生成合并的 `{species}.gff3` 并改写基因 ID |
| `--predictor-tag` | `PHS` | 基因 ID 后缀标签 |
| `--intergenic-boost` | 10.0 | intergenic 概率增益（正值 → 更保守）|
| `--viterbi-alpha` | 0.0 | 转移矩阵平滑度（0~1）|
| `--reference-gff3` | None | 参考 GFF3（用于评估指标）|
| `--genomic-offset` | 0 | 坐标偏移（子区域评估时设置）|
| `--output-dir` | `--npz-dir` | 输出目录（默认同 NPZ 目录）|

### 输出

| 文件 | 说明 |
|------|------|
| `{chrom}.gff3` | per-chrom GFF3（gene / mRNA / CDS / UTR，含 ID/Parent）|
| `{species}.gff3` | 全基因组合并文件（仅当传入 `--species-prefix` 时生成）；基因 ID 格式 `{PREFIX}{CHR:02d}G{GENE:06d}_{TAG}` |

---

## 环境要求

- Python ≥ 3.12
- CUDA GPU — PlantHelixSeek 依赖 Triton / `flash-linear-attention` 内核。
- 项目虚拟环境: `source .venv/bin/activate`
- 依赖: `torch`、`transformers`、`biopython`、`numpy`、`numba`（Viterbi JIT 加速）、`tqdm`（均在 `requirements.txt` 中）

---

## 文件

| 文件 | 用途 |
|------|------|
| `predict_genome.py` | 滑窗预测 → `.npz` |
| `pred_to_gff3.py` | NPZ → GFF3 基因模型 |
| `viterbi_utils.py` | Viterbi 解码器 + 转移概率矩阵 |
| `train_token_cls.py` | 线性链 CRF + BILOU 约束（用于 `--use-crf`）|
| `transition_probs.npz` | 预计算的 plant 转移概率 |
��）；基因 ID 格式 `{PREFIX}{CHR:02d}G{GENE:06d}_{TAG}` |

---

## 环境要求

- Python ≥ 3.12
- 加速卡：NVIDIA CUDA GPU — PlantHelixSeek 依赖 Triton / `flash-linear-attention` 内核，不支持纯 CPU 推理。
- 项目虚拟环境: `source .venv/bin/activate`
- 依赖: `torch`、`transformers`、`biopython`、`numpy`、`numba`（Viterbi JIT 加速）、`tqdm`（均在 `requirements.txt` 中）

---

## 文件

| 文件 | 用途 |
|------|------|
| `predict_genome_multigpu.py` | 滑窗预测 → `.npz` |
| `pred_to_gff3.py` | NPZ → GFF3 基因模型 |
| `viterbi_utils.py` | Viterbi 解码器 + 转移概率矩阵 |
| `train_token_cls.py` | 线性链 CRF + BILOU 约束（用于 `--use-crf`）|
| `transition_probs.npz` | 预计算的 plant 转移概率 |
