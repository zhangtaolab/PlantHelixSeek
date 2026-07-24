# PlantHelixSeek-CRE — 顺式调控元件预测

`PlantHelixSeek-CRE` 是 PlantHelixSeek 基座模型在植物顺式调控元件（Cis-Regulatory Element, CRE）二分类数据集上微调的 checkpoint。它通过全基因组滑动窗口扫描，预测每个 50 bp 区间的 CRE / 开放染色质概率，并输出标准 BigWig 轨道文件。

> 📌 English: [README.md](README.md)。

## 流程

```
下载                          预测                           Peak Calling
huggingface-cli download      →  predict_cre.py             →  call_peaks_from_bigwig.py
     (模型权重)                    (.bw BigWig)                  (.bed / .narrowPeak / .broadPeak)
```

## 模型

| Checkpoint | 基座 | 任务 | 下载 |
|------------|------|------|------|
| `PlantHelixSeek-CRE` | PlantHelixSeek 470M | 二分类 | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-CRE) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-CRE) |

权重位于 `models/PlantHelixSeek-CRE/`，加载时需 `trust_remote_code=True`。

> 💡 若 `models/PlantHelixSeek-CRE/` 已下载，可跳过[下载](#1-下载)，直接进入[预测](#2-预测-predict_crepy)。

---

## 1. 下载

从 HuggingFace 下载微调 checkpoint：

```bash
huggingface-cli download zhangtaolab/PlantHelixSeek-CRE \
    --local-dir models/PlantHelixSeek-CRE
```

**Via ModelScope** (China-friendly):

```bash
modelscope download --model zhangtaolab/PlantHelixSeek-CRE \
    --local_dir models/PlantHelixSeek-CRE

> 💡 国内镜像：`export HF_ENDPOINT=https://hf-mirror.com` 加速下载。

---

## 2. 预测 (`predict_cre.py`)

### 算法说明

| 项目 | 值 |
|------|-----|
| 窗口大小 | 500 bp |
| 步长 | 50 bp（可通过 `--stride` 调整）|
| 分箱大小 | 50 bp |
| 聚合方式 | 覆盖同一 bin 的所有窗口预测概率的算术平均 |
| TTA | 默认关闭；`--tta` 启用反向互补增强（约 2× 减速，精度略升）|

每个 50 bp bin 的得分为所有覆盖该区间的滑动窗口 class-1 概率的算术平均。由于窗口高度重叠（500 bp 窗口 / 50 bp 步长），每个 bin 最多被 10 个窗口覆盖。

### 用法

```bash
# 使用默认 PlantHelixSeek-CRE 预测
python scripts/cis_regulatory/predict_cre.py

# 指定基因组与输出目录
python scripts/cis_regulatory/predict_cre.py \
    --model_path models/PlantHelixSeek-CRE \
    --genome data/cis_regulatory/test_sequence.fa \
    --output_dir tmp/cre_predict
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model_path` | `models/PlantHelixSeek-CRE` | 微调 checkpoint 路径 |
| `--genome` | `data/cis_regulatory/test_sequence.fa` | 输入 FASTA |
| `--output_dir` | `tmp/cre_predict` | 输出目录 |
| `--batch_size` | 256 | 推理 batch size |
| `--stride` | 50 | 滑动窗口步长 bp |
| `--tta` | 关闭 | 启用反向互补 TTA |
| `--device` | 自动 | 设备，如 `cuda:0`、`cpu` |
| `--num_workers` | 0 | DataLoader 工作线程 |
| `--min_chrom_size` | 0 | 跳过小于此长度的染色体（bp），0 = 不过滤 |
| `--max_chrom_size` | 0 | 跳过大于此长度的染色体（bp），0 = 不过滤 |

### 输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `PlantHelixSeek-CRE_{species}_OCR.bw` | BigWig | 标准基因组轨道，可在 IGV / UCSC Browser 加载 |
| `PlantHelixSeek-CRE_{species}_OCR.npy` | NumPy dict | `{chrom: float32 array}`，每个元素为一个 50 bp bin 的概率 |

断点续跑：每条染色体预测完毕后立即缓存到 `chunks/<chrom>.npy`，中断后重跑会自动跳过已完成的染色体。

---

## 3. Peak Calling (`call_peaks_from_bigwig.py`)

从预测生成的 BigWig 信号文件 call peaks，输出 BED / narrowPeak / broadPeak 格式。

### 默认用法（推荐）

全基因组（CIB）预测是推荐用法：

```bash
python scripts/cis_regulatory/call_peaks_from_bigwig.py \
    --bigwig tmp/cre_predict/PlantHelixSeek-CRE_test_sequence_OCR.bw \
    --genome data/cis_regulatory/test_sequence.fa \
    --output_dir tmp/cre_predict \
    --refine_summit \
    --prefix PlantHelixSeek-CRE_test_sequence_OCR
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--bigwig` | —（必选）| 输入 BigWig 信号 |
| `--genome` | —（必选）| 基因组 FASTA（取染色体大小）|
| `--output_dir` | —（必选）| 输出目录 |
| `--threshold_method` | `mean_std` | 阈值方法: `mean_std` / `median_mad` / `percentile` / `fixed` |
| `--threshold_params` | `1.5` | mean_std/median_mad 为 k 值，percentile 为百分位，fixed 为阈值 |
| `--min_length` | 50 | Peak 最小长度 bp |
| `--max_length` | 5000 | Peak 最大长度 bp |
| `--min_score` | 0.6 | Peak 最低得分，低于此值不输出 |
| `--merge_gap` | 50 | 合并相邻高分区域的最大间距 bp |
| `--refine_summit` | 关闭 | 精修 peak summit 位置 |
| `--prefix` | （自动）| 输出文件前缀（默认从 bigwig 文件名推导）|

### 输出

| 文件 | 格式 |
|------|------|
| `{prefix}_peaks.bed` | BED6 |
| `{prefix}_peaks.narrowPeak` | BED6+4（含 summit 偏移）|
| `{prefix}_peaks.broadPeak` | BED6+3 |

---

## 环境要求

- Python ≥ 3.12
- CUDA GPU — PlantHelixSeek 依赖 Triton / `flash-linear-attention` 内核，不支持纯 CPU 推理。
- 项目虚拟环境: `source .venv/bin/activate`
- 依赖: `pyBigWig`、`biopython`、`tqdm`、`torch`、`transformers`（均在 `requirements.txt` 中）

---

## 文件

| 文件 | 用途 |
|------|------|
| `predict_cre.py` | 全基因组 CRE 预测 → BigWig |
| `call_peaks_from_bigwig.py` | BigWig → BED / narrowPeak / broadPeak |
` 内核，不支持纯 CPU 推理。
- **仅 NPU**：需 CANN Toolkit（≥ 9.0.0）环境与 `torch_npu`。NPU 下须显式传入 `--device npu:0`。
- 项目虚拟环境: `source .venv/bin/activate`
- 依赖: `pyBigWig`、`biopython`、`tqdm`、`torch`、`transformers`（均在 `requirements.txt` 中）

---

## 文件

| 文件 | 用途 |
|------|------|
| `predict_cre.py` | 全基因组 CRE 预测 → BigWig |
| `call_peaks_from_bigwig.py` | BigWig → BED / narrowPeak / broadPeak |
