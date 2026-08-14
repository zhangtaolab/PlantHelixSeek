<div align="center">

<img src="picture/PlantHelixSeek.png" width="200" alt="PlantHelixSeek Logo"/>

# PlantHelixSeek

**A Hybrid Linear Attention-Transformer Mixture-of-Experts (MoE) DNA Language Model for Plants**

*Single-nucleotide resolution · Bidirectional Masked Language Modeling (MLM) · 8,192 bp context · GPU and NPU deployment*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.49.0-yellow)](https://huggingface.co/docs/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NPU](https://img.shields.io/badge/Ascend%20NPU-supported-orange)](docs/installation.md)
[![HF Model](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/zhangtaolab/PlantHelixSeek)
[![MS Model](https://img.shields.io/badge/🧩-ModelScope-blue.svg)](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek)

</div>

> [!NOTE]
> **PlantHelixSeek** is now available! The pretrained checkpoint supports fine-tuning, mask recovery, embedding extraction, and mutagenesis analysis. **Deploy on NVIDIA GPU or Huawei Ascend NPU**.

## Table of Contents

- [Overview](#overview)
- [Model Zoo](#model-zoo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Fine-tune Datasets](#fine-tune-datasets)
- [Scripts](#scripts)
- [Performance Benchmarks](#performance-benchmarks)
- [Platform Support](#platform-support)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## Overview

**PlantHelixSeek** is a plant DNA foundation model pretrained with masked language modeling at single-nucleotide resolution. It is built on the **HelixSeek** architecture — a hybrid stack that interleaves three complementary layer types for efficient long-context bidirectional DNA modeling:

| Component                        | Role                                                                                       | Inspiration                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **Transformer layers**     | Full attention for precise positional reasoning                                            | BERT / RoBERTa                                                                                              |
| **HelixSeek-Delta layers** | Delta Attention (KDA): linear attention with gated delta rule for O(n) long-range modeling | [Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) / [Gated DeltaNet](https://arxiv.org/abs/2412.06464) |
| **HelixSeek-MLA layers**   | Multi-head Latent Attention (MLA) for KV-cache compression                                 | [Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)                                                     |

The FFN uses a sparse **Mixture-of-Experts (MoE)** with 8 experts (top-2 routing + 1 shared expert), achieving high capacity at constant inference cost.

### Model Details

| Property                    | Value                                                           |
| --------------------------- | --------------------------------------------------------------- |
| Total parameters            | 470M                                                            |
| Active parameters per token | ~240M (MoE routing)                                             |
| Hidden size                 | 512                                                             |
| Layers                      | 39 (hybrid: Transformer + KDA + MLA)                            |
| Attention heads             | 16                                                              |
| Experts                     | 8 (top-2 + 1 shared)                                            |
| Max context length          | 8,192 bp                                                        |
| Vocabulary                  | 11 tokens (single-nucleotide: A, T, C, G, N + specials)         |
| Tokenizer                   | SingleBase (1 nucleotide = 1 token)                             |
| Architecture                | HelixSeek (Transformer + KDA + MLA + MoE,`trust_remote_code`) |

---

## Model Zoo

PlantHelixSeek ships with **1 pretrained base model** and **11 task-specific fine-tuned checkpoints** covering plant regulatory element prediction, epigenomics, and translational regulation.

### Base Model

| Model Name               | Params | Pretraining | Context  | Architecture                              | Model Download                                                                                                                   |
| ------------------------ | ------ | ----------- | -------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **PlantHelixSeek** | 470M   | MLM         | 8,192 bp | HelixSeek (Transformer + KDA + MLA + MoE) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek) |

### Fine-tuned Models

All fine-tuned checkpoints are built on the PlantHelixSeek base and trained for **3 epochs** on the corresponding downstream dataset (see [Fine-tune Datasets](#fine-tune-datasets)). Each loads with `trust_remote_code=True`.

**Classification (8)**

| Model Name                               | Task                         | Fine-tune Dataset                                                                                                                                                                                 | Model Download                                                                                                                                                               |
| ---------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PlantHelixSeek-core_promoter`         | Core promoter identification | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters)               | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-core_promoter) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-core_promoter)                 |
| `PlantHelixSeek-lncRNA`                | lncRNA classification        | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-lncRNAs) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-lncRNAs)                             | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-lncRNA) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-lncRNA)                               |
| `PlantHelixSeek-open_chromatin`        | Open-chromatin region        | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-open-chromatin) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-open-chromatin)               | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-open_chromatin) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-open_chromatin)               |
| `PlantHelixSeek-sequence_conservation` | Sequence conservation        | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-sequence-conservation) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-sequence-conservation) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-sequence_conservation) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-sequence_conservation) |
| `PlantHelixSeek-H3K27ac`               | Histone mark H3K27ac         | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-histone-modifications) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-histone-modifications) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-H3K27ac) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-H3K27ac)                             |
| `PlantHelixSeek-H3K27me3`              | Histone mark H3K27me3        | ↑ same dataset (H3K27me3 subset)                                                                                                                                                                 | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-H3K27me3) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-H3K27me3)                           |
| `PlantHelixSeek-H3K4me3`               | Histone mark H3K4me3         | ↑ same dataset (H3K4me3 subset)                                                                                                                                                                  | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-H3K4me3) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-H3K4me3)                             |
| `PlantHelixSeek-translation_onoff`     | Translation on/off (leaf)    | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/cross_species_leaf_on_off_translation) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/cross_species_leaf_on_off_translation)         | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-translation_onoff) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-translation_onoff)         |

**Regression (3)**

| Model Name                              | Task                       | Fine-tune Dataset                                                                                                                                                                             | Model Download                                                                                                                                                             |
| --------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PlantHelixSeek-promoter_strength`    | Promoter strength (leaf)   | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-promoter-strength) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-promoter-strength)     | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-promoter_strength) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-promoter_strength)       |
| `PlantHelixSeek-terminator_strength`  | Terminator strength (leaf) | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-terminator-strength) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-terminator-strength) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-terminator_strength) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-terminator_strength)   |
| `PlantHelixSeek-translation_absolute` | Translation level (leaf)   | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/cross_species_leaf_absolute_translation) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/cross_species_leaf_absolute_translation) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-translation_absolute) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-translation_absolute) |

> 📖 Full model–dataset mapping, splits, and metrics: `[docs/finetune_datasets.md](docs/finetune_datasets.md)`

### Downstream Validation Tasks (Paper)

Task-specific checkpoints from the paper's downstream validation, each with its own training pipeline under `scripts/`:

| Model Name              | Task                                        | Architecture            | Model Download                                                                                                                             | Dataset                                                                                                                                               | Pipeline                                                          |
| ----------------------- | ------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `PlantHelixSeek-Anno` | Gene structure prediction (17 BILOU labels) | Token classification    | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-Anno) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-Anno) | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/PlantHelixSeek-Anno) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/PlantHelixSeek-Anno) | `[scripts/gene_annotation/](scripts/gene_annotation/README.md)` |
| `PlantHelixSeek-CRE`  | Cis-regulatory element prediction           | Sequence classification | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-CRE) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-CRE)   | [🤗 HF](https://huggingface.co/datasets/zhangtaolab/PlantHelixSeek-CRE) · [🧩 MS](https://www.modelscope.cn/datasets/zhangtaolab/PlantHelixSeek-CRE)   | `[scripts/cis_regulatory/](scripts/cis_regulatory/README.md)`   |

> These are independent of the 9 standard fine-tune datasets above — each ships its own data processing and training code.

### Download

```bash
# Base model
huggingface-cli download zhangtaolab/PlantHelixSeek --local-dir ./models/PlantHelixSeek

# Fine-tuned checkpoint (save under models/finetuned/)
huggingface-cli download zhangtaolab/PlantHelixSeek-lncRNA \
    --local-dir ./models/finetuned/PlantHelixSeek-lncRNA
```

**Via ModelScope** (no mirror needed):

```bash
modelscope download --model zhangtaolab/PlantHelixSeek --local_dir ./models/PlantHelixSeek
modelscope download --model zhangtaolab/PlantHelixSeek-lncRNA --local_dir ./models/finetuned/PlantHelixSeek-lncRNA
```

> 💡 HuggingFace China mirror: `export HF_ENDPOINT=https://hf-mirror.com`. Or use ModelScope above for native China access.

> ⚠️ Always pass `trust_remote_code=True` when loading — the HelixSeek architecture is registered via HuggingFace `auto_map`.

---

## Installation

**Requirements**: Python ≥ 3.12

```bash
git clone https://github.com/zhangtaolab/PlantHelixSeek.git
cd PlantHelixSeek
python -m venv .venv && source .venv/bin/activate
```

### NVIDIA GPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.x/13.x
pip install -r requirements.txt                                        # includes flash-linear-attention

# Verify
python scripts/huggingface/verify_install.py --model_path models/PlantHelixSeek
```

###  Ascend NPU

**Prerequisite**: [CANN Toolkit](https://www.hiascend.com/) ≥ 9.0.0 (for Ascend 910B2) — torch-npu requires the CANN runtime environment to be active (e.g. `LD_LIBRARY_PATH` pointing to the CANN libraries). Set up your CANN environment the way you normally do.

```bash
# 1. Install PyTorch + Ascend extension (versions matched to CANN 9.0.0)
pip install torch==2.9.0 torch-npu==2.9.0.post2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Triton → Ascend compiler backend for FLA kernels
#    (3.2.1 is required for CANN 9.0.0; it is not on PyPI — use the osinfra index)
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple pybind11

# 4. Override flash-linear-attention to 0.5.1 (0.5.1 adds the triton-ascend NPU backend;
#    requirements.txt pins 0.4.1). Install fla-core explicitly — the metapackage doesn't pin it.
pip install flash-linear-attention==0.5.1 fla-core==0.5.1

# 5. Verify
python scripts/huggingface/verify_install.py --model_path models/PlantHelixSeek
```

**Tested NPU environment**:

| Component              | Version                  |
| ---------------------- | ------------------------ |
| NPU hardware           | Ascend 910B2 (64 GB HBM) |
| CANN Toolkit           | 9.0.0                    |
| torch-npu              | 2.9.0.post2              |
| triton-ascend          | 3.2.1                    |
| flash-linear-attention | 0.5.1                    |

### CPU Only (Debugging, not recommended)

```bash
pip install torch
pip install -r requirements.txt
```

> ⚠️ CPU mode is for code debugging only, not recommended for actual use. The 470M-parameter model is extremely slow on CPU (tens of seconds per sequence).

📖 **Full installation guide**: [docs/installation.md](docs/installation.md)

---

## Quick Start

### Python API

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Load from local path or HuggingFace Hub
model_path = "models/PlantHelixSeek"  # or "zhangtaolab/PlantHelixSeek"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)

# Auto-detect accelerator: NVIDIA CUDA → Huawei Ascend NPU → CPU
# (FLA Triton kernels require GPU/NPU tensors; CPU is for debugging only)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch, "npu", None) and torch.npu.is_available():
    device = torch.device("npu")
else:
    device = torch.device("cpu")
model = model.to(device)

# Mask recovery: predict masked nucleotides
seq = "ATCGGCTAAGCCT<mask>GTCAGTTACCGATCG"
inputs = tokenizer(seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    logits = model(**inputs).logits

mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0]
predicted = logits[0, mask_pos].argmax(-1)
print(tokenizer.convert_ids_to_tokens(predicted))  # → ['T']
```

### Mask Recovery Inference

```bash
python scripts/huggingface/inference.py \
    --model_path models/PlantHelixSeek \
    --input_file data/test_data/inference_example.jsonl \
    --output_file tmp/mask_recovery.jsonl \
    --top_k 5
```

### Fine-tuning (Sequence Classification)

```bash
python scripts/huggingface/finetune.py \
    --pretrained_model models/PlantHelixSeek \
    --dataset_dir data/test_data/finetune_example_data \
    --output_dir tmp/finetune_output \
    --tokens_length 512 \
    --num_train_epochs 3
```

### Prediction

```bash
python scripts/huggingface/predict.py \
    --model_path tmp/finetune_output \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/predictions.csv
```

### DNALLM Quick Start

The [DNALLM](https://github.com/zhangtaolab/dnallm) package provides higher-level
APIs for common DNA model operations — embeddings, scoring, mutagenesis, and
interpretability — with a single unified interface.

```bash
pip install dnallm
```

```python
from dnallm import DNAInference

inferencer = DNAInference(
    model_path="models/PlantHelixSeek",
    trust_remote_code=True,
    device="auto",
)

# 1. Extract embeddings (mean-pooled sequence representations)
emb = inferencer.get_embeddings(
    inputs=["ATCGGCTAAGCCTGTCAGTT", "GGCCCATTGCATGCAATGCAT"],
    reduce_strategy="mean",
)

# 2. Score sequences (cross-entropy loss — lower = more likely)
scores = inferencer.scoring(
    inputs=["ATCGGCTAAGCCTGTCAGTT", "GGCCCATTGCATGCAATGCAT"],
    score_type="loss",
)
```

```bash
# 3. Saturation mutagenesis scan (identify functional positions)
python scripts/dnallm/mutagenesis.py \
    --model_path tmp/finetune_output \
    --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
    --output_dir tmp/mutagenesis

# 4. Model interpretability (Captum Integrated Gradients)
python scripts/dnallm/interpret.py \
    --model_path tmp/finetune_output \
    --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
    --method integrated_gradients \
    --output_dir tmp/interpret
```

> 📖 See `[scripts/dnallm/README.md](scripts/dnallm/README.md)` for the full list of 7 DNALLM scripts.

---

## Fine-tune Datasets

Downstream fine-tuning datasets are released in the **[PlantHelixSeek](https://huggingface.co/collections/zhangtaolab/planthelixseek)** 🤗 collection — 9 multi-species plant genomic datasets, each shipped as ready-to-use `train/dev/test.csv` splits.

| Dataset                                   | Task                             | Fine-tuned Model(s)                      |                                             HF                                             |                                          ModelScope                                          |
| ----------------------------------------- | -------------------------------- | ---------------------------------------- | :----------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| plant-multi-species-core-promoters        | Core promoter (binary)           | `PlantHelixSeek-core_promoter`         |    [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters)    |    [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters)    |
| plant-multi-species-lncRNAs               | lncRNA (binary)                  | `PlantHelixSeek-lncRNA`                |        [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-lncRNAs)        |        [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-lncRNAs)        |
| plant-multi-species-open-chromatin        | Open chromatin (binary)          | `PlantHelixSeek-open_chromatin`        |    [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-open-chromatin)    |    [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-open-chromatin)    |
| plant-multi-species-sequence-conservation | Conservation (binary)            | `PlantHelixSeek-sequence_conservation` | [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-sequence-conservation) | [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-sequence-conservation) |
| plant-multi-species-histone-modifications | Histone marks ×3 (binary)       | `H3K27ac` / `H3K27me3` / `H3K4me3` | [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-histone-modifications) | [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-histone-modifications) |
| plant-multi-species-promoter-strength     | Promoter strength (regression)   | `PlantHelixSeek-promoter_strength`     |   [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-promoter-strength)   |   [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-promoter-strength)   |
| plant-multi-species-terminator-strength   | Terminator strength (regression) | `PlantHelixSeek-terminator_strength`   |  [🤗](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-terminator-strength)  |  [🧩](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-terminator-strength)  |
| cross_species_leaf_on_off_translation     | Translation on/off (binary)      | `PlantHelixSeek-translation_onoff`     |   [🤗](https://huggingface.co/datasets/zhangtaolab/cross_species_leaf_on_off_translation)   |   [🧩](https://www.modelscope.cn/datasets/zhangtaolab/cross_species_leaf_on_off_translation)   |
| cross_species_leaf_absolute_translation   | Translation level (regression)   | `PlantHelixSeek-translation_absolute`  |  [🤗](https://huggingface.co/datasets/zhangtaolab/cross_species_leaf_absolute_translation)  |  [🧩](https://www.modelscope.cn/datasets/zhangtaolab/cross_species_leaf_absolute_translation)  |

### Download

Default save path: `data/Bio_finetune_data/`. Each dataset lands in its own subdirectory.

```bash
source .venv/bin/activate   # requires huggingface_hub (already in requirements.txt)

# Single dataset
huggingface-cli download zhangtaolab/plant-multi-species-lncRNAs \
    --repo-type dataset \
    --local-dir data/Bio_finetune_data/plant-multi-species-lncRNAs

# All 9 datasets in parallel
for ds in plant-multi-species-core-promoters \
          plant-multi-species-lncRNAs \
          plant-multi-species-open-chromatin \
          plant-multi-species-sequence-conservation \
          plant-multi-species-histone-modifications \
          plant-multi-species-promoter-strength \
          plant-multi-species-terminator-strength \
          cross_species_leaf_on_off_translation \
          cross_species_leaf_absolute_translation; do
  huggingface-cli download "zhangtaolab/$ds" --repo-type dataset \
      --local-dir "data/Bio_finetune_data/$ds" &
done; wait
```

**Via ModelScope** (fast in China, no mirror needed):

```bash
pip install modelscope

# Single dataset
modelscope download --dataset zhangtaolab/plant-multi-species-lncRNAs \
    --local_dir data/Bio_finetune_data/plant-multi-species-lncRNAs

# All 9 datasets in parallel
for ds in plant-multi-species-core-promoters \
          plant-multi-species-lncRNAs \
          plant-multi-species-open-chromatin \
          plant-multi-species-sequence-conservation \
          plant-multi-species-histone-modifications \
          plant-multi-species-promoter-strength \
          plant-multi-species-terminator-strength \
          cross_species_leaf_on_off_translation \
          cross_species_leaf_absolute_translation; do
  modelscope download --dataset "zhangtaolab/$ds" \
      --local_dir "data/Bio_finetune_data/$ds" &
done; wait
```

The **open-chromatin** dataset ships in two scales — `PlantHelixSeek-open_chromatin` is fine-tuned on the **full** version:

```bash
huggingface-cli download zhangtaolab/plant-multi-species-open-chromatin \
    --repo-type dataset \
    --include "full_datasets/*" \
    --local-dir data/Bio_finetune_data/plant-multi-species-open-chromatin
# → swap the --include for "mini_datasets/*" to get the lightweight version instead
```

> 💡 **China mirror**: `export HF_ENDPOINT=https://hf-mirror.com` for faster downloads.
>
> 📌 The bundled `data/test_data/finetune_example_data/` is only a smoke-test fixture; use the datasets above for real experiments.
>
> 📖 Detailed dataset descriptions, splits, and the full model–dataset mapping: `[docs/finetune_datasets.md](docs/finetune_datasets.md)`

---

## Scripts

### HuggingFace Transformers (`scripts/huggingface/`)

Unified backend supporting both **NVIDIA GPU** and **Huawei Ascend NPU** via `--device_target {auto,cuda,npu,cpu}`.

| Script                                                         | Task                                |
| -------------------------------------------------------------- | ----------------------------------- |
| `[inference.py](scripts/huggingface/inference.py)`           | MLM mask recovery                   |
| `[finetune.py](scripts/huggingface/finetune.py)`             | Sequence classification fine-tuning |
| `[predict.py](scripts/huggingface/predict.py)`               | Fine-tuned model prediction         |
| `[verify_install.py](scripts/huggingface/verify_install.py)` | Environment verification            |

### DNALLM Integration (`scripts/dnallm/`)

High-level wrappers using the [DNALLM](https://github.com/zhangtaolab/dnallm) package:

| Script             | Task                                           | Status |
| ------------------ | ---------------------------------------------- | ------ |
| `finetune.py`    | Classification fine-tuning                     | ✅     |
| `predict.py`     | Prediction                                     | ✅     |
| `embeddings.py`  | Embedding extraction                           | ✅     |
| `scoring.py`     | Sequence scoring (loss / probability / logits) | ✅     |
| `mutagenesis.py` | Saturation mutagenesis scan                    | ✅     |
| `interpret.py`   | Model interpretability (Captum)                | ✅     |

> ⚠️ MLM mask recovery is not supported by DNALLM — use `scripts/huggingface/inference.py` instead. See `[scripts/dnallm/README.md](scripts/dnallm/README.md)` for details.

### Downstream Task Pipelines (`scripts/gene_annotation/`, `scripts/cis_regulatory/`)

Task-specific training pipelines for the paper's downstream validation models:

| Directory                                                         | Task                                                              | Model                   |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------- |
| `[scripts/gene_annotation/](scripts/gene_annotation/README.md)` | Gene structure prediction (17 BILOU labels, token classification) | `PlantHelixSeek-Anno` |
| `[scripts/cis_regulatory/](scripts/cis_regulatory/README.md)`   | Cis-regulatory element prediction                                 | `PlantHelixSeek-CRE`  |

> These pipelines are independent of the unified `scripts/huggingface/finetune.py` flow — each has bespoke data processing, loss, and metrics.

---

## Performance Benchmarks

### Inference

Throughput (tokens/s) and Peak VRAM/HBM (GiB), forward only:

**NVIDIA GeForce RTX 5090 (32 GB)**:

| Seq  | Batch | Tokens/s | VRAM (GiB) |
| ---- | ----- | -------- | ---------- |
| 512  | 1     | 14,248   | 0.9        |
| 512  | 2     | 27,713   | 0.9        |
| 512  | 4     | 63,054   | 0.9        |
| 1024 | 1     | 28,189   | 0.9        |
| 1024 | 2     | 54,198   | 0.9        |
| 1024 | 4     | 81,996   | 1.0        |
| 2048 | 1     | 48,836   | 0.9        |
| 2048 | 2     | 73,689   | 1.0        |
| 2048 | 4     | 97,748   | 1.1        |
| 4096 | 1     | 61,694   | 1.0        |
| 4096 | 2     | 77,915   | 1.1        |
| 4096 | 4     | 89,829   | 1.3        |
| 8192 | 1     | 54,566   | 1.2        |
| 8192 | 2     | 60,538   | 1.5        |
| 8192 | 4     | 63,369   | 2.1        |

**Ascend 910B2 (64 GB HBM)**:

| Seq  | Batch | Tokens/s | HBM (GiB) |
| ---- | ----- | -------- | --------- |
| 512  | 1     | 3,305    | 1.0       |
| 512  | 2     | 6,619    | 1.0       |
| 512  | 4     | 12,873   | 1.0       |
| 1024 | 1     | 6,762    | 1.0       |
| 1024 | 2     | 13,065   | 1.0       |
| 1024 | 4     | 23,466   | 1.1       |
| 2048 | 1     | 12,859   | 1.0       |
| 2048 | 2     | 23,384   | 1.1       |
| 2048 | 4     | 32,084   | 1.2       |
| 4096 | 1     | 18,160   | 1.2       |
| 4096 | 2     | 20,989   | 1.3       |
| 4096 | 4     | 27,614   | 1.6       |
| 8192 | 1     | 18,081   | 1.4       |
| 8192 | 2     | 23,165   | 1.9       |
| 8192 | 4     | 26,424   | 2.7       |

> **GPU**: bf16 + FLA `chunk_kda` kernel. **NPU**: bf16 + SDPA (Transformer/MLA layers) + FLA via triton-ascend `fused_recurrent_kda` (KDA layers).
>
> On NPU the standard attention layers (Transformer/MLA) run via PyTorch SDPA, auto-selected by `from_pretrained`. The KDA (linear-attention) layers use `fused_recurrent_kda` (sequential scan) instead of the parallel `chunk_kda` kernel; `chunk_kda` is expected to be supported in an upcoming triton-ascend release, which will further improve NPU throughput. Additionally, the Ascend 910B2 is positioned in a different performance tier from the RTX 5090 used for GPU benchmarks. Despite these factors, NPU inference delivers practical throughput suitable for production deployment — for example, the full 7,040-sample benchmark dataset completes in ~10 minutes at 1,000 bp sequences.

📖 **Full benchmarks**: [docs/benchmarks.md](docs/benchmarks.md)

---

## Platform Support

| Platform                      | Training | Inference | Backend                                                                                    |
| ----------------------------- | -------- | --------- | ------------------------------------------------------------------------------------------ |
| NVIDIA GPU (CUDA 12.x / 13.x) | ✅       | ✅        | FLA via Triton (`chunk_kda` + `fused_recurrent_kda`)                                   |
| Huawei Ascend NPU             | ✅†     | ✅        | FLA via [triton-ascend](https://pypi.org/project/triton-ascend/) (`fused_recurrent_kda`) |
| CPU                           | —       | ⚠️ Slow | Pure PyTorch fallback                                                                      |

† See the NPU training note below.

**Train on GPU, deploy on GPU or NPU** — same model weights, same FLA kernels, zero code changes. The model auto-detects the device at runtime and selects the optimal KDA computation mode (`chunk` on GPU, `fused_recurrent` on NPU). See [Installation](#installation) for setup instructions.

> **NPU training note**: Fine-tuning runs on Ascend 910B2. For best throughput on heavy training workloads, NVIDIA GPU with `chunk_kda` is recommended.

> **How it works**: On GPU, the KimiDeltaAttention layer uses FLA's `chunk_kda` kernel (fastest). On NPU, `chunk_kda` is not yet supported by triton-ascend (planned for v3.3.0), so the model automatically falls back to `fused_recurrent_kda` — another FLA native kernel that compiles and runs on Ascend. Both produce identical numerical results.
>
> **First-run note**: The first NPU prediction call compiles Triton kernels to Ascend binaries (~~2–3 min). 

---

## Repository Structure

```
PlantHelixSeek/
├── models/
│   ├── PlantHelixSeek/        # Base model weights & code (download separately)
│   ├── PlantHelixSeek-Anno/   # Gene structure prediction checkpoint (paper)
│   ├── PlantHelixSeek-CRE/    # Cis-regulatory element checkpoint (paper)
│   └── finetuned/             # 11 standard fine-tuned checkpoints
├── tokenizer/SingleBase/      # Single-nucleotide DNA tokenizer
├── data/
│   ├── Bio_finetune_data/     # 9 standard fine-tune datasets
│   ├── gene_annotation/       # Gene structure prediction data (paper)
│   ├── cis_regulatory/        # CRE prediction data (paper)
│   └── test_data/             # Example datasets for smoke tests
│       ├── finetune_example_data/ (train/dev/test.csv)
│       ├── inference_example.jsonl
│       └── predict_example.csv
├── scripts/
│   ├── huggingface/           # Main scripts (GPU + NPU unified)
│   ├── dnallm/                # DNALLM package wrappers
│   ├── gene_annotation/       # Gene structure prediction pipeline (paper)
│   └── cis_regulatory/        # CRE prediction pipeline (paper)
├── tests/                     # End-to-end smoke tests
├── docs/                      # Installation, benchmarks & finetune datasets
├── picture/                   # Logo & visual assets
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Citation

If you use PlantHelixSeek in your research, please cite:

```bibtex
@article{PlantHelixSeek2026,
  title   = {PlantHelixSeek: A Hybrid Linear Attention-Transformer Mixture-of-Experts DNA Language Model for Plants},
  author  = {YueChao WU},
  year    = {2026},
  url     = {https://huggingface.co/zhangtaolab/PlantHelixSeek}
}
```

---

## Acknowledgments

- **[Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)** — Delta Attention (KDA) and Multi-head Latent Attention (MLA) architecture
- **[Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)** — Efficient chunkwise kernels for KDA layers
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — Model loading, training, and inference framework
- **[DNALLM](https://github.com/zhangtaolab/dnallm)** — High-level DNA language model toolkit

---

## License

- **Code**: [MIT License](LICENSE)
- **Model weights**: CC-BY-NC 4.0 (research use; contact authors for commercial licensing)

---

## Contact

- **Email**: [wuyuechao@zhangtaolab.org](mailto:wuyuechao@zhangtaolab.org)
