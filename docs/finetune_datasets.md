# Fine-tune Datasets & Model–Dataset Mapping

This document details the **9 downstream datasets** and **11 fine-tuned checkpoints** in the [PlantHelixSeek collection](https://huggingface.co/collections/zhangtaolab/planthelixseek). All datasets are multi-species plant genomic tasks, shipped as ready-to-use `train/dev/test.csv` splits. All checkpoints are fine-tuned from the PlantHelixSeek base (470M, MLM) for **3 epochs**.

> Task type (`binary` / `regression`) is read from each checkpoint's `config.json → problem_type`.

---

## Overview — Model ↔ Dataset Mapping

| Fine-tuned Model | Task Type | Dataset | Local Model Path | Local Data Path |
|------------------|:---------:|---------|-------------------|-----------------|
| `PlantHelixSeek-core_promoter` | binary | plant-multi-species-core-promoters | `models/finetuned/PlantHelixSeek-core_promoter` | `data/Bio_finetune_data/plant-multi-species-core-promoters` |
| `PlantHelixSeek-lncRNA` | binary | plant-multi-species-lncRNAs | `models/finetuned/PlantHelixSeek-lncRNA` | `data/Bio_finetune_data/plant-multi-species-lncRNAs` |
| `PlantHelixSeek-open_chromatin` | binary | plant-multi-species-open-chromatin (**full**) | `models/finetuned/PlantHelixSeek-open_chromatin` | `data/Bio_finetune_data/plant-multi-species-open-chromatin` |
| `PlantHelixSeek-sequence_conservation` | binary | plant-multi-species-sequence-conservation | `models/finetuned/PlantHelixSeek-sequence_conservation` | `data/Bio_finetune_data/plant-multi-species-sequence-conservation` |
| `PlantHelixSeek-H3K27ac` | binary | plant-multi-species-histone-modifications (H3K27ac) | `models/finetuned/PlantHelixSeek-H3K27ac` | `data/Bio_finetune_data/plant-multi-species-histone-modifications` |
| `PlantHelixSeek-H3K27me3` | binary | plant-multi-species-histone-modifications (H3K27me3) | `models/finetuned/PlantHelixSeek-H3K27me3` | ↑ same |
| `PlantHelixSeek-H3K4me3` | binary | plant-multi-species-histone-modifications (H3K4me3) | `models/finetuned/PlantHelixSeek-H3K4me3` | ↑ same |
| `PlantHelixSeek-translation_onoff` | binary | cross_species_leaf_on_off_translation | `models/finetuned/PlantHelixSeek-translation_onoff` | `data/Bio_finetune_data/cross_species_leaf_on_off_translation` |
| `PlantHelixSeek-promoter_strength` | regression | plant-multi-species-promoter-strength | `models/finetuned/PlantHelixSeek-promoter_strength` | `data/Bio_finetune_data/plant-multi-species-promoter-strength` |
| `PlantHelixSeek-terminator_strength` | regression | plant-multi-species-terminator-strength | `models/finetuned/PlantHelixSeek-terminator_strength` | `data/Bio_finetune_data/plant-multi-species-terminator-strength` |
| `PlantHelixSeek-translation_absolute` | regression | cross_species_leaf_absolute_translation | `models/finetuned/PlantHelixSeek-translation_absolute` | `data/Bio_finetune_data/cross_species_leaf_absolute_translation` |

> **Note**: The histone-modifications dataset bundles three marks (H3K27ac / H3K27me3 / H3K4me3); one dataset → three binary checkpoints.

---

## Datasets by Category

### Regulatory Elements

#### plant-multi-species-core-promoters
- **Task**: Binary classification — identify active core promoters in plants.
- **Model**: `PlantHelixSeek-core_promoter`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters)

#### plant-multi-species-promoter-strength
- **Task**: Regression — quantify promoter transcriptional strength.
- **Model**: `PlantHelixSeek-promoter_strength`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-promoter-strength) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-promoter-strength)

#### plant-multi-species-terminator-strength
- **Task**: Regression — quantify terminator strength.
- **Model**: `PlantHelixSeek-terminator_strength`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-terminator-strength) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-terminator-strength)

### Epigenomics

#### plant-multi-species-histone-modifications
- **Task**: Binary classification (×3) — predict genomic regions carrying each histone mark.
- **Subsets**: H3K27ac (active enhancer), H3K27me3 (repressive), H3K4me3 (active promoter)
- **Models**: `PlantHelixSeek-H3K27ac` / `PlantHelixSeek-H3K27me3` / `PlantHelixSeek-H3K4me3`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-histone-modifications) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-histone-modifications)

#### plant-multi-species-open-chromatin
- **Task**: Binary classification — identify open-chromatin regions (ATAC/DNase peaks).
- **Scales**: `full` (used for `PlantHelixSeek-open_chromatin`) and `mini` (lightweight).
- **Model**: `PlantHelixSeek-open_chromatin`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-open-chromatin) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-open-chromatin)

### Non-coding & Conservation

#### plant-multi-species-lncRNAs
- **Task**: Binary classification — distinguish long non-coding RNAs from coding transcripts.
- **Model**: `PlantHelixSeek-lncRNA`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-lncRNAs) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-lncRNAs)

#### plant-multi-species-sequence-conservation
- **Task**: Binary classification — predict conserved sequence regions across plant species.
- **Model**: `PlantHelixSeek-sequence_conservation`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-sequence-conservation) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/plant-multi-species-sequence-conservation)

### Translational Regulation

#### cross_species_leaf_absolute_translation
- **Task**: Regression — predict absolute translational level (leaf, cross-species).
- **Model**: `PlantHelixSeek-translation_absolute`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/cross_species_leaf_absolute_translation) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/cross_species_leaf_absolute_translation)

#### cross_species_leaf_on_off_translation
- **Task**: Binary classification — predict translation on/off state (leaf, cross-species).
- **Model**: `PlantHelixSeek-translation_onoff`
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/cross_species_leaf_on_off_translation) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/cross_species_leaf_on_off_translation)

---

## Training Summary

All checkpoints were trained from the PlantHelixSeek base with `trust_remote_code=True`, 3 epochs, on a single NVIDIA GPU. Training metrics from each checkpoint's `final_metrics.json`:

| Model | Task | Train Loss | Train Runtime |
|-------|:----:|:----------:|:-------------:|
| PlantHelixSeek-H3K27ac | binary | 0.322 | ~3.7 h |
| PlantHelixSeek-H3K27me3 | binary | 0.389 | ~3.5 h |
| PlantHelixSeek-H3K4me3 | binary | 0.207 | ~3.6 h |
| PlantHelixSeek-core_promoter | binary | 0.314 | ~28 min |
| PlantHelixSeek-lncRNA | binary | 0.293 | ~1.8 h |
| PlantHelixSeek-open_chromatin | binary | 0.544 | ~5.4 h |
| PlantHelixSeek-sequence_conservation | binary | 0.203 | ~1.9 h |
| PlantHelixSeek-translation_onoff | binary | 0.443 | ~19 min |
| PlantHelixSeek-promoter_strength | regression | 0.800 | ~24 min |
| PlantHelixSeek-terminator_strength | regression | 0.235 | ~18 min |
| PlantHelixSeek-translation_absolute | regression | 0.701 | ~19 min |

> Train loss is not directly comparable across task types (binary cross-entropy vs. MSE). Use the `predict.py` script with held-out test splits for evaluation.

---

## Download

### Datasets → `data/Bio_finetune_data/`

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com   # optional, faster in China

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

**Via ModelScope** (fast in China):

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

### Fine-tuned Models → `models/finetuned/`

```bash
# Single fine-tuned checkpoint
huggingface-cli download zhangtaolab/PlantHelixSeek-lncRNA \
    --local-dir ./models/finetuned/PlantHelixSeek-lncRNA

# All 11 checkpoints in parallel
for m in core_promoter lncRNA open_chromatin sequence_conservation \
         H3K27ac H3K27me3 H3K4me3 translation_onoff \
         promoter_strength terminator_strength translation_absolute; do
  huggingface-cli download "zhangtaolab/PlantHelixSeek-$m" \
      --local-dir "./models/finetuned/PlantHelixSeek-$m" &
done; wait
```

**Via ModelScope** (China-friendly):

```bash
# Single
modelscope download --model zhangtaolab/PlantHelixSeek-lncRNA \
    --local_dir ./models/finetuned/PlantHelixSeek-lncRNA

# All 11 in parallel
for m in core_promoter lncRNA open_chromatin sequence_conservation \
         H3K27ac H3K27me3 H3K4me3 translation_onoff \
         promoter_strength terminator_strength translation_absolute; do
  modelscope download --model "zhangtaolab/PlantHelixSeek-$m" \
      --local_dir "./models/finetuned/PlantHelixSeek-$m" &
done; wait
```

---

## Usage

```bash
# Predict with a fine-tuned classification checkpoint
python scripts/huggingface/predict.py \
    --model_path models/finetuned/PlantHelixSeek-lncRNA \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/predictions.csv
```
