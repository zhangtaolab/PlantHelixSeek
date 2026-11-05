# Fine-tune datasets and model–dataset mapping

This document describes the nine downstream datasets and eleven fine-tuned
checkpoints in the [PlantHelixSeek collection](https://huggingface.co/collections/zhangtaolab/planthelixseek).
All datasets are multi-species plant genomic tasks released as fixed
`train.csv`, `dev.csv` and `test.csv` splits. All checkpoints are fine-tuned
from the PlantHelixSeek base model (470M, MLM) for three epochs.

The same information is provided on each Hugging Face model card and dataset
card. Task type (`binary`, `multiclass` or `regression`) is stored in each
checkpoint's `config.json` under `problem_type`.

## Model–dataset mapping

| Fine-tuned model | Task type | Dataset | Train / dev / test |
|---|---|---:|---:|
| `PlantHelixSeek-core_promoter` | binary | plant-multi-species-core-promoters | 66,560 / 8,320 / 8,320 |
| `PlantHelixSeek-lncRNA` | binary | plant-multi-species-lncRNAs | 40,248 / 5,028 / 5,032 |
| `PlantHelixSeek-open_chromatin` | multiclass (3) | plant-multi-species-open-chromatin | 118,800 / 14,850 / 14,850 |
| `PlantHelixSeek-sequence_conservation` | binary | plant-multi-species-sequence-conservation | 56,320 / 7,040 / 7,040 |
| `PlantHelixSeek-H3K27ac` | binary | plant-multi-species-histone-modifications (H3K27ac) | 81,920 / 10,240 / 10,240 |
| `PlantHelixSeek-H3K27me3` | binary | plant-multi-species-histone-modifications (H3K27me3) | 79,748 / 9,968 / 9,968 |
| `PlantHelixSeek-H3K4me3` | binary | plant-multi-species-histone-modifications (H3K4me3) | 81,920 / 10,240 / 10,240 |
| `PlantHelixSeek-translation_onoff` | binary | cross_species_leaf_on_off_translation | 21,894 / 5,492 / 39,035 |
| `PlantHelixSeek-promoter_strength` | regression | plant-multi-species-promoter-strength | 58,179 / 6,825 / 7,154 |
| `PlantHelixSeek-terminator_strength` | regression | plant-multi-species-terminator-strength | 43,294 / 4,806 / 5,309 |
| `PlantHelixSeek-translation_absolute` | regression | cross_species_leaf_absolute_translation | 21,894 / 5,492 / 39,035 |

The histone-modifications dataset bundles three marks; one dataset therefore
maps to three binary checkpoints.

## Dataset catalog

### plant-multi-species-core-promoters

- **Task**: binary classification of active core promoters.
- **Provenance**: curated in PDLLMs (Liu et al., 2025).
- **Files and splits**: `train.csv` 66,560; `dev.csv` 8,320; `test.csv` 8,320.
- **Columns**: `sp` (species code), `sequence` (300 bp), `label` (0 or 1).
- **Species**: 13 (ath, barley, bd30, broomcorn, cotton, grape, maize, poplar,
  potato, rice, soybean, tomato, wheat).
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Liu et al., Molecular Plant 2025, doi:10.1016/j.molp.2024.12.006.

### plant-multi-species-lncRNAs

- **Task**: binary classification of long non-coding RNAs.
- **Provenance**: curated in PDLLMs (Liu et al., 2025); training sequences
  derived from the AgroNT plant-genomic-benchmark collection.
- **Files and splits**: `train.csv` 40,248; `dev.csv` 5,028; `test.csv` 5,032.
- **Columns**: `sp`, `sequence` (up to 2,000 bp), `label` (0 or 1).
- **Species**: 6 (broomcorn, cassava, maize, soybean, tomato, wheat).
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Liu et al., Molecular Plant 2025; Mendoza-Revilla et al.,
  Communications Biology 2024, doi:10.1038/s42003-024-06465-2.

### plant-multi-species-open-chromatin

- **Task**: three-class classification of open-chromatin state (labels 0, 1, 2).
- **Provenance**: curated in PDLLMs (Liu et al., 2025); open-chromatin peaks
  originally compiled in PlantCADB (Ding et al., 2023).
- **Files and splits**: `train.csv` 118,800; `dev.csv` 14,850; `test.csv` 14,850.
- **Columns**: `sp`, `sequence` (up to 3,000 bp), `label` (0, 1 or 2).
- **Species**: 33.
- **Primary metric**: Matthews correlation coefficient (MCC); micro- and
  weighted F1 are also reported.
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Liu et al., Molecular Plant 2025; Ding et al., GPB 2023,
  doi:10.1016/j.gpb.2022.10.005.

### plant-multi-species-sequence-conservation

- **Task**: binary classification of conserved versus non-conserved sequence.
- **Provenance**: curated in PDLLMs (Liu et al., 2025).
- **Files and splits**: `train.csv` 56,320; `dev.csv` 7,040; `test.csv` 7,040.
- **Columns**: `sp`, `sequence` (1,000 bp), `label` (0 or 1).
- **Species**: 11 (aly, ath, barley, bd30, broomcorn, maize, poplar, potato,
  rice, soybean, tomato).
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Liu et al., Molecular Plant 2025.

### plant-multi-species-histone-modifications

- **Task**: three binary classification tasks (H3K27ac, H3K27me3, H3K4me3).
- **Provenance**: curated in PDLLMs (Liu et al., 2025).
- **Files and splits**: H3K27ac 81,920 / 10,240 / 10,240; H3K27me3 79,748 /
  9,968 / 9,968; H3K4me3 81,920 / 10,240 / 10,240.
- **Columns**: `sp`, `sequence` (up to 2,000 bp), `label` (0 or 1).
- **Species**: 8 (ath, cotton, maize, potato, rice, soybean, tomato, wheat).
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Liu et al., Molecular Plant 2025.

### plant-multi-species-promoter-strength

- **Task**: regression of core-promoter strength.
- **Provenance**: AgroNT plant-genomic-benchmark (Mendoza-Revilla et al., 2024);
  the underlying measurements are plant STARR-seq core-promoter assays
  (Jores et al., 2021).
- **Files and splits**: `train.csv` 58,179; `dev.csv` 6,825; `test.csv` 7,154.
- **Columns**: `name` (candidate identifier), `sequence` (170 bp), `label`
  (continuous strength score).
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Mendoza-Revilla et al., Communications Biology 2024; Jores et
  al., Nature Plants 2021, doi:10.1038/s41477-021-00932-y.

### plant-multi-species-terminator-strength

- **Task**: regression of terminator strength.
- **Provenance**: AgroNT plant-genomic-benchmark (Mendoza-Revilla et al., 2024).
- **Files and splits**: `train.csv` 43,294; `dev.csv` 4,806; `test.csv` 5,309.
- **Columns**: `name`, `sequence` (170 bp), `label` (continuous strength score).
- **Split method**: fixed splits distributed with the dataset; no re-splitting.
- **License**: CC-BY-NC-SA 4.0.
- **Citation**: Mendoza-Revilla et al., Communications Biology 2024.

### cross_species_leaf_on_off_translation

- **Task**: binary classification of translation on/off state in leaf tissue.
- **Provenance**: PlantCAD2 fine-tuning tasks (Zhai et al., 2026).
- **Files and splits**: `train.csv` 21,894; `dev.csv` 5,492; `test.csv` 39,035.
- **Columns**: `gene`, `chr`, `strand`, `tpm`, `sequence` (500 bp), `label`
  (0 or 1).
- **Split method**: fixed splits distributed by PlantCAD2; no re-splitting.
- **License**: CC-BY-NC 4.0.
- **Citation**: Zhai et al., Cell Genomics 2026,
  doi:10.1016/j.xgen.2026.101329.

### cross_species_leaf_absolute_translation

- **Task**: regression of absolute translation level in leaf tissue.
- **Provenance**: PlantCAD2 fine-tuning tasks (Zhai et al., 2026).
- **Files and splits**: `train.csv` 21,894; `dev.csv` 5,492; `test.csv` 39,035.
- **Columns**: `gene`, `chr`, `strand`, `tpm`, `sequence` (500 bp), `label`
  (continuous translation-level score).
- **Split method**: fixed splits distributed by PlantCAD2; no re-splitting.
- **License**: CC-BY-NC-ND 4.0.
- **Citation**: Zhai et al., Cell Genomics 2026,
  doi:10.1016/j.xgen.2026.101329.

## Training protocol

All eleven checkpoints were fine-tuned from PlantHelixSeek-470M on a single GPU
with the same optimizer and schedule. Effective batch size differs because the
per-device batch size was adjusted to the sequence length of each task.

| Setting | Value |
|---|---|
| Base model | PlantHelixSeek-470M |
| Epochs | 3 |
| Learning rate | 2 × 10-5 |
| Scheduler | linear decay |
| Warm-up ratio | 0.1 |
| Weight decay | 0.01 |
| Optimizer | AdamW (fused) |
| Precision | bf16 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Hardware | one NVIDIA GPU |

| Checkpoint | Per-device batch | Gradient accumulation | Effective batch |
|---|---:|---:|---:|
| core_promoter | 6 | 2 | 12 |
| lncRNA | 1 | 16 | 16 |
| open_chromatin | 1 | 8 | 8 |
| sequence_conservation | 1 | 16 | 16 |
| H3K27ac | 1 | 16 | 16 |
| H3K27me3 | 1 | 16 | 16 |
| H3K4me3 | 1 | 16 | 16 |
| translation_onoff | 3 | 5 | 15 |
| promoter_strength | 6 | 2 | 12 |
| terminator_strength | 6 | 2 | 12 |
| translation_absolute | 3 | 5 | 15 |

## Held-out validation metrics

The values below are measured on the held-out `dev.csv` split, which is the
split used for checkpoint selection. The `test.csv` split is distributed with
each dataset and can be evaluated with `scripts/huggingface/predict.py`.

### Classification

| Checkpoint | F1 | AUROC | AUPRC | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|
| core_promoter | 0.829 | 0.909 | 0.901 | 0.656 | 0.828 |
| lncRNA | 0.904 | 0.946 | 0.946 | 0.775 | 0.890 |
| open_chromatin (3-class) | 0.663 | 0.835 | 0.740 | 0.493 | 0.661 |
| sequence_conservation | 0.914 | 0.969 | 0.970 | 0.828 | 0.914 |
| H3K27ac | 0.831 | 0.905 | 0.907 | 0.660 | 0.830 |
| H3K27me3 | 0.795 | 0.880 | 0.881 | 0.593 | 0.796 |
| H3K4me3 | 0.910 | 0.970 | 0.972 | 0.821 | 0.910 |
| translation_onoff | 0.810 | 0.805 | 0.875 | 0.439 | 0.748 |

### Regression

| Checkpoint | Spearman r | R2 | MAE |
|---|---:|---:|---:|
| promoter_strength | 0.881 | 0.778 | 0.600 |
| terminator_strength | 0.831 | 0.730 | 0.372 |
| translation_absolute | 0.520 | 0.262 | 0.447 |

## Download

### Datasets to `data/Bio_finetune_data/`

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

### Fine-tuned models to `models/finetuned/`

```bash
for m in core_promoter lncRNA open_chromatin sequence_conservation \
         H3K27ac H3K27me3 H3K4me3 translation_onoff \
         promoter_strength terminator_strength translation_absolute; do
  huggingface-cli download "zhangtaolab/PlantHelixSeek-$m" \
      --local-dir "./models/finetuned/PlantHelixSeek-$m" &
done; wait
```

## Usage

```bash
python scripts/huggingface/predict.py \
    --model_path models/finetuned/PlantHelixSeek-lncRNA \
    --input_file data/Bio_finetune_data/plant-multi-species-lncRNAs/test.csv \
    --output_file predictions.csv
```

## Citation

If you use these datasets or checkpoints, cite the PlantHelixSeek manuscript
and the original dataset publications listed in each dataset section above.
The machine-readable citation is in [CITATION.cff](../CITATION.cff).
