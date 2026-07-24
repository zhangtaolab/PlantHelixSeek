# PlantHelixSeek-CRE — Cis-Regulatory Element Prediction

PlantHelixSeek-CRE is a binary fine-tuned checkpoint of the PlantHelixSeek base on a plant cis-regulatory element (CRE) dataset. It scans a genome with sliding windows, predicts per-bin CRE / open-chromatin probability, and emits a standard BigWig track.

> 📌 中文文档见 [README.zh.md](README.zh.md)。

## Pipeline

```
Download                      Predict                       Peak Calling
huggingface-cli download      →  predict_cre.py             →  call_peaks_from_bigwig.py
     (model weights)              (.bw BigWig)                  (.bed / .narrowPeak / .broadPeak)
```

## Model

| Checkpoint             | Base                | Task                  | Download                                                                         |
| ---------------------- | ------------------- | --------------------- | -------------------------------------------------------------------------------- |
| `PlantHelixSeek-CRE` | PlantHelixSeek 470M | Binary classification | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-CRE) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-CRE) |

Weights ship at `models/PlantHelixSeek-CRE/`; load with `trust_remote_code=True`.

> 💡 If `models/PlantHelixSeek-CRE/` is already downloaded, skip [Download](#1-download) and go straight to [Prediction](#2-prediction-predict_crepy).

---

## 1. Download

Download the fine-tuned checkpoint from HuggingFace:

```bash
huggingface-cli download zhangtaolab/PlantHelixSeek-CRE \
    --local-dir models/PlantHelixSeek-CRE
```

**Via ModelScope** (China-friendly):

```bash
modelscope download --model zhangtaolab/PlantHelixSeek-CRE \
    --local_dir models/PlantHelixSeek-CRE

> 💡 China mirror: `export HF_ENDPOINT=https://hf-mirror.com` for faster downloads.

---

## 2. Prediction (`predict_cre.py`)

### Algorithm

| Item        | Value                                                                                 |
| ----------- | ------------------------------------------------------------------------------------- |
| Window size | 500 bp                                                                                |
| Stride      | 50 bp (adjustable via`--stride`)                                                    |
| Bin size    | 50 bp                                                                                 |
| Aggregation | arithmetic mean of class-1 probabilities over all windows covering a bin              |
| TTA         | off by default; `--tta` enables reverse-complement augmentation (~2x slower, slightly higher accuracy) |

Each 50 bp bin score is the arithmetic mean of class-1 probabilities across all overlapping windows (up to 10 per bin given the 500/50 overlap).

### Usage

```bash
# Predict with the default PlantHelixSeek-CRE checkpoint
python scripts/cis_regulatory/predict_cre.py

# Specify genome and output directory explicitly
python scripts/cis_regulatory/predict_cre.py \
    --model_path models/PlantHelixSeek-CRE \
    --genome data/cis_regulatory/test_sequence.fa \
    --output_dir tmp/cre_predict
```

### Arguments

| Argument             | Default                           | Description                                            |
| -------------------- | --------------------------------- | ------------------------------------------------------ |
| `--model_path`     | `models/PlantHelixSeek-CRE`     | Fine-tuned checkpoint path                             |
| `--genome`         | `data/cis_regulatory/test_sequence.fa` | Input FASTA                                            |
| `--output_dir`     | `tmp/cre_predict`               | Output directory                                       |
| `--batch_size`     | 256                               | Inference batch size                                   |
| `--stride`         | 50                                | Sliding-window stride (bp)                             |
| `--tta`            | off                               | Enable reverse-complement TTA                          |
| `--device`         | auto                              | Device, e.g.`cuda:0`, `cpu`                        |
| `--num_workers`    | 0                                 | DataLoader workers                                     |
| `--min_chrom_size` | 0                                 | Skip chromosomes smaller than this (bp); 0 = no filter |
| `--max_chrom_size` | 0                                 | Skip chromosomes larger than this (bp); 0 = no filter  |

### Output

| File                                     | Format     | Description                                           |
| ---------------------------------------- | ---------- | ----------------------------------------------------- |
| `PlantHelixSeek-CRE_{species}_OCR.bw`  | BigWig     | Standard genome track, loadable in IGV / UCSC Browser |
| `PlantHelixSeek-CRE_{species}_OCR.npy` | NumPy dict | `{chrom: float32 array}`, one entry per 50 bp bin   |

Resume: each chromosome is cached to `chunks/<chrom>.npy` on completion; interrupted runs automatically skip finished chromosomes.

---

## 3. Peak Calling (`call_peaks_from_bigwig.py`)

Call peaks from the predicted BigWig signal; outputs BED / narrowPeak / broadPeak.

### Default

Whole-genome (CIB) prediction is the recommended default:

```bash
python scripts/cis_regulatory/call_peaks_from_bigwig.py \
    --bigwig tmp/cre_predict/PlantHelixSeek-CRE_test_sequence_OCR.bw \
    --genome data/cis_regulatory/test_sequence.fa \
    --output_dir tmp/cre_predict \
    --refine_summit \
    --prefix PlantHelixSeek-CRE_test_sequence_OCR
```

### Arguments

| Argument               | Default        | Description                                                            |
| ---------------------- | -------------- | ---------------------------------------------------------------------- |
| `--bigwig`           | *(required)* | Input BigWig signal                                                    |
| `--genome`           | *(required)* | Genome FASTA (for chromosome sizes)                                    |
| `--output_dir`       | *(required)* | Output directory                                                       |
| `--threshold_method` | `mean_std`   | `mean_std` / `median_mad` / `percentile` / `fixed`             |
| `--threshold_params` | `1.5`        | k for mean_std/median_mad, percentile for percentile, cutoff for fixed |
| `--min_length`       | 50             | Minimum peak length (bp)                                               |
| `--max_length`       | 5000           | Maximum peak length (bp)                                               |
| `--min_score`        | 0.6            | Minimum peak score; peaks below this are dropped                       |
| `--merge_gap`        | 50             | Max gap to merge adjacent high-score regions (bp)                      |
| `--refine_summit`    | off            | Refine peak summit position                                            |
| `--prefix`           | *(auto)*     | Output filename prefix (derived from bigwig name if omitted)           |

### Output

| File                          | Format                      |
| ----------------------------- | --------------------------- |
| `{prefix}_peaks.bed`        | BED6                        |
| `{prefix}_peaks.narrowPeak` | BED6+4 (with summit offset) |
| `{prefix}_peaks.broadPeak`  | BED6+3                      |

---

## Requirements

- Python ≥ 3.12
- CUDA GPU — PlantHelixSeek relies on Triton / `flash-linear-attention` kernels; pure-CPU inference is not supported.
- Project venv: `source .venv/bin/activate`
- Deps: `pyBigWig`, `biopython`, `tqdm`, `torch`, `transformers` (all in `requirements.txt`)

---

## Files

| File                          | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `predict_cre.py`            | Genome-wide CRE prediction → BigWig   |
| `call_peaks_from_bigwig.py` | BigWig → BED / narrowPeak / broadPeak |
- Project venv: `source .venv/bin/activate`
- Deps: `pyBigWig`, `biopython`, `tqdm`, `torch`, `transformers` (all in `requirements.txt`)

---

## Files

| File                          | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `predict_cre.py`            | Genome-wide CRE prediction → BigWig   |
| `call_peaks_from_bigwig.py` | BigWig → BED / narrowPeak / broadPeak |
