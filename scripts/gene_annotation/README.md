# PlantHelixSeek-Anno — Gene Structure Prediction

PlantHelixSeek-Anno is a token-classification checkpoint fine-tuned from the PlantHelixSeek base. It labels every nucleotide in a FASTA sequence with one of **17 BILOU tags** spanning gene features (CDS / intron / 5′UTR / 3′UTR / intergenic), then decodes the per-base predictions into standard GFF3 gene models.

> 📌 中文文档见 [README.zh.md](README.zh.md)。

## Pipeline

```
Download                      Predict                       GFF3 Decoding
huggingface-cli download      →  predict_genome.py          →  pred_to_gff3.py
     (model weights)              (.npz per-base labels)       (.gff3 gene models)
```

## Model

| Checkpoint | Base | Task | Labels | Download |
|------------|------|------|:------:|----------|
| `PlantHelixSeek-Anno` | PlantHelixSeek 470M | Token classification | 17 (BILOU) | [🤗 HF](https://huggingface.co/zhangtaolab/PlantHelixSeek-Anno) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/PlantHelixSeek-Anno) |

Weights ship at `models/PlantHelixSeek-Anno/`; load with `trust_remote_code=True`.

> 💡 If `models/PlantHelixSeek-Anno/` is already downloaded, skip [Download](#1-download) and go straight to [Prediction](#2-prediction-predict_genomepy).

### Label scheme (17-class BILOU)

| Group | Tags |
|-------|------|
| intergenic | `O` |
| CDS | `B-CDS` `I-CDS` `L-CDS` `U-CDS` |
| Intron | `B-INTRON` `I-INTRON` `L-INTRON` `U-INTRON` |
| 5′ UTR | `B-UTR5` `I-UTR5` `L-UTR5` `U-UTR5` |
| 3′ UTR | `B-UTR3` `I-UTR3` `L-UTR3` `U-UTR3` |

---

## 1. Download

```bash
huggingface-cli download zhangtaolab/PlantHelixSeek-Anno \
    --local-dir models/PlantHelixSeek-Anno
```

**Via ModelScope** (China-friendly):

```bash
modelscope download --model zhangtaolab/PlantHelixSeek-Anno \
    --local_dir models/PlantHelixSeek-Anno

> 💡 China mirror: `export HF_ENDPOINT=https://hf-mirror.com` for faster downloads.

---

## 2. Prediction (`predict_genome.py`)

Sliding-window genome scan. Each 8 192 bp window is inferred on both strands; the valid middle region is stitched directly (no probability averaging). Attention layers are patched to SDPA to avoid O(N²) memory.

### Usage

```bash
python scripts/gene_annotation/predict_genome.py \
    --genome-fasta data/gene_annotation/test_sequence.fa \
    --model-path models/PlantHelixSeek-Anno \
    --output-dir tmp/anno_predict \
    --batch-size 4 --device cuda:0 \
    --use-crf
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--genome-fasta` | *(required)* | Input FASTA |
| `--model-path` | *(required)* | Checkpoint directory (with `config.json` + `model.safetensors`) |
| `--output-dir` | *(required)* | Output directory for `.npz` files |
| `--batch-size` | 8 | Inference batch size (RTX 5090: 4 at seq 8 192) |
| `--device` | auto | `cuda:0`, `cpu`, … |
| `--window-size` | 8192 | Window size (≤ model max Position embeddings) |
| `--stride` | 4096 | Window step (= window/2, 50% overlap) |
| `--use-crf` | off | Enable CRF Viterbi decoding per window |
| `--crf-path` | None | Path to trained CRF weights (if omitted, default constraints) |
| `--genic-bias` | 0.0 | Intergenic bias (positive → fewer genes) |
| `--chromosome` | None | Process only the specified chromosome |

### Output

Per chromosome `{chrom}.npz` containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `fwd_predictions` / `rev_predictions` | `[seq_len]` | 17-class BILOU labels (CRF-decoded if `--use-crf`) |
| `fwd_probabilities` / `rev_probabilities` | `[seq_len, 17]` | Per-base 17-class softmax probabilities |
| `fwd_feature_probs` / `rev_feature_probs` | `[seq_len, 5]` | Aggregated 5-class feature probabilities |

---

## 3. GFF3 Decoding (`pred_to_gff3.py`)

Decodes NPZ predictions into standard GFF3 gene models with ID/Parent hierarchy.

### Decoding methods

| Method | Description |
|--------|-------------|
| `viterbi` | Viterbi on 5-class feature probs + CRF label delimiting (recommended) |
| `viterbi+orf` | Same + ORF correction |
| `heuristic` | Threshold + heuristic clustering (backward compatible) |

### Usage

```bash
python scripts/gene_annotation/pred_to_gff3.py \
    --npz-dir tmp/anno_predict \
    --genome-fasta data/gene_annotation/test_sequence.fa \
    --decoding-method viterbi \
    --workers 4
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--npz-dir` | *(required)* | Directory containing `.npz` files |
| `--genome-fasta` | *(required)* | Genome FASTA (for sequence / chromosome sizes) |
| `--decoding-method` | `viterbi+orf` | `viterbi` / `viterbi+orf` / `heuristic` |
| `--workers` | 1 | Parallel workers |
| `--domain` | `plant` | Transition matrix type (`plant` / `animal`) |
| `--source` | `PlantHelixSeek-anno` | GFF3 source column |
| `--species-prefix` | `""` | Species abbreviation (e.g. `LP`); when set, a merged `{species}.gff3` is generated with normalised gene IDs |
| `--predictor-tag` | `PHS` | Gene ID suffix tag |
| `--intergenic-boost` | 10.0 | Intergenic probability boost (positive → more conservative) |
| `--viterbi-alpha` | 0.0 | Transition matrix smoothing (0–1) |
| `--reference-gff3` | None | Reference GFF3 for evaluation metrics |
| `--genomic-offset` | 0 | Coordinate offset (for sub-region evaluation) |
| `--output-dir` | `--npz-dir` | Output directory (defaults to NPZ dir) |

### Output

| File | Description |
|------|-------------|
| `{chrom}.gff3` | Per-chromosome GFF3 (gene / mRNA / CDS / UTR with ID/Parent) |
| `{species}.gff3` | Merged genome-wide GFF3 (only when `--species-prefix` is set); gene IDs follow `{PREFIX}{CHR:02d}G{GENE:06d}_{TAG}` |

---

## Requirements

- Python ≥ 3.12
- CUDA GPU — PlantHelixSeek relies on Triton / `flash-linear-attention` kernels.
- Project venv: `source .venv/bin/activate`
- Deps: `torch`, `transformers`, `biopython`, `numpy`, `numba` (Viterbi JIT), `tqdm` (all in `requirements.txt`)

---

## Files

| File | Purpose |
|------|---------|
| `predict_genome.py` | Sliding-window prediction → `.npz` |
| `pred_to_gff3.py` | NPZ → GFF3 gene models |
| `viterbi_utils.py` | Viterbi decoder + transition probability matrix |
| `train_token_cls.py` | Linear-chain CRF with BILOU constraints (for `--use-crf`) |
| `transition_probs.npz` | Pre-computed plant transition probabilities |
me-wide GFF3 (only when `--species-prefix` is set); gene IDs follow `{PREFIX}{CHR:02d}G{GENE:06d}_{TAG}` |

---

## Requirements

- Python ≥ 3.12
- Accelerator: NVIDIA CUDA GPU — PlantHelixSeek relies on Triton / `flash-linear-attention` kernels; pure-CPU inference is not supported.
- Project venv: `source .venv/bin/activate`
- Deps: `torch`, `transformers`, `biopython`, `numpy`, `numba` (Viterbi JIT), `tqdm` (all in `requirements.txt`)

---

## Files

| File | Purpose |
|------|---------|
| `predict_genome_multigpu.py` | Sliding-window prediction → `.npz` |
| `pred_to_gff3.py` | NPZ → GFF3 gene models |
| `viterbi_utils.py` | Viterbi decoder + transition probability matrix |
| `train_token_cls.py` | Linear-chain CRF with BILOU constraints (for `--use-crf`) |
| `transition_probs.npz` | Pre-computed plant transition probabilities |
