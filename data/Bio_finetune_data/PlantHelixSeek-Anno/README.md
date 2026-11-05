---
license: cc-by-nc-4.0
task_categories:
- token-classification
language:
- en
tags:
- dna
- plant
- genomics
- gene-annotation
size_categories:
- 1M<n<10M
---

# PlantHelixSeek-Anno Dataset

Fine-tuning dataset for **gene structure prediction** — 17-class BILOU token classification. Used to train [PlantHelixSeek-Anno](https://huggingface.co/zhangtaolab/PlantHelixSeek-Anno).

## Format

Parquet, 7 columns per row:

| Column | Type | Description |
|--------|------|-------------|
| `species` | string | Species name |
| `chrom` | string | Chromosome |
| `start` | int64 | Window start coordinate |
| `end` | int64 | Window end coordinate |
| `strand` | string | Strand (`+` / `-`) |
| `sequence` | string | DNA sequence (8192 bp window) |
| `labels` | list\<int32\> | Per-base 17-class BILOU labels (same length as sequence) |

## Splits

| Split | Rows | Size |
|-------|-----:|-----:|
| train | 875,050 | 3.1 GB |
| val | 133,061 | 475 MB |

## Label scheme (17-class BILOU)

`O` (intergenic), `B/I/L/U-CDS`, `B/I/L/U-INTRON`, `B/I/L/U-UTR5`, `B/I/L/U-UTR3`

## License

CC-BY-NC-4.0
