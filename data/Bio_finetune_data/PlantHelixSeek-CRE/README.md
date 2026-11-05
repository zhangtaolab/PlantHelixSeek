---
license: cc-by-nc-4.0
task_categories:
- text-classification
language:
- en
tags:
- dna
- plant
- genomics
- cis-regulatory
size_categories:
- 1M<n<10M
---

# PlantHelixSeek-CRE Dataset

Fine-tuning dataset for **cis-regulatory element (CRE) prediction** — binary classification. Used to train [PlantHelixSeek-CRE](https://huggingface.co/zhangtaolab/PlantHelixSeek-CRE).

## Format

CSV, columns:

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | string | DNA sequence |
| `label` | int | 0 = non-CRE, 1 = CRE |

## Splits

| Split | Size |
|-------|-----:|
| train | 961 MB |
| dev | 121 MB |
| test | 121 MB |

## License

CC-BY-NC-4.0
