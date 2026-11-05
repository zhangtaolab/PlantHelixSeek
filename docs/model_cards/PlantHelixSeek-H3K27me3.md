---
license: cc-by-nc-4.0
library_name: transformers
base_model: zhangtaolab/PlantHelixSeek
tags:
- dna
- plant
- genomics
- language-model
- sequence-classification
pipeline_tag: text-classification
---

# PlantHelixSeek-H3K27me3

Fine-tuned checkpoint of [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek) (470M parameters) for **binary classification of H3K27me3**.

## Task

- **Type**: binary classification of H3K27me3
- **Label**: 0 = no mark, 1 = mark present
- **Input length**: up to 2,000 bp
- **Dataset**: [plant-multi-species-histone-modifications](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-histone-modifications)

## Dataset and splits

The checkpoint uses the released `train.csv`, `dev.csv` and `test.csv` splits of `plant-multi-species-histone-modifications` (79,748 / 9,968 / 9,968 sequences). The source of the dataset is described on its dataset card.

## Training

| Setting | Value |
|---|---|
| Base model | PlantHelixSeek-470M |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 16 |
| Warm-up ratio | 0.1 |
| Weight decay | 0.01 |
| Scheduler | linear |
| Optimizer | AdamW (fused) |
| Precision | bf16 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Hardware | one NVIDIA GPU |
| Training loss | 0.389 |
| Training time | 3.5 h |

## Held-out validation metrics

Metrics on the held-out `dev.csv` split (the split used for checkpoint selection).

| Metric | Value |
|---|---:|
| f1 | 0.795 |
| AUROC | 0.880 |
| AUPRC | 0.881 |
| mcc | 0.593 |
| accuracy | 0.796 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-H3K27me3", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-H3K27me3", trust_remote_code=True
)
```

Always pass `trust_remote_code=True`; the HelixSeek architecture is registered through the Hugging Face `auto_map` mechanism.

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns transferable genome representations that capture conserved genome grammar for functional interpretation across plant species. Molecular Plant, 2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

Dataset source: Liu et al., Molecular Plant 2025.

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
