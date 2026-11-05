---
license: cc-by-nc-4.0
library_name: transformers
base_model: zhangtaolab/PlantHelixSeek
tags:
- dna
- plant
- genomics
- language-model
- regression
---

# PlantHelixSeek-promoter_strength

Fine-tuned checkpoint of [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek) (470M parameters) for **regression of core-promoter strength**.

## Task

- **Type**: regression of core-promoter strength
- **Label**: continuous strength score
- **Input length**: 170 bp
- **Dataset**: [plant-multi-species-promoter-strength](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-promoter-strength)

## Dataset and splits

The checkpoint uses the released `train.csv`, `dev.csv` and `test.csv` splits of `plant-multi-species-promoter-strength` (58,179 / 6,825 / 7,154 sequences). The source of the dataset is described on its dataset card.

## Training

| Setting | Value |
|---|---|
| Base model | PlantHelixSeek-470M |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Per-device batch size | 6 |
| Gradient accumulation | 2 |
| Effective batch size | 12 |
| Warm-up ratio | 0.1 |
| Weight decay | 0.01 |
| Scheduler | linear |
| Optimizer | AdamW (fused) |
| Precision | bf16 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Hardware | one NVIDIA GPU |
| Training loss | 0.800 |
| Training time | 0.4 h |

## Held-out validation metrics

Metrics on the held-out `dev.csv` split (the split used for checkpoint selection).

| Metric | Value |
|---|---:|
| spearmanr | 0.881 |
| r2 | 0.778 |
| mae | 0.600 |
| mse | 0.609 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-promoter_strength", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-promoter_strength", trust_remote_code=True
)
```

Always pass `trust_remote_code=True`; the HelixSeek architecture is registered through the Hugging Face `auto_map` mechanism.

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns transferable genome representations that capture conserved genome grammar for functional interpretation across plant species. Molecular Plant, 2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

Dataset source: Mendoza-Revilla et al., Communications Biology 2024; Jores et al., Nature Plants 2021, doi:10.1038/s41477-021-00932-y.

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
