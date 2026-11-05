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

# PlantHelixSeek-terminator_strength

Fine-tuned checkpoint of [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek) (470M parameters) for **regression of terminator strength**.

## Task

- **Type**: regression of terminator strength
- **Label**: continuous strength score
- **Input length**: 170 bp
- **Dataset**: [plant-multi-species-terminator-strength](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-terminator-strength)

## Dataset and splits

The checkpoint uses the released `train.csv`, `dev.csv` and `test.csv` splits of `plant-multi-species-terminator-strength` (43,294 / 4,806 / 5,309 sequences). The source of the dataset is described on its dataset card.

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
| Training loss | 0.235 |
| Training time | 0.3 h |

## Held-out validation metrics

Metrics on the held-out `dev.csv` split (the split used for checkpoint selection).

| Metric | Value |
|---|---:|
| spearmanr | 0.831 |
| r2 | 0.730 |
| mae | 0.372 |
| mse | 0.239 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-terminator_strength", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-terminator_strength", trust_remote_code=True
)
```

Always pass `trust_remote_code=True`; the HelixSeek architecture is registered through the Hugging Face `auto_map` mechanism.

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns transferable genome representations that capture conserved genome grammar for functional interpretation across plant species. Molecular Plant, 2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

Dataset source: Mendoza-Revilla et al., Communications Biology 2024.

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
