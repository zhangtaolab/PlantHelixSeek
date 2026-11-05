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

# PlantHelixSeek-sequence_conservation

Fine-tuned checkpoint of [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek) (470M parameters) for **binary classification of conserved sequence**.

## Task

- **Type**: binary classification of conserved sequence
- **Label**: 0 = non-conserved, 1 = conserved
- **Input length**: 1,000 bp
- **Dataset**: [plant-multi-species-sequence-conservation](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-sequence-conservation)

## Dataset and splits

The checkpoint uses the released `train.csv`, `dev.csv` and `test.csv` splits of `plant-multi-species-sequence-conservation` (56,320 / 7,040 / 7,040 sequences). The source of the dataset is described on its dataset card.

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
| Training loss | 0.203 |
| Training time | 1.9 h |

## Held-out validation metrics

Metrics on the held-out `dev.csv` split (the split used for checkpoint selection).

| Metric | Value |
|---|---:|
| f1 | 0.914 |
| AUROC | 0.969 |
| AUPRC | 0.970 |
| mcc | 0.828 |
| accuracy | 0.914 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-sequence_conservation", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-sequence_conservation", trust_remote_code=True
)
```

Always pass `trust_remote_code=True`; the HelixSeek architecture is registered through the Hugging Face `auto_map` mechanism.

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns transferable genome representations that capture conserved genome grammar for functional interpretation across plant species. Molecular Plant, 2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

Dataset source: Liu et al., Molecular Plant 2025.

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
