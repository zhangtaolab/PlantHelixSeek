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

# PlantHelixSeek-open_chromatin

Fine-tuned checkpoint of [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek) (470M parameters) for **three-class classification of open-chromatin state**.

## Task

- **Type**: three-class classification of open-chromatin state
- **Label**: 0, 1, 2 (three chromatin classes)
- **Input length**: up to 3,000 bp
- **Dataset**: [plant-multi-species-open-chromatin](https://huggingface.co/datasets/zhangtaolab/plant-multi-species-open-chromatin)

## Dataset and splits

The checkpoint uses the released `train.csv`, `dev.csv` and `test.csv` splits of `plant-multi-species-open-chromatin` (118,800 / 14,850 / 14,850 sequences). The source of the dataset is described on its dataset card.

## Training

| Setting | Value |
|---|---|
| Base model | PlantHelixSeek-470M |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Warm-up ratio | 0.1 |
| Weight decay | 0.01 |
| Scheduler | linear |
| Optimizer | AdamW (fused) |
| Precision | bf16 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Hardware | one NVIDIA GPU |
| Training loss | 0.544 |
| Training time | 5.4 h |

## Held-out validation metrics

Metrics on the held-out `dev.csv` split (the split used for checkpoint selection).

| Metric | Value |
|---|---:|
| f1 | 0.663 |
| AUROC | 0.835 |
| AUPRC | 0.740 |
| mcc | 0.493 |
| accuracy | 0.661 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-open_chromatin", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-open_chromatin", trust_remote_code=True
)
```

Always pass `trust_remote_code=True`; the HelixSeek architecture is registered through the Hugging Face `auto_map` mechanism.

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns transferable genome representations that capture conserved genome grammar for functional interpretation across plant species. Molecular Plant, 2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

Dataset source: Liu et al., Molecular Plant 2025; Ding et al., GPB 2023, doi:10.1016/j.gpb.2022.10.005.

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
