---
license: cc-by-nc-4.0
library_name: transformers
base_model: zhangtaolab/PlantHelixSeek
tags:
- dna
- plant
- genomics
- cis-regulatory
- sequence-classification
pipeline_tag: text-classification
---

# PlantHelixSeek-CRE

Fine-tuned from [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek)
for sequence-based prediction of cis-regulatory elements (CREs).

## Task

- **Type**: binary sequence classification (CRE versus background)
- **Input**: 8,192 bp windows
- **Output**: probability that the sequence contains an open-chromatin/CRE region

## Training data

Positive sequences were open-chromatin regions from 16 plant species in
PlantCADB (Ding et al., 2023). Negative sequences were randomly sampled
genomic regions at a 1:2 positive-to-negative ratio. Genome-wide prediction and
peak calling were performed with the pipeline in
[`scripts/cis_regulatory/`](https://github.com/zhangtaolab/PlantHelixSeek/tree/dev/scripts/cis_regulatory).

## Training

| Setting | Value |
|---|---|
| Base model | PlantHelixSeek-470M |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Precision | bf16 |
| Seed | 42 |

## Reported performance

The manuscript reports genome-wide prediction of 41,276 candidate CREs in
*L. punctata* and experimental validation of 29 promoter and 22 enhancer
candidates by dual-fluorescent reporter assays (Figure 5; Supplemental
Tables 12 and 13). Of the 51 tested candidates, 20 of 22 enhancer candidates
showed significant activity.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-CRE", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-CRE", trust_remote_code=True
)
```

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek
learns transferable genome representations that capture conserved genome
grammar for functional interpretation across plant species. Molecular Plant,
2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

Dataset source: Ding K, et al. PlantCADB. Genomics, Proteomics & Bioinformatics
2023;21(2):311-323. doi:10.1016/j.gpb.2022.10.005.

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
