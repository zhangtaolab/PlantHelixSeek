---
license: cc-by-nc-4.0
library_name: transformers
base_model: zhangtaolab/PlantHelixSeek
tags:
- dna
- plant
- genomics
- gene-annotation
- token-classification
pipeline_tag: token-classification
---

# PlantHelixSeek-Anno

Fine-tuned from [PlantHelixSeek](https://huggingface.co/zhangtaolab/PlantHelixSeek)
for nucleotide-resolution gene structure prediction. Each nucleotide is
assigned one of 17 BILOU labels covering CDS, intron, 5' UTR, 3' UTR and
intergenic regions.

## Task

- **Type**: token classification (17-class BILOU)
- **Labels**: `O`; `B/I/L/U-CDS`; `B/I/L/U-INTRON`; `B/I/L/U-UTR5`; `B/I/L/U-UTR3`
- **Input**: 8,192 bp windows with a 4,096 bp step

## Training data

Gene models from 13 plant species were converted from GFF3 annotations into
BILOU labels. Training and validation chromosomes were separated at the
chromosome level. Intergenic windows were down-sampled to approximately 15% of
the training set. See the manuscript Methods for the gene-filtering criteria.

## Training

| Setting | Value |
|---|---|
| Base model | PlantHelixSeek-470M |
| Epochs | 3 |
| Encoder learning rate | 1e-5 |
| Head learning rate | 1e-4 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Weight decay | 0.05 |
| Warm-up | 5% |
| Precision | bf16 |
| Seed | 42 |
| Loss | weighted cross-entropy with boundary weights |

## Reported performance

From the manuscript (Figure 4, Supplemental Figure 12 and Supplemental
Tables 5 and 9):

| Benchmark | Result |
|---|---|
| Mean CDS-level F1 across seven species | 0.843 |
| *L. punctata* nucleotide-level F1 | 0.917 |
| *L. punctata* CDS-level F1 | 0.895 |
| *L. punctata* transcript-level F1 | 0.813 |

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "zhangtaolab/PlantHelixSeek-Anno", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek-Anno", trust_remote_code=True
)
```

The genome-wide prediction and GFF3 decoding pipeline is in
[`scripts/gene_annotation/`](https://github.com/zhangtaolab/PlantHelixSeek/tree/dev/scripts/gene_annotation).

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek
learns transferable genome representations that capture conserved genome
grammar for functional interpretation across plant species. Molecular Plant,
2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
