---
license: cc-by-nc-4.0
library_name: transformers
language:
- en
tags:
- dna
- plant
- genomics
- language-model
- mixture-of-experts
pipeline_tag: fill-mask
base_model: []
---

# PlantHelixSeek

PlantHelixSeek is a 470M-parameter plant DNA foundation model pretrained with
masked language modeling (MLM) at single-nucleotide resolution on 53 plant
genomes. It is the base model for the PlantHelixSeek fine-tuned checkpoints.

## Model details

| Property | Value |
|---|---|
| Total parameters | 470M |
| Active parameters per token | 240M |
| Hidden size | 512 |
| Layers | 39 (27 Transformer + 9 HelixSeek-Delta + 3 HelixSeek-MLA) |
| Attention heads | 16 |
| Experts | 8 routed experts (top-2 routing) + 1 shared expert |
| Maximum context | 8,192 bp |
| Vocabulary | 11 tokens (A, T, C, G, N and special tokens) |
| Tokenizer | SingleBase (one nucleotide per token) |
| Objective | bidirectional masked language modeling |
| Precision | bf16 |

## Architecture

HelixSeek interleaves three layer types:

- **Transformer layers**: full self-attention for precise positional reasoning.
- **HelixSeek-Delta layers**: Delta Attention (KDA), a linear attention with a
  gated delta rule and O(n) complexity. Adapted from Kimi Linear
  (arXiv:2510.26692) and Gated DeltaNet (arXiv:2412.06464).
- **HelixSeek-MLA layers**: Multi-head Latent Attention (MLA) with KV-cache
  compression. Adapted from DeepSeek-V2 (arXiv:2405.04434) and Kimi Linear
  (arXiv:2510.26692).

The feed-forward network is a sparse mixture of experts with eight routed
experts (top-2 routing) and one shared expert, following Switch Transformers
(Fedus et al., JMLR 2022).

The architecture is implemented in the model repository and loaded through
`trust_remote_code=True`. The custom modules are registered through the
Hugging Face `auto_map` in `config.json`.

## Training data

The pretraining corpus contains 16.38 billion nucleotides from 53 plant
genomes: 52 angiosperms and the liverwort *Marchantia polymorpha*. The corpus
combines gene regions (7.81 billion nucleotides), cis-regulatory regions
(1.69 billion nucleotides) and genomic background (6.88 billion nucleotides).
Genomes and annotations were obtained from Ensembl Plants release 62.

## Training procedure

| Setting | Value |
|---|---|
| Objective | masked language modeling |
| Masking | 15% of tokens; 80% MASK, 10% random, 10% unchanged |
| Epochs | 10 |
| Maximum sequence length | 8,192 tokens |
| Precision | bf16 |
| Optimizer | AdamW (beta1 = 0.9, beta2 = 0.95, epsilon = 1e-6) |
| Weight decay | 0.1 |
| Learning rate | 4e-4, cosine decay |
| Per-device batch size | 1 |
| Gradient accumulation | 30 |
| Gradient clipping | 1.0 |
| Positional encoding | NTK-aware RoPE scaling above 4,096 tokens |
| Seed | 42 |
| Hardware | 8 x NVIDIA RTX 5090, approximately 20 days |

## Validation

| Metric | Value |
|---|---|
| Final validation perplexity | 2.55 |
| Masked-base recovery, 5-95% of context | 57.5% |
| Masked-base recovery, Transformer-470M baseline | 32.4% |

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained(
    "zhangtaolab/PlantHelixSeek", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "zhangtaolab/PlantHelixSeek", trust_remote_code=True
)

seq = "ATCGGCTAAGCCT<mask>GTCAGTTACCGATCG"
inputs = tokenizer(seq, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
```

## Intended use and limitations

PlantHelixSeek is intended for plant genome representation learning, variant
effect scoring, genome annotation and cis-regulatory element prediction. It is
trained on plant genomes only and is not intended for non-plant DNA. The
maximum context is 8,192 bp; the model is a masked language model and does not
generate sequences. Predictions should be validated experimentally when used
for functional claims.

## Citation

Wu Y, He Y, Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek
learns transferable genome representations that capture conserved genome
grammar for functional interpretation across plant species. Molecular Plant,
2026. Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

CC-BY-NC 4.0 (research use; contact the authors for commercial licensing).
