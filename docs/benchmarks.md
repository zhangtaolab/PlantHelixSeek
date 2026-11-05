# Compute benchmarks

This document reports the compute benchmarks summarized in the manuscript
(Supplemental Figure 2, Supplemental Figure 3 and Supplemental Table 14).
All measurements are forward-pass or training throughput measurements on the
same released PlantHelixSeek-470M weights. No model was retrained for this
comparison.

## Hardware and software

| Item | Configuration |
|---|---|
| GPU | NVIDIA RTX 5090D (32 GB); referred to as RTX 5090 in the tables below |
| NPU | HUAWEI Ascend 910B2 (64 GB HBM), CANN 9.0.0 |
| GPU software | PyTorch 2.x, bf16, Flash Linear Attention (FLA) `chunk_kda` kernel |
| NPU software | PyTorch + torch-npu, bf16; SDPA for Transformer/MLA layers; FLA `fused_recurrent_kda` for KDA layers |
| Precision | bf16 except NT-v2, which only supports fp32 |
| Measurement | Median of repeated forward passes after warm-up; peak device memory |

## 1. Architecture-level comparison

The comparison in this section is the basis of the reported **32–48% training**
and **54–79% inference** throughput gains. The denominator is the
**parameter-matched dense Transformer-470M** baseline (472M total parameters,
50 layers, hidden size 768). Transformer-220M (210M total parameters, 50 layers,
hidden size 512) is a smaller reference and is **not** the denominator.

All three models use single-nucleotide tokenization and a bidirectional masked
language modeling objective. Batch sizes were matched across models at each
sequence length.

### Training throughput (tokens s-1)

| Sequence length | Batch size | Transformer-220M | Transformer-470M | PlantHelixSeek-470M | Gain vs Transformer-470M |
|---:|---:|---:|---:|---:|---:|
| 512 | 16 | 52,778.6 | 29,540.1 | 38,897.1 | +31.7% |
| 1,024 | 8 | 45,062.5 | 25,680.0 | 35,357.3 | +37.7% |
| 2,048 | 4 | 33,667.1 | 20,126.5 | 29,866.7 | +48.4% |
| 4,096 | 1 | 19,799.1 | 12,561.1 | 17,628.2 | +40.3% |
| 8,192 | 1 | 13,087.4 | OOM | 14,730.9 | not defined (baseline OOM) |

### Inference throughput (tokens s-1)

| Sequence length | Batch size | Transformer-220M | Transformer-470M | PlantHelixSeek-470M | Gain vs Transformer-470M |
|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 212,616.6 | 111,628.9 | 172,337.0 | +54.4% |
| 1,024 | 32 | 166,002.5 | 92,469.3 | 149,213.0 | +61.4% |
| 2,048 | 32 | 122,597.3 | 73,376.0 | 121,300.0 | +65.3% |
| 4,096 | 8 | 85,607.1 | 52,648.2 | 94,080.2 | +78.7% |
| 8,192 | 1 | 48,169.4 | 31,748.8 | 54,070.9 | +70.3% |

### Peak memory (GiB)

| Task | Sequence length | Transformer-220M | Transformer-470M | PlantHelixSeek-470M |
|---|---|---:|---:|---:|
| Training | 512 | 3.923 | 8.829 | 8.214 |
| Training | 1,024 | 4.428 | 8.917 | 7.889 |
| Training | 2,048 | 6.301 | 11.521 | 9.595 |
| Training | 4,096 | 10.664 | 17.492 | 14.023 |
| Training | 8,192 | 21.827 | OOM | 24.484 |
| Inference | 512 | 1.199 | 2.748 | 2.562 |
| Inference | 1,024 | 1.214 | 2.806 | 2.545 |
| Inference | 2,048 | 1.249 | 2.835 | 2.583 |
| Inference | 4,096 | 1.343 | 2.949 | 2.648 |
| Inference | 8,192 | 1.637 | 3.256 | 2.969 |

The throughput gains correspond to a **24–33% training wall-clock reduction**
and a **35–44% inference wall-clock reduction**. PlantHelixSeek-470M is the only
470M-class model that completes 8,192-bp training within the 32-GB device budget.

## 2. Cross-model inference comparison

This section reports the six-model comparison shown in Supplemental Figure 3.
All models use their released checkpoints and a single-nucleotide tokenizer
except NT-v2 (6-mer) and DNAMamba2 (BPE). Batch sizes were chosen per sequence
length as the largest batch size that fits on the 32-GB device for every model.
NT-v2 is fp32 and does not support 4,096/8,192 tokens.

### Throughput (tokens s-1)

| Sequence length | Batch size | PlantHelixSeek | GENERanno | NT-v2 | PlantCAD2 | DNAMamba2 | PlantGFM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 215,139 | 125,698 | 35,954 | 48,343 | 105,013 | 189,487 |
| 1,024 | 16 | 203,731 | 119,349 | 30,855 | 47,996 | 105,583 | 186,804 |
| 2,048 | 8 | 183,730 | 109,050 | 24,052 | 47,876 | 105,816 | 184,164 |
| 4,096 | 4 | 152,882 | 91,845 | n/a | 48,073 | 105,991 | 180,511 |
| 8,192 | 2 | 115,475 | 70,249 | n/a | 47,912 | 106,110 | 174,951 |

### Latency (ms per forward pass)

| Sequence length | Batch size | PlantHelixSeek | GENERanno | NT-v2 | PlantCAD2 | DNAMamba2 | PlantGFM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 76.2 | 130.3 | 455.7 | 338.9 | 156.0 | 86.5 |
| 1,024 | 16 | 80.4 | 137.3 | 531.0 | 341.4 | 155.2 | 87.7 |
| 2,048 | 8 | 89.2 | 150.2 | 681.2 | 342.2 | 154.8 | 89.0 |
| 4,096 | 4 | 107.2 | 178.4 | n/a | 340.8 | 154.6 | 90.8 |
| 8,192 | 2 | 141.9 | 233.2 | n/a | 342.0 | 154.4 | 93.6 |

### Peak device memory (MB)

| Sequence length | Batch size | PlantHelixSeek | GENERanno | NT-v2 | PlantCAD2 | DNAMamba2 | PlantGFM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 1,234 | 1,439 | 3,497 | 2,103 | 1,446 | 1,397 |
| 1,024 | 16 | 1,232 | 1,440 | 4,528 | 2,095 | 1,430 | 1,402 |
| 2,048 | 8 | 1,231 | 1,440 | 6,591 | 2,091 | 1,422 | 1,411 |
| 4,096 | 4 | 1,231 | 1,441 | n/a | 2,089 | 1,418 | 1,431 |
| 8,192 | 2 | 1,230 | 1,443 | n/a | 2,088 | 1,417 | 1,471 |

At 8,192 tokens and batch size 2, PlantHelixSeek uses 1,230 MB (1.20 GiB),
the lowest peak memory among the six models.

## 3. GPU versus HUAWEI Ascend NPU

Forward-only inference of PlantHelixSeek-470M on identical weights
(Supplemental Table 14).

| Sequence length | Batch | RTX 5090 tokens s-1 | RTX 5090 GiB | Ascend 910B2 tokens s-1 | Ascend 910B2 GiB | NPU/GPU ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 14,248 | 0.9 | 3,305 | 1.0 | 0.232 |
| 512 | 2 | 27,713 | 0.9 | 6,619 | 1.0 | 0.239 |
| 512 | 4 | 63,054 | 0.9 | 12,873 | 1.0 | 0.204 |
| 1,024 | 1 | 28,189 | 0.9 | 6,762 | 1.0 | 0.240 |
| 1,024 | 2 | 54,198 | 0.9 | 13,065 | 1.0 | 0.241 |
| 1,024 | 4 | 81,996 | 1.0 | 23,466 | 1.1 | 0.286 |
| 2,048 | 1 | 48,836 | 0.9 | 12,859 | 1.0 | 0.263 |
| 2,048 | 2 | 73,689 | 1.0 | 23,384 | 1.1 | 0.317 |
| 2,048 | 4 | 97,748 | 1.1 | 32,084 | 1.2 | 0.328 |
| 4,096 | 1 | 61,694 | 1.0 | 18,160 | 1.2 | 0.294 |
| 4,096 | 2 | 77,915 | 1.1 | 20,989 | 1.3 | 0.269 |
| 4,096 | 4 | 89,829 | 1.3 | 27,614 | 1.6 | 0.307 |
| 8,192 | 1 | 54,566 | 1.2 | 18,081 | 1.4 | 0.331 |
| 8,192 | 2 | 60,538 | 1.5 | 23,165 | 1.9 | 0.383 |
| 8,192 | 4 | 63,369 | 2.1 | 26,424 | 2.7 | 0.417 |

Only PlantHelixSeek-470M was measured on the Ascend NPU. The lower NPU
throughput reflects the older accelerator generation of the Ascend 910B2;
downstream prediction accuracy is identical across platforms.

## 4. Model inclusion and exclusion

The efficiency comparison is an architecture-level comparison, not an
exhaustive comparison of all published plant DNA models. Models were included
if they have 225–493M parameters and can be executed on a 32-GB consumer GPU:

- PlantHelixSeek-470M
- GENERanno-493M
- NT-v2-493M (6-mer; fp32)
- PlantCAD2-Medium (311M published; the bidirectional checkpoint measured here
  contains 621M parameters in total across both strands)
- DNAMamba2-325M (BPE)
- PlantGFM-225M

AgroNT (1B, 6-mer, 1,024-token context) and Evo 2 (7B/40B, 1M context) were
excluded because they fall outside this parameter/context/hardware regime.
Evo 2 (1B) is included only in the fine-tuning benchmark, not in the efficiency
comparison.

## 5. Reproducibility notes

- Throughput is reported as tokens per second for a forward pass (or a training
  step) after warm-up; latency is the median wall-clock time per forward pass.
- Absolute throughput depends on the kernel implementation, batch size and
  driver stack. Comparisons should therefore be made within a table, not across
  tables measured with different batch sizes.
- The architecture-level table (Section 1) was measured with the SDPA attention
  path; the cross-model table (Section 2) uses the optimized FLA kernels. This
  explains the different peak-memory values for PlantHelixSeek at 8,192 tokens
  (2.97 GiB in Section 1 at batch size 1 versus 1.20 GiB in Section 2 at batch
  size 2).
- Raw benchmark records are in [`docs/benchmark_data/`](benchmark_data/):
  `architecture_speed.csv` and `architecture_memory.csv` for Section 1, and
  `cross_model_inference_summary.csv` and `cross_model_training_summary.csv`
  for Section 2.

## References

- Kimi Linear: https://arxiv.org/abs/2510.26692
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Switch Transformers: Fedus et al., JMLR 2022
- PlantCAD2: https://doi.org/10.1016/j.xgen.2026.101329
- GENERanno: https://doi.org/10.1101/2025.06.04.656
- PlantGFM: https://doi.org/10.1002/advs.75772
- NT-v2: https://doi.org/10.1038/s41592-025-02939-1
- AgroNT: https://doi.org/10.1038/s42003-024-06465-2
- Evo 2: https://doi.org/10.1038/s41586-026-10176-5
