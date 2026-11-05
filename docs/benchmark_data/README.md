# Benchmark data

Raw records for the tables in [docs/benchmarks.md](../benchmarks.md).

| File | Contents |
|---|---|
| `architecture_speed.csv` | Training and inference throughput (tokens s-1) for Transformer-220M, Transformer-470M and PlantHelixSeek-470M at 512-8,192 tokens. |
| `architecture_memory.csv` | Peak training and inference memory (GiB) for the same three models. |
| `cross_model_inference_summary.csv` | Inference latency, throughput and peak memory for the six-model comparison, across sequence lengths and batch sizes. |
| `cross_model_training_summary.csv` | Training latency, throughput and peak memory for the same models. |

## Columns

`architecture_speed.csv`: `task` (train or inference), `seq_len`, `batch_size`, `T-MLM-220M`, `T-MLM-470M`, `PlantHelixSeek-SDPA` (tokens s-1).

`architecture_memory.csv`: same keys, with peak memory in GiB.

`cross_model_*_summary.csv`: `model`, `short`, `arch`, `tokenizer`, `dtype`, `num_params_m`, `token_len`, `batch_size`, `median_latency_ms`, `median_tokens_per_sec`, `peak_vram_mb`, `static_vram_mb`, `n_ok`.

## Protocol

Forward-only inference or training-step measurements after warm-up on an NVIDIA RTX 5090D (32 GB) with bf16, except NT-v2 (fp32). Batch sizes are matched within each table; absolute throughput depends on kernel and batch settings, so comparisons should be made within a table. The architecture-level table uses the SDPA attention path; the six-model table uses the optimized FLA kernels.

## Provenance

`architecture_*` files are the records behind Supplemental Figure 2. `cross_model_*` files are the records behind Supplemental Figure 3. The GPU/NPU comparison is in Supplemental Table 14 and in [docs/benchmarks.md](../benchmarks.md).
