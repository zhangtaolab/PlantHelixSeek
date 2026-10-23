# Installation

## 1. Clone the repository

```bash
git clone https://github.com/zhangtaolab/PlantHelixSeek.git
cd PlantHelixSeek
```

## 2. Create a virtual environment (recommended: [uv](https://docs.astral.sh/uv/))

```bash
# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv --python 3.12 .venv
source .venv/bin/activate
```

## 3. Install PyTorch (choose your platform)

### NVIDIA GPU (CUDA 12.x)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```


### CPU only

```bash
pip install torch
```

### Apple Silicon (macOS, MPS)

```bash
pip install "torch>=2.12"   # 2.12.0 / 2.13.0 / 2.14.1 verified on macOS arm64
```

macOS has no Triton wheels, so the FLA kernels used by the KDA layers are
unavailable and the model runs a pure PyTorch fallback. Treat MPS as a
validation/debug target, not a throughput target. Two caveats:

- **torch <= 2.11**: MPS SDPA returns a query-shaped output when the value head
  dim is smaller than the query head dim, which crashes the MLA layers. Either
  upgrade to torch>=2.12 or patch a downloaded checkpoint:

  ```bash
  python scripts/huggingface/apply_mps_fix.py --model_path models/PlantHelixSeek
  ```

- **Accuracy**: the shipped pure-PyTorch KDA fallback does not match FLA
  semantics (mask recovery drops to ~chance). Pass `--kda_fallback` to also
  apply the exact recurrence:

  ```bash
  python scripts/huggingface/apply_mps_fix.py --model_path models/PlantHelixSeek --kda_fallback
  ```

Both patches are idempotent, keep the original as `attention.py.orig`, and are
scoped to the code path they fix, so CUDA/NPU runs are unchanged. Then pass
`--device mps` to any script.

## 4. Install PlantHelixSeek dependencies

```bash
pip install -r requirements.txt
# or editable install:
pip install -e .
```

### Optional: Flash Attention (NVIDIA GPU only)

```bash
pip install flash-attn>=2.0.0 flash-linear-attention==0.4.1
```


## 5. Download model weights

```bash
# Option A: huggingface-cli
huggingface-cli download zhangtaolab/PlantHelixSeek --local-dir ./models/PlantHelixSeek

# Option B: git lfs
git lfs install
git clone https://huggingface.co/zhangtaolab/PlantHelixSeek ./models/PlantHelixSeek
```

> Users in China can set the mirror: `export HF_ENDPOINT=https://hf-mirror.com`

## 6. Verify installation

```bash
python scripts/huggingface/verify_install.py --model_path models/PlantHelixSeek --device auto
```

On Apple Silicon use `--device mps`. You should see ✓ for all checks.
