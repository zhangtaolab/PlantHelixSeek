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

You should see ✓ for all checks.
