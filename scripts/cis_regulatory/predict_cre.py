"""Genome-wide cis-regulatory element (CRE) prediction with BigWig export.

Uses a binary fine-tuned PlantHelixSeek checkpoint to scan a genome with
sliding windows, predict per-bin CRE / open-chromatin probability, and emit a
standard BigWig track.

Usage:
    # Predict with the default PlantHelixSeek-CRE checkpoint
    python predict_cre.py

    # Specify model and genome explicitly
    python predict_cre.py \\
        --model_path models/PlantHelixSeek-CRE \\
        --genome path/to/genome.fa \\
        --output_dir tmp/cre_predict

    # Enable reverse-complement TTA (slower, slightly higher accuracy)
    python predict_cre.py --tta
"""

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyBigWig
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 500
STRIDE = 50
BIN_SIZE = 50

COMPLEMENT = str.maketrans("ATCGNatcgn", "TAGCNtagcn")

DEFAULT_MODEL_PATH = Path("models/PlantHelixSeek-CRE")
DEFAULT_GENOME = Path("data/cis_regulatory/test_sequence.fa")
DEFAULT_OUTPUT_DIR = Path("tmp/cre_predict")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def load_genome(fasta_path: Path) -> dict[str, str]:
    """Read a FASTA file into ``{chrom: uppercase_sequence}``."""
    genome: dict[str, str] = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        genome[rec.id] = str(rec.seq).upper()
    return genome


def _species_from_fasta(fasta_path: Path) -> str:
    """Derive species name from FASTA filename (strip ``_test`` suffix)."""
    stem = fasta_path.stem
    return re.sub(r"_test$", "", stem)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Token-ID dataset for batched DataLoader inference."""

    def __init__(self, input_ids_list: list[list[int]]) -> None:
        self.input_ids = input_ids_list

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.input_ids[idx], dtype=torch.long)


def _pad_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.size(0) for t in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(batch):
        padded[i, : t.size(0)] = t
    return padded


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------

def predict_chromosome(
    model: AutoModelForSequenceClassification,
    seq: str,
    chrom_name: str,
    device: torch.device,
    tokenizer: AutoTokenizer,
    stride: int,
    batch_size: int,
    tta: bool = False,
    num_workers: int = 0,
) -> np.ndarray:
    """Sliding-window prediction on one chromosome → BIN_SIZE-grained scores."""
    chrom_len = len(seq)
    n_bins = chrom_len // BIN_SIZE

    if n_bins == 0:
        logger.warning(
            "  %s: length %d < BIN_SIZE=%d, skipping", chrom_name, chrom_len, BIN_SIZE,
        )
        return np.array([], dtype=np.float32)

    max_pos = max(chrom_len - WINDOW_SIZE, 0)
    starts = list(range(0, max_pos + 1, stride))
    n_windows = len(starts)

    if n_windows == 0:
        logger.warning("  %s: no windows (len=%d)", chrom_name, chrom_len)
        return np.zeros(n_bins, dtype=np.float32)

    logger.info(
        "  %s: %d bp, %d windows (stride=%d)", chrom_name, chrom_len, n_windows, stride,
    )

    t0 = time.time()

    fwd_ids_list: list[list[int]] = []
    rc_ids_list: list[list[int]] = []
    bin_ranges: list[tuple[int, int]] = []

    for start in starts:
        end = start + WINDOW_SIZE
        subseq = seq[start:end]
        if len(subseq) < WINDOW_SIZE:
            subseq = subseq + "N" * (WINDOW_SIZE - len(subseq))

        enc_fwd = tokenizer(subseq, add_special_tokens=True, padding=False, truncation=False)
        fwd_ids_list.append(enc_fwd["input_ids"])
        if tta:
            rc_seq = reverse_complement(subseq)
            enc_rc = tokenizer(rc_seq, add_special_tokens=True, padding=False, truncation=False)
            rc_ids_list.append(enc_rc["input_ids"])

        bin_start = start // BIN_SIZE
        bin_end = min((end + BIN_SIZE - 1) // BIN_SIZE, n_bins)
        bin_ranges.append((bin_start, bin_end))

    fwd_probs = _run_dataloader(
        model, fwd_ids_list, device, batch_size, num_workers,
        desc=f"{chrom_name} fwd",
    )

    if tta and rc_ids_list:
        rc_probs = _run_dataloader(
            model, rc_ids_list, device, batch_size, num_workers,
            desc=f"{chrom_name} rc",
        )
        fwd_probs = (fwd_probs + rc_probs) / 2.0

    score_sum = np.zeros(n_bins, dtype=np.float64)
    coverage = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_windows):
        b_start, b_end = bin_ranges[i]
        score_sum[b_start:b_end] += fwd_probs[i]
        coverage[b_start:b_end] += 1.0

    valid = coverage > 0
    scores = np.zeros(n_bins, dtype=np.float32)
    scores[valid] = (score_sum[valid] / coverage[valid]).astype(np.float32)

    elapsed = time.time() - t0
    logger.info(
        "  %s: done in %.1fs | %.1f%% bins covered",
        chrom_name, elapsed, 100.0 * valid.sum() / n_bins,
    )
    return scores


def _run_dataloader(
    model: AutoModelForSequenceClassification,
    input_ids_list: list[list[int]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str = "Predicting",
) -> np.ndarray:
    """Batched inference via DataLoader → class-1 probability array."""
    dataset = WindowDataset(input_ids_list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_pad_collate_fn,
    )
    all_probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(input_ids=batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# BigWig output
# ---------------------------------------------------------------------------

def write_bigwig(
    output_path: Path,
    genome: dict[str, str],
    predictions: dict[str, np.ndarray],
) -> None:
    """Write predictions to a BigWig file."""
    chrom_sizes = [(chrom, len(genome[chrom])) for chrom in genome]
    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(chrom_sizes, maxZooms=10)

    for chrom in genome:
        if chrom not in predictions:
            continue
        pred = predictions[chrom]
        if len(pred) == 0:
            continue
        scores = pred.astype(np.float64)
        bw.addEntries(
            chrom, 0,
            values=scores.tolist(),
            span=BIN_SIZE,
            step=BIN_SIZE,
        )

    bw.close()
    logger.info("Wrote BigWig: %s", output_path)


# ---------------------------------------------------------------------------
# Checkpoint resume: each chromosome is cached on completion
# ---------------------------------------------------------------------------

def _chrom_cache_path(output_dir: Path, chrom: str) -> Path:
    return output_dir / "chunks" / f"{chrom}.npy"


def _load_cached_chroms(output_dir: Path) -> dict[str, np.ndarray]:
    """Load cached per-chromosome predictions from ``chunks/``."""
    cached: dict[str, np.ndarray] = {}
    chunks_dir = output_dir / "chunks"
    if not chunks_dir.is_dir():
        return cached
    for npy_file in sorted(chunks_dir.glob("*.npy")):
        chrom = npy_file.stem
        cached[chrom] = np.load(str(npy_file), allow_pickle=True)
        logger.info("  Loaded cached: %s (%d bins)", chrom, len(cached[chrom]))
    return cached


# ---------------------------------------------------------------------------
# Single-model prediction pipeline
# ---------------------------------------------------------------------------

def _prepare_remote_code(model_path: Path) -> None:
    """Pre-copy every *.py from model_path into the HF transformers_modules cache.

    The HelixSeek ``model.py → layers.py → mlp.py`` import chain must be fully
    present before HF's dynamic-module loader scans it, otherwise ``mlp.py``
    is opened from the cache before it has been copied there.
    """
    from transformers.utils import HF_MODULES_CACHE

    cache_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / model_path.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "__init__.py").touch()
    pycache = cache_dir / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    for py_file in model_path.glob("*.py"):
        shutil.copy2(py_file, cache_dir / py_file.name)
    import importlib
    importlib.invalidate_caches()


def run_model(
    model_path: Path,
    genome: dict[str, str],
    chroms: list[str],
    args: argparse.Namespace,
) -> None:
    """Load model, scan chromosomes, emit BigWig + .npy.

    Per-chromosome results are cached to ``chunks/<chrom>.npy`` so interrupted
    runs resume by skipping finished chromosomes.
    """
    output_name = model_path.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_cached_chroms(output_dir)
    skip_count = sum(1 for c in chroms if c in predictions)
    if skip_count > 0:
        logger.info("Resuming: %d/%d chromosomes already cached, skipping", skip_count, len(chroms))

    if skip_count == len(chroms):
        logger.info("All chromosomes already cached, proceeding to merge.")
    else:
        device = torch.device(args.device) if args.device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info("=" * 60)
        logger.info("Model: %s | Device: %s | Path: %s", output_name, device, model_path)

        with open(model_path / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        logger.info("  model_type=%s  vocab_size=%s", config.get("model_type"), config.get("vocab_size"))

        _prepare_remote_code(model_path)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        logger.info("Loading model …")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        model.to(device)
        logger.info("Model loaded: %s", type(model).__name__)

        total_t0 = time.time()

        for chrom in tqdm(chroms, desc="Chromosomes", unit="chrom"):
            if chrom in predictions:
                continue

            chrom_seq = genome[chrom]
            if len(chrom_seq) < WINDOW_SIZE:
                logger.warning(
                    "  %s: %d bp < WINDOW_SIZE=%d, skipping",
                    chrom, len(chrom_seq), WINDOW_SIZE,
                )
                scores = np.array([], dtype=np.float32)
            else:
                scores = predict_chromosome(
                    model, chrom_seq, chrom,
                    device=device,
                    tokenizer=tokenizer,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    tta=args.tta,
                    num_workers=args.num_workers,
                )

            predictions[chrom] = scores
            np.save(str(_chrom_cache_path(output_dir, chrom)), scores)
            logger.info("  Saved chunk: %s", _chrom_cache_path(output_dir, chrom))
            if device.type == "cuda":
                torch.cuda.empty_cache()

        total_elapsed = time.time() - total_t0
        logger.info(
            "Prediction loop finished in %.1fs (%.1f min)",
            total_elapsed, total_elapsed / 60,
        )

    species = _species_from_fasta(Path(args.genome))

    bw_path = output_dir / f"{output_name}_{species}_OCR.bw"
    write_bigwig(bw_path, genome, predictions)

    npy_path = output_dir / f"{output_name}_{species}_OCR.npy"
    np.save(str(npy_path), predictions, allow_pickle=True)
    logger.info("Saved .npy: %s", npy_path)

    logger.info("Model %s complete. Output dir: %s", output_name, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genome-wide CRE prediction with PlantHelixSeek-CRE",
    )
    parser.add_argument(
        "--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to fine-tuned PlantHelixSeek-CRE checkpoint",
    )
    parser.add_argument(
        "--genome", type=str, default=str(DEFAULT_GENOME),
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--stride", type=int, default=STRIDE,
        help=f"Sliding window stride in bp (default: {STRIDE})",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable reverse-complement test-time augmentation (slower, slightly higher accuracy)",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device override, e.g. cuda:0, cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader num_workers (default: 0)",
    )
    parser.add_argument(
        "--min_chrom_size", type=int, default=0,
        help="Skip chromosomes smaller than this (bp); 0 = no filter",
    )
    parser.add_argument(
        "--max_chrom_size", type=int, default=0,
        help="Skip chromosomes larger than this (bp); 0 = no filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not (model_path / "config.json").exists():
        logger.error("Model config not found at %s", model_path)
        sys.exit(1)
    logger.info("Model: %s", model_path)

    genome_path = Path(args.genome)
    logger.info("Loading genome: %s", genome_path)
    genome = load_genome(genome_path)
    if not genome:
        logger.error("Genome is empty or file not found: %s", genome_path)
        sys.exit(1)

    chroms = sorted(
        genome.keys(),
        key=lambda c: (
            0,
            int(re.sub(r"^chr", "", c)),
        ) if re.sub(r"^chr", "", c).isdigit() else (1, c),
    )

    if args.min_chrom_size > 0:
        skipped = [c for c in chroms if len(genome[c]) < args.min_chrom_size]
        chroms = [c for c in chroms if len(genome[c]) >= args.min_chrom_size]
        if skipped:
            logger.info(
                "Skipping %d chromosome(s) < %d bp: %s",
                len(skipped), args.min_chrom_size, skipped,
            )

    if args.max_chrom_size > 0:
        skipped = [c for c in chroms if len(genome[c]) > args.max_chrom_size]
        chroms = [c for c in chroms if len(genome[c]) <= args.max_chrom_size]
        if skipped:
            logger.info(
                "Skipping %d chromosome(s) > %d bp: %s",
                len(skipped), args.max_chrom_size, skipped,
            )

    logger.info("Chromosomes (%d): %s", len(chroms), chroms)

    run_model(model_path, genome, chroms, args)
    logger.info("All done.")


if __name__ == "__main__":
    main()
