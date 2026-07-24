"""Multi-GPU sliding-window genome gene structure prediction with middle-region extraction.

Modified from predict_genome.py to support torchrun multi-GPU inference.
Distributes chromosomes across GPUs for each species, with barrier synchronization
to handle the "some GPUs finish early" problem at the end of each species.

Key design:
  - Each rank (GPU) loads its own copy of the model.
  - For each species, chromosomes are distributed round-robin across ranks.
  - After all ranks finish their chromosomes for a species, dist.barrier() ensures
    synchronization before moving to the next species.
  - Outputs per-chromosome NPZ files, organized into per-species subdirectories.

Usage:
    source ~/DNALLM/.venv/bin/activate

    torchrun --nproc_per_node=7 predict_genome_multigpu.py \
        --species-dir LLM_anno/result/dataset/test_genome \
        --model-path LLM_anno/models/finetuned_model/PlantHelixSeek_v2 \
        --output-dir result/figure4.../data/PlantHelixSeek_v2 \
        --use-crf \
        --batch-size 4 \
        --window-size 8192 \
        --stride 4096

Can also run single-file mode (backward compatible with original):
    python predict_genome_multigpu.py \
        --genome-fasta path/to/genome.fa \
        --model-path model_dir \
        --output-dir output/ \
        --use-crf
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from Bio import SeqIO
from transformers import AutoModelForTokenClassification, AutoTokenizer

# NPU support (Huawei Ascend). Optional: torch_npu registers the `npu` device
# and the `hccl` distributed backend. Guard with try/except so the script still
# imports/runs on CUDA-only or CPU-only machines.
try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from train_token_cls import GeneStructureCRF, _patch_attention_to_sdpa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Rank %(rank)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Patch the logger to include rank info
_old_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    record.rank = getattr(_record_factory, "_rank", 0)
    return record


logging.setLogRecordFactory(_record_factory)

logger = logging.getLogger(__name__)

WINDOW_SIZE_DEFAULT = 8192
WINDOW_STEP_DEFAULT = 4096
NUM_LABELS = 17
MODEL_INPUT_LENGTH = WINDOW_SIZE_DEFAULT + 2
COMPLEMENT_MAP = str.maketrans("ACGTacgt", "TGCAtgca")

CDS_IDX = [1, 2, 3, 4]
INTRON_IDX = [5, 6, 7, 8]
UTR5_IDX = [9, 10, 11, 12]
UTR3_IDX = [13, 14, 15, 16]

_B_SWAP_L = np.array(
    [0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13, 16],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed process group for torchrun.

    Returns:
        (rank, world_size, device)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            backend = "nccl"
        elif _HAS_TORCH_NPU and torch.npu.is_available():
            backend = "hccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=12))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        elif _HAS_TORCH_NPU and torch.npu.is_available():
            torch.npu.set_device(local_rank)
            device = torch.device(f"npu:{local_rank}")
        else:
            device = torch.device("cpu")
        logger.info(
            "Distributed initialized: rank=%d/%d, local_rank=%d, device=%s, backend=%s",
            rank, world_size, local_rank, device, backend,
        )
    else:
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif _HAS_TORCH_NPU and torch.npu.is_available():
            device = torch.device("npu:0")
        else:
            device = torch.device("cpu")
        logger.info("Single-process mode: device=%s", device)

    _record_factory._rank = rank
    return rank, world_size, device


def cleanup_distributed(world_size: int) -> None:
    """Destroy process group and sync all ranks."""
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Core logic (identical to original predict_genome.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowPrediction:
    probabilities: np.ndarray
    decoded_labels: np.ndarray | None = None


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT_MAP)[::-1]


def read_fasta(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    with opener(fasta_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq).upper()
    return sequences


def create_index_windows(
    sequence_length: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Create sliding window boundaries with valid middle regions.

    Returns (N, 4) array with columns:
        starts, stops, v_starts, v_stops
    where v_starts/v_stops define the non-overlapping valid region
    within each window that collectively covers the full sequence.

    Margin = (window_size - stride) / 2 is dropped from each side.
    """
    if window_size >= sequence_length:
        return np.array([[0, sequence_length, 0, sequence_length]])

    margin = int((window_size - stride) / 2)

    starts = np.arange(0, sequence_length - window_size + 1, stride)
    stops = starts + window_size

    v_starts = np.maximum(starts, starts + margin * (starts > 0))
    v_stops = np.minimum(stops, stops - margin * (stops < sequence_length))

    return np.column_stack((starts, stops, v_starts, v_stops))


def aggregate_probs_to_features(probs_17: np.ndarray) -> np.ndarray:
    """Convert 17-class BILUO probs to 5-class feature probs (GeneCAD order).

    BILUO: 0=intergenic, 1-4=CDS, 5-8=intron, 9-12=5UTR, 13-16=3UTR
    Feature order: [intergenic, intron, 5'UTR, CDS, 3'UTR]
    """
    result = np.zeros(probs_17.shape[:-1] + (5,), dtype=probs_17.dtype)
    result[..., 0] = probs_17[..., 0]                     # intergenic
    result[..., 1] = probs_17[..., 5:9].sum(axis=-1)      # intron (BILUO 5-8)
    result[..., 2] = probs_17[..., 9:13].sum(axis=-1)     # 5'UTR (BILUO 9-12)
    result[..., 3] = probs_17[..., 1:5].sum(axis=-1)      # CDS (BILUO 1-4)
    result[..., 4] = probs_17[..., 13:17].sum(axis=-1)    # 3'UTR (BILUO 13-16)
    return result


def load_model_and_tokenizer(model_path: Path, device: torch.device) -> tuple:
    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    _patch_attention_to_sdpa(model)
    model.to(device)
    model.eval()
    logger.info(
        "Model loaded. Vocab size: %d, Num labels: %d",
        tokenizer.vocab_size,
        model.config.num_labels,
    )
    return model, tokenizer


@torch.no_grad()
def viterbi_batched_decode(
    crf: GeneStructureCRF, emissions: torch.Tensor, mask: torch.Tensor
) -> list[list[int]]:
    trans = crf._constrained_transitions()
    B, T, K = emissions.shape
    device = emissions.device

    dp = crf.start_transitions.unsqueeze(0) + emissions[:, 0]
    backpointers = torch.zeros(B, T, K, dtype=torch.long, device=device)
    for t in range(1, T):
        scores = dp.unsqueeze(-1) + trans.unsqueeze(0)  # (B, K_from, K_to): dp on from-axis
        red = scores.max(dim=1)  # reduce over K_from
        dp = torch.where(
            mask[:, t].unsqueeze(1), red.values + emissions[:, t], dp
        )
        backpointers[:, t] = red.indices

    dp = dp + crf.end_transitions.unsqueeze(0)
    best_last = torch.argmax(dp, dim=1)
    valid_lens = mask.sum(dim=1).tolist()
    bp_cpu = backpointers.cpu().numpy()
    best_last_cpu = best_last.cpu().tolist()

    results: list[list[int]] = []
    for b in range(B):
        vl = valid_lens[b]
        if vl == 0:
            results.append([])
            continue
        bp = bp_cpu[b]
        path = [0] * vl
        cur = int(best_last_cpu[b])
        path[-1] = cur
        for t in range(vl - 2, -1, -1):
            cur = int(bp[t + 1, cur])
            path[t] = cur
        results.append(path)
    return results


@torch.no_grad()
def predict_windows(
    model,
    tokenizer,
    seqs: list[str],
    device: torch.device,
    crf: GeneStructureCRF | None = None,
    genic_bias: float = 0.0,
    model_input_length: int = MODEL_INPUT_LENGTH,
    window_size: int = WINDOW_SIZE_DEFAULT,
) -> WindowPrediction:
    encoded = tokenizer(
        seqs,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=model_input_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    logits = logits[:, 1 : window_size + 1, :]

    if genic_bias > 0.0:
        logits[:, :, 0] = logits[:, :, 0] - genic_bias

    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    if crf is not None:
        crf_mask = torch.ones(
            logits.shape[0], window_size, dtype=torch.bool, device=device
        )
        decoded_paths = viterbi_batched_decode(crf, logits, crf_mask)
        decoded_labels = np.array(decoded_paths, dtype=np.int32)
        return WindowPrediction(probabilities=probs, decoded_labels=decoded_labels)

    return WindowPrediction(probabilities=probs)


def generate_windows(
    chrom_seq: str, window_size: int = WINDOW_SIZE_DEFAULT
) -> list[tuple[int, str, str]]:
    seq_len = len(chrom_seq)
    windows: list[tuple[int, str, str]] = []
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        subseq = chrom_seq[start:end]
        if len(subseq) < window_size:
            subseq = subseq + "N" * (window_size - len(subseq))
        rc_seq = reverse_complement(subseq)
        windows.append((start, subseq, rc_seq))
        start += WINDOW_STEP_DEFAULT
        if end == seq_len:
            break
    return windows


def process_chromosome(
    chrom_name: str,
    chrom_seq: str,
    model,
    tokenizer,
    batch_size: int,
    device: torch.device,
    crf: GeneStructureCRF | None = None,
    genic_bias: float = 0.0,
    window_size: int = WINDOW_SIZE_DEFAULT,
    stride: int = WINDOW_STEP_DEFAULT,
    model_input_length: int = MODEL_INPUT_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single chromosome with middle-region window extraction."""
    chrom_len = len(chrom_seq)
    logger.info("  Chromosome %s: %d bp", chrom_name, chrom_len)

    fwd_probs = np.zeros((chrom_len, NUM_LABELS), dtype=np.float32)
    rev_probs = np.zeros((chrom_len, NUM_LABELS), dtype=np.float32)
    fwd_preds = np.zeros(chrom_len, dtype=np.int32)
    rev_preds = np.zeros(chrom_len, dtype=np.int32)
    fwd_covered = np.zeros(chrom_len, dtype=bool)
    rev_covered = np.zeros(chrom_len, dtype=bool)

    windows = generate_windows(chrom_seq, window_size)

    index_windows = create_index_windows(chrom_len, window_size, stride)

    for strand_label, probs_out, preds_out, covered_out in [
        ("+", fwd_probs, fwd_preds, fwd_covered),
        ("-", rev_probs, rev_preds, rev_covered),
    ]:
        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start : batch_start + batch_size]
            seqs = [w[1] if strand_label == "+" else w[2] for w in batch_windows]
            starts = [w[0] for w in batch_windows]

            window_prediction = predict_windows(
                model, tokenizer, seqs, device,
                crf=crf, genic_bias=genic_bias,
                model_input_length=model_input_length,
                window_size=window_size,
            )
            batch_probs = window_prediction.probabilities
            batch_labels = window_prediction.decoded_labels

            for j, win_start in enumerate(starts):
                win_idx = None
                for k, (ws, we, vs, ve) in enumerate(index_windows):
                    if ws == win_start:
                        win_idx = k
                        break

                if win_idx is not None:
                    ws, we, vs, ve = index_windows[win_idx]
                    local_start = vs - ws
                    local_end = ve - ws
                else:
                    actual_win_size = min(window_size, chrom_len - win_start)
                    vs, ve = win_start, win_start + actual_win_size
                    local_start, local_end = 0, actual_win_size

                actual_len = local_end - local_start
                if actual_len <= 0:
                    continue

                if not covered_out[vs:ve].all():
                    if strand_label == "+":
                        probs_out[vs:ve] = batch_probs[j, local_start:local_end, :]
                    else:
                        probs_out[vs:ve] = batch_probs[j, local_start:local_end, :][::-1, :]

                    if batch_labels is not None:
                        if strand_label == "+":
                            preds_out[vs:ve] = batch_labels[j, local_start:local_end]
                        else:
                            preds_out[vs:ve] = batch_labels[j, local_start:local_end][::-1]

                    covered_out[vs:ve] = True

            batch_idx = batch_start // batch_size
            if batch_idx % 20 == 0:
                logger.info(
                    "    %s strand: processed %d/%d windows",
                    strand_label,
                    min(batch_start + batch_size, len(windows)),
                    len(windows),
                )

    if crf is not None:
        rev_probs = rev_probs[:, _B_SWAP_L]
        rev_preds = _B_SWAP_L[rev_preds]

    if not crf:
        fwd_preds = np.argmax(fwd_probs, axis=1).astype(np.int32)
        rev_preds = np.argmax(rev_probs, axis=1).astype(np.int32)

    fwd_feature_probs = aggregate_probs_to_features(fwd_probs)
    rev_feature_probs = aggregate_probs_to_features(rev_probs)

    return (
        fwd_preds, fwd_probs, rev_preds, rev_probs,
        fwd_feature_probs, rev_feature_probs,
    )


# ---------------------------------------------------------------------------
# Species discovery
# ---------------------------------------------------------------------------


def find_fasta(species_dir: Path) -> Path | None:
    """Find the FASTA file in a species directory.

    Supports .fna, .fa, .fasta extensions (optionally .gz compressed).
    """
    for pattern in ["*.fna", "*.fa", "*.fasta", "*.fna.gz", "*.fa.gz"]:
        files = list(species_dir.glob(pattern))
        if files:
            return files[0]
    return None


def discover_species(species_dir: Path) -> list[tuple[str, Path]]:
    """Discover all species directories with FASTA files.

    Returns list of (species_name, fasta_path) tuples, sorted by species name.
    """
    species = []
    for entry in sorted(species_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        fasta = find_fasta(entry)
        if fasta is not None:
            species.append((entry.name, fasta))
        else:
            logger.warning("No FASTA found for species: %s, skipping", entry.name)
    return species


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU sliding-window genome inference (torchrun compatible)."
    )
    # Multi-species mode (primary)
    parser.add_argument(
        "--species-dir", type=Path, default=None,
        help="Directory containing per-species subdirectories with FASTA files.",
    )
    # Single-file mode (backward compatible)
    parser.add_argument("--genome-fasta", type=Path, default=None,
                        help="Single genome FASTA file (alternative to --species-dir).")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (ignored in torchrun mode).")
    parser.add_argument("--crf-path", type=Path, default=None)
    parser.add_argument("--use-crf", action="store_true", default=False)
    parser.add_argument("--genic-bias", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE_DEFAULT)
    parser.add_argument("--stride", type=int, default=WINDOW_STEP_DEFAULT)
    parser.add_argument("--chromosome", type=str, default=None,
                        help="Only process this chromosome (by FASTA header).")
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip chromosomes whose NPZ output already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()

    model_input_length = args.window_size + 2

    # Override device if specified (single-process mode only)
    if args.device is not None and world_size == 1:
        device = torch.device(args.device)
        logger.info("Using overridden device: %s", device)

    # Load model on this GPU
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    # Setup CRF
    crf: GeneStructureCRF | None = None
    if args.crf_path is not None:
        crf = GeneStructureCRF(num_labels=NUM_LABELS)
        crf.load_state_dict(
            torch.load(args.crf_path, map_location="cpu", weights_only=True)
        )
        crf.to(device)
        crf.eval()
        logger.info("CRF loaded from %s", args.crf_path)
    elif args.use_crf:
        crf = GeneStructureCRF(num_labels=NUM_LABELS)
        crf.to(device)
        crf.eval()
        logger.info("CRF Viterbi decoding enabled (default constraints)")

    # ------------------------------------------------------------------
    # Build work items: list of (species_name, fasta_path) or (None, fasta_path)
    # ------------------------------------------------------------------
    work_items: list[tuple[str | None, Path]] = []

    if args.species_dir is not None:
        species_list = discover_species(args.species_dir)
        if rank == 0:
            logger.info("Discovered %d species in %s", len(species_list), args.species_dir)
            for name, fasta in species_list:
                logger.info("  %s -> %s", name, fasta.name)
        for name, fasta in species_list:
            work_items.append((name, fasta))
    elif args.genome_fasta is not None:
        work_items.append((None, args.genome_fasta))
    else:
        logger.error("Must specify either --species-dir or --genome-fasta")
        sys.exit(1)

    if not work_items:
        logger.error("No work items found. Check your input paths.")
        sys.exit(1)

    # Process each species / file
    total_start = time.time()

    for item_idx, (species_name, fasta_path) in enumerate(work_items, 1):
        species_start = time.time()

        if rank == 0:
            logger.info(
                "=" * 60
            )
            logger.info(
                "[Work item %d/%d] %s",
                item_idx, len(work_items),
                species_name or fasta_path.stem,
            )

        # Read FASTA (all ranks read independently — no shared storage needed)
        sequences = read_fasta(fasta_path)
        chrom_names = sorted(sequences.keys())

        # Filter to specific chromosome if requested
        if args.chromosome:
            if args.chromosome in sequences:
                chrom_names = [args.chromosome]
            else:
                matching = [n for n in sequences if args.chromosome in n]
                chrom_names = matching[:1] if matching else sorted(sequences.keys())[:1]

        # ------------------------------------------------------------------
        # Distribute chromosomes across ranks (round-robin)
        # ------------------------------------------------------------------
        my_chroms = chrom_names[rank::world_size]

        if rank == 0:
            logger.info(
                "  Species: %s, %d chromosomes, distributing across %d GPUs (round-robin)",
                species_name or fasta_path.stem, len(chrom_names), world_size,
            )
            for r in range(world_size):
                r_chroms = chrom_names[r::world_size]
                logger.info(
                    "    Rank %d: %d chromosomes %s",
                    r, len(r_chroms),
                    r_chroms[:5] if len(r_chroms) <= 5 else f"({r_chroms[0]}, ...)",
                )

        # Create output directory
        if species_name is not None:
            species_output = args.output_dir / species_name
        else:
            species_output = args.output_dir

        if rank == 0:
            species_output.mkdir(parents=True, exist_ok=True)

        # Ensure directory exists on all ranks before writing
        if world_size > 1:
            dist.barrier()

        # Each rank processes its assigned chromosomes
        total_bp = 0
        total_gene_bp = 0

        for chrom_idx, chrom_name in enumerate(my_chroms, 1):
            chrom_seq = sequences[chrom_name]
            chrom_len = len(chrom_seq)

            # Skip if output already exists
            out_path = species_output / f"{chrom_name}.npz"
            if args.skip_existing and out_path.exists():
                logger.info(
                    "  [%d/%d] Skipping %s (already exists)",
                    chrom_idx, len(my_chroms), out_path.name,
                )
                continue

            logger.info(
                "  [%d/%d] Processing %s (%d bp)...",
                chrom_idx, len(my_chroms), chrom_name, chrom_len,
            )

            (
                fwd_preds, fwd_probs, rev_preds, rev_probs,
                fwd_feature_probs, rev_feature_probs,
            ) = process_chromosome(
                chrom_name, chrom_seq, model, tokenizer,
                args.batch_size, device,
                crf=crf,
                genic_bias=args.genic_bias,
                window_size=args.window_size,
                stride=args.stride,
                model_input_length=model_input_length,
            )

            np.savez_compressed(
                out_path,
                fwd_predictions=fwd_preds,
                fwd_probabilities=fwd_probs,
                rev_predictions=rev_preds,
                rev_probabilities=rev_probs,
                fwd_feature_probs=fwd_feature_probs,
                rev_feature_probs=rev_feature_probs,
            )

            fwd_gene_bp = int(np.sum(fwd_preds != 0))
            rev_gene_bp = int(np.sum(rev_preds != 0))
            total_bp += chrom_len
            total_gene_bp += fwd_gene_bp + rev_gene_bp
            logger.info(
                "  Saved %s: + strand %d bp genic, - strand %d bp genic / %d total bp",
                out_path.name, fwd_gene_bp, rev_gene_bp, chrom_len,
            )

        # ------------------------------------------------------------------
        # Barrier: wait for ALL ranks to finish this species before proceeding.
        # This handles the "some GPUs finish early" problem — faster GPUs
        # block here until the slowest GPU completes its chromosomes.
        # ------------------------------------------------------------------
        if world_size > 1:
            logger.info(
                "  Rank %d finished %s chromosomes, waiting at barrier...",
                rank, species_name or fasta_path.stem,
            )
            dist.barrier()

        species_elapsed = time.time() - species_start
        if rank == 0:
            logger.info(
                "  Species %s completed in %.1fs (%.1f min)",
                species_name or fasta_path.stem,
                species_elapsed, species_elapsed / 60,
            )

    # Final barrier and cleanup
    if world_size > 1:
        dist.barrier()

    total_elapsed = time.time() - total_start
    if rank == 0:
        logger.info(
            "All done. %d work items, %.1f min total",
            len(work_items), total_elapsed / 60,
        )

    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
