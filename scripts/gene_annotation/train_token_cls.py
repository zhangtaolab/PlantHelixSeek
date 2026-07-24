#!/usr/bin/env python3
"""Token classification training script for ChaobaV2 gene structure prediction.

Fine-tunes a ChaobaV2 MLM checkpoint with a TokenClassification head (17 BILOU
labels) on genome Parquet datasets produced by ``build_dataset.py``.

Label schema (17 labels + PAD = -100):
    0: O          (intergenic)
    1-4:  B/I/L/U-CDS
    5-8:  B/I/L/U-INTRON
    9-12: B/I/L/U-UTR5
    13-16: B/I/L/U-UTR3

Supports both Model A (11-token vocab, softmask-insensitive) and Model B
(15-token vocab with softmask-sensitive lowercase a/t/c/g tokens).

Usage:
    python train_token_cls.py \
        --config LLM_anno/scripts/train/train_config.yaml \
        --model-path LLM_anno/models/pretrain_model/<checkpoint> \
        --train-data LLM_anno/result/dataset/softmask_insensitive/train.parquet \
        --val-data LLM_anno/result/dataset/softmask_insensitive/val.parquet \
        --output-dir LLM_anno/result/train/run_001

    # Model B (softmask-sensitive, 15-token vocab)
    python train_token_cls.py \
        --config LLM_anno/scripts/train/train_config.yaml \
        --model-path LLM_anno/models/pretrain_model/<checkpoint> \
        --train-data LLM_anno/result/dataset/softmask_sensitive/train.parquet \
        --val-data LLM_anno/result/dataset/softmask_sensitive/val.parquet \
        --softmask-sensitive \
        --output-dir LLM_anno/result/train/softmask_sensitive/run_001
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd=None
try:
    import pyarrow.parquet as pq
except ImportError:
    pq=None
import torch
import torch.nn as nn
import yaml
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES: dict[int, str] = {
    0: "O",
    1: "B-CDS",
    2: "I-CDS",
    3: "L-CDS",
    4: "U-CDS",
    5: "B-INTRON",
    6: "I-INTRON",
    7: "L-INTRON",
    8: "U-INTRON",
    9: "B-UTR5",
    10: "I-UTR5",
    11: "L-UTR5",
    12: "U-UTR5",
    13: "B-UTR3",
    14: "I-UTR3",
    15: "L-UTR3",
    16: "U-UTR3",
}

NUM_LABELS = 17
LABEL_PAD = -100

SOFTMASK_EXTRA_TOKENS: list[str] = ["a", "t", "c", "g"]


def _patch_attention_to_sdpa(model: nn.Module) -> None:
    """Replace _attention_forward on all Chaoba attention layers with SDPA.

    ChaobaV2's native attention materializes the full N×N attention matrix,
    which causes OOM at seq_len=8192 even with batch_size=1. SDPA uses
    memory-efficient (flash) kernels under the hood when available.
    """
    import torch.nn.functional as F
    from torch.nn.functional import scaled_dot_product_attention

    patched = 0
    for module in model.modules():
        if not hasattr(module, "_attention_forward"):
            continue
        if not hasattr(module, "scaling"):
            continue
        original_fn = module._attention_forward

        def make_sdpa_forward(scaling: float, dropout_p_getter):
            def _sdpa_forward(self, query, key, value, attention_mask):
                dropout_p = dropout_p_getter(self) if attention_mask is None else 0.0
                if attention_mask is not None and attention_mask.dim() == 2:
                    attention_mask = None
                attn_output = scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p if self.training else 0.0,
                    is_causal=False,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
                return attn_output, None
            return _sdpa_forward

        bound_fn = make_sdpa_forward(
            module.scaling,
            lambda m: m.attention_dropout,
        ).__get__(module, type(module))
        module._attention_forward = bound_fn
        patched += 1

    if patched > 0:
        logger.info("Patched %d attention layers to use SDPA", patched)
    else:
        logger.warning("No attention layers found for SDPA patching")
SOFTMASK_UPPER_MAP: dict[str, str] = {"a": "A", "t": "T", "c": "C", "g": "G"}

# Default training hyperparameters (used when no config file is provided)
DEFAULT_TRAINING_CONFIG: dict = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "learning_rate": 1.0e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.05,
    "bf16": True,
    "logging_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1",
    "greater_is_better": True,
    "report_to": "tensorboard",
    "gradient_accumulation_steps": 4,
    "dataloader_num_workers": 4,
    "remove_unused_columns": False,
}

DEFAULT_MODEL_CONFIG: dict = {
    "num_labels": NUM_LABELS,
    "max_seq_length": 8192,
}

DEFAULT_OUTPUT_DIR = "LLM_anno/result/train/run_001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_path(path_str: str, project_root: Path) -> Path:
    """Resolve a path string relative to *project_root* if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else project_root / p


def load_config(config_path: Path | None) -> dict:
    """Load YAML config and return raw dict. Returns empty dict if no path."""
    if config_path is None or not config_path.exists():
        logger.info("No config file provided; using defaults")
        return {}
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", config_path)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    strategy: str = "sqrt_inverse_freq",
    cap: float = 100.0,
    metadata_path: Path | None = None,
    manual_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute per-class weights for CrossEntropyLoss.

    Supported strategies:
        * ``"sqrt_inverse_freq"`` — weight_i = sqrt(N_total / (C * n_i))
        * ``"inverse_freq"`` — weight_i = N_total / (C * n_i)
        * ``"manual"`` — read weights from *manual_weights* dict
        * ``"none"`` — all weights = 1.0

    For automatic strategies, weights are normalised so O = 1.0 and clipped
    to *cap*.  For ``"manual"``, the values are used as-is after clipping.

    Args:
        strategy: Weighting strategy name.
        cap: Maximum allowed weight value.
        metadata_path: Path to metadata.json (required for auto strategies).
        manual_weights: Dict of label_name → weight (required for "manual").

    Returns:
        Float tensor of shape ``[NUM_LABELS]``.
    """
    if strategy == "none":
        return torch.ones(NUM_LABELS, dtype=torch.float32)

    if strategy == "manual":
        if manual_weights is None:
            raise ValueError("manual_weights required when strategy='manual'")
        raw_weights = np.array(
            [manual_weights.get(LABEL_NAMES[i], 1.0) for i in range(NUM_LABELS)],
            dtype=np.float64,
        )
        raw_weights = np.clip(raw_weights, 0.0, cap)
        weights = torch.tensor(raw_weights, dtype=torch.float32)
        logger.info(
            "Class weights (strategy=manual, cap=%.1f):\n%s",
            cap,
            "\n".join(
                f"  {LABEL_NAMES[i]:>12s}: {w:.4f}"
                for i, w in enumerate(weights.tolist())
            ),
        )
        return weights

    # Automatic strategies require metadata
    if metadata_path is None:
        raise ValueError(f"metadata_path required for strategy='{strategy}'")

    with open(metadata_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    train_counts: dict[str, int] = meta["label_counts"]["train"]

    counts = np.array(
        [train_counts[LABEL_NAMES[i]] for i in range(NUM_LABELS)],
        dtype=np.float64,
    )

    total = counts.sum()
    n_classes = NUM_LABELS

    if strategy == "inverse_freq":
        raw_weights = total / (n_classes * counts)
    elif strategy == "sqrt_inverse_freq":
        raw_weights = np.sqrt(total / (n_classes * counts))
    else:
        raise ValueError(f"Unknown class_weight_strategy: {strategy}")

    raw_weights /= raw_weights[0]
    raw_weights = np.clip(raw_weights, 0.0, cap)

    weights = torch.tensor(raw_weights, dtype=torch.float32)
    logger.info(
        "Class weights (strategy=%s, cap=%.1f):\n%s",
        strategy,
        cap,
        "\n".join(
            f"  {LABEL_NAMES[i]:>12s}: {w:.4f}" for i, w in enumerate(weights.tolist())
        ),
    )
    return weights


# ---------------------------------------------------------------------------
# CRF with gene-structure constraints
# ---------------------------------------------------------------------------


def _build_bilou_transitions(
    enforce_gene_order: bool = True,
    allow_utr_introns: bool = True,
) -> torch.Tensor:
    """Build a 17x17 transition constraint matrix for BILOU gene labels.

    Returns an integer mask: 1 = allowed, 0 = forbidden.

    Constraint layers (applied in order):
      1. BILOU syntax: B→{I,L}, I→{I,L}, L/U→{O,B,U}, O→{O,B,U}
      2. Gene start: O can only enter CDS or UTR5 (INTRON/UTR3 cannot start a gene)
      3. Gene end: only UTR3 can transition to O (CDS/UTR5/INTRON cannot end a gene)
      4. Gene order: per-source biological rules via ``_gene_order_valid()``

    Args:
        enforce_gene_order: Apply biological gene-structure constraints.
        allow_utr_introns: Allow introns within UTR5 regions (common in plants).
    """
    n = NUM_LABELS
    allowed = torch.zeros(n, n, dtype=torch.float32)

    gene_start_structures = {"CDS", "UTR5"}

    def parse(idx: int) -> tuple[str, str]:
        name = LABEL_NAMES[idx]
        if name == "O":
            return "O", ""
        bilou, struct = name.split("-", 1)
        return bilou, struct

    for src in range(n):
        src_bilou, src_struct = parse(src)
        for dst in range(n):
            dst_bilou, dst_struct = parse(dst)

            ok = False

            if src_bilou == "O":
                ok = dst_bilou in ("O", "B", "U")
                if ok and dst_bilou in ("B", "U"):
                    ok = dst_struct in gene_start_structures

            elif src_bilou == "B":
                ok = dst_bilou in ("I", "L") and dst_struct == src_struct

            elif src_bilou == "I":
                ok = dst_bilou in ("I", "L") and dst_struct == src_struct

            elif src_bilou in ("L", "U"):
                if dst_bilou == "O":
                    if enforce_gene_order:
                        ok = src_struct in {"CDS", "UTR3"}
                    else:
                        ok = True
                elif dst_bilou in ("B", "U"):
                    if enforce_gene_order:
                        ok = _gene_order_valid(
                            src_struct, dst_struct, dst_bilou, allow_utr_introns
                        )
                    else:
                        ok = True

            if ok:
                allowed[src, dst] = 1.0

    return allowed


def validate_transition_matrix(
    transition_mask: torch.Tensor,
    parquet_path: Path,
    max_samples: int = 10000,
) -> int:
    """Validate that training label sequences respect the transition constraints.

    Returns the number of forbidden transitions found.
    """
    table = pq.read_table(parquet_path, memory_map=True)
    n = min(max_samples, table.num_rows)
    violations = 0
    for idx in range(n):
        labs: list[int] = table.column("labels")[idx].as_py()
        for t in range(1, len(labs)):
            if labs[t - 1] >= 0 and labs[t] >= 0:
                if transition_mask[labs[t - 1], labs[t]] == 0:
                    violations += 1
                    if violations <= 10:
                        logger.warning(
                            "Forbidden transition in training data: %s(%d) -> %s(%d) at pos %d",
                            LABEL_NAMES.get(labs[t - 1], "?"),
                            labs[t - 1],
                            LABEL_NAMES.get(labs[t], "?"),
                            labs[t],
                            t,
                        )
    logger.info(
        "Transition validation: %d violations in %d samples from %s",
        violations,
        min(max_samples, table.num_rows),
        parquet_path.name,
    )
    return violations


def _gene_order_valid(
    src_struct: str,
    dst_struct: str,
    dst_bilou: str,
    allow_utr_introns: bool = True,
) -> bool:
    """Check if a transition from a span-ending tag (L/U) to a span-beginning
    tag (B/U) respects biological gene structure constraints.

    Reference constraints (per source structure):

        CDS    → INTRON(B),     UTR3(B/U)     — splicing or translation termination
        INTRON → CDS(B)                        — canonical splice acceptor
                 [UTR5(B) if UTR introns]      — UTR intron support
        UTR5   → CDS(B/U)                     — start codon
                 [INTRON(B) if UTR introns]    — UTR intron support
        UTR3   → (none — must go to O)         — gene end

    Additional structural rules:
        - No consecutive same-type spans (L-X → B-X must merge into one span).
        - No single-base introns (dst INTRON + dst_bilou U → impossible).
        - No single-base exons after intron (src INTRON + dst_bilou U →
          micro-exons extremely rare and a single base cannot carry splice signals).

    Args:
        src_struct: Source structure type (CDS, INTRON, UTR5, UTR3).
        dst_struct: Destination structure type.
        dst_bilou: BILOU tag of destination (B or U).
        allow_utr_introns: If True, allow INTRON↔UTR5 transitions for UTR introns
            (common in plant genomes: ~20% Arabidopsis, ~35% rice genes).

    Returns:
        True if the transition is biologically valid.
    """
    # Rule 1: No consecutive same-type spans
    if src_struct == dst_struct:
        return False

    # Rule 2: No single-base introns (biologically impossible — need GT...AG)
    if dst_struct == "INTRON" and dst_bilou == "U":
        return False

    # Rule 3: No single-base exons after intron (micro-exons extremely rare;
    #         single base cannot carry meaningful splice acceptor/donor signals)
    if src_struct == "INTRON" and dst_bilou == "U":
        return False

    # Rule 4: Per-source constraints
    if src_struct == "CDS":
        # CDS → INTRON (splicing donor) or UTR3 (translation termination)
        return dst_struct in {"INTRON", "UTR3"}

    if src_struct == "INTRON":
        # INTRON → CDS (canonical splice acceptor), UTR5 or UTR3 (UTR intron case)
        if dst_struct == "CDS":
            return True
        if allow_utr_introns and dst_struct in {"UTR5", "UTR3"}:
            return True
        return False

    if src_struct == "UTR5":
        # UTR5 → CDS (start codon) or INTRON (UTR intron case)
        if dst_struct == "CDS":
            return True
        if allow_utr_introns and dst_struct == "INTRON":
            return True
        return False

    if src_struct == "UTR3":
        if allow_utr_introns and dst_struct == "INTRON":
            return True
        return False

    return False


class GeneStructureCRF(nn.Module):
    """Linear-chain CRF with BILOU and gene-structure transition constraints.

    Transition scores are learned parameters, but forbidden transitions are
    masked to ``-inf`` so they can never be selected during Viterbi decoding.

    Optionally applies a soft penalty for CDS spans whose length is not a
    multiple of 3 (codon frame constraint).
    """

    def __init__(
        self,
        num_labels: int,
        enforce_gene_order: bool = True,
        allow_utr_introns: bool = True,
        cds_mod3_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.cds_mod3_penalty = cds_mod3_penalty

        mask = _build_bilou_transitions(
            enforce_gene_order=enforce_gene_order,
            allow_utr_introns=allow_utr_introns,
        )
        self.register_buffer("transition_mask", mask)

        # Learnable transitions: initialize small random values
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.02)

        # Start/end scores
        self.start_transitions = nn.Parameter(torch.randn(num_labels) * 0.02)
        self.end_transitions = nn.Parameter(torch.randn(num_labels) * 0.02)

    def _constrained_transitions(self) -> torch.Tensor:
        """Apply mask: forbidden transitions get -inf."""
        return self.transitions + (1.0 - self.transition_mask) * (-1e4)

    def forward(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss (batched, vectorized).

        Args:
            emissions: (batch, seq_len, num_labels) raw scores from classifier.
            labels: (batch, seq_len) integer label ids (PAD=-100 included).
            mask: (batch, seq_len) bool, True for real tokens.

        Returns:
            Scalar loss (mean over batch).
        """
        trans = self._constrained_transitions()

        log_norm = self._forward_algorithm_batched(emissions, trans, mask)
        gold_score = self._score_sentence_batched(emissions, labels, trans, mask)

        nll = log_norm - gold_score
        crf_loss = nll.mean()

        if self.cds_mod3_penalty > 0.0 and self.training:
            with torch.no_grad():
                pred_paths = self.decode(emissions, mask)
            crf_loss = crf_loss + self.cds_mod3_penalty * self._cds_mod3_loss_pred(
                pred_paths, mask
            )

        return crf_loss

    def _forward_algorithm_batched(
        self,
        emissions: torch.Tensor,
        trans: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-partition Z(x) for the whole batch (vectorized)."""
        # emissions: (B, T, K), mask: (B, T), trans: (K, K)
        batch_size, seq_len, num_labels = emissions.shape
        device = emissions.device
        dtype = emissions.dtype

        # alpha: (B, K)
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for t in range(1, seq_len):
            # scores[b, i, j] = alpha[b, i] + trans[i, j] + emit[b, j]
            scores = alpha.unsqueeze(-1) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)

            m = mask[:, t].unsqueeze(1).to(dtype)
            alpha = new_alpha * m + alpha * (1.0 - m)

        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

    def _score_sentence_batched(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        trans: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score gold label sequences for the whole batch (vectorized)."""
        batch_size, seq_len, num_labels = emissions.shape
        device = emissions.device
        dtype = emissions.dtype

        # Clamp PAD (-100) labels to 0 so gather/index ops don't go OOB.
        # Masked positions are excluded from the accumulated score anyway,
        # so the substituted value is irrelevant.
        safe_labels = labels.clamp(min=0)

        # Gather emission scores at gold labels: (B, T)
        emit_at_labels = emissions.gather(2, safe_labels.unsqueeze(2)).squeeze(2)

        score = self.start_transitions[safe_labels[:, 0]] + emit_at_labels[:, 0]

        for t in range(1, seq_len):
            trans_score = trans[safe_labels[:, t - 1], safe_labels[:, t]]
            step_score = trans_score + emit_at_labels[:, t]
            m = mask[:, t].to(dtype)
            score = score + step_score * m

        # Find last valid position per sequence and add end_transitions
        seq_lengths = mask.sum(dim=1).long()
        last_labels = safe_labels.gather(1, (seq_lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_labels]

        return score

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding for each sequence in the batch.

        Args:
            emissions: (batch, seq_len, num_labels)
            mask: (batch, seq_len) bool

        Returns:
            List of label id lists (without padding positions).
        """
        trans = self._constrained_transitions()
        batch_size = emissions.shape[0]
        results: list[list[int]] = []

        for b in range(batch_size):
            valid_len = int(mask[b].sum())
            if valid_len == 0:
                results.append([])
                continue
            emit = emissions[b, :valid_len]
            path = self._viterbi_single(emit, trans)
            results.append(path)

        return results

    def _viterbi_single(
        self, emissions: torch.Tensor, trans: torch.Tensor
    ) -> list[int]:
        """Viterbi for a single sequence."""
        seq_len = emissions.shape[0]
        dp = self.start_transitions + emissions[0]
        backpointers = torch.zeros(
            seq_len, self.num_labels, dtype=torch.long, device=emissions.device
        )

        for t in range(1, seq_len):
            scores = dp.unsqueeze(-1) + trans
            best = scores.argmax(dim=0)
            dp = scores.max(dim=0).values + emissions[t]
            backpointers[t] = best

        dp = dp + self.end_transitions
        best_last = dp.argmax().item()

        path = [0] * seq_len
        path[-1] = int(best_last)
        for t in range(seq_len - 2, -1, -1):
            path[t] = int(backpointers[t + 1, path[t + 1]].item())

        return path

    def _cds_mod3_loss(self, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Soft penalty: fraction of CDS spans with length not divisible by 3."""
        violations = 0.0
        total_spans = 0

        for b in range(labels.shape[0]):
            valid_len = int(mask[b].sum())
            if valid_len == 0:
                continue
            seq = labels[b, :valid_len].tolist()

            span_start = None
            for t in range(len(seq)):
                lab = seq[t]
                is_cds = lab in (1, 2, 3, 4)  # B/I/L/U-CDS

                if is_cds and span_start is None:
                    span_start = t
                elif not is_cds and span_start is not None:
                    span_len = t - span_start
                    total_spans += 1
                    if span_len % 3 != 0:
                        violations += 1.0
                    span_start = None

            if span_start is not None:
                span_len = len(seq) - span_start
                total_spans += 1
                if span_len % 3 != 0:
                    violations += 1.0

        if total_spans == 0:
            return torch.tensor(0.0, device=labels.device)

        return torch.tensor(violations / total_spans, device=labels.device)

    def _cds_mod3_loss_pred(
        self, pred_paths: list[list[int]], mask: torch.Tensor
    ) -> torch.Tensor:
        """CDS mod3 penalty on Viterbi-decoded predictions (used during training)."""
        violations = 0.0
        total_spans = 0

        for b in range(len(pred_paths)):
            valid_len = int(mask[b].sum())
            if valid_len == 0:
                continue
            seq = pred_paths[b][:valid_len]

            span_start = None
            for t in range(len(seq)):
                lab = seq[t]
                is_cds = lab in (1, 2, 3, 4)

                if is_cds and span_start is None:
                    span_start = t
                elif not is_cds and span_start is not None:
                    span_len = t - span_start
                    total_spans += 1
                    if span_len % 3 != 0:
                        violations += 1.0
                    span_start = None

            if span_start is not None:
                span_len = len(seq) - span_start
                total_spans += 1
                if span_len % 3 != 0:
                    violations += 1.0

        if total_spans == 0:
            return torch.tensor(0.0, device=mask.device)

        return torch.tensor(violations / total_spans, device=mask.device)


# ---------------------------------------------------------------------------
# Weighted Trainer (with optional CRF)
# ---------------------------------------------------------------------------


class WeightedTokenTrainer(Trainer):
    """Trainer with per-class weights and optional CRF loss.

    Overrides ``evaluate()`` to use streaming confusion-matrix accumulation
    instead of HF's default ``EvalLoopContainer`` which does O(N^2)
    ``torch.cat`` on every batch (catastrophic for large eval sets with
    long sequences).
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        crf: GeneStructureCRF | None = None,
        eval_num_workers: int | None = None,
        f1_strategy: str = "macro_f1",
        f1_weights: dict[str, float] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.crf = crf
        self._eval_num_workers = eval_num_workers
        self._f1_strategy = f1_strategy
        self._f1_weights = f1_weights

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        import torch.utils.data

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        num_workers = (
            self._eval_num_workers
            if self._eval_num_workers is not None
            else self.args.dataloader_num_workers
        )

        return torch.utils.data.DataLoader(
            eval_ds,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor
            if num_workers > 0
            else None,
        )

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            raise ValueError("evaluation requires an eval_dataset")

        dataloader = self.get_eval_dataloader(eval_ds)
        model = self.model
        model.eval()

        confusion = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
        total_loss = 0.0
        num_samples = 0
        num_steps = 0

        dataloader = self.accelerator.prepare(dataloader) if hasattr(self, "accelerator") else dataloader
        total_steps = len(dataloader) if hasattr(dataloader, "__len__") else None

        logger.info(
            "\n***** Running Evaluation *****\n  Num examples = %d\n  Batch size = %d",
            len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0,
            self.args.eval_batch_size,
        )
        start_time = time.time()

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            labels = inputs["labels"]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.args.bf16,
            ):
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            mask = labels != LABEL_PAD
            preds_np = preds[mask].cpu().numpy()
            labels_np = labels[mask].cpu().numpy()
            np.add.at(confusion, (labels_np, preds_np), 1)

            batch_samples = int(mask.sum())
            num_samples += batch_samples
            total_loss += loss.item() * batch_samples
            num_steps += 1

            del logits, outputs, preds, mask

            if total_steps and (step + 1) % max(1, total_steps // 10) == 0:
                elapsed = time.time() - start_time
                sps = num_samples / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Eval [%d/%d]  samples=%d  %.1f samples/s  elapsed=%.0fs",
                    step + 1, total_steps, num_samples, sps, elapsed,
                )

        avg_loss = total_loss / max(num_samples, 1)
        metrics = self._metrics_from_confusion(confusion)
        metrics["loss"] = avg_loss

        runtime = time.time() - start_time
        metrics["runtime"] = round(runtime, 4)
        if runtime > 0:
            metrics["samples_per_second"] = round(num_samples / runtime, 3)
            metrics["steps_per_second"] = round(num_steps / runtime, 3)

        prefixed: dict[str, float] = {}
        for k, v in metrics.items():
            prefixed[f"{metric_key_prefix}_{k}"] = v

        prefixed["epoch"] = self.state.epoch or 0.0

        self.log(prefixed)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, prefixed
        )

        return prefixed

    def _metrics_from_confusion(self, confusion: np.ndarray) -> dict[str, float]:
        tp = np.diag(confusion).astype(np.float64)
        fp = confusion.sum(axis=0).astype(np.float64) - tp
        fn = confusion.sum(axis=1).astype(np.float64) - tp

        precision = np.divide(
            tp, tp + fp,
            out=np.zeros(NUM_LABELS, dtype=np.float64),
            where=(tp + fp > 0),
        )
        recall = np.divide(
            tp, tp + fn,
            out=np.zeros(NUM_LABELS, dtype=np.float64),
            where=(tp + fn > 0),
        )
        denom_f1 = precision + recall
        f1 = np.divide(
            2.0 * precision * recall, denom_f1,
            out=np.zeros(NUM_LABELS, dtype=np.float64),
            where=(denom_f1 > 0),
        )

        total = confusion.sum()
        accuracy = float(tp.sum() / total) if total > 0 else 0.0
        macro_f1 = float(f1.mean())
        macro_precision = float(precision.mean())
        macro_recall = float(recall.mean())

        c = float(tp.sum())
        s = float(total)
        p_k = confusion.sum(axis=0).astype(np.float64)
        t_k = confusion.sum(axis=1).astype(np.float64)
        numerator = c * s - float(np.sum(p_k * t_k))
        denom = np.sqrt(
            (s * s - float(np.sum(p_k * p_k)))
            * (s * s - float(np.sum(t_k * t_k)))
        )
        mcc: float = float(numerator / denom) if denom > 0 else 0.0

        per_class_f1 = {LABEL_NAMES[i]: float(f1[i]) for i in range(NUM_LABELS)}

        if self._f1_strategy == "weighted_f1" and self._f1_weights is not None:
            total_w = sum(self._f1_weights.values())
            per_class_prec = {LABEL_NAMES[i]: float(precision[i]) for i in range(NUM_LABELS)}
            per_class_rec = {LABEL_NAMES[i]: float(recall[i]) for i in range(NUM_LABELS)}
            weighted_f1 = sum(
                self._f1_weights.get(n, 1.0) * v for n, v in per_class_f1.items()
            ) / total_w
            weighted_prec = sum(
                self._f1_weights.get(n, 1.0) * v for n, v in per_class_prec.items()
            ) / total_w
            weighted_rec = sum(
                self._f1_weights.get(n, 1.0) * v for n, v in per_class_rec.items()
            ) / total_w
        else:
            weighted_f1 = macro_f1
            weighted_prec = macro_precision
            weighted_rec = macro_recall

        # --- CDS boundary metrics (B-CDS + L-CDS combined) ---
        bcds_idx = 1   # B-CDS
        lcds_idx = 3   # L-CDS

        # Micro-averaged precision/recall for CDS boundaries:
        # pool TP/FP/FN of B-CDS and L-CDS together, then compute.
        boundary_tp = tp[bcds_idx] + tp[lcds_idx]
        boundary_fp = fp[bcds_idx] + fp[lcds_idx]
        boundary_fn = fn[bcds_idx] + fn[lcds_idx]
        boundary_prec = float(boundary_tp / (boundary_tp + boundary_fp)) if (boundary_tp + boundary_fp) > 0 else 0.0
        boundary_rec  = float(boundary_tp / (boundary_tp + boundary_fn)) if (boundary_tp + boundary_fn) > 0 else 0.0
        boundary_f1   = (
            float(2.0 * boundary_prec * boundary_rec / (boundary_prec + boundary_rec))
            if (boundary_prec + boundary_rec) > 0 else 0.0
        )

        cds_idx = np.array([1, 2, 3, 4])
        cds_tp = float(confusion[np.ix_(cds_idx, cds_idx)].sum())
        cds_fp = float(confusion[:, cds_idx].sum()) - cds_tp
        cds_fn = float(confusion[cds_idx, :].sum()) - cds_tp
        cds_f1 = float(2 * cds_tp / (2 * cds_tp + cds_fp + cds_fn)) if (2 * cds_tp + cds_fp + cds_fn) > 0 else 0.0
        cds_prec = float(cds_tp / (cds_tp + cds_fp)) if (cds_tp + cds_fp) > 0 else 0.0
        cds_rec = float(cds_tp / (cds_tp + cds_fn)) if (cds_tp + cds_fn) > 0 else 0.0

        result: dict[str, float] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "weighted_f1": weighted_f1,
            "weighted_precision": weighted_prec,
            "weighted_recall": weighted_rec,
            "mcc": mcc,
            # --- CDS nucleotide-level (B/I/L/U-CDS collapsed) ---
            "cds_f1": cds_f1,
            "cds_precision": cds_prec,
            "cds_recall": cds_rec,
            # --- CDS boundary (B-CDS + L-CDS micro-averaged) ---
            "cds_boundary_f1": boundary_f1,
            "cds_boundary_precision": boundary_prec,
            "cds_boundary_recall": boundary_rec,
            # --- Per-boundary precision/recall ---
            "precision_B-CDS": float(precision[bcds_idx]),
            "recall_B-CDS": float(recall[bcds_idx]),
            "precision_L-CDS": float(precision[lcds_idx]),
            "recall_L-CDS": float(recall[lcds_idx]),
        }
        for i in range(NUM_LABELS):
            result[f"f1_{LABEL_NAMES[i]}"] = per_class_f1[LABEL_NAMES[i]]

        return result

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, seq_len, NUM_LABELS)

        if self.crf is not None:
            crf_device = logits.device
            if next(self.crf.parameters()).device != crf_device:
                self.crf = self.crf.to(crf_device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                crf_mask = attention_mask.bool()
            else:
                crf_mask = labels != LABEL_PAD
            loss = self.crf(logits, labels, crf_mask)
        elif self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=LABEL_PAD,
            )
            loss = loss_fct(
                logits.view(-1, NUM_LABELS),
                labels.view(-1),
            )
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=LABEL_PAD)
            loss = loss_fct(
                logits.view(-1, NUM_LABELS),
                labels.view(-1),
            )

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GenomeTokenDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for genome token-classification from Parquet.

    Each row contains:
        - ``sequence``: a DNA string of length *max_length* (each character is
          a single token).
        - ``labels``: a ``list[int]`` of length *max_length* with BILOU label
          indices.

    Handles two tokenizer styles:
        * **ChaobaV2**: adds ``<cls>`` (position 0) and ``<eos>`` (last
          position), so total input length is ``max_length + 2``.  Labels at
          special-token positions are set to ``-100``.
        * **Caduceus / others**: no special tokens, input length equals
          ``max_length`` directly.

    Reads parquet by row-group on demand, caching one row-group as a pandas
    DataFrame (~0.1 GB each).  Keeps RAM bounded regardless of dataset size.
    """

    def __init__(
        self, parquet_path: Path, tokenizer: AutoTokenizer, max_length: int = 8192
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._has_cls = tokenizer.cls_token is not None
        if self._has_cls:
            self._encoded_length = max_length + 2
        else:
            self._encoded_length = max_length

        self._pf = pq.ParquetFile(str(parquet_path))
        self._num_rows = self._pf.metadata.num_rows
        self._num_row_groups = self._pf.metadata.num_row_groups
        self._rg_sizes = [
            self._pf.metadata.row_group(i).num_rows for i in range(self._num_row_groups)
        ]
        self._rg_offsets: list[int] = []
        off = 0
        for s in self._rg_sizes:
            self._rg_offsets.append(off)
            off += s

        self._rg_df: pd.DataFrame | None = None
        self._rg_index: int = -1

        logger.info(
            "ParquetFile %s: %d rows, %d row groups",
            parquet_path, self._num_rows, self._num_row_groups,
        )

    def __len__(self) -> int:
        return self._num_rows

    def _ensure_row_group(self, idx: int) -> None:
        lo, hi, rg = 0, 0, 0
        for rg in range(self._num_row_groups):
            hi = lo + self._rg_sizes[rg]
            if idx < hi:
                break
            lo = hi
        if rg == self._rg_index:
            return
        self._rg_df = self._pf.read_row_group(rg, columns=["sequence", "labels"]).to_pandas()
        self._rg_index = rg

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_row_group(idx)
        local = idx - self._rg_offsets[self._rg_index]
        row = self._rg_df.iloc[local]
        sequence: str = row["sequence"]
        labels: list[int] = row["labels"]

        if len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]
            labels = labels[: self.max_length]

        tokens = list(sequence)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self._encoded_length,
            padding="max_length",
        )

        if self._has_cls:
            labels_tensor = torch.full((self._encoded_length,), LABEL_PAD, dtype=torch.long)
            labels_tensor[1 : self.max_length + 1] = torch.tensor(labels, dtype=torch.long)
        else:
            labels_tensor = torch.full((self._encoded_length,), LABEL_PAD, dtype=torch.long)
            labels_tensor[: len(labels)] = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels_tensor,
        }


# ---------------------------------------------------------------------------
# Vocab expansion for softmask-sensitive (Model B)
# ---------------------------------------------------------------------------


def expand_vocab_softmask(
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
) -> None:
    """Add lowercase DNA tokens (a/t/c/g) and initialise their embeddings."""
    num_added = tokenizer.add_tokens(SOFTMASK_EXTRA_TOKENS)
    logger.info(
        "Added %d new tokens to tokenizer: %s", num_added, SOFTMASK_EXTRA_TOKENS
    )

    old_emb = model.get_input_embeddings()
    old_vocab_size = old_emb.weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_emb = model.get_input_embeddings()

    with torch.no_grad():
        for tok_str in SOFTMASK_EXTRA_TOKENS:
            upper_id = tokenizer.convert_tokens_to_ids(SOFTMASK_UPPER_MAP[tok_str])
            lower_id = tokenizer.convert_tokens_to_ids(tok_str)
            if upper_id == tokenizer.unk_token_id:
                logger.warning("Cannot find uppercase token '%s' for init", tok_str)
                continue
            new_emb.weight[lower_id].copy_(
                old_emb.weight[upper_id].clone()
                + torch.randn(model.config.hidden_size) * 0.02
            )

    model.config.vocab_size = len(tokenizer)
    logger.info(
        "Vocab expanded %d -> %d, embeddings re-initialised for lowercase tokens",
        old_vocab_size,
        len(tokenizer),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    eval_pred,
    f1_strategy: str = "macro_f1",
    f1_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for token classification.

    Returns accuracy, macro/weighted precision, recall, F1, and MCC.
    When *f1_strategy* is ``"weighted_f1"``, uses *f1_weights* for aggregation.

    All metrics are computed with vectorised NumPy (no sklearn dependency)
    so that ``compute_metrics`` finishes in milliseconds instead of seconds.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    mask = labels != LABEL_PAD
    true_labels = labels[mask]
    pred_labels = predictions[mask]

    accuracy: float = float((true_labels == pred_labels).mean())

    confusion = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
    np.add.at(confusion, (true_labels, pred_labels), 1)

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros(NUM_LABELS, dtype=np.float64), where=(tp + fp > 0))
    recall = np.divide(tp, tp + fn, out=np.zeros(NUM_LABELS, dtype=np.float64), where=(tp + fn > 0))
    denom_f1 = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom_f1, out=np.zeros(NUM_LABELS, dtype=np.float64), where=(denom_f1 > 0))

    per_class_f1: dict[str, float] = {
        LABEL_NAMES[i]: float(f1[i]) for i in range(NUM_LABELS)
    }
    per_class_prec: dict[str, float] = {
        LABEL_NAMES[i]: float(precision[i]) for i in range(NUM_LABELS)
    }
    per_class_rec: dict[str, float] = {
        LABEL_NAMES[i]: float(recall[i]) for i in range(NUM_LABELS)
    }

    macro_f1 = float(f1.mean())
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())

    c = float(tp.sum())
    s = float(confusion.sum())
    p_k = confusion.sum(axis=0).astype(np.float64)
    t_k = confusion.sum(axis=1).astype(np.float64)
    numerator = c * s - float(np.sum(p_k * t_k))
    denom = np.sqrt(
        (s * s - float(np.sum(p_k * p_k)))
        * (s * s - float(np.sum(t_k * t_k)))
    )
    mcc: float = float(numerator / denom) if denom > 0 else 0.0

    if f1_strategy == "weighted_f1" and f1_weights is not None:
        total_w = sum(f1_weights.values())
        weighted_f1 = (
            sum(
                f1_weights.get(name, 1.0) * per_class_f1.get(name, 0.0)
                for name in per_class_f1
            )
            / total_w
        )
        weighted_precision = (
            sum(
                f1_weights.get(n, 1.0) * per_class_prec.get(n, 0.0)
                for n in per_class_f1
            )
            / total_w
        )
        weighted_recall = (
            sum(
                f1_weights.get(n, 1.0) * per_class_rec.get(n, 0.0)
                for n in per_class_f1
            )
            / total_w
        )
    else:
        weighted_f1 = macro_f1
        weighted_precision = macro_precision
        weighted_recall = macro_recall

    bcds_idx = 1
    lcds_idx = 3
    boundary_tp = tp[bcds_idx] + tp[lcds_idx]
    boundary_fp = fp[bcds_idx] + fp[lcds_idx]
    boundary_fn = fn[bcds_idx] + fn[lcds_idx]
    boundary_prec = float(boundary_tp / (boundary_tp + boundary_fp)) if (boundary_tp + boundary_fp) > 0 else 0.0
    boundary_rec  = float(boundary_tp / (boundary_tp + boundary_fn)) if (boundary_tp + boundary_fn) > 0 else 0.0
    boundary_f1   = (
        float(2.0 * boundary_prec * boundary_rec / (boundary_prec + boundary_rec))
        if (boundary_prec + boundary_rec) > 0 else 0.0
    )

    cds_idx = np.array([1, 2, 3, 4])
    cds_tp = float(confusion[np.ix_(cds_idx, cds_idx)].sum())
    cds_fp = float(confusion[:, cds_idx].sum()) - cds_tp
    cds_fn = float(confusion[cds_idx, :].sum()) - cds_tp
    cds_f1 = float(2 * cds_tp / (2 * cds_tp + cds_fp + cds_fn)) if (2 * cds_tp + cds_fp + cds_fn) > 0 else 0.0
    cds_prec = float(cds_tp / (cds_tp + cds_fp)) if (cds_tp + cds_fp) > 0 else 0.0
    cds_rec = float(cds_tp / (cds_tp + cds_fn)) if (cds_tp + cds_fn) > 0 else 0.0

    result: dict[str, float] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "mcc": mcc,
        "cds_f1": cds_f1,
        "cds_precision": cds_prec,
        "cds_recall": cds_rec,
        "cds_boundary_f1": boundary_f1,
        "cds_boundary_precision": boundary_prec,
        "cds_boundary_recall": boundary_rec,
        "precision_B-CDS": float(precision[bcds_idx]),
        "recall_B-CDS": float(recall[bcds_idx]),
        "precision_L-CDS": float(precision[lcds_idx]),
        "recall_L-CDS": float(recall[lcds_idx]),
    }

    for i in range(NUM_LABELS):
        name = LABEL_NAMES[i]
        result[f"f1_{name}"] = per_class_f1.get(name, 0.0)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ChaobaV2 Token Classification for gene structure prediction",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: train_config.yaml in same directory)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained ChaobaV2 model",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to train.parquet",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to val.parquet",
    )
    parser.add_argument(
        "--softmask-sensitive",
        action="store_true",
        default=False,
        help="Enable softmask-sensitive mode (adds a/t/c/g tokens, 15-vocab)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to dataset metadata.json (auto-detected from train-data dir if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ---- Resolve config ----
    script_dir = Path(__file__).resolve().parent
    default_config_path = script_dir / "train_config.yaml"

    if args.config is not None:
        config_path = resolve_path(args.config, PROJECT_ROOT)
    elif default_config_path.exists():
        config_path = default_config_path
    else:
        config_path = None

    raw_cfg = load_config(config_path)
    model_cfg = deep_merge(DEFAULT_MODEL_CONFIG, raw_cfg.get("model", {}))
    training_cfg = deep_merge(DEFAULT_TRAINING_CONFIG, raw_cfg.get("training", {}))

    num_labels: int = int(model_cfg["num_labels"])
    max_seq_length: int = int(model_cfg["max_seq_length"])

    # ---- Resolve paths ----
    model_path = resolve_path(args.model_path, PROJECT_ROOT)
    train_data = resolve_path(args.train_data, PROJECT_ROOT)
    val_data = resolve_path(args.val_data, PROJECT_ROOT)

    # Resolve metadata: explicit --metadata > same dir as train parquet
    if args.metadata is not None:
        metadata_path = resolve_path(args.metadata, PROJECT_ROOT)
    else:
        metadata_path = train_data.parent / "metadata.json"
    if not metadata_path.exists():
        logger.warning(
            "metadata.json not found at %s; class weights disabled", metadata_path
        )
        metadata_path = None

    output_dir: Path = (
        resolve_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir
        else resolve_path(
            training_cfg.get("output_dir", DEFAULT_OUTPUT_DIR), PROJECT_ROOT
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project root  : %s", PROJECT_ROOT)
    logger.info("Model path    : %s", model_path)
    logger.info("Train data    : %s", train_data)
    logger.info("Val data      : %s", val_data)
    logger.info("Metadata      : %s", metadata_path)
    logger.info("Output dir    : %s", output_dir)
    logger.info(
        "Softmask      : %s",
        "sensitive (Model B)" if args.softmask_sensitive else "insensitive (Model A)",
    )
    logger.info("Num labels    : %d", num_labels)
    logger.info("Max seq length: %d", max_seq_length)

    # ---- Load tokenizer & model ----
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(
        "Tokenizer loaded  vocab_size=%d  pad_token_id=%s",
        len(tokenizer),
        tokenizer.pad_token_id,
    )

    _model_cfg = json.loads((model_path / "config.json").read_text()) if (model_path / "config.json").exists() else {}
    _no_attn_models = {"mamba", "caduceus"}
    _has_attention = _model_cfg.get("model_type", "") not in _no_attn_models
    load_kwargs: dict = dict(
        trust_remote_code=True,
        use_cache=False,
    )
    if _has_attention:
        load_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        **load_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---- Patch attention to use SDPA (memory-efficient for 8K seq) ----
    _patch_attention_to_sdpa(model)

    # ---- Disable gradient checkpointing for models that don't support it ----
    if training_cfg.get("gradient_checkpointing", False) and not getattr(
        model, "supports_gradient_checkpointing", False
    ):
        logger.warning(
            "gradient_checkpointing=true in config but %s does not support it; disabling",
            type(model).__name__,
        )
        training_cfg["gradient_checkpointing"] = False

    # ---- Softmask vocab expansion (Model B) ----
    if args.softmask_sensitive:
        expand_vocab_softmask(tokenizer, model)

    # ---- Initialise classifier head ----
    torch.manual_seed(training_cfg.get("seed", 42))
    classifier = getattr(model, "classifier", None)
    if classifier is not None and hasattr(classifier, "weight"):
        torch.nn.init.xavier_uniform_(classifier.weight)
        if classifier.bias is not None:
            torch.nn.init.zeros_(classifier.bias)
        logger.info("Classifier head initialised with Xavier uniform")
    else:
        logger.warning("Could not find classifier layer for Xavier init")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters     : %s", f"{total_params:,}")
    logger.info("Trainable parameters : %s", f"{trainable_params:,}")

    # ---- Load datasets ----
    logger.info("Loading training dataset ...")
    train_dataset = GenomeTokenDataset(
        parquet_path=train_data,
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )
    logger.info("Loading validation dataset ...")
    val_dataset = GenomeTokenDataset(
        parquet_path=val_data,
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        lr_scheduler_kwargs=training_cfg.get("lr_scheduler_kwargs"),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        bf16=bool(training_cfg.get("bf16", False)),
        logging_steps=int(training_cfg["logging_steps"]),
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=int(training_cfg["eval_steps"]),
        save_strategy=training_cfg["save_strategy"],
        save_steps=int(training_cfg["save_steps"]),
        save_total_limit=int(training_cfg["save_total_limit"]),
        load_best_model_at_end=bool(training_cfg["load_best_model_at_end"]),
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=bool(training_cfg.get("greater_is_better", True)),
        report_to=training_cfg["report_to"],
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        dataloader_num_workers=int(training_cfg["dataloader_num_workers"]),
        remove_unused_columns=False,
        seed=int(training_cfg.get("seed", 42)),
        eval_accumulation_steps=int(training_cfg.get("eval_accumulation_steps", 4)),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=int(training_cfg.get("dataloader_prefetch_factor", 2)),
    )

    # ---- Class weights ----
    cw_cfg = raw_cfg.get("class_weight", {})
    weight_strategy: str = cw_cfg.get(
        "strategy", training_cfg.get("class_weight_strategy", "sqrt_inverse_freq")
    )
    weight_cap: float = float(
        cw_cfg.get("cap", training_cfg.get("class_weight_cap", 100.0))
    )
    manual_weights: dict[str, float] | None = cw_cfg.get("manual_weights")

    class_weights: torch.Tensor | None = None
    if weight_strategy != "none":
        if weight_strategy == "manual" and manual_weights is None:
            logger.warning(
                "class_weight.strategy='manual' but no manual_weights provided; falling back to 'none'"
            )
        elif weight_strategy == "manual":
            class_weights = compute_class_weights(
                strategy="manual",
                cap=weight_cap,
                manual_weights=manual_weights,
            )
        elif metadata_path is not None:
            class_weights = compute_class_weights(
                strategy=weight_strategy,
                cap=weight_cap,
                metadata_path=metadata_path,
            )
        else:
            logger.warning("metadata.json not found; class weights disabled")

    # ---- Metrics config ----
    metrics_cfg = raw_cfg.get("metrics", {})
    f1_strategy: str = metrics_cfg.get("f1_strategy", "macro_f1")
    f1_weights: dict[str, float] | None = metrics_cfg.get("f1_weights")

    if f1_strategy == "weighted_f1" and "metric_for_best_model" not in training_cfg:
        metric_for_best = "weighted_f1"
    else:
        metric_for_best = training_cfg.get("metric_for_best_model", "macro_f1")
    training_args.metric_for_best_model = metric_for_best

    metrics_fn = functools.partial(
        compute_metrics,
        f1_strategy=f1_strategy,
        f1_weights=f1_weights if f1_strategy == "weighted_f1" else None,
    )

    # ---- CRF ----
    crf_cfg = raw_cfg.get("crf", {})
    crf_enabled: bool = bool(crf_cfg.get("enabled", False))
    crf_module: GeneStructureCRF | None = None
    if crf_enabled:
        crf_module = GeneStructureCRF(
            num_labels=num_labels,
            enforce_gene_order=bool(crf_cfg.get("enforce_gene_order", True)),
            allow_utr_introns=bool(crf_cfg.get("allow_utr_introns", True)),
            cds_mod3_penalty=float(crf_cfg.get("cds_mod3_penalty", 0.0)),
        )
        crf_module = crf_module.to(model.device)
        logger.info(
            "CRF enabled (gene_order=%s, utr_introns=%s, cds_mod3_penalty=%.3f)",
            crf_cfg.get("enforce_gene_order", True),
            crf_cfg.get("allow_utr_introns", True),
            crf_cfg.get("cds_mod3_penalty", 0.0),
        )
        crf_params = sum(p.numel() for p in crf_module.parameters())
        logger.info("CRF parameters: %s", f"{crf_params:,}")

        n_allowed = int(crf_module.transition_mask.sum().item())
        n_total = NUM_LABELS * NUM_LABELS
        logger.info(
            "Transition matrix: %d/%d allowed (%.1f%% forbidden)",
            n_allowed,
            n_total,
            100.0 * (1.0 - n_allowed / n_total),
        )

        if train_data.exists():
            validate_transition_matrix(crf_module.transition_mask, train_data)

    # ---- Trainer ----
    eval_num_workers = int(training_cfg.get("eval_dataloader_num_workers", training_cfg.get("dataloader_num_workers", 4)))
    trainer = WeightedTokenTrainer(
        class_weights=class_weights,
        crf=crf_module,
        eval_num_workers=eval_num_workers,
        f1_strategy=f1_strategy,
        f1_weights=f1_weights if f1_strategy == "weighted_f1" else None,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics_fn,
    )

    # ---- Train ----
    logger.info("Starting training ...")
    train_result = trainer.train()
    trainer.save_model()

    # ---- Save training metrics ----
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ---- Final evaluation ----
    logger.info("Running final evaluation ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ---- Save model & tokenizer ----
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Final model + tokenizer saved to %s", final_dir)

    # ---- Save CRF state dict ----
    if crf_module is not None:
        crf_path = output_dir / "crf.pt"
        torch.save(crf_module.state_dict(), crf_path)
        logger.info("CRF state dict saved to %s", crf_path)

    # ---- Persist training summary ----
    metrics_json_path = output_dir / "training_metrics.json"
    all_metrics: dict = {
        "train": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        },
        "eval": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in eval_metrics.items()
        },
        "config": {
            "model_path": str(model_path),
            "train_data": str(train_data),
            "val_data": str(val_data),
            "num_labels": num_labels,
            "max_seq_length": max_seq_length,
            "softmask_sensitive": args.softmask_sensitive,
            "class_weight_strategy": weight_strategy,
            "f1_strategy": f1_strategy,
            "crf_enabled": crf_enabled,
        },
    }
    with open(metrics_json_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2, ensure_ascii=False)
    logger.info("Training metrics saved to %s", metrics_json_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
