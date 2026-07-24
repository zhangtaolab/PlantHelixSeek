#!/usr/bin/env python3
"""Gene prediction NPZ to GFF3 decoder with PlantHelixSeek-style Viterbi decoding.

Supports three decoding methods:
  - "viterbi": PlantHelixSeek-style Viterbi on 5-class feature probs
  - "viterbi+orf": Viterbi + V7 ORF correction (default, best results)
  - "heuristic": Original threshold-based decoder (backward compatible)

Includes PlantHelixSeek-style post-processing filters and improved GFF3 output
with proper ID/Parent hierarchy.

Usage:
    python pred_to_gff3.py \
        --npz-dir predictions/ \
        --genome-fasta genome.fa \
        --decoding-method viterbi+orf \
        --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Bio import SeqIO

from viterbi_utils import (
    viterbi_decode,
    token_transition_probs,
    convert_entity_labels_to_intervals,
    aggregate_logits_np,
    convert_bilou_to_entity_labels,
    INTERVAL_ENTITY_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CDS_IDX = [1, 2, 3, 4]
INTRON_IDX = [5, 6, 7, 8]
UTR5_IDX = [9, 10, 11, 12]
UTR3_IDX = [13, 14, 15, 16]
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Heuristic decoder constants (kept for backward compatibility)
CDS_ENTRY = 0.10
CDS_CORE = 0.30
MIN_RUN = 5  # min CDS run length; 5 recovers short exons missed at 8
MIN_GENE_CDS = 36
MIN_GENE_SPAN = 80
MERGE_GAP = 800
INTRON_THRESH = 0.30
INTRON_MIN = 20
MIN_INTRON_LEN = 15
SMOOTH_W = 9
EDGE_TRIM = 0.15
MIN_ORF_AA = 15
ATG_WINDOW = 150
STOP_EXTEND_WINDOW = 150
STOP_EXTEND_MIN_PROB = 0.10
MERGE_INTRON_PROB = 0.60
MERGE_INTRON_MIN_GAP = 50
MIN_CDS_FEATURE = 6
MIN_UTR_FEATURE = 3
MIN_GENE_LEN = 30

FRAME_SHIFTS = tuple(range(-18, 19))


@dataclass
class Exon:
    start: int
    end: int
    strand: str


@dataclass
class Gene:
    gid: str
    chrom: str
    strand: str
    exons: list[Exon] = field(default_factory=list)
    utr5: list[Exon] = field(default_factory=list)
    utr3: list[Exon] = field(default_factory=list)

    @property
    def start(self):
        all_s = [e.start for e in self.exons] + [e.start for e in self.utr5] + [e.start for e in self.utr3]
        return min(all_s) if all_s else min(e.start for e in self.exons)

    @property
    def end(self):
        all_e = [e.end for e in self.exons] + [e.end for e in self.utr5] + [e.end for e in self.utr3]
        return max(all_e) if all_e else max(e.end for e in self.exons)

    @property
    def cds_len(self):
        return sum(e.end - e.start for e in self.exons)


# =========================================================================
# I/O utilities
# =========================================================================

def read_fasta(path: Path) -> dict[str, str]:
    seqs = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as h:
        for r in SeqIO.parse(h, "fasta"):
            seqs[r.id] = str(r.seq).upper()
    return seqs


def rc(s: str) -> str:
    return s.translate(str.maketrans("ATGC", "TACG"))[::-1]


# =========================================================================
# Viterbi decoding (PlantHelixSeek-style)
# =========================================================================

def decode_viterbi(
    chrom: str,
    feature_probs: np.ndarray,
    bilou_labels: np.ndarray,
    bilou_probs: np.ndarray,
    strand: str,
    gc: int,
    domain: str = "plant",
    viterbi_alpha: float = 0.0,
    min_cds_run: int = MIN_RUN,
    genic_threshold: float = 0.25,
    intergenic_boost: float = 0.0,
) -> tuple[list[Gene], int]:
    """Use CRF BILUO labels for exon boundaries, with 5-class Viterbi filtering."""
    probs = feature_probs.astype(np.float64).copy()

    if intergenic_boost > 0:
        probs[:, 0] = np.minimum(probs[:, 0] + intergenic_boost * 0.05, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
    elif intergenic_boost < 0:
        probs[:, 0] = np.maximum(probs[:, 0] + intergenic_boost * 0.05, 1e-10)
        probs = probs / probs.sum(axis=1, keepdims=True)

    transition_df = token_transition_probs(
        remove_incomplete_features=True, domain=domain
    )
    transition_matrix = transition_df.values

    viterbi_labels = viterbi_decode(
        emission_probs=probs,
        transition_matrix=transition_matrix,
        alpha=viterbi_alpha,
    )

    genic_mask = viterbi_labels != 0
    genic_fraction = float(genic_mask.mean())
    use_genic_filter = genic_fraction > 0.01

    if not use_genic_filter:
        genic_mask = np.ones(len(probs), dtype=bool)

    entity_labels = convert_bilou_to_entity_labels(bilou_labels)

    bilou_intervals = convert_entity_labels_to_intervals(
        labels=entity_labels, class_groups=INTERVAL_ENTITY_CLASSES
    )

    cds_intervals = bilou_intervals[bilou_intervals["entity"] == 5]

    genes = []
    for _, row in cds_intervals.iterrows():
        s, e = int(row["start"]), int(row["stop"]) + 1
        if e - s < min_cds_run:
            continue
        genic_frac = float(genic_mask[s:e].mean())
        if genic_frac < genic_threshold:
            continue
        gc += 1
        genes.append(Gene(f"gene_{gc:06d}", chrom, strand, [Exon(s, e, strand)]))

    genes = _merge_adjacent_cds_viterbi(genes, bilou_intervals, probs, min_cds_run)
    return genes, gc


def _merge_adjacent_cds_viterbi(
    genes: list[Gene],
    intervals: pd.DataFrame,
    probs: np.ndarray,
    min_cds_run: int,
) -> list[Gene]:
    """Merge adjacent CDS intervals connected by intron labels."""
    import pandas as pd
    if len(genes) <= 1:
        return genes

    intron_intervals = intervals[intervals["entity"] == 3]
    if intron_intervals.empty:
        return genes

    genes_by_strand = [g for g in genes if g.strand == "+"]
    if not genes_by_strand:
        genes_by_strand = [g for g in genes if g.strand == "-"]

    strands = set(g.strand for g in genes)
    result = []
    for strand in strands:
        strand_genes = sorted(
            [g for g in genes if g.strand == strand], key=lambda g: g.start
        )
        merged = []
        i = 0
        while i < len(strand_genes):
            current = strand_genes[i]
            j = i + 1
            while j < len(strand_genes):
                nxt = strand_genes[j]
                cur_end = max(e.end for e in current.exons)
                nxt_start = min(e.start for e in nxt.exons)
                gap = nxt_start - cur_end
                if gap < 0 or gap > 50000:
                    break

                gap_intron_prob = float(probs[cur_end:nxt_start, 1].mean())
                if gap_intron_prob > 0.3:
                    current.exons = sorted(
                        current.exons + nxt.exons, key=lambda e: e.start
                    )
                    j += 1
                else:
                    break
            merged.append(current)
            i = j
        result.extend(merged)

    return result if result else genes


# =========================================================================
# Heuristic decoder (original V7, kept for backward compatibility)
# =========================================================================

def smooth(a: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(a, np.ones(w) / w, mode="same") if w > 1 else a


def find_runs(cp_s: np.ndarray, cp_r: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    pos, n = 0, len(cp_s)
    while pos < n:
        if cp_s[pos] < CDS_ENTRY:
            pos += 1
            continue
        start = pos
        has_core = False
        while pos < n and cp_s[pos] >= CDS_ENTRY:
            if cp_r[pos] >= CDS_CORE:
                has_core = True
            pos += 1
        if pos - start >= MIN_RUN and has_core:
            runs.append((start, pos))
    return runs


def trim_edges(run: tuple[int, int], cp: np.ndarray) -> tuple[int, int]:
    s, e = run
    while s < e and cp[s] < EDGE_TRIM:
        s += 1
    while e > s and cp[e - 1] < EDGE_TRIM:
        e -= 1
    return (s, e)


def has_splice(seq: str, s: int, e: int, strand: str) -> bool:
    if e - s < INTRON_MIN or s < 0 or e > len(seq):
        return False
    if strand == "+":
        return seq[s : s + 2] == "GT" and seq[e - 2 : e] == "AG"
    return seq[s : s + 2] == "CT" and seq[e - 2 : e] == "AC"


LONG_INTRON_PROB = 0.80
MAX_LONG_GAP = 2000


def merge_clusters(runs, ip, seq, strand):
    if not runs:
        return []
    clusters: list[list[tuple[int, int]]] = [[runs[0]]]
    for run in runs[1:]:
        pe = clusters[-1][-1][1]
        gap = run[0] - pe
        if gap <= 0:
            last = clusters[-1][-1]
            clusters[-1][-1] = (last[0], max(last[1], run[1]))
            continue
        ip_m = float(ip[pe:run[0]].mean()) if pe < len(ip) else 0.0
        sp = has_splice(seq, pe, run[0], strand)
        if gap <= MERGE_GAP:
            if ip_m >= INTRON_THRESH or sp or gap <= 12:
                clusters[-1].append(run)
            elif gap <= 80 and ip_m >= 0.15:
                clusters[-1].append(run)
            else:
                clusters.append([run])
        elif gap <= MAX_LONG_GAP and (ip_m >= LONG_INTRON_PROB or sp):
            clusters[-1].append(run)
        else:
            clusters.append([run])
    return clusters


def resolve_strand_conflicts(fwd_probs, rev_probs, genes_f, genes_r):
    fwd_dominant = fwd_probs[:, CDS_IDX].sum(axis=1) > rev_probs[:, CDS_IDX].sum(axis=1)
    result = []
    for gene in genes_f + genes_r:
        dominant = fwd_dominant if gene.strand == "+" else ~fwd_dominant
        new_exons = []
        for exon in gene.exons:
            s, e = exon.start, exon.end
            if int(dominant[s:e].sum()) / max(e - s, 1) < 0.3:
                continue
            while s < e and not dominant[s]:
                s += 1
            while e > s and not dominant[e - 1]:
                e -= 1
            if e - s >= 6:
                new_exons.append(Exon(s, e, gene.strand))
        if sum(e.end - e.start for e in new_exons) >= MIN_GENE_CDS:
            gene.exons = new_exons
            result.append(gene)
    return result


def bilou_cds_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    pos, n = 0, len(labels)
    while pos < n:
        lbl = labels[pos]
        if lbl in (1, 4):
            start = pos
            pos += 1
            while pos < n and labels[pos] == 2:
                pos += 1
            if pos < n and labels[pos] == 3:
                pos += 1
            spans.append((start, pos))
        else:
            pos += 1
    return spans


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _has_bilou_intron(labels, start, end):
    for i in range(start, min(end, len(labels))):
        if labels[i] in (5, 6, 7, 8):
            return True
    return False


def _split_cluster_on_intergenic(cl, labels):
    if len(cl) <= 1:
        return [cl]
    result = []
    current = [cl[0]]
    for run in cl[1:]:
        prev_end = current[-1][1]
        gap_start = prev_end
        gap_end = run[0]
        gap_len = gap_end - gap_start
        if gap_len <= 0:
            current.append(run)
            continue
        gap_has_intron = _has_bilou_intron(labels, gap_start, gap_end)
        non_cds = sum(
            1 for i in range(gap_start, min(gap_end, len(labels)))
            if labels[i] not in (1, 2, 3, 4)
        )
        o_frac = (
            sum(1 for i in range(gap_start, min(gap_end, len(labels))) if labels[i] == 0)
            / gap_len
        )
        if non_cds / gap_len > 0.95 and (not gap_has_intron or o_frac > 0.20):
            result.append(current)
            current = [run]
        else:
            current.append(run)
    result.append(current)
    return result


def decode_heuristic(
    chrom, probs, seq, strand, gc, labels=None
) -> tuple[list[Gene], int]:
    cp = probs[:, CDS_IDX].sum(axis=1)
    ip = probs[:, INTRON_IDX].sum(axis=1)
    cp_s = smooth(cp, SMOOTH_W)

    runs = []
    for r in find_runs(cp_s, cp):
        t = trim_edges(r, cp)
        if t[1] - t[0] >= MIN_RUN:
            runs.append(t)

    runs.sort()
    genes = []
    for cl in merge_clusters(runs, ip, seq, strand):
        if labels is not None and len(cl) > 1:
            sub_clusters = _split_cluster_on_intergenic(cl, labels)
        else:
            sub_clusters = [cl]
        for sc in sub_clusters:
            total = sum(e - s for s, e in sc)
            span = max(e for s, e in sc) - min(s for s, e in sc)
            if total < MIN_GENE_CDS or span < MIN_GENE_SPAN:
                continue
            gc += 1
            genes.append(
                Gene(f"gene_{gc:06d}", chrom, strand, [Exon(s, e, strand) for s, e in sc])
            )
    return genes, gc


# =========================================================================
# ORF-first post-processing (V7, preserved)
# =========================================================================

def get_cds_seq(exons: list[Exon], seq: str, strand: str) -> str:
    g = "".join(seq[e.start : e.end] for e in sorted(exons, key=lambda x: x.start))
    return rc(g) if strand == "-" else g


def find_longest_orf(cds_seq: str, min_aa: int = MIN_ORF_AA) -> tuple[int, int] | None:
    best_start, best_end = -1, -1
    best_len = 0
    for frame in range(3):
        pos = frame
        while pos + 3 <= len(cds_seq):
            if cds_seq[pos : pos + 3] == "ATG":
                orf_end = pos + 3
                while orf_end + 3 <= len(cds_seq):
                    codon = cds_seq[orf_end : orf_end + 3]
                    if codon in STOP_CODONS:
                        orf_len = orf_end - pos
                        if orf_len >= min_aa * 3 and orf_len > best_len:
                            best_start = pos
                            best_end = orf_end + 3
                            best_len = orf_len
                        break
                    orf_end += 3
                else:
                    orf_len = orf_end - pos
                    remainder = orf_len % 3
                    if remainder:
                        orf_end -= remainder
                        orf_len = orf_end - pos
                    if orf_len >= min_aa * 3 and orf_len > best_len:
                        best_start = pos
                        best_end = orf_end
                        best_len = orf_len
            pos += 3
    if best_start < 0:
        return None
    return (best_start, best_end)


def cds_pos_to_genomic(cds_pos: int, sorted_exons: list[Exon], strand: str) -> int:
    total_cds = sum(e.end - e.start for e in sorted_exons)
    if strand == "-":
        forward_pos = total_cds - 1 - cds_pos
    else:
        forward_pos = cds_pos
    offset = 0
    for exon in sorted_exons:
        exon_len = exon.end - exon.start
        if offset + exon_len > forward_pos:
            return exon.start + (forward_pos - offset)
        offset += exon_len
    return sorted_exons[-1].end


def extract_orf_exons(
    orf_start_cds: int, orf_end_cds: int, exons: list[Exon], strand: str
) -> list[Exon]:
    sorted_exons = sorted(exons, key=lambda e: e.start)
    if strand == "+":
        gen_start = cds_pos_to_genomic(orf_start_cds, sorted_exons, "+")
        gen_end = cds_pos_to_genomic(orf_end_cds - 1, sorted_exons, "+") + 1
    else:
        gen_start = cds_pos_to_genomic(orf_end_cds - 1, sorted_exons, "-")
        gen_end = cds_pos_to_genomic(orf_start_cds, sorted_exons, "-") + 1
    new_exons = []
    for exon in sorted_exons:
        overlap_start = max(exon.start, gen_start)
        overlap_end = min(exon.end, gen_end)
        if overlap_end - overlap_start >= 1:
            new_exons.append(Exon(overlap_start, overlap_end, strand))
    return new_exons


def try_extend_atg(
    cds_seq: str, exons: list[Exon], seq: str, strand: str
) -> tuple[str, list[Exon]]:
    if cds_seq[:3] == "ATG":
        return cds_seq, exons
    sorted_exons = sorted(exons, key=lambda e: e.start)
    if strand == "+":
        first_start = sorted_exons[0].start
        search_start = max(0, first_start - ATG_WINDOW)
        for pos in range(first_start - 3, search_start - 1, -3):
            if pos >= 0 and pos + 3 <= len(seq) and seq[pos : pos + 3] == "ATG":
                new_first = Exon(pos, sorted_exons[0].end, strand)
                new_exons = [new_first] + sorted_exons[1:]
                return get_cds_seq(new_exons, seq, strand), new_exons
    else:
        last_end = sorted_exons[-1].end
        search_end = min(len(seq), last_end + ATG_WINDOW)
        for pos in range(last_end, min(search_end, len(seq) - 2)):
            if rc(seq[pos : pos + 3]) == "ATG":
                new_last = Exon(sorted_exons[-1].start, pos + 3, strand)
                new_exons = sorted_exons[:-1] + [new_last]
                return get_cds_seq(new_exons, seq, strand), new_exons
    return cds_seq, exons


def _try_shifted_exons(
    exons: list[Exon], seq: str, strand: str
) -> tuple[list[Exon], int] | None:
    sorted_exons = sorted(exons, key=lambda e: e.start)
    if len(sorted_exons) < 2:
        return None
    n_pairs = len(sorted_exons) - 1
    trial = list(sorted_exons)
    baseline_cds = get_cds_seq(trial, seq, strand)
    baseline_orf = find_longest_orf(baseline_cds, min_aa=MIN_ORF_AA)
    if baseline_orf is None:
        return None
    best_orf_aa = (baseline_orf[1] - baseline_orf[0]) // 3
    improved = True
    while improved:
        improved = False
        for bi in range(n_pairs):
            best_shift = 0
            best_aa_here = best_orf_aa
            for shift in FRAME_SHIFTS:
                if shift == 0:
                    continue
                test = list(trial)
                left = test[bi]
                right = test[bi + 1]
                new_left_end = left.end + shift
                new_right_start = right.start + shift
                if new_left_end <= left.start + 3 or new_right_start >= right.end - 3:
                    continue
                if new_right_start - new_left_end < MIN_INTRON_LEN:
                    continue
                test[bi] = Exon(left.start, new_left_end, strand)
                test[bi + 1] = Exon(new_right_start, right.end, strand)
                cds = get_cds_seq(test, seq, strand)
                orf = find_longest_orf(cds, min_aa=MIN_ORF_AA)
                if orf is None:
                    continue
                if cds[orf[0] : orf[0] + 3] != "ATG":
                    continue
                orf_aa = (orf[1] - orf[0]) // 3
                if orf_aa > best_aa_here:
                    best_aa_here = orf_aa
                    best_shift = shift
            if best_shift != 0:
                left = trial[bi]
                right = trial[bi + 1]
                trial[bi] = Exon(left.start, left.end + best_shift, strand)
                trial[bi + 1] = Exon(right.start + best_shift, right.end, strand)
                best_orf_aa = best_aa_here
                improved = True
    if best_orf_aa > (baseline_orf[1] - baseline_orf[0]) // 3:
        return trial, best_orf_aa
    return None


def correct_gene_orf(gene: Gene, seq: str) -> Gene | None:
    sorted_exons = sorted(gene.exons, key=lambda e: e.start)
    n_exons = len(sorted_exons)
    cds_seq = get_cds_seq(sorted_exons, seq, gene.strand)
    cds_seq, sorted_exons = try_extend_atg(cds_seq, sorted_exons, seq, gene.strand)
    orf = find_longest_orf(cds_seq, min_aa=MIN_ORF_AA)
    if n_exons >= 2:
        shifted = _try_shifted_exons(sorted_exons, seq, gene.strand)
        if shifted is not None:
            shifted_exons, shifted_orf_aa = shifted
            current_orf_aa = ((orf[1] - orf[0]) // 3) if orf else 0
            if shifted_orf_aa > current_orf_aa:
                sorted_exons = shifted_exons
                cds_seq = get_cds_seq(sorted_exons, seq, gene.strand)
                orf = find_longest_orf(cds_seq, min_aa=MIN_ORF_AA)
    if orf is None:
        return None
    orf_start, orf_end = orf
    if cds_seq[orf_start : orf_start + 3] != "ATG":
        return None
    new_exons = extract_orf_exons(orf_start, orf_end, sorted_exons, gene.strand)
    if not new_exons:
        return None
    total_cds = sum(e.end - e.start for e in new_exons)
    if total_cds < MIN_ORF_AA * 3:
        return None
    final_cds = get_cds_seq(new_exons, seq, gene.strand)
    if final_cds[:3] != "ATG":
        return None
    check_end = len(final_cds)
    if len(final_cds) >= 3 and final_cds[-3:] in STOP_CODONS:
        check_end = len(final_cds) - 3
    has_internal_stop = any(
        final_cds[p : p + 3] in STOP_CODONS for p in range(0, max(0, check_end - 2), 3)
    )
    if has_internal_stop:
        return None
    total = sum(e.end - e.start for e in new_exons)
    rem = total % 3
    if rem and gene.strand == "+":
        last = new_exons[-1]
        if last.end - last.start > rem:
            new_exons[-1] = Exon(last.start, last.end - rem, gene.strand)
    elif rem and gene.strand == "-":
        first = new_exons[0]
        if first.end - first.start > rem:
            new_exons[0] = Exon(first.start + rem, first.end, gene.strand)
    gene.exons = new_exons
    return gene


def _merge_gene_short_introns(gene: Gene) -> Gene:
    exs = sorted(gene.exons, key=lambda e: e.start)
    if len(exs) <= 1:
        return gene
    filtered = [exs[0]]
    for i in range(1, len(exs)):
        gap = exs[i].start - filtered[-1].end
        if gap < MIN_INTRON_LEN:
            prev_len = filtered[-1].end - filtered[-1].start
            curr_len = exs[i].end - exs[i].start
            if curr_len > prev_len:
                filtered[-1] = exs[i]
        else:
            filtered.append(exs[i])
    gene.exons = filtered
    return gene


def try_extend_stop(gene: Gene, seq: str, probs: np.ndarray) -> bool:
    if not gene.exons:
        return False
    cds_seq = get_cds_seq(gene.exons, seq, gene.strand)
    if len(cds_seq) >= 3 and cds_seq[-3:] in STOP_CODONS:
        return False
    sorted_exons = sorted(gene.exons, key=lambda e: e.start)
    if gene.strand == "+":
        search_start = sorted_exons[-1].end
        search_end = min(len(seq), search_start + STOP_EXTEND_WINDOW)
        for pos in range(search_start, search_end - 2, 3):
            if seq[pos : pos + 3] in STOP_CODONS:
                ext_cds_prob = float(
                    probs[search_start : pos + 3, CDS_IDX].sum(axis=1).mean()
                )
                if ext_cds_prob >= STOP_EXTEND_MIN_PROB:
                    sorted_exons[-1] = Exon(
                        sorted_exons[-1].start, pos + 3, gene.strand
                    )
                    gene.exons = sorted_exons
                    return True
                break
    else:
        search_end = sorted_exons[0].start
        search_start = max(0, search_end - STOP_EXTEND_WINDOW)
        for pos in range(search_end - 3, search_start - 1, -3):
            if rc(seq[pos : pos + 3]) in STOP_CODONS:
                ext_cds_prob = float(
                    probs[pos:search_end, CDS_IDX].sum(axis=1).mean()
                )
                if ext_cds_prob >= STOP_EXTEND_MIN_PROB:
                    sorted_exons[0] = Exon(pos, sorted_exons[0].end, gene.strand)
                    gene.exons = sorted_exons
                    return True
                break
    return False


def correct_all_genes(
    genes: list[Gene],
    seq: str,
    fwd_probs: np.ndarray | None = None,
    rev_probs: np.ndarray | None = None,
) -> list[Gene]:
    kept = 0
    dropped = 0
    corrected = []
    for gene in genes:
        result = correct_gene_orf(gene, seq)
        if result is not None:
            corrected.append(result)
            kept += 1
        else:
            dropped += 1
    revalidated = []
    for gene in corrected:
        exs = sorted(gene.exons, key=lambda e: e.start)
        has_short = any(
            exs[i + 1].start - exs[i].end < MIN_INTRON_LEN
            for i in range(len(exs) - 1)
        )
        if has_short:
            gene = _merge_gene_short_introns(gene)
            result = correct_gene_orf(gene, seq)
            if result is not None:
                revalidated.append(result)
            else:
                revalidated.append(gene)
        else:
            revalidated.append(gene)
    if fwd_probs is not None and rev_probs is not None:
        extended = 0
        for gene in revalidated:
            probs = fwd_probs if gene.strand == "+" else rev_probs
            if try_extend_stop(gene, seq, probs):
                extended += 1
        if extended:
            logger.info("Stop codon extension: %d genes extended", extended)
    MIN_CDS_WITH_STOP = 150
    before = len(revalidated)
    revalidated = [
        g
        for g in revalidated
        if (sum(e.end - e.start for e in g.exons) >= MIN_CDS_WITH_STOP)
        or (
            len(get_cds_seq(g.exons, seq, g.strand)) >= 3
            and get_cds_seq(g.exons, seq, g.strand)[-3:] in STOP_CODONS
        )
    ]
    if len(revalidated) < before:
        logger.info(
            "Stop codon filter: %d removed, %d kept",
            before - len(revalidated),
            len(revalidated),
        )
    logger.info("ORF correction: %d kept, %d dropped", kept, dropped)
    return revalidated


def merge_genes_by_intron_prob(
    genes: list[Gene], fwd_probs: np.ndarray, rev_probs: np.ndarray
) -> list[Gene]:
    if not genes:
        return genes
    merged_count = 0
    genes_sorted = sorted(genes, key=lambda g: g.start)
    result: list[Gene] = []
    for strand in ("+", "-"):
        strand_genes = [g for g in genes_sorted if g.strand == strand]
        probs = fwd_probs if strand == "+" else rev_probs
        i = 0
        while i < len(strand_genes):
            current = strand_genes[i]
            j = i + 1
            while j < len(strand_genes):
                next_gene = strand_genes[j]
                cur_exons = sorted(current.exons, key=lambda e: e.start)
                nxt_exons = sorted(next_gene.exons, key=lambda e: e.start)
                gap_start = cur_exons[-1].end
                gap_end = nxt_exons[0].start
                gap_len = gap_end - gap_start
                if gap_len < MERGE_INTRON_MIN_GAP:
                    break
                gap_intron_prob = float(
                    probs[gap_start:gap_end, INTRON_IDX].sum(axis=1).mean()
                )
                if gap_intron_prob >= MERGE_INTRON_PROB:
                    current.exons = sorted(
                        current.exons + next_gene.exons, key=lambda e: e.start
                    )
                    merged_count += 1
                    j += 1
                else:
                    break
            result.append(current)
            i = j
    if merged_count:
        logger.info("Gene merge by intron prob: %d pairs merged", merged_count)
    return result


def merge_genes_viterbi_guided(
    genes: list[Gene],
    fwd_feature_probs: np.ndarray,
    rev_feature_probs: np.ndarray,
    domain: str = "plant",
    viterbi_alpha: float = 0.0,
    max_gap: int = 50000,
    min_intron_prob: float = 0.10,
) -> list[Gene]:
    """Aggressively merge adjacent same-strand genes to fix fragmentation.

    Merges same-strand genes when:
    1. Gap between genes is within max_gap (default 50kbp)
    2. AND at least one condition is met:
       a. Mean intron probability in gap >= min_intron_prob
       b. Gap is <= 5000bp (likely a short intron)

    This fixes the gene fragmentation problem where multi-exon genes
    are split into multiple single-exon genes due to poor intron detection.
    """
    if len(genes) <= 1:
        return genes

    genes_sorted = sorted(genes, key=lambda g: g.start)
    merged_count = 0
    result: list[Gene] = []

    for strand in ("+", "-"):
        strand_genes = sorted(
            [g for g in genes_sorted if g.strand == strand],
            key=lambda g: g.start,
        )
        if not strand_genes:
            continue

        probs_17 = fwd_feature_probs if strand == "+" else rev_feature_probs
        intron_probs = probs_17[:, 1] if probs_17.shape[1] == 5 else \
                       probs_17[:, INTRON_IDX].sum(axis=1)

        i = 0
        while i < len(strand_genes):
            current = strand_genes[i]
            j = i + 1
            while j < len(strand_genes):
                nxt = strand_genes[j]
                cur_exons = sorted(current.exons, key=lambda e: e.start)
                nxt_exons = sorted(nxt.exons, key=lambda e: e.start)
                gap_start = cur_exons[-1].end
                gap_end = nxt_exons[0].start
                gap_len = gap_end - gap_start

                if gap_len <= 0 or gap_len > max_gap:
                    break

                gap_intron_prob = float(intron_probs[gap_start:gap_end].mean()) if gap_start < len(intron_probs) else 0

                should_merge = gap_intron_prob >= min_intron_prob or gap_len <= 5000
                if should_merge:
                    current.exons = sorted(
                        current.exons + nxt.exons, key=lambda e: e.start
                    )
                    merged_count += 1
                    j += 1
                else:
                    break
            result.append(current)
            i = j

    if merged_count:
        logger.info("Aggressive gene merge: %d pairs merged (max_gap=%d)", merged_count, max_gap)
    return result


# =========================================================================
# Viterbi+merge optimizations (no ORF correction, gene merge + splice snapping)
# =========================================================================


def merge_genes_for_viterbi(
    genes: list[Gene],
    fwd_feature_probs: np.ndarray,
    rev_feature_probs: np.ndarray,
    min_intron_prob: float = 0.15,
    max_gap: int = 20000,
    short_gap: int = 3000,
    min_gap: int = 50,
) -> list[Gene]:
    """Merge adjacent same-strand genes using feature probabilities.

    Graduated merge strategy:
    1. If gap <= short_gap: merge if intron_prob >= min_intron_prob * 0.5
    2. If short_gap < gap <= max_gap: merge if intron_prob >= min_intron_prob
    3. gap > max_gap or gap < min_gap: never merge

    This is less aggressive than merge_genes_viterbi_guided (heuristic only)
    but more aggressive than merge_genes_by_intron_prob (threshold 0.60).
    """
    if len(genes) <= 1:
        return genes

    genes_sorted = sorted(genes, key=lambda g: g.start)
    merged_count = 0
    result: list[Gene] = []

    for strand in ("+", "-"):
        strand_genes = sorted(
            [g for g in genes_sorted if g.strand == strand],
            key=lambda g: g.start,
        )
        if not strand_genes:
            continue

        probs = fwd_feature_probs if strand == "+" else rev_feature_probs
        # Column 1 of 5-class feature probs = intron
        if probs.shape[1] == 5:
            intron_probs = probs[:, 1]
        else:
            intron_probs = probs[:, INTRON_IDX].sum(axis=1)

        i = 0
        while i < len(strand_genes):
            current = strand_genes[i]
            j = i + 1
            while j < len(strand_genes):
                nxt = strand_genes[j]
                cur_exons = sorted(current.exons, key=lambda e: e.start)
                nxt_exons = sorted(nxt.exons, key=lambda e: e.start)
                gap_start = cur_exons[-1].end
                gap_end = nxt_exons[0].start
                gap_len = gap_end - gap_start

                if gap_len < min_gap or gap_len > max_gap:
                    break

                gap_intron_prob = (
                    float(intron_probs[gap_start:gap_end].mean())
                    if gap_start < len(intron_probs)
                    else 0.0
                )

                # Graduated threshold
                if gap_len <= short_gap:
                    threshold = min_intron_prob * 0.5
                else:
                    threshold = min_intron_prob

                if gap_intron_prob >= threshold:
                    current.exons = sorted(
                        current.exons + nxt.exons, key=lambda e: e.start
                    )
                    merged_count += 1
                    j += 1
                else:
                    break
            result.append(current)
            i = j

    if merged_count:
        logger.info("Viterbi gene merge: %d pairs merged", merged_count)
    return result


def cleanup_genes_light(genes: list[Gene], seq: str) -> list[Gene]:
    """Light gene cleanup without strict ORF requirements.

    - Merge short introns (< 15bp)
    - Fix reading frame (trim to mod 3)
    - Remove genes with internal stop codons in best frame
    - No ATG start requirement
    - No stop codon requirement
    """
    kept = []
    dropped = 0
    for gene in genes:
        exs = sorted(gene.exons, key=lambda e: e.start)
        if not exs:
            dropped += 1
            continue

        # Merge short introns
        if len(exs) > 1:
            filtered = [exs[0]]
            for i in range(1, len(exs)):
                gap = exs[i].start - filtered[-1].end
                if gap < MIN_INTRON_LEN:
                    # Merge: extend the longer exon
                    prev_len = filtered[-1].end - filtered[-1].start
                    curr_len = exs[i].end - exs[i].start
                    if curr_len > prev_len:
                        filtered[-1] = exs[i]
                    else:
                        filtered[-1] = Exon(
                            filtered[-1].start,
                            max(filtered[-1].end, exs[i].end),
                            gene.strand,
                        )
                else:
                    filtered.append(exs[i])
            exs = filtered

        gene.exons = exs
        total_cds = sum(e.end - e.start for e in exs)

        if total_cds < MIN_GENE_CDS:
            dropped += 1
            continue

        # Fix reading frame: trim to multiple of 3
        rem = total_cds % 3
        if rem:
            if gene.strand == "+":
                last = exs[-1]
                if last.end - last.start > rem:
                    exs[-1] = Exon(last.start, last.end - rem, gene.strand)
            else:
                first = exs[0]
                if first.end - first.start > rem:
                    exs[0] = Exon(first.start + rem, first.end, gene.strand)
            gene.exons = exs

        # Check for internal stop codons
        cds_seq = get_cds_seq(gene.exons, seq, gene.strand)
        has_internal_stop = False
        for p in range(0, len(cds_seq) - 2, 3):
            if p + 3 <= len(cds_seq):
                codon = cds_seq[p : p + 3]
                if codon in STOP_CODONS and p < len(cds_seq) - 3:
                    has_internal_stop = True
                    break

        if has_internal_stop:
            # Try alternative frames
            best_frame = 0
            min_stops = 999
            for frame in range(3):
                stops = 0
                for p in range(frame, len(cds_seq) - 5, 3):
                    if cds_seq[p : p + 3] in STOP_CODONS:
                        stops += 1
                if stops < min_stops:
                    min_stops = stops
                    best_frame = frame

            if best_frame > 0 and min_stops == 0:
                # Trim to use best frame
                if gene.strand == "+":
                    first = exs[0]
                    if first.end - first.start > best_frame:
                        exs[0] = Exon(
                            first.start + best_frame, first.end, gene.strand
                        )
                else:
                    last = exs[-1]
                    if last.end - last.start > best_frame:
                        exs[-1] = Exon(last.start, last.end - best_frame, gene.strand)
                gene.exons = exs
            elif min_stops > 2:
                dropped += 1
                continue

        kept.append(gene)

    if dropped:
        logger.info("Light cleanup: %d genes dropped, %d kept", dropped, len(kept))
    return kept


def snap_splice_sites(gene: Gene, seq: str, window: int = 3) -> Gene:
    """Snap internal exon boundaries to canonical GT-AG splice sites.

    For each internal exon boundary (between two consecutive exons):
    - Donor site (exon end): look for GT within ±window downstream
    - Acceptor site (exon start): look for AG within ±window upstream

    Only snap if:
    1. A canonical splice site is found within the window
    2. The shift preserves reading frame (shift is multiple of 3)
    3. The exon remains valid (start < end)

    For minus strand: splice sites are CT...AC on plus strand.
    """
    exs = sorted(gene.exons, key=lambda e: e.start)
    if len(exs) < 2:
        return gene

    seq_len = len(seq)
    changed = False

    for i in range(len(exs) - 1):
        cur = exs[i]
        nxt = exs[i + 1]

        if gene.strand == "+":
            # Donor: end of current exon should have GT right after
            best_donor_shift = 0
            for shift in range(-window, window + 1):
                pos = cur.end + shift
                if 0 <= pos and pos + 2 <= seq_len:
                    if seq[pos : pos + 2] == "GT":
                        # Check shift preserves frame
                        total_shift = sum(
                            e.end - e.start for e in exs[: i + 1]
                        ) + shift
                        if shift == 0 or abs(shift) % 3 == 0:
                            best_donor_shift = shift
                            break
                        elif best_donor_shift == 0:
                            best_donor_shift = shift

            # Acceptor: start of next exon should have AG right before
            best_acc_shift = 0
            for shift in range(-window, window + 1):
                pos = nxt.start + shift
                if 0 <= pos - 2 and pos <= seq_len:
                    if seq[pos - 2 : pos] == "AG":
                        if shift == 0 or abs(shift) % 3 == 0:
                            best_acc_shift = shift
                            break
                        elif best_acc_shift == 0:
                            best_acc_shift = shift

            # Apply shifts
            if best_donor_shift != 0:
                new_end = cur.end + best_donor_shift
                if new_end > cur.start + 3:
                    exs[i] = Exon(cur.start, new_end, gene.strand)
                    changed = True
            if best_acc_shift != 0:
                new_start = nxt.start + best_acc_shift
                if new_start < exs[i].end and new_start < nxt.end - 3:
                    # Ensure no overlap with previous exon
                    if new_start >= exs[i].end:
                        exs[i + 1] = Exon(new_start, nxt.end, gene.strand)
                        changed = True
                elif new_start >= exs[i].end:
                    exs[i + 1] = Exon(new_start, nxt.end, gene.strand)
                    changed = True

        else:
            # Minus strand: donor is at exon start (CT on plus strand)
            # Acceptor is at exon end (AC on plus strand)
            # For minus strand genes, exons are listed in genomic order
            # The "start" of each exon (lower coord) is the 3' end on minus strand
            # The "end" of each exon (higher coord) is the 5' end on minus strand

            # Donor for minus strand: at the start of the next exon (lower coord)
            # On plus strand this should be CT (rev comp of AG)
            # Actually for minus strand: the acceptor (AG on minus strand) is at
            # the exon end (higher genomic coord). On plus strand this is CT.
            # The donor (GT on minus strand) is at exon start (lower genomic coord).
            # On plus strand this is AC.

            # Acceptor at end of current exon: look for CT on plus strand
            best_end_shift = 0
            for shift in range(-window, window + 1):
                pos = cur.end + shift
                if 0 <= pos - 2 and pos <= seq_len:
                    if seq[pos - 2 : pos] == "CT":  # rev comp of AG
                        if shift == 0 or abs(shift) % 3 == 0:
                            best_end_shift = shift
                            break
                        elif best_end_shift == 0:
                            best_end_shift = shift

            # Donor at start of next exon: look for AC on plus strand
            best_start_shift = 0
            for shift in range(-window, window + 1):
                pos = nxt.start + shift
                if 0 <= pos and pos + 2 <= seq_len:
                    if seq[pos : pos + 2] == "AC":  # rev comp of GT
                        if shift == 0 or abs(shift) % 3 == 0:
                            best_start_shift = shift
                            break
                        elif best_start_shift == 0:
                            best_start_shift = shift

            if best_end_shift != 0:
                new_end = cur.end + best_end_shift
                if new_end > cur.start + 3:
                    exs[i] = Exon(cur.start, new_end, gene.strand)
                    changed = True
            if best_start_shift != 0:
                new_start = nxt.start + best_start_shift
                if new_start > exs[i].end and new_start < nxt.end - 3:
                    exs[i + 1] = Exon(new_start, nxt.end, gene.strand)
                    changed = True

    if changed:
        gene.exons = exs
    return gene


def snap_splice_sites_all(
    genes: list[Gene], seq: str, window: int = 3
) -> list[Gene]:
    """Apply splice site snapping to all genes."""
    snapped = 0
    result = []
    for gene in genes:
        original_exons = list(gene.exons)
        gene = snap_splice_sites(gene, seq, window)
        if gene.exons != original_exons:
            snapped += 1
        result.append(gene)
    if snapped:
        logger.info("Splice site snapping: %d genes adjusted", snapped)
    return result


_SPLICE_DONOR_PLUS = {"GT", "GC"}
_SPLICE_ACC_PLUS = {"AG"}
_SPLICE_ACC_MINUS = {"AC", "GC"}
_SPLICE_DONOR_MINUS = {"CT"}


def intron_canonical(seq, left_end, right_start, strand):
    if right_start - left_end < 4 or left_end < 0 or right_start > len(seq):
        return False
    pd = seq[left_end:left_end + 2]
    pa = seq[right_start - 2:right_start]
    if strand == "+":
        return pd in _SPLICE_DONOR_PLUS and pa in _SPLICE_ACC_PLUS
    return pd in _SPLICE_DONOR_MINUS and pa in _SPLICE_ACC_MINUS


def _donor_info(seq, left_end, right_start, strand):
    pd = seq[left_end:left_end + 2]
    pa = seq[right_start - 2:right_start]
    if strand == "+":
        if pd == "GT": return True, True
        if pd == "GC": return True, False
        return False, False
    if pa == "AC": return True, True
    if pa == "GC": return True, False
    return False, False


def _acceptor_info(seq, left_end, right_start, strand):
    pd = seq[left_end:left_end + 2]
    pa = seq[right_start - 2:right_start]
    if strand == "+":
        return (pa == "AG"), (pa == "AG")
    return (pd == "CT"), (pd == "CT")


def fix_splice_sites(gene, seq, window=30):
    exs = sorted(gene.exons, key=lambda e: e.start)
    if len(exs) < 2:
        return gene, 0
    strand = gene.strand
    n_fixed = 0
    for _ in range(len(exs) * 2):
        bad = []
        for i in range(len(exs) - 1):
            d_ok, _ = _donor_info(seq, exs[i].end, exs[i + 1].start, strand)
            a_ok, _ = _acceptor_info(seq, exs[i].end, exs[i + 1].start, strand)
            if not (d_ok and a_ok):
                bad.append(i)
        if not bad:
            break
        progressed = False
        for i in bad:
            d_was, _ = _donor_info(seq, exs[i].end, exs[i + 1].start, strand)
            a_was, _ = _acceptor_info(seq, exs[i].end, exs[i + 1].start, strand)
            total = sum(e.end - e.start for e in exs)
            need = (-total) % 3
            dl_iter = [0] if d_was else range(-window, window + 1)
            dr_iter = [0] if a_was else range(-window, window + 1)
            best = None
            left, right = exs[i], exs[i + 1]
            left_min_end = left.start + 3
            right_max_start = right.end - 3
            for dl in dl_iter:
                ne = left.end + dl
                if ne <= left_min_end:
                    continue
                for dr in dr_iter:
                    ns = right.start + dr
                    if ns >= right_max_start:
                        continue
                    if ns - ne < MIN_INTRON_LEN:
                        continue
                    d_canon, d_major = _donor_info(seq, ne, ns, strand)
                    a_canon, a_major = _acceptor_info(seq, ne, ns, strand)
                    if not (d_canon and a_canon):
                        continue
                    is_major = d_major and a_major
                    mod_ok = ((dl - dr) % 3) == need
                    dl_back = dl == 0 or (strand == "+" and dl > 0) or (strand == "-" and dl < 0)
                    dr_back = dr == 0 or (strand == "+" and dr > 0) or (strand == "-" and dr < 0)
                    all_back = dl_back and dr_back
                    priority = (
                        0 if is_major else 1,
                        0 if mod_ok else 1,
                        0 if all_back else 1,
                        abs(dl) + abs(dr),
                    )
                    if best is None or priority < best[0]:
                        best = (priority, dl, dr)
            if best is not None:
                _, dl, dr = best
                exs[i] = Exon(exs[i].start, exs[i].end + dl, strand)
                exs[i + 1] = Exon(exs[i + 1].start + dr, exs[i + 1].end, strand)
                n_fixed += 1
                progressed = True
                break
        if not progressed:
            break
    gene.exons = exs
    return gene, n_fixed


def fix_splice_sites_all(genes, seq, window=30):
    fixed_genes = fixed_introns = 0
    result = []
    for gene in genes:
        g, n = fix_splice_sites(gene, seq, window)
        if n:
            fixed_genes += 1
            fixed_introns += n
        result.append(g)
    if fixed_introns:
        logger.info("Splice site fix: %d introns in %d genes (window=%d)",
                    fixed_introns, fixed_genes, window)
    return result


# =========================================================================
# UTR detection
# =========================================================================

UTR_SEARCH_WINDOW = 3000


def _extract_utr_spans(
    labels: np.ndarray, region_start: int, region_end: int, utr_indices: list[int]
) -> list[tuple[int, int]]:
    spans = []
    pos = region_start
    while pos < region_end:
        if labels[pos] in utr_indices:
            s = pos
            while pos < region_end and labels[pos] in utr_indices:
                pos += 1
            if pos - s >= 3:
                spans.append((s, pos))
        else:
            pos += 1
    return spans


def detect_all_utrs(
    genes: list[Gene], fwd_labels: np.ndarray, rev_labels: np.ndarray
) -> None:
    occupied_fwd = set()
    occupied_rev = set()
    for g in genes:
        occ = occupied_fwd if g.strand == "+" else occupied_rev
        for e in g.exons:
            occ.update(range(e.start, e.end))
    fwd_genes = sorted(
        [g for g in genes if g.strand == "+"], key=lambda g: g.exons[0].start
    )
    rev_genes = sorted(
        [g for g in genes if g.strand == "-"], key=lambda g: g.exons[0].start
    )

    def _gene_boundaries(sorted_genes):
        return [
            (
                g,
                sorted(g.exons, key=lambda e: e.start)[0].start,
                sorted(g.exons, key=lambda e: e.start)[-1].end,
            )
            for g in sorted_genes
        ]

    def _clip_search(search_start, search_end, gene_cds_start, gene_cds_end, boundaries, gene_idx):
        for i, (_, other_start, other_end) in enumerate(boundaries):
            if i == gene_idx:
                continue
            if other_start < gene_cds_start and other_end > search_start:
                search_start = max(search_start, other_end)
            if other_end > gene_cds_end and other_start < search_end:
                search_end = min(search_end, other_start)
        return search_start, search_end

    for gene_list in [fwd_genes, rev_genes]:
        boundaries = _gene_boundaries(gene_list)
        for idx, (g, _, _) in enumerate(boundaries):
            labels = fwd_labels if g.strand == "+" else rev_labels
            occupied = occupied_fwd if g.strand == "+" else occupied_rev
            exs = sorted(g.exons, key=lambda e: e.start)
            first_cds_start = exs[0].start
            last_cds_end = exs[-1].end
            if g.strand == "+":
                us = max(0, first_cds_start - UTR_SEARCH_WINDOW)
                ds = min(len(labels), last_cds_end + UTR_SEARCH_WINDOW)
                us, _ = _clip_search(
                    us, first_cds_start, first_cds_start, last_cds_end, boundaries, idx
                )
                _, ds = _clip_search(
                    last_cds_end, ds, first_cds_start, last_cds_end, boundaries, idx
                )
                raw5 = _extract_utr_spans(labels, us, first_cds_start, UTR5_IDX)
                raw3 = _extract_utr_spans(labels, last_cds_end, ds, UTR3_IDX)
            else:
                us = min(len(labels), last_cds_end + UTR_SEARCH_WINDOW)
                ds = max(0, first_cds_start - UTR_SEARCH_WINDOW)
                _, us = _clip_search(
                    last_cds_end, us, first_cds_start, last_cds_end, boundaries, idx
                )
                ds, _ = _clip_search(
                    ds, first_cds_start, first_cds_start, last_cds_end, boundaries, idx
                )
                raw5 = _extract_utr_spans(labels, last_cds_end, us, UTR5_IDX)
                raw3 = _extract_utr_spans(labels, ds, first_cds_start, UTR3_IDX)
            g.utr5 = [
                Exon(s, e, g.strand)
                for s, e in raw5
                if not any(p in occupied for p in range(s, e))
            ]
            g.utr3 = [
                Exon(s, e, g.strand)
                for s, e in raw3
                if not any(p in occupied for p in range(s, e))
            ]


# =========================================================================
# PlantHelixSeek-style post-processing filters
# =========================================================================

def filter_to_min_cds_length(genes: list[Gene], min_len: int = 6) -> list[Gene]:
    before = len(genes)
    kept = [
        g
        for g in genes
        if sum(e.end - e.start for e in g.exons) >= min_len
    ]
    removed = before - len(kept)
    if removed:
        logger.info("Min CDS filter: %d removed, %d kept", removed, len(kept))
    return kept


def filter_to_min_gene_length(genes: list[Gene], min_len: int = 30) -> list[Gene]:
    before = len(genes)
    kept = [
        g
        for g in genes
        if (max(e.end for e in g.exons) - min(e.start for e in g.exons)) >= min_len
    ]
    removed = before - len(kept)
    if removed:
        logger.info("Min gene length filter: %d removed, %d kept", removed, len(kept))
    return kept


def filter_to_valid_genes(
    genes: list[Gene], require_utrs: bool = False
) -> list[Gene]:
    before = len(genes)
    kept = [g for g in genes if g.exons]
    removed = before - len(kept)
    if removed:
        logger.info("Valid gene filter: %d removed, %d kept", removed, len(kept))
    return kept


def filter_low_confidence_single_exons(
    genes: list[Gene],
    fwd_probs: np.ndarray,
    rev_probs: np.ndarray,
    min_prob: float = 0.7,
    min_cds_len: int = 200,
) -> list[Gene]:
    """Remove single-exon genes below confidence thresholds.

    A single-exon gene is removed if EITHER:
    - mean CDS probability < min_prob
    - total CDS length < min_cds_len
    """
    if min_prob <= 0 and min_cds_len <= 0:
        return genes
    cds_idx = slice(1, 5)
    kept = []
    removed = 0
    for g in genes:
        if len(g.exons) >= 2:
            kept.append(g)
            continue
        cds_len = sum(e.end - e.start for e in g.exons)
        if min_cds_len > 0 and cds_len < min_cds_len:
            removed += 1
            continue
        if min_prob > 0:
            probs = fwd_probs if g.strand == "+" else rev_probs
            ex = g.exons[0]
            mean_cds = float(probs[ex.start:ex.end, cds_idx].sum(axis=1).mean())
            if mean_cds < min_prob:
                removed += 1
                continue
        kept.append(g)
    if removed:
        logger.info(
            "Single-exon filter (prob>= %.2f, len>= %d): %d removed, %d kept",
            min_prob, min_cds_len, removed, len(kept),
        )
    return kept


# =========================================================================
# GFF3 output
# =========================================================================

def write_gff3(genes: list[Gene], path: Path, source: str = "PlantHelixSeek") -> None:
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        for i, g in enumerate(sorted(genes, key=lambda x: x.start), 1):
            exs = sorted(g.exons, key=lambda e: e.start)
            gene_start = g.start + 1
            gene_end = g.end
            gene_id = f"gene_{i}"
            rna_id = f"{gene_id}.t1"

            f.write(
                f"{g.chrom}\t{source}\tgene\t{gene_start}\t{gene_end}\t.\t{g.strand}\t.\tID={gene_id}\n"
            )
            f.write(
                f"{g.chrom}\t{source}\tmRNA\t{gene_start}\t{gene_end}\t.\t{g.strand}\t.\tID={rna_id};Parent={gene_id}\n"
            )

            for j, ex in enumerate(sorted(g.utr5, key=lambda e: e.start), 1):
                utr_id = f"{rna_id}.five_prime_UTR.{j}"
                f.write(
                    f"{g.chrom}\t{source}\tfive_prime_UTR\t{ex.start + 1}\t{ex.end}\t.\t{g.strand}\t.\tID={utr_id};Parent={rna_id}\n"
                )

            if g.strand == "-":
                phase_map = {}
                ph = 0
                for ex in reversed(exs):
                    phase_map[ex.start] = ph
                    ph = (ph + ex.end - ex.start) % 3
                for j, ex in enumerate(exs, 1):
                    cds_id = f"{rna_id}.CDS.{j}"
                    f.write(
                        f"{g.chrom}\t{source}\tCDS\t{ex.start + 1}\t{ex.end}\t.\t{g.strand}\t{phase_map[ex.start]}\tID={cds_id};Parent={rna_id}\n"
                    )
            else:
                ph = 0
                for j, ex in enumerate(exs, 1):
                    cds_id = f"{rna_id}.CDS.{j}"
                    f.write(
                        f"{g.chrom}\t{source}\tCDS\t{ex.start + 1}\t{ex.end}\t.\t{g.strand}\t{ph}\tID={cds_id};Parent={rna_id}\n"
                    )
                    ph = (ph + ex.end - ex.start) % 3

            for j, ex in enumerate(sorted(g.utr3, key=lambda e: e.start), 1):
                utr_id = f"{rna_id}.three_prime_UTR.{j}"
                f.write(
                    f"{g.chrom}\t{source}\tthree_prime_UTR\t{ex.start + 1}\t{ex.end}\t.\t{g.strand}\t.\tID={utr_id};Parent={rna_id}\n"
                )


# =========================================================================
# Evaluation
# =========================================================================

def evaluate_genes(
    genes: list[Gene], ref_path: Path, chrom: str, rend: int, rstart: int = 0,
) -> dict:
    """Evaluate predicted genes against reference GFF3.

    Returns dict with base-level F1/recall, CDS-level F1/recall, gene recall.
    """
    ref_cds: dict[str, set[int]] = {}
    with open(ref_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.strip().split("\t")
            if len(p) < 9 or p[2] != "CDS" or p[0] != chrom:
                continue
            s, e = int(p[3]), int(p[4])
            if e <= rstart or s > rend:
                continue
            s = max(s, rstart + 1); e = min(e, rend)
            ref_cds.setdefault(p[6], set()).update(range(s - 1, e))

    pred_cds: dict[str, set[int]] = {}
    for g in genes:
        for ex in g.exons:
            pred_cds.setdefault(g.strand, set()).update(range(ex.start, ex.end))

    # Base-level (nucleotide) metrics
    all_ref = set()
    all_pred = set()
    for strand in set(list(ref_cds) + list(pred_cds)):
        rs = ref_cds.get(strand, set())
        ps = pred_cds.get(strand, set())
        all_ref |= rs
        all_pred |= ps

    tp_base = len(all_ref & all_pred)
    fp_base = len(all_pred - all_ref)
    fn_base = len(all_ref - all_pred)

    base_precision = tp_base / (tp_base + fp_base) if (tp_base + fp_base) > 0 else 0
    base_recall = tp_base / (tp_base + fn_base) if (tp_base + fn_base) > 0 else 0
    base_f1 = (
        2 * base_precision * base_recall / (base_precision + base_recall)
        if (base_precision + base_recall) > 0
        else 0
    )

    # CDS-level metrics: count CDS features
    ref_cds_features = set()
    with open(ref_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.strip().split("\t")
            if len(p) < 9 or p[2] != "CDS" or p[0] != chrom:
                continue
            s, e = int(p[3]), int(p[4])
            if e <= rstart or s > rend:
                continue
            strand = p[6]
            ref_cds_features.add((strand, s, e))

    pred_cds_features = set()
    for g in genes:
        strand = g.strand
        for ex in g.exons:
            pred_cds_features.add((strand, ex.start + 1, ex.end))

    tp_cds = len(ref_cds_features & pred_cds_features)
    fp_cds = len(pred_cds_features - ref_cds_features)
    fn_cds = len(ref_cds_features - pred_cds_features)

    cds_precision = tp_cds / (tp_cds + fp_cds) if (tp_cds + fp_cds) > 0 else 0
    cds_recall = tp_cds / (tp_cds + fn_cds) if (tp_cds + fn_cds) > 0 else 0
    cds_f1 = (
        2 * cds_precision * cds_recall / (cds_precision + cds_recall)
        if (cds_precision + cds_recall) > 0
        else 0
    )

    # Gene recall: count genes with matching CDS
    ref_genes = set()
    with open(ref_path) as f:
        current_gene = None
        for line in f:
            if line.startswith("#"):
                continue
            p = line.strip().split("\t")
            if len(p) < 9 or p[0] != chrom:
                continue
            if p[2] == "gene":
                current_gene = p[8]
            elif p[2] == "CDS":
                s, e = int(p[3]), int(p[4])
                if e > rstart and s <= rend and current_gene:
                    ref_genes.add(current_gene)

    predicted_genes_with_cds = sum(1 for g in genes if g.exons)
    gene_recall = (
        predicted_genes_with_cds / len(ref_genes) if ref_genes else 0
    )

    return {
        "base_f1": base_f1,
        "base_recall": base_recall,
        "base_precision": base_precision,
        "cds_f1": cds_f1,
        "cds_recall": cds_recall,
        "cds_precision": cds_precision,
        "gene_recall": gene_recall,
        "pred_genes": predicted_genes_with_cds,
        "ref_genes": len(ref_genes),
        "tp_base": tp_base,
        "fp_base": fp_base,
        "fn_base": fn_base,
    }


# =========================================================================
# Per-chromosome decoder
# =========================================================================

def decode_one_chrom(args_tuple: tuple) -> tuple[str, int, str, dict | None]:
    (
        npz_path, genome_fasta, ref_gff3, decoding_method,
        domain, viterbi_alpha, source, genomic_offset,
        single_exon_prob, single_exon_min_cds,
        genic_threshold, intergenic_boost,
        fix_splice, splice_window, output_dir,
    ) = args_tuple

    npz_path = Path(npz_path)
    genome_fasta = Path(genome_fasta)
    ref_gff3 = Path(ref_gff3) if ref_gff3 else None
    out_dir = Path(output_dir) if output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    chrom_name = npz_path.stem
    output_gff = out_dir / f"{chrom_name}.gff3"

    try:
        d = np.load(npz_path)
        rlen = d["fwd_probabilities"].shape[0]
        seqs = read_fasta(genome_fasta)
        chrom = chrom_name
        if chrom not in seqs:
            for n in seqs:
                if n.replace("Chr", "") == chrom.replace("Chr", ""):
                    chrom = n
                    break
        seq = seqs[chrom][:rlen]

        ref_chrom = chrom_name
        for suffix in ['_0-5M', '_5-10M', '_5M', '_10M']:
            if ref_chrom.endswith(suffix):
                ref_chrom = ref_chrom[:-len(suffix)]
                break

        fwd_labels = d.get("fwd_predictions")
        rev_labels = d.get("rev_predictions")

        gc = 0
        all_genes = []

        fwd_feature = d.get("fwd_feature_probs")
        rev_feature = d.get("rev_feature_probs")
        if fwd_feature is None:
            fwd_feature = aggregate_logits_np(d["fwd_probabilities"])
        if rev_feature is None:
            rev_feature = aggregate_logits_np(d["rev_probabilities"])

        if decoding_method in ("viterbi", "viterbi+orf", "viterbi+merge"):
            gf, gc = decode_viterbi(
                chrom, fwd_feature, fwd_labels, d["fwd_probabilities"],
                "+", gc,
                domain=domain,
                viterbi_alpha=viterbi_alpha,
                genic_threshold=genic_threshold,
                intergenic_boost=intergenic_boost,
            )
            gr, gc = decode_viterbi(
                chrom, rev_feature, rev_labels, d["rev_probabilities"],
                "-", gc,
                domain=domain,
                viterbi_alpha=viterbi_alpha,
                genic_threshold=genic_threshold,
                intergenic_boost=intergenic_boost,
            )
            all_genes = gf + gr

            if decoding_method == "viterbi+orf":
                all_genes = correct_all_genes(
                    all_genes, seq, d["fwd_probabilities"], d["rev_probabilities"]
                )
                all_genes = merge_genes_by_intron_prob(
                    all_genes, d["fwd_probabilities"], d["rev_probabilities"]
                )

            elif decoding_method == "viterbi+merge":
                all_genes = merge_genes_for_viterbi(
                    all_genes, fwd_feature, rev_feature,
                )
                all_genes = cleanup_genes_light(all_genes, seq)
                all_genes = snap_splice_sites_all(all_genes, seq, window=3)
                all_genes = merge_genes_for_viterbi(
                    all_genes, fwd_feature, rev_feature,
                )
                all_genes = snap_splice_sites_all(all_genes, seq, window=3)

        elif decoding_method == "heuristic":
            gf, gc = decode_heuristic(
                chrom, d["fwd_probabilities"], seq, "+", 0, labels=fwd_labels
            )
            gr, gc = decode_heuristic(
                chrom, d["rev_probabilities"], seq, "-", gc, labels=rev_labels
            )
            all_genes = resolve_strand_conflicts(
                d["fwd_probabilities"], d["rev_probabilities"], gf, gr
            )
            all_genes = correct_all_genes(
                all_genes, seq, d["fwd_probabilities"], d["rev_probabilities"]
            )
            all_genes = merge_genes_by_intron_prob(
                all_genes, d["fwd_probabilities"], d["rev_probabilities"]
            )
            all_genes = merge_genes_viterbi_guided(
                all_genes, fwd_feature, rev_feature,
                domain=domain, viterbi_alpha=viterbi_alpha,
            )

        # Apply filters
        all_genes = filter_to_min_cds_length(all_genes, min_len=MIN_CDS_FEATURE)
        all_genes = filter_to_min_gene_length(all_genes, min_len=MIN_GENE_LEN)
        all_genes = filter_to_valid_genes(all_genes)
        all_genes = filter_low_confidence_single_exons(
            all_genes, d["fwd_probabilities"], d["rev_probabilities"],
            min_prob=single_exon_prob, min_cds_len=single_exon_min_cds,
        )

        # UTR detection
        if fwd_labels is not None and rev_labels is not None:
            detect_all_utrs(all_genes, fwd_labels, rev_labels)

        if fix_splice:
            all_genes = fix_splice_sites_all(all_genes, seq, window=splice_window)

        # Apply genomic offset for sub-region evaluations
        if genomic_offset > 0:
            for g in all_genes:
                g.chrom = ref_chrom
                for ex in g.exons:
                    ex.start += genomic_offset
                    ex.end += genomic_offset
                for ex in g.utr5:
                    ex.start += genomic_offset
                    ex.end += genomic_offset
                for ex in g.utr3:
                    ex.start += genomic_offset
                    ex.end += genomic_offset

        write_gff3(all_genes, output_gff, source=source)

        eval_results = None
        if ref_gff3:
            eval_results = evaluate_genes(all_genes, ref_gff3, ref_chrom, rlen + genomic_offset, rstart=genomic_offset)

        return chrom_name, len(all_genes), "OK", eval_results

    except Exception as e:
        logger.exception("Error decoding %s", chrom_name)
        return chrom_name, 0, f"ERROR: {e}", None


# =========================================================================
# Species-level merge with systematic gene naming
# =========================================================================

def _chrom_index(chrom_name):
    """Extract the leading integer from a chromosome name for sorting/numbering.

    'Chr1'  -> 1
    'Chr23' -> 23
    'NC_065877.1' -> 65877 (uses main accession, not version suffix)
    """
    nums = re.findall(r"\d+", str(chrom_name))
    if nums:
        return int(nums[0])
    return 0


def merge_species_gff3(npz_dir, species_prefix, predictor_tag, source):
    """Merge per-chromosome GFF3 files into a single species-level GFF3.

    Gene naming convention: {PREFIX}{CHROM:02d}G{GENE:06d}_{TAG}
      e.g. LP01G000002_PHS  ->  Landoltia punctata, Chr1, 2nd gene (by coordinate)

    - Genes are renumbered per chromosome, sorted by genomic start coordinate.
    - mRNA / CDS IDs are derived from the new gene ID (preserving .tN / .CDS.N suffixes).
    - Output is written to ``npz_dir.parent / f"{npz_dir.name}.gff3"``.

    Returns the path to the merged file (or None if nothing to merge).
    """
    npz_dir = Path(npz_dir)
    gff_files = sorted(npz_dir.glob("*.gff3"))
    if not gff_files:
        logger.warning("Merge: no per-chrom GFF3 found in %s, skip merge.", npz_dir)
        return None

    chrom_genes = {}
    for gff in gff_files:
        chrom = gff.stem
        cur_start = None
        cur_old_id = None
        cur_lines = []
        blocks = chrom_genes.setdefault(chrom, [])

        with open(gff) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9:
                    continue
                ftype = fields[2]
                start = int(fields[3])

                if ftype == "gene":
                    if cur_old_id is not None:
                        blocks.append([cur_start, cur_old_id, cur_lines])
                    m = re.search(r"ID=([^;]+)", fields[8])
                    cur_old_id = m.group(1) if m else f"gene_{len(blocks)+1}"
                    cur_start = start
                    cur_lines = [fields]
                elif cur_lines:
                    cur_lines.append(fields)
            if cur_old_id is not None:
                blocks.append([cur_start, cur_old_id, cur_lines])

    merged_path = npz_dir.parent / f"{npz_dir.name}.gff3"
    total_genes = 0
    with open(merged_path, "w") as out:
        out.write("##gff-version 3\n")
        for chrom in sorted(chrom_genes.keys(), key=_chrom_index):
            cidx = _chrom_index(chrom)
            blocks = sorted(chrom_genes[chrom], key=lambda b: b[0])
            for gnum, (gstart, old_id, lines) in enumerate(blocks, 1):
                gene_id = f"{species_prefix}{cidx:02d}G{gnum:06d}_{predictor_tag}"
                for fields in lines:
                    fields[8] = fields[8].replace(old_id, gene_id)
                    out.write("\t".join(fields) + "\n")
                total_genes += 1

    logger.info(
        "Merged %d chromosomes -> %s (%d genes, prefix=%s tag=%s)",
        len(chrom_genes), merged_path, total_genes, species_prefix, predictor_tag,
    )
    return merged_path


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--genome-fasta", type=Path, required=True)
    ap.add_argument("--reference-gff3", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--decoding-method",
        type=str,
        default="viterbi",
        choices=["viterbi", "viterbi+orf", "viterbi+merge", "heuristic"],
    )
    ap.add_argument("--domain", type=str, default="plant", choices=["plant", "animal"])
    ap.add_argument("--viterbi-alpha", type=float, default=0.0)
    ap.add_argument("--source", type=str, default="PlantHelixSeek-anno")
    ap.add_argument("--genomic-offset", type=int, default=0,
                   help="Genomic coordinate offset to add to prediction positions")
    ap.add_argument("--single-exon-prob", type=float, default=0.7,
                   help="Min mean CDS probability for single-exon genes (0=disabled)")
    ap.add_argument("--single-exon-min-cds", type=int, default=200,
                   help="Min CDS length for single-exon genes (0=disabled)")
    ap.add_argument("--genic-threshold", type=float, default=0.25,
                   help="Min genic fraction for CDS interval (higher=more conservative)")
    ap.add_argument("--intergenic-boost", type=float, default=10.0,
                   help="Boost intergenic prob: positive=more conservative(fewer genes), "
                        "negative=more aggressive(more genes)")
    ap.add_argument("--species-prefix", type=str, default="",
                   help="Species abbreviation for merged GFF3 gene IDs (e.g. LP). "
                        "If empty, no merged species-level GFF3 is produced.")
    ap.add_argument("--predictor-tag", type=str, default="PHS",
                   help="Predictor tag appended to gene IDs in merged GFF3 (e.g. PHS)")
    ap.add_argument("--fix-splice", action="store_true", default=False,
                   help="Fix non-canonical splice sites: snap to GT/GC...AG within "
                        "--splice-window, preferring offsets that restore CDS len mod 3")
    ap.add_argument("--splice-window", type=int, default=30,
                   help="Search window (bp, each side) for splice site fixing")
    ap.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory for per-chromosome GFF3 (default: npz-dir)")
    a = ap.parse_args()

    npz_files = sorted(a.npz_dir.glob("*.npz"))
    if not npz_files:
        logger.error("No NPZ files found in %s", a.npz_dir)
        return

    ref_str = str(a.reference_gff3) if a.reference_gff3 else ""
    tasks = [
        (
            str(f), str(a.genome_fasta), ref_str, a.decoding_method,
            a.domain, a.viterbi_alpha, a.source, a.genomic_offset,
            a.single_exon_prob, a.single_exon_min_cds,
            a.genic_threshold, a.intergenic_boost,
            a.fix_splice, a.splice_window, str(a.output_dir) if a.output_dir else "",
        )
        for f in npz_files
    ]

    logger.info(
        "Decoding %d chromosomes with %d workers (method=%s)",
        len(tasks), a.workers, a.decoding_method,
    )

    if a.workers <= 1:
        results = [decode_one_chrom(t) for t in tasks]
    else:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            results = pool.map(decode_one_chrom, tasks)

    all_evals = {}
    for res in results:
        name, n_genes, status, ev = res
        if status == "OK":
            logger.info("  %s: %d genes", name, n_genes)
            if ev:
                all_evals[name] = ev
                logger.info(
                    "    base F1=%.4f recall=%.4f | CDS F1=%.4f recall=%.4f | gene_recall=%.4f (%d/%d)",
                    ev["base_f1"], ev["base_recall"],
                    ev["cds_f1"], ev["cds_recall"],
                    ev["gene_recall"], ev["pred_genes"], ev["ref_genes"],
                )
        elif status == "SKIP":
            logger.info("  %s: SKIPPED", name)
        else:
            logger.error("  %s: %s", name, status)

    if all_evals:
        avg = {
            k: np.mean([e[k] for e in all_evals.values()])
            for k in [
                "base_f1", "base_recall", "base_precision",
                "cds_f1", "cds_recall", "cds_precision", "gene_recall",
            ]
        }
        logger.info("=== AVERAGE ===")
        logger.info(
            "base F1=%.4f recall=%.4f prec=%.4f | CDS F1=%.4f recall=%.4f prec=%.4f | gene_recall=%.4f",
            avg["base_f1"], avg["base_recall"], avg["base_precision"],
            avg["cds_f1"], avg["cds_recall"], avg["cds_precision"],
            avg["gene_recall"],
        )

    if a.species_prefix:
        merge_dir = a.output_dir if a.output_dir else a.npz_dir
        merge_species_gff3(merge_dir, a.species_prefix, a.predictor_tag, a.source)


if __name__ == "__main__":
    main()
