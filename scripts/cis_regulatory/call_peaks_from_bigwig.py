"""从 BigWig 信号文件中 Call Peak，输出 BED 格式的 CRE 候选区域。

支持阈值过滤 + 局部极大值精修两种模式，可输出 narrowPeak 与 broadPeak 格式。

用法:
    # 默认参数 call peaks（阈值 = mean + 1*std）
    python call_peaks_from_bigwig.py \\
        --bigwig result/predict/model_xxx/signal.bw \\
        --genome genome/species.fa \\
        --output_dir result/predict/model_xxx

    # 指定阈值
    python call_peaks_from_bigwig.py \\
        --bigwig signal.bw --genome genome.fa --output_dir out/ \\
        --threshold 0.5

    # 使用局部极大值精修 peak summit
    python call_peaks_from_bigwig.py \\
        --bigwig signal.bw --genome genome.fa --output_dir out/ \\
        --threshold 0.5 --refine_summit
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import pyBigWig
from Bio import SeqIO
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BIN_SIZE = 50  # 与 predict_open_chromatin.py 保持一致
DEFAULT_MIN_LENGTH = 50
DEFAULT_MAX_LENGTH = 5000
DEFAULT_MIN_SCORE = 0.6
DEFAULT_MERGE_GAP = 50
DEFAULT_THRESHOLD_METHOD = "mean_std"
DEFAULT_THRESHOLD_PARAMS = "1.5"


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def load_genome_chrom_sizes(fasta_path: Path) -> dict[str, int]:
    """从 FASTA 文件读取染色体名和长度。"""
    chrom_sizes: dict[str, int] = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        chrom_sizes[rec.id] = len(rec.seq)
    return chrom_sizes


def compute_threshold(
    values: np.ndarray,
    method: str,
    params: str,
) -> float:
    """根据指定方法计算信号阈值。

    Args:
        values: 信号值数组。
        method: 阈值计算方法。
            - "mean_std": mean + k * std（k 由 params 指定）
            - "median_mad": median + k * MAD（k 由 params 指定）
            - "percentile": 第 p 百分位数（p 由 params 指定）
            - "fixed": 直接使用 params 作为阈值
        params: 方法参数字符串。

    Returns:
        计算得到的阈值。
    """
    if method == "fixed":
        return float(params)

    if method == "percentile":
        p = float(params)
        return float(np.percentile(values, p))

    if method == "mean_std":
        k = float(params)
        mean = float(np.mean(values))
        std = float(np.std(values))
        threshold = mean + k * std
        logger.info("  mean=%.4f, std=%.4f, k=%.1f → threshold=%.4f", mean, std, k, threshold)
        return threshold

    if method == "median_mad":
        k = float(params)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826  # MAD → σ 估计
        threshold = median + k * mad
        logger.info("  median=%.4f, MAD=%.4f, k=%.1f → threshold=%.4f", median, mad, k, threshold)
        return threshold

    raise ValueError(f"Unknown threshold method: {method}")


def call_peaks_chromosome(
    values: np.ndarray,
    chrom: str,
    threshold: float,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    merge_gap: int = DEFAULT_MERGE_GAP,
    refine_summit: bool = False,
    summit_window: int = 100,
) -> list[dict]:
    """对单条染色体 call peaks。

    Args:
        values: 逐碱基信号值数组（1-based 对应 genome position 0..N-1）。
        chrom: 染色体名。
        threshold: 信号阈值。
        min_length: peak 最小长度（bp）。
        max_length: peak 最大长度（bp），超过则不输出。
        merge_gap: 合并间距（bp），相邻高分区域间距 <= gap 时合并。
        refine_summit: 是否精修 peak summit 位置。
        summit_window: summit 精修搜索窗口大小。

    Returns:
        peak 字典列表，每个包含 chrom, start, end, score, summit 等。
    """
    # 找到高于阈值的碱基
    above = values >= threshold
    if not above.any():
        return []

    # 提取连续高分区域
    diff = np.diff(above.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1  # 区域起点
    ends = np.where(diff == -1)[0] + 1    # 区域终点

    # 边界处理
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(values)]])

    regions = list(zip(starts.tolist(), ends.tolist()))

    # 合并间距 <= merge_gap 的相邻区域
    if merge_gap > 0 and len(regions) > 1:
        merged = [regions[0]]
        for s, e in regions[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= merge_gap:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        regions = merged

    # 过滤短区域 + 计算统计量
    peaks: list[dict] = []
    for start, end in regions:
        length = end - start
        if length < min_length or length > max_length:
            continue

        region_vals = values[start:end]
        max_score = float(np.max(region_vals))
        mean_score = float(np.mean(region_vals))

        # 找到 summit（最高点位置）
        summit_local = int(np.argmax(region_vals))
        summit_abs = start + summit_local

        # 精修 summit：在 summit 周围搜索真正的局部极大值
        if refine_summit and max_score > 0:
            win = summit_window
            search_start = max(0, summit_abs - win)
            search_end = min(len(values), summit_abs + win)
            search_vals = values[search_start:search_end]
            if len(search_vals) > 0:
                local_max = maximum_filter1d(search_vals, size=min(21, len(search_vals)))
                local_min = minimum_filter1d(search_vals, size=min(21, len(search_vals)))
                # prominence: 局部极大值 - 局部极小值
                prominence = local_max - local_min
                # 只在原 summit 附近选择
                center = summit_abs - search_start
                search_radius = min(50, len(search_vals) // 2)
                search_lo = max(0, center - search_radius)
                search_hi = min(len(search_vals), center + search_radius)
                best_idx = search_lo + int(np.argmax(search_vals[search_lo:search_hi]))
                summit_abs = search_start + best_idx

        peaks.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "length": length,
            "max_score": max_score,
            "mean_score": mean_score,
            "summit": summit_abs,
            "summit_score": float(values[summit_abs]),
        })

    return peaks


def write_bed(
    peaks: list[dict],
    output_path: Path,
    name: str = "CRE",
) -> None:
    """写入 BED6 格式文件。"""
    with open(output_path, "w") as f:
        for i, p in enumerate(peaks, 1):
            f.write(
                f"{p['chrom']}\t{p['start']}\t{p['end']}\t"
                f"{name}_{i}\t{p['max_score']:.4f}\t.\n"
            )
    logger.info("Wrote BED: %s (%d peaks)", output_path, len(peaks))


def write_narrow_peak(
    peaks: list[dict],
    output_path: Path,
    name: str = "CRE",
) -> None:
    """写入 narrowPeak (BED6+4) 格式文件。

    格式: chrom start end name score strand signalValue pValue qValue peak
    """
    with open(output_path, "w") as f:
        for i, p in enumerate(peaks, 1):
            # peak 列 = summit 相对于 peak start 的偏移
            summit_offset = p["summit"] - p["start"]
            f.write(
                f"{p['chrom']}\t{p['start']}\t{p['end']}\t"
                f"{name}_{i}\t{p['max_score']:.4f}\t.\t"
                f"{p['mean_score']:.4f}\t-1\t-1\t{summit_offset}\n"
            )
    logger.info("Wrote narrowPeak: %s (%d peaks)", output_path, len(peaks))


def write_broad_peak(
    peaks: list[dict],
    output_path: Path,
    name: str = "CRE",
) -> None:
    """写入 broadPeak (BED6+3) 格式文件。

    格式: chrom start end name score strand signalValue pValue qValue
    """
    with open(output_path, "w") as f:
        for i, p in enumerate(peaks, 1):
            f.write(
                f"{p['chrom']}\t{p['start']}\t{p['end']}\t"
                f"{name}_{i}\t{p['mean_score']:.4f}\t.\t"
                f"{p['mean_score']:.4f}\t-1\t-1\n"
            )
    logger.info("Wrote broadPeak: %s (%d peaks)", output_path, len(peaks))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 BigWig 信号文件中 Call Peak，输出 BED/narrowPeak/broadPeak 格式",
    )
    parser.add_argument(
        "--bigwig", type=str, required=True,
        help="输入 BigWig 信号文件路径",
    )
    parser.add_argument(
        "--genome", type=str, required=True,
        help="基因组 FASTA 文件路径（用于获取染色体大小）",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--threshold_method", type=str, default=DEFAULT_THRESHOLD_METHOD,
        choices=["mean_std", "median_mad", "percentile", "fixed"],
        help=f"阈值计算方法（默认: {DEFAULT_THRESHOLD_METHOD}）",
    )
    parser.add_argument(
        "--threshold_params", type=str, default=DEFAULT_THRESHOLD_PARAMS,
        help=(
            "阈值参数。"
            "mean_std/median_mad: k 值（默认 1.0）；"
            "percentile: 百分位数（如 95）；"
            "fixed: 直接指定阈值（如 0.5）"
        ),
    )
    parser.add_argument(
        "--min_length", type=int, default=DEFAULT_MIN_LENGTH,
        help=f"Peak 最小长度 bp（默认: {DEFAULT_MIN_LENGTH}）",
    )
    parser.add_argument(
        "--max_length", type=int, default=DEFAULT_MAX_LENGTH,
        help=f"Peak 最大长度 bp（默认: {DEFAULT_MAX_LENGTH}）",
    )
    parser.add_argument(
        "--min_score", type=float, default=DEFAULT_MIN_SCORE,
        help="Peak 最低得分（max_score），低于此值的 peak 不输出（默认: 0.0）",
    )
    parser.add_argument(
        "--merge_gap", type=int, default=DEFAULT_MERGE_GAP,
        help=f"合并相邻高分区域的最大间距 bp（默认: {DEFAULT_MERGE_GAP}）",
    )
    parser.add_argument(
        "--refine_summit", action="store_true",
        help="精修 peak summit 位置（局部极大值搜索）",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="输出文件前缀（默认从 bigwig 文件名推导）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bigwig_path = Path(args.bigwig)
    genome_path = Path(args.genome)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bigwig_path.exists():
        logger.error("BigWig file not found: %s", bigwig_path)
        sys.exit(1)
    if not genome_path.exists():
        logger.error("Genome FASTA not found: %s", genome_path)
        sys.exit(1)

    # 推导输出前缀
    prefix = args.prefix or bigwig_path.stem
    logger.info("Output prefix: %s", prefix)

    # 读取染色体大小
    chrom_sizes = load_genome_chrom_sizes(genome_path)
    logger.info("Loaded %d chromosome(s) from %s", len(chrom_sizes), genome_path.name)

    # 打开 BigWig
    bw = pyBigWig.open(str(bigwig_path))
    bw_chroms = bw.chroms()
    logger.info("BigWig chromosomes: %s", bw_chroms)

    # 计算阈值（先收集所有染色体信号统计）
    logger.info("Computing threshold using method=%s, params=%s ...", args.threshold_method, args.threshold_params)
    all_values: list[np.ndarray] = []
    for chrom in bw_chroms:
        if chrom not in chrom_sizes:
            logger.warning("Chromosome %s in BigWig but not in FASTA, skipping", chrom)
            continue
        vals = bw.values(chrom, 0, bw_chroms[chrom], numpy=True)
        vals = np.nan_to_num(vals, nan=0.0)
        all_values.append(vals)

    if not all_values:
        logger.error("No overlapping chromosomes between BigWig and FASTA")
        sys.exit(1)

    combined = np.concatenate(all_values)
    threshold = compute_threshold(combined, args.threshold_method, args.threshold_params)
    logger.info("Threshold = %.4f", threshold)
    del combined, all_values  # 释放内存

    # 逐染色体 call peaks
    all_peaks: list[dict] = []
    for chrom in sorted(bw_chroms.keys()):
        if chrom not in chrom_sizes:
            continue
        vals = bw.values(chrom, 0, bw_chroms[chrom], numpy=True)
        vals = np.nan_to_num(vals, nan=0.0)
        logger.info(
            "  %s: %d bp, signal range [%.4f, %.4f]",
            chrom, len(vals), vals.min(), vals.max(),
        )
        peaks = call_peaks_chromosome(
            vals, chrom, threshold,
            min_length=args.min_length,
            max_length=args.max_length,
            merge_gap=args.merge_gap,
            refine_summit=args.refine_summit,
        )
        logger.info("  %s: %d peaks called", chrom, len(peaks))
        all_peaks.extend(peaks)

    bw.close()

    # 按 chrom + start 排序
    all_peaks.sort(key=lambda p: (p["chrom"], p["start"]))
    logger.info("Total peaks (before score filter): %d", len(all_peaks))

    # min_score 过滤
    if args.min_score > 0:
        before = len(all_peaks)
        all_peaks = [p for p in all_peaks if p["max_score"] >= args.min_score]
        logger.info("Filtered by min_score=%.4f: %d → %d (removed %d)",
                     args.min_score, before, len(all_peaks), before - len(all_peaks))

    if not all_peaks:
        logger.warning("No peaks called with threshold=%.4f. Consider lowering threshold.", threshold)
        sys.exit(0)

    # 统计
    lengths = np.array([p["length"] for p in all_peaks])
    scores = np.array([p["max_score"] for p in all_peaks])
    logger.info(
        "Peak length: min=%d, median=%d, max=%d, mean=%.0f",
        lengths.min(), int(np.median(lengths)), lengths.max(), lengths.mean(),
    )
    logger.info(
        "Peak score: min=%.4f, median=%.4f, max=%.4f, mean=%.4f",
        scores.min(), np.median(scores), scores.max(), scores.mean(),
    )

    # 输出文件
    write_bed(all_peaks, output_dir / f"{prefix}_peaks.bed", name=prefix)
    write_narrow_peak(all_peaks, output_dir / f"{prefix}_peaks.narrowPeak", name=prefix)
    write_broad_peak(all_peaks, output_dir / f"{prefix}_peaks.broadPeak", name=prefix)

    logger.info("All done.")


if __name__ == "__main__":
    main()
