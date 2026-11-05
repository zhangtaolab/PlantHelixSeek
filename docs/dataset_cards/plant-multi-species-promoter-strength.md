---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
tags:
- plant
- genomics
---

# plant-multi-species-promoter-strength

Core-promoter strength (regression).

## Source and provenance

AgroNT plant-genomic-benchmark (Mendoza-Revilla et al., 2024); underlying measurements from plant STARR-seq core-promoter assays (Jores et al., 2021).

## Files and splits

Fixed train / dev / test splits: 58,179 / 6,825 / 7,154.

The released `train.csv`, `dev.csv` and `test.csv` files are used as provided by
the source dataset. No re-splitting, species-held-out filtering or additional
repeat masking was applied.

## Columns

`name` (candidate identifier), `sequence` (170 bp), `label` (continuous strength score)

- **Species scope**: multiple plant species (see source).
- **Sequence length**: 170 bp.
- **Label**: continuous strength score.
- **Primary metric**: Spearman r.

## Preprocessing

Sequences are provided as uppercase DNA strings. No additional soft-masking or
repeat filtering was applied to this fine-tuning release. Users should apply the
preprocessing of the source publication when comparing with the original
benchmark.

## Citation

Dataset source: Mendoza-Revilla J, et al. AgroNT. Communications Biology 2024;7:835. doi:10.1038/s42003-024-06465-2. Jores T, et al. Synthetic promoter designs enabled by a comprehensive analysis of plant core promoters. Nature Plants 2021;7:842-855. doi:10.1038/s41477-021-00932-y.

If you use this dataset, also cite the PlantHelixSeek manuscript: Wu Y, He Y,
Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns
transferable genome representations that capture conserved genome grammar for
functional interpretation across plant species. Molecular Plant, 2026.
Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

cc-by-nc-sa-4.0. Users must also comply with the license of the original source.
