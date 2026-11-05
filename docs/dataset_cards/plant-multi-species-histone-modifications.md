---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
tags:
- plant
- genomics
---

# plant-multi-species-histone-modifications

Three histone modifications in plants (H3K27ac, H3K27me3 and H3K4me3; binary classification).

## Source and provenance

PDLLMs (Liu et al., 2025).

## Files and splits

Fixed train / dev / test splits: H3K27ac 81,920 / 10,240 / 10,240; H3K27me3 79,748 / 9,968 / 9,968; H3K4me3 81,920 / 10,240 / 10,240.

The released `train.csv`, `dev.csv` and `test.csv` files are used as provided by
the source dataset. No re-splitting, species-held-out filtering or additional
repeat masking was applied.

## Columns

`sp` (species code), `sequence` (up to 2,000 bp), `label` (0 = no mark, 1 = mark present)

- **Species scope**: 8 species (ath, cotton, maize, potato, rice, soybean, tomato, wheat).
- **Sequence length**: up to 2,000 bp.
- **Label**: 0 = no mark; 1 = mark present.
- **Primary metric**: F1.

## Preprocessing

Sequences are provided as uppercase DNA strings. No additional soft-masking or
repeat filtering was applied to this fine-tuning release. Users should apply the
preprocessing of the source publication when comparing with the original
benchmark.

## Citation

Dataset source: Liu G, et al. PDLLMs. Molecular Plant 2025;18(2):175-178. doi:10.1016/j.molp.2024.12.006.

If you use this dataset, also cite the PlantHelixSeek manuscript: Wu Y, He Y,
Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns
transferable genome representations that capture conserved genome grammar for
functional interpretation across plant species. Molecular Plant, 2026.
Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

cc-by-nc-sa-4.0. Users must also comply with the license of the original source.
