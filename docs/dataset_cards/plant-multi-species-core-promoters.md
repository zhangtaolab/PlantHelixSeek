---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
tags:
- plant
- genomics
---

# plant-multi-species-core-promoters

Active core promoters in plants (binary classification).

## Source and provenance

PDLLMs (Liu et al., 2025).

## Files and splits

Fixed train / dev / test splits: 66,560 / 8,320 / 8,320.

The released `train.csv`, `dev.csv` and `test.csv` files are used as provided by
the source dataset. No re-splitting, species-held-out filtering or additional
repeat masking was applied.

## Columns

`sp` (species code), `sequence` (300 bp), `label` (0 = inactive, 1 = active)

- **Species scope**: 13 species (ath, barley, bd30, broomcorn, cotton, grape, maize, poplar, potato, rice, soybean, tomato, wheat).
- **Sequence length**: 300 bp.
- **Label**: 0 = inactive; 1 = active.
- **Primary metric**: F1.

## Preprocessing

Sequences are provided as uppercase DNA strings. No additional soft-masking or
repeat filtering was applied to this fine-tuning release. Users should apply the
preprocessing of the source publication when comparing with the original
benchmark.

## Citation

Dataset source: Liu G, Chen L, Wu Y, Han Y, Bao Y, Zhang T. PDLLMs: a group of tailored DNA large language models for analyzing plant genomes. Molecular Plant 2025;18(2):175-178. doi:10.1016/j.molp.2024.12.006.

If you use this dataset, also cite the PlantHelixSeek manuscript: Wu Y, He Y,
Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns
transferable genome representations that capture conserved genome grammar for
functional interpretation across plant species. Molecular Plant, 2026.
Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

cc-by-nc-sa-4.0. Users must also comply with the license of the original source.
