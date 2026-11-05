---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
tags:
- plant
- genomics
---

# plant-multi-species-open-chromatin

Open-chromatin state in plants (three-class classification).

## Source and provenance

PDLLMs (Liu et al., 2025); peaks compiled in PlantCADB (Ding et al., 2023).

## Files and splits

Fixed train / dev / test splits: 118,800 / 14,850 / 14,850.

The released `train.csv`, `dev.csv` and `test.csv` files are used as provided by
the source dataset. No re-splitting, species-held-out filtering or additional
repeat masking was applied.

## Columns

`sp` (species name), `sequence` (up to 3,000 bp), `label` (0, 1 or 2)

- **Species scope**: 33 species.
- **Sequence length**: up to 3,000 bp.
- **Label**: 0, 1, 2 (three chromatin classes).
- **Primary metric**: MCC; micro- and weighted F1 are also reported.

## Preprocessing

Sequences are provided as uppercase DNA strings. No additional soft-masking or
repeat filtering was applied to this fine-tuning release. Users should apply the
preprocessing of the source publication when comparing with the original
benchmark.

## Citation

Dataset source: Liu G, et al. PDLLMs. Molecular Plant 2025;18(2):175-178. doi:10.1016/j.molp.2024.12.006. Ding K, et al. PlantCADB. Genomics, Proteomics & Bioinformatics 2023;21(2):311-323. doi:10.1016/j.gpb.2022.10.005.

If you use this dataset, also cite the PlantHelixSeek manuscript: Wu Y, He Y,
Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns
transferable genome representations that capture conserved genome grammar for
functional interpretation across plant species. Molecular Plant, 2026.
Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

cc-by-nc-sa-4.0. Users must also comply with the license of the original source.
