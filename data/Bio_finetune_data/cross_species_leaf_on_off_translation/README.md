---
license: cc-by-nc-4.0
task_categories:
- text-classification
tags:
- plant
- genomics
---

# cross_species_leaf_on_off_translation

Translation on/off state in leaf tissue (binary classification).

## Source and provenance

PlantCAD2 fine-tuning tasks (Zhai et al., 2026).

## Files and splits

Fixed train / dev / test splits: 21,894 / 5,492 / 39,035.

The released `train.csv`, `dev.csv` and `test.csv` files are used as provided by
the source dataset. No re-splitting, species-held-out filtering or additional
repeat masking was applied.

## Columns

`gene`, `chr`, `strand`, `tpm`, `sequence` (500 bp), `label` (0 = off, 1 = on)

- **Species scope**: cross-species leaf samples.
- **Sequence length**: 500 bp.
- **Label**: 0 = off; 1 = on.
- **Primary metric**: F1.

## Preprocessing

Sequences are provided as uppercase DNA strings. No additional soft-masking or
repeat filtering was applied to this fine-tuning release. Users should apply the
preprocessing of the source publication when comparing with the original
benchmark.

## Citation

Dataset source: Zhai J, et al. PlantCAD2: a DNA foundation model for interpreting genomes across flowering plants. Cell Genomics 2026. doi:10.1016/j.xgen.2026.101329.

If you use this dataset, also cite the PlantHelixSeek manuscript: Wu Y, He Y,
Chen L, Liu Y, Wang T, Zhao H, Zhang Y, Zhang T. PlantHelixSeek learns
transferable genome representations that capture conserved genome grammar for
functional interpretation across plant species. Molecular Plant, 2026.
Manuscript MOLECULAR-PLANT-D-26-00999 (DOI to be assigned).

## License

cc-by-nc-4.0. Users must also comply with the license of the original source.
