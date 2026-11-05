# Model cards

Version-controlled copies of the Hugging Face model cards. The same content is
published on the Hugging Face and ModelScope model repositories.

## Base model

- [PlantHelixSeek](PlantHelixSeek.md)

## Fine-tuned checkpoints

- [core_promoter](PlantHelixSeek-core_promoter.md)
- [lncRNA](PlantHelixSeek-lncRNA.md)
- [open_chromatin](PlantHelixSeek-open_chromatin.md)
- [sequence_conservation](PlantHelixSeek-sequence_conservation.md)
- [H3K27ac](PlantHelixSeek-H3K27ac.md)
- [H3K27me3](PlantHelixSeek-H3K27me3.md)
- [H3K4me3](PlantHelixSeek-H3K4me3.md)
- [translation_onoff](PlantHelixSeek-translation_onoff.md)
- [promoter_strength](PlantHelixSeek-promoter_strength.md)
- [terminator_strength](PlantHelixSeek-terminator_strength.md)
- [translation_absolute](PlantHelixSeek-translation_absolute.md)

## Paper-specific checkpoints

- [PlantHelixSeek-Anno](PlantHelixSeek-Anno.md)
- [PlantHelixSeek-CRE](PlantHelixSeek-CRE.md)

Each fine-tuned card reports the training hyperparameters, the random seed and
the held-out validation metrics. The base model card reports the pretraining
configuration, the final validation perplexity and the masked-base recovery
rate.
