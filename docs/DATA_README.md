# STERN: network weights and slice features

Companion data of the STERN paper (*A quantitative coordinate system for developmental dynamics*;
reference to be added) and of its code, doi:10.5281/zenodo.23503745
(https://github.com/noHup-cc/STERN).

| file | content |
|---|---|
| `stern_weights.safetensors` | the trained network (Swin-L backbone + projector), epoch 49 of the Siamese training; load with `stern.model.load_stern` |
| `stern_weights.json` | SHA-256 of the training checkpoint and of the weights, parameter count |
| `features_<dataset>.zip` | slice features of one dataset: `<dataset>/<time>.npz` with arrays `z` (1-based slice index) and `features` (slices x 1536, float32) |
| `features_manifest.tsv` | SHA-256 and size of every `.npz` file |
| `SHA256SUMS` | SHA-256 of every file of this record (`sha256sum -c SHA256SUMS`) |

## Model

- **Architecture**: Swin Transformer, large (Swin-L; patch 4, window 7, 224 x 224 input; Liu et al., *ICCV*
  2021), in a Siamese network; a projector MLP (1536 -> 4096 -> 4096 -> 256) replaced the classifier during
  training. The slice features are the 1536-d pooled backbone output (`STERN.embed`).
- **Pretrained network**: Swin-L pretrained on ImageNet-22k, the weights of the official Swin Transformer
  release of Microsoft Research (MIT licence), loaded through timm 0.6.13 as
  `swin_large_patch4_window7_224_in22k` (`pretrained=True`).
- **Fine-tuning**: contrastive (Siamese) fine-tuning on spatially and temporally paired slices of mouse
  embryos A-C only (Methods), 50 epochs; embryo D is the test embryo. The released weights are those of the
  last epoch (index 49), exported from the training checkpoint without the optimiser state;
  `stern_weights.json` records the SHA-256 of that checkpoint and of the weights.
- **Use**: slice features for volume distances and matching of light-sheet recordings (mouse embryos,
  mouse hearts, zebrafish). It is not a classifier; other imaging modalities or tissues are untested.

Datasets: `mouse_A`-`mouse_D` (mouse embryos A-D), `heart_A`-`heart_C` (mouse hearts A-C), `zf_G1` and
`zf_G2` (zebrafish neural crest, imaging views 1 and 2 of the paper). The features were extracted with the
published network from the paper's slice images (see the code's README, "How the paper's features were
computed") and validated against the original analysis (`docs/VALIDATION.md` of the code).

Sources of the recordings:

- mouse embryos A-D: McDole et al., *Cell* 175, 859-876.e33 (2018), doi:10.1016/j.cell.2018.09.031; Image Data
  Resource idr0044, doi:10.17867/10000116 (CC BY 4.0, copyright McDole et al.);
- mouse hearts A-C: Yue et al., *Nat. Cell Biol.* 22, 332-340 (2020), doi:10.1038/s41556-020-0475-2;
  Figshare dataset *Preprocessed_Datasets.zip* (Li, Zhang, Yue and He, 2020),
  doi:10.6084/m9.figshare.11662860.v1 (CC BY 4.0);
- zebrafish views 1 and 2 (`zf_G1`, `zf_G2`): acquired for the paper (Methods); BioImage Archive S-BIAD4275,
  https://www.ebi.ac.uk/biostudies/studies/S-BIAD4275.

Usage:

```bash
sha256sum -c SHA256SUMS
for f in features_*.zip; do unzip -q "$f" -d features; done
pip install -e <code folder>  # the code record, doi:10.5281/zenodo.23503745
STERN_FEATURES=features pytest tests/test_paper.py
```

License: CC BY 4.0 (data), MIT (code).
