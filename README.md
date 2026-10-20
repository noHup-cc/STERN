# STERN

**STERN** (Spatio-Temporal Embedding for Real-time morphogeNesis) is a method for time-lapse 3D (4D)
light-sheet recordings of development. It describes each time point of a recording by features of its 2D
slices, learned from slices that are close in space or in developmental time, and compares time points
within and between recordings through a distance between these features. In the paper, this was used to
match developmental time points between mouse embryos, between mouse hearts, and between two imaging
views of a zebrafish recording.

The method has two parts:

1. **Slice features.** A Swin Transformer (Liu et al., *ICCV* 2021) maps each 2D slice of a volume to a
   1536-dimensional feature vector. The network, pretrained on ImageNet-22k, was fine-tuned on recordings
   of mouse embryos A-C: pairs of slices from nearby z positions or nearby time points served as similar
   examples, distant slices as dissimilar ones.
2. **Volume distance.** For each slice of a query volume (one time point), the Euclidean distance to the
   nearest slice of a target volume is taken; the volume distance is the sum of these distances.
   Computing it against every time point of a target recording gives a distance curve over time, and its
   minimum is the best-matching time point.

This repository contains the code of the paper *A quantitative coordinate system for developmental
dynamics*: the network, the preprocessing used in the paper, and the distance and matching analysis.
The trained weights and the features of all datasets of the paper are on Zenodo; with them, the paper's
matching results can be recomputed on a CPU, without the raw images.

| what | where |
|---|---|
| code (this repository) | GitHub; archived version: doi:10.5281/zenodo.23503745 |
| trained weights and slice features of every dataset | Zenodo, doi:10.5281/zenodo.23503600 |
| raw images: mouse embryos A-D | Image Data Resource, idr0044 (McDole et al., *Cell* 2018) |
| raw images: mouse hearts A-C | Figshare, doi:10.6084/m9.figshare.11662860.v1 (Yue et al., *Nat. Cell Biol.* 2020) |
| raw images: zebrafish, two views | BioImage Archive, S-BIAD4275 |

## Installation

```bash
git clone https://github.com/noHup-cc/STERN.git stern
cd stern
conda env create -f environment.yml     # creates the environment "stern" with current libraries
conda activate stern
pytest                                  # quick self-test (CPU, under a minute)
```

Two conda environments are provided. Both run the same code with the same preprocessing; they differ
only in the library versions:

| file | libraries | use it for |
|---|---|---|
| `environment.yml` (environment `stern`) | current versions: Python 3.11, PyTorch 2.x, timm >= 0.9, current Pillow and OpenCV | normal use; installs on current systems and GPUs |
| `environment-paper.yml` (environment `stern-paper`) | the versions used for the paper's features: Python 3.7, PyTorch 1.8.0 (CUDA 11.1), timm 0.6.13, Pillow 5.4.1, OpenCV 4.6.0.66 | reproducing the paper's features bit for bit |

The paper's versions are from 2021: Python 3.7 and PyTorch 1.8 are no longer maintained, and the CUDA 11.1
build does not support the newest GPUs. The current versions are therefore the default. Newer libraries
encode JPEG images, resize images and compute on the GPU slightly differently, so their features differ
from the paper's by up to about 0.2 %; the paper's matching results stay the same (`docs/VALIDATION.md`).

Only comparing released features? NumPy and scikit-learn are enough: `pip install -e .`
To compute features with the network, install PyTorch and timm as well: `pip install -e ".[model]"`
(add `klb` for KLB files: `pip install -e ".[model,klb]"`).

## Quick start: reproduce Fig. 2c from the released features

Download the features of mouse embryos A and D from the Zenodo record and unpack them:

```bash
mkdir features && cd features
for d in mouse_A mouse_D; do
    curl -L -o features_$d.zip "https://zenodo.org/records/23503600/files/features_$d.zip?download=1"
    unzip -q features_$d.zip
done
cd ..
```

Each dataset is a folder with one file per time point (`000000.npz`, `000001.npz`, ...) holding the
features of all its slices. Now find, for three time points of embryo D, the best-matching time point of
embryo A:

```python
from stern import FeatureSeries, distance_table, mouse_z_ranges

d = FeatureSeries.from_directory('features/mouse_D', name='D')
a = FeatureSeries.from_directory('features/mouse_A', name='A')
table = distance_table(d, a, query_times=[0, 100, 250],
                       query_ranges=mouse_z_ranges('D'), target_ranges=mouse_z_ranges('A'))

table.best_matches()        # [(0, 118, ...), (100, 215, ...), (250, 361, ...)] as in Fig. 2c
table.to_csv('D_to_A.csv')  # the whole distance curve of each query time point
```

The same from the command line:

```bash
stern match features/mouse_D features/mouse_A --query-times 0,100,250 \
    --query-ranges mouse:D --target-ranges mouse:A --out D_to_A.csv --best D_to_A_best.csv
```

This runs in a minute or two on a laptop. The scripts in `examples/` reproduce Fig. 2c for embryos A, B and C
(`match_mouse_embryos.py`) and Fig. 3h for the two zebrafish views (`match_zebrafish_views.py`).

**Slice ranges.** Only the slices that contain the specimen are compared. For the mouse embryos this range
grows during development and is given per time point by `mouse_z_ranges`; each heart has one fixed range
(`heart_z_range`); for zebrafish all slices are used. For your own data, pass a fixed range such as
`--query-ranges 20-120`, or `all`.

**Comparing a recording with itself.** To compare every time point of a recording with its earlier time
points, as in the paper's within-embryo distance matrices, compute only the pairs with target time <=
query time:

```bash
stern match features/mouse_A features/mouse_A --query-ranges mouse:A --target-ranges mouse:A \
    --lower --out A_to_A.csv
```

```python
a = FeatureSeries.from_directory('features/mouse_A', name='A')
table = distance_table(a, a, query_ranges=mouse_z_ranges('A'), target_ranges=mouse_z_ranges('A'),
                       pairs=lambda q, t: t <= q)
```

For embryo A (527 time points with slice ranges) these are about 139,000 volume distances, which takes
longer than the example above.

## Using STERN on your own recordings

Feature extraction needs the network weights (`stern_weights.safetensors` from the Zenodo record); a GPU
makes it much faster but is not required.

```bash
# download the network weights (about 0.9 GB)
curl -L -o stern_weights.safetensors \
    "https://zenodo.org/records/23503600/files/stern_weights.safetensors?download=1"

# a 4D OME-TIFF (time series): one feature file per time point
stern extract recording.ome.tif --time-series --recipe zebrafish \
    --weights stern_weights.safetensors --out my_features/

# then match two recordings
stern match my_features/ other_features/ --out distances.csv --best best_matches.csv
```

Inputs can be TIFF or OME-TIFF stacks (one file per time point, or one file with a time axis), folders of
slice images, KLB files or NumPy arrays. In Python:

```python
from stern.extract import Extractor
from stern.io import read_volume
from stern.features import save_frame

extractor = Extractor('stern_weights.safetensors', recipe='mouse')
features = extractor.embed_volume(read_volume('volume.tif'))   # array of shape (slices, 1536)
save_frame('my_features/000000.npz', features)
```

The `recipe` argument selects how a raw slice is turned into the 8-bit image the network expects. The
three recipes are the paper's settings for its three datasets (`mouse` and `heart` are identical):

| recipe | slice to 8 bit | JPEG written with | TF32 on GPU | suited to |
|---|---|---|---|---|
| `mouse`, `heart` | rounded and clipped to 0-255 | OpenCV | off | values mostly within 0-255 |
| `zebrafish` | stretched between the 1st and 99th percentile of each slice | Pillow | on | values over a wider range (16 bit) |

For your own recordings, pick the recipe by the intensity range of your images. If the values lie mostly
within 0-255 (8-bit images, or 16-bit images with low values), use `mouse`: it keeps the values as they
are and sets anything above 255 to 255. If the values spread over a wider range, as in most 16-bit
images, use `zebrafish`: it rescales each slice to 0-255.

The other two columns need no choice: each recipe sets them as the paper did for that dataset. They
affect the features only slightly; they are kept so that the paper's features can be reproduced exactly.
(TF32 is a faster, lower-precision arithmetic of NVIDIA GPUs; it is used only when running on such a GPU.)

**The JPEG step.** No JPEG files are written in either case. By default each 8-bit slice is compressed to
JPEG and decompressed again in memory, so that the network sees the same slight compression artefacts as
the JPEG files it was fine-tuned on. `--no-jpeg` (`jpeg=False`) leaves out this compression: the 8-bit
slice goes straight to resizing and the network. How this affects matching has not been evaluated; keep
the default unless you have a reason not to, and compare only features computed the same way (the
released features were computed with the JPEG step).

### Example: a mouse heart volume from its raw images

The mouse heart images of Yue et al. (2020) are on Figshare as one archive (0.7 GB) with one TIFF stack
per time point. The commands below take the stack of heart A at time point 115, compute its features and
match it against the released features of every time point of heart A (115-134):

```bash
# one TIFF stack from the archive: heart A, time point 115
curl -L -o Preprocessed_Datasets.zip https://ndownloader.figshare.com/files/21189321
unzip -j Preprocessed_Datasets.zip '*DataSetsA/115.tif' -d heart_A_raw

# its features: heart_A_115/000115.npz
stern extract heart_A_raw/115.tif --recipe heart --first-time 115 \
    --weights stern_weights.safetensors --out heart_A_115/

# the released features of heart A, and the match
curl -L -o features_heart_A.zip "https://zenodo.org/records/23503600/files/features_heart_A.zip?download=1"
unzip -q features_heart_A.zip -d released
stern match heart_A_115/ released/heart_A/ --query-ranges heart:A --target-ranges heart:A \
    --out heart_A_115.csv --best heart_A_115_best.csv
```

The best match is time point 115 itself, and `heart_A_115.csv` lists the distance to every time point.
The computed features equal the released `released/heart_A/000115.npz` (bit-identical in the paper
environment, within about 0.2 % otherwise). For your own recordings, replace the stack with your files and
choose the recipe as described above.

## Command-line options

`stern extract` writes one feature file per time point (`<out>/000115.npz` for time point 115) and
`stern match` compares two folders of such files. `stern extract --help` and `stern match --help` list the
same options.

**`stern extract INPUT ... --weights FILE --out FOLDER`**

| option | meaning |
|---|---|
| `INPUT ...` | one volume per time point, in time order: TIFF / OME-TIFF stack, KLB file, `.npy` array or folder of slice images; or one TIFF with a time axis together with `--time-series` |
| `--weights` | the network weights, `stern_weights.safetensors` |
| `--out` | output folder |
| `--recipe` | `mouse` (default), `heart` or `zebrafish`: how a slice becomes an 8-bit image (table above) |
| `--time-series` | the input is one (OME-)TIFF with a time axis; one feature file per time point |
| `--times` | time points of a time series, e.g. `0-108`, `0,50,100` or `0-108:10` (every 10th); default: all |
| `--first-time` | time point of the first input when there is one file per time point (default 0); the next files get the following numbers |
| `--channel` | channel of a multi-channel TIFF (default 0) |
| `--no-jpeg` | leave out the in-memory JPEG step (see "The JPEG step") |
| `--device` | `cuda` or `cpu` (default: `cuda` when a GPU is available) |
| `--batch-size` | slices per forward pass (default 1, as in the paper; larger batches are faster and change the features only at the level of floating-point rounding) |
| `--overwrite` | recompute time points whose feature file exists (by default they are skipped) |

**`stern match QUERY TARGET --out FILE`**

| option | meaning |
|---|---|
| `QUERY`, `TARGET` | folders of feature files (from `stern extract`, or the released features); every query time point is compared with every target time point |
| `--out` | CSV of all distances, one row per pair: `queryEmbryo, query_time, targetEmbryo, target_time, distance` |
| `--best` | CSV of the best match of each query time point: `query_time, best_target_time, distance` |
| `--query-times`, `--target-times` | time points to compare, e.g. `0,100,250`, `0-265` or `0-265:5`; default: all time points (with `mouse:` ranges, those that have a range) |
| `--query-ranges`, `--target-ranges` | slices used for the distance, as first and last slice (counted from 1, both included): `all` (default), a fixed range such as `20-120`, or the paper's ranges `mouse:A` ... `mouse:D` (one range per time point) and `heart:A` ... `heart:C` (one fixed range); see "Slice ranges" |
| `--lower` | only the pairs with target time <= query time (comparing a recording with itself) |
| `--query-name`, `--target-name` | names in the `queryEmbryo` / `targetEmbryo` columns (default: the folder names) |

## How the paper's features were computed

In the paper, every slice was converted to 8 bit with the recipe of its dataset (table above), saved as a
JPEG image (quality 95), then read back, converted to RGB, resized to 224 x 224, normalised with the
ImageNet mean and standard deviation, and passed to the network one slice at a time. The package does the
same in memory, without writing files.

**Exact reproduction.** In the paper environment (`environment-paper.yml`, see Installation) the features
computed from the raw images are bit-identical to the paper's (checked on sample time points of every
dataset). In the current environment they differ by up to about 0.2 %, and the matching results of the
paper (Fig. 2c, the heart matrices, Fig. 3h) stay the same. `docs/VALIDATION.md` reports both comparisons
in full.

## Model

- **Architecture**: Swin Transformer, large (Swin-L; patch 4, window 7, 224 x 224 input; Liu et al., *ICCV*
  2021), in a Siamese network. During training a projector MLP (1536 -> 4096 -> 4096 -> 256) replaced the
  classifier; the features are the 1536-d pooled output before it (`STERN.embed`).
- **Pretrained network**: Swin-L pretrained on ImageNet-22k, the weights of the official Swin Transformer
  release of Microsoft Research (MIT licence), loaded through timm 0.6.13 as
  `swin_large_patch4_window7_224_in22k`.
- **Fine-tuning**: contrastive (Siamese) fine-tuning on spatially and temporally paired slices of mouse
  embryos A-C only (Methods), 50 epochs; embryo D is the test embryo. The released weights are those of the
  last epoch (index 49); `stern_weights.json` records the SHA-256 of the training checkpoint and of the
  weights.
- **Intended use**: comparing and matching light-sheet recordings of developing tissue (shown for mouse
  embryos, mouse hearts and zebrafish). It is not a classifier, and other imaging modalities are untested.
- The weights load in both older (0.6) and current (>= 0.9) timm and give identical features in both.

## Tests

```bash
pytest                                                      # everything that runs without data (CPU)
STERN_FEATURES=/path/to/features pytest tests/test_paper.py   # the paper's results from the released features
```

## Citation

If you use STERN, please cite the paper (*reference to be added*) and, for the code or the data, the
Zenodo records: code doi:10.5281/zenodo.23503745; weights and features doi:10.5281/zenodo.23503600.
GitHub shows the citation under "Cite this repository" (from `CITATION.cff`).

## Contact

Maintainer: Jinzheng Ren. Questions and problems: https://github.com/noHup-cc/STERN/issues

## License

The code is released under the MIT licence (`LICENSE`); the weights and features under CC BY 4.0.
