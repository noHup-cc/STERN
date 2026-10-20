#!/usr/bin/env python3
"""Slice features of your own volumes (GPU recommended).

    python examples/extract_features.py stern_weights.safetensors volume.tif out/       # one volume
    stern extract recording.ome.tif --time-series --recipe zebrafish --weights stern_weights.safetensors --out out/

``recipe``: how a raw slice becomes the 8-bit image the network sees; use 'mouse' / 'heart' for 8-bit
light-sheet stacks (direct conversion) and 'zebrafish' for 16-bit data that needs per-slice percentile
scaling. See README, "How the paper's features were computed".
"""
import sys
from pathlib import Path

from stern.extract import Extractor
from stern.features import save_frame
from stern.io import read_volume

weights, volume_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
extractor = Extractor(weights, recipe='mouse')
volume = read_volume(volume_path)                            # (Z, Y, X)
features = extractor.embed_volume(volume)                    # (Z, 1536)
save_frame(out / '000000.npz', features)
print('%d slices -> %s' % (len(features), out / '000000.npz'))
