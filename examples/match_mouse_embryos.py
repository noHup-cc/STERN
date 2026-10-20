#!/usr/bin/env python3
"""Fig. 2c from the released features: embryo D at t = 0, 100, 250 against embryos A, B, C (CPU, minutes).

    python examples/match_mouse_embryos.py /path/to/features out/
"""
import sys
from pathlib import Path

from stern import FeatureSeries, distance_table, mouse_z_ranges, normalised_similarity

root, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else 'out')
query = FeatureSeries.from_directory(root / 'mouse_D', name='D')
for name in ('A', 'B', 'C'):
    target = FeatureSeries.from_directory(root / ('mouse_' + name), name=name)
    table = distance_table(query, target, query_times=[0, 100, 250],
                           query_ranges=mouse_z_ranges('D'), target_ranges=mouse_z_ranges(name))
    table.to_csv(out / ('D_to_%s.csv' % name))
    for qt, tt, d in table.best_matches():
        times, dist = table.curve(qt)
        sim = normalised_similarity(dist)                    # the curve as plotted in the paper
        print('D t=%3d -> %s best t=%3d (distance %.2f, %d target time points)' % (qt, name, tt, d, len(sim)))
