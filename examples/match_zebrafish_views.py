#!/usr/bin/env python3
"""Fig. 3h from the released features: every time point of zebrafish view 1 (``zf_G1``) matched to view 2
(``zf_G2``).

    python examples/match_zebrafish_views.py /path/to/features out/
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from stern import FeatureSeries, distance_table

root, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else 'out')
g1 = FeatureSeries.from_directory(root / 'zf_G1', name='G1')
g2 = FeatureSeries.from_directory(root / 'zf_G2', name='G2')
table = distance_table(g1, g2)                               # all slices of every volume, all pairs
table.write_best_matches(out / 'view1_to_view2_best_matches.csv')
best = np.array([(q, t) for q, t, _ in table.best_matches()])
print('view 1 -> view 2: Spearman rho = %.3f over %d time points' % (spearmanr(best[:, 0], best[:, 1])[0], len(best)))
