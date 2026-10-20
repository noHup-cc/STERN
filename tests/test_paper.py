"""Paper results from the released features (skipped unless STERN_FEATURES points to them).

    STERN_FEATURES=/path/to/features pytest tests/test_paper.py

The directory holds mouse_A..mouse_D, heart_A..heart_C, zf_G1, zf_G2 (one <time>.npz per time point).
Expected values (tests/data/expected_paper.json) are the published best matches and statistics.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from stern import FeatureSeries, distance_table, mouse_z_ranges

ROOT = os.environ.get('STERN_FEATURES')
EXPECTED = json.loads((Path(__file__).resolve().parent / 'data' / 'expected_paper.json').read_text())
pytestmark = pytest.mark.skipif(not ROOT, reason='set STERN_FEATURES to the released feature directory')


def load(name):
    return FeatureSeries.from_directory(Path(ROOT) / name, name=name.split('_')[-1])


def test_fig2c_best_matches_of_embryo_d():
    d = load('mouse_D')
    for target, expected in EXPECTED['fig2c_best_target_time'].items():
        t = load('mouse_' + target)
        table = distance_table(d, t, query_times=[int(q) for q in expected], query_ranges=mouse_z_ranges('D'),
                               target_ranges=mouse_z_ranges(target))
        assert {str(q): b for q, b, _ in table.best_matches()} == expected, target


def test_fig3h_zebrafish_cross_view_matching():
    from scipy.stats import spearmanr
    g1, g2 = load('zf_G1'), load('zf_G2')
    best = distance_table(g1, g2).best_matches()
    q = np.array([b[0] for b in best])
    t = np.array([b[1] for b in best])
    assert round(float(spearmanr(q, t)[0]), 3) == EXPECTED['fig3h_spearman_rho']
