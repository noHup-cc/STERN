"""Unit tests without the network: z ranges, feature files, volume distances, matching, CLI."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from stern import (FeatureSeries, distance_table, heart_z_range, load_frame, mouse_z_ranges, nearest_slices,
                   normalised_similarity, paper_z_ranges, save_frame, volume_distance)
from stern.cli import main as cli_main

DATA = Path(__file__).resolve().parent / 'data'


# ----------------------------------------------------------------------------- z ranges
def test_mouse_z_ranges_equal_the_original_function():
    """Golden values: output of the original analysis code for every time point of A-D."""
    golden = json.loads((DATA / 'mouse_z_ranges_paper.json').read_text())
    for embryo, table in golden.items():
        ours = mouse_z_ranges(embryo)
        assert sorted(ours) == sorted(int(t) for t in table)
        assert all(list(ours[int(t)]) == v for t, v in table.items())


def test_ranges_by_dataset():
    assert heart_z_range('B') == (1, 90)
    assert paper_z_ranges('heart_A') == (20, 125)
    assert paper_z_ranges('mouse_D')[0] == (265, 455)
    assert paper_z_ranges('zf_G1') is None
    with pytest.raises(ValueError):
        mouse_z_ranges('E')


# ----------------------------------------------------------------------------- features
def test_feature_files_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    f = rng.normal(size=(7, 16)).astype(np.float32)
    save_frame(tmp_path / '000003.npz', f)
    z, g = load_frame(tmp_path / '000003.npz')
    assert list(z) == list(range(1, 8)) and np.array_equal(f, g)
    s = FeatureSeries.from_directory(tmp_path)
    assert s.times == [3] and np.array_equal(s.volume(3, (2, 4)), f[1:4])
    save_frame(tmp_path / 'other' / '000001.npz', f[:3], z=[5, 6, 8])
    t = FeatureSeries.from_directory(tmp_path / 'other')
    assert np.array_equal(t.volume(1, (5, 6)), f[:2])
    with pytest.raises(ValueError):
        t.volume(1, (5, 7))                                      # slice 7 missing


# ----------------------------------------------------------------------------- distances
def legacy_volume_distance(query_frames, query_time, query_bounds, target_frames, target_time, target_bounds):
    """The volume distance of the original analysis code (per-slice loop over sklearn kneighbors)."""
    from sklearn.neighbors import NearestNeighbors
    lo, hi = target_bounds[target_time]
    target = [target_frames[target_time][z - 1][1].squeeze() for z in range(lo, hi + 1)]
    nn = NearestNeighbors(n_neighbors=len(target), algorithm='brute', metric='euclidean').fit(target)
    total = 0
    qlo, qhi = query_bounds[query_time]
    for z in range(qlo, qhi + 1):
        d, _ = nn.kneighbors([query_frames[query_time][z - 1][1].squeeze()])
        total += d[0][0]
    return total


def series(rng, times, slices, dim=32, name=None):
    return FeatureSeries.from_arrays({t: rng.normal(size=(slices, dim)).astype(np.float32) for t in times}, name)


def test_volume_distance_definition():
    rng = np.random.default_rng(1)
    q, t = rng.normal(size=(5, 8)), rng.normal(size=(9, 8))
    brute = np.sqrt(((q[:, None, :] - t[None, :, :]) ** 2).sum(-1)).min(1).sum()
    assert math.isclose(volume_distance(q, t), brute, rel_tol=1e-12)
    assert volume_distance(q, q) < 1e-6
    with pytest.raises(ValueError):
        volume_distance(q, t[:, :4])


def test_distance_table_equals_the_original_loop():
    rng = np.random.default_rng(2)
    a, d = series(rng, range(0, 6), 40, name='A'), series(rng, range(0, 4), 40, name='D')
    ranges_a = {t: (3 + t % 2, 30 + t) for t in range(0, 6)}
    ranges_d = {t: (5, 25 + t) for t in range(0, 4)}
    table = distance_table(d, a, query_ranges=ranges_d, target_ranges=ranges_a)
    legacy_frames = {s.name: {t: [[z, s.frame(t)[1][z - 1]] for z in range(1, 41)] for t in s.times} for s in (a, d)}
    for qt in d.times:
        for tt in a.times:
            ref = legacy_volume_distance(legacy_frames['D'], qt, ranges_d, legacy_frames['A'], tt, ranges_a)
            assert math.isclose(table.distance(qt, tt), float(ref), rel_tol=1e-6)
    best = table.best_matches()
    assert [b[0] for b in best] == d.times
    assert all(b[1] == int(a.times[int(np.argmin(table.curve(b[0])[1]))]) for b in best)


def test_default_times_are_those_with_z_ranges():
    rng = np.random.default_rng(8)
    a = series(rng, range(0, 6), 12, name='A')
    table = distance_table(a, a, query_ranges={t: (1, 10) for t in range(2, 6)},
                           target_ranges={t: (2, 12) for t in range(3, 6)})
    assert list(table.query_times) == [2, 3, 4, 5] and list(table.target_times) == [3, 4, 5]
    assert len(distance_table(a, a, query_ranges=(1, 5)).query_times) == 6


def test_best_match_recovers_a_shifted_time_axis():
    """Target = query delayed by 3 time points (plus noise): every best match is t + 3."""
    rng = np.random.default_rng(3)
    base = {t: rng.normal(size=(20, 16)).astype(np.float32) for t in range(0, 15)}
    query = FeatureSeries.from_arrays({t: base[t] for t in range(0, 10)}, 'Q')
    target = FeatureSeries.from_arrays({t + 3: base[t] + 0.01 * rng.normal(size=base[t].shape) for t in base}, 'T')
    best = distance_table(query, target).best_matches()
    assert [(q, t) for q, t, _ in best] == [(q, q + 3) for q in range(0, 10)]


def test_lower_triangle_and_csv_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    a = series(rng, range(0, 5), 10, name='A')
    table = distance_table(a, a, pairs=lambda q, t: t <= q)
    assert np.isnan(table.values[0, 1]) and table.distance(3, 3) < 1e-4
    table.to_csv(tmp_path / 't.csv')
    rows = list(csv.DictReader((tmp_path / 't.csv').open()))
    assert len(rows) == 15 and rows[0]['queryEmbryo'] == 'A'
    back = type(table).from_csv(tmp_path / 't.csv')
    assert np.allclose(back.values, table.values, equal_nan=True)


def test_nearest_slices_and_similarity():
    rng = np.random.default_rng(5)
    a = series(rng, range(0, 3), 12, name='A')
    query = a.frame(1)[1][4]
    top = nearest_slices(query, a, k=5)
    assert top[0][:2] == (1, 5) and top[0][2] < 1e-4 and [d for _, _, d in top] == sorted(d for _, _, d in top)
    s = normalised_similarity([3.0, 1.0, 2.0])
    assert list(s) == [0.0, 1.0, 0.5]


def test_cli_match(tmp_path, capsys):
    rng = np.random.default_rng(6)
    for name in ('Q', 'T'):
        series(rng, range(0, 4), 30).save(tmp_path / name)
    out = tmp_path / 'd.csv'
    assert cli_main(['match', str(tmp_path / 'Q'), str(tmp_path / 'T'), '--out', str(out), '--best',
                     str(tmp_path / 'b.csv'), '--query-ranges', '2-20', '--target-ranges', 'all',
                     '--query-times', '0-3']) == 0
    assert len(list(csv.DictReader(out.open()))) == 16 and len(list(csv.DictReader((tmp_path / 'b.csv').open()))) == 4


# ----------------------------------------------------------------------------- reading volumes
def test_read_tiff_stacks_and_time_series(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    from stern.io import TiffSeries, read_volume
    rng = np.random.default_rng(7)
    stack = rng.integers(0, 255, (5, 12, 10)).astype(np.uint8)
    tifffile.imwrite(str(tmp_path / 'plain.tif'), stack)                       # pages -> axes 'IYX'
    assert np.array_equal(read_volume(tmp_path / 'plain.tif'), stack)
    series = rng.integers(0, 4000, (3, 4, 12, 10)).astype(np.uint16)
    tifffile.imwrite(str(tmp_path / 'ts.ome.tif'), series, metadata={'axes': 'TZYX'})
    with TiffSeries(tmp_path / 'ts.ome.tif') as ts:
        assert ts.times == [0, 1, 2] and np.array_equal(ts.volume(2), series[2])
    folder = tmp_path / 'slices'
    folder.mkdir()
    from PIL import Image
    for k in (10, 2, 1):
        Image.fromarray(stack[k % 5]).save(str(folder / ('7z%d.png' % k)))
    assert np.array_equal(read_volume(folder), stack[[1, 2, 0]])
