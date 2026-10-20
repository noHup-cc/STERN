"""Distances between volumes (time points) and the matching of one recording against another.

Volume distance (the definition used throughout the paper): for query volume Q and target volume T, each
represented by the features of their slices,

    D(Q -> T) = sum over query slices q of  min over target slices t of  || q - t ||_2

restricted to the slice (z) ranges of each volume (``stern.zranges``). It is asymmetric. Nearest
neighbours are computed exactly (brute force, scikit-learn ``NearestNeighbors``, as in the original code);
the per-slice minima are summed in float64.
"""
import csv
import math
from pathlib import Path

import numpy as np

__all__ = ['volume_distance', 'distance_table', 'DistanceTable', 'nearest_slices', 'normalised_similarity']


def _neighbours(target):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean').fit(target)


def volume_distance(query, target):
    """D(query -> target) for two (slices, dim) feature arrays."""
    query, target = np.asarray(query), np.asarray(target)
    if query.ndim != 2 or target.ndim != 2 or query.shape[1] != target.shape[1]:
        raise ValueError('expected (slices, dim) arrays with the same dim, got %s and %s'
                         % (query.shape, target.shape))
    distances, _ = _neighbours(target).kneighbors(query)
    return float(distances[:, 0].sum(dtype=np.float64))


def _range(ranges, time):
    """z range of one time point from None (all), (lo, hi) (fixed) or {time: (lo, hi)}."""
    if ranges is None or isinstance(ranges, tuple):
        return ranges
    return tuple(ranges[time])


class DistanceTable(object):
    """D(query time -> target time) for every computed pair (NaN where not computed)."""

    def __init__(self, query_times, target_times, values, query_name=None, target_name=None):
        self.query_times = np.asarray(query_times, dtype=np.int64)
        self.target_times = np.asarray(target_times, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)
        self.query_name, self.target_name = query_name, target_name

    def distance(self, query_time, target_time):
        i = int(np.searchsorted(self.query_times, query_time))
        j = int(np.searchsorted(self.target_times, target_time))
        return float(self.values[i, j])

    def curve(self, query_time):
        """(target times, distances) of one query time point (computed pairs only)."""
        row = self.values[int(np.searchsorted(self.query_times, query_time))]
        keep = ~np.isnan(row)
        return self.target_times[keep], row[keep]

    def best_matches(self):
        """[(query time, best target time, distance)]: the target time point with the smallest distance."""
        out = []
        for i, qt in enumerate(self.query_times):
            row = self.values[i]
            if np.isnan(row).all():
                continue
            j = int(np.nanargmin(row))
            out.append((int(qt), int(self.target_times[j]), float(row[j])))
        return out

    def to_csv(self, path):
        """Long format, as the paper's result tables: queryEmbryo, query_time, targetEmbryo, target_time,
        distance (one row per computed pair, ordered by query then target time)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            w = csv.writer(handle)
            w.writerow(['queryEmbryo', 'query_time', 'targetEmbryo', 'target_time', 'distance'])
            for i, qt in enumerate(self.query_times):
                for j, tt in enumerate(self.target_times):
                    if not math.isnan(self.values[i, j]):
                        w.writerow([self.query_name, int(qt), self.target_name, int(tt),
                                    repr(float(self.values[i, j]))])

    @classmethod
    def from_csv(cls, path):
        rows = list(csv.DictReader(Path(path).open()))
        q = sorted({int(r['query_time']) for r in rows})
        t = sorted({int(r['target_time']) for r in rows})
        values = np.full((len(q), len(t)), np.nan)
        qi, ti = {v: i for i, v in enumerate(q)}, {v: i for i, v in enumerate(t)}
        for r in rows:
            values[qi[int(r['query_time'])], ti[int(r['target_time'])]] = float(r['distance'])
        names = (rows[0]['queryEmbryo'], rows[0]['targetEmbryo']) if rows else (None, None)
        return cls(q, t, values, *names)

    def write_best_matches(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            w = csv.writer(handle)
            w.writerow(['query_time', 'best_target_time', 'distance'])
            for qt, tt, d in self.best_matches():
                w.writerow([qt, tt, repr(d)])


def _default_times(series, times, ranges):
    """Given times, or every time point of the series that has a z range (all for None / fixed ranges)."""
    if times is not None:
        return list(times)
    if isinstance(ranges, dict):
        return [t for t in series.times if t in ranges]
    return list(series.times)


def distance_table(query, target, query_times=None, target_times=None, query_ranges=None, target_ranges=None,
                   pairs=None, progress=None):
    """D(query t -> target t') for the selected time points of two ``FeatureSeries``.

    query_ranges / target_ranges: None (all slices), (lo, hi) or {time: (lo, hi)} (e.g.
    ``stern.zranges.mouse_z_ranges('D')``). pairs: optional function (query time, target time) -> bool
    choosing the pairs to compute (e.g. ``lambda q, t: t <= q`` for the within-embryo matrices of the
    paper). Without explicit times, every time point that has a z range is used (as in the paper, where
    the ranges define the time span; mouse embryo A starts at t = 5). progress: optional callable(done,
    total). Each target volume is loaded and indexed once."""
    query_times = _default_times(query, query_times, query_ranges)
    target_times = _default_times(target, target_times, target_ranges)
    values = np.full((len(query_times), len(target_times)), np.nan)
    work = [(i, j) for i in range(len(query_times)) for j in range(len(target_times))
            if pairs is None or pairs(query_times[i], target_times[j])]
    by_target = {}
    for i, j in work:
        by_target.setdefault(j, []).append(i)
    queries = {}
    done = 0
    for j in sorted(by_target):
        tt = target_times[j]
        nn = _neighbours(target.volume(tt, _range(target_ranges, tt)))
        for i in by_target[j]:
            qt = query_times[i]
            if qt not in queries:
                queries[qt] = query.volume(qt, _range(query_ranges, qt))
            distances, _ = nn.kneighbors(queries[qt])
            values[i, j] = float(distances[:, 0].sum(dtype=np.float64))
            done += 1
        if progress is not None:
            progress(done, len(work))
    return DistanceTable(query_times, target_times, values, getattr(query, 'name', None),
                         getattr(target, 'name', None))


def nearest_slices(query_vector, target, k=50, target_times=None, target_ranges=None):
    """The k slices of ``target`` closest to one query slice: [(time, z, distance)] in increasing distance
    (all slices of every time point unless ranges are given)."""
    from sklearn.neighbors import NearestNeighbors
    times = list(target.times if target_times is None else target_times)
    keys, blocks = [], []
    for t in times:
        z, f = target.frame(t)
        r = _range(target_ranges, t)
        keep = np.ones(len(z), bool) if r is None else (z >= r[0]) & (z <= r[1])
        keys.extend((int(t), int(v)) for v in z[keep])
        blocks.append(f[keep])
    matrix = np.concatenate(blocks)
    nn = NearestNeighbors(n_neighbors=min(k, len(matrix)), algorithm='brute', metric='euclidean').fit(matrix)
    distances, index = nn.kneighbors(np.asarray(query_vector, dtype=matrix.dtype).reshape(1, -1))
    return [(keys[i][0], keys[i][1], float(d)) for i, d in zip(index[0], distances[0])]


def normalised_similarity(distances):
    """1 - (d - min) / (max - min): the similarity scale of the paper's curves and heat maps."""
    d = np.asarray(distances, dtype=np.float64)
    lo, hi = np.nanmin(d), np.nanmax(d)
    return np.ones_like(d) if hi == lo else 1.0 - (d - lo) / (hi - lo)
