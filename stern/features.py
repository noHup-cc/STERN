"""Storage of slice features: one ``.npz`` file per time point.

Layout (the one used for the released features)::

    <directory>/<time, 6 digits>.npz      arrays  z (n,) int, 1-based slice index
                                                  features (n, 1536) float32, row i = slice z[i]

``FeatureSeries`` holds the time points of one recording (one embryo, heart or zebrafish view).
"""
import os
import re
from pathlib import Path

import numpy as np

_STEM = re.compile(r'^(\d+)$')


def save_frame(path, features, z=None):
    """Write one time point (atomically). ``z`` defaults to 1..n."""
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError('features must be (slices, dim), got shape %s' % (features.shape,))
    z = np.arange(1, len(features) + 1) if z is None else np.asarray(z, dtype=np.int64)
    if z.shape != (len(features),):
        raise ValueError('z must have one entry per row of features')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, z=z, features=features)
    os.replace(str(tmp), str(path))


def load_frame(path):
    """-> (z (n,), features (n, d) float32)."""
    with np.load(str(path), allow_pickle=False) as data:
        return np.asarray(data['z']), np.asarray(data['features'], dtype=np.float32)


class FeatureSeries(object):
    """Slice features of one recording, indexed by time point.

    Frames are read from disk when first used and kept in memory (``cache=True``) or read on every
    access (``cache=False``, for recordings larger than memory)."""

    def __init__(self, files, name=None, cache=True):
        self._files = dict(files)
        self._frames = {}
        self.name = name
        self.cache = cache

    @classmethod
    def from_directory(cls, directory, name=None, cache=True):
        directory = Path(directory)
        files = {}
        for path in directory.glob('*.npz'):
            m = _STEM.match(path.stem)
            if m:
                files[int(m.group(1))] = path
        if not files:
            raise FileNotFoundError('no <time>.npz feature files in %s' % directory)
        return cls(files, name=name or directory.name, cache=cache)

    @classmethod
    def from_arrays(cls, frames, name=None):
        """``frames``: {time: features (n, d)} or {time: (z, features)}."""
        series = cls({}, name=name)
        for t, value in frames.items():
            z, f = value if isinstance(value, tuple) else (None, value)
            f = np.ascontiguousarray(f, dtype=np.float32)
            z = np.arange(1, len(f) + 1) if z is None else np.asarray(z)
            series._frames[int(t)] = (z, f)
        return series

    @property
    def times(self):
        return sorted(set(self._files) | set(self._frames))

    def __len__(self):
        return len(self.times)

    def __contains__(self, time):
        return time in self._files or time in self._frames

    def frame(self, time):
        """-> (z, features) of one time point."""
        if time in self._frames:
            return self._frames[time]
        if time not in self._files:
            raise KeyError('time point %r not in %s' % (time, self.name))
        value = load_frame(self._files[time])
        if self.cache:
            self._frames[time] = value
        return value

    def volume(self, time, z_range=None):
        """Feature rows of one time point, restricted to ``z_range`` = (first z, last z) inclusive.

        Every slice of the range must be present (as in the paper, a missing slice is an error)."""
        z, features = self.frame(time)
        if z_range is None:
            return features
        lo, hi = z_range
        if 1 <= lo <= hi <= len(z) and z[0] == 1 and bool((np.diff(z) == 1).all()):   # z = 1..n (usual)
            return features[lo - 1:hi]
        want = np.arange(lo, hi + 1)
        index = {int(v): i for i, v in enumerate(z)}
        missing = [int(v) for v in want if int(v) not in index]
        if missing:
            raise ValueError('%s t=%d: slices %s..%s of range %s missing' % (self.name, time, missing[0],
                                                                            missing[-1], (lo, hi)))
        return features[[index[int(v)] for v in want]]

    def save(self, directory):
        for t in self.times:
            z, f = self.frame(t)
            save_frame(Path(directory) / ('%06d.npz' % t), f, z)
