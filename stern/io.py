"""Reading volumes (z, y, x) from common image formats.

  TIFF / OME-TIFF   one volume per file (ZYX), or a time series (TZYX, e.g. the zebrafish recordings);
                    time points are read page by page, so large files are never loaded whole
  image folder      a directory of 2D slices (.png / .jpg / .tif), ordered by the slice number in the
                    file name (the paper's '<time>z<k>.jpg' names, or any name ending in a number)
  KLB               the IDR0044 mouse embryos (needs the optional ``pyklb``)
  numpy             .npy arrays (ZYX)
"""
import re
from pathlib import Path

import numpy as np

_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
_TIFF_SUFFIXES = ('.tif', '.tiff')


def _slice_number(name):
    m = re.search(r'z(\d+)$', name) or re.search(r'(\d+)$', name)
    if not m:
        raise ValueError('no slice number in file name %r' % name)
    return int(m.group(1))


class TiffSeries(object):
    """Lazy access to the time points of a (OME-)TIFF with axes ZYX or TZYX (optionally C)."""

    def __init__(self, path, channel=0):
        import tifffile
        self.path = Path(path)
        self._file = tifffile.TiffFile(str(self.path))
        series = self._file.series[0]
        axes = series.axes
        if 'Z' not in axes and sum(axes.count(a) for a in 'IQ') == 1:     # plain multi-page stack
            axes = axes.replace('I', 'Z').replace('Q', 'Z')
        self.axes, self.shape = axes, tuple(series.shape)
        if not set('ZYX') <= set(self.axes) or set(self.axes) - set('TCZYX'):
            raise ValueError('%s: unsupported axes %s (need ZYX, optionally T and C)' % (path, self.axes))
        self.channel = channel
        self.size = dict(zip(self.axes, self.shape))
        self._series = series

    @property
    def times(self):
        return list(range(self.size.get('T', 1)))

    def volume(self, time=0):
        """(Z, Y, X) array of one time point."""
        if 'T' not in self.size and time != 0:
            raise IndexError('%s has no time axis' % self.path)
        index = [slice(None)] * len(self.axes)
        if 'T' in self.size:
            index[self.axes.index('T')] = time
        if 'C' in self.size:
            index[self.axes.index('C')] = self.channel
        pages = self._page_numbers(time)
        if pages is not None:
            stack = self._file.asarray(key=pages, series=0)
            return np.asarray(stack).reshape(self.size['Z'], self.size['Y'], self.size['X'])
        return np.asarray(self._series.asarray()[tuple(index)])

    def _page_numbers(self, time):
        """Page indices of one time point when pages are stored in T, (C,) Z order."""
        order = [a for a in self.axes if a not in 'YX']
        if order not in (['Z'], ['T', 'Z'], ['T', 'C', 'Z'], ['C', 'Z']):
            return None
        z = self.size['Z']
        c = self.size.get('C', 1)
        start = (time * c + (self.channel if 'C' in self.size else 0)) * z
        return list(range(start, start + z))

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_volume(path, time=0, channel=0):
    """One (Z, Y, X) volume from a TIFF / OME-TIFF (time point ``time``), an image folder, a KLB or .npy."""
    path = Path(path)
    if path.is_dir():
        files = sorted((p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES),
                       key=lambda p: _slice_number(p.stem))
        if not files:
            raise FileNotFoundError('no image slices in %s' % path)
        from PIL import Image
        return np.stack([np.asarray(Image.open(str(p))) for p in files])
    suffix = path.suffix.lower()
    if suffix in _TIFF_SUFFIXES:
        with TiffSeries(path, channel=channel) as series:
            return series.volume(time)
    if suffix == '.klb':
        try:
            import pyklb
        except ImportError:
            raise ImportError('reading KLB needs pyklb (pip install stern[klb], see README)')
        return np.asarray(pyklb.readfull(str(path)))
    if suffix == '.npy':
        return np.load(str(path))
    raise ValueError('unsupported volume format: %s' % path)
