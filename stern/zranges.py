"""Slice (z) ranges used for the volume distances in the paper.

Mouse embryos: the range of z slices that contains the embryo grows over development. It was fixed by hand
at a few key time points and interpolated linearly in between, in the original analysis code;
``mouse_z_ranges`` reproduces that interpolation exactly, including its rounding.
Mouse hearts: one fixed range per heart. Zebrafish: all slices are used.

All ranges are 1-based and inclusive, like the slice names ``<time>z<k>.jpg`` (k = 1 is the first slice).
"""
import math

# (time, (first z, last z)) at the hand-set key time points of each mouse embryo (IDR0044 recordings).
MOUSE_KEYFRAMES = {
    'A': [(5, (415, 535)), (100, (395, 565)), (200, (335, 625)), (300, (235, 725)), (400, (155, 815)),
          (531, (85, 875))],
    'B': [(0, (385, 525)), (100, (305, 595)), (200, (235, 665)), (285, (165, 725))],
    'C': [(0, (405, 555)), (100, (365, 595)), (200, (305, 655)), (350, (195, 795))],
    'D': [(0, (265, 455)), (100, (215, 505)), (200, (145, 565)), (265, (105, 605))],
}

# Fixed z ranges of the three mouse hearts.
HEART_Z_RANGES = {'A': (20, 125), 'B': (1, 90), 'C': (10, 100)}


def _interpolate(start_time, start_bound, end_time, end_bound):
    """The interpolation of the original analysis: floor of the linear increment, applied to each bound."""
    upper_gradient = (end_bound[1] - start_bound[1]) / (end_time - start_time)
    lower_gradient = (end_bound[0] - start_bound[0]) / (end_time - start_time)
    lower, upper = start_bound
    out = {}
    for time in range(start_time, end_time + 1):
        delta = time - start_time
        if math.floor(delta * upper_gradient) != 0:
            upper = start_bound[1] + math.floor(delta * upper_gradient)
        if math.floor(delta * lower_gradient) != 0:
            lower = start_bound[0] + math.floor(delta * lower_gradient)
        out[time] = (lower, upper)
    return out


def mouse_z_ranges(embryo):
    """{time: (first z, last z)} for mouse embryo 'A', 'B', 'C' or 'D' (every time point of the recording)."""
    try:
        keys = MOUSE_KEYFRAMES[embryo]
    except KeyError:
        raise ValueError('unknown mouse embryo %r (expected one of %s)' % (embryo, ', '.join(MOUSE_KEYFRAMES)))
    out = {}
    for (t0, b0), (t1, b1) in zip(keys[:-1], keys[1:]):
        out.update(_interpolate(t0, b0, t1, b1))
    return out


def heart_z_range(heart):
    """(first z, last z) for mouse heart 'A', 'B' or 'C'."""
    try:
        return HEART_Z_RANGES[heart]
    except KeyError:
        raise ValueError('unknown heart %r (expected one of %s)' % (heart, ', '.join(HEART_Z_RANGES)))


def paper_z_ranges(dataset):
    """z ranges by dataset name as used in the paper: 'mouse_A'..'mouse_D' -> per time point,
    'heart_A'..'heart_C' -> one range for every time point, anything else (zebrafish) -> None (all slices)."""
    kind, _, name = dataset.partition('_')
    if kind == 'mouse':
        return mouse_z_ranges(name)
    if kind == 'heart':
        return heart_z_range(name)
    return None
