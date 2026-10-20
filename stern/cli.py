"""Command line: ``stern extract`` (volumes -> features) and ``stern match`` (features -> distances)."""
import argparse
import sys
import time
from pathlib import Path

from . import __version__


def _times(text):
    """'0,100,250' or '0-265' or '0-265:5' -> list of ints (None for empty)."""
    if not text:
        return None
    out = []
    for part in text.split(','):
        if '-' in part:
            span, _, step = part.partition(':')
            lo, hi = (int(v) for v in span.split('-'))
            out.extend(range(lo, hi + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return out


def _ranges(spec):
    """'all' | 'mouse:D' | 'heart:A' | 'LO-HI' -> z ranges for distance_table."""
    from .zranges import heart_z_range, mouse_z_ranges
    if spec in (None, '', 'all'):
        return None
    kind, _, name = spec.partition(':')
    if kind == 'mouse':
        return mouse_z_ranges(name)
    if kind == 'heart':
        return heart_z_range(name)
    lo, hi = (int(v) for v in spec.split('-'))
    return (lo, hi)


def cmd_extract(args):
    from .extract import Extractor
    from .features import save_frame
    from .io import TiffSeries, read_volume
    extractor = Extractor(args.weights, recipe=args.recipe, jpeg=not args.no_jpeg, device=args.device,
                          batch_size=args.batch_size)
    out = Path(args.out)
    inputs = [Path(p) for p in args.inputs]
    jobs = []
    if len(inputs) == 1 and inputs[0].suffix.lower() in ('.tif', '.tiff') and args.time_series:
        with TiffSeries(inputs[0], channel=args.channel) as series:
            times = _times(args.times) or series.times
        jobs = [(t, inputs[0], t) for t in times]
    else:
        for i, p in enumerate(inputs):
            jobs.append((int(args.first_time) + i, p, 0))
    for t, path, index in jobs:
        target = out / ('%06d.npz' % t)
        if target.exists() and not args.overwrite:
            print('skip %s (exists)' % target)
            continue
        start = time.time()
        volume = read_volume(path, time=index, channel=args.channel)
        save_frame(target, extractor.embed_volume(volume))
        print('t=%d: %d slices -> %s (%.1f s)' % (t, len(volume), target, time.time() - start), flush=True)


def cmd_match(args):
    from .distance import distance_table
    from .features import FeatureSeries
    query = FeatureSeries.from_directory(args.query, name=args.query_name)
    target = FeatureSeries.from_directory(args.target, name=args.target_name)
    pairs = (lambda q, t: t <= q) if args.lower else None
    table = distance_table(query, target, query_times=_times(args.query_times),
                           target_times=_times(args.target_times), query_ranges=_ranges(args.query_ranges),
                           target_ranges=_ranges(args.target_ranges), pairs=pairs,
                           progress=lambda d, n: print('\r%d/%d volume pairs' % (d, n), end='', flush=True))
    print()
    table.to_csv(args.out)
    if args.best:
        table.write_best_matches(args.best)
    for qt, tt, d in table.best_matches()[:10]:
        print('query t=%d -> best target t=%d (distance %.4f)' % (qt, tt, d))
    print('distances: %s' % args.out)


def build_parser():
    p = argparse.ArgumentParser(prog='stern', description='STERN slice features and volume matching')
    p.add_argument('--version', action='version', version='stern ' + __version__)
    sub = p.add_subparsers(dest='command')
    e = sub.add_parser('extract', help='volumes -> slice features (one .npz per time point)')
    e.add_argument('inputs', nargs='+', help='volume files (TIFF, KLB, .npy) or slice folders, one per time '
                                             'point; or one (OME-)TIFF time series with --time-series')
    e.add_argument('--weights', required=True, help='stern_weights.safetensors')
    e.add_argument('--recipe', default='mouse', choices=['mouse', 'heart', 'zebrafish'])
    e.add_argument('--out', required=True, help='output directory')
    e.add_argument('--time-series', action='store_true', help='the input is one TIFF with a T axis')
    e.add_argument('--times', help='time points of a time series, e.g. 0-108 or 0,50,100')
    e.add_argument('--first-time', default=0, help='time point of the first input (one input per time point)')
    e.add_argument('--channel', type=int, default=0)
    e.add_argument('--no-jpeg', action='store_true', help='skip the JPEG round trip of the paper pipeline')
    e.add_argument('--device', default=None)
    e.add_argument('--batch-size', type=int, default=1)
    e.add_argument('--overwrite', action='store_true')
    e.set_defaults(func=cmd_extract)
    m = sub.add_parser('match', help='feature directories -> volume distances and best matches')
    m.add_argument('query')
    m.add_argument('target')
    m.add_argument('--out', required=True, help='distance table (CSV)')
    m.add_argument('--best', help='best matches per query time point (CSV)')
    m.add_argument('--query-name')
    m.add_argument('--target-name')
    m.add_argument('--query-times')
    m.add_argument('--target-times')
    m.add_argument('--query-ranges', default='all', help="'all', 'mouse:D', 'heart:A' or 'LO-HI'")
    m.add_argument('--target-ranges', default='all')
    m.add_argument('--lower', action='store_true', help='only target time <= query time (within-embryo)')
    m.set_defaults(func=cmd_match)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'func', None):
        parser.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
