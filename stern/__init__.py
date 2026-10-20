"""STERN (Spatio-Temporal Embedding for Real-time morphogeNesis).

    from stern import FeatureSeries, distance_table, mouse_z_ranges
    a = FeatureSeries.from_directory('features/mouse_A')
    d = FeatureSeries.from_directory('features/mouse_D')
    table = distance_table(d, a, query_times=[0, 100, 250],
                           query_ranges=mouse_z_ranges('D'), target_ranges=mouse_z_ranges('A'))
    table.best_matches()

Feature extraction (needs torch, timm and the released weights): ``stern.extract.Extractor``.
"""
__version__ = '1.0.0'

from .distance import DistanceTable, distance_table, nearest_slices, normalised_similarity, volume_distance
from .features import FeatureSeries, load_frame, save_frame
from .zranges import heart_z_range, mouse_z_ranges, paper_z_ranges

__all__ = ['DistanceTable', 'distance_table', 'nearest_slices', 'normalised_similarity', 'volume_distance',
           'FeatureSeries', 'load_frame', 'save_frame', 'heart_z_range', 'mouse_z_ranges', 'paper_z_ranges']
