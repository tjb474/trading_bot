#!/usr/bin/env python
"""
Script to validate NR4 and NR7 features by visualizing them on OHLC charts.
"""
from plot_ohlc import plot_ohlc_with_features
import os

if __name__ == '__main__':
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_file = os.path.join(project_root, 'data', 'spy_ohlcv_new.dbn')

    # Plot a week of data to see multiple NR4/NR7 signals
    plot_ohlc_with_features(
        data_file,
        start_date='2025-06-09',
        end_date='2025-06-13'
    )