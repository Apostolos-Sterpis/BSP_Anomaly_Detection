import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class StandardScaler:
    """Z-score scaling: (x - mean) / std. Fit on training only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        x = np.asarray(data, dtype=float).reshape(-1)
        self.mean_ = float(np.mean(x))
        std = float(np.std(x))
        self.std_ = std if std != 0.0 else 1.0
        return self

    def transform(self, data):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError
        x = np.asarray(data, dtype=float).reshape(-1)
        return (x - self.mean_) / self.std_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


class RobustScaler:
    """Robust scaling: (x - median) / IQR. Fit on training only."""

    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, data):
        x = np.asarray(data, dtype=float).reshape(-1)
        self.median_ = float(np.median(x))
        q25, q75 = np.percentile(x, [25, 75])
        iqr = float(q75 - q25)
        self.iqr_ = iqr if iqr != 0.0 else 1.0
        return self

    def transform(self, data):
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError
        x = np.asarray(data, dtype=float).reshape(-1)
        return (x - self.median_) / self.iqr_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


def create_windows(data, window_size, stride=1, return_starts=False):
    """Create sliding windows from a 1D array."""
    x = np.asarray(data, dtype=float).reshape(-1)
    if len(x) < window_size:
        windows = np.empty((0, window_size), dtype=float)
        starts = np.empty((0,), dtype=int)
        return (windows, starts) if return_starts else windows

    windows = sliding_window_view(x, window_shape=window_size)[::stride]
    if return_starts:
        starts = np.arange(len(windows), dtype=int) * stride
        return windows, starts
    return windows


def create_window_labels(point_labels, window_size, stride=1, mode="any"):
    """
    Convert point-wise labels to window-wise labels.
    mode:
      - "any": window is anomalous if any point inside is 1
      - "majority": anomalous if >50% points are 1
    """
    y = np.asarray(point_labels, dtype=int).reshape(-1)
    windows = create_windows(y, window_size, stride)

    if len(windows) == 0:
        return np.empty((0,), dtype=int)

    if mode == "any":
        return windows.max(axis=1).astype(int)

    if mode == "majority":
        return (windows.mean(axis=1) > 0.5).astype(int)

    raise ValueError
